//! Color fields, tonemapping and dithering.
//!
//! Each background type computes linear RGB for a pixel (u, v); values may exceed 1.
//! u, v are in units of image height: v runs 0..1 from the top, u runs 0..aspect from the left.
//! Then: exposure -> tonemapping -> sRGB -> shading under content -> stars -> grain/dither.
//!
//! Techniques: value noise + fbm with rotated octaves, domain warping, an aurora band around a wavy
//! spine, ACES tonemapping, grain coarser than a pixel. Random draws happen in the same order as in
//! the original numpy version, so a given seed still produces the same layout.

use std::f64::consts::PI;
use std::sync::OnceLock;

use image::RgbImage;
use rayon::prelude::*;

use crate::rng::Rng;

pub type Rgb = [f64; 3];

const LUMA: Rgb = [0.2126, 0.7152, 0.0722];
/// Fields are smooth: they are computed at a lower resolution and scaled up; grain is added at full size.
const FIELD_MAX_W: usize = 1280;

// --- color --------------------------------------------------------------------------------------

fn hex2rgb(h: &str) -> Result<Rgb, String> {
    let s = h.trim().trim_start_matches('#');
    let s: String = if s.len() == 3 { s.chars().flat_map(|c| [c, c]).collect() } else { s.to_string() };
    let bad = || format!("invalid color {h:?} (expected #RRGGBB or #RGB)");
    if s.len() != 6 || !s.is_ascii() {
        return Err(bad());
    }
    let mut out = [0.0; 3];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|_| bad())? as f64 / 255.0;
    }
    Ok(out)
}

fn to_linear(c: f64) -> f64 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn to_srgb(c: f64) -> f64 {
    let c = c.max(0.0);
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

fn map3(c: Rgb, f: impl Fn(f64) -> f64) -> Rgb {
    [f(c[0]), f(c[1]), f(c[2])]
}

fn add(a: Rgb, b: Rgb) -> Rgb {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: Rgb, b: Rgb) -> Rgb {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: Rgb, k: f64) -> Rgb {
    [a[0] * k, a[1] * k, a[2] * k]
}

fn dot(a: Rgb, b: Rgb) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn clip01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = clip01((x - a) / (b - a));
    t * t * (3.0 - 2.0 * t)
}

fn reinhard(x: f64) -> f64 {
    x / (1.0 + 0.55 * x)
}

fn aces(x: f64) -> f64 {
    // ACES filmic curve fit (Narkowicz): soft highlights, stronger contrast in the shadows
    clip01((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14))
}

pub const TONEMAPS: [&str; 3] = ["none", "reinhard", "aces"];

fn tonemap_fn(name: &str) -> fn(f64) -> f64 {
    match name {
        "reinhard" => reinhard,
        "aces" => aces,
        _ => clip01,
    }
}

// --- noise --------------------------------------------------------------------------------------

/// Deterministic hash of grid points -> 0..1 (bit mixing, no sin/fract as in GLSL).
fn hash(ix: i64, iy: i64, seed: i64) -> f64 {
    const M: i64 = 0xFFFF_FFFF;
    let mut h = ((ix.wrapping_mul(0x27D4_EB2D) ^ iy.wrapping_mul(0x1656_67B1) ^ (seed.wrapping_mul(0x9E37_79B1) & M)) & M)
        as u64;
    h = ((h ^ (h >> 15)) * 0x2C1B_3C6D) & 0xFFFF_FFFF;
    h = ((h ^ (h >> 12)) * 0x297A_2D39) & 0xFFFF_FFFF;
    h ^= h >> 15;
    (h & 0xFF_FFFF) as f64 / 0xFF_FFFF as f64
}

/// Value noise: random values at grid nodes, smoothstep interpolation.
fn vnoise(x: f64, y: f64, seed: i64) -> f64 {
    let (fx, fy) = (x.floor(), y.floor());
    let (ix, iy) = (fx as i64, fy as i64);
    let (tx, ty) = (x - fx, y - fy);
    let (ux, uy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let (a, b) = (hash(ix, iy, seed), hash(ix + 1, iy, seed));
    let (c, d) = (hash(ix, iy + 1, seed), hash(ix + 1, iy + 1, seed));
    a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy
}

fn rot() -> (f64, f64) {
    (0.5f64.cos(), 0.5f64.sin())
}

/// Sum of noise octaves, each 2x denser and rotated; the rotation hides axis-aligned grid artifacts. Result ~0..1.
fn fbm(mut x: f64, mut y: f64, octaves: i64, seed: i64) -> f64 {
    let (c, s) = rot();
    let (mut total, mut amp, mut norm) = (0.0, 0.5, 0.0);
    for o in 0..octaves {
        total += amp * vnoise(x, y, seed + o * 101);
        norm += amp;
        (x, y) = ((c * x - s * y) * 2.0 + 17.3, (s * x + c * y) * 2.0 + 31.7);
        amp *= 0.5;
    }
    total / norm
}

/// Squared value noise: more dark valleys, sparser bright ridges.
fn vnoise_sq(x: f64, y: f64, seed: i64) -> f64 {
    let n = vnoise(x, y, seed);
    n * n
}

fn fbm_sq(mut x: f64, mut y: f64, seed: i64) -> f64 {
    let (c, s) = rot();
    let (mut total, mut amp) = (0.0, 0.5);
    for o in 0..4 {
        total += amp * vnoise_sq(x, y, seed + o * 101);
        (x, y) = ((c * x - s * y) * 2.0 + 100.0, (s * x + c * y) * 2.0 + 100.0);
        amp *= 0.5;
    }
    total
}

/// 2D gradient (Perlin) noise, result roughly -0.7..0.7.
fn perlin(x: f64, y: f64, seed: i64) -> f64 {
    let (fx, fy) = (x.floor(), y.floor());
    let (ix, iy) = (fx as i64, fy as i64);
    let (tx, ty) = (x - fx, y - fy);
    let corner = |dx: i64, dy: i64| {
        let ang = hash(ix + dx, iy + dy, seed) * (2.0 * PI);
        ang.cos() * (tx - dx as f64) + ang.sin() * (ty - dy as f64)
    };
    let ux = tx * tx * tx * (tx * (tx * 6.0 - 15.0) + 10.0);
    let uy = ty * ty * ty * (ty * (ty * 6.0 - 15.0) + 10.0);
    let c00 = corner(0, 0);
    let c01 = corner(0, 1);
    let top = c00 + (corner(1, 0) - c00) * ux;
    let bot = c01 + (corner(1, 1) - c01) * ux;
    top + (bot - top) * uy
}

// --- palettes -----------------------------------------------------------------------------------
// A palette is 4-6 shades of one color family ordered dark to light, not a dark
// background with colored blobs. Blending happens in OKLab, so transitions don't turn grey or muddy.

type M3 = [[f64; 3]; 3];

const M1: M3 = [
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
];
const M2: M3 = [
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
];

fn inv(m: &M3) -> M3 {
    let [[a, b, c], [d, e, f], [g, h, i]] = *m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]
}

fn inverses() -> &'static (M3, M3) {
    static INV: OnceLock<(M3, M3)> = OnceLock::new();
    INV.get_or_init(|| (inv(&M1), inv(&M2)))
}

fn mul(m: &M3, c: Rgb) -> Rgb {
    [dot(m[0], c), dot(m[1], c), dot(m[2], c)]
}

fn to_oklab(c: Rgb) -> Rgb {
    mul(&M2, map3(mul(&M1, c), f64::cbrt))
}

fn from_oklab(c: Rgb) -> Rgb {
    let (m1i, m2i) = inverses();
    map3(mul(m1i, map3(mul(m2i, c), |x| x * x * x)), |x| x.max(0.0))
}

pub struct Palette {
    pub name: &'static str,
    /// hex, any order; the ramp is sorted by luminance
    pub stops: &'static [&'static str],
    pub note: &'static str,
}

const fn pal(name: &'static str, stops: &'static [&'static str], note: &'static str) -> Palette {
    Palette { name, stops, note }
}

pub const DEFAULT_PALETTE: &str = "evening";

pub const PALETTES: &[Palette] = &[
    // dark, for the TV (light text on top)
    pal("evening", &["#120A22", "#4A1F6E", "#B2366E", "#FF5E62", "#FFB07A"], "violet through raspberry to peach"),
    pal("night", &["#070B1A", "#16236B", "#3B4FD0", "#7C83F2", "#C9C4FF"], "navy to lavender"),
    pal("dawn", &["#1A0E14", "#5E2448", "#D0577A", "#F59A73", "#FFE0A8"], "plum, pink, apricot"),
    pal("borealis", &["#0B1726", "#173F4A", "#2A8A72", "#6FD9A0", "#E4FBE8"], "night greens: navy, teal, mint"),
    pal("amethyst", &["#1F0D3D", "#4A2187", "#7F45C9", "#B07FE8", "#E6D6FB"], "violets from ink to lilac"),
    pal("deep-sea", &["#06121F", "#0D3550", "#16698A", "#3FA9BF", "#B6E8E6"], "sea navy to icy blue"),
    pal("graphite", &["#0F1012", "#2C2E33", "#5D6068", "#A3A6AD", "#E6E7EA"], "cool greys"),
    // light / pastel
    pal("blue-sky", &["#E6F2FF", "#B3D9FF", "#80B3FF", "#6699E6"], "sky blue; best with the sky type"),
    pal(
        "storm",
        &["#E8EAEE", "#B7BDC9", "#8C94A4", "#596273"],
        "rainy greys; sky + scale 52, warp 62, wind 58",
    ),
    pal(
        "night-sky",
        &["#B9C8F2", "#6E7FBF", "#4A5AA0", "#33406E"],
        "night navy; sky + scale 42, warp 46, wind 28, stars",
    ),
    pal("snow", &["#F4F7FB", "#D5DFEA", "#A9B8CB", "#7488A3"], "frosty whites and steel blue; sky + snow"),
    pal("lavender", &["#3B2A63", "#6B54A8", "#A493D6", "#D6C9EE", "#F4EEFB"], "pastel violet"),
    pal("peach", &["#5A2A3A", "#B4545E", "#EE8F7A", "#F8C39B", "#FDEBD6"], "coral and cream"),
    pal("mint", &["#123B35", "#23766A", "#4FB99A", "#A6E0C2", "#EAF6E4"], "sea greens"),
    pal("glacier", &["#15324F", "#2C6F9B", "#5FB1D1", "#B4E1E6", "#EFF8F5"], "blues"),
    pal("citrus", &["#5B3A12", "#B0712A", "#EBB64C", "#F6DD8C", "#FBF3D5"], "honey and lemon"),
    pal("rose", &["#4A1E33", "#8E3B5F", "#CF7E95", "#EDB9C2", "#FBEAEA"], "powder pink"),
    pal("sage", &["#2E4436", "#557A60", "#8FB094", "#C7DCC3", "#EEF3E7"], "sage, mist"),
    pal("sunset", &["#2A1440", "#7B2E6B", "#D9536A", "#F59A61", "#FCD9A0"], "full sunset"),
];

pub fn palette_names() -> String {
    PALETTES.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")
}

/// Palette ramp: linear RGB from darkest to lightest, plus the same colors in OKLab.
struct Ramp {
    lin: Vec<Rgb>,
    lab: Vec<Rgb>,
}

impl Ramp {
    fn new(stops: &[String]) -> Result<Self, String> {
        let mut lin = stops.iter().map(|h| hex2rgb(h).map(|c| map3(c, to_linear))).collect::<Result<Vec<_>, _>>()?;
        lin.sort_by(|a, b| dot(*a, LUMA).total_cmp(&dot(*b, LUMA)));
        let lab = lin.iter().map(|&c| to_oklab(c)).collect();
        Ok(Ramp { lin, lab })
    }

    /// Color from the ramp for t in 0..1, interpolated in OKLab.
    fn sample(&self, t: f64) -> Rgb {
        let n = self.lab.len();
        let t = clip01(t) * (n - 1) as f64;
        let i = (t as usize).min(n - 2);
        let f = t - i as f64;
        from_oklab(add(scale(self.lab[i], 1.0 - f), scale(self.lab[i + 1], f)))
    }
}

fn resolve_stops(palette: Option<&str>, colors: Option<&[String]>) -> Result<Vec<String>, String> {
    if let Some(c) = colors.filter(|c| !c.is_empty()) {
        if c.len() < 2 {
            return Err("at least 2 colors are needed".into());
        }
        return Ok(c.to_vec());
    }
    let name = palette.filter(|p| !p.is_empty()).unwrap_or(DEFAULT_PALETTE);
    PALETTES
        .iter()
        .find(|p| p.name == name)
        .map(|p| p.stops.iter().map(|s| s.to_string()).collect())
        .ok_or_else(|| format!("unknown palette {name:?}; available: {}", palette_names()))
}

// --- background types ---------------------------------------------------------------------------

/// Parameters of the sky type (weather sliders, in %).
#[derive(Clone, Copy, Debug, Default)]
pub struct SkyParams {
    pub scale: Option<f64>,
    pub warp: Option<f64>,
    pub wind: Option<f64>,
    pub time: Option<f64>,
}

/// Color field for a pixel: (pixel index in the grid, u, v) -> linear RGB.
type Field = Box<dyn Fn(usize, f64, f64) -> Rgb + Sync + Send>;

struct Ctx<'a> {
    aspect: f64,
    ramp: Ramp,
    rng: &'a mut Rng,
    n_pixels: usize,
    sky: SkyParams,
}

/// Mesh gradient: color points scattered over a grid, normalized Gaussian weights, blended in OKLab.
fn mesh(c: Ctx) -> Field {
    let lab = c.ramp.lab;
    let len = lab.len();
    let n = len.max(6);
    let base: Vec<usize> = (0..n).map(|i| i % len).collect();
    let cols = c.rng.permutation(&base);
    let gx = n.div_ceil(2);
    let gy = 2;
    let p: Vec<f64> = (0..4).map(|_| c.rng.uniform(0.0, 6.0)).collect();
    let pts: Vec<(f64, f64, f64, Rgb)> = (0..n)
        .map(|k| {
            let cx = ((k % gx) as f64 + c.rng.uniform(0.15, 0.85)) / gx as f64 * c.aspect;
            let cy = ((k / gx % gy) as f64 + c.rng.uniform(0.1, 0.9)) / gy as f64;
            // smaller r = each color keeps its own area, less averaged-out grey
            let r = c.rng.uniform(0.2, 0.28);
            (cx, cy, 2.0 * r * r, lab[cols[k]])
        })
        .collect();
    Box::new(move |_, u, v| {
        let wu = u + 0.12 * (v * 3.1 + p[0]).sin() + 0.05 * ((u + v) * 6.3 + p[1]).sin();
        let wv = v + 0.12 * (u * 2.7 + p[2]).sin() + 0.05 * ((u - v) * 5.1 + p[3]).sin();
        let (mut acc, mut wsum) = ([0.0; 3], 0.0);
        for &(cx, cy, r2, col) in &pts {
            let w = (-((wu - cx).powi(2) + (wv - cy).powi(2)) / r2).exp() + 1e-9;
            acc = add(acc, scale(col, w));
            wsum += w;
        }
        from_oklab(scale(acc, 1.0 / wsum))
    })
}

/// Swirling clouds: coordinates warped by sines a few times -> palette gradient.
fn flow(c: Ctx) -> Field {
    let pq: Vec<(f64, f64)> = (0..4).map(|_| (c.rng.uniform(0.0, 6.0), c.rng.uniform(0.0, 6.0))).collect();
    let (ramp, aspect) = (c.ramp, c.aspect);
    Box::new(move |_, u, v| {
        // mirrored: bright clouds on the right, dark on the left (under the content)
        let (mut a, mut b) = ((1.0 - u / aspect) * 3.0, v * 1.7);
        for (k, &(p, q)) in pq.iter().enumerate() {
            let k = k as f64;
            (a, b) = (a + 0.45 * (b * (1.3 + k * 0.4) + p).sin(), b + 0.45 * (a * (1.1 + k * 0.5) + q).sin());
        }
        ramp.sample(0.5 + 0.5 * (a * 1.2 + b * 0.8).sin())
    })
}

/// Stepped vertical bands glowing up from the bottom to varying heights.
fn prism(c: Ctx) -> Field {
    const COLS: f64 = 30.0;
    let peak = c.rng.uniform(0.65, 0.85);
    let phase = c.rng.uniform(0.0, 6.0);
    let (ramp, aspect) = (c.ramp, c.aspect);
    Box::new(move |_, u, v| {
        let col = (u / aspect * COLS).floor() / COLS + 0.5 / COLS;
        let reach = 0.25 + 0.55 * (-((col - peak).powi(2)) / 0.06).exp() + 0.08 * (col * 23.0 + phase).sin();
        ramp.sample(clip01((v - (1.0 - reach)) / reach).powf(1.4))
    })
}

/// Large arcs: a glowing circle edge with a halo behind it; brightness is mapped onto the palette ramp.
fn glow(c: Ctx) -> Field {
    let (ramp, aspect) = (c.ramp, c.aspect);
    let arcs = [(aspect + 0.27, -0.15, 0.95, 0.035), (aspect - 0.03, 1.30, 0.55, 0.03)];
    Box::new(move |_, u, v| {
        let mut val = 0.0;
        for &(cx, cy, r, w) in &arcs {
            let d = ((u - cx).powi(2) + (v - cy).powi(2)).sqrt();
            let ang = ((v - cy).atan2(u - cx) / PI + 1.0) / 2.0;
            let rim = (-((d - r).powi(2)) / (2.0 * w * w)).exp();
            let halo = if d > r { (-(d - r).max(0.0) / 0.25).exp() * 0.35 } else { 0.0 };
            let inner = if d < r { (-(r - d) / 0.12).exp() * 0.5 } else { 0.0 };
            val += (rim * 1.4 + halo + inner) * (0.7 + 0.3 * ang);
        }
        ramp.sample(1.0 - (-1.6 * val).exp())
    })
}

/// Domain warping fbm(p + fbm(p + fbm(p))): marbled, silky streaks.
fn warp(c: Ctx) -> Field {
    let s = c.rng.integers(0, 10_000);
    let o = [c.rng.uniform(0.0, 40.0), c.rng.uniform(0.0, 40.0)];
    let ramp = c.ramp;
    let accent = ramp.lin[1.max(ramp.lin.len() / 2)];
    Box::new(move |_, u, v| {
        let (px, py) = (u * 0.55 + o[0], v * 0.55 + o[1]);
        let (qx, qy) = (fbm(px, py, 4, s), fbm(px + 5.2, py + 1.3, 4, s + 1));
        let rx = fbm(px + 2.6 * qx + 1.7, py + 2.6 * qy + 9.2, 4, s + 2);
        let ry = fbm(px + 2.6 * qx + 8.3, py + 2.6 * qy + 2.8, 4, s + 3);
        let f = fbm(px + 2.6 * rx, py + 2.6 * ry, 5, s + 4);
        // contrast stretch: fbm clusters around 0.5
        let t = smoothstep(0.3, 0.72, f);
        let out = scale(ramp.sample(t.powf(0.9)), 1.25);
        // a second color where the warp was strong, so streaks have two hues instead of one ramp
        add(out, scale(accent, smoothstep(0.55, 0.9, rx) * (1.0 - t) * 0.5))
    })
}

/// Layers of hills with soft, ragged edges; light at the back, dark at the front.
fn dunes(c: Ctx) -> Field {
    const LAYERS: usize = 5;
    let s = c.rng.integers(0, 10_000);
    let levels: Vec<f64> = (0..LAYERS).map(|i| 0.36 + i as f64 * 0.13 + c.rng.uniform(-0.03, 0.03)).collect();
    let ramp = c.ramp;
    Box::new(move |_, u, v| {
        // sky: from the lightest color at the horizon to the background at the top
        let mut out = ramp.sample(0.3 + 0.6 * smoothstep(0.0, 0.55, v));
        // a slight warp of the whole image so edges aren't clean sine waves
        let wu = u + 0.08 * (fbm(u * 1.5, v * 1.5, 3, s) - 0.5);
        for (i, &level) in levels.iter().enumerate() {
            let (fi, ii) = (i as f64, i as i64);
            let edge = level + 0.55 * (fbm(wu * (0.9 + fi * 0.35) + fi * 13.1, fi * 7.7, 4, s + 10 + ii) - 0.5);
            let blur = 0.006 + 0.03 * fbm(wu * 2.3 + fi * 3.1, 0.5 + fi, 3, s + 30 + ii);
            let m = smoothstep(edge - blur, edge + blur, v);
            // layer color: back layers lighter; light crest, darker further down (a shift along the ramp)
            let tl = 1.0 - (fi + 0.5) / LAYERS as f64 * 0.9;
            let crest = ((v - edge) * 2.2).clamp(0.0, 0.75);
            let col = ramp.sample(clip01(tl - 0.35 * crest));
            out = add(scale(out, 1.0 - m), scale(col, m));
        }
        out
    })
}

/// Grainy sky: soft, wavy bands of brightness (light at the top) with fbm-warped borders and a
/// linear-burn darkening where bands meet. Best with coarser grain.
fn grainy(c: Ctx) -> Field {
    let s = c.rng.integers(0, 10_000);
    let o = [c.rng.uniform(0.0, 30.0), c.rng.uniform(0.0, 30.0)];
    let ramp = c.ramp;
    // three borders, each higher, more finely wavy and sharper
    const BANDS: [(f64, f64, f64, f64); 3] = [(0.72, 1.6, 0.45, 0.30), (0.48, 2.6, 0.35, 0.22), (0.24, 3.8, 0.28, 0.15)];
    Box::new(move |_, u, v| {
        let (x, y) = (u * 0.9 + o[0], v * 0.9 + o[1]);
        let (qx, qy) = (fbm(x * 0.5, y * 0.5, 4, s), fbm(x * 0.5 + 4.1, y * 0.5 + 7.7, 4, s + 1));
        let full = fbm(x + 1.6 * qx, y + 1.6 * qy, 4, s + 2);
        let (sx, sy) = (u + (full - 0.5) * 0.9, v + (full - 0.5) * 0.6);
        let mut m = [0.0; 3];
        for (i, &(level, freq, amp, soft)) in BANDS.iter().enumerate() {
            let fi = i as f64;
            let edge = level + amp * (vnoise(sx * freq + fi * 11.3, sy * 0.4 + fi * 3.7, s + 10 + i as i64) - 0.5);
            m[i] = smoothstep(edge + soft, edge - soft, sy); // 1 above the border
        }
        let t = 0.12 + 0.3 * m[0] + 0.28 * m[0] * m[1] + 0.3 * m[0] * m[1] * m[2];
        let col = ramp.sample(t);
        // linear burn at the seams: a darker band where the mask is mid-transition
        let burn = m.iter().map(|m| 4.0 * m * (1.0 - m)).sum::<f64>() / 3.0;
        let darker = ramp.sample(clip01(t - 0.25));
        let k = 0.55 * burn;
        std::array::from_fn(|ch| col[ch] + ((col[ch] + darker[ch] - 1.0).max(0.0) - col[ch]) * k)
    })
}

/// Linear burn (a + b - 1) in sRGB with opacity op: two light blues give a deep cobalt.
fn burn(base: Rgb, blend: Rgb, op: f64) -> Rgb {
    std::array::from_fn(|i| (base[i] + blend[i] - 1.0).max(0.0) * op + base[i] * (1.0 - op))
}

/// Cloudy sky: three blurred layers over a deep linear-burn background.
/// Grain is a tiny coordinate jitter, so it only shows on cloud edges, not on flat areas.
fn sky(c: Ctx) -> Field {
    let srgb: Vec<Rgb> = c.ramp.lin.iter().rev().map(|&x| map3(x, to_srgb)).collect(); // lightest first
    let n = srgb.len();
    let (high, main, mid, low) = (srgb[0], srgb[1.min(n - 1)], srgb[2.min(n - 1)], srgb[n - 1]);
    let s = c.rng.integers(0, 10_000);
    let t = match c.sky.time {
        Some(time) => time * 0.85,
        None => c.rng.uniform(0.0, 30.0), // animation "time" = a different frame
    };
    let nscale = 0.35 + c.sky.scale.unwrap_or(45.0) / 100.0 * 1.15;
    let warp = c.sky.warp.unwrap_or(50.0) / 100.0 * 0.47;
    let wind = c.sky.wind.unwrap_or(40.0) / 100.0 * 0.36;
    let (zoom, blur, grain) = (0.3971, 1.2673 * 1.5, 0.014);
    let d: Vec<Vec<f64>> =
        (0..3).map(|_| c.rng.random_vec(c.n_pixels).into_iter().map(|r| (r - 0.5) * grain).collect()).collect();
    let aspect = c.aspect;

    Box::new(move |idx, u, v| {
        let (d0, d1, d3) = (d[0][idx], d[1][idx], d[2][idx]);
        // coordinates: screen center at 0.5, y pointing up
        let x = (u - aspect / 2.0) / (2.0 * zoom) + 0.5;
        let y = (0.5 - v) / (2.0 * zoom) + 0.5;
        let (x, y) = (
            x + 2.0 * warp * perlin(x * nscale, y * nscale + 74.86 + t * 0.3, s),
            y + warp * perlin(x * nscale + 203.91, y * nscale + 10.0 + t * 0.3, s + 1),
        );
        let na = perlin(x * 18.0 + 344.9, y * 18.0 + t * 0.3, s + 2)
            + 0.5 * perlin(x * 39.6 + 723.9, y * 39.6 + t * 0.4, s + 3);
        let (x, y) = (x + na * 0.02 + d0, y + na * 0.02 - 0.09 + d0);

        let (px, py) = (x * nscale, y * nscale);
        let q = fbm_sq(px * 0.5 + wind * t, py * 0.5 + wind * t, s + 10);
        let rx = fbm_sq(px + q + 0.3 + 0.15 * t, py + q + 9.2 + 0.15 * t, s + 20);
        let ry = fbm_sq(px + q + 8.3 + 0.126 * t, py + q + 0.8 + 0.126 * t, s + 30);
        let fv = fbm_sq(px + rx - q, py + ry - q, s + 40);
        let full = ((fv + 0.6 * fv * fv + 0.7 * fv + 0.5) * 0.5).max(0.0).powf(0.55) * 0.912 - 0.5;

        #[allow(clippy::too_many_arguments)]
        let layer = |k: f64, dx: f64, dy: f64, freq: f64, off: (f64, f64), amp: f64, soft: f64, level: f64, gain: f64,
                     power: f64, seed_off: i64| {
            let (lx, ly) = (x + full * k + dx, y + full * k + dy + dx);
            let edge = vnoise_sq(lx * freq + off.0, ly * freq + off.1, s + seed_off) * amp;
            smoothstep(edge - soft * blur, edge + soft * blur, (ly - level) * gain + 0.5).powf(power)
        };
        let la = layer(1.2, d0, 0.025, 2.0, (0.0, t * 0.5), 3.0, 1.2, 0.5, 5.0, 0.8, 50);
        let lb = layer(0.85, d1, 0.025, 4.0, (293.0, t), 2.8, 0.9, 0.6, 5.0, 0.9, 60);
        let lc = layer(1.1, d3, 0.0, 6.0, (153.0, t * 1.2), 2.6, 0.7, 0.9, 6.0, 1.0, 70);

        let col = burn(main, low, 1.0 - la);
        let col = burn(col, add(main, scale(sub(mid, main), 1.0 - lb)), la);
        let top = add(main, scale(sub(high, main), 1.0 - lc));
        let col = add(col, scale(sub(top, col), la * lb));
        map3(col, |x| to_linear(clip01(x)))
    })
}

/// Aurora: curtains of light around a wavy spine; sharp bottom edge, long fade upward, rays.
fn aurora(c: Ctx) -> Field {
    let s = c.rng.integers(0, 10_000);
    let curtains: Vec<(f64, f64, f64, f64, f64)> = [(0.6, 0.07, 1.0), (0.45, 0.05, 0.5)]
        .iter()
        .map(|&(base_y, amp, gain)| (base_y, amp, gain, c.rng.uniform(0.0, 6.0), c.rng.uniform(0.0, 6.0)))
        .collect();
    let (ramp, aspect) = (c.ramp, c.aspect);
    Box::new(move |_, u, v| {
        // background: night, slightly brighter near the horizon
        let mut out = scale(ramp.lin[0], 1.0 + 0.6 * smoothstep(0.4, 1.0, v));
        let x = u / aspect;
        for (k, &(base_y, amp, gain, a, b)) in curtains.iter().enumerate() {
            let (kf, ki) = (k as f64, k as i64);
            let spine = base_y
                + amp * (0.78 * (x * 6.5 + a).sin() + 0.22 * (x * 15.0 + b).sin())
                + 0.3 * (fbm(x * 2.2 + kf * 5.0, 0.3, 3, s + ki) - 0.5);
            let d = v - spine; // >0 below the spine (sharp), <0 above it (soft)
            let sig: f64 = if d > 0.0 { 0.03 } else { 0.15 };
            let beam = (-(d * d) / (2.0 * sig * sig)).exp();
            // rays: vertically stretched noise with contrast, distinct bands instead of haze
            let rays = 0.25 + 0.85 * smoothstep(0.25, 0.85, fbm(x * 14.0 + d * 2.5 + kf * 9.0, v * 0.5, 4, s + 20 + ki));
            let fade = smoothstep(-0.05, 0.3, x) * smoothstep(1.05, 0.7, x) * 0.7 + 0.3;
            // color: lightest palette color at the bottom edge, deeper colors higher up
            let t = 1.0 - (-d).max(0.0) / 0.3 * 0.7;
            out = add(out, scale(ramp.sample(t), beam * rays * fade * gain * 0.9));
        }
        out
    })
}

/// Concentric rings from a corner: stepped brightness and thin glowing edges.
fn rings(c: Ctx) -> Field {
    let cx = c.aspect * c.rng.uniform(0.88, 1.0);
    let cy = c.rng.uniform(1.05, 1.2);
    let ramp = c.ramp;
    const R0: f64 = 0.22;
    const STEP: f64 = 0.16;
    const N: f64 = 8.0;
    Box::new(move |_, u, v| {
        let d = ((u - cx).powi(2) + (v - cy).powi(2)).sqrt();
        let k = ((d - R0) / STEP).floor().clamp(-1.0, N);
        let mut val = (if k < 0.0 { 1.0 } else { 1.0 - (k + 1.0) / (N + 1.0) }).powf(1.4) * 0.75;
        for i in 0..=N as usize {
            let fi = i as f64;
            let dr2 = (d - (R0 + fi * STEP)).powi(2);
            let line = (-dr2 / (2.0 * 0.004f64.powi(2))).exp() * 0.6 + (-dr2 / (2.0 * 0.03f64.powi(2))).exp() * 0.1;
            val += line * (1.0 - fi / (N + 2.0));
        }
        ramp.sample(clip01(val))
    })
}

/// Blurred vertical columns of light of varying width, rising from the bottom.
fn bars(c: Ctx) -> Field {
    let cols: Vec<(f64, f64, f64, f64)> = (0..14)
        .map(|_| {
            let cx = c.rng.uniform(0.3, 1.05);
            let w = c.rng.uniform(0.012, 0.06);
            let h = c.rng.uniform(0.25, 0.85) * (0.4 + 0.6 * cx); // taller towards the right
            let g = c.rng.uniform(0.4, 0.9);
            (cx, 2.0 * w * w, h, g)
        })
        .collect();
    let (ramp, aspect) = (c.ramp, c.aspect);
    Box::new(move |_, u, v| {
        let x = u / aspect;
        let mut val = 0.0;
        for &(cx, w2, h, g) in &cols {
            val += (-((x - cx).powi(2)) / w2).exp() * (-(1.0 - v).max(0.0) / h).exp() * g;
        }
        val += 0.25 * smoothstep(0.6, 1.0, v) * smoothstep(0.2, 1.0, x); // glow at the bottom
        ramp.sample(1.0 - (-1.5 * val).exp())
    })
}

pub struct Kind {
    pub name: &'static str,
    build: fn(Ctx) -> Field,
    pub tonemap: &'static str,
    pub note: &'static str,
    pub stars: f64,
    /// field with grain built into its geometry; computed at full resolution, no scaling
    full_res: bool,
}

const fn kind(name: &'static str, build: fn(Ctx) -> Field, tonemap: &'static str, note: &'static str) -> Kind {
    Kind { name, build, tonemap, note, stars: 0.0, full_res: false }
}

pub const TYPES: &[Kind] = &[
    kind("mesh", mesh, "none", "mesh gradient: color points smoothly blended in OKLab"),
    Kind {
        full_res: true,
        ..kind("sky", sky, "none", "cloudy sky: deep linear-burn background, grain on cloud edges")
    },
    kind("grainy", grainy, "none", "grainy sky: soft wavy bands, best with grain_size 2"),
    kind("flow", flow, "none", "swirling clouds across the whole palette ramp"),
    kind("prism", prism, "none", "stepped vertical bands glowing from the bottom"),
    kind("glow", glow, "none", "large arcs with a glowing edge on the right"),
    kind("warp", warp, "none", "fbm domain warping: marble / silk"),
    kind("dunes", dunes, "none", "layers of hills with soft, ragged edges"),
    Kind { stars: 0.6, ..kind("aurora", aurora, "aces", "aurora: curtains of light with rays and stars") },
    kind("rings", rings, "none", "concentric rings from the bottom-right corner"),
    kind("bars", bars, "none", "blurred columns of light rising from the bottom"),
];

pub fn type_names() -> Vec<&'static str> {
    TYPES.iter().map(|k| k.name).collect()
}

// --- post-processing ----------------------------------------------------------------------------

/// Float image: pixels row by row, N channels.
struct Plane<const N: usize> {
    w: usize,
    h: usize,
    px: Vec<[f64; N]>,
}

fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Bicubic resampling coefficients, exactly as in PIL (Resample.c, precompute_coeffs).
fn coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f64>)> {
    let sc = in_size as f64 / out_size as f64;
    let filterscale = sc.max(1.0);
    let support = 2.0 * filterscale;
    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * sc;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize;
            let mut k: Vec<f64> = (xmin..xmax).map(|x| bicubic((x as f64 - center + 0.5) / filterscale)).collect();
            let ww: f64 = k.iter().sum();
            if ww != 0.0 {
                k.iter_mut().for_each(|v| *v /= ww);
            }
            (xmin, k)
        })
        .collect()
}

/// Bicubic resize like PIL's `Image.resize(..., BICUBIC)`: horizontal pass first, then vertical.
fn resize<const N: usize>(src: Plane<N>, w: usize, h: usize) -> Plane<N> {
    let mut img = src;
    if img.w != w {
        let cs = coeffs(img.w, w);
        let mut px = vec![[0.0; N]; w * img.h];
        px.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let src = &img.px[y * img.w..(y + 1) * img.w];
            for (out, (xmin, k)) in row.iter_mut().zip(&cs) {
                for (j, kv) in k.iter().enumerate() {
                    for ch in 0..N {
                        out[ch] += src[xmin + j][ch] * kv;
                    }
                }
            }
        });
        img = Plane { w, h: img.h, px };
    }
    if img.h != h {
        let cs = coeffs(img.h, h);
        let mut px = vec![[0.0; N]; w * h];
        px.par_chunks_mut(w).zip(&cs).for_each(|(row, (ymin, k))| {
            for (j, kv) in k.iter().enumerate() {
                let src = &img.px[(ymin + j) * w..(ymin + j + 1) * w];
                for (out, s) in row.iter_mut().zip(src) {
                    for ch in 0..N {
                        out[ch] += s[ch] * kv;
                    }
                }
            }
        });
        img = Plane { w, h, px };
    }
    img
}

/// Gaussian blur of a single-channel layer (stands in for PIL's ImageFilter.GaussianBlur).
fn gaussian_blur(layer: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    if sigma < 0.05 {
        return layer.to_vec();
    }
    let r = (sigma * 3.0).ceil() as isize;
    let mut k: Vec<f64> = (-r..=r).map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f64 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    let pass = |src: &[f64], len: usize, stride: usize, lines: usize, step: usize| -> Vec<f64> {
        let mut out = vec![0.0; src.len()];
        for line in 0..lines {
            for i in 0..len {
                let mut acc = 0.0;
                for (j, kv) in k.iter().enumerate() {
                    let p = (i as isize + j as isize - r).clamp(0, len as isize - 1) as usize;
                    acc += src[line * step + p * stride] * kv;
                }
                out[line * step + i * stride] = acc;
            }
        }
        out
    };
    let tmp = pass(layer, w, 1, h, w);
    pass(&tmp, h, w, w, 1)
}

/// Shading under content: 'left' (menu/text on the left), 'bottom' (rows of tiles at the bottom), 'center'.
fn shade_mask(kind: &str, w: usize, h: usize, strength: f64) -> Option<Vec<f64>> {
    if kind.is_empty() || kind == "none" {
        return None;
    }
    let (wd, hd) = ((w.max(2) - 1) as f64, (h.max(2) - 1) as f64);
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let (xs, ys) = (x as f64 / wd, y as f64 / hd);
            let m = match kind {
                "left" => 1.0 - smoothstep(0.05, 0.7, xs),
                "bottom" => smoothstep(0.35, 1.0, ys),
                _ => 1.0 - smoothstep(0.1, 0.75, (((xs - 0.5) * w as f64 / h as f64).powi(2) + (ys - 0.5).powi(2)).sqrt()),
            };
            out.push(1.0 - strength * m);
        }
    }
    Some(out)
}

/// Sparse 1-2 px dots, more of them near the top and in dark areas.
fn add_stars(img: &mut Plane<3>, density: f64, rng: &mut Rng) {
    let (w, h) = (img.w, img.h);
    let n = (density * (w * h) as f64 / 2500.0) as usize;
    if n == 0 {
        return;
    }
    let ys: Vec<usize> = rng.random_vec(n).into_iter().map(|r| (r.powf(1.8) * h as f64) as usize).collect();
    let xs: Vec<usize> = rng.integers_vec(0, w as i64, n).into_iter().map(|x| x as usize).collect();
    let mut b: Vec<f64> = rng.random_vec(n).into_iter().map(|r| r.powi(3) * 0.8 + 0.15).collect();
    for i in 0..n {
        b[i] *= clip01(1.0 - dot(img.px[ys[i] * w + xs[i]], LUMA) * 2.2);
    }
    for (dy, dx, f) in [(0, 0, 1.0), (0, 1, 0.35), (1, 0, 0.35), (0, -1, 0.35), (-1, 0, 0.35)] {
        // like numpy fancy-index assignment: values are computed from the image before this step
        let writes: Vec<(usize, Rgb)> = (0..n)
            .filter(|&i| f == 1.0 || b[i] > 0.55) // cross shape only for bright stars
            .map(|i| {
                let yy = (ys[i] as isize + dy).clamp(0, h as isize - 1) as usize;
                let xx = (xs[i] as isize + dx).clamp(0, w as isize - 1) as usize;
                let p = yy * w + xx;
                (p, map3(img.px[p], |c| c.max(b[i] * f)))
            })
            .collect();
        for (p, c) in writes {
            img.px[p] = c;
        }
    }
}

/// Line like PIL's `ImageDraw.line` at width 1 (Bresenham over truncated coordinates).
fn draw_line(layer: &mut [f64], w: usize, h: usize, (x0, y0): (i64, i64), (x1, y1): (i64, i64), val: f64) {
    let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
    let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
    let (mut x, mut y, mut err) = (x0, y0, dx + dy);
    loop {
        if (0..w as i64).contains(&x) && (0..h as i64).contains(&y) {
            layer[y as usize * w + x as usize] = val;
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// Rain streaks: thin, slightly slanted lines, brighter on dark backgrounds; blurred so they don't look harsh.
fn add_rain(img: &mut Plane<3>, density: f64, rng: &mut Rng) {
    let (w, h) = (img.w, img.h);
    let (wf, hf) = (w as f64, h as f64);
    let dx = 12f64.to_radians().tan();
    let mut layer = vec![0.0; w * h];
    for _ in 0..(density * wf * hf / 4000.0) as usize {
        let (x, y) = (rng.uniform(-0.1 * wf, wf), rng.uniform(-0.1 * hf, hf));
        let length = rng.uniform(0.02, 0.06) * hf;
        let fill = rng.uniform(40.0, 140.0).trunc();
        let width = if rng.random() < 0.8 { 1 } else { 2 };
        let (a, b) = ((x as i64, y as i64), ((x + dx * length) as i64, (y + length) as i64));
        draw_line(&mut layer, w, h, a, b, fill);
        if width == 2 {
            draw_line(&mut layer, w, h, (a.0 + 1, a.1), (b.0 + 1, b.1), fill);
        }
    }
    let m = gaussian_blur(&layer, w, h, 0.7);
    for (p, m) in img.px.iter_mut().zip(m) {
        *p = map3(*p, |c| c + (1.0 - c) * (m / 255.0) * 0.35); // screen-like: brighter, never blows out
    }
}

/// Filled ellipse in the box [x0, x1] x [y0, y1] (like `ImageDraw.ellipse`, coordinates truncated to int).
fn draw_disc(layer: &mut [f64], w: usize, h: usize, x: f64, y: f64, r: f64, val: f64) {
    let (x0, y0, x1, y1) = ((x - r) as i64, (y - r) as i64, (x + r) as i64, (y + r) as i64);
    if x0 == x1 || y0 == y1 {
        return; // PIL draws nothing for a zero-width box
    }
    let (cx, cy) = ((x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0);
    // +0.25 px: with this margin the pixel count matches PIL for almost every size
    let (a, b) = ((x1 - x0) as f64 / 2.0 + 0.25, (y1 - y0) as f64 / 2.0 + 0.25);
    for py in y0.max(0)..=y1.min(h as i64 - 1) {
        for px in x0.max(0)..=x1.min(w as i64 - 1) {
            if ((px as f64 - cx) / a).powi(2) + ((py as f64 - cy) / b).powi(2) <= 1.0 {
                layer[py as usize * w + px as usize] = val;
            }
        }
    }
}

/// Snowflakes on two planes: small sharp ones (far away) and large blurred ones (close to the lens).
fn add_snow(img: &mut Plane<3>, density: f64, rng: &mut Rng) {
    let (w, h) = (img.w, img.h);
    let (wf, hf) = (w as f64, h as f64);
    for (count, (rmin, rmax), blur, alpha) in [(1.0, (0.6, 1.8), 0.4, 0.85), (0.08, (4.0, 9.0), 3.5, 0.45)] {
        let mut layer = vec![0.0; w * h];
        for _ in 0..(density * count * wf * hf / 3000.0) as usize {
            let (x, y) = (rng.uniform(0.0, wf), rng.uniform(0.0, hf));
            let r = rng.uniform(rmin, rmax) * hf / 1080.0;
            let fill = rng.uniform(120.0, 255.0).trunc();
            draw_disc(&mut layer, w, h, x, y, r, fill);
        }
        let m = gaussian_blur(&layer, w, h, blur * hf / 1080.0);
        for (p, m) in img.px.iter_mut().zip(m) {
            *p = map3(*p, |c| c + (1.0 - c) * (m / 255.0) * alpha);
        }
    }
}

/// Overlay noise: a grey noise layer (triangular distribution) in overlay mode, opacity noise/200.
fn overlay_noise(img: &mut Plane<3>, noise: f64, rng: &mut Rng) {
    let n = img.px.len();
    let (a, b) = (rng.random_vec(n), rng.random_vec(n));
    for (i, p) in img.px.iter_mut().enumerate() {
        let nz = (a[i] + b[i]) / 2.0;
        *p = map3(*p, |c| {
            let ov = if c < 0.5 { 2.0 * c * nz } else { 1.0 - 2.0 * (1.0 - c) * (1.0 - nz) };
            c + (ov - c) * (noise / 200.0)
        });
    }
}

const BAYER8: [[u8; 8]; 8] = [
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
];

pub const DITHERS: [&str; 3] = ["grain", "bayer", "none"];
pub const SHADES: [&str; 4] = ["none", "left", "bottom", "center"];

fn dither(img: &mut Plane<3>, mode: &str, grain: f64, grain_size: f64, rng: &mut Rng) {
    let (w, h) = (img.w, img.h);
    match mode {
        "bayer" => {
            // ordered 8x8 dithering + quantization to a few dozen levels: a visible "retro" pattern
            const LEVELS: f64 = 31.0;
            for (i, p) in img.px.iter_mut().enumerate() {
                let t = BAYER8[(i / w) % 8][(i % w) % 8] as f64 / 64.0 - 0.5;
                *p = map3(*p, |c| (c * LEVELS + 0.5 + t).floor() / LEVELS);
            }
        }
        "grain" => {
            // film grain: monochrome noise, stronger in bright areas; grain_size > 1 = coarser grain
            let gw = ((w as f64 / grain_size).round_ties_even() as usize).max(1);
            let gh = ((h as f64 / grain_size).round_ties_even() as usize).max(1);
            let noise = Plane { w: gw, h: gh, px: rng.normal_vec(gw * gh).into_iter().map(|v| [v]).collect() };
            let noise = resize(noise, w, h);
            let k = if grain_size > 1.0 { grain_size.sqrt() } else { 1.0 };
            for (p, nz) in img.px.iter_mut().zip(&noise.px) {
                let lum = (p[0] + p[1] + p[2]) / 3.0;
                let add = nz[0] * k * grain * (0.35 + 0.65 * lum);
                *p = map3(*p, |c| c + add);
            }
        }
        _ => {}
    }
    // triangular 1 LSB dither, always on: removes banding when writing 8-bit output
    let n = w * h * 3;
    let (a, b) = (rng.random_vec(n), rng.random_vec(n));
    for (i, p) in img.px.iter_mut().enumerate() {
        for ch in 0..3 {
            let j = i * 3 + ch;
            p[ch] = clip01(p[ch] + (a[j] - b[j]) / 255.0);
        }
    }
}

/// All settings for one image; defaults match the MCP tool.
#[derive(Clone, Debug)]
pub struct Opts {
    pub kind: String,
    pub palette: Option<String>,
    pub colors: Option<Vec<String>>,
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    pub dither: String,
    pub grain: f64,
    pub grain_size: f64,
    pub tonemap: Option<String>,
    pub exposure: f64,
    pub shade: String,
    pub shade_strength: f64,
    pub stars: Option<f64>,
    pub rotate: i64,
    pub noise: f64,
    pub rain: f64,
    pub snow: f64,
    pub sky: SkyParams,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            kind: "mesh".into(),
            palette: None,
            colors: None,
            width: 1920,
            height: 1080,
            seed: 1,
            dither: "grain".into(),
            grain: 0.035,
            grain_size: 1.0,
            tonemap: None,
            exposure: 1.0,
            shade: "none".into(),
            shade_strength: 0.55,
            stars: None,
            rotate: 0,
            noise: 0.0,
            rain: 0.0,
            snow: 0.0,
            sky: SkyParams::default(),
        }
    }
}

fn check(name: &str, value: &str, allowed: &[&str]) -> Result<(), String> {
    if allowed.contains(&value) { Ok(()) } else { Err(format!("{name}: {}", allowed.join(" | "))) }
}

/// Renders an 8-bit RGB image.
pub fn render(o: &Opts) -> Result<RgbImage, String> {
    let kind = TYPES
        .iter()
        .find(|k| k.name == o.kind)
        .ok_or_else(|| format!("unknown type {:?}; available: {}", o.kind, type_names().join(", ")))?;
    let (width, height) = (o.width, o.height);
    if !(16..=8192).contains(&width) || !(16..=8192).contains(&height) {
        return Err("size must be 16..8192 px".into());
    }
    let ramp = Ramp::new(&resolve_stops(o.palette.as_deref(), o.colors.as_deref())?)?;
    let tm = o.tonemap.as_deref().unwrap_or(kind.tonemap);
    check("tonemap", tm, &TONEMAPS)?;
    check("shade", &o.shade, &SHADES)?;
    check("dither", &o.dither, &DITHERS)?;
    if o.grain_size.is_nan() || o.grain_size <= 0.0 {
        return Err("grain_size must be > 0".into());
    }
    let mut rng = Rng::new(o.seed);

    let fw = if kind.full_res { width } else { width.min(FIELD_MAX_W) };
    let fh = ((height as f64 * fw as f64 / width as f64).round_ties_even() as usize).max(16);
    let rot = o.rotate.div_euclid(90).rem_euclid(4);
    let (gw, gh) = if rot % 2 == 1 { (fh, fw) } else { (fw, fh) }; // at 90/270 the field is computed with swapped dimensions
    let field_fn = (kind.build)(Ctx {
        aspect: gw as f64 / gh as f64,
        ramp,
        rng: &mut rng,
        n_pixels: gw * gh,
        sky: o.sky,
    });
    let mut grid = vec![[0.0; 3]; gw * gh];
    grid.par_chunks_mut(gw).enumerate().for_each(|(y, row)| {
        let v = (y as f64 + 0.5) / gh as f64;
        for (x, p) in row.iter_mut().enumerate() {
            *p = field_fn(y * gw + x, (x as f64 + 0.5) / gh as f64, v);
        }
    });
    drop(field_fn);

    // np.rot90(k=rot): counter-clockwise rotation, result is fh x fw
    let at = |i: usize, j: usize| -> Rgb {
        match rot {
            1 => grid[j * gw + (gw - 1 - i)],
            2 => grid[(gh - 1 - i) * gw + (gw - 1 - j)],
            3 => grid[(gh - 1 - j) * gw + i],
            _ => grid[i * gw + j],
        }
    };
    let tmf = tonemap_fn(tm);
    let mut px = Vec::with_capacity(fw * fh);
    for i in 0..fh {
        for j in 0..fw {
            px.push(map3(at(i, j), |c| to_srgb(tmf(c * o.exposure))));
        }
    }
    drop(grid);
    let mut img = resize(Plane { w: fw, h: fh, px }, width, height);
    img.px.iter_mut().for_each(|p| *p = map3(*p, clip01));

    if let Some(m) = shade_mask(&o.shade, width, height, o.shade_strength) {
        img.px.iter_mut().zip(m).for_each(|(p, m)| *p = scale(*p, m));
    }
    let dens = o.stars.unwrap_or(kind.stars);
    if dens != 0.0 {
        add_stars(&mut img, dens, &mut rng);
    }
    if o.rain != 0.0 {
        add_rain(&mut img, o.rain, &mut rng);
    }
    if o.snow != 0.0 {
        add_snow(&mut img, o.snow, &mut rng);
    }
    if o.noise != 0.0 {
        overlay_noise(&mut img, o.noise, &mut rng);
    }
    dither(&mut img, &o.dither, o.grain, o.grain_size, &mut rng);

    let bytes = img.px.iter().flat_map(|p| p.map(|c| (c * 255.0 + 0.5) as u8)).collect();
    Ok(RgbImage::from_raw(width as u32, height as u32, bytes).expect("buffer size"))
}

/// Saves a PNG, optionally with the best compression (like PIL's `optimize=True`).
pub fn save_png(img: &RgbImage, path: &std::path::Path, best: bool) -> Result<(), String> {
    use image::codecs::png::{CompressionType, FilterType, PngEncoder};
    use image::ImageEncoder;
    let file = std::fs::File::create(path).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    let comp = if best { CompressionType::Best } else { CompressionType::Default };
    PngEncoder::new_with_quality(std::io::BufWriter::new(file), comp, FilterType::Adaptive)
        .write_image(img.as_raw(), img.width(), img.height(), image::ExtendedColorType::Rgb8)
        .map_err(|e| format!("writing PNG: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pixels from the original numpy version (96x54, seed 1, dither "none"). Guards that a given
    /// seed keeps producing the same image.
    const PROBES: [(u32, u32); 8] = [(3, 5), (40, 10), (90, 50), (50, 27), (10, 45), (70, 33), (25, 20), (85, 5)];
    #[rustfmt::skip]
    const REFERENCE: &[(&str, [[u8; 3]; 8])] = &[
        ("mesh", [[246, 169, 122], [35, 22, 43], [245, 92, 95], [23, 13, 37], [79, 38, 111], [80, 35, 58], [107, 72, 89], [172, 54, 108]]),
        ("sky", [[255, 94, 99], [242, 102, 72], [18, 0, 0], [44, 12, 12], [49, 13, 14], [17, 0, 0], [249, 119, 89], [254, 128, 104]]),
        ("grainy", [[255, 175, 122], [255, 163, 116], [43, 20, 69], [185, 61, 98], [80, 32, 101], [120, 41, 94], [212, 74, 92], [254, 176, 122]]),
        ("flow", [[59, 26, 90], [116, 45, 112], [255, 153, 114], [254, 161, 117], [91, 37, 111], [255, 176, 122], [83, 35, 111], [216, 75, 106]]),
        ("prism", [[18, 10, 33], [17, 10, 34], [255, 129, 107], [17, 9, 34], [130, 48, 111], [118, 45, 112], [18, 10, 34], [18, 9, 34]]),
        ("glow", [[20, 11, 37], [43, 20, 67], [99, 41, 112], [54, 24, 82], [21, 11, 38], [129, 48, 112], [27, 14, 44], [28, 14, 49]]),
        ("warp", [[108, 44, 123], [89, 34, 98], [99, 35, 91], [50, 21, 60], [212, 69, 121], [98, 37, 109], [82, 35, 121], [58, 25, 79]]),
        ("dunes", [[116, 45, 111], [166, 53, 111], [48, 22, 75], [223, 79, 104], [53, 24, 82], [204, 68, 108], [255, 136, 109], [115, 45, 112]]),
        ("aurora", [[7, 2, 22], [10, 3, 27], [11, 5, 33], [205, 139, 102], [11, 4, 31], [76, 39, 40], [139, 26, 60], [7, 3, 21]]),
        ("rings", [[18, 10, 33], [30, 15, 51], [234, 83, 103], [88, 36, 111], [22, 12, 38], [170, 53, 110], [26, 13, 45], [54, 23, 82]]),
        ("bars", [[18, 9, 34], [78, 32, 110], [255, 105, 101], [164, 53, 111], [18, 10, 34], [230, 83, 104], [24, 13, 43], [71, 30, 106]]),
    ];

    #[test]
    fn matches_numpy_version() {
        assert_eq!(REFERENCE.len(), TYPES.len());
        for (name, expected) in REFERENCE {
            let opts = Opts { kind: name.to_string(), width: 96, height: 54, dither: "none".into(), ..Opts::default() };
            let img = render(&opts).unwrap();
            for (&(x, y), want) in PROBES.iter().zip(expected) {
                let got = img.get_pixel(x, y).0;
                let diff = got.iter().zip(want).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
                assert!(diff <= 1, "{name} ({x},{y}): {got:?} != {want:?}");
            }
        }
    }

    #[test]
    fn errors() {
        let bad = |o: Opts| render(&o).unwrap_err();
        assert!(bad(Opts { kind: "nope".into(), ..Opts::default() }).starts_with("unknown type"));
        assert!(bad(Opts { palette: Some("xyz".into()), ..Opts::default() }).starts_with("unknown palette"));
        assert!(bad(Opts { colors: Some(vec!["#fff".into()]), ..Opts::default() }).contains("2 colors"));
        assert!(bad(Opts { width: 8, ..Opts::default() }).starts_with("size"));
    }
}
