//! MCP server (stdio): generates gradient backgrounds with dithering.
//!
//! The protocol is handled by hand (JSON-RPC, one message per line), with no SDK and no async
//! runtime, so the process waiting for calls takes a few MB.

use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use base64::Engine;
use image::RgbImage;
use serde_json::{Map, Value, json};
use washi::core::{
    DEFAULT_PALETTE, DITHERS, MAX_SIZE, MIN_SIZE, Opts, PALETTES, RANGES, SHADES, SkyParams, TONEMAPS, TYPES,
    render, save_png, type_names,
};
use washi::sheet::contact_sheet;

const INSTRUCTIONS: &str = "Generates grainy gradient backgrounds (mesh, sky, aurora, ...) as PNG files. \
Start with preview_types to compare types in a palette (or preview_seeds to compare layouts), then call \
generate_gradient at the target size. Results include the saved file's path and a downscaled preview.";

/// Large buffers (images, noise) go straight to mmap/munmap. The system malloc on macOS keeps freed
/// large blocks in its cache, so after a single 4K render the process held on to ~1 GB. This way
/// memory goes back to the system as soon as a buffer is freed, and the idle server takes a few MB.
struct BigBuffersToMmap;

const BIG: usize = 128 << 10;

fn is_big(size: usize, align: usize) -> bool {
    size >= BIG && align <= 4096
}

unsafe impl std::alloc::GlobalAlloc for BigBuffersToMmap {
    unsafe fn alloc(&self, l: std::alloc::Layout) -> *mut u8 {
        if !is_big(l.size(), l.align()) {
            return unsafe { std::alloc::System.alloc(l) };
        }
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                l.size(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED { std::ptr::null_mut() } else { p.cast() }
    }

    unsafe fn alloc_zeroed(&self, l: std::alloc::Layout) -> *mut u8 {
        if is_big(l.size(), l.align()) {
            unsafe { self.alloc(l) } // pages from mmap are already zeroed
        } else {
            unsafe { std::alloc::System.alloc_zeroed(l) }
        }
    }

    unsafe fn dealloc(&self, p: *mut u8, l: std::alloc::Layout) {
        if is_big(l.size(), l.align()) {
            unsafe { libc::munmap(p.cast(), l.size()) };
        } else {
            unsafe { std::alloc::System.dealloc(p, l) }
        }
    }

    unsafe fn realloc(&self, p: *mut u8, l: std::alloc::Layout, new_size: usize) -> *mut u8 {
        if !is_big(l.size(), l.align()) && !is_big(new_size, l.align()) {
            return unsafe { std::alloc::System.realloc(p, l, new_size) };
        }
        let new_l = unsafe { std::alloc::Layout::from_size_align_unchecked(new_size, l.align()) };
        let np = unsafe { self.alloc(new_l) };
        if !np.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(p, np, l.size().min(new_size));
                self.dealloc(p, l);
            }
        }
        np
    }
}

#[global_allocator]
static ALLOC: BigBuffersToMmap = BigBuffersToMmap;

const PROTOCOL_VERSIONS: &[&str] = &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

/// Protocol version agreed on in `initialize`.
static NEGOTIATED: OnceLock<String> = OnceLock::new();

/// `resource_link` content exists since protocol version 2025-06-18; older clients don't know it.
fn supports_resource_links() -> bool {
    NEGOTIATED.get().is_some_and(|v| v.as_str() >= "2025-06-18")
}

// --- tool schemas ------------------------------------------------------------------------------

fn with_description(mut schema: Value, description: &str) -> Value {
    schema["description"] = description.into();
    schema
}

fn string_enum(values: &[&str], default: &str, description: &str) -> Value {
    with_description(json!({"type": "string", "enum": values, "default": default}), description)
}

/// Number with its allowed range taken from `RANGES`, so the schema matches the server's checks.
fn number(name: &str, default: Option<f64>, description: &str) -> Value {
    let mut s = json!({"type": "number"});
    if let Some(r) = RANGES.iter().find(|r| r.name == name) {
        s[if r.exclusive_min { "exclusiveMinimum" } else { "minimum" }] = r.min.into();
        if let Some(max) = r.max {
            s["maximum"] = max.into();
        }
    }
    if let Some(d) = default {
        s["default"] = d.into();
    }
    with_description(s, description)
}

fn size(default: usize, description: &str) -> Value {
    with_description(
        json!({"type": "integer", "minimum": MIN_SIZE, "maximum": MAX_SIZE, "default": default}),
        description,
    )
}

fn object(properties: Value) -> Value {
    json!({"type": "object", "properties": properties, "additionalProperties": false})
}

fn annotations(title: &str, read_only: bool) -> Value {
    if read_only {
        json!({"title": title, "readOnlyHint": true, "openWorldHint": false})
    } else {
        // writes a PNG; with an explicit output_path it may replace an existing PNG
        json!({"title": title, "readOnlyHint": false, "destructiveHint": true, "idempotentHint": false, "openWorldHint": false})
    }
}

fn tools() -> Value {
    let types = type_names();
    let palettes: Vec<&str> = PALETTES.iter().map(|p| p.name).collect();
    let palette = string_enum(&palettes, DEFAULT_PALETTE, "Named palette (see list_palettes). Ignored when colors is given.");
    let colors = with_description(
        json!({"type": "array", "items": {"type": "string", "pattern": "^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$"}, "minItems": 2}),
        "Custom hex colors instead of a palette, e.g. [\"#03060F\", \"#12B886\", \"#7048E8\"]. \
         3-6 shades of one color family work best; any order (they're sorted dark to light).",
    );
    let dither = string_enum(
        &DITHERS,
        "grain",
        "Final texture: grain (film grain), bayer (visible 8x8 retro pattern), none (only an invisible 1-bit dither against banding).",
    );
    let grain = number("grain", Some(0.035), "Grain strength: 0.02 subtle, 0.06 strong. Used with dither=grain.");
    let shade = string_enum(
        &SHADES,
        "none",
        "Darken part of the image so content on top stays readable: left (text on the left), bottom (rows at the bottom), center.",
    );
    let output_path = with_description(
        json!({"type": "string"}),
        "Where to save the PNG; must end in .png. Relative paths are resolved inside the output folder \
         (~/Pictures/washi, or $WASHI_OUT). Default: a new timestamped file in that folder. \
         An existing PNG at this path is replaced; any other existing file is never overwritten.",
    );
    let seed = with_description(
        json!({"type": "integer", "minimum": 0, "default": 1}),
        "Layout seed. The same seed and settings always produce the same image.",
    );

    json!([
        {
            "name": "generate_gradient",
            "title": "Generate gradient",
            "description": "Render a grainy gradient background and save it as a PNG. Returns the file's path and a downscaled JPEG preview.",
            "inputSchema": object(json!({
                "type": string_enum(&types, "mesh", "Background type. Call list_types for a description of each."),
                "palette": palette,
                "colors": colors,
                "width": size(1920, "Image width in px."),
                "height": size(1080, "Image height in px."),
                "seed": seed,
                "dither": dither,
                "grain": grain,
                "grain_size": number("grain_size", Some(1.0), "Grain size in px: 1 fine, 2-3 coarser."),
                "noise": number("noise", Some(0.0), "Grey overlay noise in %, e.g. 8. Pairs well with dither=none."),
                "shade": shade,
                "shade_strength": number("shade_strength", Some(0.55), "How strongly shade darkens, 0-1."),
                "rotate": with_description(
                    json!({"type": "integer", "enum": [0, 90, 180, 270], "default": 0}),
                    "Rotate the layout, in degrees. 180 puts the light part at the bottom.",
                ),
                "exposure": number(
                    "exposure",
                    Some(1.0),
                    "Brightness multiplier before tonemapping: below 1 is darker (e.g. 0.45 for a night sky), above 1 brighter.",
                ),
                "tonemap": with_description(
                    json!({"type": "string", "enum": TONEMAPS}),
                    "Override the type's default tonemapping (aurora uses aces, everything else none).",
                ),
                "stars": number("stars", None, "Star density, e.g. 0.6. By default only aurora has stars."),
                "rain": number("rain", Some(0.0), "Rain streak density: 0.5 light rain, 1.5 downpour."),
                "snow": number("snow", Some(0.0), "Snowflake density: 0.5 flurries, 1.5 blizzard."),
                "scale": number("scale", None, "type=sky only: cloud scale in % (default 45)."),
                "warp": number("warp", None, "type=sky only: cloud distortion in % (default 50)."),
                "wind": number("wind", None, "type=sky only: wind in % (default 40)."),
                "start_t": with_description(
                    json!({"type": "number"}),
                    "type=sky only: animation time, e.g. 20.75. Without it the frame depends on the seed.",
                ),
                "output_path": output_path,
                "preview": with_description(json!({"type": "boolean", "default": true}), "Include a downscaled JPEG preview in the result."),
            })),
            "annotations": annotations("Generate gradient", false),
        },
        {
            "name": "preview_types",
            "title": "Compare background types",
            "description": "Render a labeled comparison sheet of background types in one palette, to pick a direction before \
                generating at full size. Saves the sheet as a PNG and returns a preview.",
            "inputSchema": object(json!({
                "types": with_description(
                    json!({"type": "array", "items": {"type": "string", "enum": types}, "minItems": 1}),
                    "Types to compare. Default: all of them.",
                ),
                "palette": palette,
                "colors": colors,
                "seed": with_description(seed.clone(), "Layout seed, used for every tile on the sheet."),
                "dither": dither,
                "grain": grain,
                "shade": shade,
                "output_path": output_path,
            })),
            "annotations": annotations("Compare background types", false),
        },
        {
            "name": "preview_seeds",
            "title": "Compare seeds",
            "description": "Render a labeled comparison sheet of one type across several seeds, to pick a layout. \
                Saves the sheet as a PNG and returns a preview.",
            "inputSchema": object(json!({
                "type": string_enum(&types, "mesh", "Background type. Call list_types for a description of each."),
                "seeds": with_description(
                    json!({"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 1}),
                    "Seeds to compare. Default: 1-6.",
                ),
                "palette": palette,
                "colors": colors,
                "dither": dither,
                "grain": grain,
                "shade": shade,
                "output_path": output_path,
            })),
            "annotations": annotations("Compare seeds", false),
        },
        {
            "name": "list_types",
            "title": "List background types",
            "description": "List the background types with a description, default tonemapping and default star density.",
            "inputSchema": object(json!({})),
            "outputSchema": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "tonemap": {"type": "string", "enum": TONEMAPS},
                        "stars": {"type": "number"},
                    },
                    "required": ["description", "tonemap", "stars"],
                },
            },
            "annotations": annotations("List background types", true),
        },
        {
            "name": "list_palettes",
            "title": "List palettes",
            "description": "List the named palettes with their colors (dark to light) and a short description.",
            "inputSchema": object(json!({})),
            "outputSchema": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "colors": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["description", "colors"],
                },
            },
            "annotations": annotations("List palettes", true),
        },
    ])
}

// --- arguments ---------------------------------------------------------------------------------

struct Args<'a>(&'a Map<String, Value>);

impl Args<'_> {
    /// Argument value; missing and null both mean "use the default".
    fn get(&self, key: &str) -> Option<&Value> {
        self.0.get(key).filter(|v| !v.is_null())
    }

    fn num(&self, key: &str) -> Result<Option<f64>, String> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v.as_f64().or_else(|| v.as_str()?.trim().parse().ok()).map(Some).ok_or_else(|| bad(key, "a number")),
        }
    }

    fn f64_or(&self, key: &str, default: f64) -> Result<f64, String> {
        Ok(self.num(key)?.unwrap_or(default))
    }

    fn int_or(&self, key: &str, default: i64) -> Result<i64, String> {
        match self.num(key)? {
            None => Ok(default),
            Some(v) if v.fract() == 0.0 => Ok(v as i64),
            Some(_) => Err(bad(key, "an integer")),
        }
    }

    fn str_opt(&self, key: &str) -> Result<Option<String>, String> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(_) => Err(bad(key, "a string")),
        }
    }

    fn choice(&self, key: &str, allowed: &[&str], default: &str) -> Result<String, String> {
        let v = self.str_opt(key)?.unwrap_or_else(|| default.to_string());
        if allowed.contains(&v.as_str()) { Ok(v) } else { Err(bad(key, &format!("one of: {}", allowed.join(", ")))) }
    }

    fn bool_or(&self, key: &str, default: bool) -> Result<bool, String> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v.as_bool().ok_or_else(|| bad(key, "true/false")),
        }
    }

    /// List: a JSON array, or (leniently) a comma-separated string or an array encoded as a string.
    fn list(&self, key: &str) -> Result<Option<Vec<Value>>, String> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Array(a)) => Ok(Some(a.clone())),
            Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
                Ok(Value::Array(a)) => Ok(Some(a)),
                _ => Ok(Some(s.split(',').map(|p| Value::String(p.trim().to_string())).collect())),
            },
            Some(_) => Err(bad(key, "a list")),
        }
    }

    fn str_list(&self, key: &str) -> Result<Option<Vec<String>>, String> {
        self.list(key)?
            .map(|l| l.into_iter().map(|v| v.as_str().map(str::to_string).ok_or_else(|| bad(key, "a list of strings"))).collect())
            .transpose()
    }
}

fn bad(key: &str, expected: &str) -> String {
    format!("invalid argument {key:?}: expected {expected}")
}

/// Rejects arguments the tool's schema doesn't declare, so typos don't get silently ignored.
fn check_known_args(tool: &Value, args: &Map<String, Value>) -> Result<(), String> {
    let known = tool["inputSchema"]["properties"].as_object().map(|p| p.keys().cloned().collect::<Vec<_>>()).unwrap_or_default();
    match args.keys().find(|k| !known.contains(k)) {
        None => Ok(()),
        Some(k) if known.is_empty() => Err(format!("unknown argument {k:?}; this tool takes no arguments")),
        Some(k) => Err(format!("unknown argument {k:?}; expected one of: {}", known.join(", "))),
    }
}

/// Settings shared by the sheets and generate_gradient.
fn base_opts(a: &Args, kind_default: &str) -> Result<Opts, String> {
    let seed = a.int_or("seed", 1)?;
    if seed < 0 {
        return Err(bad("seed", "a non-negative integer"));
    }
    Ok(Opts {
        kind: a.choice("type", &type_names(), kind_default)?,
        palette: a.str_opt("palette")?,
        colors: a.str_list("colors")?,
        seed: seed as u64,
        dither: a.choice("dither", &DITHERS, "grain")?,
        grain: a.f64_or("grain", 0.035)?,
        shade: a.choice("shade", &SHADES, "none")?,
        ..Opts::default()
    })
}

// --- output files ------------------------------------------------------------------------------

fn out_dir() -> PathBuf {
    match std::env::var_os("WASHI_OUT") {
        Some(p) => PathBuf::from(p),
        None => home().join("Pictures").join("washi"),
    }
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default()
}

fn timestamp() -> String {
    // local time, like time.strftime('%Y%m%d-%H%M%S')
    unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        format!(
            "{:04}{:02}{:02}-{:02}{:02}{:02}",
            tm.tm_year + 1900,
            tm.tm_mon + 1,
            tm.tm_mday,
            tm.tm_hour,
            tm.tm_min,
            tm.tm_sec
        )
    }
}

fn is_png(path: &Path) -> bool {
    let mut sig = [0u8; 8];
    std::fs::File::open(path).and_then(|mut f| f.read_exact(&mut sig)).is_ok() && sig == *b"\x89PNG\r\n\x1a\n"
}

/// Resolves where to save. An explicit path must end in .png and may only replace an existing PNG;
/// a default path is a new timestamped file that never replaces anything.
fn out_path(path: Option<String>, stem: &str) -> Result<PathBuf, String> {
    let dir = out_dir();
    let p = match path {
        Some(raw) => {
            let p = match raw.strip_prefix("~/") {
                Some(rest) => home().join(rest),
                None => PathBuf::from(&raw),
            };
            let p = if p.is_relative() { dir.join(p) } else { p };
            if !p.extension().is_some_and(|e| e.eq_ignore_ascii_case("png")) {
                return Err(format!("output_path must end in .png (got {raw:?})"));
            }
            if p.is_dir() {
                return Err(format!("{} is a folder, not a file", p.display()));
            }
            if p.exists() && !is_png(&p) {
                return Err(format!("{} already exists and isn't a PNG; refusing to overwrite it", p.display()));
            }
            p
        }
        None => {
            let base = format!("{stem}-{}", timestamp());
            let mut p = dir.join(format!("{base}.png"));
            let mut n = 2;
            while p.exists() {
                p = dir.join(format!("{base}-{n}.png"));
                n += 1;
            }
            p
        }
    };
    if let Some(parent) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    Ok(p)
}

// --- results -----------------------------------------------------------------------------------

fn text(s: impl Into<String>) -> Value {
    json!({"type": "text", "text": s.into()})
}

/// `file://` URI for an absolute path, percent-encoding everything but unreserved characters and '/'.
fn file_uri(path: &Path) -> String {
    let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let mut uri = String::from("file://");
    for b in abs.to_string_lossy().bytes() {
        if b.is_ascii_alphanumeric() || b"-._~/".contains(&b) {
            uri.push(b as char);
        } else {
            uri.push_str(&format!("%{b:02X}"));
        }
    }
    uri
}

/// Link to the saved PNG, for clients that support `resource_link` content.
fn png_link(path: &Path) -> Option<Value> {
    supports_resource_links().then(|| {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        json!({"type": "resource_link", "uri": file_uri(path), "name": name, "mimeType": "image/png"})
    })
}

/// Downscaled JPEG preview for the response.
fn preview(img: &RgbImage, max_w: u32) -> Result<Value, String> {
    let small;
    let img = if img.width() > max_w {
        let h = (img.height() as f64 * max_w as f64 / img.width() as f64).round() as u32;
        small = image::imageops::resize(img, max_w, h, image::imageops::FilterType::Lanczos3);
        &small
    } else {
        img
    };
    let mut buf = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 88)
        .encode_image(img)
        .map_err(|e| format!("JPEG preview: {e}"))?;
    Ok(json!({"type": "image", "data": base64::engine::general_purpose::STANDARD.encode(&buf), "mimeType": "image/jpeg"}))
}

/// Text summary, link to the file and (optionally) a preview.
fn saved(msg: String, path: &Path, img: Option<(&RgbImage, u32)>) -> Result<Vec<Value>, String> {
    let mut content = vec![text(msg)];
    content.extend(png_link(path));
    if let Some((img, max_w)) = img {
        content.push(preview(img, max_w)?);
    }
    Ok(content)
}

// --- tools -------------------------------------------------------------------------------------

fn generate_gradient(a: &Args) -> Result<Vec<Value>, String> {
    let t0 = Instant::now();
    let rotate = a.int_or("rotate", 0)?;
    if ![0, 90, 180, 270].contains(&rotate) {
        return Err(bad("rotate", "0, 90, 180 or 270"));
    }
    let size = |key, default| -> Result<usize, String> {
        let v = a.int_or(key, default)?;
        usize::try_from(v).map_err(|_| format!("size must be {MIN_SIZE}..{MAX_SIZE} px"))
    };
    let opts = Opts {
        width: size("width", 1920)?,
        height: size("height", 1080)?,
        grain_size: a.f64_or("grain_size", 1.0)?,
        tonemap: match a.str_opt("tonemap")? {
            Some(t) if !TONEMAPS.contains(&t.as_str()) => return Err(bad("tonemap", "none, reinhard or aces")),
            t => t,
        },
        exposure: a.f64_or("exposure", 1.0)?,
        shade_strength: a.f64_or("shade_strength", 0.55)?,
        stars: a.num("stars")?,
        rotate,
        noise: a.f64_or("noise", 0.0)?,
        rain: a.f64_or("rain", 0.0)?,
        snow: a.f64_or("snow", 0.0)?,
        sky: SkyParams { scale: a.num("scale")?, warp: a.num("warp")?, wind: a.num("wind")?, time: a.num("start_t")? },
        ..base_opts(a, "mesh")?
    };
    let path = out_path(a.str_opt("output_path")?, &opts.kind)?;
    let img = render(&opts)?;
    save_png(&img, &path, true)?;
    let msg = format!("Saved {} ({}x{}, {:.1}s)", path.display(), opts.width, opts.height, t0.elapsed().as_secs_f64());
    saved(msg, &path, a.bool_or("preview", true)?.then_some((&img, 768)))
}

fn sheet_result(variants: Vec<(String, Opts)>, path: &Path) -> Result<Vec<Value>, String> {
    let sheet = contact_sheet(&variants)?;
    save_png(&sheet, path, false)?;
    saved(format!("Sheet: {}", path.display()), path, Some((&sheet, 1200)))
}

fn preview_types(a: &Args) -> Result<Vec<Value>, String> {
    let base = base_opts(a, "mesh")?;
    let names = type_names();
    let types = match a.str_list("types")? {
        Some(t) if !t.is_empty() => {
            if let Some(bad_t) = t.iter().find(|t| !names.contains(&t.as_str())) {
                return Err(format!("unknown type {bad_t:?}; available: {}", names.join(", ")));
            }
            t
        }
        _ => names.iter().map(|s| s.to_string()).collect(),
    };
    let path = out_path(a.str_opt("output_path")?, "sheet")?;
    let variants = types.into_iter().map(|t| (t.clone(), Opts { kind: t, ..base.clone() })).collect();
    sheet_result(variants, &path)
}

fn preview_seeds(a: &Args) -> Result<Vec<Value>, String> {
    let base = base_opts(a, "mesh")?;
    let seeds: Vec<u64> = match a.list("seeds")? {
        Some(l) if !l.is_empty() => l
            .iter()
            .map(|v| {
                v.as_u64()
                    .or_else(|| v.as_f64().filter(|f| f.fract() == 0.0 && *f >= 0.0).map(|f| f as u64))
                    .or_else(|| v.as_str()?.trim().parse().ok())
                    .ok_or_else(|| bad("seeds", "a list of non-negative integers"))
            })
            .collect::<Result<_, _>>()?,
        _ => vec![1, 2, 3, 4, 5, 6],
    };
    let path = out_path(a.str_opt("output_path")?, &format!("{}-seeds", base.kind))?;
    let variants = seeds.iter().map(|&s| (format!("{} · seed {s}", base.kind), Opts { seed: s, ..base.clone() })).collect();
    sheet_result(variants, &path)
}

fn structured(v: Value) -> Value {
    json!({"content": [text(serde_json::to_string_pretty(&v).unwrap_or_default())], "structuredContent": v, "isError": false})
}

fn list_types() -> Value {
    let m: Map<String, Value> = TYPES
        .iter()
        .map(|k| (k.name.to_string(), json!({"description": k.note, "tonemap": k.tonemap, "stars": k.stars})))
        .collect();
    structured(Value::Object(m))
}

fn list_palettes() -> Value {
    let m: Map<String, Value> =
        PALETTES.iter().map(|p| (p.name.to_string(), json!({"description": p.note, "colors": p.stops}))).collect();
    structured(Value::Object(m))
}

fn call_tool(params: &Value) -> Result<Value, (i64, String)> {
    let name = params.get("name").and_then(Value::as_str).unwrap_or_default();
    let tools = tools();
    let Some(tool) = tools.as_array().and_then(|t| t.iter().find(|t| t["name"] == name)) else {
        return Err((-32602, format!("Unknown tool: {name}")));
    };
    let empty = Map::new();
    let raw = params.get("arguments").and_then(Value::as_object).unwrap_or(&empty);
    let args = Args(raw);
    let content = |c: Vec<Value>| json!({"content": c, "isError": false});
    let result = check_known_args(tool, raw).and_then(|()| match name {
        "generate_gradient" => generate_gradient(&args).map(content),
        "preview_types" => preview_types(&args).map(content),
        "preview_seeds" => preview_seeds(&args).map(content),
        "list_types" => Ok(list_types()),
        _ => Ok(list_palettes()),
    });
    // tool failures go back as results the model can read and act on, not as protocol errors
    Ok(result.unwrap_or_else(|e| json!({"content": [text(format!("Error in {name}: {e}"))], "isError": true})))
}

// --- JSON-RPC loop -----------------------------------------------------------------------------

fn handle(msg: &Value) -> Option<Value> {
    let id = msg.get("id")?.clone(); // no id = a notification, no reply
    let method = msg.get("method").and_then(Value::as_str).unwrap_or_default();
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    let result = match method {
        "initialize" => {
            let asked = params.get("protocolVersion").and_then(Value::as_str).unwrap_or_default();
            let version = if PROTOCOL_VERSIONS.contains(&asked) { asked } else { PROTOCOL_VERSIONS[0] };
            let _ = NEGOTIATED.set(version.to_string());
            Ok(json!({
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": false}},
                "serverInfo": {"name": "washi", "title": "washi", "version": env!("CARGO_PKG_VERSION")},
                "instructions": INSTRUCTIONS,
            }))
        }
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({"tools": tools()})),
        "tools/call" => call_tool(&params),
        "resources/list" => Ok(json!({"resources": []})),
        "resources/templates/list" => Ok(json!({"resourceTemplates": []})),
        "prompts/list" => Ok(json!({"prompts": []})),
        _ => Err((-32601, format!("Method not found: {method}"))),
    };
    Some(match result {
        Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
        Err((code, message)) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}),
    })
}

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(Value::Array(batch)) => {
                let r: Vec<Value> = batch.iter().filter_map(handle).collect();
                (!r.is_empty()).then(|| Value::Array(r))
            }
            Ok(msg) => handle(&msg),
            Err(e) => Some(json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": format!("Parse error: {e}")}})),
        };
        if let Some(r) = reply {
            let _ = writeln!(out, "{r}");
            let _ = out.flush();
        }
    }
}
