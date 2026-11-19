//! MCP server (stdio): generates gradient backgrounds with dithering.
//!
//! The protocol is handled by hand (JSON-RPC, one message per line), with no SDK and no async
//! runtime, so the process waiting for calls takes a few MB.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use base64::Engine;
use washi::core::{DEFAULT_PALETTE, DITHERS, Opts, PALETTES, SHADES, SkyParams, TONEMAPS, TYPES, render, save_png, type_names};
use washi::sheet::contact_sheet;
use image::RgbImage;
use serde_json::{Map, Value, json};

const INSTRUCTIONS: &str = "Generates gradient backgrounds (mesh, flow, aurora, ...) as PNG with grain/dithering. \
Start with preview_types to compare types for a palette, then call generate_gradient at the target size. \
The returned path is the finished file; the preview in the response is downscaled.";

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

// --- tool schemas -------------------------------------------------------------------------------

fn tools() -> Value {
    let types = type_names();
    let str_enum = |vals: &[&str], default: &str| json!({"type": "string", "enum": vals, "default": default});
    let num = |default: f64| json!({"type": "number", "default": default});
    let opt_num = || json!({"anyOf": [{"type": "number"}, {"type": "null"}], "default": null});
    let opt_str = || json!({"anyOf": [{"type": "string"}, {"type": "null"}], "default": null});
    let colors = json!({"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}], "default": null});

    json!([
        {
            "name": "generate_gradient",
            "description": "Generates a background and saves it as PNG. Returns the path and (optionally) a downscaled preview.\n\n\
type: background type (list_types describes each).\n\
palette: a named palette (list_palettes); ignored when colors is given.\n\
colors: custom hex colors (3-6 shades of one family work best), any order - sorted by luminance.\n\
rotate: rotates the layout by 0/90/180/270 degrees (e.g. 180 = light at the bottom instead of the top).\n\
noise: overlay noise in % (grey overlay layer, opacity noise/200); pairs well with dither=\"none\".\n\
rain: rain streak density (0.5 light rain, 1.5 downpour). snow: snowflakes (0.5 flurries, 1.5 blizzard).\n\
scale, warp, wind: type=\"sky\" only - cloud size, distortion and wind in % (default 45/50/40).\n\
start_t: sky only - animation time (e.g. 20.75); without it the frame depends on the seed.\n\
seed: a different seed = a different layout with the same colors.\n\
dither: grain (film grain), bayer (visible 8x8 retro pattern), none (only an invisible 1 LSB dither against banding).\n\
grain: grain strength, 0.02 subtle .. 0.06 strong. grain_size: grain size in px (1 fine, 2-3 coarser).\n\
tonemap: overrides the type's default (reinhard / aces / none). exposure: brightness multiplier before tonemapping.\n\
shade: darkening under content (left - text on the left, bottom - rows at the bottom, center); shade_strength 0..1.\n\
stars: star density (by default only aurora, ~0.6).\n\
output_path: where to save; default ~/Pictures/washi/<type>-<time>.png.",
            "inputSchema": {"type": "object", "properties": {
                "type": str_enum(&types, "mesh"),
                "palette": {"type": "string", "default": DEFAULT_PALETTE},
                "colors": colors,
                "width": {"type": "integer", "default": 1920},
                "height": {"type": "integer", "default": 1080},
                "seed": {"type": "integer", "default": 1},
                "dither": str_enum(&DITHERS, "grain"),
                "grain": num(0.035),
                "grain_size": num(1.0),
                "tonemap": {"anyOf": [{"type": "string", "enum": TONEMAPS}, {"type": "null"}], "default": null},
                "exposure": num(1.0),
                "shade": str_enum(&SHADES, "none"),
                "shade_strength": num(0.55),
                "stars": opt_num(),
                "rotate": {"type": "integer", "enum": [0, 90, 180, 270], "default": 0},
                "noise": num(0.0),
                "rain": num(0.0),
                "snow": num(0.0),
                "scale": opt_num(),
                "warp": opt_num(),
                "wind": opt_num(),
                "start_t": opt_num(),
                "output_path": opt_str(),
                "preview": {"type": "boolean", "default": true},
            }},
        },
        {
            "name": "preview_types",
            "description": "Comparison sheet: several background types (all by default) in one palette, small labeled tiles.\n\n\
For choosing a direction before generating at full size. Also saves the sheet as PNG.",
            "inputSchema": {"type": "object", "properties": {
                "palette": {"type": "string", "default": DEFAULT_PALETTE},
                "colors": colors,
                "types": {"anyOf": [{"type": "array", "items": {"type": "string", "enum": types}}, {"type": "null"}], "default": null},
                "seed": {"type": "integer", "default": 1},
                "dither": str_enum(&DITHERS, "grain"),
                "grain": num(0.035),
                "shade": str_enum(&SHADES, "none"),
                "output_path": opt_str(),
            }},
        },
        {
            "name": "preview_seeds",
            "description": "Comparison sheet of one type across several seeds (1..6 by default) - for picking a layout.",
            "inputSchema": {"type": "object", "properties": {
                "type": str_enum(&types, "mesh"),
                "palette": {"type": "string", "default": DEFAULT_PALETTE},
                "colors": colors,
                "seeds": {"anyOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "null"}], "default": null},
                "dither": str_enum(&DITHERS, "grain"),
                "grain": num(0.035),
                "shade": str_enum(&SHADES, "none"),
                "output_path": opt_str(),
            }},
        },
        {
            "name": "list_types",
            "description": "Background types with a description and their default tonemapping.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "list_palettes",
            "description": "Named palettes: colors from dark to light and a short description.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ])
}

// --- arguments ----------------------------------------------------------------------------------

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

// --- tools --------------------------------------------------------------------------------------

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

fn out_path(path: Option<String>, stem: &str) -> Result<PathBuf, String> {
    let p = match path {
        Some(p) if p == "~" => home(),
        Some(p) if p.starts_with("~/") => home().join(&p[2..]),
        Some(p) => PathBuf::from(p),
        None => out_dir().join(format!("{stem}-{}.png", timestamp())),
    };
    if let Some(dir) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    Ok(p)
}

fn text(s: impl Into<String>) -> Value {
    json!({"type": "text", "text": s.into()})
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

fn generate_gradient(a: &Args) -> Result<Vec<Value>, String> {
    let t0 = Instant::now();
    let rotate = a.int_or("rotate", 0)?;
    if ![0, 90, 180, 270].contains(&rotate) {
        return Err(bad("rotate", "0, 90, 180 or 270"));
    }
    let size = |key, default| -> Result<usize, String> {
        let v = a.int_or(key, default)?;
        usize::try_from(v).map_err(|_| "size must be 16..8192 px".to_string())
    };
    let opts = Opts {
        width: size("width", 1920)?,
        height: size("height", 1080)?,
        grain_size: a.f64_or("grain_size", 1.0)?,
        tonemap: match a.str_opt("tonemap")? {
            Some(t) if !TONEMAPS.contains(&t.as_str()) => return Err(bad("tonemap", "none, reinhard, aces or null")),
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
    let img = render(&opts)?;
    let path = out_path(a.str_opt("output_path")?, &opts.kind)?;
    save_png(&img, &path, true)?;
    let msg = format!(
        "Saved {} ({}x{}, {:.1}s)",
        path.display(),
        opts.width,
        opts.height,
        t0.elapsed().as_secs_f64()
    );
    if a.bool_or("preview", true)? { Ok(vec![text(msg), preview(&img, 768)?]) } else { Ok(vec![text(msg)]) }
}

fn sheet_result(variants: Vec<(String, Opts)>, path: &Path) -> Result<Vec<Value>, String> {
    let sheet = contact_sheet(&variants)?;
    save_png(&sheet, path, false)?;
    Ok(vec![text(format!("Sheet: {}", path.display())), preview(&sheet, 1200)?])
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
    let variants = types.into_iter().map(|t| (t.clone(), Opts { kind: t, ..base.clone() })).collect();
    sheet_result(variants, &out_path(a.str_opt("output_path")?, "sheet")?)
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
    let variants = seeds.iter().map(|&s| (format!("{} · seed {s}", base.kind), Opts { seed: s, ..base.clone() })).collect();
    sheet_result(variants, &out_path(a.str_opt("output_path")?, &format!("{}-seeds", base.kind))?)
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
    let empty = Map::new();
    let args = Args(params.get("arguments").and_then(Value::as_object).unwrap_or(&empty));
    let result = match name {
        "generate_gradient" => generate_gradient(&args),
        "preview_types" => preview_types(&args),
        "preview_seeds" => preview_seeds(&args),
        "list_types" => return Ok(list_types()),
        "list_palettes" => return Ok(list_palettes()),
        _ => return Err((-32602, format!("Unknown tool: {name}"))),
    };
    Ok(match result {
        Ok(content) => json!({"content": content, "isError": false}),
        Err(e) => json!({"content": [text(format!("Error in {name}: {e}"))], "isError": true}),
    })
}

// --- JSON-RPC loop ------------------------------------------------------------------------------

fn handle(msg: &Value) -> Option<Value> {
    let id = msg.get("id")?.clone(); // no id = a notification, no reply
    let method = msg.get("method").and_then(Value::as_str).unwrap_or_default();
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    let result = match method {
        "initialize" => {
            let asked = params.get("protocolVersion").and_then(Value::as_str).unwrap_or_default();
            let version = if PROTOCOL_VERSIONS.contains(&asked) { asked } else { PROTOCOL_VERSIONS[0] };
            Ok(json!({
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": false}},
                "serverInfo": {"name": "washi", "version": env!("CARGO_PKG_VERSION")},
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
