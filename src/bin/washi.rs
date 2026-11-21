//! washi <type> <out.png> [options] - the same as the MCP tool, from the terminal.

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use clap::builder::TypedValueParser;
use washi::core::{DEFAULT_PALETTE, Opts, SkyParams, TYPES, palette_names, render, save_png, type_names};
use washi::sheet::contact_sheet;

#[derive(Parser)]
#[command(name = "washi", about = "washi <type> <out.png> [options] - the same as the MCP tool, from the terminal.")]
struct Args {
    /// background type (mesh, sky, grainy, flow, prism, glow, warp, dunes, aurora, rings, bars); 'sheet' = all types on one sheet
    #[arg(value_name = "TYPE")]
    kind: String,
    /// output file (.png)
    out: PathBuf,
    /// palette (see list_palettes)
    #[arg(short, long, default_value = DEFAULT_PALETTE)]
    palette: String,
    /// custom colors instead of a palette: '#hex,#hex,...' (sorted by luminance)
    #[arg(short, long)]
    colors: Option<String>,
    #[arg(short = 'W', long, default_value_t = 1920)]
    width: usize,
    #[arg(short = 'H', long, default_value_t = 1080)]
    height: usize,
    #[arg(short, long, default_value_t = 1)]
    seed: u64,
    /// final texture; hatch = two-color print pattern
    #[arg(short, long, default_value = "grain", value_parser = ["grain", "bayer", "hatch", "none"])]
    dither: String,
    /// grain strength (0.02 subtle, 0.06 strong); with hatch: how irregular the pattern is
    #[arg(short, long, default_value_t = 0.035)]
    grain: f64,
    /// grain size in px (1 = fine); with hatch: size of one pattern square
    #[arg(long, default_value_t = 1.0)]
    grain_size: f64,
    #[arg(long, value_parser = ["none", "reinhard", "aces"])]
    tonemap: Option<String>,
    #[arg(long, default_value_t = 1.0)]
    exposure: f64,
    #[arg(long, default_value = "none", value_parser = ["none", "left", "bottom", "center"])]
    shade: String,
    /// star density (by default only aurora has stars)
    #[arg(long)]
    stars: Option<f64>,
    /// overlay noise, in % (opacity noise/200)
    #[arg(long, default_value_t = 0.0)]
    noise: f64,
    /// sky: cloud scale % (default 45)
    #[arg(long)]
    scale: Option<f64>,
    /// sky: cloud distortion % (default 50)
    #[arg(long)]
    warp: Option<f64>,
    /// sky: wind % (default 40)
    #[arg(long)]
    wind: Option<f64>,
    /// sky: animation time (e.g. 20.75); without it the frame depends on the seed
    #[arg(long)]
    time: Option<f64>,
    /// rain streak density (0.5 light, 1.5 downpour)
    #[arg(long, default_value_t = 0.0)]
    rain: f64,
    /// snowflake density (0.5 flurries, 1.5 blizzard)
    #[arg(long, default_value_t = 0.0)]
    snow: f64,
    /// rotates the layout
    #[arg(long, default_value_t = 0, value_parser = clap::builder::PossibleValuesParser::new(["0", "90", "180", "270"]).map(|s| s.parse::<i64>().unwrap()))]
    rotate: i64,
}

fn main() {
    let a = Args::parse();
    let opts = Opts {
        kind: a.kind.clone(),
        palette: Some(a.palette),
        colors: a.colors.map(|c| c.split(',').map(|s| s.trim().to_string()).collect()),
        width: a.width,
        height: a.height,
        seed: a.seed,
        dither: a.dither,
        grain: a.grain,
        grain_size: a.grain_size,
        tonemap: a.tonemap,
        exposure: a.exposure,
        shade: a.shade,
        shade_strength: 0.55,
        stars: a.stars,
        rotate: a.rotate,
        noise: a.noise,
        rain: a.rain,
        snow: a.snow,
        sky: SkyParams { scale: a.scale, warp: a.warp, wind: a.wind, time: a.time },
    };
    let t0 = Instant::now();
    let result = if a.kind == "sheet" {
        let variants: Vec<_> =
            TYPES.iter().map(|k| (k.name.to_string(), Opts { kind: k.name.to_string(), ..opts.clone() })).collect();
        contact_sheet(&variants).and_then(|s| save_png(&s, &a.out, false))
    } else {
        render(&opts).and_then(|img| save_png(&img, &a.out, true))
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        if e.starts_with("unknown palette") || e.starts_with("unknown type") {
            eprintln!("types: {}\npalettes: {}", type_names().join(", "), palette_names());
        }
        std::process::exit(1);
    }
    println!("OK {} ({:.1}s)", a.out.display(), t0.elapsed().as_secs_f64());
}
