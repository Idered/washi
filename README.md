# washi

Grainy, textured gradient backgrounds, rendered as PNG. Named after the fibrous Japanese paper.

washi is two programs from one Rust crate:

- `washi`, a command-line tool
- `washi-mcp`, an MCP server, so Claude and other MCP clients can generate backgrounds

```bash
washi aurora out.png -p borealis
washi sky storm.png -p storm --rain 1 --noise 8
washi sheet types.png -p peach        # every type side by side
```

## Install

```bash
cargo install --path .
```

This puts `washi` and `washi-mcp` in `~/.cargo/bin`. To add the server to Claude Code:

```bash
claude mcp add -s user washi -- ~/.cargo/bin/washi-mcp
```

## Types

| Type | Look |
|---|---|
| `mesh` | Soft color points blended into each other |
| `sky` | Clouds over a deep sky, with grain only on the cloud edges |
| `grainy` | Wavy bands of light; best with `--grain-size 2` |
| `flow` | Swirling clouds across the whole palette |
| `prism` | Stepped vertical bands glowing from the bottom |
| `glow` | Large arcs with a glowing edge |
| `warp` | Marbled, silky streaks |
| `dunes` | Layered hills with soft, ragged edges |
| `aurora` | Curtains of light with rays and stars |
| `rings` | Concentric rings from a corner |
| `bars` | Blurred columns of light rising from the bottom |

## Palettes

Each palette is 4–6 shades of one color family. Colors are blended in OKLab, so transitions stay clean instead of turning grey.

| Group | Palettes |
|---|---|
| Dark | `evening` (default), `night`, `dawn`, `borealis`, `amethyst`, `deep-sea`, `graphite` |
| Skies | `blue-sky`, `storm`, `night-sky`, `snow` |
| Light | `lavender`, `peach`, `mint`, `glacier`, `citrus`, `rose`, `sage`, `sunset` |

You can also pass your own colors in any order with `-c "#03060F,#12B886,#7048E8"`. They're sorted from dark to light.

## Options

| CLI | MCP | Default | What it does |
|---|---|---|---|
| `-p` | `palette` | `evening` | Named palette |
| `-c` | `colors` | | Custom hex colors instead of a palette |
| `-W`, `-H` | `width`, `height` | 1920×1080 | Size in px (16–8192) |
| `-s` | `seed` | 1 | Layout; the same seed always gives the same image |
| `-d` | `dither` | `grain` | `grain` (film grain), `bayer` (retro 8×8 pattern), `none` |
| `-g` | `grain` | 0.035 | Grain strength: 0.02 subtle, 0.06 strong |
| `--grain-size` | `grain_size` | 1 | Grain size in px; 2–3 is coarser |
| `--noise` | `noise` | 0 | Grey overlay noise, in % |
| `--shade` | `shade` | `none` | Darken under content: `left`, `bottom`, `center` |
| `--rotate` | `rotate` | 0 | Rotate the layout: 0, 90, 180, 270 |
| `--exposure` | `exposure` | 1 | Brightness before tonemapping |
| `--tonemap` | `tonemap` | per type | `none`, `reinhard`, `aces` |
| `--stars` | `stars` | aurora only | Star density |
| `--rain` | `rain` | 0 | Rain streaks: 0.5 light, 1.5 downpour |
| `--snow` | `snow` | 0 | Snowflakes: 0.5 flurries, 1.5 blizzard |
| `--scale`, `--warp`, `--wind` | same | 45, 50, 40 | `sky` only: cloud size, distortion, wind (in %) |
| `--time` | `start_t` | from seed | `sky` only: animation frame |

Some combinations that work well:

```bash
washi sky storm.png -p storm -d none --noise 8 --scale 52 --warp 62 --wind 58 --rain 1
washi sky night.png -p night-sky --rotate 180 --exposure 0.45 --stars 0.7
washi sky snow.png -p snow --snow 1
washi grainy sand.png -p citrus --grain-size 2 --grain 0.05
washi mesh tv.png -p night --shade left
```

## MCP tools

| Tool | What it does |
|---|---|
| `generate_gradient` | Renders a background, saves the PNG and returns its path with a small preview |
| `preview_types` | A sheet comparing types in one palette |
| `preview_seeds` | A sheet comparing seeds for one type |
| `list_types` | Types with descriptions |
| `list_palettes` | Palettes with their colors |

Images are saved to `~/Pictures/washi`, or to `$WASHI_OUT` if it's set. Each call creates a new timestamped file there unless you pass `output_path`:

- It must end in `.png`.
- Relative paths are resolved inside the output folder.
- An existing PNG at that path is replaced, but any other existing file is never overwritten.

Every parameter has a description and limits in the tool schema, and the server enforces the same limits. Unknown arguments are rejected instead of ignored. Results include a text summary, a downscaled JPEG preview and, for clients on protocol 2025-06-18 or newer, a `resource_link` to the saved PNG.

## How it works

Each type computes a color field at up to 1280 px wide, which is then scaled to the final size. After that come exposure, tonemapping, sRGB conversion, shading, stars and weather, and finally grain or dithering. A faint 1-bit dither is always added so smooth gradients don't band in 8-bit PNGs.

The MCP server speaks JSON-RPC over stdio directly, without an SDK or async runtime. It uses about 2 MB while idle, including after large renders.

| File | Contents |
|---|---|
| `src/core.rs` | Types, palettes, color math and the render pipeline |
| `src/rng.rs` | Random number generator (a port of numpy's PCG64, so seeds match the original Python version) |
| `src/sheet.rs` | Comparison sheets with labels |
| `src/bin/washi.rs` | CLI |
| `src/bin/washi-mcp.rs` | MCP server |

## Development

```bash
cargo test --release    # image checks (seeds still render the same) and MCP protocol tests
cargo build --release
```
