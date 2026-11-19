//! Preview sheet: several variants side by side, with labels.

use ab_glyph::{Font, FontVec, PxScale, ScaleFont};
use image::{Rgb, RgbImage};
use rayon::prelude::*;

use crate::core::{Opts, render};

fn font() -> Option<FontVec> {
    for (path, index) in [
        ("/System/Library/Fonts/SFNS.ttf", 0),
        ("/System/Library/Fonts/Helvetica.ttc", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    ] {
        if let Some(f) = std::fs::read(path).ok().and_then(|d| FontVec::try_from_vec_and_index(d, index).ok()) {
            return Some(f);
        }
    }
    None
}

/// Antialiased text; (x, y) is the top-left corner of the line (top of the ascender, as in PIL).
fn draw_text(img: &mut RgbImage, font: &FontVec, size: f32, x: f32, y: f32, text: &str, color: [u8; 3]) {
    // size as in PIL: em height in pixels (ab_glyph's PxScale is ascent - descent)
    let scale = match font.units_per_em() {
        Some(upem) => PxScale::from(size * font.height_unscaled() / upem),
        None => PxScale::from(size),
    };
    let sf = font.as_scaled(scale);
    let baseline = y + sf.ascent();
    let mut pen = x;
    let mut prev = None;
    for ch in text.chars() {
        let id = sf.glyph_id(ch);
        if let Some(p) = prev {
            pen += sf.kern(p, id);
        }
        let glyph = id.with_scale_and_position(scale, ab_glyph::point(pen, baseline));
        pen += sf.h_advance(id);
        prev = Some(id);
        let Some(outline) = font.outline_glyph(glyph) else { continue };
        let b = outline.px_bounds();
        outline.draw(|gx, gy, cov| {
            let (px, py) = (b.min.x as i32 + gx as i32, b.min.y as i32 + gy as i32);
            if px < 0 || py < 0 || px >= img.width() as i32 || py >= img.height() as i32 {
                return;
            }
            let p = img.get_pixel_mut(px as u32, py as u32);
            let a = cov.clamp(0.0, 1.0);
            for i in 0..3 {
                p.0[i] = (p.0[i] as f32 * (1.0 - a) + color[i] as f32 * a).round() as u8;
            }
        });
    }
}

/// variants: a list of (label, render settings). 480x270 tiles, 3 columns.
pub fn contact_sheet(variants: &[(String, Opts)]) -> Result<RgbImage, String> {
    const TILE_W: u32 = 480;
    const TILE_H: u32 = 270;
    const COLS: u32 = 3;
    const GAP: u32 = 12;
    let n = variants.len() as u32;
    let rows = n.div_ceil(COLS);
    let mut sheet = RgbImage::from_pixel(COLS * TILE_W + (COLS + 1) * GAP, rows * TILE_H + (rows + 1) * GAP, Rgb([0x11, 0x11, 0x14]));
    let tiles = variants
        .par_iter()
        .map(|(_, o)| render(&Opts { width: TILE_W as usize, height: TILE_H as usize, ..o.clone() }))
        .collect::<Result<Vec<_>, _>>()?;
    let font = font();
    let size = (TILE_H / 14).max(12) as f32;
    for (i, ((label, _), tile)) in variants.iter().zip(tiles).enumerate() {
        let i = i as u32;
        let x = GAP + (i % COLS) * (TILE_W + GAP);
        let y = GAP + (i / COLS) * (TILE_H + GAP);
        image::imageops::replace(&mut sheet, &tile, x as i64, y as i64);
        if let Some(f) = &font {
            let (tx, ty) = (x as f32 + 10.0, (y + TILE_H) as f32 - size - 12.0);
            draw_text(&mut sheet, f, size, tx + 1.0, ty + 1.0, label, [0, 0, 0]);
            draw_text(&mut sheet, f, size, tx, ty, label, [0xF2, 0xF2, 0xF5]);
        }
    }
    Ok(sheet)
}
