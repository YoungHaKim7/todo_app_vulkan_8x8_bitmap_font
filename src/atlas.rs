//! Glyph atlas: packs the 8x8 font into a single texture and maps cells to UVs.
//!
//! Layout: cell 0 is a solid white square used for filled rectangles, cells 1.. hold the
//! glyphs for ASCII 32..=126.

use crate::font;

pub(crate) const ATLAS_COLS: u32 = 16;
pub(crate) const CELL_PX: u32 = 8;
pub(crate) const ATLAS_CELLS: u32 = 96;
pub(crate) const ATLAS_ROWS: u32 = ATLAS_CELLS.div_ceil(ATLAS_COLS);
pub(crate) const ATLAS_W: u32 = ATLAS_COLS * CELL_PX;
pub(crate) const ATLAS_H: u32 = ATLAS_ROWS * CELL_PX;

pub(crate) fn white_uv() -> [f32; 2] {
    [
        (CELL_PX as f32 * 0.5) / ATLAS_W as f32,
        (CELL_PX as f32 * 0.5) / ATLAS_H as f32,
    ]
}

pub(crate) fn cell_uv(cell: u32, fx: f32, fy: f32) -> [f32; 2] {
    let col = cell % ATLAS_COLS;
    let row = cell / ATLAS_COLS;
    [
        (col as f32 * CELL_PX as f32 + fx * CELL_PX as f32) / ATLAS_W as f32,
        (row as f32 * CELL_PX as f32 + fy * CELL_PX as f32) / ATLAS_H as f32,
    ]
}

pub(crate) fn build_atlas() -> Vec<u8> {
    let mut px = vec![0u8; (ATLAS_W * ATLAS_H) as usize];
    let mut put = |cell: u32, gx: u32, gy: u32, v: u8| {
        let col = cell % ATLAS_COLS;
        let row = cell / ATLAS_COLS;
        let x = col * CELL_PX + gx;
        let y = row * CELL_PX + gy;
        px[(y * ATLAS_W + x) as usize] = v;
    };
    for gy in 0..CELL_PX {
        for gx in 0..CELL_PX {
            put(0, gx, gy, 255);
        }
    }
    for (i, glyph) in font::FONT_8X8.iter().enumerate() {
        let cell = 1 + i as u32;
        for (gy, bits) in glyph.iter().enumerate() {
            for gx in 0..8 {
                if bits & (1 << gx) != 0 {
                    put(cell, gx, gy as u32, 255);
                }
            }
        }
    }
    px
}
