//! Immediate-mode UI core: vertex building, rectangles, and bitmap-glyph text.
//!
//! Widgets and screen layout live in the [`widgets`] and [`screen`] submodules; colors in
//! [`theme`].

pub(crate) mod screen;
pub(crate) mod theme;
pub(crate) mod widgets;

use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

use crate::{
    atlas::{cell_uv, white_uv},
    font,
};

/// Horizontal advance of one glyph cell, in atlas pixels.
pub(crate) const GLYPH_ADV: f32 = 8.0;
pub(crate) const SCALE_TITLE: f32 = 3.0;
pub(crate) const SCALE_TEXT: f32 = 2.0;

#[derive(BufferContents, Clone, Copy, Vertex)]
#[repr(C)]
pub(crate) struct UiVertex {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
    #[format(R32G32B32A32_SFLOAT)]
    color: [f32; 4],
}

#[derive(Clone, Copy)]
pub(crate) struct Rect {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) w: f32,
    pub(crate) h: f32,
}

impl Rect {
    pub(crate) fn contains(self, p: [f32; 2]) -> bool {
        p[0] >= self.x && p[0] < self.x + self.w && p[1] >= self.y && p[1] < self.y + self.h
    }

    pub(crate) fn inset(self, d: f32) -> Rect {
        Rect {
            x: self.x + d,
            y: self.y + d,
            w: self.w - 2.0 * d,
            h: self.h - 2.0 * d,
        }
    }
}

/// One frame's draw list plus the input state it was built against.
pub(crate) struct Ui {
    pub(crate) verts: Vec<UiVertex>,
    mouse: [f32; 2],
    pub(crate) clicks: Vec<[f32; 2]>,
    pub(crate) pointer: bool,
}

impl Ui {
    pub(crate) fn new(mouse: [f32; 2]) -> Self {
        Self {
            verts: Vec::new(),
            mouse,
            clicks: Vec::new(),
            pointer: false,
        }
    }

    pub(crate) fn hovered(&self, r: Rect) -> bool {
        r.contains(self.mouse)
    }

    pub(crate) fn take_click(&mut self, r: Rect) -> bool {
        match self.clicks.iter().position(|p| r.contains(*p)) {
            Some(i) => {
                self.clicks.remove(i);
                true
            }
            None => false,
        }
    }

    fn quad_rot(&mut self, center: [f32; 2], half: [f32; 2], angle: f32, color: [f32; 4]) {
        let (s, c) = angle.sin_cos();
        const CORNERS: [[f32; 2]; 6] = [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ];
        let uv = white_uv();
        for cnr in CORNERS {
            let lx = cnr[0] * half[0];
            let ly = cnr[1] * half[1];
            self.verts.push(UiVertex {
                pos: [center[0] + lx * c - ly * s, center[1] + lx * s + ly * c],
                uv,
                color,
            });
        }
    }

    pub(crate) fn rect(&mut self, r: Rect, color: [f32; 4]) {
        self.quad_rot(
            [r.x + r.w * 0.5, r.y + r.h * 0.5],
            [r.w * 0.5, r.h * 0.5],
            0.0,
            color,
        );
    }

    pub(crate) fn line(&mut self, a: [f32; 2], b: [f32; 2], thickness: f32, color: [f32; 4]) {
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-3 {
            return;
        }
        self.quad_rot(
            [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5],
            [len * 0.5, thickness * 0.5],
            dy.atan2(dx),
            color,
        );
    }

    fn glyph(&mut self, x: f32, y: f32, byte: u8, scale: f32, color: [f32; 4]) -> f32 {
        if byte >= 32
            && let Some(cell) = font::cell_index(byte)
        {
            let g = GLYPH_ADV * scale;
            const FRACS: [[f32; 2]; 6] = [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ];
            for f in FRACS {
                self.verts.push(UiVertex {
                    pos: [x + f[0] * g, y + f[1] * g],
                    uv: cell_uv(cell, f[0], f[1]),
                    color,
                });
            }
        }
        x + GLYPH_ADV * scale
    }

    pub(crate) fn text_at(&mut self, x: f32, y: f32, s: &str, scale: f32, color: [f32; 4]) -> f32 {
        let mut cx = x;
        for b in s.bytes() {
            cx = self.glyph(cx, y, b, scale, color);
        }
        cx
    }

    pub(crate) fn text_clipped(
        &mut self,
        x: f32,
        y: f32,
        s: &str,
        scale: f32,
        color: [f32; 4],
        max_x: f32,
    ) -> f32 {
        let mut cx = x;
        for b in s.bytes() {
            if cx + GLYPH_ADV * scale > max_x {
                break;
            }
            cx = self.glyph(cx, y, b, scale, color);
        }
        cx
    }
}

pub(crate) fn text_width(s: &str, scale: f32) -> f32 {
    s.chars().count() as f32 * GLYPH_ADV * scale
}

pub(crate) fn fit_width(s: &str, scale: f32, max_w: f32) -> f32 {
    let adv = GLYPH_ADV * scale;
    let mut x = 0.0;
    for _ in s.bytes() {
        if x + adv > max_w {
            break;
        }
        x += adv;
    }
    x
}
