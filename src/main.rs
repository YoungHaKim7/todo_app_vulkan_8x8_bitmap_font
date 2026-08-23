mod font;

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use vulkano::{
    DeviceSize, Validated, Version, VulkanError, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo, CopyImageToBufferInfo,
        PrimaryCommandBufferAbstract, RenderingAttachmentInfo, RenderingInfo,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorImageInfo, DescriptorSet, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    format::Format,
    image::{
        Image, ImageCreateInfo, ImageType, ImageUsage,
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        DynamicState, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
        layout::{PipelineLayoutCreateInfo, push_constant_ranges_from_stages},
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    shader::ShaderStages,
    swapchain::{
        Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image,
    },
    sync::{self, GpuFuture},
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{CursorIcon, Window, WindowId},
};

mod ui_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "../assets/ui_vs.vert",
    }
}

mod ui_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "../assets/ui_fs.frag",
    }
}

#[derive(BufferContents, Clone, Copy, Vertex)]
#[repr(C)]
struct UiVertex {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
    #[format(R32G32B32A32_SFLOAT)]
    color: [f32; 4],
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct Push {
    screen: [f32; 4],
}

const MAX_VERTICES: usize = 1 << 16;
const SAVE_FILE: &str = "todos.txt";

const ATLAS_COLS: u32 = 16;
const CELL_PX: u32 = 8;
const ATLAS_CELLS: u32 = 96;
const ATLAS_ROWS: u32 = ATLAS_CELLS.div_ceil(ATLAS_COLS);
const ATLAS_W: u32 = ATLAS_COLS * CELL_PX;
const ATLAS_H: u32 = ATLAS_ROWS * CELL_PX;

const GLYPH_ADV: f32 = 8.0;
const SCALE_TITLE: f32 = 3.0;
const SCALE_TEXT: f32 = 2.0;

const COL_BG: [f32; 4] = [0.055, 0.06, 0.078, 1.0];
const COL_ROW_ALT: [f32; 4] = [0.085, 0.09, 0.115, 1.0];
const COL_PANEL: [f32; 4] = [0.13, 0.135, 0.17, 1.0];
const COL_PANEL_HOVER: [f32; 4] = [0.18, 0.19, 0.235, 1.0];
const COL_FIELD: [f32; 4] = [0.09, 0.095, 0.125, 1.0];
const COL_BORDER: [f32; 4] = [0.24, 0.25, 0.31, 1.0];
const COL_ACCENT: [f32; 4] = [0.23, 0.52, 0.93, 1.0];
const COL_ACCENT_HOVER: [f32; 4] = [0.32, 0.62, 1.0, 1.0];
const COL_ACCENT_DISABLED: [f32; 4] = [0.13, 0.20, 0.32, 1.0];
const COL_TEXT: [f32; 4] = [0.92, 0.93, 0.96, 1.0];
const COL_TEXT_DIM: [f32; 4] = [0.44, 0.46, 0.55, 1.0];
const COL_PLACEHOLDER: [f32; 4] = [0.38, 0.40, 0.48, 1.0];
const COL_CHECK: [f32; 4] = [0.30, 0.78, 0.49, 1.0];
const COL_DANGER_HOVER: [f32; 4] = [0.98, 0.45, 0.43, 1.0];

fn white_uv() -> [f32; 2] {
    [
        (CELL_PX as f32 * 0.5) / ATLAS_W as f32,
        (CELL_PX as f32 * 0.5) / ATLAS_H as f32,
    ]
}

fn cell_uv(cell: u32, fx: f32, fy: f32) -> [f32; 2] {
    let col = cell % ATLAS_COLS;
    let row = cell / ATLAS_COLS;
    [
        (col as f32 * CELL_PX as f32 + fx * CELL_PX as f32) / ATLAS_W as f32,
        (row as f32 * CELL_PX as f32 + fy * CELL_PX as f32) / ATLAS_H as f32,
    ]
}

fn build_atlas() -> Vec<u8> {
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

#[derive(Clone, Copy)]
struct Rect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

impl Rect {
    fn contains(self, p: [f32; 2]) -> bool {
        p[0] >= self.x && p[0] < self.x + self.w && p[1] >= self.y && p[1] < self.y + self.h
    }

    fn inset(self, d: f32) -> Rect {
        Rect {
            x: self.x + d,
            y: self.y + d,
            w: self.w - 2.0 * d,
            h: self.h - 2.0 * d,
        }
    }
}

struct Ui {
    verts: Vec<UiVertex>,
    mouse: [f32; 2],
    clicks: Vec<[f32; 2]>,
    pointer: bool,
}

impl Ui {
    fn new(mouse: [f32; 2]) -> Self {
        Self {
            verts: Vec::new(),
            mouse,
            clicks: Vec::new(),
            pointer: false,
        }
    }

    fn hovered(&self, r: Rect) -> bool {
        r.contains(self.mouse)
    }

    fn take_click(&mut self, r: Rect) -> bool {
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

    fn rect(&mut self, r: Rect, color: [f32; 4]) {
        self.quad_rot(
            [r.x + r.w * 0.5, r.y + r.h * 0.5],
            [r.w * 0.5, r.h * 0.5],
            0.0,
            color,
        );
    }

    fn line(&mut self, a: [f32; 2], b: [f32; 2], thickness: f32, color: [f32; 4]) {
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

    fn text_at(&mut self, x: f32, y: f32, s: &str, scale: f32, color: [f32; 4]) -> f32 {
        let mut cx = x;
        for b in s.bytes() {
            cx = self.glyph(cx, y, b, scale, color);
        }
        cx
    }

    fn text_clipped(
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

fn text_width(s: &str, scale: f32) -> f32 {
    s.chars().count() as f32 * GLYPH_ADV * scale
}

fn fit_width(s: &str, scale: f32, max_w: f32) -> f32 {
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

fn sanitize(c: char) -> Option<char> {
    if c == '\t' {
        Some(' ')
    } else if c.is_ascii_graphic() || c == ' ' {
        Some(c)
    } else {
        None
    }
}

struct Todo {
    text: String,
    done: bool,
}

fn parse_save_line(line: &str) -> Option<Todo> {
    let (flag, text) = line.split_once('\t')?;
    let text: String = text.chars().filter_map(sanitize).collect();
    let text = text.trim().to_string();
    (!text.is_empty()).then_some(Todo {
        text,
        done: flag.trim() == "1",
    })
}

fn encode_save_line(todo: &Todo) -> String {
    format!("{}\t{}", u8::from(todo.done), todo.text)
}

struct Todos {
    items: Vec<Todo>,
    input: String,
    focused: bool,
    caret_since: Instant,
    scroll: f32,
    max_scroll: f32,
}

impl Todos {
    fn load(path: &Path) -> Self {
        let items = fs::read_to_string(path)
            .map(|data| data.lines().filter_map(parse_save_line).collect())
            .unwrap_or_default();
        Self {
            items,
            input: String::new(),
            focused: false,
            caret_since: Instant::now(),
            scroll: 0.0,
            max_scroll: 0.0,
        }
    }

    fn save(&self, path: &Path) {
        let body: String = self
            .items
            .iter()
            .map(|t| encode_save_line(t) + "\n")
            .collect();
        let _ = fs::write(path, body);
    }

    fn add_task(&mut self, path: &Path) {
        let text = self.input.trim().to_string();
        if text.is_empty() {
            return;
        }
        self.items.push(Todo { text, done: false });
        self.input.clear();
        self.caret_since = Instant::now();
        self.save(path);
    }

    fn open_count(&self) -> usize {
        self.items.iter().filter(|t| !t.done).count()
    }

    fn done_count(&self) -> usize {
        self.items.iter().filter(|t| t.done).count()
    }
}

struct BtnStyle {
    base: [f32; 4],
    hover: [f32; 4],
    disabled: [f32; 4],
    text: [f32; 4],
}

const BTN_PRIMARY: BtnStyle = BtnStyle {
    base: COL_ACCENT,
    hover: COL_ACCENT_HOVER,
    disabled: COL_ACCENT_DISABLED,
    text: COL_TEXT,
};

const BTN_GHOST: BtnStyle = BtnStyle {
    base: COL_PANEL,
    hover: COL_PANEL_HOVER,
    disabled: COL_PANEL,
    text: COL_TEXT_DIM,
};

fn button(ui: &mut Ui, r: Rect, label: &str, style: &BtnStyle, enabled: bool) -> bool {
    let hot = enabled && ui.hovered(r);
    if hot {
        ui.pointer = true;
    }
    let clicked = enabled && ui.take_click(r);
    let bg = if !enabled {
        style.disabled
    } else if hot {
        style.hover
    } else {
        style.base
    };
    ui.rect(r, bg);
    let tw = text_width(label, SCALE_TEXT);
    let tc = if enabled { style.text } else { COL_TEXT_DIM };
    ui.text_at(
        r.x + (r.w - tw) * 0.5,
        r.y + (r.h - GLYPH_ADV * SCALE_TEXT) * 0.5,
        label,
        SCALE_TEXT,
        tc,
    );
    clicked
}

fn checkbox(ui: &mut Ui, r: Rect, checked: bool) -> bool {
    let hot = ui.hovered(r);
    if hot {
        ui.pointer = true;
    }
    let clicked = ui.take_click(r);
    let active = checked || hot;
    ui.rect(r, if active { COL_CHECK } else { COL_BORDER });
    ui.rect(r.inset(2.0), COL_FIELD);
    if checked {
        let m = 0.28 * r.h;
        let p1 = [r.x + m, r.y + r.h * 0.55];
        let p2 = [r.x + r.w * 0.42, r.y + r.h - m * 0.7];
        let p3 = [r.x + r.w - m * 0.6, r.y + r.h * 0.25];
        ui.line(p1, p2, 2.5, COL_CHECK);
        ui.line(p2, p3, 2.5, COL_CHECK);
    }
    clicked
}

fn delete_button(ui: &mut Ui, r: Rect) -> bool {
    let hot = ui.hovered(r);
    if hot {
        ui.pointer = true;
    }
    let clicked = ui.take_click(r);
    if hot {
        ui.rect(r, [0.86, 0.37, 0.35, 0.18]);
    }
    let c = [r.x + r.w * 0.5, r.y + r.h * 0.5];
    let d = r.w * 0.26;
    let col = if hot { COL_DANGER_HOVER } else { COL_TEXT_DIM };
    ui.line([c[0] - d, c[1] - d], [c[0] + d, c[1] + d], 2.0, col);
    ui.line([c[0] - d, c[1] + d], [c[0] + d, c[1] - d], 2.0, col);
    clicked
}

fn caret_blinking(since: Instant) -> bool {
    (since.elapsed().as_millis() / 450).is_multiple_of(2)
}

fn draw_ui(todos: &mut Todos, save_path: &Path, ui: &mut Ui, w: f32, h: f32) {
    ui.pointer = false;
    ui.verts.clear();

    let pad = 26.0;
    let content_w = w - 2.0 * pad;

    ui.text_at(pad, pad, "ToDo", SCALE_TITLE, COL_ACCENT_HOVER);
    let counts = format!("{} open / {} done", todos.open_count(), todos.done_count());
    ui.text_at(
        w - pad - text_width(&counts, SCALE_TEXT),
        pad + 10.0,
        &counts,
        SCALE_TEXT,
        COL_TEXT_DIM,
    );

    let y0 = pad + GLYPH_ADV * SCALE_TITLE + 16.0;
    let row_h = 38.0;
    let add_w = 88.0;
    let gap = 10.0;

    let field = Rect {
        x: pad,
        y: y0,
        w: content_w - add_w - gap,
        h: row_h,
    };
    let was_focused = todos.focused;
    let field_clicked = ui.take_click(field);
    if field_clicked {
        todos.caret_since = Instant::now();
    }
    if ui.hovered(field) {
        ui.pointer = true;
    }
    ui.rect(
        field,
        if todos.focused {
            COL_ACCENT
        } else {
            COL_BORDER
        },
    );
    ui.rect(field.inset(1.5), COL_FIELD);

    let ty = field.y + (field.h - GLYPH_ADV * SCALE_TEXT) * 0.5;
    let tx = field.x + 12.0;
    let max_tx = field.x + field.w - 12.0;
    if todos.input.is_empty() && !todos.focused {
        ui.text_at(tx, ty, "What needs doing?", SCALE_TEXT, COL_PLACEHOLDER);
    } else {
        ui.text_clipped(tx, ty, &todos.input, SCALE_TEXT, COL_TEXT, max_tx);
    }
    if todos.focused && caret_blinking(todos.caret_since) {
        let caret_x = (tx + fit_width(&todos.input, SCALE_TEXT, max_tx - tx)).min(max_tx - 2.0);
        ui.rect(
            Rect {
                x: caret_x,
                y: ty - 3.0,
                w: 2.0,
                h: GLYPH_ADV * SCALE_TEXT + 6.0,
            },
            COL_ACCENT_HOVER,
        );
    }

    let add_btn = Rect {
        x: w - pad - add_w,
        y: y0,
        w: add_w,
        h: row_h,
    };
    let add_clicked = button(
        ui,
        add_btn,
        "Add",
        &BTN_PRIMARY,
        !todos.input.trim().is_empty(),
    );
    if add_clicked {
        todos.add_task(save_path);
    }

    let list_top = y0 + row_h + 16.0;
    let list_bottom = h - 48.0;
    let pitch = 46.0;
    let item_h = 40.0;
    let visible_h = (list_bottom - list_top).max(0.0);

    todos.max_scroll = (todos.items.len() as f32 * pitch - visible_h).max(0.0);
    todos.scroll = todos.scroll.clamp(0.0, todos.max_scroll);

    let first = (todos.scroll / pitch).floor() as usize;
    let visible = (visible_h / pitch).ceil() as usize + 1;

    let mut interacted_elsewhere = false;

    for i in first..todos.items.len().min(first + visible) {
        let ry = list_top + i as f32 * pitch - todos.scroll;
        let top = ry;
        let bottom = ry + item_h;
        if top < list_top || bottom > list_bottom {
            continue;
        }
        if i % 2 == 1 {
            ui.rect(
                Rect {
                    x: pad - 8.0,
                    y: ry,
                    w: content_w + 16.0,
                    h: item_h,
                },
                COL_ROW_ALT,
            );
        }

        let cb = Rect {
            x: pad + 2.0,
            y: ry + (item_h - 24.0) * 0.5,
            w: 24.0,
            h: 24.0,
        };
        if checkbox(ui, cb, todos.items[i].done) {
            todos.items[i].done = !todos.items[i].done;
            todos.save(save_path);
            interacted_elsewhere = true;
        }

        let text_x = cb.x + cb.w + 14.0;
        let del_btn = Rect {
            x: w - pad - 28.0,
            y: ry + (item_h - 28.0) * 0.5,
            w: 28.0,
            h: 28.0,
        };
        let text_col = if todos.items[i].done {
            COL_TEXT_DIM
        } else {
            COL_TEXT
        };
        let tw = ui.text_clipped(
            text_x,
            ry + (item_h - GLYPH_ADV * SCALE_TEXT) * 0.5,
            &todos.items[i].text,
            SCALE_TEXT,
            text_col,
            del_btn.x - 14.0,
        );
        if todos.items[i].done && tw > text_x {
            ui.line(
                [text_x, ry + item_h * 0.5],
                [tw, ry + item_h * 0.5],
                2.0,
                COL_TEXT_DIM,
            );
        }
        if delete_button(ui, del_btn) {
            todos.items.remove(i);
            todos.save(save_path);
            interacted_elsewhere = true;
            break;
        }
    }

    if todos.items.is_empty() {
        let msg = "No tasks yet. Type above and press Enter.";
        ui.text_at(
            (w - text_width(msg, SCALE_TEXT)) * 0.5,
            list_top + visible_h * 0.5 - GLYPH_ADV * SCALE_TEXT * 0.5,
            msg,
            SCALE_TEXT,
            COL_PLACEHOLDER,
        );
    }

    if todos.max_scroll > 0.0 {
        let track = Rect {
            x: w - 5.0,
            y: list_top,
            w: 3.0,
            h: visible_h,
        };
        ui.rect(track, COL_ROW_ALT);
        let thumb_h = (visible_h * visible_h / (visible_h + todos.max_scroll)).max(24.0);
        let thumb_y = track.y + (track.h - thumb_h) * (todos.scroll / todos.max_scroll.max(1e-3));
        ui.rect(
            Rect {
                x: track.x,
                y: thumb_y,
                w: track.w,
                h: thumb_h,
            },
            COL_BORDER,
        );
    }

    let hint = "Enter: add · Esc: quit";
    ui.text_at(pad, h - 36.0, hint, SCALE_TEXT, COL_TEXT_DIM);

    let done_n = todos.done_count();
    let clear_label = format!("Clear completed ({})", done_n);
    let clear_w = text_width(&clear_label, SCALE_TEXT) + 24.0;
    let clear_btn = Rect {
        x: w - pad - clear_w,
        y: h - 42.0,
        w: clear_w,
        h: 30.0,
    };
    if button(ui, clear_btn, &clear_label, &BTN_GHOST, done_n > 0) {
        todos.items.retain(|t| !t.done);
        todos.save(save_path);
        interacted_elsewhere = true;
    }

    if !ui.clicks.is_empty() {
        interacted_elsewhere = true;
        ui.clicks.clear();
    }
    todos.focused = field_clicked || add_clicked || (was_focused && !interacted_elsewhere);
    if todos.focused != was_focused {
        todos.caret_since = Instant::now();
    }
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    sampler: Arc<Sampler>,
    atlas: Arc<ImageView>,
    todos: Todos,
    save_path: PathBuf,
    mouse: [f32; 2],
    pending_clicks: Vec<[f32; 2]>,
    cursor_is_pointer: bool,
    dump_done: bool,
    rcx: Option<RenderContext>,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    attachment_image_views: Vec<Arc<ImageView>>,
    pipeline: Arc<GraphicsPipeline>,
    descriptor_set: Arc<DescriptorSet>,
    viewport: Viewport,
    // One vertex buffer per swapchain image. The image we just acquired is guaranteed to no
    // longer be in use by the GPU, so its buffer can always be written without a conflict.
    vertex_buffers: Vec<Subbuffer<[UiVertex]>>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        println!("Vulkan ToDo");
        println!(
            "Controls: type + Enter = add task · click checkbox = toggle · X = delete · scroll = move list · Esc = quit"
        );

        let library = unsafe { VulkanLibrary::new() }.unwrap();

        let required_extensions = Surface::required_extensions(event_loop);

        let instance = Instance::new(
            &library,
            &InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: &required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
            })
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("no suitable physical device found");

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        if physical_device.api_version() < Version::V1_3 {
            device_extensions.khr_dynamic_rendering = true;
        }

        let (device, mut queues) = Device::new(
            &physical_device,
            &DeviceCreateInfo {
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: &device_extensions,
                enabled_features: &DeviceFeatures {
                    dynamic_rendering: true,
                    ..DeviceFeatures::empty()
                },
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            &device,
            &Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &Default::default(),
        ));

        let sampler = Sampler::new(
            &device,
            &SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )
        .unwrap();

        let atlas = {
            let atlas_image = Image::new(
                &memory_allocator,
                &ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R8_UNORM,
                    extent: [ATLAS_W, ATLAS_H, 1],
                    usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                &AllocationCreateInfo::default(),
            )
            .unwrap();

            let staging = Buffer::from_iter(
                &memory_allocator,
                &BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                build_atlas(),
            )
            .unwrap();

            let mut uploads = AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            uploads
                .copy_buffer_to_image(CopyBufferToImageInfo::new(staging, atlas_image.clone()))
                .unwrap();
            uploads
                .build()
                .unwrap()
                .execute(queue.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .map_err(Validated::unwrap)
                .unwrap()
                .wait(None)
                .map_err(Validated::unwrap)
                .unwrap();

            ImageView::new_default(&atlas_image).unwrap()
        };

        let save_path = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(SAVE_FILE);

        let todos = Todos::load(&save_path);
        println!(
            "{} task(s) loaded from {}",
            todos.items.len(),
            save_path.display()
        );

        Self {
            instance,
            device,
            queue,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            sampler,
            atlas,
            todos,
            save_path,
            mouse: [-1000.0; 2],
            pending_clicks: Vec::new(),
            cursor_is_pointer: false,
            dump_done: false,
            rcx: None,
        }
    }

    fn handle_keyboard(&mut self, event: KeyEvent) {
        if event.state != ElementState::Pressed {
            return;
        }
        if !self.todos.focused {
            return;
        }
        match event.logical_key {
            Key::Named(NamedKey::Enter) => self.todos.add_task(&self.save_path),
            Key::Named(NamedKey::Backspace) => {
                if self.todos.input.pop().is_some() {
                    self.todos.caret_since = Instant::now();
                }
            }
            Key::Named(NamedKey::Space) => self.type_char(' '),
            Key::Character(text) => {
                for c in text.as_str().chars().filter_map(sanitize) {
                    self.type_char(c);
                }
            }
            _ => {}
        }
    }

    fn type_char(&mut self, c: char) {
        if self.todos.input.chars().count() < 80 {
            self.todos.input.push(c);
            self.todos.caret_since = Instant::now();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Vulkan ToDo")
                        .with_inner_size(LogicalSize::new(940.0, 640.0))
                        .with_min_inner_size(LogicalSize::new(560.0, 420.0)),
                )
                .unwrap(),
        );
        let surface = Surface::from_window(&self.instance, &window).unwrap();
        let window_size = window.inner_size();

        let (swapchain, images) = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, &Default::default())
                .unwrap();

            let (image_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, &Default::default())
                .unwrap()[0];

            Swapchain::new(
                &self.device,
                &surface,
                &SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window_size.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let attachment_image_views = images
            .iter()
            .map(|image| ImageView::new_default(image).unwrap())
            .collect::<Vec<_>>();

        let (pipeline, descriptor_set) = {
            let vs = unsafe { ui_vs::load(&self.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = unsafe { ui_fs::load(&self.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();

            let vertex_input_state = UiVertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(&vs),
                PipelineShaderStageCreateInfo::new(&fs),
            ];

            let set_layout = DescriptorSetLayout::new(
                &self.device,
                &DescriptorSetLayoutCreateInfo {
                    bindings: &[
                        DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_count: 1,
                            stages: ShaderStages::FRAGMENT,
                            immutable_samplers: &[&self.sampler],
                            ..DescriptorSetLayoutBinding::new(DescriptorType::Sampler)
                        },
                        DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_count: 1,
                            stages: ShaderStages::FRAGMENT,
                            ..DescriptorSetLayoutBinding::new(DescriptorType::SampledImage)
                        },
                    ],
                    ..Default::default()
                },
            )
            .unwrap();

            let layout = PipelineLayout::new(
                &self.device,
                &PipelineLayoutCreateInfo {
                    set_layouts: &[&set_layout],
                    push_constant_ranges: &push_constant_ranges_from_stages(&stages),
                    ..Default::default()
                },
            )
            .unwrap();

            let subpass = PipelineRenderingCreateInfo {
                color_attachment_formats: &[Some(swapchain.image_format())],
                ..Default::default()
            };

            let pipeline = GraphicsPipeline::new(
                &self.device,
                None,
                &GraphicsPipelineCreateInfo {
                    stages: &stages,
                    vertex_input_state: Some(&vertex_input_state),
                    input_assembly_state: Some(&InputAssemblyState::default()),
                    viewport_state: Some(&ViewportState::default()),
                    rasterization_state: Some(&RasterizationState::default()),
                    multisample_state: Some(&MultisampleState::default()),
                    color_blend_state: Some(&ColorBlendState {
                        attachments: &[ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend::alpha()),
                            ..Default::default()
                        }],
                        ..Default::default()
                    }),
                    dynamic_state: &[DynamicState::Viewport],
                    subpass: Some((&subpass).into()),
                    ..GraphicsPipelineCreateInfo::new(&layout)
                },
            )
            .unwrap();

            let descriptor_set = DescriptorSet::new(
                &self.descriptor_set_allocator,
                &pipeline.layout().set_layouts()[0],
                &[WriteDescriptorSet::image(
                    1,
                    &DescriptorImageInfo {
                        image_view: Some(&self.atlas),
                        ..Default::default()
                    },
                )],
                &[],
            )
            .unwrap();

            (pipeline, descriptor_set)
        };

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let vertex_buffers = (0..attachment_image_views.len())
            .map(|_| {
                Buffer::new_slice(
                    &self.memory_allocator,
                    &BufferCreateInfo {
                        usage: BufferUsage::VERTEX_BUFFER,
                        ..Default::default()
                    },
                    &AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    MAX_VERTICES as DeviceSize,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        self.rcx = Some(RenderContext {
            window,
            swapchain,
            attachment_image_views,
            pipeline,
            descriptor_set,
            viewport,
            vertex_buffers,
            recreate_swapchain: false,
            previous_frame_end: Some(sync::now(self.device.clone()).boxed()),
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                if let Some(rcx) = self.rcx.as_mut() {
                    rcx.recreate_swapchain = true;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse = [position.x as f32, position.y as f32];
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left && state == ElementState::Released {
                    self.pending_clicks.push(self.mouse);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                self.todos.scroll =
                    (self.todos.scroll - lines * 40.0).clamp(0.0, self.todos.max_scroll);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Key::Named(NamedKey::Escape) = event.logical_key {
                    if event.state == ElementState::Pressed {
                        event_loop.exit();
                    }
                } else {
                    self.handle_keyboard(event);
                }
            }
            WindowEvent::Focused(false) => {
                self.todos.focused = false;
            }
            WindowEvent::RedrawRequested => {
                self.redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if !self.dump_done
            && let Some(path) = std::env::var_os("TODO_DUMP_FRAME")
        {
            self.dump_done = true;
            self.dump_frame(&path.to_string_lossy());
            event_loop.exit();
            return;
        }
        if let Some(rcx) = self.rcx.as_ref() {
            rcx.window.request_redraw();
        }
    }
}

impl App {
    fn redraw(&mut self) {
        let window_size = match self.rcx.as_ref() {
            Some(rcx) => rcx.window.inner_size(),
            None => return,
        };

        if window_size.width == 0 || window_size.height == 0 {
            return;
        }

        let rcx = self.rcx.as_mut().unwrap();

        rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

        let memory_allocator = self.memory_allocator.clone();

        if rcx.recreate_swapchain {
            let (new_swapchain, new_images) = rcx
                .swapchain
                .recreate(&SwapchainCreateInfo {
                    image_extent: window_size.into(),
                    ..rcx.swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            rcx.swapchain = new_swapchain;
            rcx.attachment_image_views = new_images
                .iter()
                .map(|image| ImageView::new_default(image).unwrap())
                .collect::<Vec<_>>();
            rcx.vertex_buffers = (0..rcx.attachment_image_views.len())
                .map(|_| {
                    Buffer::new_slice(
                        &memory_allocator,
                        &BufferCreateInfo {
                            usage: BufferUsage::VERTEX_BUFFER,
                            ..Default::default()
                        },
                        &AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                            ..Default::default()
                        },
                        MAX_VERTICES as DeviceSize,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            rcx.viewport.extent = window_size.into();
            rcx.recreate_swapchain = false;
        }

        let (w, h) = (window_size.width as f32, window_size.height as f32);

        let mut ui = Ui::new(self.mouse);
        ui.clicks = std::mem::take(&mut self.pending_clicks);
        draw_ui(&mut self.todos, &self.save_path, &mut ui, w, h);
        if ui.verts.len() > MAX_VERTICES {
            ui.verts.truncate(MAX_VERTICES);
        }
        let vertex_count = ui.verts.len() as u32;

        let want_pointer = ui.pointer;
        if want_pointer != self.cursor_is_pointer {
            self.cursor_is_pointer = want_pointer;
            rcx.window.set_cursor(if want_pointer {
                CursorIcon::Pointer
            } else {
                CursorIcon::Default
            });
        }

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(rcx.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    rcx.recreate_swapchain = true;
                    rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            rcx.recreate_swapchain = true;
        }

        // The image we just acquired cannot be in use by the GPU anymore, so the vertex buffer
        // belonging to it is safe to overwrite. Writing here (instead of before acquiring) is
        // what prevents `AccessConflict(DeviceRead)` when frames are still in flight.
        let vertex_buffer = rcx.vertex_buffers[image_index as usize].clone();
        {
            let mut guard = vertex_buffer.write().unwrap();
            guard[..ui.verts.len()].copy_from_slice(&ui.verts);
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_rendering(RenderingInfo {
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some(COL_BG.into()),
                    ..RenderingAttachmentInfo::new(
                        rcx.attachment_image_views[image_index as usize].clone(),
                    )
                })],
                ..Default::default()
            })
            .unwrap()
            .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(rcx.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                rcx.pipeline.layout().clone(),
                0,
                rcx.descriptor_set.clone(),
            )
            .unwrap()
            .push_constants(
                rcx.pipeline.layout().clone(),
                0,
                Push {
                    screen: [w, h, 0.0, 0.0],
                },
            )
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .unwrap();

        unsafe { builder.draw(vertex_count, 1, 0, 0) }.unwrap();

        builder.end_rendering().unwrap();

        let command_buffer = builder.build().unwrap();

        let future = sync::now(self.device.clone())
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::new(rcx.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                rcx.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                rcx.recreate_swapchain = true;
                rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                println!("failed to flush future: {e}");
                rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }
    }

    fn dump_frame(&mut self, path: &str) {
        let width = 940u32;
        let height = 640u32;

        let rcx = match self.rcx.as_ref() {
            Some(rcx) => rcx,
            None => return,
        };
        let pipeline = rcx.pipeline.clone();
        let descriptor_set = rcx.descriptor_set.clone();
        let layout = pipeline.layout().clone();
        let color_format = rcx.swapchain.image_format();

        let mut ui = Ui::new([-1000.0; 2]);
        draw_ui(
            &mut self.todos,
            &self.save_path,
            &mut ui,
            width as f32,
            height as f32,
        );
        let vertices = ui.verts;
        let vertex_count = vertices.len() as u32;

        let vertex_buffer = Buffer::from_iter(
            &self.memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        let target = Image::new(
            &self.memory_allocator,
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: color_format,
                extent: [width, height, 1],
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();
        let view = ImageView::new_default(&target).unwrap();

        let readback: Subbuffer<[u8]> = Buffer::from_iter(
            &self.memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            std::iter::repeat_n(0u8, (width * height * 4) as usize),
        )
        .unwrap();

        if let Some(rcx) = self.rcx.as_mut()
            && let Some(previous) = rcx.previous_frame_end.take()
        {
            drop(previous);
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_rendering(RenderingInfo {
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some(COL_BG.into()),
                    ..RenderingAttachmentInfo::new(view)
                })],
                ..Default::default()
            })
            .unwrap()
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [width as f32, height as f32],
                    min_depth: 0.0,
                    max_depth: 1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .bind_pipeline_graphics(pipeline)
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                layout.clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .push_constants(
                layout,
                0,
                Push {
                    screen: [width as f32, height as f32, 0.0, 0.0],
                },
            )
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer)
            .unwrap();

        unsafe { builder.draw(vertex_count, 1, 0, 0) }.unwrap();

        builder.end_rendering().unwrap();
        builder
            .copy_image_to_buffer(CopyImageToBufferInfo::new(target, readback.clone()))
            .unwrap();

        let command_buffer = builder.build().unwrap();
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .map_err(Validated::unwrap)
            .unwrap()
            .wait(None)
            .map_err(Validated::unwrap)
            .unwrap();

        let data = readback.read().unwrap();
        let bgra = matches!(
            color_format,
            Format::B8G8R8A8_UNORM | Format::B8G8R8A8_SRGB | Format::B8G8R8A8_SNORM
        );
        let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
        for px in data.as_chunks::<4>().0 {
            if bgra {
                ppm.extend_from_slice(&[px[2], px[1], px[0]]);
            } else {
                ppm.extend_from_slice(&[px[0], px[1], px[2]]);
            }
        }
        std::fs::write(path, ppm).unwrap();
        println!("debug frame written to {path}");
    }
}

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font_covers_printable_ascii() {
        assert_eq!(font::cell_index(b' '), Some(1));
        assert_eq!(font::cell_index(b'~'), Some(95));
        assert_eq!(font::cell_index(b'\n'), None);
        assert!(font::FONT_8X8[b'!' as usize - 32].iter().any(|b| *b != 0));
    }

    #[test]
    fn sanitize_keeps_only_renderable_ascii() {
        assert_eq!(sanitize('a'), Some('a'));
        assert_eq!(sanitize(' '), Some(' '));
        assert_eq!(sanitize('\t'), Some(' '));
        assert_eq!(sanitize('\n'), None);
        assert_eq!(sanitize('\u{e9}'), None);
    }

    #[test]
    fn save_file_roundtrip() {
        let path =
            std::env::temp_dir().join(format!("vulkan_todo_test_{}.txt", std::process::id()));
        let mut todos = Todos {
            items: vec![
                Todo {
                    text: "buy milk".into(),
                    done: false,
                },
                Todo {
                    text: "ship release 1.0!".into(),
                    done: true,
                },
            ],
            input: String::new(),
            focused: false,
            caret_since: Instant::now(),
            scroll: 0.0,
            max_scroll: 0.0,
        };
        todos.save(&path);
        let loaded = Todos::load(&path);
        assert_eq!(loaded.items.len(), 2);
        assert_eq!(loaded.items[0].text, "buy milk");
        assert!(!loaded.items[0].done);
        assert_eq!(loaded.items[1].text, "ship release 1.0!");
        assert!(loaded.items[1].done);
        let _ = fs::remove_file(&path);
    }
}
