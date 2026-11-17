//! Windows for the headless platform client.
//!
//! A headless window has no compositor surface: layout, text shaping, and
//! entity plumbing run normally. Without the `headless-rendering` feature it
//! has no GPU either: `draw` discards the scene, and the sprite atlas hands out
//! tiles without uploading pixels (the same `HeadlessAtlas` GPUI's `TestWindow`
//! uses). This lets command-line tools drive real `Window`-based code paths
//! without a display server.
//!
//! With `headless-rendering`, a window draws each scene into an offscreen
//! texture through `gpui_wgpu`'s headless renderer, a timer drives its frames,
//! and `render_to_image` reads the last frame back, so that tests can drive and
//! capture the app without a display server.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

#[cfg(feature = "headless-rendering")]
use gpui::PlatformHeadlessRenderer as _;

use gpui::{
    Bounds, Capslock, DispatchEventResult, GpuSpecs, HeadlessAtlas, Modifiers, Pixels,
    PlatformAtlas, PlatformDisplay, PlatformInput, PlatformInputHandler, PlatformWindow, Point,
    PromptButton, PromptLevel, RequestFrameOptions, Scene, Size, WindowAppearance,
    WindowBackgroundAppearance, WindowBounds, WindowControlArea, WindowParams, WindowVisibility,
};

struct HeadlessWindowState {
    bounds: Bounds<Pixels>,
    /// Held while this window exists, so the platform can tell that headless windows are open.
    _lease: Rc<()>,
    input_handler: Option<PlatformInputHandler>,
    title: Option<String>,
    is_fullscreen: bool,
    #[cfg(feature = "headless-rendering")]
    frames: rendering::Frames,
}

pub(crate) struct HeadlessWindow(Rc<RefCell<HeadlessWindowState>>);

impl raw_window_handle::HasWindowHandle for HeadlessWindow {
    fn window_handle(
        &self,
    ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        // Headless windows are not backed by a native window.
        Err(raw_window_handle::HandleError::NotSupported)
    }
}

impl raw_window_handle::HasDisplayHandle for HeadlessWindow {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Err(raw_window_handle::HandleError::NotSupported)
    }
}

impl HeadlessWindow {
    pub(crate) fn new(params: WindowParams, lease: Rc<()>, executors: &HeadlessExecutors) -> Self {
        let state = Rc::new(RefCell::new(HeadlessWindowState {
            bounds: params.bounds,
            _lease: lease,
            input_handler: None,
            title: None,
            is_fullscreen: false,
            #[cfg(feature = "headless-rendering")]
            frames: rendering::Frames::new(),
        }));
        #[cfg(feature = "headless-rendering")]
        rendering::start(&state, executors);
        #[cfg(not(feature = "headless-rendering"))]
        let _ = executors;
        Self(state)
    }
}

impl PlatformWindow for HeadlessWindow {
    fn bounds(&self) -> Bounds<Pixels> {
        self.0.borrow().bounds
    }

    fn is_maximized(&self) -> bool {
        false
    }

    fn window_bounds(&self) -> WindowBounds {
        WindowBounds::Windowed(self.bounds())
    }

    fn content_size(&self) -> Size<Pixels> {
        self.bounds().size
    }

    fn resize(&mut self, size: Size<Pixels>) {
        let mut state = self.0.borrow_mut();
        state.bounds.size = size;
        // GPUI asks for the size while it updates the window, so the window
        // learns it on the next frame, as from a display server.
        #[cfg(feature = "headless-rendering")]
        {
            state.frames.pending_size = Some(size);
        }
    }

    fn scale_factor(&self) -> f32 {
        1.0
    }

    fn appearance(&self) -> WindowAppearance {
        WindowAppearance::Dark
    }

    fn display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        None
    }

    fn mouse_position(&self) -> Point<Pixels> {
        Point::default()
    }

    fn modifiers(&self) -> Modifiers {
        Modifiers::default()
    }

    fn capslock(&self) -> Capslock {
        Capslock::default()
    }

    fn set_input_handler(&mut self, input_handler: PlatformInputHandler) {
        self.0.borrow_mut().input_handler = Some(input_handler);
    }

    fn take_input_handler(&mut self) -> Option<PlatformInputHandler> {
        self.0.borrow_mut().input_handler.take()
    }

    fn prompt(
        &self,
        _level: PromptLevel,
        _msg: &str,
        _detail: Option<&str>,
        _answers: &[PromptButton],
    ) -> Option<futures::channel::oneshot::Receiver<usize>> {
        // Fall back to GPUI's rendered prompts.
        None
    }

    fn activate(&self) {}

    fn is_active(&self) -> bool {
        false
    }

    fn visibility(&self) -> WindowVisibility {
        // There is no display server: `draw` discards every scene.
        WindowVisibility::Hidden
    }

    fn is_hovered(&self) -> bool {
        false
    }

    fn background_appearance(&self) -> WindowBackgroundAppearance {
        WindowBackgroundAppearance::Opaque
    }

    fn set_title(&mut self, title: &str) {
        self.0.borrow_mut().title = Some(title.to_owned());
    }

    fn get_title(&self) -> String {
        self.0.borrow().title.clone().unwrap_or_default()
    }

    fn set_background_appearance(&self, _background: WindowBackgroundAppearance) {}

    fn minimize(&self) {}

    fn zoom(&self) {}

    fn toggle_fullscreen(&self) {
        let mut state = self.0.borrow_mut();
        state.is_fullscreen = !state.is_fullscreen;
    }

    fn is_fullscreen(&self) -> bool {
        self.0.borrow().is_fullscreen
    }

    // Without `headless-rendering` no compositor or timer drives a frame loop,
    // so frame and status callbacks are dropped: anything that awaits a frame
    // will never resolve headlessly.
    fn on_request_frame(&self, callback: Box<dyn FnMut(RequestFrameOptions)>) {
        #[cfg(feature = "headless-rendering")]
        {
            self.0.borrow_mut().frames.request_frame = Some(callback);
        }
        #[cfg(not(feature = "headless-rendering"))]
        drop(callback);
    }

    fn on_input(&self, _callback: Box<dyn FnMut(PlatformInput) -> DispatchEventResult>) {}

    fn on_active_status_change(&self, _callback: Box<dyn FnMut(bool)>) {}

    fn on_visibility_change(&self, _callback: Box<dyn FnMut(WindowVisibility)>) {}

    fn on_hover_status_change(&self, _callback: Box<dyn FnMut(bool)>) {}

    fn on_resize(&self, callback: Box<dyn FnMut(Size<Pixels>, f32)>) {
        #[cfg(feature = "headless-rendering")]
        {
            self.0.borrow_mut().frames.resize = Some(callback);
        }
        #[cfg(not(feature = "headless-rendering"))]
        drop(callback);
    }

    fn on_moved(&self, _callback: Box<dyn FnMut()>) {}

    fn on_display_changed(&self, _callback: Box<dyn FnMut()>) {}

    fn on_should_close(&self, _callback: Box<dyn FnMut() -> bool>) {}

    fn on_close(&self, _callback: Box<dyn FnOnce()>) {}

    fn on_hit_test_window_control(&self, _callback: Box<dyn FnMut() -> Option<WindowControlArea>>) {
    }

    fn on_appearance_changed(&self, _callback: Box<dyn FnMut()>) {}

    fn draw(&self, scene: &Scene) {
        #[cfg(feature = "headless-rendering")]
        {
            let mut state = self.0.borrow_mut();
            let size = rendering::device_size(state.bounds.size);
            if let Some(renderer) = state.frames.renderer.as_mut()
                && let Err(error) = renderer.render_scene(scene, size)
            {
                log::error!("Failed to draw a headless frame: {error:#}");
            }
        }
        #[cfg(not(feature = "headless-rendering"))]
        let _ = scene;
    }

    fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        #[cfg(feature = "headless-rendering")]
        if let Some(renderer) = self.0.borrow().frames.renderer.as_ref() {
            return renderer.sprite_atlas();
        }
        Arc::new(HeadlessAtlas::default())
    }

    #[cfg(feature = "headless-rendering")]
    fn render_to_image(&self, scene: &Scene) -> anyhow::Result<image::RgbaImage> {
        let mut state = self.0.borrow_mut();
        let size = rendering::device_size(state.bounds.size);
        let renderer = state
            .frames
            .renderer
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("this headless window found no GPU adapter to draw with"))?;
        renderer.render_scene_to_image(scene, size)
    }

    fn is_subpixel_rendering_supported(&self) -> bool {
        false
    }

    fn update_ime_position(&self, _bounds: Bounds<Pixels>) {}

    fn gpu_specs(&self) -> Option<GpuSpecs> {
        None
    }
}

/// The executors a headless window schedules its frames on.
// Only `headless-rendering` schedules frames.
#[cfg_attr(not(feature = "headless-rendering"), allow(dead_code))]
pub(crate) struct HeadlessExecutors {
    pub(crate) foreground: gpui::ForegroundExecutor,
    pub(crate) background: gpui::BackgroundExecutor,
}

#[cfg(feature = "headless-rendering")]
mod rendering {
    use std::{cell::RefCell, rc::Rc, time::Duration};

    use gpui::{DevicePixels, FrameRequestSource, Pixels, RequestFrameOptions, Size, Task};
    use gpui_wgpu::WgpuHeadlessRenderer;

    use super::{HeadlessExecutors, HeadlessWindowState};

    /// The time between two frames, a display's 60 Hz.
    const FRAME_INTERVAL: Duration = Duration::from_millis(16);

    /// The offscreen renderer of a headless window and the callbacks its frames drive.
    pub(super) struct Frames {
        pub(super) renderer: Option<WgpuHeadlessRenderer>,
        pub(super) request_frame: Option<Box<dyn FnMut(RequestFrameOptions)>>,
        pub(super) resize: Option<Box<dyn FnMut(Size<Pixels>, f32)>>,
        /// A size GPUI asked for that the window has not reported back yet.
        pub(super) pending_size: Option<Size<Pixels>>,
        ticker: Option<Task<()>>,
    }

    impl Frames {
        pub(super) fn new() -> Self {
            let renderer = WgpuHeadlessRenderer::new()
                .map_err(|error| log::warn!("Headless windows draw nothing: {error:#}"))
                .ok();
            Self {
                renderer,
                request_frame: None,
                resize: None,
                pending_size: None,
                ticker: None,
            }
        }
    }

    /// Returns `size` in device pixels, at the headless scale factor of 1.
    pub(super) fn device_size(size: Size<Pixels>) -> Size<DevicePixels> {
        Size {
            width: DevicePixels(f32::from(size.width).round() as i32),
            height: DevicePixels(f32::from(size.height).round() as i32),
        }
    }

    /// Ticks the frames of `state` until the window goes away.
    pub(super) fn start(state: &Rc<RefCell<HeadlessWindowState>>, executors: &HeadlessExecutors) {
        let window = Rc::downgrade(state);
        let background = executors.background.clone();
        let ticker = executors.foreground.spawn(async move {
            loop {
                background.timer(FRAME_INTERVAL).await;
                let Some(state) = window.upgrade() else {
                    break;
                };
                tick(&state);
            }
        });
        state.borrow_mut().frames.ticker = Some(ticker);
    }

    /// Reports a pending size, then asks GPUI for a frame. Each callback runs
    /// without the window's state borrowed, because GPUI asks the window for its
    /// bounds while it handles either.
    fn tick(state: &Rc<RefCell<HeadlessWindowState>>) {
        let pending = state.borrow_mut().frames.pending_size.take();
        if let Some(size) = pending {
            let callback = state.borrow_mut().frames.resize.take();
            if let Some(mut callback) = callback {
                callback(size, 1.0);
                state.borrow_mut().frames.resize.get_or_insert(callback);
            }
        }
        let callback = state.borrow_mut().frames.request_frame.take();
        if let Some(mut callback) = callback {
            callback(RequestFrameOptions {
                signal_source: FrameRequestSource::LocalSchedule,
                ..RequestFrameOptions::default()
            });
            state.borrow_mut().frames.request_frame.get_or_insert(callback);
        }
    }
}

#[cfg(all(test, feature = "headless-rendering"))]
mod tests {
    use std::{cell::Cell, time::Duration};

    use gpui::{
        BackgroundExecutor, BorderStyle, ContentMask, Corners, Edges, ForegroundExecutor, Hsla,
        Point, Quad, TestDispatcher, WindowKind,
    };

    use super::*;

    fn setup_window(width: f32, height: f32) -> (HeadlessWindow, TestDispatcher) {
        let dispatcher = TestDispatcher::new(0);
        let shared = Arc::new(dispatcher.clone());
        let executors = HeadlessExecutors {
            foreground: ForegroundExecutor::new(shared.clone()),
            background: BackgroundExecutor::new(shared),
        };
        let params = WindowParams {
            bounds: Bounds {
                origin: Point::default(),
                size: Size {
                    width: width.into(),
                    height: height.into(),
                },
            },
            titlebar: None,
            kind: WindowKind::Normal,
            is_movable: true,
            app_owns_titlebar_drag: false,
            is_resizable: true,
            is_minimizable: true,
            focus: false,
            show: false,
            icon: None,
            display_id: None,
            app_id: None,
            window_min_size: None,
        };
        let window = HeadlessWindow::new(params, Rc::new(()), &executors);
        (window, dispatcher)
    }

    fn solid_quad(x: f32, width: f32, color: Hsla) -> Quad {
        let bounds = Bounds {
            origin: Point { x: x.into(), y: 0.0.into() },
            size: Size { width: width.into(), height: 32.0.into() },
        };
        Quad {
            order: 0,
            border_style: BorderStyle::Solid,
            bounds,
            content_mask: ContentMask { bounds },
            background: color.into(),
            border_color: color,
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        }
    }

    #[test]
    fn window_draws_its_scene_into_an_image() -> anyhow::Result<()> {
        let (window, _) = setup_window(64.0, 32.0);
        let mut scene = Scene::default();
        scene.insert_primitive(solid_quad(0.0, 32.0, gpui::red()));
        scene.insert_primitive(solid_quad(32.0, 32.0, gpui::blue()));
        scene.finish();
        window.draw(&scene);
        let image = window.render_to_image(&scene)?;
        assert_eq!(image.dimensions(), (64, 32));
        let red = image.get_pixel(16, 16).0;
        let blue = image.get_pixel(48, 16).0;
        assert!(red[0] > 200 && red[2] < 50, "{red:?}");
        assert!(blue[2] > 200 && blue[0] < 50, "{blue:?}");
        Ok(())
    }

    #[test]
    fn resize_reaches_gpui_with_the_next_frame() {
        let (mut window, dispatcher) = setup_window(100.0, 100.0);
        let reported = Rc::new(Cell::new(None));
        let frames = Rc::new(Cell::new(0));
        window.on_resize(Box::new({
            let reported = reported.clone();
            move |size, scale| reported.set(Some((size, scale)))
        }));
        window.on_request_frame(Box::new({
            let frames = frames.clone();
            move |_| frames.set(frames.get() + 1)
        }));
        let size = Size {
            width: 200.0.into(),
            height: 150.0.into(),
        };
        window.resize(size);
        assert_eq!(window.content_size(), size);
        assert_eq!(reported.get(), None);
        dispatcher.advance_clock(Duration::from_millis(16));
        dispatcher.run_until_parked();
        assert_eq!(reported.get(), Some((size, 1.0)));
        assert!(frames.get() >= 1);
    }
}
