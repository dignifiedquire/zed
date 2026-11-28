use std::{cell::Cell, sync::OnceLock};

use gpui_util::ResultExt;
use windows::{
    UI::{
        Color,
        ViewManagement::{UIColorType, UISettings},
    },
    Win32::{Foundation::*, Graphics::Dwm::*, UI::WindowsAndMessaging::*},
    core::BOOL,
};

use crate::*;
use gpui::*;

pub(crate) trait HiLoWord {
    fn hiword(&self) -> u16;
    fn loword(&self) -> u16;
    fn signed_hiword(&self) -> i16;
    fn signed_loword(&self) -> i16;
}

impl HiLoWord for WPARAM {
    fn hiword(&self) -> u16 {
        ((self.0 >> 16) & 0xFFFF) as u16
    }

    fn loword(&self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }

    fn signed_hiword(&self) -> i16 {
        ((self.0 >> 16) & 0xFFFF) as i16
    }

    fn signed_loword(&self) -> i16 {
        (self.0 & 0xFFFF) as i16
    }
}

impl HiLoWord for LPARAM {
    fn hiword(&self) -> u16 {
        ((self.0 >> 16) & 0xFFFF) as u16
    }

    fn loword(&self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }

    fn signed_hiword(&self) -> i16 {
        ((self.0 >> 16) & 0xFFFF) as i16
    }

    fn signed_loword(&self) -> i16 {
        (self.0 & 0xFFFF) as i16
    }
}

pub(crate) unsafe fn get_window_long(hwnd: HWND, nindex: WINDOW_LONG_PTR_INDEX) -> isize {
    #[cfg(target_pointer_width = "64")]
    unsafe {
        GetWindowLongPtrW(hwnd, nindex)
    }
    #[cfg(target_pointer_width = "32")]
    unsafe {
        GetWindowLongW(hwnd, nindex) as isize
    }
}

pub(crate) unsafe fn set_window_long(
    hwnd: HWND,
    nindex: WINDOW_LONG_PTR_INDEX,
    dwnewlong: isize,
) -> isize {
    #[cfg(target_pointer_width = "64")]
    unsafe {
        SetWindowLongPtrW(hwnd, nindex, dwnewlong)
    }
    #[cfg(target_pointer_width = "32")]
    unsafe {
        SetWindowLongW(hwnd, nindex, dwnewlong as i32) as isize
    }
}

pub(crate) fn windows_credentials_target_name(url: &str) -> String {
    format!("zed:url={}", url)
}

pub(crate) fn load_cursor(style: CursorStyle) -> Option<HCURSOR> {
    static ARROW: OnceLock<SafeCursor> = OnceLock::new();
    static IBEAM: OnceLock<SafeCursor> = OnceLock::new();
    static CROSS: OnceLock<SafeCursor> = OnceLock::new();
    static HAND: OnceLock<SafeCursor> = OnceLock::new();
    static SIZEWE: OnceLock<SafeCursor> = OnceLock::new();
    static SIZENS: OnceLock<SafeCursor> = OnceLock::new();
    static SIZENWSE: OnceLock<SafeCursor> = OnceLock::new();
    static SIZENESW: OnceLock<SafeCursor> = OnceLock::new();
    static NO: OnceLock<SafeCursor> = OnceLock::new();
    let (lock, name) = match style {
        CursorStyle::IBeam | CursorStyle::IBeamCursorForVerticalLayout => (&IBEAM, IDC_IBEAM),
        CursorStyle::Crosshair => (&CROSS, IDC_CROSS),
        CursorStyle::PointingHand | CursorStyle::DragLink => (&HAND, IDC_HAND),
        CursorStyle::ResizeLeft
        | CursorStyle::ResizeRight
        | CursorStyle::ResizeLeftRight
        | CursorStyle::ResizeColumn => (&SIZEWE, IDC_SIZEWE),
        CursorStyle::ResizeUp
        | CursorStyle::ResizeDown
        | CursorStyle::ResizeUpDown
        | CursorStyle::ResizeRow => (&SIZENS, IDC_SIZENS),
        CursorStyle::ResizeUpLeftDownRight => (&SIZENWSE, IDC_SIZENWSE),
        CursorStyle::ResizeUpRightDownLeft => (&SIZENESW, IDC_SIZENESW),
        CursorStyle::OperationNotAllowed => (&NO, IDC_NO),
        _ => (&ARROW, IDC_ARROW),
    };
    Some(
        *(*lock.get_or_init(|| {
            HCURSOR(
                unsafe { LoadImageW(None, name, IMAGE_CURSOR, 0, 0, LR_DEFAULTSIZE | LR_SHARED) }
                    .log_err()
                    .unwrap_or_default()
                    .0,
            )
            .into()
        })),
    )
}

thread_local! {
    /// The appearance `Platform::set_window_appearance` forces on every window's frame, if any.
    static FORCED_APPEARANCE: Cell<Option<WindowAppearance>> = const { Cell::new(None) };
}

/// Forces `appearance` on the frame of every window, or lets the frames follow the system again
/// for `None`. Windows opened later and later system changes keep it.
pub(crate) fn force_appearance(appearance: Option<WindowAppearance>) {
    FORCED_APPEARANCE.with(|forced| forced.set(appearance));
}

/// Returns the appearance forced on the windows' frames, if any.
pub(crate) fn forced_appearance() -> Option<WindowAppearance> {
    FORCED_APPEARANCE.with(Cell::get)
}

/// This function is used to configure the dark mode for the window built-in title bar.
/// An appearance forced by `force_appearance` wins over `appearance`.
pub(crate) fn configure_dwm_dark_mode(hwnd: HWND, appearance: WindowAppearance) {
    let appearance = forced_appearance().unwrap_or(appearance);
    let dark_mode_enabled: BOOL = match appearance {
        WindowAppearance::Dark | WindowAppearance::VibrantDark => true.into(),
        WindowAppearance::Light | WindowAppearance::VibrantLight => false.into(),
    };
    unsafe {
        DwmSetWindowAttribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            &dark_mode_enabled as *const _ as _,
            std::mem::size_of::<BOOL>() as u32,
        )
        .log_err();
    }
}

#[inline]
pub(crate) fn logical_point(x: f32, y: f32, scale_factor: f32) -> Point<Pixels> {
    Point {
        x: px(x / scale_factor),
        y: px(y / scale_factor),
    }
}

// https://learn.microsoft.com/en-us/windows/apps/desktop/modernize/apply-windows-themes
#[inline]
pub(crate) fn system_appearance() -> Result<WindowAppearance> {
    let ui_settings = UISettings::new()?;
    let foreground_color = ui_settings.GetColorValue(UIColorType::Foreground)?;
    // If the foreground is light, then is_color_light will evaluate to true,
    // meaning Dark mode is enabled.
    if is_color_light(&foreground_color) {
        Ok(WindowAppearance::Dark)
    } else {
        Ok(WindowAppearance::Light)
    }
}

#[inline(always)]
fn is_color_light(color: &Color) -> bool {
    ((5 * color.G as u32) + (2 * color.R as u32) + color.B as u32) > (8 * 128)
}

#[cfg(test)]
mod tests {
    use gpui::WindowAppearance;
    use windows::{
        Win32::{
            Foundation::HWND,
            Graphics::Dwm::{DWMWA_USE_IMMERSIVE_DARK_MODE, DwmGetWindowAttribute},
            UI::WindowsAndMessaging::{
                CreateWindowExW, DestroyWindow, WINDOW_EX_STYLE, WS_OVERLAPPEDWINDOW,
            },
        },
        core::{BOOL, w},
    };

    use super::{configure_dwm_dark_mode, force_appearance};

    fn frame_is_dark(hwnd: HWND) -> bool {
        let mut dark = BOOL::default();
        unsafe {
            DwmGetWindowAttribute(
                hwnd,
                DWMWA_USE_IMMERSIVE_DARK_MODE,
                &mut dark as *mut _ as _,
                std::mem::size_of::<BOOL>() as u32,
            )
        }
        .unwrap();
        dark.as_bool()
    }

    #[test]
    fn test_forced_appearance_wins_over_the_system_until_cleared() {
        let hwnd = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE::default(),
                w!("STATIC"),
                w!("gpui-test"),
                WS_OVERLAPPEDWINDOW,
                0,
                0,
                100,
                100,
                None,
                None,
                None,
                None,
            )
        }
        .unwrap();

        force_appearance(Some(WindowAppearance::Dark));
        configure_dwm_dark_mode(hwnd, WindowAppearance::Light);
        let forced = frame_is_dark(hwnd);
        force_appearance(None);
        configure_dwm_dark_mode(hwnd, WindowAppearance::Light);
        let followed = frame_is_dark(hwnd);
        unsafe { DestroyWindow(hwnd) }.unwrap();

        assert!(forced);
        assert!(!followed);
    }
}
