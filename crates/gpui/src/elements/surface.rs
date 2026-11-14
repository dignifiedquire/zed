use crate::{
    App, Bounds, Element, ElementId, GlobalElementId, InspectorElementId, IntoElement, LayoutId,
    ObjectFit, Pixels, Style, StyleRefinement, Styled, Window,
};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use objc2_core_foundation::CFRetained;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use objc2_core_video::{CVPixelBuffer, CVPixelBufferGetHeight, CVPixelBufferGetWidth};
use refineable::Refineable;

/// A source of a surface's content.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SurfaceSource {
    /// A CoreVideo image buffer
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    Surface(CFRetained<CVPixelBuffer>),
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl From<CFRetained<CVPixelBuffer>> for SurfaceSource {
    fn from(value: CFRetained<CVPixelBuffer>) -> Self {
        SurfaceSource::Surface(value)
    }
}

/// A surface element.
pub struct Surface {
    source: SurfaceSource,
    object_fit: ObjectFit,
    style: StyleRefinement,
}

/// Create a new surface element.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn surface(source: impl Into<SurfaceSource>) -> Surface {
    Surface {
        source: source.into(),
        object_fit: ObjectFit::Contain,
        style: Default::default(),
    }
}

impl Surface {
    /// Set the object fit for the image.
    pub fn object_fit(mut self, object_fit: ObjectFit) -> Self {
        self.object_fit = object_fit;
        self
    }
}

impl Element for Surface {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut style = Style::default();
        style.refine(&self.style);
        let layout_id = window.request_layout(style, [], cx);
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Self::PrepaintState {
    }

    fn paint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        #[cfg_attr(
            not(any(target_os = "macos", target_os = "ios")),
            allow(unused_variables)
        )]
        bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        #[cfg_attr(
            not(any(target_os = "macos", target_os = "ios")),
            allow(unused_variables)
        )]
        window: &mut Window,
        _: &mut App,
    ) {
        match &self.source {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            SurfaceSource::Surface(surface) => {
                let size = crate::size(
                    CVPixelBufferGetWidth(surface).into(),
                    CVPixelBufferGetHeight(surface).into(),
                );
                let image_bounds = self.object_fit.get_bounds(bounds, size);
                let mut style = Style::default();
                style.refine(&self.style);
                let corner_radii = style.corner_radii.to_pixels(window.rem_size());
                window.paint_surface(bounds, image_bounds, corner_radii, surface.clone());
            }
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }
}

impl IntoElement for Surface {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Styled for Surface {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use std::ptr;

    use objc2_core_video::{CVPixelBufferCreate, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange};

    use super::*;
    use crate::{Bounds, ParentElement as _, TestAppContext, div, point, px, size};

    fn setup_buffer(width: usize, height: usize) -> CFRetained<CVPixelBuffer> {
        let mut buffer = ptr::null_mut();
        // SAFETY: The output pointer is writable, and no attributes are passed.
        let status = unsafe {
            CVPixelBufferCreate(
                None,
                width,
                height,
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                None,
                ptr::NonNull::from(&mut buffer),
            )
        };
        assert_eq!(status, 0);
        // SAFETY: A successful create call transfers ownership of the non-null buffer.
        unsafe { CFRetained::from_raw(ptr::NonNull::new(buffer).unwrap()) }
    }

    #[crate::test]
    fn cover_shows_the_middle_of_a_wider_buffer_with_its_radii_clamped(cx: &mut TestAppContext) {
        let buffer = setup_buffer(200, 100);
        let window = cx.add_empty_window();
        window.draw(point(px(0.), px(0.)), size(px(100.), px(100.)), |_, _| {
            div()
                .size(px(100.))
                .child(
                    surface(buffer.clone())
                        .size_full()
                        .rounded(px(100.))
                        .object_fit(ObjectFit::Cover),
                )
                .into_any_element()
        });
        let (painted, scale_factor) = window.update(|window, _| {
            let painted = window.rendered_frame.scene.surfaces.last().map(|surface| {
                (
                    surface.bounds,
                    surface.texture_bounds,
                    surface.corner_radii.top_left,
                )
            });
            (painted, window.scale_factor())
        });
        let (bounds, texture_bounds, radius) = painted.unwrap();
        assert_eq!(bounds.size.width, px(100.).scale(scale_factor));
        assert_eq!(bounds.size.height, px(100.).scale(scale_factor));
        let middle = Bounds {
            origin: point(0.25, 0.),
            size: size(0.5, 1.),
        };
        assert_eq!(texture_bounds, middle);
        assert_eq!(radius, px(50.).scale(scale_factor));
    }
}
