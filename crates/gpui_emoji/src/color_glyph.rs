//! Painting of COLR glyphs, version 0 and 1, with skrifa and tiny-skia.

use std::collections::BTreeMap;

use skrifa::{
    FontRef, GlyphId, MetadataProvider as _,
    color::{
        Brush, ColorGlyphCollection, ColorPainter, ColorStop, CompositeMode, Extend,
        Transform as FontTransform,
    },
    instance::{LocationRef, Size},
    outline::{DrawSettings, OutlinePen},
    raw::{TableProvider as _, types::BoundingBox},
    string::StringId,
};
use tiny_skia::{
    BlendMode, Color, FillRule, GradientStop, LinearGradient, Mask, Paint, Path, PathBuilder,
    Pixmap, PixmapPaint, Point, RadialGradient, Rect, Shader, SpreadMode, Transform,
};

/// A colour glyph painted at a size, in straight RGBA.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorImage {
    /// The left edge in pixels, right of the glyph's origin.
    pub left: i32,
    /// The top edge in pixels, above the baseline.
    pub top: i32,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Returns the family of the font in `data` when the font has COLRv1 glyphs,
/// which only the painter draws.
pub fn colrv1_family(data: &[u8]) -> Option<String> {
    let font = FontRef::new(data).ok()?;
    if font.colr().ok()?.version() < 1 {
        return None;
    }
    let family = font
        .localized_strings(StringId::FAMILY_NAME)
        .english_or_first()?;
    Some(family.to_string())
}

/// Returns whether the font in `data` has a COLR glyph for `glyph`.
pub fn has_color_glyph(data: &[u8], glyph: u32) -> bool {
    FontRef::new(data).is_ok_and(|font| {
        ColorGlyphCollection::new(&font)
            .get(GlyphId::new(glyph))
            .is_some()
    })
}

/// Returns the pixel bounds of the COLR glyph `glyph` of the font in `data`
/// at `pixels` per em, as left, top, width and height.
pub fn color_glyph_bounds(data: &[u8], glyph: u32, pixels: f32) -> Option<(i32, i32, u32, u32)> {
    let font = FontRef::new(data).ok()?;
    let color_glyph = ColorGlyphCollection::new(&font).get(GlyphId::new(glyph))?;
    let bounds = color_glyph.bounding_box(LocationRef::default(), Size::unscaled())?;
    let scale = pixels / f32::from(font.head().ok()?.units_per_em());
    Some(pixel_bounds(bounds, scale))
}

/// Paints the COLR glyph `glyph` of the font in `data` at `pixels` per em.
pub fn paint_color_glyph(data: &[u8], glyph: u32, pixels: f32) -> Option<ColorImage> {
    let font = FontRef::new(data).ok()?;
    let color_glyph = ColorGlyphCollection::new(&font).get(GlyphId::new(glyph))?;
    let bounds = color_glyph.bounding_box(LocationRef::default(), Size::unscaled())?;
    let scale = pixels / f32::from(font.head().ok()?.units_per_em());
    let (left, top, width, height) = pixel_bounds(bounds, scale);
    let base = Transform::from_row(scale, 0., 0., -scale, -left as f32, top as f32);
    let mut painter = Painter {
        palette: palette(&font),
        font,
        layers: vec![Pixmap::new(width, height)?],
        transforms: vec![base],
        clips: Vec::new(),
        paths: BTreeMap::new(),
    };
    color_glyph
        .paint(LocationRef::default(), &mut painter)
        .ok()?;
    let pixmap = painter.layers.pop()?;
    let rgba = pixmap
        .pixels()
        .iter()
        .flat_map(|pixel| {
            let color = pixel.demultiply();
            [color.red(), color.green(), color.blue(), color.alpha()]
        })
        .collect();
    Some(ColorImage {
        left,
        top,
        width,
        height,
        rgba,
    })
}

/// Returns the whole pixels covering `bounds` at `scale`, as left, top, width
/// and height.
fn pixel_bounds(bounds: BoundingBox<f32>, scale: f32) -> (i32, i32, u32, u32) {
    let left = (bounds.x_min * scale).floor();
    let top = (bounds.y_max * scale).ceil();
    let width = ((bounds.x_max * scale).ceil() - left).max(1.);
    let height = (top - (bounds.y_min * scale).floor()).max(1.);
    (left as i32, top as i32, width as u32, height as u32)
}

/// Returns the font's first palette, or none.
fn palette(font: &FontRef<'_>) -> Vec<Color> {
    let Ok(cpal) = font.cpal() else {
        return Vec::new();
    };
    let Some(Ok(records)) = cpal.color_records_array() else {
        return Vec::new();
    };
    let first = cpal
        .color_record_indices()
        .first()
        .map_or(0, |index| usize::from(index.get()));
    let count = usize::from(cpal.num_palette_entries());
    records
        .iter()
        .skip(first)
        .take(count)
        .map(|record| Color::from_rgba8(record.red, record.green, record.blue, record.alpha))
        .collect()
}

struct Pen(PathBuilder);

impl OutlinePen for Pen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.move_to(x, y);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to(x, y);
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.0.quad_to(cx0, cy0, x, y);
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.0.cubic_to(cx0, cy0, cx1, cy1, x, y);
    }

    fn close(&mut self) {
        self.0.close();
    }
}

/// Paints a COLR glyph's paint graph into a stack of layers.
struct Painter<'a> {
    font: FontRef<'a>,
    palette: Vec<Color>,
    layers: Vec<Pixmap>,
    transforms: Vec<Transform>,
    clips: Vec<Option<Mask>>,
    paths: BTreeMap<u32, Option<Path>>,
}

impl Painter<'_> {
    fn color(&self, index: u16, alpha: f32) -> Color {
        let mut color = self
            .palette
            .get(usize::from(index))
            .copied()
            .unwrap_or(Color::BLACK);
        color.apply_opacity(alpha);
        color
    }

    fn stops(&self, stops: &[ColorStop]) -> Vec<GradientStop> {
        stops
            .iter()
            .map(|stop| GradientStop::new(stop.offset, self.color(stop.palette_index, stop.alpha)))
            .collect()
    }

    fn first_stop_color(&self, stops: &[ColorStop]) -> Color {
        stops.first().map_or(Color::TRANSPARENT, |stop| {
            self.color(stop.palette_index, stop.alpha)
        })
    }

    fn path(&mut self, glyph: GlyphId) -> Option<Path> {
        let font = &self.font;
        self.paths
            .entry(glyph.to_u32())
            .or_insert_with(|| {
                let outline = font.outline_glyphs().get(glyph)?;
                let mut pen = Pen(PathBuilder::new());
                outline
                    .draw(
                        DrawSettings::unhinted(Size::unscaled(), LocationRef::default()),
                        &mut pen,
                    )
                    .ok()?;
                pen.0.finish()
            })
            .clone()
    }

    fn current(&self) -> Transform {
        self.transforms
            .last()
            .copied()
            .unwrap_or(Transform::identity())
    }

    fn size(&self) -> (u32, u32) {
        self.layers
            .first()
            .map_or((1, 1), |layer| (layer.width(), layer.height()))
    }

    /// Returns the paint of `brush`, whose shader takes `shader_transform`.
    fn paint(&self, brush: &Brush<'_>, shader_transform: Transform) -> Paint<'static> {
        let shader = match brush {
            Brush::Solid {
                palette_index,
                alpha,
            } => Shader::SolidColor(self.color(*palette_index, *alpha)),
            Brush::LinearGradient {
                p0,
                p1,
                color_stops,
                extend,
            } => LinearGradient::new(
                Point::from_xy(p0.x, p0.y),
                Point::from_xy(p1.x, p1.y),
                self.stops(color_stops),
                spread(*extend),
                shader_transform,
            )
            .unwrap_or(Shader::SolidColor(self.first_stop_color(color_stops))),
            Brush::RadialGradient {
                c0,
                r0,
                c1,
                r1,
                color_stops,
                extend,
            } => RadialGradient::new(
                Point::from_xy(c0.x, c0.y),
                Point::from_xy(c1.x, c1.y),
                *r1,
                self.radial_stops(*r0, *r1, color_stops),
                spread(*extend),
                shader_transform,
            )
            .unwrap_or(Shader::SolidColor(self.first_stop_color(color_stops))),
            // tiny-skia 0.11 has no sweep gradient, and Noto Color Emoji uses none.
            Brush::SweepGradient { color_stops, .. } => {
                Shader::SolidColor(self.first_stop_color(color_stops))
            }
        };
        Paint {
            shader,
            anti_alias: true,
            ..Paint::default()
        }
    }

    /// Returns the stops of a radial gradient, moved for a start radius that
    /// tiny-skia 0.11 lacks.
    fn radial_stops(&self, r0: f32, r1: f32, stops: &[ColorStop]) -> Vec<GradientStop> {
        // skrifa moves the start circle out to the first stop. For concentric
        // circles, moving the stops back towards the centre is exact.
        let inner = if r1 > 0. { (r0 / r1).clamp(0., 1.) } else { 0. };
        stops
            .iter()
            .map(|stop| {
                GradientStop::new(
                    inner + stop.offset * (1. - inner),
                    self.color(stop.palette_index, stop.alpha),
                )
            })
            .collect()
    }

    fn push_clip(&mut self, path: Option<Path>) {
        let transform = self.current();
        let (width, height) = self.size();
        let parent = self.clips.last().cloned().flatten();
        let mask = path.and_then(|path| match parent {
            Some(mut mask) => {
                mask.intersect_path(&path, FillRule::Winding, true, transform);
                Some(mask)
            }
            None => {
                let mut mask = Mask::new(width, height)?;
                mask.fill_path(&path, FillRule::Winding, true, transform);
                Some(mask)
            }
        });
        self.clips.push(mask);
    }
}

impl ColorPainter for Painter<'_> {
    fn push_transform(&mut self, transform: FontTransform) {
        let next = self.current().pre_concat(to_tiny_skia(transform));
        self.transforms.push(next);
    }

    fn pop_transform(&mut self) {
        self.transforms.pop();
    }

    fn push_clip_glyph(&mut self, glyph: GlyphId) {
        let path = self.path(glyph);
        self.push_clip(path);
    }

    fn push_clip_box(&mut self, clip_box: BoundingBox<f32>) {
        let path = Rect::from_ltrb(
            clip_box.x_min,
            clip_box.y_min,
            clip_box.x_max,
            clip_box.y_max,
        )
        .map(PathBuilder::from_rect);
        self.push_clip(path);
    }

    fn pop_clip(&mut self) {
        self.clips.pop();
    }

    fn fill(&mut self, brush: Brush<'_>) {
        let paint = self.paint(&brush, self.current());
        let mask = self.clips.last().cloned().flatten();
        let Some(layer) = self.layers.last_mut() else {
            return;
        };
        let Some(rect) = Rect::from_xywh(0., 0., layer.width() as f32, layer.height() as f32)
        else {
            return;
        };
        layer.fill_rect(rect, &paint, Transform::identity(), mask.as_ref());
    }

    fn fill_glyph(
        &mut self,
        glyph: GlyphId,
        brush_transform: Option<FontTransform>,
        brush: Brush<'_>,
    ) {
        let Some(path) = self.path(glyph) else {
            return;
        };
        // tiny-skia applies the path's transform to the shader as well, so the
        // shader takes the brush's own transform alone.
        let shader_transform = brush_transform.map_or(Transform::identity(), to_tiny_skia);
        let paint = self.paint(&brush, shader_transform);
        let transform = self.current();
        let mask = self.clips.last().cloned().flatten();
        if let Some(layer) = self.layers.last_mut() {
            layer.fill_path(&path, &paint, FillRule::Winding, transform, mask.as_ref());
        }
    }

    fn push_layer(&mut self, _: CompositeMode) {
        let (width, height) = self.size();
        if let Some(layer) = Pixmap::new(width, height) {
            self.layers.push(layer);
        }
    }

    fn pop_layer_with_mode(&mut self, mode: CompositeMode) {
        if self.layers.len() < 2 {
            return;
        }
        let Some(top) = self.layers.pop() else {
            return;
        };
        let paint = PixmapPaint {
            blend_mode: blend_mode(mode),
            ..PixmapPaint::default()
        };
        if let Some(layer) = self.layers.last_mut() {
            layer.draw_pixmap(0, 0, top.as_ref(), &paint, Transform::identity(), None);
        }
    }
}

fn to_tiny_skia(transform: FontTransform) -> Transform {
    Transform::from_row(
        transform.xx,
        transform.yx,
        transform.xy,
        transform.yy,
        transform.dx,
        transform.dy,
    )
}

fn spread(extend: Extend) -> SpreadMode {
    match extend {
        Extend::Repeat => SpreadMode::Repeat,
        Extend::Reflect => SpreadMode::Reflect,
        _ => SpreadMode::Pad,
    }
}

fn blend_mode(mode: CompositeMode) -> BlendMode {
    match mode {
        CompositeMode::Clear => BlendMode::Clear,
        CompositeMode::Src => BlendMode::Source,
        CompositeMode::Dest => BlendMode::Destination,
        CompositeMode::DestOver => BlendMode::DestinationOver,
        CompositeMode::SrcIn => BlendMode::SourceIn,
        CompositeMode::DestIn => BlendMode::DestinationIn,
        CompositeMode::SrcOut => BlendMode::SourceOut,
        CompositeMode::DestOut => BlendMode::DestinationOut,
        CompositeMode::SrcAtop => BlendMode::SourceAtop,
        CompositeMode::DestAtop => BlendMode::DestinationAtop,
        CompositeMode::Xor => BlendMode::Xor,
        CompositeMode::Plus => BlendMode::Plus,
        CompositeMode::Screen => BlendMode::Screen,
        CompositeMode::Overlay => BlendMode::Overlay,
        CompositeMode::Darken => BlendMode::Darken,
        CompositeMode::Lighten => BlendMode::Lighten,
        CompositeMode::ColorDodge => BlendMode::ColorDodge,
        CompositeMode::ColorBurn => BlendMode::ColorBurn,
        CompositeMode::HardLight => BlendMode::HardLight,
        CompositeMode::SoftLight => BlendMode::SoftLight,
        CompositeMode::Difference => BlendMode::Difference,
        CompositeMode::Exclusion => BlendMode::Exclusion,
        CompositeMode::Multiply => BlendMode::Multiply,
        CompositeMode::HslHue => BlendMode::Hue,
        CompositeMode::HslSaturation => BlendMode::Saturation,
        CompositeMode::HslColor => BlendMode::Color,
        CompositeMode::HslLuminosity => BlendMode::Luminosity,
        _ => BlendMode::SourceOver,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOTO_COLRV1_SUBSET: &[u8] =
        include_bytes!("../test_data/noto-color-emoji/NotoColorEmoji-COLRv1-Subset.ttf");

    fn glyph_of(character: char) -> u32 {
        let font = FontRef::new(NOTO_COLRV1_SUBSET).expect("font");
        font.charmap().map(character).expect("glyph").to_u32()
    }

    fn pixel(image: &ColorImage, x: u32, y: u32) -> [u8; 4] {
        let start = ((y * image.width + x) * 4) as usize;
        let mut rgba = [0; 4];
        rgba.copy_from_slice(&image.rgba[start..start + 4]);
        rgba
    }

    #[test]
    fn grinning_face_shades_from_a_yellow_centre_to_an_orange_rim() {
        let image = paint_color_glyph(NOTO_COLRV1_SUBSET, glyph_of('😀'), 90.).expect("image");
        let (cx, cy) = (image.width / 2, image.height / 2);
        let [r, g, b, a] = pixel(&image, cx, cy / 3 + 4);
        assert_eq!(a, 255);
        let rim = pixel(&image, cx, image.height - 4);
        assert!(rim[3] > 200, "{rim:?}");
        assert!(
            r > 230 && g > 180 && b < 120,
            "centre-top {:?}",
            [r, g, b, a]
        );
        assert!(
            u16::from(rim[1]) + 20 < u16::from(g),
            "rim {rim:?} not more orange than the top {:?}",
            [r, g, b, a]
        );
    }

    #[test]
    fn painted_image_fills_the_raster_bounds_at_about_an_em() {
        for pixels in [16., 33.8, 90.] {
            let glyph = glyph_of('🔥');
            let image = paint_color_glyph(NOTO_COLRV1_SUBSET, glyph, pixels).expect("image");
            let (left, top, width, height) =
                color_glyph_bounds(NOTO_COLRV1_SUBSET, glyph, pixels).expect("bounds");
            assert_eq!(
                (image.left, image.top, image.width, image.height),
                (left, top, width, height)
            );
            assert_eq!(image.rgba.len(), (width * height * 4) as usize);
            let size = width.max(height) as f32;
            assert!(
                size > pixels * 0.8 && size < pixels * 1.3,
                "{size} at {pixels}"
            );
        }
    }

    #[test]
    fn radial_gradient_keeps_its_first_colour_inside_the_start_radius() {
        let mut painter = Painter {
            font: FontRef::new(NOTO_COLRV1_SUBSET).expect("font"),
            palette: vec![
                Color::from_rgba8(255, 0, 0, 255),
                Color::from_rgba8(0, 0, 255, 255),
            ],
            layers: vec![Pixmap::new(41, 41).expect("pixmap")],
            transforms: vec![Transform::identity()],
            clips: Vec::new(),
            paths: BTreeMap::new(),
        };
        let stops = [
            ColorStop {
                offset: 0.,
                palette_index: 0,
                alpha: 1.,
            },
            ColorStop {
                offset: 1.,
                palette_index: 1,
                alpha: 1.,
            },
        ];
        let centre = skrifa::raw::types::Point::new(20.5, 20.5);
        painter.fill(Brush::RadialGradient {
            c0: centre,
            r0: 10.,
            c1: centre,
            r1: 20.,
            color_stops: &stops,
            extend: Extend::Pad,
        });
        let pixel = painter.layers[0].pixel(28, 20).expect("pixel").demultiply();
        assert_eq!((pixel.red(), pixel.blue()), (255, 0));
    }

    #[test]
    fn colrv1_fonts_name_their_family() {
        assert_eq!(
            colrv1_family(NOTO_COLRV1_SUBSET).as_deref(),
            Some("Noto Color Emoji")
        );
        assert_eq!(colrv1_family(b"not a font"), None);
    }

    #[test]
    fn only_colr_glyphs_count_as_color_glyphs() {
        assert!(has_color_glyph(NOTO_COLRV1_SUBSET, glyph_of('😀')));
        assert!(!has_color_glyph(NOTO_COLRV1_SUBSET, 0));
        assert!(!has_color_glyph(b"not a font", 1));
    }
}
