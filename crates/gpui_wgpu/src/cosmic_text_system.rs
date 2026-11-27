use anyhow::{Context as _, Ok, Result};
use collections::HashMap;
use cosmic_text::{
    Attrs, AttrsList, Ellipsize, Family, Font as CosmicTextFont,
    FontFeatures as CosmicFontFeatures, FontSystem, ShapeBuffer, ShapeLine, Stretch, Style, Weight,
};
use gpui::{
    Bounds, DevicePixels, FallbackFontClass, Font, FontFallbacks, FontFeatures, FontId,
    FontMetrics, FontRun, GlyphId, IsZero as _, LineLayout, MissingGlyph, MissingGlyphSink, Pixels,
    PlatformTextSystem, RenderGlyphParams, SUBPIXEL_VARIANTS_X, SUBPIXEL_VARIANTS_Y, ShapedGlyph,
    ShapedRun, SharedString, Size, TextRenderingMode, point, size,
};

use gpui_emoji::{emoji_presentation_clusters, emoji_scale};
use itertools::Itertools;
use parking_lot::RwLock;
use smallvec::SmallVec;
use std::{borrow::Cow, ops::Range, sync::Arc};
use swash::{
    NormalizedCoord, Tag,
    scale::{Render, ScaleContext, Source, StrikeWith},
    tag_from_bytes,
    zeno::{Format, Vector},
};
use unicode_segmentation::UnicodeSegmentation;

// Keep in sync with the Linux list of `EMOJI_FONT_FAMILIES` in gpui's svg_renderer.rs.
const EMOJI_FONT_FAMILIES: &[&str] = &[
    "Noto Color Emoji",
    "Emoji One",
    "Twitter Color Emoji",
    "JoyPixels",
];

const WEIGHT_AXIS: Tag = tag_from_bytes(b"wght");

pub struct CosmicTextSystem(RwLock<CosmicTextSystemState>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FontKey {
    family: SharedString,
    features: FontFeatures,
    fallbacks: Option<FontFallbacks>,
}

impl FontKey {
    fn new(family: SharedString, features: FontFeatures, fallbacks: Option<FontFallbacks>) -> Self {
        Self {
            family,
            features,
            fallbacks,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LoadedFontKey {
    database_id: cosmic_text::fontdb::ID,
    font: FontKey,
    weight: Weight,
}

struct CosmicTextSystemState {
    font_system: FontSystem,
    scratch: ShapeBuffer,
    swash_scale_context: ScaleContext,
    pending_glyph_images: HashMap<RenderGlyphParams, swash::scale::image::Image>,
    /// Contains all already loaded fonts, including all faces. Indexed by `FontId`.
    loaded_fonts: Vec<LoadedFont>,
    loaded_font_ids_by_key: HashMap<LoadedFontKey, FontId>,
    /// Caches the `FontId`s associated with a specific family to avoid iterating the font database
    /// for every font face in a family.
    font_ids_by_family_cache: HashMap<FontKey, SmallVec<[FontId; 4]>>,
    system_font_fallback: String,
    missing_glyph_sink: Option<Arc<dyn MissingGlyphSink>>,
    /// The colour emoji face that draws clusters asking for emoji presentation.
    emoji_face: Option<(cosmic_text::fontdb::ID, SharedString)>,
}

struct LoadedFont {
    font: Arc<CosmicTextFont>,
    /// The face's own weight, or the weight its weight axis is set to.
    weight: Weight,
    /// The face's variation coordinates at `weight`, empty for a face without axes.
    coords: Vec<NormalizedCoord>,
    features: CosmicFontFeatures,
    is_known_emoji_font: bool,
    /// The face's index in its font file when the face has COLRv1 glyphs, which
    /// gpui_emoji's painter draws.
    color_face: Option<u32>,
    /// resolved at load time so `layout_line` shares one chain across faces.
    /// `Arc` keeps clone cheap on the per-run hot path.
    user_fallback_chain: Arc<[(FontId, SharedString)]>,
}

struct FontMatchProperties {
    primary_family_name: SharedString,
    stretch: Stretch,
    style: Style,
    weight: Weight,
    features: CosmicFontFeatures,
    fallback_chain: Arc<[(FontId, SharedString)]>,
}

impl FontMatchProperties {
    fn attributes<'a>(&'a self, font_id: FontId, family_name: &'a str) -> Attrs<'a> {
        Attrs::new()
            .metadata(font_id.0)
            .family(Family::Name(family_name))
            .stretch(self.stretch)
            .style(self.style)
            .weight(self.weight)
            .font_features(self.features.clone())
    }
}

impl CosmicTextSystem {
    /// Returns the selected face's weight and style, which may differ from the request.
    pub fn font_weight_and_style(
        &self,
        font_id: FontId,
    ) -> Result<(gpui::FontWeight, gpui::FontStyle)> {
        let state = self.0.read();
        let font = state
            .loaded_fonts
            .get(font_id.0)
            .context("invalid font ID")?;
        let face = state
            .font_system
            .db()
            .face(font.font.id())
            .context("font face not found")?;
        let style = match face.style {
            cosmic_text::Style::Normal => gpui::FontStyle::Normal,
            cosmic_text::Style::Italic => gpui::FontStyle::Italic,
            cosmic_text::Style::Oblique => gpui::FontStyle::Oblique,
        };
        Ok((gpui::FontWeight(f32::from(font.weight.0)), style))
    }

    /// Builds reports for unresolved source indices after an outer text system
    /// has applied its own fallback.
    pub fn missing_glyphs(
        &self,
        text: &str,
        font_runs: &[FontRun],
        missing_text_indices: impl IntoIterator<Item = usize>,
    ) -> Vec<MissingGlyph> {
        self.0
            .read()
            .missing_glyphs(text, font_runs, missing_text_indices)
    }

    pub fn new(system_font_fallback: &str) -> Self {
        let font_system = FontSystem::new();
        let emoji_face = find_emoji_face(font_system.db());

        Self(RwLock::new(CosmicTextSystemState {
            font_system,
            scratch: ShapeBuffer::default(),
            swash_scale_context: ScaleContext::new(),
            pending_glyph_images: HashMap::default(),
            loaded_fonts: Vec::new(),
            loaded_font_ids_by_key: HashMap::default(),
            font_ids_by_family_cache: HashMap::default(),
            system_font_fallback: system_font_fallback.to_string(),
            missing_glyph_sink: None,
            emoji_face,
        }))
    }

    pub fn new_without_system_fonts(system_font_fallback: &str) -> Self {
        let font_system = FontSystem::new_with_locale_and_db(
            "en-US".to_string(),
            cosmic_text::fontdb::Database::new(),
        );
        let emoji_face = find_emoji_face(font_system.db());

        Self(RwLock::new(CosmicTextSystemState {
            font_system,
            scratch: ShapeBuffer::default(),
            swash_scale_context: ScaleContext::new(),
            pending_glyph_images: HashMap::default(),
            loaded_fonts: Vec::new(),
            loaded_font_ids_by_key: HashMap::default(),
            font_ids_by_family_cache: HashMap::default(),
            system_font_fallback: system_font_fallback.to_string(),
            missing_glyph_sink: None,
            emoji_face,
        }))
    }
}

impl PlatformTextSystem for CosmicTextSystem {
    fn add_fonts(&self, fonts: Vec<Cow<'static, [u8]>>) -> Result<()> {
        self.0.write().add_fonts(fonts)
    }

    fn set_missing_glyph_sink(&self, sink: Option<Arc<dyn MissingGlyphSink>>) {
        self.0.write().missing_glyph_sink = sink;
    }

    fn all_font_names(&self) -> Vec<String> {
        let mut result = self
            .0
            .read()
            .font_system
            .db()
            .faces()
            .filter_map(|face| face.families.first().map(|family| family.0.clone()))
            .collect_vec();
        result.sort_unstable();
        result.dedup();
        result
    }

    fn font_id(&self, font: &Font) -> Result<FontId> {
        let mut state = self.0.write();
        let key = FontKey::new(
            font.family.clone(),
            font.features.clone(),
            font.fallbacks.clone(),
        );
        let candidates = if let Some(font_ids) = state.font_ids_by_family_cache.get(&key) {
            font_ids.as_slice()
        } else {
            let font_ids =
                state.load_family(&font.family, &font.features, font.fallbacks.as_ref())?;
            state.font_ids_by_family_cache.insert(key.clone(), font_ids);
            state.font_ids_by_family_cache[&key].as_ref()
        };

        let ix = find_best_match(font, candidates, &state)?;
        let font_id = candidates[ix];

        state.font_id_at_weight(font_id, &key, font.weight)
    }

    fn prewarm_fonts(&self, font_ids: &[FontId]) {
        self.0.write().prewarm_fonts(font_ids);
    }

    fn font_metrics(&self, font_id: FontId) -> FontMetrics {
        let state = self.0.read();
        let loaded_font = state.loaded_font(font_id);
        let metrics = loaded_font.font.as_swash().metrics(&loaded_font.coords);

        FontMetrics {
            units_per_em: metrics.units_per_em as u32,
            ascent: metrics.ascent,
            descent: -metrics.descent,
            line_gap: metrics.leading,
            underline_position: metrics.underline_offset,
            underline_thickness: metrics.stroke_size,
            cap_height: metrics.cap_height,
            x_height: metrics.x_height,
            bounding_box: Bounds {
                origin: point(0.0, 0.0),
                size: size(metrics.max_width, metrics.ascent + metrics.descent),
            },
        }
    }

    fn typographic_bounds(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Bounds<f32>> {
        let lock = self.0.read();
        let loaded_font = lock.loaded_font(font_id);
        let glyph_metrics = loaded_font
            .font
            .as_swash()
            .glyph_metrics(&loaded_font.coords);
        let glyph_id = glyph_id.0 as u16;
        Ok(Bounds {
            origin: point(0.0, 0.0),
            size: size(
                glyph_metrics.advance_width(glyph_id),
                glyph_metrics.advance_height(glyph_id),
            ),
        })
    }

    fn advance(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Size<f32>> {
        self.0.read().advance(font_id, glyph_id)
    }

    fn glyph_for_char(&self, font_id: FontId, ch: char) -> Option<GlyphId> {
        self.0.read().glyph_for_char(font_id, ch)
    }

    fn glyph_raster_bounds(&self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>> {
        self.0.write().raster_bounds(params)
    }

    fn rasterize_glyph(
        &self,
        params: &RenderGlyphParams,
        raster_bounds: Bounds<DevicePixels>,
    ) -> Result<(Size<DevicePixels>, Vec<u8>)> {
        self.0.write().rasterize_glyph(params, raster_bounds)
    }

    fn layout_line(&self, text: &str, font_size: Pixels, runs: &[FontRun]) -> LineLayout {
        self.0.write().layout_line(text, font_size, runs)
    }

    fn recommended_rendering_mode(
        &self,
        _font_id: FontId,
        _font_size: Pixels,
    ) -> TextRenderingMode {
        TextRenderingMode::Subpixel
    }
}

impl CosmicTextSystemState {
    fn loaded_font(&self, font_id: FontId) -> &LoadedFont {
        &self.loaded_fonts[font_id.0]
    }

    /// Returns the index of `font`'s face in its font file when the face has COLRv1 glyphs.
    fn color_face(&self, font: &CosmicTextFont) -> Option<u32> {
        let face = self.font_system.db().face(font.id())?;
        has_colr_v1_glyphs(font.as_swash()).then_some(face.index)
    }

    fn font_match_properties(&self, font_id: FontId) -> Option<FontMatchProperties> {
        let loaded_font = self.loaded_font(font_id);
        let Some(face) = self.font_system.db().face(loaded_font.font.id()) else {
            log::warn!("font face not found in database for font_id {:?}", font_id);
            return None;
        };
        let Some(first_family) = face.families.first() else {
            log::warn!("font face has no family names for font_id {:?}", font_id);
            return None;
        };

        Some(FontMatchProperties {
            primary_family_name: first_family.0.clone().into(),
            stretch: face.stretch,
            style: face.style,
            weight: loaded_font.weight,
            features: loaded_font.features.clone(),
            fallback_chain: Arc::clone(&loaded_font.user_fallback_chain),
        })
    }

    fn prewarm_fonts(&mut self, font_ids: &[FontId]) {
        for &font_id in font_ids {
            let Some(properties) = self.font_match_properties(font_id) else {
                continue;
            };
            let primary_attributes =
                properties.attributes(font_id, &properties.primary_family_name);
            self.font_system.get_font_matches(&primary_attributes);

            for (fallback_id, fallback_name) in &*properties.fallback_chain {
                let fallback_attributes = properties.attributes(*fallback_id, fallback_name);
                self.font_system.get_font_matches(&fallback_attributes);
            }
        }
    }

    #[profiling::function]
    fn add_fonts(&mut self, fonts: Vec<Cow<'static, [u8]>>) -> Result<()> {
        self.font_ids_by_family_cache.clear();
        let db = self.font_system.db_mut();
        for bytes in fonts {
            let added = db.load_font_source(cosmic_text::fontdb::Source::Binary(Arc::new(bytes)));
            for id in added {
                remove_other_faces_of_emoji_family(db, id, &self.loaded_fonts);
            }
        }
        self.emoji_face = find_emoji_face(self.font_system.db());
        Ok(())
    }

    fn emoji_font(&mut self) -> Option<(FontId, SharedString)> {
        let (face_id, family) = self.emoji_face.clone()?;
        match self.font_id_for_cosmic_id(face_id) {
            std::result::Result::Ok(font_id) => Some((font_id, family)),
            Err(error) => {
                log::warn!("failed to load emoji font {family}: {error:#}");
                None
            }
        }
    }

    #[profiling::function]
    fn load_family(
        &mut self,
        name: &str,
        features: &FontFeatures,
        fallbacks: Option<&FontFallbacks>,
    ) -> Result<SmallVec<[FontId; 4]>> {
        let loaded_font_key = FontKey::new(
            SharedString::from(name.to_owned()),
            features.clone(),
            fallbacks.cloned(),
        );

        // recurse with `fallbacks = None` so a fallback family cannot pull in
        // another chain. missing fallback families are dropped so a typo in
        // settings still lets the primary family load.
        let user_fallback_chain: Arc<[(FontId, SharedString)]> = match fallbacks {
            Some(fallbacks) if !fallbacks.fallback_list().is_empty() => {
                let mut chain: Vec<(FontId, SharedString)> = Vec::new();
                for fallback_name in fallbacks.fallback_list() {
                    let fb_key = FontKey::new(
                        SharedString::from(fallback_name.clone()),
                        features.clone(),
                        None,
                    );
                    let fb_ids = if let Some(cached) = self.font_ids_by_family_cache.get(&fb_key) {
                        cached.clone()
                    } else {
                        let loaded = self.load_family(fallback_name, features, None)?;
                        self.font_ids_by_family_cache.insert(fb_key, loaded.clone());
                        loaded
                    };
                    let Some(&fb_id) = fb_ids.first() else {
                        continue;
                    };
                    let db_id = self.loaded_fonts[fb_id.0].font.id();
                    if let Some(face) = self.font_system.db().face(db_id)
                        && let Some(family) = face.families.first()
                    {
                        chain.push((fb_id, SharedString::from(family.0.clone())));
                    }
                }
                Arc::from(chain)
            }
            _ => Arc::from(Vec::new()),
        };

        let name = gpui::font_name_with_fallbacks(name, &self.system_font_fallback);

        let families = self
            .font_system
            .db()
            .faces()
            .filter(|face| face.families.iter().any(|family| *name == family.0))
            .map(|face| (face.id, face.post_script_name.clone(), face.weight))
            .collect::<SmallVec<[_; 4]>>();

        let cosmic_features = cosmic_font_features(features)?;

        let mut loaded_font_ids = SmallVec::new();
        for (database_id, postscript_name, weight) in families {
            let key = LoadedFontKey {
                database_id,
                font: loaded_font_key.clone(),
                weight,
            };
            if let Some(&font_id) = self.loaded_font_ids_by_key.get(&key) {
                self.loaded_fonts[font_id.0].user_fallback_chain = Arc::clone(&user_fallback_chain);
                loaded_font_ids.push(font_id);
                continue;
            }

            let font = self
                .font_system
                .get_font(database_id, cosmic_text::Weight::NORMAL)
                .context("Could not load font")?;

            // HACK: To let the storybook run and render Windows caption icons. We should actually do better font fallback.
            let allowed_bad_font_names = [
                "SegoeFluentIcons", // NOTE: Segoe fluent icons postscript name is inconsistent
                "Segoe Fluent Icons",
            ];

            // This only decides whether a face can be a primary font. The
            // editor measures layout with the width of 'm' and would panic
            // without it.
            //
            // Don't remove the face from the font database. Arabic, Devanagari
            // and emoji fonts often have no Latin glyphs, and they are exactly
            // what cosmic-text falls back to when the primary font lacks a
            // glyph.
            if font.as_swash().charmap().map('m') == 0
                && !allowed_bad_font_names.contains(&postscript_name.as_str())
            {
                continue;
            };

            let font_id = FontId(self.loaded_fonts.len());
            loaded_font_ids.push(font_id);
            let color_face = self.color_face(&font);
            self.loaded_fonts.push(LoadedFont {
                coords: weight_coords(&font, weight),
                font,
                weight,
                features: cosmic_features.clone(),
                is_known_emoji_font: check_is_known_emoji_font(&postscript_name),
                color_face,
                user_fallback_chain: Arc::clone(&user_fallback_chain),
            });
            self.loaded_font_ids_by_key.insert(key, font_id);
        }

        Ok(loaded_font_ids)
    }

    /// Returns the font of `font_id`'s face at `weight` when the face has a weight
    /// axis, and `font_id` otherwise.
    fn font_id_at_weight(
        &mut self,
        font_id: FontId,
        font: &FontKey,
        weight: gpui::FontWeight,
    ) -> Result<FontId> {
        let loaded_font = self.loaded_font(font_id);
        let Some(axis) = loaded_font
            .font
            .as_swash()
            .variations()
            .find_by_tag(WEIGHT_AXIS)
        else {
            return Ok(font_id);
        };
        let weight = Weight(weight.0.clamp(axis.min_value(), axis.max_value()).round() as u16);
        if weight == loaded_font.weight {
            return Ok(font_id);
        }
        let key = LoadedFontKey {
            database_id: loaded_font.font.id(),
            font: font.clone(),
            weight,
        };
        let user_fallback_chain = Arc::clone(&loaded_font.user_fallback_chain);
        let features = loaded_font.features.clone();
        let is_known_emoji_font = loaded_font.is_known_emoji_font;
        let color_face = loaded_font.color_face;
        if let Some(&instance_id) = self.loaded_font_ids_by_key.get(&key) {
            self.loaded_fonts[instance_id.0].user_fallback_chain = user_fallback_chain;
            return Ok(instance_id);
        }

        let instance = self
            .font_system
            .get_font(key.database_id, weight)
            .context("Could not load font")?;
        let instance_id = FontId(self.loaded_fonts.len());
        self.loaded_fonts.push(LoadedFont {
            coords: weight_coords(&instance, weight),
            font: instance,
            weight,
            features,
            is_known_emoji_font,
            color_face,
            user_fallback_chain,
        });
        self.loaded_font_ids_by_key.insert(key, instance_id);
        Ok(instance_id)
    }

    fn advance(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Size<f32>> {
        let loaded_font = self.loaded_font(font_id);
        let glyph_metrics = loaded_font
            .font
            .as_swash()
            .glyph_metrics(&loaded_font.coords);
        Ok(Size {
            width: glyph_metrics.advance_width(glyph_id.0 as u16),
            height: glyph_metrics.advance_height(glyph_id.0 as u16),
        })
    }

    fn glyph_for_char(&self, font_id: FontId, ch: char) -> Option<GlyphId> {
        let glyph_id = self.loaded_font(font_id).font.as_swash().charmap().map(ch);
        if glyph_id == 0 {
            None
        } else {
            Some(GlyphId(glyph_id.into()))
        }
    }

    fn raster_bounds(&mut self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>> {
        let image = self.render_glyph_image(params)?;
        let bounds = Bounds {
            origin: point(image.placement.left.into(), (-image.placement.top).into()),
            size: size(image.placement.width.into(), image.placement.height.into()),
        };
        if !bounds.is_zero() {
            self.pending_glyph_images.insert(params.clone(), image);
        }
        Ok(bounds)
    }

    #[profiling::function]
    fn rasterize_glyph(
        &mut self,
        params: &RenderGlyphParams,
        glyph_bounds: Bounds<DevicePixels>,
    ) -> Result<(Size<DevicePixels>, Vec<u8>)> {
        if glyph_bounds.size.width.0 == 0 || glyph_bounds.size.height.0 == 0 {
            anyhow::bail!("glyph bounds are empty");
        }

        let mut image = match self.pending_glyph_images.remove(params) {
            Some(image) => image,
            None => self.render_glyph_image(params)?,
        };
        let bitmap_size = glyph_bounds.size;
        match image.content {
            swash::scale::image::Content::Color | swash::scale::image::Content::SubpixelMask => {
                // Convert from RGBA to BGRA.
                for pixel in image.data.chunks_exact_mut(4) {
                    pixel.swap(0, 2);
                }
                Ok((bitmap_size, image.data))
            }
            swash::scale::image::Content::Mask => {
                if params.subpixel_rendering {
                    // We must always return RGBA data when subpixel rendering is requested.
                    let expanded = image.data.iter().flat_map(|&a| [a, a, a, a]).collect();
                    Ok((bitmap_size, expanded))
                } else {
                    Ok((bitmap_size, image.data))
                }
            }
        }
    }

    fn render_glyph_image(
        &mut self,
        params: &RenderGlyphParams,
    ) -> Result<swash::scale::image::Image> {
        let loaded_font = &self.loaded_fonts[params.font_id.0];
        let font_ref = loaded_font.font.as_swash();
        let pixel_size = if params.is_emoji && params.glyph_id.0 != 0 {
            f32::from(params.font_size) * emoji_scale(f32::from(params.font_size))
        } else {
            f32::from(params.font_size)
        };
        // swash draws COLRv1 glyphs blank, so gpui_emoji's painter draws them.
        if let Some(index) = loaded_font.color_face {
            let data = loaded_font.font.data();
            if params.is_emoji && gpui_emoji::has_color_glyph(data, index, params.glyph_id.0) {
                let pixels = pixel_size * params.scale_factor;
                match gpui_emoji::paint_color_glyph(data, index, params.glyph_id.0, pixels) {
                    Some(image) => return Ok(color_image(image)),
                    None => log::warn!("gpui_emoji could not paint {params:?}"),
                }
            }
        }

        let subpixel_offset = Vector::new(
            params.subpixel_variant.x as f32 / SUBPIXEL_VARIANTS_X as f32 / params.scale_factor,
            params.subpixel_variant.y as f32 / SUBPIXEL_VARIANTS_Y as f32 / params.scale_factor,
        );

        let mut scaler = self
            .swash_scale_context
            .builder(font_ref)
            .size(pixel_size * params.scale_factor)
            .normalized_coords(&loaded_font.coords)
            .hint(true)
            .build();

        let sources: &[Source] = if params.is_emoji {
            &[
                Source::ColorOutline(0),
                Source::ColorBitmap(StrikeWith::BestFit),
                Source::Outline,
            ]
        } else {
            &[Source::Bitmap(StrikeWith::ExactSize), Source::Outline]
        };

        let mut renderer = Render::new(sources);
        if params.subpixel_rendering {
            // There seems to be a bug in Swash where the B and R values are swapped.
            renderer
                .format(Format::subpixel_bgra())
                .offset(subpixel_offset);
        } else {
            renderer.format(Format::Alpha).offset(subpixel_offset);
        }

        let glyph_id: u16 = params.glyph_id.0.try_into()?;
        renderer
            .render(&mut scaler, glyph_id)
            .with_context(|| format!("unable to render glyph via swash for {params:?}"))
    }

    /// This is used when cosmic_text has chosen a fallback font instead of using the requested
    /// font, typically to handle some unicode characters. When this happens, `loaded_fonts` may not
    /// yet have an entry for this fallback font, and so one is added.
    ///
    /// Note that callers shouldn't use this `FontId` somewhere that will retrieve the corresponding
    /// `LoadedFont.features`, as it will have an arbitrarily chosen or empty value. The only
    /// current use of this field is for the *input* of `layout_line`, and so it's fine to use
    /// `font_id_for_cosmic_id` when computing the *output* of `layout_line`.
    fn font_id_for_cosmic_id(&mut self, id: cosmic_text::fontdb::ID) -> Result<FontId> {
        if let Some(ix) = self
            .loaded_fonts
            .iter()
            .position(|loaded_font| loaded_font.font.id() == id)
        {
            Ok(FontId(ix))
        } else {
            let font = self
                .font_system
                .get_font(id, cosmic_text::Weight::NORMAL)
                .context("failed to get fallback font from cosmic-text font system")?;
            let face = self
                .font_system
                .db()
                .face(id)
                .context("fallback font face not found in cosmic-text database")?;

            let is_emoji_face = self
                .emoji_face
                .as_ref()
                .is_some_and(|(emoji_face_id, _)| *emoji_face_id == id);
            let font_id = FontId(self.loaded_fonts.len());
            let color_face = self.color_face(&font);
            self.loaded_fonts.push(LoadedFont {
                coords: weight_coords(&font, face.weight),
                font,
                weight: face.weight,
                features: CosmicFontFeatures::new(),
                is_known_emoji_font: is_emoji_face
                    || check_is_known_emoji_font(&face.post_script_name),
                color_face,
                user_fallback_chain: Arc::from(Vec::new()),
            });

            Ok(font_id)
        }
    }

    #[profiling::function]
    fn layout_line(&mut self, text: &str, font_size: Pixels, font_runs: &[FontRun]) -> LineLayout {
        if contains_paragraph_separator(text) {
            self.layout_line_with_separators(text, font_size, font_runs)
        } else {
            self.layout_line_no_separators(text, font_size, font_runs)
        }
    }

    fn layout_line_with_separators(
        &mut self,
        text: &str,
        font_size: Pixels,
        font_runs: &[FontRun],
    ) -> LineLayout {
        let mut layout = LineLayout {
            font_size,
            len: text.len(),
            ..Default::default()
        };
        let mut paragraph_start = 0;

        for (separator_start, separator) in text
            .char_indices()
            .filter(|(_, character)| is_paragraph_separator(*character))
        {
            let separator_end = separator_start + separator.len_utf8();
            self.shape_segment(
                text,
                paragraph_start..separator_start,
                font_size,
                font_runs,
                &mut layout,
            );
            self.shape_segment(
                text,
                separator_start..separator_end,
                font_size,
                font_runs,
                &mut layout,
            );
            paragraph_start = separator_end;
        }

        self.shape_segment(
            text,
            paragraph_start..text.len(),
            font_size,
            font_runs,
            &mut layout,
        );

        layout
    }

    fn shape_segment(
        &mut self,
        text: &str,
        range: Range<usize>,
        font_size: Pixels,
        font_runs: &[FontRun],
        layout: &mut LineLayout,
    ) {
        if range.is_empty() {
            return;
        }

        let segment_font_runs = clip_font_runs(font_runs, range.clone());
        let segment =
            self.layout_line_no_separators(&text[range.clone()], font_size, &segment_font_runs);

        let mut segment_runs = segment.runs;
        for run in &mut segment_runs {
            for glyph in &mut run.glyphs {
                glyph.index += range.start;
                glyph.position.x += layout.width;
            }
        }

        for mut run in segment_runs {
            if let Some(same_run) = layout
                .runs
                .last_mut()
                .filter(|last| last.font_id == run.font_id)
            {
                same_run.glyphs.append(&mut run.glyphs);
            } else {
                layout.runs.push(run);
            }
        }

        layout.width += segment.width;
        layout.ascent = layout.ascent.max(segment.ascent);
        layout.descent = layout.descent.max(segment.descent);
    }

    fn layout_line_no_separators(
        &mut self,
        text: &str,
        font_size: Pixels,
        font_runs: &[FontRun],
    ) -> LineLayout {
        let emoji_font = if text.is_ascii() {
            None
        } else {
            self.emoji_font()
        };
        let mut attrs_list = AttrsList::new(&Attrs::new());
        let mut offs = 0;
        for run in font_runs {
            let run_end = offs + run.len;

            let Some(properties) = self.font_match_properties(run.font_id) else {
                offs = run_end;
                continue;
            };

            let primary_attrs = properties.attributes(run.font_id, &properties.primary_family_name);
            let fallback_attrs: SmallVec<[Attrs<'_>; 4]> = properties
                .fallback_chain
                .iter()
                .map(|(font_id, family_name)| properties.attributes(*font_id, family_name))
                .collect();

            let spans = if properties.fallback_chain.is_empty() {
                let mut spans = SmallVec::<[RunSpan; 4]>::new();
                spans.push(RunSpan {
                    start: offs,
                    end: run_end,
                    slot: None,
                    font_id: run.font_id,
                });
                spans
            } else {
                let loaded_fonts = &self.loaded_fonts;
                let covers = |id: FontId, ch: char| charmap_covers(loaded_fonts, id, ch);
                compute_run_spans(
                    text,
                    offs,
                    run.len,
                    run.font_id,
                    &properties.fallback_chain,
                    &covers,
                )
            };

            for span in spans {
                let attrs = match span.slot {
                    None => &primary_attrs,
                    Some(ix) => &fallback_attrs[ix],
                };
                attrs_list.add_span(span.start..span.end, attrs);
            }

            if let Some((emoji_font_id, emoji_family)) = &emoji_font {
                // An emoji font has a single face, and the run's weight or style would
                // make cosmic-text pass over it.
                let emoji_attrs = Attrs::new()
                    .metadata(emoji_font_id.0)
                    .family(Family::Name(emoji_family));
                let loaded_fonts = &self.loaded_fonts;
                let primary_covers = |ch| charmap_covers(loaded_fonts, run.font_id, ch);
                let emoji_font_covers = |ch| charmap_covers(loaded_fonts, *emoji_font_id, ch);
                for cluster in emoji_presentation_clusters(
                    text,
                    offs..run_end,
                    &primary_covers,
                    &emoji_font_covers,
                ) {
                    attrs_list.add_span(cluster, &emoji_attrs);
                }
            }
            offs = run_end;
        }

        let line = ShapeLine::new(
            &mut self.font_system,
            text,
            &attrs_list,
            cosmic_text::Shaping::Advanced,
            4,
        );
        let mut layout_lines = Vec::with_capacity(1);
        line.layout_to_buffer(
            &mut self.scratch,
            f32::from(font_size),
            None, // We do our own wrapping
            cosmic_text::Wrap::None,
            Ellipsize::None,
            None,
            &mut layout_lines,
            None,
            cosmic_text::Hinting::Disabled,
        );

        let Some(layout) = layout_lines.first() else {
            return LineLayout {
                font_size,
                width: Pixels::ZERO,
                ascent: Pixels::ZERO,
                descent: Pixels::ZERO,
                runs: Vec::new(),
                len: text.len(),
            };
        };

        let missing_glyphs = self.missing_glyph_sink.as_ref().map(|_| {
            self.missing_glyphs(
                text,
                font_runs,
                layout
                    .glyphs
                    .iter()
                    .filter(|glyph| glyph.glyph_id == 0)
                    .map(|glyph| glyph.start),
            )
        });

        let emoji_scale = emoji_scale(f32::from(font_size));
        // Emoji draw larger than shaped. Each entry holds an emoji's left edge and the advance
        // it gains, and the line's ascent and descent grow with the emoji font's.
        let mut emoji_growth: SmallVec<[(f32, f32); 4]> = SmallVec::new();
        let mut emoji_ascent: f32 = 0.;
        let mut emoji_descent: f32 = 0.;
        let mut runs: Vec<ShapedRun> = Vec::new();
        for glyph in &layout.glyphs {
            let mut font_id = FontId(glyph.metadata);
            let mut loaded_font = self.loaded_font(font_id);
            if loaded_font.font.id() != glyph.font_id {
                match self.font_id_for_cosmic_id(glyph.font_id) {
                    std::result::Result::Ok(resolved_id) => {
                        font_id = resolved_id;
                        loaded_font = self.loaded_font(font_id);
                    }
                    Err(error) => {
                        log::warn!(
                            "failed to resolve cosmic font id {:?}: {error:#}",
                            glyph.font_id
                        );
                        continue;
                    }
                }
            }
            let is_emoji = loaded_font.is_known_emoji_font;

            // Shaping replaces a variation selector the emoji font leaves unused with the
            // font's space glyph, and a bitmap-only emoji font has no image for it.
            if is_emoji
                && glyph.glyph_id != 0
                && glyph.glyph_id == loaded_font.font.as_swash().charmap().map(' ')
            {
                continue;
            }

            if is_emoji && glyph.glyph_id != 0 && emoji_scale > 1. {
                emoji_growth.push((glyph.x, glyph.w * (emoji_scale - 1.)));
                let metrics = loaded_font.font.as_swash().metrics(&[]);
                let units = f32::from(font_size) * emoji_scale / f32::from(metrics.units_per_em);
                emoji_ascent = emoji_ascent.max(metrics.ascent * units);
                emoji_descent = emoji_descent.max(metrics.descent * units);
            }

            let shaped_glyph = ShapedGlyph {
                id: GlyphId(glyph.glyph_id as u32),
                position: point(
                    (glyph.x + glyph.x_offset * f32::from(font_size)).into(),
                    (glyph.y - glyph.y_offset * f32::from(font_size)).into(),
                ),
                index: glyph.start,
                is_emoji,
            };

            if let Some(last_run) = runs
                .last_mut()
                .filter(|last_run| last_run.font_id == font_id)
            {
                last_run.glyphs.push(shaped_glyph);
            } else {
                runs.push(ShapedRun {
                    font_id,
                    glyphs: vec![shaped_glyph],
                });
            }
        }

        // Every glyph moves right by the advance the emoji left of it gain. A right-to-left
        // line lists its glyphs from the right, so the emoji are sorted by position first.
        emoji_growth.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut growth = 0.;
        for (_, added) in &mut emoji_growth {
            growth += *added;
            *added = growth;
        }
        for glyph in runs.iter_mut().flat_map(|run| run.glyphs.iter_mut()) {
            let x = f32::from(glyph.position.x);
            let left = emoji_growth.partition_point(|(emoji_x, _)| *emoji_x < x);
            if let Some((_, shift)) = left
                .checked_sub(1)
                .and_then(|index| emoji_growth.get(index))
            {
                glyph.position.x += Pixels::from(*shift);
            }
        }

        if let Some((sink, missing_glyphs)) = self.missing_glyph_sink.as_ref().zip(missing_glyphs) {
            sink.report(missing_glyphs);
        }

        LineLayout {
            font_size,
            width: (layout.w + growth).into(),
            ascent: layout.max_ascent.max(emoji_ascent).into(),
            descent: layout.max_descent.max(emoji_descent).into(),
            runs,
            len: text.len(),
        }
    }

    fn missing_glyphs(
        &self,
        text: &str,
        font_runs: &[FontRun],
        missing_text_indices: impl IntoIterator<Item = usize>,
    ) -> Vec<MissingGlyph> {
        let mut missing_text_indices = missing_text_indices.into_iter().peekable();
        if missing_text_indices.peek().is_none() {
            return Vec::new();
        }
        let mut missing_text_indices = missing_text_indices.collect::<Vec<_>>();
        missing_text_indices.sort_unstable();
        missing_text_indices.dedup();

        let mut font_run_index = 0;
        let mut font_run_end = font_runs.first().map_or(0, |font_run| font_run.len);
        let mut missing_glyphs = Vec::new();
        let mut missing_index = 0;
        for (grapheme_start, grapheme) in text.grapheme_indices(true) {
            let grapheme_end = grapheme_start + grapheme.len();
            while missing_text_indices
                .get(missing_index)
                .is_some_and(|text_index| *text_index < grapheme_start)
            {
                missing_index += 1;
            }
            let Some(&text_index) = missing_text_indices.get(missing_index) else {
                break;
            };
            if text_index >= grapheme_end {
                continue;
            }

            while font_run_end <= text_index && font_run_index + 1 < font_runs.len() {
                font_run_index += 1;
                let Some(font_run) = font_runs.get(font_run_index) else {
                    break;
                };
                font_run_end += font_run.len;
            }
            let font_class = self.fallback_font_class(
                font_runs
                    .get(font_run_index)
                    .or_else(|| font_runs.last())
                    .map(|font_run| font_run.font_id),
            );
            missing_glyphs.push(MissingGlyph::new(grapheme.into(), font_class));
            while missing_text_indices
                .get(missing_index)
                .is_some_and(|text_index| *text_index < grapheme_end)
            {
                missing_index += 1;
            }
        }
        missing_glyphs
    }

    fn fallback_font_class(&self, font_id: Option<FontId>) -> FallbackFontClass {
        let Some(font_id) = font_id else {
            return FallbackFontClass::Proportional;
        };
        let loaded_font = self.loaded_font(font_id);
        let is_monospace = self
            .font_system
            .db()
            .face(loaded_font.font.id())
            .is_some_and(|face| face.monospaced);
        if is_monospace {
            FallbackFontClass::Monospace
        } else {
            FallbackFontClass::Proportional
        }
    }
}

#[inline(always)]
/// Returns `font`'s variation coordinates with its weight axis at `weight`.
fn weight_coords(font: &CosmicTextFont, weight: Weight) -> Vec<NormalizedCoord> {
    font.as_swash()
        .variations()
        .normalized_coords([(WEIGHT_AXIS, f32::from(weight.0))])
        .collect()
}

fn is_paragraph_separator(character: char) -> bool {
    unicode_bidi::bidi_class(character) == unicode_bidi::BidiClass::B
}

fn contains_paragraph_separator(text: &str) -> bool {
    if text
        .bytes()
        .any(|byte| matches!(byte, b'\n' | b'\r' | 0x1c | 0x1d | 0x1e))
    {
        return true;
    }

    !text.is_ascii() && text.chars().any(is_paragraph_separator)
}

fn clip_font_runs(font_runs: &[FontRun], range: Range<usize>) -> SmallVec<[FontRun; 4]> {
    let mut clipped = SmallVec::new();
    let mut offs = 0;
    for run in font_runs {
        let run_start = offs;
        offs += run.len;
        if offs <= range.start {
            continue;
        }
        if run_start >= range.end {
            break;
        }
        let start = run_start.max(range.start);
        let end = offs.min(range.end);
        if start < end {
            clipped.push(FontRun {
                len: end - start,
                font_id: run.font_id,
            });
        }
    }
    clipped
}

#[cfg(feature = "font-kit")]
fn find_best_match(
    font: &Font,
    candidates: &[FontId],
    state: &CosmicTextSystemState,
) -> Result<usize> {
    let candidate_properties = candidates
        .iter()
        .map(|font_id| {
            let database_id = state.loaded_font(*font_id).font.id();
            let face_info = state
                .font_system
                .db()
                .face(database_id)
                .context("font face not found in database")?;
            Ok(face_info_into_properties(face_info))
        })
        .collect::<Result<SmallVec<[_; 4]>>>()?;

    let ix =
        font_kit::matching::find_best_match(&candidate_properties, &font_into_properties(font))
            .context("requested font family contains no font matching the other parameters")?;

    Ok(ix)
}

#[cfg(not(feature = "font-kit"))]
fn find_best_match(
    font: &Font,
    candidates: &[FontId],
    state: &CosmicTextSystemState,
) -> Result<usize> {
    if candidates.is_empty() {
        anyhow::bail!("requested font family contains no font matching the other parameters");
    }
    if candidates.len() == 1 {
        return Ok(0);
    }

    let target_weight = font.weight.0;
    let target_italic = matches!(
        font.style,
        gpui::FontStyle::Italic | gpui::FontStyle::Oblique
    );

    let mut best_index = 0;
    let mut best_score = u32::MAX;

    for (index, font_id) in candidates.iter().enumerate() {
        let database_id = state.loaded_font(*font_id).font.id();
        let face_info = state
            .font_system
            .db()
            .face(database_id)
            .context("font face not found in database")?;

        let is_italic = matches!(
            face_info.style,
            cosmic_text::Style::Italic | cosmic_text::Style::Oblique
        );
        let style_penalty: u32 = if is_italic == target_italic { 0 } else { 1000 };
        let weight_diff = (face_info.weight.0 as i32 - target_weight as i32).unsigned_abs();
        let score = style_penalty + weight_diff;

        if score < best_score {
            best_score = score;
            best_index = index;
        }
    }

    Ok(best_index)
}

/// one contiguous slice of a `FontRun` that maps to a single slot. `slot` is
/// `None` for the primary font and `Some(ix)` for `fallback_chain[ix]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RunSpan {
    start: usize,
    end: usize,
    slot: Option<usize>,
    font_id: FontId,
}

/// walks `text[run_offset..run_offset + run_len]` and groups codepoints into
/// spans. inheriting codepoints stay in the current span so shaping clusters
/// like emoji zwj sequences and combining marks are not torn apart.
fn compute_run_spans(
    text: &str,
    run_offset: usize,
    run_len: usize,
    primary: FontId,
    fallback_chain: &[(FontId, SharedString)],
    covers: &impl Fn(FontId, char) -> bool,
) -> SmallVec<[RunSpan; 4]> {
    let mut spans = SmallVec::new();
    let run_end = run_offset + run_len;
    if run_end <= run_offset {
        return spans;
    }
    if fallback_chain.is_empty() {
        spans.push(RunSpan {
            start: run_offset,
            end: run_end,
            slot: None,
            font_id: primary,
        });
        return spans;
    }
    let run_text = &text[run_offset..run_end];
    let mut span_start = run_offset;
    let mut span_slot: Option<usize> = None;
    let mut span_font_id = primary;
    for (grapheme_idx, grapheme) in run_text.grapheme_indices(true) {
        let abs = run_offset + grapheme_idx;
        let ch = grapheme.chars().next().unwrap_or('\0');
        let next_slot = pick_covering_slot(ch, span_slot, primary, fallback_chain, covers);
        if next_slot == span_slot {
            continue;
        }
        if abs > span_start {
            spans.push(RunSpan {
                start: span_start,
                end: abs,
                slot: span_slot,
                font_id: span_font_id,
            });
        }
        span_start = abs;
        span_slot = next_slot;
        span_font_id = slot_font_id(next_slot, primary, fallback_chain);
    }
    if span_start < run_end {
        spans.push(RunSpan {
            start: span_start,
            end: run_end,
            slot: span_slot,
            font_id: span_font_id,
        });
    }
    spans
}

fn slot_font_id(
    slot: Option<usize>,
    primary: FontId,
    fallback_chain: &[(FontId, SharedString)],
) -> FontId {
    match slot {
        None => primary,
        Some(ix) => fallback_chain[ix].0,
    }
}

fn pick_covering_slot(
    ch: char,
    current: Option<usize>,
    primary: FontId,
    fallback_chain: &[(FontId, SharedString)],
    covers: &impl Fn(FontId, char) -> bool,
) -> Option<usize> {
    if (ch as u32) <= 0x7F {
        return None;
    }
    if covers(primary, ch) {
        return None;
    }
    let current_id = slot_font_id(current, primary, fallback_chain);
    if covers(current_id, ch) {
        return current;
    }

    fallback_chain
        .iter()
        .position(|(fb_id, _)| covers(*fb_id, ch))
}

fn charmap_covers(loaded_fonts: &[LoadedFont], id: FontId, ch: char) -> bool {
    loaded_fonts
        .get(id.0)
        .is_some_and(|loaded| loaded.font.as_swash().charmap().map(ch) != 0)
}

fn cosmic_font_features(features: &FontFeatures) -> Result<CosmicFontFeatures> {
    let mut result = CosmicFontFeatures::new();
    for feature in features.0.iter() {
        let name_bytes: [u8; 4] = feature
            .0
            .as_bytes()
            .try_into()
            .context("Incorrect feature flag format")?;

        let tag = cosmic_text::FeatureTag::new(&name_bytes);

        result.set(tag, feature.1);
    }
    Ok(result)
}

#[cfg(feature = "font-kit")]
fn font_into_properties(font: &gpui::Font) -> font_kit::properties::Properties {
    font_kit::properties::Properties {
        style: match font.style {
            gpui::FontStyle::Normal => font_kit::properties::Style::Normal,
            gpui::FontStyle::Italic => font_kit::properties::Style::Italic,
            gpui::FontStyle::Oblique => font_kit::properties::Style::Oblique,
        },
        weight: font_kit::properties::Weight(font.weight.0),
        stretch: Default::default(),
    }
}

#[cfg(feature = "font-kit")]
fn face_info_into_properties(
    face_info: &cosmic_text::fontdb::FaceInfo,
) -> font_kit::properties::Properties {
    font_kit::properties::Properties {
        style: match face_info.style {
            cosmic_text::Style::Normal => font_kit::properties::Style::Normal,
            cosmic_text::Style::Italic => font_kit::properties::Style::Italic,
            cosmic_text::Style::Oblique => font_kit::properties::Style::Oblique,
        },
        weight: font_kit::properties::Weight(face_info.weight.0.into()),
        stretch: match face_info.stretch {
            cosmic_text::Stretch::Condensed => font_kit::properties::Stretch::CONDENSED,
            cosmic_text::Stretch::Expanded => font_kit::properties::Stretch::EXPANDED,
            cosmic_text::Stretch::ExtraCondensed => font_kit::properties::Stretch::EXTRA_CONDENSED,
            cosmic_text::Stretch::ExtraExpanded => font_kit::properties::Stretch::EXTRA_EXPANDED,
            cosmic_text::Stretch::Normal => font_kit::properties::Stretch::NORMAL,
            cosmic_text::Stretch::SemiCondensed => font_kit::properties::Stretch::SEMI_CONDENSED,
            cosmic_text::Stretch::SemiExpanded => font_kit::properties::Stretch::SEMI_EXPANDED,
            cosmic_text::Stretch::UltraCondensed => font_kit::properties::Stretch::ULTRA_CONDENSED,
            cosmic_text::Stretch::UltraExpanded => font_kit::properties::Stretch::ULTRA_EXPANDED,
        },
    }
}

fn find_emoji_face(
    db: &cosmic_text::fontdb::Database,
) -> Option<(cosmic_text::fontdb::ID, SharedString)> {
    EMOJI_FONT_FAMILIES.iter().find_map(|&family| {
        let face = db.faces().find(|face| {
            face.families.iter().any(|(name, _)| name == family)
                && db
                    .with_face_data(face.id, |data, index| {
                        swash::FontRef::from_index(data, index as usize)
                            .is_some_and(has_paintable_color_glyphs)
                    })
                    .unwrap_or(false)
        })?;
        Some((face.id, SharedString::new_static(family)))
    })
}

/// Returns whether the font has color glyphs that GPUI paints: swash paints CBDT and sbix
/// bitmaps and COLR version 0 layers, and gpui_emoji's painter paints COLRv1 glyphs.
fn has_paintable_color_glyphs(font: swash::FontRef) -> bool {
    let table = |tag| font.table(swash::tag_from_bytes(tag));
    let has_colr_layers = table(b"COLR")
        .and_then(|colr| colr.get(2..4))
        .is_some_and(|base_glyph_count| base_glyph_count.iter().any(|byte| *byte != 0));
    table(b"CBDT").is_some()
        || table(b"sbix").is_some()
        || has_colr_layers
        || has_colr_v1_glyphs(font)
}

/// Returns whether the font has a COLR table of version 1 with a BaseGlyphList.
fn has_colr_v1_glyphs(font: swash::FontRef) -> bool {
    font.table(swash::tag_from_bytes(b"COLR"))
        .is_some_and(|colr| {
            colr.get(0..2) == Some(&[0, 1])
                && colr
                    .get(14..18)
                    .is_some_and(|offset| offset.iter().any(|byte| *byte != 0))
        })
}

/// Removes the faces of `id`'s emoji family other than `id` when GPUI can paint `id`, so
/// that the copy of an emoji font the app adds draws in place of the system's. A face that
/// `loaded` already uses stays.
fn remove_other_faces_of_emoji_family(
    db: &mut cosmic_text::fontdb::Database,
    id: cosmic_text::fontdb::ID,
    loaded: &[LoadedFont],
) {
    let is_paintable = db
        .with_face_data(id, |data, index| {
            swash::FontRef::from_index(data, index as usize).is_some_and(has_paintable_color_glyphs)
        })
        .unwrap_or(false);
    let family = db.face(id).and_then(|face| {
        face.families
            .iter()
            .map(|(name, _)| name.clone())
            .find(|name| EMOJI_FONT_FAMILIES.contains(&name.as_str()))
    });
    let Some(family) = family.filter(|_| is_paintable) else {
        return;
    };
    let others: Vec<_> = db
        .faces()
        .filter(|face| face.id != id && face.families.iter().any(|(name, _)| *name == family))
        .map(|face| face.id)
        .collect();
    for other in others {
        if loaded.iter().any(|font| font.font.id() == other) {
            log::warn!("the {family} face that text already uses stays beside the added copy");
            continue;
        }
        db.remove_face(other);
    }
}

/// Returns a glyph that gpui_emoji painted as swash's color image, which the atlas takes.
fn color_image(image: gpui_emoji::ColorImage) -> swash::scale::image::Image {
    swash::scale::image::Image {
        source: Source::ColorOutline(0),
        content: swash::scale::image::Content::Color,
        placement: swash::zeno::Placement {
            left: image.left,
            top: image.top,
            width: image.width,
            height: image.height,
        },
        data: image.rgba,
    }
}

fn check_is_known_emoji_font(postscript_name: &str) -> bool {
    // TODO: Include other common emoji fonts
    postscript_name == "NotoColorEmoji"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "test-support")]
    use std::{cell::RefCell, rc::Rc};
    use std::{
        sync::mpsc,
        thread,
        time::{Duration, Instant},
    };

    #[test]
    fn all_font_names_tracks_available_families() -> Result<()> {
        let text_system = gpui::TextSystem::new(Arc::new(
            CosmicTextSystem::new_without_system_fonts("IBM Plex Sans"),
        ));
        assert!(text_system.all_font_names().is_empty());

        text_system.add_fonts(vec![Cow::Borrowed(include_bytes!(
            "../../../assets/fonts/lilex/Lilex-Regular.ttf"
        ))])?;
        assert_eq!(text_system.all_font_names(), ["Lilex"]);

        text_system.add_fonts(vec![
            Cow::Borrowed(IBM_PLEX),
            Cow::Borrowed(include_bytes!("../../../assets/fonts/lilex/Lilex-Bold.ttf")),
        ])?;
        assert_eq!(text_system.all_font_names(), ["IBM Plex Sans", "Lilex"]);
        Ok(())
    }

    fn fid(i: usize) -> FontId {
        FontId(i)
    }

    fn chain(ids: &[usize]) -> SmallVec<[(FontId, SharedString); 4]> {
        ids.iter()
            .map(|&i| (fid(i), SharedString::from(format!("fb{i}"))))
            .collect()
    }

    fn span(start: usize, end: usize, slot: Option<usize>, font_id: FontId) -> RunSpan {
        RunSpan {
            start,
            end,
            slot,
            font_id,
        }
    }

    const IBM_PLEX: &[u8] =
        include_bytes!("../../../assets/fonts/ibm-plex-sans/IBMPlexSans-Regular.ttf");
    const LILEX: &[u8] = include_bytes!("../../../assets/fonts/lilex/Lilex-Regular.ttf");

    /// Every code point of `Bidi_Class=B`, each of which starts a new bidi
    /// paragraph and so can split one line into mixed-direction paragraphs.
    const SEPARATORS: &[char] = &[
        '\u{000a}', '\u{000d}', '\u{001c}', '\u{001d}', '\u{001e}', '\u{0085}', '\u{2029}',
    ];

    fn text_system() -> Result<CosmicTextSystem> {
        let text_system = CosmicTextSystem::new_without_system_fonts("IBM Plex Sans");
        text_system.add_fonts(vec![Cow::Borrowed(IBM_PLEX)])?;
        Ok(text_system)
    }

    #[test]
    fn cached_font_resolution_does_not_wait_for_cold_lookups() -> Result<()> {
        for family in [".ZedSans", "Missing Font"] {
            let platform_text_system = Arc::new(text_system()?);
            let text_system = gpui::TextSystem::new(platform_text_system.clone());
            let cached = gpui::font(family);
            let expected = platform_text_system.font_id(&gpui::font(".ZedSans"))?;
            assert_eq!(text_system.resolve_font(&cached), expected);
            let (sender, receiver) = mpsc::channel();

            thread::scope(|scope| {
                let backend = platform_text_system.0.read();
                let deadline = Instant::now() + Duration::from_secs(5);
                let cold = scope.spawn(|| text_system.resolve_font(&cached.clone().bold()));
                while !platform_text_system.0.is_locked_exclusive() && Instant::now() < deadline {
                    thread::yield_now();
                }
                let cold_waiting = platform_text_system.0.is_locked_exclusive();
                let warm = scope.spawn(|| sender.send(text_system.resolve_font(&cached)));
                let result =
                    receiver.recv_timeout(deadline.saturating_duration_since(Instant::now()));
                drop(backend);
                let cold_result = cold.join();
                let warm_result = warm.join();

                assert!(cold_waiting);
                assert_eq!(result.expect("cached lookup blocked"), expected);
                assert_eq!(cold_result.expect("cold lookup panicked"), expected);
                warm_result
                    .expect("warm lookup panicked")
                    .expect("receiver dropped");
            });
        }
        Ok(())
    }

    #[test]
    fn adding_fonts_retries_cached_font_failures() -> Result<()> {
        let platform_text_system = Arc::new(CosmicTextSystem::new_without_system_fonts("Lilex"));
        let text_system = gpui::TextSystem::new(platform_text_system.clone());
        text_system.add_fonts(vec![Cow::Borrowed(IBM_PLEX)])?;
        let original_fallback = platform_text_system.font_id(&gpui::font(".ZedSans"))?;
        let requests = [
            gpui::Font::default(),
            gpui::font("Lilex"),
            gpui::font("Missing Font"),
        ];
        for requested in &requests {
            assert_eq!(text_system.resolve_font(requested), original_fallback);
        }
        assert_eq!(
            text_system.get_font_for_id(original_fallback),
            Some(gpui::font(".ZedSans"))
        );

        text_system.add_fonts(vec![Cow::Borrowed(LILEX)])?;
        for (requested, resolved) in requests.iter().zip([".SystemUIFont", "Lilex", ".ZedMono"]) {
            let installed = platform_text_system.font_id(&gpui::font(resolved))?;
            assert_ne!(installed, original_fallback);
            assert_eq!(text_system.resolve_font(requested), installed);
        }
        Ok(())
    }

    #[test]
    fn font_properties_describe_the_selected_face() -> Result<()> {
        let text_system = text_system()?;
        let regular = gpui::font("IBM Plex Sans");
        let regular_id = text_system.font_id(&regular)?;
        for (weight, style) in [
            (gpui::FontWeight::MEDIUM, gpui::FontStyle::Normal),
            (gpui::FontWeight::BOLD, gpui::FontStyle::Italic),
            (gpui::FontWeight::NORMAL, gpui::FontStyle::Oblique),
        ] {
            let requested = Font {
                weight,
                style,
                ..regular.clone()
            };
            let font_id = text_system.font_id(&requested)?;
            assert_eq!(font_id, regular_id);
            assert_eq!(
                text_system.font_weight_and_style(font_id)?,
                (gpui::FontWeight::NORMAL, gpui::FontStyle::Normal)
            );
        }
        Ok(())
    }

    fn layout_text(text_system: &CosmicTextSystem, text: &str) -> Result<LineLayout> {
        let font_id = text_system.font_id(&gpui::font("IBM Plex Sans"))?;
        let runs = [FontRun {
            len: text.len(),
            font_id,
        }];
        Ok(text_system.layout_line(text, gpui::px(14.0), &runs))
    }

    #[test]
    fn combining_marks_are_positioned_over_their_base() -> Result<()> {
        let text_system = text_system()?;
        let layout = layout_text(&text_system, "v\u{0304}v")?;
        let glyphs = layout
            .runs
            .iter()
            .flat_map(|run| run.glyphs.iter())
            .collect::<Vec<_>>();
        assert_eq!(glyphs.len(), 3);
        let base = glyphs[0].position;
        let mark = glyphs[1].position;
        let next = glyphs[2].position;
        assert!(mark.x > base.x);
        assert!(
            mark.x < next.x,
            "combining mark positioned after its base: mark.x={:?}, next.x={:?}",
            mark.x,
            next.x
        );
        Ok(())
    }

    /// Mirrors the original crash: mixed-direction text reaching the shaper
    /// through `shape_text`, which only splits lines on `\n`.
    #[test]
    fn shape_text_with_mixed_direction_paragraphs() -> Result<()> {
        let platform_text_system = Arc::new(text_system()?);
        let text_system = Arc::new(gpui::TextSystem::new(platform_text_system));
        let window_text_system = gpui::WindowTextSystem::new(text_system);

        let text: SharedString = "first line\n\u{05d0}\u{001c}A".into();
        let runs = [gpui::TextRun {
            len: text.len(),
            font: gpui::font("IBM Plex Sans"),
            ..Default::default()
        }];

        let lines = window_text_system.shape_text(text, gpui::px(14.0), &runs, None, None)?;

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1].len(), "\u{05d0}\u{001c}A".len());
        assert!(lines[1].width() > Pixels::ZERO);
        Ok(())
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn reports_graphemes_that_exhaust_font_fallback() -> Result<()> {
        let platform_text_system = Arc::new(text_system()?);
        let dispatcher = gpui::TestDispatcher::new(0);
        let cx =
            gpui::TestAppContext::build_with_text_system(dispatcher, None, platform_text_system);
        let observed = Rc::new(RefCell::new(Vec::new()));
        let _subscription = cx.update(|cx| {
            let observed = observed.clone();
            cx.on_missing_glyphs(move |missing_glyphs, _| {
                observed.borrow_mut().extend_from_slice(missing_glyphs);
            })
        });
        let text: SharedString = "界".into();

        cx.update(|cx| {
            let text_system = gpui::WindowTextSystem::new(cx.text_system().clone());
            let runs = [gpui::TextRun {
                len: text.len(),
                font: gpui::font("IBM Plex Sans"),
                ..Default::default()
            }];
            text_system.shape_line(text, gpui::px(14.0), &runs, None);
        });
        cx.run_until_parked();

        let observed = observed.borrow();
        assert_eq!(observed.len(), 1);
        assert_eq!(observed[0].grapheme(), "界");
        assert_eq!(
            observed[0].font_class(),
            gpui::FallbackFontClass::Proportional
        );
        Ok(())
    }

    #[test]
    fn combines_missing_glyphs_from_one_grapheme() -> Result<()> {
        let text_system = text_system()?;
        let font_id = text_system.font_id(&gpui::font("IBM Plex Sans"))?;
        let text = "x\u{0301}";
        let runs = [FontRun {
            len: text.len(),
            font_id,
        }];

        let missing_glyphs = text_system
            .0
            .read()
            .missing_glyphs(text, &runs, [0, "x".len()]);

        assert_eq!(missing_glyphs.len(), 1);
        assert_eq!(missing_glyphs[0].grapheme(), text);
        Ok(())
    }

    #[test]
    fn adding_fonts_invalidates_cached_line_layouts() -> Result<()> {
        let platform_text_system = Arc::new(text_system()?);
        let text_system = Arc::new(gpui::TextSystem::new(platform_text_system.clone()));
        let window_text_system = gpui::WindowTextSystem::new(text_system.clone());
        let text: SharedString = "cached text".into();
        let runs = [gpui::TextRun {
            len: text.len(),
            font: gpui::font("IBM Plex Sans"),
            ..Default::default()
        }];

        let first_layout = window_text_system.shape_line(text.clone(), gpui::px(14.0), &runs, None);
        let cached_layout =
            window_text_system.shape_line(text.clone(), gpui::px(14.0), &runs, None);
        assert!(std::ptr::eq::<LineLayout>(
            &**first_layout,
            &**cached_layout
        ));
        let loaded_font_count = platform_text_system.0.read().loaded_fonts.len();

        text_system.add_fonts(vec![Cow::Borrowed(LILEX)])?;

        let refreshed_layout = window_text_system.shape_line(text, gpui::px(14.0), &runs, None);
        assert!(!std::ptr::eq::<LineLayout>(
            &**first_layout,
            &**refreshed_layout
        ));
        assert_eq!(
            platform_text_system.0.read().loaded_fonts.len(),
            loaded_font_count
        );
        Ok(())
    }

    #[test]
    fn layout_line_with_mixed_direction_paragraphs() -> Result<()> {
        let text_system = text_system()?;

        for separator in SEPARATORS {
            for text in [
                format!("\u{05d0}{separator}A"),
                format!("A{separator}\u{05d0}"),
            ] {
                let layout = layout_text(&text_system, &text)?;

                assert_eq!(layout.len, text.len(), "{text:?}");
                assert!(layout.width > Pixels::ZERO, "{text:?}");
                assert!(
                    layout.runs.iter().any(|run| !run.glyphs.is_empty()),
                    "{text:?}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn layout_line_with_separators_at_line_edges() -> Result<()> {
        let text_system = text_system()?;

        for text in [
            "\u{001c}",
            "\u{001c}\u{001c}",
            "\u{001c}\u{05d0}",
            "\u{05d0}\u{001c}",
            "\u{05d0}\u{001c}\u{001c}A",
            "\u{001c}\u{05d0}\u{001c}A\u{001c}",
        ] {
            let layout = layout_text(&text_system, text)?;
            assert_eq!(layout.len, text.len(), "{text:?}");
        }

        Ok(())
    }

    /// Glyph indices must stay absolute and positions ordered across segment
    /// boundaries, otherwise cursor placement and hit testing desync. Uses
    /// single-direction text so visual order matches logical order.
    #[test]
    fn layout_line_keeps_indices_and_positions_ordered_across_paragraphs() -> Result<()> {
        let text_system = text_system()?;
        let text = "ab\u{001c}cd\u{2029}ef";
        let layout = layout_text(&text_system, text)?;

        let glyphs: Vec<_> = layout.runs.iter().flat_map(|run| &run.glyphs).collect();
        assert!(!glyphs.is_empty());

        for glyph in &glyphs {
            assert!(glyph.index < text.len(), "{:?}", glyph.index);
            assert!(text.is_char_boundary(glyph.index), "{:?}", glyph.index);
        }
        for pair in glyphs.windows(2) {
            assert!(pair[0].index < pair[1].index);
            assert!(pair[0].position.x <= pair[1].position.x);
        }

        // Every segment contributes width, so the whole line is wider than its
        // leading paragraph alone.
        assert!(layout.width > layout_text(&text_system, "ab")?.width);
        Ok(())
    }

    /// A font run boundary that does not line up with a paragraph boundary must
    /// still be clipped to the right segments.
    #[test]
    fn layout_line_with_font_run_straddling_a_separator() -> Result<()> {
        let text_system = text_system()?;
        let font_id = text_system.font_id(&gpui::font("IBM Plex Sans"))?;
        let text = "ab\u{001c}\u{05d0}\u{05d1}";

        // The run boundary falls inside the trailing RTL paragraph.
        let runs = [
            FontRun {
                len: "ab\u{001c}\u{05d0}".len(),
                font_id,
            },
            FontRun {
                len: "\u{05d1}".len(),
                font_id,
            },
        ];
        let layout = text_system.layout_line(text, gpui::px(14.0), &runs);

        assert_eq!(layout.len, text.len());
        assert!(layout.width > Pixels::ZERO);
        Ok(())
    }

    /// Lines with no separator take the fast path and must be shaped exactly as
    /// they were before paragraph splitting existed.
    #[test]
    fn layout_line_without_separators_takes_fast_path() -> Result<()> {
        let text_system = text_system()?;

        for text in [
            "hello world",
            "\u{05d0}\u{05d1}\u{05d2}",
            "mixed \u{05d0}\u{05d1}",
        ] {
            assert!(!contains_paragraph_separator(text), "{text:?}");
            let layout = layout_text(&text_system, text)?;
            assert_eq!(layout.len, text.len(), "{text:?}");
            assert!(layout.width > Pixels::ZERO, "{text:?}");
        }

        Ok(())
    }

    /// cosmic-text sums word widths to get a line's width but accumulates glyph
    /// advances to position glyphs, so a trailing zero-advance glyph (here a
    /// zero-width space) can land a few ulps past the width. When that glyph is a
    /// wrap boundary, the row before it extends past the line's width, and hit
    /// testing in that sliver used to panic (ZED-BW8, ZED-75K, ZED-81Z).
    #[test]
    fn index_for_position_past_line_width() -> Result<()> {
        let text_system = Arc::new(gpui::TextSystem::new(Arc::new(text_system()?)));
        let window_text_system = gpui::WindowTextSystem::new(text_system);
        let text: SharedString = "Warning: this will delete files\u{200b}".into();
        let runs = [gpui::TextRun {
            len: text.len(),
            font: gpui::font("IBM Plex Sans"),
            ..Default::default()
        }];
        let lines =
            window_text_system.shape_text(text, gpui::px(14.), &runs, Some(gpui::px(4.)), None)?;
        let line = &lines[0];
        let width = line.unwrapped_layout.width;
        let boundary_glyph = |row: usize| {
            let boundary = line.wrap_boundaries()[row];
            &line.runs()[boundary.run_ix].glyphs[boundary.glyph_ix]
        };
        let row = (1..line.wrap_boundaries().len())
            .find(|row| boundary_glyph(*row).position.x > width)
            .expect("trailing zero-width space should be a wrap boundary past the line width");
        let row_start_x = f32::from(boundary_glyph(row - 1).position.x);
        let row_end = boundary_glyph(row);

        let mut x = f32::from(width) - row_start_x;
        while x + row_start_x < f32::from(width) {
            x = x.next_up();
        }
        assert!(gpui::px(x + row_start_x) < row_end.position.x);

        let line_height = gpui::px(20.);
        let position = gpui::point(gpui::px(x), line_height * row as f32 + gpui::px(1.));
        assert_eq!(
            line.index_for_position(position, line_height),
            Err(row_end.index)
        );
        Ok(())
    }

    #[test]
    fn paragraph_separator_detection() {
        for separator in SEPARATORS {
            assert!(is_paragraph_separator(*separator), "{separator:?}");
            assert!(contains_paragraph_separator(&format!("a{separator}b")));
        }

        for text in [
            "",
            "plain ascii",
            "\u{05d0}",
            "tab\there",
            "emoji \u{1f600}",
        ] {
            assert!(!contains_paragraph_separator(text), "{text:?}");
        }
    }

    #[test]
    fn font_runs_are_clipped_to_segment() {
        let runs = [
            FontRun {
                len: 3,
                font_id: fid(1),
            },
            FontRun {
                len: 4,
                font_id: fid(2),
            },
        ];

        assert_eq!(clip_font_runs(&runs, 0..7).as_slice(), &runs);
        assert_eq!(
            clip_font_runs(&runs, 2..5).as_slice(),
            &[
                FontRun {
                    len: 1,
                    font_id: fid(1)
                },
                FontRun {
                    len: 2,
                    font_id: fid(2)
                },
            ]
        );
        assert_eq!(
            clip_font_runs(&runs, 3..7).as_slice(),
            &[FontRun {
                len: 4,
                font_id: fid(2)
            }]
        );
        assert!(clip_font_runs(&runs, 5..5).is_empty());
    }

    #[test]
    fn primary_wins_over_current_fallback_when_primary_covers() {
        let primary = fid(0);
        let fb = chain(&[1, 2]);
        let covers = |id: FontId, _: char| id == fid(0) || id == fid(1);
        assert_eq!(
            pick_covering_slot('a', Some(0), primary, &fb, &covers),
            None
        );
    }

    #[test]
    fn primary_preferred_over_fallback_when_both_cover() {
        let primary = fid(0);
        let fb = chain(&[1]);
        let covers = |_: FontId, _: char| true;
        assert_eq!(pick_covering_slot('a', None, primary, &fb, &covers), None);
    }

    #[test]
    fn falls_through_chain_in_order() {
        let primary = fid(0);
        let fb = chain(&[1, 2, 3]);
        // only fallback 2 at index 1 covers.
        let covers = |id: FontId, _: char| id == fid(2);
        assert_eq!(
            pick_covering_slot('字', None, primary, &fb, &covers),
            Some(1)
        );
    }

    #[test]
    fn no_coverage_returns_primary() {
        let primary = fid(0);
        let fb = chain(&[1, 2]);
        let covers = |_: FontId, _: char| false;
        // nothing covers. return `None` so the `cosmic-text` built in script
        // fallback can take over during shaping.
        assert_eq!(
            pick_covering_slot('\u{1F600}', Some(1), primary, &fb, &covers),
            None
        );
    }

    #[test]
    fn empty_chain_always_returns_primary() {
        let primary = fid(0);
        let fb: SmallVec<[(FontId, SharedString); 4]> = SmallVec::new();
        let covers = |_: FontId, _: char| false;
        assert_eq!(pick_covering_slot('a', None, primary, &fb, &covers), None);
    }

    #[test]
    fn slot_font_id_resolution() {
        let primary = fid(7);
        let fb = chain(&[10, 20]);
        assert_eq!(slot_font_id(None, primary, &fb), fid(7));
        assert_eq!(slot_font_id(Some(0), primary, &fb), fid(10));
        assert_eq!(slot_font_id(Some(1), primary, &fb), fid(20));
    }

    #[test]
    fn run_spans_with_no_chain_emit_one_primary_span() {
        let primary = fid(0);
        let fb: SmallVec<[(FontId, SharedString); 4]> = SmallVec::new();
        let covers = |_: FontId, _: char| false;
        let text = "hello";
        let spans = compute_run_spans(text, 0, text.len(), primary, &fb, &covers);
        assert_eq!(spans.as_slice(), &[span(0, text.len(), None, primary)]);
    }

    #[test]
    fn run_spans_use_byte_offsets_for_multibyte_chars() {
        let primary = fid(0);
        let fb = chain(&[1]);
        // primary covers ascii. fallback covers cjk.
        let covers = |id: FontId, ch: char| {
            if id == primary {
                ch.is_ascii()
            } else {
                !ch.is_ascii()
            }
        };
        let text = "a字b";
        let spans = compute_run_spans(text, 0, text.len(), primary, &fb, &covers);
        // '字' is 3 bytes so split is at 1 then 4.
        assert_eq!(
            spans.as_slice(),
            &[
                span(0, 1, None, primary),
                span(1, 4, Some(0), fid(1)),
                span(4, 5, None, primary),
            ]
        );
    }

    #[test]
    fn run_spans_respect_run_offset() {
        let primary = fid(0);
        let fb = chain(&[1]);
        let covers = |id: FontId, ch: char| {
            if id == primary {
                ch.is_ascii()
            } else {
                !ch.is_ascii()
            }
        };
        // outer text has a prefix that is not part of this run.
        let text = "xx字y";
        let run_offset = 2;
        let run_len = text.len() - run_offset;
        let spans = compute_run_spans(text, run_offset, run_len, primary, &fb, &covers);
        assert_eq!(
            spans.as_slice(),
            &[span(2, 5, Some(0), fid(1)), span(5, 6, None, primary)]
        );
    }

    #[test]
    fn run_spans_keep_combining_marks_with_base_in_fallback() {
        let primary = fid(0);
        let fb = chain(&[1]);
        // primary covers ascii only. fallback covers the base char.
        // combining mark must stay in the fallback span even when fallback
        // does not advertise coverage of it.
        let covers = |id: FontId, ch: char| {
            if id == primary {
                ch.is_ascii()
            } else {
                ch == '\u{0905}'
            }
        };
        // \u{0905} devanagari short a + \u{0902} candrabindu mark.
        let text = "\u{0905}\u{0902}";
        let spans = compute_run_spans(text, 0, text.len(), primary, &fb, &covers);
        assert_eq!(spans.as_slice(), &[span(0, text.len(), Some(0), fid(1))]);
    }

    #[test]
    fn run_spans_keep_zwj_inside_emoji_cluster() {
        let primary = fid(0);
        let fb = chain(&[1]);
        // only fallback covers the emoji codepoints. zwj must not split.
        let covers = |id: FontId, ch: char| id == fid(1) && ch != '\u{200D}';
        // family zwj sequence woman zwj girl.
        let text = "\u{1F469}\u{200D}\u{1F467}";
        let spans = compute_run_spans(text, 0, text.len(), primary, &fb, &covers);
        assert_eq!(spans.as_slice(), &[span(0, text.len(), Some(0), fid(1))]);
    }

    #[test]
    fn run_spans_collapse_adjacent_same_slot() {
        let primary = fid(0);
        let fb = chain(&[1]);
        let covers = |id: FontId, ch: char| {
            if id == primary {
                ch.is_ascii()
            } else {
                !ch.is_ascii()
            }
        };
        let text = "字字字";
        let spans = compute_run_spans(text, 0, text.len(), primary, &fb, &covers);
        assert_eq!(spans.as_slice(), &[span(0, text.len(), Some(0), fid(1))]);
    }

    #[test]
    fn run_spans_empty_run_returns_no_spans() {
        let primary = fid(0);
        let fb = chain(&[1]);
        let covers = |_: FontId, _: char| true;
        let spans = compute_run_spans("anything", 3, 0, primary, &fb, &covers);
        assert!(spans.is_empty());
    }

    // Subsets of DejaVu Sans 2.37 and Noto Color Emoji 2.047 as Ubuntu 24.04 ships
    // them, made with fontTools' subset for the code points the tests below shape.
    const DEJAVU_SANS_SUBSET: &[u8] =
        include_bytes!("../test_data/dejavu-sans/DejaVuSans-Subset.ttf");
    const NOTO_COLOR_EMOJI_SUBSET: &[u8] =
        include_bytes!("../test_data/noto-color-emoji/NotoColorEmoji-Subset.ttf");
    const NOTO_COLOR_EMOJI_NO_COLOR_SUBSET: &[u8] =
        include_bytes!("../test_data/noto-color-emoji/NotoColorEmoji-NoColor-Subset.ttf");
    const NOTO_COLOR_EMOJI_COLRV1_SUBSET: &[u8] = include_bytes!(
        "../../gpui_emoji/test_data/noto-color-emoji/NotoColorEmoji-COLRv1-Subset.ttf"
    );

    // A subset of Noto Sans Hebrew 2.003, a variable font whose weight axis runs
    // from 100 to 900, made with fontTools' subset for "Hilm ".
    const NOTO_SANS_HEBREW_SUBSET: &[u8] =
        include_bytes!("../test_data/noto-sans-hebrew/NotoSansHebrew-Subset.ttf");

    fn ink(text_system: &CosmicTextSystem, font_id: FontId, character: char) -> Result<u32> {
        let glyph_id = text_system
            .glyph_for_char(font_id, character)
            .context("no glyph")?;
        let params = RenderGlyphParams {
            font_id,
            glyph_id,
            font_size: gpui::px(32.0),
            subpixel_variant: point(0, 0),
            scale_factor: 1.0,
            is_emoji: false,
            subpixel_rendering: false,
            dilation: 0,
        };
        let bounds = text_system.glyph_raster_bounds(&params)?;
        let (_, pixels) = text_system.rasterize_glyph(&params, bounds)?;
        Ok(pixels.iter().map(|alpha| u32::from(*alpha)).sum())
    }

    #[test]
    fn bold_in_a_variable_font_shapes_and_draws_at_the_bold_weight() -> Result<()> {
        let text_system = CosmicTextSystem::new_without_system_fonts("IBM Plex Sans");
        text_system.add_fonts(vec![Cow::Borrowed(NOTO_SANS_HEBREW_SUBSET)])?;
        let regular = gpui::font("Noto Sans Hebrew");
        let bold = Font {
            weight: gpui::FontWeight::BOLD,
            ..regular.clone()
        };
        let regular_id = text_system.font_id(&regular)?;
        let bold_id = text_system.font_id(&bold)?;
        assert_eq!(
            text_system.font_weight_and_style(bold_id)?,
            (gpui::FontWeight::BOLD, gpui::FontStyle::Normal)
        );
        let regular_width = layout_in(&text_system, &regular, "Hilm")?.width;
        let bold_width = layout_in(&text_system, &bold, "Hilm")?.width;
        assert!(
            bold_width > regular_width,
            "{bold_width:?} <= {regular_width:?}"
        );
        let regular_ink = ink(&text_system, regular_id, 'm')?;
        let bold_ink = ink(&text_system, bold_id, 'm')?;
        assert!(
            bold_ink > regular_ink * 5 / 4,
            "{bold_ink} <= {regular_ink} * 1.25"
        );
        Ok(())
    }

    fn emoji_text_system(fonts: &[&'static [u8]]) -> Result<CosmicTextSystem> {
        let text_system = CosmicTextSystem::new_without_system_fonts("IBM Plex Sans");
        text_system.add_fonts(fonts.iter().map(|font| Cow::Borrowed(*font)).collect())?;
        Ok(text_system)
    }

    // On Linux, cosmic-text's own fallback list puts DejaVu Sans before the emoji
    // font, and DejaVu Sans maps ❤, ⚠ and 😀. A user fallback makes it win on
    // every OS.
    fn plex_falling_back_to_dejavu() -> Font {
        let mut font = gpui::font("IBM Plex Sans");
        font.fallbacks = Some(FontFallbacks::from_fonts(vec!["DejaVu Sans".to_owned()]));
        font
    }

    fn layout_in(text_system: &CosmicTextSystem, font: &Font, text: &str) -> Result<LineLayout> {
        let font_id = text_system.font_id(font)?;
        let runs = [FontRun {
            len: text.len(),
            font_id,
        }];
        Ok(text_system.layout_line(text, gpui::px(32.0), &runs))
    }

    fn glyph_faces(layout: &LineLayout) -> Vec<(FontId, bool)> {
        layout
            .runs
            .iter()
            .flat_map(|run| run.glyphs.iter().map(|glyph| (run.font_id, glyph.is_emoji)))
            .collect()
    }

    fn emoji_font_id(text_system: &CosmicTextSystem) -> Result<FontId> {
        let (font_id, _) = text_system
            .0
            .write()
            .emoji_font()
            .context("no emoji font")?;
        Ok(font_id)
    }

    fn is_paintable(tables: &[(&[u8; 4], &[u8])]) -> bool {
        let mut font = 0x0001_0000_u32.to_be_bytes().to_vec();
        font.extend_from_slice(&(tables.len() as u16).to_be_bytes());
        font.extend_from_slice(&[0; 6]);
        let mut offset = 12 + 16 * tables.len();
        for (tag, data) in tables {
            font.extend_from_slice(*tag);
            font.extend_from_slice(&[0; 4]);
            font.extend_from_slice(&(offset as u32).to_be_bytes());
            font.extend_from_slice(&(data.len() as u32).to_be_bytes());
            offset += data.len();
        }
        for (_, data) in tables {
            font.extend_from_slice(data);
        }
        swash::FontRef::from_index(&font, 0).is_some_and(has_paintable_color_glyphs)
    }

    #[test]
    fn paintable_color_glyphs_need_bitmaps_colr_layers_or_a_colr_v1_glyph_list() {
        let cmap: (&[u8; 4], &[u8]) = (b"cmap", &[0; 4]);
        assert!(is_paintable(&[(b"CBDT", &[0, 3, 0, 0]), cmap]));
        assert!(is_paintable(&[(b"COLR", &[0, 0, 0, 1, 0, 0, 0, 14]), cmap]));
        let mut colr_v1 = vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34];
        colr_v1.resize(34, 0);
        assert!(is_paintable(&[(b"COLR", &colr_v1), cmap]));
        assert!(!is_paintable(&[
            (b"COLR", &[0, 1, 0, 0, 0, 0, 0, 34]),
            cmap
        ]));
        assert!(!is_paintable(&[cmap]));
    }

    #[test]
    fn layout_line_draws_emoji_sequences_from_the_emoji_font() -> Result<()> {
        let text_system =
            emoji_text_system(&[IBM_PLEX, DEJAVU_SANS_SUBSET, NOTO_COLOR_EMOJI_SUBSET])?;
        let emoji_font = emoji_font_id(&text_system)?;
        let font = plex_falling_back_to_dejavu();
        for text in ["❤\u{fe0f}", "❤", "😀", "⚠\u{fe0f}", "1\u{fe0f}\u{20e3}"] {
            let layout = layout_in(&text_system, &font, text)?;
            assert_eq!(glyph_faces(&layout), [(emoji_font, true)], "{text:?}");
        }
        let plex = text_system.font_id(&font)?;
        let one = layout_in(&text_system, &font, "1")?;
        assert_eq!(glyph_faces(&one), [(plex, false)]);
        Ok(())
    }

    #[test]
    fn text_default_emoji_keeps_a_primary_font_that_maps_it() -> Result<()> {
        let text_system = emoji_text_system(&[DEJAVU_SANS_SUBSET, NOTO_COLOR_EMOJI_SUBSET])?;
        let emoji_font = emoji_font_id(&text_system)?;
        let font = gpui::font("DejaVu Sans");
        let dejavu = text_system.font_id(&font)?;
        let heart = layout_in(&text_system, &font, "❤")?;
        assert_eq!(glyph_faces(&heart), [(dejavu, false)]);
        let selected = layout_in(&text_system, &font, "❤\u{fe0f}")?;
        assert_eq!(glyph_faces(&selected), [(emoji_font, true)]);
        Ok(())
    }

    fn colored_pixels(text_system: &CosmicTextSystem, text: &str) -> Result<usize> {
        let layout = layout_in(text_system, &plex_falling_back_to_dejavu(), text)?;
        let run = layout.runs.first().context("no run")?;
        let glyph = run.glyphs.first().context("no glyph")?;
        anyhow::ensure!(glyph.is_emoji, "{text:?} drew as text");
        let params = RenderGlyphParams {
            font_id: run.font_id,
            glyph_id: glyph.id,
            font_size: gpui::px(32.0),
            subpixel_variant: point(0, 0),
            scale_factor: 1.0,
            is_emoji: glyph.is_emoji,
            subpixel_rendering: false,
            dilation: 0,
        };
        let bounds = text_system.glyph_raster_bounds(&params)?;
        let (_, pixels) = text_system.rasterize_glyph(&params, bounds)?;
        Ok(pixels
            .chunks_exact(4)
            .filter(|pixel| {
                let channels = pixel.iter().take(3);
                let (min, max) = channels.fold((u8::MAX, u8::MIN), |(min, max), channel| {
                    (min.min(*channel), max.max(*channel))
                });
                max - min > 48
            })
            .count())
    }

    #[test]
    fn colrv1_emoji_rasterize_in_color() -> Result<()> {
        let text_system =
            emoji_text_system(&[IBM_PLEX, DEJAVU_SANS_SUBSET, NOTO_COLOR_EMOJI_COLRV1_SUBSET])?;
        for text in ["😀", "❤\u{fe0f}", "🇩🇪", "🔥"] {
            let colored = colored_pixels(&text_system, text)?;
            assert!(colored >= 300, "{text:?}: {colored} colored pixels");
        }
        Ok(())
    }

    #[test]
    fn added_emoji_font_replaces_the_systems_copy_of_its_family() -> Result<()> {
        let text_system =
            emoji_text_system(&[IBM_PLEX, DEJAVU_SANS_SUBSET, NOTO_COLOR_EMOJI_SUBSET])?;
        text_system.add_fonts(vec![Cow::Borrowed(NOTO_COLOR_EMOJI_COLRV1_SUBSET)])?;
        let emoji_font = emoji_font_id(&text_system)?;
        let layout = layout_in(&text_system, &plex_falling_back_to_dejavu(), "😀")?;
        assert_eq!(glyph_faces(&layout), [(emoji_font, true)]);
        let state = text_system.0.read();
        let font = state.loaded_fonts[emoji_font.0].font.as_swash();
        assert!(font.table(swash::tag_from_bytes(b"COLR")).is_some());
        assert!(font.table(swash::tag_from_bytes(b"CBDT")).is_none());
        let copies = state
            .font_system
            .db()
            .faces()
            .filter(|face| {
                face.families
                    .iter()
                    .any(|(name, _)| name == "Noto Color Emoji")
            })
            .count();
        assert_eq!(copies, 1);
        drop(state);
        let colored = colored_pixels(&text_system, "😀")?;
        assert!(colored >= 300, "{colored} colored pixels");
        Ok(())
    }

    fn emoji_face_tables(text_system: &CosmicTextSystem) -> Result<(bool, bool)> {
        let emoji_font = emoji_font_id(text_system)?;
        let state = text_system.0.read();
        let font = state.loaded_fonts[emoji_font.0].font.as_swash();
        let has = |tag| font.table(swash::tag_from_bytes(tag)).is_some();
        Ok((has(b"CBDT"), has(b"COLR")))
    }

    #[test]
    fn added_emoji_font_without_color_glyphs_leaves_the_systems_copy() -> Result<()> {
        let text_system =
            emoji_text_system(&[IBM_PLEX, DEJAVU_SANS_SUBSET, NOTO_COLOR_EMOJI_SUBSET])?;
        text_system.add_fonts(vec![Cow::Borrowed(NOTO_COLOR_EMOJI_NO_COLOR_SUBSET)])?;
        assert_eq!(emoji_face_tables(&text_system)?, (true, false));
        let colored = colored_pixels(&text_system, "😀")?;
        assert!(colored >= 300, "{colored} colored pixels");
        Ok(())
    }

    #[test]
    fn emoji_face_in_use_stays_beside_an_added_copy() -> Result<()> {
        let text_system =
            emoji_text_system(&[IBM_PLEX, DEJAVU_SANS_SUBSET, NOTO_COLOR_EMOJI_SUBSET])?;
        let used = emoji_font_id(&text_system)?;
        let used_face = text_system.0.read().loaded_fonts[used.0].font.id();
        text_system.add_fonts(vec![Cow::Borrowed(NOTO_COLOR_EMOJI_COLRV1_SUBSET)])?;
        assert!(
            text_system
                .0
                .read()
                .font_system
                .db()
                .face(used_face)
                .is_some()
        );
        let colored = colored_pixels(&text_system, "😀")?;
        assert!(colored >= 300, "{colored} colored pixels");
        Ok(())
    }

    #[test]
    fn emoji_glyphs_rasterize_in_color() -> Result<()> {
        let text_system =
            emoji_text_system(&[IBM_PLEX, DEJAVU_SANS_SUBSET, NOTO_COLOR_EMOJI_SUBSET])?;
        let layout = layout_in(&text_system, &plex_falling_back_to_dejavu(), "❤\u{fe0f}")?;
        let run = layout.runs.first().context("no run")?;
        let glyph = run.glyphs.first().context("no glyph")?;
        assert!(glyph.is_emoji);
        let params = RenderGlyphParams {
            font_id: run.font_id,
            glyph_id: glyph.id,
            font_size: gpui::px(32.0),
            subpixel_variant: point(0, 0),
            scale_factor: 1.0,
            is_emoji: glyph.is_emoji,
            subpixel_rendering: false,
            dilation: 0,
        };
        let bounds = text_system.glyph_raster_bounds(&params)?;
        let (_, pixels) = text_system.rasterize_glyph(&params, bounds)?;
        let colored = pixels
            .chunks_exact(4)
            .filter(|pixel| {
                let channels = pixel.iter().take(3);
                let (min, max) = channels.fold((u8::MAX, u8::MIN), |(min, max), channel| {
                    (min.min(*channel), max.max(*channel))
                });
                max - min > 48
            })
            .count();
        assert!(colored >= 300, "{colored} coloured pixels");
        Ok(())
    }

    #[test]
    fn adding_an_emoji_font_routes_later_layouts() -> Result<()> {
        let text_system = emoji_text_system(&[IBM_PLEX, DEJAVU_SANS_SUBSET])?;
        let font = plex_falling_back_to_dejavu();
        let before = layout_in(&text_system, &font, "❤\u{fe0f}")?;
        assert_eq!(glyph_faces(&before).len(), 2);
        assert!(glyph_faces(&before).iter().all(|(_, is_emoji)| !is_emoji));
        text_system.add_fonts(vec![Cow::Borrowed(NOTO_COLOR_EMOJI_SUBSET)])?;
        let emoji_font = emoji_font_id(&text_system)?;
        let after = layout_in(&text_system, &font, "❤\u{fe0f}")?;
        assert_eq!(glyph_faces(&after), [(emoji_font, true)]);
        Ok(())
    }

    fn layout_at(text_system: &CosmicTextSystem, text: &str, font_size: f32) -> Result<LineLayout> {
        let font_id = text_system.font_id(&plex_falling_back_to_dejavu())?;
        let runs = [FontRun {
            len: text.len(),
            font_id,
        }];
        Ok(text_system.layout_line(text, gpui::px(font_size), &runs))
    }

    fn emoji_advance_and_height_per_em(
        text_system: &CosmicTextSystem,
        font_size: f32,
    ) -> Result<(f32, f32)> {
        let layout = layout_at(text_system, "😀😀", font_size)?;
        let run = layout.runs.first().context("no run")?;
        let [first, second] = run.glyphs.as_slice() else {
            anyhow::bail!("{} glyphs", run.glyphs.len());
        };
        let params = RenderGlyphParams {
            font_id: run.font_id,
            glyph_id: first.id,
            font_size: gpui::px(font_size),
            subpixel_variant: point(0, 0),
            scale_factor: 1.0,
            is_emoji: first.is_emoji,
            subpixel_rendering: false,
            dilation: 0,
        };
        let height = text_system.glyph_raster_bounds(&params)?.size.height.0 as f32;
        let advance = f32::from(second.position.x - first.position.x);
        Ok((advance / font_size, height / font_size))
    }

    #[test]
    fn emoji_at_text_sizes_draw_and_advance_larger_than_large_emoji() -> Result<()> {
        let text_system =
            emoji_text_system(&[IBM_PLEX, DEJAVU_SANS_SUBSET, NOTO_COLOR_EMOJI_SUBSET])?;
        let (body_advance, body_height) = emoji_advance_and_height_per_em(&text_system, 16.)?;
        let (large_advance, large_height) = emoji_advance_and_height_per_em(&text_system, 32.)?;
        let (larger_advance, larger_height) = emoji_advance_and_height_per_em(&text_system, 64.)?;
        assert!(
            body_height > large_height * 1.1,
            "{body_height} <= {large_height} * 1.1"
        );
        assert!(
            body_advance > large_advance * 1.1,
            "{body_advance} <= {large_advance} * 1.1"
        );
        assert!(
            (larger_height - large_height).abs() < 0.05,
            "{larger_height} != {large_height}"
        );
        assert!(
            (larger_advance - large_advance).abs() < 0.01,
            "{larger_advance} != {large_advance}"
        );
        Ok(())
    }

    #[test]
    fn emoji_in_right_to_left_text_push_the_glyphs_right_of_them() -> Result<()> {
        let text_system =
            emoji_text_system(&[IBM_PLEX, DEJAVU_SANS_SUBSET, NOTO_COLOR_EMOJI_SUBSET])?;
        let emoji_advance = f32::from(layout_at(&text_system, "😀", 16.)?.width);
        let layout = layout_at(&text_system, "\u{05d0}😀\u{05d1}😀\u{05d2}", 16.)?;
        let mut positions: Vec<(f32, bool)> = layout
            .runs
            .iter()
            .flat_map(|run| &run.glyphs)
            .map(|glyph| (f32::from(glyph.position.x), glyph.is_emoji))
            .collect();
        positions.sort_by(|a, b| a.0.total_cmp(&b.0));
        let emoji_and_right: Vec<(f32, f32)> = positions
            .windows(2)
            .filter_map(|pair| match pair {
                [(emoji_x, true), (right_x, _)] => Some((*emoji_x, *right_x)),
                _ => None,
            })
            .collect();
        assert_eq!(emoji_and_right.len(), 2, "{positions:?}");
        for (emoji_x, right_x) in emoji_and_right {
            assert!(
                (right_x - emoji_x - emoji_advance).abs() < 0.01,
                "{right_x} - {emoji_x} != {emoji_advance} in {positions:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn emoji_at_text_sizes_stay_inside_the_line_ascent_and_descent() -> Result<()> {
        let text_system =
            emoji_text_system(&[IBM_PLEX, DEJAVU_SANS_SUBSET, NOTO_COLOR_EMOJI_SUBSET])?;
        for font_size in 12..=19 {
            let font_size = font_size as f32;
            let layout = layout_at(&text_system, "😀", font_size)?;
            let run = layout.runs.first().context("no run")?;
            let glyph = run.glyphs.first().context("no glyph")?;
            let params = RenderGlyphParams {
                font_id: run.font_id,
                glyph_id: glyph.id,
                font_size: gpui::px(font_size),
                subpixel_variant: point(0, 0),
                scale_factor: 1.0,
                is_emoji: glyph.is_emoji,
                subpixel_rendering: false,
                dilation: 0,
            };
            let bounds = text_system.glyph_raster_bounds(&params)?;
            let top = -bounds.origin.y.0 as f32;
            let bottom = (bounds.origin.y.0 + bounds.size.height.0) as f32;
            let (ascent, descent) = (f32::from(layout.ascent), f32::from(layout.descent));
            assert!(
                top <= ascent + 1.,
                "{font_size} px: top {top} above {ascent}"
            );
            assert!(
                bottom <= descent + 1.,
                "{font_size} px: bottom {bottom} below {descent}"
            );
        }
        Ok(())
    }

    #[test]
    fn missing_glyphs_of_the_emoji_font_keep_their_advance() -> Result<()> {
        let text_system =
            emoji_text_system(&[IBM_PLEX, DEJAVU_SANS_SUBSET, NOTO_COLOR_EMOJI_SUBSET])?;
        let missing_advance_per_em = |font_size: f32| -> Result<f32> {
            let layout = layout_at(&text_system, "😀🏽x", font_size)?;
            let glyphs: Vec<&ShapedGlyph> =
                layout.runs.iter().flat_map(|run| &run.glyphs).collect();
            let missing = glyphs
                .iter()
                .position(|glyph| glyph.id.0 == 0)
                .context("no missing glyph")?;
            let Some([missing, next]) = glyphs.get(missing..missing + 2) else {
                anyhow::bail!("no glyph after the missing one");
            };
            assert!(missing.is_emoji);
            Ok(f32::from(next.position.x - missing.position.x) / font_size)
        };
        let body = missing_advance_per_em(16.)?;
        let large = missing_advance_per_em(32.)?;
        assert!((body - large).abs() < 0.01, "{body} != {large}");
        Ok(())
    }

    #[test]
    fn components_of_an_unjoined_emoji_sequence_make_room() -> Result<()> {
        let text_system =
            emoji_text_system(&[IBM_PLEX, DEJAVU_SANS_SUBSET, NOTO_COLOR_EMOJI_SUBSET])?;
        let emoji_advance = f32::from(layout_at(&text_system, "😀", 16.)?.width);
        let layout = layout_at(&text_system, "😀\u{200d}😀", 16.)?;
        let run = layout.runs.first().context("no run")?;
        let [first, second] = run.glyphs.as_slice() else {
            anyhow::bail!("{} glyphs", run.glyphs.len());
        };
        assert_eq!(first.index, second.index);
        let distance = f32::from(second.position.x - first.position.x);
        assert!(
            (distance - emoji_advance).abs() < 0.01,
            "{distance} != {emoji_advance}"
        );
        Ok(())
    }
}
