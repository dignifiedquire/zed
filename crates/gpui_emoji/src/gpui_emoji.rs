//! Emoji rules that GPUI's text systems share.

use std::ops::Range;

use smallvec::SmallVec;
use unicode_properties::{EmojiStatus, UnicodeEmoji as _};
use unicode_segmentation::UnicodeSegmentation as _;

/// How a grapheme cluster asks to be drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presentation {
    Text,
    /// As emoji where the primary font lacks it, as a text-default emoji
    /// without a selector does.
    TextDefault,
    Emoji,
}

impl Presentation {
    pub fn of(cluster: &str) -> Self {
        if cluster.contains('\u{fe0e}') {
            Self::Text
        } else if cluster.chars().any(|character| {
            matches!(character, '\u{fe0f}' | '\u{20e3}') || is_emoji_presentation(character)
        }) {
            Self::Emoji
        } else if cluster.chars().any(is_text_default_emoji) {
            Self::TextDefault
        } else {
            Self::Text
        }
    }
}

fn is_emoji_presentation(character: char) -> bool {
    matches!(
        character.emoji_status(),
        EmojiStatus::EmojiPresentation
            | EmojiStatus::EmojiPresentationAndModifierBase
            | EmojiStatus::EmojiPresentationAndEmojiComponent
            | EmojiStatus::EmojiPresentationAndModifierAndEmojiComponent
    )
}

// The keycap bases are the only ASCII characters with Emoji=Yes, and they stay
// text without a selector.
fn is_text_default_emoji(character: char) -> bool {
    !character.is_ascii()
        && matches!(
            character.emoji_status(),
            EmojiStatus::EmojiOther
                | EmojiStatus::EmojiModifierBase
                | EmojiStatus::EmojiOtherAndEmojiComponent
        )
}

/// Returns the factor emoji grow by at a font size in pixels.
pub fn emoji_scale(font_size: f32) -> f32 {
    // CoreText draws Apple Color Emoji 1.25 times as large up to 16 px, and the factor
    // falls to 1 at 24 px. Noto Color Emoji's art fills about 1.1 times the em of Apple's,
    // so 0.9 of that factor matches macOS. Large emoji keep their size. Other emoji fonts
    // take the same factor as an approximation.
    let core_text_size = (font_size * 1.25).min(12. + font_size / 2.).max(font_size);
    (0.9 * core_text_size / font_size).max(1.)
}

/// Returns the byte ranges of the grapheme clusters in `text[run]` that the
/// emoji font draws.
pub fn emoji_presentation_clusters(
    text: &str,
    run: Range<usize>,
    primary_covers: &impl Fn(char) -> bool,
    emoji_font_covers: &impl Fn(char) -> bool,
) -> SmallVec<[Range<usize>; 4]> {
    let Some(run_text) = text.get(run.clone()) else {
        return SmallVec::new();
    };
    if run_text.is_ascii() {
        return SmallVec::new();
    }
    run_text
        .grapheme_indices(true)
        .filter(|(_, cluster)| {
            let Some(first) = cluster.chars().next() else {
                return false;
            };
            let is_emoji = match Presentation::of(cluster) {
                Presentation::Text => false,
                Presentation::TextDefault => !primary_covers(first),
                Presentation::Emoji => true,
            };
            is_emoji && emoji_font_covers(first)
        })
        .map(|(start, cluster)| {
            let start = run.start + start;
            start..start + cluster.len()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emoji_grow_at_text_sizes_and_keep_large_sizes() {
        for font_size in [12., 14., 16.] {
            assert!((emoji_scale(font_size) - 1.125).abs() < 1e-6, "{font_size}");
        }
        assert!(emoji_scale(18.) > 1. && emoji_scale(18.) < 1.125);
        for font_size in [20., 24., 32., 64.] {
            assert_eq!(emoji_scale(font_size), 1., "{font_size}");
        }
    }

    #[test]
    fn emoji_presentation_follows_selectors() {
        let expected = [
            ("❤\u{fe0f}", Presentation::Emoji),
            ("❤", Presentation::TextDefault),
            ("❤\u{fe0e}", Presentation::Text),
            ("1\u{fe0f}\u{20e3}", Presentation::Emoji),
            ("#\u{fe0f}\u{20e3}", Presentation::Emoji),
            ("1", Presentation::Text),
            ("#", Presentation::Text),
            ("😀", Presentation::Emoji),
            ("😀\u{fe0e}", Presentation::Text),
            ("👋🏽", Presentation::Emoji),
            ("☝🏻", Presentation::Emoji),
            ("🇩🇪", Presentation::Emoji),
            ("🏳\u{fe0f}\u{200d}🌈", Presentation::Emoji),
            ("©", Presentation::TextDefault),
            ("a", Presentation::Text),
        ];
        for (cluster, presentation) in expected {
            assert_eq!(Presentation::of(cluster), presentation, "{cluster:?}");
        }
    }

    fn drawn_clusters(
        text: &str,
        primary_covers: impl Fn(char) -> bool,
        emoji_font_covers: impl Fn(char) -> bool,
    ) -> Vec<&str> {
        emoji_presentation_clusters(text, 3..text.len(), &primary_covers, &emoji_font_covers)
            .into_iter()
            .map(|range| &text[range])
            .collect()
    }

    #[test]
    fn emoji_presentation_clusters_span_whole_sequences() {
        let text = "xx a ❤\u{fe0f} 1\u{fe0f}\u{20e3} ❤ 😀\u{fe0e} 😀";
        let ascii = |character: char| character.is_ascii();
        assert_eq!(
            drawn_clusters(text, ascii, |_| true),
            ["❤\u{fe0f}", "1\u{fe0f}\u{20e3}", "❤", "😀"]
        );
        let ascii_and_heart = |character: char| character.is_ascii() || character == '❤';
        assert_eq!(
            drawn_clusters(text, ascii_and_heart, |_| true),
            ["❤\u{fe0f}", "1\u{fe0f}\u{20e3}", "😀"]
        );
        assert!(drawn_clusters(text, ascii, |_| false).is_empty());
    }
}
