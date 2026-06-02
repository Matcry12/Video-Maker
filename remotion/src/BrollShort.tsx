import React, { useMemo } from "react";
import {
  AbsoluteFill,
  Img,
  OffthreadVideo,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

// ── Types ─────────────────────────────────────────────────────────────────

export interface WordTiming {
  word: string;
  start: number; // seconds
  end: number;   // seconds
}

export interface BrollShortProps {
  clipsFile: string;
  paperFile: string;
  fontFile: string;
  fps: number;
  durationInFrames: number;
  words: WordTiming[];
  hookText?: string;
}

// ── Constants ─────────────────────────────────────────────────────────────

const TARGET_WORDS  = 4;    // words per single-line caption (soft target)
const MAX_CHARS     = 17;   // char budget per card so one line never overflows at FONT_SIZE
const CAPTION_Y_TOP = 1300; // top of caption zone (higher up = smaller number)
const FONT_SIZE     = 96;
const CAPTION_AVAIL = 1080 - 60 - 60; // usable width between left/right margins
const GLYPH_K       = 0.52; // approx glyph advance / fontSize for Changa One
const MIN_FONT      = 54;   // floor for the auto-shrink safety net
const MIN_CARD_SEC  = 1.0;  // min seconds a caption card stays on screen (anti-flash)
const CARD_GAP      = 0.04; // min gap kept between consecutive cards (s)
const TAIL_EXTEND   = 0.4;  // extra hold for the final card (s)

// ── Phrase chunking ───────────────────────────────────────────────────────

interface Phrase {
  words: WordTiming[];
  start: number;
  end: number;
  displayStart?: number;
  displayEnd?: number;
}

// A token ends a sentence if it closes with . ! ? … (optionally quoted/bracketed).
function isSentenceEnd(token: string): boolean {
  return /[.!?…]+["')\]]?$/.test(token.trim());
}

// Build single-line caption cards:
//   1. split the word stream into sentences (a card never crosses a sentence end),
//   2. split each sentence into evenly balanced ~TARGET_WORDS chunks (no awkward
//      "5 + 1" tail — e.g. 7 words → 4+3, not 4+2+1).
function buildPhrases(words: WordTiming[]): Phrase[] {
  const sentences: WordTiming[][] = [];
  let cur: WordTiming[] = [];
  for (const w of words) {
    cur.push(w);
    if (isSentenceEnd(w.word)) {
      sentences.push(cur);
      cur = [];
    }
  }
  if (cur.length) sentences.push(cur);

  const phrases: Phrase[] = [];
  for (const sent of sentences) {
    const n = sent.length;
    // total characters incl. single spaces between words
    const totalChars = sent.reduce((a, w) => a + w.word.length, 0) + (n - 1);
    // enough cards to satisfy BOTH the word target and the char budget,
    // so a few long words never produce one over-wide line.
    const cardsByWords = Math.ceil(n / TARGET_WORDS);
    const cardsByChars = Math.ceil(totalChars / MAX_CHARS);
    const cards = Math.max(1, cardsByWords, cardsByChars);
    const per = Math.ceil(n / cards);
    for (let i = 0; i < n; i += per) {
      const chunk = sent.slice(i, i + per);
      if (!chunk.length) continue;
      phrases.push({
        words: chunk,
        start: chunk[0].start,
        end: chunk[chunk.length - 1].end,
      });
    }
  }
  return phrases;
}

// Give each card a guaranteed minimum on-screen time by stretching its display
// window into the silent gap before the next card (without overlapping it). Stops
// fast-spoken cards from flashing past too quickly to read. The per-word yellow
// highlight still uses real word timings, so karaoke stays in sync while the card
// lingers.
function computeDisplayWindows(phrases: Phrase[]): Phrase[] {
  return phrases.map((p, i) => {
    const next = phrases[i + 1];
    const displayStart = p.start;
    let displayEnd: number;
    if (next) {
      const wantEnd = Math.max(p.end + 0.12, p.start + MIN_CARD_SEC);
      const hardEnd = next.start - CARD_GAP;
      displayEnd = Math.max(p.end + 0.05, Math.min(wantEnd, hardEnd));
    } else {
      displayEnd = Math.max(p.end + 0.12, p.start + MIN_CARD_SEC, p.end + TAIL_EXTEND);
    }
    return { ...p, displayStart, displayEnd };
  });
}

// Auto-shrink safety net: pick a font size that guarantees the phrase fits on
// one line within CAPTION_AVAIL, capped at FONT_SIZE and floored at MIN_FONT.
function fitFontSize(words: WordTiming[]): number {
  const chars = words.reduce((a, w) => a + w.word.length, 0) + Math.max(0, words.length - 1);
  const ideal = CAPTION_AVAIL / (GLYPH_K * Math.max(1, chars));
  return Math.max(MIN_FONT, Math.min(FONT_SIZE, Math.floor(ideal)));
}

// ── Karaoke caption layer ─────────────────────────────────────────────────

const KaraokeWord: React.FC<{
  word: WordTiming;
  t: number;
  frame: number;
  fps: number;
}> = ({ word, t, frame, fps }) => {
  const isActive = t >= word.start && t < word.end;

  // Active word: color change only (no scale/pop, no movement).
  return (
    <span
      style={{
        display: "inline-block",
        color: isActive ? "#FFE34D" : "#FFFFFF",
        // Thick dark outline for legibility over any footage
        textShadow: [
          "-3px -3px 0 #000",
          " 3px -3px 0 #000",
          "-3px  3px 0 #000",
          " 3px  3px 0 #000",
          "-3px  0   0 #000",
          " 3px  0   0 #000",
          " 0   -3px 0 #000",
          " 0    3px 0 #000",
          // Extra glow for the active word
          ...(isActive ? ["0 0 18px #FFE34D66"] : []),
        ].join(", "),
      }}
    >
      {word.word}
    </span>
  );
};

const KaraokeCaptions: React.FC<{
  words: WordTiming[];
  fps: number;
}> = ({ words, fps }) => {
  const frame = useCurrentFrame();
  const t     = frame / fps;

  const phrases = useMemo(() => computeDisplayWindows(buildPhrases(words)), [words]);

  // Find which card to show, using its (min-duration) display window.
  const activePhrase = useMemo(() => {
    for (const p of phrases) {
      const ds = p.displayStart ?? p.start;
      const de = p.displayEnd ?? p.end;
      if (t >= ds && t <= de) return p;
    }
    return null;
  }, [phrases, t]);

  if (!activePhrase) return null;

  // Fade the whole card in/out over its display window.
  const ds = activePhrase.displayStart ?? activePhrase.start;
  const de = activePhrase.displayEnd ?? activePhrase.end;
  const fadeIn  = interpolate(t, [ds, ds + 0.08], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const fadeOut = interpolate(t, [de - 0.1, de], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const opacity = Math.min(fadeIn, fadeOut);

  const fontSize = fitFontSize(activePhrase.words);

  return (
    <div
      style={{
        position: "absolute",
        top: CAPTION_Y_TOP,
        left: 60,
        right: 60,
        display: "flex",
        flexWrap: "nowrap",        // single line — never wrap to a 2nd row
        justifyContent: "center",
        alignItems: "baseline",
        columnGap: "0.25em",
        whiteSpace: "nowrap",
        opacity,
        fontFamily: '"Changa One", sans-serif',
        fontSize,
        fontWeight: 400, // Changa One is inherently heavy
        lineHeight: 1.15,
        textAlign: "center",
      }}
    >
      {activePhrase.words.map((w, i) => (
        <KaraokeWord key={`${w.word}-${i}-${w.start}`} word={w} t={t} frame={frame} fps={fps} />
      ))}
    </div>
  );
};

// ── Main composition ──────────────────────────────────────────────────────

export const BrollShort: React.FC<BrollShortProps> = ({
  clipsFile,
  paperFile,
  fontFile,
  fps,
  words,
  hookText = "",
}) => {
  const { width, height } = useVideoConfig();

  // Square clip dimensions and vertical centering
  const squareSize = 1080;
  const squareTop  = Math.round((height - squareSize) / 2); // (1920 - 1080) / 2 = 420

  // Inject @font-face once so Remotion's headless Chrome picks up Changa One
  const fontFaceStyle = `
    @font-face {
      font-family: "Changa One";
      src: url("${staticFile(fontFile)}") format("truetype");
      font-weight: 400;
      font-style: normal;
    }
  `;

  return (
    <AbsoluteFill style={{ width, height, overflow: "hidden", background: "#111" }}>
      {/* Inject font */}
      <style dangerouslySetInnerHTML={{ __html: fontFaceStyle }} />

      {/* Layer 1: Paper grid background — cover-fill 1080×1920 */}
      <AbsoluteFill>
        <Img
          src={staticFile(paperFile)}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
          }}
        />
      </AbsoluteFill>

      {/* Layer 2: Square clip montage — 1080×1080 centered vertically */}
      <div
        style={{
          position: "absolute",
          top: squareTop,
          left: 0,
          width: squareSize,
          height: squareSize,
          borderRadius: 24,
          overflow: "hidden",
          boxShadow: "0 8px 48px rgba(0,0,0,0.55), 0 2px 12px rgba(0,0,0,0.35)",
        }}
      >
        <OffthreadVideo
          src={staticFile(clipsFile)}
          style={{
            width: squareSize,
            height: squareSize,
            objectFit: "cover",
          }}
          muted
        />
      </div>

      {/* Top clickbait hook headline — centered in the paper band above the square */}
      {hookText ? (
        <div
          style={{
            position: "absolute",
            top: 40,
            left: 44,
            right: 44,
            // Clip strictly to the paper band above the square — never overlaps video.
            height: squareTop - 40 - 24,
            overflow: "hidden",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div
            style={{
              width: "100%",
              textAlign: "center",
              lineHeight: 1.22,
              // Balance the wrap so the two lines are roughly equal length.
              textWrap: "balance",
              wordBreak: "keep-all",
            } as React.CSSProperties}
          >
            <span
              style={{
                // Bigger title. Long ones simply wrap to 2 balanced lines; the
                // band has overflow:hidden so it can never spill onto the video.
                fontSize: hookText.length > 22 ? 92 : hookText.length > 13 ? 106 : 118,
                fontFamily: '"Changa One", sans-serif',
                textTransform: "uppercase",
                // No box — bold yellow text with a thick dark outline so it still
                // pops off the paper background.
                color: "#FFE34D",
                WebkitTextStroke: "3px #111",
                textShadow: [
                  "-4px -4px 0 #111", " 4px -4px 0 #111",
                  "-4px  4px 0 #111", " 4px  4px 0 #111",
                  "0 6px 0 #111", "0 12px 26px rgba(0,0,0,0.5)",
                ].join(", "),
              }}
            >
              {hookText}
            </span>
          </div>
        </div>
      ) : null}

      {/* Layer 3: Animated karaoke captions */}
      <KaraokeCaptions words={words} fps={fps} />
    </AbsoluteFill>
  );
};
