// captions.mjs — pure caption logic extracted from BrollShort.tsx
// Plain ESM, no TypeScript types. Importable by both TSX (via bundler) and node tests.

/**
 * Returns true if `token` ends a sentence (.  !  ?  …  optionally followed by
 * a closing quote/bracket).
 * @param {string} token
 * @returns {boolean}
 */
export function isSentenceEnd(token) {
  return /[.!?…]+["')\]]?$/.test(token.trim());
}

/**
 * Split a WordTiming[] into sentence-aware, balanced ~targetWords-word cards.
 * A card never crosses a sentence boundary.
 *
 * @param {Array<{word: string, start: number, end: number}>} words
 * @param {{ targetWords?: number, maxChars?: number }} opts
 * @returns {Array<{words: Array<{word:string,start:number,end:number}>, start:number, end:number}>}
 */
export function buildPhrases(words, { targetWords = 4, maxChars = 17 } = {}) {
  // Split stream into sentences
  const sentences = [];
  let cur = [];
  for (const w of words) {
    cur.push(w);
    if (isSentenceEnd(w.word)) {
      sentences.push(cur);
      cur = [];
    }
  }
  if (cur.length) sentences.push(cur);

  const phrases = [];
  for (const sent of sentences) {
    const n = sent.length;
    // total chars including single spaces between words
    const totalChars = sent.reduce((a, w) => a + w.word.length, 0) + (n - 1);
    const cardsByWords = Math.ceil(n / targetWords);
    const cardsByChars = Math.ceil(totalChars / maxChars);
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

/**
 * Auto-shrink safety net: pick the largest font size that fits `words` on one
 * line within `avail` pixels, clamped to [minFont, maxFont].
 *
 * @param {Array<{word: string}>} words
 * @param {{ avail?: number, glyphK?: number, maxFont?: number, minFont?: number }} opts
 * @returns {number}
 */
export function fitFontSize(words, { avail = 960, glyphK = 0.52, maxFont = 96, minFont = 54 } = {}) {
  const chars = words.reduce((a, w) => a + w.word.length, 0) + Math.max(0, words.length - 1);
  const ideal = avail / (glyphK * Math.max(1, chars));
  return Math.max(minFont, Math.min(maxFont, Math.floor(ideal)));
}

/**
 * Stretch each phrase's display window so fast-spoken cards stay visible for
 * at least `minCardSec` seconds, without overlapping the next card.
 *
 * Each returned phrase gains `displayStart` and `displayEnd` fields:
 *   displayStart = phrase.start  (unchanged)
 *   non-last: displayEnd = min(wantEnd, nextPhrase.start - gap), at least phrase.end + 0.05
 *   last:     displayEnd = max(phrase.end + 0.12, phrase.start + minCardSec, phrase.end + tailExtend)
 *
 * @param {Array<{words: any[], start: number, end: number}>} phrases
 * @param {{ minCardSec?: number, gap?: number, tailExtend?: number }} opts
 * @returns {Array<{words: any[], start: number, end: number, displayStart: number, displayEnd: number}>}
 */
export function computeDisplayWindows(phrases, { minCardSec = 1.0, gap = 0.04, tailExtend = 0.4 } = {}) {
  return phrases.map((phrase, i) => {
    const displayStart = phrase.start;
    let displayEnd;

    if (i < phrases.length - 1) {
      const next = phrases[i + 1];
      const wantEnd = Math.max(phrase.end + 0.12, phrase.start + minCardSec);
      const hardEnd = next.start - gap;
      // Never let displayEnd be less than phrase.end + 0.05; never overlap next card
      displayEnd = Math.max(phrase.end + 0.05, Math.min(wantEnd, hardEnd));
    } else {
      // Last card: extend tail
      displayEnd = Math.max(phrase.end + 0.12, phrase.start + minCardSec, phrase.end + tailExtend);
    }

    return { ...phrase, displayStart, displayEnd };
  });
}

/**
 * Convenience: buildPhrases then computeDisplayWindows in one call.
 *
 * @param {Array<{word: string, start: number, end: number}>} words
 * @param {object} opts  — merged into both buildPhrases and computeDisplayWindows
 * @returns {Array<{words: any[], start: number, end: number, displayStart: number, displayEnd: number}>}
 */
export function buildCaptionCards(words, opts = {}) {
  return computeDisplayWindows(buildPhrases(words, opts), opts);
}
