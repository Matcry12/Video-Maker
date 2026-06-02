// captions.test.mjs — node built-in test runner
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildPhrases, computeDisplayWindows, buildCaptionCards, isSentenceEnd } from "./captions.mjs";

// ── helpers ───────────────────────────────────────────────────────────────────

function w(word, start, end) {
  return { word, start, end };
}

// ── isSentenceEnd ─────────────────────────────────────────────────────────────

test("isSentenceEnd: period", () => {
  assert.ok(isSentenceEnd("hello."));
  assert.ok(isSentenceEnd("hello!"));
  assert.ok(isSentenceEnd("hello?"));
  assert.ok(isSentenceEnd("hello…"));
  assert.ok(isSentenceEnd('hello."'));
  assert.ok(!isSentenceEnd("hello"));
  assert.ok(!isSentenceEnd("hello,"));
});

// ── buildPhrases: sentence boundary ──────────────────────────────────────────

test("buildPhrases: never crosses a sentence boundary", () => {
  // "a." is a sentence end; "b" and "c" belong to the next sentence.
  const words = [w("a.", 0, 0.3), w("b", 0.4, 0.6), w("c", 0.7, 0.9)];
  const phrases = buildPhrases(words);

  // First card must be exactly ["a."]
  assert.equal(phrases[0].words.length, 1);
  assert.equal(phrases[0].words[0].word, "a.");

  // No card should contain words from both sentences
  for (const phrase of phrases) {
    const hasEnd = phrase.words.some((w) => isSentenceEnd(w.word));
    const hasNonEnd = phrase.words.some((w) => !isSentenceEnd(w.word));
    // A card CAN have a sentence-end word + other words from the SAME sentence
    // but must never have a sentence-end word that is not the last word unless
    // those remaining words also belong to the same sentence.
    // Simpler check: if there's a sentence-ender it must be the last word of the card.
    const sentEndIdx = phrase.words.findIndex((w) => isSentenceEnd(w.word));
    if (sentEndIdx !== -1) {
      assert.equal(sentEndIdx, phrase.words.length - 1,
        `sentence-ender must be last word of card, got idx ${sentEndIdx} in [${phrase.words.map(w=>w.word)}]`);
    }
  }
});

test("buildPhrases: multi-sentence stream, each sentence isolated", () => {
  // Two sentences: "Hello world." and "Foo bar."
  const words = [
    w("Hello", 0, 0.3), w("world.", 0.4, 0.7),
    w("Foo", 0.8, 1.0), w("bar.", 1.1, 1.3),
  ];
  const phrases = buildPhrases(words);
  // Every phrase must come entirely from one sentence
  for (const phrase of phrases) {
    const hasFirstSentEnd = phrase.words.some(x => x.word === "world.");
    const hasSecondSent = phrase.words.some(x => x.word === "Foo" || x.word === "bar.");
    assert.ok(!(hasFirstSentEnd && hasSecondSent),
      "card crosses sentence boundary");
  }
});

// ── computeDisplayWindows: min on-screen time ─────────────────────────────────

test("fast 4-word single sentence gets minCardSec display window (last card)", () => {
  // All spoken in 0.2s, next card far away (or it's the last card)
  const words = [w("a", 0.0, 0.05), w("b", 0.06, 0.1), w("c", 0.11, 0.15), w("d", 0.16, 0.2)];
  const phrases = buildPhrases(words);
  const cards = computeDisplayWindows(phrases, { minCardSec: 1.0 });

  // Should be a single card (one sentence, 4 words)
  assert.equal(cards.length, 1);
  const dur = cards[0].displayEnd - cards[0].displayStart;
  assert.ok(dur >= 1.0, `expected displayEnd-displayStart >= 1.0, got ${dur}`);
});

test("fast card followed by next card: displayEnd does not exceed next.start - gap", () => {
  // Two fast sentences, back-to-back
  const words1 = [w("hello.", 0.0, 0.2)];
  const words2 = [w("world.", 0.5, 0.7)];
  const allWords = [...words1, ...words2];
  const phrases = buildPhrases(allWords);
  const cards = computeDisplayWindows(phrases, { minCardSec: 1.0, gap: 0.04 });

  assert.equal(cards.length, 2);
  const gap = 0.04;
  assert.ok(
    cards[0].displayEnd <= cards[1].displayStart - gap + 1e-9,
    `card[0].displayEnd=${cards[0].displayEnd} must be <= card[1].start - gap (${cards[1].displayStart - gap})`
  );
});

test("non-last card: displayEnd is at least phrase.end + 0.05", () => {
  // Two cards where the next card starts far away (3s gap)
  const words = [
    w("hello.", 0.0, 0.1),
    w("world.", 3.0, 3.1),
  ];
  const phrases = buildPhrases(words);
  const cards = computeDisplayWindows(phrases, { minCardSec: 1.0, gap: 0.04 });

  assert.ok(cards[0].displayEnd >= cards[0].end + 0.05 - 1e-9,
    `displayEnd ${cards[0].displayEnd} < end+0.05 ${cards[0].end + 0.05}`);
});

test("last card: tail extension applied", () => {
  const words = [w("done.", 2.0, 2.3)];
  const phrases = buildPhrases(words);
  const cards = computeDisplayWindows(phrases, { minCardSec: 1.0, tailExtend: 0.4 });

  // displayEnd >= phrase.end + tailExtend
  assert.ok(cards[0].displayEnd >= cards[0].end + 0.4 - 1e-9,
    `tail extend: displayEnd=${cards[0].displayEnd}, end+0.4=${cards[0].end + 0.4}`);
});

test("two back-to-back cards: no overlap (gap respected)", () => {
  const words = [
    w("first.", 0.0, 0.5),
    w("second.", 0.6, 1.0),
  ];
  const phrases = buildPhrases(words);
  const cards = computeDisplayWindows(phrases, { minCardSec: 1.0, gap: 0.04 });

  assert.equal(cards.length, 2);
  assert.ok(
    cards[0].displayEnd <= cards[1].displayStart - 0.04 + 1e-9,
    `overlap: card[0].displayEnd=${cards[0].displayEnd} > card[1].displayStart-gap=${cards[1].displayStart - 0.04}`
  );
});

// ── buildCaptionCards convenience ─────────────────────────────────────────────

test("buildCaptionCards: has displayStart and displayEnd on every card", () => {
  const words = [
    w("The", 0.0, 0.1), w("quick.", 0.15, 0.3),
    w("Brown", 0.4, 0.5), w("fox.", 0.55, 0.65),
  ];
  const cards = buildCaptionCards(words);
  assert.ok(cards.length > 0);
  for (const c of cards) {
    assert.ok("displayStart" in c, "missing displayStart");
    assert.ok("displayEnd" in c, "missing displayEnd");
    assert.ok(c.displayEnd > c.displayStart, "displayEnd <= displayStart");
  }
});
