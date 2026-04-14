"""
Script linter — validates generated scripts before FFmpeg rendering.

Pure deterministic scoring, no LLM calls. Called after script generation
and before video rendering to gate quality.
"""

import re
from collections import Counter


_GENERIC_OPENERS = [
    r"in a catastrophic event",
    r"in history",
    r"throughout history",
    r"in a shocking turn",
    r"it was a day that",
    r"what if i told you",
    r"did you know",
]

_WEAK_ENDINGS = [
    r"for generations to come\.?$",
    r"changed the world forever\.?$",
    r"a haunting reminder\.?$",
    r"never be the same\.?$",
    r"the world would never forget\.?$",
    r"a lesson for\b.*$",
]

_DRAMATIC_ADJECTIVES = {
    "staggering", "devastating", "massive", "haunting", "catastrophic",
    "unprecedented", "deadly", "horrifying", "shocking", "tragic",
    "terrifying", "brutal", "horrific", "unimaginable",
}

# Vietnamese lint rules
_VN_GENERIC_OPENERS = [
    r"vào năm",
    r"trong lịch sử",
    r"câu chuyện về",
    r"ngày xửa ngày xưa",
    r"từ thuở xa xưa",
    r"có một",
]

_VN_WEAK_ENDINGS = [
    r"mãi mãi\.?$",
    r"cho muôn đời sau\.?$",
    r"không bao giờ quên\.?$",
    r"thay đổi thế giới\.?$",
    r"bài học cho\.?.*$",
]

_VN_DRAMATIC_ADJECTIVES = {
    "kinh hoàng", "thảm khốc", "khủng khiếp", "chấn động",
    "bi thảm", "tàn khốc", "khốc liệt", "thương tâm",
}

_NUMBER_RE = re.compile(r"\d+[.,]?\d*")


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return re.findall(r"[a-z]+", text.lower())


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def lint_script(
    script: dict,
    *,
    strict: bool = False,
) -> dict:
    """Score a generated script for quality.

    Returns:
        {
            "score": int,          # 0-100
            "status": str,         # "pass" | "soft_fail" | "hard_fail"
            "hard_fail": bool,
            "issues": [
                {"code": str, "severity": str, "detail": str, "block_index": int | None}
            ]
        }

    Decision rules:
        score >= 75 -> "pass"
        60-74       -> "soft_fail" (regenerate once)
        < 60        -> "hard_fail" (do not render)
    """
    issues: list[dict] = []
    score = 100

    blocks = script.get("blocks", [])
    texts = [b.get("text", "") for b in blocks]

    # --- TOO_FEW_BLOCKS (-15) ---
    if len(blocks) < 3:
        score -= 15
        issues.append({
            "code": "TOO_FEW_BLOCKS",
            "severity": "medium",
            "detail": f"Script has only {len(blocks)} block(s), minimum recommended is 3.",
            "block_index": None,
        })

    # --- TOO_SHORT (-10) ---
    total_chars = sum(len(t) for t in texts)
    if total_chars < 200:
        score -= 10
        issues.append({
            "code": "TOO_SHORT",
            "severity": "medium",
            "detail": f"Total text length is {total_chars} characters, minimum recommended is 200.",
            "block_index": None,
        })

    # --- GENERIC_OPENER (-25) ---
    if texts:
        first_lower = texts[0].strip().lower()
        for pattern in _GENERIC_OPENERS:
            if re.match(pattern, first_lower):
                score -= 25
                issues.append({
                    "code": "GENERIC_OPENER",
                    "severity": "high",
                    "detail": f"First block starts with generic opener matching /{pattern}/.",
                    "block_index": 0,
                })
                break

    # --- WEAK_ENDING (-20) ---
    if texts:
        last_lower = texts[-1].strip().lower()
        for pattern in _WEAK_ENDINGS:
            if re.search(pattern, last_lower):
                score -= 20
                issues.append({
                    "code": "WEAK_ENDING",
                    "severity": "high",
                    "detail": f"Last block ends with weak ending matching /{pattern}/.",
                    "block_index": len(texts) - 1,
                })
                break

    # --- REPEATED_IMPACT (-15 each) ---
    number_counts: Counter = Counter()
    for text in texts:
        block_numbers = set(_NUMBER_RE.findall(text))
        for num in block_numbers:
            number_counts[num] += 1

    for num, count in number_counts.items():
        if count > 2:
            score -= 15
            issues.append({
                "code": "REPEATED_IMPACT",
                "severity": "medium",
                "detail": f"Number '{num}' appears in {count} blocks (max 2).",
                "block_index": None,
            })

    # --- REPEATED_ADJECTIVE (-10 each) ---
    adjective_counts: Counter = Counter()
    for text in texts:
        words = set(_tokenize(text))
        for adj in _DRAMATIC_ADJECTIVES & words:
            adjective_counts[adj] += 1

    for adj, count in adjective_counts.items():
        if count > 1:
            score -= 10
            issues.append({
                "code": "REPEATED_ADJECTIVE",
                "severity": "low",
                "detail": f"Dramatic adjective '{adj}' used {count} times across blocks.",
                "block_index": None,
            })

    # --- HIGH_AVG_SENTENCE_LENGTH (-10) ---
    all_sentences: list[str] = []
    for text in texts:
        sentences = re.split(r"[.!?]+", text)
        all_sentences.extend(s.strip() for s in sentences if s.strip())

    if all_sentences:
        avg_words = sum(len(s.split()) for s in all_sentences) / len(all_sentences)
        if avg_words > 22:
            score -= 10
            issues.append({
                "code": "HIGH_AVG_SENTENCE_LENGTH",
                "severity": "low",
                "detail": f"Average sentence length is {avg_words:.1f} words (max 22).",
                "block_index": None,
            })

    # --- LOW_BLOCK_NOVELTY (-15 each pair) ---
    for i in range(len(texts) - 1):
        words_a = set(_tokenize(texts[i]))
        words_b = set(_tokenize(texts[i + 1]))
        similarity = _jaccard(words_a, words_b)
        if similarity > 0.45:
            score -= 15
            issues.append({
                "code": "LOW_BLOCK_NOVELTY",
                "severity": "medium",
                "detail": f"Blocks {i} and {i + 1} have high word overlap (Jaccard={similarity:.2f}, threshold 0.45).",
                "block_index": i,
            })

    # --- LANGUAGE-AWARE RULES ---
    lang = script.get("language", "en-US").lower()

    if lang.startswith("vi"):
        # Vietnamese: GENERIC_OPENER
        if texts:
            first_lower = texts[0].strip().lower()
            for pattern in _VN_GENERIC_OPENERS:
                if re.match(pattern, first_lower):
                    score -= 25
                    issues.append({
                        "code": "VN_GENERIC_OPENER",
                        "severity": "high",
                        "detail": f"First block starts with Vietnamese generic opener matching /{pattern}/.",
                        "block_index": 0,
                    })
                    break

        # Vietnamese: WEAK_ENDING
        if texts:
            last_lower = texts[-1].strip().lower()
            for pattern in _VN_WEAK_ENDINGS:
                if re.search(pattern, last_lower):
                    score -= 20
                    issues.append({
                        "code": "VN_WEAK_ENDING",
                        "severity": "high",
                        "detail": f"Last block ends with Vietnamese weak ending matching /{pattern}/.",
                        "block_index": len(texts) - 1,
                    })
                    break

        # Vietnamese: DRAMATIC_ADJECTIVES
        for idx, text in enumerate(texts):
            text_lower = text.lower()
            for adj in _VN_DRAMATIC_ADJECTIVES:
                count = text_lower.count(adj)
                if count > 1:
                    score -= 10
                    issues.append({
                        "code": "VN_DRAMATIC_ADJECTIVE",
                        "severity": "low",
                        "detail": f"Vietnamese dramatic adjective '{adj}' used {count} times in block {idx}.",
                        "block_index": idx,
                    })

    # Clamp score
    score = max(score, 0)

    # Determine status
    if score >= 75:
        status = "pass"
    elif score >= 60:
        status = "soft_fail"
    else:
        status = "hard_fail"

    return {
        "score": score,
        "status": status,
        "hard_fail": status == "hard_fail",
        "issues": issues,
    }
