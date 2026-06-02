"""Extract chapters from a light novel PDF into individual .md files.

Usage:
    .venv/bin/python scripts/extract_chapters.py <pdf_path> [out_dir]

Each chapter is written to <out_dir>/<nn>_<slug>.md with a frontmatter header
so an LLM can identify source, title, and position at a glance.

Output dir defaults to the same folder as the PDF, under a `chapters/` subdir.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pdfplumber

# ---------------------------------------------------------------------------
# Chapter boundary patterns — order matters (most specific first)
# ---------------------------------------------------------------------------
_BOUNDARIES = [
    # Prologue / named intro section
    (r"^Coquelicots\s+Blooming\s+Across\s+the\s+Battlefield", "00_prologue"),
    # Numbered chapters
    (r"^Chapter\s+1\b", "01_chapter_1"),
    (r"^Chapter\s+2\b", "02_chapter_2"),
    (r"^Chapter\s+3\b", "03_chapter_3"),
    (r"^Interlude[:\s]+The\s+Headless\s+Knight(?!\s+I)", "04_interlude_1"),
    (r"^Chapter\s+4\b", "05_chapter_4"),
    (r"^Interlude[:\s]+The\s+Headless\s+Knight\s+II(?!I)", "06_interlude_2"),
    (r"^Chapter\s+5\b", "07_chapter_5"),
    (r"^Interlude[:\s]+The\s+Headless\s+Knight\s+III(?!I)", "08_interlude_3"),
    (r"^Chapter\s+6\b", "09_chapter_6"),
    (r"^Interlude[:\s]+The\s+Headless\s+Knight\s+IV", "10_interlude_4"),
    (r"^Chapter\s+7\b", "11_chapter_7"),
    (r"^Epilogue[:\s]+The\s+Bloodstained", "12_epilogue_1"),
    (r"^Epilogue\s+II", "13_epilogue_2"),
    (r"^Afterword", "14_afterword"),
]

_COMPILED = [(re.compile(pat, re.IGNORECASE | re.MULTILINE), slug)
             for pat, slug in _BOUNDARIES]

# Full titles for frontmatter
_TITLES = {
    "00_prologue": "Coquelicots Blooming Across the Battlefield",
    "01_chapter_1": "Chapter 1: A Battlefield with Zero Casualties",
    "02_chapter_2": "Chapter 2: All Quiet on the Skeletal Front",
    "03_chapter_3": "Chapter 3: To Your Gallant Visage at the Underworld's Edge",
    "04_interlude_1": "Interlude: The Headless Knight",
    "05_chapter_4": "Chapter 4: I Am Legion, for We Are Many",
    "06_interlude_2": "Interlude: The Headless Knight II",
    "07_chapter_5": "Chapter 5: Fuckin' Glory to the Spearhead Squadron",
    "08_interlude_3": "Interlude: The Headless Knight III",
    "09_chapter_6": "Chapter 6: Fiat Justitia Ruat Caelum",
    "10_interlude_4": "Interlude: The Headless Knight IV",
    "11_chapter_7": "Chapter 7: Good-bye",
    "12_epilogue_1": "Epilogue: The Bloodstained Queen's Sojourn",
    "13_epilogue_2": "Epilogue II: Reboot",
    "14_afterword": "Afterword",
}

# Noise patterns to strip before writing
_NOISE = re.compile(
    r"(Page\|\d+\s*Goldenagato\|[^\n]*|Goldenagato\|[^\n]*|Page\|\d+)",
    re.IGNORECASE,
)


def _detect_boundary(text: str) -> str | None:
    """Return slug if text starts a new chapter section, else None."""
    stripped = text.lstrip()
    for pat, slug in _COMPILED:
        if pat.match(stripped):
            return slug
    return None


def _clean(text: str) -> str:
    text = _NOISE.sub("", text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract(pdf_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    sections: dict[str, list[str]] = {}   # slug → [page texts]
    order: list[str] = []
    current: str | None = None

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"PDF: {total} pages")
        for i, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            slug = _detect_boundary(text)
            if slug and slug != current:
                current = slug
                if current not in sections:
                    sections[current] = []
                    order.append(current)
                print(f"  p{i+1:03d} → {current}")
            if current:
                sections[current].append(text)

    written: list[Path] = []
    for slug in order:
        title = _TITLES.get(slug, slug)
        body = _clean("\n\n".join(sections[slug]))
        frontmatter = (
            f"---\n"
            f"source: 86—EIGHTY-SIX Vol. 1\n"
            f"section: {slug}\n"
            f"title: {title}\n"
            f"---\n\n"
        )
        out_path = out_dir / f"{slug}.md"
        out_path.write_text(frontmatter + body, encoding="utf-8")
        words = len(body.split())
        print(f"  wrote {out_path.name}  ({words} words)")
        written.append(out_path)

    return written


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pdf_path = Path(sys.argv[1]).expanduser().resolve()
    out_dir = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 \
        else pdf_path.parent / "chapters"

    written = extract(pdf_path, out_dir)
    print(f"\nDone — {len(written)} files in {out_dir}")


if __name__ == "__main__":
    main()
