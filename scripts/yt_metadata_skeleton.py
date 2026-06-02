"""
Print a fill-in YouTube metadata template for a /video run.

Replaces ~700 tokens of inline Step-6 prose in the /video skill with ~200 tokens
of template output. The LLM fills in the placeholders using script + topic in
its context.

Usage:
    .venv/bin/python scripts/yt_metadata_skeleton.py <script_path> [topic] [skill_id]
"""

import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: yt_metadata_skeleton.py <script_path> [topic] [skill_id]", file=sys.stderr)
        sys.exit(2)

    script_path = Path(sys.argv[1])
    topic = sys.argv[2] if len(sys.argv) > 2 else script_path.parent.name
    skill = sys.argv[3] if len(sys.argv) > 3 else ""

    # Title formulas by skill family
    title_hint = {
        "comparison": '"X vs Y — The One Nobody Saw Coming"',
        "easter_eggs": '"5 [Topic] Details You Walked Right Past"',
        "dark_secrets": '"The [Topic] Truth They Tried To Bury"',
        "lore_deep_dive": '"The REAL Reason [Topic] [Did Something]"',
        "brainrot": '"[Topic] Was BUILT Different"',
    }.get(skill, '"[Topic] — [INTENSITY WORD] Truth"')

    print(f"""# Fill in below using script text + topic={topic!r} skill={skill!r}
# Title rules: <=60 chars; ONE of [question hook | controversial claim | "everyone got it wrong" | number reveal];
#   include intensity word (ACTUALLY, REAL TRUTH, NOBODY TALKS ABOUT, BROKEN, INSANE);
#   suggested formula for this skill: {title_hint}

TITLE
<title here>

# Description rules: 4 lines.
#   L1: punchier version of script's hook
#   L2: tease the twist, no spoiler
#   L3: comment-driving question
#   L4: subscribe CTA + <=2 emojis

DESCRIPTION
<line 1>
<line 2>
<line 3>
<line 4>

# Hashtag mix (8-10 total): 1-2 broad (#anime #shorts) + 2-3 franchise + 2-3 character/topic + 1-2 trend (#vsbattle #anitokk)

HASHTAGS
#tag1 #tag2 ...

# Tag rules (15-20, lowercase, comma-separated, no #):
#   subjects (full + abbreviated), franchise (full + abbreviated),
#   techniques/scenes mentioned in script, format tags
#   (vs battle, who would win, anime explained, anime shorts, anime tier list)

TAGS
tag1, tag2, ...
""")


if __name__ == "__main__":
    main()
