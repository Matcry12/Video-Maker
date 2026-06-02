"""
Render a faceless Psychology-Facts stock-b-roll YouTube Short from a topic.

Usage:
    .venv/bin/python scripts/render_broll.py --topic "why you wake up at 3am"
    .venv/bin/python scripts/render_broll.py --topic "the sunk cost fallacy" --words 150
    .venv/bin/python scripts/render_broll.py --topic "why you procrastinate" --bgm-vol 0.12
    .venv/bin/python scripts/render_broll.py --topic "social media dopamine" --force

Exit codes:
    0  success
    1  error
    3  duplicate topic (use --force to override)
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a faceless Psychology-Facts b-roll YouTube Short."
    )
    parser.add_argument("--topic", required=True, help="Video topic (plain English).")
    parser.add_argument(
        "--words",
        type=int,
        default=130,
        metavar="N",
        help="Target voiceover word count (default: 130).",
    )
    parser.add_argument(
        "--bgm-vol",
        type=float,
        default=0.16,
        metavar="V",
        help="Background music volume 0.0–1.0 (default: 0.16).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip duplicate check and re-render even if topic was already produced.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Render only; skip the auto-upload to YouTube.",
    )
    parser.add_argument(
        "--channel",
        help="Channel from profile youtube.channels "
             "(default: youtube.pipeline_channels.broll).",
    )
    parser.add_argument(
        "--privacy",
        choices=["private", "unlisted", "public"],
        help="Upload immediately with this privacy instead of scheduling "
             "(e.g. --privacy private for a quick test).",
    )
    parser.add_argument(
        "--publish-at",
        help="Explicit scheduled publish time (ISO; held private until then). "
             "No tz => profile tz. Overrides the default next-8AM schedule.",
    )
    args = parser.parse_args()

    from src.broll import build_broll, BrollDuplicateError

    try:
        result = build_broll(
            args.topic,
            target_words=args.words,
            bgm_vol=args.bgm_vol,
            skip_dupe_check=args.force,
        )
    except BrollDuplicateError:
        print(
            f"⚠ Already produced this topic (use --force to override): {args.topic!r}"
        )
        sys.exit(3)

    print(f"FINAL VIDEO: {result.final_video}")
    yt = result.youtube or {}
    if yt.get("title"):
        print(f"\nYouTube title:       {yt['title']}")
    if yt.get("description"):
        print(f"\nYouTube description:\n{yt['description']}")
    if yt.get("hashtags"):
        ht = yt["hashtags"]
        print(f"\nYouTube hashtags:    {' '.join(ht) if isinstance(ht, list) else ht}")
    if yt.get("tags"):
        tags = yt["tags"]
        tags_str = ", ".join(tags) if isinstance(tags, list) else tags
        print(f"\nYouTube tags:        {tags_str}")

    # --- Auto-upload to YouTube (psychology-facts -> @UntoldPsychologyT) ---
    # Like /writer: by default schedule for the next profile slot (held private,
    # auto-publishes public). --privacy or --publish-at override that.
    if args.no_upload:
        return
    from src.agent_config import youtube_settings
    from src.youtube_upload import upload_run, meta_from_dict

    channel = args.channel or youtube_settings().get(
        "pipeline_channels", {}
    ).get("broll", "untold")
    meta = meta_from_dict(
        yt, privacy="private", publish_at=None, category_id="24", made_for_kids=False,
    )
    schedule_next = not (args.privacy or args.publish_at)
    try:
        url = upload_run(
            result.work_dir,
            video=result.final_video,
            meta=meta,
            channel=channel,
            privacy=args.privacy,
            publish_at=args.publish_at,
            schedule_next=schedule_next,
        )
        if url:
            when = "scheduled (next slot)" if schedule_next else (
                args.publish_at or args.privacy)
            print(f"\n✅ Uploaded to '{channel}' [{when}]: {url}")
    except Exception as exc:
        print(f"\n⚠ Upload to '{channel}' failed ({exc}). "
              f"Video is saved; retry: .venv/bin/python scripts/upload_to_youtube.py "
              f"{result.final_video} --channel {channel} --schedule-next")


if __name__ == "__main__":
    main()
