"""Upload a rendered run to YouTube via the Data API v3.

Usage:
    .venv/bin/python scripts/upload_to_youtube.py <run_dir | video.mp4> \
        [--script script.json] \
        [--publish-at "2026-06-01T18:00:00"] \
        [--privacy private|unlisted|public] \
        [--category 24] [--thumb thumb.jpg] [--dry-run]

Notes:
    - --publish-at takes an ISO time; no timezone => interpreted in the profile's
      youtube.timezone, then converted to UTC. Scheduling forces privacy=private.
    - First run opens a browser to grant access (see YOUTUBE_SETUP.md).
    - --dry-run prints the request body without uploading (no auth needed).
"""
import argparse
import logging
import sys
from pathlib import Path

# Allow running as a plain script: ensure project root on sys.path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.youtube_upload import upload_run  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Upload a rendered run to YouTube.")
    p.add_argument("target", help="Run directory, or a path to the .mp4 file.")
    p.add_argument("--script", help="Override path to script.json.")
    p.add_argument("--publish-at", help="Scheduled publish time (ISO; forces private).")
    p.add_argument("--schedule-next", action="store_true",
                   help="Schedule for the next profile schedule_time (e.g. 08:00); forces private.")
    p.add_argument("--privacy", choices=["private", "unlisted", "public"],
                   help="Privacy status (default from profile).")
    p.add_argument("--category", help="YouTube category ID (default from profile).")
    p.add_argument("--thumb", help="Override thumbnail path.")
    p.add_argument("--title", help="Upload an mp4 with this title and skip the "
                                   "script.json lookup (handy for quick test uploads).")
    p.add_argument("--description", default="", help="Description when using --title.")
    p.add_argument("--channel", help="Channel name from profile youtube.channels "
                                     "(default: youtube.default_channel). First use "
                                     "of a new channel opens a browser to pick it.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the request body without uploading.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    target = Path(args.target)
    if target.is_file() and target.suffix == ".mp4":
        run_dir = target.parent
        video = target
    else:
        run_dir = target
        video = None

    # --title => build metadata inline, no script.json required.
    meta = None
    if args.title:
        from src.youtube_upload import meta_from_dict
        meta = meta_from_dict(
            {"title": args.title, "description": args.description, "tags": []},
            privacy=args.privacy or "private", publish_at=None,
            category_id=args.category or "24", made_for_kids=False,
        )

    url = upload_run(
        run_dir,
        video=video,
        script=Path(args.script) if args.script else None,
        privacy=args.privacy,
        publish_at=args.publish_at,
        schedule_next=args.schedule_next,
        category_id=args.category,
        thumb=Path(args.thumb) if args.thumb else None,
        channel=args.channel,
        meta=meta,
        dry_run=args.dry_run,
    )
    if url:
        print(f"\nUploaded: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
