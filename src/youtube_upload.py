"""YouTube Data API v3 uploader for rendered videos.

Pushes a finished MP4 (+ thumbnail + metadata) to YouTube via the official API
with OAuth2. Supports scheduled publishing (status.publishAt).

One-time setup (see YOUTUBE_SETUP.md):
  1. Google Cloud project -> enable "YouTube Data API v3"
  2. OAuth client ID -> "Desktop app" -> download as client_secret.json (project root)
  3. First run opens a browser to consent; the refresh token is cached in token.json.

Quota note: each videos.insert costs ~1600 units of the default 10k/day quota
(~6 uploads/day). Audit note: until the Cloud project passes a YouTube audit,
API-uploaded videos are forced to private and publishAt will NOT make them
public — they stay private until you publish manually.

Programmatic entry: upload_run(run_dir, ...) -> watch URL.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CLIENT_SECRET = _PROJECT_ROOT / "client_secret.json"
_TOKEN_FILE = _PROJECT_ROOT / "token.json"

# youtube.upload covers videos.insert AND thumbnails.set -> single consent scope.
_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# YouTube field limits.
_TITLE_MAX = 100
_DESC_MAX = 5000

_VALID_PRIVACY = {"private", "unlisted", "public"}


@dataclass
class VideoMeta:
    title: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    category_id: str = "24"  # 24 = Entertainment
    privacy: str = "private"
    publish_at: Optional[str] = None  # RFC3339 UTC, e.g. "2026-06-01T18:00:00Z"
    made_for_kids: bool = False

    def to_body(self) -> dict:
        title = (self.title or "Untitled").strip()[:_TITLE_MAX]
        status: dict = {
            "privacyStatus": self.privacy,
            "selfDeclaredMadeForKids": bool(self.made_for_kids),
        }
        # publishAt requires privacyStatus=private (API rule).
        if self.publish_at:
            status["privacyStatus"] = "private"
            status["publishAt"] = self.publish_at
        return {
            "snippet": {
                "title": title,
                "description": (self.description or "").strip()[:_DESC_MAX],
                "tags": [str(t).strip() for t in self.tags if str(t).strip()],
                "categoryId": str(self.category_id),
            },
            "status": status,
        }


# ── auth ──────────────────────────────────────────────────────────────────


def get_authenticated_service(
    client_secret: Path = _CLIENT_SECRET,
    token_file: Path = _TOKEN_FILE,
):
    """Return an authorized youtube API client.

    Loads cached token; refreshes if expired; runs the browser consent flow on
    first use. Raises FileNotFoundError if client_secret.json is missing.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired YouTube token.")
            creds.refresh(Request())
        else:
            if not client_secret.exists():
                raise FileNotFoundError(
                    f"Missing OAuth client secret at {client_secret}. "
                    "See YOUTUBE_SETUP.md to create one."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), _SCOPES)
            creds = flow.run_local_server(port=0)
        token_file.write_text(creds.to_json(), encoding="utf-8")
        logger.info("Saved YouTube token to %s", token_file)

    return build("youtube", "v3", credentials=creds)


# ── channel selection ───────────────────────────────────────────────────────


def resolve_token_file(channel: Optional[str] = None) -> Path:
    """Map a channel name to its cached OAuth token path.

    A single Google account can own several YouTube channels (Brand Accounts);
    each channel needs its own token (chosen on the consent screen). Channels are
    declared in profiles/default.json -> youtube.channels as {name: token_file}.
    Relative token paths resolve against the project root.

    Falls back to the legacy single-token behaviour (token.json) when no channels
    map is configured. Raises KeyError if `channel` is given but not declared.
    """
    from src.agent_config import youtube_settings

    cfg = youtube_settings()
    channels: dict = cfg.get("channels") or {}
    name = channel or cfg.get("default_channel", "main")

    if not channels:
        return _TOKEN_FILE  # legacy: no channels configured

    if name not in channels:
        raise KeyError(
            f"Channel {name!r} not found in profile youtube.channels "
            f"(have: {sorted(channels)}). Add it to profiles/default.json."
        )
    token = Path(channels[name])
    if not token.is_absolute():
        token = _PROJECT_ROOT / token
    return token


# ── time helpers ────────────────────────────────────────────────────────────


def to_rfc3339_utc(when: str, tz_name: str = "UTC") -> str:
    """Normalize a publish time to RFC3339 UTC (e.g. '2026-06-01T18:00:00Z').

    Accepts an ISO-8601 string. If it carries no timezone, it is interpreted in
    `tz_name` and converted to UTC.
    """
    s = when.strip()
    # Allow a trailing Z by mapping to +00:00 for fromisoformat.
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz_name))
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def next_occurrence(hhmm: str, tz_name: str = "UTC", *, now: Optional[datetime] = None) -> str:
    """Next future occurrence of a local HH:MM, as RFC3339 UTC.

    If today's HH:MM (in tz_name) has already passed, returns tomorrow's.
    `now` is for testing; defaults to the real current time.
    """
    tz = ZoneInfo(tz_name)
    hour, minute = (int(x) for x in hhmm.strip().split(":"))
    current = (now.astimezone(tz) if now else datetime.now(tz))
    target = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= current:
        target += timedelta(days=1)
    return target.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── upload ────────────────────────────────────────────────────────────────


def upload_video(youtube, video_path: Path, meta: VideoMeta) -> str:
    """Resumable upload of one video. Returns the new video ID."""
    from googleapiclient.http import MediaFileUpload

    if meta.privacy not in _VALID_PRIVACY:
        raise ValueError(f"privacy must be one of {_VALID_PRIVACY}, got {meta.privacy!r}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
    request = youtube.videos().insert(
        part="snippet,status", body=meta.to_body(), media_body=media
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            logger.info("Upload progress: %d%%", int(status.progress() * 100))
    video_id = response["id"]
    logger.info("Uploaded video id=%s", video_id)
    return video_id


def set_thumbnail(youtube, video_id: str, thumb_path: Path) -> None:
    """Attach a custom thumbnail (requires a verified channel)."""
    from googleapiclient.http import MediaFileUpload

    if not thumb_path.exists():
        logger.warning("Thumbnail not found, skipping: %s", thumb_path)
        return
    youtube.thumbnails().set(
        videoId=video_id, media_body=MediaFileUpload(str(thumb_path))
    ).execute()
    logger.info("Set thumbnail for %s", video_id)


# ── metadata extraction ─────────────────────────────────────────────────────


def build_meta_from_script(script_path: Path, *, privacy: str, publish_at: Optional[str],
                           category_id: str, made_for_kids: bool) -> VideoMeta:
    """Build VideoMeta from a script.json.

    Uses the structured `youtube` field when present (long-form). Falls back to
    `topic` for the title and the first narration block for the description
    (Shorts, which lack a youtube field).
    """
    data = json.loads(script_path.read_text(encoding="utf-8"))
    yt = data.get("youtube") or {}

    title = yt.get("title") or data.get("topic") or script_path.parent.name
    description = yt.get("description") or ""
    chapters = yt.get("chapters") or []
    hashtags = yt.get("hashtags") or []
    tags = yt.get("tags") or []

    if chapters:
        description = (description + "\n\n" + "\n".join(chapters)).strip()
    if hashtags:
        description = (description + "\n\n" + " ".join(hashtags)).strip()

    # Shorts fallback: no youtube field at all -> seed description from block text.
    if not yt and not description:
        blocks = data.get("blocks") or []
        if blocks:
            description = (blocks[0].get("text") or "")[:500]

    # Per-video category from the youtube field wins over the profile/CLI default.
    category_id = yt.get("category_id") or category_id

    return VideoMeta(
        title=str(title),
        description=str(description),
        tags=[str(t) for t in tags],
        category_id=str(category_id),
        privacy=privacy,
        publish_at=publish_at,
        made_for_kids=made_for_kids,
    )


def meta_from_dict(yt: dict, *, privacy: str, publish_at: Optional[str],
                   category_id: str, made_for_kids: bool) -> VideoMeta:
    """Build VideoMeta from an in-memory youtube dict ({title, description, tags}).

    Used by pipelines (e.g. /broll) that already hold their metadata and never
    write a script.json. Mirrors build_meta_from_script's field handling: a
    `hashtags` list is appended to the end of the description.
    """
    description = str(yt.get("description") or "")
    hashtags = yt.get("hashtags") or []
    if hashtags:
        description = (description + "\n\n" + " ".join(str(h) for h in hashtags)).strip()
    return VideoMeta(
        title=str(yt.get("title") or "Untitled"),
        description=description,
        tags=[str(t) for t in (yt.get("tags") or [])],
        category_id=str(yt.get("category_id") or category_id),
        privacy=privacy,
        publish_at=publish_at,
        made_for_kids=made_for_kids,
    )


def _find_artifacts(run_dir: Path, video: Optional[Path],
                    script: Optional[Path]) -> tuple[Path, Path, Optional[Path]]:
    """Resolve (script.json, video.mp4, thumbnail) from a run dir or explicit paths."""
    script = script or (run_dir / "script.json")
    if not script.exists():
        raise FileNotFoundError(f"No script.json (looked at {script})")

    if video is None:
        mp4s = sorted(run_dir.glob("*.mp4"))
        if not mp4s:
            # Common case: final video lives in output/videos/<run>.mp4
            guess = _PROJECT_ROOT / "output" / "videos" / f"{run_dir.name}.mp4"
            if guess.exists():
                mp4s = [guess]
        if not mp4s:
            raise FileNotFoundError(
                f"No .mp4 found in {run_dir} or output/videos/{run_dir.name}.mp4 "
                "— pass the video path explicitly."
            )
        video = mp4s[0]

    thumb = video.with_name(video.stem + "_thumb.jpg")
    return script, video, (thumb if thumb.exists() else None)


def upload_run(
    run_dir: Path,
    *,
    video: Optional[Path] = None,
    script: Optional[Path] = None,
    privacy: Optional[str] = None,
    publish_at: Optional[str] = None,
    schedule_next: bool = False,
    category_id: Optional[str] = None,
    thumb: Optional[Path] = None,
    channel: Optional[str] = None,
    meta: Optional[VideoMeta] = None,
    dry_run: bool = False,
) -> Optional[str]:
    """Upload a finished run to YouTube. Returns the watch URL (or None on dry-run).

    Profile defaults (profiles/default.json -> "youtube") fill any unset args.
    `publish_at` is normalized to RFC3339 UTC using the configured timezone.
    `schedule_next` ignores publish_at and targets the next profile schedule_time.
    `channel` selects which channel's token to use (see youtube.channels); the
    first upload to a new channel opens a browser to pick it on the consent screen.
    """
    from src.agent_config import youtube_settings

    cfg = youtube_settings()
    privacy = privacy or cfg.get("default_privacy", "private")
    category_id = category_id or cfg.get("category_id", "24")
    made_for_kids = bool(cfg.get("made_for_kids", False))
    tz_name = cfg.get("timezone", "UTC")

    if schedule_next and not publish_at:
        publish_at = next_occurrence(cfg.get("schedule_time", "08:00"), tz_name)
    elif publish_at:
        publish_at = to_rfc3339_utc(publish_at, tz_name)

    if meta is None:
        # Build metadata from a script.json (long-form / runs that write one).
        script_path, video_path, found_thumb = _find_artifacts(
            run_dir,
            Path(video) if video else None,
            Path(script) if script else None,
        )
        meta = build_meta_from_script(
            script_path,
            privacy=privacy,
            publish_at=publish_at,
            category_id=category_id,
            made_for_kids=made_for_kids,
        )
    else:
        # Caller supplied metadata (e.g. /broll holds its own youtube dict and
        # never writes a script.json). Resolve only the video + thumbnail, and
        # let profile/CLI privacy + scheduling win over whatever the meta carried.
        if video is not None:
            video_path = Path(video)
        else:
            mp4s = sorted(run_dir.glob("*.mp4"))
            if not mp4s:
                guess = _PROJECT_ROOT / "output" / "videos" / f"{run_dir.name}.mp4"
                if guess.exists():
                    mp4s = [guess]
            if not mp4s:
                raise FileNotFoundError(
                    f"No .mp4 found in {run_dir} — pass the video path explicitly."
                )
            video_path = mp4s[0]
        guess_thumb = video_path.with_name(video_path.stem + "_thumb.jpg")
        found_thumb = guess_thumb if guess_thumb.exists() else None
        meta.privacy = privacy
        meta.publish_at = publish_at
        meta.made_for_kids = made_for_kids
        if category_id:
            meta.category_id = category_id

    thumb_path = Path(thumb) if thumb else found_thumb

    token_file = resolve_token_file(channel)

    logger.info(
        "Prepared upload: video=%s thumb=%s privacy=%s publish_at=%s channel=%s token=%s",
        video_path, thumb_path, meta.privacy, meta.publish_at,
        channel or cfg.get("default_channel", "main"), token_file.name,
    )

    if dry_run:
        print(json.dumps({
            "video": str(video_path),
            "thumbnail": str(thumb_path) if thumb_path else None,
            "channel": channel or cfg.get("default_channel", "main"),
            "token_file": str(token_file),
            "body": meta.to_body(),
        }, indent=2, ensure_ascii=False))
        return None

    youtube = get_authenticated_service(token_file=token_file)
    video_id = upload_video(youtube, video_path, meta)
    if thumb_path:
        try:
            set_thumbnail(youtube, video_id, thumb_path)
        except Exception as exc:  # non-fatal: thumbnail needs a verified channel
            logger.warning("Thumbnail upload failed (channel may be unverified): %s", exc)

    url = f"https://youtu.be/{video_id}"
    if meta.publish_at:
        logger.info(
            "Scheduled (private until %s). NOTE: unaudited API projects stay "
            "private and will NOT auto-publish.", meta.publish_at,
        )
    return url
