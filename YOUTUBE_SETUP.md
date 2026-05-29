# YouTube Auto-Upload Setup

Push rendered videos to YouTube with the official Data API v3, including
scheduled publishing. One-time Google setup, then fully scriptable.

## 1. One-time Google Cloud setup (you do this — it's in your account)

1. Go to <https://console.cloud.google.com/> and create a project (or reuse one).
2. **APIs & Services → Library →** search **"YouTube Data API v3" → Enable**.
3. **APIs & Services → OAuth consent screen:**
   - User type: **External**, fill app name + your email, **Save**.
   - **Test users → add your own Google account** (the channel owner). While the
     app is in "Testing" mode only listed test users can authorize — that's fine
     for personal use and avoids the public-verification process.
4. **APIs & Services → Credentials → Create credentials → OAuth client ID:**
   - Application type: **Desktop app** → Create.
   - **Download JSON**, save it as `client_secret.json` in the **project root**
     (next to `requirements.txt`). It is gitignored.

## 2. First authorization

Run any real upload (or just the auth itself). A browser window opens asking you
to grant access; approve it. The refresh token is cached to `token.json` (also
gitignored), so subsequent uploads are headless.

```bash
.venv/bin/python -c "from src.youtube_upload import get_authenticated_service as g; g(); print('auth OK')"
```

## 3. Upload a rendered run

```bash
# Upload from a run directory (auto-finds the mp4 in output/videos/<run>.mp4)
.venv/bin/python scripts/upload_to_youtube.py output/runs/<run_name>

# Or point directly at the mp4
.venv/bin/python scripts/upload_to_youtube.py output/videos/<name>.mp4 \
    --script output/runs/<run_name>/script.json

# Schedule it (forces privacy=private until the publish time)
.venv/bin/python scripts/upload_to_youtube.py output/runs/<run_name> \
    --publish-at "2026-06-01T18:00:00"     # interpreted in profile youtube.timezone

# Inspect the request without uploading (no auth needed)
.venv/bin/python scripts/upload_to_youtube.py output/runs/<run_name> --dry-run
```

Flags: `--privacy private|unlisted|public`, `--category <id>`, `--thumb <path>`,
`--script <path>`, `--dry-run`. Defaults come from `profiles/default.json → "youtube"`.

## Where metadata comes from

- **Long-form** runs: the structured `script.json → youtube` field
  (title, description, chapters, hashtags, tags).
- **Shorts** (no `youtube` field): title falls back to `topic`, description to
  the first narration block. Edit `script.json` before uploading if you want
  tighter copy.

## Two limits worth knowing

- **Quota:** the API gives 10,000 units/day and **each upload costs ~1,600** —
  about **6 uploads/day**. Past that you get a quota error until the next day
  (Pacific midnight). Request more in the Cloud console if needed.
- **Scheduling = private-until-publish (by design):** a `publishAt` upload is held
  **private** until that time, then YouTube flips it to public automatically. This
  is a YouTube API rule, not a bug. Public + scheduled publishing was tested and
  **works on this channel without any API audit**. (If you ever clone this to an
  account where API uploads come back forced-private, that account needs the
  one-time YouTube API compliance audit — request it from the API quota page.)

## Config (`profiles/default.json → "youtube"`)

```json
"youtube": {
  "category_id": "24",        // 24 = Entertainment
  "default_privacy": "private",
  "timezone": "UTC",          // how naive --publish-at times are interpreted
  "made_for_kids": false
}
```
