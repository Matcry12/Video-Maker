"""
Web UI for the Video Maker pipeline.
Run: python -m src.web
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from flask import Flask, render_template, request, jsonify, send_from_directory

from .manager import VideoManager, PROJECT_ROOT, OUTPUT_DIR, Profile

logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)

# Global state
jobs = {}  # job_id -> {status, progress, output, error}

VOICES = [
    {"id": "NamMinh", "label": "NamMinh (nam)"},
    {"id": "HoaiMy", "label": "HoaiMy (nữ)"},
]



def create_manager():
    logger.info("Initializing VideoManager...")
    mgr = VideoManager()
    logger.info("VideoManager ready.")
    return mgr


def api_error(
    message: str,
    *,
    code: str = "UNKNOWN_ERROR",
    status: int = 400,
    details: str | None = None,
    hint: str | None = None,
):
    payload = {
        "error": {
            "code": code,
            "message": message,
        }
    }
    if details:
        payload["error"]["details"] = details
    if hint:
        payload["error"]["hint"] = hint
    return jsonify(payload), status


def output_to_payload(path: Path) -> dict:
    stat = path.stat()
    created_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    encoded_name = quote(path.name)
    return {
        "filename": path.name,
        "size_bytes": stat.st_size,
        "created_at": created_dt.isoformat(),
        "created_ts": stat.st_mtime,
        "url": f"/output/{encoded_name}",
        "download_url": f"/output/{encoded_name}?download=1",
    }


@app.route("/")
def index():
    # List existing scripts
    scripts_dir = PROJECT_ROOT / "json_scripts"
    scripts = sorted(scripts_dir.glob("*.json")) if scripts_dir.exists() else []
    script_names = [s.stem for s in scripts]

    # List existing outputs
    outputs = sorted(OUTPUT_DIR.glob("*.mp4")) if OUTPUT_DIR.exists() else []
    output_names = [o.name for o in outputs]

    subtitle_presets = ["minimal", "energetic", "cinematic"]
    default_subtitle_preset = "minimal"
    profile_path = PROJECT_ROOT / "profiles" / "default.json"
    if profile_path.exists():
        try:
            profile_raw = json.loads(profile_path.read_text())
            profile = Profile(**profile_raw)
            subtitle_presets = sorted(profile.subtitle.presets.keys())
            default_subtitle_preset = profile.subtitle.default_preset
        except Exception:
            logger.exception("Failed to load subtitle presets from profile")

    return render_template(
        "index.html",
        voices=VOICES,
        scripts=script_names,
        outputs=output_names,
        subtitle_presets=subtitle_presets,
        default_subtitle_preset=default_subtitle_preset,
    )


@app.route("/api/scripts/<name>")
def get_script(name):
    path = PROJECT_ROOT / "json_scripts" / f"{name}.json"
    if not path.exists():
        return api_error(
            "Script not found.",
            code="SCRIPT_NOT_FOUND",
            status=404,
            hint="Select an existing script from the list or save a new script name first.",
        )
    try:
        return jsonify(json.loads(path.read_text()))
    except Exception as exc:
        logger.exception("Failed to parse script: %s", path)
        return api_error(
            "Script file is invalid JSON.",
            code="INVALID_SCRIPT_JSON",
            status=400,
            details=str(exc),
            hint="Fix JSON content in the script file, then reload.",
        )


@app.route("/api/scripts")
def list_scripts():
    scripts_dir = PROJECT_ROOT / "json_scripts"
    scripts = sorted(scripts_dir.glob("*.json")) if scripts_dir.exists() else []
    return jsonify([s.stem for s in scripts])


@app.route("/api/scripts/<name>", methods=["POST"])
def save_script(name):
    data = request.get_json()
    if data is None:
        return api_error(
            "Request body must be JSON.",
            code="INVALID_REQUEST_JSON",
            status=400,
            hint="Send a valid JSON script payload.",
        )
    path = PROJECT_ROOT / "json_scripts" / f"{name}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return jsonify({"ok": True})
    except Exception as exc:
        logger.exception("Failed to save script: %s", path)
        return api_error(
            "Failed to save script.",
            code="SAVE_SCRIPT_FAILED",
            status=500,
            details=str(exc),
            hint="Check file permissions and free disk space, then try again.",
        )


@app.route("/api/scripts/<name>", methods=["DELETE"])
def delete_script(name):
    path = PROJECT_ROOT / "json_scripts" / f"{name}.json"
    try:
        if path.exists():
            path.unlink()
        return jsonify({"ok": True})
    except Exception as exc:
        logger.exception("Failed to delete script: %s", path)
        return api_error(
            "Failed to delete script.",
            code="DELETE_SCRIPT_FAILED",
            status=500,
            details=str(exc),
            hint="Close any process using this file and retry.",
        )


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    if not isinstance(data, dict):
        return api_error(
            "Request body must be JSON.",
            code="INVALID_REQUEST_JSON",
            status=400,
            hint="Send script_name and script as JSON.",
        )
    script_name = data.get("script_name", "untitled")
    output_name = data.get("output_name") or script_name
    script_data = data.get("script")
    request_meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    save_script = bool(data.get("save_script", True))

    if not script_data or not script_data.get("blocks"):
        return api_error(
            "Script must include at least one block.",
            code="INVALID_SCRIPT",
            status=400,
            hint="Add a block with non-empty text, then generate again.",
        )

    # Persist script payload (either to saved scripts or temporary job script)
    if save_script:
        script_path = PROJECT_ROOT / "json_scripts" / f"{script_name}.json"
        cleanup_script = False
    else:
        script_path = PROJECT_ROOT / "tmp" / "job_scripts" / f"{output_name}_{int(time.time() * 1000)}.json"
        cleanup_script = True
    try:
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(json.dumps(script_data, ensure_ascii=False, indent=2))
    except Exception as exc:
        logger.exception("Failed to persist script before generation: %s", script_path)
        return api_error(
            "Unable to save script before generation.",
            code="SAVE_SCRIPT_FAILED",
            status=500,
            details=str(exc),
            hint="Check write permissions and disk space, then retry.",
        )

    # Create job
    job_id = f"{output_name}_{int(time.time())}"
    jobs[job_id] = {
        "status": "starting",
        "stage": "loading",
        "progress": "Queued...",
        "current_block": None,
        "total_blocks": len(script_data.get("blocks", [])),
        "meta": request_meta,
        "output": None,
        "error": None,
    }

    # Run in background thread
    thread = threading.Thread(target=_run_job, args=(job_id, script_path, output_name, cleanup_script))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})


def _run_job(job_id, script_path, output_name, cleanup_script=False):
    mgr = None

    def update_job(**fields):
        job = jobs.get(job_id)
        if not job:
            return
        job.update(fields)

    try:
        update_job(
            status="loading",
            stage="loading",
            progress="Loading models...",
            current_block=0,
        )
        mgr = create_manager()

        def on_progress(progress_event: dict):
            update_job(
                status="processing",
                stage=progress_event.get("stage", "rendering"),
                progress=progress_event.get("message", "Processing..."),
                current_block=progress_event.get("current_block"),
                total_blocks=progress_event.get("total_blocks"),
            )

        output = mgr.process_script(script_path, output_name, progress_callback=on_progress)

        total_blocks = jobs.get(job_id, {}).get("total_blocks")
        update_job(
            status="done",
            stage="done",
            output=output.name,
            progress="Complete!",
            current_block=total_blocks,
        )
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        failed_stage = jobs.get(job_id, {}).get("stage") or "rendering"
        update_job(
            status="error",
            stage=failed_stage,
            progress="Failed to generate video.",
            error={
                "code": "GENERATION_FAILED",
                "message": "Video generation failed.",
                "details": str(e),
                "hint": "Check background/image paths, voice settings, and block text, then retry.",
            },
        )
    finally:
        if mgr:
            mgr.close()
        if cleanup_script and script_path.exists():
            script_path.unlink(missing_ok=True)


@app.route("/api/jobs/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return api_error(
            "Job not found.",
            code="JOB_NOT_FOUND",
            status=404,
            hint="Start a new generation and poll the returned job ID.",
        )
    return jsonify(job)


@app.route("/output/<filename>")
def serve_output(filename):
    download_flag = request.args.get("download", "").strip().lower()
    as_attachment = download_flag in {"1", "true", "yes"}
    return send_from_directory(str(OUTPUT_DIR), filename, as_attachment=as_attachment)


@app.route("/api/outputs")
def list_outputs():
    outputs = sorted(OUTPUT_DIR.glob("*.mp4")) if OUTPUT_DIR.exists() else []
    payload = [output_to_payload(p) for p in outputs]
    payload.sort(key=lambda x: x.get("created_ts", 0), reverse=True)
    return jsonify(payload)


@app.route("/api/outputs/<filename>", methods=["DELETE"])
def delete_output(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return api_error(
            "Output file not found.",
            code="OUTPUT_NOT_FOUND",
            status=404,
            hint="Refresh the output list and retry.",
        )
    try:
        path.unlink()
        return jsonify({"ok": True})
    except Exception as exc:
        logger.exception("Failed to delete output: %s", path)
        return api_error(
            "Failed to delete output file.",
            code="DELETE_OUTPUT_FAILED",
            status=500,
            details=str(exc),
            hint="Close any media player using the file, then retry.",
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("\n  Video Maker Web UI")
    print("  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
