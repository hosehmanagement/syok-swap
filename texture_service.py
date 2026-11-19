"""Texture pass as a small job service (no Modal, no GPU): the logic behind the strength slider.

A "job" is a folder under a root directory (on Modal: a Volume):
    meta.json      state of the two stages ("prepare" then "render") and the numbers the UI needs
    src.mp4        the swapped clip the user uploaded
    ref.<ext>      the reference photo
    tracks.json / layer.npy / prepared.json   what texture_transfer.prepare() found (strength independent)
    final.mp4      the last full render

Everything is plain files so it can be unit-tested with a fake face detector and no network.
"""
import json
import re
import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

import swapkit as sk
import texture_transfer as tt

MAX_VIDEO_BYTES = 60 * 1024 * 1024
MAX_PHOTO_BYTES = 8 * 1024 * 1024
PHOTO_EXTS = (".png", ".jpg", ".jpeg", ".webp")
JOB_TTL_SECONDS = 3 * 24 * 3600          # same 3 days the results are kept in the user's Cloudflare storage
_ID = re.compile(r"^[0-9a-f]{16}$")


class JobError(Exception):
    """A problem the caller can be told about (bad input, unknown job, wrong stage)."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _dir(root, job):
    if not isinstance(job, str) or not _ID.match(job):
        raise JobError("Unknown job.", 404)
    d = Path(root) / job
    if not d.is_dir():
        raise JobError("Unknown job (it may have expired after 3 days).", 404)
    return d


def _read_meta(d):
    return json.loads((d / "meta.json").read_text())


def _write_meta(d, meta):
    tmp = d / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(d / "meta.json")


def _set(d, stage, **kw):
    meta = _read_meta(d)
    meta.setdefault(stage, {}).update(kw, at=time.time())      # "at" lets the status call notice a step that died without saying so
    _write_meta(d, meta)
    return meta


def new_job(root, video, photo, photo_ext, consent, video_name="src.mp4", meta=None):
    """Store the two uploads and return the new job id. The slow work happens later in run_prepare().

    The swap service (swap_service.py) reuses this with video_name="raw.mp4": the upload is the ORIGINAL clip there, and
    src.mp4 only appears once the swap has made the swapped clip the texture slider works on. `meta` adds keys to meta.json."""
    if consent != "yes":
        raise JobError("Only use photos and videos of people who agreed to it (or fictional characters). Send consent=yes.")
    ext = (photo_ext or "").lower()
    if ext not in PHOTO_EXTS:
        raise JobError("Reference photo must be png, jpg or webp.")
    if not video or len(video) > MAX_VIDEO_BYTES:
        raise JobError(f"Video is missing or larger than {MAX_VIDEO_BYTES // 2**20} MB.")
    if not photo or len(photo) > MAX_PHOTO_BYTES:
        raise JobError(f"Photo is missing or larger than {MAX_PHOTO_BYTES // 2**20} MB.")
    job = uuid.uuid4().hex[:16]
    d = Path(root) / job
    d.mkdir(parents=True)
    try:
        (d / video_name).write_bytes(video)
        (d / f"ref{ext}").write_bytes(photo)
        try:
            info = sk.probe(d / video_name)
        except Exception:
            raise JobError("That video could not be read. Use an MP4.")
        if info["duration"] > sk.MAX_SECONDS and video_name == "src.mp4":
            raise JobError(f"Video is longer than {int(sk.MAX_SECONDS)} seconds.")
        base = {"job": job, "created": time.time(), "ref": f"ref{ext}",
                "prepare": {"state": "queued"}, "render": {"state": "idle"}}
        base.update(meta or {})
        _write_meta(d, base)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    return job


def run_prepare(root, job, detector=tt.detect_face):
    """Find the face in the photo and track it through the clip (strength independent). Never raises: errors go in meta."""
    d = _dir(root, job)
    _set(d, "prepare", state="preparing")
    try:
        meta = _read_meta(d)
        p = tt.prepare(d / "src.mp4", d / meta["ref"], detector=detector)
        np.save(d / "layer.npy", p["layer"])
        (d / "tracks.json").write_text(json.dumps(p["tracks"]))
        info = {k: p[k] for k in ("info", "layer_size", "frames", "detected")}
        info["layer_size"] = list(info["layer_size"])
        (d / "prepared.json").write_text(json.dumps(info))
        _set(d, "prepare", state="ready", frames=p["frames"], detected=p["detected"],
             width=p["info"]["width"], height=p["info"]["height"], fps=p["info"]["fps"])
    except ValueError as e:                      # our own refusals: safe to show
        _set(d, "prepare", state="error", message=str(e))
    except Exception:                            # anything else: do not leak internals
        _set(d, "prepare", state="error", message="The texture pass could not read this clip.")


def _load_prepared(d):
    if _read_meta(d)["prepare"].get("state") != "ready":
        raise JobError("The clip is not ready yet.", 409)
    info = json.loads((d / "prepared.json").read_text())
    tracks = [(tuple(int(v) for v in b) if b is not None else None, float(w)) for b, w in json.loads((d / "tracks.json").read_text())]
    return {"info": info["info"], "tracks": tracks, "layer": np.load(d / "layer.npy"),
            "layer_size": tuple(info["layer_size"]), "frames": info["frames"], "detected": info["detected"]}


def status(root, job):
    d = _dir(root, job)
    m = _read_meta(d)
    return {"job": job, "prepare": m["prepare"], "render": m["render"],
            "expires_at": m["created"] + JOB_TTL_SECONDS}


def preview_jpeg(root, job, index, strength):
    """One finished frame at this strength, as JPEG bytes. Costs a fraction of a second of CPU."""
    d = _dir(root, job)
    try:
        index = int(index)
        strength = tt.clamp_strength(strength)
    except (TypeError, ValueError):
        raise JobError("frame must be a whole number and strength a number.")
    try:
        frame = tt.preview_frame(d / "src.mp4", _load_prepared(d), index, strength)
    except IndexError as e:
        raise JobError(str(e))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise JobError("Could not encode the preview.", 500)
    return buf.tobytes()


def start_render(root, job, strength):
    """Validate and mark the render as queued. The caller then runs run_render() (on Modal: in its own function)."""
    d = _dir(root, job)
    strength = tt.clamp_strength(strength)
    _load_prepared(d)
    if _read_meta(d)["render"].get("state") in ("queued", "rendering"):
        raise JobError("A render is already running for this clip.", 409)
    (d / "final.mp4").unlink(missing_ok=True)
    _set(d, "render", state="queued", strength=strength)
    return strength


def run_render(root, job):
    """Write the whole clip at the strength chosen in start_render(). Never raises: errors go in meta."""
    d = _dir(root, job)
    try:
        strength = _read_meta(d)["render"]["strength"]
        _set(d, "render", state="rendering")
        stats = tt.render_prepared(d / "src.mp4", _load_prepared(d), d / "final.mp4", strength)
        _set(d, "render", state="done", strength=stats["strength"], frames=stats["frames"], applied=stats["applied"],
             bytes=(d / "final.mp4").stat().st_size)
    except Exception:
        _set(d, "render", state="error", message="The render failed. Try again.")


def result_path(root, job):
    d = _dir(root, job)
    if _read_meta(d)["render"].get("state") != "done" or not (d / "final.mp4").exists():
        raise JobError("The render is not finished.", 409)
    return d / "final.mp4"


def delete_job(root, job):
    shutil.rmtree(_dir(root, job))


def sweep(root, now=None, ttl=JOB_TTL_SECONDS):
    """Delete jobs older than ttl. Returns how many were removed. Folders with no readable meta are removed too."""
    now = time.time() if now is None else now
    removed = 0
    for d in Path(root).iterdir() if Path(root).is_dir() else []:
        if not d.is_dir() or not _ID.match(d.name):
            continue
        try:
            created = _read_meta(d)["created"]
        except Exception:
            created = d.stat().st_mtime
        if now - created > ttl:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed


# Public names for the swap service, which keeps the swap stages in the same job folder and meta.json.
job_dir, read_meta, write_meta, set_stage = _dir, _read_meta, _write_meta, _set
