"""Face-lock: an OPTIONAL second pass after the swap that puts the exact face from the user's photo onto the swapped clip.

Wan2.2-Animate redraws the whole person from one photo, so the face is close but not the same person. When the user ticks
"face-lock", the swapped clip (src.mp4) and the face photo are sent to Magic Hour's "Face swap video" API on the user's OWN
Magic Hour account (BYOK: their key, their credits, commercial use under Magic Hour's paid plans). Nothing is sent to Magic Hour
unless the user asked for it.

No Modal here, and every network call goes through an injectable `http` transport, so the whole flow is unit-tested with a fake
Magic Hour and no network. The default transport is stdlib urllib, so nothing new enters the light image.

The Magic Hour calls mirror syok-ai-router/src/effects.ts exactly (docs: https://docs.magichour.ai):
    POST /v1/files/upload-urls        -> presigned PUT URLs + file_path per file (free)
    PUT  <upload_url>                 -> the file bytes (presigned, no API key)
    POST /v1/face-swap                -> { id, credits_charged }   ONE call, never retried: it spends credits
    GET  /v1/video-projects/{id}      -> { status, downloads:[{url}], error:{message, code}, credits_charged }

Files in the job folder:  src_plain.mp4 (the swap as Wan made it, kept for good)   src.mp4 (replaced by the face-locked clip)
                          face.<ext> (optional close-up; otherwise ref.<ext> is sent)

The key is a function argument only: it is never written to meta.json or any file, never logged, and scrubbed from every
message that is stored or raised.
"""
import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import swapkit as sk
import texture_service as tsvc

MH_BASE = "https://api.magichour.ai"
POLL_SECONDS = 5.0
MAX_WAIT_SECONDS = 25 * 60              # overall cap from the start of the run; facelock_job's Modal limit is 30 min
MAX_RESULT_BYTES = 200 * 2**20
DURATION_TOLERANCE = 0.6                # seconds between the swapped clip and what Magic Hour sends back
API_TIMEOUT = 60
TRANSFER_TIMEOUT = 600
CREATE_NAME = "Syok Swap face-lock"
FREE_PLAN_PX = 576                      # Magic Hour's free plan caps the output's longest edge here

REFUND = " Magic Hour refunds the credits of a failed job."
UNCHANGED = " Your swapped clip is unchanged: try again, or skip face-lock to use it as it is."
NO_FACE = "Magic Hour could not find a face in your photo. Add a sharp front-facing close-up."
BAD_KEY = "Magic Hour didn't accept that key. Copy it again from Magic Hour → Developer."
NO_CREDIT = "Your Magic Hour account is out of free credits. Top it up on magichour.ai, then try again."
BUSY = "Magic Hour is busy right now (rate limit) — wait a moment and try again."
FAILED = "Face-lock failed." + UNCHANGED
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".mp4": "video/mp4"}


class FaceLockError(Exception):
    """A failure whose message is written for the user (no key, no paths)."""


class Unreachable(Exception):
    """The transport could not talk to the server at all (DNS, TLS, connection reset, time-out)."""


class TooLarge(Exception):
    """A download went past its size cap."""


# ------------------------------------------------------------------ transport

class _Redirects(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect of a call that carries the API key; others only to https."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if req.has_header("Authorization") or urllib.parse.urlsplit(newurl).scheme != "https":
            return None                                   # urllib then raises HTTPError with the 3xx code
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_Redirects)


def urllib_http(method, url, headers=None, body=None, dest=None, max_bytes=None, timeout=API_TIMEOUT):
    """The real transport. `body` is bytes or a Path (streamed). With `dest`, a 2xx response body is streamed to that file
    (raising TooLarge past `max_bytes`) and b"" is returned. Returns (status, body bytes); a non-2xx status is returned, not
    raised. Network failures raise Unreachable without any detail (the URL or headers never end up in a message)."""
    headers = dict(headers or {})
    auth = headers.pop("Authorization", None)
    fh = None
    data = body
    if isinstance(body, Path):
        fh = open(body, "rb")
        data = fh
        headers["Content-Length"] = str(body.stat().st_size)
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    if auth:
        req.add_unredirected_header("Authorization", auth)   # never copied onto a redirect
    try:
        try:
            r = _OPENER.open(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            try:
                return e.code, e.read(4096)
            finally:
                e.close()
        with r:
            if dest is None:
                return r.status, r.read(16 * 2**20)
            n = 0
            with open(dest, "wb") as out:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    n += len(chunk)
                    if max_bytes is not None and n > max_bytes:
                        raise TooLarge()
                    out.write(chunk)
            return r.status, b""
    except (TooLarge, FaceLockError):
        raise
    except Exception:
        raise Unreachable() from None
    finally:
        if fh is not None:
            fh.close()


# ------------------------------------------------------------------ Magic Hour helpers

def scrub(text, key):
    """Remove the key (and anything that looks like a Magic Hour key) from text that may be shown or stored."""
    text = str(text or "")
    if key:
        text = text.replace(key, "[key]")
    return re.sub(r"mhk_[A-Za-z0-9_\-]+", "[key]", text)


def mh_error(status, raw):
    """A friendly message for a failed Magic Hour call, like effects.ts mhError (bad key / no credit / busy / other)."""
    raw = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw or "")
    if "no_source_face" in raw:
        return NO_FACE
    if status in (401, 403):
        return BAD_KEY
    if status == 402 or re.search(r"credit|quota|insufficient|balance|payment", raw, re.I):
        return NO_CREDIT
    if status == 429:
        return BUSY
    return f"Magic Hour error ({status}). {raw[:200]}".strip()


def _json(raw):
    try:
        v = json.loads(raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _https(url):
    return isinstance(url, str) and urllib.parse.urlsplit(url).scheme == "https"


def upload_host_ok(url):
    """Presigned upload links must be https on magichour.ai or one of its subdomains (its storage, e.g. videos.magichour.ai),
    the same rule as the router's /v1/effects/magichour/upload."""
    if not _https(url):
        return False
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return host == "magichour.ai" or host.endswith(".magichour.ai")


def _auth(key):
    return {"Authorization": f"Bearer {key}"}


def _call(http, method, url, key, payload=None, unreachable="Couldn't reach Magic Hour. Check your internet and try again."):
    headers = _auth(key)
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    try:
        return http(method, url, headers=headers, body=body, timeout=API_TIMEOUT)
    except Unreachable:
        raise FaceLockError(unreachable)


def upload_urls(http, key, items):
    status, raw = _call(http, "POST", f"{MH_BASE}/v1/files/upload-urls", key, {"items": items})
    if not 200 <= status < 300:
        raise FaceLockError(mh_error(status, raw))
    got = _json(raw).get("items")
    if not isinstance(got, list) or len(got) != len(items):
        raise FaceLockError("Magic Hour returned no upload URLs.")
    for it in got:
        if not isinstance(it, dict) or not isinstance(it.get("file_path"), str) or not it["file_path"]:
            raise FaceLockError("Magic Hour returned no upload URLs.")
        if not upload_host_ok(it.get("upload_url")):
            raise FaceLockError("Magic Hour returned an upload link that is not a Magic Hour address, so nothing was sent.")
    return got


def put_file(http, upload_url, path):
    try:
        status, _ = http("PUT", upload_url, headers={"Content-Type": MIME.get(path.suffix.lower(), "application/octet-stream")},
                         body=path, timeout=TRANSFER_TIMEOUT)
    except Unreachable:
        raise FaceLockError("Couldn't upload the file to Magic Hour. Try again.")
    if not 200 <= status < 300:
        raise FaceLockError(f"Upload to Magic Hour failed ({status}). Try again.")


def create_face_swap(http, key, image_file_path, video_file_path, end_seconds):
    """Start the job. ONE POST, NEVER retried: it spends the user's credits. Returns (id, credits_charged)."""
    body = {"name": CREATE_NAME, "start_seconds": 0, "end_seconds": max(0.1, round(float(end_seconds), 3)),
            "style": {"version": "default"},
            "assets": {"face_swap_mode": "all-faces", "image_file_path": image_file_path, "video_source": "file",
                       "video_file_path": video_file_path}}
    status, raw = _call(http, "POST", f"{MH_BASE}/v1/face-swap", key, body,
                        unreachable="Couldn't reach Magic Hour while starting the job. Check your Magic Hour dashboard "
                                    "before you try again, in case it started.")
    if not 200 <= status < 300:
        raise FaceLockError(mh_error(status, raw))
    data = _json(raw)
    if not data.get("id"):
        raise FaceLockError("Magic Hour didn't return a job id.")
    return str(data["id"]), _int(data.get("credits_charged"))


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def project_error(data, state):
    err = data.get("error") if isinstance(data.get("error"), dict) else {}
    if err.get("code") == "no_source_face":
        return NO_FACE + REFUND
    if state == "canceled":
        return "The face-lock job was canceled on Magic Hour." + UNCHANGED
    msg = str(err.get("message") or "Magic Hour couldn't finish this one").strip()[:200].rstrip(".")
    code = f" ({err['code']})" if err.get("code") else ""
    return f"Magic Hour could not finish the face-lock: {msg}{code}." + REFUND + UNCHANGED


def wait_for_project(http, key, project_id, deadline, clock, sleep, poll_seconds, on_credits=lambda c: None):
    """Poll the project until complete / error / canceled, or the deadline. Polling is a free GET, so a busy or unreachable
    Magic Hour is simply asked again; only a refused key or a missing project stops it. Returns the project's JSON."""
    url = f"{MH_BASE}/v1/video-projects/{urllib.parse.quote(project_id, safe='')}"
    while True:
        try:
            status, raw = http("GET", url, headers=_auth(key), timeout=API_TIMEOUT)
        except Unreachable:
            status, raw = 0, b""
        if 200 <= status < 300:
            data = _json(raw)
            c = _int(data.get("credits_charged"))
            if c is not None:
                on_credits(c)
            state = data.get("status")
            if state == "complete":
                return data
            if state in ("error", "canceled"):
                raise FaceLockError(project_error(data, state))
        elif status in (401, 403, 404):
            raise FaceLockError(mh_error(status, raw) if status != 404 else "Magic Hour lost the face-lock job." + UNCHANGED)
        if clock() + poll_seconds > deadline:
            raise FaceLockError(f"Magic Hour did not finish within {MAX_WAIT_SECONDS // 60} minutes. Check your Magic Hour "
                                "dashboard (a job that fails there is refunded)." + UNCHANGED)
        sleep(poll_seconds)


def download(http, url, dest, max_bytes=None):
    max_bytes = MAX_RESULT_BYTES if max_bytes is None else max_bytes
    if not _https(url):
        raise FaceLockError("Magic Hour sent a download link that is not https, so it was not used." + UNCHANGED)
    try:
        status, _ = http("GET", url, headers={}, dest=dest, max_bytes=max_bytes, timeout=TRANSFER_TIMEOUT)
    except TooLarge:
        raise FaceLockError(f"Magic Hour's result is larger than {max_bytes // 2**20} MB, so it was not used." + UNCHANGED)
    except Unreachable:
        raise FaceLockError("Couldn't download the result from Magic Hour. Try again within a few hours; the result stays on "
                            "your Magic Hour account." + UNCHANGED)
    if not 200 <= status < 300:
        raise FaceLockError(f"Couldn't download the result from Magic Hour ({status})." + UNCHANGED)


# ------------------------------------------------------------------ video plumbing

def fit_to(result, plain_info, out):
    """Re-encode Magic Hour's result onto the swapped clip's timeline: its fps, its width x height (aspect kept, lanczos,
    padded if the aspect differs by rounding), exactly its duration (the last frame is held if the result is a little short).
    Returns the size Magic Hour sent back as (w, h)."""
    info = sk.probe(result)
    W, H, fps, dur = plain_info["width"], plain_info["height"], plain_info["fps"], plain_info["duration"]
    vf = []
    if abs(info["fps"] - fps) > 0.01:
        vf.append(f"fps={fps:.6f}")
    if (info["width"], info["height"]) != (W, H):
        vf.append(f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos")
        vf.append(f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2")
    vf += ["setsar=1", f"tpad=stop_mode=clone:stop_duration={DURATION_TOLERANCE + 0.1}"]
    sk.sh(["ffmpeg", "-y", "-v", "error", "-i", str(result), "-vf", ",".join(vf), "-t", f"{dur:.3f}", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", str(out)])
    return info["width"], info["height"]


def build_locked_clip(result, plain, out, work):
    """Check Magic Hour's result, fit it to the swapped clip and put the swapped clip's soundtrack back on it.
    Returns an informational note (or None). Raises FaceLockError when the result cannot be used."""
    plain_info = sk.probe(plain)
    try:
        info = sk.probe(result)
    except Exception:
        raise FaceLockError("Magic Hour sent back a video that could not be read." + UNCHANGED)
    if abs(info["duration"] - plain_info["duration"]) > DURATION_TOLERANCE:
        raise FaceLockError(f"Magic Hour sent back {info['duration']:.1f} s of video but the swapped clip is "
                            f"{plain_info['duration']:.1f} s, so it was not used." + UNCHANGED)
    fitted = work / "fitted.mp4"
    w, h = fit_to(result, plain_info, fitted)
    note = None
    if w * h < plain_info["width"] * plain_info["height"]:
        note = (f"Magic Hour sent back {w}x{h}; it was scaled back to {plain_info['width']}x{plain_info['height']}. "
                f"Magic Hour's free plan limits face swap videos to {FREE_PLAN_PX} px on the longest edge, so a smaller "
                "result is normal there; a paid plan keeps more detail.")
    sk.mux_original_audio(fitted, plain, out)
    return note


# ------------------------------------------------------------------ the job

def face_photo(d, meta):
    """The photo whose face Magic Hour copies: the optional face close-up, else the photo that drove the swap."""
    return d / (meta.get("face") or meta["ref"])


def _queue_prepare_unless_ready(d):
    """After a failed face-lock the slider must still work on the plain swap: queue it unless it is ready or on its way."""
    st = tsvc.read_meta(d).get("prepare", {}).get("state")
    if st in ("ready", "queued", "preparing"):
        return False
    tsvc.set_stage(d, "prepare", state="queued", message=None)
    return True


def run(root, job, key, http=urllib_http, clock=time.monotonic, sleep=time.sleep, poll_seconds=POLL_SECONDS,
        max_wait=MAX_WAIT_SECONDS, on_progress=lambda: None):
    """Do the whole face-lock for one job. Never raises. Returns True when the caller must spawn the slider preparation."""
    d = tsvc.job_dir(root, job)
    meta = tsvc.read_meta(d)
    if meta.get("facelock", {}).get("state") != "queued":       # skipped meanwhile, or a stray second call: spend nothing
        return False
    deadline = clock() + max_wait
    work = d / "facelock_tmp"
    credits = {"n": None}

    def progress(message, **kw):
        tsvc.set_stage(d, "facelock", state="working", message=message, **kw)
        on_progress()

    try:
        return _run(d, meta, key, http, clock, sleep, poll_seconds, deadline, work, credits, progress, on_progress)
    except FaceLockError as e:
        message = scrub(e, key)
    except Exception:                                           # anything else: never leak internals, paths or the key
        message = FAILED
    finally:
        shutil.rmtree(work, ignore_errors=True)
        (d / "src.mp4.tmp").unlink(missing_ok=True)
    tsvc.set_stage(d, "facelock", state="error", message=message, credits=credits["n"])
    return _queue_prepare_unless_ready(d)


def _run(d, meta, key, http, clock, sleep, poll_seconds, deadline, work, credits, progress, on_progress):
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    if not (d / "src_plain.mp4").exists():                 # the swap as Wan made it; never overwritten
        tmp = d / "src_plain.mp4.tmp"
        shutil.copyfile(d / "src.mp4", tmp)
        tmp.replace(d / "src_plain.mp4")
    plain = d / "src_plain.mp4"
    face = face_photo(d, meta)
    duration = sk.probe(plain)["duration"]
    progress("Uploading your clip and photo to Magic Hour", credits=None, note=None, project=None)
    urls = upload_urls(http, key, [{"type": "video", "extension": "mp4"},
                                   {"type": "image", "extension": face.suffix.lower().lstrip(".")}])
    put_file(http, urls[0]["upload_url"], plain)
    put_file(http, urls[1]["upload_url"], face)
    project, c = create_face_swap(http, key, urls[1]["file_path"], urls[0]["file_path"], duration)
    credits["n"] = c
    progress("Magic Hour is swapping the face", credits=c, project=project)

    def seen(c):
        if c != credits["n"]:
            credits["n"] = c
            tsvc.set_stage(d, "facelock", credits=c)
            on_progress()

    data = wait_for_project(http, key, project, deadline, clock, sleep, poll_seconds, seen)
    downloads = data.get("downloads") if isinstance(data.get("downloads"), list) else []
    url = downloads[0].get("url") if downloads and isinstance(downloads[0], dict) else None
    if not url:
        raise FaceLockError("Magic Hour finished but sent no video." + UNCHANGED)
    progress("Downloading the result")
    result = work / "magichour.mp4"
    download(http, url, result)
    progress("Checking the result")
    locked = work / "locked.mp4"
    note = build_locked_clip(result, plain, locked, work)
    tmp = d / "src.mp4.tmp"
    shutil.copyfile(locked, tmp)
    tmp.replace(d / "src.mp4")
    for name in ("layer.npy", "tracks.json", "prepared.json", "final.mp4"):
        (d / name).unlink(missing_ok=True)
    tsvc.set_stage(d, "render", state="idle")
    tsvc.set_stage(d, "facelock", state="done", message=None, credits=credits["n"], note=note, finished=time.time())
    tsvc.set_stage(d, "prepare", state="queued", message=None)
    return True
