"""Syok Swap as ONE job, from the first upload to the saved clip. No Modal, no GPU here, so every rule is unit-testable.

A job is a folder under a root directory (on Modal: the `syok-swap-cache` Volume, under jobs/). One meta.json carries four stages:

    mask     "queued" -> "preparing" -> "ready" | "error"   small GPU, cents: finds the person, red-overlay preview (preview.mp4)
    swap     "idle" -> "queued" -> "generating" -> "done" | "error"   H100, the paid step: the new person is generated
    prepare  "idle" -> "queued" -> "preparing" -> "ready" | "error"   CPU, starts by itself when the swap is done:
                                                           tracks the face in the SWAPPED clip for the texture slider
    render   "idle" -> "queued" -> "rendering" -> "done" | "error"   CPU: the full clip at the slider's strength (final.mp4)

The texture slider is not a separate tool: the swapped clip becomes src.mp4 in this same folder and the same reference photo
(ref.<ext>) that drove the swap gives the skin grain, so the user never uploads anything twice.

Files:  raw.mp4 (original clip)  ref.<ext> (new person)  preview.mp4 / mask.mp4 / section.mp4 / proc/ (mask step)
        generated.mp4 (raw generation)  src.mp4 (swapped clip)  layer.npy, tracks.json, prepared.json (slider)  final.mp4 (result)
"""
import os
import shutil
import time
from pathlib import Path

import swapkit as sk
import texture_transfer as tt
import texture_service as tsvc
from texture_service import JobError

# What the paid H100 step costs on the user's own Modal account. These are ESTIMATES from our own test renders (480p: about
# $0.07 per source second; 720p: about $0.22, read from the Modal dashboard) plus a fixed model-load share. The card shows them
# as "about" and tells the user to read the real figure on their Modal dashboard.
RATE_PER_SECOND = {"480p": 0.08, "720p": 0.23}
SWAP_FIXED_USD = 0.10
MASK_USD = {"480p": 0.10, "720p": 0.30}
MIN_PER_SECOND = {"480p": 1.5, "720p": 4.2}      # minutes of H100 time per second of source (measured: 5 s took 5-8 min / ~20 min)
MIN_FIXED = 4


def estimate(resolution, seconds):
    """Rough cost and time of the whole swap for a section of `seconds`. Always says it is an estimate."""
    swap = round(SWAP_FIXED_USD + RATE_PER_SECOND[resolution] * float(seconds), 2)
    return {"approx": True, "mask_usd": MASK_USD[resolution], "swap_usd": swap,
            "total_usd": round(swap + MASK_USD[resolution], 2),
            "swap_minutes": int(round(MIN_FIXED + MIN_PER_SECOND[resolution] * float(seconds)))}


def _num(v, name):
    try:
        n = float(v or 0)
    except (TypeError, ValueError):
        raise JobError(f"{name} must be a number of seconds.")
    if n != n or n < 0 or n > 86400:
        raise JobError(f"{name} must be a number of seconds, 0 or more.")
    return n


def new_swap_job(root, video, photo, photo_ext, consent, resolution="480p", margin="normal", start=0.0, length=0.0,
                 skin=None, skin_ext="", detector=tt.detect_face):
    """Store the original clip and the new person's photo. Returns the job id; the mask step then runs on a small GPU."""
    resolution = resolution or "480p"
    margin = margin or "normal"
    if resolution not in sk.RESOLUTIONS:
        raise JobError("Resolution must be 480p or 720p.")
    if margin not in sk.MARGINS:
        raise JobError("Margin must be tight, normal or loose.")
    start, length = _num(start, "Start"), _num(length, "Length")
    job = tsvc.new_job(root, video, photo, photo_ext, consent, video_name="raw.mp4", skin=skin, skin_ext=skin_ext)
    d = tsvc.job_dir(root, job)
    try:
        try:
            s, l = sk.plan_section(sk.probe(d / "raw.mp4")["duration"], start, length)
        except ValueError as e:
            raise JobError(str(e))
        meta = tsvc.read_meta(d)
        check_slider_photo(d, meta, detector)
        meta.update(kind="swap", request={"resolution": resolution, "margin": margin, "start": s, "length": l},
                    mask={"state": "queued"}, swap={"state": "idle"}, prepare={"state": "idle"},
                    estimate=estimate(resolution, l))
        tsvc.write_meta(d, meta)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    return job


def check_slider_photo(d, meta, detector=tt.detect_face):
    """BEFORE anything is paid for: can the photo (or the skin close-up) feed the realism slider? Refuse if not."""
    try:
        width = tt.face_width_in(d / "raw.mp4", detector)
    except Exception:
        width = None
    sp = tsvc.skin_path(d, meta)
    try:
        tt.check_reference(sp or (d / meta["ref"]), width, detector=detector, need_face=not sp)
    except ValueError as e:
        tip = ("" if sp else " You can also add a skin close-up of the same person (used only for skin grain).")
        raise JobError(f"The realism slider cannot use this photo: {e}{tip} Nothing was started and nothing was charged.", 422)


def retexture(root, job, skin=None, skin_ext=""):
    """Run the slider preparation again on a finished swap (CPU only), optionally with a new skin close-up."""
    d = tsvc.job_dir(root, job)
    meta = tsvc.read_meta(d)
    if meta.get("swap", {}).get("state") != "done" or not (d / "src.mp4").exists():
        raise JobError("The swap is not finished.", 409)
    if meta.get("prepare", {}).get("state") in ("queued", "preparing"):
        raise JobError("The slider is already being prepared.", 409)
    if meta.get("render", {}).get("state") in ("queued", "rendering"):
        raise JobError("A render is running. Wait for it to finish.", 409)
    if skin:
        tsvc.set_skin(root, job, skin, skin_ext)
    for name in ("layer.npy", "tracks.json", "prepared.json", "final.mp4"):
        (d / name).unlink(missing_ok=True)
    tsvc.set_stage(d, "render", state="idle")
    tsvc.set_stage(d, "prepare", state="queued", message=None)


def request_of(root, job):
    """What the mask step needs: file names and the options chosen at upload."""
    d = tsvc.job_dir(root, job)
    meta = tsvc.read_meta(d)
    return {"dir": d, "ref": meta["ref"], **meta["request"]}


def mask_ready(root, job, section, stats, head=None):
    tsvc.set_stage(tsvc.job_dir(root, job), "mask", state="ready", section=list(section), stats=stats, head=head,
                   warnings=list(stats.get("warnings") or []))


def fail(root, job, stage, message):
    """Record a problem the user may read. `message` must already be safe to show (no paths, no internals)."""
    tsvc.set_stage(tsvc.job_dir(root, job), stage, state="error", message=message)


def mask_path(root, job):
    d = tsvc.job_dir(root, job)
    if tsvc.read_meta(d).get("mask", {}).get("state") != "ready" or not (d / "preview.mp4").exists():
        raise JobError("The mask preview is not ready yet.", 409)
    return d / "preview.mp4"


def start_generate(root, job, relight=True):
    """Check the mask was made and looks usable, and that this is not a second paid run, then queue the swap.
    Refuses an empty mask and refuses to start again once a swap exists or is running (a double click must not bill twice)."""
    d = tsvc.job_dir(root, job)
    meta = tsvc.read_meta(d)
    mask, swap = meta.get("mask", {}), meta.get("swap", {})
    if mask.get("state") != "ready":
        raise JobError("Check the mask first: the swap needs it.", 409)
    bad = [w for w in mask.get("warnings", []) if "empty" in w]
    if bad:
        raise JobError("Refusing to swap: " + "; ".join(bad))
    if swap.get("state") in ("queued", "generating"):
        raise JobError("The swap is already running for this clip.", 409)
    if swap.get("state") == "done":
        raise JobError("This clip is already swapped. Use the slider, or start over to swap again.", 409)
    tsvc.set_stage(d, "swap", state="queued", relight=bool(relight), started=time.time())
    return bool(relight)


def finish_swap(root, job, final_path, had_audio):
    """The GPU step made `final_path`: it becomes src.mp4, the clip the slider works on, and the slider's face tracking is queued."""
    d = tsvc.job_dir(root, job)
    tmp = d / "src.mp4.tmp"
    shutil.copyfile(final_path, tmp)
    tmp.replace(d / "src.mp4")
    tsvc.set_stage(d, "swap", state="done", had_audio=bool(had_audio), bytes=(d / "src.mp4").stat().st_size, finished=time.time())
    tsvc.set_stage(d, "prepare", state="queued")


# A step that is "running" for longer than its Modal time limit (plus a margin) died without writing an answer (out of credits,
# a restart, a time-out). Say so instead of leaving the card spinning for ever.
ACTIVE = {"mask": ("queued", "preparing"), "swap": ("queued", "generating"), "prepare": ("queued", "preparing"),
          "render": ("queued", "rendering")}
STUCK_AFTER_SECONDS = {"mask": 3600 + 300, "swap": 10800 + 300, "prepare": 1800 + 300, "render": 3600 + 300}
STUCK_MESSAGE = {
    "mask": "The mask step stopped without finishing (Modal's time limit, a restart, or no credit left). Start over.",
    "swap": "The swap stopped without finishing (Modal's time limit, a restart, or no credit left). Check your Modal dashboard, then try again.",
    "prepare": "Preparing the texture slider stopped without finishing. Try again.",
    "render": "The render stopped without finishing. Try again."}


def _expire_stuck(d, meta, now):
    for stage, active in ACTIVE.items():
        st = meta.get(stage) or {}
        if st.get("state") in active and now - (st.get("at") or meta["created"]) > STUCK_AFTER_SECONDS[stage]:
            meta = tsvc.set_stage(d, stage, state="error", message=STUCK_MESSAGE[stage])
    return meta


def weights_state(weights_dir, flag_dir, min_bytes=40 * 10**9, now=None, stale=3 * 3600):
    """"ready" | "downloading" | "missing" for the ~50 GB of model files in the user's Modal volume.
    The download writes .weights-downloading when it starts and .weights-ok when it ends. A volume filled earlier with
    `modal run app.py::fetch` has neither, so a model folder of the full size also counts as ready."""
    flag_dir, w = Path(flag_dir), Path(weights_dir)
    now = time.time() if now is None else now
    dl = flag_dir / ".weights-downloading"
    if dl.exists():
        try:
            started = float(dl.read_text())
        except (OSError, ValueError):
            started = dl.stat().st_mtime
        if now - started < stale:
            return "downloading"
    if (flag_dir / ".weights-ok").exists():
        return "ready"
    if w.is_dir() and sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(w) for f in fs) >= min_bytes:
        return "ready"
    return "missing"


def status(root, job, now=None):
    d = tsvc.job_dir(root, job)
    m = _expire_stuck(d, tsvc.read_meta(d), time.time() if now is None else now)
    return {"job": job, "mask": m.get("mask", {}), "swap": m.get("swap", {}), "prepare": m.get("prepare", {}),
            "render": m.get("render", {}), "request": m.get("request", {}), "estimate": m.get("estimate"),
            "expires_at": m["created"] + tsvc.JOB_TTL_SECONDS}


def swapped_path(root, job):
    """The swapped clip as the GPU step made it (no texture): the fallback when the slider cannot be used."""
    d = tsvc.job_dir(root, job)
    if tsvc.read_meta(d).get("swap", {}).get("state") != "done" or not (d / "src.mp4").exists():
        raise JobError("The swap is not finished.", 409)
    return d / "src.mp4"
