"""Syok Swap as ONE job, from the first upload to the saved clip. No Modal, no GPU here, so every rule is unit-testable.

A job is a folder under a root directory (on Modal: the `syok-swap-cache` Volume, under jobs/). One meta.json carries five stages:

    mask     "queued" -> "preparing" -> "ready" | "error"   small GPU, cents: finds the person, red-overlay preview (preview.mp4)
    swap     "idle" -> "queued" -> "generating" -> "done" | "error"   H100, the paid step: the new person is generated
    facelock "idle" -> "waiting" -> "queued" -> "working" -> "done" | "error" | "skipped"   OPTIONAL, CPU + the user's own
                                                           Magic Hour account: puts the exact face from the photo onto the
                                                           swapped clip (facelock.py). Only when the user ticked the box.
    prepare  "idle" -> "queued" -> "preparing" -> "ready" | "error"   CPU, starts by itself when the swap is done (or when the
                                                           face-lock is done, failed or skipped):
                                                           tracks the face in the SWAPPED clip for the texture slider
    render   "idle" -> "queued" -> "rendering" -> "done" | "error"   CPU: the full clip at the slider's strength (final.mp4)

The texture slider is not a separate tool: the swapped clip becomes src.mp4 in this same folder and the same reference photo
(ref.<ext>) that drove the swap gives the skin grain, so the user never uploads anything twice.

Face-lock: with request.facelock the swap ends in facelock "waiting" and the slider is NOT prepared yet (the clip will change).
POST /facelock/{job} (with the user's Magic Hour key in a header, never stored) runs it; POST /facelock/{job}/skip uses the swap
as it is. Either way the slider is then prepared on whatever src.mp4 is.

Files:  raw.mp4 (original clip)  ref.<ext> (new person)  preview.mp4 / mask.mp4 / section.mp4 / proc/ (mask step)
        generated.mp4 (raw generation)  src.mp4 (swapped clip)  layer.npy, tracks.json, prepared.json (slider)  final.mp4 (result)
        face.<ext> (the photo face-lock sends to Magic Hour, chosen by picker.py)  src_plain.mp4 (the swap before face-lock, kept)
        ref_collage.png (EXPERIMENTAL, only with request.collage: the one reference image Wan generates from)

Collage (EXPERIMENTAL, off by default, not measured yet): with request.collage, ref_collage.png = the main photo + a
head-and-shoulders crop of one added face photo side by side (collage.py), made at upload. request_of() hands it to the mask
step as the reference (--refer_path, which becomes proc/src_ref.png for generate.py). Everything else (the slider check, the
slider's skin grain, face-lock and its pick) keeps using the original main photo ref.<ext>. The close-up is the picker's
winner, or the best ADDED photo when the main photo won, so the main photo never appears twice. Needs at least one face photo.

Best-photo pick: POST /swap and POST /facelock/{job} take the multipart field `face` 1 to 5 times (png/jpg/webp, each at most
MAX_PHOTO_BYTES, all together at most 12 MB). The main photo ref.<ext> is always one more candidate. picker.py scores them
(face size, then how front-facing, then sharpness, then exposure; one person only) and the winner is stored as face.<ext>
(a copy of ref.<ext> when the main photo wins) and meta["face"]; the other candidates are deleted, never kept. The result is
meta["facelock"]["pick"] = {"chosen": N (1-based), "of": M, "reason": str, "scores": [{"n", "face_px", "front", "sharp",
"ok", "note", "score"}]}. Numbering: the added photos in upload order, then the main photo last (N = M). When no photo has a
face the main photo is used (no face.<ext>) and the warning says so; nothing is ever blocked by the pick.
"""
import os
import shutil
import time
from pathlib import Path

import cv2

import picker
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
                 skin=None, skin_ext="", detector=tt.detect_face, face=None, face_ext="", facelock=False, faces=None,
                 face_detector=None, collage=False, mask_shape=""):
    """Store the original clip and the new person's photo. Returns the job id; the mask step then runs on a small GPU.
    `faces` is a list of (bytes, ext) candidate photos for face-lock (`face`/`face_ext` is the one-photo form); the best of
    them and the main photo is kept (picker.py). `facelock` asks for the face-lock pass after the swap.
    `collage` (EXPERIMENTAL, needs at least one face photo): Wan gets ref_collage.png (main photo + close-up, collage.py) as
    its reference instead of the main photo; the slider, the face check and face-lock still use the main photo.
    `mask_shape`: "silhouette" (default: the red mask follows the person) or "box" (the old whole-bounding-box mask);
    swapkit.MASK_SHAPES. Stored in the request only when it is "box", so a default job's request is exactly as before."""
    resolution = resolution or "480p"
    margin = margin or "normal"
    if resolution not in sk.RESOLUTIONS:
        raise JobError("Resolution must be 480p or 720p.")
    if margin not in sk.MARGINS:
        raise JobError("Margin must be tight, normal or loose.")
    mask_shape = (mask_shape or sk.DEFAULT_MASK_SHAPE).strip().lower()
    if mask_shape not in sk.MASK_SHAPES:
        raise JobError("Mask shape must be silhouette or box.")
    start, length = _num(start, "Start"), _num(length, "Length")
    uploads = face_uploads(faces, face, face_ext)
    if collage and not uploads:
        raise JobError("Collage needs a face close-up.", 422)
    job = tsvc.new_job(root, video, photo, photo_ext, consent, video_name="raw.mp4", skin=skin, skin_ext=skin_ext)
    d = tsvc.job_dir(root, job)
    try:
        try:
            s, l = sk.plan_section(sk.probe(d / "raw.mp4")["duration"], start, length)
        except ValueError as e:
            raise JobError(str(e))
        meta = tsvc.read_meta(d)
        check_slider_photo(d, meta, detector)
        request = {"resolution": resolution, "margin": margin, "start": s, "length": l}
        if facelock:                                  # only present when asked for, so a job without it is exactly as before
            request["facelock"] = True
        if collage:                                   # same rule: only present when on
            request["collage"] = True
        if mask_shape != sk.DEFAULT_MASK_SHAPE:       # same rule: only present when not the default
            request["mask_shape"] = mask_shape
        meta.update(kind="swap", request=request,
                    mask={"state": "queued"}, swap={"state": "idle"}, prepare={"state": "idle"},
                    facelock={"state": "idle"}, estimate=estimate(resolution, l))
        if uploads:
            pick_face(d, meta, uploads, _picker_detector(detector, face_detector))
        elif facelock:
            _set_facelock_warning(d, meta, detector)
        if collage:
            make_collage_ref(d, meta, uploads, _picker_detector(detector, face_detector))
        tsvc.write_meta(d, meta)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    return job


COLLAGE_NAME = "ref_collage.png"


def collage_face_index(pick, n_added):
    """1-based number of the added photo the collage uses: the picker's winner when it is an added photo; when the main photo
    won (or no photo had a face), the best ADDED photo by the picker's own order (face found, not a profile, score, face size,
    earlier photo), so the main photo never appears twice in the collage."""
    if pick["chosen"] <= n_added:
        return pick["chosen"]
    added = pick["scores"][:n_added]
    return 1 + max(range(n_added), key=lambda i: (added[i]["face_px"] > 0, added[i]["front"] >= picker.PROFILE_FRONT,
                                                  added[i]["score"], added[i]["face_px"], -i))


def make_collage_ref(d, meta, uploads, face_detector):
    """EXPERIMENTAL: write ref_collage.png (main photo + the chosen close-up) and record which close-up in meta["collage"].
    Runs after pick_face, so meta["facelock"]["pick"] is there. The main photo ref.<ext> is not touched."""
    import collage
    n = collage_face_index(meta["facelock"]["pick"], len(uploads))
    try:
        png = collage.make_collage((d / meta["ref"]).read_bytes(), uploads[n - 1][0], detector=face_detector)
    except ValueError as e:
        raise JobError(f"The collage could not be made: {e}") from None
    tmp = d / (COLLAGE_NAME + ".tmp")
    tmp.write_bytes(png)
    tmp.replace(d / COLLAGE_NAME)
    meta["collage"] = {"file": COLLAGE_NAME, "face": n, "of": len(uploads)}


def generation_ref(d, meta):
    """The file name Wan's preprocessing reads as the reference (--refer_path -> proc/src_ref.png -> generate.py):
    ref_collage.png when the job asked for the collage, else the main photo."""
    if meta.get("request", {}).get("collage"):
        if not (d / COLLAGE_NAME).exists():
            raise JobError("The collage reference is missing. Start over.", 409)
        return COLLAGE_NAME
    return meta["ref"]


FACELOCK_MIN_FACE_PX = 256          # below this Magic Hour has little to copy (its docs ask for >= 512 x 512 px for the face)


def _face_ext(face, face_ext):
    """The optional face close-up for face-lock: png/jpg/webp, at most MAX_PHOTO_BYTES."""
    if not face:
        return ""
    ext = (face_ext or "").lower()
    if ext not in tsvc.PHOTO_EXTS:
        raise JobError("Face photo must be png, jpg or webp.")
    if len(face) > tsvc.MAX_PHOTO_BYTES:
        raise JobError(f"Face photo is larger than {tsvc.MAX_PHOTO_BYTES // 2**20} MB.")
    return ext


def face_uploads(faces=None, face=None, face_ext=""):
    """Check the candidate face photos BEFORE anything is stored: 1 to 5 non-empty files (empty parts, as a browser form sends
    for an unused file field, are ignored), each png/jpg/webp and at most MAX_PHOTO_BYTES, all together at most 12 MB.
    Returns [(bytes, ext)]."""
    items = list(faces or [])
    if face:
        items.insert(0, (face, face_ext))
    items = [(b, e) for b, e in items if b]
    if len(items) > picker.MAX_CANDIDATES:
        raise JobError(f"Add at most {picker.MAX_CANDIDATES} face photos.")
    out = [(b, _face_ext(b, e)) for b, e in items]
    if sum(len(b) for b, _ in out) > picker.MAX_TOTAL_BYTES:
        raise JobError(f"The face photos together are larger than {picker.MAX_TOTAL_BYTES // 2**20} MB.")
    return out


def _picker_detector(detector, face_detector):
    """The picker's detector: given, else YuNet in production, else the injected (box, weight) detector (tests' fakes)."""
    if face_detector is not None:
        return face_detector
    return picker.yunet_faces if detector is tt.detect_face else picker.from_box_detector(detector)


def pick_face(d, meta, uploads, face_detector):
    """Store the candidates, keep the best as face.<ext> (or nothing when the main photo has to be used), delete the rest,
    and record the pick and its warning in meta (not written here). Replaces any face photo from before."""
    cands = []
    try:
        for i, (data, ext) in enumerate(uploads):
            p = d / f"face_cand{i + 1}{ext}"
            p.write_bytes(data)
            cands.append(p)
        ref = d / meta["ref"]
        paths = cands + [ref]
        m = len(paths)
        labels = [f"photo {i + 1}" for i in range(m - 1)] + [f"photo {m} (your main photo)"]
        res = picker.pick([picker.read_image(p) for p in paths], face_detector, labels)
        if meta.get("face"):
            (d / meta["face"]).unlink(missing_ok=True)
            meta.pop("face", None)
        chosen = res["chosen"]
        if chosen is not None:
            src = paths[chosen - 1]
            dest = d / f"face{src.suffix.lower()}"
            if src == ref:
                tmp = d / "face.tmp"
                shutil.copyfile(ref, tmp)
                tmp.replace(dest)
            else:
                src.replace(dest)
            meta["face"] = dest.name
    finally:
        for p in cands:
            p.unlink(missing_ok=True)
    meta.setdefault("facelock", {})["pick"] = {"chosen": chosen if chosen is not None else m, "of": m,
                                               "reason": res["reason"], "scores": res["scores"]}
    _store_warning(meta, pick_warning(res))


def pick_warning(res):
    """The face-lock warning after a pick, from the CHOSEN photo's face (or None)."""
    m = res["of"]
    if res["chosen"] is None:
        return (f"We could not find a clear face in any of your {m} photos, so your main photo is used and Magic Hour may not "
                "find a face either. Add a sharp head-and-shoulders close-up for a closer likeness.")
    width = res["scores"][res["chosen"] - 1]["face_px"]
    if res["all_small"]:
        return (f"Every photo you added shows the face under {FACELOCK_MIN_FACE_PX} px wide: the best one is only about "
                f"{width} px wide, so Magic Hour has little to copy. Add a sharp head-and-shoulders close-up for a closer "
                "likeness. Magic Hour asks for a face area of at least 512 x 512 px.")
    return _small_face_text(width) if width < FACELOCK_MIN_FACE_PX else None


def _small_face_text(width):
    return (f"The face in your photo is only about {width} px wide, so Magic Hour has little to copy. Add a sharp "
            "head-and-shoulders close-up for a closer likeness. Magic Hour asks for a face area of at least 512 x 512 px.")


def facelock_warning(photo_path, detector=tt.detect_face):
    """A plain-English warning when the face Magic Hour would copy is small (or not found), else None. Never blocks."""
    img = cv2.imread(str(photo_path))
    if img is None:
        return "We could not read the face photo, so Magic Hour may not either. Add a sharp head-and-shoulders close-up."
    try:
        box, _ = detector(img)
    except Exception:
        box = None
    if box is None:
        return ("We could not find a clear face in your photo, so Magic Hour may not either. "
                "Add a sharp head-and-shoulders close-up for a closer likeness.")
    width = int(round(float(box[2])))
    if width >= FACELOCK_MIN_FACE_PX:
        return None
    return _small_face_text(width)


def _set_facelock_warning(d, meta, detector):
    _store_warning(meta, facelock_warning(d / (meta.get("face") or meta["ref"]), detector))


def _store_warning(meta, w):
    if w:
        meta["facelock_warning"] = w
        meta.setdefault("facelock", {})["warning"] = w
    else:
        meta.pop("facelock_warning", None)
        meta.setdefault("facelock", {}).pop("warning", None)


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
    if meta.get("facelock", {}).get("state") in ("queued", "working"):
        raise JobError("Face-lock is running. Wait for it to finish.", 409)
    if skin:
        tsvc.set_skin(root, job, skin, skin_ext)
    for name in ("layer.npy", "tracks.json", "prepared.json", "final.mp4"):
        (d / name).unlink(missing_ok=True)
    tsvc.set_stage(d, "render", state="idle")
    tsvc.set_stage(d, "prepare", state="queued", message=None)


def request_of(root, job):
    """What the mask step needs: file names and the options chosen at upload. `ref` is the image Wan generates from (the
    collage when asked for); `photo` is always the main photo."""
    d = tsvc.job_dir(root, job)
    meta = tsvc.read_meta(d)
    return {"dir": d, "ref": generation_ref(d, meta), "photo": meta["ref"], **meta["request"]}


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
    """The GPU step made `final_path`: it becomes src.mp4, the clip the slider works on, and the slider's face tracking is queued.
    With face-lock asked for, face-lock waits for the user's Magic Hour key instead and the slider is NOT queued (the clip will
    change). Returns True when the caller must spawn the slider preparation."""
    d = tsvc.job_dir(root, job)
    tmp = d / "src.mp4.tmp"
    shutil.copyfile(final_path, tmp)
    tmp.replace(d / "src.mp4")
    tsvc.set_stage(d, "swap", state="done", had_audio=bool(had_audio), bytes=(d / "src.mp4").stat().st_size, finished=time.time())
    if tsvc.read_meta(d).get("request", {}).get("facelock"):
        tsvc.set_stage(d, "facelock", state="waiting")
        return False
    tsvc.set_stage(d, "prepare", state="queued")
    return True


FACELOCK_KEY_MIN = 8


def start_facelock(root, job, key, face=None, face_ext="", detector=tt.detect_face, faces=None, face_detector=None):
    """Check face-lock may start now and queue it. The key is only checked here, never stored: the caller passes it straight to
    the spawned function. Optional `faces` (list of (bytes, ext); `face` is the one-photo form) are new candidates: the best
    of them and the main photo replaces the face photo from before."""
    if not isinstance(key, str) or len(key.strip()) < FACELOCK_KEY_MIN:
        raise JobError("Connect your Magic Hour key first.", 400)
    d = tsvc.job_dir(root, job)
    meta = tsvc.read_meta(d)
    if meta.get("swap", {}).get("state") != "done" or not (d / "src.mp4").exists():
        raise JobError("The swap is not finished.", 409)
    state = meta.get("facelock", {}).get("state", "idle")
    if state not in ("idle", "waiting", "error", "skipped"):
        raise JobError("Face-lock is already running or done for this clip." if state != "done"
                       else "Face-lock is already done for this clip.", 409)
    if meta.get("prepare", {}).get("state") in ("queued", "preparing"):
        raise JobError("The slider is being prepared. Wait for it to finish, then lock the face.", 409)
    if meta.get("render", {}).get("state") in ("queued", "rendering"):
        raise JobError("A render is running. Wait for it to finish.", 409)
    uploads = face_uploads(faces, face, face_ext)
    if uploads:
        pick_face(d, meta, uploads, _picker_detector(detector, face_detector))
        tsvc.write_meta(d, meta)
    tsvc.set_stage(d, "facelock", state="queued", message=None, note=None, credits=None, started=time.time())


def skip_facelock(root, job):
    """Use the swap as it is: only while face-lock waits for the key or after it failed. Returns True when the caller must
    spawn the slider preparation (not when the slider is already ready or on its way, e.g. queued by the failed face-lock)."""
    d = tsvc.job_dir(root, job)
    meta = tsvc.read_meta(d)
    state = meta.get("facelock", {}).get("state", "idle")
    if state not in ("waiting", "error"):
        raise JobError("A finished face-lock cannot be skipped." if state == "done"
                       else "Face-lock can only be skipped while it waits or after it failed.", 409)
    tsvc.set_stage(d, "facelock", state="skipped", message=None)
    if meta.get("prepare", {}).get("state") in ("ready", "queued", "preparing"):
        return False
    tsvc.set_stage(d, "prepare", state="queued", message=None)
    return True


def facelock_blocks_render(root, job):
    """A render must not start while face-lock may replace the clip under it."""
    if tsvc.read_meta(tsvc.job_dir(root, job)).get("facelock", {}).get("state") in ("queued", "working"):
        raise JobError("Face-lock is running. Wait for it to finish.", 409)


# A step that is "running" for longer than its Modal time limit (plus a margin) died without writing an answer (out of credits,
# a restart, a time-out). Say so instead of leaving the card spinning for ever.
ACTIVE = {"mask": ("queued", "preparing"), "swap": ("queued", "generating"), "prepare": ("queued", "preparing"),
          "render": ("queued", "rendering"), "facelock": ("queued", "working")}
STUCK_AFTER_SECONDS = {"mask": 3600 + 300, "swap": 10800 + 300, "prepare": 1800 + 300, "render": 3600 + 300,
                       "facelock": 1800 + 300}
STUCK_MESSAGE = {
    "mask": "The mask step stopped without finishing (Modal's time limit, a restart, or no credit left). Start over.",
    "swap": "The swap stopped without finishing (Modal's time limit, a restart, or no credit left). Check your Modal dashboard, then try again.",
    "prepare": "Preparing the texture slider stopped without finishing. Try again.",
    "render": "The render stopped without finishing. Try again.",
    "facelock": "Face-lock stopped without finishing (Modal's time limit or a restart). Check your Magic Hour dashboard, "
                "then try again or skip it to use the swap as it is."}


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
    facelock = dict(m.get("facelock") or {"state": "idle"})
    if m.get("facelock_warning"):
        facelock["warning"] = m["facelock_warning"]
    out = {"job": job, "mask": m.get("mask", {}), "swap": m.get("swap", {}), "prepare": m.get("prepare", {}),
           "render": m.get("render", {}), "facelock": facelock, "request": m.get("request", {}), "estimate": m.get("estimate"),
           "expires_at": m["created"] + tsvc.JOB_TTL_SECONDS}
    if m.get("collage"):                      # EXPERIMENTAL collage reference: which added photo it used
        out["collage"] = {"face": m["collage"]["face"], "of": m["collage"]["of"]}
    return out


def swapped_path(root, job):
    """The swapped clip as the GPU step made it (no texture): the fallback when the slider cannot be used."""
    d = tsvc.job_dir(root, job)
    if tsvc.read_meta(d).get("swap", {}).get("state") != "done" or not (d / "src.mp4").exists():
        raise JobError("The swap is not finished.", 409)
    return d / "src.mp4"
