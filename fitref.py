"""Fit the reference to the clip: the new person is painted into the FIRST and the LAST frame, and those two views become Wan's reference.

Wan2.2-Animate takes ONE reference image and the person in it may differ from the person in the clip in build, pose, framing and light.
An image-editing model (Qwen-Image-Edit, Apache-2.0, run on the user's own Modal GPU: app.py `fit_job`) replaces the old person in the
first and the last frame with the person from the photo, keeping the scene. The two edited person crops go side by side into ONE
reference image (like collage.py), so Wan sees the new person at the pose, size, light and scene of both ends of the clip.

This module is the part that runs anywhere (CPU, tested): which frames, which crop, the prompt, the guards that refuse a bad edit, and
the reference image. The model itself is only a function `edit(scene_bgr, person_bgr, prompt) -> edited scene bgr` that app.py supplies.

    run_fit(job_dir, edit) -> dict   writes fit_first.png, fit_last.png, fit_ref.png into job_dir (only the ones that passed)

Guards (a bad edit is dropped, never fed to Wan): the scene outside the person must still be the scene (mean change at most
MAX_BACKGROUND_CHANGE of 255) and the person's area must really have changed (at least MIN_PERSON_CHANGE). If neither frame passes, the
original photo stays the reference and the result says why.
"""
import io
from pathlib import Path

import cv2
import numpy as np

CROP_PAD = 0.30                  # context around the person, in person-box sizes, on each side
MIN_CROP = 320                   # px: never edit a crop smaller than this on its shorter side (the model needs context)
MAX_BACKGROUND_CHANGE = 30.0     # mean |edited - original| outside the person, 0-255: more means the scene was redrawn
MIN_PERSON_CHANGE = 12.0         # mean |edited - original| inside the person: less means nothing was replaced
PERSON_PAD = 0.08                # tight person crop for the reference, in person-box sizes
NEUTRAL = (180, 180, 180)        # BGR background of the reference canvas
GAP_FRACTION = 0.02

PROMPT = ("Replace the person in Picture 1 with the person shown in Picture 2: the same face, hair, skin tone and body build as Picture 2. "
          "Keep Picture 1 exactly as it is otherwise: the same background, camera angle, framing, lighting, colour grading, the same pose "
          "and body position, and the same kind of clothing. Photorealistic, natural skin, no extra people.")


def first_last_frames(video):
    """(first, last) frame of a video as BGR arrays; (None, None) for an empty one."""
    cap = cv2.VideoCapture(str(video))
    first = last = None
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if first is None:
                first = f
            last = f
    finally:
        cap.release()
    return first, last


def person_bbox(mask):
    """(x0, y0, x1, y1) of the True pixels of a boolean HxW mask, or None when empty. x1/y1 are exclusive."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_box(bbox, shape, pad=CROP_PAD, min_side=MIN_CROP):
    """The scene crop to edit: the person's box grown by `pad` of its size on each side, at least `min_side` on the shorter side
    (when the photo allows), kept inside the frame. shape = (h, w). Returns (x0, y0, x1, y1)."""
    h, w = shape[:2]
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = x0 - bw * pad, x1 + bw * pad
    y0, y1 = y0 - bh * pad, y1 + bh * pad
    def fit(lo, hi, size):
        if hi - lo < min(min_side, size):
            mid = (lo + hi) / 2
            lo, hi = mid - min(min_side, size) / 2, mid + min(min_side, size) / 2
        if lo < 0:
            hi, lo = hi - lo, 0
        if hi > size:
            lo, hi = lo - (hi - size), size
        return max(0, int(round(lo))), min(size, int(round(hi)))
    x0, x1 = fit(x0, x1, w)
    y0, y1 = fit(y0, y1, h)
    return x0, y0, x1, y1


def mask_at(video_frame_mask, shape):
    """A mask-video frame (BGR/grey) as a boolean mask at (h, w) = shape[:2]."""
    g = video_frame_mask.max(axis=2) if video_frame_mask.ndim == 3 else video_frame_mask
    g = cv2.resize(g, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return g > 127


def check_edit(original, edited, person_mask):
    """Is `edited` an acceptable replacement of `original` (same size, BGR)? Returns (ok, outside_change, inside_change, reason)."""
    if edited is None or edited.shape != original.shape:
        return False, None, None, "The image editor returned a picture of another size."
    d = np.abs(original.astype(np.float32) - edited.astype(np.float32)).max(axis=2)
    grown = cv2.dilate(person_mask.astype(np.uint8), np.ones((max(3, int(0.05 * min(original.shape[:2]))) | 1,) * 2, np.uint8)).astype(bool)
    outside, inside = d[~grown], d[person_mask]
    if not len(outside) or not len(inside):
        return False, None, None, "No person area to check."
    oc, ic = float(outside.mean()), float(inside.mean())
    if oc > MAX_BACKGROUND_CHANGE:
        return False, oc, ic, "The editor redrew the scene, not only the person."
    if ic < MIN_PERSON_CHANGE:
        return False, oc, ic, "The editor did not replace the person."
    return True, oc, ic, ""


def person_crop(scene, person_mask, pad=PERSON_PAD):
    """The person alone (box of the mask grown by `pad`), from the edited scene."""
    bb = person_bbox(person_mask)
    if bb is None:
        return scene
    x0, y0, x1, y1 = bb
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    h, w = scene.shape[:2]
    return scene[max(0, int(y0 - py)):min(h, int(y1 + py)), max(0, int(x0 - px)):min(w, int(x1 + px))]


def make_fit_reference(crops, max_side=1536):
    """One PNG (bytes): the edited person crops side by side at one height on a neutral background (never upscaled past the tallest
    crop; the whole canvas fits `max_side` on its longer side)."""
    if not crops:
        raise ValueError("No fitted person to build a reference from.")
    h = max(c.shape[0] for c in crops)
    gap = max(4, int(round(GAP_FRACTION * h)))
    ws = [int(round(c.shape[1] * h / c.shape[0])) for c in crops]
    total_w = sum(ws) + gap * (len(crops) - 1)
    scale = min(1.0, max_side / max(total_w, h))
    H = max(16, int(round(h * scale)))
    ws = [max(16, int(round(w * scale))) for w in ws]
    gap = max(2, int(round(gap * scale)))
    canvas = np.zeros((H, sum(ws) + gap * (len(crops) - 1), 3), np.uint8)
    canvas[:] = NEUTRAL
    x = 0
    for c, w in zip(crops, ws):
        canvas[:, x:x + w] = cv2.resize(c, (w, H), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        x += w + gap
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("Could not encode the fitted reference.")
    return buf.tobytes()


def _read_ref(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("The reference photo could not be read.")
    return img


def run_fit(job_dir, edit, ref_photo=None):
    """Edit the first and last frame of the job's section.mp4 with `edit(scene_bgr, person_bgr, prompt) -> bgr` and write
    fit_first.png / fit_last.png (the edited scene crops that passed the guards) and fit_ref.png (the reference for Wan).
    Returns {"state": "done"|"skipped", "frames": {"first": {...}, "last": {...}}, "message": str}; never raises for a bad edit
    (a frame the editor spoiled is simply not used); `edit` itself raising makes that frame "error"."""
    d = Path(job_dir)
    first, last = first_last_frames(d / "section.mp4")
    if first is None:
        return {"state": "skipped", "frames": {}, "message": "The clip has no frames."}
    person_photo = _read_ref(ref_photo)
    mask_path = d / "core.mp4" if (d / "core.mp4").exists() else d / "mask.mp4"
    mfirst, mlast = first_last_frames(mask_path)
    if mfirst is None:
        return {"state": "skipped", "frames": {}, "message": "The mask video has no frames."}
    result, crops = {}, []
    for name, frame, mframe in (("first", first, mfirst), ("last", last, mlast)):
        info = {"state": "skipped"}
        result[name] = info
        pm = mask_at(mframe, frame.shape)
        bb = person_bbox(pm)
        if bb is None:
            info["message"] = "No person found in this frame."
            continue
        x0, y0, x1, y1 = crop_box(bb, frame.shape)
        scene = frame[y0:y1, x0:x1].copy()
        pcrop_mask = pm[y0:y1, x0:x1]
        try:
            edited = edit(scene, person_photo, PROMPT)
        except Exception as e:                     # the model failed on this frame: keep going with the other one
            info.update(state="error", message=f"The image editor failed: {type(e).__name__}")
            continue
        if edited is not None and edited.shape[:2] != scene.shape[:2]:
            edited = cv2.resize(edited, (scene.shape[1], scene.shape[0]), interpolation=cv2.INTER_CUBIC)
        ok, oc, ic, why = check_edit(scene, edited, pcrop_mask)
        info.update(outside_change=oc, inside_change=ic)
        if not ok:
            info.update(state="rejected", message=why)
            continue
        cv2.imwrite(str(d / f"fit_{name}.png"), edited)
        crops.append(person_crop(edited, pcrop_mask))
        info["state"] = "done"
    if not crops:
        msgs = [f"{k}: {v.get('message', v['state'])}" for k, v in result.items()]
        return {"state": "skipped", "frames": result, "message": "No fitted frame passed the checks, so your own photo is used. " + "; ".join(msgs)}
    (d / "fit_ref.png").write_bytes(make_fit_reference(crops))
    n = len(crops)
    return {"state": "done", "frames": result, "message": f"Fitted the new person to {'both ends' if n == 2 else 'one end'} of the clip."}
