"""Syok Swap helpers: CPU-only video plumbing around Wan2.2-Animate (Apache-2.0).

Ideas taken from the two reference projects, re-implemented here (no code copied):
  * Openjutsu: margin presets (tight/normal/loose), section (start/length) of the source,
    keep the swap region only and leave the rest of the picture alone.
  * open-genjutsu: review the mask BEFORE paying for generation, and deliver the
    ORIGINAL soundtrack untouched on the result.
"""
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

# (iterations, kernel) for Wan's mask dilation in preprocess_data.py. normal == Wan's default.
MARGINS = {"tight": (2, 5), "normal": (3, 7), "loose": (4, 9)}
# (w_len, h_len): the grid of Wan's get_aug_mask (wan/modules/animate/preprocess/utils.py), passed as --w_len / --h_len to
# preprocess_data.py. Wan fills every grid cell of the person's bounding box that touches the (dilated) SAM2 mask. Wan's CLI
# default is 1 x 1: ONE cell, so the WHOLE bounding box is filled (the wide red rectangle over the hedge, found 20 Nov 2026).
# "silhouette" uses get_aug_mask's own default grid (10 x 20 cells), so the mask follows the person's outline in cells of
# 1/10 of the person's width and 1/20 of their height. "box" is the old behaviour, kept reachable.
MASK_SHAPES = {"silhouette": (10, 20), "box": (1, 1)}
DEFAULT_MASK_SHAPE = "silhouette"


def preprocess_mask_args(margin, mask_shape=DEFAULT_MASK_SHAPE):
    """The mask flags for Wan's preprocess_data.py in replace mode: dilation (margin) and grid (mask shape)."""
    if margin not in MARGINS:
        raise ValueError("Margin must be tight, normal or loose.")
    if mask_shape not in MASK_SHAPES:
        raise ValueError("Mask shape must be silhouette or box.")
    it, k = MARGINS[margin]
    wl, hl = MASK_SHAPES[mask_shape]
    return ["--iterations", str(it), "--k", str(k), "--w_len", str(wl), "--h_len", str(hl)]
RESOLUTIONS = {"480p": (832, 480), "720p": (1280, 720)}  # target AREA as [w, h]
MAX_SECONDS = 30.0


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-800:]}")
    return r.stdout


def probe(path):
    """Return dict(duration, fps, width, height, has_audio)."""
    j = json.loads(sh(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    num, den = v["avg_frame_rate"].split("/")
    return {
        "duration": float(j["format"]["duration"]),
        "fps": float(num) / float(den) if float(den) else 30.0,
        "width": int(v["width"]),
        "height": int(v["height"]),
        "has_audio": any(s["codec_type"] == "audio" for s in j["streams"]),
    }


def plan_section(duration, start=0.0, length=0.0):
    """Clamp a (start, length) request to the clip. length 0 = to the end. Returns (start, length)."""
    if duration <= 0:
        raise ValueError("Source video has no duration.")
    start = max(0.0, float(start))
    if start >= duration - 1.0:
        raise ValueError("Section starts too close to the end of the video.")
    end = duration if not length else min(duration, start + float(length))
    length = end - start
    if length < 1.0:
        raise ValueError("Section must be at least 1 second.")
    if length > MAX_SECONDS:
        raise ValueError(f"Section is {length:.1f}s; the limit is {MAX_SECONDS:.0f}s per job.")
    return round(start, 3), round(length, 3)


def cut_section(src, dst, start, length):
    """Re-encode the chosen part of the source, with its audio, so later steps share one timeline."""
    sh(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(length), "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", str(dst)])
    return dst


def frames(path):
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield f  # BGR
    finally:
        cap.release()


class Writer:
    """Pipe BGR frames to ffmpeg (H.264, yuv420p)."""

    def __init__(self, path, w, h, fps):
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
             "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)

    def put(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait():
            raise RuntimeError("ffmpeg writer failed")


def source_frame_index(i, out_fps, src_frames, src_fps):
    """Frame of the source that is on screen at output frame i."""
    return min(src_frames - 1, int(round(i / out_fps * src_fps - 1e-9)))


def composite_over_original(original, generated, mask, out, feather=7, full_res=False, blend_original=None):
    """Paste the generated person over the ORIGINAL frames at the original size.

    Everything outside the (feathered) mask is the original pixels, untouched. Frames are matched by
    time, so a generated clip at another fps still lines up. Returns the number of frames written.

    full_res=True writes the result at the GENERATED size instead, so the generated person keeps all
    its detail (the original background is enlarged to match, so it is softer than the person).

    blend_original: default False at every size (2026-11-18: final.mp4 at 448x768 showed the same bug). The mask is a coarse region, not a person silhouette
    (on the 720p job the top ~31% of the frame was never masked), so original pixels outside it can be the ORIGINAL
    PERSON's head: found 2026-11-18, the original red-haired head came back through the composite in the head-turn
    frames (100-118) while the raw generation was correct. At full size the enlarged original is softer than the
    generated frame anyway, so the generated frames are used as they are; pass blend_original=True to opt back in.
    """
    if blend_original is None:
        blend_original = False
    info = probe(original)
    ow, oh, ofps = info["width"], info["height"], info["fps"]
    orig = list(frames(original))
    gen = list(frames(generated))
    msk = list(frames(mask))
    if not (orig and gen and msk):
        raise ValueError("Empty video passed to composite.")
    ginfo = probe(generated)
    gfps = ginfo["fps"]
    if full_res:
        ow, oh = ginfo["width"], ginfo["height"]
    n = int(round(len(gen) / gfps * ofps))
    k = (feather * 2 + 1) | 1
    w = Writer(out, ow, oh, ofps)
    try:
        for i in range(n):
            o = orig[min(len(orig) - 1, i)]
            if o.shape[1] != ow or o.shape[0] != oh:
                o = cv2.resize(o, (ow, oh), interpolation=cv2.INTER_CUBIC)
            gi = min(len(gen) - 1, int(round(i / ofps * gfps - 1e-9)))
            g = cv2.resize(gen[gi], (ow, oh), interpolation=cv2.INTER_CUBIC)
            m = cv2.resize(msk[min(len(msk) - 1, gi)], (ow, oh), interpolation=cv2.INTER_LINEAR)
            m = (m.max(axis=2) if m.ndim == 3 else m).astype(np.float32) / 255.0
            m = (m > 0.5).astype(np.float32)
            if not blend_original:
                m = np.ones_like(m)
            if feather and blend_original:
                m = cv2.GaussianBlur(m, (k, k), 0)
            m = m[..., None]
            w.put(o.astype(np.float32) * (1 - m) + g.astype(np.float32) * m)
    finally:
        w.close()
    return n


def mask_preview(original, mask, out, color=(0, 0, 255), alpha=0.45):
    """Red overlay of the area that will be regenerated, so the user can approve before GPU spend."""
    info = probe(original)
    w = Writer(out, info["width"], info["height"], info["fps"])
    msk = list(frames(mask))
    try:
        for i, f in enumerate(frames(original)):
            m = cv2.resize(msk[min(len(msk) - 1, i)], (f.shape[1], f.shape[0]))
            m = ((m.max(axis=2) if m.ndim == 3 else m) > 127)[..., None]
            tint = np.zeros_like(f)
            tint[:] = color
            w.put(np.where(m, f * (1 - alpha) + tint * alpha, f))
    finally:
        w.close()
    return out


def _find_yunet():
    for p in (Path(__file__).with_name("face_detection_yunet_2023mar.onnx"),
              Path(__file__).parent / "lora" / "face_detection_yunet_2023mar.onnx"):
        if p.exists():
            return p
    return None


def detect_face_box(bgr):
    """Largest face as (x, y, w, h) in this frame's pixels from YuNet, or None. Profile and back-of-head views are often missed."""
    model = _find_yunet()
    if model is None or not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("YuNet face model missing (face_detection_yunet_2023mar.onnx).")
    k = 320 / min(bgr.shape[:2])
    small = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    det = cv2.FaceDetectorYN.create(str(model), "", (w, h), 0.5, 0.3, 5000)
    _, faces = det.detect(small)
    if faces is None:
        return None
    f = max(faces, key=lambda r: r[2] * r[3])
    return (float(f[0]) / k, float(f[1]) / k, float(f[2]) / k, float(f[3]) / k)


def _median_filter(vals, window=9):
    out, half = [], window // 2
    for i in range(len(vals)):
        seg = [v for v in vals[max(0, i - half): i + half + 1] if v is not None]
        out.append(float(np.median(seg)) if seg else None)
    return out


def head_boxes(masks, faces):
    """Wan's replace mask can stop at the face or neck: Wan prompts SAM2 with body points and then fills the mask's whole
    bounding box (with the 1 x 1 grid: MASK_SHAPES["box"]; head_ellipses() is the silhouette version), so when SAM2 leaves the head out the ORIGINAL head stays in the background video and comes back through
    the result (found 2026-11-18: the top 31% of the mask was never marked and the original red-haired head returned in
    the head-turn frames).
    masks: per-frame 2-D arrays (>127 = person); faces: per-frame (x, y, w, h) in mask pixels or None, detected on the
    original frames. Returns per-frame (x0, y0, x1, y1) boxes to add to the mask, from above the hair down to the mask's
    top edge (None = nothing to add). Frames without a detection (profile, turned away) take the position interpolated
    from the frames that have one, then everything is smoothed over time; a detected face is never left out.
    Raises if no face was found anywhere: the head cannot be placed without guessing."""
    if not any(f is not None for f in faces):
        raise ValueError("No face was detected in any frame, so the head cannot be added to the mask.")
    idx = [i for i, f in enumerate(faces) if f is not None]

    def per_frame(vals):
        v = np.interp(np.arange(len(faces)), idx, [vals[i] for i in idx])    # holds the nearest value beyond the ends
        return _median_filter(list(v))

    tops = per_frame([None if f is None else f[1] - 0.9 * f[3] for f in faces])        # hair sits above the face box
    x0s = per_frame([None if f is None else f[0] - 0.9 * f[2] for f in faces])
    x1s = per_frame([None if f is None else f[0] + 1.9 * f[2] for f in faces])
    out = []
    for m, f, t, a, z in zip(masks, faces, tops, x0s, x1s):
        ys, xs = np.nonzero(m > 127)
        if len(ys) == 0:
            out.append(None)
            continue
        y0 = int(ys.min())
        if f is not None:
            t, a, z = min(t, f[1] - 0.9 * f[3]), min(a, f[0] - 0.9 * f[2]), max(z, f[0] + 1.9 * f[2])
        t -= 0.1 * (y0 - t) if y0 > t else 0                      # a little extra above the hair
        if t >= y0:
            out.append(None)
            continue
        out.append((int(max(0, a)), int(max(0, t)), int(min(m.shape[1], z)), y0))
    return out


HEAD_HALF_WIDTH = 1.2      # head ellipse half-width, in face-box widths (the old rectangle was 2.8 face widths wide)
HEAD_TOP = 1.0             # top of the head ellipse above the face box top, in face-box heights (hair room)
HEAD_BOTTOM = 1.15         # bottom of the head ellipse below the face box top, in face-box heights (chin and a bit of neck)


def head_ellipses(masks, faces):
    """Head-shaped version of head_boxes() for the silhouette mask: per frame an ellipse around the face and the hair, as
    dict(ellipse=(cx, cy, half_w, half_h), face=(x0, y0, x1, y1) or None), or None when the frame's mask is empty.
    The ellipse spans HEAD_TOP face heights above the face box to HEAD_BOTTOM below its top and HEAD_HALF_WIDTH face widths
    either side of its centre, which always contains the face box itself; the detected face box is ALSO returned so it is
    filled as a rectangle too (a detected face is never left outside the mask, whatever the ellipse maths does).
    Frames without a detection take the position interpolated from the frames that have one, smoothed over time, as in
    head_boxes(). Raises if no face was found anywhere."""
    if not any(f is not None for f in faces):
        raise ValueError("No face was detected in any frame, so the head cannot be added to the mask.")
    idx = [i for i, f in enumerate(faces) if f is not None]

    def per_frame(vals):
        v = np.interp(np.arange(len(faces)), idx, [vals[i] for i in idx])
        return _median_filter(list(v))

    cxs = per_frame([None if f is None else f[0] + f[2] / 2 for f in faces])
    tops = per_frame([None if f is None else f[1] - HEAD_TOP * f[3] for f in faces])
    bots = per_frame([None if f is None else f[1] + HEAD_BOTTOM * f[3] for f in faces])
    hws = per_frame([None if f is None else HEAD_HALF_WIDTH * f[2] for f in faces])
    out = []
    for m, f, cx, t, b, hw in zip(masks, faces, cxs, tops, bots, hws):
        if not (m > 127).any():
            out.append(None)
            continue
        face = None
        if f is not None:                       # this frame's own detection wins over the smoothed track
            cx, face = f[0] + f[2] / 2, (f[0], f[1], f[0] + f[2], f[1] + f[3])
            t, b, hw = min(t, f[1] - HEAD_TOP * f[3]), max(b, f[1] + HEAD_BOTTOM * f[3]), max(hw, HEAD_HALF_WIDTH * f[2])
        out.append({"ellipse": (cx, (t + b) / 2, hw, (b - t) / 2), "face": face})
    return out


def head_region(shape, head):
    """Boolean (h, w) array of one head_ellipses() entry: the ellipse plus the face box."""
    h, w = shape
    r = np.zeros((h, w), np.uint8)
    if head is None:
        return r.astype(bool)
    cx, cy, hw, hh = head["ellipse"]
    cv2.ellipse(r, (int(round(cx)), int(round(cy))), (int(np.ceil(hw)) + 1, int(np.ceil(hh)) + 1), 0, 0, 360, 1, -1)
    if head["face"] is not None:
        x0, y0, x1, y1 = head["face"]
        r[max(0, int(np.floor(y0))):max(0, int(np.ceil(y1)) + 1), max(0, int(np.floor(x0))):max(0, int(np.ceil(x1)) + 1)] = 1
    return r.astype(bool)


def _bbox_fill(m):
    """What Wan's 1 x 1 grid makes of a mask: its whole bounding box."""
    out = np.zeros_like(m, dtype=bool)
    ys, xs = np.nonzero(m)
    if len(ys):
        out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    return out


def cover_head_files(proc_dir, section, face_fn=None, shape="box"):
    """Rebuild Wan's src_mask.mp4 and src_bg.mp4 in proc_dir with the head inside the mask. Faces are detected on the
    ORIGINAL section frames (the background video has the lower face blacked out when the mask reaches it), matched by time
    and scaled to the mask's size. The old files are kept as *_nohead.mp4.
    shape "box" (the old behaviour): a rectangle from above the hair down to the mask's top (head_boxes()).
    shape "silhouette": a head-shaped ellipse plus the detected face box (head_ellipses()), so the head is covered without a
    wide block of background. In both, the background video is blacked out exactly where the mask is added, as Wan builds
    it (src_bg = frame * (1 - mask)), and the two videos keep the same frames.
    Returns stats: frames, frames with a detected face, frames extended, mean extension in pixels, and "coverage": the mean
    share of the frame in Wan's mask before the head was added (wan_mean), after (final_mean), and what the old box path
    would have covered on this same mask (box_mean: Wan's mask filled to its bounding box plus the old head rectangle)."""
    if shape not in MASK_SHAPES:
        raise ValueError("Mask shape must be silhouette or box.")
    proc = Path(proc_dir)
    face_fn = face_fn or detect_face_box
    bg_p, mk_p = proc / "src_bg.mp4", proc / "src_mask.mp4"
    bg, mk = list(frames(bg_p)), list(frames(mk_p))
    if not bg or len(bg) != len(mk):
        raise ValueError(f"Background and mask videos differ in length ({len(bg)} vs {len(mk)} frames).")
    info, sinfo = probe(mk_p), probe(section)
    sec = list(frames(section))
    sx, sy = info["width"] / sec[0].shape[1], info["height"] / sec[0].shape[0]
    faces, cache = [], {}
    for j in range(len(mk)):
        si = min(len(sec) - 1, int(round(j / info["fps"] * sinfo["fps"] - 1e-9)))
        if si not in cache:
            cache[si] = face_fn(sec[si])
        f = cache[si]
        faces.append(None if f is None else (f[0] * sx, f[1] * sy, f[2] * sx, f[3] * sy))
    gray = [f.max(axis=2) for f in mk]
    hh, ww = gray[0].shape
    # what the old box path makes of this same person mask: bounding box + head rectangle (for the coverage comparison)
    filled = [_bbox_fill(g > 127) for g in gray]
    old_boxes = head_boxes([f.astype(np.uint8) * 255 for f in filled], faces)
    box_cover = []
    for f, b in zip(filled, old_boxes):
        f = f.copy()
        if b is not None:
            f[b[1]:b[3], b[0]:b[2]] = True
        box_cover.append(float(f.mean()))
    if shape == "box":
        boxes = head_boxes(gray, faces)
        adds = []
        for b in boxes:
            a = np.zeros((hh, ww), bool)
            if b is not None:
                a[b[1]:b[3], b[0]:b[2]] = True
            adds.append(a)
    else:
        heads = head_ellipses(gray, faces)
        adds = [head_region((hh, ww), x) for x in heads]
    shutil.copy(bg_p, proc / "src_bg_nohead.mp4")
    shutil.copy(mk_p, proc / "src_mask_nohead.mp4")
    wb = Writer(bg_p, info["width"], info["height"], info["fps"])
    wm = Writer(mk_p, info["width"], info["height"], info["fps"])
    wan_cover, final_cover, ext = [], [], []
    try:
        for f, g, a in zip(bg, gray, adds):
            on = g > 127
            wan_cover.append(float(on.mean()))
            new = a & ~on
            if new.any():
                ys = np.nonzero(on)[0]
                top = int(ys.min()) if len(ys) else hh
                ext.append(max(0, top - int(np.nonzero(a)[0].min())))
            g = g.copy()
            f = f.copy()
            g[a] = 255
            f[g > 127] = 0          # bg = frame * (1 - mask) everywhere, as Wan builds it (also clears codec bleed at old edges)
            final_cover.append(float((g > 127).mean()))
            wm.put(np.repeat(g[..., None], 3, axis=2))
            wb.put(f)
    finally:
        wb.close()
        wm.close()
    return {"frames": len(bg), "faces_detected": sum(1 for f in faces if f is not None),
            "frames_extended": len(ext), "mean_extension_px": float(np.mean(ext)) if ext else 0.0,
            "head_shape": "rectangle" if shape == "box" else "ellipse", "mask_shape": shape,
            "coverage": {"wan_mean": float(np.mean(wan_cover)), "final_mean": float(np.mean(final_cover)),
                         "box_mean": float(np.mean(box_cover))}}


def mask_stats(mask):
    """Share of the frame covered by the mask, per video: (min, mean, max). Flags empty/huge masks."""
    cover = [float((((f.max(axis=2)) > 127).mean())) for f in frames(mask)]
    if not cover:
        raise ValueError("Mask video has no frames.")
    stats = {"min": min(cover), "mean": float(np.mean(cover)), "max": max(cover), "frames": len(cover)}
    stats["warnings"] = []
    if stats["mean"] < 0.01:
        stats["warnings"].append("Mask is almost empty: no person was found. Do not generate.")
    if stats["max"] > 0.85:
        stats["warnings"].append("At some point the person fills most of the picture, so most of it (background included) is redrawn there. This is normal for close-ups.")
    return stats


def mux_original_audio(video, original, out):
    """Copy the video as is and attach the original soundtrack; refuse to trim/stretch beyond 0.5 s."""
    if not probe(original)["has_audio"]:
        sh(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-c", "copy", str(out)])
        return out, False
    vd, ad = probe(video)["duration"], probe(original)["duration"]
    if abs(vd - ad) > 0.5:
        raise ValueError(f"Video is {vd:.2f}s but the original audio is {ad:.2f}s; refusing to trim or stretch it.")
    sh(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(original), "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", str(min(vd, ad)), str(out)])
    return out, True


def normalize_mask(mask, out):
    """Turn Wan's mask into a clean grayscale video. Wan marks the PERSON as white (src_bg = frame * (1 - mask) in
    wan/modules/animate/preprocess/process_pipepline.py), so the polarity is kept exactly as Wan wrote it - it is never
    guessed from coverage (a close-up portrait legitimately fills most of the frame). Returns (path, inverted=False)."""
    fs = list(frames(mask))
    if not fs:
        raise ValueError("Mask video has no frames.")
    info = probe(mask)
    w = Writer(out, info["width"], info["height"], info["fps"])
    try:
        for f in fs:
            g = f.max(axis=2)
            w.put(np.repeat(g[..., None], 3, axis=2))
    finally:
        w.close()
    return out, False
