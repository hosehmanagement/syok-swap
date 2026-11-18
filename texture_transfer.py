"""Texture transfer: copy the reference photo's fine skin detail (stubble, pores) onto the swapped face.

Why: the video model's encoder drops high-frequency detail, so the generated face is smoother than the
reference. This step takes the reference's HIGH-FREQUENCY layer (the photo minus its blurred copy, so no
lighting or colour comes along), attaches it to the tracked face box and adds it to the luminance only.
Because it is anchored to the face box (not the screen), it moves with the head. Stubble and pores are
random patterns, so exact feature alignment is not needed, only the right density in the right area.

CPU only, OpenCV only (Haar cascades ship inside opencv), no new model weights, so no licence issue.
Limits: box tracking, not a 3D face fit; strong head turns and profile views get a weaker effect.
"""
import cv2
import numpy as np

import swapkit as sk

PAD = (0.10, 0.10, 1.20, 1.30)  # box expanded: x-0.10w, y-0.10h, width*1.2, height*1.3 (adds chin/jaw stubble)


def pad_box(box, pad=PAD):
    x, y, w, h = box
    return (x - pad[0] * w, y - pad[1] * h, w * pad[2], h * pad[3])


def skin_prob(bgr):
    """Soft skin mask 0..1 from chroma (works on stubble too, it stays skin-coloured)."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[..., 1].astype(np.float32), ycc[..., 2].astype(np.float32)
    m = ((cr > 128) & (cr < 182) & (cb > 72) & (cb < 132)).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 2.0)
    return np.clip(m, 0, 1)


def make_detail_layer(ref_bgr, ref_box, size_wh, sigma_frac=0.010, clip_mult=2.5):
    """High-frequency luminance layer of the reference face, resized to size_wh (w, h).

    Returns float32 (h, w): zero mean, soft-clipped so strong edges (brows, nostrils) are not pasted.
    """
    x, y, w, h = [int(round(v)) for v in pad_box(ref_box)]
    H, W = ref_bgr.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    crop = ref_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("Reference face box is outside the image.")
    # match the video's pixel scale BEFORE taking the high-pass, otherwise detail would be far finer than the video can show
    crop = cv2.resize(crop, size_wh, interpolation=cv2.INTER_AREA)
    sk_mask = skin_prob(crop)
    y_ch = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sigma = max(1.0, sigma_frac * size_wh[0])
    hf = y_ch - cv2.GaussianBlur(y_ch, (0, 0), sigma)
    hf *= sk_mask
    # keep real texture, drop copied outlines: attenuate near large-scale edges (jaw, hair line, ear) of the reference
    coarse = cv2.GaussianBlur(y_ch, (0, 0), sigma * 3.0)
    gx, gy = cv2.Sobel(coarse, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(coarse, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), sigma * 2.0)
    g0 = 2.0 * float(np.median(mag[sk_mask > 0.5])) + 1e-3 if (sk_mask > 0.5).any() else float(mag.mean()) + 1e-3
    hf *= 1.0 / (1.0 + (mag / g0) ** 4)
    # feather the borders so the crop edge can never draw a line
    hh, ww = hf.shape
    fy = np.clip(np.minimum(np.arange(hh), hh - 1 - np.arange(hh)) / (0.10 * hh), 0, 1)
    fx = np.clip(np.minimum(np.arange(ww), ww - 1 - np.arange(ww)) / (0.10 * ww), 0, 1)
    hf *= np.outer(fy, fx).astype(np.float32)
    sd = float(hf[sk_mask > 0.5].std()) if (sk_mask > 0.5).any() else float(hf.std())
    if sd < 1e-6:
        return np.zeros_like(hf)
    c = clip_mult * sd
    hf = c * np.tanh(hf / c)
    return hf.astype(np.float32)


def apply_texture(frame_bgr, box, hf_layer, strength=1.0, weight=1.0):
    """Add the detail layer to one frame inside the (padded) face box. Returns a new uint8 frame."""
    if box is None or strength <= 0 or weight <= 0:
        return frame_bgr
    H, W = frame_bgr.shape[:2]
    x, y, w, h = [int(round(v)) for v in pad_box(box)]
    if w < 8 or h < 8:
        return frame_bgr
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return frame_bgr
    layer = cv2.resize(hf_layer, (w, h), interpolation=cv2.INTER_LINEAR)[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = frame_bgr[y0:y1, x0:x1]
    # inscribed ellipse x skin colour: keeps the effect off hair, background and clothes
    ell = np.zeros(roi.shape[:2], np.float32)
    cv2.ellipse(ell, ((x1 - x0) // 2, (y1 - y0) // 2), (int((x1 - x0) * 0.46), int((y1 - y0) * 0.48)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(ell, (0, 0), max(2.0, 0.04 * w)) * skin_prob(roi)
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    ycc[..., 0] = np.clip(ycc[..., 0] + strength * weight * mask * layer, 0, 255)
    out = frame_bgr.copy()
    out[y0:y1, x0:x1] = cv2.cvtColor(ycc.round().astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return out


# ---------------- face tracking (OpenCV Haar, frontal first then profile, mirrored for the other side) ----------------
_CASC = {}


def _cascade(name):
    if name not in _CASC:
        c = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if c.empty():
            raise RuntimeError(f"OpenCV cascade missing: {name}")
        _CASC[name] = c
    return _CASC[name]


MIN_SKIN = 0.35


def box_is_skin(frame_bgr, box, min_skin=MIN_SKIN):
    """True when the middle of the box is mostly skin coloured (rejects windows, hair, cloth)."""
    x, y, w, h = [int(v) for v in box]
    H, W = frame_bgr.shape[:2]
    x0, y0 = max(0, x + w // 4), max(0, y + h // 4)
    x1, y1 = min(W, x + 3 * w // 4), min(H, y + 3 * h // 4)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return False
    return float(skin_prob(frame_bgr[y0:y1, x0:x1]).mean()) >= min_skin


def reject_outliers(dets, lo=0.4, hi=2.5):
    """Drop boxes whose area is far from the clip's median box area (single-frame false hits)."""
    areas = [b[2] * b[3] for b, _ in dets if b is not None]
    if len(areas) < 3:
        return list(dets)
    med = float(np.median(areas))
    return [(b, w) if b is not None and lo * med <= b[2] * b[3] <= hi * med else (None, 0.0) for b, w in dets]


def detect_face(frame_bgr):
    """Return ((x, y, w, h), weight) or (None, 0). Weight 1.0 frontal, 0.7 profile."""
    g = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    H, W = g.shape
    ms = (max(24, min(H, W) // 10),) * 2
    for name in ("haarcascade_frontalface_alt2.xml", "haarcascade_frontalface_default.xml"):
        f = _cascade(name).detectMultiScale(g, 1.1, 5, minSize=ms)
        f = [b for b in f if box_is_skin(frame_bgr, b)]
        if len(f):
            return tuple(max(f, key=lambda b: b[2] * b[3])), 1.0
    prof = _cascade("haarcascade_profileface.xml")
    cands = [tuple(b) for b in prof.detectMultiScale(g, 1.1, 4, minSize=ms)]
    gm = cv2.flip(g, 1)
    cands += [(W - b[0] - b[2], b[1], b[2], b[3]) for b in prof.detectMultiScale(gm, 1.1, 4, minSize=ms)]
    cands = [b for b in cands if box_is_skin(frame_bgr, b)]
    if cands:
        return max(cands, key=lambda b: b[2] * b[3]), 0.7
    return None, 0.0


def smooth_tracks(dets, max_gap=10, window=5):
    """dets: list of (box|None, weight). Fill short gaps by interpolation, then smooth the boxes.

    Returns list of (box|None, weight); interpolated frames get weight 0.5, long gaps stay None.
    """
    n = len(dets)
    boxes = [None if b is None else np.array(b, np.float32) for b, _ in dets]
    wts = [w for _, w in dets]
    idx = [i for i, b in enumerate(boxes) if b is not None]
    out = [(None, 0.0)] * n
    if not idx:
        return out
    for i in range(n):
        if boxes[i] is not None:
            continue
        prev = max((j for j in idx if j < i), default=None)
        nxt = min((j for j in idx if j > i), default=None)
        if prev is not None and nxt is not None and nxt - prev <= max_gap + 1:
            t = (i - prev) / (nxt - prev)
            boxes[i] = boxes[prev] * (1 - t) + boxes[nxt] * t
            wts[i] = 0.5
        elif prev is not None and i - prev <= 3:
            boxes[i], wts[i] = boxes[prev], 0.5
        elif nxt is not None and nxt - i <= 3:
            boxes[i], wts[i] = boxes[nxt], 0.5
    half = window // 2
    for i in range(n):
        if boxes[i] is None:
            continue
        nb = [boxes[j] for j in range(max(0, i - half), min(n, i + half + 1)) if boxes[j] is not None]
        out[i] = (tuple(float(v) for v in np.mean(nb, axis=0)), wts[i])
    return out


def process_video(src, ref_path, out, strength=1.0, sigma_frac=0.010, detector=detect_face):
    """Apply the texture transfer to a swapped clip. Keeps frame count, size and audio. Returns stats dict."""
    ref = cv2.imread(str(ref_path))
    if ref is None:
        raise ValueError("Reference image unreadable.")
    ref_box, _ = detector(ref)
    if ref_box is None:
        raise ValueError("No face found in the reference image.")
    info = sk.probe(src)
    frames = list(sk.frames(src))
    dets = [detector(f) for f in frames]
    tracks = smooth_tracks(reject_outliers(dets))
    widths = [b[2] for b, w in tracks if b is not None and w >= 0.7]
    if not widths:
        raise ValueError("No face tracked in the clip; nothing to do.")
    size = int(round(float(np.median(widths)) * PAD[2])), int(round(float(np.median(widths)) * PAD[3]))
    layer = make_detail_layer(ref, ref_box, size, sigma_frac)
    tmp = str(out) + ".noaudio.mp4"
    wr = sk.Writer(tmp, info["width"], info["height"], info["fps"])
    applied = 0
    try:
        for f, (b, wgt) in zip(frames, tracks):
            g = apply_texture(f, b, layer, strength, wgt)
            applied += int(b is not None)
            wr.put(g)
    finally:
        wr.close()
    sk.mux_original_audio(tmp, src, out)
    return {"frames": len(frames), "applied": applied, "detected": sum(b is not None for b, _ in dets),
            "layer_size": size, "strength": strength}
