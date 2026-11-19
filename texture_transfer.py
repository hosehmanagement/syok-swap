"""Texture transfer: copy the reference photo's fine skin detail (stubble, pores) onto the swapped face.

Why: the video model's encoder drops high-frequency detail, so the generated face is smoother than the
reference. This step takes the reference's HIGH-FREQUENCY layer (the photo minus its blurred copy, so no
lighting or colour comes along), attaches it to the tracked face box and adds it to the luminance only.
Because it is anchored to the face box (not the screen), it moves with the head. Stubble and pores are
random patterns, so exact feature alignment is not needed, only the right density in the right area.

v2 (found in the live test): copying the reference's high-pass layer pixel for pixel also copied its eyes, brows,
nose lines and mouth, which showed up as a mask on the swapped face. So the layer is now NEW grain, made from the
reference's clean-skin statistics only (same pore size and strength, no features, no layout), and apply_texture
keeps it off the swapped face's own eyes, glasses, brows, lips and hard edges.

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


def _radial_profile(patches, nbins=32):
    """Average isotropic amplitude spectrum (cycles/pixel 0..0.5) of zero-mean patches. Returns (freqs, amps)."""
    p = patches[0].shape[0]
    win = np.outer(np.hanning(p), np.hanning(p)).astype(np.float32)
    pw = np.zeros((p, p), np.float64)
    for q in patches:
        q = (q - q.mean()) * win
        pw += np.abs(np.fft.fft2(q)) ** 2
    pw /= len(patches)
    fy, fx = np.fft.fftfreq(p)[:, None], np.fft.fftfreq(p)[None, :]
    fr = np.hypot(fy, fx)
    edges = np.linspace(0.0, 0.5, nbins + 1)
    idx = np.clip(np.digitize(fr, edges) - 1, 0, nbins - 1)
    amps = np.array([np.sqrt(pw[idx == k].mean()) if (idx == k).any() else 0.0 for k in range(nbins)])
    return 0.5 * (edges[:-1] + edges[1:]), amps


def _clean_skin_mask(y_ch, hf, sk_mask, sigma, k_e=1.6):
    """True where the reference is plain skin: skin coloured, no feature edges, no outlier brightness or contrast."""
    skin = sk_mask > 0.8
    if not skin.any():
        return skin
    coarse = cv2.GaussianBlur(y_ch, (0, 0), sigma * 3.0)
    gx, gy = cv2.Sobel(coarse, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(coarse, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), sigma * 2.0)
    energy = cv2.GaussianBlur(np.abs(hf), (0, 0), sigma * 3.0)
    lum = cv2.GaussianBlur(y_ch, (0, 0), sigma * 3.0)
    ok = skin & (mag < k_e * np.median(mag[skin])) & (energy < k_e * np.median(energy[skin]))
    ok &= np.abs(lum - np.median(lum[skin])) < 0.22 * max(1.0, float(np.median(lum[skin])))
    k = max(1, int(round(sigma)))
    return cv2.erode(ok.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool)


def make_detail_layer(ref_bgr, ref_box, size_wh, sigma_frac=0.005, clip_mult=2.5, seed=1234):
    """NEW skin grain with the reference's clean-skin statistics, sized size_wh (w, h).

    Returns float32 (h, w): zero mean, soft-clipped. It has the reference's pore size and strength, but none of its
    features (no eyes, brows, nose or mouth), so nothing is pasted at the wrong place. Same input -> same layer,
    so a preview and the full render always match.
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
    hh, ww = hf.shape
    patches, clean = [], None
    for k_e in (1.6, 2.2, 3.0):                                  # strict first; looser only if the photo has little plain skin
        clean = _clean_skin_mask(y_ch, hf, sk_mask, sigma, k_e)
        for p in (64, 48, 32, 24, 16):
            if min(hh, ww) < p * 2:
                continue
            step = max(4, p // 3)
            patches = [hf[j:j + p, i:i + p] for j in range(0, hh - p + 1, step) for i in range(0, ww - p + 1, step)
                       if clean[j:j + p, i:i + p].mean() >= 0.96]
            if len(patches) >= 3:
                break
        if len(patches) >= 3 and int(clean.sum()) >= 400:
            break
    if len(patches) < 3 or int(clean.sum()) < 400:
        raise ValueError("Not enough plain skin in the reference photo (glasses, hair, beard, shadows or heavy make-up "
                         "cover it). Use a sharper, evenly lit, front-facing photo.")
    freqs, amps = _radial_profile(patches)
    sd = float(hf[clean].std())                                   # strength of the reference's plain-skin grain
    if sd < 1e-6:
        return np.zeros_like(hf)
    # filter white noise so its spectrum matches the reference skin's, then set its strength to the reference's
    fy, fx = np.fft.fftfreq(hh)[:, None], np.fft.rfftfreq(ww)[None, :]
    filt = np.interp(np.hypot(fy, fx), freqs, amps, left=amps[0], right=amps[-1]).astype(np.float32)
    filt[0, 0] = 0.0                                              # no low-frequency / brightness drift at all
    noise = np.random.default_rng(seed).standard_normal((hh, ww)).astype(np.float32)
    layer = np.fft.irfft2(np.fft.rfft2(noise) * filt, s=(hh, ww)).astype(np.float32)
    layer -= layer.mean()
    layer *= sd / max(float(layer.std()), 1e-9)
    c = clip_mult * sd
    layer = c * np.tanh(layer / c)
    # feather the borders so the crop edge can never draw a line
    fy2 = np.clip(np.minimum(np.arange(hh), hh - 1 - np.arange(hh)) / (0.10 * hh), 0, 1)
    fx2 = np.clip(np.minimum(np.arange(ww), ww - 1 - np.arange(ww)) / (0.10 * ww), 0, 1)
    layer *= np.outer(fy2, fx2).astype(np.float32)
    return layer.astype(np.float32)


def protect_mask(roi_bgr, skin):
    """0..1 weight: 1 on plain skin of the SWAPPED face, falling to 0 on its eyes, glasses, brows, lips and hard edges."""
    h, w = roi_bgr.shape[:2]
    y_ch = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sigma = max(1.0, 0.012 * w)
    sel = skin > 0.5
    if not sel.any():
        return np.zeros((h, w), np.float32)
    coarse = cv2.GaussianBlur(y_ch, (0, 0), sigma * 1.5)
    gx, gy = cv2.Sobel(coarse, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(coarse, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), sigma)
    g0 = 2.5 * float(np.median(mag[sel])) + 1e-3
    w_edge = 1.0 / (1.0 + (mag / g0) ** 2)                        # frames, brows, lip and eye edges, hairline
    lum = cv2.GaussianBlur(y_ch, (0, 0), sigma)
    ls = float(np.percentile(lum[sel], 75))                       # skin is brighter than hair, brows and frames
    w_dark = np.clip((lum - 0.55 * ls) / (0.25 * ls + 1e-3), 0, 1)  # glasses frames, brows, pupils, hair
    ycc = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    cr = cv2.GaussianBlur(ycc[..., 1], (0, 0), sigma)
    crs = float(np.median(cr[sel & (lum > 0.8 * ls)]) if (sel & (lum > 0.8 * ls)).any() else np.median(cr[sel]))
    w_lip = np.clip(1.0 - (cr - (crs + 6.0)) / 10.0, 0, 1)           # redder than the skin around it = lips
    # eyes sit in a fixed band of a face box: keep the grain off the eye sockets and the glasses' lens area
    eyes = np.zeros((h, w), np.float32)
    for cx in (0.30, 0.70):
        cv2.ellipse(eyes, (int(cx * w), int(0.37 * h)), (int(0.17 * w), int(0.075 * h)), 0, 0, 360, 1.0, -1)
    w_eye = 1.0 - cv2.GaussianBlur(eyes, (0, 0), max(1.5, 0.03 * w))
    return (w_edge * w_dark * w_lip * w_eye).astype(np.float32)


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
    skin = skin_prob(roi)
    mask = cv2.GaussianBlur(ell, (0, 0), max(2.0, 0.04 * w)) * skin * protect_mask(roi, skin)
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


DEFAULT_STRENGTH = 2.0   # measured on one fictional face: 2.0 carries 0.98-1.04 of the reference's fine detail
MIN_STRENGTH = 0.0       # 0 = texture pass off (lets a user compare with and without)
MAX_STRENGTH = 3.0       # 3.0 already overshoots the reference (1.4x) and looks gritty by eye; do not offer more


def clamp_strength(strength):
    """Slider value -> a safe strength in [MIN_STRENGTH, MAX_STRENGTH]. Refuses NaN/None/non-numbers."""
    try:
        v = float(strength)
    except (TypeError, ValueError):
        raise ValueError("Strength must be a number.")
    if v != v:
        raise ValueError("Strength must be a number.")
    return min(MAX_STRENGTH, max(MIN_STRENGTH, v))


def prepare(src, ref_path, sigma_frac=0.005, detector=detect_face):
    """Do the slow, strength-independent work ONCE: find the face in the reference and track it through the clip.

    Returns a dict that preview_frame()/render_prepared() reuse for any strength, so a slider never re-detects.
    """
    ref = cv2.imread(str(ref_path))
    if ref is None:
        raise ValueError("Reference image unreadable.")
    ref_box, _ = detector(ref)
    if ref_box is None:
        raise ValueError("No face found in the reference image.")
    info = sk.probe(src)
    dets = [detector(f) for f in sk.frames(src)]
    tracks = smooth_tracks(reject_outliers(dets))
    widths = [b[2] for b, w in tracks if b is not None and w >= 0.7]
    if not widths:
        raise ValueError("No face tracked in the clip; nothing to do.")
    size = int(round(float(np.median(widths)) * PAD[2])), int(round(float(np.median(widths)) * PAD[3]))
    layer = make_detail_layer(ref, ref_box, size, sigma_frac)
    return {"info": info, "tracks": tracks, "layer": layer, "layer_size": size,
            "frames": len(dets), "detected": sum(b is not None for b, _ in dets)}


def preview_frame(src, prepared, index, strength=DEFAULT_STRENGTH):
    """One finished frame (BGR uint8) at the given strength: what a slider shows while it is dragged."""
    strength = clamp_strength(strength)
    if not 0 <= index < prepared["frames"]:
        raise IndexError(f"Frame {index} is outside 0..{prepared['frames'] - 1}.")
    for i, f in enumerate(sk.frames(src)):
        if i == index:
            b, wgt = prepared["tracks"][i]
            return apply_texture(f, b, prepared["layer"], strength, wgt)
    raise IndexError(f"Frame {index} not found in the clip.")


def render_prepared(src, prepared, out, strength=DEFAULT_STRENGTH):
    """Write the whole clip at the chosen strength (audio kept). Returns the same stats dict as process_video."""
    strength = clamp_strength(strength)
    info = prepared["info"]
    tmp = str(out) + ".noaudio.mp4"
    wr = sk.Writer(tmp, info["width"], info["height"], info["fps"])
    applied = 0
    try:
        for f, (b, wgt) in zip(sk.frames(src), prepared["tracks"]):
            wr.put(apply_texture(f, b, prepared["layer"], strength, wgt))
            applied += int(b is not None)
    finally:
        wr.close()
    sk.mux_original_audio(tmp, src, out)
    return {"frames": prepared["frames"], "applied": applied, "detected": prepared["detected"],
            "layer_size": prepared["layer_size"], "strength": strength}


def process_video(src, ref_path, out, strength=1.0, sigma_frac=0.005, detector=detect_face):
    """Apply the texture transfer to a swapped clip. Keeps frame count, size and audio. Returns stats dict.

    Kept as the one-call form (old default strength 1.0, unchanged so existing callers behave the same).
    A slider UI should call prepare() once, then preview_frame() while dragging and render_prepared() on confirm.
    """
    return render_prepared(src, prepare(src, ref_path, sigma_frac, detector), out, strength)
