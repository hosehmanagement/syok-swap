"""Post-pass that takes the 'AI-smooth' look off a swapped clip. CPU only, no GPU cost.

Diffusion output is denoised toward retouched averages, so pores and micro-contrast vanish and
sharpening only makes it look waxy. What works: put fine detail and film-like noise back
(luminance-only, so no colour speckle), keep it OFF flat backgrounds, and apply it mostly where
the swapped person is. Original pixels outside the mask are left alone when a mask is given.
"""
import cv2
import numpy as np

import swapkit as sk


def add_texture(frame, mask=None, grain=3.0, detail=0.35, seed=0):
    """frame BGR uint8. grain = sigma of luminance noise (0-8 sensible). detail = local-contrast gain."""
    f = frame.astype(np.float32)
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycc[..., 0]
    if detail:
        # restore micro-contrast: difference between fine and slightly blurred luminance
        fine = y - cv2.GaussianBlur(y, (0, 0), 1.6)
        y = y + detail * fine * 2.0
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, grain, y.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.6)  # slightly clumped like film, not pixel salt
    # grain strongest in midtones, weaker in deep shadows/highlights
    mid = 1.0 - np.abs(y - 128.0) / 128.0
    y = y + noise * (0.4 + 0.6 * mid)
    ycc[..., 0] = np.clip(y, 0, 255)
    out = cv2.cvtColor(ycc.round().astype(np.uint8), cv2.COLOR_YCrCb2BGR).astype(np.float32)
    if mask is not None:
        m = mask.astype(np.float32)
        m = (m.max(axis=2) if m.ndim == 3 else m) / 255.0
        m = cv2.GaussianBlur(m, (0, 0), 4)[..., None]
        out = f * (1 - m) + out * m
    return np.clip(out, 0, 255).astype(np.uint8)


def process_video(src, out, mask_video=None, grain=3.0, detail=0.35, seed=0):
    """Apply add_texture per frame (new noise each frame, as real sensors do), keep the audio."""
    info = sk.probe(src)
    tmp = str(out) + ".noaudio.mp4"
    w = sk.Writer(tmp, info["width"], info["height"], info["fps"])
    masks = list(sk.frames(mask_video)) if mask_video else None
    n = 0
    try:
        for i, fr in enumerate(sk.frames(src)):
            m = None
            if masks:
                m = cv2.resize(masks[min(i, len(masks) - 1)], (fr.shape[1], fr.shape[0]))
            w.put(add_texture(fr, m, grain, detail, seed + i))
            n += 1
    finally:
        w.close()
    sk.mux_original_audio(tmp, src, out)
    return n
