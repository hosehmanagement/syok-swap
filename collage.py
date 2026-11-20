"""EXPERIMENTAL collage reference for the Wan2.2-Animate swap. Off by default; nobody has measured yet whether it helps.

Wan2.2-Animate takes ONE reference image. In a full-body photo the face is tiny, so the idea to test is: give Wan one image that
holds the full-body photo AND a sharp head-and-shoulders close-up of the same person, side by side. It may give a closer likeness;
it may also make Wan draw two people or a split image. Only the generation sees the collage: the realism slider, the face check
and face-lock keep using the original main photo.

    make_collage(main_bytes, face_bytes, *, max_side=1536, detector=None) -> PNG bytes

Layout (pure CPU, deterministic: the same inputs always give the same bytes):
    [ main photo, whole ] [gap] [ close-up, cropped to head and shoulders ]
  * Both photos are decoded with their EXIF orientation applied; transparent parts (RGBA, LA, P with transparency) are
    flattened onto the background colour.
  * Background colour: per-channel median of the main photo's border strip (BORDER_FRACTION of its shorter side, at least
    1 px), using only pixels that are at least half opaque; NEUTRAL when the border is fully transparent.
  * Close-up crop: the largest face the detector finds (picker.yunet_faces by default, the same detector as the best-photo
    pick). Box (x, y, w, h) -> centre cx = x + w/2; crop width CROP_W x w, from y - CROP_ABOVE x h down to
    y + h + CROP_BELOW x h, shifted inside the photo where possible, then clipped to it. No face: a centre-top crop whose width
    is the photo's shorter side and whose height is FALLBACK_TALL x that width (clipped to the photo).
  * Both parts are scaled to the same height H (aspect ratios kept: each width = round(H x w/h)), with a gap of
    max(GAP_MIN, round(GAP_FRACTION x H)) px between them. H is the largest height, at most the main photo's own height
    (never upscaled), at which the whole canvas fits in max_side on its longer side. The close-up may be scaled up to H.
"""
import io

import cv2
import numpy as np

BORDER_FRACTION = 0.02
NEUTRAL = (180, 180, 180)            # RGB, only when the main photo's border is fully transparent
CROP_W = 2.4                         # close-up crop width in face-box widths
CROP_ABOVE = 0.6                     # hair above the face box, in face-box heights
CROP_BELOW = 1.3                     # neck and shoulders below the face box, in face-box heights
FALLBACK_TALL = 1.25                 # no face found: crop height = 1.25 x crop width, from the top centre
GAP_FRACTION = 0.02
GAP_MIN = 4
MIN_SIDE = 64


def _decode(data):
    """Bytes -> (RGB uint8 array, alpha float array 0..1 or None), EXIF orientation applied."""
    from PIL import Image, ImageOps
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception:
        raise ValueError("A collage photo could not be read. Use png, jpg or webp.") from None
    im = ImageOps.exif_transpose(im)
    has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
    if has_alpha:
        rgba = np.asarray(im.convert("RGBA"))
        return np.ascontiguousarray(rgba[..., :3]), rgba[..., 3].astype(np.float32) / 255.0
    return np.ascontiguousarray(np.asarray(im.convert("RGB"))), None


def background_colour(rgb, alpha=None):
    """Per-channel median of the border strip (only at least half-opaque pixels), as an RGB tuple of ints."""
    h, w = rgb.shape[:2]
    b = max(1, int(round(BORDER_FRACTION * min(h, w))))
    mask = np.zeros((h, w), bool)
    mask[:b, :] = mask[-b:, :] = True
    mask[:, :b] = mask[:, -b:] = True
    if alpha is not None:
        mask &= alpha >= 0.5
    if not mask.any():
        return NEUTRAL
    px = rgb[mask]
    return tuple(int(v) for v in np.median(px, axis=0).round())


def _flatten(rgb, alpha, bg):
    if alpha is None:
        return rgb
    a = alpha[..., None]
    out = rgb.astype(np.float32) * a + np.asarray(bg, np.float32) * (1.0 - a)
    return np.clip(out.round(), 0, 255).astype(np.uint8)


def _span(start, length, limit):
    """[start, start + length) shifted inside [0, limit) where it fits, then clipped. Returns ints (a, b)."""
    if length >= limit:
        return 0, limit
    start = min(max(0.0, start), limit - length)
    a = min(int(round(start)), limit - 1)
    return a, min(limit, a + max(1, int(round(length))))


def head_crop_box(shape, faces):
    """(x0, y0, x1, y1) of the head-and-shoulders crop for an image of `shape` from detector output `faces`."""
    h, w = shape[:2]
    faces = [f for f in (faces or []) if f and f.get("box") is not None and float(f["box"][2]) > 0]
    if not faces:
        cw = min(w, h)
        x0, x1 = _span((w - cw) / 2.0, cw, w)
        return x0, 0, x1, min(h, int(round(FALLBACK_TALL * cw)))
    fx, fy, fw, fh = (float(v) for v in max(faces, key=lambda f: float(f["box"][2]))["box"])
    fh = fh if fh > 0 else fw
    cw = CROP_W * fw
    x0, x1 = _span(fx + fw / 2.0 - cw / 2.0, cw, w)
    top = fy - CROP_ABOVE * fh
    y0, y1 = _span(top, fh * (1 + CROP_ABOVE + CROP_BELOW), h)
    return x0, y0, x1, y1


def layout(main_wh, face_wh, max_side=1536):
    """Sizes of the canvas: returns {"height", "main_w", "face_w", "gap", "width"}. Same height for both parts, aspect kept,
    longer canvas side <= max_side, the main photo never upscaled."""
    mw, mh = main_wh
    fw, fh = face_wh
    am, af = mw / mh, fw / fh
    gap = lambda H: max(GAP_MIN, int(round(GAP_FRACTION * H)))
    parts = lambda H: (max(1, int(round(H * am))), max(1, int(round(H * af))))
    H = int(min(mh, max_side, max_side / (am + af) + 1))        # no taller height can fit; the largest that fits wins
    while H > 1:
        a, b = parts(H)
        if max(H, a + gap(H) + b) <= max_side:
            break
        H -= 1
    a, b = parts(H)
    return {"height": H, "main_w": a, "face_w": b, "gap": gap(H), "width": a + gap(H) + b}


def _resize(rgb, w, h):
    interp = cv2.INTER_AREA if w < rgb.shape[1] or h < rgb.shape[0] else cv2.INTER_CUBIC
    return cv2.resize(rgb, (w, h), interpolation=interp)


def make_collage(main_bytes, face_bytes, *, max_side=1536, detector=None):
    """The collage reference as PNG bytes (see the module docstring). `detector` takes a BGR image and returns picker-style
    faces ([{"box": (x, y, w, h), ...}]); default picker.yunet_faces. Raises ValueError for an unreadable photo."""
    if int(max_side) < MIN_SIDE:
        raise ValueError(f"max_side must be at least {MIN_SIDE}.")
    if detector is None:
        import picker
        detector = picker.yunet_faces
    main, main_a = _decode(main_bytes)
    face, face_a = _decode(face_bytes)
    bg = background_colour(main, main_a)
    main = _flatten(main, main_a, bg)
    face = _flatten(face, face_a, bg)
    try:
        faces = list(detector(np.ascontiguousarray(face[..., ::-1])) or [])
    except Exception:
        faces = []
    x0, y0, x1, y1 = head_crop_box(face.shape, faces)
    face = face[y0:y1, x0:x1]
    L = layout((main.shape[1], main.shape[0]), (face.shape[1], face.shape[0]), int(max_side))
    H = L["height"]
    canvas = np.empty((H, L["width"], 3), np.uint8)
    canvas[:] = bg
    canvas[:, :L["main_w"]] = _resize(main, L["main_w"], H)
    canvas[:, L["main_w"] + L["gap"]:] = _resize(face, L["face_w"], H)
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(canvas[..., ::-1]))
    if not ok:
        raise ValueError("The collage could not be encoded.")
    return buf.tobytes()
