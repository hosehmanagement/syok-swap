"""Best-photo picker for face-lock: the user may add up to 5 candidate photos of the new person (full body, side profile,
close-up...); Magic Hour's face swap takes ONE face image, so this picks the photo whose face it can copy best. The main reference
photo (ref.<ext>) is always one more candidate, so one added photo is simply a 2-way pick: that photo vs the main photo.

No Modal here and the face detector is injectable, like the rest of the job service, so every rule is unit-tested.

A detector takes a BGR image and returns a list of faces, each a dict:
    {"box": (x, y, w, h)  in that image's pixels,
     "landmarks": [(x, y) x 5] or None   YuNet order: right eye, left eye, nose tip, right mouth corner, left mouth corner
                                         (the person's right, so the "right eye" is on the image's left)
     "front": 0..1, optional              used only when there are no landmarks (e.g. a Haar frontal / profile hit)}
The default, `yunet_faces`, runs OpenCV's YuNet (face_detection_yunet_2023mar.onnx, MIT, 233 KB) and falls back to the Haar
cascades of texture_transfer.detect_face (frontal = 1.0, profile = 0.3) when the model file or cv2.FaceDetectorYN is missing.
`from_box_detector` turns the swap service's (box, weight) detector into this shape (the tests' fake detectors).

Per candidate, from its LARGEST face (by box width):
    face_px  face box width in the photo's own pixels.
    sharp    R = var(Laplacian(g)) / max(var(g), 100), where g is the grey face crop resized to 128 x 128 (so a big and a
             small crop compare on the same grid) and dividing by the crop's own variance cancels its contrast and lighting.
             The floor of 100 (grey levels squared, a standard deviation of 10) stops a nearly flat crop (a smeared face,
             a blank wall) from looking sharp because of its last bits of noise; real face crops measure 2000-4000.
             sharp = min(1, R / 0.15). Measured on real portraits: in focus R = 0.13-0.29, Gaussian blur sigma 3 -> 0.01-0.02.
    front    from the 5 landmarks, measured along the eye line u (unit vector right eye -> left eye) with d = eye distance:
                 nose_off  = |(nose - eye_mid) . u| / (d/2)       0 = nose centred between the eyes, >= 1 = beyond an eye
                 mouth_off = |(mouth_mid - eye_mid) . u| / (d/2)
                 roll      = angle of the eye line from horizontal, in degrees (0..90)
                 front = (0.6 * clamp01(1 - nose_off) + 0.4 * clamp01(1 - mouth_off)) * (0.8 + 0.2 * clamp01(1 - roll/45))
             1 = straight on. Below PROFILE_FRONT (0.35) the photo counts as a profile.
    ok       exposure: mean luma (0..255) of the face crop within [60, 200].
    multi    another face at least 40% as wide as the largest is in the photo (which person Magic Hour copies is then
             ambiguous). Tiny faces far in the background do not count.

Score (all factors in 0..1, so the score is in 0..1):
    score = S^1.5 * front * (0.4 + 0.6 * sharp) * E * M
        S = min(face_px, 512) / 512           size dominates (Magic Hour asks for a face of at least 512 x 512 px); no gain past 512
        E = 1 inside [60, 200], else max(0.5, 1 - distance_outside / 100)
        M = 0.6 when multi, else 1
    The winner is the highest by the key (front >= 0.35, round(score, 4), face_px, -input_index): a profile can never beat a
    photo that is not a profile, whatever their sizes; equal scores go to the larger face, then to the earlier photo.
    Photos with no face are never chosen. When no photo has a face, the main photo is used (nothing is blocked).
"""
from pathlib import Path

import cv2
import numpy as np

MAX_CANDIDATES = 5                 # face photos the user may add (the main photo comes on top)
MAX_TOTAL_BYTES = 12 * 2**20       # all added face photos together
SIZE_CAP_PX = 512
PROFILE_FRONT = 0.35
SHARP_REF = 0.15
VAR_FLOOR = 100.0
LUMA_LO, LUMA_HI = 60.0, 200.0
MULTI_PENALTY = 0.6
MULTI_MIN_FRACTION = 0.4
MIN_FACE_PX = 256                  # same threshold as swap_service.FACELOCK_MIN_FACE_PX
GRID = 128

YUNET_NAME = "face_detection_yunet_2023mar.onnx"


def _find_yunet():
    here = Path(__file__).resolve().parent
    for p in (here / YUNET_NAME, here / "lora" / YUNET_NAME):
        if p.exists():
            return p
    return None


def _yunet_once(model, bgr, k):
    img = bgr if k == 1 else cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(str(model), "", (w, h), 0.5, 0.3, 5000)
    _, rows = det.detect(img)
    if rows is None:
        return []
    out = []
    for r in rows:
        r = [float(v) / k for v in r[:14]]
        out.append({"box": tuple(r[:4]), "landmarks": [(r[4 + 2 * i], r[5 + 2 * i]) for i in range(5)]})
    return out


def yunet_faces(bgr):
    """All faces YuNet finds. First on a copy whose longest side is at most 640 px (keeps small faces in full-body shots);
    if none, on a copy with a 320 px short side (YuNet misses faces that fill a big frame). Coordinates are in `bgr` pixels."""
    model = _find_yunet()
    if model is None or not hasattr(cv2, "FaceDetectorYN"):
        return _haar_faces(bgr)
    h, w = bgr.shape[:2]
    k1 = min(1.0, 640.0 / max(h, w))
    faces = _yunet_once(model, bgr, k1)
    k2 = 320.0 / min(h, w)
    if not faces and abs(k2 - k1) > 1e-6:
        faces = _yunet_once(model, bgr, k2)
    return faces


def _haar_faces(bgr):
    import texture_transfer as tt
    return from_box_detector(tt.detect_face)(bgr)


def from_box_detector(detector):
    """Adapt a (box, weight) detector (texture_transfer.detect_face or a test fake) to this module's shape: one face at most,
    no landmarks; weight 1.0 (frontal) -> front 1.0, anything lower (profile) -> 0.3."""
    def faces(bgr):
        box, weight = detector(bgr)
        if box is None:
            return []
        return [{"box": tuple(float(v) for v in box), "landmarks": None, "front": 1.0 if float(weight) >= 1.0 else 0.3}]
    return faces


# ------------------------------------------------------------------ the measures

def _clamp01(v):
    return float(min(1.0, max(0.0, v)))


def frontality(landmarks):
    """0..1 from the 5 landmarks (formula in the module docstring); 1 = straight on."""
    p = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    re, le, nose, mr, ml = p
    v = le - re
    d = float(np.hypot(*v))
    if d < 1.0:
        return 0.0
    u = v / d
    mid = (re + le) / 2
    nose_off = abs(float(np.dot(nose - mid, u))) / (d / 2)
    mouth_off = abs(float(np.dot((mr + ml) / 2 - mid, u))) / (d / 2)
    roll = abs(float(np.degrees(np.arctan2(v[1], v[0]))))
    roll = min(roll, 180.0 - roll)
    return round((0.6 * _clamp01(1 - nose_off) + 0.4 * _clamp01(1 - mouth_off)) * (0.8 + 0.2 * _clamp01(1 - roll / 45.0)), 4)


def _crop(bgr, box):
    x, y, w, h = box
    H, W = bgr.shape[:2]
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(W, int(round(x + w))), min(H, int(round(y + h)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return bgr[y0:y1, x0:x1]


def sharpness(crop):
    """0..1: var(Laplacian) / max(var(grey), 100) on the crop resized to 128 x 128, divided by 0.15 and capped at 1."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    interp = cv2.INTER_AREA if min(g.shape) >= GRID else cv2.INTER_LINEAR
    g = cv2.resize(g, (GRID, GRID), interpolation=interp).astype(np.float64)
    r = float(cv2.Laplacian(g, cv2.CV_64F).var()) / max(float(g.var()), VAR_FLOOR)
    return round(min(1.0, r / SHARP_REF), 4)


def luma(crop):
    return float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())


def exposure_factor(y):
    if LUMA_LO <= y <= LUMA_HI:
        return 1.0
    dist = (LUMA_LO - y) if y < LUMA_LO else (y - LUMA_HI)
    return max(0.5, 1.0 - dist / 100.0)


def score_parts(face_px, front, sharp, exp_factor, multi):
    s = min(float(face_px), SIZE_CAP_PX) / SIZE_CAP_PX
    return round(s ** 1.5 * front * (0.4 + 0.6 * sharp) * exp_factor * (MULTI_PENALTY if multi else 1.0), 4)


def measure(bgr, detector=yunet_faces):
    """Everything the score needs for one decoded photo. `face` False when no face was found (or the photo is unreadable)."""
    if bgr is None:
        return {"face": False, "face_px": 0, "front": 0.0, "sharp": 0.0, "ok": False, "multi": False, "score": 0.0,
                "note": "could not be read"}
    try:
        faces = list(detector(bgr) or [])
    except Exception:
        faces = []
    faces = [f for f in faces if f and f.get("box") is not None and float(f["box"][2]) > 0]
    if not faces:
        return {"face": False, "face_px": 0, "front": 0.0, "sharp": 0.0, "ok": False, "multi": False, "score": 0.0,
                "note": "no face found"}
    big = max(faces, key=lambda f: float(f["box"][2]))
    width = float(big["box"][2])
    others = sum(1 for f in faces if f is not big and float(f["box"][2]) >= MULTI_MIN_FRACTION * width)
    front = frontality(big["landmarks"]) if big.get("landmarks") is not None else round(_clamp01(big.get("front", 1.0)), 4)
    crop = _crop(bgr, big["box"])
    sharp = sharpness(crop) if crop is not None else 0.0
    y = luma(crop) if crop is not None else 0.0
    ok = LUMA_LO <= y <= LUMA_HI
    px = int(round(width))
    score = score_parts(px, front, sharp, exposure_factor(y), others > 0)
    return {"face": True, "face_px": px, "front": front, "sharp": sharp, "ok": ok, "multi": others > 0, "score": score,
            "note": _note(px, front, sharp, ok, y, others)}


def _note(px, front, sharp, ok, y, others):
    bits = [("front-facing" if front >= 0.8 else "slightly turned" if front >= 0.6 else "turned"
             if front >= PROFILE_FRONT else "side profile"), f"about {px} px wide"]
    if sharp < 0.4:
        bits.append("blurry")
    if not ok:
        bits.append("too dark" if y < LUMA_LO else "too bright")
    if others:
        bits.append("more than one face")
    return ", ".join(bits)


def _key(i, m):
    return (m["face"], m["front"] >= PROFILE_FRONT, m["score"], m["face_px"], -i)


def pick(images, detector=yunet_faces, labels=None):
    """`images`: decoded BGR images (None = unreadable), in input order. Returns
        {"chosen": 1-based index or None when no image has a face, "of": count, "reason": one sentence,
         "scores": [{"n", "face_px", "front", "sharp", "ok", "note", "score"}], "all_small": bool}
    `labels` optional per-image names for the reason ("your main photo"); default "photo N"."""
    ms = [measure(img, detector) for img in images]
    n = len(ms)
    labels = labels or [f"photo {i + 1}" for i in range(n)]
    scores = [{"n": i + 1, "face_px": m["face_px"], "front": m["front"], "sharp": m["sharp"], "ok": m["ok"],
               "note": m["note"], "score": m["score"]} for i, m in enumerate(ms)]
    found = [i for i, m in enumerate(ms) if m["face"]]
    if not found:
        return {"chosen": None, "of": n, "scores": scores, "all_small": False,
                "reason": f"No face was found in any of the {n} photos, so your main photo is used."}
    best = max(found, key=lambda i: _key(i, ms[i]))
    b = ms[best]
    desc = ("front-facing" if b["front"] >= 0.8 else "the most front-facing" if b["front"] >= PROFILE_FRONT
            else "a side profile, the best of the photos with a face")
    extra = "" if b["sharp"] >= 0.4 else ", though a little blurry"
    reason = f"Using {labels[best]}: {desc} and about {b['face_px']} px wide{extra}."
    return {"chosen": best + 1, "of": n, "scores": scores, "reason": reason[0].upper() + reason[1:],
            "all_small": all(ms[i]["face_px"] < MIN_FACE_PX for i in found)}


def read_image(path):
    """Decode a photo file (png/jpg/webp) to BGR, or None."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except Exception:
        return None
