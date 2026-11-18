"""Consent-gated dataset prep for a Wan 2.2 identity/skin-detail LoRA. CPU only, no GPU spend.

A LoRA bakes one person's face into model weights, so consent is checked for the WHOLE image set:
the record binds to a digest of every image's SHA-256 (add, swap or remove one photo and it no longer verifies).
Quality gates keep the set honest: sharp, unique, big enough, contains a face. Captions are explicit about skin detail.
"""
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

import consent

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_KEPT = 12          # identity LoRAs are unreliable below this
MIN_SIDE = 640
MIN_SHARP = 60.0       # variance of Laplacian on a 512px version; below this the photo is soft
DUP_HAMMING = 5        # average-hash bits that may differ before two photos count as duplicates
MIN_FACE_FRAC = 0.18   # face width / short side; pores need a close-up
DEFAULT_CAPTION = ("{t}, close-up photo of {t}, sharp focus, natural skin with visible pores, "
                   "fine skin texture and stubble, no skin smoothing")


def list_images(src):
    return sorted(p for p in Path(src).iterdir() if p.suffix.lower() in EXTS and p.is_file())


def dataset_digest(paths):
    """One hash for the whole set (order independent)."""
    hs = sorted(consent.sha256_file(p) for p in paths)
    return hashlib.sha256("\n".join(hs).encode()).hexdigest()


def make_dataset_record(paths, kind, attest=None, who="", contact="", valid_days=consent.DEFAULT_VALID_DAYS, now=None):
    """Same rules as consent.make_record, but bound to the set instead of one image."""
    if kind not in ("fictional", "real"):
        raise ValueError("kind must be 'fictional' or 'real'.")
    attest = dict(attest or {})
    if kind == "real":
        missing = [k for k in consent.REQUIRED_FOR_REAL if attest.get(k) is not True]
        if missing:
            raise PermissionError("Consent incomplete, not allowed to proceed: " + ", ".join(missing))
        if not who.strip() or not contact.strip():
            raise PermissionError("Consent needs the name of the person and a contact for them.")
    now = time.time() if now is None else now
    return {"kind": kind, "dataset_sha256": dataset_digest(paths), "n_images": len(list(paths)), "attest": attest,
            "who": who, "contact": contact, "created": now, "expires": now + valid_days * 86400}


def verify_dataset_record(record, paths, now=None):
    now = time.time() if now is None else now
    if record.get("dataset_sha256") != dataset_digest(paths):
        raise PermissionError("Consent record does not match this exact set of images.")
    if now > record.get("expires", 0):
        raise PermissionError("Consent has expired; ask the person again.")
    if record.get("kind") == "real":
        missing = [k for k in consent.REQUIRED_FOR_REAL if record.get("attest", {}).get(k) is not True]
        if missing:
            raise PermissionError("Consent incomplete: " + ", ".join(missing))
    return True


def sharpness(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = 512.0 / max(g.shape)
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def ahash(bgr, n=8):
    g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (n, n), interpolation=cv2.INTER_AREA)
    return (g > g.mean()).flatten()


YUNET = Path(__file__).with_name("face_detection_yunet_2023mar.onnx")  # opencv_zoo YuNet, MIT licence, 233 KB


def _yunet_fraction(bgr):
    """Face width / short side from YuNet on a 320-px-short-side copy (it misses faces that fill a big frame). None if unavailable."""
    if not YUNET.exists() or not hasattr(cv2, "FaceDetectorYN"):
        return None
    k = 320 / min(bgr.shape[:2])
    small = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    det = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), 0.5, 0.3, 5000)
    _, faces = det.detect(small)
    return 0.0 if faces is None else max(float(f[2]) for f in faces) / min(h, w)


def face_fraction(bgr):
    """Largest face width as a fraction of the short side (0 if none).

    YuNet first: it handles 3/4 and turned heads, which the Haar cascades miss (found 2026-11-18: Haar scored 0.0 on
    plainly visible 3/4 portraits). Haar frontal+profile (also mirrored) is the fallback if the model file is absent."""
    y = _yunet_fraction(bgr)
    if y is not None:
        return y
    short0 = min(bgr.shape[:2])
    if short0 > 512:
        k = 512 / short0
        bgr = cv2.resize(bgr, (round(bgr.shape[1] * k), round(bgr.shape[0] * k)), interpolation=cv2.INTER_AREA)
    g = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    short = min(g.shape)
    ms = (max(24, short // 12),) * 2
    best = 0
    for name in ("haarcascade_frontalface_alt2.xml", "haarcascade_profileface.xml"):
        c = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if c.empty():
            raise RuntimeError(f"OpenCV face cascade missing ({name}); install opencv-python-headless<5")
        for img in (g, cv2.flip(g, 1)):
            for (x, y2, w, h) in c.detectMultiScale(img, 1.1, 5, minSize=ms):
                best = max(best, w)
    return best / short


def screen(paths, face_fn=None):
    """Return (kept, rejected) where rejected is [(path, reason)]. Never raises on a bad file."""
    face_fn = face_fn or face_fraction
    kept, rejected, hashes = [], [], []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            rejected.append((p, "unreadable"))
            continue
        if min(img.shape[:2]) < MIN_SIDE:
            rejected.append((p, f"too small (short side {min(img.shape[:2])} < {MIN_SIDE})"))
            continue
        sh = sharpness(img)
        if sh < MIN_SHARP:
            rejected.append((p, f"blurry (sharpness {sh:.0f} < {MIN_SHARP:.0f})"))
            continue
        ff = face_fn(img)
        if ff < MIN_FACE_FRAC:
            rejected.append((p, "no close face found" if ff == 0 else f"face too small ({ff:.2f} of frame)"))
            continue
        h = ahash(img)
        if any(int((h != o).sum()) <= DUP_HAMMING for o in hashes):
            rejected.append((p, "near-duplicate of an earlier photo"))
            continue
        hashes.append(h)
        kept.append(p)
    return kept, rejected


def read_captions(csv_path):
    """Optional captions.csv with columns file,caption."""
    if not csv_path or not Path(csv_path).exists():
        return {}
    with open(csv_path, newline="") as f:
        return {r["file"]: r["caption"] for r in csv.DictReader(f)}


def toml_text(images_dir, cache_dir, resolution=768, repeats=1):
    return (f"[general]\nresolution = [{resolution}, {resolution}]\ncaption_extension = \".txt\"\nbatch_size = 1\n"
            f"enable_bucket = true\nbucket_no_upscale = false\n\n[[datasets]]\nimage_directory = \"{images_dir}\"\n"
            f"cache_directory = \"{cache_dir}\"\nnum_repeats = {repeats}\n")


def prepare(src_dir, out_dir, consent_path, trigger="sksperson", resolution=768, captions_csv=None, cache_dir=None, now=None, face_fn=None):
    """Verify consent for the WHOLE set, screen quality, write images+captions+dataset.toml+manifest.json."""
    all_paths = list_images(src_dir)
    if not all_paths:
        raise ValueError("No images found.")
    verify_dataset_record(consent.load(consent_path), all_paths, now=now)   # consent covers every file in the folder
    kept, rejected = screen(all_paths, face_fn)
    if len(kept) < MIN_KEPT:
        raise ValueError(f"Only {len(kept)} usable photos (need {MIN_KEPT}+). Rejected: "
                         + "; ".join(f"{p.name}: {r}" for p, r in rejected))
    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    caps = read_captions(captions_csv)
    for p in kept:
        img = cv2.imread(str(p))
        s = resolution * 1.25 / min(img.shape[:2])
        if s < 1:                                       # never upscale; only cap very large photos
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        stem = p.stem
        cv2.imwrite(str(img_dir / f"{stem}.png"), img)
        cap = caps.get(p.name) or DEFAULT_CAPTION.format(t=trigger)
        if trigger not in cap:
            cap = f"{trigger}, {cap}"
        (img_dir / f"{stem}.txt").write_text(cap)
    (out / "dataset.toml").write_text(toml_text(str(img_dir), str(cache_dir or out / "cache"), resolution))
    manifest = {"trigger": trigger, "kept": [p.name for p in kept], "rejected": [[p.name, r] for p, r in rejected],
                "resolution": resolution, "dataset_sha256": dataset_digest(all_paths)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
