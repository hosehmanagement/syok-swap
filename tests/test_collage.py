"""EXPERIMENTAL collage reference (collage.py) and how the job service and the HTTP layer use it.

Synthetic images only (solid colours, drawn faces, generated skin texture); no photos of real people."""
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import collage
import facelock as fl
import picker
import swap_service as ss
import texture_service as svc
import texture_transfer as tt
from test_texture_service import FAKE, photo_bytes

pytest.importorskip("fastapi")
from test_facelock import KEY, UP_IMAGE, FaceRig, FakeMagicHour, mh_result
from test_picker import Scripted, drawn, face, needs_yunet, photo
from test_swap_service import clip

NO_FACE = lambda bgr: []
APP = (Path(__file__).resolve().parent.parent / "app.py").read_text()


def rgb_png(rgb):
    b = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(b, "PNG")
    return b.getvalue()


def rgba_png(rgba):
    b = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba), "RGBA").save(b, "PNG")
    return b.getvalue()


def solid(w, h, rgb):
    a = np.empty((h, w, 3), np.uint8)
    a[:] = rgb
    return a


def decode(png):
    im = Image.open(io.BytesIO(png))
    assert im.format == "PNG" and im.mode == "RGB"
    return np.asarray(im)


# ------------------------------------------------------------------ layout

def test_layout_same_height_aspect_kept_gap_and_exact_sizes():
    main = solid(400, 1000, (10, 200, 10))
    close = solid(600, 600, (200, 10, 10))
    out = decode(collage.make_collage(rgb_png(main), rgb_png(close), detector=NO_FACE))
    # H = main height (1000, fits); main 400 wide, gap round(0.02 * 1000) = 20, close-up (600 x 600 top-centre crop) 1000 wide
    assert out.shape == (1000, 400 + 20 + 1000, 3)
    assert (out[:, :400] == (10, 200, 10)).all()                          # the whole main photo, left
    assert (out[:, 420:] == (200, 10, 10)).all()                          # the close-up, right
    bg = np.array((10, 200, 10))                                          # the gap is the main photo's border colour
    assert (out[:, 400:420] == bg).all()


@pytest.mark.parametrize("mw,mh,fw,fh,max_side", [(400, 1000, 600, 600, 1536), (3000, 4000, 900, 1200, 1536),
                                                  (3000, 4000, 900, 1200, 512), (300, 3000, 100, 100, 1536),
                                                  (4000, 1000, 500, 700, 1536), (999, 1333, 777, 555, 700)])
def test_layout_fits_max_side_and_keeps_both_aspect_ratios(mw, mh, fw, fh, max_side):
    L = collage.layout((mw, mh), (fw, fh), max_side)
    H = L["height"]
    assert max(H, L["width"]) <= max_side and L["width"] == L["main_w"] + L["gap"] + L["face_w"]
    assert H <= mh                                                        # the main photo is never upscaled
    assert abs(L["main_w"] - H * mw / mh) <= 0.5 and abs(L["face_w"] - H * fw / fh) <= 0.5     # rounding only, no stretch
    assert L["gap"] == max(collage.GAP_MIN, round(collage.GAP_FRACTION * H))
    # as large as possible: one pixel taller would break the limit (or the main photo's own height)
    if H < mh:
        H2 = H + 1
        w2 = round(H2 * mw / mh) + max(collage.GAP_MIN, round(collage.GAP_FRACTION * H2)) + round(H2 * fw / fh)
        assert max(H2, w2) > max_side


def test_max_side_is_respected_on_a_real_collage():
    main = np.random.default_rng(1).integers(0, 255, (4000, 3000, 3), dtype=np.uint8)
    close = np.random.default_rng(2).integers(0, 255, (1200, 900, 3), dtype=np.uint8)
    for side in (1536, 512, 100):
        out = decode(collage.make_collage(rgb_png(main), rgb_png(close), max_side=side, detector=NO_FACE))
        assert max(out.shape[:2]) <= side
        L = collage.layout((3000, 4000), (900, 900 * 5 // 4 if 900 * 5 // 4 <= 1200 else 1200), side)
        assert out.shape[:2] == (L["height"], L["width"])
    with pytest.raises(ValueError):
        collage.make_collage(rgb_png(main), rgb_png(close), max_side=10, detector=NO_FACE)


def test_a_small_main_photo_is_not_upscaled_but_the_close_up_is_scaled_to_its_height():
    out = decode(collage.make_collage(rgb_png(solid(200, 300, (0, 0, 255))), rgb_png(solid(64, 80, (255, 0, 0))),
                                      detector=NO_FACE))
    # close-up: 64 x 80 -> top-centre crop 64 x 80 (1.25 x 64 = 80) -> 240 x 300
    assert out.shape == (300, 200 + 6 + 240, 3)


# ------------------------------------------------------------------ the close-up crop

def test_head_and_shoulders_crop_from_the_largest_face_with_margin():
    faces = [{"box": (100, 100, 50, 60)}, {"box": (400, 300, 200, 240)}]
    x0, y0, x1, y1 = collage.head_crop_box((1200, 1000, 3), faces)
    # largest face: centre x 500, width 2.4 x 200 = 480 -> 260..740; from 300 - 0.6 x 240 = 156 to 300 + 240 + 1.3 x 240 = 852
    assert (x0, y0, x1, y1) == (260, 156, 740, 852)


def test_a_crop_near_the_edge_is_shifted_inside_then_clipped():
    assert collage.head_crop_box((1000, 1000, 3), [{"box": (0, 0, 200, 200)}]) == (0, 0, 480, 580)
    assert collage.head_crop_box((1000, 1000, 3), [{"box": (850, 850, 150, 150)}]) == (640, 565, 1000, 1000)
    # face wider than the photo allows: the crop is the whole width
    assert collage.head_crop_box((400, 300, 3), [{"box": (50, 50, 200, 200)}])[0::2] == (0, 300)


def test_no_face_falls_back_to_a_centre_top_crop():
    assert collage.head_crop_box((900, 600, 3), []) == (0, 0, 600, 750)            # tall: full width, 1.25 x width high
    assert collage.head_crop_box((600, 900, 3), None) == (150, 0, 750, 600)        # wide: a centred square-ish top crop
    crash = lambda bgr: 1 / 0
    a = collage.make_collage(rgb_png(solid(100, 200, (0, 0, 0))), rgb_png(solid(900, 600, (9, 9, 9))), detector=crash)
    b = collage.make_collage(rgb_png(solid(100, 200, (0, 0, 0))), rgb_png(solid(900, 600, (9, 9, 9))), detector=NO_FACE)
    assert a == b                                                                   # a crashing detector = no face


def test_the_crop_really_is_the_face_region():
    img = solid(600, 800, (0, 0, 200))
    img[300:400, 250:350] = (0, 220, 0)                                             # the "face" box: 100 x 100 at (250, 300)
    seen = []
    det = lambda bgr: seen.append(bgr.shape) or [{"box": (250, 300, 100, 100)}]
    out = decode(collage.make_collage(rgb_png(solid(300, 600, (50, 50, 50))), rgb_png(img), detector=det))
    assert seen == [(800, 600, 3)]                                                  # detector gets the BGR close-up
    # crop 240 x 290 (x 180..420, y 240..530), scaled to 600 high -> 497 wide; the face is x 70..170 / y 60..160 of the crop
    right = out[:, 300 + 12:]
    assert right.shape[:2] == (600, round(600 * 240 / 290))
    k = 600 / 290
    cy, cx = int((300 - 240 + 50) * k), int((250 - 180 + 50) * k)
    assert tuple(right[cy, cx]) == (0, 220, 0)
    assert tuple(right[5, 5]) == (0, 0, 200)


@needs_yunet
def test_real_yunet_crops_around_a_drawn_face():
    img = np.full((900, 700, 3), (200, 210, 220), np.uint8)
    img[100:500, 150:550] = drawn(fw=200, size=400)                                # BGR drawn face, centre (350, 300)
    faces = picker.yunet_faces(img)
    x0, y0, x1, y1 = collage.head_crop_box(img.shape, faces)
    assert abs((x0 + x1) / 2 - 350) < 25 and y0 < 200 and y1 > 450 and 400 < x1 - x0 < 620
    main = rgb_png(solid(300, 600, (90, 90, 90)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert collage.make_collage(main, rgb_png(rgb)) == collage.make_collage(main, rgb_png(rgb))


# ------------------------------------------------------------------ decoding: RGBA, EXIF, background, errors

def test_rgba_inputs_are_flattened_onto_the_background_from_the_main_photos_border():
    main = np.zeros((300, 200, 4), np.uint8)
    main[..., :3] = (30, 60, 90)
    main[..., 3] = 255
    main[100:200, 50:150, 3] = 0                                                    # transparent hole in the middle
    close = np.zeros((100, 100, 4), np.uint8)
    close[40:60, 40:60] = (250, 0, 0, 255)                                          # only a red square is opaque
    out = decode(collage.make_collage(rgba_png(main), rgba_png(close), detector=NO_FACE))
    assert tuple(out[150, 100]) == (30, 60, 90)                                     # hole -> border colour
    right = out[:, 200 + 6:]
    assert tuple(right[0, 0]) == (30, 60, 90) and tuple(right[right.shape[0] // 2, right.shape[1] // 2]) == (250, 0, 0)


def test_a_fully_transparent_border_uses_the_neutral_colour_and_half_alpha_blends():
    main = np.zeros((200, 200, 4), np.uint8)
    main[50:150, 50:150] = (0, 0, 0, 255)
    out = decode(collage.make_collage(rgba_png(main), rgb_png(solid(50, 50, (1, 2, 3))), detector=NO_FACE))
    assert tuple(out[0, 0]) == collage.NEUTRAL and tuple(out[100, 100]) == (0, 0, 0)
    half = np.zeros((10, 10, 4), np.uint8)
    half[...] = (200, 100, 0, 128)
    rgb, a = collage._decode(rgba_png(half))
    flat = collage._flatten(rgb, a, (0, 100, 200))
    exp = np.round(np.array((200, 100, 0)) * (128 / 255) + np.array((0, 100, 200)) * (1 - 128 / 255))
    assert (flat[0, 0] == exp).all()


def test_background_is_the_median_of_the_border_strip():
    img = solid(100, 100, (200, 200, 200))
    img[0, :30] = (0, 0, 0)                                                         # a minority of dark border pixels
    img[20:80, 20:80] = (0, 0, 255)                                                 # the middle does not count
    assert collage.background_colour(img) == (200, 200, 200)


def exif_jpeg(rgb, orientation):
    b = io.BytesIO()
    ex = Image.Exif()
    ex[0x0112] = orientation
    Image.fromarray(rgb).save(b, "JPEG", quality=95, exif=ex.tobytes())
    return b.getvalue()


def test_exif_orientation_is_applied_to_both_photos():
    stored = solid(300, 200, (0, 0, 255))                                           # 300 wide x 200 high as stored
    stored[:60, :60] = (255, 0, 0)                                                  # red block top-left as stored
    seen = []
    det = lambda bgr: seen.append(bgr.shape) or []
    out = decode(collage.make_collage(exif_jpeg(stored, 6), exif_jpeg(stored, 6), detector=det))
    # orientation 6 = rotate 90 degrees clockwise to view: 200 wide x 300 high, the red block moves to the top-right
    assert seen == [(300, 200, 3)]
    assert out.shape[0] == 300 and out[:, :200].shape == (300, 200, 3)
    r, g, b = (int(v) for v in out[20, 180])
    assert r > 200 and b < 60
    r, g, b = (int(v) for v in out[20, 20])
    assert b > 200 and r < 60


def test_the_same_inputs_give_the_same_bytes_and_unreadable_bytes_are_refused():
    rng = np.random.default_rng(7)
    main = rng.integers(0, 255, (777, 333, 3), dtype=np.uint8)
    close = rng.integers(0, 255, (640, 480, 3), dtype=np.uint8)
    det = lambda bgr: [{"box": (100.4, 90.6, 120.2, 140.9)}]
    a = collage.make_collage(rgb_png(main), rgb_png(close), detector=det)
    assert a == collage.make_collage(rgb_png(main), rgb_png(close), detector=det)
    assert a[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError):
        collage.make_collage(b"not an image", rgb_png(close), detector=det)
    with pytest.raises(ValueError):
        collage.make_collage(rgb_png(main), b"", detector=det)


# ------------------------------------------------------------------ the job service and the HTTP layer

def files(tmp, faces=(), main=None):
    fs = [("video", ("v.mp4", clip(tmp), "video/mp4")), ("photo", ("p.png", main or photo_bytes(), "image/png"))]
    return fs + [("face", f) for f in faces]


def meta_of(r, job):
    return json.loads((r.root / job / "meta.json").read_text())


def comparable(meta):
    """meta.json without what differs between any two jobs (id, times)."""
    m = json.loads(json.dumps(meta))
    m.pop("job"); m.pop("created")
    for v in m.values():
        if isinstance(v, dict):
            v.pop("at", None)
    return m


@pytest.mark.parametrize("value", ["yes", "1", "true", "TRUE", " Yes ", "True"])
def test_collage_on_stores_the_collage_and_keeps_the_main_photo(tmp_path, value):
    r = FaceRig(tmp_path)
    close = photo_bytes(seed=4)
    res = r.c.post("/swap", files=files(tmp_path, [("c.png", close, "image/png")]), data={"consent": "yes", "collage": value})
    assert res.status_code == 200, res.text
    job = res.json()["job"]
    d = r.root / job
    meta = meta_of(r, job)
    assert meta["request"]["collage"] is True and "facelock" not in meta["request"]
    assert meta["ref"] == "ref.png" and (d / "ref.png").read_bytes() == photo_bytes()        # the original, untouched
    assert meta["collage"] == {"file": "ref_collage.png", "face": 1, "of": 1}
    exp = collage.make_collage(photo_bytes(), close, detector=picker.from_box_detector(FAKE))
    assert (d / "ref_collage.png").read_bytes() == exp
    st = r.c.get(f"/status/{job}").json()
    assert st["request"]["collage"] is True and st["collage"] == {"face": 1, "of": 1}
    assert r.calls["mask"] == 1


@pytest.mark.parametrize("value", [None, "", "no", "0", "false", "on", "y", "collage"])
def test_collage_off_leaves_the_job_exactly_as_before(tmp_path, value):
    r = FaceRig(tmp_path)
    close = [("c.png", photo_bytes(seed=4), "image/png")]
    base = r.c.post("/swap", files=files(tmp_path, close), data={"consent": "yes"}).json()["job"]
    data = {"consent": "yes"} if value is None else {"consent": "yes", "collage": value}
    job = r.c.post("/swap", files=files(tmp_path, close), data=data).json()["job"]
    meta = meta_of(r, job)
    assert comparable(meta) == comparable(meta_of(r, base))
    assert "collage" not in meta["request"] and "collage" not in meta
    assert not (r.root / job / "ref_collage.png").exists()
    assert "collage" not in r.c.get(f"/status/{job}").json()
    assert ss.request_of(r.root, job)["ref"] == "ref.png"
    # and without any face photo, an off collage is not an error either
    assert r.c.post("/swap", files=files(tmp_path), data=data).status_code == 200


def test_meta_json_of_a_job_without_collage_is_byte_for_byte_the_old_format(tmp_path, monkeypatch):
    """Same inputs, same clock, same id: the bytes of meta.json are equal with the collage argument off and absent."""
    import time
    import uuid
    monkeypatch.setattr(time, "time", lambda: 1234.5)
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID("12345678123456781234567812345678"))
    blobs = []
    for i, kw in enumerate(({}, {"collage": False})):
        root = tmp_path / f"r{i}"; root.mkdir()
        job = ss.new_swap_job(root, clip(tmp_path), photo_bytes(), ".png", "yes", detector=FAKE,
                              faces=[(photo_bytes(seed=4), ".png")], **kw)
        blobs.append((root / job / "meta.json").read_bytes())
        assert sorted(p.name for p in (root / job).iterdir()) == ["face.png", "meta.json", "raw.mp4", "ref.png"]
    assert blobs[0] == blobs[1] and b"collage" not in blobs[0]


def test_collage_without_a_face_close_up_is_refused_and_nothing_is_stored(tmp_path):
    r = FaceRig(tmp_path)
    for fs in (files(tmp_path), files(tmp_path, [("c.png", b"", "image/png")])):          # none, or an empty file part
        res = r.c.post("/swap", files=fs, data={"consent": "yes", "collage": "yes", "facelock": "yes"})
        assert res.status_code == 422 and res.json() == {"error": "Collage needs a face close-up."}
    assert list(r.root.iterdir()) == [] and r.calls["mask"] == 0
    with pytest.raises(svc.JobError) as e:
        ss.new_swap_job(r.root, clip(tmp_path), photo_bytes(), ".png", "yes", detector=FAKE, collage=True)
    assert e.value.status == 422 and str(e.value) == "Collage needs a face close-up."
    assert list(r.root.iterdir()) == []


def test_consent_is_still_required_with_collage(tmp_path):
    r = FaceRig(tmp_path)
    res = r.c.post("/swap", files=files(tmp_path, [("c.png", photo_bytes(seed=4), "image/png")]),
                   data={"consent": "", "collage": "yes"})
    assert res.status_code == 400 and "agreed" in res.json()["error"]
    assert list(r.root.iterdir()) == [] and r.calls["mask"] == 0


def test_a_close_up_that_cannot_be_decoded_is_refused_and_nothing_is_kept(tmp_path):
    r = FaceRig(tmp_path)
    res = r.c.post("/swap", files=files(tmp_path, [("c.png", b"\x89PNG broken", "image/png")]),
                   data={"consent": "yes", "collage": "yes"})
    assert res.status_code == 400 and res.json()["error"].startswith("The collage could not be made:")
    assert list(r.root.iterdir()) == [] and r.calls["mask"] == 0


SCRIPT = {11: [face(100, 100, 120)], 12: [face(50, 50, 420)], 15: [face(60, 60, 200)], 16: [face(60, 60, 150)],
          13: [face(0, 0, 560, yaw=1.3)]}


def test_the_collage_uses_the_pickers_winner(tmp_path):
    det = Scripted(SCRIPT)
    r = FaceRig(tmp_path, face_detector=det)
    res = r.c.post("/swap", files=files(tmp_path, [("a.png", photo(11), "image/png"), ("b.png", photo(12), "image/png")],
                                        main=photo(15)), data={"consent": "yes", "collage": "yes", "facelock": "yes"})
    job = res.json()["job"]
    meta = meta_of(r, job)
    assert meta["facelock"]["pick"]["chosen"] == 2 and meta["collage"]["face"] == 2
    assert (r.root / job / "ref_collage.png").read_bytes() == collage.make_collage(photo(15), photo(12), detector=Scripted(SCRIPT))


def test_when_the_main_photo_wins_the_collage_uses_the_best_added_photo_not_the_main_photo_twice(tmp_path):
    r = FaceRig(tmp_path, face_detector=Scripted(SCRIPT))
    added = [("a.png", photo(11), "image/png"), ("p.png", photo(13), "image/png"), ("b.png", photo(16), "image/png")]
    job = r.c.post("/swap", files=files(tmp_path, added, main=photo(15)),
                   data={"consent": "yes", "collage": "yes"}).json()["job"]
    meta = meta_of(r, job)
    assert meta["facelock"]["pick"]["chosen"] == 4                                 # the main photo (200 px) won face-lock's pick
    assert meta["collage"] == {"file": "ref_collage.png", "face": 3, "of": 3}       # 150 px frontal beats 120 px and a profile
    exp = collage.make_collage(photo(15), photo(16), detector=Scripted(SCRIPT))
    assert (r.root / job / "ref_collage.png").read_bytes() == exp
    assert exp != collage.make_collage(photo(15), photo(15), detector=Scripted(SCRIPT))


def test_collage_face_index_rules():
    s = lambda px, front, score: {"face_px": px, "front": front, "score": score}
    assert ss.collage_face_index({"chosen": 2, "scores": [s(1, 1, 1)] * 4}, 3) == 2
    pick = {"chosen": 4, "scores": [s(0, 0, 0), s(300, 0.2, 0.9), s(100, 0.9, 0.1), s(400, 1, 1)]}
    assert ss.collage_face_index(pick, 3) == 3                                     # a profile never beats a frontal face
    pick = {"chosen": 3, "scores": [s(0, 0, 0), s(0, 0, 0), s(0, 0, 0)]}
    assert ss.collage_face_index(pick, 2) == 1                                     # no face anywhere: the first added photo
    pick = {"chosen": 3, "scores": [s(100, 1, 0.5), s(100, 1, 0.5), s(500, 1, 1)]}
    assert ss.collage_face_index(pick, 2) == 1                                     # ties go to the earlier photo


# ------------------------------------------------------------------ what each step reads

def test_generation_reads_the_collage_and_everything_else_the_original_photo(tmp_path, monkeypatch):
    checked, prepared = [], []
    real_check, real_prepare = tt.check_reference, tt.prepare
    monkeypatch.setattr(tt, "check_reference", lambda p, *a, **k: checked.append(Path(p).name) or real_check(p, *a, **k))
    monkeypatch.setattr(tt, "prepare", lambda src, ref, **k: prepared.append(Path(ref).name) or real_prepare(src, ref, **k))
    r = FaceRig(tmp_path)
    job = r.c.post("/swap", files=files(tmp_path, [("c.png", photo_bytes(seed=4), "image/png")]),
                   data={"consent": "yes", "collage": "yes"}).json()["job"]
    q = ss.request_of(r.root, job)
    assert q["ref"] == "ref_collage.png" and q["photo"] == "ref.png" and q["collage"] is True     # what the mask step uses
    assert checked == ["ref.png"]                                                  # the slider check before anything is paid
    assert r.c.post(f"/generate/{job}").status_code == 200
    assert prepared == ["ref.png"]                                                 # the slider's skin grain
    assert r.c.get(f"/status/{job}").json()["prepare"]["state"] == "ready"
    (r.root / job / "ref_collage.png").unlink()
    with pytest.raises(svc.JobError) as e:                                         # never falls back silently
        ss.request_of(r.root, job)
    assert e.value.status == 409


def test_face_lock_sends_the_original_photo_never_the_collage(tmp_path):
    mh = FakeMagicHour(tmp_path, result=mh_result(tmp_path))
    r = FaceRig(tmp_path, mh)
    close = photo_bytes(seed=4)
    job = r.c.post("/swap", files=files(tmp_path, [("c.png", close, "image/png")]),
                   data={"consent": "yes", "collage": "yes", "facelock": "yes"}).json()["job"]
    d = r.root / job
    assert r.c.post(f"/generate/{job}").status_code == 200
    assert fl.face_photo(d, meta_of(r, job)).name == "face.png"
    assert r.lock(job).status_code == 200
    assert mh.uploaded[UP_IMAGE] == close and mh.uploaded[UP_IMAGE] != (d / "ref_collage.png").read_bytes()
    assert r.c.get(f"/status/{job}").json()["facelock"]["state"] == "done"


def test_face_lock_falls_back_to_the_main_photo_not_the_collage_when_the_main_photo_wins(tmp_path):
    mh = FakeMagicHour(tmp_path, result=mh_result(tmp_path))
    r = FaceRig(tmp_path, mh, face_detector=Scripted(SCRIPT))
    job = r.c.post("/swap", files=files(tmp_path, [("a.png", photo(11), "image/png")], main=photo(15)),
                   data={"consent": "yes", "collage": "yes", "facelock": "yes"}).json()["job"]
    assert r.c.post(f"/generate/{job}").status_code == 200
    assert r.lock(job).status_code == 200
    assert mh.uploaded[UP_IMAGE] == photo(15)                                      # the main photo as uploaded
    assert meta_of(r, job)["collage"]["face"] == 1


def test_health_and_the_modal_app_wiring():
    from test_swap_service import Rig
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        assert Rig(Path(t)).c.get("/health").json()["collage"] is True
    mask_fn = APP[APP.index("def mask_job("):APP.index("def swap_job(")]
    assert 'q = ss.request_of(JOBS, job)' in mask_fn and 'd / q["ref"]' in mask_fn      # the generation reference
    core = APP[APP.index("def _preview_core("):APP.index("def make_preview(")]
    assert '"--refer_path", str(ref)' in core
    light = APP[APP.index("light = ("):APP.index("WEIGHTS_FLAGS =")]
    assert 'HERE / "collage.py", "/opt/collage.py"' in light and '"pillow' in light
