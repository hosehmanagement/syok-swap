"""Best-photo picker for face-lock (picker.py) and how the job service and the HTTP layer use it.

Three kinds of cases:
  * scripted: a fake detector returns exact boxes and 5 landmarks, so every scoring rule is checked on its own;
  * real YuNet: the shipped face_detection_yunet_2023mar.onnx runs on simple drawn faces (no image files in the repo), so the
    default detector, the landmark-based frontality, sharpness and the two-face rule are checked end to end;
  * real photos: only when SYOK_FACE_SAMPLES points at a folder of real portraits (none are in the repo).
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import picker
import swap_service as ss
import texture_service as svc
from test_texture_service import FAKE, photo_bytes, skin_photo

pytest.importorskip("fastapi")
from test_facelock import KEY, UP_IMAGE, FaceRig, FakeMagicHour, mh_result
from test_swap_service import clip


# ------------------------------------------------------------------ scripted faces

def face(x, y, w, yaw=0.0, roll=0.0, mouth=None):
    """A face box with YuNet-ordered landmarks. yaw = how far the nose sits from the eye midpoint, in half eye distances
    (0 straight on, >= 1 beyond an eye: a profile). The mouth follows the nose unless given. roll in degrees."""
    d = 0.4 * w
    cx, cy = x + w / 2, y + 0.4 * w
    u = np.array([np.cos(np.radians(roll)), np.sin(np.radians(roll))])
    nrm = np.array([-u[1], u[0]])
    mid = np.array([cx, cy])
    re, le = mid - u * d / 2, mid + u * d / 2
    nose = mid + u * yaw * d / 2 + nrm * 0.25 * w
    mo = yaw if mouth is None else mouth
    mmid = mid + u * mo * d / 2 + nrm * 0.45 * w
    mr, ml = mmid - u * 0.2 * w, mmid + u * 0.2 * w
    return {"box": (x, y, w, w * 1.2), "landmarks": [tuple(p) for p in (re, le, nose, mr, ml)]}


def tagged(tag, size=600, img=None):
    """A skin-coloured photo whose top-left pixel carries `tag`, so the scripted detector knows which photo it sees."""
    img = skin_photo(seed=tag % 50 + 1, size=size) if img is None else img.copy()
    img[0, 0] = (tag, tag, tag)
    return img


def png(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class Scripted:
    """Detector: tag -> list of faces. Unknown tags have no face."""

    def __init__(self, script):
        self.script = script
        self.seen = []

    def __call__(self, bgr):
        tag = int(bgr[0, 0, 0])
        self.seen.append(tag)
        return [dict(f) for f in self.script.get(tag, [])]


def run(script, tags, size=600, imgs=None):
    det = Scripted(script)
    images = imgs if imgs is not None else [tagged(t, size) for t in tags]
    return picker.pick(images, det)


# ------------------------------------------------------------------ the measures

def test_frontality_is_1_straight_on_and_low_for_a_profile():
    assert picker.frontality(face(0, 0, 300)["landmarks"]) == 1.0
    assert picker.frontality(face(0, 0, 300, yaw=1.2)["landmarks"]) < picker.PROFILE_FRONT
    assert picker.frontality(face(0, 0, 300, yaw=-1.2)["landmarks"]) < picker.PROFILE_FRONT        # either side
    half = picker.frontality(face(0, 0, 300, yaw=0.5)["landmarks"])
    assert 0.35 < half < 0.8
    # a tilted but straight-on head stays front-facing (roll only takes up to 20 %)
    tilt = picker.frontality(face(0, 0, 300, roll=30)["landmarks"])
    assert tilt == pytest.approx((0.6 + 0.4) * (0.8 + 0.2 * (1 - 30 / 45)), abs=1e-3) and tilt > 0.8
    # the nose alone off-centre (mouth centred) is still penalised; eyes on top of each other -> 0
    assert picker.frontality(face(0, 0, 300, yaw=1.0, mouth=0.0)["landmarks"]) == pytest.approx(0.4, abs=1e-3)
    assert picker.frontality([(10, 10)] * 5) == 0.0


def test_sharpness_compares_big_and_small_crops_on_the_same_grid_and_flat_is_zero():
    rng = np.random.default_rng(0)
    tex = np.clip(128 + rng.normal(0, 30, (512, 512, 3)), 0, 255).astype(np.uint8)
    tex = cv2.GaussianBlur(tex, (0, 0), 0.8)
    big = picker.sharpness(tex)
    small = picker.sharpness(cv2.resize(tex, (200, 200), interpolation=cv2.INTER_AREA))
    blurred = picker.sharpness(cv2.GaussianBlur(tex, (0, 0), 6))
    assert big > 0.9 and small > 0.9 and blurred < 0.2
    assert picker.sharpness(np.full((300, 300, 3), 128, np.uint8)) == 0.0


def test_exposure_window_and_its_penalty():
    assert picker.exposure_factor(60) == picker.exposure_factor(200) == picker.exposure_factor(130) == 1.0
    assert picker.exposure_factor(30) == pytest.approx(0.7) and picker.exposure_factor(250) == pytest.approx(0.5)
    assert picker.exposure_factor(0) == 0.5


def test_score_formula_as_documented():
    assert picker.score_parts(512, 1.0, 1.0, 1.0, False) == 1.0
    assert picker.score_parts(900, 1.0, 1.0, 1.0, False) == 1.0                                   # capped at 512
    assert picker.score_parts(256, 0.8, 0.5, 0.7, True) == round(0.5 ** 1.5 * 0.8 * 0.7 * 0.7 * 0.6, 4)


# ------------------------------------------------------------------ the pick rules (scripted detector)

def test_prefers_the_larger_face():
    r = run({1: [face(100, 100, 120)], 2: [face(50, 50, 400)], 3: [face(80, 80, 250)]}, [1, 2, 3])
    assert r["chosen"] == 2 and r["of"] == 3
    assert [s["face_px"] for s in r["scores"]] == [120, 400, 250]
    assert r["reason"] == "Using photo 2: front-facing and about 400 px wide."


def test_size_counts_up_to_512_then_ties_go_to_the_larger_face_then_the_earlier_photo():
    r = run({1: [face(0, 0, 520)], 2: [face(0, 0, 580)]}, [1, 2], size=700)
    assert r["scores"][0]["score"] == r["scores"][1]["score"] and r["chosen"] == 2
    r = run({1: [face(0, 0, 300)], 2: [face(0, 0, 300)], 3: [face(0, 0, 300)]}, [1, 2, 3])
    assert r["chosen"] == 1
    again = [run({1: [face(0, 0, 300)], 2: [face(0, 0, 300)]}, [1, 2]) for _ in range(3)]
    assert all(a == again[0] for a in again)                                                     # deterministic


def test_prefers_frontal_over_profile_even_when_the_profile_is_much_larger():
    r = run({1: [face(0, 0, 560, yaw=1.3)], 2: [face(100, 100, 160)]}, [1, 2], size=700)
    assert r["chosen"] == 2
    assert r["scores"][0]["front"] < picker.PROFILE_FRONT and r["scores"][0]["note"].startswith("side profile")
    # and a decent turned face (front >= 0.35) still beats a profile; among profiles only, the best profile wins
    r = run({1: [face(0, 0, 500, yaw=1.3)], 2: [face(0, 0, 200, yaw=0.55)]}, [1, 2])
    assert r["chosen"] == 2
    r = run({1: [face(0, 0, 200, yaw=1.3)], 2: [face(0, 0, 450, yaw=1.3)]}, [1, 2])
    assert r["chosen"] == 2 and "side profile" in r["reason"]


def test_sharp_beats_blurred_at_equal_size():
    rng = np.random.default_rng(1)
    sharp = np.clip(150 + rng.normal(0, 25, (600, 600, 3)), 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(sharp, (0, 0), 4)
    r = run({1: [face(100, 100, 300)], 2: [face(100, 100, 300)]}, None, imgs=[tagged(1, img=blur), tagged(2, img=sharp)])
    assert r["chosen"] == 2 and r["scores"][0]["sharp"] < 0.4 < r["scores"][1]["sharp"]
    assert "blurry" in r["scores"][0]["note"]


def test_a_dark_face_loses_to_a_well_lit_one_and_is_flagged_not_ok():
    dark = np.full((600, 600, 3), 20, np.uint8)
    lit = np.full((600, 600, 3), 140, np.uint8)
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 6, (600, 600, 1))
    dark = np.clip(dark + noise, 0, 255).astype(np.uint8)
    lit = np.clip(lit + noise, 0, 255).astype(np.uint8)
    r = run({1: [face(100, 100, 300)], 2: [face(100, 100, 300)]}, None, imgs=[tagged(1, img=dark), tagged(2, img=lit)])
    assert r["chosen"] == 2 and r["scores"][0]["ok"] is False and r["scores"][1]["ok"] is True
    assert "too dark" in r["scores"][0]["note"]


def test_two_people_in_a_photo_are_penalised_but_a_tiny_background_face_is_not():
    r = run({1: [face(0, 0, 300), face(320, 0, 280)], 2: [face(0, 0, 300)]}, [1, 2])
    assert r["chosen"] == 2 and "more than one face" in r["scores"][0]["note"]
    assert r["scores"][0]["score"] == pytest.approx(r["scores"][1]["score"] * picker.MULTI_PENALTY, abs=1e-3)
    r = run({1: [face(0, 0, 300), face(400, 400, 60)], 2: [face(0, 0, 300)]}, [1, 2])
    assert r["scores"][0]["score"] == r["scores"][1]["score"] and r["chosen"] == 1               # 60 < 40 % of 300


def test_a_photo_without_a_face_is_skipped_and_an_unreadable_one_too():
    r = picker.pick([tagged(1), None, tagged(3)], Scripted({3: [face(0, 0, 100)]}))
    assert r["chosen"] == 3
    assert r["scores"][0] == {"n": 1, "face_px": 0, "front": 0.0, "sharp": 0.0, "ok": False, "note": "no face found",
                              "score": 0.0}
    assert r["scores"][1]["note"] == "could not be read"


def test_no_face_anywhere_returns_no_choice():
    r = run({}, [1, 2, 3])
    assert r["chosen"] is None and r["of"] == 3 and "No face was found in any of the 3 photos" in r["reason"]


def test_a_detector_that_crashes_counts_as_no_face():
    def boom(_):
        raise RuntimeError("model broke")
    assert picker.pick([tagged(1)], boom)["chosen"] is None


def test_box_detector_adapter_maps_frontal_and_profile_weights():
    f = picker.from_box_detector(lambda img: ((1, 2, 30, 30), 1.0))(None)
    assert f == [{"box": (1.0, 2.0, 30.0, 30.0), "landmarks": None, "front": 1.0}]
    assert picker.from_box_detector(lambda img: ((1, 2, 30, 30), 0.7))(None)[0]["front"] == 0.3
    assert picker.from_box_detector(lambda img: (None, 0.0))(None) == []


# ------------------------------------------------------------------ real YuNet on drawn faces

needs_yunet = pytest.mark.skipif(picker._find_yunet() is None or not hasattr(cv2, "FaceDetectorYN"),
                                 reason="YuNet model or cv2.FaceDetectorYN missing")


def drawn(fw=200, size=400, turn=0.0, seed=0):
    """A simple drawn face YuNet detects (checked 2026-11-20 with opencv 4.14: scores 0.76-0.92). `turn` slides the eyes,
    nose and mouth sideways inside the head outline, which YuNet's landmarks read as a turned head."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), (200, 210, 220), np.uint8)
    cx, cy = size // 2, size // 2
    a, b = fw // 2, int(fw * 0.65)
    cv2.ellipse(img, (cx, cy), (a, b), 0, 0, 360, (150, 170, 215), -1)
    cv2.ellipse(img, (cx, cy - int(b * 0.75)), (int(a * 1.05), int(b * 0.45)), 0, 180, 360, (30, 30, 40), -1)
    s = int(turn * a)
    ey, ex = cy - int(b * 0.15), int(a * 0.4)
    for x in (cx - ex + s, cx + ex + s):
        cv2.ellipse(img, (x, ey), (int(a * 0.18), int(a * 0.09)), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (x, ey), int(a * 0.08), (40, 30, 30), -1)
        cv2.line(img, (x - int(a * 0.2), ey - int(a * 0.2)), (x + int(a * 0.2), ey - int(a * 0.22)), (30, 30, 40),
                 max(2, a // 25))
    nx = cx + int(1.6 * s)
    pts = np.array([[nx, ey + int(a * 0.1)], [nx - int(a * 0.12), cy + int(b * 0.22)], [nx + int(a * 0.12), cy + int(b * 0.22)]])
    cv2.fillPoly(img, [pts], (120, 140, 190))
    cv2.ellipse(img, (cx + int(1.2 * s), cy + int(b * 0.45)), (int(a * 0.3), int(a * 0.1)), 0, 0, 360, (80, 80, 170), -1)
    img = np.clip(img.astype(np.float32) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


@needs_yunet
def test_real_yunet_finds_a_drawn_face_with_front_facing_landmarks():
    m = picker.measure(drawn())
    assert m["face"] and 160 <= m["face_px"] <= 230 and m["front"] > 0.8 and m["ok"] and not m["multi"]


@needs_yunet
def test_real_yunet_prefers_the_larger_frontal_face():
    r = picker.pick([drawn(fw=120), drawn(fw=200)])
    assert r["chosen"] == 2 and r["scores"][0]["face_px"] < r["scores"][1]["face_px"]


@needs_yunet
def test_real_yunet_reads_a_turned_face_as_a_profile_and_prefers_the_smaller_frontal_one():
    r = picker.pick([drawn(fw=200, turn=0.6), drawn(fw=120)])
    assert r["scores"][0]["front"] < picker.PROFILE_FRONT <= r["scores"][1]["front"]
    assert r["chosen"] == 2


@needs_yunet
def test_real_yunet_blurred_loses_to_sharp():
    img = drawn()
    r = picker.pick([cv2.GaussianBlur(img, (0, 0), 3), img])
    assert r["chosen"] == 2 and r["scores"][0]["sharp"] < r["scores"][1]["sharp"]


@needs_yunet
def test_real_yunet_two_people_are_penalised():
    two = np.concatenate([drawn(fw=150), drawn(fw=150, seed=1)], axis=1)
    m = picker.measure(two)
    assert m["face"] and m["multi"] and "more than one face" in m["note"]
    assert picker.pick([two, drawn(fw=150, seed=2)])["chosen"] == 2


@needs_yunet
def test_real_yunet_finds_no_face_in_a_plain_photo_and_in_a_huge_close_up_rescales():
    assert picker.measure(skin_photo(3))["face"] is False
    close = cv2.resize(drawn(fw=360, size=400), (1600, 1600), interpolation=cv2.INTER_LINEAR)    # face fills a big frame
    m = picker.measure(close)
    assert m["face"] and m["face_px"] > 1000 and m["front"] > 0.8


SAMPLES = os.environ.get("SYOK_FACE_SAMPLES")


@needs_yunet
@pytest.mark.skipif(not SAMPLES, reason="set SYOK_FACE_SAMPLES to a folder of real portraits to run this")
def test_real_portraits_beat_their_blurred_and_shrunk_copies():
    paths = sorted(p for p in Path(SAMPLES).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    assert paths
    for p in paths:
        img = picker.read_image(p)
        m = picker.measure(img)
        assert m["face"], p.name
        blur = cv2.GaussianBlur(img, (0, 0), 3)
        small = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        assert picker.pick([blur, small, img])["chosen"] == 3, p.name


# ------------------------------------------------------------------ the job service and the HTTP layer

SCRIPT = {
    11: [face(100, 100, 120)],                    # small frontal
    12: [face(50, 50, 420)],                      # big frontal: the one to pick
    13: [face(0, 0, 560, yaw=1.3)],               # big profile
    14: [face(0, 0, 300), face(320, 0, 280)],     # two people
    15: [face(60, 60, 200)],                      # the main photo: medium frontal
    16: [face(60, 60, 150)],
}


def photo(tag):
    return png(tagged(tag))


def upload(r, faces, ref_tag=15, **extra):
    files = [("video", ("v.mp4", clip(r.tmp), "video/mp4")), ("photo", ("p.png", photo(ref_tag), "image/png"))]
    files += [("face", f) for f in faces]
    return r.c.post("/swap", files=files, data={"consent": "yes", "facelock": "yes", **extra})


def job_files(d):
    return sorted(p.name for p in Path(d).iterdir() if p.is_file())


def test_repeated_face_fields_are_all_candidates_with_the_main_photo_last(tmp_path):
    det = Scripted(SCRIPT)
    r = FaceRig(tmp_path, face_detector=det)
    res = upload(r, [("a.jpg", photo(11), "image/jpeg"), ("b.webp", photo(12), "image/webp"),
                     ("c.png", photo(13), "image/png"), ("d.png", photo(14), "image/png")])
    assert res.status_code == 200, res.text
    job = res.json()["job"]
    d = r.root / job
    assert det.seen == [11, 12, 13, 14, 15]                       # the four added photos in order, then the main photo
    meta = json.loads((d / "meta.json").read_text())
    assert meta["face"] == "face.webp" and (d / "face.webp").read_bytes() == photo(12)
    pick = r.c.get(f"/status/{job}").json()["facelock"]["pick"]
    assert pick["chosen"] == 2 and pick["of"] == 5
    assert pick["reason"] == "Using photo 2: front-facing and about 420 px wide."
    assert [s["n"] for s in pick["scores"]] == [1, 2, 3, 4, 5]
    assert set(pick["scores"][0]) == {"n", "face_px", "front", "sharp", "ok", "note", "score"}
    assert pick["scores"][2]["note"].startswith("side profile") and "more than one face" in pick["scores"][3]["note"]
    assert pick == meta["facelock"]["pick"]
    # the other candidates are not kept anywhere in the job folder
    assert not any(n.startswith("face_cand") for n in job_files(d))
    assert [n for n in job_files(d) if n.startswith("face")] == ["face.webp"]
    blob = b"".join(p.read_bytes() for p in d.rglob("*") if p.is_file())
    for t in (11, 13, 14):
        assert photo(t) not in blob
    assert "warning" not in r.c.get(f"/status/{job}").json()["facelock"]                     # 420 px is plenty


def test_the_main_photo_is_always_a_candidate_and_can_win(tmp_path):
    r = FaceRig(tmp_path, face_detector=Scripted(SCRIPT))
    job = upload(r, [("a.jpg", photo(11), "image/jpeg")]).json()["job"]                       # 120 px vs the main photo's 200
    d = r.root / job
    meta = json.loads((d / "meta.json").read_text())
    assert meta["face"] == "face.png" and (d / "face.png").read_bytes() == (d / "ref.png").read_bytes() == photo(15)
    assert not (d / "face.jpg").exists()
    pick = meta["facelock"]["pick"]
    assert pick["chosen"] == 2 and pick["of"] == 2
    assert pick["reason"] == "Using photo 2 (your main photo): front-facing and about 200 px wide."
    assert "about 200 px wide" in meta["facelock_warning"]                                    # the CHOSEN photo's size


def test_a_single_face_photo_that_is_better_than_the_main_photo_is_kept_as_before(tmp_path):
    r = FaceRig(tmp_path, face_detector=Scripted(SCRIPT))
    job = upload(r, [("close.jpg", photo(12), "image/jpeg")]).json()["job"]
    d = r.root / job
    meta = json.loads((d / "meta.json").read_text())
    assert meta["face"] == "face.jpg" and (d / "face.jpg").read_bytes() == photo(12)
    assert meta["facelock"]["pick"]["chosen"] == 1 and meta["facelock"]["pick"]["of"] == 2
    assert "facelock_warning" not in meta


def test_limits_five_files_twelve_mb_together_each_photo_limit_and_type_nothing_stored(tmp_path):
    r = FaceRig(tmp_path, face_detector=Scripted(SCRIPT))
    six = [(f"{i}.png", photo(11), "image/png") for i in range(6)]
    res = upload(r, six)
    assert res.status_code == 400 and res.json()["error"] == "Add at most 5 face photos."
    seven = 7 * 2**20
    res = upload(r, [("a.png", b"x" * seven, "image/png"), ("b.png", b"y" * seven, "image/png")])
    assert res.status_code == 400 and res.json()["error"] == "The face photos together are larger than 12 MB."
    res = upload(r, [("a.png", b"x" * (svc.MAX_PHOTO_BYTES + 1), "image/png")])
    assert res.status_code == 400 and res.json()["error"] == "Face photo is larger than 8 MB."
    res = upload(r, [("a.png", photo(11), "image/png"), ("b.gif", photo(12), "image/gif")])
    assert res.status_code == 400 and res.json()["error"] == "Face photo must be png, jpg or webp."
    assert list(r.root.iterdir()) == [] and r.calls["mask"] == 0
    five = [(f"{i}.png", photo(11), "image/png") for i in range(5)]                          # exactly 5 is fine
    assert upload(r, five).status_code == 200
    h = r.c.get("/health").json()["limits"]
    assert h["faces"] == 5 and h["faces_mb"] == 12


def test_limits_at_the_service_layer_and_empty_parts_are_ignored():
    assert ss.face_uploads([(b"", ".png"), (b"abc", ".PNG")]) == [(b"abc", ".png")]
    assert ss.face_uploads(None, b"abc", ".jpg") == [(b"abc", ".jpg")]
    assert ss.face_uploads() == []
    with pytest.raises(svc.JobError):
        ss.face_uploads([(b"a", ".png")] * 6)
    with pytest.raises(svc.JobError):
        ss.face_uploads([(b"a" * (6 * 2**20 + 1), ".png")] * 2)
    ss.face_uploads([(b"a" * (6 * 2**20), ".png")] * 2)                                       # exactly 12 MB is fine


def test_no_face_in_any_photo_uses_the_main_photo_and_says_so(tmp_path):
    r = FaceRig(tmp_path, face_detector=Scripted({}))
    job = upload(r, [("a.png", photo(11), "image/png"), ("b.png", photo(12), "image/png")]).json()["job"]
    d = r.root / job
    meta = json.loads((d / "meta.json").read_text())
    assert "face" not in meta and not any(n.startswith("face") for n in job_files(d))
    st = r.c.get(f"/status/{job}").json()["facelock"]
    assert st["pick"]["chosen"] == 3 and st["pick"]["of"] == 3
    assert st["pick"]["reason"] == "No face was found in any of the 3 photos, so your main photo is used."
    assert st["warning"].startswith("We could not find a clear face in any of your 3 photos, so your main photo is used")
    assert st["state"] == "idle" and r.calls["mask"] == 1                                     # never blocks


def test_every_candidate_under_256_px_gets_its_own_warning(tmp_path):
    r = FaceRig(tmp_path, face_detector=Scripted(SCRIPT))
    job = upload(r, [("a.png", photo(11), "image/png"), ("b.png", photo(16), "image/png")]).json()["job"]
    w = r.c.get(f"/status/{job}").json()["facelock"]["warning"]
    assert w.startswith("Every photo you added shows the face under 256 px wide: the best one is only about 200 px wide")
    assert "512 x 512" in w


def test_facelock_sends_the_picked_photo_to_magic_hour(tmp_path):
    mh = FakeMagicHour(tmp_path, result=mh_result(tmp_path))
    r = FaceRig(tmp_path, mh, face_detector=Scripted(SCRIPT))
    job = upload(r, [("a.png", photo(11), "image/png"), ("b.webp", photo(12), "image/webp")]).json()["job"]
    assert r.c.post(f"/generate/{job}").status_code == 200
    assert r.lock(job).status_code == 200
    st = r.c.get(f"/status/{job}").json()["facelock"]
    assert st["state"] == "done" and st["pick"]["chosen"] == 2                                # the pick survives the run
    up = next(c for c in mh.calls if c["url"].endswith("upload-urls"))["json"]
    assert up["items"][1] == {"type": "image", "extension": "webp"}
    assert mh.uploaded[UP_IMAGE] == photo(12)


def test_new_candidates_at_facelock_time_replace_the_earlier_pick(tmp_path):
    mh = FakeMagicHour(tmp_path, result=mh_result(tmp_path))
    r = FaceRig(tmp_path, mh, face_detector=Scripted(SCRIPT))
    job = upload(r, [("a.webp", photo(12), "image/webp")]).json()["job"]
    d = r.root / job
    assert (d / "face.webp").exists()
    assert r.c.post(f"/generate/{job}").status_code == 200
    files = [("face", ("x.jpg", photo(11), "image/jpeg")), ("face", ("y.jpg", photo(16), "image/jpeg"))]
    res = r.c.post(f"/facelock/{job}", headers={"X-Magic-Hour-Key": KEY}, files=files)
    assert res.status_code == 200
    meta = json.loads((d / "meta.json").read_text())
    assert not (d / "face.webp").exists()                                                     # the old face photo is gone
    assert meta["face"] == "face.png" and meta["facelock"]["pick"]["chosen"] == 3             # the main photo (200 px) wins
    assert mh.uploaded[UP_IMAGE] == photo(15)
    assert [n for n in job_files(d) if n.startswith("face")] == ["face.png"]
    # too many files at face-lock time: refused, nothing queued
    j2 = upload(r, []).json()["job"]
    r.c.post(f"/generate/{j2}")
    many = [("face", (f"{i}.png", photo(11), "image/png")) for i in range(6)]
    res = r.c.post(f"/facelock/{j2}", headers={"X-Magic-Hour-Key": KEY}, files=many)
    assert res.status_code == 400 and res.json()["error"] == "Add at most 5 face photos."
    assert r.c.get(f"/status/{j2}").json()["facelock"]["state"] == "waiting"


def test_without_face_photos_nothing_is_picked_and_the_old_warning_logic_runs(tmp_path):
    root = tmp_path / "jobs"; root.mkdir()
    job = ss.new_swap_job(root, clip(tmp_path), photo_bytes(), ".png", "yes", facelock=True, detector=FAKE)
    meta = svc.read_meta(svc.job_dir(root, job))
    assert "pick" not in meta["facelock"] and "face" not in meta
    assert "about 80 px wide" in meta["facelock_warning"]


def test_service_defaults_to_yunet_and_tests_fakes_are_adapted():
    assert ss._picker_detector(ss.tt.detect_face, None) is picker.yunet_faces
    assert ss._picker_detector(FAKE, None)(skin_photo(3))[0]["box"] == (40.0, 40.0, 80.0, 80.0)
    own = lambda img: []
    assert ss._picker_detector(FAKE, own) is own
