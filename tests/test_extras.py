import sys, time
from pathlib import Path
import numpy as np, cv2, pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import swapkit as sk, longclip as lc, consent as cs, realism as rl
from test_swapkit import make_video


def test_plan_chunks_short_is_single():
    assert lc.plan_chunks(20, 24) == [(0.0, 20.0)]


def test_plan_chunks_long_covers_everything_with_overlap():
    for dur in (31, 60, 95.5, 120):
        ch = lc.plan_chunks(dur, 24)
        assert ch[0][0] == 0 and abs(ch[-1][0] + ch[-1][1] - dur) < 0.05
        for (s1, l1), (s2, l2) in zip(ch, ch[1:]):
            assert s2 < s1 + l1  # they overlap
            assert s2 > s1       # and progress
        assert all(l <= sk.MAX_SECONDS for _, l in ch)
        assert all(abs(s * 24 - round(s * 24)) < 1e-6 for s, _ in ch)  # frame aligned


def test_plan_chunks_rejects_bad_settings():
    with pytest.raises(ValueError):
        lc.plan_chunks(60, 24, chunk=40)
    with pytest.raises(ValueError):
        lc.plan_chunks(60, 24, chunk=10, overlap=10)


def test_stitch_frame_count_and_crossfade(tmp_path):
    a = make_video(tmp_path / "a.mp4", (0, 0, 0), n=24, fps=24)
    b = make_video(tmp_path / "b.mp4", (255, 255, 255), n=24, fps=24)
    out = tmp_path / "o.mp4"
    total = lc.stitch([a, b], 6, out)
    assert total == 24 + 24 - 6
    fr = list(sk.frames(out))
    assert len(fr) == total
    mid = [int(f.mean()) for f in fr[18:24]]
    assert mid == sorted(mid) and 0 < mid[0] < mid[-1] < 255  # smooth ramp, no jump


def test_consent_real_needs_everything(tmp_path):
    img = tmp_path / "f.png"; cv2.imwrite(str(img), np.full((16, 16, 3), 90, np.uint8))
    with pytest.raises(PermissionError):
        cs.make_record(img, "real", attest={}, who="A", contact="a@b.c")
    full = {k: True for k in cs.REQUIRED_FOR_REAL}
    with pytest.raises(PermissionError):
        cs.make_record(img, "real", attest=full, who="", contact="")
    rec = cs.make_record(img, "real", attest=full, who="Ann", contact="ann@example.com")
    assert cs.verify(rec, img)


def test_consent_bound_to_image_and_expiry_and_revoke(tmp_path):
    img = tmp_path / "f.png"; cv2.imwrite(str(img), np.full((16, 16, 3), 90, np.uint8))
    other = tmp_path / "g.png"; cv2.imwrite(str(other), np.full((16, 16, 3), 91, np.uint8))
    full = {k: True for k in cs.REQUIRED_FOR_REAL}
    rec = cs.make_record(img, "real", attest=full, who="Ann", contact="x@y.z", valid_days=1, now=1000)
    with pytest.raises(PermissionError):
        cs.verify(rec, other, now=1001)
    with pytest.raises(PermissionError):
        cs.verify(rec, img, now=1000 + 2 * 86400)
    p = tmp_path / "c.json"; cs.save(rec, p); cs.revoke(p)
    with pytest.raises(PermissionError):
        cs.verify(cs.load(p), img, now=1001)


def test_consent_fictional_needs_no_person(tmp_path):
    img = tmp_path / "f.png"; cv2.imwrite(str(img), np.full((16, 16, 3), 5, np.uint8))
    assert cs.verify(cs.make_record(img, "fictional"), img)


def test_texture_adds_detail_not_colour_shift():
    flat = np.full((64, 64, 3), (120, 140, 180), np.uint8)
    out = rl.add_texture(flat, grain=4, detail=0.3, seed=1)
    spatial = lambda a: float(a.astype(np.float32)[..., 1].std())  # variation within one channel
    assert spatial(flat) == 0.0 and spatial(out) > 1.0
    assert abs(float(out.mean()) - float(flat.mean())) < 3.0
    assert np.array_equal(out, rl.add_texture(flat, grain=4, detail=0.3, seed=1))  # reproducible


def test_texture_respects_mask():
    fr = np.full((64, 64, 3), 128, np.uint8)
    mask = np.zeros((64, 64), np.uint8); mask[:, :32] = 255
    out = rl.add_texture(fr, mask, grain=6, detail=0)
    assert out[:, :20].std() > 1.0
    assert out[:, 50:].std() < 0.01  # far outside the mask stays untouched


def test_texture_video_keeps_length_and_audio(tmp_path):
    v = make_video(tmp_path / "v.mp4", (100, 120, 140), n=24, fps=24, audio=True)
    out = tmp_path / "o.mp4"
    assert rl.process_video(v, out) == 24
    info = sk.probe(out)
    assert info["has_audio"] and abs(info["duration"] - 1.0) < 0.2
