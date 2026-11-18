import sys
from pathlib import Path
import numpy as np, cv2, pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import texture_transfer as tt, swapkit as sk
from test_swapkit import make_video

SKIN = (140, 160, 205)  # BGR, inside the skin chroma range


def skin_frame(w=200, h=200, seed=0):
    f = np.zeros((h, w, 3), np.uint8); f[:] = SKIN
    return f


def hf_energy(f, box):
    x, y, w, h = [int(v) for v in box]
    g = cv2.cvtColor(f[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float((g - cv2.GaussianBlur(g, (0, 0), 1.5)).std())


def test_skin_prob_accepts_skin_rejects_blue():
    assert tt.skin_prob(skin_frame()).mean() > 0.9
    blue = np.zeros((50, 50, 3), np.uint8); blue[:] = (200, 60, 30)
    assert tt.skin_prob(blue).mean() < 0.1


def test_detail_layer_has_no_low_frequency_and_is_bounded():
    rng = np.random.default_rng(0)
    ref = (skin_frame(400, 400).astype(np.float32) + rng.normal(0, 6, (400, 400, 3))).clip(0, 255).astype(np.uint8)
    layer = tt.make_detail_layer(ref, (100, 100, 200, 200), (120, 150))
    assert layer.shape == (150, 120)
    assert abs(float(layer.mean())) < 1.0
    assert np.abs(layer).max() < 40


def test_apply_adds_detail_inside_box_only():
    rng = np.random.default_rng(1)
    layer = rng.normal(0, 5, (130, 120)).astype(np.float32)
    f = skin_frame()
    box = (60, 50, 80, 80)
    out = tt.apply_texture(f, box, layer, strength=1.0)
    assert hf_energy(out, (80, 70, 40, 40)) > hf_energy(f, (80, 70, 40, 40)) + 0.5
    assert np.array_equal(out[:10], f[:10])  # far from the box: untouched
    assert out.dtype == np.uint8 and out.shape == f.shape


def test_apply_noop_cases():
    f = skin_frame(); layer = np.ones((10, 10), np.float32)
    assert tt.apply_texture(f, None, layer) is f
    assert tt.apply_texture(f, (10, 10, 50, 50), layer, strength=0) is f
    assert tt.apply_texture(f, (500, 500, 50, 50), layer) is not None  # box outside: no crash


def test_apply_skips_non_skin_areas():
    f = np.zeros((200, 200, 3), np.uint8); f[:] = (200, 60, 30)  # blue "background"
    layer = np.random.default_rng(2).normal(0, 8, (130, 120)).astype(np.float32)
    out = tt.apply_texture(f, (60, 50, 80, 80), layer)
    assert np.abs(out.astype(int) - f.astype(int)).max() <= 1


def test_smooth_tracks_fills_short_gaps_not_long_ones():
    d = [((100, 100, 50, 50), 1.0)] * 3 + [(None, 0.0)] * 2 + [((110, 100, 50, 50), 1.0)] * 3 + [(None, 0.0)] * 30 + [((100, 100, 50, 50), 1.0)]
    t = tt.smooth_tracks(d)
    assert all(t[i][0] is not None for i in range(8))
    assert t[3][1] == 0.5  # interpolated frames are weaker
    assert t[20][0] is None  # long gap stays empty


def test_smooth_tracks_all_none():
    assert tt.smooth_tracks([(None, 0.0)] * 4) == [(None, 0.0)] * 4


def test_process_video_with_fake_detector(tmp_path):
    src = make_video(tmp_path / "s.mp4", SKIN, n=12, fps=24, w=160, h=160)
    ref = tmp_path / "r.png"; cv2.imwrite(str(ref), np.random.default_rng(3).integers(100, 180, (300, 300, 3), dtype=np.uint8))
    box = (40, 40, 80, 80)
    out = tmp_path / "o.mp4"
    st = tt.process_video(src, ref, out, strength=1.0, detector=lambda f: (box, 1.0))
    assert st["frames"] == 12 and st["applied"] == 12
    assert len(list(sk.frames(out))) == 12
    assert sk.probe(out)["width"] == 160


def test_process_video_refuses_when_no_face(tmp_path):
    src = make_video(tmp_path / "s.mp4", SKIN, n=6, fps=24, w=64, h=64)
    ref = tmp_path / "r.png"; cv2.imwrite(str(ref), np.full((64, 64, 3), 120, np.uint8))
    with pytest.raises(ValueError):
        tt.process_video(src, ref, tmp_path / "o.mp4", detector=lambda f: (None, 0.0))


def test_box_is_skin_rejects_blue_window_accepts_skin():
    import numpy as np
    import texture_transfer as tt
    win = np.zeros((100, 100, 3), np.uint8); win[...] = (160, 90, 30)       # bluish (BGR)
    skin = np.zeros((100, 100, 3), np.uint8); skin[...] = (120, 150, 200)   # skin-ish (BGR)
    assert not tt.box_is_skin(win, (10, 10, 80, 80))
    assert tt.box_is_skin(skin, (10, 10, 80, 80))


def test_reject_outliers_drops_tiny_false_box():
    import texture_transfer as tt
    dets = [((0, 0, 200, 200), 1.0)] * 6 + [((5, 5, 40, 40), 1.0), (None, 0.0)]
    out = tt.reject_outliers(dets)
    assert out[6] == (None, 0.0) and out[0][0] == (0, 0, 200, 200) and out[7] == (None, 0.0)
