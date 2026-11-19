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


def skin_photo(seed=3, size=300, sd=6.0):
    """A plain skin-coloured photo with fine grain: what a real, evenly lit reference looks like to the engine."""
    rng = np.random.default_rng(seed)
    img = np.empty((size, size, 3), np.float32); img[:] = SKIN
    return np.clip(img + rng.normal(0, sd, (size, size, 1)), 0, 255).astype(np.uint8)


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
    ref = tmp_path / "r.png"; cv2.imwrite(str(ref), skin_photo(3))
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


def _hf(img_bgr, size_wh, sigma):
    g = cv2.cvtColor(cv2.resize(img_bgr, size_wh, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), sigma)


def _ref_with_features(seed=1, sd=6.0):
    """Skin grain plus things a face has: a dark 'eye' blob, a red 'mouth' bar and a bright 'tooth' stripe."""
    img = skin_photo(seed, 400, sd)
    cv2.ellipse(img, (150, 170), (40, 18), 0, 0, 360, (20, 20, 25), -1)
    cv2.ellipse(img, (260, 170), (40, 18), 0, 0, 360, (20, 20, 25), -1)
    cv2.rectangle(img, (150, 290), (260, 320), (90, 90, 200), -1)
    cv2.rectangle(img, (170, 300), (240, 308), (240, 240, 240), -1)
    return img


def test_layer_is_new_grain_not_a_copy_of_the_reference_features():
    """Found in the live test: pasting the reference's high-pass layer put its eyes, nose and mouth on the swapped face."""
    ref = _ref_with_features()
    box = (100, 100, 200, 200)
    size = (240, 280)
    layer = tt.make_detail_layer(ref, box, size)
    x, y, w, h = [int(round(v)) for v in tt.pad_box(box)]
    ref_hf = _hf(ref[y:y + h, x:x + w], size, max(1.0, 0.005 * size[0]))
    r = float(np.corrcoef(layer.ravel(), ref_hf.ravel())[0, 1])
    assert abs(r) < 0.1, r                                   # no copied structure at all
    # and the grain is even: no hot spot where the reference's eyes / mouth were
    e = cv2.GaussianBlur(np.abs(layer), (0, 0), 6)
    inner = e[int(.15 * size[1]):int(.85 * size[1]), int(.15 * size[0]):int(.85 * size[0])]
    assert inner.max() < 1.8 * np.median(inner), (inner.max(), np.median(inner))


def test_layer_strength_follows_the_references_grain_and_is_repeatable():
    box = (100, 100, 200, 200); size = (240, 280)
    fine = tt.make_detail_layer(skin_photo(1, 400, 4.0), box, size)
    rough = tt.make_detail_layer(skin_photo(1, 400, 8.0), box, size)
    assert 1.5 < rough.std() / fine.std() < 2.6              # twice the reference grain -> about twice the layer
    assert abs(float(rough.mean())) < 0.5 and fine.shape == (280, 240)
    again = tt.make_detail_layer(skin_photo(1, 400, 8.0), box, size)
    assert np.array_equal(rough, again)                      # same photo -> same layer, so preview == render


def test_reference_without_plain_skin_is_refused_with_a_plain_message():
    blue = np.zeros((300, 300, 3), np.uint8); blue[:] = (200, 60, 30)
    with pytest.raises(ValueError, match="plain skin"):
        tt.make_detail_layer(blue, (60, 60, 180, 180), (120, 140))


def test_swapped_faces_own_features_are_left_alone():
    """Eyes, glasses frame, brows and lips of the swapped face must not receive grain; plain cheeks must."""
    f = skin_frame(300, 300)
    cv2.ellipse(f, (105, 120), (22, 11), 0, 0, 360, (25, 25, 30), -1)      # eye
    cv2.rectangle(f, (70, 95), (230, 101), (30, 30, 35), -1)              # glasses frame / brow bar
    cv2.rectangle(f, (110, 205), (190, 225), (95, 95, 205), -1)           # lips (redder than skin)
    f = cv2.GaussianBlur(f, (0, 0), 1.2)
    box = (50, 50, 200, 200)
    layer = np.random.default_rng(5).normal(0, 6, (260, 240)).astype(np.float32)
    out = tt.apply_texture(f, box, layer, strength=3.0)
    d = np.abs(out.astype(int) - f.astype(int)).sum(axis=2).astype(float)
    cheek = d[150:190, 80:130].mean()
    assert cheek > 1.0                                        # grain is added on plain skin
    assert d[110:130, 90:120].mean() < 0.35 * cheek           # eye
    assert d[94:103, 80:220].mean() < 0.35 * cheek            # frame / brow
    assert d[208:222, 115:185].mean() < 0.35 * cheek          # lips


# ---- slider support: default 2.0, range 0..3, prepare once / preview any strength / render on confirm ----

def test_slider_constants_and_clamp():
    assert tt.DEFAULT_STRENGTH == 2.0 and tt.MIN_STRENGTH == 0.0 and tt.MAX_STRENGTH == 3.0
    assert tt.clamp_strength(2.0) == 2.0
    assert tt.clamp_strength(-1) == 0.0
    assert tt.clamp_strength(9) == 3.0
    assert tt.clamp_strength("1.5") == 1.5
    for bad in (None, "x", float("nan")):
        with pytest.raises(ValueError):
            tt.clamp_strength(bad)


def _prepared(tmp_path, n=6):
    src = make_video(tmp_path / "s.mp4", SKIN, n=n, fps=24, w=160, h=160)
    ref = tmp_path / "r.png"; cv2.imwrite(str(ref), skin_photo(3))
    box = (40, 40, 80, 80)
    return src, tt.prepare(src, ref, detector=lambda f: (box, 1.0)), box


def test_preview_frame_matches_full_render_at_same_strength(tmp_path):
    src, prep, box = _prepared(tmp_path)
    out = tmp_path / "o.mp4"
    tt.render_prepared(src, prep, out, strength=2.0)
    full = list(sk.frames(out))
    prev = tt.preview_frame(src, prep, 3, 2.0)
    # the video is re-encoded, so allow codec noise; the preview must be the same picture, not a different strength
    assert np.abs(full[3].astype(int) - prev.astype(int)).mean() < 3.0


def test_preview_detail_grows_with_strength_and_zero_is_untouched(tmp_path):
    src, prep, box = _prepared(tmp_path)
    e = {s: hf_energy(tt.preview_frame(src, prep, 2, s), box) for s in (0, 1.0, 2.0, 3.0)}
    assert e[0] < e[1.0] < e[2.0] < e[3.0]
    orig = next(iter(sk.frames(src)))
    assert np.array_equal(tt.preview_frame(src, prep, 0, 0), orig)


def test_preview_clamps_and_rejects_bad_frame(tmp_path):
    src, prep, box = _prepared(tmp_path)
    assert np.array_equal(tt.preview_frame(src, prep, 1, 99), tt.preview_frame(src, prep, 1, 3.0))
    with pytest.raises(IndexError):
        tt.preview_frame(src, prep, 6, 2.0)
    with pytest.raises(IndexError):
        tt.preview_frame(src, prep, -1, 2.0)


def test_prepare_refuses_when_no_face(tmp_path):
    src = make_video(tmp_path / "s.mp4", SKIN, n=4, fps=24, w=64, h=64)
    ref = tmp_path / "r.png"; cv2.imwrite(str(ref), np.full((64, 64, 3), 120, np.uint8))
    with pytest.raises(ValueError):
        tt.prepare(src, ref, detector=lambda f: (None, 0.0))
