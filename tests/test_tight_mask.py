"""Silhouette mask (20 Nov 2026): the red "will be replaced" area should follow the person, not fill their bounding box.

Wan's own mask code (tests/wan_mask_upstream.py, a verbatim copy of the pinned Wan2.2 file, with patch_wan_mask.py applied the
same way the image build applies it) runs here on SYNTHETIC frames: a moving ellipse "person" with outstretched arms and a
round head on a textured "hedge", where the simulated SAM2 mask misses the head (the real failure seen 2026-11-18). No real
person, no GPU, no SAM2: what SAM2 itself returns on real footage is NOT tested here."""
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import patch_wan_mask
import swapkit as sk

UPSTREAM = (HERE / "wan_mask_upstream.py").read_text()


def _module(text, name):
    m = types.ModuleType(name)
    exec(compile(text, name, "exec"), m.__dict__)
    return m


wan_orig = _module(UPSTREAM, "wan_orig")
wan = _module(patch_wan_mask.patch_text(UPSTREAM), "wan_patched")

W, H, N, FPS = 192, 256, 12, 24
HEAD_COLOUR = (40, 40, 230)             # BGR red-ish head, so the fake face detector can find it
R = 16                                  # head radius


def person(i):
    """Ground truth for frame i: (body mask, head mask, head centre)."""
    cx = 70 + 4 * i
    body = np.zeros((H, W), np.uint8)
    cv2.ellipse(body, (cx, 160), (22, 70), 0, 0, 360, 1, -1)       # torso and legs
    cv2.ellipse(body, (cx, 112), (50, 6), 0, 0, 360, 1, -1)        # outstretched arms
    head = np.zeros((H, W), np.uint8)
    cv2.circle(head, (cx, 70), R, 1, -1)
    return body.astype(bool), head.astype(bool), (cx, 70)


def hedge(seed=3):
    rng = np.random.default_rng(seed)
    f = np.empty((H, W, 3), np.uint8)
    f[..., 0] = 30
    f[..., 1] = 110 + (rng.integers(0, 60, (H, W)))                # green, noisy like leaves
    f[..., 2] = 40
    f[::7, :, 1] = 60                                              # some structure
    return f


def frame(i):
    f = hedge()
    body, head, _ = person(i)
    f[body] = (120, 90, 60)
    f[head] = HEAD_COLOUR
    return f


def fake_face(fr):
    """Face box (x, y, w, h) from the head colour, the lower 2/3 of the head (the hair is above it), or None."""
    m = (fr[..., 2] > 170) & (fr[..., 1] < 90) & (fr[..., 0] < 90)
    ys, xs = np.nonzero(m)
    if len(ys) < 20:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    hw, hh = x1 - x0, y1 - y0
    return (float(x0 + 0.2 * hw), float(y0 + 0.35 * hh), float(0.6 * hw), float(0.6 * hh))


def wan_proc(tmp_path, mask_shape, margin="normal", sec_scale=2):
    """Write what Wan's preprocess writes in replace mode (process_pipepline.py lines 102-124 at the pinned commit), with its
    own get_mask_body_img / get_aug_mask, plus the original section at another size. SAM2's mask = the body WITHOUT the head."""
    it, k = sk.MARGINS[margin]
    wl, hl = sk.MASK_SHAPES[mask_shape]
    proc = tmp_path / mask_shape
    proc.mkdir()
    wm = sk.Writer(proc / "src_mask.mp4", W, H, FPS)
    wb = sk.Writer(proc / "src_bg.mp4", W, H, FPS)
    ws = sk.Writer(proc / "section.mp4", W * sec_scale, H * sec_scale, FPS)
    for i in range(N):
        fr = frame(i)
        body, _, _ = person(i)
        _, dil = wan.get_mask_body_img(fr, body.astype(np.uint8), iterations=it, k=k)
        aug = wan.get_aug_mask(dil, w_len=wl, h_len=hl)
        wb.put(fr * (1 - aug[:, :, None]))
        wm.put(np.stack([aug * 255] * 3, axis=2))
        ws.put(cv2.resize(fr, (W * sec_scale, H * sec_scale), interpolation=cv2.INTER_NEAREST))
    wm.close(); wb.close(); ws.close()
    return proc


def masks_of(path):
    return [f.max(axis=2) > 127 for f in sk.frames(path)]


def run(tmp_path, mask_shape, face_fn=fake_face, **kw):
    proc = wan_proc(tmp_path, mask_shape, **kw)
    st = sk.cover_head_files(proc, proc / "section.mp4", face_fn=face_fn, shape=mask_shape)
    return proc, st


def _erode(m, px=2):
    return cv2.erode(m.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)).astype(bool)


# ---- Wan's grid, and the build-time patch ----

def test_patch_matches_the_pinned_wan_file_and_is_idempotent():
    once = patch_wan_mask.patch_text(UPSTREAM)
    assert once != UPSTREAM and patch_wan_mask.patch_text(once) == once
    with pytest.raises(SystemExit):
        patch_wan_mask.patch_text(UPSTREAM.replace("np.int32(bbox_wh[0] / w_len)", "bbox_wh[0] // w_len"))


def test_wan_default_grid_1x1_is_the_whole_bounding_box():
    """The root cause: preprocess_data.py's --w_len/--h_len default to 1, so get_aug_mask fills the whole box."""
    body, _, _ = person(0)
    aug = wan.get_aug_mask(body.astype(np.uint8), w_len=1, h_len=1).astype(bool)
    ys, xs = np.nonzero(body)
    box = np.zeros_like(body)
    box[ys.min():ys.max(), xs.min():xs.max()] = True            # Wan's range() stops before the last row/column
    assert (aug == (box | body)).all()
    assert aug.mean() > 1.8 * body.mean()


def test_patched_grid_equals_upstream_on_every_mask_upstream_can_handle():
    rng = np.random.default_rng(0)
    for _ in range(40):
        m = np.zeros((96, 96), np.uint8)
        x, y = rng.integers(0, 60, 2)
        h, w = rng.integers(25, 36), rng.integers(15, 36)          # at least 20 x 10 px, so upstream does not crash
        m[y:y + h, x:x + w] = rng.random((h, w)) > 0.8
        m[y, x] = m[y + h - 1, x + w - 1] = 1
        for wl, hl in ((1, 1), (10, 20)):
            assert (wan.get_aug_mask(m.copy(), wl, hl) == wan_orig.get_aug_mask(m.copy(), wl, hl)).all()


def test_patched_grid_survives_a_tiny_mask_that_crashes_upstream():
    m = np.zeros((64, 64), np.uint8)
    m[10:15, 10:30] = 1                                          # 4 px tall: shorter than h_len=20
    with pytest.raises(ValueError):
        wan_orig.get_aug_mask(m.copy(), 10, 20)
    out = wan.get_aug_mask(m.copy(), 10, 20)
    assert (out[m.astype(bool)] == 1).all() and out.sum() == m.sum()


def test_preprocess_args_carry_the_grid_and_the_margin():
    assert sk.DEFAULT_MASK_SHAPE == "silhouette"
    assert sk.MASK_SHAPES["box"] == (1, 1)
    assert sk.preprocess_mask_args("normal") == ["--iterations", "3", "--k", "7", "--w_len", "10", "--h_len", "20"]
    assert sk.preprocess_mask_args("tight", "box") == ["--iterations", "2", "--k", "5", "--w_len", "1", "--h_len", "1"]
    for bad in (("huge", "box"), ("normal", "blob")):
        with pytest.raises(ValueError):
            sk.preprocess_mask_args(*bad)


# ---- the whole mask step on synthetic frames ----

@pytest.fixture(scope="module")
def both(tmp_path_factory):
    t = tmp_path_factory.mktemp("tight")
    return {"silhouette": run(t, "silhouette"), "box": run(t, "box")}


def test_silhouette_covers_much_less_than_the_box(both):
    _, s = both["silhouette"]
    _, b = both["box"]
    assert s["coverage"]["final_mean"] < 0.7 * b["coverage"]["final_mean"]
    # box_mean (what the box path would have covered, computed from the silhouette run's mask) matches the real box run
    assert abs(s["coverage"]["box_mean"] - b["coverage"]["final_mean"]) < 0.02
    assert abs(b["coverage"]["box_mean"] - b["coverage"]["final_mean"]) < 0.01
    assert s["coverage"]["wan_mean"] <= s["coverage"]["final_mean"]
    assert s["mask_shape"] == "silhouette" and s["head_shape"] == "ellipse" and b["head_shape"] == "rectangle"


@pytest.mark.parametrize("shape", ["silhouette", "box"])
def test_person_head_and_face_are_fully_inside_the_mask(both, shape):
    proc, st = both[shape]
    ms = masks_of(proc / "src_mask.mp4")
    assert len(ms) == N and st["faces_detected"] == N and st["frames_extended"] == N
    for i, m in enumerate(ms):
        body, head, _ = person(i)
        assert m[body].all(), f"body pixel outside the mask in frame {i}"
        assert m[head].all(), f"original head pixel outside the mask in frame {i}"
        x, y, w, h = fake_face(frame(i))
        assert m[int(y):int(np.ceil(y + h)) + 1, int(x):int(np.ceil(x + w)) + 1].all()


def test_background_corners_and_the_hedge_beside_the_body_stay_outside(both):
    proc, _ = both["silhouette"]
    bproc, _ = both["box"]
    for i, (m, mb) in enumerate(zip(masks_of(proc / "src_mask.mp4"), masks_of(bproc / "src_mask.mp4"))):
        for ys, xs in ((slice(0, 10), slice(0, 10)), (slice(0, 10), slice(-10, None)),
                       (slice(-10, None), slice(0, 10)), (slice(-10, None), slice(-10, None))):
            assert not m[ys, xs].any()
        cx = 70 + 4 * i
        for x in (cx - 44, cx + 44):                         # hedge beside the hips, under the arms
            assert not m[195:205, x - 1:x + 2].any(), f"hedge inside the silhouette mask in frame {i}"
            assert mb[195:205, x - 1:x + 2].all()            # ... and inside the old box (the reported bug)
        for x0 in (cx - 25, cx + 21):                        # the old head rectangle's top corners: hedge beside the hair
            assert not m[47:51, x0:x0 + 4].any()
            assert mb[47:51, x0:x0 + 4].all()


@pytest.mark.parametrize("shape", ["silhouette", "box"])
def test_background_video_stays_in_step_with_the_mask(both, shape):
    proc, _ = both[shape]
    ms = masks_of(proc / "src_mask.mp4")
    bg = list(sk.frames(proc / "src_bg.mp4"))
    assert len(bg) == len(ms) == N
    assert sk.probe(proc / "src_bg.mp4")["duration"] == pytest.approx(sk.probe(proc / "src_mask.mp4")["duration"], abs=1e-3)
    for i, (m, b) in enumerate(zip(ms, bg)):
        inside, outside = _erode(m), _erode(~m)
        assert b[inside].max() < 40                          # blacked out where Wan redraws (src_bg = frame * (1 - mask))
        diff = np.abs(b.astype(int) - frame(i).astype(int))[outside]
        assert diff.mean() < 12                              # the original picture everywhere else
    assert (proc / "src_mask_nohead.mp4").exists() and (proc / "src_bg_nohead.mp4").exists()


def test_undetected_frames_still_get_the_head(tmp_path):
    calls = {"n": 0}

    def flaky(fr):                                           # every third frame: no detection (profile, turned away)
        calls["n"] += 1
        return None if calls["n"] % 3 == 0 else fake_face(fr)

    proc, st = run(tmp_path, "silhouette", face_fn=flaky)
    assert st["faces_detected"] == N - N // 3
    for i, m in enumerate(masks_of(proc / "src_mask.mp4")):
        _, head, _ = person(i)
        assert m[head].all(), f"head left out in frame {i}"


def test_no_face_anywhere_is_refused_in_both_shapes(tmp_path):
    for shape in ("silhouette", "box"):
        proc = wan_proc(tmp_path, shape)
        with pytest.raises(ValueError):
            sk.cover_head_files(proc, proc / "section.mp4", face_fn=lambda fr: None, shape=shape)
    with pytest.raises(ValueError):
        sk.cover_head_files(tmp_path / "silhouette", tmp_path / "silhouette" / "section.mp4", face_fn=fake_face, shape="blob")


def test_head_ellipse_always_contains_the_face_box():
    m = np.zeros((200, 200), np.uint8)
    m[150:190, 80:120] = 255
    for f in [(90.0, 60.0, 20.0, 24.0), (10.0, 5.0, 40.0, 30.0), (150.0, 100.0, 45.0, 60.0)]:
        r = sk.head_region((200, 200), sk.head_ellipses([m], [f])[0])
        x, y, w, h = f
        assert r[int(y):int(y + h) + 1, int(x):int(x + w) + 1].all()
        assert r[max(0, int(y - 0.8 * h)), int(x + w / 2)]  # hair room above the face
    assert sk.head_ellipses([np.zeros((10, 10), np.uint8)], [(1.0, 1.0, 2.0, 2.0)]) == [None]


# ---- the job service and the web app ----

def test_mask_shape_in_the_job_request(tmp_path):
    pytest.importorskip("fastapi")
    import swap_service as ss
    from test_swap_service import Rig
    r = Rig(tmp_path)
    h = r.c.get("/health").json()
    assert h["mask_shapes"] == ["box", "silhouette"] and h["mask_shape"] == "silhouette"
    job = r.upload().json()["job"]
    assert "mask_shape" not in ss.request_of(r.root, job)                 # default: silhouette, request as before
    job = r.upload(mask_shape="Box").json()["job"]
    assert ss.request_of(r.root, job)["mask_shape"] == "box"
    n = r.calls["mask"]
    bad = r.upload(mask_shape="blob")
    assert bad.status_code == 400 and "Mask shape" in bad.json()["error"] and r.calls["mask"] == n


def test_modal_app_wiring():
    app = (HERE.parent / "app.py").read_text()
    core = app[app.index("def _preview_core("):app.index("def make_preview(")]
    assert "sk.preprocess_mask_args(margin, mask_shape)" in core and '"--replace_flag", *mask_args]' in core
    assert "sk.cover_head_files(d / \"proc\", sec, shape=mask_shape)" in core
    assert 'stats["box_mean"] = head["coverage"]["box_mean"]' in core
    mask_fn = app[app.index("def mask_job("):app.index("def swap_job(")]
    assert 'q.get("mask_shape")' in mask_fn
    img = app[app.index("image = ("):app.index("# The small CPU image")]
    patch = img.index('"/opt/patch_wan_mask.py", copy=True')
    assert img.index("git clone https://github.com/Wan-Video/Wan2.2") < patch
    assert "python /opt/patch_wan_mask.py /opt/wan/wan/modules/animate/preprocess/utils.py" in img
    assert patch < img.index('"swapkit.py", "/opt/swapkit.py"')    # copy=True build steps come before the plain mounts
