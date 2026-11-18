import sys, subprocess, tempfile
from pathlib import Path
import numpy as np, cv2, pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import swapkit as sk


def make_video(path, color, n=24, fps=24, w=64, h=48, box=None, audio=False):
    wr = sk.Writer(path.with_suffix(".v.mp4") if audio else path, w, h, fps)
    for _ in range(n):
        f = np.zeros((h, w, 3), np.uint8); f[:] = color
        if box: f[box[1]:box[3], box[0]:box[2]] = 255
        wr.put(f)
    wr.close()
    if audio:
        sk.sh(["ffmpeg", "-y", "-v", "error", "-i", str(path.with_suffix(".v.mp4")), "-f", "lavfi", "-i",
               f"sine=frequency=440:duration={n / fps}", "-c:v", "copy", "-c:a", "aac", "-shortest", str(path)])
    return path


def test_plan_section():
    assert sk.plan_section(10) == (0.0, 10.0)
    assert sk.plan_section(10, 2, 3) == (2.0, 3.0)
    assert sk.plan_section(10, 8, 99) == (8.0, 2.0)
    for bad in [(10, 9.5, 0), (40, 0, 0), (0, 0, 0)]:
        with pytest.raises(ValueError):
            sk.plan_section(*bad)


def test_margins_normal_is_wan_default():
    assert sk.MARGINS["normal"] == (3, 7)
    assert sk.MARGINS["tight"][0] < sk.MARGINS["loose"][0]


def test_composite_keeps_background_exact_and_fps_remap(tmp_path):
    orig = make_video(tmp_path / "o.mp4", (30, 60, 90), n=24, fps=24)
    gen = make_video(tmp_path / "g.mp4", (200, 10, 10), n=15, fps=15, w=32, h=24)  # other size AND fps
    msk = make_video(tmp_path / "m.mp4", (0, 0, 0), n=15, fps=15, w=32, h=24, box=(12, 8, 20, 16))
    out = tmp_path / "c.mp4"
    n = sk.composite_over_original(orig, gen, msk, out, feather=0)
    assert n == 24
    info = sk.probe(out)
    assert (info["width"], info["height"]) == (64, 48)
    f = next(sk.frames(out)); o = next(sk.frames(orig))
    assert np.abs(f[2, 2].astype(int) - o[2, 2].astype(int)).max() < 12      # background ~ original (codec only)
    assert f[24, 32, 0] > 150 and f[24, 32, 2] < 80   # centre is the generated colour (BGR blue-dominant)


def test_mask_stats_warns_on_empty(tmp_path):
    empty = make_video(tmp_path / "e.mp4", (0, 0, 0))
    assert sk.mask_stats(empty)["warnings"]
    ok = make_video(tmp_path / "k.mp4", (0, 0, 0), box=(10, 10, 40, 40))
    assert not sk.mask_stats(ok)["warnings"]


def test_mux_original_audio_and_refusal(tmp_path):
    orig = make_video(tmp_path / "o.mp4", (1, 2, 3), n=48, fps=24, audio=True)
    vid = make_video(tmp_path / "v.mp4", (9, 9, 9), n=48, fps=24)
    out, had = sk.mux_original_audio(vid, orig, tmp_path / "f.mp4")
    assert had and sk.probe(out)["has_audio"]
    short = make_video(tmp_path / "s.mp4", (9, 9, 9), n=12, fps=24)
    with pytest.raises(ValueError):
        sk.mux_original_audio(short, orig, tmp_path / "x.mp4")


def test_mask_preview_runs(tmp_path):
    orig = make_video(tmp_path / "o.mp4", (0, 0, 0))
    msk = make_video(tmp_path / "m.mp4", (0, 0, 0), box=(10, 10, 30, 30))
    sk.mask_preview(orig, msk, tmp_path / "p.mp4")
    f = next(sk.frames(tmp_path / "p.mp4"))
    assert f[20, 20, 2] > 60 and f[2, 2, 2] < 20


def test_normalize_mask_keeps_wan_polarity_even_for_close_up_portraits(tmp_path):
    # Regression for the real failure: a close-up portrait fills ~98% of the frame. Wan marks the person white, so a
    # mask that is mostly white must STAY white (it was wrongly inverted to a thin border before).
    wr = sk.Writer(tmp_path / "w.mp4", 64, 48, 24)
    for _ in range(12):
        f = np.full((48, 64, 3), 255, np.uint8); f[0:2, :] = 0; wr.put(f)
    wr.close()
    out, inv = sk.normalize_mask(tmp_path / "w.mp4", tmp_path / "n.mp4")
    assert not inv
    f = next(sk.frames(out))
    assert f[25, 30, 0] > 200 and f[0, 30, 0] < 50
    # and a small person (white box on black) is unchanged too
    out2, inv2 = sk.normalize_mask(make_video(tmp_path / "k.mp4", (0, 0, 0), box=(10, 10, 40, 40)), tmp_path / "n2.mp4")
    assert not inv2
    g = next(sk.frames(out2))
    assert g[25, 25, 0] > 200 and g[2, 2, 0] < 50

def test_cc_fallback_matches_contract():
    import torch
    from sam2_cc_fallback import get_connected_components_cpu
    m = torch.zeros(2, 1, 8, 8, dtype=torch.uint8)
    m[0, 0, 1:3, 1:3] = 1      # 4-pixel blob
    m[0, 0, 5:8, 5:8] = 1      # 9-pixel blob
    m[1, 0, 0, 0] = 1          # 1-pixel blob
    m[1, 0, 1, 1] = 1          # touches only diagonally -> separate under 4-connectivity
    labels, counts = get_connected_components_cpu(m)
    assert labels.shape == m.shape and counts.shape == m.shape
    assert counts[0, 0, 1, 1].item() == 4 and counts[0, 0, 6, 6].item() == 9
    assert counts[0, 0, 0, 0].item() == 0 and labels[0, 0, 0, 0].item() == 0
    assert labels[0, 0, 1, 1] != labels[0, 0, 6, 6]
    assert counts[1, 0, 0, 0].item() == 1 and counts[1, 0, 1, 1].item() == 1


def test_composite_full_res_uses_generated_size_and_keeps_detail(tmp_path):
    orig = make_video(tmp_path / "o.mp4", (10, 200, 10), n=24, fps=24, w=64, h=48)
    gen = make_video(tmp_path / "g.mp4", (200, 10, 10), n=24, fps=24, w=128, h=96)
    msk = make_video(tmp_path / "m.mp4", (0, 0, 0), n=24, fps=24, w=128, h=96, box=(40, 30, 88, 66))
    out = tmp_path / "c.mp4"
    n = sk.composite_over_original(orig, gen, msk, out, feather=0, full_res=True, blend_original=True)
    info = sk.probe(out)
    assert n == 24 and (info["width"], info["height"]) == (128, 96)
    f = next(sk.frames(out))
    assert f[48, 64, 0] > 150 and f[48, 64, 2] < 80          # centre is the generated colour
    assert f[3, 3, 1] > 150                                 # background is the (enlarged) original


def test_composite_full_res_default_never_lets_original_pixels_through(tmp_path):
    """Regression (2026-11-18): the mask was a coarse region, so the original person's head came back through the
    composite in head-turn frames. At full size the generated frame is used as it is, even where the mask is 0."""
    orig = make_video(tmp_path / "o.mp4", (10, 200, 10), n=24, fps=24, w=64, h=48)
    gen = make_video(tmp_path / "g.mp4", (200, 10, 10), n=24, fps=24, w=128, h=96)
    msk = make_video(tmp_path / "m.mp4", (0, 0, 0), n=24, fps=24, w=128, h=96, box=(40, 30, 88, 66))
    out = tmp_path / "c.mp4"
    sk.composite_over_original(orig, gen, msk, out, feather=7, full_res=True)
    f = next(sk.frames(out))
    assert f[3, 3, 0] > 150 and f[3, 3, 1] < 80             # corner is generated red, not original green
    assert f[90, 120, 0] > 150 and f[90, 120, 1] < 80
