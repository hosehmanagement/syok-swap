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


def test_normalize_mask_inverts_when_background_is_white(tmp_path):
    # white background, black person box => mean coverage > 0.5 => must be inverted so the person is white
    m = make_video(tmp_path / "m.mp4", (255, 255, 255), box=None)
    wr = sk.Writer(tmp_path / "w.mp4", 64, 48, 24)
    for _ in range(12):
        f = np.full((48, 64, 3), 255, np.uint8); f[10:40, 20:40] = 0; wr.put(f)
    wr.close()
    out, inv = sk.normalize_mask(tmp_path / "w.mp4", tmp_path / "n.mp4")
    assert inv
    f = next(sk.frames(out))
    assert f[25, 30, 0] > 200 and f[2, 2, 0] < 50
    out2, inv2 = sk.normalize_mask(make_video(tmp_path / "k.mp4", (0, 0, 0), box=(10, 10, 40, 40)), tmp_path / "n2.mp4")
    assert not inv2


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
