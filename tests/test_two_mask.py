"""Two-mask composite: Wan redraws inside a loose mask, the final clip keeps the ORIGINAL pixels everywhere the new person is not."""
import sys
from pathlib import Path
import numpy as np, cv2, pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import swapkit as sk
import patch_wan_raw_mask as pr

H, W = 144, 256
rng = np.random.default_rng(7)
HEDGE = rng.integers(40, 110, (H, W, 3)).astype(np.uint8)
HEDGE[..., 1] = np.clip(HEDGE[..., 1] + 60, 0, 255)          # noisy green "hedge"
OLD = (30, 30, 200)       # old person: red (BGR)
NEW = (200, 120, 20)      # new person: blue-ish


def rect(box):
    m = np.zeros((H, W), bool)
    m[box[1]:box[3], box[0]:box[2]] = True
    return m


def write(path, frs, fps=24):
    w = sk.Writer(path, frs[0].shape[1], frs[0].shape[0], fps)
    for f in frs:
        w.put(f)
    w.close()
    return path


def scene(tmp_path, n=6, shift=0, redraw_bg=False, new_wide=False):
    old = rect((100, 30, 140, 130))                       # old person
    grid = rect((80, 20, 160, 134))                       # loose generation mask (bigger than the person)
    new_box = (92, 30, 172, 130) if new_wide else (100, 30, 140, 130)
    orig = HEDGE.copy(); orig[old] = OLD
    gen = HEDGE.astype(np.int32).copy()
    if redraw_bg:                                          # Wan redrew the hedge: other leaves, same overall colour
        gen = np.clip(gen + rng.integers(-35, 35, gen.shape), 0, 255)
    gen = np.clip(gen + shift, 0, 255).astype(np.uint8)
    nb = rect(new_box)
    gen[nb] = NEW
    g = np.repeat((grid * 255).astype(np.uint8)[..., None], 3, axis=2)
    c = np.repeat((old * 255).astype(np.uint8)[..., None], 3, axis=2)
    return (write(tmp_path / "o.mp4", [orig] * n), write(tmp_path / "g.mp4", [gen] * n),
            write(tmp_path / "m.mp4", [g] * n), write(tmp_path / "c.mp4", [c] * n), orig, nb)


def run(tmp_path, **kw):
    o, g, m, c, orig, nb = scene(tmp_path, **kw)
    out = tmp_path / "out.mp4"
    n = sk.composite_two_mask(o, g, m, c, out, feather=3)
    f = next(sk.frames(out))
    return n, f, next(sk.frames(o)), nb          # the DECODED input frame: pure noise does not survive H.264 unchanged


def err(a, b):
    return float(np.abs(a.astype(int) - b.astype(int)).mean())


def test_background_outside_the_mask_is_the_original_even_when_wan_redrew_it(tmp_path):
    n, f, orig, nb = run(tmp_path, redraw_bg=True, shift=9)
    assert n == 6
    far = ~cv2.dilate(rect((80, 20, 160, 134)).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    assert err(f[far], orig[far]) < 6                      # codec only, not Wan's redrawn / shifted hedge


def test_new_person_is_generated_and_the_old_one_is_gone(tmp_path):
    _, f, orig, nb = run(tmp_path)
    inner = rect((108, 40, 132, 120))
    assert np.abs(f[inner].astype(int) - np.array(NEW)).mean() < 25
    assert np.abs(f[inner].astype(int) - np.array(OLD)).mean() > 100


def test_ring_cells_Wan_redrew_keep_the_original_hedge(tmp_path):
    """Grid cells next to the person hold only hedge. Wan redrew the leaves; the final clip keeps the original leaves."""
    _, f, orig, nb = run(tmp_path, redraw_bg=True)
    ring = rect((84, 30, 96, 120))                         # inside the grid mask, outside the old person's outline
    assert err(f[ring], orig[ring]) < 8


def test_a_new_body_wider_than_the_old_one_is_not_cut_off(tmp_path):
    _, f, orig, nb = run(tmp_path, new_wide=True)
    wider = rect((95, 40, 99, 120))                        # new body, outside the old person's outline
    assert np.abs(f[wider].astype(int) - np.array(NEW)).mean() < 40


def test_vae_colour_shift_is_removed_outside_the_person(tmp_path):
    _, f, orig, nb = run(tmp_path, shift=14)
    near = rect((10, 10, 60, 60))
    assert err(f[near], orig[near]) < 6


def test_colour_gain_bias_needs_enough_background_and_is_smoothed():
    o = np.full((60, 60, 3), 100, np.uint8); g = np.full((60, 60, 3), 100, np.uint8)
    g[:] = o + 10
    none = np.zeros((60, 60), bool)
    gain, bias = sk.colour_gain_bias(o, g, none)
    assert (gain == 1).all() and (bias == 0).all()         # nothing to measure: untouched
    o2 = np.random.default_rng(1).integers(0, 255, (60, 60, 3)).astype(np.uint8)
    g2 = np.clip(o2.astype(int) + 10, 0, 255).astype(np.uint8)
    gain, bias = sk.colour_gain_bias(o2, g2, np.ones((60, 60), bool))
    assert abs(gain.mean() - 1) < 0.05 and -14 < bias.mean() < -6
    gain2, bias2 = sk.colour_gain_bias(o2, g2, np.ones((60, 60), bool), prev=(gain, bias), smooth=0.4)
    assert np.allclose(gain2, gain, atol=0.05)


def test_empty_core_still_works_from_the_colour_difference(tmp_path):
    o, g, m, c, orig, nb = scene(tmp_path, redraw_bg=True)
    z = write(tmp_path / "z.mp4", [np.zeros((H, W, 3), np.uint8)] * 6)
    out = tmp_path / "o2.mp4"
    sk.composite_two_mask(o, g, m, z, out, feather=3)
    f = next(sk.frames(out))
    assert np.abs(f[rect((108, 40, 132, 120))].astype(int) - np.array(NEW)).mean() < 40


def test_empty_video_is_refused(tmp_path):
    with pytest.raises(Exception):
        sk.composite_two_mask(tmp_path / "none.mp4", tmp_path / "none.mp4", tmp_path / "none.mp4", tmp_path / "none.mp4", tmp_path / "x.mp4")


def test_raw_mask_patch_applies_once_to_the_pinned_wan_file_and_is_idempotent():
    text = (Path(__file__).parent / "wan_pipeline_upstream.py.txt").read_text()
    out = pr.patch_text(text)
    assert out.count("src_mask_raw.mp4") == 1 and pr.patch_text(out) == out
    compile(out, "process_pipepline.py", "exec")
    with pytest.raises(SystemExit):
        pr.patch_text("nothing to patch here")


def test_blocky_grid_is_coarser_than_silhouette():
    assert sk.MASK_SHAPES["blocky"][0] < sk.MASK_SHAPES["silhouette"][0] and sk.MASK_SHAPES["blocky"][1] < sk.MASK_SHAPES["silhouette"][1]
    assert "6" in sk.preprocess_mask_args("normal", "blocky")


def test_cover_head_files_writes_the_core_mask_when_the_raw_mask_exists(tmp_path):
    n = 4
    def put(name, fr):
        return write(tmp_path / name, [fr] * n)
    person = np.zeros((H, W, 3), np.uint8); person[30:130, 100:140] = 255
    grid = np.zeros((H, W, 3), np.uint8); grid[20:134, 80:160] = 255
    bg = np.full((H, W, 3), 90, np.uint8); bg[grid.max(axis=2) > 0] = 0
    put("src_bg.mp4", bg); put("src_mask.mp4", grid); put("src_mask_raw.mp4", person); put("section.mp4", np.full((H, W, 3), 90, np.uint8))
    st = sk.cover_head_files(tmp_path, tmp_path / "section.mp4", face_fn=lambda fr: (112.0, 34.0, 16.0, 18.0), shape="silhouette")
    assert st["has_core"] is True and (tmp_path / "src_core.mp4").exists()
    core = next(sk.frames(tmp_path / "src_core.mp4")).max(axis=2) > 127
    assert core[80, 120] and not core[80, 90]               # person's outline, not the grid


def test_without_the_raw_mask_nothing_new_is_written(tmp_path):
    n = 3
    grid = np.zeros((H, W, 3), np.uint8); grid[20:134, 80:160] = 255
    write(tmp_path / "src_bg.mp4", [np.full((H, W, 3), 50, np.uint8)] * n); write(tmp_path / "src_mask.mp4", [grid] * n)
    write(tmp_path / "section.mp4", [np.full((H, W, 3), 50, np.uint8)] * n)
    st = sk.cover_head_files(tmp_path, tmp_path / "section.mp4", face_fn=lambda fr: (112.0, 34.0, 16.0, 18.0), shape="silhouette")
    assert st["has_core"] is False and not (tmp_path / "src_core.mp4").exists()
