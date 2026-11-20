import sys
from pathlib import Path
import numpy as np, cv2, pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import swapkit as sk
import fitref as fr

H, W = 360, 640
rng = np.random.default_rng(3)
HEDGE = np.clip(rng.integers(40, 110, (H, W, 3)) + np.array([0, 60, 0]), 0, 255).astype(np.uint8)
BOX = (260, 60, 380, 330)           # the old person


def write(path, frames, fps=24):
    w = sk.Writer(path, frames[0].shape[1], frames[0].shape[0], fps)
    for f in frames:
        w.put(f)
    w.close()
    return path


def job(tmp_path, move=0):
    f = HEDGE.copy(); f[BOX[1]:BOX[3], BOX[0]:BOX[2]] = (30, 30, 200)
    g = HEDGE.copy(); g[BOX[1]:BOX[3], BOX[0] + move:BOX[2] + move] = (30, 30, 200)
    write(tmp_path / "section.mp4", [f, f, g])
    m = np.zeros((H, W, 3), np.uint8); m[BOX[1]:BOX[3], BOX[0]:BOX[2]] = 255
    m2 = np.zeros((H, W, 3), np.uint8); m2[BOX[1]:BOX[3], BOX[0] + move:BOX[2] + move] = 255
    write(tmp_path / "core.mp4", [m, m, m2])
    cv2.imwrite(str(tmp_path / "ref.jpg"), np.full((200, 120, 3), (200, 120, 20), np.uint8))
    return tmp_path


def paint_new_person(scene, photo, prompt):
    out = scene.copy()
    h, w = out.shape[:2]
    # the old person was red; the editor paints a blue person where red was
    red = (out[..., 2] > 150) & (out[..., 0] < 80)
    out[red] = (200, 120, 20)
    return out


def test_both_ends_fitted_and_reference_built(tmp_path):
    d = job(tmp_path, move=60)
    r = fr.run_fit(d, paint_new_person, d / "ref.jpg")
    assert r["state"] == "done" and r["frames"]["first"]["state"] == "done" and r["frames"]["last"]["state"] == "done"
    assert (d / "fit_first.png").exists() and (d / "fit_last.png").exists()
    ref = cv2.imread(str(d / "fit_ref.png"))
    assert ref is not None and ref.shape[1] > ref.shape[0] * 0.5          # two persons side by side
    assert "both ends" in r["message"]


def test_an_editor_that_redraws_the_scene_is_rejected(tmp_path):
    d = job(tmp_path)
    def scene_wrecker(scene, photo, prompt):
        return np.full_like(scene, (10, 200, 200))
    r = fr.run_fit(d, scene_wrecker, d / "ref.jpg")
    assert r["state"] == "skipped" and not (d / "fit_ref.png").exists()
    assert r["frames"]["first"]["state"] == "rejected" and "scene" in r["frames"]["first"]["message"]
    assert "your own photo is used" in r["message"]


def test_an_editor_that_changes_nothing_is_rejected(tmp_path):
    d = job(tmp_path)
    r = fr.run_fit(d, lambda scene, photo, prompt: scene.copy(), d / "ref.jpg")
    assert r["state"] == "skipped" and "did not replace" in r["frames"]["first"]["message"]


def test_one_frame_failing_keeps_the_other(tmp_path):
    d = job(tmp_path)
    calls = []
    def flaky(scene, photo, prompt):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("out of memory")
        return paint_new_person(scene, photo, prompt)
    r = fr.run_fit(d, flaky, d / "ref.jpg")
    assert r["state"] == "done" and r["frames"]["last"]["state"] == "error" and "one end" in r["message"]
    assert (d / "fit_first.png").exists() and not (d / "fit_last.png").exists()


def test_editor_output_of_another_size_is_resized_then_checked(tmp_path):
    d = job(tmp_path)
    def other_size(scene, photo, prompt):
        small = cv2.resize(paint_new_person(scene, photo, prompt), (scene.shape[1] // 2 + 16, scene.shape[0] // 2 + 16))
        return small
    r = fr.run_fit(d, other_size, d / "ref.jpg")
    assert r["frames"]["first"]["state"] in ("done", "rejected")          # resized, then judged; never a crash


def test_crop_box_pads_clamps_and_keeps_a_minimum():
    x0, y0, x1, y1 = fr.crop_box((250, 50, 390, 340), (360, 640))
    assert 0 <= x0 < 250 and x1 > 390 and y0 == 0 and y1 == 360
    x0, y0, x1, y1 = fr.crop_box((300, 150, 340, 200), (360, 640), pad=0.0, min_side=320)
    assert min(x1 - x0, y1 - y0) >= 320 - 1
    x0, y0, x1, y1 = fr.crop_box((0, 0, 40, 40), (100, 100), min_side=320)
    assert (x0, y0, x1, y1) == (0, 0, 100, 100)                            # photo smaller than the minimum: the whole photo


def test_person_bbox_and_empty():
    m = np.zeros((10, 10), bool); m[2:5, 3:7] = True
    assert fr.person_bbox(m) == (3, 2, 7, 5) and fr.person_bbox(np.zeros((4, 4), bool)) is None


def test_make_fit_reference_layout_and_limits():
    a = np.full((300, 100, 3), 50, np.uint8); b = np.full((200, 80, 3), 120, np.uint8)
    png = fr.make_fit_reference([a, b], max_side=400)
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) <= 400 and img.shape[0] > 100
    with pytest.raises(ValueError):
        fr.make_fit_reference([])


def test_no_core_falls_back_to_the_grid_mask_and_empty_video_is_skipped(tmp_path):
    d = job(tmp_path)
    (d / "core.mp4").rename(d / "mask.mp4")
    assert fr.run_fit(d, paint_new_person, d / "ref.jpg")["state"] == "done"
    empty = tmp_path / "e"; empty.mkdir()
    r = fr.run_fit(empty, paint_new_person, d / "ref.jpg")
    assert r["state"] == "skipped" and "no frames" in r["message"]
