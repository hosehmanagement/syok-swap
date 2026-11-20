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


def test_mux_original_audio_and_refusal(tmp_path):
    orig = make_video(tmp_path / "o.mp4", (1, 2, 3), n=48, fps=24, audio=True)
    vid = make_video(tmp_path / "v.mp4", (9, 9, 9), n=48, fps=24)
    out, had = sk.mux_original_audio(vid, orig, tmp_path / "f.mp4")
    assert had and sk.probe(out)["has_audio"]
    short = make_video(tmp_path / "s.mp4", (9, 9, 9), n=12, fps=24)
    with pytest.raises(ValueError):
        sk.mux_original_audio(short, orig, tmp_path / "x.mp4")
