"""Check a job's proc dir AFTER cover_head_files: how much of every detected face the new mask covers, and a tile of the
background video (the original pixels that survive) at a few frames. Usage: python tools/check_head.py proc_dir section.mp4 [tile.png]"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import swapkit as sk


def main(proc, section, tile=None):
    proc = Path(proc)
    mk = [f.max(axis=2) for f in sk.frames(proc / "src_mask.mp4")]
    old = [f.max(axis=2) for f in sk.frames(proc / "src_mask_nohead.mp4")]
    bg = list(sk.frames(proc / "src_bg.mp4"))
    info, sinfo = sk.probe(proc / "src_mask.mp4"), sk.probe(section)
    sec = list(sk.frames(section))
    sx, sy = info["width"] / sec[0].shape[1], info["height"] / sec[0].shape[0]
    cov, det = [], 0
    for j in range(len(mk)):
        si = min(len(sec) - 1, int(round(j / info["fps"] * sinfo["fps"] - 1e-9)))
        f = sk.detect_face_box(sec[si])
        if f is None:
            continue
        det += 1
        x, y, w, h = [int(round(v)) for v in (f[0] * sx, f[1] * sy, f[2] * sx, f[3] * sy)]
        x, y = max(0, x), max(0, y)
        cov.append(float((mk[j][y:y + h, x:x + w] > 127).mean()))
    top = int(mk[0].shape[0] * 0.31)
    print(f"frames {len(mk)}, faces detected {det}")
    print(f"detected face covered by new mask: min {min(cov):.3f} mean {np.mean(cov):.3f}")
    print(f"top 31% of frame marked: old {np.mean([(m[:top] > 127).mean() for m in old]):.3f} new {np.mean([(m[:top] > 127).mean() for m in mk]):.3f}")
    print("extension rows per frame (min/mean/max):", [int(v) for v in (min(((n > 127).any(axis=1).argmax() for n in mk)), 0, 0)][:1],
          "new mask first marked row, mean", round(float(np.mean([(m > 127).any(axis=1).argmax() for m in mk])), 1),
          "old", round(float(np.mean([(m > 127).any(axis=1).argmax() for m in old])), 1))
    if tile:
        picks = [0, 40, 80, 100, 125, 147]
        row = [cv2.resize(bg[min(i, len(bg) - 1)], (224, 384)) for i in picks]
        cv2.imwrite(tile, np.hstack(row))


if __name__ == "__main__":
    main(*sys.argv[1:4])
