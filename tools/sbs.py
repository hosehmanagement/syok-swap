"""Before/after contact sheet: python tools/sbs.py original.mp4 swapped.mp4 out.png [t_ms t_ms ...]"""
import sys

import cv2
import numpy as np


def frame(path, t_ms):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read {path} at {t_ms} ms")
    return f


orig, swapped, out = sys.argv[1:4]
times = [float(x) for x in sys.argv[4:]] or [500, 1900, 3200]
rows = []
for t in times:
    a = frame(orig, t)
    b = cv2.resize(frame(swapped, t), (a.shape[1], a.shape[0]))
    rows.append(np.hstack([a, b]))
sheet = np.vstack(rows)
cv2.imwrite(out, cv2.resize(sheet, None, fx=0.5, fy=0.5))
print("wrote", out)
