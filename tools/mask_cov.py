"""Per-frame share of the image where the final composite differs from the raw generation (i.e. where original pixels
were kept because the mask was 0), split into the head band and the rest.

    python tools/mask_cov.py swap720_raw.mp4 swap720_full.mp4 [step]
"""
import sys

import cv2
import numpy as np

raw, full = cv2.VideoCapture(sys.argv[1]), cv2.VideoCapture(sys.argv[2])
step = int(sys.argv[3]) if len(sys.argv) > 3 else 10
i = 0
print("frame  differs_total  differs_head_band(y 40-100%)")
while True:
    a, b = raw.read()[1], full.read()[1]
    if a is None or b is None:
        break
    if i % step == 0 or i == 118:
        d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) > 25
        h = d.shape[0]
        print(f"{i:5d}  {d.mean():.2f}  {d[int(h * .4):].mean():.2f}")
    i += 1
