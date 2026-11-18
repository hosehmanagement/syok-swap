"""Measure and show what texture_transfer did.  python tools/tt_report.py before.mp4 after.mp4 ref.png out.png [frame ...]

Prints, per frame, the fine-detail energy of the lower-face skin (cheeks/chin, where stubble lives):
before, after, and the reference photo scaled to the same face size. 'carried' = after / reference.
The picture is  [reference | before | after]  per frame, 2x enlarged, so pores can be judged by eye.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import texture_transfer as tt


def lower_face_energy(bgr, box, sigma=1.2):
    x, y, w, h = [int(v) for v in box]
    X0, Y0, X1, Y1 = x + int(.15 * w), y + int(.45 * h), x + int(.85 * w), y + int(1.0 * h)
    roi = bgr[max(0, Y0):Y1, max(0, X0):X1]
    if roi.size == 0:
        return float("nan")
    m = tt.skin_prob(roi) > 0.5
    if m.sum() < 50:
        return float("nan")
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hf = g - cv2.GaussianBlur(g, (0, 0), sigma)
    return float(hf[m].std())


def crop(bgr, box, scale=2):
    x, y, w, h = [int(v) for v in tt.pad_box(box)]
    H, W = bgr.shape[:2]
    c = bgr[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
    c = cv2.resize(c, (320, 380), interpolation=cv2.INTER_AREA if c.shape[1] > 320 else cv2.INTER_CUBIC)
    return cv2.resize(c, None, fx=scale / 2, fy=scale / 2, interpolation=cv2.INTER_CUBIC)


def frame_at(path, i):
    c = cv2.VideoCapture(str(path)); c.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = c.read(); c.release()
    return f if ok else None


def main(before, after, ref_path, out, idx):
    ref = cv2.imread(ref_path)
    rbox, _ = tt.detect_face(ref)
    rows, table = [], []
    for i in idx:
        fb, fa = frame_at(before, i), frame_at(after, i)
        box, wgt = tt.detect_face(fb)
        if box is None:
            table.append((i, None)); continue
        # reference scaled so its face box has the same width as the video face box
        s = box[2] / rbox[2]
        rs = cv2.resize(ref, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        rb = tuple(v * s for v in rbox)
        eb, ea, er = lower_face_energy(fb, box), lower_face_energy(fa, box), lower_face_energy(rs, rb)
        table.append((i, round(eb, 2), round(ea, 2), round(er, 2), round(ea / er, 2) if er else None, wgt))
        rows.append(np.hstack([crop(rs, rb), crop(fb, box), crop(fa, box)]))
    for t in table:
        print(t)
    print("columns: frame, before, after, reference, carried=after/reference, detector weight (1 frontal, .7 profile)")
    if rows:
        cv2.imwrite(out, np.vstack(rows))
        print("wrote", out)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], a[4], [int(v) for v in a[5:]] or [10, 25, 40, 60, 80, 100, 120])
