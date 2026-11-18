"""Dump full frames (with the detected face box drawn) from before/after videos.

python tools/frame_dump.py before.mp4 after.mp4 out_prefix frame [frame ...]
Writes <out_prefix><frame>.png = [before with box | after], and prints the detection.
"""
import sys

import cv2

import texture_transfer as tt


def grab(path, n):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def main(argv):
    before, after, prefix = argv[0], argv[1], argv[2]
    for n in (int(x) for x in argv[3:]):
        a, b = grab(before, n), grab(after, n)
        if a is None or b is None:
            print(n, "frame missing")
            continue
        det = tt.detect_face(a)
        shown = a.copy()
        if det is not None:
            x, y, w, h = det[0]
            cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print(n, det, a.shape)
        cv2.imwrite(f"{prefix}{n}.png", cv2.hconcat([shown, b]))


if __name__ == "__main__":
    main(sys.argv[1:])
