"""Build-time patch for Wan2.2's wan/modules/animate/preprocess/utils.py (Apache-2.0), get_aug_mask().

get_aug_mask() splits the person's bounding box into a w_len x h_len grid with cells of int(width / w_len) by
int(height / h_len) pixels and fills every cell that touches the mask. When the box is narrower than w_len or shorter than
h_len pixels (a person leaving the frame, a tiny SAM2 speck) a cell size is 0 and range(..., 0) raises, killing the mask
step. This makes a cell at least 1 pixel; for every box big enough to work today the result is unchanged.

Run once inside the image after Wan is cloned:  python /opt/patch_wan_mask.py /opt/wan/wan/modules/animate/preprocess/utils.py
"""
import sys
from pathlib import Path

OLD = ("    w_slice = np.int32(bbox_wh[0] / w_len)\n"
       "    h_slice = np.int32(bbox_wh[1] / h_len)\n")
NEW = ("    w_slice = max(1, np.int32(bbox_wh[0] / w_len))  # syok-swap: never a 0-px cell\n"
       "    h_slice = max(1, np.int32(bbox_wh[1] / h_len))\n")


def patch_text(text):
    if NEW in text:
        return text
    if text.count(OLD) != 1:
        raise SystemExit("Wan preprocess utils.py layout changed - update patch_wan_mask.py")
    return text.replace(OLD, NEW)


if __name__ == "__main__":
    p = Path(sys.argv[1])
    p.write_text(patch_text(p.read_text()))
    print("patched", p)
