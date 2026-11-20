"""Build-time patch for Wan2.2's wan/modules/animate/preprocess/process_pipepline.py (Apache-2.0), replace mode.

Wan writes src_mask.mp4 AFTER dilating SAM2's person mask and filling a coarse grid (get_aug_mask), so the file no longer says
where the person actually is. Syok Swap's composite needs the person's own outline (a tight matte) next to that loose mask, so
this makes Wan also write SAM2's raw per-frame mask as src_mask_raw.mp4 (white = person), in the same folder, same fps.
Nothing Wan itself reads is changed.

Run once inside the image after Wan is cloned:  python /opt/patch_wan_raw_mask.py /opt/wan/wan/modules/animate/preprocess/process_pipepline.py
"""
import sys
from pathlib import Path

OLD = ("            mpy.ImageSequenceClip(aug_masks_new, fps=fps).write_videofile(src_mask_path)\n")
NEW = (OLD +
       "            # syok-swap: also keep SAM2's own person mask (before dilation and grid), for the tight composite matte\n"
       "            raw_masks_new = [np.stack([m.astype(np.uint8) * 255] * 3, axis=2) for m in masks]\n"
       "            mpy.ImageSequenceClip(raw_masks_new, fps=fps).write_videofile(os.path.join(output_path, 'src_mask_raw.mp4'))\n")


def patch_text(text):
    if "src_mask_raw.mp4" in text:
        return text
    if text.count(OLD) != 1:
        raise SystemExit("Wan process_pipepline.py layout changed - update patch_wan_raw_mask.py")
    return text.replace(OLD, NEW)


if __name__ == "__main__":
    p = Path(sys.argv[1])
    p.write_text(patch_text(p.read_text()))
    print("patched", p)
