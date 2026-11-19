"""Check that the slider preview equals the full render at the same strength.

    python tools/check_slider.py swapped.mp4 reference.png out.txt FRAME:render_at_2.0.mp4:2.0 [FRAME:render.mp4:STRENGTH ...]

For each FRAME:VIDEO:STRENGTH it prints the mean absolute pixel difference (0-255) between
texture_transfer.preview_frame(src, prepared, FRAME, STRENGTH) and frame FRAME of VIDEO (a clip made by
process_video/render_prepared at the same strength). A small number (codec noise, about 1-3) means the slider
shows the same picture the final video will contain.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import swapkit as sk
import texture_transfer as tt


def main(argv):
    src, ref, out = argv[0], argv[1], argv[2]
    prep = tt.prepare(src, ref)
    lines = [f"prepared frames={prep['frames']} detected={prep['detected']}"]
    cache = {}
    for spec in argv[3:]:
        frame, video, strength = spec.split(":")
        frame, strength = int(frame), float(strength)
        if video not in cache:
            cache[video] = list(sk.frames(video))
        prev = tt.preview_frame(src, prep, frame, strength)
        diff = float(np.abs(prev.astype(int) - cache[video][frame].astype(int)).mean())
        lines.append(f"frame={frame} strength={strength} video={video} mean_abs_diff={diff:.3f}")
    Path(out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
