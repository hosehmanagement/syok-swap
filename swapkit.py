"""Syok Swap helpers: CPU-only video plumbing around Wan2.2-Animate (Apache-2.0).

Ideas taken from the two reference projects, re-implemented here (no code copied):
  * Openjutsu: margin presets (tight/normal/loose), section (start/length) of the source,
    keep the swap region only and leave the rest of the picture alone.
  * open-genjutsu: review the mask BEFORE paying for generation, and deliver the
    ORIGINAL soundtrack untouched on the result.
"""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

# (iterations, kernel) for Wan's mask dilation in preprocess_data.py. normal == Wan's default.
MARGINS = {"tight": (2, 5), "normal": (3, 7), "loose": (4, 9)}
RESOLUTIONS = {"480p": (832, 480), "720p": (1280, 720)}  # target AREA as [w, h]
MAX_SECONDS = 30.0


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-800:]}")
    return r.stdout


def probe(path):
    """Return dict(duration, fps, width, height, has_audio)."""
    j = json.loads(sh(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    num, den = v["avg_frame_rate"].split("/")
    return {
        "duration": float(j["format"]["duration"]),
        "fps": float(num) / float(den) if float(den) else 30.0,
        "width": int(v["width"]),
        "height": int(v["height"]),
        "has_audio": any(s["codec_type"] == "audio" for s in j["streams"]),
    }


def plan_section(duration, start=0.0, length=0.0):
    """Clamp a (start, length) request to the clip. length 0 = to the end. Returns (start, length)."""
    if duration <= 0:
        raise ValueError("Source video has no duration.")
    start = max(0.0, float(start))
    if start >= duration - 1.0:
        raise ValueError("Section starts too close to the end of the video.")
    end = duration if not length else min(duration, start + float(length))
    length = end - start
    if length < 1.0:
        raise ValueError("Section must be at least 1 second.")
    if length > MAX_SECONDS:
        raise ValueError(f"Section is {length:.1f}s; the limit is {MAX_SECONDS:.0f}s per job.")
    return round(start, 3), round(length, 3)


def cut_section(src, dst, start, length):
    """Re-encode the chosen part of the source, with its audio, so later steps share one timeline."""
    sh(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(length), "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", str(dst)])
    return dst


def frames(path):
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield f  # BGR
    finally:
        cap.release()


class Writer:
    """Pipe BGR frames to ffmpeg (H.264, yuv420p)."""

    def __init__(self, path, w, h, fps):
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
             "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)

    def put(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait():
            raise RuntimeError("ffmpeg writer failed")


def source_frame_index(i, out_fps, src_frames, src_fps):
    """Frame of the source that is on screen at output frame i."""
    return min(src_frames - 1, int(round(i / out_fps * src_fps - 1e-9)))


def composite_over_original(original, generated, mask, out, feather=7, full_res=False, blend_original=None):
    """Paste the generated person over the ORIGINAL frames at the original size.

    Everything outside the (feathered) mask is the original pixels, untouched. Frames are matched by
    time, so a generated clip at another fps still lines up. Returns the number of frames written.

    full_res=True writes the result at the GENERATED size instead, so the generated person keeps all
    its detail (the original background is enlarged to match, so it is softer than the person).

    blend_original: default False at every size (2026-11-18: final.mp4 at 448x768 showed the same bug). The mask is a coarse region, not a person silhouette
    (on the 720p job the top ~31% of the frame was never masked), so original pixels outside it can be the ORIGINAL
    PERSON's head: found 2026-11-18, the original red-haired head came back through the composite in the head-turn
    frames (100-118) while the raw generation was correct. At full size the enlarged original is softer than the
    generated frame anyway, so the generated frames are used as they are; pass blend_original=True to opt back in.
    """
    if blend_original is None:
        blend_original = False
    info = probe(original)
    ow, oh, ofps = info["width"], info["height"], info["fps"]
    orig = list(frames(original))
    gen = list(frames(generated))
    msk = list(frames(mask))
    if not (orig and gen and msk):
        raise ValueError("Empty video passed to composite.")
    ginfo = probe(generated)
    gfps = ginfo["fps"]
    if full_res:
        ow, oh = ginfo["width"], ginfo["height"]
    n = int(round(len(gen) / gfps * ofps))
    k = (feather * 2 + 1) | 1
    w = Writer(out, ow, oh, ofps)
    try:
        for i in range(n):
            o = orig[min(len(orig) - 1, i)]
            if o.shape[1] != ow or o.shape[0] != oh:
                o = cv2.resize(o, (ow, oh), interpolation=cv2.INTER_CUBIC)
            gi = min(len(gen) - 1, int(round(i / ofps * gfps - 1e-9)))
            g = cv2.resize(gen[gi], (ow, oh), interpolation=cv2.INTER_CUBIC)
            m = cv2.resize(msk[min(len(msk) - 1, gi)], (ow, oh), interpolation=cv2.INTER_LINEAR)
            m = (m.max(axis=2) if m.ndim == 3 else m).astype(np.float32) / 255.0
            m = (m > 0.5).astype(np.float32)
            if not blend_original:
                m = np.ones_like(m)
            if feather and blend_original:
                m = cv2.GaussianBlur(m, (k, k), 0)
            m = m[..., None]
            w.put(o.astype(np.float32) * (1 - m) + g.astype(np.float32) * m)
    finally:
        w.close()
    return n


def mask_preview(original, mask, out, color=(0, 0, 255), alpha=0.45):
    """Red overlay of the area that will be regenerated, so the user can approve before GPU spend."""
    info = probe(original)
    w = Writer(out, info["width"], info["height"], info["fps"])
    msk = list(frames(mask))
    try:
        for i, f in enumerate(frames(original)):
            m = cv2.resize(msk[min(len(msk) - 1, i)], (f.shape[1], f.shape[0]))
            m = ((m.max(axis=2) if m.ndim == 3 else m) > 127)[..., None]
            tint = np.zeros_like(f)
            tint[:] = color
            w.put(np.where(m, f * (1 - alpha) + tint * alpha, f))
    finally:
        w.close()
    return out


def mask_stats(mask):
    """Share of the frame covered by the mask, per video: (min, mean, max). Flags empty/huge masks."""
    cover = [float((((f.max(axis=2)) > 127).mean())) for f in frames(mask)]
    if not cover:
        raise ValueError("Mask video has no frames.")
    stats = {"min": min(cover), "mean": float(np.mean(cover)), "max": max(cover), "frames": len(cover)}
    stats["warnings"] = []
    if stats["mean"] < 0.01:
        stats["warnings"].append("Mask is almost empty: no person was found. Do not generate.")
    if stats["max"] > 0.85:
        stats["warnings"].append("Mask covers almost the whole frame: the result will be a full re-render.")
    return stats


def mux_original_audio(video, original, out):
    """Copy the video as is and attach the original soundtrack; refuse to trim/stretch beyond 0.5 s."""
    if not probe(original)["has_audio"]:
        sh(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-c", "copy", str(out)])
        return out, False
    vd, ad = probe(video)["duration"], probe(original)["duration"]
    if abs(vd - ad) > 0.5:
        raise ValueError(f"Video is {vd:.2f}s but the original audio is {ad:.2f}s; refusing to trim or stretch it.")
    sh(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(original), "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", str(min(vd, ad)), str(out)])
    return out, True


def normalize_mask(mask, out):
    """Turn Wan's mask into a clean grayscale video. Wan marks the PERSON as white (src_bg = frame * (1 - mask) in
    wan/modules/animate/preprocess/process_pipepline.py), so the polarity is kept exactly as Wan wrote it - it is never
    guessed from coverage (a close-up portrait legitimately fills most of the frame). Returns (path, inverted=False)."""
    fs = list(frames(mask))
    if not fs:
        raise ValueError("Mask video has no frames.")
    info = probe(mask)
    w = Writer(out, info["width"], info["height"], info["fps"])
    try:
        for f in fs:
            g = f.max(axis=2)
            w.put(np.repeat(g[..., None], 3, axis=2))
    finally:
        w.close()
    return out, False
