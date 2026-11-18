"""Clips longer than 30 s: split into overlapping chunks, swap each, crossfade the overlaps back together.

Why: one GPU job is capped at MAX_SECONDS. Wan-Animate itself already works in segments with
temporal guidance from the previous segment; here we do the same one level up. Each chunk starts
`overlap` seconds BEFORE the previous one ended, so the two renders show the same moment and can
be blended without a visible jump. Pure CPU, no GPU cost.
"""
import numpy as np

import swapkit as sk


def plan_chunks(duration, fps, chunk=24.0, overlap=1.0, min_tail=3.0):
    """Return [(start_s, length_s)], frame-aligned. Every chunk <= MAX_SECONDS; neighbours overlap."""
    if duration <= 0 or fps <= 0:
        raise ValueError("Bad duration/fps.")
    if not (0 < overlap < chunk <= sk.MAX_SECONDS):
        raise ValueError("Need 0 < overlap < chunk <= MAX_SECONDS.")
    if duration <= sk.MAX_SECONDS:
        return [(0.0, round(duration, 3))]
    step = chunk - overlap
    out, start = [], 0.0
    while start < duration - 1e-6:
        end = min(duration, start + chunk)
        out.append([start, end])
        if end >= duration:
            break
        start += step
    # a very short last chunk is not worth a job: stretch the previous start back instead
    if len(out) > 1 and out[-1][1] - out[-1][0] < min_tail:
        out[-1][0] = max(out[-2][0] + step, out[-1][1] - chunk)
    snap = lambda t: round(round(t * fps) / fps, 3)
    return [(snap(s), round(snap(e) - snap(s), 3)) for s, e in out]


def blend_overlap(tail, head):
    """Linear crossfade of two equal-length frame lists (older -> newer)."""
    n = len(tail)
    if n != len(head):
        raise ValueError("Overlap parts must have the same number of frames.")
    out = []
    for i, (a, b) in enumerate(zip(tail, head)):
        w = (i + 1) / (n + 1)
        out.append((a.astype(np.float32) * (1 - w) + b.astype(np.float32) * w).round().astype(np.uint8))
    return out


def stitch(chunk_videos, overlap_frames, out):
    """Join swapped chunk videos (same size/fps). Returns total frames written."""
    if not chunk_videos:
        raise ValueError("No chunks.")
    info = sk.probe(chunk_videos[0])
    w = sk.Writer(out, info["width"], info["height"], info["fps"])
    total, carry = 0, None
    try:
        for ci, path in enumerate(chunk_videos):
            fr = list(sk.frames(path))
            if len(fr) <= overlap_frames * 2 and ci < len(chunk_videos) - 1:
                raise ValueError(f"Chunk {ci} is too short for the overlap.")
            head = fr[:overlap_frames] if ci else []
            if ci:
                for f in blend_overlap(carry, head):
                    w.put(f); total += 1
            body_end = len(fr) - (overlap_frames if ci < len(chunk_videos) - 1 else 0)
            for f in fr[overlap_frames if ci else 0:body_end]:
                w.put(f); total += 1
            carry = fr[body_end:] if ci < len(chunk_videos) - 1 else None
    finally:
        w.close()
    return total
