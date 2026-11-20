"""Syok Swap video plumbing (CPU only): ffprobe, frame reading/writing through ffmpeg, and putting the original soundtrack
back on a clip. Used by the H3 engine (h3_engine / h3_api / h3_app: probe) and by texture_service / texture_transfer.
"""
import json
import subprocess

import cv2
import numpy as np

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
