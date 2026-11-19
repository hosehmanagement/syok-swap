"""Syok Swap texture pass on Modal: CPU only, no GPU, no model weights (a few cents at most per clip).

This is the small app a user deploys into THEIR OWN Modal account so the Syok "Recast" card can offer the strength slider.
It is separate from app.py (the 50 GB Wan model) on purpose: the texture pass only needs OpenCV and ffmpeg.

    modal deploy texture_app.py          # prints the endpoint URL; protect it with a Proxy Token (see docs/TEXTURE.md)

The endpoint requires a Modal Proxy Token (Modal-Key / Modal-Secret headers), so only the user's own Syok Worker can call it.
"""
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
ROOT = "/jobs"

app = modal.App("syok-swap-texture")
jobs = modal.Volume.from_name("syok-swap-texture-jobs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "opencv-python-headless", "fastapi[standard]")
    .add_local_file(HERE / "swapkit.py", "/opt/swapkit.py")
    .add_local_file(HERE / "texture_transfer.py", "/opt/texture_transfer.py")
    .add_local_file(HERE / "texture_service.py", "/opt/texture_service.py")
    .add_local_file(HERE / "texture_api.py", "/opt/texture_api.py")
)


def _svc():
    if "/opt" not in sys.path:
        sys.path.insert(0, "/opt")
    import texture_service
    return texture_service


@app.function(image=image, volumes={ROOT: jobs}, cpu=2, timeout=1800)
def prepare_job(job: str):
    jobs.reload()
    _svc().run_prepare(ROOT, job)
    jobs.commit()


@app.function(image=image, volumes={ROOT: jobs}, cpu=2, timeout=3600)
def render_job(job: str):
    jobs.reload()
    _svc().run_render(ROOT, job)
    jobs.commit()


@app.function(image=image, volumes={ROOT: jobs}, schedule=modal.Period(hours=6), timeout=600)
def sweep_old_jobs():
    """Jobs (and the uploaded clip) are deleted after 3 days, the same time the result is kept in the user's Cloudflare."""
    jobs.reload()
    n = _svc().sweep(ROOT)
    jobs.commit()
    return n


@app.function(image=image, volumes={ROOT: jobs}, cpu=2, timeout=600, max_containers=3)
@modal.concurrent(max_inputs=4)
@modal.asgi_app(requires_proxy_auth=True)
def web():
    if "/opt" not in sys.path:
        sys.path.insert(0, "/opt")
    import texture_api

    def reload():
        try:
            jobs.reload()
        except Exception:      # a volume with files open by a concurrent request cannot reload; the next request will
            pass

    return texture_api.build_app(ROOT, prepare_job.spawn, render_job.spawn, before_read=reload, after_write=jobs.commit)
