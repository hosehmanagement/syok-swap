"""HTTP face of the texture-pass job service. Plain FastAPI, no Modal, so it is testable with TestClient.

Routes (all behind Modal's proxy token when deployed):
    GET    /health                 -> {"ok": true, ...}  free check the router uses to prove the key and URL
    POST   /prepare                multipart: video, photo, consent=yes  -> {"job": id}   (face tracking runs in the background)
    GET    /status/{job}           -> prepare/render state
    GET    /preview/{job}?frame=N&strength=S   -> image/jpeg (one finished frame)
    POST   /render/{job}?strength=S            -> starts the full render
    GET    /result/{job}           -> the finished mp4 (409 until done)
    DELETE /job/{job}
"""
from pathlib import Path

import texture_service as svc
import texture_transfer as tt


def build_app(root, spawn_prepare, spawn_render, before_read=lambda: None, after_write=lambda: None):
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, Response

    app = FastAPI(title="Syok Swap texture pass", docs_url=None, redoc_url=None)

    @app.exception_handler(svc.JobError)
    async def _job_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=exc.status)

    @app.get("/health")
    def health():
        return {"ok": True, "app": "syok-swap-texture", "version": 1,
                "strength": {"min": tt.MIN_STRENGTH, "max": tt.MAX_STRENGTH, "start": tt.DEFAULT_STRENGTH},
                "limits": {"video_mb": svc.MAX_VIDEO_BYTES // 2**20, "photo_mb": svc.MAX_PHOTO_BYTES // 2**20,
                           "seconds": int(svc.sk.MAX_SECONDS), "keep_days": svc.JOB_TTL_SECONDS // 86400}}

    @app.post("/prepare")
    async def prepare(video: UploadFile = File(...), photo: UploadFile = File(...), consent: str = Form("")):
        before_read()
        job = svc.new_job(root, await video.read(), await photo.read(), Path(photo.filename or "").suffix, consent)
        after_write()
        spawn_prepare(job)
        return {"job": job}

    @app.get("/status/{job}")
    def status(job: str):
        before_read()
        return svc.status(root, job)

    @app.get("/preview/{job}")
    def preview(job: str, frame: int = 0, strength: float = tt.DEFAULT_STRENGTH):
        before_read()
        return Response(svc.preview_jpeg(root, job, frame, strength), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.post("/render/{job}")
    def render(job: str, strength: float = tt.DEFAULT_STRENGTH):
        before_read()
        used = svc.start_render(root, job, strength)
        after_write()
        spawn_render(job)
        return {"state": "queued", "strength": used}

    @app.get("/result/{job}")
    def result(job: str):
        before_read()
        return FileResponse(svc.result_path(root, job), media_type="video/mp4", filename=f"syok-swap-{job}.mp4")

    @app.delete("/job/{job}")
    def delete(job: str):
        before_read()
        svc.delete_job(root, job)
        after_write()
        return {"deleted": True}

    return app
