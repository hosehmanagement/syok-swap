"""HTTP face of Syok Swap: one job from upload to saved clip. Plain FastAPI, no Modal, so it is testable with TestClient.

Routes (all behind Modal's proxy token when deployed):
    GET    /health                 -> {"ok": true, "app": "syok-swap", "weights": "ready|downloading|missing", ...}   free check
    POST   /setup                  one-time: download the model files into the user's Modal volume (starts it if missing)
    POST   /swap                   multipart: video, photo, consent=yes, resolution, margin, start, length, optional skin,
                                   optional face (close-up for face-lock), facelock=yes|""
                                   -> {"job", "estimate"}. Refused (422, nothing charged) when the photo cannot feed the
                                   realism slider. (the mask step starts at once on a small GPU: cents)
    POST   /facelock/{job}         header X-Magic-Hour-Key (the user's own key; never stored or logged), optional multipart face.
                                   Puts the exact face from the photo onto the swapped clip with Magic Hour (the user's credits),
                                   then prepares the slider again. -> {"facelock": {"state": "queued"}}
    POST   /facelock/{job}/skip    use the swap as it is (while face-lock waits or after it failed); the slider is prepared
    POST   /retexture/{job}        multipart: optional skin close-up. Prepare the slider again on a finished swap (CPU only)
    GET    /status/{job}           -> mask / swap / prepare / render / facelock stages, the estimate, expires_at
    GET    /mask/{job}             -> the red-overlay preview video (video/mp4) once the mask is ready
    POST   /generate/{job}?relight=true   the PAID step: generate the new person on an H100 (refuses a bad mask or a second run).
                                   When it ends, the slider's face tracking starts by itself.
    GET    /preview/{job}?frame=N&strength=S   -> image/jpeg: one finished frame of the swapped clip with the texture slider applied
    POST   /render/{job}?strength=S            full clip at that strength (CPU)
    GET    /result/{job}           -> the finished mp4 (409 until done)
    GET    /swapped/{job}          -> the swapped clip WITHOUT texture (fallback when the slider could not be prepared)
    DELETE /job/{job}
"""
from pathlib import Path
from typing import Optional

import swap_service as ss
import texture_service as svc
import texture_transfer as tt


def build_app(root, spawn_mask, spawn_swap, spawn_render, spawn_weights, weights_state,
              before_read=lambda: None, after_write=lambda: None, spawn_prepare=lambda job: None, detector=tt.detect_face,
              spawn_facelock=lambda job, key: None):
    from fastapi import FastAPI, File, Form, Header, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, Response

    app = FastAPI(title="Syok Swap", docs_url=None, redoc_url=None)

    @app.exception_handler(svc.JobError)
    async def _job_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=exc.status)

    @app.get("/health")
    def health():
        before_read()
        return {"ok": True, "app": "syok-swap", "version": 3, "weights": weights_state(), "facelock": True,
                "strength": {"min": tt.MIN_STRENGTH, "max": tt.MAX_STRENGTH, "start": tt.DEFAULT_STRENGTH},
                "limits": {"video_mb": svc.MAX_VIDEO_BYTES // 2**20, "photo_mb": svc.MAX_PHOTO_BYTES // 2**20,
                           "seconds": int(svc.sk.MAX_SECONDS), "keep_days": svc.JOB_TTL_SECONDS // 86400}}

    @app.post("/setup")
    def setup():
        before_read()
        state = weights_state()
        if state == "missing":
            spawn_weights()
            state = "downloading"
        return {"weights": state}

    @app.post("/swap")
    async def swap(video: UploadFile = File(...), photo: UploadFile = File(...), consent: str = Form(""),
                   resolution: str = Form("480p"), margin: str = Form("normal"), start: str = Form("0"), length: str = Form("0"),
                   skin: Optional[UploadFile] = File(None), face: Optional[UploadFile] = File(None), facelock: str = Form("")):
        before_read()
        if weights_state() != "ready":
            raise svc.JobError("One-time setup is not finished: the model files are not on your Modal account yet.", 409)
        skin_bytes = (await skin.read()) if skin is not None else None
        face_bytes = (await face.read()) if face is not None else None
        job = ss.new_swap_job(root, await video.read(), await photo.read(), Path(photo.filename or "").suffix, consent,
                              resolution, margin, start, length,
                              skin=skin_bytes or None, skin_ext=Path((skin.filename if skin is not None else "") or "").suffix,
                              detector=detector, face=face_bytes or None,
                              face_ext=Path((face.filename if face is not None else "") or "").suffix,
                              facelock=(facelock or "").strip().lower() == "yes")
        after_write()
        spawn_mask(job)
        return {"job": job, "estimate": ss.status(root, job)["estimate"]}

    @app.post("/retexture/{job}")
    async def retexture(job: str, skin: Optional[UploadFile] = File(None), retry: str = Form("")):
        before_read()
        skin_bytes = (await skin.read()) if skin is not None else None
        ss.retexture(root, job, skin_bytes or None, Path((skin.filename if skin is not None else "") or "").suffix)
        after_write()
        spawn_prepare(job)
        return {"job": job, "prepare": {"state": "queued"}}

    @app.post("/facelock/{job}")
    async def facelock_start(job: str, face: Optional[UploadFile] = File(None),
                             x_magic_hour_key: Optional[str] = Header(None)):
        before_read()
        face_bytes = (await face.read()) if face is not None else None
        ss.start_facelock(root, job, x_magic_hour_key, face_bytes or None,
                          Path((face.filename if face is not None else "") or "").suffix, detector=detector)
        after_write()
        try:
            spawn_facelock(job, x_magic_hour_key.strip())          # the only place the key goes; never into meta or a log
        except Exception:
            ss.fail(root, job, "facelock", "Face-lock could not start. Try again, or skip it to use the swap as it is.")
            after_write()
            raise svc.JobError("Face-lock could not start. Try again.", 500) from None
        return {"facelock": {"state": "queued"}}

    @app.post("/facelock/{job}/skip")
    def facelock_skip(job: str):
        before_read()
        spawn = ss.skip_facelock(root, job)
        after_write()
        if spawn:
            spawn_prepare(job)
        st = ss.status(root, job)
        return {"facelock": {"state": "skipped"}, "prepare": {"state": st["prepare"].get("state", "queued")}}

    @app.get("/status/{job}")
    def status(job: str):
        before_read()
        return ss.status(root, job)

    @app.get("/mask/{job}")
    def mask(job: str):
        before_read()
        return FileResponse(ss.mask_path(root, job), media_type="video/mp4", headers={"Cache-Control": "no-store"})

    @app.post("/generate/{job}")
    def generate(job: str, relight: bool = True):
        before_read()
        used = ss.start_generate(root, job, relight)
        after_write()
        spawn_swap(job)
        return {"state": "queued", "relight": used}

    @app.get("/preview/{job}")
    def preview(job: str, frame: int = 0, strength: float = tt.DEFAULT_STRENGTH):
        before_read()
        return Response(svc.preview_jpeg(root, job, frame, strength), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.post("/render/{job}")
    def render(job: str, strength: float = tt.DEFAULT_STRENGTH):
        before_read()
        ss.facelock_blocks_render(root, job)
        used = svc.start_render(root, job, strength)
        after_write()
        spawn_render(job)
        return {"state": "queued", "strength": used}

    @app.get("/result/{job}")
    def result(job: str):
        before_read()
        return FileResponse(svc.result_path(root, job), media_type="video/mp4", filename=f"syok-swap-{job}.mp4")

    @app.get("/swapped/{job}")
    def swapped(job: str):
        before_read()
        return FileResponse(ss.swapped_path(root, job), media_type="video/mp4", filename=f"syok-swap-{job}.mp4")

    @app.delete("/job/{job}")
    def delete(job: str):
        before_read()
        svc.delete_job(root, job)
        after_write()
        return {"deleted": True}

    return app
