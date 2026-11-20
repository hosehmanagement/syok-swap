"""HTTP face of Syok Swap: one job from upload to saved clip. Plain FastAPI, no Modal, so it is testable with TestClient.

Routes (all behind Modal's proxy token when deployed):
    GET    /health                 -> {"ok": true, "app": "syok-swap", "weights": "ready|downloading|missing", ...}   free check
    POST   /setup                  one-time: download the model files into the user's Modal volume (starts it if missing)
    POST   /swap                   multipart: video, photo, consent=yes, resolution, margin, start, length, optional skin,
                                   optional face (1 to 5 candidate photos for face-lock: repeat the field; each png/jpg/
                                   webp <= photo_mb, all together <= faces_mb, else 400), facelock=yes|"",
                                   collage=yes|1|true (EXPERIMENTAL, off otherwise: Wan gets the main photo + the chosen
                                   close-up side by side as its one reference; needs at least one face, else 422
                                   "Collage needs a face close-up.", nothing stored; slider and face-lock keep the main photo),
                                   fit=yes|1|true: before the swap an image-editing model on your Modal GPU paints the new person into
                                   the clip's first and last frame and Wan gets those two views as its reference (status.fit says
                                   what happened; a bad edit is dropped and your photo is used),
                                   mask_shape=silhouette|blocky|box (default silhouette: the red mask follows the person's outline;
                                   box: the old whole-bounding-box mask; anything else 400, nothing stored)
                                   -> {"job", "estimate"}. The best of the face photos and the main photo is kept
                                   (picker.py); status facelock.pick says which and why. Refused (422, nothing charged) when the photo cannot feed the
                                   realism slider. (the mask step starts at once on a small GPU: cents)
    POST   /facelock/{job}         header X-Magic-Hour-Key (the user's own key; never stored or logged), optional multipart face
                                   (1 to 5, same rules as /swap; the main photo is again a candidate; replaces the earlier pick).
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
from typing import List, Optional

import picker
import swap_service as ss
import texture_service as svc
import texture_transfer as tt


def collage_on(value):
    """The `collage` form field: "yes", "1" or "true" (any case, spaces ignored) turn it on; anything else is off."""
    return (value or "").strip().lower() in ("yes", "1", "true")


def build_app(root, spawn_mask, spawn_swap, spawn_render, spawn_weights, weights_state,
              before_read=lambda: None, after_write=lambda: None, spawn_prepare=lambda job: None, detector=tt.detect_face,
              spawn_facelock=lambda job, key: None, face_detector=None):
    from fastapi import FastAPI, File, Form, Header, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, Response

    app = FastAPI(title="Syok Swap", docs_url=None, redoc_url=None)

    async def _read_faces(files):
        """The repeated `face` field as [(bytes, ext)]. Stops reading past the limits so a huge upload is refused early."""
        out, total = [], 0
        for f in files or []:
            if f is None:
                continue
            data = await f.read(svc.MAX_PHOTO_BYTES + 1)
            if not data:
                continue
            total += len(data)
            out.append((data, Path(f.filename or "").suffix))
            if len(out) > picker.MAX_CANDIDATES or total > picker.MAX_TOTAL_BYTES:
                break                                   # face_uploads() refuses it with the plain message
        return out

    @app.exception_handler(svc.JobError)
    async def _job_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=exc.status)

    @app.get("/health")
    def health():
        before_read()
        return {"ok": True, "app": "syok-swap", "version": 5, "weights": weights_state(), "facelock": True, "collage": True, "composite": "two-mask", "fit": True,
                "mask_shapes": sorted(svc.sk.MASK_SHAPES), "mask_shape": svc.sk.DEFAULT_MASK_SHAPE,
                "strength": {"min": tt.MIN_STRENGTH, "max": tt.MAX_STRENGTH, "start": tt.DEFAULT_STRENGTH},
                "limits": {"video_mb": svc.MAX_VIDEO_BYTES // 2**20, "photo_mb": svc.MAX_PHOTO_BYTES // 2**20,
                           "seconds": int(svc.sk.MAX_SECONDS), "keep_days": svc.JOB_TTL_SECONDS // 86400,
                           "faces": picker.MAX_CANDIDATES, "faces_mb": picker.MAX_TOTAL_BYTES // 2**20}}

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
                   skin: Optional[UploadFile] = File(None), face: Optional[List[UploadFile]] = File(None),
                   facelock: str = Form(""), collage: str = Form(""), mask_shape: str = Form(""), fit: str = Form("")):
        before_read()
        if weights_state() != "ready":
            raise svc.JobError("One-time setup is not finished: the model files are not on your Modal account yet.", 409)
        skin_bytes = (await skin.read()) if skin is not None else None
        faces = await _read_faces(face)
        job = ss.new_swap_job(root, await video.read(), await photo.read(), Path(photo.filename or "").suffix, consent,
                              resolution, margin, start, length,
                              skin=skin_bytes or None, skin_ext=Path((skin.filename if skin is not None else "") or "").suffix,
                              detector=detector, faces=faces, face_detector=face_detector,
                              facelock=(facelock or "").strip().lower() == "yes", collage=collage_on(collage),
                              mask_shape=mask_shape, fit=collage_on(fit))
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
    async def facelock_start(job: str, face: Optional[List[UploadFile]] = File(None),
                             x_magic_hour_key: Optional[str] = Header(None)):
        before_read()
        faces = await _read_faces(face)
        ss.start_facelock(root, job, x_magic_hour_key, detector=detector, faces=faces, face_detector=face_detector)
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
