"""HTTP face of the Syok Swap H3 engine (MiniMax H3 through openjutsu, on the user's own Modal). Plain FastAPI, no Modal
import, so it is testable with TestClient. h3_app.web wires it to the real Modal
functions (check_job / swap_job / fetch_models) behind Modal's proxy token (requires_proxy_auth=True).

Territory (H3 Community License excludes the US, the EU, the UK and South Korea; see docs/H3_TERRITORY.md):
    The country comes ONLY from the X-Syok-Country request header, which the Syok router Worker sets from Cloudflare's
    request.cf.country (and strips from anything the browser sent). This endpoint answers only to the Modal proxy token,
    which only the router holds. A form field of the same or a similar name is never read. The value goes through
    h3_territory.check_country and FAILS CLOSED: a missing, empty, malformed, "XX" (unknown) or "T1" (Tor) value, or a
    blocked country, is refused with 451 {"error": "territory", ...}.
    Checked on: POST /swap (header AND the tick box territory=yes), POST /generate/{job} (the paid H3 run), POST
    /fixfaces/{job} (the optional paid H3 face run) and POST /setup (downloads the H3 model files). Not checked on /health,
    /status, /mask, /result, /faces and DELETE /job, so a user can always read and delete their own job (the router checks
    the country on every route anyway).

Routes:
    GET    /health          -> {"ok", "app": "syok-swap-h3", "version", "engine", "weights", "label", "territory",
                                "limits", "resolutions", "margins"}
    POST   /setup           starts the model download if files are missing -> {"weights": "downloading" | "ready"}
    POST   /swap            multipart: video, photo, consent=yes, territory=yes, margin, resolution, start, length
                            -> {"job", "estimate"}; the mask check starts at once (small GPU)
    GET    /status/{job}    -> h3_engine.status(...) plus "estimate"
    GET    /mask/{job}      -> check.mp4 (video/mp4) once the mask check is ready (409 before)
    POST   /generate/{job}  the PAID step (big GPU) -> {"state": "queued", "resume": bool}
    GET    /result/{job}    -> h3.mp4 (video/mp4), 409 until done
    POST   /fixfaces/{job}  json {"mode": "photo" | "mouth"} (default "photo"): the optional PAID Fix faces run on the
                            finished swap (big GPU) -> {"state": "queued"}. Opt-in only: never started by anything else.
    GET    /faces/{job}     -> faces.mp4 (video/mp4), 409 until done
    DELETE /job/{job}       -> {"deleted": true} (the swap and the Fix faces result go with the job)

Fix faces licence note (shown to the user by the web card too): it uses the InsightFace buffalo_l face model, licensed for
non-commercial research only, and Ultralytics YOLO weights (AGPL-3.0). That is why it is opt-in and never automatic.
"""
import json
from pathlib import Path

import h3_engine as h3
import h3_territory as terr
import texture_service as tsvc

APP_NAME = "syok-swap-h3"
VERSION = 1
COUNTRY_HEADER = "X-Syok-Country"
TERRITORY_FIELD = "territory"            # the tick box form field; only the exact value "yes" counts
TERRITORY_SUMMARY = "US, EU, UK and South Korea are excluded"
TERRITORY_ERROR = "MiniMax H3 is not available where you are (US, EU, UK, South Korea)."
CONSENT_ERROR = ("Only use photos and videos of people who agreed to it (or fictional characters). Tick the consent box "
                 "(consent=yes).")
TICK_ERROR = "Tick '" + terr.TERRITORY_CONSENT_TEXT + "' to use " + terr.ENGINE_LABEL + "."
FACES_DOWNLOADING = ("The Fix faces model files (about 0.5 GB) are downloading to your Modal account. Try again in a few "
                     "minutes.")
FACES_UNAVAILABLE = "Fix faces is not set up on this engine."

# What one live run measured (contract / docs/H3.md): the mask check took about 17 minutes on a cold L4, and a 6 s clip at
# 480p swapped in about 3 minutes on an L40S. Only that one data point exists, so other resolutions get no swap time.
CHECK_MINUTES_COLD = 17
SWAP_MINUTES_PER_SECOND_480P = 3 / 6
ESTIMATE_BASIS = ("Measured once: the mask check took about 17 minutes on a cold start, and 6 seconds at 480p swapped in "
                  "about 3 minutes. Modal bills the exact amount to your own account.")


class TerritoryRefused(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def territory_gate(header_value):
    """Raise TerritoryRefused unless the router's country header is an allowed country. Returns the normalised code."""
    r = terr.check_country(header_value)
    if r["status"] != terr.ALLOWED:
        raise TerritoryRefused(r["reason"])
    return r["country"]


def estimate(request, clips):
    """Rough time for a job. Always says it is approximate; no money figure (none has been measured for H3)."""
    seconds = float(request.get("length") or 0)
    res = request.get("resolution")
    swap = int(max(1, round(SWAP_MINUTES_PER_SECOND_480P * seconds))) if res == "480p" else None
    return {"approx": True, "clips": clips, "seconds": seconds, "resolution": res,
            "check_minutes": CHECK_MINUTES_COLD, "swap_minutes": swap, "basis": ESTIMATE_BASIS}


def build_h3_app(root, spawn_check, spawn_swap, spawn_weights, weights_state, before_read=lambda: None,
                 after_write=lambda: None, probe=None, spawn_faces=None, faces_weights_state=None):
    """`spawn_check(job)` / `spawn_swap(job)` / `spawn_faces(job)` start the GPU steps, `spawn_weights()` starts the model
    download (it fetches every missing file, the Fix faces ones included), `weights_state()` and `faces_weights_state()`
    -> "ready" | "downloading" | "missing" (swap + check files / Fix faces files). `probe(path)` reads the video
    (swapkit.probe by default). Without `spawn_faces` the Fix faces routes answer 501."""
    from fastapi import FastAPI, File, Form, Header, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse

    if probe is None:
        import swapkit
        probe = swapkit.probe

    app = FastAPI(title="Syok Swap H3", docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(tsvc.JobError)
    async def _job_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=exc.status)

    @app.exception_handler(TerritoryRefused)
    async def _territory(request, exc):
        return JSONResponse({"error": "territory", "message": TERRITORY_ERROR}, status_code=451)

    @app.get("/health")
    def health():
        before_read()
        return {"ok": True, "app": APP_NAME, "version": VERSION, "engine": "minimax-h3", "weights": weights_state(),
                "label": terr.ENGINE_LABEL, "territory": TERRITORY_SUMMARY,
                "limits": {"video_mb": tsvc.MAX_VIDEO_BYTES // 2**20, "photo_mb": tsvc.MAX_PHOTO_BYTES // 2**20,
                           "seconds": int(h3.MAX_SECTION_SECONDS), "keep_days": tsvc.JOB_TTL_SECONDS // 86400},
                "resolutions": list(h3.PRESETS), "margins": list(h3.MARGINS)}

    @app.post("/setup")
    def setup(x_syok_country: str = Header(None)):
        territory_gate(x_syok_country)
        before_read()
        state = weights_state()
        if state == "missing":
            spawn_weights()
            state = "downloading"
        return {"weights": state}

    @app.post("/swap")
    async def swap(video: UploadFile = File(...), photo: UploadFile = File(...), consent: str = Form(""),
                   territory: str = Form(""), margin: str = Form("normal"), resolution: str = Form(h3.DEFAULT_RESOLUTION),
                   start: str = Form("0"), length: str = Form("0"), x_syok_country: str = Header(None)):
        # The licence gate comes first: nothing is read, stored or started for a refused country.
        country = territory_gate(x_syok_country)
        if not terr.territory_consent_given(territory):
            raise TerritoryRefused("tick box not ticked")
        if consent != "yes":
            raise tsvc.JobError(CONSENT_ERROR, 422)
        before_read()
        if weights_state() != "ready":
            raise tsvc.JobError("One-time setup is not finished: the model files are not on your Modal account yet.", 409)
        record = terr.make_territory_record(country, territory)       # raises PermissionError only if a check above was skipped
        job = h3.new_h3_job(root, await video.read(), await photo.read(), Path(photo.filename or "").suffix, consent, probe,
                            margin=margin, resolution=resolution, start=start, length=length)
        d = tsvc.job_dir(root, job)
        meta = tsvc.read_meta(d)
        meta.update(h3_territory="yes", country=country, territory_record=record)
        tsvc.write_meta(d, meta)
        h3.start_check(root, job)
        after_write()
        try:
            spawn_check(job)
        except Exception:
            h3.fail(root, job, "check", "The mask check could not start. Delete this job and try again.")
            after_write()
            raise tsvc.JobError("The mask check could not start on your Modal account. Try again.", 500) from None
        st = h3.status(root, job)
        return {"job": job, "estimate": estimate(st["request"], st["clips"])}

    @app.get("/status/{job}")
    def status(job: str):
        before_read()
        st = h3.status(root, job)
        return {**st, "estimate": estimate(st["request"], st["clips"])}

    @app.get("/mask/{job}")
    def mask(job: str):
        before_read()
        return FileResponse(h3.result_path(root, job, "check"), media_type="video/mp4", headers={"Cache-Control": "no-store"})

    @app.post("/generate/{job}")
    def generate(job: str, x_syok_country: str = Header(None)):
        territory_gate(x_syok_country)
        before_read()
        resume = h3.start_swap(root, job)          # refuses a missing/failed check, a running swap and a finished one (409)
        after_write()
        try:
            spawn_swap(job)
        except Exception:
            h3.fail(root, job, "swap", "The swap could not start. Start it again: nothing was generated.")
            after_write()
            raise tsvc.JobError("The swap could not start on your Modal account. Try again.", 500) from None
        return {"state": "queued", "resume": resume}

    @app.get("/result/{job}")
    def result(job: str):
        before_read()
        return FileResponse(h3.result_path(root, job, "swap"), media_type="video/mp4", filename=f"syok-h3-{job}.mp4")

    @app.post("/fixfaces/{job}")
    async def fixfaces(job: str, request: Request, x_syok_country: str = Header(None)):
        # Licence gate first: the body is not even read for a refused country.
        territory_gate(x_syok_country)
        if spawn_faces is None or faces_weights_state is None:
            raise tsvc.JobError(FACES_UNAVAILABLE, 501)
        raw = await request.body()
        try:
            body = json.loads(raw) if raw.strip() else {}
        except ValueError:
            raise tsvc.JobError("Send JSON like {\"mode\": \"photo\"}.") from None
        if not isinstance(body, dict):
            raise tsvc.JobError("Send JSON like {\"mode\": \"photo\"}.")
        before_read()
        _, mode, _ = h3.check_faces_start(root, job, body.get("mode"))     # 404 / 409 / 400, nothing written yet
        state = faces_weights_state()
        if state != "ready":
            if state == "missing":
                spawn_weights()
            raise tsvc.JobError(FACES_DOWNLOADING, 409)
        h3.start_faces(root, job, mode)
        after_write()
        try:
            spawn_faces(job)
        except Exception:
            h3.fail(root, job, "faces", "Fix faces could not start. Try again: nothing was generated.")
            after_write()
            raise tsvc.JobError("Fix faces could not start on your Modal account. Try again.", 500) from None
        return {"state": "queued"}

    @app.get("/faces/{job}")
    def faces(job: str):
        before_read()
        return FileResponse(h3.result_path(root, job, "faces"), media_type="video/mp4", filename=f"syok-h3-faces-{job}.mp4")

    @app.delete("/job/{job}")
    def delete(job: str):
        before_read()
        tsvc.delete_job(root, job)
        after_write()
        return {"deleted": True}

    return app
