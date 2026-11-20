"""Syok Swap H3: openjutsu's 'Simple' swap and 'Check mask' run headlessly in ComfyUI on the user's own Modal account.

A SEPARATE Modal app ("syok-swap-h3") with its own two Volumes; the Wan engine (app.py, "syok-swap") is not touched.
NOTHING HERE HAS BEEN RUN ON MODAL YET. See docs/H3.md for every unverified item before the first deploy.
The web endpoint (web(), h3_api.py) is the one the Syok router calls; it has not been deployed yet either.

    modal run h3_app.py::fetch                                 # once: model files into the syok-h3-models Volume (see docs)
    modal run h3_app.py::new --video v.mp4 --photo p.png --consent yes [--margin normal --start 0 --length 6 --seed 0 --resolution 480p]
    modal run h3_app.py::check --job <id> [--out check.mp4]    # small GPU: SAM 3.1 mask preview (cyan = regenerated)
    modal run h3_app.py::swap  --job <id> [--out h3.mp4]       # big GPU, the paid step; run it again after an error to RESUME
    modal run h3_app.py::status --job <id>

Settings chosen at deploy/run time through environment variables (read where `modal deploy` / `modal run` runs):
    SYOK_H3_GPU          GPU for the swap (default "L40S"; openjutsu was only tested on a 16 GB RTX 4060 Ti + 64 GB RAM)
    SYOK_H3_CHECK_GPU    GPU for the mask check (default "L4")
    SYOK_H3_REGION       optional Modal region(s), comma separated, e.g. "ap" or "ap-southeast". Unset = no pin (Modal picks).
                         Modal bills a region pin at 1.15x (broad, e.g. "ap") or 1.75x (narrow, e.g. "ap-southeast") of the
                         base price, and whether the user's Modal plan may pin a region at all is NOT verified: see docs/H3.md.
    SYOK_H3_HF_SECRET    optional name of a Modal Secret holding HF_TOKEN, only if a model repo turns out to be gated.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import modal

HERE = Path(__file__).parent

# ComfyUI core: openjutsu needs 0.41.2 or newer (README "Requirements"). Pinned to that documented minimum; tag verified to exist
# (refs/tags/v0.41.2 = b53f77a74e21091cbdebfdf7b85fc80326932646 on 21 Nov 2026).
COMFYUI_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
COMFYUI_REF = "b53f77a74e21091cbdebfdf7b85fc80326932646"   # v0.41.2

# Only the five packs the Simple and Check mask workflows need (openjutsu CUSTOM_NODES_REPORT.md section 1). Revisions:
#   recorded in CUSTOM_NODES_REPORT.md section 4 as a commit: Fantastic PromptBuilder 01b6d8d (1.8.1), VideoHelperSuite 115de7a (1.7.9)
#   recorded only as a version, resolved here to the commit whose pyproject.toml first carries that version:
#       Context Loop 0.7.3 -> 293ed6b, KJNodes 1.5.2 -> d3cfe21
#   NOT recorded (TODO): comfyui-h3-oneclick-swap -> pinned to its HEAD as seen on 21 Nov 2026 (e44a56e), not a tested revision
PACKS = [
    ("ComfyUI-MiniMaxH3-Context-Loop", "https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop.git",
     "293ed6b6255ff684598302090f1411a4a1eb5648"),
    ("ComfyUI-Fantastic-MiniMaxH3-PromptBuilder", "https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder.git",
     "01b6d8d91180c6994840e34312a65febe02cc4f5"),
    ("ComfyUI-KJNodes", "https://github.com/kijai/ComfyUI-KJNodes.git", "d3cfe21625e5170126ce06fbfcfe1d88108688c3"),
    ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
     "115de7a9d9e34410cffb9ecfd268e993b11a50fb"),
    ("comfyui-h3-oneclick-swap", "https://github.com/D3voz/comfyui-h3-oneclick-swap.git",
     "e44a56e92920000e675d14be484aea0d7498d63e"),
]

MODELS = "/h3models"        # syok-h3-models Volume: ComfyUI's models directory (--models-directory)
JOBS = "/h3jobs"            # syok-h3-jobs Volume: job folders (meta.json, uploads, ComfyUI output + Context Loop checkpoints)
COMFY = "/opt/ComfyUI"
PORT = 8188

GPU_SWAP = os.environ.get("SYOK_H3_GPU", "L40S")
GPU_CHECK = os.environ.get("SYOK_H3_CHECK_GPU", "L4")
SWAP_MINUTES = int(os.environ.get("SYOK_H3_SWAP_MINUTES", "240"))   # hard cap on the paid GPU step (the first live test sets 45)
REGION = [r.strip() for r in os.environ.get("SYOK_H3_REGION", "").split(",") if r.strip()] or None
HF_SECRET = os.environ.get("SYOK_H3_HF_SECRET", "").strip()
SECRETS = [modal.Secret.from_name(HF_SECRET)] if HF_SECRET else []

app = modal.App("syok-swap-h3")
models = modal.Volume.from_name("syok-h3-models", create_if_missing=True)
jobs = modal.Volume.from_name("syok-h3-jobs", create_if_missing=True)


def _pack_cmd(name, url, rev):
    d = f"{COMFY}/custom_nodes/{name}"
    return (f"git clone -q {url} {d} && cd {d} && git checkout -q {rev} "
            f"&& if [ -f requirements.txt ]; then pip install -r requirements.txt; fi")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")       # ffmpeg on PATH: Context Loop's final assembly needs it
    .run_commands(f"git clone -q {COMFYUI_REPO} {COMFY} && cd {COMFY} && git checkout -q {COMFYUI_REF}")
    .run_commands(f"pip install -r {COMFY}/requirements.txt")
    .run_commands(*[_pack_cmd(*p) for p in PACKS])
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Fail the cheap CPU build, not a paid GPU run, when a pack does not import (ComfyUI only logs "IMPORT FAILED").
    .run_commands(f"cd {COMFY} && python main.py --cpu --quick-test-for-ci > /tmp/qt.log 2>&1; cat /tmp/qt.log | tail -40; "
                  "! grep -q 'IMPORT FAILED' /tmp/qt.log")
    .add_local_file(HERE / "h3_engine.py", "/opt/syok/h3_engine.py")
    .add_local_file(HERE / "texture_service.py", "/opt/syok/texture_service.py")     # job folder + meta.json helpers
    .add_local_file(HERE / "texture_transfer.py", "/opt/syok/texture_transfer.py")   # imported by texture_service
    .add_local_file(HERE / "swapkit.py", "/opt/syok/swapkit.py")                     # probe() (ffprobe)
    .add_local_dir(HERE / "workflows" / "openjutsu", "/opt/syok/workflows/openjutsu")
)
light = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "opencv-python-headless<5", "huggingface_hub[hf_transfer]", "fastapi[standard]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(HERE / "h3_engine.py", "/opt/syok/h3_engine.py")
    .add_local_file(HERE / "h3_api.py", "/opt/syok/h3_api.py")                       # the web endpoint (web() below)
    .add_local_file(HERE / "h3_territory.py", "/opt/syok/h3_territory.py")           # the licence territory gate
    .add_local_file(HERE / "texture_service.py", "/opt/syok/texture_service.py")
    .add_local_file(HERE / "texture_transfer.py", "/opt/syok/texture_transfer.py")
    .add_local_file(HERE / "swapkit.py", "/opt/syok/swapkit.py")
    .add_local_dir(HERE / "workflows" / "openjutsu", "/opt/syok/workflows/openjutsu")
)


def _eng():
    if "/opt/syok" not in sys.path:
        sys.path.insert(0, "/opt/syok")
    import h3_engine
    return h3_engine


# ---------------------------------------------------------------- ComfyUI process (one per call, per job folders)

def _http(method, path, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def _start_comfy(job_dir):
    """Start ComfyUI with this job's input/output folders on the jobs Volume (so the Context Loop's per-clip checkpoints
    survive a crash and a rerun resumes) and the models Volume as its models directory."""
    out, inp, tmp = job_dir / "comfy_out", job_dir / "comfy_in", Path("/tmp/comfy_temp")
    for p in (out, inp, tmp):
        p.mkdir(parents=True, exist_ok=True)
    log = open("/tmp/comfy.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT), "--disable-auto-launch",
         "--models-directory", MODELS, "--input-directory", str(inp), "--output-directory", str(out),
         "--temp-directory", str(tmp)], cwd=COMFY, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):                                          # custom nodes can take a while to import
        if proc.poll() is not None:
            raise RuntimeError("ComfyUI exited at start:\n" + Path("/tmp/comfy.log").read_text()[-3000:])
        try:
            _http("GET", "/system_stats", timeout=5)
            return proc, {"output": out, "temp": tmp / "temp"}   # ComfyUI v0.41.2 main.py: <--temp-directory>/temp
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(1)
    proc.kill()
    raise RuntimeError("ComfyUI did not start within 10 minutes.")


def _run_graph(eng, job_dir, graph, node_id, timeout_s):
    """Submit, wait, return the file the node made. Commits the jobs Volume every minute so checkpoints are kept."""
    proc, folders = _start_comfy(job_dir)
    since = time.time()
    try:
        try:
            r = _http("POST", "/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})
        except urllib.error.HTTPError as e:                     # ComfyUI refused the graph (missing node/model, bad input)
            raise RuntimeError("ComfyUI refused the workflow: " + e.read().decode(errors="replace")[:3000]) from None
        pid = r["prompt_id"]
        last_commit, deadline = time.time(), time.time() + timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("ComfyUI stopped during the run:\n" + Path("/tmp/comfy.log").read_text()[-3000:])
            entry = (_http("GET", f"/history/{pid}") or {}).get(pid)
            state = eng.history_state(entry)
            if state == "error":
                raise RuntimeError("ComfyUI error: " + eng.history_error(entry))
            if state == "success":
                files = [p for p in eng.output_files(entry, node_id, folders) if p.suffix.lower() == ".mp4"]
                found = files[0] if files else eng.newest_mp4(folders["output"], since) or eng.newest_mp4(folders["temp"], since)
                if not found:
                    raise RuntimeError("ComfyUI finished but no MP4 was found for node " + str(node_id))
                return found
            if time.time() - last_commit > 60:
                jobs.commit()
                last_commit = time.time()
            time.sleep(5)
        raise RuntimeError("The run did not finish in time.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def _stage(job, stage, node_id, timeout_s, failure):
    eng = _eng()
    jobs.reload()
    models.reload()
    d = Path(JOBS) / job
    eng.begin(JOBS, job, stage)
    jobs.commit()
    try:
        missing = eng.models_missing(MODELS, stage)
        if missing:
            raise ValueError("Model files are missing on your Modal account: " + ", ".join(missing) + ". Run the fetch step.")
        import swapkit
        if eng.add_silent_audio(d / eng.VIDEO_NAME, swapkit.probe):   # jobs made before this fix may hold a silent clip
            jobs.commit()
        graph = eng.graph_for(JOBS, job, stage)
        produced = _run_graph(eng, d, graph, node_id, timeout_s)
        eng.finish(JOBS, job, stage, produced)
    except Exception as e:
        eng.fail(JOBS, job, stage, str(e) if isinstance(e, ValueError) else failure)
        raise
    finally:
        jobs.commit()
    return eng.status(JOBS, job)


@app.function(image=image, gpu=GPU_CHECK, region=REGION, memory=32768, volumes={MODELS: models, JOBS: jobs},
              timeout=3600, max_containers=2)
def check_job(job: str):
    """The mask check: SAM 3.1 + the edit mask over the section, cyan/orange preview next to the grey mannequin."""
    eng = _eng()
    return _stage(job, "check", eng.MASK_NODES["preview"], 3300,
                  "The mask check failed. Nothing was swapped and no swap was charged.")


@app.function(image=image, gpu=GPU_SWAP, region=REGION, memory=65536, volumes={MODELS: models, JOBS: jobs},
              timeout=SWAP_MINUTES * 60, max_containers=1)
def swap_job(job: str):
    """The paid step. Run it again on the same job after an error: the Context Loop resumes from the last finished clip."""
    eng = _eng()
    return _stage(job, "swap", eng.NODES["final"], SWAP_MINUTES * 60 - 300,
                  "The swap failed. Start it again on the same job: it resumes from the last finished clip.")


@app.function(image=light, volumes={MODELS: models}, secrets=SECRETS, cpu=4, timeout=4 * 3600)
def fetch_models(sources: dict):
    """Download each model file straight into the models Volume (huggingface_hub, inside Modal; nothing on any PC).
    `sources` = {file: {"repo_id", "path_in_repo"?, "revision"?}}. Refuses files without a source: nothing is guessed."""
    eng = _eng()
    from huggingface_hub import hf_hub_download
    src = eng.resolve_sources(sources)
    flag = Path(MODELS) / eng.FETCH_FLAG             # lets the web endpoint say "downloading" (h3_engine.weights_state)
    flag.write_text(str(time.time()))
    models.commit()
    got = []
    try:
        for m in eng.MODEL_FILES:
            dest = Path(MODELS) / m["folder"] / m["file"]
            if dest.is_file():
                got.append(f"{m['folder']}/{m['file']} (already there)")
                continue
            s = src[m["file"]]
            tmp = Path(MODELS) / ".partial"
            p = Path(hf_hub_download(s["repo_id"], s["path_in_repo"], revision=s.get("revision"), local_dir=str(tmp)))
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), dest)
            models.commit()
            got.append(f"{m['folder']}/{m['file']}")
        shutil.rmtree(Path(MODELS) / ".partial", ignore_errors=True)
    finally:
        flag.unlink(missing_ok=True)                 # a failed fetch reads as "missing" again, so /setup can retry it
        models.commit()
    return got


@app.function(image=light, volumes={MODELS: models}, cpu=1, timeout=300)
def model_sizes():
    """Size of every expected model file on the Volume (bytes), so a short or missing download is visible."""
    eng = _eng()
    models.reload()
    out = []
    for m in eng.MODEL_FILES:
        f = Path(MODELS) / m["folder"] / m["file"]
        out.append((f"{m['folder']}/{m['file']}", f.stat().st_size if f.is_file() else None))
    return out


@app.function(image=light, volumes={JOBS: jobs}, cpu=2, timeout=600)
def new_job(video: bytes, photo: bytes, photo_ext: str, consent: str, settings: dict):
    eng = _eng()
    import swapkit
    jobs.reload()
    job = eng.new_h3_job(JOBS, video, photo, photo_ext, consent, swapkit.probe, **settings)
    jobs.commit()
    return eng.status(JOBS, job)


@app.function(image=light, volumes={JOBS: jobs}, cpu=1, timeout=120)
def job_status(job: str):
    jobs.reload()
    return _eng().status(JOBS, job)


@app.function(image=light, volumes={JOBS: jobs}, cpu=1, timeout=300)
def queue(job: str, stage: str):
    """Mark a stage queued (refuses a second paid run, or a swap before the mask check). Returns True for a resume."""
    eng = _eng()
    jobs.reload()
    out = eng.start_check(JOBS, job) if stage == "check" else eng.start_swap(JOBS, job)
    jobs.commit()
    return bool(out)


@app.function(image=light, volumes={JOBS: jobs}, cpu=1, timeout=600)
def read_result(job: str, stage: str):
    jobs.reload()
    return _eng().result_path(JOBS, job, stage).read_bytes()


@app.function(image=light, volumes={JOBS: jobs}, schedule=modal.Period(hours=6), timeout=600)
def sweep_old_jobs():
    """Job folders (uploads, ComfyUI outputs and checkpoints, results) are deleted after 3 days, as in the Wan engine."""
    eng = _eng()
    jobs.reload()
    n = eng.tsvc.sweep(JOBS)
    jobs.commit()
    return n


@app.function(image=light, volumes={MODELS: models, JOBS: jobs}, cpu=2, timeout=600, max_containers=3)
@modal.concurrent(max_inputs=4)
@modal.asgi_app(requires_proxy_auth=True)
def web():
    """The HTTP API (h3_api.py) the Syok router calls with the user's Modal proxy token. The router adds X-Syok-Country
    (from Cloudflare's request.cf.country) and strips any client-sent copy; the API fails closed without it."""
    eng = _eng()
    import h3_api

    def reload():
        for vol in (jobs, models):
            try:
                vol.reload()
            except Exception:  # a volume with files open by a concurrent request cannot reload; the next request will
                pass

    def weights_state():
        return eng.weights_state(MODELS)

    return h3_api.build_h3_app(JOBS, check_job.spawn, swap_job.spawn, lambda: fetch_models.spawn({}), weights_state,
                               before_read=reload, after_write=jobs.commit)


# ---------------------------------------------------------------- local entry points

@app.local_entrypoint()
def fetch(sources: str = ""):
    """--sources (optional): a JSON file {file: {"repo_id": "owner/name", "path_in_repo": "...", "revision": "..."}} that
    overrides the verified sources built into h3_engine.MODEL_SOURCES. Without it the built-in sources are used."""
    override = json.loads(Path(sources).read_text()) if sources else {}
    print("\n".join(fetch_models.remote(override)))


@app.local_entrypoint()
def sizes():
    total = 0
    for name, n in model_sizes.remote():
        print(f"{name}: {'MISSING' if n is None else f'{n / 1e9:.2f} GB'}")
        total += n or 0
    print(f"total {total / 1e9:.1f} GB")


@app.local_entrypoint()
def new(video: str, photo: str, consent: str = "", margin: str = "normal", start: float = 0.0, length: float = 0.0,
        seed: int = 0, resolution: str = "480p", role: str = "", target: str = "person", hat_fix: bool = False,
        no_lip_sync: bool = False, motion: int = 512, clip_seconds: float = 6.0):
    if consent != "yes":
        raise SystemExit("Only use photos and videos of people who agreed to it (or fictional characters). Re-run with --consent yes.")
    settings = dict(margin=margin, start=start, length=length, seed=seed, resolution=resolution, role=role, target=target,
                    hat_fix=hat_fix, no_lip_sync=no_lip_sync, motion=motion, clip_seconds=clip_seconds)
    st = new_job.remote(Path(video).read_bytes(), Path(photo).read_bytes(), Path(photo).suffix, consent, settings)
    print(json.dumps(st, indent=1))
    print(f"Next: modal run h3_app.py::check --job {st['job']}")


@app.local_entrypoint()
def check(job: str, out: str = "check.mp4"):
    queue.remote(job, "check")
    print(json.dumps(check_job.remote(job), indent=1))
    Path(out).write_bytes(read_result.remote(job, "check"))
    print("Saved", out)


@app.local_entrypoint()
def swap(job: str, out: str = "h3.mp4"):
    if queue.remote(job, "swap"):
        print("Resuming from the last finished clip.")
    print(json.dumps(swap_job.remote(job), indent=1))
    Path(out).write_bytes(read_result.remote(job, "swap"))
    print("Saved", out)


@app.local_entrypoint()
def status(job: str):
    print(json.dumps(job_status.remote(job), indent=1))
