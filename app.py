"""Syok Swap: replace the person in a video with a character from one photo.

Engine: Wan2.2-Animate-14B (Apache-2.0) in replacement mode, on Modal.
Two steps so you never pay for a bad mask:
  1. preview  (small GPU, cents): finds the person, writes a red-overlay mask video + coverage stats.
  2. render   (H100): generates, pastes the result over the ORIGINAL frames, restores the ORIGINAL soundtrack.

    modal run app.py::fetch                                   # one time, downloads ~50 GB of weights
    modal run app.py::preview --video v.mp4 --image face.png --consent yes
    modal run app.py::render  --job <id>

The same two steps (plus Syok's texture slider on the result, and the optional face-lock pass that uses the user's own Magic
Hour account) are also served as ONE web app for the Syok "Recast -> Syok Swap" card, in the user's own Modal account:
    modal deploy app.py     (see docs/SWAP.md).
"""
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import modal

WAN_REF = "1ea34ff48f87168174e12956e200b1d908b1c5ff"  # Wan-Video/Wan2.2, pinned
WEIGHTS = "/cache/Wan2.2-Animate-14B"
JOBS = "/cache/jobs"

app = modal.App("syok-swap")
cache = modal.Volume.from_name("syok-swap-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.6.0", "torchvision==0.21.0")
    .run_commands(f"git clone https://github.com/Wan-Video/Wan2.2 /opt/wan && cd /opt/wan && git checkout {WAN_REF}")
    .run_commands("cd /opt/wan && grep -v flash_attn requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt")
    .run_commands("cd /opt/wan && grep -v -i sam2 requirements_animate.txt > /tmp/reqa.txt && pip install -r /tmp/reqa.txt")
    # SAM2's setup.py imports torch while building, so build without isolation; skip its CUDA extension (no nvcc here)
    .run_commands("pip install hydra-core iopath 'moviepy<2' && git clone -q https://github.com/facebookresearch/sam2.git /tmp/sam2 && cd /tmp/sam2 && git checkout -q 0e78a118995e66bb27d78518c4bd9a3e95b4e266 && SP=\"$(python -c 'import site;print(site.getsitepackages()[0])')\" && cp -r sam2 \"$SP/sam2\" && cp -r sam2_configs \"$SP/sam2_configs\" && python -c 'import sam2, os; print(sam2.__file__)'")
    # SAM2's connected-components op is a CUDA extension we don't build; patch in the tested CPU fallback
    .add_local_file(Path(__file__).parent / "sam2_cc_fallback.py", "/opt/sam2_cc_fallback.py", copy=True)
    .add_local_file(Path(__file__).parent / "patch_sam2.py", "/opt/patch_sam2.py", copy=True)
    .run_commands("python /opt/patch_sam2.py")
    # Wan's mask grid (--w_len/--h_len): a person narrower than 10 px or shorter than 20 px in some frame would make a grid cell
    # 0 px and crash preprocess; patch the cell size to at least 1 px (fails this cheap CPU build if Wan's file ever changes)
    .add_local_file(Path(__file__).parent / "patch_wan_mask.py", "/opt/patch_wan_mask.py", copy=True)
    .run_commands("python /opt/patch_wan_mask.py /opt/wan/wan/modules/animate/preprocess/utils.py")
    # SAM2's own person mask next to Wan's loose grid mask (src_mask_raw.mp4): the tight matte for the two-mask composite
    .add_local_file(Path(__file__).parent / "patch_wan_raw_mask.py", "/opt/patch_wan_raw_mask.py", copy=True)
    .run_commands("python /opt/patch_wan_raw_mask.py /opt/wan/wan/modules/animate/preprocess/process_pipepline.py")
    .pip_install("huggingface_hub[hf_transfer]")
    # imported by Wan's package but missing from its requirements files (found when a paid H100 run failed on einops)
    .pip_install("einops", "regex", "safetensors", "librosa", "scipy")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": "/opt/wan"})
    # fail the cheap CPU image build, not a paid GPU run, if any import Wan needs is missing (Wan itself touches CUDA on import, so check the packages)
    .run_commands("python -c 'import einops, regex, safetensors, librosa, scipy, decord, peft, onnxruntime, pandas, matplotlib, loguru, sentencepiece, easydict, ftfy, dashscope, imageio, cv2, diffusers, transformers, tokenizers, accelerate, omegaconf, hydra, moviepy, sam2, PIL, tqdm; print(\"third-party imports ok\")'")
    # Wan's attention asserts FlashAttention-2 is installed (it was left out of the requirements because it is normally compiled
    # from source). Use the official prebuilt wheel that matches torch 2.6 / CUDA 12 / python 3.11.
    .run_commands("pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl' && python -c \"import importlib.metadata as m; print('flash_attn', m.version('flash_attn'))\"")
    .add_local_file(Path(__file__).parent / "swapkit.py", "/opt/swapkit.py")
    .add_local_file(Path(__file__).parent / "texture_transfer.py", "/opt/texture_transfer.py")
    .add_local_file(Path(__file__).parent / "texture_service.py", "/opt/texture_service.py")
    .add_local_file(Path(__file__).parent / "swap_service.py", "/opt/swap_service.py")
    .add_local_file(Path(__file__).parent / "picker.py", "/opt/picker.py")            # swap_service imports it
    .add_local_file(Path(__file__).parent / "lora" / "face_detection_yunet_2023mar.onnx", "/opt/face_detection_yunet_2023mar.onnx")  # head detection for the mask
    .add_local_python_source("lora")   # lora/merge.py: merge an identity LoRA into the base weights at render time
)

# The small CPU image for everything that needs no model: the web endpoint, the slider's face tracking and render, the sweeper.
HERE = Path(__file__).parent
light = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "opencv-python-headless<5", "fastapi[standard]", "pillow<13")
    .add_local_file(HERE / "swapkit.py", "/opt/swapkit.py")
    .add_local_file(HERE / "texture_transfer.py", "/opt/texture_transfer.py")
    .add_local_file(HERE / "texture_service.py", "/opt/texture_service.py")
    .add_local_file(HERE / "swap_service.py", "/opt/swap_service.py")
    .add_local_file(HERE / "swap_api.py", "/opt/swap_api.py")
    .add_local_file(HERE / "facelock.py", "/opt/facelock.py")
    .add_local_file(HERE / "picker.py", "/opt/picker.py")              # best-photo pick for face-lock
    .add_local_file(HERE / "collage.py", "/opt/collage.py")            # EXPERIMENTAL collage reference, made at upload
    .add_local_file(HERE / "lora" / "face_detection_yunet_2023mar.onnx", "/opt/face_detection_yunet_2023mar.onnx")  # its detector
)
WEIGHTS_FLAGS = "/cache"      # .weights-downloading / .weights-ok live next to the model folder


@app.function(image=image, volumes={"/cache": cache}, timeout=7200, cpu=4)
def fetch_weights():
    from huggingface_hub import snapshot_download
    flag, ok = Path(WEIGHTS_FLAGS) / ".weights-downloading", Path(WEIGHTS_FLAGS) / ".weights-ok"
    flag.write_text(str(time.time()))
    cache.commit()                       # so the web endpoint can tell "downloading" from "missing"
    snapshot_download("Wan-AI/Wan2.2-Animate-14B", local_dir=WEIGHTS)
    ok.write_text("ok")
    flag.unlink(missing_ok=True)
    cache.commit()
    return sorted(p.name for p in Path(WEIGHTS).iterdir())


def _sk():
    import sys
    sys.path.insert(0, "/opt")
    import swapkit
    return swapkit


def _preview_core(sk, d, raw, ref, resolution, margin, start, length, mask_shape=None):
    """Find the person and write mask.mp4 + preview.mp4 (red overlay) into job folder `d`. Returns the numbers the UI shows.
    mask_shape "silhouette" (default): Wan's mask grid follows the person and the head is added as an ellipse; "box": the old
    whole-bounding-box mask plus head rectangle (swapkit.MASK_SHAPES)."""
    mask_shape = mask_shape or sk.DEFAULT_MASK_SHAPE
    info = sk.probe(raw)
    s, l = sk.plan_section(info["duration"], start, length)
    sec = d / "section.mp4"
    sk.cut_section(raw, sec, s, l)
    mask_args = sk.preprocess_mask_args(margin, mask_shape)
    w, h = sk.RESOLUTIONS[resolution]
    pre = "/opt/wan/wan/modules/animate/preprocess"
    (d / "proc").mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["python", "preprocess_data.py", "--ckpt_path", f"{WEIGHTS}/process_checkpoint", "--video_path", str(sec),
         "--refer_path", str(ref), "--save_path", str(d / "proc"), "--resolution_area", str(w), str(h),
         "--replace_flag", *mask_args],
        cwd=pre, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("preprocess failed:\n" + (r.stdout + r.stderr)[-2500:])
    # Wan's mask can leave the head out (SAM2 body points), which keeps the ORIGINAL head in the background
    head = sk.cover_head_files(d / "proc", sec, shape=mask_shape)
    mask, inverted = sk.normalize_mask(d / "proc" / "src_mask.mp4", d / "mask.mp4")
    if (d / "proc" / "src_core.mp4").exists():        # the person's own outline + head: the tight matte for the final composite
        shutil.copy(d / "proc" / "src_core.mp4", d / "core.mp4")
    stats = sk.mask_stats(mask)
    stats["mask_shape"] = mask_shape
    stats["box_mean"] = head["coverage"]["box_mean"]     # what the old box mask would have covered on this clip
    sk.mask_preview(sec, mask, d / "preview.mp4")
    return {"section": [s, l], "resolution": resolution, "margin": margin, "mask_shape": mask_shape,
            "mask_inverted": inverted, "stats": stats, "head": head}


@app.function(image=image, gpu=["L4", "A10G", "L40S"], volumes={"/cache": cache}, timeout=1800)
def make_preview(video: bytes, photo: bytes, photo_ext: str, resolution: str, margin: str, start: float, length: float,
                 mask_shape: str = "silhouette"):
    sk = _sk()
    cache.reload()
    job = uuid.uuid4().hex[:10]
    d = Path(JOBS) / job
    (d / "proc").mkdir(parents=True)
    raw = d / "raw.mp4"
    raw.write_bytes(video)
    ref = d / f"person{photo_ext}"
    ref.write_bytes(photo)
    meta = {"job": job, **_preview_core(sk, d, raw, ref, resolution, margin, start, length, mask_shape)}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    cache.commit()
    return meta, (d / "preview.mp4").read_bytes()


def _composite(sk, folder, section, gen, mask, out, feather, full_res):
    """Final clip = ORIGINAL pixels everywhere the new person is not (two-mask composite) when the job has the person's own
    outline (core.mp4); older jobs fall back to the previous behaviour (Wan's whole frame)."""
    core = Path(folder) / "core.mp4"
    if core.exists():
        return sk.composite_two_mask(section, gen, mask, core, out, feather=feather, full_res=full_res)
    return sk.composite_over_original(section, gen, mask, out, feather=feather, full_res=full_res)


def _render_core(sk, d, relight, feather, full_res, lora, lora_strength):
    """Generate the new person on the GPU, paste it over the ORIGINAL frames, restore the ORIGINAL soundtrack.
    Leaves generated.mp4 and final.mp4 in job folder `d` and returns whether the original had audio."""
    # work on local disk, not the network volume: generate.py reads src_ref.png with cv2.imread, which returned None
    # for the volume copy inside the GPU container (the same file reads fine from a CPU container)
    w = Path("/tmp/work")
    shutil.rmtree(w, ignore_errors=True)
    shutil.copytree(d / "proc", w / "proc")
    for name in ("section.mp4", "mask.mp4", "core.mp4"):
        if (d / name).exists():                        # core.mp4 is missing on jobs masked before the two-mask composite
            shutil.copy(d / name, w / name)
    import cv2
    # Inside generate.py, cv2.imread of a PNG dies with "libpng error: bad parameters to zlib" (a zlib clash with another
    # library loaded in that process). cv2.imread picks the decoder from the file CONTENT, so store the reference as BMP
    # bytes under the name Wan expects: same pixels, no zlib involved.
    ref_png = w / "proc" / "src_ref.png"
    img = cv2.imread(str(ref_png))
    if img is None:
        raise RuntimeError(f"Reference image unreadable: {ref_png.stat().st_size} bytes")
    ok, buf = cv2.imencode(".bmp", img)
    if not ok:
        raise RuntimeError("Could not re-encode the reference image")
    ref_png.write_bytes(buf.tobytes())
    gen = w / "generated.mp4"
    cmd = ["python", "generate.py", "--task", "animate-14B", "--ckpt_dir", WEIGHTS,
           "--src_root_path", str(w / "proc"), "--refert_num", "1", "--replace_flag", "--save_file", str(gen)]
    if relight:
        cmd.append("--use_relighting_lora")
    env = dict(os.environ)
    if lora:                                       # path of a .safetensors LoRA on the volume, e.g. /cache/loras/me_low.safetensors
        from lora.merge import patch_animate_source
        src = Path("/opt/wan/wan/animate.py")
        src.write_text(patch_animate_source(src.read_text()))
        env.update(SYOK_LORA=lora, SYOK_LORA_STRENGTH=str(lora_strength))
    r = subprocess.run(cmd, cwd="/opt/wan", capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError("generate failed:\n" + (r.stdout + r.stderr)[-2500:])
    comp = w / "composite.mp4"
    _composite(sk, w, w / "section.mp4", gen, w / "mask.mp4", comp, feather, full_res)
    final, had_audio = sk.mux_original_audio(comp, w / "section.mp4", w / "final.mp4")
    shutil.copy(gen, d / "generated.mp4")
    shutil.copy(final, d / "final.mp4")
    return had_audio




@app.function(image=image, gpu="H100", volumes={"/cache": cache}, timeout=3600)
def render_job(job: str, relight: bool = True, feather: int = 7, full_res: bool = False, lora: str = "", lora_strength: float = 1.0):
    sk = _sk()
    cache.reload()
    d = Path(JOBS) / job
    meta = json.loads((d / "meta.json").read_text())
    if meta["stats"]["warnings"] and any("empty" in x for x in meta["stats"]["warnings"]):
        raise ValueError("Refusing to render: " + "; ".join(meta["stats"]["warnings"]))
    had_audio = _render_core(sk, d, relight, feather, full_res, lora, lora_strength)
    meta.update(done=True, had_audio=had_audio)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    cache.commit()
    return (d / "final.mp4").read_bytes(), (d / "generated.mp4").read_bytes()


# ---- the one-flow job service used by the Syok card (swap_service.py / swap_api.py): mask -> swap -> slider -> save ----

def _svc():
    if "/opt" not in sys.path:
        sys.path.insert(0, "/opt")
    import swap_service
    import texture_service
    return swap_service, texture_service


def _friendly(e, fallback):
    """Our own refusals (ValueError) are written for the user and safe to show; anything else gets the plain fallback."""
    return str(e) if isinstance(e, ValueError) else fallback


@app.function(image=image, gpu=["L4", "A10G", "L40S"], volumes={"/cache": cache}, timeout=3600, max_containers=2)
def mask_job(job: str):
    """Step 1 (cents): find the person and make the red-overlay preview. A bad mask costs almost nothing to find out here."""
    sk = _sk()
    ss, ts = _svc()
    cache.reload()
    q = ss.request_of(JOBS, job)        # q["ref"]: the collage (ref_collage.png) when the job asked for it, else the main photo
    d = q["dir"]
    ts.set_stage(d, "mask", state="preparing")
    cache.commit()
    try:
        m = _preview_core(sk, d, d / "raw.mp4", d / q["ref"], q["resolution"], q["margin"], q["start"], q["length"],
                          q.get("mask_shape"))
        ss.mask_ready(JOBS, job, m["section"], m["stats"], m["head"])
    except Exception as e:
        ss.fail(JOBS, job, "mask", _friendly(e, "The mask step failed. Nothing was swapped and no swap was charged. Try a clearer clip."))
        raise
    finally:
        cache.commit()


@app.function(image=image, gpu="H100", volumes={"/cache": cache}, timeout=10800, max_containers=1)
def swap_job(job: str):
    """Step 2 (the paid one): generate the new person, then hand the swapped clip to the slider's face tracking."""
    sk = _sk()
    ss, ts = _svc()
    cache.reload()
    d = ts.job_dir(JOBS, job)
    meta = ts.read_meta(d)
    relight = bool(meta.get("swap", {}).get("relight", True))
    ts.set_stage(d, "swap", state="generating")
    cache.commit()
    try:
        had_audio = _render_core(sk, d, relight, 7, False, "", 1.0)
        prepare = ss.finish_swap(JOBS, job, d / "final.mp4", had_audio)
        (d / "final.mp4").unlink(missing_ok=True)        # final.mp4 is the TEXTURED result from here on; the swapped clip is src.mp4
        cache.commit()
    except Exception as e:
        ss.fail(JOBS, job, "swap", _friendly(e, "The swap failed. Try again; the mask is kept, so nothing needs to be uploaded again."))
        cache.commit()
        raise
    if not prepare:                                      # face-lock asked for: it waits for the user's Magic Hour key first
        return
    try:
        prepare_job.spawn(job)                           # CPU, cents: the slider can be used as soon as this is done
    except Exception:
        ss.fail(JOBS, job, "prepare", "The swap is done, but preparing the texture slider could not start. You can still download the swapped clip.")
        cache.commit()
        raise


@app.function(image=light, volumes={"/cache": cache}, cpu=2, timeout=1800)
def prepare_job(job: str):
    cache.reload()
    _svc()[1].run_prepare(JOBS, job)
    cache.commit()


@app.function(image=light, volumes={"/cache": cache}, cpu=2, timeout=1800)
def facelock_job(job: str, key: str):
    """Optional face-lock (CPU, the user's own Magic Hour credits): the exact face from the photo onto the swapped clip, then
    the slider is prepared again. The key is only an argument here: never logged, never written to the volume."""
    cache.reload()
    if "/opt" not in sys.path:
        sys.path.insert(0, "/opt")
    import facelock
    ss, ts = _svc()
    prepare = facelock.run(JOBS, job, key, on_progress=cache.commit)
    cache.commit()
    if not prepare:
        return
    try:
        prepare_job.spawn(job)
    except Exception:
        ss.fail(JOBS, job, "prepare", "Preparing the texture slider could not start. Try again.")
        cache.commit()
        raise RuntimeError("prepare_job could not be spawned") from None


@app.function(image=light, volumes={"/cache": cache}, cpu=2, timeout=3600)
def texture_render_job(job: str):
    cache.reload()
    _svc()[1].run_render(JOBS, job)
    cache.commit()


@app.function(image=light, volumes={"/cache": cache}, schedule=modal.Period(hours=6), timeout=600)
def sweep_old_jobs():
    """Jobs (uploads, the swapped clip, the result) are deleted after 3 days, the same time the result is kept in the user's Cloudflare."""
    cache.reload()
    n = _svc()[1].sweep(JOBS)
    cache.commit()
    return n


@app.function(image=light, volumes={"/cache": cache}, cpu=2, timeout=600, max_containers=3)
@modal.concurrent(max_inputs=4)
@modal.asgi_app(requires_proxy_auth=True)
def web():
    if "/opt" not in sys.path:
        sys.path.insert(0, "/opt")
    import swap_api
    import swap_service

    def reload():
        try:
            cache.reload()
        except Exception:      # a volume with files open by a concurrent request cannot reload; the next request will
            pass

    def weights_state():
        return swap_service.weights_state(WEIGHTS, WEIGHTS_FLAGS)

    return swap_api.build_app(JOBS, mask_job.spawn, swap_job.spawn, texture_render_job.spawn, fetch_weights.spawn,
                              weights_state, before_read=reload, after_write=cache.commit,
                              spawn_prepare=prepare_job.spawn, spawn_facelock=facelock_job.spawn)


@app.function(image=image, volumes={"/cache": cache}, cpu=4, timeout=900)
def recomposite_job(job: str, feather: int = 7):
    """CPU only, no GPU cost: rebuild the final clip at the GENERATED size from files a finished render already saved."""
    sk = _sk()
    cache.reload()
    d = Path(JOBS) / job
    for name in ("generated.mp4", "section.mp4", "mask.mp4"):
        if not (d / name).exists():
            raise FileNotFoundError(f"{name} not found in job {job}; render it first.")
    w = Path("/tmp/recomp")
    shutil.rmtree(w, ignore_errors=True)
    w.mkdir()
    comp = w / "composite.mp4"
    _composite(sk, d, d / "section.mp4", d / "generated.mp4", d / "mask.mp4", comp, feather, True)
    final, _ = sk.mux_original_audio(comp, d / "section.mp4", w / "final_full.mp4")
    shutil.copy(final, d / "final_full.mp4")
    cache.commit()
    return (d / "final_full.mp4").read_bytes()


@app.local_entrypoint()
def recomposite(job: str, out: str = "swap_full.mp4"):
    Path(out).write_bytes(recomposite_job.remote(job))
    print("Saved", out)


@app.function(image=image, volumes={"/cache": cache}, timeout=300)
def inspect_job(job: str):
    """Cheap CPU helper: list a job's files and check the reference image reads (debugging aid)."""
    import cv2
    cache.reload()
    d = Path(JOBS) / job
    out = []
    for p in sorted(d.rglob("*")):
        if p.is_file():
            out.append(f"{p.relative_to(d)}  {p.stat().st_size}")
    ref = d / "proc" / "src_ref.png"
    out.append(f"imread(src_ref.png) -> {None if cv2.imread(str(ref)) is None else cv2.imread(str(ref)).shape}")
    return "\n".join(out)


@app.local_entrypoint()
def inspect(job: str):
    print(inspect_job.remote(job))


@app.local_entrypoint()
def fetch():
    print(fetch_weights.remote())


@app.local_entrypoint()
def preview(video: str, image: str, consent: str = "", resolution: str = "480p", margin: str = "normal",
            start: float = 0.0, length: float = 0.0, out: str = "preview.mp4", mask_shape: str = "silhouette"):
    if consent != "yes":
        raise SystemExit("Only use photos and videos of people who agreed to it (or fictional characters). Re-run with --consent yes.")
    ext = Path(image).suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        raise SystemExit("Character photo must be png, jpg or webp.")
    meta, pv = make_preview.remote(Path(video).read_bytes(), Path(image).read_bytes(), ext, resolution, margin, start, length,
                                   mask_shape)
    Path(out).write_bytes(pv)
    print(json.dumps(meta, indent=1))
    print(f"Mask preview saved to {out}. Check it, then: modal run app.py::render --job {meta['job']}")


@app.local_entrypoint()
def render(job: str, out: str = "swap.mp4", relight: bool = True, full_res: bool = False, lora: str = "", lora_strength: float = 1.0):
    final, raw = render_job.remote(job, relight, 7, full_res, lora, lora_strength)
    Path(out).write_bytes(final)
    Path(out.replace(".mp4", "_raw.mp4")).write_bytes(raw)
    print("Saved", out)


# ---- test helper: make a FICTIONAL character photo (SDXL base 1.0, OpenRAIL++; ungated - FLUX.1-schnell needs a gated HF login) so tests never use real faces ----
face_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "diffusers>=0.32", "transformers", "accelerate", "sentencepiece", "protobuf")
)


@app.function(image=face_image, gpu="L40S", volumes={"/cache": cache}, timeout=1200)
def test_face(prompt: str, seed: int = 7):
    import torch
    from diffusers import StableDiffusionXLPipeline
    p = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
                                              variant="fp16", cache_dir="/cache/hf").to("cuda")
    img = p(prompt, height=1024, width=768, num_inference_steps=28, guidance_scale=6.0,
            generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    import io
    b = io.BytesIO(); img.save(b, "PNG")
    cache.commit()
    return b.getvalue()


@app.local_entrypoint()
def face(prompt: str = "photorealistic portrait of a fictional middle-aged man with short black hair, thick eyebrows, brown eyes, stubble, dark blue jacket, plain grey background, soft daylight, looking at the camera",
         out: str = "fictional_man.png", seed: int = 7):
    Path(out).write_bytes(test_face.remote(prompt, seed))
    print("saved", out)
