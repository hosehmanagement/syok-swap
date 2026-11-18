"""Train a Wan 2.2 14B LoRA on Modal with musubi-tuner (Apache-2.0). Nothing here runs unless you call it.

    modal run lora/train_modal.py::doctor                       # CPU, cents: pins/prints the musubi commit, checks every flag exists
    modal run lora/train_modal.py::fetch                        # CPU, ~40 GB download into the volume (no GPU cost)
    modal run lora/train_modal.py::upload --src prepared/       # copy the output of lora/dataset.py::prepare into the volume
    modal run lora/train_modal.py::train --name me --yes        # H100; prints the cost ceiling first and refuses without --yes

Cost control: the GPU function has a hard 1-hour timeout (about $4 at the estimated H100 rate), steps are capped,
and a checkpoint is saved every 100 steps to the volume, so a stopped run keeps its partial LoRA.
Defaults train ONLY the low-noise expert (skin texture/detail). The high-noise expert (layout/motion) is optional.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lora import commands as C  # noqa: E402

MUSUBI_REPO = "https://github.com/kohya-ss/musubi-tuner"
MUSUBI_REF = "main"    # NOT PINNED YET: `doctor` prints the commit; copy it here once the flags check passes
VOL = "/data"
MODELS = f"{VOL}/models"
TIMEOUT_S = 3600

app = modal.App("syok-lora")
vol = modal.Volume.from_name("syok-lora-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", "huggingface_hub[hf_transfer]")
    .run_commands(f"git clone {MUSUBI_REPO} /opt/musubi && cd /opt/musubi && git checkout {MUSUBI_REF} && pip install -e . && git rev-parse HEAD")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("lora")   # ship our own package; without this the container fails with "No module named 'lora'"
)

FILES = [  # (repo, path in repo, local name) verified present on the Hub on 2026-11-18
    ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors", "wan2.2_t2v_low_noise_14B_fp16.safetensors"),
    ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan_2.1_vae.safetensors", "wan_2.1_vae.safetensors"),
    ("Wan-AI/Wan2.1-T2V-14B", "models_t5_umt5-xxl-enc-bf16.pth", "models_t5_umt5-xxl-enc-bf16.pth"),
]
HIGH = ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors", "wan2.2_t2v_high_noise_14B_fp16.safetensors")


def _help(script):
    r = subprocess.run(["python", script, "--help"], cwd="/opt/musubi", capture_output=True, text=True)
    return r.stdout + r.stderr


@app.function(image=image, cpu=2, timeout=900)
def doctor():
    """Cheap pre-flight: commit in use, scripts exist, and every flag we pass is accepted by this musubi version."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd="/opt/musubi", capture_output=True, text=True).stdout.strip()
    problems = []
    for script, flags in ((C.CACHE_LATENTS, C.FLAGS_CACHE_LATENTS), (C.CACHE_TEXT, C.FLAGS_CACHE_TEXT), (C.TRAINER, C.FLAGS_TRAIN)):
        if not Path("/opt/musubi", script).exists():
            problems.append(f"missing script {script}")
            continue
        h = _help(script)
        miss = C.flags_missing(h, flags)
        if miss:
            problems.append(f"{script}: unknown flags {miss}")
            print(f"--- raw `--help` output for {script} (last 700 chars) ---\n{h[-700:]}\n---")
    anim = subprocess.run("grep -ril animate /opt/musubi/src /opt/musubi/docs | head -5", shell=True, capture_output=True, text=True).stdout
    print("musubi commit:", commit)
    print("mentions of 'animate' in musubi (training the Animate checkpoint directly):", anim.strip() or "none")
    print("PROBLEMS:" if problems else "all flags accepted", *problems, sep="\n  ")
    return {"commit": commit, "problems": problems, "animate_mentions": anim.strip()}


@app.function(image=image, volumes={VOL: vol}, cpu=4, timeout=7200)
def fetch(include_high: bool = False):
    from huggingface_hub import hf_hub_download
    Path(MODELS).mkdir(parents=True, exist_ok=True)
    for repo, path, local in FILES + ([HIGH] if include_high else []):
        dst = Path(MODELS) / local
        if dst.exists() and dst.stat().st_size > 1_000_000:
            continue
        p = hf_hub_download(repo, path, local_dir="/tmp/hf")
        shutil.move(p, dst)
    vol.commit()
    return {p.name: round(p.stat().st_size / 1e9, 1) for p in Path(MODELS).iterdir()}


@app.function(image=image, volumes={VOL: vol}, gpu="H100", timeout=TIMEOUT_S)
def train_job(name: str, stage: str = "low", steps: int = 600, rank: int = 16, lr: float = 1e-4):
    plan = C.Plan(name=name, dataset_toml=f"{VOL}/lora/{name}/dataset.toml", models_dir=MODELS,
                  out_dir=f"{VOL}/lora/{name}/out", stage=stage, steps=steps, rank=rank, lr=lr, save_every=100)
    Path(plan.out_dir).mkdir(parents=True, exist_ok=True)
    for cmd in (C.cache_latents_cmd(plan), C.cache_text_cmd(plan), C.train_cmd(plan)):
        print("RUN", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd="/opt/musubi")
        vol.commit()                                   # keep caches and checkpoints even if a later step fails
        if r.returncode:
            raise RuntimeError(f"step failed ({r.returncode}): {' '.join(cmd[:3])}")
    return sorted(p.name for p in Path(plan.out_dir).glob("*.safetensors"))


@app.local_entrypoint()
def upload(src: str, name: str = "me"):
    """src = folder written by lora.dataset.prepare (images/, dataset.toml, manifest.json)."""
    s = Path(src)
    if not (s / "dataset.toml").exists() or not (s / "manifest.json").exists():
        raise SystemExit("Run lora.dataset.prepare first (it checks consent and photo quality).")
    with vol.batch_upload(force=True) as b:
        b.put_directory(str(s / "images"), f"/lora/{name}/images")
        # the toml must point at the volume paths, not at your local paths
        (s / "dataset.volume.toml").write_text((s / "dataset.toml").read_text()
                                               .replace(str(s / "images"), f"{VOL}/lora/{name}/images")
                                               .replace(str(s / "cache"), f"{VOL}/lora/{name}/cache"))
        b.put_file(str(s / "dataset.volume.toml"), f"/lora/{name}/dataset.toml")
        b.put_file(str(s / "manifest.json"), f"/lora/{name}/manifest.json")
    print("uploaded", name)


@app.local_entrypoint()
def train(name: str = "me", stage: str = "low", steps: int = 600, rank: int = 16, yes: bool = False):
    C.train_cmd(C.Plan(name, "x", "x", "x", stage=stage, steps=steps, rank=rank))   # validates stage and the step cap
    print(f"H100 hard cap {TIMEOUT_S // 60} min = about ${C.cost_ceiling(TIMEOUT_S)} at the estimated rate (check the dashboard). "
          f"Plan: stage={stage} steps={steps} rank={rank}; a checkpoint every 100 steps is kept.")
    if not yes:
        raise SystemExit("Not started. Re-run with --yes to spend this.")
    print(train_job.remote(name, stage, steps, rank))
