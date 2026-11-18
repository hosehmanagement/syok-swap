"""Pure builders for musubi-tuner command lines (Wan 2.2 14B). No I/O, so they are unit-tested on CPU.

Settings come from reports in musubi-tuner discussion #455 (character LoRAs: low ranks work, 1e-4 to 3e-5,
low-noise expert for detail, shift 1.0 for low noise). They are starting points, not tuned for this project.
UNVERIFIED until `modal run lora/train_modal.py::doctor` passes: exact flag names for the pinned musubi version.
"""
from dataclasses import dataclass

TRAINER = "src/musubi_tuner/wan_train_network.py"
CACHE_LATENTS = "src/musubi_tuner/wan_cache_latents.py"
CACHE_TEXT = "src/musubi_tuner/wan_cache_text_encoder_outputs.py"

# every long flag we pass; `doctor` greps --help for each so a typo costs cents, not a GPU hour
FLAGS_TRAIN = ["--task", "--dit", "--vae", "--t5", "--dataset_config", "--network_module", "--network_dim",
               "--network_alpha", "--learning_rate", "--max_train_steps", "--save_every_n_steps", "--timestep_sampling",
               "--discrete_flow_shift", "--min_timestep", "--max_timestep", "--fp8_base", "--fp8_scaled",
               "--gradient_checkpointing", "--optimizer_type", "--output_dir", "--output_name", "--seed", "--sdpa",
               "--mixed_precision"]
FLAGS_CACHE_LATENTS = ["--dataset_config", "--vae"]
FLAGS_CACHE_TEXT = ["--dataset_config", "--t5", "--batch_size"]

STAGES = {
    # name: (dit file stem, min_timestep, max_timestep, discrete_flow_shift)
    "low": ("wan2.2_t2v_low_noise_14B_fp16.safetensors", 0, 875, 1.0),     # skin texture, fine detail
    "high": ("wan2.2_t2v_high_noise_14B_fp16.safetensors", 875, 1000, 5.0),  # layout/motion; optional for stills
}


@dataclass
class Plan:
    name: str                 # output LoRA name
    dataset_toml: str
    models_dir: str
    out_dir: str
    stage: str = "low"
    steps: int = 1500
    rank: int = 16
    lr: float = 1e-4
    save_every: int = 250
    seed: int = 42


def cache_latents_cmd(p):
    return ["python", CACHE_LATENTS, "--dataset_config", p.dataset_toml, "--vae", f"{p.models_dir}/wan_2.1_vae.safetensors"]


def cache_text_cmd(p):
    return ["python", CACHE_TEXT, "--dataset_config", p.dataset_toml,
            "--t5", f"{p.models_dir}/models_t5_umt5-xxl-enc-bf16.pth", "--batch_size", "8"]


def train_cmd(p):
    if p.stage not in STAGES:
        raise ValueError(f"stage must be one of {list(STAGES)}")
    if not (100 <= p.steps <= 6000):
        raise ValueError("steps must be 100..6000 (a hard cap, so a typo cannot run up a bill)")
    dit, lo, hi, shift = STAGES[p.stage]
    return ["accelerate", "launch", "--num_cpu_threads_per_process", "1", "--mixed_precision", "bf16", TRAINER,
            "--task", "t2v-A14B", "--dit", f"{p.models_dir}/{dit}", "--vae", f"{p.models_dir}/wan_2.1_vae.safetensors",
            "--t5", f"{p.models_dir}/models_t5_umt5-xxl-enc-bf16.pth", "--dataset_config", p.dataset_toml,
            "--sdpa", "--mixed_precision", "bf16", "--fp8_base", "--fp8_scaled", "--gradient_checkpointing",
            "--optimizer_type", "adamw8bit", "--learning_rate", str(p.lr),
            "--network_module", "networks.lora_wan", "--network_dim", str(p.rank), "--network_alpha", str(p.rank),
            "--timestep_sampling", "shift", "--discrete_flow_shift", str(shift),
            "--min_timestep", str(lo), "--max_timestep", str(hi),
            "--max_train_steps", str(p.steps), "--save_every_n_steps", str(p.save_every),
            "--seed", str(p.seed), "--output_dir", p.out_dir, "--output_name", f"{p.name}_{p.stage}"]


def flags_missing(help_text, flags):
    return [f for f in flags if f not in help_text]


# H100 80GB on Modal, estimate only (check modal.com/pricing); used for a printed cost ceiling, never as a guarantee
H100_PER_HOUR = 3.95


def cost_ceiling(timeout_s, per_hour=H100_PER_HOUR):
    return round(timeout_s / 3600 * per_hour, 2)
