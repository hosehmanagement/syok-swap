# Wan 2.2 14B LoRA pipeline (skin detail / identity)

Goal: a LoRA for the person's face that keeps pores and stubble, trained with captions that name that detail.

## Steps
1. Put 12+ sharp, close-up, different photos in a folder.
2. Consent for the WHOLE set: `lora.dataset.make_dataset_record(paths, "real" or "fictional", attest=..., who=..., contact=...)`, save with `consent.save`. Adding or swapping one photo invalidates it.
3. `lora.dataset.prepare(src, out, consent_path)` rejects small, blurry, duplicate and faceless photos, writes captions (trigger word + "visible pores, fine skin texture, stubble, no skin smoothing"), `dataset.toml`, `manifest.json`.
4. `modal run lora/train_modal.py::doctor` (cents), `::fetch`, `::upload --src out`, `::train --name me --yes`.

## What is tested vs not
Tested on CPU (tests/test_lora.py): consent binding, quality gates, captions, toml, command lines, step cap.
NOT tested: anything on Modal. `doctor` exists to catch wrong flag names for the installed musubi version for cents instead of GPU hours.

## Unverified (do not assume)
- That a LoRA trained on the T2V low-noise model loads into, and helps, **Wan2.2-Animate-14B** (our swap engine). Only the relight LoRA is known to load there. Musubi may not train the Animate checkpoint directly; `doctor` greps for it. A ComfyUI-format Animate file exists (wan2.2_animate_14B_bf16.safetensors).
- The settings (rank 16, lr 1e-4, shift 1.0, steps 600) come from community reports (musubi discussion #455), not from tests on this project.
- Musubi is pinned to f8a1b037 (2026-11-18). `doctor` passed: every flag we use appears in its source. That checks names only, not that training works. Its `--help` crashes without a GPU, so flags are read from source.
- Musubi has no Animate support (no mention in its source), so a T2V-trained LoRA cannot be tested or trained on the Animate checkpoint with it.
- Photos-only training may not teach motion; the low-noise expert is trained alone.

## Cost
Hard 1 h cap on an H100 (est. about $3.95, check Modal's price page). Not run yet.
