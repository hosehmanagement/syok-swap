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

## Verified on 2026-11-18 (from real file headers, no GPU)
- All 400 attention/FFN linear layers of Wan2.2-T2V-A14B (low noise) exist in Wan2.2-Animate-14B with identical names and shapes; Animate only adds k_img/v_img and a face adapter. A musubi-style LoRA trained on T2V low-noise maps onto the ORIGINAL Wan Animate checkpoint: 400 of 400 modules, 0 unmatched (checked against its real 1441-weight index). Same check on the relight LoRA: 480 of 480.
- Wan's Animate loader accepts one PEFT adapter (rank 128), used by the relight LoRA, so the identity LoRA is MERGED into the base weights (lora/merge.py, tested: math, three key styles, dtype, refuses a wrong base model). `modal run app.py::render --job X --lora /cache/loras/me_low.safetensors --lora-strength 1.0` patches Wan's animate.py (patch tested against the real file at the pinned commit) so the merge happens before the relight LoRA loads.
- musubi-tuner pinned to f8a1b037; `doctor` passed (flag names exist in its source). Its `--help` needs a GPU, so flags are read from source.

## Still unverified (needs a paid training run, est. up to ~$3.95)
- That the merged LoRA actually makes the swapped face look better (pores/stubble) rather than worse or changing identity. Compatibility is proven; benefit is not.
- Musubi cannot train the Animate checkpoint itself (no Animate support), so the LoRA is trained on T2V low-noise and merged; strength needs tuning.
- Training settings (rank 16, lr 1e-4, shift 1.0, 600 steps) are community starting points.
- Photos-only training may not teach motion; only the low-noise expert is trained.

## Cost
Hard 1 h cap on an H100 (est. about $3.95, check Modal's price page). Not run yet.
