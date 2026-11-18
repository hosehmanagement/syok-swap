"""Merge a trained LoRA into Wan2.2-Animate's base weights (free, CPU-testable).

Why merge: Wan's Animate loader accepts ONE PEFT adapter (rank 128), and the relight LoRA already uses it.
Verified 2026-11-18 from the safetensors headers: the 400 attention/FFN linears of Wan2.2-T2V-A14B (low noise) exist in
Wan2.2-Animate-14B with identical names and shapes (Animate adds k_img/v_img and a face adapter, which a T2V LoRA never touches),
so a LoRA trained on T2V low-noise maps onto Animate module-for-module. NOT verified: that the merged result looks better.

Accepted LoRA key styles (musubi/kohya, ComfyUI, PEFT):
  lora_unet_blocks_0_self_attn_q.lora_down.weight / .lora_up.weight / .alpha
  diffusion_model.blocks.0.self_attn.q.lora_down.weight / .lora_up.weight
  blocks.0.self_attn.q.lora_A.weight / .lora_B.weight
Update per layer:  W += strength * (alpha / rank) * (up @ down)
"""
import re

SUFFIXES = (("lora_down.weight", "lora_up.weight"), ("lora_A.weight", "lora_B.weight"))
PREFIXES = ("lora_unet_", "diffusion_model.", "transformer.", "base_model.model.")


def _norm(name):
    for p in PREFIXES:
        if name.startswith(p):
            name = name[len(p):]
    return name.replace(".", "_")


def group_lora(lora_sd):
    """{normalised module name: (down, up, alpha_or_None)} from a flat LoRA state dict."""
    groups = {}
    for key in lora_sd:
        for down_s, up_s in SUFFIXES:
            if key.endswith("." + down_s):
                mod = key[: -len(down_s) - 1]
                up_key = mod + "." + up_s
                if up_key not in lora_sd:
                    raise KeyError(f"LoRA has {key} but no {up_key}")
                a = lora_sd.get(mod + ".alpha")
                groups[_norm(mod)] = (lora_sd[key], lora_sd[up_key], None if a is None else float(a))
    if not groups:
        raise ValueError("No LoRA down/up pairs found in this file.")
    return groups


def plan(weight_names, lora_sd):
    """Match LoRA modules to base weights. Returns ({weight name: lora module key}, unmatched LoRA modules)."""
    by_mod = {_norm(re.sub(r"\.weight$", "", n)): n for n in weight_names if n.endswith(".weight")}
    groups = group_lora(lora_sd)
    hit = {by_mod[m]: m for m in groups if m in by_mod}
    return hit, sorted(m for m in groups if m not in by_mod)


def merge_into(state, lora_sd, strength=1.0, max_unmatched=0):
    """In place: add the LoRA delta to every matching weight in `state` (dict name -> tensor). Returns a report.

    Refuses if more than max_unmatched LoRA modules have no base weight (a sign of a wrong base model),
    or if a delta does not fit its weight's shape.
    """
    hit, unmatched = plan(state.keys(), lora_sd)
    if len(unmatched) > max_unmatched:
        raise ValueError(f"{len(unmatched)} LoRA modules match nothing in the base model, e.g. {unmatched[:3]}")
    groups = group_lora(lora_sd)
    for wname, mod in hit.items():
        down, up, alpha = groups[mod]
        rank = down.shape[0]
        scale = strength * (1.0 if alpha is None else alpha / rank)
        w = state[wname]
        delta = (up.float() @ down.float()) * scale
        if tuple(delta.shape) != tuple(w.shape):
            raise ValueError(f"{wname}: LoRA delta {tuple(delta.shape)} does not fit weight {tuple(w.shape)}")
        state[wname] = (w.float() + delta).to(w.dtype)
    return {"merged": len(hit), "unmatched": unmatched, "strength": strength}


def merge_into_model(model, lora_path, strength=1.0):
    """Merge a .safetensors LoRA into a loaded torch model's linear weights (used inside the render container)."""
    import torch
    from safetensors.torch import load_file
    lora_sd = load_file(lora_path)
    sd = {n: p for n, p in model.named_parameters() if n.endswith(".weight")}
    hit, unmatched = plan(sd.keys(), lora_sd)
    if unmatched:
        raise ValueError(f"{len(unmatched)} LoRA modules match nothing in the model, e.g. {unmatched[:3]}")
    groups = group_lora(lora_sd)
    with torch.no_grad():
        for wname, mod in hit.items():
            down, up, alpha = groups[mod]
            scale = strength * (1.0 if alpha is None else float(alpha) / down.shape[0])
            p = sd[wname]
            p.add_(((up.to(p.device).float() @ down.to(p.device).float()) * scale).to(p.dtype))
    return {"merged": len(hit), "strength": strength}


ANCHOR = '        if use_lora:\n            logging.info("Loading Relighting Lora. ")'
HOOK = (
    '        if os.environ.get("SYOK_LORA"):\n'
    '            from lora.merge import merge_into_model\n'
    '            print("SYOK_LORA merge:", merge_into_model(model, os.environ["SYOK_LORA"], float(os.environ.get("SYOK_LORA_STRENGTH", "1"))), flush=True)\n'
)


def patch_animate_source(text):
    """Insert the merge hook just before Wan's relight-LoRA block (base weights get the identity LoRA first,
    then the single PEFT adapter slot still loads the relight LoRA as before). Idempotent; refuses if Wan's code moved."""
    if "SYOK_LORA" in text:
        return text
    if text.count(ANCHOR) != 1:
        raise RuntimeError("Wan's animate.py changed: cannot find the one place to merge the LoRA. Re-check the pinned commit.")
    return text.replace(ANCHOR, HOOK + ANCHOR)
