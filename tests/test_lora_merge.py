import pytest

torch = pytest.importorskip("torch")
from lora import merge as M


def _lora(name, out_f, in_f, r, alpha, style):
    g = torch.Generator().manual_seed(0)
    down, up = torch.randn(r, in_f, generator=g), torch.randn(out_f, r, generator=g)
    if style == "kohya":
        base = "lora_unet_" + name.replace(".", "_")
        return {f"{base}.lora_down.weight": down, f"{base}.lora_up.weight": up, f"{base}.alpha": torch.tensor(float(alpha))}, down, up
    if style == "comfy":
        return {f"diffusion_model.{name}.lora_down.weight": down, f"diffusion_model.{name}.lora_up.weight": up}, down, up
    return {f"{name}.lora_A.weight": down, f"{name}.lora_B.weight": up}, down, up


@pytest.mark.parametrize("style", ["kohya", "comfy", "peft"])
def test_merge_math_and_all_key_styles(style):
    w = torch.randn(6, 4)
    state = {"blocks.0.self_attn.q.weight": w.clone(), "blocks.0.self_attn.q.bias": torch.zeros(6), "patch_embedding.weight": torch.ones(2, 2)}
    lsd, down, up = _lora("blocks.0.self_attn.q", 6, 4, 2, 4, style)
    rep = M.merge_into(state, lsd, strength=0.5)
    scale = 0.5 * (4 / 2 if style == "kohya" else 1.0)      # alpha/rank only when an alpha is stored
    assert rep["merged"] == 1
    assert torch.allclose(state["blocks.0.self_attn.q.weight"], w + scale * (up @ down), atol=1e-5)
    assert torch.equal(state["patch_embedding.weight"], torch.ones(2, 2))          # untouched
    assert torch.equal(state["blocks.0.self_attn.q.bias"], torch.zeros(6))


def test_dtype_preserved_and_strength_zero_is_noop():
    w = torch.randn(6, 4).to(torch.bfloat16)
    state = {"blocks.1.ffn.0.weight": w.clone()}
    lsd, _, _ = _lora("blocks.1.ffn.0", 6, 4, 2, 2, "kohya")
    M.merge_into(state, lsd, strength=0.0)
    assert state["blocks.1.ffn.0.weight"].dtype == torch.bfloat16 and torch.equal(state["blocks.1.ffn.0.weight"], w)


def test_refuses_wrong_base_model_and_bad_shape():
    state = {"blocks.0.self_attn.q.weight": torch.randn(6, 4)}
    other, _, _ = _lora("blocks.9.cross_attn.v", 6, 4, 2, 2, "kohya")
    with pytest.raises(ValueError, match="match nothing"):
        M.merge_into(state, other)
    bad, _, _ = _lora("blocks.0.self_attn.q", 7, 4, 2, 2, "kohya")
    with pytest.raises(ValueError, match="does not fit"):
        M.merge_into(state, bad)
    with pytest.raises(ValueError):
        M.group_lora({"foo": torch.zeros(1)})


def test_ffn_names_with_digits_do_not_collide():
    state = {"blocks.0.ffn.0.weight": torch.zeros(3, 3), "blocks.0.ffn.2.weight": torch.zeros(3, 3)}
    lsd, d, u = _lora("blocks.0.ffn.2", 3, 3, 1, 1, "comfy")
    M.merge_into(state, lsd)
    assert torch.count_nonzero(state["blocks.0.ffn.0.weight"]) == 0 and torch.count_nonzero(state["blocks.0.ffn.2.weight"]) > 0


def test_patch_animate_source_inserts_once_before_relight_block():
    src = ("class X:\n    def f(self):\n        model = 1\n" + M.ANCHOR + "\n            pass\n")
    out = M.patch_animate_source(src)
    assert out.count("SYOK_LORA") >= 2 and out.index("SYOK_LORA") < out.index("Loading Relighting Lora")
    assert M.patch_animate_source(out) == out                      # idempotent
    with pytest.raises(RuntimeError, match="changed"):
        M.patch_animate_source("def f():\n    pass\n")
