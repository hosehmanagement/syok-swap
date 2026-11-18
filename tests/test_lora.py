import json
import numpy as np
import cv2
import pytest

import consent
from lora import commands as C
from lora import dataset as D


def _face_like(path, seed, size=900):
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), (150, 170, 200), np.uint8)
    img = np.clip(img + rng.normal(0, 25, img.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def test_digest_order_independent_and_changes(tmp_path):
    ps = []
    for i in range(3):
        p = tmp_path / f"{i}.png"; _face_like(p, i); ps.append(p)
    assert D.dataset_digest(ps) == D.dataset_digest(ps[::-1])
    d1 = D.dataset_digest(ps); _face_like(ps[1], 77)
    assert D.dataset_digest(ps) != d1


def test_real_record_needs_every_attestation_and_name(tmp_path):
    p = tmp_path / "a.png"; _face_like(p, 1)
    with pytest.raises(PermissionError):
        D.make_dataset_record([p], "real", attest={"is_adult": True}, who="A", contact="a@x")
    with pytest.raises(PermissionError):
        D.make_dataset_record([p], "real", attest={k: True for k in consent.REQUIRED_FOR_REAL})
    r = D.make_dataset_record([p], "real", attest={k: True for k in consent.REQUIRED_FOR_REAL}, who="A", contact="a@x")
    assert D.verify_dataset_record(r, [p])


def test_record_breaks_when_set_changes_or_expires(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"; _face_like(a, 1); _face_like(b, 2)
    r = D.make_dataset_record([a], "fictional", now=1000)
    with pytest.raises(PermissionError):
        D.verify_dataset_record(r, [a, b], now=1001)     # photo added after sign-off
    with pytest.raises(PermissionError):
        D.verify_dataset_record(r, [a], now=1000 + 91 * 86400)


def test_screen_rejects_small_blurry_unreadable(tmp_path):
    small = tmp_path / "small.png"; cv2.imwrite(str(small), np.zeros((200, 200, 3), np.uint8))
    blur = tmp_path / "blur.png"; cv2.imwrite(str(blur), np.full((900, 900, 3), 128, np.uint8))
    bad = tmp_path / "bad.png"; bad.write_bytes(b"not an image")
    kept, rej = D.screen([small, blur, bad], face_fn=lambda im: 0.5)
    reasons = {p.name: r for p, r in rej}
    assert kept == [] and "too small" in reasons["small.png"] and "unreadable" == reasons["bad.png"] and "blurry" in reasons["blur.png"]


def test_prepare_refuses_without_valid_consent_and_with_too_few(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    for i in range(3):
        _face_like(src / f"{i}.png", i)
    ps = D.list_images(src)
    rec = D.make_dataset_record(ps, "fictional")
    cp = tmp_path / "c.json"; consent.save(rec, cp)
    with pytest.raises(ValueError, match="usable photos"):
        D.prepare(src, tmp_path / "out", cp, face_fn=lambda im: 0.5)
    _face_like(src / "extra.png", 50)                    # set changed after sign-off
    with pytest.raises(PermissionError):
        D.prepare(src, tmp_path / "out", cp)


def test_toml_and_default_caption_have_trigger_and_detail():
    t = D.toml_text("/i", "/c")
    assert 'image_directory = "/i"' in t and "enable_bucket = true" in t and "num_repeats = 1" in t
    cap = D.DEFAULT_CAPTION.format(t="sksperson")
    assert "sksperson" in cap and "pores" in cap and "stubble" in cap


def test_train_cmd_low_stage_and_caps():
    p = C.Plan("me", "/d.toml", "/m", "/o")
    cmd = C.train_cmd(p)
    s = " ".join(cmd)
    assert "t2v-A14B" in s and "wan2.2_t2v_low_noise_14B_fp16.safetensors" in s
    assert cmd[cmd.index("--min_timestep") + 1] == "0" and cmd[cmd.index("--max_timestep") + 1] == "875"
    assert cmd[cmd.index("--output_name") + 1] == "me_low"
    for f in C.FLAGS_TRAIN:
        assert f in cmd, f
    with pytest.raises(ValueError):
        C.train_cmd(C.Plan("me", "/d.toml", "/m", "/o", steps=50000))
    with pytest.raises(ValueError):
        C.train_cmd(C.Plan("me", "/d.toml", "/m", "/o", stage="mid"))


def test_high_stage_range_and_flags_missing_and_cost():
    cmd = C.train_cmd(C.Plan("me", "/d.toml", "/m", "/o", stage="high"))
    assert cmd[cmd.index("--min_timestep") + 1] == "875" and cmd[cmd.index("--max_timestep") + 1] == "1000"
    assert C.flags_missing("--task --dit", ["--task", "--vae"]) == ["--vae"]
    assert C.cost_ceiling(3600) == C.H100_PER_HOUR


def test_prepare_happy_path_writes_images_captions_toml(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    rng = np.random.default_rng(0)
    for i in range(13):                                   # sharp, distinct images (random blocks differ by hash)
        blocks = (rng.random((8, 8)) * 200 + 20).astype(np.uint8)           # distinct coarse layout per image
        img = cv2.resize(blocks, (900, 900), interpolation=cv2.INTER_NEAREST)[..., None].repeat(3, 2)
        img = np.clip(img + rng.normal(0, 30, img.shape), 0, 255).astype(np.uint8)  # plus sharp fine noise
        cv2.imwrite(str(src / f"p{i}.png"), img)
    ps = D.list_images(src)
    cp = tmp_path / "c.json"; consent.save(D.make_dataset_record(ps, "fictional"), cp)
    m = D.prepare(src, tmp_path / "out", cp, trigger="zq", face_fn=lambda im: 0.5)
    assert len(m["kept"]) >= D.MIN_KEPT
    out = tmp_path / "out"
    assert (out / "dataset.toml").exists() and json.loads((out / "manifest.json").read_text())["trigger"] == "zq"
    txt = next((out / "images").glob("*.txt")).read_text()
    assert txt.startswith("zq") and "pores" in txt
