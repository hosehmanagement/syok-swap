import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import swapkit as sk
import texture_service as svc
import texture_transfer as tt
from test_swapkit import make_video


SKIN = (140, 160, 205)
BOX = (40, 40, 80, 80)
FAKE = lambda f: (BOX, 1.0)


def skin_photo(seed=3, size=300, sd=6.0):
    """A plain skin-coloured photo with fine grain: what a real, evenly lit reference looks like to the engine."""
    rng = np.random.default_rng(seed)
    img = np.empty((size, size, 3), np.float32); img[:] = SKIN
    return np.clip(img + rng.normal(0, sd, (size, size, 1)), 0, 255).astype(np.uint8)


def photo_bytes(seed=3):
    img = skin_photo(seed)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def video_bytes(tmp_path, n=12, w=160, h=160):
    return make_video(tmp_path / "v.mp4", SKIN, n=n, fps=24, w=w, h=h, audio=True).read_bytes()


def ready_job(root, tmp_path, **kw):
    job = svc.new_job(root, video_bytes(tmp_path, **kw), photo_bytes(), ".png", "yes")
    svc.run_prepare(root, job, detector=FAKE)
    return job


def test_new_job_refuses_without_consent_bad_photo_type_and_empty_files(tmp_path):
    v = video_bytes(tmp_path)
    for args in [(v, photo_bytes(), ".png", ""), (v, photo_bytes(), ".png", "no"), (v, photo_bytes(), ".gif", "yes"),
                 (b"", photo_bytes(), ".png", "yes"), (v, b"", ".png", "yes")]:
        with pytest.raises(svc.JobError):
            svc.new_job(tmp_path / "jobs", *args)
    assert not (tmp_path / "jobs").exists() or not any((tmp_path / "jobs").iterdir())


def test_new_job_refuses_unreadable_video_and_cleans_up(tmp_path):
    root = tmp_path / "jobs"; root.mkdir()
    with pytest.raises(svc.JobError):
        svc.new_job(root, b"not a video at all", photo_bytes(), ".png", "yes")
    assert list(root.iterdir()) == []


def test_new_job_refuses_oversize(tmp_path, monkeypatch):
    monkeypatch.setattr(svc, "MAX_VIDEO_BYTES", 10)
    with pytest.raises(svc.JobError):
        svc.new_job(tmp_path, b"x" * 11, photo_bytes(), ".png", "yes")


def test_prepare_then_status_ready(tmp_path):
    root = tmp_path / "jobs"; root.mkdir()
    job = ready_job(root, tmp_path)
    st = svc.status(root, job)
    assert st["prepare"]["state"] == "ready" and st["prepare"]["frames"] == 12 and st["prepare"]["width"] == 160
    assert st["render"]["state"] == "idle"
    assert st["expires_at"] - time.time() > 2.9 * 86400


def test_prepare_error_is_recorded_not_raised(tmp_path):
    root = tmp_path / "jobs"; root.mkdir()
    job = svc.new_job(root, video_bytes(tmp_path), photo_bytes(), ".png", "yes")
    svc.run_prepare(root, job, detector=lambda f: (None, 0.0))
    st = svc.status(root, job)["prepare"]
    assert st["state"] == "error" and "face" in st["message"].lower()


def test_preview_matches_direct_engine_and_strength_changes_detail(tmp_path):
    root = tmp_path / "jobs"; root.mkdir()
    job = ready_job(root, tmp_path)
    def decode(b):
        return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    zero, low, high = (decode(svc.preview_jpeg(root, job, 3, s)) for s in (0, 1.0, 3.0))
    src = next(f for i, f in enumerate(sk.frames(root / job / "src.mp4")) if i == 3)
    # strength 0 is the untouched clip frame (only JPEG rounding differs)
    assert np.abs(zero.astype(int) - src.astype(int)).mean() < 2.0
    def detail(img):
        g = cv2.cvtColor(img[40:120, 40:120], cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float((g - cv2.GaussianBlur(g, (0, 0), 1.5)).std())
    assert detail(zero) < detail(low) < detail(high)


def test_preview_rejects_bad_input_and_clamps_strength(tmp_path):
    root = tmp_path / "jobs"; root.mkdir()
    job = ready_job(root, tmp_path)
    for frame, strength in [(99, 1), (-1, 1), ("x", 1), (0, "abc"), (0, float("nan"))]:
        with pytest.raises(svc.JobError):
            svc.preview_jpeg(root, job, frame, strength)
    assert svc.preview_jpeg(root, job, 0, 99) == svc.preview_jpeg(root, job, 0, tt.MAX_STRENGTH)  # clamped


def test_preview_before_ready_is_409(tmp_path):
    root = tmp_path / "jobs"; root.mkdir()
    job = svc.new_job(root, video_bytes(tmp_path), photo_bytes(), ".png", "yes")
    with pytest.raises(svc.JobError) as e:
        svc.preview_jpeg(root, job, 0, 2.0)
    assert e.value.status == 409


def test_job_ids_are_validated_no_path_tricks(tmp_path):
    for bad in ["../etc", "..", "", "ABCDEF0123456789", "a" * 15, "a" * 17, "zzzzzzzzzzzzzzzz"]:
        with pytest.raises(svc.JobError) as e:
            svc.status(tmp_path, bad)
        assert e.value.status == 404


def test_render_keeps_frames_audio_and_matches_preview(tmp_path):
    root = tmp_path / "jobs"; root.mkdir()
    job = ready_job(root, tmp_path)
    assert svc.start_render(root, job, 2.0) == 2.0
    svc.run_render(root, job)
    st = svc.status(root, job)["render"]
    assert st["state"] == "done" and st["frames"] == 12 and st["applied"] == 12 and st["strength"] == 2.0
    out = svc.result_path(root, job)
    assert sk.probe(out)["has_audio"] and len(list(sk.frames(out))) == 12
    # the preview frame at the same strength is close to the same frame of the full render
    prev = cv2.imdecode(np.frombuffer(svc.preview_jpeg(root, job, 5, 2.0), np.uint8), cv2.IMREAD_COLOR)
    full = next(f for i, f in enumerate(sk.frames(out)) if i == 5)
    other = cv2.imdecode(np.frombuffer(svc.preview_jpeg(root, job, 5, 0.0), np.uint8), cv2.IMREAD_COLOR)
    assert np.abs(prev.astype(int) - full.astype(int)).mean() < np.abs(other.astype(int) - full.astype(int)).mean()


def test_render_twice_refused_while_running_and_result_needs_done(tmp_path):
    root = tmp_path / "jobs"; root.mkdir()
    job = ready_job(root, tmp_path)
    with pytest.raises(svc.JobError) as e:
        svc.result_path(root, job)
    assert e.value.status == 409
    svc.start_render(root, job, 1.0)
    with pytest.raises(svc.JobError) as e:
        svc.start_render(root, job, 1.0)
    assert e.value.status == 409


def test_re_render_with_other_strength_replaces_result(tmp_path):
    root = tmp_path / "jobs"; root.mkdir()
    job = ready_job(root, tmp_path)
    svc.start_render(root, job, 1.0); svc.run_render(root, job)
    first = svc.result_path(root, job).read_bytes()
    svc.start_render(root, job, 3.0); svc.run_render(root, job)
    assert svc.status(root, job)["render"]["strength"] == 3.0
    assert svc.result_path(root, job).read_bytes() != first


def test_sweep_removes_only_jobs_older_than_three_days(tmp_path):
    root = tmp_path / "jobs"; root.mkdir()
    old = ready_job(root, tmp_path); new = ready_job(root, tmp_path)
    meta = json.loads((root / old / "meta.json").read_text()); meta["created"] = time.time() - 3.1 * 86400
    (root / old / "meta.json").write_text(json.dumps(meta))
    (root / "not-a-job").mkdir()
    assert svc.sweep(root) == 1
    assert not (root / old).exists() and (root / new).exists() and (root / "not-a-job").exists()


def test_sweep_on_missing_root_is_zero(tmp_path):
    assert svc.sweep(tmp_path / "nope") == 0
