"""The one-flow Syok Swap: upload -> mask check -> paid swap -> texture slider -> saved clip. The GPU steps are replaced by fakes
that write the same files and stage states the real Modal functions write, so every rule around them is tested here."""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import swap_service as ss
import texture_service as svc
from test_swapkit import make_video
from test_texture_service import BOX, FAKE, SKIN, photo_bytes

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient
import swap_api


def clip(tmp_path, n=40, w=160, h=160, fps=24):
    return make_video(tmp_path / "orig.mp4", SKIN, n=n, fps=fps, w=w, h=h, audio=True).read_bytes()


class Rig:
    """Wires the API to fake GPU steps. `calls` records how often each paid or GPU step was started."""

    def __init__(self, tmp_path, weights="ready", mask_warnings=None, mask_runs=True):
        self.root = tmp_path / "jobs"; self.root.mkdir()
        self.tmp = tmp_path
        self.calls = {"mask": 0, "swap": 0, "render": 0, "weights": 0, "prepare": 0}
        self.weights = weights
        self.mask_warnings = mask_warnings or []
        self.mask_runs = mask_runs
        self.events = []
        app = swap_api.build_app(self.root, self.mask, self.swap, self.render, self.fetch_weights, lambda: self.weights,
                                 before_read=lambda: self.events.append("r"), after_write=lambda: self.events.append("w"),
                                 spawn_prepare=self.prepare, detector=FAKE)
        self.c = TestClient(app)

    def mask(self, job):
        self.calls["mask"] += 1
        if not self.mask_runs:
            return
        d = svc.job_dir(self.root, job)
        make_video(d / "preview.mp4", (0, 0, 120), n=4, fps=24, w=32, h=32)
        ss.mask_ready(self.root, job, [0.0, 0.5], {"min": 0.2, "mean": 0.3, "max": 0.4, "frames": 12, "warnings": self.mask_warnings})

    def swap(self, job):                                  # what the H100 step leaves behind, then the CPU face tracking
        self.calls["swap"] += 1
        d = svc.job_dir(self.root, job)
        final = make_video(d / "fake_final.mp4", SKIN, n=40, fps=24, w=160, h=160, audio=True)
        ss.finish_swap(self.root, job, final, True)
        svc.run_prepare(self.root, job, detector=FAKE)

    def prepare(self, job):                               # the CPU slider preparation, run again on request
        self.calls["prepare"] += 1
        svc.run_prepare(self.root, job, detector=FAKE)

    def render(self, job):
        self.calls["render"] += 1
        svc.run_render(self.root, job)

    def fetch_weights(self):
        self.calls["weights"] += 1

    def upload(self, data=None, files=None, **extra):
        files = files or {"video": ("v.mp4", clip(self.tmp), "video/mp4"), "photo": ("p.png", photo_bytes(), "image/png")}
        d = {"consent": "yes", **extra} if data is None else data
        return self.c.post("/swap", files=files, data=d)


def test_whole_flow_through_the_http_layer(tmp_path):
    r = Rig(tmp_path)
    h = r.c.get("/health").json()
    assert h["ok"] and h["app"] == "syok-swap" and h["version"] == 5 and h["weights"] == "ready" and h["facelock"] is True
    assert h["strength"] == {"min": 0.0, "max": 3.0, "start": 2.0} and h["limits"]["keep_days"] == 3
    up = r.upload(resolution="480p", margin="normal")
    assert up.status_code == 200
    job = up.json()["job"]
    assert up.json()["estimate"]["approx"] is True
    st = r.c.get(f"/status/{job}").json()
    assert st["mask"]["state"] == "ready" and st["swap"]["state"] == "idle" and st["prepare"]["state"] == "idle"
    mv = r.c.get(f"/mask/{job}")
    assert mv.status_code == 200 and mv.headers["content-type"] == "video/mp4" and len(mv.content) > 500
    d = r.root / job
    assert (d / "raw.mp4").exists() and not (d / "src.mp4").exists()          # the upload is the ORIGINAL clip
    assert r.c.get(f"/preview/{job}", params={"frame": 1}).status_code == 409      # no slider before the swap exists
    g = r.c.post(f"/generate/{job}")
    assert g.status_code == 200 and g.json() == {"state": "queued", "relight": True}
    st = r.c.get(f"/status/{job}").json()
    assert st["swap"]["state"] == "done" and st["prepare"]["state"] == "ready" and st["prepare"]["frames"] == 40
    assert (d / "src.mp4").exists() and not (d / "final.mp4").exists()
    p = r.c.get(f"/preview/{job}", params={"frame": 2, "strength": 2.0})
    assert p.status_code == 200 and p.headers["content-type"] == "image/jpeg" and p.content[:2] == b"\xff\xd8"
    assert r.c.get(f"/result/{job}").status_code == 409
    assert r.c.post(f"/render/{job}", params={"strength": 1.5}).json() == {"state": "queued", "strength": 1.5}
    res = r.c.get(f"/result/{job}")
    assert res.status_code == 200 and res.headers["content-type"] == "video/mp4" and len(res.content) > 1000
    assert r.c.get(f"/swapped/{job}").status_code == 200                              # the plain swap stays reachable
    assert r.calls == {"mask": 1, "swap": 1, "render": 1, "weights": 0, "prepare": 0}
    assert r.c.delete(f"/job/{job}").json() == {"deleted": True}
    assert r.c.get(f"/status/{job}").status_code == 404


def test_the_paid_step_cannot_start_twice_or_without_a_good_mask(tmp_path):
    r = Rig(tmp_path)
    job = r.upload().json()["job"]
    assert r.c.post(f"/generate/{job}").status_code == 200
    again = r.c.post(f"/generate/{job}")                                                # double click / second tab
    assert again.status_code == 409 and "already swapped" in again.json()["error"]
    assert r.calls["swap"] == 1

    r2 = Rig(tmp_path / "b", mask_runs=False) if (tmp_path / "b").mkdir() is None else None
    j2 = r2.upload().json()["job"]
    early = r2.c.post(f"/generate/{j2}")                                                # mask not made yet
    assert early.status_code == 409 and "mask" in early.json()["error"].lower()
    assert r2.c.get(f"/mask/{j2}").status_code == 409
    assert r2.calls["swap"] == 0

    (tmp_path / "c").mkdir()
    r3 = Rig(tmp_path / "c", mask_warnings=["Mask is almost empty: no person was found. Do not generate."])
    j3 = r3.upload().json()["job"]
    empty = r3.c.post(f"/generate/{j3}")
    assert empty.status_code == 400 and "Refusing" in empty.json()["error"]
    assert r3.calls["swap"] == 0


def test_a_running_swap_is_refused_and_a_failed_one_can_be_retried(tmp_path):
    r = Rig(tmp_path)
    job = r.upload().json()["job"]
    d = svc.job_dir(r.root, job)
    svc.set_stage(d, "swap", state="generating")
    assert r.c.post(f"/generate/{job}").status_code == 409
    ss.fail(r.root, job, "swap", "The swap failed. Nothing was kept.")
    st = r.c.get(f"/status/{job}").json()
    assert st["swap"]["state"] == "error" and st["swap"]["message"] == "The swap failed. Nothing was kept."
    assert r.c.post(f"/generate/{job}", params={"relight": "false"}).json() == {"state": "queued", "relight": False}


def test_uploads_are_checked_before_anything_is_stored(tmp_path):
    r = Rig(tmp_path)
    files = {"video": ("v.mp4", clip(tmp_path), "video/mp4"), "photo": ("p.png", photo_bytes(), "image/png")}
    cases = [({"consent": ""}, 400, "agreed"), ({"consent": "yes", "resolution": "4k"}, 400, "480p or 720p"),
             ({"consent": "yes", "margin": "huge"}, 400, "tight, normal or loose"), ({"consent": "yes", "start": "-3"}, 400, "seconds"),
             ({"consent": "yes", "length": "abc"}, 400, "seconds"), ({"consent": "yes", "start": "50"}, 400, "close to the end")]
    for data, code, word in cases:
        res = r.c.post("/swap", files=files, data=data)
        assert res.status_code == code and word in res.json()["error"], (data, res.json())
    assert not any(r.root.iterdir())                                                    # no folder left behind
    assert r.calls["mask"] == 0


def test_section_start_and_length_are_kept_and_the_estimate_follows_them(tmp_path):
    r = Rig(tmp_path)
    long = make_video(tmp_path / "long.mp4", SKIN, n=96, fps=24, w=160, h=160, audio=True).read_bytes()     # 4 s
    job = r.upload(files={"video": ("v.mp4", long, "video/mp4"), "photo": ("p.png", photo_bytes(), "image/png")},
                   start="1", length="2", resolution="720p").json()["job"]
    st = r.c.get(f"/status/{job}").json()
    assert st["request"] == {"resolution": "720p", "margin": "normal", "start": 1.0, "length": 2.0}
    assert st["estimate"] == ss.estimate("720p", 2.0)


def test_estimate_grows_with_length_and_resolution_and_is_labelled_approximate():
    a, b, c = ss.estimate("480p", 5), ss.estimate("480p", 10), ss.estimate("720p", 10)
    assert a["approx"] and a["swap_usd"] < b["swap_usd"] < c["swap_usd"]
    assert a["swap_minutes"] < b["swap_minutes"] < c["swap_minutes"]
    assert a["total_usd"] == round(a["swap_usd"] + a["mask_usd"], 2)
    assert ss.estimate("720p", 26)["swap_usd"] > 5                                       # a 26 s 720p clip is NOT cheap: the card must show that


def test_one_time_setup_is_started_once_and_blocks_swaps_until_ready(tmp_path):
    r = Rig(tmp_path, weights="missing")
    blocked = r.upload()
    assert blocked.status_code == 409 and "setup" in blocked.json()["error"].lower()
    assert not any(r.root.iterdir())
    assert r.c.get("/health").json()["weights"] == "missing"
    assert r.c.post("/setup").json() == {"weights": "downloading"} and r.calls["weights"] == 1
    r.weights = "downloading"
    assert r.c.post("/setup").json() == {"weights": "downloading"} and r.calls["weights"] == 1   # never a second 50 GB download
    r.weights = "ready"
    assert r.c.post("/setup").json() == {"weights": "ready"} and r.calls["weights"] == 1
    assert r.upload().status_code == 200


def test_swapped_clip_is_available_even_when_the_slider_could_not_be_prepared(tmp_path):
    r = Rig(tmp_path)
    job = r.upload().json()["job"]
    assert r.c.get(f"/swapped/{job}").status_code == 409
    r.swap = None
    d = svc.job_dir(r.root, job)
    final = make_video(d / "f.mp4", SKIN, n=40, fps=24, w=160, h=160, audio=True)
    ss.finish_swap(r.root, job, final, True)
    ss.fail(r.root, job, "prepare", "Not enough plain skin in the reference photo.")
    st = r.c.get(f"/status/{job}").json()
    assert st["swap"]["state"] == "done" and st["prepare"]["state"] == "error"
    assert r.c.get(f"/swapped/{job}").status_code == 200
    assert r.c.post(f"/render/{job}").status_code == 409


def test_read_and_write_hooks_bracket_every_request(tmp_path):
    r = Rig(tmp_path)
    job = r.upload().json()["job"]
    r.c.get(f"/status/{job}"); r.c.post(f"/generate/{job}")
    assert r.events.count("w") == 2 and r.events.count("r") >= 3


def test_old_jobs_are_swept_with_their_files(tmp_path):
    r = Rig(tmp_path)
    job = r.upload().json()["job"]
    meta = json.loads((r.root / job / "meta.json").read_text()); meta["created"] = time.time() - 3.1 * 86400
    (r.root / job / "meta.json").write_text(json.dumps(meta))
    assert svc.sweep(r.root) == 1 and not (r.root / job).exists()


# ---- steps that die without saying so ----

def test_a_step_that_never_finishes_is_reported_as_failed_not_spinning_for_ever(tmp_path):
    r = Rig(tmp_path)
    job = r.upload().json()["job"]
    d = svc.job_dir(r.root, job)
    svc.set_stage(d, "swap", state="generating")
    fresh = ss.status(r.root, job)
    assert fresh["swap"]["state"] == "generating"                                        # still inside its time limit: left alone
    later = ss.status(r.root, job, now=time.time() + ss.STUCK_AFTER_SECONDS["swap"] + 5)
    assert later["swap"]["state"] == "error" and "stopped without finishing" in later["swap"]["message"]
    assert later["mask"]["state"] == "ready"                                             # finished stages are never touched
    assert r.c.post(f"/generate/{job}").status_code == 200                               # and the user can try again
    for stage, active in ss.ACTIVE.items():                                              # every active stage has a limit and a message
        assert stage in ss.STUCK_AFTER_SECONDS and stage in ss.STUCK_MESSAGE and active


def test_a_queued_step_with_no_timestamp_is_judged_from_the_job_creation_time(tmp_path):
    r = Rig(tmp_path, mask_runs=False)
    job = r.upload().json()["job"]
    assert ss.status(r.root, job)["mask"]["state"] == "queued"
    assert ss.status(r.root, job, now=time.time() + ss.STUCK_AFTER_SECONDS["mask"] + 5)["mask"]["state"] == "error"


# ---- model files ----

def test_weights_state_downloading_ready_missing_and_legacy_volume(tmp_path):
    w, flags = tmp_path / "model", tmp_path / "flags"
    w.mkdir(); flags.mkdir()
    now = time.time()
    assert ss.weights_state(w, flags, min_bytes=100, now=now) == "missing"
    (w / "a.bin").write_bytes(b"x" * 50)
    assert ss.weights_state(w, flags, min_bytes=100, now=now) == "missing"               # a half-finished folder is NOT ready
    (flags / ".weights-downloading").write_text(str(now))
    assert ss.weights_state(w, flags, min_bytes=100, now=now + 60) == "downloading"
    assert ss.weights_state(w, flags, min_bytes=100, now=now + 4 * 3600) == "missing"     # a download that died long ago is not "downloading"
    (flags / ".weights-ok").write_text("ok")
    assert ss.weights_state(w, flags, min_bytes=100, now=now + 4 * 3600) == "ready"
    (flags / ".weights-downloading").unlink(); (flags / ".weights-ok").unlink()
    (w / "b.bin").write_bytes(b"x" * 60)
    assert ss.weights_state(w, flags, min_bytes=100, now=now) == "ready"                 # filled earlier by `modal run app.py::fetch`


# ---- the Modal app file ----

APP = (Path(__file__).resolve().parent.parent / "app.py").read_text()


def test_modal_app_file_is_one_locked_app_with_cheap_cpu_steps_and_one_paid_step():
    compile(APP, "app.py", "exec")
    assert APP.count("requires_proxy_auth=True") == 1 and 'modal.App("syok-swap")' in APP
    assert APP.count('gpu="H100"') == 2                                                  # the old CLI render and the one-flow swap, nothing else
    assert "max_containers=1" in APP                                                     # one paid H100 run at a time
    for name in ("prepare_job", "texture_render_job", "sweep_old_jobs", "web"):
        block = APP[APP.index(f"def {name}("):]
        head = APP[:APP.index(f"def {name}(")].rsplit("@app.function", 1)[1]
        assert "image=light" in head and "gpu" not in head, name                         # CPU image, no GPU
    assert "Period(hours=6)" in APP
    assert "prepare_job.spawn(job)" in APP                                               # the slider's tracking starts by itself after the swap


# ---------- the realism slider is checked BEFORE anything is paid for, and can be prepared again afterwards ----------
def png(img):
    import cv2
    return cv2.imencode(".png", img)[1].tobytes()


def no_skin_photo(size=300):
    import numpy as np
    rng = np.random.default_rng(5)
    return np.clip(np.full((size, size, 3), (200, 90, 40), np.float32) + rng.normal(0, 6, (size, size, 1)), 0, 255).astype(np.uint8)


def test_a_photo_that_cannot_feed_the_slider_is_refused_before_any_gpu_step(tmp_path):
    r = Rig(tmp_path)
    bad = r.upload(files={"video": ("v.mp4", clip(tmp_path), "video/mp4"), "photo": ("p.png", png(no_skin_photo()), "image/png")})
    assert bad.status_code == 422
    msg = bad.json()["error"]
    assert "realism slider" in msg and "nothing was charged" in msg and "skin close-up" in msg
    assert r.calls["mask"] == 0 and list(r.root.iterdir()) == []            # nothing started, nothing kept


def test_a_skin_close_up_rescues_that_photo_and_is_used_only_for_the_grain(tmp_path):
    r = Rig(tmp_path)
    files = {"video": ("v.mp4", clip(tmp_path), "video/mp4"), "photo": ("p.png", png(no_skin_photo()), "image/png"),
             "skin": ("s.png", photo_bytes(), "image/png")}
    up = r.upload(files=files)
    assert up.status_code == 200 and r.calls["mask"] == 1
    job = up.json()["job"]
    d = r.root / job
    assert (d / "skin.png").exists() and (d / "ref.png").read_bytes() == png(no_skin_photo())   # the face photo is untouched
    r.c.post(f"/generate/{job}")
    assert r.c.get(f"/status/{job}").json()["prepare"]["state"] == "ready"
    # a bad skin photo is refused the same way, and so is a wrong file type
    files["skin"] = ("s.png", png(no_skin_photo()), "image/png")
    assert r.upload(files=files).status_code == 422
    files["skin"] = ("s.gif", photo_bytes(), "image/gif")
    assert r.upload(files=files).status_code == 400


def test_the_slider_can_be_prepared_again_on_a_finished_swap_without_a_new_swap(tmp_path):
    r = Rig(tmp_path)
    job = r.upload().json()["job"]
    assert r.c.post(f"/retexture/{job}", data={"retry": "1"}).status_code == 409          # no swap yet
    r.c.post(f"/generate/{job}")
    d = r.root / job
    svc.set_stage(d, "prepare", state="error", message="Not enough plain skin in the reference photo")   # as the real case did
    for name in ("layer.npy", "tracks.json", "prepared.json"):
        (d / name).unlink()
    again = r.c.post(f"/retexture/{job}", files={"skin": ("s.png", photo_bytes(seed=9), "image/png")}, data={"retry": "1"})
    assert again.status_code == 200 and again.json() == {"job": job, "prepare": {"state": "queued"}}
    st = r.c.get(f"/status/{job}").json()
    assert st["prepare"]["state"] == "ready" and not st["prepare"].get("message") and st["render"]["state"] == "idle"
    assert (d / "skin.png").exists() and r.calls == {"mask": 1, "swap": 1, "render": 0, "weights": 0, "prepare": 1}   # no second paid swap
    assert r.c.get(f"/preview/{job}", params={"frame": 1, "strength": 2}).status_code == 200
    # while it is being prepared a second request is refused
    svc.set_stage(d, "prepare", state="preparing")
    assert r.c.post(f"/retexture/{job}", data={"retry": "1"}).status_code == 409
    assert r.c.post("/retexture/0000000000000000", data={"retry": "1"}).status_code == 404
