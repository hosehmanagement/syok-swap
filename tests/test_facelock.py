"""Face-lock: the optional Magic Hour pass after the swap. Magic Hour is replaced by a fake transport (no network, no credits);
the video plumbing (fit, scale back, soundtrack) runs on real ffmpeg with tiny generated clips like the other tests."""
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import facelock as fl
import swap_service as ss
import swapkit as sk
import texture_service as svc
from test_swapkit import make_video
from test_texture_service import FAKE, SKIN, photo_bytes

pytest.importorskip("fastapi")
from test_swap_service import Rig, clip

KEY = "mhk_live_TOPSECRET0123456789"
UP_VIDEO = "https://videos.magichour.ai/api-assets/abc/video.mp4?X-Sig=1"
UP_IMAGE = "https://videos.magichour.ai/api-assets/abc/image.png?X-Sig=2"
DL = "https://videos.magichour.ai/results/abc/out.mp4?sig=3"
LOCKED = (60, 200, 60)                 # the colour of what "Magic Hour" sends back, so the test can see the clip changed


class FakeMagicHour:
    """Answers like api.magichour.ai. `polls` is the list of project states returned one by one (the last one repeats)."""

    def __init__(self, tmp, polls=None, result=None, upload_status=200, create_status=200, create_body=None,
                 upload_items=None, download_url=DL, poll_status=200):
        self.tmp = Path(tmp)
        self.calls = []
        self.polls = list(polls or [{"status": "queued", "credits_charged": 30}, {"status": "rendering", "credits_charged": 30},
                                    {"status": "complete", "credits_charged": 28, "downloads": [{"url": DL}]}])
        self.result = result
        self.upload_status, self.create_status, self.poll_status = upload_status, create_status, poll_status
        self.create_body = create_body if create_body is not None else b'{"id": "proj_1", "credits_charged": 30}'
        self.upload_items = upload_items or [{"upload_url": UP_VIDEO, "file_path": "api-assets/abc/video.mp4"},
                                             {"upload_url": UP_IMAGE, "file_path": "api-assets/abc/image.png"}]
        self.download_url = download_url
        self.uploaded = {}

    def count(self, method, part):
        return sum(1 for c in self.calls if c["method"] == method and part in c["url"])

    def __call__(self, method, url, headers=None, body=None, dest=None, max_bytes=None, timeout=None):
        headers = dict(headers or {})
        self.calls.append({"method": method, "url": url, "headers": headers,
                           "json": json.loads(body) if isinstance(body, bytes) else None})
        if method == "POST" and url == "https://api.magichour.ai/v1/files/upload-urls":
            if self.upload_status != 200:
                return self.upload_status, b'{"message": "Invalid API key ' + KEY.encode() + b'"}'
            return 200, json.dumps({"items": self.upload_items}).encode()
        if method == "PUT":
            assert isinstance(body, Path) and body.exists()
            self.uploaded[url] = body.read_bytes()
            return 200, b""
        if method == "POST" and url == "https://api.magichour.ai/v1/face-swap":
            return self.create_status, self.create_body
        if method == "GET" and url.startswith("https://api.magichour.ai/v1/video-projects/"):
            if self.poll_status != 200:
                return self.poll_status, b""
            state = self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]
            if state.get("status") == "complete" and self.download_url != DL:
                state = {**state, "downloads": [{"url": self.download_url}]}
            return 200, json.dumps(state).encode()
        if method == "GET" and dest is not None:
            data = Path(self.result).read_bytes()
            if max_bytes is not None and len(data) > max_bytes:
                raise fl.TooLarge()
            Path(dest).write_bytes(data)
            return 200, b""
        raise AssertionError(f"unexpected call {method} {url}")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def mh_result(tmp, n=40, w=160, h=160, name="mh.mp4"):
    return make_video(Path(tmp) / name, LOCKED, n=n, fps=24, w=w, h=h, audio=False)


class FaceRig(Rig):
    """The swap API wired like app.py: the swap stops at face-lock "waiting", POST /facelock runs the fake Magic Hour, and the
    slider preparation runs whenever the service says to spawn it."""

    def __init__(self, tmp_path, mh=None, detector=FAKE):
        super().__init__(tmp_path)
        import swap_api
        from fastapi.testclient import TestClient
        self.mh = mh
        self.facelock_calls = []
        self.clock = Clock()
        app = swap_api.build_app(self.root, self.mask, self.swap, self.render, self.fetch_weights, lambda: self.weights,
                                 spawn_prepare=self.prepare, detector=detector, spawn_facelock=self.facelock)
        self.c = TestClient(app)

    def swap(self, job):
        self.calls["swap"] += 1
        d = svc.job_dir(self.root, job)
        final = make_video(d / "fake_final.mp4", SKIN, n=40, fps=24, w=160, h=160, audio=True)
        if ss.finish_swap(self.root, job, final, True):
            self.prepare(job)

    def facelock(self, job, key):
        self.facelock_calls.append(job)
        if self.mh is not None and fl.run(self.root, job, key, http=self.mh, clock=self.clock, sleep=self.clock.sleep,
                                          poll_seconds=5, max_wait=120):
            self.prepare(job)

    def swapped_job(self, **extra):
        job = self.upload(facelock="yes", **extra).json()["job"]
        assert self.c.post(f"/generate/{job}").status_code == 200
        return job

    def lock(self, job, key=KEY, files=None):
        return self.c.post(f"/facelock/{job}", headers={"X-Magic-Hour-Key": key} if key is not None else {}, files=files)


def everything_written(d):
    """All bytes of every file in the job folder."""
    return b"".join(p.read_bytes() for p in Path(d).rglob("*") if p.is_file())


def assert_key_nowhere(r, job, *texts):
    assert KEY.encode() not in everything_written(r.root / job)
    assert KEY not in json.dumps(r.c.get(f"/status/{job}").json())
    for t in texts:
        assert KEY not in str(t)


# ------------------------------------------------------------------ upload: the box, the face photo, the warning

def test_upload_with_facelock_waits_after_the_swap_and_does_not_prepare_the_slider(tmp_path):
    r = FaceRig(tmp_path)
    job = r.swapped_job()
    st = r.c.get(f"/status/{job}").json()
    assert st["request"]["facelock"] is True
    assert st["swap"]["state"] == "done" and st["facelock"]["state"] == "waiting" and st["prepare"]["state"] == "idle"
    assert r.calls["prepare"] == 0 and r.facelock_calls == []          # nothing is sent to Magic Hour without the key
    assert r.c.get(f"/swapped/{job}").status_code == 200


def test_face_close_up_is_stored_and_a_small_face_warns_but_never_blocks(tmp_path):
    r = FaceRig(tmp_path)                                    # FAKE finds an 80 px face
    up = r.upload(facelock="yes", files={"video": ("v.mp4", clip(tmp_path), "video/mp4"),
                                         "photo": ("p.png", photo_bytes(), "image/png"),
                                         "face": ("f.jpg", photo_bytes(seed=4), "image/jpeg")})
    assert up.status_code == 200 and r.calls["mask"] == 1
    job = up.json()["job"]
    meta = json.loads((r.root / job / "meta.json").read_text())
    assert meta["face"] == "face.jpg" and (r.root / job / "face.jpg").read_bytes() == photo_bytes(seed=4)
    w = r.c.get(f"/status/{job}").json()["facelock"]["warning"]
    assert w == meta["facelock_warning"] and "about 80 px wide" in w and "512 x 512" in w
    bad = r.upload(files={"video": ("v.mp4", clip(tmp_path), "video/mp4"), "photo": ("p.png", photo_bytes(), "image/png"),
                          "face": ("f.gif", photo_bytes(), "image/gif")})
    assert bad.status_code == 400 and "Face photo" in bad.json()["error"]


def test_a_40_px_face_triggers_the_warning_but_the_job_is_created(tmp_path):
    root = tmp_path / "jobs"; root.mkdir()
    tiny = lambda f: ((10, 10, 40, 40), 1.0)
    job = ss.new_swap_job(root, clip(tmp_path), photo_bytes(), ".png", "yes", facelock=True, detector=tiny)
    st = ss.status(root, job)
    assert st["mask"]["state"] == "queued"
    assert st["facelock"]["warning"] == ("The face in your photo is only about 40 px wide, so Magic Hour has little to copy. "
                                         "Add a sharp head-and-shoulders close-up for a closer likeness. Magic Hour asks "
                                         "for a face area of at least 512 x 512 px.")
    big = lambda f: ((0, 0, 300, 300), 1.0)
    j2 = ss.new_swap_job(root, clip(tmp_path), photo_bytes(), ".png", "yes", facelock=True, detector=big)
    assert "warning" not in ss.status(root, j2)["facelock"]
    assert "facelock_warning" not in svc.read_meta(svc.job_dir(root, j2))


def test_a_job_without_facelock_behaves_exactly_as_before(tmp_path):
    r = FaceRig(tmp_path)
    job = r.upload().json()["job"]
    d = r.root / job
    meta = json.loads((d / "meta.json").read_text())
    assert "facelock" not in meta["request"] and "facelock_warning" not in meta and "face" not in meta
    assert meta["facelock"] == {"state": "idle"}
    r.c.post(f"/generate/{job}")
    assert (d / "src.mp4").read_bytes() == (d / "fake_final.mp4").read_bytes()            # byte for byte the swap
    assert not (d / "src_plain.mp4").exists() and r.facelock_calls == []
    st = r.c.get(f"/status/{job}").json()
    assert st["facelock"] == {"state": "idle"} and st["prepare"]["state"] == "ready" and r.calls["prepare"] == 1
    root2 = tmp_path / "j2"; root2.mkdir()
    j2 = ss.new_swap_job(root2, clip(tmp_path), photo_bytes(), ".png", "yes", detector=FAKE)
    final = make_video(tmp_path / "f.mp4", SKIN, n=40, fps=24, w=160, h=160, audio=True)
    assert ss.finish_swap(root2, j2, final, True) is True
    assert ss.status(root2, j2)["prepare"]["state"] == "queued"


# ------------------------------------------------------------------ the happy path

def test_facelock_replaces_the_clip_keeps_the_plain_swap_and_prepares_the_slider_again(tmp_path):
    mh = FakeMagicHour(tmp_path, result=mh_result(tmp_path))
    r = FaceRig(tmp_path, mh)
    job = r.swapped_job()
    d = r.root / job
    plain = (d / "src.mp4").read_bytes()
    res = r.lock(job)
    assert res.status_code == 200 and res.json() == {"facelock": {"state": "queued"}}
    st = r.c.get(f"/status/{job}").json()
    assert st["facelock"]["state"] == "done" and st["facelock"]["credits"] == 28 and not st["facelock"].get("message")
    assert st["facelock"].get("note") is None                                          # same size: nothing to say
    assert st["prepare"]["state"] == "ready" and r.calls["prepare"] == 1
    assert (d / "src_plain.mp4").read_bytes() == plain and (d / "src.mp4").read_bytes() != plain
    f = next(sk.frames(d / "src.mp4"))
    assert abs(int(f[80, 80, 1]) - 200) < 15                                           # the clip IS Magic Hour's result
    info, pinfo = sk.probe(d / "src.mp4"), sk.probe(d / "src_plain.mp4")
    assert (info["width"], info["height"]) == (160, 160) and abs(info["duration"] - pinfo["duration"]) < 0.1
    assert info["has_audio"]                                                           # the swap's soundtrack is back on it
    assert not (d / "facelock_tmp").exists() and not (d / "src.mp4.tmp").exists()
    # exactly what was sent to Magic Hour
    assert mh.count("POST", "/v1/face-swap") == 1 and mh.count("POST", "/upload-urls") == 1
    up = next(c for c in mh.calls if c["url"].endswith("upload-urls"))["json"]
    assert up == {"items": [{"type": "video", "extension": "mp4"}, {"type": "image", "extension": "png"}]}
    body = next(c for c in mh.calls if c["url"].endswith("/v1/face-swap"))["json"]
    assert body["start_seconds"] == 0 and abs(body["end_seconds"] - pinfo["duration"]) < 0.01
    assert body["style"] == {"version": "default"}
    assert body["assets"] == {"face_swap_mode": "all-faces", "image_file_path": "api-assets/abc/image.png",
                              "video_source": "file", "video_file_path": "api-assets/abc/video.mp4"}
    assert mh.uploaded[UP_VIDEO] == plain and mh.uploaded[UP_IMAGE] == (d / "ref.png").read_bytes()
    for c in mh.calls:                                     # the key only goes to the API, never to the storage links
        if c["url"].startswith("https://api.magichour.ai/"):
            assert c["headers"]["Authorization"] == f"Bearer {KEY}"
        else:
            assert "Authorization" not in c["headers"] and KEY not in c["url"]
    assert_key_nowhere(r, job, res.text)
    # a done face-lock cannot run again or be skipped
    assert r.lock(job).status_code == 409
    assert r.c.post(f"/facelock/{job}/skip").status_code == 409


def test_a_face_close_up_sent_with_the_request_is_the_face_magic_hour_gets(tmp_path):
    mh = FakeMagicHour(tmp_path, result=mh_result(tmp_path))
    r = FaceRig(tmp_path, mh)
    job = r.swapped_job()
    res = r.lock(job, files={"face": ("close.webp", photo_bytes(seed=8), "image/webp")})
    assert res.status_code == 200
    d = r.root / job
    assert json.loads((d / "meta.json").read_text())["face"] == "face.webp"
    up = next(c for c in mh.calls if c["url"].endswith("upload-urls"))["json"]
    assert up["items"][1] == {"type": "image", "extension": "webp"}
    assert mh.uploaded[UP_IMAGE] == photo_bytes(seed=8)


def test_a_smaller_result_is_scaled_back_with_a_note_about_the_free_plan(tmp_path):
    mh = FakeMagicHour(tmp_path, result=mh_result(tmp_path, w=96, h=96))
    r = FaceRig(tmp_path, mh)
    job = r.swapped_job()
    assert r.lock(job).status_code == 200
    st = r.c.get(f"/status/{job}").json()["facelock"]
    assert st["state"] == "done" and "96x96" in st["note"] and "576 px" in st["note"] and "160x160" in st["note"]
    info = sk.probe(r.root / job / "src.mp4")
    assert (info["width"], info["height"]) == (160, 160) and info["has_audio"]


def test_fit_keeps_the_aspect_and_holds_the_last_frame_of_a_slightly_short_result(tmp_path):
    plain = make_video(tmp_path / "plain.mp4", SKIN, n=48, fps=24, w=160, h=96, audio=True)        # 2.0 s, 5:3
    short = make_video(tmp_path / "short.mp4", LOCKED, n=40, fps=24, w=80, h=60, audio=False)      # 1.67 s, 4:3
    note = fl.build_locked_clip(short, plain, tmp_path / "out.mp4", tmp_path)
    info = sk.probe(tmp_path / "out.mp4")
    assert (info["width"], info["height"]) == (160, 96) and abs(info["duration"] - 2.0) < 0.1 and info["has_audio"]
    assert "80x60" in note
    f = next(sk.frames(tmp_path / "out.mp4"))
    assert f[48, 4].max() < 30 and abs(int(f[48, 80, 1]) - 200) < 20                    # padded sides, picture in the middle


# ------------------------------------------------------------------ failures: plain messages, clip untouched, slider still works

@pytest.mark.parametrize("kind", ["bad_key", "no_credit", "no_source_face", "canceled", "timeout", "wrong_duration",
                                  "error_echoes_key", "bad_upload_host", "http_download", "too_large", "no_downloads"])
def test_failures_leave_the_swap_untouched_and_the_slider_usable(tmp_path, kind, monkeypatch):
    result = mh_result(tmp_path)
    mh = {
        "bad_key": lambda: FakeMagicHour(tmp_path, result=result, upload_status=401),
        "no_credit": lambda: FakeMagicHour(tmp_path, result=result, create_status=402,
                                           create_body=b'{"code": "insufficient_credits", "message": "Not enough credits"}'),
        "no_source_face": lambda: FakeMagicHour(tmp_path, result=result, polls=[
            {"status": "rendering"}, {"status": "error", "credits_charged": 0,
                                      "error": {"code": "no_source_face", "message": "No face"}}]),
        "canceled": lambda: FakeMagicHour(tmp_path, result=result, polls=[{"status": "canceled"}]),
        "timeout": lambda: FakeMagicHour(tmp_path, result=result, polls=[{"status": "rendering"}]),
        "wrong_duration": lambda: FakeMagicHour(tmp_path, result=mh_result(tmp_path, n=80, name="long.mp4")),
        "error_echoes_key": lambda: FakeMagicHour(tmp_path, result=result, create_status=500,
                                                  create_body=b"internal error for key " + KEY.encode()),
        "bad_upload_host": lambda: FakeMagicHour(tmp_path, result=result, upload_items=[
            {"upload_url": "https://evil.example.com/v.mp4", "file_path": "a"},
            {"upload_url": UP_IMAGE, "file_path": "b"}]),
        "http_download": lambda: FakeMagicHour(tmp_path, result=result, download_url="http://videos.magichour.ai/x.mp4"),
        "too_large": lambda: FakeMagicHour(tmp_path, result=result),
        "no_downloads": lambda: FakeMagicHour(tmp_path, result=result, polls=[{"status": "complete", "downloads": []}]),
    }[kind]()
    if kind == "too_large":
        monkeypatch.setattr(fl, "MAX_RESULT_BYTES", 100)
    r = FaceRig(tmp_path, mh)
    job = r.swapped_job()
    d = r.root / job
    plain = (d / "src.mp4").read_bytes()
    res = r.lock(job)
    assert res.status_code == 200                                    # the start is accepted; the outcome is in status
    st = r.c.get(f"/status/{job}").json()
    msg = st["facelock"]["message"]
    assert st["facelock"]["state"] == "error" and msg
    assert (d / "src.mp4").read_bytes() == plain                     # untouched
    assert st["prepare"]["state"] == "ready" and r.calls["prepare"] == 1   # the slider works on the plain swap
    assert str(d) not in msg and "/" not in msg.replace("Magic Hour → Developer", "")
    assert not (d / "facelock_tmp").exists()
    assert mh.count("POST", "/v1/face-swap") <= 1                    # never retried
    expect = {"bad_key": "didn't accept that key", "no_credit": "out of free credits",
              "no_source_face": "could not find a face in your photo", "canceled": "canceled",
              "timeout": "did not finish within 25 minutes", "wrong_duration": "3.3 s of video but the swapped clip is 1.7 s",
              "error_echoes_key": "Magic Hour error (500)", "bad_upload_host": "not a Magic Hour address",
              "http_download": "not https", "too_large": "larger than 0 MB", "no_downloads": "sent no video"}[kind]
    assert expect in msg, msg
    if kind == "no_source_face":
        assert "refunds" in msg
    if kind == "bad_key":
        assert mh.count("POST", "/v1/face-swap") == 0 and mh.count("PUT", "") == 0
    if kind == "bad_upload_host":
        assert mh.count("PUT", "") == 0                              # nothing sent to a host that is not Magic Hour
    assert_key_nowhere(r, job, res.text, msg)
    # the user can skip now (the slider is already ready, so it is not prepared twice) ...
    sk_ = r.c.post(f"/facelock/{job}/skip")
    assert sk_.status_code == 200 and sk_.json() == {"facelock": {"state": "skipped"}, "prepare": {"state": "ready"}}
    assert r.calls["prepare"] == 1
    # ... or retry after skipping
    r.mh = FakeMagicHour(tmp_path, result=mh_result(tmp_path, name="ok.mp4"))
    monkeypatch.setattr(fl, "MAX_RESULT_BYTES", 200 * 2**20)
    assert r.lock(job).status_code == 200
    st = r.c.get(f"/status/{job}").json()
    assert st["facelock"]["state"] == "done" and st["prepare"]["state"] == "ready"
    assert (d / "src_plain.mp4").read_bytes() == plain               # the plain swap is never overwritten


def test_out_of_credits_spends_one_create_call_only(tmp_path):
    mh = FakeMagicHour(tmp_path, result=mh_result(tmp_path), create_status=402, create_body=b"{}")
    r = FaceRig(tmp_path, mh)
    job = r.swapped_job()
    r.lock(job)
    assert mh.count("POST", "/v1/face-swap") == 1 and mh.count("GET", "video-projects") == 0


def test_polling_rides_out_a_busy_magic_hour_but_stops_on_a_refused_key(tmp_path):
    clock = Clock()
    mh = FakeMagicHour(tmp_path, result=mh_result(tmp_path), poll_status=429)
    with pytest.raises(fl.FaceLockError) as e:
        fl.wait_for_project(mh, KEY, "p", clock() + 60, clock, clock.sleep, 5)
    assert "did not finish" in str(e.value) and mh.count("GET", "video-projects") == 13      # t = 0, 5, ... 60 s
    mh = FakeMagicHour(tmp_path, result=mh_result(tmp_path), poll_status=401)
    with pytest.raises(fl.FaceLockError) as e:
        fl.wait_for_project(mh, KEY, "p", clock() + 60, clock, clock.sleep, 5)
    assert str(e.value) == fl.BAD_KEY and mh.count("GET", "video-projects") == 1


# ------------------------------------------------------------------ the routes: key, preconditions, race, skip, stuck

def test_no_key_or_a_short_key_is_refused_and_nothing_is_queued(tmp_path):
    r = FaceRig(tmp_path, FakeMagicHour(tmp_path, result=mh_result(tmp_path)))
    job = r.swapped_job()
    for key in (None, "", "short"):
        res = r.lock(job, key=key)
        assert res.status_code == 400 and res.json()["error"] == "Connect your Magic Hour key first."
    assert r.facelock_calls == [] and r.c.get(f"/status/{job}").json()["facelock"]["state"] == "waiting"


def test_preconditions_and_the_race_rule(tmp_path):
    r = FaceRig(tmp_path)                                               # no fake Magic Hour: spawn only records
    job = r.upload(facelock="yes").json()["job"]
    d = r.root / job
    assert r.lock(job).status_code == 409                               # swap not done
    assert r.lock("0000000000000000").status_code == 404
    r.c.post(f"/generate/{job}")
    for stage, state in (("prepare", "queued"), ("prepare", "preparing"), ("render", "queued"), ("render", "rendering")):
        svc.set_stage(d, stage, state=state)
        res = r.lock(job)
        assert res.status_code == 409, (stage, state)
        svc.set_stage(d, stage, state="idle")
    assert r.facelock_calls == []
    assert r.lock(job).status_code == 200 and r.facelock_calls == [job]
    st = r.c.get(f"/status/{job}").json()["facelock"]
    assert st["state"] == "queued"
    assert r.lock(job).status_code == 409                               # a double click does not start a second paid job
    assert r.c.post(f"/retexture/{job}").status_code == 409             # nothing that touches the clip runs meanwhile
    svc.set_stage(d, "prepare", state="ready")
    assert r.c.post(f"/render/{job}").status_code == 409
    svc.set_stage(d, "facelock", state="working")
    assert r.lock(job).status_code == 409 and r.c.post(f"/facelock/{job}/skip").status_code == 409
    assert r.facelock_calls == [job]
    assert_key_nowhere(r, job)


def test_skip_prepares_the_slider_on_the_plain_swap(tmp_path):
    r = FaceRig(tmp_path)
    job = r.upload(facelock="yes").json()["job"]
    assert r.c.post(f"/facelock/{job}/skip").status_code == 409         # idle: nothing to skip yet
    r.c.post(f"/generate/{job}")
    res = r.c.post(f"/facelock/{job}/skip")
    assert res.status_code == 200 and res.json()["facelock"] == {"state": "skipped"}
    st = r.c.get(f"/status/{job}").json()
    assert st["facelock"]["state"] == "skipped" and st["prepare"]["state"] == "ready" and r.calls["prepare"] == 1
    assert r.c.post(f"/facelock/{job}/skip").status_code == 409         # only once
    assert r.c.get(f"/preview/{job}", params={"frame": 1}).status_code == 200


def test_a_face_lock_that_never_finishes_is_reported_and_can_be_skipped(tmp_path):
    r = FaceRig(tmp_path)
    job = r.swapped_job()
    assert r.lock(job).status_code == 200
    svc.set_stage(r.root / job, "facelock", state="working")
    assert ss.status(r.root, job)["facelock"]["state"] == "working"
    later = ss.status(r.root, job, now=time.time() + ss.STUCK_AFTER_SECONDS["facelock"] + 5)
    assert later["facelock"]["state"] == "error" and "stopped without finishing" in later["facelock"]["message"]
    assert ss.ACTIVE["facelock"] == ("queued", "working") and ss.STUCK_AFTER_SECONDS["facelock"] == 2100
    assert r.c.post(f"/facelock/{job}/skip").status_code == 200
    assert r.c.get(f"/status/{job}").json()["prepare"]["state"] == "ready"


def test_a_spawn_failure_reports_an_error_without_the_key(tmp_path):
    r = FaceRig(tmp_path)
    job = r.swapped_job()

    def boom(job, key):
        raise RuntimeError(f"could not reach modal with {key}")
    r.facelock = boom
    import swap_api
    from fastapi.testclient import TestClient
    c = TestClient(swap_api.build_app(r.root, r.mask, r.swap, r.render, r.fetch_weights, lambda: "ready",
                                      spawn_prepare=r.prepare, detector=FAKE, spawn_facelock=boom),
                   raise_server_exceptions=False)
    res = c.post(f"/facelock/{job}", headers={"X-Magic-Hour-Key": KEY})
    assert res.status_code == 500 and KEY not in res.text
    st = c.get(f"/status/{job}").json()["facelock"]
    assert st["state"] == "error" and KEY not in json.dumps(st)
    assert c.post(f"/facelock/{job}/skip").status_code == 200


def test_the_slider_can_be_prepared_again_on_a_locked_clip(tmp_path):
    mh = FakeMagicHour(tmp_path, result=mh_result(tmp_path))
    r = FaceRig(tmp_path, mh)
    job = r.swapped_job()
    r.lock(job)
    d = r.root / job
    locked = (d / "src.mp4").read_bytes()
    assert r.c.post(f"/render/{job}", params={"strength": 1.0}).status_code == 200
    assert (d / "final.mp4").exists()
    again = r.c.post(f"/retexture/{job}", files={"skin": ("s.png", photo_bytes(seed=9), "image/png")})
    assert again.status_code == 200
    st = r.c.get(f"/status/{job}").json()
    assert st["prepare"]["state"] == "ready" and st["facelock"]["state"] == "done" and st["render"]["state"] == "idle"
    assert (d / "src.mp4").read_bytes() == locked and not (d / "final.mp4").exists()
    assert r.c.get(f"/preview/{job}", params={"frame": 2, "strength": 2}).status_code == 200


def test_delete_removes_the_face_photo_and_the_plain_swap_with_the_job(tmp_path):
    mh = FakeMagicHour(tmp_path, result=mh_result(tmp_path))
    r = FaceRig(tmp_path, mh)
    job = r.swapped_job()
    r.lock(job, files={"face": ("f.png", photo_bytes(seed=2), "image/png")})
    assert (r.root / job / "face.png").exists() and (r.root / job / "src_plain.mp4").exists()
    assert r.c.delete(f"/job/{job}").json() == {"deleted": True} and not (r.root / job).exists()


def test_run_spends_nothing_unless_queued(tmp_path):
    mh = FakeMagicHour(tmp_path, result=mh_result(tmp_path))
    r = FaceRig(tmp_path)
    job = r.swapped_job()                                          # "waiting", never queued
    assert fl.run(r.root, job, KEY, http=mh) is False and mh.calls == []
    r.c.post(f"/facelock/{job}/skip")
    assert fl.run(r.root, job, KEY, http=mh) is False and mh.calls == []


# ------------------------------------------------------------------ helpers

def test_scrub_and_friendly_errors():
    assert KEY not in fl.scrub(f"boom {KEY} and mhk_live_other_9", KEY)
    assert fl.mh_error(403, b"") == fl.BAD_KEY and fl.mh_error(402, b"") == fl.NO_CREDIT
    assert fl.mh_error(400, b'{"message":"Insufficient balance"}') == fl.NO_CREDIT
    assert fl.mh_error(429, b"") == fl.BUSY and fl.mh_error(422, b'{"code":"no_source_face"}') == fl.NO_FACE
    assert fl.upload_host_ok(UP_VIDEO) and fl.upload_host_ok("https://magichour.ai/x")
    for bad in ("http://videos.magichour.ai/x", "https://magichour.ai.evil.com/x", "https://evilmagichour.ai/x", None):
        assert not fl.upload_host_ok(bad)


def test_the_real_transport_never_follows_a_redirect_with_the_key_and_only_to_https():
    h = fl._Redirects()
    with_key = urllib.request.Request("https://api.magichour.ai/v1/x")
    with_key.add_unredirected_header("Authorization", "Bearer x")
    assert h.redirect_request(with_key, None, 302, "Found", {}, "https://elsewhere.example/") is None
    plain = urllib.request.Request("https://videos.magichour.ai/out.mp4")
    assert h.redirect_request(plain, None, 302, "Found", {}, "http://cdn.example/out.mp4") is None
    assert h.redirect_request(plain, None, 302, "Found", {}, "https://cdn.example/out.mp4") is not None


def test_the_modal_function_is_cheap_and_spawns_the_slider_only_when_told():
    app = (Path(__file__).resolve().parent.parent / "app.py").read_text()
    head = app[:app.index("def facelock_job(")].rsplit("@app.function", 1)[1]
    assert "image=light" in head and "gpu" not in head and "timeout=1800" in head and "cpu=2" in head
    assert '"facelock.py"' in app and "spawn_facelock=facelock_job.spawn" in app
    body = app[app.index("def swap_job("):app.index("def prepare_job(")]
    assert "prepare = ss.finish_swap(" in body and "if not prepare:" in body
    src = (Path(__file__).resolve().parent.parent / "facelock.py").read_text()
    assert "import modal" not in src and "requests" not in src
