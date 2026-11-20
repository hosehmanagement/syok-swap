"""H3 web API (h3_api.build_h3_app): every route through FastAPI's TestClient, with fake GPU steps. No Modal, no GPU.

The territory rules (H3 Community License: no US, EU, UK or South Korea) are pinned here: the country is read ONLY from
the X-Syok-Country header the router sets, fails closed, and is never taken from the multipart body."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3_engine as h3
import h3_territory as terr
import texture_service as svc
from test_swapkit import make_video
from test_texture_service import SKIN, photo_bytes

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient
import h3_api

OK = {"X-Syok-Country": "MY"}
EU27 = "AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE".split()
TERRITORY_JSON = {"error": "territory", "message": "MiniMax H3 is not available where you are (US, EU, UK, South Korea)."}


class Rig:
    """Wires the API to fake GPU steps. `calls` counts how often each GPU / download step was started."""

    def __init__(self, tmp_path, weights="ready", check_ok=True, seconds=8.0, check_runs=True):
        self.root = tmp_path / "jobs"
        self.root.mkdir()
        self.tmp = tmp_path
        self.weights = weights
        self.check_ok = check_ok
        self.check_runs = check_runs
        self.calls = {"check": 0, "swap": 0, "weights": 0}
        self.events = []
        self.seconds = seconds
        app = h3_api.build_h3_app(self.root, self.check, self.swap, self.fetch, lambda: self.weights,
                                  before_read=lambda: self.events.append("r"), after_write=lambda: self.events.append("w"),
                                  probe=lambda p: {"duration": self.seconds, "has_audio": True})
        self.c = TestClient(app)

    def _clip(self, name):
        return make_video(self.tmp / name, SKIN, n=4, fps=24, w=32, h=32)

    def check(self, job):                                   # what check_job does on the small GPU
        self.calls["check"] += 1
        if not self.check_runs:
            return
        h3.begin(self.root, job, "check")
        if self.check_ok:
            h3.finish(self.root, job, "check", self._clip("check_out.mp4"))
        else:
            h3.fail(self.root, job, "check", "The mask check failed. Nothing was swapped and no swap was charged.")

    def swap(self, job):                                    # what swap_job does on the big GPU (the paid step)
        self.calls["swap"] += 1
        h3.begin(self.root, job, "swap")
        h3.finish(self.root, job, "swap", make_video(self.tmp / "h3_out.mp4", SKIN, n=24, fps=24, w=64, h=64))

    def fetch(self):
        self.calls["weights"] += 1

    def upload(self, data=None, headers=OK, files=None, **extra):
        files = files or {"video": ("v.mp4", self._clip("src.mp4").read_bytes(), "video/mp4"),
                          "photo": ("p.png", photo_bytes(), "image/png")}
        d = {"consent": "yes", "territory": "yes", **extra} if data is None else data
        return self.c.post("/swap", files=files, data=d, headers=headers or {})

    def jobs(self):
        return sorted(p.name for p in self.root.iterdir())


# ---------------------------------------------------------------- /health and /setup

def test_health_describes_the_engine_without_any_country(tmp_path):
    r = Rig(tmp_path, weights="downloading")
    h = r.c.get("/health")                                   # no header: health is a free check
    assert h.status_code == 200
    assert h.json() == {"ok": True, "app": "syok-swap-h3", "version": 1, "engine": "minimax-h3", "weights": "downloading",
                        "label": "MiniMax H3", "territory": "US, EU, UK and South Korea are excluded",
                        "limits": {"video_mb": 60, "photo_mb": 8, "seconds": 30, "keep_days": 3},
                        "resolutions": ["480p", "576p", "736p", "768p", "1080p"], "margins": ["tight", "normal", "loose"]}
    assert h.json()["label"] == terr.ENGINE_LABEL


def test_setup_starts_the_download_once_and_only_when_missing(tmp_path):
    r = Rig(tmp_path, weights="missing")
    assert r.c.post("/setup", headers=OK).json() == {"weights": "downloading"}
    assert r.calls["weights"] == 1
    r.weights = "downloading"
    assert r.c.post("/setup", headers=OK).json() == {"weights": "downloading"}
    r.weights = "ready"
    assert r.c.post("/setup", headers=OK).json() == {"weights": "ready"}
    assert r.calls["weights"] == 1


@pytest.mark.parametrize("headers", [{}, {"X-Syok-Country": ""}, {"X-Syok-Country": "US"}, {"X-Syok-Country": "XX"}])
def test_setup_is_refused_outside_the_licensed_territory(tmp_path, headers):
    r = Rig(tmp_path, weights="missing")
    s = r.c.post("/setup", headers=headers)
    assert s.status_code == 451 and s.json() == TERRITORY_JSON
    assert r.calls["weights"] == 0


# ---------------------------------------------------------------- the whole flow

def test_whole_flow_through_the_http_layer(tmp_path):
    r = Rig(tmp_path)
    up = r.upload(margin="tight", resolution="480p", start="1", length="6")
    assert up.status_code == 200, up.text
    job = up.json()["job"]
    est = up.json()["estimate"]
    assert est["approx"] is True and est["seconds"] == 6.0 and est["resolution"] == "480p"
    assert est["check_minutes"] == 17 and est["swap_minutes"] == 3 and est["clips"] == h3.clip_count(6.0, 6.0)
    assert r.calls == {"check": 1, "swap": 0, "weights": 0}             # the mask check started by itself, nothing paid
    meta = svc.read_meta(svc.job_dir(r.root, job))
    assert meta["h3_territory"] == "yes" and meta["country"] == "MY"
    assert meta["territory_record"]["ip_country"] == "MY" and terr.verify_territory_record(meta["territory_record"])
    assert meta["request"]["margin"] == "tight" and meta["request"]["start"] == 1.0 and meta["request"]["length"] == 6.0

    st = r.c.get(f"/status/{job}").json()
    assert st["job"] == job and st["engine"] == "minimax-h3" and st["check"]["state"] == "ready"
    assert st["swap"] == {"state": "idle", "attempts": 0} and st["estimate"] == est and "expires_at" in st
    mv = r.c.get(f"/mask/{job}")
    assert mv.status_code == 200 and mv.headers["content-type"] == "video/mp4" and len(mv.content) > 200
    assert r.c.get(f"/result/{job}").status_code == 409

    g = r.c.post(f"/generate/{job}", headers=OK)
    assert g.status_code == 200 and g.json() == {"state": "queued", "resume": False}
    st = r.c.get(f"/status/{job}").json()
    assert st["swap"]["state"] == "done" and st["swap"]["attempts"] == 1
    res = r.c.get(f"/result/{job}")
    assert res.status_code == 200 and res.headers["content-type"] == "video/mp4" and len(res.content) > 500
    assert f"syok-h3-{job}.mp4" in res.headers["content-disposition"]
    assert r.calls == {"check": 1, "swap": 1, "weights": 0}

    assert r.c.delete(f"/job/{job}").json() == {"deleted": True}
    assert r.c.get(f"/status/{job}").status_code == 404
    assert r.jobs() == []


def test_read_and_write_hooks_bracket_the_requests(tmp_path):
    r = Rig(tmp_path)
    job = r.upload().json()["job"]
    assert r.events[0] == "r" and "w" in r.events                      # read the Volume first, commit after writing
    r.events.clear()
    r.c.get(f"/status/{job}")
    assert r.events == ["r"]
    r.events.clear()
    r.c.post(f"/generate/{job}", headers=OK)
    assert r.events == ["r", "w"]


# ---------------------------------------------------------------- territory on /swap (fail closed, header only)

@pytest.mark.parametrize("code", ["US", "GB", "KR", "us", " gb "] + EU27)
def test_blocked_countries_get_451_and_nothing_is_stored(tmp_path, code):
    r = Rig(tmp_path)
    s = r.upload(headers={"X-Syok-Country": code})
    assert s.status_code == 451 and s.json() == TERRITORY_JSON
    assert r.jobs() == [] and r.calls["check"] == 0


@pytest.mark.parametrize("value", ["", "   ", "XX", "T1", "ZZ", "EU", "M", "MYS", "M1", "my,us"])
def test_unknown_or_malformed_country_fails_closed(tmp_path, value):
    r = Rig(tmp_path)
    s = r.upload(headers={"X-Syok-Country": value})
    assert s.status_code == 451 and s.json()["error"] == "territory"
    assert r.jobs() == [] and r.calls["check"] == 0


def test_a_missing_country_header_is_refused(tmp_path):
    r = Rig(tmp_path)
    s = r.upload(headers={})
    assert s.status_code == 451 and s.json() == TERRITORY_JSON
    assert r.jobs() == [] and r.calls["check"] == 0


@pytest.mark.parametrize("field", ["X-Syok-Country", "x-syok-country", "x_syok_country", "country", "h3_country"])
def test_the_country_is_never_taken_from_the_multipart_body(tmp_path, field):
    r = Rig(tmp_path)
    s = r.upload(headers={}, **{field: "MY"})                           # no header: a body field must not stand in for it
    assert s.status_code == 451 and s.json()["error"] == "territory"
    s = r.upload(headers={"X-Syok-Country": "US"}, **{field: "MY"})     # and it cannot override a blocked header either
    assert s.status_code == 451
    assert r.jobs() == [] and r.calls["check"] == 0


@pytest.mark.parametrize("tick", [None, "", "no", "YES", "true", "1", " yes"])
def test_without_the_territory_tick_box_it_is_451(tmp_path, tick):
    r = Rig(tmp_path)
    data = {"consent": "yes"} if tick is None else {"consent": "yes", "territory": tick}
    s = r.upload(data=data)
    assert s.status_code == 451 and s.json() == TERRITORY_JSON
    assert r.jobs() == [] and r.calls["check"] == 0


def test_the_old_h3_territory_field_name_does_not_count_as_the_tick(tmp_path):
    r = Rig(tmp_path)
    s = r.upload(data={"consent": "yes", "h3_territory": "yes"})       # the contract's field is `territory`
    assert s.status_code == 451 and r.jobs() == []


@pytest.mark.parametrize("code", ["MY", "sg", " jp ", "AU", "ID", "TH"])
def test_allowed_countries_pass_and_are_recorded_normalised(tmp_path, code):
    r = Rig(tmp_path)
    s = r.upload(headers={"X-Syok-Country": code})
    assert s.status_code == 200, s.text
    meta = svc.read_meta(svc.job_dir(r.root, s.json()["job"]))
    assert meta["country"] == code.strip().upper() and meta["h3_territory"] == "yes"


# ---------------------------------------------------------------- consent and bad input

@pytest.mark.parametrize("consent", [None, "", "no", "YES"])
def test_without_consent_it_is_422_and_nothing_is_stored(tmp_path, consent):
    r = Rig(tmp_path)
    data = {"territory": "yes"} if consent is None else {"territory": "yes", "consent": consent}
    s = r.upload(data=data)
    assert s.status_code == 422 and "agreed" in s.json()["error"]
    assert r.jobs() == [] and r.calls["check"] == 0


def test_a_blocked_country_is_refused_before_consent_is_even_looked_at(tmp_path):
    r = Rig(tmp_path)
    s = r.upload(data={"territory": "yes"}, headers={"X-Syok-Country": "DE"})
    assert s.status_code == 451


def test_missing_files_are_422_from_fastapi(tmp_path):
    r = Rig(tmp_path)
    s = r.c.post("/swap", data={"consent": "yes", "territory": "yes"}, headers=OK)
    assert s.status_code == 422 and r.jobs() == []


@pytest.mark.parametrize("extra,match", [({"margin": "huge"}, "Margin"), ({"resolution": "4k"}, "Resolution"),
                                         ({"start": "abc"}, "Start"), ({"length": "-1"}, "Length")])
def test_bad_settings_are_400_and_nothing_is_stored(tmp_path, extra, match):
    r = Rig(tmp_path)
    s = r.upload(**extra)
    assert s.status_code == 400 and match in s.json()["error"]
    assert r.jobs() == [] and r.calls["check"] == 0


def test_a_section_over_the_limit_is_refused(tmp_path):
    r = Rig(tmp_path, seconds=90)
    s = r.upload()
    assert s.status_code == 400 and "limit" in s.json()["error"] and r.jobs() == []


def test_a_bad_photo_type_is_refused(tmp_path):
    r = Rig(tmp_path)
    files = {"video": ("v.mp4", r._clip("src.mp4").read_bytes(), "video/mp4"), "photo": ("p.gif", b"GIF89a", "image/gif")}
    s = r.upload(files=files)
    assert s.status_code == 400 and "png, jpg or webp" in s.json()["error"] and r.jobs() == []


def test_swap_needs_the_model_files_first(tmp_path):
    r = Rig(tmp_path, weights="downloading")
    s = r.upload()
    assert s.status_code == 409 and "setup" in s.json()["error"]
    assert r.jobs() == [] and r.calls["check"] == 0


def test_a_check_that_cannot_start_is_marked_failed_not_left_queued(tmp_path):
    r = Rig(tmp_path)

    def boom(job):
        raise RuntimeError("modal down")
    app = h3_api.build_h3_app(r.root, boom, r.swap, r.fetch, lambda: "ready", probe=lambda p: {"duration": 8, "has_audio": True})
    c = TestClient(app)
    files = {"video": ("v.mp4", r._clip("src.mp4").read_bytes(), "video/mp4"), "photo": ("p.png", photo_bytes(), "image/png")}
    s = c.post("/swap", files=files, data={"consent": "yes", "territory": "yes"}, headers=OK)
    assert s.status_code == 500 and "modal down" not in s.text
    (job,) = r.jobs()
    assert h3.status(r.root, job)["check"]["state"] == "error"


# ---------------------------------------------------------------- /generate: the paid step

def test_generate_twice_is_refused_and_runs_once(tmp_path):
    r = Rig(tmp_path)
    job = r.upload().json()["job"]
    assert r.c.post(f"/generate/{job}", headers=OK).status_code == 200
    again = r.c.post(f"/generate/{job}", headers=OK)
    assert again.status_code == 409 and "already swapped" in again.json()["error"]
    assert r.calls["swap"] == 1


def test_generate_while_the_swap_is_running_is_refused(tmp_path):
    r = Rig(tmp_path)
    job = r.upload().json()["job"]
    h3.start_swap(r.root, job)                                          # queued by an earlier request, GPU not done yet
    s = r.c.post(f"/generate/{job}", headers=OK)
    assert s.status_code == 409 and "already running" in s.json()["error"] and r.calls["swap"] == 0


def test_generate_needs_a_ready_mask_check(tmp_path):
    r = Rig(tmp_path, check_runs=False)                                 # the check is still queued
    job = r.upload().json()["job"]
    s = r.c.post(f"/generate/{job}", headers=OK)
    assert s.status_code == 409 and "Check the mask" in s.json()["error"] and r.calls["swap"] == 0


def test_generate_refuses_a_failed_mask_check(tmp_path):
    r = Rig(tmp_path, check_ok=False)
    job = r.upload().json()["job"]
    st = r.c.get(f"/status/{job}").json()
    assert st["check"]["state"] == "error" and "no swap was charged" in st["check"]["message"]
    assert r.c.get(f"/mask/{job}").status_code == 409
    assert r.c.post(f"/generate/{job}", headers=OK).status_code == 409 and r.calls["swap"] == 0


def test_a_failed_swap_can_be_resumed(tmp_path):
    r = Rig(tmp_path)
    job = r.upload().json()["job"]
    h3.start_swap(r.root, job); h3.begin(r.root, job, "swap")
    h3.fail(r.root, job, "swap", "The swap failed. Start it again on the same job: it resumes from the last finished clip.")
    g = r.c.post(f"/generate/{job}", headers=OK)
    assert g.status_code == 200 and g.json() == {"state": "queued", "resume": True}
    st = r.c.get(f"/status/{job}").json()["swap"]
    assert st["state"] == "done" and st["attempts"] == 2


@pytest.mark.parametrize("headers", [{}, {"X-Syok-Country": ""}, {"X-Syok-Country": "KR"}, {"X-Syok-Country": "T1"}])
def test_generate_is_refused_outside_the_licensed_territory(tmp_path, headers):
    r = Rig(tmp_path)
    job = r.upload().json()["job"]
    g = r.c.post(f"/generate/{job}", headers=headers)
    assert g.status_code == 451 and g.json() == TERRITORY_JSON
    assert r.calls["swap"] == 0 and r.c.get(f"/status/{job}").json()["swap"]["state"] == "idle"


def test_generate_country_is_not_taken_from_a_query_or_body(tmp_path):
    r = Rig(tmp_path)
    job = r.upload().json()["job"]
    g = r.c.post(f"/generate/{job}?x_syok_country=MY", data={"X-Syok-Country": "MY"})
    assert g.status_code == 451 and r.calls["swap"] == 0


def test_a_swap_that_cannot_start_can_be_started_again(tmp_path):
    r = Rig(tmp_path)
    job = r.upload().json()["job"]

    def boom(j):
        raise RuntimeError("modal down")
    c = TestClient(h3_api.build_h3_app(r.root, r.check, boom, r.fetch, lambda: "ready"))
    s = c.post(f"/generate/{job}", headers=OK)
    assert s.status_code == 500 and "modal down" not in s.text
    assert h3.status(r.root, job)["swap"]["state"] == "error"
    assert r.c.post(f"/generate/{job}", headers=OK).json()["resume"] is True


# ---------------------------------------------------------------- status / mask / result / delete

def test_status_mask_result_409_while_not_ready(tmp_path):
    r = Rig(tmp_path, check_runs=False)
    job = r.upload().json()["job"]
    st = r.c.get(f"/status/{job}").json()
    assert st["check"]["state"] == "queued" and st["swap"]["state"] == "idle"
    m = r.c.get(f"/mask/{job}")
    assert m.status_code == 409 and "not ready" in m.json()["error"]
    res = r.c.get(f"/result/{job}")
    assert res.status_code == 409 and "not ready" in res.json()["error"]


def test_a_wan_job_on_the_same_volume_is_409_not_an_h3_job(tmp_path):
    r = Rig(tmp_path)
    clip = make_video(tmp_path / "w.mp4", SKIN, n=12, fps=24, w=64, h=64).read_bytes()
    wan = svc.new_job(r.root, clip, photo_bytes(), ".png", "yes")
    for path in (f"/status/{wan}", f"/mask/{wan}", f"/result/{wan}"):
        s = r.c.get(path)
        assert s.status_code == 409 and "not an H3 job" in s.json()["error"], path
    assert r.c.post(f"/generate/{wan}", headers=OK).status_code == 409


@pytest.mark.parametrize("job", ["0123456789abcdef", "nothex", "../../etc"])
def test_unknown_or_malformed_job_ids_are_404(tmp_path, job):
    r = Rig(tmp_path)
    for method, path in (("get", f"/status/{job}"), ("get", f"/mask/{job}"), ("get", f"/result/{job}"),
                         ("delete", f"/job/{job}")):
        s = getattr(r.c, method)(path)
        assert s.status_code == 404, (method, path)
    assert r.c.post(f"/generate/{job}", headers=OK).status_code == 404


def test_status_and_delete_work_without_a_country(tmp_path):
    """A user can always read and delete their own job (the router checks the country on every route anyway)."""
    r = Rig(tmp_path)
    job = r.upload().json()["job"]
    assert r.c.get(f"/status/{job}").status_code == 200
    assert r.c.get(f"/mask/{job}").status_code == 200
    assert r.c.delete(f"/job/{job}").status_code == 200 and r.jobs() == []


# ---------------------------------------------------------------- weights_state and the Modal wiring

def _fill_models(d):
    for m in h3.MODEL_FILES:
        (d / m["folder"]).mkdir(parents=True, exist_ok=True)
        (d / m["folder"] / m["file"]).write_bytes(b"x")


def test_weights_state_missing_downloading_ready_and_stale_flag(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    assert h3.weights_state(d) == "missing"
    (d / h3.FETCH_FLAG).write_text("1000")
    assert h3.weights_state(d, now=1000 + 60) == "downloading"
    assert h3.weights_state(d, now=1000 + h3.FETCH_STALE_SECONDS + 1) == "missing"     # a fetch that died
    (d / h3.FETCH_FLAG).write_text("garbage")                                          # unreadable flag: its mtime counts
    assert h3.weights_state(d) == "downloading"
    _fill_models(d)
    assert h3.weights_state(d) == "ready"                                              # files present wins over the flag
    (d / h3.MODEL_FILES[-1]["folder"] / h3.MODEL_FILES[-1]["file"]).unlink()           # SAM: needed by both stages
    (d / h3.FETCH_FLAG).unlink()
    assert h3.weights_state(d) == "missing"


H3_APP = (ROOT / "h3_app.py").read_text()
H3_API = (ROOT / "h3_api.py").read_text()


def test_the_api_module_does_not_import_modal():
    assert "import modal" not in H3_API and "from modal" not in H3_API


def test_h3_app_web_endpoint_is_locked_cpu_only_and_wired_to_the_real_steps():
    compile(H3_APP, "h3_app.py", "exec")
    assert H3_APP.count("requires_proxy_auth=True") == 1 and 'modal.App("syok-swap-h3")' in H3_APP
    head = H3_APP[:H3_APP.index("def web(")].rsplit("@app.function", 1)[1]
    assert "image=light" in head and "gpu" not in head and "@modal.asgi_app(requires_proxy_auth=True)" in head
    body = H3_APP[H3_APP.index("def web("):].split("\n# ----", 1)[0]
    assert "h3_api.build_h3_app(JOBS, check_job.spawn, swap_job.spawn, lambda: fetch_models.spawn({})" in body
    assert "eng.weights_state(MODELS)" in body and "after_write=jobs.commit" in body
    assert "jobs, models" in body and ".reload()" in body                        # both Volumes are read fresh per request
    light = H3_APP[H3_APP.index("light = ("):H3_APP.index("def _eng(")]
    for f in ("h3_api.py", "h3_territory.py", "h3_engine.py", "fastapi"):
        assert f in light, f


def test_fetch_models_writes_and_always_clears_the_downloading_flag():
    body = H3_APP[H3_APP.index("def fetch_models("):H3_APP.index("def model_sizes(")]
    assert "eng.FETCH_FLAG" in body and "flag.write_text(" in body
    fin = body.split("finally:", 1)[1]
    assert "flag.unlink(missing_ok=True)" in fin and "models.commit()" in fin


def test_no_secret_or_country_is_ever_logged_or_put_in_meta_beyond_the_code(tmp_path):
    r = Rig(tmp_path)
    job = r.upload(headers={"X-Syok-Country": "MY", "Authorization": "Bearer not-a-real-token"}).json()["job"]
    raw = (svc.job_dir(r.root, job) / "meta.json").read_text()
    assert "not-a-real-token" not in raw and "Bearer" not in raw
    assert json.loads(raw)["country"] == "MY"
