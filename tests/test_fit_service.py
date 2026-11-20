import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pytest.importorskip("fastapi")
import swap_service as ss
from test_swap_service import Rig


def test_health_says_fit_is_available(tmp_path):
    assert Rig(tmp_path).c.get("/health").json()["fit"] is True


def test_fit_is_off_by_default_and_the_request_is_as_before(tmp_path):
    r = Rig(tmp_path)
    up = r.upload()
    assert "fit" not in ss.request_of(r.root, up.json()["job"])
    assert "fit_usd" not in up.json()["estimate"]


@pytest.mark.parametrize("value", ["yes", "YES", " 1 ", "true"])
def test_fit_on_is_stored_and_priced(tmp_path, value):
    r = Rig(tmp_path)
    base = r.upload().json()["estimate"]
    up = r.upload(fit=value)
    assert up.status_code == 200
    assert ss.request_of(r.root, up.json()["job"])["fit"] is True
    est = up.json()["estimate"]
    assert est["fit_usd"] == ss.FIT_USD and est["total_usd"] == round(base["total_usd"] + ss.FIT_USD, 2)
    assert est["swap_minutes"] == base["swap_minutes"] + 3


@pytest.mark.parametrize("value", ["no", "", "0", "maybe"])
def test_anything_else_is_off(tmp_path, value):
    r = Rig(tmp_path)
    up = r.upload(fit=value)
    assert up.status_code == 200 and "fit" not in ss.request_of(r.root, up.json()["job"])


def test_status_reports_what_the_fit_step_did_and_nothing_before(tmp_path):
    r = Rig(tmp_path)
    job = r.upload(fit="yes").json()["job"]
    assert "fit" not in r.c.get(f"/status/{job}").json()
    ss.record_fit(r.root, job, {"state": "skipped", "message": "No fitted frame passed the checks, so your own photo is used.",
                                "frames": {"first": {"state": "rejected", "outside_change": 41.5, "inside_change": 30.0}}})
    st = r.c.get(f"/status/{job}").json()["fit"]
    assert st["state"] == "skipped" and "your own photo" in st["message"] and st["frames"]["first"]["state"] == "rejected"


def test_a_fit_stuck_for_over_an_hour_is_reported_not_left_spinning(tmp_path):
    r = Rig(tmp_path)
    job = r.upload(fit="yes").json()["job"]
    ss.tsvc.set_stage(ss.tsvc.job_dir(r.root, job), "fit", state="working")
    st = ss.status(r.root, job, now=ss.tsvc.read_meta(ss.tsvc.job_dir(r.root, job))["created"] + 10**5)
    assert st["fit"]["state"] == "error" and "stopped without finishing" in st["fit"]["message"]
