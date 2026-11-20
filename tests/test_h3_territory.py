import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import consent as cs
import h3_territory as h3

EU27 = "AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE".split()
ALL_BLOCKED = ["US", "GB", "KR"] + EU27
WORKER = ROOT / "docs" / "h3-territory-worker.js"


def test_blocked_list_is_exactly_us_gb_kr_and_eu27():
    assert len(EU27) == 27
    assert h3.EU_MEMBER_CODES == frozenset(EU27)
    assert h3.BLOCKED_COUNTRY_CODES == frozenset(ALL_BLOCKED)
    assert len(h3.BLOCKED_COUNTRY_CODES) == 30


@pytest.mark.parametrize("code", ALL_BLOCKED)
def test_every_blocked_code_is_blocked(code):
    r = h3.check_country(code)
    assert r["status"] == h3.BLOCKED and r["country"] == code and r["reason"]
    assert not h3.is_allowed(code)


@pytest.mark.parametrize("code", ALL_BLOCKED)
def test_every_blocked_code_is_blocked_in_lowercase(code):
    assert h3.check_country(code.lower())["status"] == h3.BLOCKED


@pytest.mark.parametrize("code", ["MY", "SG", "JP", "AU"])
def test_allowed_examples(code):
    r = h3.check_country(code)
    assert r == {"status": h3.ALLOWED, "country": code, "reason": f"{code} is not in the H3 excluded list."}


@pytest.mark.parametrize("raw,norm", [("my", "MY"), ("sg", "SG"), (" jp ", "JP"), ("Au", "AU")])
def test_lowercase_and_spaces_are_normalised(raw, norm):
    r = h3.check_country(raw)
    assert r["status"] == h3.ALLOWED and r["country"] == norm


@pytest.mark.parametrize("bad", [None, "", "   ", "XX", "T1", "ZZ", "EU", "M", "MYS", "M1", "12", "U.S.", 458, b"MY"])
def test_unknown_missing_or_malformed_is_blocked(bad):
    r = h3.check_country(bad)
    assert r["status"] == h3.BLOCKED and r["reason"]
    assert not h3.is_allowed(bad)


def test_none_and_empty_reasons_say_unknown():
    assert "unknown" in h3.check_country(None)["reason"]
    assert "unknown" in h3.check_country("")["reason"]


def test_ui_strings_are_exact():
    assert h3.ENGINE_LABEL == "MiniMax H3"
    assert h3.TERRITORY_CONSENT_TEXT == "I am outside the US, EU, UK and South Korea"
    assert "MiniMax H3" in h3.LICENCE_NOTICE and "H3 Community License" in h3.LICENCE_NOTICE
    for place in ("US", "European Union", "United Kingdom", "South Korea"):
        assert place in h3.LICENCE_NOTICE


@pytest.mark.parametrize("value,ok", [("yes", True), ("", False), (None, False), ("no", False), ("YES", False), ("on", False)])
def test_tick_box_follows_the_consent_yes_rule(value, ok):
    assert h3.territory_consent_given(value) is ok


def test_record_saved_and_loaded_like_existing_consent(tmp_path):
    rec = h3.make_territory_record("my", "yes", now=1000.0)
    assert rec["ip_country"] == "MY" and rec["text"] == h3.TERRITORY_CONSENT_TEXT and rec["created"] == 1000.0
    p = tmp_path / "territory.json"
    cs.save(rec, p)
    assert h3.verify_territory_record(cs.load(p)) is True


def test_record_refused_without_tick_or_from_blocked_country():
    with pytest.raises(PermissionError):
        h3.make_territory_record("MY", "")
    for code in ("US", "de", None, ""):
        with pytest.raises(PermissionError):
            h3.make_territory_record(code, "yes")


def test_record_refused_after_wording_change_or_tamper():
    rec = h3.make_territory_record("SG", "yes", now=1.0)
    with pytest.raises(PermissionError):
        h3.verify_territory_record({**rec, "text_sha256": "0" * 64})
    with pytest.raises(PermissionError):
        h3.verify_territory_record({**rec, "ip_country": "GB"})
    with pytest.raises(PermissionError):
        h3.verify_territory_record({**rec, "attest": {}})


def test_worker_blocked_list_matches_python():
    js = WORKER.read_text()
    block = js.split("const BLOCKED = new Set([", 1)[1].split("]);", 1)[0]
    assert set(re.findall(r'"([A-Z]{2})"', block)) == set(h3.BLOCKED_COUNTRY_CODES)
    assert "request.cf" in js and "status: 403" in js and "NOT DEPLOYED" in js


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_worker_returns_403_for_blocked_or_missing(tmp_path):
    mod = tmp_path / "worker.mjs"
    mod.write_text(WORKER.read_text())
    script = tmp_path / "run.mjs"
    script.write_text(
        "import w, { h3Blocked } from './worker.mjs';\n"
        "globalThis.fetch = async () => new Response('ok', { status: 200 });\n"
        "const out = {};\n"
        "for (const c of ['US','gb','KR','DE','SE','MY','sg','JP','AU','', 'XX', 'T1', null]) {\n"
        "  const req = new Request('https://x.test/'); Object.defineProperty(req, 'cf', { value: c === null ? undefined : { country: c } });\n"
        "  out[String(c)] = (await w.fetch(req, {}, {})).status;\n"
        "}\n"
        "out.noCf = (await w.fetch(new Request('https://x.test/'), {}, {})).status;\n"
        "console.log(JSON.stringify(out));\n")
    r = subprocess.run(["node", str(script)], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    import json
    out = json.loads(r.stdout)
    for c in ("US", "gb", "KR", "DE", "SE", "", "XX", "T1", "null", "noCf"):
        assert out[c] == 403, c
    for c in ("MY", "sg", "JP", "AU"):
        assert out[c] == 200, c
