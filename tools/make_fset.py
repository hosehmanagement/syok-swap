"""Rebuild the fictional one-identity training set from the generated PNGs and run lora.dataset.prepare on it.

    PYTHONPATH=. python tools/make_fset.py fset_raw fset_raw2 prepared_fset

The 14 names below were picked by eye on 2026-11-18 as one consistent-looking identity (turned-head shots; the straight-on
shots and h04/h05/h10 came out as a different-looking man). The consent record is 'fictional' and is bound to exactly these
files, so changing any image invalidates it. Expected dataset_sha256 is checked at the end.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

from lora import dataset as D

PICK = {"g": ["g02", "g07", "g08", "g09", "g14"], "h": ["h01", "h02", "h03", "h06", "h07", "h08", "h09", "h11", "h12"]}
EXPECTED = "8512f209ef069562c8362ad748c2711db504c64f63a9608b0439becd0f316a96"

if __name__ == "__main__":
    raw1, raw2, out = (Path(a) for a in sys.argv[1:4])
    src = Path(tempfile.mkdtemp())
    for d, key in ((raw1, "g"), (raw2, "h")):
        for n in PICK[key]:
            shutil.copy(d / f"{n}.png", src / f"{n}.png")
    paths = sorted(src.glob("*.png"))
    rec = D.make_dataset_record(paths, "fictional")
    consent_path = out.parent / "fset_consent.json"
    consent_path.write_text(json.dumps(rec, indent=1))
    m = D.prepare(src, out, consent_path)
    print("kept", len(m["kept"]), "rejected", m["rejected"])
    print("digest", m["dataset_sha256"], "MATCH" if m["dataset_sha256"] == EXPECTED else "DIFFERENT")
