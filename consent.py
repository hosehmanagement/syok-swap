"""Consent gate for REAL people's faces. Nothing here touches a GPU.

A swap that uses a real person's face needs a signed-off record BEFORE any job starts. The record
binds the attestation to the exact image (SHA-256), so approving one photo never covers another.
Fictional/synthetic faces use kind="fictional" and need no person, but are still recorded.
"""
import hashlib
import json
import time
from pathlib import Path

REQUIRED_FOR_REAL = (
    "is_adult",              # the person shown is 18+
    "is_me_or_has_written_permission",
    "not_a_public_figure_impersonation",
    "no_deception_or_harm",  # not for fraud, harassment, fake endorsements, intimate content
    "may_be_labelled_ai",    # output may carry an AI-generated label
)
DEFAULT_VALID_DAYS = 90


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def make_record(image_path, kind, attest=None, who="", contact="", valid_days=DEFAULT_VALID_DAYS, now=None):
    """kind: 'fictional' or 'real'. For 'real' every item in REQUIRED_FOR_REAL must be True."""
    if kind not in ("fictional", "real"):
        raise ValueError("kind must be 'fictional' or 'real'.")
    attest = dict(attest or {})
    if kind == "real":
        missing = [k for k in REQUIRED_FOR_REAL if attest.get(k) is not True]
        if missing:
            raise PermissionError("Consent incomplete, not allowed to proceed: " + ", ".join(missing))
        if not who.strip() or not contact.strip():
            raise PermissionError("Consent needs the name of the person and a contact for them.")
    now = time.time() if now is None else now
    return {"kind": kind, "image_sha256": sha256_file(image_path), "attest": attest, "who": who,
            "contact": contact, "created": now, "expires": now + valid_days * 86400}


def verify(record, image_path, now=None):
    """Raise PermissionError unless the record covers exactly this image and is still valid."""
    now = time.time() if now is None else now
    if record.get("image_sha256") != sha256_file(image_path):
        raise PermissionError("Consent record is for a different image.")
    if now > record.get("expires", 0):
        raise PermissionError("Consent has expired; ask the person again.")
    if record.get("kind") == "real":
        missing = [k for k in REQUIRED_FOR_REAL if record.get("attest", {}).get(k) is not True]
        if missing:
            raise PermissionError("Consent incomplete: " + ", ".join(missing))
    return True


def save(record, path):
    Path(path).write_text(json.dumps(record, indent=2))


def load(path):
    return json.loads(Path(path).read_text())


def revoke(path):
    """A revoked record is expired immediately; the file is kept as an audit trail."""
    r = load(path)
    r["expires"] = 0
    r["revoked"] = True
    save(r, path)
