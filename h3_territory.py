"""Territory gate and UI strings for a possible MiniMax H3 swap engine. Pure Python, no network, nothing touches a GPU.

Used by the H3 engine's web API (h3_api.py, behind h3_app.web).

The H3 Community License excludes the US, the EU, the UK and South Korea. This module is a SAFEGUARD written with the
licence's section V.5 in mind (see docs/H3_TERRITORY.md). It does NOT make anything legally compliant: the country comes
from the visitor's IP address (Cloudflare's request.cf.country), which a VPN or proxy defeats, and it needs a lawyer's review.

Rules:
  * BLOCKED = US, GB, KR and the 27 EU member states.
  * An unknown, missing or malformed country code is BLOCKED (fail closed). Cloudflare's own "XX" (unknown) and "T1" (Tor)
    are therefore blocked too, because they are not real ISO country codes of an allowed place.
  * The user must also tick TERRITORY_CONSENT_TEXT; it is sent as `h3_territory=yes`, the same way the existing
    `consent=yes` form field works, and can be kept as a consent.py-style JSON record (make_territory_record / consent.save).
"""
import hashlib
import re
import time

# The 27 EU member states (ISO 3166-1 alpha-2). Greece is "GR" in ISO (the EU itself sometimes writes "EL").
EU_MEMBER_CODES = frozenset((
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
    "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
))
OTHER_BLOCKED_CODES = frozenset(("US", "GB", "KR"))
BLOCKED_COUNTRY_CODES = EU_MEMBER_CODES | OTHER_BLOCKED_CODES

ALLOWED = "allowed"
BLOCKED = "blocked"

_CODE = re.compile(r"^[A-Z]{2}$")
# Two-letter values that are not a country (Cloudflare: XX = unknown, T1 = Tor exit). Always blocked.
NOT_A_COUNTRY = frozenset(("XX", "T1", "ZZ", "EU", "AP"))

# ---- UI strings (exact text; tests pin them so a change is deliberate) ----
ENGINE_LABEL = "MiniMax H3"
TERRITORY_CONSENT_TEXT = "I am outside the US, EU, UK and South Korea"
TERRITORY_CONSENT_FIELD = "h3_territory"
LICENCE_NOTICE = (
    "This engine uses MiniMax H3 under the H3 Community License, which does not allow use in the US, "
    "the European Union, the United Kingdom or South Korea. It is not offered there."
)
BLOCKED_MESSAGE = (
    "MiniMax H3 is not available in your region. Its licence does not allow use in the US, the EU, the UK or "
    "South Korea, and we could not confirm you are outside them."
)


def check_country(code):
    """Return {"status": "allowed"|"blocked", "country": <normalised code or "">, "reason": <plain text>}.
    Input is the two-letter country code (e.g. Cloudflare's request.cf.country). Case and surrounding spaces are ignored.
    Anything that is not a recognisable allowed two-letter code is blocked."""
    if code is None:
        return {"status": BLOCKED, "country": "", "reason": "No country code: blocked because the location is unknown."}
    if not isinstance(code, str):
        return {"status": BLOCKED, "country": "", "reason": "Country code is not text: blocked because the location is unknown."}
    c = code.strip().upper()
    if not c:
        return {"status": BLOCKED, "country": "", "reason": "Empty country code: blocked because the location is unknown."}
    if not _CODE.match(c):
        return {"status": BLOCKED, "country": c, "reason": f"'{c}' is not a two-letter country code: blocked because the location is unknown."}
    if c in NOT_A_COUNTRY:
        return {"status": BLOCKED, "country": c, "reason": f"'{c}' is not a country (unknown location or anonymiser): blocked."}
    if c in EU_MEMBER_CODES:
        return {"status": BLOCKED, "country": c, "reason": f"{c} is an EU member state, excluded by the H3 Community License."}
    if c in OTHER_BLOCKED_CODES:
        return {"status": BLOCKED, "country": c, "reason": f"{c} is excluded by the H3 Community License."}
    return {"status": ALLOWED, "country": c, "reason": f"{c} is not in the H3 excluded list."}


def is_allowed(code):
    return check_country(code)["status"] == ALLOWED


def territory_consent_given(value):
    """The tick box form field: only the exact value "yes" counts, the same rule as the existing `consent=yes` field."""
    return value == "yes"


def text_sha256(text=TERRITORY_CONSENT_TEXT):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_territory_record(country_code, ticked, now=None):
    """A consent.py-style record (plain dict, saved with consent.save / read with consent.load) that the user ticked the box,
    binding it to the exact wording (SHA-256 of TERRITORY_CONSENT_TEXT) and the country the IP check saw.
    Raises PermissionError when the box was not ticked or the country is blocked; nothing is recorded then."""
    if not territory_consent_given(ticked):
        raise PermissionError("Tick '" + TERRITORY_CONSENT_TEXT + "' to use " + ENGINE_LABEL + ".")
    result = check_country(country_code)
    if result["status"] != ALLOWED:
        raise PermissionError(BLOCKED_MESSAGE)
    now = time.time() if now is None else now
    return {"kind": "h3_territory", "engine": ENGINE_LABEL, "text": TERRITORY_CONSENT_TEXT,
            "text_sha256": text_sha256(), "attest": {"outside_excluded_territories": True},
            "ip_country": result["country"], "created": now}


def verify_territory_record(record):
    """Raise PermissionError unless the record is for the current wording and an allowed country."""
    if record.get("kind") != "h3_territory" or record.get("attest", {}).get("outside_excluded_territories") is not True:
        raise PermissionError("No territory confirmation on record.")
    if record.get("text_sha256") != text_sha256():
        raise PermissionError("The territory wording changed; ask the user to tick the box again.")
    if not is_allowed(record.get("ip_country")):
        raise PermissionError(BLOCKED_MESSAGE)
    return True
