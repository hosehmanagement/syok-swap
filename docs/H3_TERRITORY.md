# MiniMax H3 territory safeguard (not wired in, not deployed)

The live engine is still **Wan2.2-Animate** (Apache-2.0). This page describes groundwork for a possible **MiniMax H3** engine.
Nothing here is deployed, and `app.py` / `swap_api.py` do not use it yet.

## Important: this is NOT legal compliance

- It is a **safeguard** written with **section V.5** of the H3 Community License in mind. Sections V.2, V.4 and IV.2 were also
  named as relevant. The section numbers come from the project owner's reading; this repo does not hold the licence text,
  so they were **not** checked against it here.
- The country check is **IP-based** (Cloudflare's `request.cf.country`). A **VPN, proxy or travel** gets around it or gets it wrong.
  The tick box is the user's own statement; it is not proof.
- Using these files does **not** make Syok Swap, Syok AI or anyone else compliant with the H3 Community License or any law.
- **A lawyer must review** the licence, this approach and the wording below before H3 is offered to anyone.

## What is here

| File | What it does |
|---|---|
| `h3_territory.py` | Blocked list (US, GB, KR + the 27 EU member states), `check_country(code)` -> allowed/blocked with a reason, the UI strings, and a consent record helper |
| `docs/h3-territory-worker.js` | Example Cloudflare Worker: reads `request.cf.country`, returns **403** for blocked or missing codes. **Example only, not deployed.** |
| `tests/test_h3_territory.py` | Every blocked code, allowed examples (MY, SG, JP, AU), lowercase, `None`, empty string, the worker list matches the Python list |

## Rules

1. Blocked: `US`, `GB`, `KR`, and `AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE`.
2. **Unknown or missing country = blocked** (fail closed). That includes `None`, empty text, anything not two letters, and
   Cloudflare's `XX` (unknown) and `T1` (Tor).
3. Case and spaces are ignored (`" my "` is read as `MY`).
4. Both checks are needed: the IP country must be allowed **and** the user must tick the box.

## UI strings (exact)

- Engine label shown on the UI: **MiniMax H3**
- Tick box (form field `h3_territory`, value `yes`, same rule as the existing `consent=yes`):
  **I am outside the US, EU, UK and South Korea**
- Licence notice: *This engine uses MiniMax H3 under the H3 Community License, which does not allow use in the US, the
  European Union, the United Kingdom or South Korea. It is not offered there.*
- Blocked message: *MiniMax H3 is not available in your region. Its licence does not allow use in the US, the EU, the UK or
  South Korea, and we could not confirm you are outside them. You can still use the standard engine.*

The tick can be kept like the existing consent records: `h3_territory.make_territory_record(country, "yes")` gives a plain dict
(bound to the exact wording by SHA-256, plus the IP country) that `consent.save` / `consent.load` store as JSON.
`verify_territory_record` refuses it if the wording later changes.

## Open points before any use

- Lawyer review of the licence sections and wording.
- Where the check runs: the Worker in front of the H3 route, and the same check again on the server (never trust the browser).
- Whether stricter location evidence is needed (for example billing country), and what to do for users who travel.
