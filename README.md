# Syok Swap

Replace the person in a video with a character from **one photo**. Open, self-hosted on [Modal](https://modal.com), public repo.

- **Engine:** MiniMax H3 through ComfyUI, using openjutsu's Simple swap and Check mask workflows (`h3_engine.py`,
  `h3_app.py`, `h3_api.py`). Modal app name: `syok-swap-h3`. Details: `docs/H3.md`.
- **Fix faces (optional, opt-in, paid):** openjutsu's Fix faces on the finished swap. It uses the InsightFace `buffalo_l`
  face model, which is for non-commercial research use only (and AGPL-3.0 YOLO weights), so it never runs by itself.
- **Licence territory:** MiniMax H3's licence excludes the US, the EU, the UK and South Korea. The web API refuses those
  countries (fail closed) on the router's `X-Syok-Country` header plus a tick box; this is a safeguard, not legal
  compliance. See `docs/H3_TERRITORY.md`.

## Use

```
pip install modal && modal token new
modal run h3_app.py::fetch                                   # once, about 51 GB of model files into a Modal Volume
modal run h3_app.py::new   --video clip.mp4 --photo face.png --consent yes --length 6
modal run h3_app.py::check --job <id printed above>          # small GPU: mask preview (cyan = regenerated)
modal run h3_app.py::swap  --job <id>                        # big GPU, the paid step
modal run h3_app.py::fixfaces --job <id> --mode photo         # optional, paid: Fix faces on the finished swap
```

Or use the GitHub Actions workflows in your own fork: **deploy Syok Swap H3 engine to my Modal** (`deploy-h3.yml`) and
**test Syok Swap H3 engine on my Modal** (`test-h3.yml`).

`--consent yes` is required: only use photos and videos of people who agreed, or fictional characters. No public figures.

## Cost (Modal, per second billing)

The mask check runs on an L4, the swap on an L40S by default. First live test (21 Nov 2026, 6 s clip at 480p): mask check
about 17 minutes including a cold start, swap about 3 minutes. No money figure has been measured; see `docs/H3.md`.

## Free CI

Tests run on GitHub Actions. Public repos get free standard-runner minutes; nothing here runs a GPU on GitHub.

## Licences

Code: Apache-2.0. Third-party parts and model licences: see `THIRD-PARTY.md`.
