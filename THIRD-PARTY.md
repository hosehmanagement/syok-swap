# Third party

- H3 engine (`h3_engine.py`, `h3_app.py`): `workflows/openjutsu/simple.api.json` and `mask.api.json` are verbatim copies of
  D3voz/openjutsu's workflow exports (commit 7fe118b), and the graph patching in `h3_engine.py` is a Python port of openjutsu's
  `lib/simple.ts` and `lib/clips.ts`. MIT, Copyright (c) 2026 D3voz; the licence text is in `workflows/openjutsu/LICENSE`.
- The H3 engine installs at build time (not stored here): ComfyUI (GPL-3.0) and the custom-node packs listed in `h3_app.py`
  (check each pack's own licence), and loads MiniMax H3, Qwen3-VL and SAM 3.1 weights whose licences are separate. MiniMax
  H3's licence has regional restrictions (see README); check it before any commercial use.
- Ideas only (no code copied) from abdullayevziyodulla/open-genjutsu (MIT): delivering the original soundtrack untouched
  (`swapkit.mux_original_audio`).
