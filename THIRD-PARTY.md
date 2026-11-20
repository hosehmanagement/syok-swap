# Third party

- Wan2.2 (Wan-Video/Wan2.2) and Wan2.2-Animate-14B weights: Apache-2.0. Downloaded at build/fetch time, not stored here.
- SAM 2, ViTPose, YOLOv10 come inside Wan's `process_checkpoint`; check their licences before shipping a product.
- Ideas only (no code copied) from D3voz/openjutsu (MIT) and abdullayevziyodulla/open-genjutsu (MIT), for the Wan engine.
- H3 engine (`h3_engine.py`, `h3_app.py`): `workflows/openjutsu/simple.api.json` and `mask.api.json` are verbatim copies of
  D3voz/openjutsu's workflow exports (commit 7fe118b), and the graph patching in `h3_engine.py` is a Python port of openjutsu's
  `lib/simple.ts` and `lib/clips.ts`. MIT, Copyright (c) 2026 D3voz; the licence text is in `workflows/openjutsu/LICENSE`.
- The H3 engine installs at build time (not stored here): ComfyUI (GPL-3.0) and the custom-node packs listed in `h3_app.py`
  (check each pack's own licence), and loads MiniMax H3, Qwen3-VL and SAM 3.1 weights whose licences are separate. MiniMax
  H3's licence has regional restrictions (see README); check it before any commercial use.
- Magic Hour (magichour.ai) "Face swap video" API: an external paid service, used only by the optional face-lock step, only
  when the user asks for it, with the user's own API key and credits. Nothing from Magic Hour is stored in this repo;
  commercial use of its output is covered by Magic Hour's paid plans (check their terms).
- `tests/wan_mask_upstream.py`: a verbatim copy of three small mask functions from Wan2.2's `wan/modules/animate/preprocess/utils.py` (Apache-2.0, Copyright The Alibaba Wan Team Authors), used by the tests only.
