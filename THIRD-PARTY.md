# Third party

- H3 engine (`h3_engine.py`, `h3_app.py`): `workflows/openjutsu/simple.api.json`, `mask.api.json` and `face.api.json` are
  verbatim copies of D3voz/openjutsu's workflow exports (commit 7fe118b, 7fe118bb5a61fcfac0148e1bfea3dd29884dbf26), and the
  graph patching in `h3_engine.py` is a Python port of openjutsu's `lib/simple.ts`, `lib/clips.ts` and `lib/face.ts`. MIT, Copyright (c) 2026 D3voz; the licence text is in `workflows/openjutsu/LICENSE`.
- The H3 engine installs at build time (not stored here): ComfyUI (GPL-3.0) and the custom-node packs listed in `h3_app.py`
  (check each pack's own licence), and loads MiniMax H3, Qwen3-VL and SAM 3.1 weights whose licences are separate. MiniMax
  H3's licence has regional restrictions (see README); check it before any commercial use.
- Ideas only (no code copied) from abdullayevziyodulla/open-genjutsu (MIT): delivering the original soundtrack untouched
  (`swapkit.mux_original_audio`).
- Fix faces (optional, opt-in only; `face.api.json`, `h3_engine.prune_face`, `h3_app.faces_job`), installed or downloaded at
  build / fetch / first-run time, not stored here:
  - Node packs: Carasibana/ComfyUI-H3-FaceRefine at 79a97ce (MIT) and ltdrdata/ComfyUI-Impact-Pack at 5bed9d5 (GPL-3.0; only
    its SAMLoader is used); D3voz/comfyui-h3-oneclick-swap (MIT, already installed) supplies H3 Face Fix Chunks / Join.
  - InsightFace (Python package 0.7.3; the code is MIT) and its `buffalo_l` face model, which InsightFace downloads into the
    models Volume on the first Fix faces run. **`buffalo_l` is licensed for non-commercial research use only.**
  - `face_yolov8m.pt` and `person_yolov8m-seg.pt` from Bingsu/adetailer (revision 53cc19d): Ultralytics YOLO weights,
    **AGPL-3.0** (openjutsu's README; the Hugging Face card says Apache-2.0 for the repo).
  - `sam_vit_b_01ec64.pth` from Meta's segment-anything (Apache-2.0), downloaded from the official
    `dl.fbaipublicfiles.com` URL and checked against its sha256.
  - Because of the `buffalo_l` and AGPL terms, Fix faces is never automatic: the user starts it, and the web card says so.
