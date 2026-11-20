# Third party

- Wan2.2 (Wan-Video/Wan2.2) and Wan2.2-Animate-14B weights: Apache-2.0. Downloaded at build/fetch time, not stored here.
- SAM 2, ViTPose, YOLOv10 come inside Wan's `process_checkpoint`; check their licences before shipping a product.
- Ideas only (no code copied) from D3voz/openjutsu (MIT) and abdullayevziyodulla/open-genjutsu (MIT).
- Magic Hour (magichour.ai) "Face swap video" API: an external paid service, used only by the optional face-lock step, only
  when the user asks for it, with the user's own API key and credits. Nothing from Magic Hour is stored in this repo;
  commercial use of its output is covered by Magic Hour's paid plans (check their terms).
- `tests/wan_mask_upstream.py`: a verbatim copy of three small mask functions from Wan2.2's `wan/modules/animate/preprocess/utils.py` (Apache-2.0, Copyright The Alibaba Wan Team Authors), used by the tests only.
