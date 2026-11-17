# First end-to-end result (2026-11-18)

Everything below used **fictional faces only**: the test clip is a fictional red-haired woman, the replacement
character is a fictional man generated with SDXL (`modal run app.py::face`).

## What ran

| Step | Command | GPU | Wall time | Rough cost* |
|---|---|---|---|---|
| Weights (one time, ~50 GB) | `modal run app.py::fetch` | none (CPU) | a few minutes | cents |
| Fictional face | `modal run app.py::face` | L40S | ~1 min | ~$0.03 |
| Mask preview | `modal run app.py::preview --video test_wide.mp4 --image fictional_man.png --consent yes` | L4/A10G/L40S fallback | ~6-7 min | ~$0.10 |
| Swap render | `modal run app.py::render --job <id>` | H100 | ~5 min incl. cold start and 25 s model load | ~$0.35 |

*Estimates from Modal's list prices (H100 about $3.95/h, L4 about $0.80/h). Check the Modal billing page for exact figures.

Input: 3.88 s, 704x704, with audio. Output: same length and size, original AAC audio restored.
The before/after contact sheet is made with `python tools/sbs.py before.mp4 after.mp4 sheet.png`.

## What it looks like

* The woman is replaced by the man; head turns and expressions from the source are followed.
* The blue-grey background outside the mask is kept from the original frames.
* Original soundtrack is attached unchanged.

## Known weaknesses (not hidden)

* A faint, slightly different-tinted rectangle can be seen around the character in some frames (the mask is Wan's
  blocky body box; the feather is 7 px). Next step: larger feather and colour matching at the seam.
* Background colours from the reference photo can leak in (thin coloured strip near the shoulder).
* One person, one section of at most 30 s. No multi-person, no hat/face second pass yet.
* The test clip is a tight portrait; a full-body clip with real motion still has to be tested.

## Bugs found and fixed on the way (each one is a commit)

1. SAM2 would not build (needs nvcc) -> use its pure-python package at a pinned commit, plus its `sam2_configs`.
2. SAM2's CUDA connected-components op missing -> tested CPU OpenCV fallback, patched in at image build.
3. **Mask polarity guess was wrong.** Wan writes the person as white. The code inverted any mask covering over half the
   frame, which is exactly what a close-up portrait does. Now Wan's polarity is kept as written; regression test added.
4. Packages Wan imports but does not list: `einops`, `regex`, `safetensors`, `librosa`, `scipy`, `moviepy<2` ->
   installed, and the image build now checks the imports so a miss fails on a cheap CPU build, not a paid GPU run.
5. Wan asserts FlashAttention-2 -> official prebuilt wheel for torch 2.6 / CUDA 12 / Python 3.11.
6. `libpng error: bad parameters to zlib` when `generate.py` reads `src_ref.png` -> store the reference as BMP bytes under
   that name (`cv2.imread` picks the decoder from content).
7. L4 GPUs were scarce (9-13 min queue) -> GPU fallback list `["L4", "A10G", "L40S"]`; the preview started in 5 seconds.
