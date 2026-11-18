# Making swaps look human, and handling more than one person / long clips / consent

Status key: **built** = code + tests in this repo (free, CPU). **to test** = needs a paid GPU run (cost stated first).

## 1. Less "AI-made" faces
Why: diffusion models denoise toward retouched averages, so pores, stubble and asymmetry vanish. Sharpening makes it worse.

1. **Better reference photo (biggest effect, free).** Use a reference with visible pores, freckles, stubble, flyaway hair, slight dust/lint, uneven colour. Neutral grey background, even soft light, no beauty filter. A smooth reference gives a smooth result. *(guideline)*
2. **Character sheet on grey.** Front, 3/4 and side views on one flat mid-grey background; same outfit in all. Grey keeps colour spill off the face. *(guideline; to test as multi-reference)*
3. **Film-texture post-pass (built):** `realism.py` adds luminance-only grain and fine local contrast, mostly inside the person mask, original background untouched, audio kept. `realism.process_video(src, out, mask_video, grain=3, detail=0.35)`.
4. **Relight LoRA (to test):** `render_job` already accepts `--use_relighting_lora`; helps skin catch light instead of glowing. Needs one H100 run (~$0.55 like earlier tests).
5. **Skin-texture/face-restore upscaler (to test, only if 3 is not enough):** check the licence first (many are non-commercial).

## 2. More than one person
Wan-Animate replaces one character per pass. Plan: one SAM2 mask per person, one pass per person, feeding each output in as the next pass's source. Cost grows linearly (about $0.55 per person for a short clip). *(to test; not built)*

## 3. Clips over 30 seconds (built)
`longclip.plan_chunks(duration, fps)` splits into frame-aligned chunks of at most 24 s with 1 s overlap; `longclip.stitch(videos, overlap_frames, out)` crossfades the overlaps. Run each chunk as its own job, then stitch and attach the original audio. Cost is roughly proportional to length.

## 4. Consent for real people (built)
`consent.py`: a real face needs adult + own face or written permission + no impersonation + no deception/harm + AI-label OK, plus a name and contact. The record is tied to the image's SHA-256, expires (90 days), and can be revoked. Call `consent.verify(record, image)` before ANY job. Fictional faces are recorded as `kind="fictional"`. Not yet wired into the gateway.
