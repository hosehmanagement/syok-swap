# Syok Swap on your own Modal (swap + texture slider + optional face-lock, one flow)

One Modal app (`app.py`, name `syok-swap`). Modal bills YOUR account. Idle costs nothing. Results are removed after 3 days.
Every swap result has the texture strength slider (0 to 3, starts at 2.0) built in: there is no separate texture step.

## One-time setup (about 10 minutes)
1. **Fork** this repo on GitHub. Do the next steps in your fork.
2. In Modal (modal.com) > Settings > **API Tokens** > create a token (`ak-...` / `as-...`).
3. In your fork: Settings > Secrets and variables > Actions > add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.
4. Fork > Actions > **deploy Syok Swap to my Modal** > Run workflow. The run summary shows `Endpoint address: https://...modal.run`.
5. In Modal > Settings > **Proxy Auth Tokens** > create one (`wk-...` / `ws-...`).
6. In Syok (Creative Studio > Recast > Syok Swap) paste the endpoint address and the proxy token. Syok keeps them in your own Cloudflare Worker.
7. First use only: the card offers "Download the model" (about 50 GB into your Modal volume, one time).

The endpoint refuses every request without your proxy token (`requires_proxy_auth=True`).

## Flow (what the Syok card does)
1. `POST /swap` (video, photo, consent=yes, resolution, margin, start, length, optional mask_shape) returns the job and an approximate cost estimate. A cheap GPU builds the mask preview.
2. You check the mask (`GET /mask/{job}`), then confirm. `POST /generate/{job}` runs the paid H100 swap (once per job).
3. Face tracking starts automatically on the swapped clip. Then the slider works: `GET /preview/{job}?frame=N&strength=S` (one frame),
   `POST /render/{job}?strength=S` (full render), `GET /result/{job}`. `GET /swapped/{job}` is the plain swap without texture.
4. The realism slider takes its skin grain from the face in the photo; when the face shows too little plain skin (glasses,
   or a tiny face in a full-body photo) it uses plain skin anywhere in the photo. `POST /swap` also takes an optional `skin`
   close-up, and refuses (422, nothing charged) a photo that cannot feed the slider. `POST /retexture/{job}` (optional `skin`)
   prepares the slider again on a finished swap, CPU only.
5. `GET /status/{job}`, `DELETE /job/{job}`, `GET /health`, `POST /setup`.

## Mask shape: silhouette (default) or box

**What was wrong (reported 20 Nov 2026 from a live mask check):** the red "will be replaced" area was a wide rectangle around
the person (the hedge on both sides of the body was inside it) plus another rectangle over the head. Wan redraws everything
inside the mask, so that background changed in the result.

**Why:** Wan's `preprocess_data.py` (pinned commit `1ea34ff`) makes the mask in three steps: SAM2 finds the person from body
points (`process_pipepline.py` `get_mask`), the mask is dilated by the margin (`utils.get_mask_body_img`, `--iterations/--k`),
then `utils.get_aug_mask` splits the person's bounding box into a `--w_len` x `--h_len` grid and fills every cell the person
touches. Both default to **1**, so there is one cell and the **whole bounding box** is filled. We never passed them. On top of
that, our head fix (`swapkit.head_boxes`) added a rectangle 2.8 face-widths wide above the mask.

**What changed:**
- `mask_shape=silhouette` (default): `--w_len 10 --h_len 20` (the grid `get_aug_mask` itself defaults to), so the mask follows
  the person's outline in cells of 1/10 of their width and 1/20 of their height. The margin still dilates the outline first,
  so the person is never cut: every SAM2 pixel stays inside.
- The head is added as an ellipse around the face and hair (2.4 face-widths wide, from 1 face-height above the face box to just
  below the chin) plus the detected face box itself, so a detected face can never be outside the mask, and frames where the
  face is not detected use the smoothed position from the frames where it is (same as before).
- `src_bg.mp4` is blacked out wherever the final mask is, exactly as Wan builds it (`frame * (1 - mask)`), and keeps the same
  frames as `src_mask.mp4`.
- A small build-time patch (`patch_wan_mask.py`) keeps Wan's grid from crashing when the person is tinier than the grid
  (under 10 px wide or 20 px tall in a frame); it changes nothing for anything larger. The image build fails if Wan's file
  ever changes, so this can never break silently in a paid run.
- `mask_shape=box` on `POST /swap` (or `--mask-shape box` on `modal run app.py::preview`) gives the old behaviour.
- `status.mask.stats` now has `mask_shape` and `box_mean`: the share of the picture the old box mask would have covered on the
  same clip, next to `mean` (what this mask covers), so the card can show "covers X% instead of Y%". `status.mask.head.coverage`
  has the details.

**What is NOT measured:** this has only been tested on synthetic frames on a CPU (a moving ellipse "person" on a noisy green
"hedge", with Wan's own grid code). Nobody has run it on a GPU with real footage yet, so it is unknown how SAM2's real outline,
hair, loose clothes or a held object come out, or whether the tighter mask lets the original body shape show through ("shape
leakage", which Wan's own guide warns can happen with a finer mask). **Always look at the red preview (about $0.10 at 480p)
before paying for the swap.** If the red area cuts into the person, use margin `loose`, or `mask_shape=box`.

## Face-lock (optional, uses YOUR Magic Hour account)
Wan2.2-Animate redraws the whole person from one photo, so the face is close but not exactly the person in the photo.
Face-lock is an optional second pass after the swap: Magic Hour's "Face swap video" API puts the exact face from the photo
onto the swapped clip. Then the realism slider is prepared on the face-locked clip as usual.

- **Only when you tick the box.** `POST /swap` with `facelock=yes` (and optionally up to 5 `face` photos of the same person,
  see "Best-photo pick" below; without them the main photo is used). Nothing is sent to Magic Hour otherwise.
- After the swap, `status.facelock.state` is `waiting` and the slider is not prepared yet (the clip is about to change).
  `POST /facelock/{job}` with the header `X-Magic-Hour-Key: <your key>` (optional `face` files, which replace the earlier pick)
  sends the swapped clip and the chosen face photo to Magic Hour and puts the result back in place.
  `POST /facelock/{job}/skip` uses the swap as it is.
- **Credits come from your own Magic Hour account** (magichour.ai > Developer for the key). Commercial use is covered by Magic
  Hour's paid plans. `status.facelock.credits` shows what Magic Hour charged; Magic Hour refunds a job that fails on their side.
  The job is created once and never retried automatically, so a retry is always your click.
- Your key is used for that one job only: it is not saved in the job folder, the status or any log.
- **Free plan:** Magic Hour limits face swap videos to 576 px on the longest edge. The result is scaled back to the swapped
  clip's size and `status.facelock.note` says so; a paid plan keeps more detail.
- **Photo:** Magic Hour asks for a face area of at least 512 x 512 px. A face under about 256 px wide does not block anything,
  but `status.facelock.warning` suggests a sharper head-and-shoulders close-up (also when every photo you added is under
  256 px, or none of them shows a face; then the main photo is used).
- If face-lock fails, the swapped clip is untouched and the slider still works on it; try again or skip. The swap as Wan made
  it is kept as `src_plain.mp4` in the job folder (removed with the job after 3 days).
- States: `idle`, `waiting`, `queued`, `working`, `done`, `error`, `skipped` (plus `message`, `credits`, `warning`, `note`, `pick`).

### Best-photo pick (picker.py)
Magic Hour copies the face from ONE photo. You may add several (a full-body shot, a side profile, a close-up...) and the
code picks the one it can copy best.

- The multipart field `face` may repeat **1 to 5 times** on `POST /swap` and `POST /facelock/{job}`: png/jpg/webp, each up to
  8 MB, **all together up to 12 MB** (else 400 with a plain message, nothing stored). Empty file parts are ignored.
  `GET /health` reports `limits.faces` (5) and `limits.faces_mb` (12).
- The **main photo is always one more candidate**. Numbering: your added photos in upload order, then the main photo last.
  One added photo is simply a 2-way pick (that photo or the main photo).
- The winner is stored as `face.<ext>` (a copy of the main photo when it wins); the other candidates are deleted, never kept.
  `status.facelock.pick` = `{"chosen": N, "of": M, "reason": "Using photo 2: front-facing and about 640 px wide.",
  "scores": [{"n", "face_px", "front", "sharp", "ok", "note", "score"}]}` (N is 1-based; `ok` = the face is neither too dark
  nor too bright; `note` is a short plain-English description; `score` is the number below).
- If **no photo has a face**, nothing is blocked: the main photo is used, `pick.chosen` is the main photo and the warning says so.
- How it scores, from the largest face YuNet finds in each photo (exact formulas in picker.py's docstring):
  - `face_px` = face box width; `S = min(face_px, 512) / 512`.
  - `front` (0..1) from the 5 landmarks: nose and mouth centred between the eyes (and eyes level, a small share); below 0.35
    it counts as a profile.
  - `sharp` (0..1) = variance of the Laplacian of the face resized to 128 x 128, divided by the crop's own variance (floor 100),
    divided by 0.15, capped at 1.
  - `E` = 1 when the face's mean brightness is 60-200, less outside (never below 0.5); `M` = 0.6 when another face at least
    40% as wide is in the photo (which person to copy would be ambiguous).
  - `score = S^1.5 x front x (0.4 + 0.6 x sharp) x E x M`. A profile never beats a photo that is not a profile; equal scores
    go to the larger face, then to the earlier photo. Photos without a face are never chosen.
## Collage reference (EXPERIMENTAL, off by default)
Wan2.2-Animate generates the new person from ONE reference image. In a full-body photo the face is small, so this option gives
Wan one image that holds the full-body photo AND a sharp head-and-shoulders close-up of the same person, side by side.

**Nobody has measured yet whether this gives a better likeness.** It is an experiment to test. Wan may also misread the
collage and draw two people, or a split image. Leave it off unless you are testing it.

- `POST /swap` with `collage=yes` (also `1` or `true`; anything else, or no field, means off) and at least one `face` photo.
  With `collage` on and no `face` photo: 422 `{"error": "Collage needs a face close-up."}`, nothing stored, nothing charged.
  `consent=yes` is still required.
- The close-up is the face photo the best-photo pick chose. When your main photo wins that pick, the best of the photos you
  ADDED is used instead (same scoring), so the main photo never appears twice.
- `collage.py` builds `ref_collage.png` at upload (CPU, no GPU): main photo on the left, the close-up cropped to head and
  shoulders (YuNet, the same detector as the pick; a top-centre crop when no face is found) on the right, both scaled to the
  same height without stretching, a small gap, background colour taken from the main photo's border, transparent parts
  flattened onto it, EXIF orientation applied, at most 1536 px on the longer side. The same photos always give the same image.
- Only the generation uses the collage (it is the reference the mask step hands to Wan). The realism slider, the photo check
  before paying and face-lock keep using your ORIGINAL main photo, exactly as without the collage.
- `meta.json` / `status.request` get `"collage": true` only when it is on (a job without it is exactly as before);
  `status.collage` = `{"face": N, "of": M}` says which added photo was used. `GET /health` reports `"collage": true`.

Limits: video up to 60 MB, photo up to 8 MB, section up to 30 s. Clothes come from the reference photo.
