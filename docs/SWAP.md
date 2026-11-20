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
1. `POST /swap` (video, photo, consent=yes, resolution, margin, start, length) returns the job and an approximate cost estimate. A cheap GPU builds the mask preview.
2. You check the mask (`GET /mask/{job}`), then confirm. `POST /generate/{job}` runs the paid H100 swap (once per job).
3. Face tracking starts automatically on the swapped clip. Then the slider works: `GET /preview/{job}?frame=N&strength=S` (one frame),
   `POST /render/{job}?strength=S` (full render), `GET /result/{job}`. `GET /swapped/{job}` is the plain swap without texture.
4. The realism slider takes its skin grain from the face in the photo; when the face shows too little plain skin (glasses,
   or a tiny face in a full-body photo) it uses plain skin anywhere in the photo. `POST /swap` also takes an optional `skin`
   close-up, and refuses (422, nothing charged) a photo that cannot feed the slider. `POST /retexture/{job}` (optional `skin`)
   prepares the slider again on a finished swap, CPU only.
5. `GET /status/{job}`, `DELETE /job/{job}`, `GET /health`, `POST /setup`.

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
Limits: video up to 60 MB, photo up to 8 MB, section up to 30 s. Clothes come from the reference photo.
