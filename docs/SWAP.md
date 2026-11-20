# Syok Swap on your own Modal (swap + texture slider, one flow)

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
Limits: video up to 60 MB, photo up to 8 MB, section up to 30 s. Clothes come from the reference photo.
