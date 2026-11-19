# Texture pass endpoint (strength slider) on your own Modal

A small CPU-only Modal app (`texture_app.py`). It needs no GPU and no model weights. Modal bills YOUR account, a few cents at most per clip;
when nobody is using it, it costs nothing. Results are removed after 3 days.

## One-time setup (about 5 minutes)
1. **Fork** this repo on GitHub (button top right). Do the next steps in your fork.
2. In Modal (modal.com) > Settings > **API Tokens** > create a token. Copy the Token ID (`ak-...`) and Token Secret (`as-...`).
3. In your fork: Settings > Secrets and variables > Actions > New repository secret. Add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.
4. Fork > Actions > **deploy texture pass to my Modal** > Run workflow. When it ends, the run summary shows `Endpoint address: https://...modal.run`.
5. In Modal > Settings > **Proxy Auth Tokens** > create one. Copy its Token ID (`wk-...`) and Secret (`ws-...`).
6. In Syok (Creative Studio > Recast > Texture pass) paste the endpoint address and the proxy token. Syok keeps them in your own Cloudflare Worker.

The endpoint refuses every request that does not carry your proxy token (`requires_proxy_auth=True`).

## API (what the Syok card calls)
`GET /health` | `POST /prepare` (multipart: video, photo, consent=yes) | `GET /status/{job}` | `GET /preview/{job}?frame=N&strength=S` |
`POST /render/{job}?strength=S` | `GET /result/{job}` | `DELETE /job/{job}`. Strength is 0 to 3; the card starts the slider at 2.0.
Limits: video up to 60 MB and 30 s, photo up to 8 MB.
