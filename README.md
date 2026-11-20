# Syok Swap

Replace the person in a video with a character from **one photo**. Open, self-hosted on [Modal](https://modal.com), public repo.

- **Engine:** [Wan2.2-Animate-14B](https://github.com/Wan-Video/Wan2.2) in replacement mode (Apache-2.0, weights and code, commercial use allowed).
- **No paid API, no MiniMax H3** (its licence excludes the US, EU, UK and Korea).
- Pinned to Wan2.2 commit `1ea34ff`.

## What we took from the two reference projects (re-implemented, no code copied)

| Idea | From | Here |
|---|---|---|
| Preview the mask before paying for generation | open-genjutsu | `preview` step on a small GPU, red overlay + coverage stats, refuses empty masks |
| Deliver the **original soundtrack** untouched | open-genjutsu | `mux_original_audio` (refuses to trim/stretch more than 0.5 s) |
| Margin presets tight / normal / loose | Openjutsu | maps to Wan's mask dilation (`--iterations/--k`); the mask shape (silhouette/box) maps to Wan's grid (`--w_len/--h_len`) |
| Work on a section (start, length) of the source | Openjutsu | `--start --length` |
| Keep everything outside the swapped person | Openjutsu | generated person is pasted over the **original frames at original size** (feathered) |

**Texture slider (0 to 3, default 2.0) is part of every swap**, see `docs/SWAP.md` for the one-flow web endpoint and deploy.

**Optional face-lock** after the swap puts the exact face from the photo on the swapped clip with Magic Hour's face swap API,
on the user's own Magic Hour account (BYOK; only when the user ticks the box), see `docs/SWAP.md`.

**Collage reference (EXPERIMENTAL, off by default):** `collage=yes` gives Wan one image with the full-body photo and a face
close-up side by side, instead of the full-body photo alone. Nobody has measured yet whether it helps the likeness, and Wan may
draw two people or a split image. The slider and face-lock still use the original photo. See `docs/SWAP.md`.

**Mask follows the person (silhouette, default since 20 Nov 2026; NOT yet checked on a real GPU run):** Wan's mask step used
to fill the person's whole bounding box (its `--w_len/--h_len` grid defaults to 1 x 1), plus a wide rectangle over the head,
so background beside the person (a hedge, a wall) was inside the red area and got redrawn. Now Wan's own grid is 10 x 20 cells,
so the mask follows the person's outline in small blocks, and the head is added as an ellipse plus the detected face box
instead of a rectangle. `mask_shape=box` keeps the old behaviour. On synthetic test frames the mask covers about a third less
of the picture with the head and body still fully inside; on real footage this is **not measured**: always look at the red
mask preview (about $0.10 at 480p) before paying for the swap. See `docs/SWAP.md`.

Not done yet: multi-person, hat fix, long clips above 30 s.

## Use

```
pip install modal && modal token new
modal run app.py::fetch                                      # once, ~50 GB of weights into a Modal volume
modal run app.py::preview --video clip.mp4 --image face.png --consent yes   # add --mask-shape box for the old box mask
modal run app.py::render  --job <id printed above>
```

`--consent yes` is required: only use photos and videos of people who agreed, or fictional characters. No public figures.

## Cost (Modal, per second billing)

Preview uses an L4, render uses an H100. Measured numbers are in `docs/RESULTS.md`.

## Free CI

Tests run on GitHub Actions. Public repos get free standard-runner minutes; nothing here runs a GPU on GitHub.

## Licences

Code: Apache-2.0. Wan2.2: Apache-2.0 (see `THIRD-PARTY.md`).
