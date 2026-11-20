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
| Margin presets tight / normal / loose | Openjutsu | maps to Wan's mask dilation (`--iterations/--k`) |
| Work on a section (start, length) of the source | Openjutsu | `--start --length` |
| Keep everything outside the swapped person | Openjutsu | generated person is pasted over the **original frames at original size** (feathered) |

**Texture slider (0 to 3, default 2.0) is part of every swap**, see `docs/SWAP.md` for the one-flow web endpoint and deploy.

**Optional face-lock** after the swap puts the exact face from the photo on the swapped clip with Magic Hour's face swap API,
on the user's own Magic Hour account (BYOK; only when the user ticks the box), see `docs/SWAP.md`.

**Collage reference (EXPERIMENTAL, off by default):** `collage=yes` gives Wan one image with the full-body photo and a face
close-up side by side, instead of the full-body photo alone. Nobody has measured yet whether it helps the likeness, and Wan may
draw two people or a split image. The slider and face-lock still use the original photo. See `docs/SWAP.md`.

Not done yet: multi-person, hat fix, long clips above 30 s.

## Use

```
pip install modal && modal token new
modal run app.py::fetch                                      # once, ~50 GB of weights into a Modal volume
modal run app.py::preview --video clip.mp4 --image face.png --consent yes
modal run app.py::render  --job <id printed above>
```

`--consent yes` is required: only use photos and videos of people who agreed, or fictional characters. No public figures.

## Cost (Modal, per second billing)

Preview uses an L4, render uses an H100. Measured numbers are in `docs/RESULTS.md`.

## Free CI

Tests run on GitHub Actions. Public repos get free standard-runner minutes; nothing here runs a GPU on GitHub.

## Licences

Code: Apache-2.0. Wan2.2: Apache-2.0 (see `THIRD-PARTY.md`).
