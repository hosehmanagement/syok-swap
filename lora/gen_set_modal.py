"""Generate a FICTIONAL training set (one invented adult identity) with Z-Image Turbo on Modal. Cents, not dollars.

    modal run lora/gen_set_modal.py [--batch 2]  # writes PNGs to ./fset_raw/ and to the Modal volume (syok-lora-data:/fset/raw)

Why this exists: the free Hugging Face Spaces ran out of GPU quota, and Pollinations' free model draws smooth, watermarked faces.
Everything generated here is a made-up person, so it can pass lora.dataset's consent gate as "fictional".

The identity text is identical in every prompt; only angle, light and expression change, with the same seed, which is what
keeps the face stable. Consistency is checked by eye afterwards (text-only generation cannot guarantee it).
"""
from pathlib import Path

import modal

BASE = (
    "Raw unretouched close-up portrait photograph of one fictional adult man, about 38 years old, olive skin, "
    "short dark brown hair, heavy straight eyebrows, deep-set hazel eyes, many clearly visible brown freckles across "
    "nose and cheekbones, a small scar through the left eyebrow, light dark stubble. Dull matte skin with no glow or "
    "sheen, uneven skin tone, visible open pores on nose and cheeks, fine wrinkles at the eyes, subtle pigmentation "
    "patches, faint reddish blood-vessel veins on the cheeks and nose, tiny vellus hairs catching the light, blemishes "
    "and imperfections left as they are, no skin smoothing, no beauty filter. "
)
TAILS = [
    "Soft overcast window light from the left, natural shadows under brow and nose, neutral grey background, 85mm lens, sharp focus on the eyes, highly detailed skin texture, looking straight at camera, neutral expression.",
    "Soft overcast window light from the left, natural shadows under brow and nose, neutral grey background, 85mm lens, sharp focus on the eyes, highly detailed skin texture, head turned slightly to his left, looking at camera, neutral expression.",
    "Soft overcast window light from the left, natural shadows under brow and nose, neutral grey background, 85mm lens, sharp focus on the eyes, highly detailed skin texture, head turned slightly to his right, looking at camera, slight smile.",
    "Hard side light from the right, deep natural shadows, neutral grey background, 85mm lens, sharp focus on the eyes, highly detailed skin texture, looking straight at camera, neutral expression.",
    "Soft overcast window light from the left, neutral grey background, 85mm lens, sharp focus on the eyes, highly detailed skin texture, head tilted slightly down, eyes looking up at camera.",
    "Soft diffused light from above, neutral grey background, 85mm lens, sharp focus on the eyes, highly detailed skin texture, chin slightly raised, looking at camera, neutral expression.",
    "Warm late-afternoon window light from the right, neutral background, 85mm lens, sharp focus on the eyes, highly detailed skin texture, three-quarter view, looking at camera, relaxed expression.",
    "Cool flat daylight, neutral grey background, 100mm macro lens, sharp focus, highly detailed skin texture, looking straight at camera, eyes narrowed in a squint.",
    "Soft overcast window light from the left, neutral grey background, 85mm lens, sharp focus on the eyes, highly detailed skin texture, head turned slightly to his left, eyes looking sideways, serious expression.",
    "Soft window light from the front, neutral grey background, 85mm lens, sharp focus on the eyes, highly detailed skin texture, looking at camera, mouth slightly open as if about to speak.",
    "Dim indoor light from the left with a soft shadow on the right side of the face, neutral background, 85mm lens, sharp focus on the eyes, highly detailed skin texture, looking at camera, neutral expression.",
    "Soft overcast window light from the right, neutral grey background, 85mm lens, sharp focus on the eyes, highly detailed skin texture, three-quarter view facing his left, looking at camera, slight smile.",
    "Soft overcast window light from the left, neutral grey background, 85mm lens, sharp focus on the eyes, highly detailed skin texture, looking straight at camera, eyebrows raised slightly.",
    "Even studio light, neutral grey background, 85mm lens, sharp focus on the eyes, highly detailed skin texture, head turned slightly to his right, looking away, thoughtful expression.",
]

TAILS2 = [  # batch 2: turned heads (the identity Z-Image keeps stable) with different framing, background and light, so images are not near-duplicates
    "Wider framing with collar and shoulders visible, dark charcoal background, soft key light from the left, 85mm lens, sharp focus on the eyes, highly detailed skin texture, head turned slightly to his right, looking at camera, neutral expression.",
    "Tight crop from forehead to chin, outdoors in open shade, blurred green foliage background, 100mm lens, sharp focus on the eyes, highly detailed skin texture, head turned three-quarters to his left, looking at camera, slight smile.",
    "Wider framing showing neck and plain navy t-shirt collar, warm beige wall background, late-afternoon sun from the right, 85mm lens, sharp focus, highly detailed skin texture, head turned slightly to his left, looking past the camera.",
    "Tight macro crop on eyes, nose and cheeks, flat neutral light, 100mm macro lens, sharp focus, highly detailed skin texture, head turned slightly to his right, looking at camera.",
    "Wider framing, blurred office background, cool fluorescent light from above, 85mm lens, sharp focus on the eyes, highly detailed skin texture, head turned three-quarters to his right, looking at camera, neutral expression.",
    "Close portrait, dark moody background, single warm lamp light from the left leaving the right side of the face in soft shadow, 85mm lens, sharp focus on the eyes, highly detailed skin texture, head turned slightly to his left, serious expression.",
    "Wider framing with shoulders visible, white tiled wall background, soft bounce light from below the face, 85mm lens, sharp focus on the eyes, highly detailed skin texture, head tilted slightly, looking at camera, faint smile.",
    "Tight crop, overcast daylight outdoors on a street, blurred buildings behind, 85mm lens, sharp focus on the eyes, highly detailed skin texture, head turned three-quarters to his right, eyes looking down, calm expression.",
    "Wider framing, bookshelf background out of focus, soft window light from the right, 85mm lens, sharp focus on the eyes, highly detailed skin texture, head turned slightly to his left, looking at camera, relaxed expression.",
    "Close portrait, plain light blue background, even soft light, 85mm lens, sharp focus on the eyes, highly detailed skin texture, head turned slightly to his right, chin down, looking up at camera.",
    "Wider framing showing the upper chest in a grey hoodie, concrete wall background, hard noon sun from above-left casting shadows under the brow and nose, 85mm lens, sharp focus, highly detailed skin texture, head turned slightly to his left, squinting slightly.",
    "Tight crop, golden hour backlight with a soft rim of light on the hair, front of the face in open shade, 85mm lens, sharp focus on the eyes, highly detailed skin texture, head turned three-quarters to his left, looking at camera.",
]
SEED = 7
STEPS = 9

app = modal.App("syok-fset")
vol = modal.Volume.from_name("syok-lora-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers", "accelerate", "sentencepiece", "pillow", "huggingface_hub[hf_transfer]",
                 "diffusers>=0.36")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, gpu="L40S", timeout=1500, volumes={"/data": vol})
def generate(batch: int = 1) -> list:
    """~3 min model download + ~5 s per image. Hard 25-minute cap (about $0.80 worst case on an L40S)."""
    import io

    import torch
    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16).to("cuda")
    out = Path("/data/fset/raw")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    tails, prefix = (TAILS, "g") if batch == 1 else (TAILS2, "h")
    for i, tail in enumerate(tails, 1):
        g = torch.Generator("cuda").manual_seed(SEED)
        img = pipe(prompt=BASE + tail, height=1024, width=1024, num_inference_steps=STEPS, guidance_scale=0.0, generator=g).images[0]
        name = f"{prefix}{i:02d}.png"
        img.save(out / name)  # lossless: webp/jpeg compression would blur the pores we are trying to keep
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        results.append((name, buf.getvalue()))
        print("ok", name, flush=True)
    vol.commit()
    return results


@app.local_entrypoint()
def main(out: str = "fset_raw", batch: int = 1):
    d = Path(out)
    d.mkdir(exist_ok=True)
    for name, data in generate.remote(batch):
        (d / name).write_bytes(data)
        print("saved", d / name, len(data))
