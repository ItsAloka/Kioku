"""Image loading, CLIP preprocessing and the blur (sharpness) score."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # HEIC/HEIF from iPhones; optional so the app still runs without it.
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF = True
except ImportError:  # pragma: no cover
    HEIF = False

Image.MAX_IMAGE_PIXELS = 200_000_000  # large panoramas are fine; decompression bombs are not

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
if HEIF:
    EXTENSIONS |= {".heic", ".heif"}

# openai/clip-vit-base-patch16 preprocessor_config.json
CLIP_SIZE = 224
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in EXTENSIONS


def load_rgb(path: Path, max_side: int | None = None) -> Image.Image:
    """Open an image upright (EXIF orientation applied) as RGB, optionally downscaled."""
    with Image.open(path) as im:
        if max_side and hasattr(im, "draft"):
            im.draft("RGB", (max_side, max_side))  # JPEG: decode at a smaller scale, much faster
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGBA").convert("RGB") if "A" in im.getbands() else im.convert("RGB")
        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return im.copy()


# google/siglip2-base-patch16-224 preprocessor_config.json
SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SIGLIP_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def clip_pixels(im: Image.Image) -> np.ndarray:
    """CLIP input: shortest side to 224 (bicubic), centre crop 224x224, normalise, CHW float32."""
    w, h = im.size
    scale = CLIP_SIZE / min(w, h)
    nw, nh = max(CLIP_SIZE, round(w * scale)), max(CLIP_SIZE, round(h * scale))
    im = im.resize((nw, nh), Image.Resampling.BICUBIC)
    left, top = (nw - CLIP_SIZE) // 2, (nh - CLIP_SIZE) // 2
    im = im.crop((left, top, left + CLIP_SIZE, top + CLIP_SIZE))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    return arr.transpose(2, 0, 1)


def siglip_pixels(im: Image.Image) -> np.ndarray:
    """SigLIP input: the whole photo squashed to 224x224 (bilinear, no crop), scaled to −1…1."""
    im = im.resize((CLIP_SIZE, CLIP_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - SIGLIP_MEAN) / SIGLIP_STD
    return arr.transpose(2, 0, 1)


def tiny_signature(im: Image.Image) -> np.ndarray:
    """8×8 RGB miniature (192 bytes). Two copies of one photo match here; two different photos
    of the same kind of scene (which CLIP rates as near-identical) usually do not."""
    return np.asarray(im.resize((8, 8), Image.Resampling.BOX), dtype=np.uint8).reshape(-1)


def blur_score(im: Image.Image) -> float:
    """Sharpness: mean of the strongest 0.3% Laplacian responses on a 512px greyscale copy.

    The usual variance-of-Laplacian averages over the whole frame, so sharp photos with few
    edges (clear sky, screenshots, flat graphics) score as "blurry". Looking only at the
    strongest edges asks the real question: is anything in this photo in focus? Sharp photos
    score roughly 20–500; clearly blurred or shaken ones stay under ~12.
    """
    g = im.convert("L")
    if max(g.size) > 512:
        g = g.copy()
        g.thumbnail((512, 512), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    lap = np.abs(a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4.0 * a[1:-1, 1:-1])
    flat = lap.ravel()
    k = max(1, flat.size * 3 // 1000)
    return float(np.partition(flat, flat.size - k)[-k:].mean())
