"""Download the ONNX model files into models/<name>/, each checked by SHA-256.

    .venv\\Scripts\\python.exe packaging\\fetch_model.py            # CLIP (the default model)
    .venv\\Scripts\\python.exe packaging\\fetch_model.py siglip2-b16-224

Sources on Hugging Face, each pinned to one repo revision:
  siglip2-b16-224  onnx-community/siglip2-base-patch16-224-ONNX (google/siglip2-base-patch16-224,
                   Apache-2.0)
  clip-vit-b16     Xenova/clip-vit-base-patch16 (openai/clip-vit-base-patch16, MIT)
Files already present with the right hash are skipped.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"

# name → (repo, revision, ((repo path, local name, sha256), ...))
MODELS = {
    "siglip2-b16-224": (
        "onnx-community/siglip2-base-patch16-224-ONNX",
        "ba1f3b0843f24bc5417d38e19c37b287d719b2f4",
        (
            ("onnx/vision_model.onnx", "vision_model.onnx",
             "c0573e3f4140c3a7c4e9cc5912bd6b26a033b46a6a8e8af26cbea262b163bcad"),
            ("onnx/text_model_quantized.onnx", "text_model_quantized.onnx",
             "3a0603d3a00c05a80a6ded4743c16aaac7b1e62cdcc7e362e7ce418659b96400"),
            ("tokenizer.json", "tokenizer.json",
             "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322"),
        ),
    ),
    "clip-vit-b16": (
        "Xenova/clip-vit-base-patch16",
        "342fdf2f67aded64d138ff074745fb4a5d2bba5f",
        (
            ("onnx/vision_model.onnx", "vision_model.onnx",
             "b5170b47c0ecc667c9af5b42ed460c341377706d1466d177798c65fe49ea1e6b"),
            ("onnx/text_model_quantized.onnx", "text_model_quantized.onnx",
             "9106b51e6c663a56b99182ec617c2b3d53577b037e7e24a7717eb78048a0c97a"),
            ("tokenizer.json", "tokenizer.json",
             "72ed5c96db5729294468543e4bc75fce14ca63f58e37300290189ba1c1e52b85"),
        ),
    ),
}  # fmt: skip
DEFAULT = "clip-vit-b16"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(repo: str, revision: str, repo_path: str, target: Path, expected: str) -> None:
    if target.is_file() and sha256(target) == expected:
        print(f"ok    {target.name}")
        return
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{repo_path}"
    partial = target.with_suffix(target.suffix + ".part")
    print(f"get   {target.name}")
    with urllib.request.urlopen(url, timeout=60) as resp, partial.open("wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    got = sha256(partial)
    if got != expected:
        partial.unlink()
        raise SystemExit(f"{target.name}: sha256 {got} does not match the pin {expected}")
    partial.replace(target)
    print(f"ok    {target.name}")


def main(argv: list[str]) -> int:
    name = argv[0] if argv else DEFAULT
    if name not in MODELS:
        raise SystemExit(f"unknown model {name!r}; choose from {', '.join(MODELS)}")
    repo, revision, files = MODELS[name]
    dest = MODELS_ROOT / name
    dest.mkdir(parents=True, exist_ok=True)
    for repo_path, local, expected in files:
        fetch(repo, revision, repo_path, dest / local, expected)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
