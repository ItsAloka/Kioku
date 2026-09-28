"""Image/text embedding model on ONNX Runtime: images and text into one shared vector space.

Two models are supported, picked by folder name under models/ (see ``kioku.MODEL``):
  clip-vit-b16     CLIP ViT-B/16 (OpenAI, 2021): 512-d, the default (smaller download)
  siglip2-b16-224  SigLIP 2 ViT-B/16 (Google, 2025): 768-d, more accurate, ~280 MB bigger

The image model runs on the GPU through DirectML when one is available (any DirectX 12 GPU:
NVIDIA, AMD or Intel) and falls back to the CPU. The text model is the int8-quantised export
and always runs on the CPU: it encodes one short query at a time, so it is fast either way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

from . import MODEL, model_dir
from .imaging import clip_pixels, siglip_pixels


@dataclass(frozen=True)
class Spec:
    """Everything that differs between the supported models."""

    dim: int
    preprocess: Callable[[Image.Image], np.ndarray]
    image_output: str
    text_output: str
    context: int  # text length in tokens
    pad_to_context: bool  # SigLIP was trained on text padded to full length; CLIP was not
    # Search tuning: text→image scores below ``min_text_score`` are noise; results further
    # than ``text_margin`` below the best hit are dropped. Duplicate slider default in %.
    min_text_score: float
    text_margin: float
    dup_default: int


SPECS = {
    "siglip2-b16-224": Spec(
        dim=768, preprocess=siglip_pixels, image_output="pooler_output",
        text_output="pooler_output", context=64, pad_to_context=True,
        min_text_score=0.05, text_margin=0.05, dup_default=95,
    ),
    "clip-vit-b16": Spec(
        dim=512, preprocess=clip_pixels, image_output="image_embeds",
        text_output="text_embeds", context=77, pad_to_context=False,
        min_text_score=0.22, text_margin=0.03, dup_default=95,
    ),
}  # fmt: skip


def _normalise(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class Clip:
    def __init__(self, name: str | None = None, use_gpu: bool = True) -> None:
        self.name = name or MODEL
        if self.name not in SPECS:
            raise ValueError(f"unknown model {self.name!r}; choose from {', '.join(SPECS)}")
        self.spec = SPECS[self.name]
        root = model_dir(self.name)
        missing = [n for n in ("vision_model.onnx", "text_model_quantized.onnx", "tokenizer.json")
                   if not (root / n).is_file()]  # fmt: skip
        if missing:
            raise FileNotFoundError(
                f"model files missing in {root}: {', '.join(missing)} "
                f"(run packaging\\fetch_model.py {self.name})"
            )
        providers = ["CPUExecutionProvider"]
        if use_gpu and "DmlExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "DmlExecutionProvider")
        vis_opts = ort.SessionOptions()
        if providers[0] == "DmlExecutionProvider":
            vis_opts.enable_mem_pattern = False  # required by DirectML
            vis_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            self.vision = ort.InferenceSession(
                str(root / "vision_model.onnx"), vis_opts, providers=providers
            )
        except Exception:  # a broken GPU driver must not stop the app: retry on the CPU
            providers = ["CPUExecutionProvider"]
            self.vision = ort.InferenceSession(
                str(root / "vision_model.onnx"), providers=providers
            )
        self.device = "GPU (DirectML)" if self.vision.get_providers()[0] == "DmlExecutionProvider" else "CPU"
        self.text = ort.InferenceSession(
            str(root / "text_model_quantized.onnx"), providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self._pad_id = self.tokenizer.token_to_id("<pad>") or 0
        self._vis_in = self.vision.get_inputs()[0].name
        self._vis_out = self._output_name(self.vision, self.spec.image_output)
        self._txt_inputs = {i.name for i in self.text.get_inputs()}
        self._txt_out = self._output_name(self.text, self.spec.text_output)

    @staticmethod
    def _output_name(session: ort.InferenceSession, wanted: str) -> str:
        names = [o.name for o in session.get_outputs()]
        if wanted not in names:
            raise RuntimeError(f"model has no {wanted!r} output (outputs: {names})")
        return wanted

    def preprocess(self, im: Image.Image) -> np.ndarray:
        """An RGB image → the (3, 224, 224) float32 input this model expects."""
        return self.spec.preprocess(im)

    def encode_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """(n, 3, 224, 224) float32 → (n, dim) unit vectors."""
        out = self.vision.run([self._vis_out], {self._vis_in: pixels.astype(np.float32)})[0]
        return _normalise(out.astype(np.float32))

    def encode_text(self, text: str) -> np.ndarray:
        """One query → a (dim,) unit vector."""
        ctx = self.spec.context
        ids = self.tokenizer.encode(text.strip().lower()).ids[:ctx]
        if not self.spec.pad_to_context and len(ids) == ctx:
            ids[-1] = self.tokenizer.token_to_id("<|endoftext|>")
        mask = [1] * len(ids)
        if self.spec.pad_to_context:
            pad = ctx - len(ids)
            ids, mask = ids + [self._pad_id] * pad, mask + [0] * pad
        feed = {"input_ids": np.array([ids], dtype=np.int64)}
        if "attention_mask" in self._txt_inputs:
            feed["attention_mask"] = np.array([mask], dtype=np.int64)
        out = self.text.run([self._txt_out], feed)[0]
        return _normalise(out.astype(np.float32))[0]

    def encode_query(self, query: str) -> np.ndarray:
        """Search vector for a user query: the query and "a photo of <query>" averaged.

        Both models were trained on image captions, so the caption-style prompt usually lands
        closer to real photos; keeping the raw query too helps with things like "screenshot of
        code".
        """
        q = query.strip()
        vec = self.encode_text(q) + self.encode_text(f"a photo of {q}")
        return _normalise(vec)
