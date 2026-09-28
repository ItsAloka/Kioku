# Third-party notices

Kioku itself is MIT-licensed (see `LICENSE`). The installer also ships:

| Component | Licence | Notes |
|---|---|---|
| CLIP ViT-B/16 (OpenAI), ONNX export by Xenova | MIT | `models/clip-vit-b16/` — image model (fp32) and text model (int8) |
| PyQt6 / Qt 6 | GPL v3 / LGPL v3 | GUI |
| ONNX Runtime + DirectML | MIT | inference on CPU / any DirectX 12 GPU |
| Hugging Face Tokenizers | Apache 2.0 | text tokenizer |
| NumPy | BSD-3-Clause | |
| Pillow | MIT-CMU (HPND) | image decoding |
| pillow-heif / libheif | BSD-3-Clause / LGPL v3 | HEIC photos |
| Send2Trash | BSD-3-Clause | move to Recycle Bin |

Because the build includes PyQt6 (GPL v3), distributed binaries of Kioku are offered under
terms compatible with the GPL v3; the source code is available at
https://github.com/ItsAloka/Kioku.
