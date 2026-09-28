"""Kioku (記憶): offline text search, duplicate finder and blur finder for your photo folders."""

from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "1.0.0"

APP_NAME = "Kioku"


def resource_root() -> Path:
    """Where bundled read-only files live: the PyInstaller bundle, or the project root in dev."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[2]


# The image/text model in use (a folder under models/). KIOKU_MODEL=siglip2-b16-224 switches to
# the more accurate but bigger SigLIP 2; changing it re-indexes every photo on the next start.
MODEL = os.environ.get("KIOKU_MODEL", "clip-vit-b16")


def model_dir(name: str | None = None) -> Path:
    return resource_root() / "models" / (name or MODEL)


def data_dir() -> Path:
    """Per-user writable data (the index database). Override with KIOKU_DATA_DIR."""
    override = os.environ.get("KIOKU_DATA_DIR")
    base = Path(override) if override else Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base
