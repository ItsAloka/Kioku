"""Scan folders and bring the index up to date: new or changed photos in, deleted ones out.

Plain functions (no Qt) so they can be tested and run from a worker thread. Decoding happens
on a small thread pool (Pillow releases the GIL), embedding on the GPU in batches.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

from .clip import Clip
from .imaging import blur_score, is_image, load_rgb, tiny_signature
from .store import Store

BATCH = 32
DECODE_SIDE = 512  # decode at most this big: plenty for the model (224) and the blur score
SKIP_DIRS = {"$recycle.bin", "system volume information", ".thumbnails", "node_modules", ".git"}

# (done, total, current file)
Progress = Callable[[int, int, str], None]


@dataclass
class Report:
    added: int = 0
    removed: int = 0
    failed: int = 0
    cancelled: bool = False


def walk_images(folder: str) -> Iterator[tuple[str, float, int]]:
    """(path, mtime, size) of every image under ``folder``; hidden and system dirs skipped, and
    hidden files too (macOS leaves a 4 KB "._IMG_1234.jpg" metadata file next to each photo)."""
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d.lower() not in SKIP_DIRS]
        for name in files:
            if not name.startswith(".") and is_image(Path(name)):
                path = os.path.join(root, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size > 0:
                    yield path, st.st_mtime, st.st_size


def _prepare(item: tuple[str, float, int], preprocess: Callable[[Image.Image], np.ndarray]):
    """Decode one photo → ((path, mtime, size, width, height, blur, tiny, pixels), None)
    or (item, error)."""
    path, mtime, size = item
    try:
        with Image.open(path) as im:
            width, height = im.size
            orientation = im.getexif().get(0x0112, 1)
        if orientation in (5, 6, 7, 8):  # rotated 90°: the upright photo is taller than wide
            width, height = height, width
        im = load_rgb(Path(path), max_side=DECODE_SIDE)
        return (path, mtime, size, width, height, blur_score(im), tiny_signature(im), preprocess(im)), None
    except Exception as exc:  # corrupt, truncated, unsupported: record it, keep going
        return item, f"{type(exc).__name__}: {exc}"


def update_index(
    store: Store,
    clip: Clip,
    folders: list[str] | None = None,
    progress: Progress | None = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> Report:
    report = Report()
    work: list[tuple[str, tuple[str, float, int]]] = []  # (folder, item)
    for folder in folders if folders is not None else store.folders():
        known = store.known(folder)
        seen: set[str] = set()
        if os.path.isdir(folder):  # an unplugged drive: keep its photos, just skip it
            for path, mtime, size in walk_images(folder):
                seen.add(path)
                if known.get(path) != (mtime, size):
                    work.append((folder, (path, mtime, size)))
            gone = [p for p in known if p not in seen]
            store.delete_paths(gone)
            report.removed += len(gone)
        if cancelled():
            report.cancelled = True
            return report

    total = len(work)
    done = 0
    if progress:
        progress(0, total, "")
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        for start in range(0, total, BATCH):
            if cancelled():
                report.cancelled = True
                break
            chunk = work[start:start + BATCH]
            results = list(pool.map(partial(_prepare, preprocess=clip.preprocess), [item for _f, item in chunk]))
            ready, pixels, folders_of = [], [], []
            for (folder, _item), (row, error) in zip(chunk, results, strict=True):
                if error:
                    store.mark_failed(*row, error)
                    report.failed += 1
                else:
                    ready.append(row)
                    pixels.append(row[7])
                    folders_of.append(folder)
            if ready:
                embs = clip.encode_pixels(np.stack(pixels))
                by_folder: dict[str, list] = {}
                for row, folder, emb in zip(ready, folders_of, embs, strict=True):
                    by_folder.setdefault(folder, []).append((*row[:7], emb))
                for folder, rows in by_folder.items():
                    store.upsert(folder, rows)
                report.added += len(ready)
            done += len(chunk)
            if progress:
                progress(done, total, chunk[-1][1][0])
    return report
