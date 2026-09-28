from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from kioku import model_dir
from kioku.imaging import blur_score, clip_pixels
from kioku.indexer import update_index
from kioku.search import duplicate_groups, top_k
from kioku.store import Store


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def shape_image(path: Path, kind: str, color: str, size=(640, 480)) -> Path:
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    box = (size[0] // 4, size[1] // 5, size[0] * 3 // 4, size[1] * 4 // 5)
    (d.ellipse if kind == "circle" else d.rectangle)(box, fill=color)
    im.save(path)
    return path


# --- search ------------------------------------------------------------------------------------


def test_top_k_orders_best_first_and_applies_floor():
    m = np.stack([unit([1, 0, 0]), unit([0, 1, 0]), unit([1, 1, 0])])
    hits = top_k(unit([1, 0.1, 0]), m, k=3, min_score=0.5)
    assert [i for i, _ in hits] == [0, 2]
    assert hits[0][1] > hits[1][1]


def test_top_k_empty_matrix():
    assert top_k(unit([1, 0]), np.zeros((0, 2), dtype=np.float32)) == []


def test_duplicate_groups_are_transitive_and_sorted():
    a = unit([1, 0, 0, 0])
    near_a = unit([1, 0.2, 0, 0])
    near_near = unit([1, 0.4, 0, 0])  # close to near_a, a bit further from a
    b = unit([0, 0, 1, 0])
    near_b = unit([0, 0, 1, 0.1])
    lone = unit([0, 1, 0, 0])
    m = np.stack([a, b, near_a, lone, near_b, near_near])
    groups = duplicate_groups(m, threshold=0.97, block=2)  # small block exercises the blocking
    assert groups == [[0, 2, 5], [1, 4]]


def test_duplicate_groups_need_matching_colours_too():
    m = np.stack([unit([1, 0]), unit([1, 0.01]), unit([1, 0.02])])  # CLIP: all "the same"
    red, red2, blue = [255, 0, 0] * 64, [250, 5, 0] * 64, [0, 0, 255] * 64
    tiny = np.array([red, red2, blue], dtype=np.uint8)
    assert duplicate_groups(m, 0.95, tiny) == [[0, 1]]


def test_top_k_margin_keeps_only_close_matches():
    m = np.stack([unit([1, 0]), unit([1, 0.05]), unit([1, 1])])
    assert [i for i, _ in top_k(unit([1, 0]), m, k=3, margin=0.06)] == [0, 1]


# --- blur --------------------------------------------------------------------------------------


def test_blur_score_separates_sharp_from_blurry(tmp_path):
    im = Image.open(shape_image(tmp_path / "a.png", "rect", "black"))
    sharp = blur_score(im)
    blurry = blur_score(im.filter(ImageFilter.GaussianBlur(6)))
    assert sharp > 10 * blurry


def test_sharp_photo_with_few_edges_is_not_called_blurry(tmp_path):
    # A flat, low-contrast graphic: variance-of-Laplacian calls this blurry; we must not.
    im = Image.open(shape_image(tmp_path / "y.png", "circle", "yellow"))
    assert blur_score(im) > 15 > blur_score(im.filter(ImageFilter.GaussianBlur(3)))


def test_clip_pixels_shape_and_centre_crop():
    arr = clip_pixels(Image.new("RGB", (1000, 300), "white"))
    assert arr.shape == (3, 224, 224)
    assert arr.dtype == np.float32


# --- store -------------------------------------------------------------------------------------


def test_folders_never_nest(tmp_path):
    store = Store(tmp_path / "i.db")
    parent, child = tmp_path / "photos", tmp_path / "photos" / "2024"
    child.mkdir(parents=True)
    store.add_folder(child)
    store.add_folder(parent)  # folds the child into the parent
    assert store.folders() == [str(parent.resolve())]
    with pytest.raises(ValueError):
        store.add_folder(child)


# --- indexer (fake CLIP: the pipeline without the 400 MB model) ---------------------------------


class FakeClip:
    device = "CPU"
    preprocess = staticmethod(clip_pixels)

    def encode_pixels(self, pixels: np.ndarray) -> np.ndarray:
        feats = pixels.reshape(len(pixels), 3, -1).mean(axis=2)  # mean colour per channel
        feats = np.concatenate([feats, np.zeros((len(pixels), 509), dtype=np.float32)], axis=1)
        return feats / np.linalg.norm(feats, axis=1, keepdims=True)


def test_update_index_adds_changes_and_removes(tmp_path):
    photos = tmp_path / "photos"
    (photos / "sub").mkdir(parents=True)
    (photos / ".hidden").mkdir()
    a = shape_image(photos / "a.jpg", "circle", "red")
    shape_image(photos / "sub" / "b.png", "rect", "blue")
    shape_image(photos / ".hidden" / "skip.png", "rect", "green")
    (photos / "broken.jpg").write_bytes(b"not an image")
    (photos / "._a.jpg").write_bytes(b"\x00\x05\x16\x07Mac OS X")  # macOS metadata, not a photo
    (photos / "notes.txt").write_text("ignored")

    store = Store(tmp_path / "i.db")
    store.add_folder(photos)
    report = update_index(store, FakeClip())
    assert (report.added, report.failed) == (2, 1)
    listed, matrix, tiny = store.load()
    assert sorted(Path(p.path).name for p in listed) == ["a.jpg", "b.png"]
    assert matrix.shape == (2, 512)
    assert tiny.shape == (2, 192)

    # nothing changed → nothing to do (the broken file is remembered, not retried)
    report = update_index(store, FakeClip())
    assert (report.added, report.failed, report.removed) == (0, 0, 0)

    # a changed file is re-indexed, a deleted one is dropped
    shape_image(a, "circle", "red", size=(800, 600))
    os.utime(a, (1_900_000_000, 1_900_000_000))
    os.remove(photos / "sub" / "b.png")
    report = update_index(store, FakeClip())
    assert (report.added, report.removed) == (1, 1)
    listed, _, _ = store.load()
    assert [(Path(p.path).name, p.width) for p in listed] == [("a.jpg", 800)]


def test_changing_model_forgets_photos_but_keeps_folders(tmp_path):
    photos = tmp_path / "p"
    photos.mkdir()
    shape_image(photos / "a.png", "rect", "red")
    store = Store(tmp_path / "i.db")
    store.add_folder(photos)
    assert store.use_model("model-a") is False  # empty index: nothing to redo
    update_index(store, FakeClip())
    assert store.use_model("model-a") is False  # same model: keep everything
    assert store.count() == 1
    assert store.use_model("model-b") is True  # new model: old vectors can't be compared
    assert store.count() == 0
    assert store.folders() == [str(photos.resolve())]
    assert update_index(store, FakeClip()).added == 1


def test_exif_rotated_photo_reports_upright_size(tmp_path):
    photos = tmp_path / "p"
    photos.mkdir()
    im = Image.new("RGB", (400, 200), "white")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° CW on display
    im.save(photos / "r.jpg", exif=exif)
    store = Store(tmp_path / "i.db")
    store.add_folder(photos)
    update_index(store, FakeClip())
    (photo,), _, _ = store.load()
    assert (photo.width, photo.height) == (200, 400)


# --- the real model (skipped if the ONNX files are not downloaded) -----------------------------

MODELS = ["siglip2-b16-224", "clip-vit-b16"]


@pytest.mark.parametrize("name", MODELS)
def test_real_model_matches_text_to_images(tmp_path, name):
    if not (model_dir(name) / "text_model_quantized.onnx").is_file():
        pytest.skip(f"run packaging/fetch_model.py {name}")
    from kioku.clip import SPECS, Clip

    clip = Clip(name)
    names = ["red circle", "blue square"]
    pixels = np.stack([
        clip.preprocess(Image.open(shape_image(tmp_path / "c.png", "circle", "red")).convert("RGB")),
        clip.preprocess(Image.open(shape_image(tmp_path / "s.png", "rect", "blue")).convert("RGB")),
    ])  # fmt: skip
    emb = clip.encode_pixels(pixels)
    assert emb.shape == (2, SPECS[name].dim)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-4)
    for i, name in enumerate(names):
        assert int(np.argmax(emb @ clip.encode_query(f"a {name}"))) == i
