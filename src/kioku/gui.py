"""Kioku's window: Search / Duplicates / Blurry tabs over one shared photo index."""

from __future__ import annotations

import os
import subprocess
import traceback
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PyQt6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    QRunnable,
    QSize,
    Qt,
    QThread,
    QThreadPool,
    pyqtSignal,
)
from PyQt6.QtGui import QAction, QColor, QGuiApplication, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from send2trash import send2trash

from . import APP_NAME, MODEL, data_dir
from .clip import SPECS, Clip
from .imaging import EXTENSIONS, load_rgb
from .indexer import Report, update_index
from .search import duplicate_groups, top_k
from .store import Photo, Store

THUMB = 168
CELL = QSize(THUMB + 16, THUMB + 40)


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


# --- background helpers ------------------------------------------------------------------------


class _Relay(QObject):
    done = pyqtSignal(object, object)  # (result, error text or None)


class _Job(QRunnable):
    def __init__(self, fn: Callable[[], object], relay: _Relay) -> None:
        super().__init__()
        self.fn, self.relay = fn, relay

    def run(self) -> None:
        try:
            self.relay.done.emit(self.fn(), None)
        except Exception:
            self.relay.done.emit(None, traceback.format_exc())


def run_later(parent: QObject, fn: Callable[[], object], on_done: Callable[[object, object], None]) -> None:
    """Run ``fn`` on the thread pool; ``on_done(result, error)`` then runs on the GUI thread."""
    relay = _Relay(parent)
    relay.done.connect(lambda result, error: (on_done(result, error), relay.deleteLater()))
    QThreadPool.globalInstance().start(_Job(fn, relay))


class ThumbCache(QObject):
    """Thumbnails decoded on the thread pool, kept as an LRU of QPixmaps."""

    ready = pyqtSignal(str)

    class _Signals(QObject):
        loaded = pyqtSignal(str, QImage)

    class _Load(QRunnable):
        def __init__(self, path: str, signals: ThumbCache._Signals) -> None:
            super().__init__()
            self.path, self.signals = path, signals

        def run(self) -> None:
            try:
                im = load_rgb(Path(self.path), max_side=THUMB * 2)
                data = im.tobytes("raw", "RGB")
                qimg = QImage(data, im.width, im.height, 3 * im.width, QImage.Format.Format_RGB888).copy()
            except Exception:
                qimg = QImage()
            self.signals.loaded.emit(self.path, qimg)

    def __init__(self, limit: int = 3000) -> None:
        super().__init__()
        self.limit = limit
        self.cache: OrderedDict[str, QPixmap] = OrderedDict()
        self.pending: set[str] = set()
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(max(2, (os.cpu_count() or 4) // 2))
        self.signals = ThumbCache._Signals()
        self.signals.loaded.connect(self._loaded)
        self.placeholder = QPixmap(THUMB, THUMB)
        self.placeholder.fill(QColor("#262a33"))
        self.broken = QPixmap(THUMB, THUMB)
        self.broken.fill(QColor("#3a2328"))

    def get(self, path: str) -> QPixmap:
        pix = self.cache.get(path)
        if pix is not None:
            self.cache.move_to_end(path)
            return pix
        if path not in self.pending:
            self.pending.add(path)
            self.pool.start(ThumbCache._Load(path, self.signals))
        return self.placeholder

    def forget(self, paths: list[str]) -> None:
        for p in paths:
            self.cache.pop(p, None)

    def _loaded(self, path: str, img: QImage) -> None:
        self.pending.discard(path)
        pix = self.broken if img.isNull() else QPixmap.fromImage(img).scaled(
            THUMB, THUMB, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.cache[path] = pix
        while len(self.cache) > self.limit:
            self.cache.popitem(last=False)
        self.ready.emit(path)


# --- grid --------------------------------------------------------------------------------------


class PhotoModel(QAbstractListModel):
    def __init__(self, thumbs: ThumbCache) -> None:
        super().__init__()
        self.thumbs = thumbs
        self.items: list[tuple[Photo, str, str]] = []  # (photo, caption, tooltip extra)
        self.rows_by_path: dict[str, list[int]] = {}
        thumbs.ready.connect(self._thumb_ready)

    def set_items(self, items: list[tuple[Photo, str, str]]) -> None:
        self.beginResetModel()
        self.items = items
        self.rows_by_path = {}
        for row, (photo, _c, _t) in enumerate(items):
            self.rows_by_path.setdefault(photo.path, []).append(row)
        self.endResetModel()

    def photo(self, row: int) -> Photo:
        return self.items[row][0]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008, N802
        return 0 if parent.isValid() else len(self.items)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        photo, caption, extra = self.items[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return caption
        if role == Qt.ItemDataRole.DecorationRole:
            return self.thumbs.get(photo.path)
        if role == Qt.ItemDataRole.ToolTipRole:
            lines = [photo.path, f"{photo.width}×{photo.height} · {human_size(photo.size)}"]
            return "\n".join(lines + ([extra] if extra else []))
        return None

    def _thumb_ready(self, path: str) -> None:
        for row in self.rows_by_path.get(path, ()):
            idx = self.index(row)
            self.dataChanged.emit(idx, idx, [Qt.ItemDataRole.DecorationRole])


class PhotoGrid(QListView):
    """Thumbnail grid. Double-click opens; right-click offers open / show / similar / recycle."""

    similar_requested = pyqtSignal(object)  # Photo
    trash_requested = pyqtSignal(list)  # list[Photo]

    def __init__(self, thumbs: ThumbCache) -> None:
        super().__init__()
        self.model_ = PhotoModel(thumbs)
        self.setModel(self.model_)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setIconSize(QSize(THUMB, THUMB))
        self.setGridSize(CELL)
        self.setUniformItemSizes(True)
        self.setWordWrap(True)
        self.setSpacing(4)
        self.setLayoutMode(QListView.LayoutMode.Batched)
        self.setBatchSize(200)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.doubleClicked.connect(lambda idx: open_file(self.model_.photo(idx.row()).path))

    def set_items(self, items: list[tuple[Photo, str, str]]) -> None:
        self.model_.set_items(items)
        self.scrollToTop()

    def selected(self) -> list[Photo]:
        rows = sorted({i.row() for i in self.selectedIndexes()})
        return [self.model_.photo(r) for r in rows]

    def _menu(self, pos) -> None:
        idx = self.indexAt(pos)
        if not idx.isValid():
            return
        if not self.selectionModel().isSelected(idx):
            self.clearSelection()
            self.selectionModel().select(idx, self.selectionModel().SelectionFlag.Select)
        photo = self.model_.photo(idx.row())
        chosen = self.selected()
        menu = QMenu(self)
        menu.addAction("Open", lambda: open_file(photo.path))
        menu.addAction("Show in folder", lambda: show_in_folder(photo.path))
        menu.addAction("Find similar photos", lambda: self.similar_requested.emit(photo))
        menu.addAction("Copy path", lambda: QGuiApplication.clipboard().setText(photo.path))
        menu.addSeparator()
        label = "Move to Recycle Bin" if len(chosen) == 1 else f"Move {len(chosen)} photos to Recycle Bin"
        menu.addAction(label, lambda: self.trash_requested.emit(chosen))
        menu.exec(self.viewport().mapToGlobal(pos))


def open_file(path: str) -> None:
    try:
        os.startfile(path)  # type: ignore[attr-defined]
    except OSError as exc:
        QMessageBox.warning(None, APP_NAME, f"Could not open the file:\n{exc}")


def show_in_folder(path: str) -> None:
    subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])


# --- folders dialog ----------------------------------------------------------------------------


class FoldersDialog(QDialog):
    def __init__(self, store: Store, parent: QWidget) -> None:
        super().__init__(parent)
        self.store = store
        self.changed = False
        self.setWindowTitle("Photo folders")
        self.resize(560, 320)
        self.list = QListWidget()
        add = QPushButton("Add folder…")
        remove = QPushButton("Remove")
        add.clicked.connect(self._add)
        remove.clicked.connect(self._remove)
        note = QLabel("Kioku only reads these folders. Removing one forgets its photos from the "
                      "index; the files themselves are never touched.")  # fmt: skip
        note.setWordWrap(True)
        note.setObjectName("muted")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        row = QHBoxLayout()
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch()
        lay = QVBoxLayout(self)
        lay.addWidget(self.list)
        lay.addLayout(row)
        lay.addWidget(note)
        lay.addWidget(buttons)
        self._refresh()

    def _refresh(self) -> None:
        self.list.clear()
        self.list.addItems(self.store.folders())

    def _add(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a photo folder")
        if not path:
            return
        try:
            self.store.add_folder(path)
        except ValueError as exc:
            QMessageBox.information(self, APP_NAME, str(exc))
            return
        self.changed = True
        self._refresh()

    def _remove(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        self.store.remove_folder(item.text())
        self.changed = True
        self._refresh()


# --- indexing thread ---------------------------------------------------------------------------


class IndexThread(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(object)  # Report
    failed = pyqtSignal(str)

    def __init__(self, db_path: Path, clip: Clip) -> None:
        super().__init__()
        self.db_path, self.clip = db_path, clip
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        store = Store(self.db_path)
        try:
            report = update_index(
                store, self.clip, progress=self.progress.emit, cancelled=lambda: self._stop
            )
            self.finished_ok.emit(report)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            store.close()


# --- main window -------------------------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1280, 820)
        self.db_path = data_dir() / "index.db"
        self.store = Store(self.db_path)
        self.model_changed = self.store.use_model(MODEL)  # a new model: every photo re-indexed
        self.clip: Clip | None = None
        self.photos: list[Photo] = []
        self.matrix = np.zeros((0, 512), dtype=np.float32)
        self.tiny = np.zeros((0, 192), dtype=np.uint8)
        self.indexer: IndexThread | None = None
        self.thumbs = ThumbCache()

        # top bar
        self.folders_btn = QPushButton("Folders…")
        self.update_btn = QPushButton("Update index")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.hide()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setTextVisible(True)
        self.progress.hide()
        self.status = QLabel("Loading the AI model…")
        self.status.setObjectName("muted")
        self.folders_btn.clicked.connect(self._folders)
        self.update_btn.clicked.connect(self.start_indexing)
        self.stop_btn.clicked.connect(self._stop_indexing)
        top = QHBoxLayout()
        title = QLabel("記憶 Kioku")
        title.setObjectName("title")
        top.addWidget(title)
        top.addSpacing(16)
        top.addWidget(self.folders_btn)
        top.addWidget(self.update_btn)
        top.addWidget(self.stop_btn)
        top.addWidget(self.progress)
        top.addStretch()
        top.addWidget(self.status)

        # tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._search_tab(), "Search")
        self.tabs.addTab(self._dupes_tab(), "Duplicates")
        self.tabs.addTab(self._blurry_tab(), "Blurry")

        # empty state (no folders yet)
        empty = QWidget()
        el = QVBoxLayout(empty)
        el.addStretch()
        hello = QLabel("Search your photos by describing them.")
        hello.setObjectName("hero")
        hint = QLabel("Add a folder of photos to start. Everything stays on this PC.")
        hint.setObjectName("muted")
        pick = QPushButton("Add a photo folder…")
        pick.setObjectName("primary")
        pick.clicked.connect(self._first_folder)
        for w in (hello, hint):
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
            el.addWidget(w)
        el.addSpacing(12)
        el.addWidget(pick, alignment=Qt.AlignmentFlag.AlignCenter)
        el.addStretch()
        self.stack = QStackedWidget()
        self.stack.addWidget(empty)
        self.stack.addWidget(self.tabs)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.addLayout(top)
        lay.addWidget(self.stack)
        self.setCentralWidget(central)

        find = QAction(self)
        find.setShortcut("Ctrl+F")
        find.triggered.connect(lambda: (self.tabs.setCurrentIndex(0), self.query.setFocus(), self.query.selectAll()))
        self.addAction(find)

        self._set_busy_controls(True)
        self._reload()
        run_later(self, Clip, self._clip_loaded)

    # --- tabs ----------------------------------------------------------------------------------

    def _search_tab(self) -> QWidget:
        self.query = QLineEdit()
        self.query.setPlaceholderText(
            "Describe a photo… e.g. beach at sunset, screenshot of code, handwritten notes, a cat on a bed"
        )
        self.query.setClearButtonEnabled(True)
        self.query.returnPressed.connect(self.search)
        go = QPushButton("Search")
        go.setObjectName("primary")
        go.clicked.connect(self.search)
        by_photo = QPushButton("Search by photo…")
        by_photo.setToolTip("Pick any image (indexed or not) and find photos that look like it")
        by_photo.clicked.connect(self.search_by_photo)
        self.search_info = QLabel("")
        self.search_info.setObjectName("muted")
        self.search_grid = PhotoGrid(self.thumbs)
        self.search_grid.similar_requested.connect(self.similar)
        self.search_grid.trash_requested.connect(self.trash)
        row = QHBoxLayout()
        row.addWidget(self.query, 1)
        row.addWidget(go)
        row.addWidget(by_photo)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addLayout(row)
        lay.addWidget(self.search_info)
        lay.addWidget(self.search_grid, 1)
        return w

    def _dupes_tab(self) -> QWidget:
        self.dup_slider = QSlider(Qt.Orientation.Horizontal)
        self.dup_slider.setRange(88, 99)
        self.dup_slider.setValue(SPECS[MODEL].dup_default if MODEL in SPECS else 95)
        self.dup_slider.setMaximumWidth(220)
        self.dup_label = QLabel()
        self.dup_slider.valueChanged.connect(self._dup_label_text)
        self._dup_label_text()
        find = QPushButton("Find duplicates")
        find.setObjectName("primary")
        find.clicked.connect(self.find_duplicates)
        pick = QPushButton("Select extras")
        pick.setToolTip("In every group, select all photos except the best one "
                        "(largest resolution, then sharpest).")  # fmt: skip
        pick.clicked.connect(self._select_extras)
        trash = QPushButton("Move selected to Recycle Bin")
        trash.clicked.connect(lambda: self.trash(self.dup_grid.selected()))
        self.dup_info = QLabel("Finds photos that look almost the same: burst shots, resized or "
                               "re-saved copies, the same photo from WhatsApp and the camera.")  # fmt: skip
        self.dup_info.setObjectName("muted")
        self.dup_grid = PhotoGrid(self.thumbs)
        self.dup_grid.similar_requested.connect(self.similar)
        self.dup_grid.trash_requested.connect(self.trash)
        self.dup_groups: list[list[Photo]] = []
        row = QHBoxLayout()
        row.addWidget(QLabel("Similarity"))
        row.addWidget(self.dup_slider)
        row.addWidget(self.dup_label)
        row.addWidget(find)
        row.addStretch()
        row.addWidget(pick)
        row.addWidget(trash)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addLayout(row)
        lay.addWidget(self.dup_info)
        lay.addWidget(self.dup_grid, 1)
        return w

    def _blurry_tab(self) -> QWidget:
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setRange(4, 40)
        self.blur_slider.setValue(15)
        self.blur_slider.setMaximumWidth(220)
        self.blur_label = QLabel()
        self.blur_slider.valueChanged.connect(lambda v: self.blur_label.setText(f"sharpness < {v}"))
        self.blur_slider.valueChanged.connect(self.find_blurry)
        self.blur_label.setText(f"sharpness < {self.blur_slider.value()}")
        trash = QPushButton("Move selected to Recycle Bin")
        trash.clicked.connect(lambda: self.trash(self.blur_grid.selected()))
        self.blur_info = QLabel("")
        self.blur_info.setObjectName("muted")
        self.blur_grid = PhotoGrid(self.thumbs)
        self.blur_grid.similar_requested.connect(self.similar)
        self.blur_grid.trash_requested.connect(self.trash)
        row = QHBoxLayout()
        row.addWidget(QLabel("Show photos with"))
        row.addWidget(self.blur_slider)
        row.addWidget(self.blur_label)
        row.addStretch()
        row.addWidget(trash)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addLayout(row)
        lay.addWidget(self.blur_info)
        lay.addWidget(self.blur_grid, 1)
        return w

    # --- state ---------------------------------------------------------------------------------

    def _reload(self) -> None:
        """Re-read the index from SQLite (after indexing or deleting)."""
        self.photos, self.matrix, self.tiny = self.store.load()
        self.stack.setCurrentIndex(1 if self.store.folders() else 0)
        self._update_status()
        self.find_blurry()

    def _update_status(self) -> None:
        parts = [f"{len(self.photos):,} photos", f"{len(self.store.folders())} folder(s)"]
        parts.append(self.clip.device if self.clip else "loading AI model…")
        self.status.setText(" · ".join(parts))

    def _set_busy_controls(self, busy: bool) -> None:
        self.update_btn.setEnabled(not busy and self.clip is not None)
        self.folders_btn.setEnabled(not busy)

    def _clip_loaded(self, clip: Clip | None, error: str | None) -> None:
        if error:
            self.status.setText("AI model failed to load")
            QMessageBox.critical(self, APP_NAME, f"The AI model could not be loaded.\n\n{error[-1500:]}")
            return
        self.clip = clip
        self._set_busy_controls(False)
        self._update_status()
        if self.model_changed:
            self.statusBar().showMessage(
                "The AI model has changed: re-indexing your photos once.", 15000
            )
        if self.store.folders():
            self.start_indexing()  # pick up photos added since last time

    # --- actions -------------------------------------------------------------------------------

    def _first_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a photo folder")
        if not path:
            return
        try:
            self.store.add_folder(path)
        except ValueError as exc:
            QMessageBox.information(self, APP_NAME, str(exc))
            return
        self._reload()
        self.start_indexing()

    def _folders(self) -> None:
        dlg = FoldersDialog(self.store, self)
        dlg.exec()
        if dlg.changed:
            self._reload()
            self.start_indexing()

    def start_indexing(self) -> None:
        if self.clip is None or self.indexer is not None or not self.store.folders():
            return
        self.indexer = IndexThread(self.db_path, self.clip)
        self.indexer.progress.connect(self._index_progress)
        self.indexer.finished_ok.connect(self._index_done)
        self.indexer.failed.connect(self._index_failed)
        self.indexer.finished.connect(self._index_thread_ended)
        self._set_busy_controls(True)
        self.stop_btn.show()
        self.progress.show()
        self.progress.setRange(0, 0)
        self.progress.setFormat("Scanning folders…")
        self.indexer.start()

    def _stop_indexing(self) -> None:
        if self.indexer:
            self.indexer.stop()
            self.stop_btn.setEnabled(False)

    def _index_progress(self, done: int, total: int, _current: str) -> None:
        if total == 0:
            return
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.progress.setFormat(f"Indexing {done:,} / {total:,}")

    def _index_done(self, report: Report) -> None:
        self._reload()
        bits = []
        if report.added:
            bits.append(f"{report.added:,} indexed")
        if report.removed:
            bits.append(f"{report.removed:,} removed")
        if report.failed:
            bits.append(f"{report.failed:,} unreadable")
        if report.cancelled:
            bits.append("stopped")
        if bits:
            self.statusBar().showMessage("Index updated: " + ", ".join(bits), 8000)

    def _index_failed(self, error: str) -> None:
        QMessageBox.critical(self, APP_NAME, f"Indexing failed.\n\n{error[-1500:]}")

    def _index_thread_ended(self) -> None:
        self.indexer = None
        self.progress.hide()
        self.stop_btn.hide()
        self.stop_btn.setEnabled(True)
        self._set_busy_controls(False)

    def search(self) -> None:
        text = self.query.text().strip()
        if not text:
            return
        if self.clip is None:
            self.search_info.setText("The AI model is still loading…")
            return
        if not self.photos:
            self.search_info.setText("No photos indexed yet.")
            return
        spec = self.clip.spec
        hits = top_k(
            self.clip.encode_query(text), self.matrix, k=300, min_score=spec.min_text_score, margin=spec.text_margin
        )
        self.search_grid.set_items(
            [(self.photos[i], Path(self.photos[i].path).name, f"match {s:.2f}") for i, s in hits]
        )
        self.search_info.setText(
            f"Best matches for “{text}” ({len(hits)}), closest first." if hits
            else f"Nothing matched “{text}”. Try describing it differently."
        )

    def similar(self, photo: Photo) -> None:
        try:
            i = next(n for n, p in enumerate(self.photos) if p.id == photo.id)
        except StopIteration:
            return
        self._show_similar(self.matrix[i], Path(photo.path).name, skip=i)

    def search_by_photo(self) -> None:
        if self.clip is None or not self.photos:
            self.search_info.setText("Index some photos first." if self.clip else "The AI model is still loading…")
            return
        exts = " ".join(f"*{e}" for e in sorted(EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(self, "Find photos like…", "", f"Images ({exts})")
        if not path:
            return
        try:
            vec = self.clip.encode_pixels(self.clip.preprocess(load_rgb(Path(path), max_side=512))[None])[0]
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not read that image:\n{exc}")
            return
        self._show_similar(vec, Path(path).name)

    def _show_similar(self, vec: np.ndarray, name: str, skip: int = -1) -> None:
        hits = [(j, s) for j, s in top_k(vec, self.matrix, k=101) if j != skip][:100]
        self.tabs.setCurrentIndex(0)
        self.query.clear()
        self.search_grid.set_items(
            [(self.photos[j], Path(self.photos[j].path).name, f"similarity {s:.2f}") for j, s in hits]
        )
        self.search_info.setText(f"Photos that look like {name}, most similar first")

    def _dup_label_text(self) -> None:
        self.dup_label.setText(f"{self.dup_slider.value()}%")

    def find_duplicates(self) -> None:
        if not self.photos:
            return
        threshold = self.dup_slider.value() / 100
        matrix, tiny, photos = self.matrix, self.tiny, list(self.photos)
        self.dup_info.setText("Comparing every photo with every other…")

        def done(groups, error) -> None:
            if error:
                self.dup_info.setText("Duplicate search failed.")
                return
            self.dup_groups = [[photos[i] for i in g] for g in groups]
            items = []
            for n, group in enumerate(self.dup_groups, 1):
                for p in group:
                    items.append((p, f"Group {n} · {p.width}×{p.height}", f"group {n} of {len(groups)}"))
            self.dup_grid.set_items(items)
            extra = sum(len(g) - 1 for g in self.dup_groups)
            wasted = sum(sum(p.size for p in g) - max(p.size for p in g) for g in self.dup_groups)
            self.dup_info.setText(
                f"{len(groups)} groups, {extra} extra copies (about {human_size(wasted)}). "
                "“Select extras” keeps the best photo of each group."
                if groups else "No duplicates found at this similarity."
            )

        run_later(self, lambda: duplicate_groups(matrix, threshold, tiny), done)

    def _select_extras(self) -> None:
        keep = {max(g, key=lambda p: (p.width * p.height, p.blur)).id for g in self.dup_groups}
        sel = self.dup_grid.selectionModel()
        sel.clearSelection()
        model = self.dup_grid.model_
        for row in range(model.rowCount()):
            if model.photo(row).id not in keep:
                sel.select(model.index(row), sel.SelectionFlag.Select)

    def find_blurry(self) -> None:
        limit = self.blur_slider.value()
        blurry = sorted((p for p in self.photos if p.blur < limit), key=lambda p: p.blur)
        self.blur_grid.set_items(
            [(p, f"sharpness {p.blur:.0f}", "lower = blurrier") for p in blurry]
        )
        self.blur_info.setText(
            f"{len(blurry)} of {len(self.photos):,} photos look blurry or out of focus, blurriest first. "
            "Some may be intentional (night shots, soft backgrounds): check before deleting."
        )

    def trash(self, photos: list[Photo]) -> None:
        if not photos:
            return
        size = human_size(sum(p.size for p in photos))
        answer = QMessageBox.question(
            self, APP_NAME,
            f"Move {len(photos)} photo(s) ({size}) to the Recycle Bin?\n\n"
            "You can restore them from the Recycle Bin later.",
        )  # fmt: skip
        if answer != QMessageBox.StandardButton.Yes:
            return
        moved, failed = [], []
        for p in photos:
            try:
                send2trash(p.path)
                moved.append(p.path)
            except Exception as exc:
                failed.append(f"{p.path}: {exc}")
        self.store.delete_paths(moved)
        self.thumbs.forget(moved)
        self._reload()
        gone = set(moved)
        for grid in (self.search_grid, self.dup_grid):
            grid.set_items([it for it in grid.model_.items if it[0].path not in gone])
        self.statusBar().showMessage(f"Moved {len(moved)} photo(s) to the Recycle Bin", 8000)
        if failed:
            QMessageBox.warning(self, APP_NAME, "Some files could not be moved:\n\n" + "\n".join(failed[:10]))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.indexer is not None:
            self.indexer.stop()
            self.indexer.wait(10000)
        self.store.close()
        super().closeEvent(event)


STYLE = """
QWidget { background: #16181d; color: #e6e6eb; font-family: 'Segoe UI'; font-size: 10pt; }
QLabel#title { font-size: 15pt; font-weight: 600; color: #f2f2f7; }
QLabel#hero { font-size: 18pt; font-weight: 600; }
QLabel#muted { color: #9a9cab; }
QLineEdit { background: #1f222a; border: 1px solid #2f3340; border-radius: 8px; padding: 8px 10px; font-size: 11pt; }
QLineEdit:focus { border-color: #9b7bff; }
QPushButton { background: #252935; border: 1px solid #333848; border-radius: 8px; padding: 7px 14px; }
QPushButton:hover { background: #2d3240; }
QPushButton:disabled { color: #6b6e7c; }
QPushButton#primary { background: #8b6cf6; border: none; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #9b7fff; }
QTabWidget::pane { border: none; }
QTabBar::tab { background: transparent; padding: 8px 18px; color: #9a9cab; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #f2f2f7; border-bottom: 2px solid #8b6cf6; }
QListView { background: #1a1d24; border: 1px solid #262a33; border-radius: 10px; padding: 6px; }
QListView::item { border-radius: 8px; padding: 4px; color: #b9bbc8; }
QListView::item:selected { background: #3b3160; color: white; }
QListView::item:hover { background: #242836; }
QProgressBar { background: #1f222a; border: 1px solid #2f3340; border-radius: 6px; text-align: center; height: 18px; }
QProgressBar::chunk { background: #8b6cf6; border-radius: 6px; }
QSlider::groove:horizontal { height: 4px; background: #2f3340; border-radius: 2px; }
QSlider::handle:horizontal { background: #8b6cf6; width: 14px; margin: -6px 0; border-radius: 7px; }
QMenu { background: #1f222a; border: 1px solid #333848; }
QMenu::item { padding: 6px 18px; }
QMenu::item:selected { background: #3b3160; }
QStatusBar { color: #9a9cab; }
QToolTip { background: #1f222a; color: #e6e6eb; border: 1px solid #333848; }
"""


def app_icon() -> QIcon:
    # Bundled next to this module in both dev and the frozen build (see packaging/Kioku.spec).
    ico = Path(__file__).resolve().parent / "resources" / "app.ico"
    return QIcon(str(ico)) if ico.is_file() else QIcon()
