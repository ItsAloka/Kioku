"""The photo index: SQLite for rows, one in-memory float32 matrix for the embeddings.

Each thread opens its own ``Store`` (sqlite3 connections are per-thread). The GUI thread
reads; the indexer thread writes.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TINY = 8 * 8 * 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS folders (
    path TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS photos (
    id     INTEGER PRIMARY KEY,
    path   TEXT NOT NULL UNIQUE,
    folder TEXT NOT NULL,
    mtime  REAL NOT NULL,
    size   INTEGER NOT NULL,
    width  INTEGER NOT NULL,
    height INTEGER NOT NULL,
    blur   REAL NOT NULL,
    tiny   BLOB NOT NULL,
    emb    BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS photos_folder ON photos(folder);
CREATE TABLE IF NOT EXISTS failed (
    path  TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    size  INTEGER NOT NULL,
    error TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Photo:
    id: int
    path: str
    size: int
    width: int
    height: int
    blur: float


def norm_path(path: str | Path) -> str:
    """One spelling per file, so the same photo is never indexed twice."""
    return str(Path(path).resolve())


class Store:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.con = sqlite3.connect(db_path, timeout=30)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.executescript(SCHEMA)
        self.con.commit()

    def close(self) -> None:
        self.con.close()

    def use_model(self, name: str) -> bool:
        """Record which model the embeddings come from. Embeddings from different models can't
        be compared, so on a change every photo is forgotten (folders are kept) and the next
        index update rebuilds them. Returns True if that happened."""
        row = self.con.execute("SELECT value FROM meta WHERE key = 'model'").fetchone()
        if row and row[0] == name:
            return False
        had_photos = self.count() > 0
        with self.con:
            self.con.execute("DELETE FROM photos")
            self.con.execute("DELETE FROM failed")
            self.con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('model', ?)", (name,))
        return had_photos

    # --- folders ---------------------------------------------------------------------------

    def folders(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT path FROM folders ORDER BY path")]

    def add_folder(self, path: str | Path) -> str:
        """Add a folder to index. Folders never nest: one inside a known folder is refused
        (it is already covered), and known folders inside the new one are folded into it."""
        p = norm_path(path)
        for existing in self.folders():
            if _inside(p, existing):
                raise ValueError(f"Already covered by {existing}")
        with self.con:
            for existing in self.folders():
                if _inside(existing, p):
                    self.con.execute("DELETE FROM folders WHERE path = ?", (existing,))
                    self.con.execute("UPDATE photos SET folder = ? WHERE folder = ?", (p, existing))
            self.con.execute("INSERT OR IGNORE INTO folders(path) VALUES (?)", (p,))
        return p

    def remove_folder(self, path: str) -> None:
        """Forget a folder and every photo indexed from it. Files on disk are not touched."""
        with self.con:
            self.con.execute("DELETE FROM folders WHERE path = ?", (path,))
            self.con.execute("DELETE FROM photos WHERE folder = ?", (path,))
            self.con.execute(
                "DELETE FROM failed WHERE path LIKE ? ESCAPE '\\'",
                (_like_prefix(path),),
            )

    # --- photos ----------------------------------------------------------------------------

    def known(self, folder: str) -> dict[str, tuple[float, int]]:
        """``{path: (mtime, size)}`` of indexed and failed files under a folder."""
        rows = self.con.execute("SELECT path, mtime, size FROM photos WHERE folder = ?", (folder,))
        out = {p: (m, s) for p, m, s in rows}
        rows = self.con.execute(
            "SELECT path, mtime, size FROM failed WHERE path LIKE ? ESCAPE '\\'",
            (_like_prefix(folder),),
        )
        out.update({p: (m, s) for p, m, s in rows})
        return out

    def upsert(self, folder: str, rows: list[tuple]) -> None:
        """rows: (path, mtime, size, width, height, blur, tiny signature, embedding)."""
        with self.con:
            self.con.executemany(
                """INSERT INTO photos(path, folder, mtime, size, width, height, blur, tiny, emb)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(path) DO UPDATE SET folder=excluded.folder, mtime=excluded.mtime,
                     size=excluded.size, width=excluded.width, height=excluded.height,
                     blur=excluded.blur, tiny=excluded.tiny, emb=excluded.emb""",
                [
                    (p, folder, m, s, w, h, b, np.asarray(t, dtype=np.uint8).tobytes(),
                     np.asarray(e, dtype=np.float32).tobytes())
                    for p, m, s, w, h, b, t, e in rows
                ],
            )
            self.con.executemany("DELETE FROM failed WHERE path = ?", [(r[0],) for r in rows])

    def mark_failed(self, path: str, mtime: float, size: int, error: str) -> None:
        with self.con:
            self.con.execute(
                "INSERT OR REPLACE INTO failed(path, mtime, size, error) VALUES (?, ?, ?, ?)",
                (path, mtime, size, error[:500]),
            )

    def delete_paths(self, paths: list[str]) -> None:
        with self.con:
            self.con.executemany("DELETE FROM photos WHERE path = ?", [(p,) for p in paths])
            self.con.executemany("DELETE FROM failed WHERE path = ?", [(p,) for p in paths])

    def count(self) -> int:
        return self.con.execute("SELECT COUNT(*) FROM photos").fetchone()[0]

    def load(self) -> tuple[list[Photo], np.ndarray, np.ndarray]:
        """All photos, their (n, dim) embeddings and (n, 192) tiny signatures, rows aligned."""
        rows = self.con.execute(
            "SELECT id, path, size, width, height, blur, tiny, emb FROM photos ORDER BY path"
        ).fetchall()
        photos = [Photo(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]
        if not rows:
            return photos, np.zeros((0, 0), np.float32), np.zeros((0, TINY), np.uint8)
        tiny = np.frombuffer(b"".join(r[6] for r in rows), dtype=np.uint8).reshape(-1, TINY)
        matrix = np.frombuffer(b"".join(r[7] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        return photos, matrix.copy(), tiny.copy()


def _inside(child: str, parent: str) -> bool:
    """True if ``child`` is ``parent`` or somewhere below it (case-insensitive, like Windows)."""
    c, p = Path(child), Path(parent)
    return c == p or p in c.parents


def _like_prefix(folder: str) -> str:
    esc = folder.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    sep = "\\\\" if not esc.endswith("\\\\") else ""
    return esc + sep + "%"
