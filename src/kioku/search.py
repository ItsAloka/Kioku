"""Vector search over unit-length embeddings: nearest photos and near-duplicate groups."""

from __future__ import annotations

import numpy as np

# Per-model score floors and margins live in clip.SPECS: text→image scores are on a different
# scale for each model. What matters is how a photo compares with the best one, not the raw number.

def top_k(
    query: np.ndarray,
    matrix: np.ndarray,
    k: int = 200,
    min_score: float = -1.0,
    margin: float | None = None,
) -> list[tuple[int, float]]:
    """Rows of ``matrix`` most similar to ``query`` (cosine, both unit length), best first.

    ``min_score`` is an absolute floor; ``margin`` also drops anything further than that
    below the best hit.
    """
    if matrix.shape[0] == 0:
        return []
    scores = matrix @ query
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    floor = min_score
    if margin is not None:
        floor = max(floor, float(scores[idx[0]]) - margin)
    return [(int(i), float(scores[i])) for i in idx if scores[i] >= floor]


def duplicate_groups(
    matrix: np.ndarray,
    threshold: float = 0.95,
    tiny: np.ndarray | None = None,
    max_color_diff: float = 0.12,
    block: int = 2048,
) -> list[list[int]]:
    """Groups of rows whose image embeddings are at least ``threshold`` similar.

    CLIP compares meaning, so two *different* photos of the same kind of scene can score 0.95+.
    With ``tiny`` (8×8 RGB miniatures) a pair must also look alike pixel-wise: mean colour
    difference at most ``max_color_diff`` (0–1). Pairs are found block by block (never the
    full n×n matrix in memory) and merged with union-find, so A~B and B~C land in one group.
    Groups are sorted biggest first.
    """
    small = None if tiny is None else tiny.astype(np.float32) / 255.0
    n = matrix.shape[0]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, n, block):
        sims = matrix[start:start + block] @ matrix.T
        rows, cols = np.nonzero(sims >= threshold)
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            a, b = start + r, c
            if a < b and (small is None or np.abs(small[a] - small[b]).mean() <= max_color_diff):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    found = [g for g in groups.values() if len(g) > 1]
    found.sort(key=lambda g: (-len(g), g[0]))
    return found
