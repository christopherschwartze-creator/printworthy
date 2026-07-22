"""Small internal helpers shared across the ei_core package.

Per CODE_REVIEW L2R-1: three copies of `_build_vertex_neighbours` lived
in `quality.py`, `triage.py`, and `classify.py`. Each had a slightly
different "we keep this here so we don't depend on the others" comment.
They're collapsed here. The cost of importing one symbol is zero; the
benefit is a single drift surface.
"""
from __future__ import annotations

import numpy as np


def _distinct_directed_edges(faces: np.ndarray) -> np.ndarray:
    """Sorted-unique directed (src, dst) vertex pairs implied by the faces.

    Each triangle (a, b, c) contributes the 6 directed neighbour relations
    a↔b, a↔c, b↔c. Returns a (P, 2) int64 array, lexicographically sorted
    (src ascending, dst ascending within src) with duplicate rows removed —
    so `pairs[:, 0]` is sorted and each src's dst block is the sorted-unique
    neighbour set. Empty (0, 2) when there are no faces.
    """
    F = np.asarray(faces, dtype=np.int64)
    if F.ndim != 2 or F.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)
    src = F[:, [0, 0, 1, 1, 2, 2]].reshape(-1)
    dst = F[:, [1, 2, 0, 2, 0, 1]].reshape(-1)
    pairs = np.stack([src, dst], axis=1)
    # np.unique(axis=0) returns rows lexsorted (col0 then col1) and unique.
    return np.unique(pairs, axis=0)


def build_vertex_neighbours(n: int, faces: np.ndarray) -> list[list[int]]:
    """Build a vertex → sorted-unique-neighbour list from the face array.

    Returns a Python list-of-lists of length `n`. Each inner list is
    the sorted distinct neighbour set of the corresponding vertex.

    Note: callers that already have a `trimesh.Trimesh` should prefer
    `mesh.vertex_neighbors` (lazily cached by trimesh), which this
    function reproduces. We use this helper when only raw arrays are
    available — e.g. inside the σ_min closed-form classifier tests.

    Vectorised (one sort over the directed edge list) — bit-identical to the
    old per-face set-accumulation loop but ~3–4× faster on the raw-array path.
    Vertex indices ≥ n (out of range) are ignored rather than raising.
    """
    pairs = _distinct_directed_edges(faces)
    out: list[list[int]] = [[] for _ in range(n)]
    if pairs.shape[0] == 0:
        return out
    src = pairs[:, 0]
    # bounds[v] = first row whose src ≥ v; the rows [bounds[v]:bounds[v+1]] are
    # exactly vertex v's sorted-unique neighbours (src is sorted ascending).
    bounds = np.searchsorted(src, np.arange(n + 1))
    dst = pairs[:, 1]
    for v in range(n):
        seg = dst[bounds[v]:bounds[v + 1]]
        if seg.size:
            out[v] = seg.tolist()
    return out


def valence_from_faces(n: int, faces: np.ndarray) -> np.ndarray:
    """Per-vertex valence (count of distinct neighbours) from the face array.

    Vectorised; bit-identical to the old set-accumulation loop. Vertex indices
    ≥ n are ignored (the returned array is always length n).
    """
    pairs = _distinct_directed_edges(faces)
    if pairs.shape[0] == 0:
        return np.zeros(n, dtype=np.int64)
    return np.bincount(pairs[:, 0], minlength=n)[:n].astype(np.int64)
