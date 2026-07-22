"""Per-vertex frame-conditioning (σ*) and normal-coherence (ρ) metrics.

Three primitives, each with a single-vertex form (`vertex_*`) and a
field-version that vectorises over the whole mesh (`vertex_*_field`).

    vertex_sigma_min(vertices, v_idx, v2v) -> (σ_min, FrameStatus)
        Smallest singular value of the 3×k unit-edge matrix at vertex v.
        Captures frame conditioning — collapses for sliver fans, hair
        vertices, near-coplanar 1-rings.

    vertex_rho_diameter(face_normals_at_v)
        Maximum pairwise angle (radians) between incident face normals.
        Max-pairwise → spike-sensitive. Single normal flip → ≈ π.

    vertex_rho_deficit(face_normals_at_v, areas=None)
        1 − ‖mean(unit normals)‖.  Averaged → robust to a single jittery
        normal; lights up only when a region of normals disagrees.

Per PHASE_2_EI_CORE.md §2.4, ρ_deficit is the product default — the
classifier feeds it (rather than ρ_diameter) into REPAIR_TABLE dispatch.

Every field function returns (values, status) per the locked viz contract.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np

from ._types import FrameStatus

# Re-export so consumers can `from printworthy.core.quality import FrameStatus`
# without going through .types — keeps quality.py self-contained for
# anyone reading the math.
__all__ = [
    "FrameStatus",
    "vertex_sigma_min",
    "vertex_sigma_min_field",
    "vertex_rho_diameter",
    "vertex_rho_diameter_field",
    "vertex_rho_deficit",
    "vertex_rho_deficit_field",
    "sigma_min_2d_closed_form",
]


# ── σ* — frame conditioning ───────────────────────────────────────────────


def sigma_min_2d_closed_form(cos_theta: float | np.ndarray) -> float | np.ndarray:
    """Closed-form σ_min for the rank-deficient (k = 2) case.

    For two unit edges u₁, u₂ with cos(angle) = c:
        UᵀU = [[1, c], [c, 1]]   eigenvalues 1+c, 1-c
        σ_min(U) = √(1 − |c|)

    Algebraically identical to `closed_form_sigma_min(theta)` from
    `Applied/FEM/benchmarks/_check_sigma_star_2d.py` and to the 2D PCA
    `sigma_min_per_vertex` in `Applied/PCA/ei_reference.py`. Canonical
    home is here (per audit/canonicals.md §"Per-vertex 2D σ_min").

    Accepts either a Python float or a numpy array — vectorises naturally.
    """
    return np.sqrt(np.clip(1.0 - np.abs(cos_theta), 0.0, 1.0))


def vertex_sigma_min(
    vertices: np.ndarray,
    v_idx: int,
    v2v: Sequence[Sequence[int]],
) -> tuple[float, FrameStatus]:
    """σ_min at a single vertex.

    vertices : (N, 3) array of mesh vertex positions
    v_idx    : index of the vertex to evaluate
    v2v      : sequence-of-sequences, neighbour indices for every vertex
               (e.g. `trimesh.Trimesh.vertex_neighbors`)

    Returns (σ_min, FrameStatus). Semantics:
        valence ≥ 3 → OK,             σ_min via SVD of 3×k unit-edge matrix
        valence = 2 → RANK_DEFICIENT, σ_min via 2D PCA closed form
        valence ≤ 1 → UNDEFINED,      σ_min := 1.0 placeholder
    """
    neighbours = list(v2v[v_idx])
    k = len(neighbours)

    if k <= 1:
        return 1.0, FrameStatus.UNDEFINED

    v = vertices[v_idx]
    edges = vertices[neighbours] - v                     # (k, 3)
    norms = np.linalg.norm(edges, axis=1)
    # Drop zero-length edges (duplicate vertices that slipped past the weld).
    nonzero = norms > 0.0
    if nonzero.sum() < 2:
        return 1.0, FrameStatus.UNDEFINED
    edges = edges[nonzero]
    norms = norms[nonzero]
    unit_edges = edges / norms[:, None]                  # (k', 3)
    k = unit_edges.shape[0]

    if k == 2:
        cos_theta = float(unit_edges[0] @ unit_edges[1])
        return float(sigma_min_2d_closed_form(cos_theta)), FrameStatus.RANK_DEFICIENT

    # k >= 3 — SVD on the 3×k unit-edge matrix.
    U = unit_edges.T                                     # (3, k)
    s = np.linalg.svd(U, compute_uv=False)
    return float(s.min()), FrameStatus.OK


def vertex_sigma_min_field(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    v2v: Optional[Sequence[Sequence[int]]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex (σ_min, status) arrays, both shape (N,).

    vertices : (N, 3)
    faces    : (F, 3) int
    v2v      : optional precomputed vertex-neighbour table. If None, built
               from the face array.

    Returns:
        values : float64 ndarray, shape (N,)
        status : uint8   ndarray, shape (N,), entries from FrameStatus

    Vectorisation (CODE_REVIEW H2R-1): vertices are bucketed by valence
    `k`; within a bucket the unit-edge matrices share shape (m, 3, k) and
    `np.linalg.svd` runs batched. The k=2 bucket uses the closed-form
    2D-PCA expression (no SVD). The previous Python-loop version cost
    ~10 s on a 200 k-vertex Hunyuan mesh; the batched form is ~0.3 s.
    """
    n = int(vertices.shape[0])
    if v2v is None:
        v2v = _build_vertex_neighbours(n, faces)

    values = np.empty(n, dtype=np.float64)
    status = np.empty(n, dtype=np.uint8)

    # Compute per-vertex valence + bucket vertices by k. Drop zero-length
    # edges per vertex (residue from non-welded inputs) before deciding k.
    raw_counts = np.fromiter((len(nbrs) for nbrs in v2v),
                             dtype=np.int64, count=n)

    # Build a (vertex, neighbour_position, 3) padded array of edge vectors
    # bucketed by k. Cheap path: process each unique k value separately.
    # For very large meshes most vertices live in 2-3 buckets (typical
    # valences 5/6/7), keeping bucket count small.
    handled = np.zeros(n, dtype=bool)

    # Handle k ≤ 1 first (UNDEFINED).
    undef_mask = raw_counts <= 1
    if undef_mask.any():
        values[undef_mask] = 1.0
        status[undef_mask] = int(FrameStatus.UNDEFINED)
        handled |= undef_mask

    remaining = np.flatnonzero(~handled)
    if remaining.size == 0:
        return values, status

    unique_ks = np.unique(raw_counts[remaining])
    for k_val in unique_ks:
        if k_val <= 1:
            continue
        bucket_idx = remaining[raw_counts[remaining] == k_val]
        if bucket_idx.size == 0:
            continue
        # Build (m, k, 3) edge stack and (m, k) length stack.
        nbr_arr = np.empty((bucket_idx.size, int(k_val)), dtype=np.int64)
        for j, v in enumerate(bucket_idx):
            nbrs = v2v[int(v)]
            # Length matches by construction (k_val == len(nbrs)).
            nbr_arr[j] = np.asarray(nbrs, dtype=np.int64)
        anchor = vertices[bucket_idx][:, None, :]      # (m, 1, 3)
        edges = vertices[nbr_arr] - anchor             # (m, k, 3)
        lengths = np.linalg.norm(edges, axis=2)        # (m, k)
        ok_edges = lengths > 0

        # If ANY vertex in the bucket has zero-length edges, that vertex
        # might effectively have a lower k; we handle those by falling
        # back to the single-vertex path. Cheap because the bucket they
        # were originally in shrinks.
        clean_rows = ok_edges.all(axis=1)
        clean_bucket = bucket_idx[clean_rows]
        if clean_bucket.size > 0:
            sub_edges = edges[clean_rows]
            sub_lengths = lengths[clean_rows]
            units = sub_edges / sub_lengths[..., None]  # (m', k, 3)

            if k_val == 2:
                cos_t = (units[:, 0, :] * units[:, 1, :]).sum(axis=1)
                values[clean_bucket] = sigma_min_2d_closed_form(cos_t)
                status[clean_bucket] = int(FrameStatus.RANK_DEFICIENT)
            else:
                # U has shape (m', 3, k); batched SVD returns (m', 3).
                U = np.transpose(units, (0, 2, 1))
                s = np.linalg.svd(U, compute_uv=False)
                values[clean_bucket] = s.min(axis=1)
                status[clean_bucket] = int(FrameStatus.OK)
            handled[clean_bucket] = True

        # For the dirty rows (zero-length edges), fall back to the per-
        # vertex helper — small in number and rare in practice.
        dirty = bucket_idx[~clean_rows]
        for v in dirty:
            s_val, s_status = vertex_sigma_min(vertices, int(v), v2v)
            values[int(v)] = s_val
            status[int(v)] = int(s_status)
            handled[int(v)] = True

    # Anything still unhandled (shouldn't happen) gets the per-vertex
    # fallback so we never return uninitialised data.
    leftover = np.flatnonzero(~handled)
    for v in leftover:
        s_val, s_status = vertex_sigma_min(vertices, int(v), v2v)
        values[int(v)] = s_val
        status[int(v)] = int(s_status)

    return values, status


# ── ρ — normal coherence ──────────────────────────────────────────────────


def vertex_rho_diameter(face_normals_at_v: np.ndarray) -> float:
    """Maximum pairwise angle (radians) between incident face normals.

    face_normals_at_v : (m, 3) unit normals of the faces incident to v.

    Semantics:
        m == 0 → 0.0      (vertex with no incident faces)
        m == 1 → 0.0      (single face, nothing to compare)
        m >= 2 → max_{i<j} arccos(clip(nᵢ · nⱼ, -1, 1))

    Spike-sensitive by design: one flipped face among m sane ones makes
    ρ_diameter ≈ π. Useful for catching isolated normal flips; less
    useful for catching a regional fold (use ρ_deficit for that).
    """
    m = face_normals_at_v.shape[0]
    if m < 2:
        return 0.0
    # Cosines of all pairs via outer product on the (m, 3) normal stack.
    gram = face_normals_at_v @ face_normals_at_v.T       # (m, m)
    # Minimum cosine (over off-diagonal entries) = maximum angle.
    iu = np.triu_indices(m, k=1)
    min_cos = float(np.clip(gram[iu].min(), -1.0, 1.0))
    return float(np.arccos(min_cos))


def vertex_rho_deficit(
    face_normals_at_v: np.ndarray,
    areas: Optional[np.ndarray] = None,
) -> float:
    """1 − ‖mean of unit normals‖.

    face_normals_at_v : (m, 3) unit normals of the faces incident to v.
    areas             : optional (m,) per-face area weights.
                        When supplied, the mean is area-weighted, which
                        damps spike contributions from tiny incident
                        triangles. Defaults to equal weighting.

    Semantics:
        m == 0 → 0.0
        m == 1 → 0.0
        m >= 2 → 1 − ‖(Σ wᵢ nᵢ) / Σ wᵢ‖

    All-coherent normals (e.g. flat patch) → 0. One flip among many →
    small positive value. Full random scatter → up to 1.0.

    Robust to a single outlier — this is exactly what we want for the
    classifier dispatch: flag a *region* of bad normals rather than fire
    on a single jittery face.
    """
    m = face_normals_at_v.shape[0]
    if m < 2:
        return 0.0
    if areas is None:
        mean_n = face_normals_at_v.mean(axis=0)
    else:
        w = areas.reshape(-1, 1)
        mean_n = (face_normals_at_v * w).sum(axis=0) / w.sum()
    norm = float(np.linalg.norm(mean_n))
    return float(max(0.0, 1.0 - norm))


def _rho_field(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    weight_by_area: bool,
    use_diameter: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared kernel for the two ρ field flavours.

    Vectorised: a profile (2026-06-03) showed the previous per-vertex
    Python loop dominated triage — `vertex_rho_diameter` was called once
    per vertex (471k× on a 117k-vertex mesh) and recomputed `triu_indices`
    inside each call. This version builds the vertex→face incidence with a
    single argsort, computes ρ_deficit as a segment-sum, and computes
    ρ_diameter by bucketing vertices by valence so the pairwise gram is a
    batched matmul per distinct valence (~10–20 buckets) with one
    `triu_indices` per bucket. Output is numerically identical to the
    per-vertex reference (see test_rho_field_vectorised_matches_reference).
    """
    n = int(vertices.shape[0])
    f = int(faces.shape[0])

    # Per-face unit normals + areas (one vectorised pass).
    tri = vertices[faces]                                 # (F, 3, 3)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(e1, e2)                              # (F, 3)
    area2 = np.linalg.norm(cross, axis=1)                 # 2 × triangle area
    areas = 0.5 * area2
    # Avoid division by zero for degenerate triangles.
    safe = np.where(area2 > 0, area2, 1.0)
    normals = cross / safe[:, None]
    # Degenerate triangles → zero normal; they contribute nothing.

    values = np.zeros(n, dtype=np.float64)
    status = np.empty(n, dtype=np.uint8)

    if f == 0:
        status.fill(int(FrameStatus.UNDEFINED))
        return values, status

    # Vertex→face incidence via one stable argsort. Each face contributes
    # three (vertex, face) incidences; sorting by vertex groups a vertex's
    # incident faces into a contiguous CSR slice.
    vert_ids = faces.reshape(-1).astype(np.int64)         # (3F,)
    face_ids = np.repeat(np.arange(f, dtype=np.int64), 3) # (3F,)
    order = np.argsort(vert_ids, kind="stable")
    vert_sorted = vert_ids[order]
    face_sorted = face_ids[order]                         # incident-face id per slot
    counts = np.bincount(vert_ids, minlength=n)           # incident-face count per vertex
    offsets = np.zeros(n, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)[:-1]                  # CSR start per vertex

    # Status: 0 faces → UNDEFINED; 1 face → BOUNDARY_ONLY; ≥2 → OK.
    status.fill(int(FrameStatus.OK))
    status[counts == 0] = int(FrameStatus.UNDEFINED)
    status[counts == 1] = int(FrameStatus.BOUNDARY_ONLY)

    has_pair = counts >= 2                                # only these get a value

    if not use_diameter:
        # ρ_deficit = 1 − ‖Σ wᵢ nᵢ / Σ wᵢ‖ — a per-vertex segment-sum.
        w = areas if weight_by_area else np.ones(f, dtype=np.float64)
        contrib = normals[face_sorted] * w[face_sorted][:, None]   # (3F, 3)
        sum_n = np.zeros((n, 3), dtype=np.float64)
        np.add.at(sum_n, vert_sorted, contrib)
        sum_w = np.bincount(vert_sorted, weights=w[face_sorted], minlength=n)
        safe_w = np.where(sum_w > 0, sum_w, 1.0)
        mean_norm = np.linalg.norm(sum_n, axis=1) / safe_w
        deficit = np.clip(1.0 - mean_norm, 0.0, None)
        values[has_pair] = deficit[has_pair]
        return values, status

    # ρ_diameter = arccos(min_{i<j} nᵢ·nⱼ) — bucket vertices by valence so
    # each distinct group size is one batched (k, m, m) gram + one
    # triu_indices.
    valences = np.unique(counts[has_pair])
    for m in valences:
        m = int(m)
        vids = np.flatnonzero(has_pair & (counts == m))   # (k,)
        if vids.size == 0:
            continue
        starts = offsets[vids]                            # (k,)
        slot_idx = starts[:, None] + np.arange(m)[None, :]  # (k, m) into *_sorted
        fkm = face_sorted[slot_idx]                       # (k, m) incident face ids
        nrm = normals[fkm]                                # (k, m, 3)
        gram = nrm @ nrm.transpose(0, 2, 1)               # (k, m, m)
        iu0, iu1 = np.triu_indices(m, k=1)                # once per valence
        min_cos = gram[:, iu0, iu1].min(axis=1)           # (k,)
        values[vids] = np.arccos(np.clip(min_cos, -1.0, 1.0))
    return values, status


def vertex_rho_diameter_field(
    vertices: np.ndarray, faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex (ρ_diameter, status). See `vertex_rho_diameter` for math."""
    return _rho_field(vertices, faces, weight_by_area=False, use_diameter=True)


def vertex_rho_deficit_field(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    weight_by_area: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex (ρ_deficit, status). See `vertex_rho_deficit` for math.

    The product default is area-weighted; for benchmark reproductions
    against unweighted reference results pass weight_by_area=False.
    """
    return _rho_field(vertices, faces, weight_by_area=weight_by_area, use_diameter=False)


# ── helpers ───────────────────────────────────────────────────────────────


# Re-export the shared helper so existing internal callers keep working.
# Per CODE_REVIEW L2R-1, the implementation lives in `_util.build_vertex_neighbours`
# now; the three-way drift surface is gone.
from ._util import build_vertex_neighbours as _build_vertex_neighbours

# NOTE (vendor trim): the tangent-projected σ_min variants
# (vertex_sigma_min_tangent / vertex_sigma_min_tangent_field) were dropped in
# the printworthy port — no consumer in the closure (only vertex_sigma_min_field
# is used, by _print_fem) and they were never in __all__. The reference
# implementation remains in the Forge source tree (ei_core/quality.py).


