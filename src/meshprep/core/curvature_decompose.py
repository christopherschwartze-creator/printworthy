"""Curvature-based surface decomposition: segment at CONVEXITY CHANGES.

WHY THIS EXISTS
===============
The bundled ``ei_decompose_hierarchical`` clusters faces by NORMAL DIRECTION.
On a smooth region of CONSTANT curvature whose normals fan around (a cylinder
body, a sphere), the normals point every which way, so direction-clustering
shatters one uniform region into many compass-sector strips -> a highly cyclic
chart graph (betti1 huge), which defeats the bulk+blob TREE the decoupling needs.

The RIGHT signal is the SECOND MOMENT OF CURVATURE (the curvature tensor): it is
CONSTANT within a region of uniform bending and only changes where the surface's
CONVEXITY changes (cylinder body -> spherical cap, bulk -> blob across a saddle
neck, flat face -> flat face across a sharp crease). Cutting at those convexity
changes yields the natural bulk+blob regions.

THE SIGNAL (second moment of curvature, Taubin 1995)
----------------------------------------------------
Per vertex v with unit normal n, for each 1-ring neighbour u:
    d   = u - v
    T   = normalize( (I - n nᵀ) d )            (tangent component of the edge)
    kap = -2 (n · d) / (d · d)                 (Euler normal curvature in dir T;
                                                sign: convex/sphere => kap > 0)
    M  += w · kap · T Tᵀ                        (the SECOND MOMENT of curvature)
M is a symmetric 3x3 with one ~0 eigenvalue along n; its two tangent eigenvalues
m_hi >= m_lo invert (Taubin) to the principal curvatures
    k1 = 3 m_hi - m_lo ,   k2 = 3 m_lo - m_hi .
(The absolute scale depends on the weights and is irrelevant to segmentation —
only the SIGN PATTERN and HOMOGENEITY of (k1,k2) matter.)

CONVEXITY TYPE (where the cuts come from)
-----------------------------------------
With a scale-relative zero band eps:
    |k1|,|k2| < eps                  -> PLANAR        (flat)
    one ~0, the other != 0           -> DEVELOPABLE   (cylinder-like)
    k1,k2 same sign, both >= eps      -> ELLIPTIC      (+ convex / - concave)
    opposite signs                    -> HYPERBOLIC    (saddle)
Two adjacent faces are in the SAME region iff they share a convexity type AND are
not separated by a sharp dihedral crease. Connected components of that relation
are the charts; a convexity change (type boundary) or a crease is a CUT.

Permissive: numpy + trimesh only. NO GPL. Self-contained (does NOT need
Applied.MeshDecomp). Never raises: on any failure returns a single-chart labeling.
"""
from __future__ import annotations

import numpy as np

# convexity type codes
PLANAR = 0
DEV_CONVEX = 1      # developable, positive (ridge)
DEV_CONCAVE = 2     # developable, negative (valley)
ELLIPTIC_CONVEX = 3  # both > 0 (sphere-like bump / bulk)
ELLIPTIC_CONCAVE = 4  # both < 0 (bowl)
HYPERBOLIC = 5      # saddle (the neck between bulk and blob)


def principal_curvatures(mesh):
    """Per-vertex (k1, k2, dir1) from the SECOND MOMENT OF CURVATURE tensor.

    Returns
    -------
    k1, k2 : (V,) float, principal curvatures with k1 >= k2 (sign convention:
             a convex sphere with outward normals gives k1, k2 > 0).
    dir1   : (V, 3) unit principal direction of k1 (max curvature). Zeros where
             undefined (degenerate/boundary vertex).

    Pure numpy + trimesh. The accumulation is vectorised over the half-edges; the
    per-vertex 3x3 eigensolve is a single loop (cheap for typical mesh sizes).
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    N = np.asarray(mesh.vertex_normals, dtype=np.float64)
    n = len(V)
    # directed half-edges from unique undirected edges (both orientations)
    e = np.asarray(mesh.edges_unique, dtype=np.int64)
    A = np.concatenate([e[:, 0], e[:, 1]])     # tail vertex (where we accumulate)
    B = np.concatenate([e[:, 1], e[:, 0]])     # head vertex

    d = V[B] - V[A]                              # (E2, 3) edge vectors
    dd = np.einsum("ij,ij->i", d, d)            # |d|^2
    na = N[A]
    nd = np.einsum("ij,ij->i", na, d)           # n . d
    # tangent component of the edge direction
    T = d - nd[:, None] * na
    Tn = np.linalg.norm(T, axis=1)
    valid = (dd > 1e-18) & (Tn > 1e-9)
    T = np.where(valid[:, None], T / np.maximum(Tn[:, None], 1e-18), 0.0)
    # Euler normal curvature in direction T. Sign flip so convex (sphere) => +.
    kap = np.where(valid, -2.0 * nd / np.maximum(dd, 1e-18), 0.0)

    # accumulate M_a = sum_b kap * T Tᵀ  (and a weight count) per tail vertex A
    M = np.zeros((n, 3, 3), dtype=np.float64)
    outer = kap[:, None, None] * (T[:, :, None] * T[:, None, :])  # (E2,3,3)
    np.add.at(M, A, outer)
    wsum = np.zeros(n)
    np.add.at(wsum, A, valid.astype(np.float64))
    M = M / np.maximum(wsum[:, None, None], 1.0)

    k1 = np.zeros(n)
    k2 = np.zeros(n)
    dir1 = np.zeros((n, 3))
    for a in range(n):
        if wsum[a] < 1.0:
            continue
        evals, evecs = np.linalg.eigh(M[a])          # ascending eigenvalues
        order = np.argsort(np.abs(evals))            # |.| ascending; [0]=normal(~0)
        m_lo, m_hi = evals[order[1]], evals[order[2]]
        if m_hi < m_lo:
            m_lo, m_hi = m_hi, m_lo
            hi_vec = evecs[:, order[1]]
        else:
            hi_vec = evecs[:, order[2]]
        ka = 3.0 * m_hi - m_lo                       # Taubin inversion
        kb = 3.0 * m_lo - m_hi
        k1[a] = max(ka, kb)
        k2[a] = min(ka, kb)
        dir1[a] = hi_vec
    return k1, k2, dir1


def smooth_curvature_field(mesh, k1, k2, iters=16):
    """Uniform 1-ring averaging of the (k1, k2) field — damps discrete-curvature
    NOISE so a nominally-uniform region (a sphere, a smooth bump bulk) does not
    fragment into speckle at classification time.

    A few Laplacian/uniform smoothing passes pull each vertex's (k1, k2) toward
    the mean of its 1-ring neighbours. This collapses the thin sign-flicker rings
    that appear in transition zones (e.g. the base of a Gaussian bump) WITHOUT
    moving a genuinely uniform region (whose neighbours already agree) or erasing
    a real convexity step (which is wide and survives a few passes). The principal
    ORDER k1 >= k2 is re-imposed after smoothing.

    Pure numpy + trimesh.edges_unique. Returns (k1_s, k2_s). On any failure
    returns the inputs unchanged (never raises).
    """
    try:
        iters = int(iters)
        if iters <= 0:
            return np.asarray(k1, float), np.asarray(k2, float)
        n = len(k1)
        e = np.asarray(mesh.edges_unique, dtype=np.int64)
        if len(e) == 0:
            return np.asarray(k1, float), np.asarray(k2, float)
        A = np.concatenate([e[:, 0], e[:, 1]])
        B = np.concatenate([e[:, 1], e[:, 0]])
        deg = np.zeros(n)
        np.add.at(deg, A, 1.0)
        deg = np.maximum(deg, 1.0)
        a = np.array(k1, dtype=np.float64)
        b = np.array(k2, dtype=np.float64)
        for _ in range(iters):
            sa = np.zeros(n)
            sb = np.zeros(n)
            np.add.at(sa, A, a[B])
            np.add.at(sb, A, b[B])
            # 0.5-weight uniform Laplacian step: stable, gentle.
            a = 0.5 * a + 0.5 * (sa / deg)
            b = 0.5 * b + 0.5 * (sb / deg)
        # re-impose k1 >= k2 (smoothing the two channels independently can cross
        # them at a saddle; the convexity test assumes k1 is the max).
        hi = np.maximum(a, b)
        lo = np.minimum(a, b)
        return hi, lo
    except Exception:
        return np.asarray(k1, float), np.asarray(k2, float)


def convexity_type(k1, k2, eps):
    """Map (k1, k2) arrays to a convexity-type code with zero band eps."""
    k1 = np.asarray(k1, float)
    k2 = np.asarray(k2, float)
    a0 = np.abs(k1) < eps
    b0 = np.abs(k2) < eps
    out = np.empty(len(k1), dtype=np.int64)
    out[:] = -1
    planar = a0 & b0
    dev = (a0 ^ b0)                                   # exactly one ~0
    same = (~planar) & (~dev) & (np.sign(k1) == np.sign(k2))
    saddle = (~planar) & (~dev) & (np.sign(k1) != np.sign(k2))
    out[planar] = PLANAR
    # developable sign = sign of the non-zero curvature
    dev_sign = np.where(np.abs(k1) >= np.abs(k2), np.sign(k1), np.sign(k2))
    out[dev & (dev_sign >= 0)] = DEV_CONVEX
    out[dev & (dev_sign < 0)] = DEV_CONCAVE
    out[same & (k1 > 0)] = ELLIPTIC_CONVEX
    out[same & (k1 <= 0)] = ELLIPTIC_CONCAVE
    out[saddle] = HYPERBOLIC
    return out


def decompose_by_curvature(
    mesh,
    *,
    eps_rel: float = 0.28,
    crease_angle: float = 0.6,
    min_region_faces: int = 4,
    smooth_iters: int = 16,
) -> np.ndarray:
    """Segment ``mesh`` into charts at CONVEXITY CHANGES; return per-face labels.

    Two adjacent faces share a chart iff they have the same convexity type AND
    are not separated by a dihedral crease > ``crease_angle`` (radians). Connected
    components of that relation are the charts.

    Parameters
    ----------
    eps_rel : zero band for "flat" as a FRACTION of a robust curvature scale
              (the 75th percentile of |k1|), so the test is scale-invariant.
    crease_angle : dihedral angle (rad) above which an edge is always a cut.
    min_region_faces : regions smaller than this are merged into a SAME-TYPE,
              non-crease-separated neighbour (kills speckle from curvature noise).
              A small region with NO valid same-type neighbour (e.g. a small cube
              side, crease-bounded on all sides) is LEFT INTACT rather than
              destroyed -- legitimate small regions survive.
    smooth_iters : number of uniform 1-ring averaging passes applied to the
              (k1, k2) field before classification. Damps discrete-curvature
              noise so a nominally-uniform region does not fragment into speckle.
              0 disables smoothing.

    Returns
    -------
    labels : (F,) int, contiguous chart ids (0..C-1). Feed to
             ``build_chart_tree(mesh, face_labels=labels)``.

    Never raises: on any failure returns all-zeros (single chart).
    """
    try:
        faces = np.asarray(mesh.faces, dtype=np.int64)
        nf = len(faces)
        if nf == 0:
            return np.zeros(0, dtype=np.int64)

        k1, k2, _ = principal_curvatures(mesh)
        # W1 fix: damp discrete-curvature noise before classifying, so smooth
        # uniform regions do not shatter into sign-flicker speckle.
        k1, k2 = smooth_curvature_field(mesh, k1, k2, iters=smooth_iters)
        scale = float(np.percentile(np.abs(k1), 75))
        eps = max(eps_rel * scale, 1e-9)
        vtype = convexity_type(k1, k2, eps)
        # per-face type = mode of its 3 vertices (ties -> the max-|curvature| vertex)
        ftype = np.empty(nf, dtype=np.int64)
        kmag = np.abs(k1) + np.abs(k2)
        for f in range(nf):
            vs = faces[f]
            ts = vtype[vs]
            if ts[0] == ts[1] == ts[2]:
                ftype[f] = ts[0]
            else:
                # majority, breaking ties toward the strongest-curvature vertex
                vals, counts = np.unique(ts, return_counts=True)
                top = counts.max()
                cands = vals[counts == top]
                if len(cands) == 1:
                    ftype[f] = cands[0]
                else:
                    ftype[f] = ts[int(np.argmax(kmag[vs]))]

        # union-find over face adjacencies; merge iff same type AND not a crease
        parent = np.arange(nf)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        fadj = np.asarray(mesh.face_adjacency, dtype=np.int64)
        try:
            ang = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
        except Exception:
            ang = np.zeros(len(fadj))
        is_crease = ang > crease_angle
        for idx, (fa, fb) in enumerate(fadj):
            if ftype[fa] == ftype[fb] and not is_crease[idx]:
                union(int(fa), int(fb))

        roots = np.array([find(f) for f in range(nf)])
        # contiguous relabel
        _, labels = np.unique(roots, return_inverse=True)
        labels = labels.astype(np.int64)

        # W2 fix: speckle merge is crease- AND type-aware. A small region folds
        # only into a SAME-TYPE neighbour across a NON-crease boundary; a small
        # region with no such neighbour (a crease-bounded cube side) is kept.
        labels = _merge_small_regions(
            labels, fadj, ftype, is_crease, min_region_faces)
        return labels
    except Exception:
        return np.zeros(len(np.asarray(mesh.faces)), dtype=np.int64)


def _merge_small_regions(labels, fadj, ftype, is_crease, min_region_faces):
    """Fold a speckle region (< min_region_faces) into a SAME-TYPE neighbour it
    shares a NON-crease boundary with (longest such boundary wins). A small
    region with no valid same-type non-crease neighbour is LEFT INTACT.

    This is the W2 fix: the old version merged a small region into its
    largest-boundary neighbour REGARDLESS of type or crease, so it collapsed
    every (legitimately small) cube side across the 90deg creases into one
    chart. Restricting the merge to same-type non-crease neighbours keeps real
    small regions (box -> ~6) while still erasing genuine curvature speckle
    (which is surrounded by a same-type smooth region with no crease between).

    Repeats until stable (bounded). Deterministic: small regions are processed
    in sorted id order and ties broken by neighbour id.
    """
    from collections import defaultdict

    labels = np.asarray(labels, dtype=np.int64).copy()
    fadj = np.asarray(fadj, dtype=np.int64)
    ftype = np.asarray(ftype, dtype=np.int64)
    is_crease = np.asarray(is_crease, dtype=bool)
    for _ in range(8):  # a few passes converge; bounded for safety
        ids, counts = np.unique(labels, return_counts=True)
        if len(ids) <= 1:
            break
        size = {int(i): int(c) for i, c in zip(ids, counts)}
        small = set(i for i, c in size.items() if c < min_region_faces)
        if not small:
            break
        # NON-crease shared boundary length between region pairs (only these
        # edges are eligible to merge across; crease edges are hard cuts).
        shared = defaultdict(int)
        for idx in range(len(fadj)):
            if is_crease[idx]:
                continue
            fa, fb = int(fadj[idx, 0]), int(fadj[idx, 1])
            la, lb = int(labels[fa]), int(labels[fb])
            if la != lb:
                shared[(la, lb)] += 1
                shared[(lb, la)] += 1
        # dominant convexity type of each region (regions are type-homogeneous
        # by construction; recompute defensively after each pass).
        region_type: dict = {}
        for f in range(len(labels)):
            region_type.setdefault(int(labels[f]), int(ftype[f]))
        changed = False
        for s in sorted(small):
            cands = [
                (cnt, other)
                for (a, other), cnt in shared.items()
                if a == s and region_type.get(other) == region_type.get(s)
                and other not in small         # don't merge into another speckle
            ]
            if not cands:
                # no same-type non-crease neighbour -> keep this region intact
                continue
            cands.sort(reverse=True)           # longest boundary; tie -> larger id
            target = cands[0][1]
            labels[labels == s] = target
            changed = True
        if not changed:
            break
        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.astype(np.int64)
    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int64)
