"""Per-part quad-grid placement on a segmented mesh.

THESIS (why segmentation unlocks this). Whole-mesh seamless quad retopology
needs a global mixed-integer parameterization with interior cones (the MIQ/CoMISo
kernel, which is GPL and forbidden here). But once the shape is cut into PARTS,
every part is a topological disk or tube:

    * CAP  (1 cut boundary,  chi=1): a fingertip, a dome. ZERO interior cones are
      forced (Poincare-Hopf: all the curvature goes to the boundary). One pole
      vertex (valence Nv) is the single, expected irregular vertex.
    * TUBE (2 cut boundaries, chi=0): a limb segment, a vase body. ZERO cones,
      ZERO irregular interior vertices -- a perfectly regular grid exists.
    * JUNCTION (k>=3 cut boundaries, chi=2-k): the bulk after limbs are removed.
      Poincare-Hopf REQUIRES |4(2-k)| quarter-cones here -- irregular vertices are
      mandatory and belong AT the junctions (exactly where good hand-retopo puts
      them). v1 flattens it (LSCM) and reports the irregulars honestly.

So per-part the grid is the integer iso-lines of a FREE harmonic parameterization
(no integer program). The ONLY global coupling is that two parts sharing a cut
agree on the SAME integer subdivision N of that loop (a per-loop integer, driven
by one global target edge length) -- then their boundary vertices coincide and
the grids weld watertight.

PI's design principle (sphere+bump): "directions solved separately, perpendicular
to the cut, scaled continuous." This is exactly the harmonic field u (=0 on the
cut, rising inward): its iso-lines run PARALLEL to the cut, its gradient runs
PERPENDICULAR to it, and matching N across the cut makes the two grids continuous.

Permissive deps only (numpy, scipy, trimesh, matplotlib.tri for inverse map). NO
GPL. The LSCM junction-fallback is the validated Levy-2002 solve in
forge.retopo.stitch_kernel.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from collections import defaultdict, deque


# ---------------------------------------------------------------------------
#  submesh + topology
# ---------------------------------------------------------------------------
def submesh(V, F, face_ids):
    """Local submesh for a set of faces. Returns (Vloc, Floc, used_global)."""
    f = F[face_ids]
    used = np.unique(f)
    remap = -np.ones(V.shape[0], np.int64)
    remap[used] = np.arange(len(used))
    return V[used].copy(), remap[f].astype(np.int64), used


def boundary_loops(Floc, n):
    """Ordered boundary vertex loops of a triangle submesh (local indices).

    A boundary edge belongs to exactly one face. Returns list of loops, each a
    list of vertex indices in traversal order (open polylines closed implicitly).
    """
    ecount = defaultdict(int)
    for tri in Floc:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = (a, b) if a < b else (b, a)
            ecount[e] += 1
    bedges = [e for e, c in ecount.items() if c == 1]
    if not bedges:
        return []
    adj = defaultdict(list)
    for a, b in bedges:
        adj[a].append(b); adj[b].append(a)
    loops, seen = [], set()
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxts = [x for x in adj[cur] if x != prev]
            if not nxts:
                break
            nxt = nxts[0]
            if nxt == start:
                break
            if nxt in seen:
                break
            loop.append(nxt); seen.add(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(loop)
    return loops


# ---------------------------------------------------------------------------
#  cotangent Laplacian + harmonic solve
# ---------------------------------------------------------------------------
def cotan_laplacian(V, F):
    """Cotangent Laplacian L (n x n), L[i,i]=sum w, L[i,j]=-w_ij. Robust:
    clamps tiny areas; falls back to combinatorial weights where cot blows up."""
    n = len(V)
    I, J, W = [], [], []
    for tri in F:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        vi, vj, vk = V[i], V[j], V[k]
        for (a, b, c) in ((i, j, k), (j, k, i), (k, i, j)):
            # angle at c, opposite edge (a,b) -> weight on edge (a,b)
            u = V[a] - V[c]; w = V[b] - V[c]
            cross = np.linalg.norm(np.cross(u, w))
            if cross < 1e-18:
                cot = 0.0
            else:
                cot = float(np.dot(u, w)) / cross
            cot = np.clip(cot, -1e4, 1e4)
            I += [a, b]; J += [b, a]; W += [0.5 * cot, 0.5 * cot]
    Woff = sp.csr_matrix((W, (I, J)), shape=(n, n))
    # symmetrize accumulation, build L = D - W
    Woff = 0.5 * (Woff + Woff.T)
    d = np.asarray(Woff.sum(axis=1)).ravel()
    L = sp.diags(d) - Woff
    return L.tocsr()


def harmonic(V, F, fixed_idx, fixed_val):
    """Solve the Dirichlet harmonic problem L u = 0, u[fixed]=val. Returns u (n,).
    Falls back to a combinatorial (graph) Laplacian if the cotan solve fails."""
    n = len(V)
    fixed_idx = np.asarray(fixed_idx, np.int64)
    fixed_val = np.asarray(fixed_val, float)
    free = np.setdiff1d(np.arange(n), fixed_idx)
    for L in (cotan_laplacian(V, F), _graph_laplacian(V, F)):
        try:
            A = L[free][:, free]
            b = -L[free][:, fixed_idx] @ fixed_val
            uf = spla.spsolve(A.tocsc(), b)
            if np.all(np.isfinite(uf)):
                u = np.empty(n); u[fixed_idx] = fixed_val; u[free] = uf
                return u
        except Exception:
            continue
    u = np.zeros(n); u[fixed_idx] = fixed_val
    return u


def _graph_laplacian(V, F):
    n = len(V); I, J = [], []
    for tri in F:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            I += [a, b]; J += [b, a]
    W = sp.csr_matrix((np.ones(len(I)), (I, J)), shape=(n, n))
    W.data[:] = 1.0
    d = np.asarray(W.sum(1)).ravel()
    return (sp.diags(d) - W).tocsr()


def graph_dist_from(V, F, sources):
    """BFS hop+euclidean distance from a set of source vertices (for tip-finding)."""
    n = len(V)
    adj = defaultdict(list)
    for tri in F:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    d = np.full(n, np.inf); dq = deque()
    for s in sources:
        d[s] = 0.0; dq.append(s)
    while dq:
        x = dq.popleft()
        for y in adj[x]:
            nd = d[x] + np.linalg.norm(V[x] - V[y])
            if nd < d[y]:
                d[y] = nd; dq.append(y)
    d[~np.isfinite(d)] = 0.0
    return d


# ---------------------------------------------------------------------------
#  circumferential coordinate (v) about the part axis
# ---------------------------------------------------------------------------
def _cut_normal(P):
    """Best-fit-plane normal of a cut loop (smallest-variance SVD direction)."""
    Q = P - P.mean(0)
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    return Vt[2]


def _medial_deviation(V, u, c0, axis, clen, n=10):
    """Max perpendicular deviation of the (heavily smoothed) u-ring medial curve
    from the straight chord, normalized by chord length. ~0 for a straight axis
    (even a flaring revolution body); large for a genuinely bent limb. Robust to
    centroid noise via wide Gaussian-in-u smoothing."""
    u = np.asarray(u, float)
    umin, umax = float(u.min()), float(u.max())
    if umax - umin < 1e-9:
        return 0.0
    sig = 1.5 * (umax - umin) / n
    dev = 0.0
    for s in np.linspace(umin, umax, n):
        w = np.exp(-((u - s) / sig) ** 2)
        c = (w[:, None] * V).sum(0) / (w.sum() + 1e-12)
        d = c - c0
        perp = d - (d @ axis) * axis        # component off the chord line
        dev = max(dev, float(np.linalg.norm(perp)))
    return dev / clen


def part_axis(V):
    """Principal (elongation) axis via PCA. Returns (axis, centroid, e1, e2)."""
    c = V.mean(0); Q = V - c
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    axis = Vt[0]
    e1, e2 = Vt[1], Vt[2]
    return axis, c, e1, e2


def angle_coord(V, axis, c, e1, e2):
    """Periodic circumferential coordinate v in [0,1) about a SINGLE straight axis.
    Adequate only for straight axisymmetric parts; folds on bent limbs. Kept as a
    fallback; prefer medial_frame_v."""
    Q = V - c
    x = Q @ e1; y = Q @ e2
    return (np.arctan2(y, x) / (2 * np.pi)) % 1.0


def medial_frame_v(V, F, u, n_rings=28, straight_thr=0.93):
    """Circumferential coordinate v in [0,1), choosing the right construction.

    Builds the part's MEDIAL curve (u-ring centroids). If the medial curve is
    nearly STRAIGHT (chord/arclen > straight_thr), the part is a surface-of-
    revolution-like body and a single-axis angle is exact and twist-free (medial
    parallel-transport would only inject spurious twist on a flaring vase -- the
    measured 85deg->66deg regression). If the medial curve BENDS (a limb), use the
    parallel-transported per-ring frame so v follows the bend without folding
    (the panel's #1 fix). Returns (v, intrinsic). Pure numpy."""
    u = np.asarray(u, float)
    umin, umax = float(u.min()), float(u.max())
    diag = float(np.linalg.norm(V.max(0) - V.min(0))) + 1e-12
    if umax - umin < 1e-9:
        ax, c, e1, e2 = part_axis(V)
        return angle_coord(V, ax, c, e1, e2), False
    sig = (umax - umin) / n_rings
    s_samp = np.linspace(umin, umax, n_rings)
    C = np.empty((n_rings, 3))                      # smooth medial curve
    for r, s in enumerate(s_samp):
        w = np.exp(-((u - s) / (sig + 1e-12)) ** 2)
        C[r] = (w[:, None] * V).sum(0) / (w.sum() + 1e-12)
    spread = np.linalg.norm(C - C.mean(0), axis=1).max()
    if spread < 1e-3 * diag:                        # near-spherical: medial degenerate
        ax, c, e1, e2 = part_axis(V)
        return angle_coord(V, ax, c, e1, e2), False
    chord = float(np.linalg.norm(C[-1] - C[0]))
    arclen = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum()) + 1e-12
    straightness = chord / arclen
    if straightness > straight_thr:                 # STRAIGHT -> single axis (no twist)
        axis = (C[-1] - C[0]); axis /= (np.linalg.norm(axis) + 1e-12)
        e1, e2 = _basis_perp(axis)
        return angle_coord(V, axis, C.mean(0), e1, e2), True
    # BENT -> parallel-transported per-ring frame
    T = np.gradient(C, axis=0)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    E1 = np.empty((n_rings, 3)); E1[0] = _basis_perp(T[0])[0]
    for r in range(1, n_rings):
        E1[r] = _rotate_between(T[r - 1], T[r], E1[r - 1])
        E1[r] -= (E1[r] @ T[r]) * T[r]
        E1[r] /= np.linalg.norm(E1[r]) + 1e-12
    E2 = np.cross(T, E1)
    idx = np.clip(np.searchsorted(s_samp, u) - 1, 0, n_rings - 2)
    f = (u - s_samp[idx]) / (s_samp[idx + 1] - s_samp[idx] + 1e-12)
    c_i = C[idx] * (1 - f[:, None]) + C[idx + 1] * f[:, None]
    e1_i = E1[idx] * (1 - f[:, None]) + E1[idx + 1] * f[:, None]
    e2_i = E2[idx] * (1 - f[:, None]) + E2[idx + 1] * f[:, None]
    Q = V - c_i
    x = (Q * e1_i).sum(1); y = (Q * e2_i).sum(1)
    return (np.arctan2(y, x) / (2 * np.pi)) % 1.0, True


def _rotate_between(a, b, vec):
    """Rotate `vec` by the minimal rotation taking unit a -> unit b (Rodrigues)."""
    a = a / (np.linalg.norm(a) + 1e-12); b = b / (np.linalg.norm(b) + 1e-12)
    ax = np.cross(a, b); s = np.linalg.norm(ax); cth = float(np.dot(a, b))
    if s < 1e-9:
        return vec if cth > 0 else -vec
    ax = ax / s
    return (vec * cth + np.cross(ax, vec) * s + ax * (ax @ vec) * (1 - cth))


# ---------------------------------------------------------------------------
#  per-part parameterization
# ---------------------------------------------------------------------------
def param_part(V, F, loops):
    """Dispatch by #boundary loops. Returns (u, v, kind, info).

    u : harmonic axial/radial coordinate (0 at cut(s)).
    v : periodic circumferential coordinate in [0,1).
    kind: 'cap' | 'tube' | 'junction' | 'closed'.
    """
    n = len(V); nb = len(loops)
    axis, c, e1, e2 = part_axis(V)
    if nb == 1:
        # CAP: u = 0 on the cut, 1 at the farthest interior vertex (the tip).
        bnd = np.asarray(loops[0], np.int64)
        dist = graph_dist_from(V, F, bnd)
        tip = int(np.argmax(dist))
        u = harmonic(V, F, np.append(bnd, tip), np.append(np.zeros(len(bnd)), 1.0))
        v, intrinsic = medial_frame_v(V, F, u)   # surface-following (fixes bent-limb fold)
        return u, v, "cap", {"tip": tip, "loops": 1, "v_intrinsic": intrinsic}
    if nb == 2:
        # TUBE: u = 0 on loop A, 1 on loop B (axial).
        A = np.asarray(loops[0], np.int64); B = np.asarray(loops[1], np.int64)
        u = harmonic(V, F, np.concatenate([A, B]),
                     np.concatenate([np.zeros(len(A)), np.ones(len(B))]))
        # Decide straight vs bent from the CUT-PLANE NORMALS (clean boundary loops;
        # the medial-centroid curve is too noisy on small meshes). Straight tube =
        # both cut normals parallel to the chord -> a single global axis is exact
        # (and twist-free). Bent tube -> diverging normals -> medial frame.
        cA, cB = V[A].mean(0), V[B].mean(0)
        chord = cB - cA; clen = np.linalg.norm(chord) + 1e-12; chord /= clen
        nA = _cut_normal(V[A]) * np.sign(_cut_normal(V[A]) @ chord + 1e-12)
        nB = _cut_normal(V[B]) * np.sign(_cut_normal(V[B]) @ chord + 1e-12)
        bend = np.degrees(np.arccos(np.clip(nA @ nB, -1, 1)))
        mdev = _medial_deviation(V, u, cA, chord, clen)   # axis-curvature, robust
        # BENT only if BOTH the cut normals diverge AND the medial axis genuinely
        # curves -> avoids the false-positive on a flaring/wavy revolution neck
        # (diverging normals but a straight axis).
        if bend >= 30.0 and mdev > 0.12:         # BENT
            v, intrinsic = medial_frame_v(V, F, u)
        else:                                    # STRAIGHT (revolution-like)
            e1, e2 = _basis_perp(chord)
            v = angle_coord(V, chord, 0.5 * (cA + cB), e1, e2); intrinsic = True
        return u, v, "tube", {"loops": 2, "bend": float(bend),
                              "mdev": float(mdev), "v_intrinsic": intrinsic}
    if nb == 0:
        # CLOSED component (no cut): treat like a sphere -> two poles.
        d0 = graph_dist_from(V, F, [0])
        p1 = int(np.argmax(d0))
        d1 = graph_dist_from(V, F, [p1])
        p0 = int(np.argmax(d1))
        u = harmonic(V, F, [p0, p1], [0.0, 1.0])
        axis = V[p1] - V[p0]; axis = axis / (np.linalg.norm(axis) + 1e-12)
        e1, e2 = _basis_perp(axis)
        v = angle_coord(V, axis, V.mean(0), e1, e2)
        return u, v, "closed", {"poles": (p0, p1)}
    # JUNCTION (k>=3): u = harmonic 0 on all cuts, 1 at interior max; v = LSCM-ish
    # angle. v1 flags this; the grid will carry the topologically-required
    # irregular vertices at the branch.
    allb = np.concatenate([np.asarray(l, np.int64) for l in loops])
    dist = graph_dist_from(V, F, allb)
    core = int(np.argmax(dist))
    u = harmonic(V, F, np.append(allb, core), np.append(np.zeros(len(allb)), 1.0))
    v = angle_coord(V, axis, c, e1, e2)
    return u, v, "junction", {"loops": nb, "core": core}


def _basis_perp(axis):
    """Two orthonormal vectors spanning the plane perpendicular to `axis`."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = t - (t @ a) * a; e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(a, e1)
    return e1, e2


# ---------------------------------------------------------------------------
#  grid extraction (integer lattice in (u,v) -> 3D)
# ---------------------------------------------------------------------------
def _uv_interpolator(V, F, u, v):
    """Build a robust (u,v)->3D interpolator over the part, with the periodic v
    seam duplicated on both sides. Returns (interp, uvtree, pts, V2)."""
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import cKDTree
    loC = v < 0.5; hiC = v >= 0.5
    u2 = np.concatenate([u, u[loC], u[hiC]])
    v2 = np.concatenate([v, v[loC] + 1.0, v[hiC] - 1.0])
    V2 = np.vstack([V, V[loC], V[hiC]])
    pts = np.column_stack([u2, v2])
    interp = LinearNDInterpolator(pts, V2)   # Qhull Delaunay; robust to polar slivers
    return interp, cKDTree(pts), pts, V2


def _sample_grid(interp, uvtree, V2, Nu, Nv, pole, vshift=0.0):
    """Sample the integer (u,v) lattice -> (Nu+1, Nv, 3), NN-fallback for hull
    misses. vshift rotates the angular origin (phase alignment for welding)."""
    eps = 1e-3
    uu = np.linspace(eps, 1 - eps, Nu + 1)
    vv = (np.linspace(0, 1, Nv, endpoint=False) + vshift) % 1.0
    UU, VV = np.meshgrid(uu, vv, indexing="ij")
    Q = np.column_stack([UU.ravel(), VV.ravel()])
    P = interp(Q)
    bad = ~np.all(np.isfinite(P), axis=1)
    if bad.any():
        _, nn = uvtree.query(Q[bad])
        P[bad] = V2[nn]
    return P.reshape(Nu + 1, Nv, 3)


def build_grid(V, F, u, v, kind, Nu, Nv):
    """Sample the integer (u,v) lattice back to 3D via linear interpolation over
    the (u,v) triangulation. Returns (Vq, quads, valence, ok).

    Nu = rings along u (0..1), Nv = sectors around v (periodic). For 'cap'/'closed'
    the u=1 end collapses to a pole; for 'tube' both ends are open rings.
    """
    try:
        interp, uvtree, pts, V2 = _uv_interpolator(V, F, u, v)
    except Exception:
        return None, None, None, False
    pole = kind in ("cap", "closed")
    grid = _sample_grid(interp, uvtree, V2, Nu, Nv, pole)
    # assemble vertices + quads
    Vq, quads, vid = [], [], {}
    def vert(i, j):
        j %= Nv
        key = (i, j) if not (pole and i == Nu) else ("pole",)
        if key not in vid:
            vid[key] = len(Vq)
            Vq.append(grid[i, j] if key != ("pole",) else np.nanmean(grid[Nu], 0))
        return vid[key]
    top = Nu if not pole else Nu - 1
    for i in range(top):
        for j in range(Nv):
            a, b, c, d = vert(i, j), vert(i, j + 1), vert(i + 1, j + 1), vert(i + 1, j)
            quads.append([a, b, c, d])
    if pole:
        pj = vert(Nu, 0)  # the single pole vertex
        for j in range(Nv):
            a, b = vert(Nu - 1, j), vert(Nu - 1, j + 1)
            quads.append([a, b, pj, pj])  # degenerate-as-triangle fan at pole
    Vq = np.array(Vq); quads = np.array(quads, np.int64)
    val = _valence(quads, len(Vq))
    return Vq, quads, val, True


def _seam_faces(F, v, nV, seam):
    """Faces using duplicated seam vertices so triangles spanning v~0/1 don't
    wrap the long way. A face straddles if its v-range exceeds 0.5."""
    dup = {int(g): nV + i for i, g in enumerate(np.flatnonzero(seam))}
    out = []
    for tri in F:
        vs = v[tri]
        if vs.max() - vs.min() > 0.5:
            newtri = [dup.get(int(t), int(t)) if v[int(t)] < 0.5 else int(t) for t in tri]
            out.append(newtri)
        else:
            out.append([int(t) for t in tri])
    return np.array(out, np.int64)


def _valence(quads, n):
    val = np.zeros(n, np.int64)
    edges = defaultdict(set)
    for q in quads:
        qq = [x for x in q]
        for a, b in zip(qq, qq[1:] + qq[:1]):
            if a != b:
                edges[a].add(b); edges[b].add(a)
    for i in range(n):
        val[i] = len(edges[i])
    return val


# ---------------------------------------------------------------------------
#  metrics
# ---------------------------------------------------------------------------
def scaled_jacobian(Vq, quads):
    """Min/mean scaled Jacobian over faces (1 = right-angled corner; ->0 = sliver).
    A face stored with a REPEATED vertex ([a,b,c,c]) is a genuine TRIANGLE (from an
    odd-loop pinwheel cap) and is measured as a triangle, NOT as a zero-area quad
    -- so the metric reports real geometry, not the storage artifact."""
    sj = []
    for q in quads:
        verts = [int(x) for x in q]
        uniq = list(dict.fromkeys(verts))            # unique, order-preserving
        n = len(uniq)
        if n < 3:
            continue                                  # truly degenerate -> skip
        P = Vq[uniq]
        cs = []
        for t in range(n):
            e1 = P[(t + 1) % n] - P[t]
            e2 = P[(t - 1) % n] - P[t]
            a, b = np.linalg.norm(e1), np.linalg.norm(e2)
            if a < 1e-12 or b < 1e-12:
                continue
            cs.append(np.linalg.norm(np.cross(e1, e2)) / (a * b))
        if cs:
            sj.append(min(cs))
    sj = np.array(sj) if sj else np.array([0.0])
    return float(sj.min()), float(sj.mean())


def edge_cv(Vq, quads):
    lens = []
    for q in quads:
        qq = list(q)
        for a, b in zip(qq, qq[1:] + qq[:1]):
            if a != b:
                lens.append(np.linalg.norm(Vq[a] - Vq[b]))
    lens = np.array([x for x in lens if x > 1e-9])
    return float(lens.std() / (lens.mean() + 1e-12)) if len(lens) else float("nan")


def irregular_fraction(val, boundary_mask=None):
    """Fraction of INTERIOR vertices with valence != 4."""
    interior = np.ones(len(val), bool)
    if boundary_mask is not None:
        interior = ~boundary_mask
    interior &= (val > 0)
    if interior.sum() == 0:
        return 0.0, 0
    irr = (val[interior] != 4)
    return float(irr.mean()), int(irr.sum())


def fidelity(mesh, Vq, scale):
    """One-sided max + RMS distance from quad vertices to the input surface / scale."""
    try:
        cp, dist, _ = mesh.nearest.on_surface(Vq)
        return float(dist.max() / scale), float(np.sqrt((dist ** 2).mean()) / scale)
    except Exception:
        return float("nan"), float("nan")


def project_to_surface(mesh, Vq):
    """Snap grid vertices onto the input surface (the LinearND interpolator chords
    the surface; this removes the chord error). Returns projected copy."""
    try:
        cp, _, _ = mesh.nearest.on_surface(Vq)
        return np.asarray(cp, float)
    except Exception:
        return Vq


def perpendicularity(Vq, quads, cut_pts):
    """Median angle (deg) between u-family grid edges touching a cut and the cut
    tangent. ~90 = perpendicular (the PI's requirement). cut_pts: (N,3) shared
    boundary ring. Returns (u_perp_deg, v_para_deg)."""
    from scipy.spatial import cKDTree
    ct = cKDTree(cut_pts)
    # cut tangents at each ring point
    tang = np.roll(cut_pts, -1, 0) - np.roll(cut_pts, 1, 0)
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)
    u_ang, v_ang = [], []
    for q in quads:
        P = Vq[q]
        for t in range(4):
            a, b = q[t], q[(t + 1) % 4]
            if a == b:
                continue
            mid = 0.5 * (Vq[a] + Vq[b])
            d, k = ct.query(mid)
            if d > 0.25 * np.linalg.norm(np.ptp(cut_pts, axis=0)):
                continue
            e = Vq[b] - Vq[a]; ne = np.linalg.norm(e)
            if ne < 1e-12:
                continue
            ang = np.degrees(np.arccos(np.clip(abs(np.dot(e / ne, tang[k])), 0, 1)))
            # edges near the cut: classify as along-cut (small angle) or across
            (v_ang if ang < 45 else u_ang).append(ang)
    um = float(np.median(u_ang)) if u_ang else float("nan")
    vm = float(np.median(v_ang)) if v_ang else float("nan")
    return um, vm


def largest_component_faces(mesh, face_ids):
    """Keep only the largest connected face-component within a label region and
    drop slivers (fixes disconnected label regions, panel finding c)."""
    fset = set(int(x) for x in face_ids)
    fadj = np.asarray(mesh.face_adjacency, np.int64)
    nbr = defaultdict(list)
    for a, b in fadj:
        a, b = int(a), int(b)
        if a in fset and b in fset:
            nbr[a].append(b); nbr[b].append(a)
    seen, comps = set(), []
    for f in fset:
        if f in seen:
            continue
        comp, dq = [], deque([f]); seen.add(f)
        while dq:
            x = dq.popleft(); comp.append(x)
            for y in nbr[x]:
                if y not in seen:
                    seen.add(y); dq.append(y)
        comps.append(comp)
    if not comps:
        return np.asarray(list(fset), np.int64)
    return np.asarray(max(comps, key=len), np.int64)


def _loop_len(V, loop):
    P = V[np.asarray(loop, np.int64)]
    return float(np.linalg.norm(P - np.roll(P, 1, 0), axis=1).sum())


def grid_part_metrics(mesh, V, F, u, v, kind, h, scale, loops=None):
    """Full per-part metric bundle at target edge length h. Nv is driven by the
    actual cut-loop perimeter (= the shared-N stitch quantity); Nu by the
    axial/radial physical span of the u=0..1 sweep."""
    # circumference to subdivide = mean boundary-loop length (the stitch N)
    if loops:
        perim = float(np.mean([_loop_len(V, l) for l in loops]))
    else:
        perim = _level_perimeter(V, F, u, 1e-3) or (2 * np.pi * scale * 0.2)
    height = _u_height(V, F, u) or (0.3 * scale)
    Nu = max(2, int(round(height / h)))
    Nv = max(6, int(round(perim / h)))
    Nv += Nv % 2  # even
    Vq, quads, val, ok = build_grid(V, F, u, v, kind, Nu, Nv)
    if not ok or Vq is None or len(quads) == 0:
        return {"kind": kind, "ok": False, "Nu": Nu, "Nv": Nv}
    sj_min, sj_mean = scaled_jacobian(Vq, quads)
    cv = edge_cv(Vq, quads)
    bmask = _boundary_verts(quads, len(Vq))
    irr_frac, irr_n = irregular_fraction(val, bmask)
    fmax, frms = fidelity(mesh, Vq, scale)
    # Poincare-Hopf excess: topological minimum irregulars for this kind
    ph_min = {"cap": 1, "tube": 0, "closed": 2, "junction": None}[kind]
    return {"kind": kind, "ok": True, "Nu": Nu, "Nv": Nv,
            "n_quads": int(len(quads)), "sj_min": sj_min, "sj_mean": sj_mean,
            "edge_cv": cv, "irr_frac": irr_frac, "irr_n": irr_n, "ph_min": ph_min,
            "fid_max": fmax, "fid_rms": frms, "Vq": Vq, "quads": quads, "val": val}


def _u_height(V, F, u):
    """Physical length of the u=0..1 sweep ~ |grad u|^-1 integrated; estimate as
    mean distance between u~0 and u~1 vertices."""
    lo = V[u < 0.1]; hi = V[u > 0.9]
    if len(lo) and len(hi):
        return float(np.linalg.norm(hi.mean(0) - lo.mean(0)))
    return None


def _level_perimeter(V, F, u, lvl):
    """Approximate perimeter of the u~lvl iso-loop = a representative ring length."""
    band = V[np.abs(u - 0.5) < 0.15]
    if len(band) < 3:
        return None
    c = band.mean(0)
    r = np.linalg.norm(band - c, axis=1).mean()
    return float(2 * np.pi * r)


def _boundary_verts(quads, n):
    cnt = defaultdict(int)
    for q in quads:
        qq = list(q)
        for a, b in zip(qq, qq[1:] + qq[:1]):
            if a != b:
                e = (a, b) if a < b else (b, a)
                cnt[e] += 1
    bm = np.zeros(n, bool)
    for (a, b), c in cnt.items():
        if c == 1:
            bm[a] = True; bm[b] = True
    return bm


# ---------------------------------------------------------------------------
#  multi-part: cut graph, shared-N stitch, pinned welding
# ---------------------------------------------------------------------------
def partition_junction(V, F, loops):
    """Split a k-boundary JUNCTION into k territories by harmonic dominance.

    For each cut loop i solve h_i = harmonic(1 on L_i, 0 on all other loops); the
    territory of a face = argmax_i h_i. Each territory touches exactly one original
    socket and meets its neighbours along internal seams that join at the branch
    triple-points -- so a territory is a TUBE (socket loop + internal core loop) or
    a CAP, which the per-part gridder + shared-cut weld already handle. The triple
    points become the topologically-required |4(2-k)| irregular vertices, AT the
    branch (where good hand-retopo also puts them). Returns per-face labels 0..k-1.
    """
    k = len(loops)
    H = np.zeros((k, len(V)))
    allb = np.concatenate([np.asarray(l, np.int64) for l in loops])
    for i, li in enumerate(loops):
        vals = np.zeros(len(allb))
        off = 0
        for j, lj in enumerate(loops):
            vals[off:off + len(lj)] = 1.0 if j == i else 0.0
            off += len(lj)
        H[i] = harmonic(V, F, allb, vals)
    face_h = H[:, F].mean(axis=2)            # (k, nF)
    terr = np.argmax(face_h, axis=0).astype(np.int64)
    return terr


def _trace_loops(edges):
    """Order a set of undirected edges into vertex loops (handles several)."""
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    loops, seen_e = [], set()
    for s in list(adj):
        for nb0 in adj[s]:
            e0 = (s, nb0) if s < nb0 else (nb0, s)
            if e0 in seen_e:
                continue
            loop = [s]; prev, cur = s, nb0
            seen_e.add(e0); loop.append(cur)
            while cur != s:
                nxts = [x for x in adj[cur] if x != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                e = (cur, nxt) if cur < nxt else (nxt, cur)
                if e in seen_e:
                    break
                seen_e.add(e); prev, cur = cur, nxt
                if cur != s:
                    loop.append(cur)
            if len(loop) >= 3:
                loops.append(loop)
    return loops


def cut_graph(mesh, labels):
    """Build the inter-part cut structure from face adjacency + labels.

    Returns:
      cuts      : list of dicts {parts: frozenset{l1,l2}, loop: [global verts],
                  perim: float}
      part_cuts : dict label -> list of cut indices touching it
      free_loops: dict label -> list of mesh-boundary loops (open-mesh edges)
    """
    F = np.asarray(mesh.faces, np.int64)
    V = np.asarray(mesh.vertices, float)
    fadj = np.asarray(mesh.face_adjacency, np.int64)
    fae = np.asarray(mesh.face_adjacency_edges, np.int64)
    pair_edges = defaultdict(list)
    for (f1, f2), (a, b) in zip(fadj, fae):
        l1, l2 = int(labels[f1]), int(labels[f2])
        if l1 != l2:
            pair_edges[frozenset((l1, l2))].append((int(a), int(b)))
    cuts, part_cuts = [], defaultdict(list)
    for pair, edges in pair_edges.items():
        for loop in _trace_loops(set(tuple(sorted(e)) for e in edges)):
            P = V[loop]
            perim = float(np.linalg.norm(P - np.roll(P, 1, 0), axis=1).sum())
            ci = len(cuts)
            cuts.append({"parts": pair, "loop": loop, "perim": perim})
            for l in pair:
                part_cuts[l].append(ci)
    # mesh-boundary (open) loops, per part
    ecount = defaultdict(list)
    for tri in F:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = (a, b) if a < b else (b, a)
            ecount[e].append(0)
    return cuts, part_cuts


def assign_shared_N(cuts, kind_of_part, part_cuts, h):
    """Assign each cut an integer subdivision N=round(perim/h), with TUBE parts
    forcing their two cuts to equal N (union-find). Returns N_of_cut: list[int]."""
    nC = len(cuts)
    par = list(range(nC))
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for lbl, cis in part_cuts.items():
        if kind_of_part.get(lbl) == "tube" and len(cis) == 2:
            a, b = find(cis[0]), find(cis[1])
            if a != b:
                par[max(a, b)] = min(a, b)
    # per-class mean perimeter -> N
    cls = defaultdict(list)
    for ci in range(nC):
        cls[find(ci)].append(ci)
    N_of_cut = [0] * nC
    for root, members in cls.items():
        Lbar = float(np.mean([cuts[ci]["perim"] for ci in members]))
        N = max(6, int(round(Lbar / h)))
        N += N % 2
        for ci in members:
            N_of_cut[ci] = N
    return N_of_cut


def resample_loop(V, loop, N):
    """N equal-arc-length points around the closed polyline `loop` (global verts).
    Returns (N,3) ordered CCW in the loop's best-fit plane."""
    P = V[np.asarray(loop, np.int64)]
    # order CCW in best-fit plane for a canonical start/direction
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    e1, e2, nrm = Vt[0], Vt[1], Vt[2]
    ang = np.arctan2((P - c) @ e2, (P - c) @ e1)
    order = np.argsort(ang)
    P = P[order]
    seg = np.linalg.norm(P - np.roll(P, -1, 0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    targets = (np.arange(N) / N) * total
    Pcl = np.vstack([P, P[0]])
    out = np.empty((N, 3))
    for i, t in enumerate(targets):
        k = np.searchsorted(s, t, side="right") - 1
        k = min(max(k, 0), len(P) - 1)
        f = (t - s[k]) / (seg[k] + 1e-12)
        out[i] = Pcl[k] * (1 - f) + Pcl[k + 1] * f
    return out


def _resample_pts(P, N):
    """Resample a closed sequence of 3D points to exactly N points (equal index
    spacing, linear interp). Handles a tube whose two cuts ended up with different
    N (a boundary loop split between two neighbour labels)."""
    P = np.asarray(P, float); M = len(P)
    if M == N:
        return P
    t = (np.arange(N) / N) * M
    k = np.floor(t).astype(int) % M; f = (t - np.floor(t))[:, None]
    return P[k] * (1 - f) + P[(k + 1) % M] * f


def _align_ring_to(ref, pts):
    """Reorder `pts` (cyclic shift + optional reversal) to best match `ref`
    point-for-point in 3D. Fixes the tube seam-phase Dj + orientation flip."""
    N = len(ref)
    pts = _resample_pts(pts, N)
    best = (np.inf, pts)
    for d in (1, -1):
        cand = pts[(np.arange(N) * d) % N]
        for s in range(N):
            rolled = np.roll(cand, -s, axis=0)
            cost = float(np.linalg.norm(ref - rolled, axis=1).sum())
            if cost < best[0]:
                best = (cost, rolled)
    return best[1]


def build_grid_pinned(V, F, u, v, kind, Nu, pin_rings):
    """Grid with boundary ring(s) PINNED to shared cut points (watertight weld).

    pin_rings: dict end->{'pts': (N,3)} where end=0 is the u~0 boundary and end=1
    the u~1 boundary (tube only). N (=len pts) sets the circumferential count Nv;
    interior columns are angle-aligned to the pinned points so the ring connects
    with minimal shear. Returns (Vq, quads, val, ok, ring_vids).
    """
    pts0 = pin_rings[0]["pts"]
    Nv = len(pts0)
    try:
        interp, uvtree, ptsuv, V2 = _uv_interpolator(V, F, u, v)
    except Exception:
        return None, None, None, False, {}
    pole = kind in ("cap", "closed")
    from scipy.spatial import cKDTree
    vtree = cKDTree(V)
    # IMPORTANT (twist fix): the ring-0 column order is taken AS GIVEN by the
    # caller, NOT re-sorted by this part's own v. The caller propagates ONE
    # consistent column order along the part chain (phase propagation), so adjacent
    # parts agree on which column is the v=0 meridian -> no rotational jump at the
    # seam. Interior columns are sampled at the v-value of each given ring-0 point.
    _, nn0 = vtree.query(pts0)
    vcol = v[nn0]                                   # angular coord per column (given order)
    grid = np.empty((Nu + 1, Nv, 3))
    uu = np.linspace(1e-3, 1 - 1e-3, Nu + 1)
    Qrows = []
    for ui in uu:
        Q = np.column_stack([np.full(Nv, ui), vcol])
        P = interp(Q)
        bad = ~np.all(np.isfinite(P), axis=1)
        if bad.any():
            _, nn = uvtree.query(Q[bad]); P[bad] = V2[nn]
        Qrows.append(P)
    grid = np.array(Qrows)                          # (Nu+1, Nv, 3)
    grid[0] = pts0                                  # PIN ring 0 (weld, given order)
    ring_vids = {}
    if not pole and 1 in pin_rings:
        pts1 = pin_rings[1]["pts"]
        if pin_rings[1].get("fixed"):
            grid[Nu] = _resample_pts(pts1, Nv)       # order fixed by a parent (cycle)
        else:
            # align ring-1 to ring-0 (cyclic shift + reflection) so the tube does
            # not spiral internally; this DEFINES ring-1's order, which the caller
            # then propagates to the next part across that cut.
            grid[Nu] = _align_ring_to(pts0, pts1)
    # assemble verts/quads
    Vq, quads, vid = [], [], {}
    def vert(i, j):
        j %= Nv
        key = ("pole",) if (pole and i == Nu) else (i, j)
        if key not in vid:
            vid[key] = len(Vq)
            Vq.append(np.nanmean(grid[Nu], 0) if key == ("pole",) else grid[i, j])
        return vid[key]
    top = Nu if not pole else Nu - 1
    for i in range(top):
        for j in range(Nv):
            quads.append([vert(i, j), vert(i, j + 1), vert(i + 1, j + 1), vert(i + 1, j)])
    if pole:
        pj = vert(Nu, 0)
        for j in range(Nv):
            quads.append([vert(Nu - 1, j), vert(Nu - 1, j + 1), pj, pj])
    Vq = np.array(Vq); quads = np.array(quads, np.int64)
    # record the global ring-0 vertex ids (row 0) for welding across parts
    ring0 = [vid[(0, j)] for j in range(Nv)]
    ring_vids[0] = (ring0, pts0)
    if not pole and 1 in pin_rings:
        ring1 = [vid[(Nu, j)] for j in range(Nv)]
        ring_vids[1] = (ring1, grid[Nu])
    return Vq, quads, _valence(quads, len(Vq)), True, ring_vids


def _match_external(ctr, per, external_cuts, tol=0.35):
    """Match a boundary loop (centroid ctr, perimeter per) to an external cut
    passed from a parent call (a junction body's socket = a global limb cut).
    Returns the matching external-cut dict or None."""
    if not external_cuts:
        return None
    best = None
    for ec in external_cuts:
        d = float(np.linalg.norm(ctr - ec["centroid"]))
        if d < tol * ec["perim"] and 0.55 < per / (ec["perim"] + 1e-9) < 1.8:
            if best is None or d < best[0]:
                best = (d, ec)
    return best[1] if best else None


def place_grids_multipart(mesh, labels, h, verbose=True, _depth=0, external_cuts=None,
                          topo_clean=False):
    """Grid every part and weld at shared cuts. Returns a result dict with the
    merged quad mesh + per-part and global metrics."""
    V = np.asarray(mesh.vertices, float); F = np.asarray(mesh.faces, np.int64)
    scale = float(np.linalg.norm(V.max(0) - V.min(0)))
    parts = [int(x) for x in np.unique(labels)]
    cuts, part_cuts = cut_graph(mesh, labels)

    # classify each part by its boundary-loop count
    kind_of, part_data = {}, {}
    for lbl in parts:
        fids = largest_component_faces(mesh, np.flatnonzero(labels == lbl))
        Vloc, Floc, used = submesh(V, F, fids)
        loops = boundary_loops(Floc, len(Vloc))
        nb = len(loops)
        kind = {0: "closed", 1: "cap", 2: "tube"}.get(nb, "junction")
        kind_of[lbl] = kind
        part_data[lbl] = (Vloc, Floc, used, loops, kind)

    N_of_cut = assign_shared_N(cuts, kind_of, part_cuts, h)
    # resample each cut ONCE -> shared 3D points used by BOTH incident parts
    cut_pts = {ci: resample_loop(V, cuts[ci]["loop"], N_of_cut[ci]) for ci in range(len(cuts))}

    # --- BODY-FIRST PRE-PASS: Blossom each junction body; its socket boundaries
    # become the fixed rings the LIMBS conform to. We OVERRIDE cut_pts for each
    # body-limb cut with the body's actual Blossom socket boundary loop, so the
    # limb tube pins to it EXACTLY -> clean weld, no interface ring of irregulars.
    junction_quads = {}
    if _depth == 0:
        import trimesh as _tmj
        from ._blossom import blossom_quad_patch
        for lbl in parts:
            Vloc, Floc, used, loops, kind = part_data[lbl]
            if kind != "junction" or len(loops) < 3:
                continue
            try:
                jm = _tmj.Trimesh(Vloc, Floc, process=False)
                Vb, Qb, binfo = blossom_quad_patch(jm, h)
                Vb = project_to_surface(mesh, Vb)
            except Exception as _be:
                continue
            junction_quads[lbl] = (Vb, Qb)
            my = part_cuts.get(lbl, [])
            if not my:
                continue
            cut_ctr = {ci: V[np.asarray(cuts[ci]["loop"], np.int64)].mean(0) for ci in my}
            for bl in _quad_boundary_loops(Qb, len(Vb)):
                if len(bl) < 4:
                    continue
                ctr = Vb[np.asarray(bl, np.int64)].mean(0)
                ci = min(my, key=lambda c: np.linalg.norm(ctr - cut_ctr[c]))
                if np.linalg.norm(ctr - cut_ctr[ci]) < 0.5 * cuts[ci]["perim"]:
                    clean = _regularize_socket_loop(mesh, Vb, bl)   # smooth+even ring
                    Vb[np.asarray(bl, np.int64)] = clean            # body side (in-place)
                    cut_pts[ci] = clean                             # limb conforms to CLEAN ring

    # build each part's grid. Process in BFS order over the part-adjacency graph
    # and PROPAGATE one consistent column order (the v=0 meridian) across every
    # shared cut, so adjacent parts agree on phase -> no rotational twist at the
    # seams (the vase-twist bug).
    allV, allQ, voff = [], [], 0
    part_metrics, weld_index = {}, defaultdict(list)
    padj = defaultdict(list)
    for ci, c in enumerate(cuts):
        pr = [p for p in c["parts"] if p in part_data]
        if len(pr) == 2:
            padj[pr[0]].append((pr[1], ci)); padj[pr[1]].append((pr[0], ci))
    if hasattr(mesh, "area_faces"):
        part_area = {l: float(mesh.area_faces[np.flatnonzero(labels == l)].sum()) for l in parts}
    else:
        part_area = {l: 1.0 for l in parts}
    order_bfs, seen = [], set()
    for seed in sorted(parts, key=lambda l: -part_area.get(l, 0)):
        if seed in seen:
            continue
        seen.add(seed); dq = deque([seed])
        while dq:
            x = dq.popleft(); order_bfs.append(x)
            for nb, ci in padj.get(x, []):
                if nb not in seen:
                    seen.add(nb); dq.append(nb)
    cut_order = {}                       # ci -> column-ordered (N,3) pts (propagated meridian)

    for lbl in order_bfs:
        Vloc, Floc, used, loops, kind = part_data[lbl]
        u, v, k2, info = param_part(Vloc, Floc, loops)
        my_cuts = part_cuts.get(lbl, [])
        # Pinnable boundaries = internal cuts + EXTERNAL-cut matches (a junction
        # body's socket loops matched to the global limb cuts passed from the
        # parent -> the body territories weld to the limbs, closing the gaps).
        pinnable = [{"pts": cut_order.get(ci, cut_pts[ci]), "perim": cuts[ci]["perim"],
                     "fixed": ci in cut_order, "cut": ci} for ci in my_cuts]
        if external_cuts:
            int_ctrs = [V[np.asarray(cuts[ci]["loop"], np.int64)].mean(0) for ci in my_cuts]
            for loop in loops:
                gpts = Vloc[np.asarray(loop, np.int64)]; ctr = gpts.mean(0)
                per = float(np.linalg.norm(gpts - np.roll(gpts, 1, 0), axis=1).sum())
                if any(np.linalg.norm(ctr - ic) < 0.25 * (per + 1e-9) for ic in int_ctrs):
                    continue                                  # already an internal cut
                ext = _match_external(ctr, per, external_cuts)
                if ext is not None:
                    pinnable.append({"pts": ext["pts"], "perim": per, "fixed": True, "cut": -1})
        if kind in ("cap", "tube") and pinnable:
            pin = {}
            pinnable.sort(key=lambda p: -p["perim"])
            # ring 0 = a FIXED boundary (external/inherited) when present, so its N
            # and phase drive the grid and it welds to the neighbour; else largest.
            p0 = next((p for p in pinnable if p["fixed"]), pinnable[0])
            rest = [p for p in pinnable if p is not p0]
            pin[0] = {"pts": p0["pts"], "cut": p0["cut"], "fixed": p0["fixed"]}
            if kind == "tube" and rest:
                pin[1] = {"pts": rest[0]["pts"], "cut": rest[0]["cut"], "fixed": rest[0]["fixed"]}
            height = _u_height(Vloc, Floc, u) or (0.3 * scale)
            # axial resolution matches the CIRCUMFERENTIAL spacing (square quads),
            # not the global h -- so a THIN limb (small ring, slivers at h) gets
            # finer rings instead of tall slivers. Clamp the spacing to [0.5h, 2h].
            ring0 = np.asarray(pin[0]["pts"], float)
            circ = float(np.linalg.norm(ring0 - np.roll(ring0, 1, 0), axis=1).sum())
            spacing = np.clip(circ / max(len(ring0), 1), 0.5 * h, 2.0 * h)
            Nu = max(2, int(round(height / spacing)))
            Vq, quads, val, ok, rings = build_grid_pinned(Vloc, Floc, u, v, kind, Nu, pin)
            if not ok:
                part_metrics[lbl] = {"kind": kind, "ok": False}
                continue
            Vq = project_to_surface(mesh, Vq)        # snap interior off the chord
            for end, (rvids, rpts) in rings.items():  # keep weld points EXACT
                Vq[np.asarray(rvids, np.int64)] = rpts
            sj_min, sj_mean = scaled_jacobian(Vq, quads)
            bmask = _boundary_verts(quads, len(Vq))
            irr_frac, irr_n = irregular_fraction(val, bmask)
            uperp, vpar = perpendicularity(Vq, quads, pin[0]["pts"])
            part_metrics[lbl] = {"kind": kind, "ok": True, "n_quads": int(len(quads)),
                                 "sj_min": sj_min, "sj_mean": sj_mean,
                                 "edge_cv": edge_cv(Vq, quads), "irr_n": irr_n,
                                 "Nu": Nu, "Nv": len(pin[0]["pts"]),
                                 "u_perp": uperp, "v_para": vpar}
            # register weld rings + PROPAGATE the agreed column order to neighbours
            for end, (rvids, rpts) in rings.items():
                ci = pin[end]["cut"]
                if ci >= 0 and ci not in cut_order:        # ci=-1 is an external pin
                    cut_order[ci] = np.asarray(rpts)       # this part fixes the meridian here
                if ci >= 0:
                    for local_vid, p in zip(rvids, rpts):
                        weld_index[ci].append((voff + local_vid, tuple(np.round(p, 6))))
            allV.append(Vq); allQ.append(quads + voff); voff += len(Vq)
        elif kind == "junction" and len(loops) >= 3 and lbl in junction_quads:
            # JUNCTION body: emit the Blossom quads computed in the body-first
            # pre-pass (the limbs already conformed to its socket boundaries).
            Vb, Qb = junction_quads[lbl]
            allV.append(Vb); allQ.append(Qb + voff); voff += len(Vb)
            part_metrics[lbl] = {"kind": "junction", "ok": True, "method": "blossom",
                                 "n_quads": int(len(Qb))}
        else:
            part_metrics[lbl] = {"kind": kind, "ok": False, "reason": "closed/nested (v1 TODO)"}

    if not allV:
        return {"ok": False, "cuts": cuts, "kind_of": kind_of, "part_metrics": part_metrics}
    Vall = np.vstack(allV); Qall = np.vstack(allQ)
    # WELD: merge vertices that are the same shared cut point (by rounded position)
    Vall, Qall, n_merged = _weld_vertices(Vall, Qall)
    # TOLERANCE weld: close the Blossom-body <-> limb-tube interface (same socket
    # curve, slightly different vertices). Only boundary vertices, within ~0.4h.
    if _depth == 0:
        Vall, Qall, n_tol = _tolerance_weld_boundary(Vall, Qall, tol=0.45 * h)
    else:
        n_tol = 0
    # cap residual triple-point holes (junction branch irregulars)
    Vall, Qall, n_capped = _cap_border_holes(mesh, Vall, Qall)
    # relax + targeted untangle + drop any zero-area cells (sjMin>0 guarantee)
    n_nonman = 0
    if _depth == 0 and len(Qall):
        Vall = relax_quads(mesh, Vall, Qall, iters=8, lam=0.5)
        Vall = untangle_quads(mesh, Vall, Qall)
        # final close: stitch any residual border slits (too small for the loop-
        # capper) then re-cap. On a closed input ANY leftover border is a defect,
        # so escalate the weld tolerance until watertight.
        for _tol in (0.7, 1.2, 2.0):
            if _quad_watertight(Qall)[0]:
                break
            Vall, Qall, _ = _tolerance_weld_boundary(Vall, Qall, tol=_tol * h)
            Vall, Qall, _ = _cap_border_holes(mesh, Vall, Qall)
        # ENFORCE MANIFOLD: dedup + drop zero-area + remove excess faces on any
        # edge shared by >2 (from welds / Blossom) + cap -> a clean manifold mesh,
        # required for the topological cleanup operators to fire.
        Vall, Qall = _enforce_manifold(mesh, Vall, Qall)
        # topological cleanup (optional): reduces the worst high-valence vertices
        # (manifold-safe, never-worse). Off by default -- marginal count benefit at
        # high cost; the irreducible irregulars are inherent to Blossom matching.
        if topo_clean:
            try:
                # _quad_cleanup is not shipped in meshprep; topo_clean
                # (default OFF) degrades to a no-op via this guard.
                from ._quad_cleanup import topo_cleanup  # type: ignore
                Vall, Qall = topo_cleanup(Vall, Qall, mesh=mesh, max_passes=2)
                Vall, Qall = _enforce_manifold(mesh, Vall, Qall)
            except Exception:
                pass
        # FINAL untangle: smooth any cap/enforce degenerate cells -> sjMin > 0.
        Vall = untangle_quads(mesh, Vall, Qall)
    _, _, n_nonman = quad_manifold_stats(Qall)
    watertight, n_border = _quad_watertight(Qall)
    fmax, frms = fidelity(mesh, Vall, scale)
    val_all = _valence(Qall, len(Vall))
    bmask = _boundary_verts(Qall, len(Vall))
    irr_frac, irr_n = irregular_fraction(val_all, bmask)
    out = {"ok": True, "cuts": cuts, "kind_of": kind_of, "N_of_cut": N_of_cut,
           "part_metrics": part_metrics, "Vall": Vall, "Qall": Qall,
           "n_quads": int(len(Qall)), "n_verts": int(len(Vall)), "n_merged": n_merged,
           "watertight": watertight, "n_border_edges": n_border, "n_capped": n_capped,
           "n_nonmanifold": n_nonman, "manifold": bool(n_nonman == 0 and n_border == 0),
           "fid_max": fmax, "fid_rms": frms, "irr_frac": irr_frac, "irr_n": irr_n,
           "scale": scale}
    return out


def _weld_vertices(V, Q, tol=6):
    """Merge vertices at identical (rounded) positions; remap quads."""
    keys = {}
    remap = np.empty(len(V), np.int64)
    newV = []
    for i, p in enumerate(V):
        key = tuple(np.round(p, tol))
        if key not in keys:
            keys[key] = len(newV); newV.append(p)
        remap[i] = keys[key]
    Q2 = remap[Q]
    return np.array(newV), Q2, len(V) - len(newV)


def _cap_border_holes(mesh, V, Q, max_edges=200):
    """Fill residual border-hole loops with a fan to a surface-projected center.
    Closes BOTH the junction triple-points AND the larger gaps where a multi-way
    body core (k tubes at different N) cannot tile -- guaranteeing a complete,
    watertight mesh ('enclosure is always possible'). Each fan center is an
    irregular vertex AT the branch/core. Returns (V, Q, n_capped)."""
    cnt = defaultdict(int)
    for q in Q:
        qq = list(q)
        for a, b in zip(qq, qq[1:] + qq[:1]):
            if a != b:
                e = (a, b) if a < b else (b, a)
                cnt[e] += 1
    bedges = [e for e, c in cnt.items() if c == 1]
    if not bedges:
        return V, Q, 0
    loops = _trace_loops(set(bedges))
    V = list(V); Q = list(Q); n_capped = 0
    for loop in loops:
        n = len(loop)
        if not (3 <= n <= max_edges):
            continue
        # PINWHEEL cap: pair consecutive loop edges to a center vertex -> quads
        # [loop[i], loop[i+1], loop[i+2], center], all 4 DISTINCT (non-degenerate,
        # unlike the old [a,b,ci,ci] triangle fan), reusing the ORIGINAL loop edges
        # (no boundary subdivision -> no T-junctions). Center valence n/2 (half the
        # fan). For EVEN n it tiles cleanly; for ODD n the last wedge is a single
        # triangle-quad. The odd-index loop verts become valence-2 doublets that
        # topological cleanup removes. Watertight by construction.
        ctr = project_to_surface(mesh, np.mean([V[i] for i in loop], axis=0)[None])[0]
        ci = len(V); V.append(ctr); n_capped += 1
        i = 0
        while i < n:
            a = loop[i]; b = loop[(i + 1) % n]; c = loop[(i + 2) % n]
            if i + 1 < n:
                Q.append([a, b, c, ci]); i += 2
            else:                                   # last odd edge -> triangle-quad
                Q.append([a, b, ci, ci]); i += 1
    return np.array(V), np.array(Q, np.int64), n_capped


def relax_quads(mesh, V, Q, iters=8, lam=0.5, pin_mask=None):
    """Tangential Laplacian relaxation with surface reprojection. A regular grid is
    already near its own Laplacian average, so this barely moves the clean limb
    tubes but RELAXES the rough fan-capped cores toward uniform quads. Reprojection
    to the surface prevents shrinkage and preserves fidelity. Open-boundary
    vertices (and any pin_mask) are held fixed. Returns relaxed V."""
    V = np.asarray(V, float).copy()
    adj = defaultdict(set)
    for q in Q:
        qq = [int(x) for x in q]
        for a, b in zip(qq, qq[1:] + qq[:1]):
            if a != b:
                adj[a].add(b); adj[b].add(a)
    hold = _boundary_verts(Q, len(V))
    if pin_mask is not None:
        hold = hold | np.asarray(pin_mask, bool)
    nbr_list = [np.array(sorted(adj[i]), np.int64) if adj[i] else np.array([], np.int64)
                for i in range(len(V))]
    for _ in range(iters):
        newV = V.copy()
        for i in range(len(V)):
            if hold[i] or len(nbr_list[i]) == 0:
                continue
            newV[i] = (1 - lam) * V[i] + lam * V[nbr_list[i]].mean(0)
        V = project_to_surface(mesh, newV)
    return V


def _regularize_socket_loop(mesh, Vb, loop_vids):
    """Smooth + uniform-arclength resample a Blossom socket boundary loop (same
    count) so the conforming limb pins to a CLEAN, evenly-spaced ring instead of a
    jagged one -> kills the interface irregular cluster (adversarial reviewer #1).
    Returns new (M,3) points in loop order; caller writes them to BOTH the body
    boundary and the limb cut so the weld stays exact."""
    idx = np.asarray(loop_vids, np.int64)
    P = Vb[idx].astype(float)
    M = len(P)
    # gentle Taubin-style smoothing of the closed loop (shrink then unshrink)
    for lam in (0.5, -0.52) * 4:
        Pm = 0.5 * (np.roll(P, 1, 0) + np.roll(P, -1, 0))
        P = P + lam * (Pm - P)
    # uniform arc-length resample to the SAME count M
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)]); total = s[-1] + 1e-12
    targ = np.linspace(0, total, M, endpoint=False)
    Pcl = np.vstack([P, P[0]])
    out = np.empty((M, 3))
    for d in range(3):
        out[:, d] = np.interp(targ, s, Pcl[:, d])
    return project_to_surface(mesh, out)


def untangle_quads(mesh, V, Q, sj_floor=0.06, iters=14):
    """Targeted interior smoothing of low-quality quads' vertices (boundary held
    fixed -> weld + watertightness preserved). Kills residual near-degenerate
    quads without blurring the clean grid (only bad-quad verts move)."""
    V = np.asarray(V, float).copy()
    hold = _boundary_verts(Q, len(V))
    adj = defaultdict(set)
    for q in Q:
        qq = [int(x) for x in q]
        for a, b in zip(qq, qq[1:] + qq[:1]):
            if a != b:
                adj[a].add(b); adj[b].add(a)
    for _ in range(iters):
        # per-quad min corner quality
        bad = set()
        worst = 1.0
        for q in Q:
            s, _ = scaled_jacobian(V, [q]); worst = min(worst, s)
            if s < sj_floor:
                bad.update(int(x) for x in q)
        if worst >= sj_floor or not bad:
            break
        moved = V.copy()
        for i in bad:
            if hold[i] or not adj[i]:
                continue
            moved[i] = V[sorted(adj[i])].mean(0)
        V = project_to_surface(mesh, moved)
    return V


def _drop_degenerate_quads(V, Q):
    """Remove quads with <3 distinct vertices (zero-area, e.g. collapsed cells).
    These cover no area, so removal leaves no hole. Returns filtered Q."""
    keep = [q for q in Q if len(set(int(x) for x in q)) >= 3]
    return np.asarray(keep, np.int64) if keep else Q


def _quad_boundary_loops(Q, nV):
    """Ordered boundary vertex loops of a quad mesh (edges on one face only)."""
    cnt = defaultdict(int)
    for q in Q:
        qq = [int(x) for x in q]
        for a, b in zip(qq, qq[1:] + qq[:1]):
            if a != b:
                e = (a, b) if a < b else (b, a); cnt[e] += 1
    bedges = set(e for e, c in cnt.items() if c == 1)
    return [[int(i) for i in loop] for loop in _trace_loops(bedges)]


def _tolerance_weld_boundary(V, Q, tol):
    """Merge BOUNDARY vertices (on open edges) that lie within `tol` of each other.
    Closes the interface between an independently-meshed junction body (Blossom)
    and the limb tube rings, which sit on the same socket curve but at slightly
    different vertices. Interior vertices are never merged. Returns (V, Q, n)."""
    from scipy.spatial import cKDTree
    bmask = _boundary_verts(Q, len(V))
    bidx = np.flatnonzero(bmask)
    if len(bidx) < 2:
        return V, Q, 0
    tree = cKDTree(V[bidx])
    pairs = tree.query_pairs(tol)
    parent = np.arange(len(V))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in pairs:
        a, b = find(bidx[i]), find(bidx[j])
        if a != b:
            parent[max(a, b)] = min(a, b)
    remap = np.array([find(i) for i in range(len(V))], np.int64)
    uniq, inv = np.unique(remap, return_inverse=True)
    Vn = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    for i in range(len(V)):
        Vn[inv[i]] += V[i]; cnt[inv[i]] += 1
    Vn /= cnt[:, None]
    return Vn, inv[Q], len(V) - len(uniq)


def _repair_manifold(V, Q):
    """Make the quad mesh manifold: drop zero-area cells (<3 distinct verts) and
    remove DUPLICATE faces (same vertex set, created when the tolerance weld merges
    a quad onto its neighbour). These are the source of edges shared by >2 faces.
    Returns filtered Q."""
    out, seen = [], set()
    for q in Q:
        u = set(int(x) for x in q)
        if len(u) < 3:                       # zero-area, safe to drop
            continue
        key = frozenset(u)
        if key in seen:                      # duplicate face
            continue
        seen.add(key); out.append([int(x) for x in q])
    return np.asarray(out, np.int64) if out else Q


def _enforce_manifold(mesh, V, Q, max_passes=5):
    """GUARANTEE a manifold quad mesh: repeatedly dedup + drop zero-area cells,
    then on any edge shared by >2 faces keep the 2 best-quality faces and remove
    the rest, capping the resulting holes. Converges to every edge shared by 1-2
    faces. Returns (V, Q)."""
    for _ in range(max_passes):
        Q = _repair_manifold(V, Q)
        ef = defaultdict(list)
        for fi, q in enumerate(Q):
            qq = [int(x) for x in q]
            for a, b in zip(qq, qq[1:] + qq[:1]):
                if a != b:
                    e = (a, b) if a < b else (b, a); ef[e].append(fi)
        nm = [e for e, fs in ef.items() if len(fs) > 2]
        if not nm:
            break
        remove = set()
        for e in nm:
            fs = [fi for fi in ef[e] if fi not in remove]
            if len(fs) <= 2:
                continue
            fs.sort(key=lambda fi: scaled_jacobian(V, [Q[fi]])[0])  # worst first
            remove.update(fs[:len(fs) - 2])
        if not remove:
            break
        Q = np.asarray([q for fi, q in enumerate(Q) if fi not in remove], np.int64)
        V, Q, _ = _cap_border_holes(mesh, V, Q)
    return V, _repair_manifold(V, Q)


def quad_manifold_stats(Q):
    """Return (is_manifold, n_border, n_nonmanifold) for a quad mesh.
    manifold = every undirected edge shared by exactly 1 (border) or 2 faces."""
    cnt = defaultdict(int)
    for q in Q:
        qq = [int(x) for x in q]
        for a, b in zip(qq, qq[1:] + qq[:1]):
            if a != b:
                e = (a, b) if a < b else (b, a); cnt[e] += 1
    border = sum(1 for c in cnt.values() if c == 1)
    nonman = sum(1 for c in cnt.values() if c > 2)
    return (border == 0 and nonman == 0), border, nonman


def _quad_watertight(Q):
    """Is the quad mesh closed? (every undirected edge shared by exactly 2 faces).
    Returns (watertight bool, n_border_edges)."""
    cnt = defaultdict(int)
    for q in Q:
        qq = list(q)
        for a, b in zip(qq, qq[1:] + qq[:1]):
            if a != b:
                e = (a, b) if a < b else (b, a)
                cnt[e] += 1
    border = sum(1 for c in cnt.values() if c == 1)
    return (border == 0), border


# ---------------------------------------------------------------------------
#  analytic test parts
# ---------------------------------------------------------------------------
def shrinkwrap_quads(mesh, n_lat=26, n_lon=36, inflate=1.7):
    """GUARANTEED-COVERAGE fallback: enclose the shape in a closed UV-sphere quad
    cage (PCA-fit ellipsoid) and project every cage vertex onto the surface
    (closest point). Because the cage is closed, the result is ALWAYS a complete,
    watertight quad mesh -- coverage is never in question (the PI's "enclose and
    shrink" idea). TRADE-OFF: the cage topology does not follow the part structure
    or feature flow, and it WEBS across deep concavities / between separated limbs
    (a cage vertex in the gap between two fingers projects onto the nearest finger),
    so fidelity degrades on branchy/concave shapes. Best used as a coverage
    fallback or a base to refine, NOT a replacement for the feature-aligned
    per-part grids. Returns (Vq, quads, info)."""
    V = np.asarray(mesh.vertices, float)
    c = V.mean(0)
    # PCA frame + extents so the cage is an oriented ellipsoid hugging the shape
    Q = V - c
    _, _, Rt = np.linalg.svd(Q, full_matrices=False)
    ext = np.abs(Q @ Rt.T).max(0) * inflate
    lats = np.linspace(0, np.pi, n_lat + 1)
    lons = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    # cage vertices on a (lat x lon) grid, poles at lat 0 and n_lat
    P = np.empty((n_lat + 1, n_lon, 3))
    for i, th in enumerate(lats):
        for j, ph in enumerate(lons):
            local = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]) * ext
            P[i, j] = c + local @ Rt
    flat = P.reshape(-1, 3)
    proj, dist, _ = mesh.nearest.on_surface(flat)
    proj = proj.reshape(n_lat + 1, n_lon, 3)
    # assemble quad mesh (fan the two poles)
    Vq, quads, vid = [], [], {}
    def vert(i, j):
        j %= n_lon
        key = ("N",) if i == 0 else ("S",) if i == n_lat else (i, j)
        if key not in vid:
            vid[key] = len(Vq)
            Vq.append(proj[0, 0] if key == ("N",) else proj[n_lat, 0] if key == ("S",) else proj[i, j])
        return vid[key]
    for i in range(1, n_lat):
        for j in range(n_lon):
            if i == 1:
                quads.append([vert(0, 0), vert(1, j), vert(1, j + 1), vert(0, 0)])
            if i == n_lat - 1:
                quads.append([vert(n_lat - 1, j), vert(n_lat, 0), vert(n_lat, 0), vert(n_lat - 1, j + 1)])
            if 1 <= i < n_lat - 1:
                quads.append([vert(i, j), vert(i, j + 1), vert(i + 1, j + 1), vert(i + 1, j)])
    Vq = np.array(Vq); quads = np.array(quads, np.int64)
    scale = float(np.linalg.norm(V.max(0) - V.min(0)))
    fmax, frms = fidelity(mesh, Vq, scale)
    wt, border = _quad_watertight(quads)
    # reverse coverage: how far the input surface is from the wrap (webbing reveal)
    try:
        samp = mesh.triangles_center
        from scipy.spatial import cKDTree
        d2 = cKDTree(Vq).query(samp)[0]
        cover_max = float(d2.max() / scale)
    except Exception:
        cover_max = float("nan")
    return Vq, quads, {"watertight": wt, "border": border, "fid_max": fmax,
                       "fid_rms": frms, "coverage_max": cover_max, "n_quads": len(quads)}


def make_cap(noise=0.0, seed=0):
    """Irregular hemispherical cap (1 boundary)."""
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    keep = m.triangles_center[:, 2] > 0.15
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    if noise:
        rng = np.random.default_rng(seed)
        m.vertices += rng.normal(0, noise, m.vertices.shape)
    return m


def make_tube(noise=0.0, seed=0, nu=24, nv=40):
    """Open cylinder lateral surface (2 boundaries)."""
    th = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    zz = np.linspace(0, 2.0, nu)
    V, F = [], []
    for i, z in enumerate(zz):
        for j, t in enumerate(th):
            V.append([np.cos(t), np.sin(t), z])
    V = np.array(V)
    if noise:
        rng = np.random.default_rng(seed)
        V += rng.normal(0, noise, V.shape)
    def vid(i, j): return i * nv + (j % nv)
    for i in range(nu - 1):
        for j in range(nv):
            a, b, c, d = vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)
            F += [[a, b, c], [a, c, d]]
    import trimesh
    return trimesh.Trimesh(V, np.array(F), process=False)


def make_bent_tube(R=2.0, r=0.5, nu=30, nv=40, bend=np.pi / 2, noise=0.0, seed=0):
    """A BENT open tube (centerline = circular arc). The case PCA-axis v folds on."""
    phis = np.linspace(0, bend, nu)
    th = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    V = []
    for phi in phis:
        c = np.array([R * np.cos(phi), R * np.sin(phi), 0.0])
        T = np.array([-np.sin(phi), np.cos(phi), 0.0])
        up = np.array([0, 0, 1.0]); side = np.cross(T, up)
        for t in th:
            V.append(c + r * (np.cos(t) * side + np.sin(t) * up))
    V = np.array(V)
    if noise:
        V += np.random.default_rng(seed).normal(0, noise, V.shape)
    F = []
    def vid(i, j): return i * nv + (j % nv)
    for i in range(nu - 1):
        for j in range(nv):
            a, b, c2, d = vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)
            F += [[a, b, c2], [a, c2, d]]
    import trimesh
    return trimesh.Trimesh(V, np.array(F), process=False)


def make_capsule_labeled(stretch=2.6, noise=0.005, seed=1):
    """Elongated icosphere split into bottom-cap / tube / top-cap (3 labels).
    A watertight cap-tube-cap test for the multi-part weld."""
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    Vv = np.asarray(m.vertices, float).copy(); Vv[:, 2] *= stretch
    if noise:
        Vv += np.random.default_rng(seed).normal(0, noise, Vv.shape)
    m = trimesh.Trimesh(Vv, np.asarray(m.faces), process=False)
    zc = m.triangles_center[:, 2]; zmax = float(np.abs(zc).max())
    labels = np.ones(len(m.faces), np.int64)
    labels[zc < -0.45 * zmax] = 0
    labels[zc > 0.45 * zmax] = 2
    return m, labels


if __name__ == "__main__":
    import trimesh
    print("=== analytic single-part grids ===")
    print(f"{'part':10} {'kind':9} {'Nu':>3} {'Nv':>3} {'quads':>6} "
          f"{'sjMin':>6} {'sjMean':>6} {'edgeCV':>6} {'irr%':>6} {'irrN':>5} "
          f"{'phMin':>5} {'fidMax':>7} {'fidRMS':>7}")
    cases = [("hemisphere", make_cap(noise=0.01)),
             ("cylinder", make_tube(noise=0.01))]
    for nm, m in cases:
        V = np.asarray(m.vertices, float); F = np.asarray(m.faces, np.int64)
        scale = float(np.linalg.norm(V.max(0) - V.min(0)))
        loops = boundary_loops(F, len(V))
        u, v, kind, info = param_part(V, F, loops)
        h = 0.12 * scale
        r = grid_part_metrics(m, V, F, u, v, kind, h, scale, loops=loops)
        if not r["ok"]:
            print(f"{nm:10} {kind:9}  FAILED build (Nu={r['Nu']} Nv={r['Nv']})")
            continue
        print(f"{nm:10} {r['kind']:9} {r['Nu']:3d} {r['Nv']:3d} {r['n_quads']:6d} "
              f"{r['sj_min']:6.3f} {r['sj_mean']:6.3f} {r['edge_cv']:6.3f} "
              f"{100*r['irr_frac']:6.1f} {r['irr_n']:5d} {str(r['ph_min']):>5} "
              f"{r['fid_max']:7.4f} {r['fid_rms']:7.4f}")
    print("\nnote: cap expects ph_min=1 (the pole); tube expects ph_min=0 "
          "(fully regular). irrN at/near that floor = clean grid.")

    # --- bent tube: medial-v vs PCA-v perpendicularity (the panel's #1 fix) ---
    print("\n=== bent tube: perpendicularity to cut (90=perp; PCA-v folds) ===")
    m = make_bent_tube(noise=0.005)
    V = np.asarray(m.vertices, float); F = np.asarray(m.faces, np.int64)
    scale = float(np.linalg.norm(V.max(0) - V.min(0)))
    loops = boundary_loops(F, len(V))
    u, v_med, kind, info = param_part(V, F, loops)
    ax, c, e1, e2 = part_axis(V); v_pca = angle_coord(V, ax, c, e1, e2)
    h = 0.10 * scale; Nu = 10; Nv = 24
    for nm2, vv in [("PCA-axis v", v_pca), ("medial-frame v", v_med)]:
        Vq, quads, val, ok = build_grid(V, F, u, vv, kind, Nu, Nv)
        if not ok:
            print(f"  {nm2:16} build FAILED"); continue
        Vq = project_to_surface(m, Vq)
        cutpts = resample_loop(V, loops[0], Nv)
        up, vp = perpendicularity(Vq, quads, cutpts)
        sjm, _ = scaled_jacobian(Vq, quads)
        print(f"  {nm2:16} u-family(cross)={up:5.1f}deg  v-family(along)={vp:4.1f}deg  "
              f"sjMin={sjm:6.3f}")
    print("  want u-family ~90 (perpendicular). PCA-v skews on the bend; medial-v fixes it.")

    # --- multi-part WELD: capsule (cap-tube-cap), watertight stitch ---
    print("\n=== multi-part weld: capsule (cap+tube+cap) ===")
    m, labels = make_capsule_labeled()
    scale = float(np.linalg.norm(np.asarray(m.vertices).max(0) - np.asarray(m.vertices).min(0)))
    res = place_grids_multipart(m, labels, h=0.12 * scale)
    if res["ok"]:
        print(f"  parts={res['kind_of']}  N_of_cut={res['N_of_cut']}")
        print(f"  total quads={res['n_quads']}  verts={res['n_verts']}  "
              f"welded_pairs={res['n_merged']}")
        print(f"  WATERTIGHT={res['watertight']}  border_edges={res['n_border_edges']}  "
              f"irregular={res['irr_n']} ({100*res['irr_frac']:.1f}%)  "
              f"fidMax={res['fid_max']:.4f} fidRMS={res['fid_rms']:.4f}")
        for lbl, pm in sorted(res["part_metrics"].items()):
            if pm.get("ok"):
                print(f"    part {lbl} [{pm['kind']:5}] quads={pm['n_quads']:4d} "
                      f"sjMin={pm['sj_min']:.3f} edgeCV={pm['edge_cv']:.3f} "
                      f"irr={pm['irr_n']} uPerp={pm.get('u_perp',float('nan')):.0f}")
            else:
                print(f"    part {lbl} [{pm['kind']:5}] {pm.get('reason','build failed')}")
    else:
        print("  multipart FAILED:", res.get("part_metrics"))
