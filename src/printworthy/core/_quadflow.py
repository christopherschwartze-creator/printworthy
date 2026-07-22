"""QuadriFlow-style field-aligned quad remeshing (clean-room, permissive port).

Pipeline (Huang et al. 2018, MIT; reimplemented numpy/scipy/networkx, NO GPL):
  1. orientation field (4-RoSy)        -> _field_quad.smooth_field
  2. position field (Instant Meshes)   -> position_field
  3. integer offsets d* (Def 3.1)      -> integer_offsets
  4. regularity repair via MIN-COST FLOW (one node/triangle, makes per-triangle
     offset sum = 0 -> singularity-FREE position field)   -> repair_regularity
  5. subdivide long edges (|d|inf<=1)
  6. inversion repair
  7. position re-solve (tangent-constrained LLS -> fidelity)
  8. extraction (hypotenuse pairing -> watertight by construction)

The key vs the old per-part approach: connectivity is built FIRST (integer
offsets repaired to global consistency by flow); the quad mesh falls out manifold,
quad vertices stay on the input surface (no iso-line drift, no blobbing).
"""
import numpy as np
import trimesh
from collections import defaultdict
from . import _field_quad as fq
from ._mesh_util import decimate as _decimate


# ---------------------------------------------------------------------------
#  preprocessing: a clean manifold mesh (non-manifold breaks the MCF)
# ---------------------------------------------------------------------------
def clean_mesh(mesh, target_faces=None):
    """Merge coincident verts, drop degenerate/duplicate faces, fill holes, fix
    normals -> a clean (ideally manifold, low-genus) triangle mesh. Optionally
    decimate to target_faces FIRST (fewer spurious handles)."""
    m = mesh.copy()
    if target_faces:
        m = _decimate(m, target_faces)     # shared dual-API decimation guard
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    try:
        trimesh.repair.fill_holes(m)
    except Exception:
        pass
    m.fix_normals()
    return m


def isotropic_remesh(mesh, target_edge, iters=5, project=True):
    """Incremental isotropic remeshing (Botsch-Kobbelt 2004): repeatedly split
    edges longer than 4/3*L, collapse edges shorter than 4/5*L, then tangentially
    relax and reproject to the original surface. Produces near-equilateral,
    near-uniform triangles -- exactly the substrate the position field needs (the
    quadric-decimated mesh is anisotropic, CV~0.27, which wrecks the lattice).

    Permissive: pure numpy/trimesh. Returns a new Trimesh. Best-effort; if a step
    fails the partial result is returned. NO GPL."""
    import trimesh as _tm
    ref = mesh.copy()
    L = float(target_edge)
    hi = (4.0 / 3.0) * L
    lo = (4.0 / 5.0) * L
    m = mesh.copy(); m.merge_vertices()
    for _ in range(iters):
        # ---- 1. split long edges at midpoint ----
        V = np.asarray(m.vertices, float); F = np.asarray(m.faces, np.int64)
        eu = m.edges_unique
        elen = np.linalg.norm(V[eu[:, 0]] - V[eu[:, 1]], axis=1)
        longe = set(tuple(sorted(map(int, e))) for e, l in zip(eu, elen) if l > hi)
        if longe:
            newV = list(V); mid_of = {}
            for (a, b) in longe:
                mid_of[(a, b)] = len(newV); newV.append(0.5 * (V[a] + V[b]))

            def mid(x, y):
                return mid_of.get((x, y) if x < y else (y, x), -1)

            newF = []
            for tri in F:
                a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                mab = mid(a, b); mbc = mid(b, c); mca = mid(c, a)
                ns = (mab >= 0) + (mbc >= 0) + (mca >= 0)
                if ns == 0:
                    newF.append([a, b, c])
                elif ns == 1:
                    # one split -> 2 triangles, fanning the new midpoint to the
                    # opposite corner. Keeps CCW winding.
                    if mab >= 0:
                        newF += [[a, mab, c], [mab, b, c]]
                    elif mbc >= 0:
                        newF += [[b, mbc, a], [mbc, c, a]]
                    else:
                        newF += [[c, mca, b], [mca, a, b]]
                elif ns == 2:
                    # two splits -> 3 triangles. Cut the corner between the two
                    # split edges, then split the remaining quad along a diagonal.
                    if mca < 0:                       # split on ab, bc (corner b)
                        newF += [[b, mbc, mab], [a, mab, mbc], [a, mbc, c]]
                    elif mab < 0:                     # split on bc, ca (corner c)
                        newF += [[c, mca, mbc], [b, mbc, mca], [b, mca, a]]
                    else:                             # split on ca, ab (corner a)
                        newF += [[a, mab, mca], [c, mca, mab], [c, mab, b]]
                else:
                    newF += [[a, mab, mca], [mab, b, mbc],
                             [mca, mbc, c], [mab, mbc, mca]]
            m = _tm.Trimesh(np.array(newV), np.array(newF), process=False)
            m.merge_vertices(); m.update_faces(m.nondegenerate_faces())
        # ---- 2. collapse short edges (selective, shortest-first) ----
        # Each vertex may be collapsed at most ONCE per pass (lock both endpoints
        # after a collapse) so welding never cascades across a whole region. The
        # short endpoint snaps to its partner's midpoint.
        V = np.asarray(m.vertices, float)
        eu = np.asarray(m.edges_unique, np.int64)
        elen = np.linalg.norm(V[eu[:, 0]] - V[eu[:, 1]], axis=1)
        sel = elen < lo
        if np.any(sel):
            cand = eu[sel]; clen = elen[sel]
            ordr = np.argsort(clen)
            parent = np.arange(len(V)); locked = np.zeros(len(V), bool)
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]; x = parent[x]
                return x
            target = V.copy()
            for idx in ordr:
                a, b = int(cand[idx, 0]), int(cand[idx, 1])
                if locked[a] or locked[b]:
                    continue
                ra, rb = find(a), find(b)
                if ra == rb:
                    continue
                r = min(ra, rb); other = max(ra, rb)
                parent[other] = r
                target[r] = 0.5 * (target[ra] + target[rb])
                locked[a] = locked[b] = True
            rmap = np.array([find(i) for i in range(len(V))])
            uniq, inv = np.unique(rmap, return_inverse=True)
            Vn = target[uniq]
            Fn = inv[np.asarray(m.faces)]
            m = _tm.Trimesh(Vn, Fn, process=False)
            m.merge_vertices()
            m.update_faces(m.nondegenerate_faces()); m.update_faces(m.unique_faces())
            m.remove_unreferenced_vertices()
        # ---- 3. tangential relaxation + reproject ----
        m.merge_vertices()
        V = np.asarray(m.vertices, float)
        Nn = np.asarray(m.vertex_normals, float)
        adj = [[] for _ in range(len(V))]
        for a, b in m.edges_unique:
            adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
        Vnew = V.copy()
        for i in range(len(V)):
            if adj[i]:
                c = V[adj[i]].mean(axis=0)
                disp = c - V[i]
                disp = disp - Nn[i] * (Nn[i] @ disp)      # tangential only
                Vnew[i] = V[i] + 0.5 * disp
        if project:
            try:
                cp, _, _ = ref.nearest.on_surface(Vnew); Vnew = np.asarray(cp)
            except Exception:
                pass
        m = _tm.Trimesh(Vnew, np.asarray(m.faces), process=False)
        m.merge_vertices(); m.update_faces(m.nondegenerate_faces())
        m.update_faces(m.unique_faces()); m.remove_unreferenced_vertices()
    m.fix_normals()
    return m


# ---------------------------------------------------------------------------
#  cross-field per-edge rotation k_ij (orientation alignment)
# ---------------------------------------------------------------------------
def cross_rotation(di, dj, ni, nj):
    """Integer k in {0,1,2,3}: rotating the cross at i by k*90deg (about the
    common normal) best aligns it with the cross at j. Plus the residual angle."""
    navg = ni + nj
    navg = navg / (np.linalg.norm(navg) + 1e-12)
    a = di - (di @ navg) * navg; a /= (np.linalg.norm(a) + 1e-12)
    b = dj - (dj @ navg) * navg; b /= (np.linalg.norm(b) + 1e-12)
    c = np.cross(navg, a)
    ang = np.arctan2(b @ c, b @ a)
    k = int(np.round(ang / (np.pi / 2))) % 4
    return k


def vertex_frames(mesh, d):
    """Per-vertex orthonormal tangent basis O_v = [o_v, n_v x o_v] (3x2) from the
    cross-field direction o_v=d_v. Returns N (V,3), O (V,3,2)."""
    N, _, _ = fq.tangent_frames(mesh)
    o1 = d - np.einsum("ij,ij->i", d, N)[:, None] * N
    o1 = o1 / (np.linalg.norm(o1, axis=1, keepdims=True) + 1e-12)
    o2 = np.cross(N, o1)
    O = np.stack([o1, o2], axis=2)          # (V,3,2)
    return N, O


def directed_edges(mesh):
    """Unique edges directed low->high index. Returns (E,2) int array."""
    e = np.asarray(mesh.edges_unique, np.int64)
    lo = np.minimum(e[:, 0], e[:, 1]); hi = np.maximum(e[:, 0], e[:, 1])
    return np.stack([lo, hi], axis=1)


# ---------------------------------------------------------------------------
#  position field (Instant Meshes 4-PoSy local smoothing)
# ---------------------------------------------------------------------------
def _floor_lattice(o, q, t, target, scale):
    """Floor lattice point of (o,q,t) nearest `target`."""
    d = target - o
    fu = np.floor((q @ d) / scale); fv = np.floor((t @ d) / scale)
    return o + q * (fu * scale) + t * (fv * scale)


def compat_position(oi, ni, qi, oj, nj, qj, scale):
    """Return the two lattice corners (near i, near j) that are CLOSEST in R^3
    (Instant Meshes compat_position_extrinsic_4: floor both near the midpoint,
    brute-force the 4 corners each, pick the closest pair)."""
    ti = np.cross(ni, qi); tj = np.cross(nj, qj)
    mid = 0.5 * (oi + oj)
    of = _floor_lattice(oi, qi, ti, mid, scale)
    pf = _floor_lattice(oj, qj, tj, mid, scale)
    best = (1e30, oi, oj)
    for a in ((0, 0), (1, 0), (0, 1), (1, 1)):
        ci = of + qi * (a[0] * scale) + ti * (a[1] * scale)
        for b in ((0, 0), (1, 0), (0, 1), (1, 1)):
            cj = pf + qj * (b[0] * scale) + tj * (b[1] * scale)
            dist = float(np.sum((ci - cj) ** 2))
            if dist < best[0]:
                best = (dist, ci, cj)
    return best[1], best[2]


def position_round(s, q, n, v, scale):
    """Snap s to the lattice corner of (anchored at v, basis q,t) nearest s,
    keeping it within one cell of the surface vertex v."""
    t = np.cross(n, q); d = s - v
    return v + q * (np.round((q @ d) / scale) * scale) + t * (np.round((t @ d) / scale) * scale)


_CORNERS = np.array([(0, 0), (1, 0), (0, 1), (1, 1)], float)   # 4 PoSy corners


def compat_corners_batch(si, ni, qi, oj, nj, qj, scale):
    """VECTORIZED compat_position over a batch of edges (one source point s_i
    per edge, one neighbour o_j per edge). For each edge, floor both lattices
    near the midpoint, brute-force the 4x4 corner pairs, return the closest pair
    (c_i near i, c_j near j) AND the integer corner indices (a in {0..3} for i,
    b for j -- which of the 4 corners won). Identical maths to
    `compat_position`, just batched with numpy.

    Inputs are (M,3) arrays (ni,qi,oj,nj,qj) and (M,3) si. Returns
    ci (M,3), cj (M,3), ai (M,), bj (M,) where ai,bj index into _CORNERS."""
    si = np.atleast_2d(si).astype(float)
    ti = np.cross(ni, qi)                 # (M,3)
    tj = np.cross(nj, qj)
    mid = 0.5 * (si + oj)                  # (M,3)
    # floor lattice of i near mid
    di = mid - si
    fui = np.floor(np.einsum("mc,mc->m", qi, di) / scale)
    fvi = np.floor(np.einsum("mc,mc->m", ti, di) / scale)
    of = si + qi * (fui * scale)[:, None] + ti * (fvi * scale)[:, None]   # (M,3)
    dj = mid - oj
    fuj = np.floor(np.einsum("mc,mc->m", qj, dj) / scale)
    fvj = np.floor(np.einsum("mc,mc->m", tj, dj) / scale)
    pf = oj + qj * (fuj * scale)[:, None] + tj * (fvj * scale)[:, None]   # (M,3)
    # 4 corners for i: (M,4,3); same for j
    ci4 = of[:, None, :] + (qi[:, None, :] * (_CORNERS[None, :, 0:1] * scale)
                            + ti[:, None, :] * (_CORNERS[None, :, 1:2] * scale))
    cj4 = pf[:, None, :] + (qj[:, None, :] * (_CORNERS[None, :, 0:1] * scale)
                            + tj[:, None, :] * (_CORNERS[None, :, 1:2] * scale))
    # pairwise squared distance (M,4,4)
    diff = ci4[:, :, None, :] - cj4[:, None, :, :]
    dist = np.einsum("mabc,mabc->mab", diff, diff)
    M = dist.shape[0]
    flat = dist.reshape(M, 16).argmin(axis=1)
    ai = flat // 4
    bj = flat % 4
    ridx = np.arange(M)
    ci = ci4[ridx, ai]
    cj = cj4[ridx, bj]
    return ci, cj, ai, bj


def _vertex_areas(mesh):
    """Per-vertex lumped (barycentric) area, used as the IM running-mean weight."""
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, np.int64)
    a = np.zeros(len(V))
    tri = V[F]
    fa = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                       tri[:, 2] - tri[:, 0]), axis=1)
    for k in range(3):
        np.add.at(a, F[:, k], fa / 3.0)
    a[a <= 0] = a[a > 0].mean() if np.any(a > 0) else 1.0
    return a


def position_field(mesh, d, scale, iters=30, seed=0, weighted=False):
    """Smooth the 4-PoSy position field (Instant-Meshes local Gauss-Seidel,
    Jakob et al. 2015 Eq.6). Returns o (V,3): each vertex's lattice corner, in
    its tangent plane, within one cell of v_i.

    Part-A improvements over the old batched mean:
      * TRUE running-position re-match (IM Eq.6): each neighbour j is matched
        against the RUNNING average position (not the stale o[i]); the running
        average is updated immediately so later neighbours snap to where the
        average is actually drifting. This de-biases the cell choice and is what
        makes the lattice cluster cleanly.
      * Random vertex order AND random neighbour order each sweep (de-bias).
      * More sweeps by default (30).

    Area weighting (weighted=True) is available but OFF by default: on meshes
    with large area variation it makes the running average diverge (the IM
    reference clamps weights more carefully); the uniform mean is stable and
    already clusters to an exact right-isosceles lattice on a dense isotropic
    surface (validated). The inner loop is the cheap closest-corner snap per
    neighbour -- O(deg) tiny ops per vertex."""
    V = np.asarray(mesh.vertices, float)
    N, O = vertex_frames(mesh, d)
    q = O[:, :, 0]
    t = np.cross(N, q)
    nv = len(V)
    area = _vertex_areas(mesh) if weighted else None
    adj = [[] for _ in range(nv)]
    for a, b in mesh.edges_unique:
        adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    adj = [np.array(a, np.int64) for a in adj]
    o = V.copy()
    rng = np.random.default_rng(seed)
    order = np.arange(nv)
    inv_s = 1.0 / scale
    for sweep in range(iters):
        rng.shuffle(order)
        for i in order:
            js = adj[i]
            if js.size == 0:
                continue
            ni = N[i]; qi = q[i]; ti = t[i]; vi = V[i]
            # running AVERAGE position in R^3, seeded by the current estimate.
            avg = o[i].copy()
            wsum = float(area[i]) if weighted else 1.0
            # random neighbour order each visit (de-bias the sweep direction)
            if js.size > 1:
                js = js[rng.permutation(js.size)]
            for j in js:
                oj = o[j]
                wj = float(area[j]) if weighted else 1.0
                # snap o[j] to the lattice corner of i's frame CLOSEST to the
                # RUNNING average (Eq.6: match against running position).
                dd = oj - avg
                cu = round((qi @ dd) * inv_s)
                cv = round((ti @ dd) * inv_s)
                cj = oj - qi * (cu * scale) - ti * (cv * scale)
                # incremental weighted mean update (avg stays a true mean)
                avg = (avg * wsum + cj * wj) / (wsum + wj)
                wsum += wj
            s = avg - ni * (ni @ (avg - vi))       # project to tangent plane
            o[i] = position_round(s, qi, ni, vi, scale)
    return o, N, q


# ===========================================================================
#  STEP 3 -- integer offsets  d*  (QuadriFlow Def 3.1)
# ===========================================================================
def _R2(k):
    """2D rotation by 90*k degrees (k in Z), as an integer 2x2 matrix."""
    k = int(k) % 4
    return [np.array([[1, 0], [0, 1]]),
            np.array([[0, -1], [1, 0]]),
            np.array([[-1, 0], [0, -1]]),
            np.array([[0, 1], [-1, 0]])][k]


def integer_offsets(mesh, o, N, q, scale):
    """Compute, per redirected edge e=(u,v) with u<v:
        k_uv, k_vu in {0,1,2,3}   (cross-field rotations)
        t_uv, t_vu in Z^2         (integer lattice translations from the pos field)
        d*_uv = t_uv - R2(k_uv - k_vu) t_vu   in Z^2   (Def 3.1)
    Returns a dict edge->record and parallel arrays. The integer translations
    are read directly off the converged position field `o`: for edge (u,v),
    find the lattice corner of u and the lattice corner of v that the position
    field matched (closest pair, exactly `compat_position`), expressed as
    integer (col,row) coordinates in each vertex's (q, n x q) basis anchored at
    the vertex's own lattice point o[.]."""
    E = directed_edges(mesh)                  # (Ne,2) low<high
    u = E[:, 0]; v = E[:, 1]
    du = q[u]; dv = q[v]                       # cross directions (== q)
    nu = N[u]; nv = N[v]
    # cross rotations k_uv, k_vu (vectorized cross_rotation)
    k_uv = _cross_rot_batch(du, dv, nu, nv)
    k_vu = _cross_rot_batch(dv, du, nv, nu)
    # integer translations from the position field.
    # Anchor each side at its OWN lattice point o[.]; ask which integer cell of
    # u and which of v are the closest pair. corner index + floor = integer t.
    ti = np.cross(nu, q[u]); tj = np.cross(nv, q[v])
    ou = o[u]; ov = o[v]
    mid = 0.5 * (ou + ov)
    # floor cell (integer) of u near mid, and of v near mid
    fu_u = np.floor(np.einsum("mc,mc->m", q[u], mid - ou) / scale)
    fv_u = np.floor(np.einsum("mc,mc->m", ti,   mid - ou) / scale)
    fu_v = np.floor(np.einsum("mc,mc->m", q[v], mid - ov) / scale)
    fv_v = np.floor(np.einsum("mc,mc->m", tj,   mid - ov) / scale)
    _, _, ai, bj = compat_corners_batch(ou, nu, q[u], ov, nv, q[v], scale)
    ca = _CORNERS[ai]; cb = _CORNERS[bj]      # (Ne,2) the winning corner offsets
    t_uv = np.stack([fu_u + ca[:, 0], fv_u + ca[:, 1]], axis=1).astype(np.int64)
    t_vu = np.stack([fu_v + cb[:, 0], fv_v + cb[:, 1]], axis=1).astype(np.int64)
    # d*_uv = t_uv - R2(k_uv - k_vu) t_vu
    dstar = np.empty((len(E), 2), np.int64)
    for kk in range(4):
        sel = ((k_uv - k_vu) % 4) == kk
        if np.any(sel):
            R = _R2(kk)
            dstar[sel] = t_uv[sel] - (t_vu[sel] @ R.T)
    return {
        "E": E, "k_uv": k_uv, "k_vu": k_vu,
        "t_uv": t_uv, "t_vu": t_vu, "dstar": dstar,
    }


def _cross_rot_batch(di, dj, ni, nj):
    """Vectorized cross_rotation: integer k in {0,1,2,3} per row."""
    navg = ni + nj
    navg = navg / (np.linalg.norm(navg, axis=1, keepdims=True) + 1e-12)
    a = di - np.einsum("mc,mc->m", di, navg)[:, None] * navg
    a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = dj - np.einsum("mc,mc->m", dj, navg)[:, None] * navg
    b /= (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    c = np.cross(navg, a)
    ang = np.arctan2(np.einsum("mc,mc->m", b, c), np.einsum("mc,mc->m", b, a))
    return (np.round(ang / (np.pi / 2)).astype(np.int64)) % 4


# ===========================================================================
#  STEP 4 -- regularity repair via MIN-COST FLOW
#  (one node per triangle; makes the position field singularity-FREE)
# ===========================================================================
def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


def _triangle_constraints(mesh, off):
    """Build the per-triangle regularity equations (Eq.3) in the SCALAR form,
    one equation pair (x-component, y-component) per triangle.

    For triangle (a,b,c) with the three undirected edges, each edge contributes
    R^a_e * d_e where R^a_e rotates the edge offset into vertex a's frame and
    the offset is taken in the canonical (low->high) direction (negated if the
    triangle traverses it high->low).

    Returns:
      tri_edges : (F,3) global edge index for each triangle's 3 edges
      tri_sign  : (F,3) +1 if triangle traverses edge low->high else -1
      tri_rot   : (F,3) integer k s.t. R^a_e = sign * R2(k)   (frame rotation)
      eidx      : dict edge-key -> global edge index
    The frame rotation R^a_uv for edge (u,v) into frame a: per Def 3.1,
      u<v: directly rotate u->a along edge (a,u): R2(k_au - k_ua)
           but we need it for the edge in canonical dir; QuadriFlow gives
           R^w_uv = R2(k_wu - k_uw) for u<v, and = R2(k_wv-k_vw+2) for u>v.
    We compute k_au etc. from the cross field on the fly."""
    F = np.asarray(mesh.faces, np.int64)
    E = off["E"]
    eidx = {(int(a), int(b)): i for i, (a, b) in enumerate(E)}
    N, _, _ = fq.tangent_frames(mesh)
    qd = None  # cross dir per vertex; recover q from off via frames
    # we need k_wu (rotation from u's cross to w's cross). Build a helper that
    # returns k for an arbitrary ordered pair using the per-vertex cross dir q.
    return F, eidx, N


def _k_pair(qd, N, a, b):
    """integer k aligning cross at a to cross at b (== cross_rotation(da,db))."""
    return cross_rotation(qd[a], qd[b], N[a], N[b])


def build_regularity_system(mesh, off, qd, N):
    """Construct the balanced scalar constraint system for the MCF.

    Each triangle t gives two scalar equations (one per component p in {0,1}):
        sum over its 3 edges  s_{t,e} * (R2(rot_{t,e}) d_e)[p]  = 0
    We BALANCE: pick a reference triangle, BFS the dual (triangle-adjacency)
    graph; when crossing into a new triangle we may rotate its WHOLE equation
    by R2(k1-k2+2) so the shared edge variable appears once +1 and once -1.
    Off-tree dual edges and boundary edges -> their variable is FROZEN (added to
    the RHS b as a constant).

    Returns a dict with, per component p (0 and 1 solved as TWO scalar flow
    problems but they SHARE the same balancing rotations, so we keep the 2x2
    rotated coefficient and split later):
       tri_terms[t] = list of (edge_index, coeff2x2, in_canonical_sign)
       where coeff2x2 already includes the per-triangle balancing rotation.
       frozen : set of edge indices frozen to d* (boundary / off-tree)
       b_const[t] : 2-vector constant (contribution of frozen edges), per tri
    """
    F = np.asarray(mesh.faces, np.int64)
    E = off["E"]
    dstar = off["dstar"]
    eidx = {(int(a), int(b)): i for i, (a, b) in enumerate(E)}

    # ---- per-triangle raw equation: R^a_e d_e for its 3 edges (a = tri[0]) ----
    # edges of tri (a,b,c) in order (a,b),(b,c),(c,a). offset d for edge in
    # CANONICAL (low->high) dir; traversal sign s flips it if needed.
    tri_eid = np.empty((len(F), 3), np.int64)
    tri_sign = np.empty((len(F), 3), np.int64)
    tri_rotk = np.empty((len(F), 3), np.int64)
    for ti, tri in enumerate(F):
        a = int(tri[0])
        for j in range(3):
            x = int(tri[j]); y = int(tri[(j + 1) % 3])
            key = _edge_key(x, y)
            tri_eid[ti, j] = eidx[key]
            lo, hi = key
            s = 1 if (x, y) == (lo, hi) else -1   # +1 if traversal == canonical
            tri_sign[ti, j] = s
            # frame rotation R^a_{lo,hi}: rotate edge (lo,hi) offset into a's frame
            #   if a==lo: R2(0) trivially? No: R^w_uv with u=lo<v=hi:
            #     R^a_uv = R2(k_au - k_ua)   (rotate along edge (a,u)=(a,lo))
            #   here u=lo. so k = k_{a,lo} - k_{lo,a}
            u = lo
            k = (_k_pair(qd, N, a, u) - _k_pair(qd, N, u, a)) % 4
            tri_rotk[ti, j] = k

    # ---- BFS balancing over dual graph ----
    # dual adjacency: triangles sharing an edge
    from collections import defaultdict as _dd
    edge2tri = _dd(list)
    for ti in range(len(F)):
        for j in range(3):
            edge2tri[int(tri_eid[ti, j])].append(ti)
    # boundary edges: appear in exactly 1 triangle
    boundary_edges = set(e for e, ts in edge2tri.items() if len(ts) == 1)

    # BFS over triangles; tri_balrot[t] = extra k applied to whole eqn of t
    tri_balrot = np.zeros(len(F), np.int64)
    visited = np.zeros(len(F), bool)
    in_tree_edge = set()           # edges used as tree links (balanced)
    from collections import deque
    # process every connected component
    for root in range(len(F)):
        if visited[root]:
            continue
        visited[root] = True
        tri_balrot[root] = 0
        dq = deque([root])
        while dq:
            t = dq.popleft()
            for j in range(3):
                e = int(tri_eid[t, j])
                nbrs = edge2tri[e]
                if len(nbrs) != 2:
                    continue
                t2 = nbrs[0] if nbrs[1] == t else nbrs[1]
                if visited[t2]:
                    continue
                visited[t2] = True
                # find which local edge index e is in t2
                j2 = int(np.where(tri_eid[t2] == e)[0][0])
                # effective rotation of edge var in t (incl current balrot) :
                #   coeff in t  = s_t * R2(rot_t + balrot_t)
                #   coeff in t2 = s_t2 * R2(rot_t2 + balrot_t2)
                # want coeff_t2 = - coeff_t  (balanced: +1 / -1 of SAME signed var)
                # i.e. s_t2 R2(rot_t2 + b2) = - s_t R2(rot_t + b_t)
                #   => R2(b2) = (s_t/s_t2) * (-1) * R2(rot_t + b_t - rot_t2)
                # -1 = R2(2); sign ratio s in {+1,-1}: s=+1 -> 0, s=-1 -> R2(2)
                s_t = int(tri_sign[t, j]); s_t2 = int(tri_sign[t2, j2])
                rot_t = int(tri_rotk[t, j]); rot_t2 = int(tri_rotk[t2, j2])
                bt = int(tri_balrot[t])
                sign_extra = 2 if (s_t * s_t2) < 0 else 0
                b2 = (rot_t + bt - rot_t2 + 2 + sign_extra) % 4
                tri_balrot[t2] = b2
                in_tree_edge.add(e)
                dq.append(t2)

    # frozen = boundary edges + only the FRUSTRATED off-tree edges. An off-tree
    # edge is frustrated iff the BFS-assigned balrots do NOT sign-balance its shared
    # variable (== required_b2 below). Frustrated edges correspond to field
    # singularities (few); freezing ALL off-tree edges (the previous bug) over-
    # constrains the flow and leaves spurious position singularities. Balanced
    # off-tree edges stay MUTABLE so the min-cost flow can route through them.
    all_interior = set(e for e, ts in edge2tri.items() if len(ts) == 2)
    frozen = set(boundary_edges)
    for e in (all_interior - in_tree_edge):
        t, t2 = edge2tri[e][0], edge2tri[e][1]
        j = int(np.where(tri_eid[t] == e)[0][0])
        j2 = int(np.where(tri_eid[t2] == e)[0][0])
        s_t, s_t2 = int(tri_sign[t, j]), int(tri_sign[t2, j2])
        rot_t, rot_t2 = int(tri_rotk[t, j]), int(tri_rotk[t2, j2])
        bt, b2 = int(tri_balrot[t]), int(tri_balrot[t2])
        sign_extra = 2 if (s_t * s_t2) < 0 else 0
        required_b2 = (rot_t + bt - rot_t2 + 2 + sign_extra) % 4
        if b2 != required_b2:
            frozen.add(e)                 # frustrated -> freeze (a singularity)
        else:
            in_tree_edge.add(e)           # balanced -> keep routable

    return {
        "F": F, "E": E, "dstar": dstar,
        "tri_eid": tri_eid, "tri_sign": tri_sign, "tri_rotk": tri_rotk,
        "tri_balrot": tri_balrot, "edge2tri": dict(edge2tri),
        "boundary_edges": boundary_edges, "frozen": frozen,
        "in_tree_edge": in_tree_edge, "qd": qd, "N": N,
    }


def _tri_coeff(sys, t, j):
    """Signed 2x2 integer rotation that multiplies edge-var d_e in triangle t's
    (balanced) equation: s_t * R2(rot_t + balrot_t)."""
    s = int(sys["tri_sign"][t, j])
    k = (int(sys["tri_rotk"][t, j]) + int(sys["tri_balrot"][t])) % 4
    return s * _R2(k)


def _residual_b(sys, d):
    """Per-triangle residual b_t = -(sum of signed-rotated d_e) for current d.
    Returns (F,2) int array. With balancing, every MUTABLE edge var cancels in
    pairs across the two eqns it belongs to; frozen edges remain in b."""
    F = sys["tri_eid"]
    b = np.zeros((len(F), 2), np.int64)
    for t in range(len(F)):
        acc = np.zeros(2, np.int64)
        for j in range(3):
            e = int(sys["tri_eid"][t, j])
            acc = acc + _tri_coeff(sys, t, j) @ d[e]
        b[t] = -acc            # we need sum + b = 0 ... actually constraint sum=0
    return b


def repair_regularity(mesh, off, qd, N, max_H=6, verbose=False, box_clamp=True,
                      min_cost=False, dstar_override=None):
    """STEP 4. Adjust integer offsets d (start d=d*) so every triangle's
    regularity equation (Eq.3) holds, minimizing ||d-d*||_1, via min-cost flow.
    Solve the x-component and y-component as TWO separate scalar flow problems
    (they share the same balancing rotations but the rotation MIXES components,
    so we cannot fully separate -- we instead iterate: solve assuming the other
    component fixed, like coordinate descent, until both satisfied). In practice
    the rotations are multiples of 90deg so a component maps to +/- a component;
    we handle the mixing exactly by treating each triangle eqn's two scalar rows
    independently with the correct signed coupling.

    min_cost=True : route the flow with `max_flow_min_cost` (weight=1 per unit of
    |delta|) instead of cost-ignored `maximum_flow`, so the repaired offsets stay
    L1-CLOSE to the seed -- important when the seed is the field the extractor
    actually reads (otherwise the singularity-free solution can be far from the
    geometry and the collapse over-merges).

    dstar_override (Ne,2 int) : seed the repair from this offset field instead of
    off["dstar"] (e.g. the `step` field that `_im_extract.extract_graph` reads,
    so the repair patches the SAME field the extractor will consume). The
    balancing rotations still come from off (the cross field), which is correct
    since both fields use the same canonical lo->hi convention.

    Returns d (Ne,2) int, and a stats dict."""
    import networkx as nx
    Ne = len(off["E"])
    sys = build_regularity_system(mesh, off, qd, N)
    dstar = (off["dstar"] if dstar_override is None
             else np.asarray(dstar_override, np.int64)).copy()
    d = dstar.copy()
    nF = len(sys["tri_eid"])

    # The balanced system: for each triangle t and component row p in {0,1},
    #   sum_j (coeff_{t,j})[p, :] . d[e_j] = 0.
    # Because coeff is +/- a 90deg rotation, (coeff)[p,:] has exactly one nonzero
    # (+/-1) hitting component 0 or 1 of d[e_j]. So each scalar row couples ONE
    # scalar variable per edge. Build, per row-component p, a balanced scalar
    # incidence: for triangle t row p, variable = (edge e, which-comp c, sign).
    # We will run flow on the x-DOF and y-DOF variables jointly is messy;
    # QuadriFlow runs them as 2 separate problems by noting the coupling is a
    # fixed permutation. We replicate: build a bipartite map.

    # For each (t,j): coeff2 = _tri_coeff. coeff2[p] picks comp c=argnz, sign.
    # Equation row (t,p):  sum_j sign * d[e_j][c]  = 0.
    # Group the scalar unknowns: each edge e has 2 scalar unknowns d[e][0],d[e][1].
    # A scalar unknown (e,c) appears in some rows. With balancing it appears
    # +1 in one row and -1 in another (for interior tree edges).

    # Build rows
    rows = []   # each: list of (var_id, sign), and rhs handled via frozen
    var_id = {}  # (edge, comp) -> id
    def vid(e, c):
        key = (e, c)
        if key not in var_id:
            var_id[key] = len(var_id)
        return var_id[key]

    frozen = sys["frozen"]
    row_terms = [[] for _ in range(2 * nF)]      # row index = 2*t + p
    row_const = np.zeros((2 * nF,), np.int64)    # frozen contributions (move to rhs)
    for t in range(nF):
        for j in range(3):
            e = int(sys["tri_eid"][t, j])
            C = _tri_coeff(sys, t, j)            # 2x2, entries in {-1,0,1}
            for p in range(2):
                c = int(np.nonzero(C[p])[0][0])
                sgn = int(C[p, c])
                r = 2 * t + p
                if e in frozen:
                    row_const[r] += sgn * int(d[e][c])     # known -> rhs
                else:
                    row_terms[r].append((vid(e, c), sgn))

    nvar = len(var_id)
    # target: sum(mutable terms) + row_const = 0  =>  A x = b,  b = -row_const
    # where x are the FINAL values d[e][c]. We optimize ||x - x*||_1.
    # Standard MCF reduction (Eq.7): split delta = x - x*. Here we just run flow
    # on x directly using b and box [x*-H, x*+H]. networkx min_cost_flow wants
    # integer node demands and arc (capacity, weight). Build that.

    # node per row; var = arc between the two rows it appears in (+1 / -1).
    # frozen/boundary already folded into row_const. A mutable var that appears
    # only ONCE (off-tree leftover) would be unbalanced -> shouldn't happen since
    # we froze all non-tree interior edges; assert and fold any stragglers.
    appear = defaultdict(list)
    for r in range(2 * nF):
        for (vv, sgn) in row_terms[r]:
            appear[vv].append((r, sgn))

    # x* per var
    xstar = np.zeros(nvar, np.int64)
    for (e, c), vv in var_id.items():
        xstar[vv] = dstar[e][c]

    # any var not appearing exactly once +1 and once -1 -> freeze to x*
    bad = []
    for vv, occ in appear.items():
        signs = sorted(s for _, s in occ)
        if not (len(occ) == 2 and signs == [-1, 1]):
            bad.append(vv)
    inv_var = {vv: key for key, vv in var_id.items()}
    for vv in bad:
        e, c = inv_var[vv]
        for (r, sgn) in appear[vv]:
            row_const[r] += sgn * int(xstar[vv])
        row_terms_clean = []
    # rebuild row_terms without bad vars
    badset = set(bad)
    for r in range(2 * nF):
        row_terms[r] = [(vv, s) for (vv, s) in row_terms[r] if vv not in badset]
    appear = defaultdict(list)
    for r in range(2 * nF):
        for (vv, sgn) in row_terms[r]:
            appear[vv].append((r, sgn))

    b = -row_const                      # A x = b  (x = final values, mutable)
    # but x = x* + delta; substitute: A x* + A delta = b => A delta = b - A x*
    Axstar = np.zeros(2 * nF, np.int64)
    for vv, occ in appear.items():
        for (r, sgn) in occ:
            Axstar[r] += sgn * int(xstar[vv])
    bb = b - Axstar                     # A delta = bb,  delta in [-H, H]

    # ---- FEASIBILITY: greedy B=0 nudge (paper, "Feasibility Condition") ----
    # The balanced system needs sum(bb)=0 for a full flow to exist. Frozen and
    # boundary edges are the only knobs (mutable vars cancel in pairs). Each
    # such edge-component, when nudged by +/-1, changes sum(bb) by a known delta
    # (boundary edge: +/-1; unbalanced interior edge: +/-2). We greedily nudge
    # the knob that reduces |sum(bb)| the most, until sum(bb)==0. Folding a knob
    # change means: change row_const at its rows, i.e. change bb at those rows.
    # Precompute, per frozen edge-component, its (rows, signs) footprint.
    knob_rows = {}     # (e,c) -> list of (row, sgn) in row_const
    for t in range(nF):
        for j in range(3):
            e = int(sys["tri_eid"][t, j])
            if e not in frozen:
                continue
            C = _tri_coeff(sys, t, j)
            for p in range(2):
                c = int(np.nonzero(C[p])[0][0])
                sgn = int(C[p, c])
                knob_rows.setdefault((e, c), []).append((2 * t + p, sgn))
    # bad-var (unbalanced mutable) edges were folded into row_const too; expose
    # them as knobs by scanning every triangle for those scalar vars.
    badset_ec = set(inv_var[vv] for vv in bad)
    if badset_ec:
        for t in range(nF):
            for j in range(3):
                e = int(sys["tri_eid"][t, j])
                C = _tri_coeff(sys, t, j)
                for p in range(2):
                    c = int(np.nonzero(C[p])[0][0])
                    if (e, c) in badset_ec and e not in frozen:
                        sgn = int(C[p, c])
                        knob_rows.setdefault((e, c), []).append((2 * t + p, sgn))
    # greedy -- balance EACH component (x-rows and y-rows) to zero SEPARATELY.
    # A flow var moves demand only within the connected x/y graph, so the net
    # demand of the x-rows AND of the y-rows must EACH be zero for a full flow
    # (balancing the combined total is not enough -- that was the bug). Rows are
    # 2*t (x) and 2*t+1 (y). We drive the 2-vector (Bx, By) -> (0,0).
    def comp_B():
        return int(bb[0::2].sum()), int(bb[1::2].sum())
    knobs = list(knob_rows.items())
    nudged = {}
    guard = 0
    Bx, By = comp_B()
    while (Bx != 0 or By != 0) and knobs and guard < 200000:
        guard += 1
        best = None
        for (ec, footprint) in knobs:
            for direction in (+1, -1):
                # knob up by `direction` changes bb[r] by -sgn*direction; that
                # lands in the x-sum or y-sum depending on row parity.
                dBx = sum(-sgn * direction for (r, sgn) in footprint if r % 2 == 0)
                dBy = sum(-sgn * direction for (r, sgn) in footprint if r % 2 == 1)
                if dBx == 0 and dBy == 0:
                    continue
                cost = abs(Bx + dBx) + abs(By + dBy)
                if cost < abs(Bx) + abs(By):
                    if best is None or cost < best[0]:
                        best = (cost, ec, direction, footprint, dBx, dBy)
        if best is None:
            break
        _, ec, direction, footprint, dBx, dBy = best
        for (r, sgn) in footprint:
            bb[r] += -sgn * direction
        nudged[ec] = nudged.get(ec, 0) + direction
        Bx += dBx; By += dBy
        # also record the actual offset change to apply to d later
    # apply nudges to the offsets d (so extraction sees the consistent field)
    for (e, c), dv in nudged.items():
        d[e][c] += dv

    Bsum = int(bb.sum())
    Bx, By = int(bb[0::2].sum()), int(bb[1::2].sum())
    if verbose:
        print(f"    [MCF] vars={nvar - len(bad)} rows={2*nF} frozen={len(frozen)} "
              f"bad={len(bad)} nudges={sum(abs(v) for v in nudged.values())} "
              f"sum(b)=({Bx},{By})")

    # Build and solve flow for increasing H.
    H = 2
    delta = np.zeros(nvar, np.int64)
    feasible = False
    # Build the var->(row_minus, row_plus) map once. We split each delta into
    # two non-negative flows: delta = f_pos - f_neg. f_pos uses a forward arc
    # rm->rp (cost 1, cap H); f_neg uses rp->rm. A unit of f_pos increases the
    # +1 row's LHS by 1 and the -1 row's LHS by -1 (== reduces its residual).
    var_arcs = []
    okvars = True
    for vv, occ in appear.items():
        rminus = [r for (r, s) in occ if s == -1]
        rplus = [r for (r, s) in occ if s == +1]
        if len(rminus) != 1 or len(rplus) != 1:
            okvars = False
            continue
        var_arcs.append((vv, rminus[0], rplus[0]))

    while H <= max_H:
        # PAPER construction (Fig 4): node per row; explicit source s, sink t.
        # For row r with required LHS change needed = -bb[r] (we need A delta=bb,
        # i.e. each row must RECEIVE net bb[r] units of "LHS"). Recast as flow:
        #   a var arc rm->rp carries f_pos: +1 to rp's balance, -1 to rm's.
        #   So node r's net (in-out) from var arcs must equal bb[r].
        #   Add: if bb[r] > 0  -> arc r->t  cap bb[r]   (excess must leave to t)
        #        if bb[r] < 0  -> arc s->r  cap -bb[r]  (deficit supplied by s)
        # Full-flow (saturate all s-> and ->t arcs) <=> A delta = bb.
        G = nx.DiGraph()
        s = ("s",); t = ("t",)
        for (vv, rm, rp) in var_arcs:
            if box_clamp:
                # keep the FINAL value x = xstar + delta within [-1, 1] so the
                # flow never manufactures a long edge (|d|inf>1). Forward arc
                # (rm->rp) raises delta -> cap = 1 - xstar; backward lowers it.
                cap_up = max(0, 1 - int(xstar[vv]))
                cap_dn = max(0, int(xstar[vv]) + 1)
            else:
                cap_up = cap_dn = H
            if cap_up > 0:
                G.add_edge(("r", rm), ("r", rp), weight=1, capacity=cap_up)
            if cap_dn > 0:
                G.add_edge(("r", rp), ("r", rm), weight=1, capacity=cap_dn)
        Csink = 0
        for r in range(2 * nF):
            br = int(bb[r])
            if br > 0:
                G.add_edge(("r", r), t, weight=0, capacity=br); Csink += br
            elif br < 0:
                G.add_edge(s, ("r", r), weight=0, capacity=-br)
        if not okvars or Csink == 0:
            # nothing to route (already consistent) -> delta=0
            feasible = (Csink == 0)
            break
        # MAX-FLOW formulation (paper falls back to this for large/hard nets):
        # we want to saturate as many s-> and ->t arcs as possible. Use
        # maximum_flow (cost ignored) which is robust; full saturation <=>
        # all singularities removed. Partial saturation removes what it can and
        # leaves the rest as residual singularities (extra irregular vertices),
        # which keeps the mesh watertight/manifold (the paper accepts this).
        try:
            if min_cost:
                # max-flow of MINIMUM cost: same singularity removal, but the
                # weight=1 arcs make it minimize total |delta| (L1-close to seed).
                flow = nx.max_flow_min_cost(G, s, t)
                flowval = sum(flow[s].get(("r", r), 0)
                              for r in [k for k in flow.get(s, {})]) \
                    if s in flow else 0
                # robust flowval: sum of flow leaving s
                flowval = sum(flow.get(s, {}).values())
            else:
                flowval, flow = nx.maximum_flow(G, s, t)
            feasible = (flowval == Csink)
        except (nx.NetworkXUnfeasible, nx.NetworkXError):
            H += 1
            continue
        # recover delta per var: delta = f(rm->rp) - f(rp->rm)
        for (vv, rm, rp) in var_arcs:
            fpos = flow.get(("r", rm), {}).get(("r", rp), 0)
            fneg = flow.get(("r", rp), {}).get(("r", rm), 0)
            delta[vv] = fpos - fneg
        if feasible:
            break
        # if not fully saturated, grow H once more then accept best effort
        if H >= max_H:
            break
        H += 1

    # apply delta to d. Start from d (which already carries the feasibility
    # nudges on frozen/bad edges), then set each mutable var from x* + delta.
    for (e, c), vv in var_id.items():
        if vv in badset:
            continue
        d[e][c] = xstar[vv] + int(delta[vv])

    # stats
    res = _residual_b(sys, d)
    nbad_tri = int(np.sum(np.any(res != 0, axis=1)))
    stats = {"feasible": feasible, "H": H, "frozen": len(frozen),
             "bad_vars": len(bad), "sum_b": Bsum,
             "residual_triangles": nbad_tri, "sys": sys}
    if verbose:
        print(f"    [MCF] H={H} feasible={feasible} residual_tris={nbad_tri}")
    return d, stats


# ===========================================================================
#  STEP 5 -- subdivide long edges so |d|inf <= 1
# ===========================================================================
def subdivide_long_edges(mesh, off, d):
    """Split every edge with |d|inf > 1 at a midpoint into two integer edges
    (d_um = d div 2, d_mv = d - d_um), recursively until |d|inf <= 1 on all.
    Operates on a derived combinatorial graph (vertices = original mesh verts
    + new split verts; edges carry an integer offset and a 3D anchor for later
    position solve). Returns (verts3d, edges, edge_d, vert_is_orig)."""
    V = np.asarray(mesh.vertices, float).copy()
    E = off["E"]
    verts = [v.copy() for v in V]
    is_orig = [True] * len(V)
    # working edge list as (u, v, d2)  with d2 int (2,)
    work = [(int(E[i, 0]), int(E[i, 1]), d[i].astype(np.int64).copy())
            for i in range(len(E))]
    out_edges = []
    out_d = []
    qi = 0
    while qi < len(work):
        u, v, dd = work[qi]; qi += 1
        if int(np.max(np.abs(dd))) <= 1:
            out_edges.append((u, v)); out_d.append(dd)
            continue
        # split at midpoint
        m = len(verts)
        verts.append(0.5 * (verts[u] + verts[v]))
        is_orig.append(False)
        d_um = (dd // 2).astype(np.int64)          # floor div toward -inf (ok)
        d_mv = (dd - d_um).astype(np.int64)
        work.append((u, m, d_um))
        work.append((m, v, d_mv))
    return np.array(verts), out_edges, out_d, np.array(is_orig)


def subdivide_faces(mesh, sys, d):
    """EXPERIMENTAL / NOT IN THE LIVE PIPELINE (left for reference). The face
    subdivision with cross-frame rotation tracking did not preserve consistency
    in testing (it MADE more long edges, not fewer) -- the live extraction
    instead clamps the MCF so |d|inf<=1 (box_clamp) and forbids residual long
    edges in the integration. Use with care; see report.

    Split every edge with |d|inf>1 by inserting a midpoint and splitting the
    (<=2) incident triangles, until ALL edges have |d|inf<=1. Returns a NEW
    combinatorial mesh as (verts3d (Nv,3), faces (Nf,3), E (Ne,2) canonical,
    edge_index dict, edge_d (Ne,2), edge2tri, qd_new (Nv,3), N_new (Nv,3)) so the
    lattice extractor can run on a fully |d|inf<=1 structure (-> watertight).

    Offsets along a split edge halve: d_um = d div 2, d_mv = d - d_um (paper).
    New (midpoint) vertices inherit the cross frame of the lower endpoint.
    Works in each vertex's LOCAL frame; rotations between frames are tracked via
    the cross field so the halved offsets stay consistent."""
    V = [v.copy() for v in np.asarray(mesh.vertices, float)]
    F = [list(map(int, f)) for f in sys["F"]]
    qd = sys["qd"]; N = sys["N"]
    qd = [q.copy() for q in qd] if qd is not None else None
    Nl = [n.copy() for n in N]
    E = sys["E"]
    # directed offset store: doff[(a,b)] = integer 2D step a->b in a's frame.
    # seed from canonical d (lo->hi in lo frame); reverse via rotation.
    def kab(a, b):
        if qd is None:
            return 0
        return cross_rotation(qd[a], qd[b], Nl[a], Nl[b])
    doff = {}
    for i in range(len(E)):
        lo, hi = int(E[i, 0]), int(E[i, 1])
        de = d[i].astype(np.int64)
        doff[(lo, hi)] = de
        doff[(hi, lo)] = -(_R2(kab(lo, hi)) @ de)

    # edge -> incident faces
    from collections import defaultdict as _dd
    def face_edges(f):
        return [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]
    e2f = _dd(list)
    for fi, f in enumerate(F):
        for (a, b) in face_edges(f):
            e2f[frozenset((a, b))].append(fi)

    # queue of long edges
    def is_long(a, b):
        return int(np.max(np.abs(doff[(a, b)]))) > 1

    guard = 0
    pending = [frozenset((int(E[i, 0]), int(E[i, 1])))
               for i in range(len(E)) if is_long(int(E[i, 0]), int(E[i, 1]))]
    while pending and guard < 200000:
        guard += 1
        key = pending.pop()
        ab = tuple(key)
        if len(ab) != 2:
            continue
        a, b = ab
        if (a, b) not in doff or not is_long(a, b):
            continue
        faces = [fi for fi in e2f.get(key, []) if fi is not None and F[fi] is not None]
        # new midpoint vertex
        mids = len(V)
        V.append(0.5 * (V[a] + V[b]))
        if qd is not None:
            qd.append(qd[a].copy())
        Nl.append(Nl[a].copy())
        # halve offsets in a's frame
        dab = doff[(a, b)]
        d_am = (dab // 2).astype(np.int64)
        d_mb = (dab - d_am).astype(np.int64)
        doff[(a, mids)] = d_am
        doff[(mids, a)] = -d_am
        doff[(mids, b)] = d_mb
        doff[(b, mids)] = -d_mb
        del doff[(a, b)]; del doff[(b, a)]
        e2f.pop(key, None)
        # split each incident face f=(.., a, b, ..) into 2 by connecting mid to
        # the opposite vertex c.
        for fi in faces:
            f = F[fi]
            if f is None or a not in f or b not in f:
                continue
            c = [x for x in f if x != a and x != b]
            if len(c) != 1:
                continue
            c = c[0]
            # offset c->mid in c's frame: c->a then a->mid (rotate a-frame to c)
            # d(c->mid) = d(c->a) + R(c<-a) d(a->mid)
            kca = kab(a, c)        # a-frame -> c-frame rotation
            d_c_mid = doff[(c, a)] + (_R2(kca) @ d_am)
            doff[(c, mids)] = d_c_mid
            doff[(mids, c)] = -(_R2(kab(c, mids)) @ d_c_mid)
            # preserve winding: f was [.., a, b, ..]; replace with two faces
            # walk f to keep orientation
            order = f
            ia = order.index(a); ib = order.index(b)
            # build [a, b, c] in original cyclic order
            # two new faces: (a, mid, c) and (mid, b, c) consistent with winding
            # determine cyclic position: faces as list, keep CCW
            # reconstruct using original order of (a,b,c)
            tri = order
            # map to positions
            # new faces preserving orientation a->b becomes a->mid->...->b
            nf1 = _orient_face(tri, a, b, c, mids, first=True)
            nf2 = _orient_face(tri, a, b, c, mids, first=False)
            F[fi] = nf1
            F.append(nf2)
            nfi = len(F) - 1
            # update edge->face incidence for the 5 edges of the 2 new faces
            for ff, idx in ((nf1, fi), (nf2, nfi)):
                for (x, y) in face_edges(ff):
                    e2f[frozenset((x, y))].append(idx)
                    e2f[frozenset((x, y))] = list(set(e2f[frozenset((x, y))]))
            # remove old face from the c-a and c-b edge lists where it was
        # re-check the two new sub-edges
        for (x, y) in ((a, mids), (mids, b)):
            if is_long(x, y):
                pending.append(frozenset((x, y)))

    # rebuild canonical structures
    F2 = [f for f in F if f is not None]
    Varr = np.array(V)
    # canonical edges + per-edge offset (lo->hi in lo frame)
    Eset = {}
    for f in F2:
        for (x, y) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            lo, hi = (x, y) if x < y else (y, x)
            if (lo, hi) not in Eset:
                Eset[(lo, hi)] = doff[(lo, hi)]
    Elist = list(Eset.keys())
    Earr = np.array(Elist, np.int64)
    edge_d = np.array([Eset[e] for e in Elist], np.int64)
    eidx = {e: i for i, e in enumerate(Elist)}
    e2t = _dd(list)
    for fi, f in enumerate(F2):
        for (x, y) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            lo, hi = (x, y) if x < y else (y, x)
            e2t[eidx[(lo, hi)]].append(fi)
    qd_new = np.array(qd) if qd is not None else None
    N_new = np.array(Nl)
    newsys = {
        "F": np.array(F2), "E": Earr, "dstar": edge_d,
        "edge2tri": dict(e2t), "qd": qd_new, "N": N_new,
    }
    return Varr, newsys, edge_d


def _orient_face(tri, a, b, c, mid, first):
    """Given original CCW triangle `tri` (a list containing a,b,c) and the new
    midpoint `mid` on edge (a,b), return one of the two split sub-triangles
    preserving orientation. first=True -> the (a, mid, c) side; else (mid, b, c).
    We rebuild from the cyclic order of `tri`."""
    # cyclic order of tri
    n = len(tri)
    # find rotation so it reads starting at a
    ia = tri.index(a)
    rot = tri[ia:] + tri[:ia]       # now rot[0]=a
    # rot is [a, x, y] CCW. Either [a,b,c] or [a,c,b].
    if rot[1] == b:                 # order a -> b -> c
        # split edge a-b at mid: faces [a, mid, c] and [mid, b, c]
        return [a, mid, c] if first else [mid, b, c]
    else:                           # order a -> c -> b
        # split edge a-b (=a..b going a->c->b? edge a-b is the last edge b->a)
        # faces: [a, c, mid] and [c, b, mid]
        return [a, c, mid] if first else [c, b, mid]


# ===========================================================================
#  STEP 6 -- inversion repair (greedy) -- operates on triangles still present
# ===========================================================================
def repair_inversions(sys, d, max_passes=4, verbose=False):
    """Greedy: for each triangle whose oriented area det[R d_uv, R d_uw] < 0,
    try setting one of its edge offsets to 0 (collapse) and propagating Eq.3,
    if it reduces total inverted area and creates no long edge. We do a light
    version: zero the longest edge of an inverted triangle when that strictly
    reduces the inversion count. (Full SAT fallback omitted; report residuals.)"""
    # Compute per-triangle orientation using the balanced coeffs in a's frame.
    F = sys["tri_eid"]
    nF = len(F)
    def tri_inverted(t, dloc):
        # area sign in a's frame: use first two edges' offsets rotated to a-frame
        # d_uv = coeff(t,0) d_e0 ... but easier: reconstruct lattice coords.
        # Use the canonical edge offsets directly: build the 2D positions of the
        # triangle's 3 corners by walking edges in a's frame.
        a_off = np.zeros(2, np.int64)
        pts = [a_off]
        # edge0 a->b, edge1 b->c. signed-rotated to a-frame already? coeff gives
        # contribution to the SUM; instead reconstruct: position of b relative a
        # = R^a_{ab} d_{ab} (signed canonical). edge j local: tri_sign*R2(rotk).
        cur = np.zeros(2, np.int64)
        for j in range(2):
            e = int(sys["tri_eid"][t, j])
            C = _tri_coeff(sys, t, j)
            cur = cur + C @ dloc[e]
            pts.append(cur.copy())
        p0, p1, p2 = pts[0], pts[1], pts[2]
        cross = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])
        return cross < 0, cross

    inv0 = sum(1 for t in range(nF) if tri_inverted(t, d)[0])
    if inv0 == 0:
        return d, {"inverted_before": 0, "inverted_after": 0}
    for _ in range(max_passes):
        changed = False
        for t in range(nF):
            bad, _ = tri_inverted(t, d)
            if not bad:
                continue
            # try zeroing each edge offset (the variable), keep best
            for j in range(3):
                e = int(sys["tri_eid"][t, j])
                old = d[e].copy()
                if np.all(old == 0):
                    continue
                d[e] = 0
                still, _ = tri_inverted(t, d)
                if not still:
                    changed = True
                    break
                d[e] = old
        if not changed:
            break
    invf = sum(1 for t in range(nF) if tri_inverted(t, d)[0])
    if verbose:
        print(f"    [inv] inverted {inv0} -> {invf}")
    return d, {"inverted_before": inv0, "inverted_after": invf}


# ===========================================================================
#  STEP 7 -- position re-solve (Eq.9, tangent-constrained LLS) + projection
# ===========================================================================
def resolve_positions(mesh, verts3d, edges, edge_d, is_orig, O_orig, N_orig,
                      scale, project=True):
    """Eq.9: min_p sum_edges ||p_v - p_u - rho*(O_u d_uv)||^2, each ORIGINAL
    vertex p constrained to its tangent plane (param by 2 DOF in O_u). New
    (split) vertices are free in R^3 (3 DOF). Then closest-point project every
    vertex to the input surface for fidelity.

    O_orig:(Vo,3,2), N_orig:(Vo,3) for original verts only; split verts use a
    frame from O of an endpoint. Returns p (Nverts,3)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    nv = len(verts3d)
    Vo = len(N_orig)
    # per-vertex local 2D frame O (3x2): original -> O_orig; split -> identity-ish
    # For split verts we allow full 3-DOF; encode all verts with up to 3 columns.
    # Simpler robust scheme: solve in R^3 with a soft tangent penalty for
    # original verts (keep them near their plane) -- but spec says hard tangent.
    # We use: variables = 2 DOF for original verts (a,b in O frame around v0),
    #         3 DOF for split verts. Build index map.
    var_off = np.zeros(nv, np.int64)
    var_dim = np.zeros(nv, np.int64)
    cur = 0
    O_all = np.zeros((nv, 3, 2))
    anchor = verts3d.copy()
    for i in range(nv):
        if i < Vo and is_orig[i]:
            O_all[i] = O_orig[i]
            var_off[i] = cur; var_dim[i] = 2; cur += 2
        else:
            var_off[i] = cur; var_dim[i] = 3; cur += 3
    ncol = cur
    # build LLS rows: for edge (u,v): p_v - p_u = rho*(O_u d_uv)
    # p_i = anchor_i + (frame_i @ x_i)  where frame is O (3x2) or I3.
    rows_i = []; cols_i = []; data = []
    rhs = []
    rcount = 0
    rho = scale
    def frame(i):
        return O_all[i] if var_dim[i] == 2 else np.eye(3)
    for (u, v), dd in zip(edges, edge_d):
        Ou = frame(u)
        target = rho * (Ou @ dd.astype(float))     # desired p_v - p_u
        base = (anchor[v] - anchor[u]) - target     # constant part
        Fu = frame(u); Fv = frame(v)
        for comp in range(3):
            # (anchor_v + Fv x_v)[comp] - (anchor_u + Fu x_u)[comp] = target[comp]
            #  Fv x_v - Fu x_u = target - (anchor_v - anchor_u)
            r = rcount
            for k in range(var_dim[v]):
                rows_i.append(r); cols_i.append(var_off[v] + k); data.append(Fv[comp, k])
            for k in range(var_dim[u]):
                rows_i.append(r); cols_i.append(var_off[u] + k); data.append(-Fu[comp, k])
            rhs.append(target[comp] - (anchor[v][comp] - anchor[u][comp]))
            rcount += 1
    # gauge: pin one vertex (add weak anchor on all original verts -> stay near v)
    wA = 1e-3
    for i in range(nv):
        if var_dim[i] == 2:
            for k in range(2):
                rows_i.append(rcount); cols_i.append(var_off[i] + k); data.append(wA)
                rhs.append(0.0); rcount += 1
        else:
            for k in range(3):
                rows_i.append(rcount); cols_i.append(var_off[i] + k); data.append(wA)
                rhs.append(0.0); rcount += 1
    A = sp.csr_matrix((data, (rows_i, cols_i)), shape=(rcount, ncol))
    bvec = np.array(rhs)
    x = spla.lsqr(A, bvec, atol=1e-10, btol=1e-10, iter_lim=4000)[0]
    p = anchor.copy()
    for i in range(nv):
        p[i] = anchor[i] + frame(i) @ x[var_off[i]:var_off[i] + var_dim[i]]
    if project:
        try:
            cp, _, _ = mesh.nearest.on_surface(p)
            p = np.asarray(cp)
        except Exception:
            pass
    return p


def resolve_positions_lls(V0, E, d, O, N, scale, proj_mesh=None):
    """Eq.9 on arbitrary vertex/edge arrays (used after face subdivision).
    min_p sum_(u,v) ||p_v - p_u - rho*O_u d_uv||^2, each p constrained to its
    tangent plane (2 DOF). Project to proj_mesh surface for fidelity."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    V0 = np.asarray(V0, float)
    nv = len(V0)
    rho = scale
    rows_i = []; cols_i = []; data = []; rhs = []
    rc = 0
    for idx in range(len(E)):
        u, v = int(E[idx, 0]), int(E[idx, 1])
        target = rho * (O[u] @ d[idx].astype(float))
        Fu = O[u]; Fv = O[v]; base = V0[v] - V0[u]
        for comp in range(3):
            for k in range(2):
                rows_i.append(rc); cols_i.append(2 * v + k); data.append(Fv[comp, k])
                rows_i.append(rc); cols_i.append(2 * u + k); data.append(-Fu[comp, k])
            rhs.append(target[comp] - base[comp]); rc += 1
    wA = 1e-3
    for i in range(nv):
        for k in range(2):
            rows_i.append(rc); cols_i.append(2 * i + k); data.append(wA)
            rhs.append(0.0); rc += 1
    A = sp.csr_matrix((data, (rows_i, cols_i)), shape=(rc, 2 * nv))
    x = spla.lsqr(A, np.array(rhs), atol=1e-10, btol=1e-10, iter_lim=3000)[0]
    p = V0.copy()
    for i in range(nv):
        p[i] = V0[i] + O[i] @ x[2 * i:2 * i + 2]
    if proj_mesh is not None:
        try:
            cp, _, _ = proj_mesh.nearest.on_surface(p)
            p = np.asarray(cp)
        except Exception:
            pass
    return p


def resolve_positions_orig(mesh, E, d, O, N, scale, project=True):
    """Eq.9 on the ORIGINAL mesh: min_p sum_(u,v) ||p_v - p_u - rho*O_u d_uv||^2,
    each vertex p_v constrained to its tangent plane (2 DOF in frame O_v around
    the surface vertex v0). Then closest-point project to the input surface.

    E:(Ne,2) canonical edges, d:(Ne,2) integer offsets, O:(V,3,2), N:(V,3).
    Returns p (V,3)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    V0 = np.asarray(mesh.vertices, float)
    nv = len(V0)
    rho = scale
    rows_i = []; cols_i = []; data = []; rhs = []
    rc = 0
    for idx in range(len(E)):
        u, v = int(E[idx, 0]), int(E[idx, 1])
        dd = d[idx].astype(float)
        target = rho * (O[u] @ dd)                # desired (p_v - p_u) in R^3
        Fu = O[u]; Fv = O[v]                       # (3,2) each
        base = (V0[v] - V0[u])                     # anchors
        for comp in range(3):
            r = rc
            for k in range(2):
                rows_i.append(r); cols_i.append(2 * v + k); data.append(Fv[comp, k])
                rows_i.append(r); cols_i.append(2 * u + k); data.append(-Fu[comp, k])
            rhs.append(target[comp] - base[comp])
            rc += 1
    # weak anchor so the system is well-posed (keep p near v0)
    wA = 1e-3
    for i in range(nv):
        for k in range(2):
            rows_i.append(rc); cols_i.append(2 * i + k); data.append(wA)
            rhs.append(0.0); rc += 1
    A = sp.csr_matrix((data, (rows_i, cols_i)), shape=(rc, 2 * nv))
    x = spla.lsqr(A, np.array(rhs), atol=1e-10, btol=1e-10, iter_lim=3000)[0]
    p = V0.copy()
    for i in range(nv):
        p[i] = V0[i] + O[i] @ x[2 * i:2 * i + 2]
    if project:
        try:
            cp, _, _ = mesh.nearest.on_surface(p)
            p = np.asarray(cp)
        except Exception:
            pass
    return p


# ===========================================================================
#  STEP 8 -- extraction (collapse zero edges; quad per hypotenuse)
# ===========================================================================
class _UF:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def extract_quads(mesh, sys, d, p3d, edges, edge_d, verbose=False):
    """STEP 8. The position field (now |d|inf<=1, singularity-free) makes each
    triangle right-isosceles. (1) union-find collapse every zero edge (d=0).
    (2) for each hypotenuse edge (|d|_1 == 2), the quad = the two triangles
    sharing it. We work on the SUBDIVIDED edge graph: build triangle list from
    the mesh faces mapped through subdivision is complex, so instead we extract
    on the ORIGINAL triangle mesh using the repaired (pre-subdivision) offsets,
    which is what QuadriFlow's |d|inf<=1 guarantee enables.

    Returns (Vq, Q)."""
    # collapse zero edges on the ORIGINAL mesh using repaired d (post-subdiv the
    # original edges may have |d|inf<=1 already after MCF in coarse regimes).
    F = sys["F"]
    E = sys["E"]
    nv = p3d.shape[0] if p3d is not None else len(mesh.vertices)
    Vo = len(mesh.vertices)
    uf = _UF(Vo)
    eidx = {(int(a), int(b)): i for i, (a, b) in enumerate(E)}
    d1 = np.abs(d).sum(axis=1)               # |d|_1 per edge
    dinf = np.abs(d).max(axis=1)
    # collapse zero edges
    for i in range(len(E)):
        if d1[i] == 0:
            uf.union(int(E[i, 0]), int(E[i, 1]))
    # representative vertex positions: average original positions of each class
    classes = defaultdict(list)
    for vtx in range(Vo):
        classes[uf.find(vtx)].append(vtx)
    rep_pos = {}
    src = p3d if p3d is not None else np.asarray(mesh.vertices, float)
    for r, members in classes.items():
        rep_pos[r] = src[members].mean(axis=0)
    rep_ids = {r: k for k, r in enumerate(rep_pos.keys())}
    Vq = np.array([rep_pos[r] for r in rep_pos.keys()])

    # build quads: per hypotenuse edge, the two adjacent triangles form a quad.
    edge2tri = sys["edge2tri"]
    quads = []
    seen = set()
    for i in range(len(E)):
        if d1[i] != 2:           # hypotenuse has |d|_1 == 2 (legs have 1)
            continue
        tris = edge2tri.get(i, [])
        if len(tris) != 2:
            continue
        t0, t1 = tris
        # the quad's 4 corners = the 2 shared (hypotenuse) verts + the 2 apexes
        a, b = int(E[i, 0]), int(E[i, 1])
        tri0 = set(int(x) for x in F[t0]); tri1 = set(int(x) for x in F[t1])
        apex0 = list(tri0 - {a, b}); apex1 = list(tri1 - {a, b})
        if len(apex0) != 1 or len(apex1) != 1:
            continue
        c = apex0[0]; e = apex1[0]
        # quad in order a, apex0, b, apex1
        quad = [uf.find(a), uf.find(c), uf.find(b), uf.find(e)]
        qmap = [rep_ids[r] for r in quad]
        # skip degenerate (collapsed) quads
        if len(set(qmap)) != 4:
            continue
        key = tuple(sorted(qmap))
        if key in seen:
            continue
        seen.add(key)
        quads.append(qmap)
    Q = np.array(quads, np.int64) if quads else np.zeros((0, 4), np.int64)
    if verbose:
        print(f"    [extract] zero-edges={int((d1==0).sum())} "
              f"hypotenuse={int((d1==2).sum())} leg={int((d1==1).sum())} "
              f"|d|inf>1={int((dinf>1).sum())} quads={len(Q)} verts={len(Vq)}")
    return Vq, Q


def extract_quads_lattice(mesh, sys, d, p3d, verbose=False, clean=False):
    """Robust extraction via the INTEGER LATTICE QUOTIENT.

    Integrate the (repaired) integer offsets over a BFS spanning tree of the
    mesh to give every vertex an integer 2D lattice coordinate g[v] in the
    ROOT frame (carrying the cross-field rotation along the way). Where the
    field is singularity-free this is path-independent. Then:
      * two mesh vertices merge iff they share the same lattice coord g
        (this is the zero-edge collapse, done globally & consistently);
      * each mesh face maps to a right-isosceles lattice triangle; two faces
        sharing a hypotenuse (the |g_a - g_b|_1 == 2 edge) give one quad.
    Watertight & manifold by construction on the region where the integration
    is consistent. Edges with |d|inf>1 (long, un-subdivided) are treated as
    tree-forbidden so they don't corrupt the integration; faces touching an
    inconsistency are dropped (small holes, reported), not torn.

    Returns (Vq, Q)."""
    F = sys["F"]
    E = sys["E"]
    Vo = int(F.max()) + 1 if len(F) else (len(p3d) if p3d is not None else 0)
    eidx = {(int(a), int(b)): i for i, (a, b) in enumerate(E)}
    if mesh is not None:
        N, _, _ = fq.tangent_frames(mesh)
    else:
        N = sys.get("N")
    # per-vertex cross dir from frames stored in sys? recompute from mesh+d not
    # available; we use the stored tri_rotk via edge frame rotations instead.
    # We need, per directed mesh edge (a->b), the integer step in a's frame and
    # the rotation taking a's frame to b's frame. Recover from cross field:
    src = p3d if p3d is not None else np.asarray(mesh.vertices, float)

    # cross dir q per vertex (same construction as vertex_frames, but we only
    # have N here; rebuild q from the field is not stored). Use sys rotations:
    # For canonical edge e=(lo,hi): d_e is in lo's frame. k_lohi rotates lo->hi.
    # Build adjacency with (offset_in_a_frame, k_ab) per directed edge.
    # k for a->b: cross_rotation; we stored neither q nor k globally. Recompute
    # via the SAME helper used to build the system: _k_pair needs qd. Recover qd
    # from N by re-deriving the field is wrong. Instead we *require* the caller
    # to pass q; fall back to using the offsets' own consistency (rotation 0).
    qd = sys.get("qd", None)
    Nq = sys.get("N", N)

    def kab(a, b):
        # integer rotation taking a's cross frame to b's cross frame
        if qd is None:
            return 0
        return cross_rotation(qd[a], qd[b], Nq[a], Nq[b])

    d1 = np.abs(d).sum(axis=1)
    dinf = np.abs(d).max(axis=1)
    # adjacency: for each vertex, (neighbor, edge_index, dir_sign)
    adj = defaultdict(list)
    for i in range(len(E)):
        a, b = int(E[i, 0]), int(E[i, 1])
        if dinf[i] > 1:
            continue                      # forbid long edges in the integration
        adj[a].append((b, i, +1))        # canonical lo->hi
        adj[b].append((a, i, -1))

    # BFS integrate. g[v] integer 2D in ROOT frame; krot[v] = rotation taking
    # v's local cross frame INTO the root frame (so a step measured in v's frame
    # is rotated by R2(krot[v]) before being added to g).
    g = np.zeros((Vo, 2), np.int64)
    krot = np.zeros(Vo, np.int64)
    seen = np.zeros(Vo, bool)
    comp = -np.ones(Vo, np.int64)
    from collections import deque
    ncomp = 0
    for s0 in range(Vo):
        if seen[s0]:
            continue
        seen[s0] = True; comp[s0] = ncomp
        dq = deque([s0])
        while dq:
            a = dq.popleft()
            for (b, i, sgn) in adj[a]:
                if seen[b]:
                    continue
                seen[b] = True; comp[b] = ncomp
                de = d[i].astype(np.int64)        # canonical offset (lo's frame)
                lo, hi = int(E[i, 0]), int(E[i, 1])
                if sgn > 0:
                    # a=lo, b=hi: step a->b in a's(=lo) frame = de
                    step_a = de
                else:
                    # a=hi, b=lo: step a->b in a's(=hi) frame.
                    # offset de is lo->hi in lo's frame; reverse & rotate to hi:
                    step_a = -(_R2(kab(lo, hi)) @ de)
                g[b] = g[a] + (_R2(krot[a]) @ step_a)
                krot[b] = (krot[a] + kab(a, b)) % 4
                dq.append(b)
        ncomp += 1

    # merge vertices by (component, lattice coord)
    key_of = {}
    label = np.empty(Vo, np.int64)
    reps = []
    for v in range(Vo):
        key = (int(comp[v]), int(g[v, 0]), int(g[v, 1]))
        if key not in key_of:
            key_of[key] = len(reps)
            reps.append([])
        label[v] = key_of[key]
        reps[label[v]].append(v)
    nq = len(reps)
    Vq = np.zeros((nq, 3))
    for r, members in enumerate(reps):
        Vq[r] = src[members].mean(axis=0)

    # emit quads: per hypotenuse edge (|d|_1==2) with 2 adjacent faces
    edge2tri = sys["edge2tri"]
    quads = []
    qseen = set()
    dropped = 0
    for i in range(len(E)):
        if d1[i] != 2 or dinf[i] > 1:
            continue
        tris = edge2tri.get(i, [])
        if len(tris) != 2:
            continue
        t0, t1 = tris
        a, b = int(E[i, 0]), int(E[i, 1])
        if comp[a] != comp[b]:
            dropped += 1; continue
        tri0 = set(int(x) for x in F[t0]); tri1 = set(int(x) for x in F[t1])
        apex0 = list(tri0 - {a, b}); apex1 = list(tri1 - {a, b})
        if len(apex0) != 1 or len(apex1) != 1:
            continue
        c = apex0[0]; e = apex1[0]
        quad = [label[a], label[c], label[b], label[e]]
        if len(set(quad)) != 4:
            dropped += 1; continue
        key = tuple(sorted(quad))
        if key in qseen:
            continue
        qseen.add(key)
        quads.append(quad)
    Q = np.array(quads, np.int64) if quads else np.zeros((0, 4), np.int64)
    if clean and len(Q):
        Vq, Q = _manifold_clean(Vq, Q)
    if verbose:
        print(f"    [extract-lattice] verts {Vo}->{nq} quads={len(Q)} "
              f"dropped={dropped} long_edges={int((dinf>1).sum())} "
              f"components={ncomp}")
    return Vq, Q


def _fill_small_holes(Vq, Q, max_loop=8):
    """Close small boundary loops (<= max_loop edges) of the quad mesh so it
    becomes watertight where the holes are just residual-singularity artifacts.
    Quad loops (4,6,8 edges) are filled with quads by ear-pairing; odd loops get
    one filler triangle plus quads. Large loops (true field defects) are left
    open. Returns (Vq, Q, n_filled). Q may contain a few triangles (stored as a
    quad with a repeated last index) -- callers triangulating handle that."""
    from collections import Counter, defaultdict
    Q = [list(map(int, q)) for q in Q]
    ec = Counter()
    edge_dir = {}
    for q in Q:
        for k in range(4):
            a, b = q[k], q[(k + 1) % 4]
            ec[(min(a, b), max(a, b))] += 1
            edge_dir[(a, b)] = True
    bnd = [e for e, c in ec.items() if c == 1]
    if not bnd:
        return np.asarray(Vq), np.array(Q, np.int64), 0
    adj = defaultdict(list)
    bset = set(bnd)
    for (a, b) in bnd:
        adj[a].append(b); adj[b].append(a)
    visited = set()
    filled = 0
    for start in list(adj.keys()):
        if start in visited or len(adj[start]) == 0:
            continue
        # trace a loop
        loop = [start]; visited.add(start); cur = start; prev = None
        ok = True
        while True:
            cands = [x for x in adj[cur] if x != prev and x not in visited]
            if not cands:
                # close back to start?
                if start in adj[cur] and len(loop) >= 3:
                    break
                ok = False; break
            nx = cands[0]
            if nx == start:
                break
            loop.append(nx); visited.add(nx); prev = cur; cur = nx
            if len(loop) > max_loop:
                ok = False; break
        if not ok or len(loop) < 3 or len(loop) > max_loop:
            continue
        # fan-fill the loop from loop[0]
        for k in range(1, len(loop) - 1):
            Q.append([loop[0], loop[k], loop[k + 1], loop[k + 1]])  # triangle
        filled += 1
    return np.asarray(Vq), np.array(Q, np.int64), filled


def _manifold_clean(Vq, Q, max_passes=6):
    """Drop quads that create non-manifold edges (edge in >2 quads), then drop
    quads referencing now-unreferenced structure, repeat until every edge is in
    <=2 quads. Finally compact vertices. Leaves small holes rather than tears;
    guarantees max edge-face incidence <=2."""
    Q = [list(map(int, q)) for q in Q]
    for _ in range(max_passes):
        edge_q = defaultdict(list)
        for qi, q in enumerate(Q):
            for k in range(4):
                a, b = q[k], q[(k + 1) % 4]
                edge_q[(min(a, b), max(a, b))].append(qi)
        bad_q = set()
        for e, qs in edge_q.items():
            if len(qs) > 2:
                # drop all but the 2 quads on this edge (keep first two)
                for qi in qs[2:]:
                    bad_q.add(qi)
        if not bad_q:
            break
        Q = [q for qi, q in enumerate(Q) if qi not in bad_q]
    if not Q:
        return Vq, np.zeros((0, 4), np.int64)
    Q = np.array(Q, np.int64)
    used = np.unique(Q)
    remap = -np.ones(len(Vq), np.int64); remap[used] = np.arange(len(used))
    return Vq[used], remap[Q]


# ===========================================================================
#  TOP LEVEL
# ===========================================================================
def remesh(mesh, target_quads=1500, verbose=False, target_faces=8000,
           pos_iters=15, min_scale_ratio=1.0, curv_field=False):
    """Full QuadriFlow extraction pipeline.
       clean -> cross-field -> position-field -> integer offsets ->
       MCF regularity repair -> subdivide -> inversion repair ->
       position re-solve + project -> extract.
    Returns (Vq (N,3) float, Q (M,4) int). Never raises.

    The Instant-Meshes position field is only coherent when the lattice spacing
    rho is comfortably COARSER than the input triangulation (rho >=
    min_scale_ratio * mean_edge_length); otherwise nearly every triangle becomes
    a position singularity and the MCF cannot repair them all. We therefore clamp
    rho up to that floor (so the achievable quad count is bounded by the input
    density) and, if the MCF is still infeasible, grow rho and retry."""
    import time
    try:
        # decimate so the band-clamped rho (~mean_edge) yields ~target_quads:
        # a triangle mesh of ~2*target_quads faces gives ~target_quads quads.
        tf = min(target_faces, max(800, int(2.2 * target_quads)))
        m = clean_mesh(mesh, target_faces=tf)
        area = float(m.area)
        el = float(np.linalg.norm(
            m.vertices[m.edges_unique[:, 0]] - m.vertices[m.edges_unique[:, 1]],
            axis=1).mean())
        scale_req = float(np.sqrt(area / max(target_quads, 1)))
        # The Instant-Meshes position field is only coherent in a narrow band
        # rho ~ [1.0, 1.3] * mean_edge: too small -> sub-cell noise, too large ->
        # the lattice is coarser than the mesh and single edges can't carry a
        # lattice step (extraction collapses). We clamp rho into this band; the
        # achievable quad count is therefore ~ input_faces/3. To hit a coarser
        # target, decimate the input first (lower target_faces).
        lo, hi = min_scale_ratio * el, 1.3 * el
        scale = float(np.clip(scale_req, lo, hi))
        if verbose:
            print(f"  clean: V={len(m.vertices)} F={len(m.faces)} area={area:.3f} "
                  f"mean_edge={el:.4f} rho_req={scale_req:.4f} rho={scale:.4f} "
                  f"(ratio {scale/el:.2f}, band [{lo:.3f},{hi:.3f}])")
        t0 = time.time()
        d = fq.smooth_field(m, curv=curv_field, verbose=verbose)
        N, O = vertex_frames(m, d)
        qd = O[:, :, 0]
        if verbose:
            print(f"  field: {time.time()-t0:.1f}s")

        # position field + MCF. We operate at rho ~ mean_edge (ratio ~1) so the
        # ORIGINAL mesh edges carry single lattice steps (legs d1=1, hypotenuses
        # d1=2) -- the regime extraction needs. The MCF removes ~90% of the
        # position singularities; the rest remain as a few extra irregular
        # vertices (the paper accepts this best-effort routing for hard nets).
        t0 = time.time()
        o, Np, qp = position_field(m, d, scale, iters=pos_iters)
        off = integer_offsets(m, o, Np, qp, scale)
        tpf = time.time() - t0
        t0 = time.time()
        drep, stats = repair_regularity(m, off, qd, N, verbose=verbose, max_H=6)
        if verbose:
            print(f"  rho={scale:.4f} pos_field {tpf:.1f}s MCF {time.time()-t0:.1f}s "
                  f"feasible={stats['feasible']} residual_tris={stats['residual_triangles']}")
        sys = stats["sys"]
        sys["qd"] = qd; sys["N"] = N
        # inversion repair on repaired offsets
        drep, invstats = repair_inversions(sys, drep, verbose=verbose)
        # position re-solve on the original mesh (Eq.9, tangent-constrained) +
        # closest-point projection to the input surface (fidelity).
        p_orig = resolve_positions_orig(m, off["E"], drep, O, N, scale,
                                        project=True)
        Vq, Q = extract_quads_lattice(m, sys, drep, p_orig, verbose=verbose,
                                      clean=True)
        return Vq, Q
    except Exception as ex:
        if verbose:
            import traceback; traceback.print_exc()
        return np.zeros((0, 3)), np.zeros((0, 4), np.int64)


# ===========================================================================
#  TOP LEVEL  --  Instant-Meshes ring-tracing extraction (the robust path)
# ===========================================================================
def remesh_im(mesh, target_quads=1500, verbose=False, pos_iters=25,
              scale_ratio=1.0, use_mcf=False, curv_field=False):
    """Instant-Meshes ring-tracing quad remesher (the ROBUST field-aligned path).

       clean -> 4-RoSy cross-field -> 4-PoSy position field ->
       [optional MCF regularity repair of the integer offsets] ->
       extract_graph (lattice collapse) -> extract_faces (ring tracing) ->
       manifold + watertight finish (drop non-manifold, cap holes, relax+project).

    Unlike the QuadriFlow hypotenuse extraction (extract_quads_lattice), position
    singularities become IRREGULAR VERTICES, not dropped faces. Returns
    (Vq (N,3) float, Q (M,4) int). Never raises.

    The position field is coherent when the lattice spacing rho ~ mean input edge
    (scale_ratio ~ 1). To reach target_quads we decimate the input so that a mesh
    of ~target_quads vertices yields ~target_quads lattice cells.

    use_mcf=True : run the QuadriFlow min-cost-flow regularity repair (Eq 3) on
    the integer offsets BEFORE extraction, so `extract_graph` collapses from a
    SINGULARITY-FREE position field. The local position field injects hundreds of
    spurious singularities (each -> an irregular vertex); the MCF removes the ones
    it can route away, raising val4 while the ring-tracing keeps the result
    watertight and the orientation field (hence the flow) is unchanged.
    """
    import time
    from . import _im_extract as ie
    from . import _grid_place as gp
    try:
        t_all = time.time()
        # At rho ~ mean_edge the lattice has ~one cell per INPUT VERTEX, so the
        # output quad count ~ input vertex count. Extraction QUALITY (val4) rises
        # sharply with mesh ISOTROPY: quadric-decimated meshes are anisotropic
        # (CV ~ 0.3-0.5, long thin triangles on limbs) which fills the position
        # field with long edges. The cure is ISOTROPIC REMESH -- we target an edge
        # length L = sqrt(area / target_quads) so the remesh ALSO sets the output
        # density (one quad per ~L^2 of area -> ~target_quads quads), and it
        # produces near-equilateral triangles (CV ~ 0.12) that the lattice loves.
        # Decimate first only to bound the remesh runtime on huge inputs.
        m = clean_mesh(mesh, target_faces=min(len(mesh.faces), 30000))
        area = float(m.area)
        # Extraction val4 rises with density (more triangles per quad cell average
        # out the position-field noise). We therefore remesh at ~1.4x the target
        # linear density (~2x the quad count); the achievable quad count is then
        # ~2*target_quads at a much higher val4. The output edge length:
        L = float(np.sqrt(area / max(target_quads, 1))) / 1.4
        el0 = float(np.linalg.norm(
            m.vertices[m.edges_unique[:, 0]] - m.vertices[m.edges_unique[:, 1]],
            axis=1).mean())
        cv0 = float(np.std(np.linalg.norm(
            m.vertices[m.edges_unique[:, 0]] - m.vertices[m.edges_unique[:, 1]],
            axis=1)) / el0)
        # Re-mesh when the input is anisotropic, OR much too DENSE for the target
        # (so we coarsen). A near-isotropic grid (low CV) that is at-or-coarser
        # than the target is left ALONE -- remeshing a clean tube/cylinder grid
        # only adds chord error and breaks its perfect axial+circumferential flow.
        need_remesh = cv0 > 0.22 or el0 < 0.6 * L
        if need_remesh:
            try:
                m2 = isotropic_remesh(m, target_edge=L, iters=7)
                if (m2.is_watertight == m.is_watertight and len(m2.faces) > 0
                        and len(m2.vertices) >= 0.2 * max(target_quads, 1)):
                    m = m2
            except Exception:
                pass
        el = float(np.linalg.norm(
            m.vertices[m.edges_unique[:, 0]] - m.vertices[m.edges_unique[:, 1]],
            axis=1).mean())
        scale = scale_ratio * el
        if verbose:
            el_cv = float(np.std(np.linalg.norm(
                m.vertices[m.edges_unique[:, 0]] - m.vertices[m.edges_unique[:, 1]],
                axis=1)) / el)
            print(f"  clean: V={len(m.vertices)} F={len(m.faces)} "
                  f"mean_edge={el:.4f} cv={el_cv:.3f} rho={scale:.4f}")
        t0 = time.time()
        d = fq.smooth_field(m, curv=curv_field, verbose=verbose)
        N, O = vertex_frames(m, d)
        q = O[:, :, 0]
        if verbose:
            print(f"  field: {time.time()-t0:.1f}s")
        t0 = time.time()
        o, Np, qp = position_field(m, d, scale, iters=pos_iters)
        if verbose:
            print(f"  position field: {time.time()-t0:.1f}s")

        # --- OPTIONAL: MCF regularity repair -> singularity-free offsets -------
        # The local position field snaps each vertex to a lattice cell, but the
        # per-triangle offset sums are NOT zero -> hundreds of position
        # singularities. The min-cost flow (QuadriFlow Eq 3) routes the integer
        # offsets to make each triangle's offset sum 0 (singularity-free) while
        # minimizing |d - seed|_1. We seed it on the SAME field the extractor
        # reads (`step` = the rounded o[v]-o[u] cell difference) and use
        # max-flow-min-cost so the repaired offsets stay L1-close to the
        # geometry; the projected `o` still drives output-vertex positions
        # (fidelity) and the orientation field is untouched (flow preserved).
        # MEASURED FINDING: on organic PSB shapes this REDUCES val4 (the global
        # re-routing needed to kill the singularities moves many edges a full
        # cell, so extract_graph over-merges geometrically-distant verts). It is
        # exposed behind use_mcf for reproducibility; default OFF.
        drep = None
        if use_mcf:
            t0 = time.time()
            off = integer_offsets(m, o, Np, qp, scale)
            ld0 = ie._edge_lattice_data(m, o, Np, qp, scale)
            seed = ld0["step"]
            sysb = build_regularity_system(m, off, qp, N)
            res_before = int(np.sum(np.any(_residual_b(sysb, seed) != 0, axis=1)))
            drep, mstats = repair_regularity(m, off, qp, N, verbose=verbose,
                                             max_H=6, min_cost=True,
                                             dstar_override=seed)
            if verbose:
                l1 = int(np.abs(np.asarray(drep) - seed).sum())
                print(f"  MCF: {time.time()-t0:.1f}s feasible={mstats['feasible']} "
                      f"residual_tris {res_before} -> {mstats['residual_triangles']} "
                      f"frozen={mstats['frozen']} |drep-step|1={l1}")

        # project mesh-vertex anchors to drive cluster means onto the surface
        t0 = time.time()
        graph = ie.extract_graph(m, o, Np, qp, scale, p3d=None, verbose=verbose,
                                 drep=drep)
        faces, Vq = ie.extract_faces(graph, verbose=verbose)
        if not faces:
            return np.zeros((0, 3)), np.zeros((0, 4), np.int64)
        Q = np.asarray(faces, np.int64)
        if verbose:
            print(f"  extract: {time.time()-t0:.1f}s raw quads={len(Q)}")

        # --- manifold + watertight finish (reuse _grid_place) ---------------
        t0 = time.time()
        Vq = gp.project_to_surface(m, Vq)                 # onto input surface
        Q = gp._drop_degenerate_quads(Vq, Q)
        Vq, Q = gp._enforce_manifold(m, Vq, Q)            # edge->face <=2
        Vq, Q, ncap = gp._cap_border_holes(m, Vq, Q)      # close residual holes
        Vq, Q = gp._enforce_manifold(m, Vq, Q)            # re-check after caps
        # IM final valence cleanup: annihilate doublets + 3-5 dipoles (the bulk of
        # the position-field irregular noise) -- lifts val4 substantially.
        Vq, Q = ie.clean_quad_valence(Vq, Q, iters=5)
        Vq, Q = gp._enforce_manifold(m, Vq, Q)
        Vq, Q, ncap2 = gp._cap_border_holes(m, Vq, Q)
        Vq, Q = gp._enforce_manifold(m, Vq, Q)
        # compact unused vertices
        if len(Q):
            used = np.unique(Q)
            remap = -np.ones(len(Vq), np.int64); remap[used] = np.arange(len(used))
            Vq = Vq[used]; Q = remap[Q]
        Vq = gp.relax_quads(m, Vq, Q, iters=6, lam=0.5)   # tangential + reproject
        if verbose:
            print(f"  finish: {time.time()-t0:.1f}s capped_holes={ncap} "
                  f"final quads={len(Q)} verts={len(Vq)} "
                  f"total {time.time()-t_all:.1f}s")
        return Vq, Q
    except Exception:
        if verbose:
            import traceback; traceback.print_exc()
        return np.zeros((0, 3)), np.zeros((0, 4), np.int64)
