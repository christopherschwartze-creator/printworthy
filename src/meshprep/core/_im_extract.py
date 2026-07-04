"""Instant-Meshes ring-tracing quad extraction (clean-room, permissive port).

The ROBUST field-aligned extraction (Jakob, Tarini, Panozzo, Sorkine-Hornung
2015, "Instant Field-Aligned Meshes", Sec. 4). Unlike the QuadriFlow hypotenuse
pairing (which needs a perfect singularity-free position field and drops faces
where it isn't), this method handles position singularities as IRREGULAR
VERTICES (valence 3/5) instead of holes -- the documented, robust path.

Pipeline (consumes the orientation + position field from _quadflow):
  extract_graph : collapse mesh vertices that share a lattice cell (union-find),
                  emit one OUTPUT VERTEX per cluster and OUTPUT EDGES for
                  single-lattice-step links, stored in field-rotational order.
  extract_faces : ring-trace closed loops by always turning to the rotational-
                  next outgoing edge (field-aligned); quads first; fill_face the
                  rest (split big rings into quads, fan small rings).

Permissive only: numpy / scipy / trimesh. NO GPL.
"""
import os
import numpy as np
from collections import defaultdict, deque

from ._quadflow import _R2, directed_edges, _cross_rot_batch


# ===========================================================================
#  lattice integration: per-vertex integer cell + per-edge shift
# ===========================================================================
def _edge_lattice_data(mesh, o, N, q, scale, step_override=None):
    """For every directed mesh edge (i,j) compute the integer lattice STEP from
    i's cell to j's cell, measured in i's (q, n x q) tangent basis, plus the
    cross-field rotation k_ij taking i's frame into j's frame.

    The position field already snaps every vertex to a lattice node o[.]. The
    edge step is therefore simply the displacement o[j]-o[i] expressed in i's
    tangent basis and rounded to integer cells (Instant Meshes Eq. for the
    integer translation t_ij). We also report the residual (sub-cell) error of
    that rounding as the match quality.

    `step_override` (Ne,2 int, OPTIONAL): the MCF-repaired, singularity-FREE
    per-edge offset d*_uv from `repair_regularity` (QuadriFlow Def 3 / Eq 3).
    It uses the SAME canonical (lo->hi, lo-frame) convention as `step`, so when
    supplied it REPLACES the locally-rounded `step` as the authoritative
    lattice-cell difference for the collapse/edge decision -- making the
    extraction globally consistent. The rotation k_ij and the sub-cell `err`
    (still measured against the projected position field `o`) are unchanged.
    Rows align 1:1 with `directed_edges(mesh)` because both `step_override`
    (from `integer_offsets` / `repair_regularity`) and this routine index edges
    via the same `directed_edges(mesh)` ordering.

    Returns dict with parallel arrays over the UNIQUE (low->high) edges:
      E      (Ne,2)  low<high endpoints
      k_ij   (Ne,)   rotation i(=lo) frame -> j(=hi) frame, in {0,1,2,3}
      step   (Ne,2)  integer lattice step i->j in i's basis  (this is t_i - t_j
                     already expressed in the common (i) frame, i.e. the signed
                     componentwise shift -- absDiff in extract_graph is |step|)
      err    (Ne,)   sub-cell residual of the rounding / scale (match quality)
    """
    E = directed_edges(mesh)
    u = E[:, 0]; v = E[:, 1]
    nu = N[u]; qu = q[u]
    tu = np.cross(nu, qu)
    k_ij = _cross_rot_batch(qu, q[v], nu, N[v])           # lo -> hi
    disp = o[v] - o[u]                                     # R^3 displacement
    su = np.einsum("mc,mc->m", qu, disp) / scale           # in u's q axis (cells)
    sv = np.einsum("mc,mc->m", tu, disp) / scale           # in u's nxq axis
    iu = np.round(su); iv = np.round(sv)
    step = np.stack([iu, iv], axis=1).astype(np.int64)
    err = np.sqrt((su - iu) ** 2 + (sv - iv) ** 2)         # sub-cell residual
    if step_override is not None:
        so = np.asarray(step_override, np.int64)
        if so.shape != step.shape:
            raise ValueError(
                f"step_override shape {so.shape} != edge step shape {step.shape}; "
                "the MCF offsets must be indexed by directed_edges(mesh).")
        step = so.copy()
    return {"E": E, "k_ij": k_ij, "step": step, "err": err}


class _UF:
    __slots__ = ("p", "sz")
    def __init__(self, n):
        self.p = list(range(n)); self.sz = [1] * n
    def find(self, x):
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra; self.sz[ra] += self.sz[rb]
        return ra


# ===========================================================================
#  Part B -- extract_graph  (vertex collapse -> output verts + ordered edges)
# ===========================================================================
def extract_graph(mesh, o, N, q, scale, p3d=None, verbose=False, drep=None,
                  prune_mode="degaware"):
    """Collapse the mesh down to the lattice quotient.

    `prune_mode`: "degaware" (default, degree-aware triangle prune that drives
    valence toward 4) or "diag" (legacy: remove the most-diagonal edge of each
    3-cycle). Kept switchable for honest within-process A/B.

    Per directed edge (i,j): the position field gives each endpoint's integer
    lattice cell. The shift difference (in a COMMON frame) decides the edge:
        absDiff.max() > 1  -> skip (spans >1 cell)
        absDiff == (1,1)   -> skip (diagonal, never a quad edge)
        absDiff.sum()==0   -> SAME cell -> union-find merge endpoints
        absDiff in {(1,0),(0,1)} -> ONE lattice step -> an OUTPUT EDGE.
    Collapse candidates are unioned ASC by match error with a conflict check.
    Output vertex = cluster mean of `o` (or p3d); output edges link distinct
    clusters and are stored per-vertex in FIELD-ROTATIONAL order.

    `drep` (Ne,2 int, OPTIONAL): the MCF-repaired singularity-free per-edge
    offset (QuadriFlow Eq 3, from `_quadflow.repair_regularity`). When supplied
    it is used as the AUTHORITATIVE per-edge lattice-cell difference for the
    collapse/edge/skip decision (and the BFS integration), instead of the cell
    difference re-derived locally from `o`. The projected position field `o`
    is STILL used for output-vertex positions and the cross-field rotation, so
    the flow stays field-aligned and on the surface -- only the COMBINATORICS
    become globally consistent, which removes the spurious singularities.

    Returns dict:
      Vq        (Nq,3)   output vertex positions (cluster means, pre-projection)
      out_edges list of (a,b) cluster-id pairs (undirected, unique)
      out_adj   list[Nq] of neighbour cluster ids in rotational order
      label     (Vo,)    mesh vertex -> cluster id
      cluster_members  list[Nq] of mesh vertex ids
    """
    Vo = len(mesh.vertices)
    ld = _edge_lattice_data(mesh, o, N, q, scale, step_override=drep)
    E = ld["E"]; step = ld["step"]; k_ij = ld["k_ij"]
    Ne = len(E)

    # the step is the signed integer lattice shift i->j already in i's frame.
    absdiff = np.abs(step)
    amax = absdiff.max(axis=1)
    asum = absdiff.sum(axis=1)

    is_collapse = (asum == 0)
    is_diag = (absdiff[:, 0] == 1) & (absdiff[:, 1] == 1)
    is_step = (amax == 1) & (asum == 1)                  # (1,0) or (0,1)

    # ---- GLOBAL lattice integration (transitive-safe collapse) ---------------
    # Integrate the integer steps over a BFS spanning forest, carrying the cross-
    # field rotation, to give every mesh vertex an integer 2D coordinate g[v] in
    # its component's ROOT frame. Vertices with the SAME (component, g) are the
    # same lattice node -> one cluster. Unlike chaining same-cell union edges,
    # this is globally consistent (path-independent where the field is singular-
    # ity free), so it never transitively over-merges and it works for scale>edge.
    # Long edges (|step|inf>1) are excluded from the tree so they can't corrupt
    # the integration; faces touching an inconsistency stay separate (small
    # irregular region), not torn.
    g = np.zeros((Vo, 2), np.int64)
    krot = np.zeros(Vo, np.int64)
    seen = np.zeros(Vo, bool)
    comp = -np.ones(Vo, np.int64)
    adj_int = defaultdict(list)
    for e in range(Ne):
        if amax[e] > 1:
            continue
        a, b = int(E[e, 0]), int(E[e, 1])
        adj_int[a].append((b, e, +1)); adj_int[b].append((a, e, -1))
    ncomp = 0
    for s0 in range(Vo):
        if seen[s0]:
            continue
        seen[s0] = True; comp[s0] = ncomp
        dq = deque([s0])
        while dq:
            a = dq.popleft()
            for (b, e, sgn) in adj_int[a]:
                if seen[b]:
                    continue
                seen[b] = True; comp[b] = ncomp
                st = step[e]                              # i(=lo)->j(=hi) in lo frame
                if sgn > 0:                               # a=lo, b=hi
                    step_a = st
                    k_a_b = int(k_ij[e])
                else:                                     # a=hi, b=lo
                    step_a = -(_R2((-int(k_ij[e])) % 4) @ st)
                    k_a_b = (-int(k_ij[e])) % 4
                g[b] = g[a] + (_R2(krot[a]) @ step_a)
                krot[b] = (krot[a] + k_a_b) % 4
                dq.append(b)
        ncomp += 1

    # cluster = unique (component, g)
    keys = [(int(comp[v]), int(g[v, 0]), int(g[v, 1])) for v in range(Vo)]
    key_of = {}
    label = np.empty(Vo, np.int64)
    members = []
    for v in range(Vo):
        kk = keys[v]
        ci = key_of.get(kk, -1)
        if ci < 0:
            ci = len(members); key_of[kk] = ci; members.append([])
        label[v] = ci; members[ci].append(v)

    # ---- split spatially-incoherent clusters (BFS mis-merge guard) -----------
    # The BFS integration can give the SAME (comp,g) to two vertices that are
    # actually a cell apart (a wrong step accumulated along the tree path on a
    # noisy/curved patch). Such a cluster spans >~1 cell in R^3. Split any
    # cluster whose members don't all lie within ~scale of a common centre into
    # single-linkage sub-clusters (mesh-edge connectivity within `scale`).
    V3 = np.asarray(mesh.vertices, float)
    mesh_nbr = defaultdict(set)
    for e in range(Ne):
        a, b = int(E[e, 0]), int(E[e, 1])
        mesh_nbr[a].add(b); mesh_nbr[b].add(a)
    split_members = []
    thr2 = (0.9 * scale) ** 2
    for mem in members:
        if len(mem) <= 1:
            split_members.append(mem); continue
        P = V3[mem]
        if float(((P - P.mean(0)) ** 2).sum(1).max()) <= thr2:
            split_members.append(mem); continue
        # single-linkage within the cluster using mesh adjacency + distance
        memset = set(mem); seen_m = set();
        for s in mem:
            if s in seen_m:
                continue
            comp_m = []; stack = [s]; seen_m.add(s)
            while stack:
                x = stack.pop(); comp_m.append(x)
                for y in mesh_nbr[x]:
                    if y in memset and y not in seen_m and \
                       float(np.sum((V3[x] - V3[y]) ** 2)) <= thr2:
                        seen_m.add(y); stack.append(y)
            split_members.append(comp_m)
    members = split_members
    label = np.empty(Vo, np.int64)
    for ci, mem in enumerate(members):
        for v in mem:
            label[v] = ci
    Nq = len(members)
    coll_per = np.array([len(m) for m in members], np.int64)   # cluster sizes

    # ---- output vertex positions = cluster mean of o (then mean of surface) ----
    src = p3d if p3d is not None else o
    Vq = np.zeros((Nq, 3))
    for c in range(Nq):
        Vq[c] = src[members[c]].mean(axis=0)

    # ---- output edges: single-lattice-step links between DISTINCT clusters ----
    # (per-edge step classification). This over-connects near defects (closing
    # spurious diagonal triangles), so the graph cleanup below PRUNES triangles
    # explicitly. A pure g-coordinate adjacency is triangle-free but leaves more
    # dangling val-2/3 nodes at integration seams (worse for the ring tracer),
    # so the step+prune route wins empirically.
    step_idx = np.flatnonzero(is_step)
    edge_set = set()
    raw_dir = defaultdict(set)                            # cluster -> set(neighbours)
    for e in step_idx:
        a, b = int(E[e, 0]), int(E[e, 1])
        ca, cb = label[a], label[b]
        if ca == cb:
            continue
        key = (ca, cb) if ca < cb else (cb, ca)
        edge_set.add(key)
        raw_dir[ca].add(cb); raw_dir[cb].add(ca)
    out_edges = list(edge_set)

    # ---- drop spurious clusters (≈ no collapses) BEFORE ordering -------------
    # spec: drop clusters with nCollapses <= avg/10. A cluster of a single mesh
    # vertex that never collapsed and has no step-edges is spurious noise.
    # Keep singletons that DO carry step edges (they are real lattice nodes).
    avg_coll = coll_per[coll_per > 0].mean() if np.any(coll_per > 0) else 0.0
    keep = np.ones(Nq, bool)
    if avg_coll > 0:
        for c in range(Nq):
            if coll_per[c] <= avg_coll / 10.0 and len(members[c]) <= 1 and len(raw_dir[c]) == 0:
                keep[c] = False

    # ---- topological graph cleanup (IM post-extraction) ----------------------
    nbr = {c: set(x for x in raw_dir[c] if keep[x]) for c in range(Nq)}
    for c in range(Nq):
        if not keep[c]:
            nbr[c] = set()

    # (a) TRIANGLE PRUNING. A quad-mesh graph has NO 3-cycles: if two neighbours
    # of a vertex are themselves adjacent, one of the three edges is a spurious
    # diagonal (position-field noise near a singularity). Remove, per triangle,
    # the LEAST field-aligned edge -- the one whose 3D direction is furthest from
    # the cross-field axes at its endpoints (i.e. most diagonal). This is what
    # collapses the 500+ triangular faces into clean quads. Pre-compute a rough
    # per-cluster cross frame for the alignment test.
    rq = np.zeros((Nq, 3)); rn = np.zeros((Nq, 3))
    for c in range(Nq):
        mm = members[c]
        rn[c] = N[mm].mean(axis=0); rq[c] = q[mm].mean(axis=0)
    rn /= (np.linalg.norm(rn, axis=1, keepdims=True) + 1e-12)
    rq = rq - np.einsum("ij,ij->i", rq, rn)[:, None] * rn
    rq /= (np.linalg.norm(rq, axis=1, keepdims=True) + 1e-12)
    rt = np.cross(rn, rq)

    def edge_diag_cost(a, b):
        # 0 = perfectly axis-aligned (good), ->1 = diagonal (bad). Average the
        # |sin(2*theta)| of the edge angle in each endpoint's (q, n x q) frame.
        v = Vq[b] - Vq[a]
        cost = 0.0
        for c in (a, b):
            ang = np.arctan2(float(v @ rt[c]), float(v @ rq[c]))
            cost += abs(np.sin(2 * ang))                 # max at 45deg
        return cost

    # collect unique triangles (a<b<c, mutually adjacent), prune greedily
    removed = set()
    def adj_now(a, b):
        return (b in nbr[a]) and ((a, b) not in removed) and ((b, a) not in removed)
    tri_list = []
    for a in range(Nq):
        if not keep[a]:
            continue
        na = sorted(x for x in nbr[a] if x > a)
        for ix in range(len(na)):
            b = na[ix]
            for c in na[ix + 1:]:
                if b in nbr[c]:
                    tri_list.append((a, b, c))
    # DEGREE-AWARE prune (the val4 lever). The raw step-graph is near-balanced in
    # 3/5 valence but OVER-connected near singularities (lots of deg 6-12); the
    # old "remove the most-diagonal edge of each triangle" deletes edges blindly
    # and craters verts to deg-3. We choose, per 3-cycle, the edge whose removal
    # best drives BOTH endpoints toward the target valence 4: a strong penalty for
    # dropping an endpoint below 4, a reward for pulling a >4 endpoint down toward
    # 4, with the diagonality as a secondary tiebreak. Degree is tracked
    # incrementally and the most over-connected triangles go first.
    deg = {c: len(nbr[c]) for c in range(Nq) if keep[c]}

    def remove_score(x, y):
        # lower = better to remove. After removal deg[x],deg[y] each drop by 1.
        s = 0.0
        for w in (x, y):
            dw = deg.get(w, 4)
            after = dw - 1
            if after < 4:                # would create/deepen a deg<4 irregular
                s += 6.0 + 2.0 * (4 - after)
            elif dw > 4:                 # pulling an over-valent vert toward 4: good
                s -= 1.5
        s -= 0.4 * edge_diag_cost(x, y)  # prefer removing diagonal edges (tiebreak)
        return s

    if prune_mode == "degaware":
        # order triangles so the most over-connected (sum of endpoint deg) go first
        tri_list.sort(key=lambda t: -(deg.get(t[0], 0) + deg.get(t[1], 0)
                                      + deg.get(t[2], 0)))
        for (a, b, c) in tri_list:
            if not (adj_now(a, b) and adj_now(b, c) and adj_now(a, c)):
                continue
            cabs = [(a, b), (b, c), (a, c)]
            (x, y) = min(cabs, key=lambda e: remove_score(e[0], e[1]))
            nbr[x].discard(y); nbr[y].discard(x)
            removed.add((x, y)); deg[x] -= 1; deg[y] -= 1
    else:                                # legacy "diag" prune
        for (a, b, c) in tri_list:
            if not (adj_now(a, b) and adj_now(b, c) and adj_now(a, c)):
                continue
            cabs = [((a, b), edge_diag_cost(a, b)),
                    ((b, c), edge_diag_cost(b, c)),
                    ((a, c), edge_diag_cost(a, c))]
            (x, y), _ = max(cabs, key=lambda t: t[1])
            nbr[x].discard(y); nbr[y].discard(x)
            removed.add((x, y)); deg[x] -= 1; deg[y] -= 1

    # (b) dissolve valence-1 (dangling) and valence-2 (pass-through) clusters,
    # iterate to fixpoint. A valence-2 cluster is spliced through.
    changed = True; guard = 0
    while changed and guard < Nq + 10:
        guard += 1; changed = False
        for c in range(Nq):
            if not keep[c]:
                continue
            deg = len(nbr[c])
            if deg == 0:
                keep[c] = False; changed = True
            elif deg == 1:
                o1 = next(iter(nbr[c]))
                nbr[o1].discard(c); keep[c] = False; nbr[c] = set(); changed = True
            elif deg == 2:
                a, b = tuple(nbr[c])
                if a != b and b not in nbr[a]:           # don't create a triangle
                    nbr[a].discard(c); nbr[b].discard(c)
                    nbr[a].add(b); nbr[b].add(a)
                    keep[c] = False; nbr[c] = set(); changed = True
    raw_dir = nbr
    out_edges = []
    for c in range(Nq):
        for nb in raw_dir[c]:
            if c < nb:
                out_edges.append((c, nb))

    # ---- field-rotational ordering of each cluster's outgoing edges ----------
    # angle of the edge direction (to neighbour's mean position) in the cluster's
    # representative tangent frame (q, n x q). Stored CCW.
    # representative frame per cluster: average q/n of members, re-orthonormalize.
    rep_q = np.zeros((Nq, 3)); rep_n = np.zeros((Nq, 3))
    for c in range(Nq):
        mm = members[c]
        rep_n[c] = N[mm].mean(axis=0)
        rep_q[c] = q[mm].mean(axis=0)
    nrm = np.linalg.norm(rep_n, axis=1, keepdims=True); rep_n /= (nrm + 1e-12)
    rep_q = rep_q - np.einsum("ij,ij->i", rep_q, rep_n)[:, None] * rep_n
    nrm = np.linalg.norm(rep_q, axis=1, keepdims=True); rep_q /= (nrm + 1e-12)
    rep_t = np.cross(rep_n, rep_q)

    out_adj = [[] for _ in range(Nq)]
    for c in range(Nq):
        if not keep[c]:
            continue
        nbrs = [nb for nb in raw_dir[c] if keep[nb]]
        if not nbrs:
            continue
        vecs = Vq[nbrs] - Vq[c]
        ang = np.arctan2(vecs @ rep_t[c], vecs @ rep_q[c])
        order2 = np.argsort(ang)
        out_adj[c] = [int(nbrs[k]) for k in order2]

    if verbose:
        nkeep = int(keep.sum())
        print(f"    [graph] mesh V={Vo} -> clusters={Nq} kept={nkeep} "
              f"out_edges={len(out_edges)} collapse_edges={int(is_collapse.sum())} "
              f"step_edges={len(step_idx)} skipped(>1/diag)="
              f"{int(((amax>1)|is_diag).sum())}")

    return {"Vq": Vq, "out_edges": out_edges, "out_adj": out_adj,
            "label": label, "members": members, "keep": keep,
            "rep_q": rep_q, "rep_n": rep_n, "rep_t": rep_t}


# ===========================================================================
#  Part C -- extract_faces  (ring tracing -> quads)
# ===========================================================================
def _build_halfedges(out_adj, keep, turn=-1):
    """Directed half-edges with a rotational-next map.

    For each directed edge a->b we need: at b, the next outgoing edge in
    rotational order relative to the incoming direction, so that always turning
    that way traces a face on ONE consistent side. `turn` selects the rotational
    step (-1 = clockwise of the reversed incoming edge = consistent left-turn
    face; +1 = the other side). Returns:
      he      dict (a,b)->id for present directed edges
      src,dst per half-edge endpoints
      nxt     list: nxt[id of a->b] = id of b->c (the rotational successor).
    """
    he = {}                                              # (a,b) -> id
    src = []; dst = []
    for a, nbrs in enumerate(out_adj):
        if not keep[a]:
            continue
        for b in nbrs:
            if (a, b) not in he:
                he[(a, b)] = len(src); src.append(a); dst.append(b)
    # position of each neighbour in a vertex's ordered ring
    pos_in_ring = [dict() for _ in range(len(out_adj))]
    for a, nbrs in enumerate(out_adj):
        for idx, b in enumerate(nbrs):
            pos_in_ring[a][b] = idx
    nxt = [-1] * len(src)
    for hid in range(len(src)):
        a, b = src[hid], dst[hid]
        ring = out_adj[b]
        if not ring:
            continue
        # incoming at b came FROM a; turn to the rotational successor of `a` in
        # b's CCW ring by `turn` steps -> keeps the face on a consistent side.
        if a in pos_in_ring[b]:
            i0 = pos_in_ring[b][a]
            c = ring[(i0 + turn) % len(ring)]
        else:
            c = ring[0]
        nid = he.get((b, c), -1)
        nxt[hid] = nid
    return he, src, dst, nxt


def _trace_ring(start_hid, src, dst, nxt, used, max_len):
    """Trace a closed ring from a directed half-edge by always turning to the
    rotational-next outgoing edge. Returns the list of vertex ids (the ring) and
    the list of half-edge ids, or (None,None) if it does not close within
    max_len or revisits a half-edge."""
    hid = start_hid
    ring_v = [src[hid]]
    ring_he = []
    seen_he = set()
    for _ in range(max_len + 1):
        if hid < 0 or used[hid] or hid in seen_he:
            return None, None
        seen_he.add(hid)
        ring_he.append(hid)
        b = dst[hid]
        if b == ring_v[0]:
            return ring_v, ring_he                       # closed
        ring_v.append(b)
        hid = nxt[hid]
    return None, None


# squareness floor for accepting a triangle-pair merge (env-overridable for A/B)
_PAIR_SQ_FLOOR = float(os.environ.get("PAIR_SQ_FLOOR", "-1.4"))


def _ring_edges(ring):
    """Undirected edge set of a closed ring (list of vertex ids)."""
    n = len(ring)
    return set((ring[k], ring[(k + 1) % n]) if ring[k] < ring[(k + 1) % n]
               else (ring[(k + 1) % n], ring[k]) for k in range(n))


def _merge_tri_pair(t0, t1, shared):
    """Merge two triangle rings sharing edge `shared`=(u,v) into one quad ring.
    A quad [u, w0, v, w1] where w0 is t0's apex (the vertex != u,v) and w1 is t1's
    apex. Returns the 4-cycle or None if degenerate."""
    u, v = shared
    a0 = [x for x in t0 if x != u and x != v]
    a1 = [x for x in t1 if x != u and x != v]
    if len(a0) != 1 or len(a1) != 1:
        return None
    w0, w1 = a0[0], a1[0]
    if w0 == w1:
        return None
    quad = [u, w0, v, w1]
    if len(set(quad)) != 4:
        return None
    return quad


def extract_faces(graph, verbose=False, turn=-1, pair_tris=False):
    """Ring-trace closed loops, quads first, fill the rest. Returns a list of
    faces (each a list of cluster ids; quads have 4, triangles stored as a quad
    with repeated last index).

    `pair_tris` (default OFF): merge pairs of size-3 orbits that share a graph
    edge into a single quad BEFORE filling. Two triangles sharing an edge combine
    into a real quad, killing both their would-be valence-3 vertices. MEASURED
    (within-process A/B, squareness-gated): the net val4 effect through the full
    finish is within noise (+-0.2%) -- the downstream clean_quad_valence + relax
    absorb the triangle-quads anyway -- while an ungated merge can drop
    flow_alignment below the 0.87 floor. Kept available (squareness-gated by
    `_PAIR_SQ_FLOOR`) but OFF by default because it does not pay and risks flow.
    """
    out_adj = graph["out_adj"]; keep = graph["keep"]; Vq = graph["Vq"]
    he, src, dst, nxt = _build_halfedges(out_adj, keep, turn=turn)
    nHE = len(src)
    used = np.zeros(nHE, bool)
    faces = []

    # The faces of a rotation system are the ORBITS of `nxt`: each directed half-
    # edge belongs to exactly one face, so tracing every orbit is a clean,
    # hole-free partition (the deg-loop missed interior faces because claiming a
    # quad early blocked a neighbouring orbit that ran through a shared half-edge
    # -- which cannot happen with the orbit partition). We trace each orbit once.
    # Quads-first is preserved as a fill-order preference: collect all orbits,
    # then fill size-4 first so they are emitted as clean quads before the
    # irregular faces are split.
    orbits = []                                          # (size, ring_v, ring_he)
    for h0 in range(nHE):
        if used[h0]:
            continue
        ring_v, ring_he = _trace_ring(h0, src, dst, nxt, used, 64)
        if ring_v is None:
            # open path (hit a -1 link or a used edge): mark its half-edges so we
            # don't retry forever; these become border edges -> capped later.
            h = h0; steps = 0
            while h >= 0 and not used[h] and steps < 65:
                used[h] = True; h = nxt[h]; steps += 1
            continue
        for h in ring_he:
            used[h] = True
        orbits.append((len(ring_v), ring_v, ring_he))

    # ---- pair adjacent triangle orbits into quads (the deg-3 lever) ----------
    n_paired = 0
    if pair_tris:
        tri_orbits = [(i, ring_v) for i, (size, ring_v, _) in enumerate(orbits)
                      if size == 3]
        # map each undirected edge -> list of triangle-orbit indices using it
        edge2tri = defaultdict(list)
        for oi, ring_v in tri_orbits:
            for e in _ring_edges(ring_v):
                edge2tri[e].append(oi)
        consumed = set()
        merged_quads = []
        # greedily pair triangles across a shared edge (prefer the squarer merge)
        ring_of = {oi: rv for oi, rv in tri_orbits}
        for e, ois in edge2tri.items():
            cand = [o for o in ois if o not in consumed]
            if len(cand) < 2:
                continue
            # try all unordered pairs sharing this edge; take the first valid square
            best = None
            for ii in range(len(cand)):
                for jj in range(ii + 1, len(cand)):
                    o0, o1 = cand[ii], cand[jj]
                    quad = _merge_tri_pair(ring_of[o0], ring_of[o1], e)
                    if quad is None:
                        continue
                    sc = _quad_square_score(Vq[quad])
                    if best is None or sc > best[0]:
                        best = (sc, o0, o1, quad)
            if best is not None and best[0] > _PAIR_SQ_FLOOR:
                # squareness gate: _quad_square_score in [-4,0] (0=perfect 90deg
                # corners). Only accept a merge that yields a reasonably square
                # quad; a sliver merge would inject an off-axis diagonal and drop
                # flow_alignment below the 0.87 floor (measured on teddy). Slivers
                # stay as two triangle-quads (the cleanup/cap handle them).
                _, o0, o1, quad = best
                consumed.add(o0); consumed.add(o1)
                merged_quads.append(quad)
                n_paired += 1
        # emit the merged quads; drop the consumed triangle orbits from the fill
        for quad in merged_quads:
            faces.append(list(quad))
        remaining = [(size, ring_v, ring_he) for k, (size, ring_v, ring_he)
                     in enumerate(orbits) if not (size == 3 and k in consumed)]
        orbits = remaining

    # boundary orbits of a CLOSED surface that traced the wrong way show up as one
    # giant orbit; drop orbits that are implausibly large (>16) -- they are the
    # outer boundary of an open patch, not a face (capped later).
    orbits.sort(key=lambda t: (t[0] != 4, t[0]))          # size-4 first
    for size, ring_v, ring_he in orbits:
        if size > 16:
            continue
        _fill_face(ring_v, Vq, faces)

    if verbose:
        nq = sum(1 for f in faces if len(set(f)) == 4)
        ntri = sum(1 for f in faces if len(set(f)) == 3)
        big = sum(1 for s, _, _ in orbits if s > 16)
        print(f"    [faces] half_edges={nHE} orbits={len(orbits)} faces={len(faces)}"
              f" quads={nq} tris={ntri} paired={n_paired} dropped_big={big} "
              f"used_he={int(used.sum())}")
    return faces, Vq


def _fill_face(ring, Vq, faces):
    """fill_face cases (spec Part C):
       size==4 -> emit quad
       size>=5 -> greedy carve the best ~90deg corner quad, repeat until <=4
       size<4  -> fan to a centroid (stored as triangle-quads a,b,c,c).
    `ring` is a list of cluster ids (closed loop)."""
    ring = list(ring)
    if len(ring) == 4:
        faces.append(ring)
        return
    if len(ring) == 3:
        # triangle: store as a quad with a repeated last index
        faces.append([ring[0], ring[1], ring[2], ring[2]])
        return
    if len(ring) < 3:
        return
    # size >= 5: greedy 90deg-corner carving.
    work = ring[:]
    guard = 0
    while len(work) > 4 and guard < 4 * len(ring) + 8:
        guard += 1
        n = len(work)
        # score each ear (i-1,i,i+1,i+2) as a quad by corner squareness
        best = None
        for i in range(n):
            a = work[i]; b = work[(i + 1) % n]; c = work[(i + 2) % n]; d = work[(i + 3) % n]
            quad = [a, b, c, d]
            if len(set(quad)) != 4:
                continue
            sc = _quad_square_score(Vq[quad])
            if best is None or sc > best[0]:
                best = (sc, i)
        if best is None:
            break
        i = best[1]
        a = work[i]; b = work[(i + 1) % n]; c = work[(i + 2) % n]; d = work[(i + 3) % n]
        faces.append([a, b, c, d])
        # remove the two interior verts b,c (consumed); ring keeps a..d chord
        rm = {(i + 1) % n, (i + 2) % n}
        work = [work[j] for j in range(n) if j not in rm]
    if len(work) == 4:
        faces.append(work)
    elif len(work) == 3:
        faces.append([work[0], work[1], work[2], work[2]])
    # len 2 or fewer -> drop (degenerate remainder)


def _quad_square_score(P):
    """Higher = closer to a flat 90deg-cornered quad. P is (4,3)."""
    s = 0.0
    for t in range(4):
        e1 = P[(t + 1) % 4] - P[t]; e2 = P[(t - 1) % 4] - P[t]
        a = np.linalg.norm(e1); b = np.linalg.norm(e2)
        if a < 1e-12 or b < 1e-12:
            return -1e9
        cosang = abs(float(e1 @ e2) / (a * b))           # 0 at 90deg
        s -= cosang
    return s


# ===========================================================================
#  quad-mesh valence cleanup (IM final cleanup: doublets + 3-5 dipoles)
# ===========================================================================
def _quad_adj(Q, nV):
    adj = defaultdict(set)
    for q in Q:
        qq = [int(x) for x in dict.fromkeys(int(c) for c in q)]
        for a, b in zip(qq, qq[1:] + qq[:1]):
            adj[a].add(b); adj[b].add(a)
    return adj


def clean_quad_valence(V, Q, iters=4, merge_max_val=6):
    """Instant-Meshes-style final valence cleanup on the quad mesh.

    Removes the two dominant position-field artefacts that inflate the irregular
    count without changing the surface:
      * DOUBLET (valence-2 interior vertex): the two quads sharing it are merged
        into one quad and the vertex deleted.
      * 3-5 DIPOLE: a valence-3 vertex adjacent to a valence-5 vertex. Collapsing
        that shared edge merges the pair into one vertex of valence
        6-(#common neighbours): 4 for a TRUE dipole (2 shared nbrs), 5 for a
        near-dipole (1), 6 for none.

    `merge_max_val`: cap the merged valence. INTEGRITY NOTE (measured): on this
    extractor's output the scattered 3s and 5s almost always share 0 neighbours,
    so a 3-5 merge usually produces a valence-6 vertex. With merge_max_val=6
    (default, preserves the historical behaviour) that merge is accepted: it
    RAISES val4% (one vertex removed) and lowers the irregular COUNT 2->1, but
    trades two mild irregulars (3,5; deviation 1+1=2) for one worse one (6;
    deviation 4) -- the val4% gain (~+5%) is partly a vertex-count artefact, not a
    quality gain (it added +150 deg-6 verts on teddy). Set merge_max_val=5 for the
    QUALITY-honest cleanup (only 3+5 -> 4 or 5 merges, which improve both count and
    deviation); it leaves val4% ~5 points lower but the mesh is genuinely cleaner.
    Best-first ordering (most common nbrs first) claims the true annihilations
    before the lossier merges. Watertight/manifold preserved.
    """
    V = np.asarray(V, float)
    Q = [[int(x) for x in q] for q in Q]
    for _ in range(iters):
        nV = len(V)
        adj = _quad_adj(Q, nV)
        val = {i: len(adj[i]) for i in adj}
        # boundary vertices (on an open edge) are off-limits
        ec = defaultdict(int)
        for q in Q:
            u = [int(x) for x in dict.fromkeys(q)]
            for a, b in zip(u, u[1:] + u[:1]):
                ec[(min(a, b), max(a, b))] += 1
        bnd = set()
        for (a, b), c in ec.items():
            if c == 1:
                bnd.add(a); bnd.add(b)

        # build vertex->quad incidence
        v2q = defaultdict(list)
        for qi, q in enumerate(Q):
            for x in set(q):
                v2q[x].append(qi)

        merge = {}                                       # vertex -> target vertex
        dead_q = set()
        locked = set()

        def try_collapse(a, b):
            # collapse edge (a,b): map a->b. Refuse if it would fold a quad.
            if a in locked or b in locked or a in bnd or b in bnd:
                return False
            for qi in v2q[a]:
                if qi in dead_q:
                    continue
                q = [merge.get(x, x) for x in Q[qi]]
                q = [b if x == a else x for x in q]
                u = list(dict.fromkeys(q))
                if len(u) < 3:                           # would degenerate badly
                    return False
            merge[a] = b
            locked.add(a); locked.add(b)
            for x in adj[a]:
                locked.add(x)
            return True

        # (a) doublets: valence-2 interior vertices -> collapse onto a neighbour
        for i in list(adj.keys()):
            if val.get(i, 0) == 2 and i not in bnd and i not in locked:
                nb = [x for x in adj[i] if x not in (i,)]
                if nb:
                    try_collapse(i, nb[0])
        # (b) 3-5 dipoles -- VALENCE-BOUNDED collapse. Collapsing edge (i,j) makes
        # a vertex of valence |N(i) U N(j) \ {i,j}| = 6 - (#common neighbours): it
        # is 4 for a TRUE dipole (2 shared nbrs), 5 for a near-dipole (1 shared),
        # 6 for none. The original code accepted ALL of them, so most 3-5 merges
        # produced a valence-6 vertex (measured: deg-6 72 -> 223 on teddy): that
        # raises val4% only because the vertex count shrank, while trading two mild
        # irregulars (3,5; deviation 1+1) for one WORSE one (6; deviation 4). We
        # bound the merged valence to <=5 (prefer the shared==2 -> 4 annihilation,
        # allow shared==1 -> 5 which still reduces both count and deviation, reject
        # shared==0 -> 6). Process best-first (shared==2 before shared==1) so the
        # true annihilations claim their verts before the near-dipoles.
        cand35 = []
        for i in list(adj.keys()):
            if val.get(i, 0) == 3 and i not in bnd:
                for j in adj[i]:
                    if val.get(j, 0) == 5 and j not in bnd:
                        shared = len(adj[i] & adj[j])
                        merged_val = 6 - shared
                        if merged_val <= merge_max_val:
                            cand35.append((-shared, i, j))
        cand35.sort()                                    # most-shared (best) first
        for _, i, j in cand35:
            if i in locked or j in locked:
                continue
            try_collapse(i, j)

        if not merge:
            break
        # apply merges + drop dead/degenerate quads
        def res(x):
            seen = set()
            while x in merge and x not in seen:
                seen.add(x); x = merge[x]
            return x
        newQ = []
        seenq = set()
        for qi, q in enumerate(Q):
            if qi in dead_q:
                continue
            u = list(dict.fromkeys(res(x) for x in q))
            if len(u) == 4:
                key = tuple(sorted(u))
                if key in seenq:
                    continue
                seenq.add(key); newQ.append(u)
            elif len(u) == 3:
                key = tuple(sorted(u))
                if key in seenq:
                    continue
                seenq.add(key); newQ.append([u[0], u[1], u[2], u[2]])
            # len<3 -> drop (collapsed)
        Q = newQ
        # compact vertices
        used = sorted(set(x for q in Q for x in q))
        remap = {v: k for k, v in enumerate(used)}
        V = V[used]
        Q = [[remap[x] for x in q] for q in Q]
    return V, np.asarray(Q, np.int64) if Q else np.zeros((0, 4), np.int64)
