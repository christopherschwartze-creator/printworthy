"""Blossom tri-to-quad meshing for junction patches (Remacle et al. 2012).

Replaces the degenerate valence-k fan at multi-way junction cores with a clean
quad-dominant fill, via minimum-cost perfect matching on the triangle dual graph
(networkx = true Edmonds blossom, permissive). Pipeline:
  coarsen patch (to ~2h) -> blossom match (eta angle-quality) -> Catmull-Clark
  uniform split (all-quad, T-junction-free, lands at ~h) -> relax + project.
Boundary vertices are never moved by the matching, so the patch welds.

Permissive: numpy, scipy, trimesh, networkx. NO GPL, no `triangle` (non-permissive).
"""
import numpy as np
import networkx as nx
from collections import defaultdict
from . import _grid_place as g
from ._mesh_util import decimate as _decimate


def quad_eta(P):
    """Angle quality of a quad (Remacle eq.3): 1 = all 90deg, 0 = degenerate."""
    worst = 0.0
    for t in range(4):
        e1 = P[(t + 1) % 4] - P[t]; e2 = P[(t - 1) % 4] - P[t]
        n1 = np.linalg.norm(e1); n2 = np.linalg.norm(e2)
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        c = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
        worst = max(worst, abs(np.degrees(np.arccos(c)) - 90.0))
    return max(0.0, 1.0 - worst / 90.0)


def _quad_field_align(P, di, ni):
    """Mean |cos| of a quad's 4 edges to the nearest cross direction {di, ni x di}.
    1 = the quad's edges follow the field; ~0.7 = unaligned."""
    c = np.cross(ni, di); al, cnt = 0.0, 0
    for k in range(4):
        e = P[(k + 1) % 4] - P[k]; ne = np.linalg.norm(e)
        if ne < 1e-9:
            continue
        e = e / ne; al += max(abs(e @ di), abs(e @ c)); cnt += 1
    return al / cnt if cnt else 0.7


def blossom_match(V, F, field=None, field_w=0.6):
    """Min-cost perfect matching of triangles into quads. Returns (quads, tris)
    where quads are [a,o0,b,o1] CCW and tris are leftover [a,b,c].
    field=(d, N): bias the matching toward quads whose edges follow the cross-field
    (field_w weights alignment vs squareness) -> field-following flow."""
    emap = defaultdict(list)
    for fi, tri in enumerate(F):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = (int(a), int(b)) if a < b else (int(b), int(a))
            emap[e].append(fi)
    G = nx.Graph(); G.add_nodes_from(range(len(F)))
    for e, fs in emap.items():
        if len(fs) != 2:
            continue
        f0, f1 = fs
        sh = set(int(x) for x in F[f0]) & set(int(x) for x in F[f1])
        if len(sh) != 2:
            continue
        o0 = [int(x) for x in F[f0] if int(x) not in sh][0]
        o1 = [int(x) for x in F[f1] if int(x) not in sh][0]
        a, b = e
        P = V[[a, o0, b, o1]]
        eta = quad_eta(P)
        w = (1.0 - eta)
        if field is not None:
            d, N = field
            fa = _quad_field_align(P, d[a], N[a])
            w = (1.0 - field_w) * w + field_w * (1.0 - fa)
        G.add_edge(f0, f1, weight=w + 1e-6)   # min-cost => squarest + field-aligned
    if G.number_of_edges() == 0:
        return [], [list(map(int, F[fi])) for fi in range(len(F))]
    M = nx.min_weight_matching(G)
    quads, used = [], set()
    for f0, f1 in M:
        sh = set(int(x) for x in F[f0]) & set(int(x) for x in F[f1])
        o0 = [int(x) for x in F[f0] if int(x) not in sh][0]
        o1 = [int(x) for x in F[f1] if int(x) not in sh][0]
        a, b = sorted(sh)
        quads.append([int(a), int(o0), int(b), int(o1)]); used |= {f0, f1}
    tris = [list(map(int, F[fi])) for fi in range(len(F)) if fi not in used]
    return quads, tris


def catmull_clark_split(V, faces):
    """One uniform Catmull-Clark-style split: every quad->4 quads, every tri->3
    quads, via shared edge-midpoints + face centers. All-quad, T-junction-free.
    `faces` = list of index lists (len 3 or 4). Returns (Vnew, quads)."""
    V = [np.asarray(p, float) for p in V]
    edgemid = {}
    def emid(a, b):
        k = (a, b) if a < b else (b, a)
        if k not in edgemid:
            edgemid[k] = len(V); V.append(0.5 * (V[a] + V[b]))
        return edgemid[k]
    out = []
    for f in faces:
        n = len(f)
        o = len(V); V.append(np.mean([V[i] for i in f], axis=0))
        ms = [emid(f[i], f[(i + 1) % n]) for i in range(n)]
        for i in range(n):
            out.append([f[i], ms[i], o, ms[(i - 1) % n]])
    return np.array(V), np.array(out, np.int64)


def cleanup_quads(V, Q, min_edge_frac=0.18):
    """Topological cleanup: collapse short edges (removes valence-2 doublets,
    degenerate/inverted quads, and many scattered valence-3 irregulars). Merging
    vertices preserves watertightness; quads that collapse below 4 unique verts are
    dropped (their tiny holes get capped downstream). Returns (V, Q)."""
    V = np.asarray(V, float)
    Q = [[int(x) for x in q] for q in Q]
    el = [np.linalg.norm(V[a] - V[b]) for q in Q for a, b in zip(q, q[1:] + q[:1]) if a != b]
    if not el:
        return V, np.asarray(Q, np.int64)
    thr = min_edge_frac * float(np.median(el))
    parent = list(range(len(V)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    for _ in range(3):                                # a few passes
        for q in Q:
            for a, b in zip(q, q[1:] + q[:1]):
                if a != b and np.linalg.norm(V[find(a)] - V[find(b)]) < thr:
                    union(a, b)
    newQ = []
    for q in Q:
        qq = [find(x) for x in q]
        rq = [x for i, x in enumerate(qq) if x != qq[(i - 1) % 4]]   # drop consecutive dups
        if len(rq) == 4:
            newQ.append(rq)
    if not newQ:
        return V, np.asarray(Q, np.int64)
    # average merged vertex positions; reindex
    grp = defaultdict(list)
    for i in range(len(V)):
        grp[find(i)].append(i)
    used = sorted(grp.keys()); rmap = {g: i for i, g in enumerate(used)}
    Vn = np.array([V[grp[g]].mean(0) for g in used])
    Qn = np.array([[rmap[x] for x in q] for q in newQ], np.int64)
    return Vn, Qn


def blossom_quad_patch(patch, h, coarsen=True, cc_split=True, relax_iters=6,
                       field_bias=False):
    """Quad-mesh a surface patch via Blossom. Returns (Vq, Q, info).
    field_bias=True biases the tri-pairing toward the cross-field (flow-following)."""
    V = np.asarray(patch.vertices, float); F = np.asarray(patch.faces, np.int64)
    area = float(patch.area)
    n0 = len(F)
    if coarsen:
        # coarse edge length ~2h (CC split halves it back to ~h). A triangle of
        # edge L covers ~0.43 L^2, so n_tris ~ area/(0.43*(2h)^2). HARD-CAP at 700
        # so the O(N^3) blossom matching stays fast (a few seconds).
        L = (2.0 if cc_split else 1.0) * h
        target = int(np.clip(area / (0.43 * L * L), 40, 700))
        if target < len(F):
            pc = _decimate(patch, target)      # shared dual-API decimation guard
            if pc is not patch and len(pc.faces) >= 4:
                pc.fix_normals(); patch = pc
                V = np.asarray(pc.vertices, float); F = np.asarray(pc.faces, np.int64)
    fld = None
    if field_bias:
        try:
            from . import _field_quad as _fq
            _d = _fq.smooth_field(patch); _N, _, _ = _fq.tangent_frames(patch)
            fld = (_d, _N)
        except Exception:
            fld = None
    quads, tris = blossom_match(V, F, field=fld)
    faces = quads + tris
    if cc_split:
        Vn, Q = catmull_clark_split(list(V), faces)
    else:
        # pad tris to degenerate quads
        Vn = V; Q = np.array([q if len(q) == 4 else [q[0], q[1], q[2], q[2]] for q in faces], np.int64)
    Vn = g.project_to_surface(patch, Vn)
    if relax_iters:
        Vn = g.relax_quads(patch, Vn, Q, iters=relax_iters)
    Vn, Q = cleanup_quads(Vn, Q)                      # collapse short edges / doublets
    Vn = g.project_to_surface(patch, Vn)
    if relax_iters:
        Vn = g.relax_quads(patch, Vn, Q, iters=relax_iters // 2)
    info = {"tris_in": n0, "coarse_tris": len(F), "quads_matched": len(quads),
            "leftover_tris": len(tris), "n_quads": len(Q)}
    return Vn, Q, info
