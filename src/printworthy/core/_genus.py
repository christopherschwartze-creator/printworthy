"""Genus repair for field-aligned quad remeshes (GPL-free: numpy/scipy/networkx/trimesh).

WHY
---
The mcf2 integer-grid extraction (`_seamless.remesh_mcf2`) can stitch together two
lattice cells that landed on the same integer position but are FAR apart on the
real surface. The result is a closed, manifold, "watertight" quad mesh whose
QUAD COMPLEX carries spurious genus -- small phantom tunnels. trimesh's
`is_watertight` and the eval's `manifold_stats` both MISS this (the surface is a
perfectly good 2-manifold, just of the wrong topological type). The print lens
exposes it: teddy genus 0 -> 2, hand 0 -> 2, fourleg 5 -> 11, octopus 1 -> 2
(measured 2026-06-23). Each phantom handle is an unprintable tunnel.

WHAT
----
`genus_repair(V, Q, ref_mesh) -> (V, Q)` detects the EXCESS handles (relative to
`ref_mesh`'s genus) and removes them, preserving watertightness, the quad
structure (mostly), and surface fidelity (target two-sided < 0.01 of bbox-diag).

HOW (what is actually implemented + measured)
----------------------------------------------
(a) HANDLE-LOOP detection (CORRECT tree-cotree, Erickson-Whittlesey): edges of the
    quad mesh partition into a primal spanning tree T, a dual spanning tree C*, and
    exactly 2g GENERATOR edges. Each generator closes a non-contractible loop. The
    SPURIOUS grid-stitch handles are the geometrically SHORT loops (3-5 quad edges,
    3D length ~0.07-0.2 of bbox-diag); the REAL body loops (a leg, an octopus arm)
    are LONG (20-140 edges). Sorting loops shortest-first isolates the spurious
    ones, so the real topology is never touched (octopus keeps its 1 real handle).

(b) HANDLE REMOVAL (quad-preserving cut + surface-domed cap): around the shortest
    generator loop we grow the SMALLEST ball region whose removal+capping drops the
    genus by exactly one (never below ref). The resulting boundary loop is capped
    with a fan that is then refined by INTERIOR 1->3 splits and CONTINUOUSLY
    reprojected onto ref_mesh, so the cap hugs the surface (protects the surf->out
    coverage fidelity). Quad structure is preserved everywhere except the small
    cap. We iterate until quad-genus == ref genus.

    The "messy grid-maps" lever (a grid-stitch tunnel = the position-field folding
    two sheets together) is exactly the short-non-contractible-loop signature in
    (a); it is folded into the detector, not a separate pass.

(c) VOXEL FALLBACK -- TRIED AND REJECTED (documented honestly): voxelizing a thin
    shell creates dozens of voxel-scale handles (global force_solid took teddy
    genus 0 -> 707), so a voxel rebuild makes genus WORSE, not better. It is NOT
    used. The ring-growth in (b) is the universal robust path; for the small
    fraction of "large fold" handles it still succeeds, at a fidelity cost (below).

MEASURED (PSB, target_quads=1500, 2026-06-23). HARD constraints all hold: quad
genus == ref, quad mesh stays a closed 2-manifold (every edge 2-incident).
  teddy   g2->0  fid 0.0022->0.0023  quad-frac 0.98  (tiny tunnels: ideal)
  fourleg g11->5 fid 0.0051->0.0083  quad-frac 0.94  (drops 6 spurious, keeps 5 real)
  hand    g2->0  fid 0.0031->0.018   quad-frac 0.68  (LARGE folds: see caveat)
  octopus g2->1  fid 0.0047->0.015   quad-frac 0.77  (keeps the 1 real arm handle)
CAVEAT: tunnel-type handles (teddy/fourleg) repair with fidelity < 0.01 and high
quad retention. "Large fold" handles (hand/octopus) -- where the grid stitched two
genuinely distant surface regions together -- require removing a finger-sized
region, so the domed cap cannot reproduce the removed protrusion: fidelity lands
~0.015-0.018 (over the 0.01 target) and quad retention drops. The genus + closed-
manifold guarantees still hold. quad_genus()/quad_euler() are unit-validated
(cube chi=2 g0, torus chi=0 g1) and are the correct measure for the quad complex;
the print gate's quads_to_trimesh euler is confounded by PRE-EXISTING quad-diagonal
artifacts in the mcf2 output (this module REDUCES them, e.g. hand 5->1).

Never raises; on any failure returns the input unchanged (caller still gates).
"""
import numpy as np

# ----------------------------------------------------------------------------- #
#  topology helpers (operate on the quad complex directly)                      #
# ----------------------------------------------------------------------------- #


def _poly(q):
    """Quad index list with consecutive duplicates removed (degenerate quads that
    the extractor stored as triangles collapse to a 3-gon)."""
    qq = [int(x) for x in q]
    u = [qq[i] for i in range(len(qq)) if qq[i] != qq[(i - 1) % len(qq)]]
    return u


def quad_euler(V, Q):
    """Euler characteristic chi = Vu - E + F of the quad complex (Vu = used
    vertices, E = unique undirected edges, F = #faces). Returns (chi, Vu, E, F)."""
    edges = set()
    verts = set()
    F = 0
    for q in Q:
        u = _poly(q)
        if len(u) < 3:
            continue
        F += 1
        for a, b in zip(u, u[1:] + u[:1]):
            if a != b:
                edges.add((a, b) if a < b else (b, a))
                verts.add(a)
                verts.add(b)
    return len(verts) - len(edges) + F, len(verts), len(edges), F


def quad_genus(V, Q):
    """Genus of the quad mesh, assuming a single closed orientable component:
    g = (2 - n_components - chi) // 2 ... but we report the per-surface number
    g = (2 * n_comp - chi) // 2 collapsed to the usual (2 - chi)//2 for one comp.
    We compute components on the edge graph and aggregate."""
    chi, _, _, _ = quad_euler(V, Q)
    ncomp = _n_components(V, Q)
    # chi = sum over components (2 - 2 g_i); total genus G with C comps:
    #   chi = 2 C - 2 G  =>  G = (2 C - chi) / 2
    return int(round((2 * ncomp - chi) / 2.0)), ncomp


def _edge_graph(Q):
    """networkx graph of the quad mesh 1-skeleton."""
    import networkx as nx

    G = nx.Graph()
    for q in Q:
        u = _poly(q)
        if len(u) < 3:
            continue
        for a, b in zip(u, u[1:] + u[:1]):
            if a != b:
                G.add_edge(a, b)
    return G


def _n_components(V, Q):
    import networkx as nx

    G = _edge_graph(Q)
    if G.number_of_nodes() == 0:
        return 0
    return nx.number_connected_components(G)


def _edge_to_faces(Q):
    """undirected edge -> list of face indices (manifold => len<=2)."""
    e2f = {}
    for fi, q in enumerate(Q):
        u = _poly(q)
        if len(u) < 3:
            continue
        for a, b in zip(u, u[1:] + u[:1]):
            if a == b:
                continue
            e = (a, b) if a < b else (b, a)
            e2f.setdefault(e, []).append(fi)
    return e2f


# ----------------------------------------------------------------------------- #
#  (a)+(b) shortest spurious-handle loop detection (tree-cotree homology)        #
# ----------------------------------------------------------------------------- #


def _dual_graph(Q, e2f):
    """Face-adjacency (dual) graph: nodes = faces, edges = shared quad edges,
    tagged with the primal edge so we can recover it."""
    import networkx as nx

    D = nx.Graph()
    D.add_nodes_from(range(len(Q)))
    for e, fs in e2f.items():
        if len(fs) == 2:
            D.add_edge(fs[0], fs[1], primal=e)
    return D


def _handle_loops(V, Q, ref_genus, verbose=False):
    """Find the homology-generator loops, sorted most-tunnel-like first.

    CORRECT tree-cotree (Erickson-Whittlesey): edges partition into a primal
    spanning tree T, a dual spanning tree C* (built from dual edges whose primal
    edge is NOT in T), and exactly 2g GENERATOR edges in neither. Each generator
    edge (u,v) closes a non-contractible loop through T's u..v path. We score each
    loop by the grid-stitch-tunnel signature (short in 3D, sides 3D-near but
    graph-far) and return them sorted (most tunnel-like first)."""
    import networkx as nx

    V = np.asarray(V, float)
    e2f = _edge_to_faces(Q)
    G = _edge_graph(Q)
    if G.number_of_nodes() == 0:
        return []
    D = _dual_graph(Q, e2f)

    # 1) primal spanning tree T (union-find over SHORT 3D edges so loops close
    #    through geometrically-short paths => small tunnels surface first)
    parent = {n: n for n in G.nodes()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[rx] = ry
        return True

    edges = sorted(G.edges(), key=lambda e: np.linalg.norm(V[e[0]] - V[e[1]]))
    T = set()
    P = nx.Graph()
    P.add_nodes_from(G.nodes())
    for a, b in edges:
        if union(a, b):
            ee = (a, b) if a < b else (b, a)
            T.add(ee)
            P.add_edge(a, b, w=float(np.linalg.norm(V[a] - V[b])))

    # 2) dual spanning tree C* over dual edges whose primal edge is NOT in T
    p2 = {n: n for n in D.nodes()}

    def find2(x):
        while p2[x] != x:
            p2[x] = p2[p2[x]]
            x = p2[x]
        return x

    def union2(x, y):
        rx, ry = find2(x), find2(y)
        if rx == ry:
            return False
        p2[rx] = ry
        return True

    Cstar = set()
    for a, b, data in D.edges(data=True):
        pe = data["primal"]
        if pe in T:
            continue
        if union2(a, b):
            Cstar.add(pe)

    # 3) generators = edges in neither T nor C*
    gens = [e for e in e2f if (e not in T and e not in Cstar)]
    if not gens:
        return []

    diag = float(np.linalg.norm(V.max(0) - V.min(0))) + 1e-12
    scored = []
    for (a, b) in gens:
        try:
            path = nx.shortest_path(P, a, b, weight="w")
        except Exception:
            continue
        loop = path + [a]  # close via the generator edge a-b
        sc = _tunnel_score(V, loop, G, diag)
        if sc is not None:
            scored.append((sc, loop))
    scored.sort(key=lambda t: t[0]["score"])
    if verbose:
        for sc, lp in scored:
            print("   gen loop len=%d 3d=%.4f tunnelfrac=%.2f near=%.4f score=%.4f"
                  % (len(sc["loop_v"]), sc["len3d"], sc["tunnel_frac"],
                     sc["near3d"], sc["score"]))
    return scored


def _tunnel_score(V, loop, G, diag):
    """Score a non-contractible loop for being a *spurious grid-stitch tunnel*
    rather than a real body loop (leg, arm, octopus arm).

    A grid-stitch tunnel: the loop is SHORT in 3D, and the surface on the two
    sides of the loop is 3D-CLOSE but GRAPH-FAR. We approximate by: take the loop
    vertices, find for each a 3D-near vertex that is GRAPH-far (>= K hops) -- if a
    large fraction of the loop has such a near-but-graph-far partner, it's a
    pinched tunnel. score = len3d/diag * (1 + (1 - tunnel_frac)); lower = more
    tunnel-like.

    GEOMETRY DOMINATES: across the corpus the SPURIOUS handles are the
    geometrically SHORT loops (3-5 quad edges, 3D length ~0.07-0.2 of bbox-diag)
    and the REAL body loops (a leg, an octopus arm) are LONG (20-140 edges). So
    the primary sort key is len3d/diag; the tunnel signature (sides 3D-near but
    graph-far) is a cheap tie-breaker layered on top, computed only at the loop
    vertices (O(loop) KD queries, no per-loop BFS -> fast)."""
    from scipy.spatial import cKDTree

    loop_v = loop[:-1] if loop[0] == loop[-1] else loop
    if len(loop_v) < 3:
        return None
    len3d = float(np.sum(np.linalg.norm(np.diff(V[loop], axis=0), axis=1)))
    near_thr = 0.04 * diag
    # cheap tunnel signal: a loop vertex has a 3D-near neighbour that is NOT one
    # of its loop neighbours nor a 2-ring graph neighbour along the loop. We
    # approximate "graph-far" by "not in the loop vertex set + their direct
    # mesh neighbours" -- enough to tell a pinched neck from an open patch.
    tree = cKDTree(V)
    loopset = set(loop_v)
    ring1 = set(loopset)
    for lv in loop_v:
        ring1.update(G.neighbors(lv))
    tunnel_hits = 0
    near3d = np.inf
    for i, lv in enumerate(loop_v):
        dd, jj = tree.query(V[lv], k=min(16, len(V)))
        for d, j in zip(np.atleast_1d(dd), np.atleast_1d(jj)):
            j = int(j)
            if j == lv or d <= 1e-12:
                continue
            if d > near_thr:
                break
            if j not in ring1:  # 3D-near, not a graph neighbour of the loop
                tunnel_hits += 1
                near3d = min(near3d, float(d))
                break
    tunnel_frac = tunnel_hits / max(1, len(loop_v))
    if not np.isfinite(near3d):
        near3d = near_thr
    # short loop => low score (repaired first); tunnel signature gives a mild
    # discount so a short-AND-pinched loop sorts ahead of a short-but-open one.
    score = (len3d / diag) * (1.0 - 0.25 * tunnel_frac)
    return dict(score=score, len3d=len3d, near3d=near3d,
                tunnel_frac=tunnel_frac, graph_far=tunnel_hits,
                loop_v=loop_v)


# ----------------------------------------------------------------------------- #
#  region + boundary-loop helpers                                                #
# ----------------------------------------------------------------------------- #


def _region_faces(Q, seed_verts, ring, G=None):
    """faces within `ring` quad-edge hops of any seed vertex."""
    import networkx as nx

    if G is None:
        G = _edge_graph(Q)
    near = set(seed_verts)
    for s in seed_verts:
        try:
            near.update(nx.single_source_shortest_path_length(G, s, cutoff=ring).keys())
        except Exception:
            pass
    region = set(fi for fi, q in enumerate(Q) if any(v in near for v in _poly(q)))
    return region, near


def _boundary_loops(face_idx, Q):
    """Ordered boundary loops of a face subset (edges used by exactly one face)."""
    cnt = {}
    for fi in face_idx:
        u = _poly(Q[fi])
        for a, b in zip(u, u[1:] + u[:1]):
            if a == b:
                continue
            e = (a, b) if a < b else (b, a)
            cnt[e] = cnt.get(e, 0) + 1
    bnd = [e for e, c in cnt.items() if c == 1]
    adj = {}
    for a, b in bnd:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    loops = []
    seen = set()
    for start in list(adj):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nxts = [x for x in adj.get(cur, []) if x != prev and (x not in seen or x == start)]
            if not nxts:
                break
            nx_ = nxts[0]
            if nx_ == start:
                break
            loop.append(nx_)
            seen.add(nx_)
            prev, cur = cur, nx_
        if len(loop) >= 3:
            loops.append(loop)
    return loops


# ----------------------------------------------------------------------------- #
#  (a)+(b) quad-preserving handle pinch: cut a band around the loop, cap the     #
#          resulting boundary loop(s) with a reprojected fan of quads/tris.      #
# ----------------------------------------------------------------------------- #


def _cap_domed(nV, nQ, new_idx, loop, subdiv):
    """Cap one boundary `loop` with a fan to its centroid, then refine the cap by
    INTERIOR 1->3 centroid splits so the cap can dome toward the surface. We add
    UNPROJECTED centroids here (their indices are recorded in `new_idx` so the
    caller can batch-project them CONTINUOUSLY onto ref_mesh -- a continuous
    projection maps distinct inputs to distinct outputs, avoiding the coincident-
    duplicate / non-manifold collapse that a discrete-sample snap caused).

    CRITICAL: never subdivide a boundary-loop edge (shared with the kept mesh ->
    a boundary midpoint = non-manifold T-junction); only strictly-interior
    centroids are added. subdiv=0 => plain flat fan (one apex)."""
    pts = np.array([nV[v] for v in loop])
    c = pts.mean(0)
    ci = len(nV)
    nV.append(list(c))
    new_idx.append(ci)
    tris = [[int(a), int(b), ci] for a, b in zip(loop, loop[1:] + loop[:1])]
    for _ in range(max(0, subdiv)):
        nt = []
        for a, b, d in tris:
            tc = (np.asarray(nV[a]) + np.asarray(nV[b]) + np.asarray(nV[d])) / 3.0
            tci = len(nV)
            nV.append(list(tc))
            new_idx.append(tci)
            nt += [[a, b, tci], [b, d, tci], [d, a, tci]]
        tris = nt
    for t in tris:
        nQ.append([t[0], t[1], t[2], t[2]])


def _project_to_ref(pts, ref_mesh):
    """Continuous closest-point of pts onto ref_mesh's surface (distinct inputs ->
    distinct outputs, unlike a discrete-sample snap). Falls back to a dense-sample
    KDTree if the proximity BVH is unavailable. Returns (N,3)."""
    pts = np.asarray(pts, float)
    if len(pts) == 0:
        return pts
    try:
        cp, _, _ = ref_mesh.nearest.on_surface(pts)
        return np.asarray(cp, float)
    except Exception:
        from scipy.spatial import cKDTree
        import trimesh
        samp, _ = trimesh.sample.sample_surface(ref_mesh, 200000)
        tree = cKDTree(np.asarray(samp))
        _, idx = tree.query(pts, k=1)
        return np.asarray(samp)[idx]


def _cut_cap_region(V, Q, region, ref_mesh, cap_subdiv=2):
    """Delete `region` faces and cap the resulting boundary loop(s) with a domed
    fan whose interior vertices are CONTINUOUSLY reprojected onto ref_mesh (so the
    cap hugs the surface => coverage fidelity held, and no coincident duplicates).
    Returns (V2, Q2) or None. Caller checks the genus.

    cap_subdiv scales with the loop size so small caps stay cheap and big caps
    (which would otherwise flat-span a chunk of removed surface) get domed."""
    if not region:
        return None
    keep = [fi for fi in range(len(Q)) if fi not in region]
    bloops = _boundary_loops(keep, Q)
    if not bloops:
        return None
    nV = [list(x) for x in V]
    nQ = [[int(x) for x in Q[fi]] for fi in keep]
    new_idx = []
    for loop in bloops:
        # subdivide more for big loops (they span more removed surface, so the cap
        # must hug the surface harder to hold the surf->out coverage fidelity)
        if len(loop) >= 6:
            sd = cap_subdiv
        elif len(loop) >= 4:
            sd = 1
        else:
            sd = 0
        _cap_domed(nV, nQ, new_idx, loop, sd)
    nV = np.array(nV, float)
    if new_idx:  # batch CONTINUOUS reprojection of all new cap verts at once
        ni = np.array(new_idx, int)
        nV[ni] = _project_to_ref(nV[ni], ref_mesh)
    return nV, nQ


def _remove_one_handle(V, Q, loop_v, ref_mesh, cur_g, ref_g, G=None,
                       max_ring=18, max_frac=0.45):
    """Remove EXACTLY ONE handle localized at `loop_v` by growing a ball region
    around the loop until cut+cap drops the genus by one (and no further -- never
    below ref_g). Returns (V2, Q2, new_genus, ring_used) or None.

    Approach (a)/(b): the spurious grid-stitch handle is a tube around the
    non-contractible loop; the smallest ball that ENCLOSES the tube, once removed
    and capped, drops genus by one. We search the smallest such ball so the fewest
    quads are sacrificed. Monotone ring growth + accept-first-drop = minimal patch.
    """
    V = np.asarray(V, float)
    if G is None:
        G = _edge_graph(Q)
    nfaces = len(Q)
    best = None
    for ring in range(1, max_ring + 1):
        region, _ = _region_faces(Q, loop_v, ring, G=G)
        if len(region) >= max_frac * nfaces:
            break
        res = _cut_cap_region(V, Q, region, ref_mesh)
        if res is None:
            continue
        Vn, Qn = _finalize(res[0], res[1], ref_mesh)
        gn, _ = quad_genus(Vn, Qn)
        if not _is_closed_manifold(Qn):
            continue
        # accept the SMALLEST region that drops genus by exactly one and does not
        # undershoot the reference (never remove a real handle below ref_g)
        if gn == cur_g - 1 and gn >= ref_g:
            return Vn, Qn, gn, ring, len(region)
        # if a region drops by >1 (merged tube), keep as a fallback but prefer a
        # clean -1; only use it if no -1 region is found by max_ring
        if best is None and ref_g <= gn < cur_g:
            best = (Vn, Qn, gn, ring, len(region))
    if best is not None:
        return best
    return None


# ----------------------------------------------------------------------------- #
#  finishing: weld + watertight guarantee on the quad mesh                       #
# ----------------------------------------------------------------------------- #


def _finalize(V, Q, ref_mesh, weld_eps=1e-7):
    """Weld EXACTLY-coincident vertices, drop unreferenced ones, re-index.

    The cap reprojection snaps new vertices onto the reference surface; some land
    exactly on an existing surface vertex -> a coincident duplicate. Those MUST be
    welded or the triangulated proxy (quads_to_trimesh, process=True) re-merges
    them and changes the genus the print gate reads (seen: 23 dup pairs flipped
    tri-euler 2->1). We weld ONLY exact coincidences (<=weld_eps via a rounded-
    coordinate hash) -- never the near-but-distinct dome points (a KDTree-radius
    reweld corrupted those into non-manifold). Then compact + drop degenerate
    quads."""
    V = np.asarray(V, float)
    # exact-coincidence weld via quantized-coordinate hashing
    if len(V):
        key = np.round(V / weld_eps).astype(np.int64)
        seen = {}
        vmap = np.empty(len(V), np.int64)
        for i in range(len(V)):
            k = (int(key[i, 0]), int(key[i, 1]), int(key[i, 2]))
            if k in seen:
                vmap[i] = seen[k]
            else:
                seen[k] = i
                vmap[i] = i
    else:
        vmap = np.arange(len(V))
    nQ = []
    for q in Q:
        qm = [int(vmap[int(x)]) for x in q]
        u = _poly(qm)
        if len(u) == 3:
            nQ.append([u[0], u[1], u[2], u[2]])
        elif len(u) == 4:
            nQ.append([u[0], u[1], u[2], u[3]])
        # <3 unique verts => degenerate, drop
    if not nQ:
        return V, np.zeros((0, 4), int)
    used = sorted({int(x) for q in nQ for x in q})
    remap = {old: i for i, old in enumerate(used)}
    nV2 = V[used]
    nQ2 = [[remap[int(x)] for x in q] for q in nQ]
    return nV2, np.asarray(nQ2, int)


# ----------------------------------------------------------------------------- #
#  PUBLIC ENTRY                                                                   #
# ----------------------------------------------------------------------------- #


def genus_repair(V, Q, ref_mesh, max_handles=12, verbose=False, return_stats=False):
    """Remove spurious handles from a quad mesh so its genus matches `ref_mesh`.

    Parameters
    ----------
    V : (N,3) float vertices
    Q : (M,4) int quads (degenerate quads = a repeated index, the project convention)
    ref_mesh : trimesh.Trimesh -- the input the remesh approximates (sets the
               target genus and the surface for fidelity reprojection)
    max_handles : safety cap on pinch iterations.

    Returns
    -------
    (V, Q) with genus == ref genus where achievable, watertight preserved, quad
    structure preserved outside the small repaired patches. Never raises;
    returns the input on failure (the caller still gates on genus).

    With return_stats=True also returns a dict of measured numbers.
    """
    stats = {"applied": [], "ok": False}
    try:
        V0 = np.asarray(V, float).copy()
        Q0 = np.asarray(Q, int).copy()
        ref_g = (2 - int(ref_mesh.euler_number)) // 2
        cur_g, ncomp = quad_genus(V0, Q0)
        stats["ref_genus"] = ref_g
        stats["start_genus"] = cur_g
        stats["start_quads"] = int(len(Q0))
        if cur_g <= ref_g:
            # nothing spurious to remove (or genus already too LOW -- we never ADD
            # handles; a too-low genus is a different defect out of scope here)
            stats["ok"] = (cur_g == ref_g)
            stats["end_genus"] = cur_g
            stats["quad_frac_kept"] = 1.0
            return (V0, Q0, stats) if return_stats else (V0, Q0)

        Vc, Qc = V0, Q0
        n_quad_faces_start = sum(1 for q in Q0 if len(set(_poly(q))) == 4)
        for it in range(max_handles):
            cur_g, _ = quad_genus(Vc, Qc)
            if cur_g <= ref_g:
                break
            scored = _handle_loops(Vc, Qc, ref_g, verbose=verbose)
            if not scored:
                break
            G = _edge_graph(Qc)  # reused for region growth this iteration
            progressed = False
            # The SHORTEST generator loops are the spurious handles (real body
            # loops -- a leg/arm -- are long). Try them shortest-first; the first
            # whose minimal enclosing ball, cut+capped, drops the genus by one
            # (without undershooting ref_g) wins. Never touch the long real loops.
            n_try = min(len(scored), max(6, (cur_g - ref_g) + 5))
            for sc, loop in scored[:n_try]:
                out = _remove_one_handle(Vc, Qc, sc["loop_v"], ref_mesh,
                                         cur_g, ref_g, G=G)
                if out is None:
                    continue
                Vn, Qn, gn, ring, rsize = out
                Vc, Qc = Vn, Qn
                stats["applied"].append(
                    {"iter": it, "loop_len": len(sc["loop_v"]), "ring": ring,
                     "region_faces": rsize, "len3d": round(sc["len3d"], 4),
                     "genus": int(gn)})
                progressed = True
                break
            if not progressed:
                break

        end_g, _ = quad_genus(Vc, Qc)
        n_quad_faces_end = sum(1 for q in Qc if len(set(_poly(q))) == 4)
        stats["end_genus"] = end_g
        stats["end_quads"] = int(len(Qc))
        stats["ok"] = (end_g == ref_g)
        stats["quad_frac_kept"] = round(n_quad_faces_end / max(1, n_quad_faces_start), 3)
        stats["manifold"] = _is_closed_manifold(Qc)
        if not stats["ok"] and end_g >= stats["start_genus"]:
            # made no progress; return input untouched
            return (V0, Q0, stats) if return_stats else (V0, Q0)
        return (Vc, Qc, stats) if return_stats else (Vc, Qc)
    except Exception as e:  # never raise
        stats["error"] = f"{type(e).__name__}: {e}"
        if verbose:
            import traceback
            traceback.print_exc()
        return (np.asarray(V, float), np.asarray(Q, int), stats) if return_stats \
            else (np.asarray(V, float), np.asarray(Q, int))


def _is_closed_manifold(Q):
    """True if every quad edge has exactly 2 incident faces (closed 2-manifold)."""
    e2f = _edge_to_faces(Q)
    if not e2f:
        return False
    return all(len(fs) == 2 for fs in e2f.values())
