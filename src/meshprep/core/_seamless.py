"""SEAMLESS (rotation-aware) integer-grid offsets for field-aligned quad remesh.

THE FIX (QuadriFlow Def 3.1, done right). The old `_stripe.stripe_offsets`
computes the two integer-offset families (u and v) INDEPENDENTLY per scalar phase,
ignoring the cross-field per-edge rotation k_ij. Across an edge where the cross
rotates 90deg (k_ij=1) the u/v axes SWAP, so the per-vertex integer coords of the
two endpoints live in DIFFERENT frames and must be reconciled by R2(k_ij) before
differencing. Without that the seed offsets are inconsistent EVERYWHERE near
singularities and the MCF (`repair_regularity`) has to move them |drep-seed|1~9458
-> wrecked collapse geometry -> low val4.

THE SEAMLESS OFFSET (the whole point). Assign each vertex a per-vertex INTEGER
lattice coord t_v in Z^2, in v's OWN tangent frame (q_v, n_v x q_v). For the
canonical directed edge e=(lo,hi) with cross rotation k = k_{lo->hi} (the SAME
rotation `_im_extract._edge_lattice_data` / `extract_graph`'s BFS use), the
seamless per-edge step (in lo's frame, the convention `extract_graph` reads) is

        step_seamless[e] = R2(k) @ t_hi - t_lo .

Derivation: `extract_graph`'s BFS sets g[b] = g[a] + R2(krot[a]) @ step_a with
krot[b] = krot[a] + k_{a->b}. If g[v] = R2(krot[v]) @ t_v then path-independence
forces exactly step[e] = R2(k) @ t_hi - t_lo. This is a COBOUNDARY in the twisted
(rotation-cocycle) cohomology -> the per-triangle regularity sum (QuadriFlow Eq.3)
telescopes to ZERO on every triangle whose rotation holonomy is trivial (the
~all non-singular triangles). Only the genuine singularities (holonomy = +-90deg,
sum index = chi by Poincare-Hopf) can't be made consistent -- exactly the few the
MCF must touch. So |drep-seed|1 collapses from ~9458 to ~the number of singular
triangles.

We obtain t_v by INTEGRATING the locally-rounded position-field steps over a BFS
spanning forest carrying the rotation (this is `extract_graph`'s own g, converted
to each vertex's local frame: t_v = R2(-krot[v]) @ g[v]). On the spanning tree the
seamless step reproduces the local step EXACTLY; off-tree edges get the
seamless-corrected value -- the minimal change that makes them tree-consistent.

Permissive only: numpy / scipy / trimesh / networkx. NO GPL.
"""
import numpy as np
from collections import defaultdict, deque

from . import _field_quad as fq
from . import _quadflow as qf
from . import _im_extract as ie
from ._quadflow import _R2


# ---------------------------------------------------------------------------
#  per-vertex integer lattice coords by rotation-carrying BFS integration
# ---------------------------------------------------------------------------
def lattice_coords(mesh, step, k_ij, E, seed_vertex_phase=None):
    """Integrate the per-edge integer steps over a BFS spanning forest of the
    mesh, carrying the cross-field rotation, to give every vertex a global
    integer 2D lattice coord g[v] in its component's ROOT frame, and the rotation
    krot[v] taking v's local frame into that root frame. Then the per-vertex
    LOCAL integer coord is t_v = R2(-krot[v]) @ g[v].

    `step` (Ne,2 int): canonical (lo->hi, lo-frame) integer lattice step.
    `k_ij` (Ne,) int:  cross rotation lo-frame -> hi-frame.
    `E`    (Ne,2) int:  canonical edges (lo<hi).
    Long edges (|step|inf>1) are excluded from the spanning forest (they cannot
    carry a single lattice step) so they never corrupt the integration -- exactly
    as `extract_graph` does.

    Returns t (Vo,2 int), g (Vo,2 int), krot (Vo,) int, comp (Vo,) int.
    Identical integration maths to _im_extract.extract_graph (so the seamless
    offsets agree with what the extractor will read on the tree)."""
    Vo = int(mesh.vertices.shape[0])
    Ne = len(E)
    absmax = np.abs(step).max(axis=1)
    adj = defaultdict(list)
    for e in range(Ne):
        if absmax[e] > 1:
            continue
        a, b = int(E[e, 0]), int(E[e, 1])
        adj[a].append((b, e, +1))
        adj[b].append((a, e, -1))
    g = np.zeros((Vo, 2), np.int64)
    krot = np.zeros(Vo, np.int64)
    comp = -np.ones(Vo, np.int64)
    seen = np.zeros(Vo, bool)
    ncomp = 0
    for s0 in range(Vo):
        if seen[s0]:
            continue
        seen[s0] = True
        comp[s0] = ncomp
        dq = deque([s0])
        while dq:
            a = dq.popleft()
            for (b, e, sgn) in adj[a]:
                if seen[b]:
                    continue
                seen[b] = True
                comp[b] = ncomp
                st = step[e]
                if sgn > 0:                                # a=lo, b=hi
                    step_a = st
                    k_a_b = int(k_ij[e])
                else:                                      # a=hi, b=lo
                    step_a = -(_R2((-int(k_ij[e])) % 4) @ st)
                    k_a_b = (-int(k_ij[e])) % 4
                g[b] = g[a] + (_R2(int(krot[a])) @ step_a)
                krot[b] = (int(krot[a]) + k_a_b) % 4
                dq.append(b)
        ncomp += 1
    # local coord t_v = R2(-krot[v]) @ g[v]
    t = np.zeros((Vo, 2), np.int64)
    for v in range(Vo):
        t[v] = _R2((-int(krot[v])) % 4) @ g[v]
    return t, g, krot, comp


def seamless_offsets(mesh, o, N, q, scale, step_local=None, k_ij=None, E=None):
    """SEAMLESS per-edge integer offsets via Def 3.1.

    1. local per-edge step + rotation k_ij from the position field `o`
       (_im_extract._edge_lattice_data) -- unless supplied.
    2. per-vertex integer lattice coords t_v by rotation-carrying BFS
       (`lattice_coords`).
    3. seamless step[e] = R2(k_ij[e]) @ t_hi - t_lo   (lo frame).

    Returns dict:
      E        (Ne,2)  canonical edges
      k_ij     (Ne,)   cross rotations
      step     (Ne,2)  SEAMLESS integer offsets (lo frame)  <- the seed
      step_local (Ne,2) the locally-rounded offsets (for the |seamless-local|1 diag)
      t        (Vo,2)  per-vertex integer lattice coords
      comp,krot
    """
    if step_local is None or k_ij is None or E is None:
        ld = ie._edge_lattice_data(mesh, o, N, q, scale)
        E = ld["E"]; step_local = ld["step"]; k_ij = ld["k_ij"]
    t, g, krot, comp = lattice_coords(mesh, step_local, k_ij, E)
    Ne = len(E)
    step = np.zeros((Ne, 2), np.int64)
    for e in range(Ne):
        lo, hi = int(E[e, 0]), int(E[e, 1])
        step[e] = _R2(int(k_ij[e])) @ t[hi] - t[lo]
    return {"E": E, "k_ij": k_ij, "step": step, "step_local": step_local,
            "t": t, "g": g, "krot": krot, "comp": comp}


# ---------------------------------------------------------------------------
#  STEP 5 -- subdivide long edges so |step|inf <= 1 (rotation-correct)
# ---------------------------------------------------------------------------
def subdivide_seamless(mesh, S, scale, project=True):
    """Split every seamless edge with |step|inf > 1 into a chain of unit/short
    edges, inserting midpoint vertices, until ALL edges have |step|inf <= 1. This
    is QuadriFlow STEP 5 done correctly for the seamless field: the consistent
    seed concentrates the unavoidable winding (Poincare-Hopf) into a few long
    edges; subdividing turns each long edge into a chain of grid cells the
    extractor CAN integrate (instead of dropping it).

    The seamless field has a GLOBAL lattice coord g[v] (root frame) per original
    vertex (path-independent where non-singular). A long edge (lo,hi) spans
    g[hi]-g[lo] = several cells; we place one midpoint per intermediate integer
    cell along the straight lattice segment from g[lo] to g[hi], with its 3D
    position linearly interpolated (then surface-projected). Every sub-edge then
    carries a UNIT (or zero) lattice step in the ROOT frame -- rotation k=0 on all
    sub-edges (they live in the single root frame of the component), so no
    rotation bookkeeping is needed inside a subdivided edge.

    Returns dict:
      P     (Nv,3)   3D positions (originals + midpoints)
      g     (Nv,2)   global lattice coord (root frame), integer
      comp  (Nv,)    component id
      edges list of (a,b) augmented-vertex pairs
      estep (Ne,2)   per-aug-edge integer lattice step a->b in ROOT frame
                     (|.|inf <= 1 by construction)
      Vo    int      number of original vertices (first Vo of P)
    """
    Vo = int(mesh.vertices.shape[0])
    E = S["E"]
    g = S["g"].astype(np.int64).copy()
    comp = S["comp"]
    V3 = np.asarray(mesh.vertices, float)
    P = [V3[i].copy() for i in range(Vo)]
    gg = [g[i].copy() for i in range(Vo)]
    cc = [int(comp[i]) for i in range(Vo)]
    edges = []
    estep = []
    Ne = len(E)
    for e in range(Ne):
        lo, hi = int(E[e, 0]), int(E[e, 1])
        if comp[lo] != comp[hi]:
            continue                                   # cross-component (rare)
        # global lattice displacement lo->hi in ROOT frame
        glo = g[lo]; ghi = g[hi]
        dg = ghi - glo                                 # integer 2D (root frame)
        M = int(np.abs(dg).max())
        if M <= 1:
            edges.append((lo, hi)); estep.append(dg.copy()); continue
        # chain lo -> m_1 -> ... -> m_{M-1} -> hi, one cell of dg per step.
        # distribute dg as evenly as possible into M unit-ish steps.
        steps = _split_disp(dg, M)                     # list of M int2 each |.|inf<=1
        prev = lo; prev_g = glo.copy()
        for s in range(M):
            if s == M - 1:
                cur = hi; cur_g = ghi
            else:
                cur = len(P)
                cur_g = prev_g + steps[s]
                frac = (s + 1) / M
                P.append((1 - frac) * V3[lo] + frac * V3[hi])
                gg.append(cur_g.copy())
                cc.append(int(comp[lo]))
            edges.append((prev, cur)); estep.append(steps[s].copy())
            prev = cur; prev_g = cur_g
    Parr = np.array(P, float)
    if project:
        try:
            cp, _, _ = mesh.nearest.on_surface(Parr)
            Parr = np.asarray(cp, float)
        except Exception:
            pass
    return {"P": Parr, "g": np.array(gg, np.int64), "comp": np.array(cc, np.int64),
            "edges": edges, "estep": np.array(estep, np.int64) if estep
            else np.zeros((0, 2), np.int64), "Vo": Vo}


def _split_disp(dg, M):
    """Split integer 2D displacement dg into M steps each with |.|inf<=1, summing
    to dg. Bresenham-like: spread the |dg[0]| and |dg[1]| unit moves across the M
    slots as evenly as possible."""
    dg = np.asarray(dg, np.int64)
    out = np.zeros((M, 2), np.int64)
    for c in range(2):
        n = int(dg[c]); s = 1 if n >= 0 else -1; n = abs(n)
        # place n unit moves into M slots, evenly
        if n == 0:
            continue
        # slot indices: round((k+0.5)*M/n) for k in range(n)
        for k in range(n):
            slot = int((k + 0.5) * M / n)
            if slot >= M:
                slot = M - 1
            out[slot, c] += s
    return [out[i] for i in range(M)]


# ---------------------------------------------------------------------------
#  STEP 8 -- extraction on the (subdivided) seamless lattice
# ---------------------------------------------------------------------------
def extract_seamless(mesh, sub, N, q, scale, verbose=False):
    """Robust lattice extraction on the subdivided seamless graph. Because every
    augmented vertex carries a GLOBAL integer lattice coord g (root frame) and
    every sub-edge a unit step, clustering is trivial and exact: merge all
    vertices that share (component, g). Output edges = unit-step links between
    distinct clusters. Then ring-trace via the SAME _im_extract.extract_faces
    rotation-system tracer (we build the graph dict it expects).

    Returns (faces, Vq) exactly like _im_extract.extract_faces."""
    P = sub["P"]; g = sub["g"]; comp = sub["comp"]
    edges = sub["edges"]; estep = sub["estep"]
    Nv = len(P)

    # cluster = unique (component, g)
    key_of = {}
    label = np.empty(Nv, np.int64)
    members = []
    for v in range(Nv):
        kk = (int(comp[v]), int(g[v, 0]), int(g[v, 1]))
        ci = key_of.get(kk, -1)
        if ci < 0:
            ci = len(members); key_of[kk] = ci; members.append([])
        label[v] = ci; members[ci].append(v)

    # split spatially-incoherent clusters (a wrong winding can co-locate two cells)
    mesh_nbr = defaultdict(set)
    for (a, b) in edges:
        mesh_nbr[a].add(b); mesh_nbr[b].add(a)
    thr2 = (0.9 * scale) ** 2
    split_members = []
    for mem in members:
        if len(mem) <= 1:
            split_members.append(mem); continue
        Pm = P[mem]
        if float(((Pm - Pm.mean(0)) ** 2).sum(1).max()) <= thr2:
            split_members.append(mem); continue
        memset = set(mem); seen_m = set()
        for s in mem:
            if s in seen_m:
                continue
            cm = []; stack = [s]; seen_m.add(s)
            while stack:
                x = stack.pop(); cm.append(x)
                for y in mesh_nbr[x]:
                    if y in memset and y not in seen_m and \
                       float(np.sum((P[x] - P[y]) ** 2)) <= thr2:
                        seen_m.add(y); stack.append(y)
            split_members.append(cm)
    members = split_members
    label = np.empty(Nv, np.int64)
    for ci, mem in enumerate(members):
        for v in mem:
            label[v] = ci
    Nq = len(members)

    # cluster representative frames (for rotational ordering) + positions
    Vq = np.zeros((Nq, 3))
    rep_n = np.zeros((Nq, 3)); rep_q = np.zeros((Nq, 3))
    # map augmented vertex -> nearest original vertex for frame lookup
    V3 = np.asarray(mesh.vertices, float)
    from scipy.spatial import cKDTree
    tree = cKDTree(V3)
    for c in range(Nq):
        mem = members[c]
        Vq[c] = P[mem].mean(axis=0)
        # frames from nearest original verts
        idxs = tree.query(P[mem])[1]
        rep_n[c] = N[idxs].mean(axis=0)
        rep_q[c] = q[idxs].mean(axis=0)
    rep_n /= (np.linalg.norm(rep_n, axis=1, keepdims=True) + 1e-12)
    rep_q = rep_q - np.einsum("ij,ij->i", rep_q, rep_n)[:, None] * rep_n
    rep_q /= (np.linalg.norm(rep_q, axis=1, keepdims=True) + 1e-12)
    rep_t = np.cross(rep_n, rep_q)

    # output edges: unit-step links between DISTINCT clusters
    asum = np.abs(estep).sum(axis=1)
    raw = defaultdict(set)
    for ei, (a, b) in enumerate(edges):
        if asum[ei] != 1:                              # only single lattice steps
            continue
        ca, cb = int(label[a]), int(label[b])
        if ca == cb:
            continue
        raw[ca].add(cb); raw[cb].add(ca)

    keep = np.ones(Nq, bool)
    nbr = {c: set(raw[c]) for c in range(Nq)}

    # triangle pruning (a quad graph has no 3-cycles): drop the most-diagonal edge
    def edge_diag_cost(a, b):
        v = Vq[b] - Vq[a]; cost = 0.0
        for c in (a, b):
            ang = np.arctan2(float(v @ rep_t[c]), float(v @ rep_q[c]))
            cost += abs(np.sin(2 * ang))
        return cost
    removed = set()
    def adj_now(a, b):
        return (b in nbr[a]) and ((a, b) not in removed) and ((b, a) not in removed)
    tris = []
    for a in range(Nq):
        na = sorted(x for x in nbr[a] if x > a)
        for ix in range(len(na)):
            b = na[ix]
            for c in na[ix + 1:]:
                if b in nbr[c]:
                    tris.append((a, b, c))
    for (a, b, c) in tris:
        if not (adj_now(a, b) and adj_now(b, c) and adj_now(a, c)):
            continue
        cabs = [((a, b), edge_diag_cost(a, b)), ((b, c), edge_diag_cost(b, c)),
                ((a, c), edge_diag_cost(a, c))]
        (x, y), _ = max(cabs, key=lambda tt: tt[1])
        nbr[x].discard(y); nbr[y].discard(x); removed.add((x, y))

    # dissolve valence-1/2
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
                o1 = next(iter(nbr[c])); nbr[o1].discard(c)
                keep[c] = False; nbr[c] = set(); changed = True
            elif deg == 2:
                a, b = tuple(nbr[c])
                if a != b and b not in nbr[a]:
                    nbr[a].discard(c); nbr[b].discard(c)
                    nbr[a].add(b); nbr[b].add(a)
                    keep[c] = False; nbr[c] = set(); changed = True

    # rotational ordering
    out_adj = [[] for _ in range(Nq)]
    for c in range(Nq):
        if not keep[c]:
            continue
        nbrs = [nb for nb in nbr[c] if keep[nb]]
        if not nbrs:
            continue
        vecs = Vq[nbrs] - Vq[c]
        ang = np.arctan2(vecs @ rep_t[c], vecs @ rep_q[c])
        out_adj[c] = [int(nbrs[k]) for k in np.argsort(ang)]

    if verbose:
        print(f"    [seamless-extract] aug_V={Nv} clusters={Nq} "
              f"kept={int(keep.sum())} long_subdiv_edges={len(edges)}")

    graph = {"Vq": Vq, "out_adj": out_adj, "keep": keep,
             "rep_q": rep_q, "rep_n": rep_n, "rep_t": rep_t}
    faces, Vq2 = ie.extract_faces(graph, verbose=verbose)
    return faces, Vq2


# ---------------------------------------------------------------------------
#  TOP LEVEL -- remesh_seamless
# ---------------------------------------------------------------------------
def _prep_mesh(mesh, target_quads):
    """clean + isotropic-remesh into the rho ~ mean-edge regime (same as
    remesh_im) so the lattice has ~one cell per vertex and the field is coherent."""
    m = qf.clean_mesh(mesh, target_faces=min(len(mesh.faces), 30000))
    area = float(m.area)
    L = float(np.sqrt(area / max(target_quads, 1))) / 1.4
    el0 = float(np.linalg.norm(
        m.vertices[m.edges_unique[:, 0]] - m.vertices[m.edges_unique[:, 1]], axis=1).mean())
    cv0 = float(np.std(np.linalg.norm(
        m.vertices[m.edges_unique[:, 0]] - m.vertices[m.edges_unique[:, 1]], axis=1)) / el0)
    if cv0 > 0.22 or el0 < 0.6 * L:
        try:
            m2 = qf.isotropic_remesh(m, target_edge=L, iters=7)
            if m2.is_watertight == m.is_watertight and len(m2.faces) > 0:
                m = m2
        except Exception:
            pass
    return m


def remesh_seamless(mesh, target_quads=1500, verbose=False, mode="auto",
                    curv_field=False):
    """SEAMLESS (rotation-aware) integer-grid quad remesh.

       clean + isotropic-remesh  ->  4-RoSy cross field  ->  4-PoSy position field
       ->  SEAMLESS integer offsets (Def 3.1, R2(k)t_hi - t_lo; residual_tris =
           #cross-field singularities, the Poincare-Hopf floor)
       ->  subdivide long edges so |step|inf <= 1 (rotation-correct chains)
       ->  extract_seamless (global-lattice clustering + ring tracing)
       ->  manifold/watertight finish (reuse _grid_place).

    mode="seamless" : always use the seamless+subdivide path.
    mode="local"    : use the local (non-seamless) extract_graph seed (the IM
                      baseline) -- kept for honest A/B and because, empirically,
                      the local seed extracts at HIGHER val4 on organic shapes
                      (the seamless field's topologically-forced winding
                      concentration, though correctly subdivided, fragments more
                      than the extractor's native irregular-vertex handling).
    mode="auto"     : pick per-shape by extracting both and keeping the higher
                      val4 (default; honest best-of).

    Returns (Vq (N,3), Q (M,4)). Never raises."""
    import time
    from . import _grid_place as gp
    try:
        t_all = time.time()
        m = _prep_mesh(mesh, target_quads)
        el = float(np.linalg.norm(
            m.vertices[m.edges_unique[:, 0]] - m.vertices[m.edges_unique[:, 1]],
            axis=1).mean())
        scale = el
        d = fq.smooth_field(m, curv=curv_field, verbose=verbose)
        N, O = qf.vertex_frames(m, d); q = O[:, :, 0]
        o, Np, qp = qf.position_field(m, d, scale, iters=25)
        ld = ie._edge_lattice_data(m, o, N, q, scale)

        def finish(faces, Vq):
            if not faces:
                return np.zeros((0, 3)), np.zeros((0, 4), np.int64)
            Q = np.asarray(faces, np.int64)
            Vq = gp.project_to_surface(m, Vq)
            Q = gp._drop_degenerate_quads(Vq, Q)
            Vq, Q = gp._enforce_manifold(m, Vq, Q)
            Vq, Q, _ = gp._cap_border_holes(m, Vq, Q)
            Vq, Q = ie.clean_quad_valence(Vq, Q, iters=5)
            Vq, Q = gp._enforce_manifold(m, Vq, Q)
            if len(Q):
                used = np.unique(Q); remap = -np.ones(len(Vq), np.int64)
                remap[used] = np.arange(len(used)); Vq = Vq[used]; Q = remap[Q]
            Vq = gp.relax_quads(m, Vq, Q, iters=6, lam=0.5)
            return Vq, Q

        def val4(Q, n):
            adj = defaultdict(set)
            for qq in Q:
                u = [int(x) for x in dict.fromkeys(int(c) for c in qq)]
                for a, b in zip(u, u[1:] + u[:1]):
                    adj[a].add(b); adj[b].add(a)
            vals = [len(adj[i]) for i in adj]
            return 100.0 * np.mean([v == 4 for v in vals]) if vals else 0.0

        results = {}
        if mode in ("local", "auto"):
            g0 = ie.extract_graph(m, o, N, q, scale, p3d=None, drep=ld["step"],
                                  verbose=verbose)
            f0, V0 = ie.extract_faces(g0, verbose=verbose)
            results["local"] = finish(f0, V0)
        if mode in ("seamless", "auto"):
            S = seamless_offsets(m, o, N, q, scale, step_local=ld["step"],
                                 k_ij=ld["k_ij"], E=ld["E"])
            if verbose:
                rl = triangle_residual(m, S["E"], ld["step"], S["k_ij"])
                rs = triangle_residual(m, S["E"], S["step"], S["k_ij"])
                idx = fq.singularities(m, d)
                print(f"  seamless: residual_tris LOCAL={rl} SEAMLESS={rs} "
                      f"(#cross_sing={int((idx!=0).sum())})")
            sub = subdivide_seamless(m, S, scale, project=True)
            f1, V1 = extract_seamless(m, sub, N, q, scale, verbose=verbose)
            results["seamless"] = finish(f1, V1)

        if mode == "auto":
            best = max(results.items(),
                       key=lambda kv: (val4(kv[1][1], len(kv[1][0])), len(kv[1][1])))
            if verbose:
                for k, (Vv, Qq) in results.items():
                    print(f"  [{k}] val4={val4(Qq, len(Vv)):.1f}% q={len(Qq)}")
                print(f"  AUTO -> {best[0]} ({time.time()-t_all:.0f}s)")
            return best[1]
        Vq, Q = results[mode]
        if verbose:
            print(f"  total {time.time()-t_all:.0f}s")
        return Vq, Q
    except Exception:
        if verbose:
            import traceback; traceback.print_exc()
        return np.zeros((0, 3)), np.zeros((0, 4), np.int64)


# ---------------------------------------------------------------------------
#  TOP LEVEL -- remesh_mcf2  (convention-matched MCF regularity repair)
# ---------------------------------------------------------------------------
def _mcf2_lattice(mesh, drep, k_ij, E):
    """Build the global lattice integration (g, comp, krot) of the REPAIRED mcf2
    field, then a seamless-style step from those coords so the long-edge
    subdivision machinery (`subdivide_seamless`) can run on the mcf2 path too.

    The local mcf2 seed leaves ~535 edges with |step|inf>1 (the position field
    genuinely stretches >1 cell on curved/elongated regions); `extract_graph`
    SKIPS those, dropping connections and creating a valence-3 glut. Subdividing
    them into unit-step chains recovers the missing connections."""
    t, g, krot, comp = lattice_coords(mesh, drep, k_ij, E)
    Ne = len(E)
    step = np.zeros((Ne, 2), np.int64)
    for e in range(Ne):
        lo, hi = int(E[e, 0]), int(E[e, 1])
        if comp[lo] == comp[hi]:
            step[e] = _R2(int(k_ij[e])) @ t[hi] - t[lo]
        else:
            step[e] = drep[e]
    return {"E": E, "k_ij": k_ij, "step": step, "t": t, "g": g,
            "krot": krot, "comp": comp}


def remesh_mcf2(mesh, target_quads=1500, verbose=False, seed="local",
                box_clamp=True, return_stats=False, curv_field=False,
                subdiv=False, merge_max_val=5):
    """Field-aligned quad remesh using the CONVENTION-MATCHED min-cost-flow
    regularity repair (`_mcf2.repair`).

       clean + isotropic-remesh -> 4-RoSy cross field -> 4-PoSy position field
       -> per-edge LOCAL lattice step (`_edge_lattice_data`)
       -> MCF repair in the PATH-INTEGRAL convention (the SAME residual the
          extractor reads; box_clamp keeps every final |step|inf<=1 so the
          topologically-forced winding SPREADS across unit edges)
       -> _im_extract.extract_graph(drep=repaired) -> extract_faces
       -> manifold/watertight finish (reuse _grid_place).

    seed : which offset field the flow patches.
       "local"    -> the locally-rounded position-field step (already SPREAD;
                     the recommended seed -- the flow only nudges it consistent).
       "seamless" -> the BFS-integrated seamless step (CONCENTRATES winding into
                     a few long edges; kept for the honest A/B).
    box_clamp : keep final |step|inf <= 1 (spread; default).  False lets the flow
                make long edges (concentrate) -- for the A/B only.

    Returns (Vq (N,3), Q (M,4)); with return_stats also a dict of measured
    numbers (residual_tris, cross_sing, long edges before/after, |drep-seed|1).
    Never raises."""
    import time
    from . import _grid_place as gp
    from . import _mcf2 as mcf2
    stats = {}
    try:
        t_all = time.time()
        m = _prep_mesh(mesh, target_quads)
        el = float(np.linalg.norm(
            m.vertices[m.edges_unique[:, 0]] - m.vertices[m.edges_unique[:, 1]],
            axis=1).mean())
        scale = el
        d = fq.smooth_field(m, curv=curv_field, verbose=verbose)
        N, O = qf.vertex_frames(m, d); q = O[:, :, 0]
        o, Np, qp = qf.position_field(m, d, scale, iters=25)
        ld = ie._edge_lattice_data(m, o, N, q, scale)
        E = ld["E"]; k_ij = ld["k_ij"]; step_local = ld["step"]

        idx = fq.singularities(m, d)
        nsing = int(np.sum(idx != 0))

        if seed == "seamless":
            S = seamless_offsets(m, o, N, q, scale, step_local=step_local,
                                 k_ij=k_ij, E=E)
            seed_step = S["step"]
        else:
            seed_step = step_local
        drep, mst = mcf2.repair(m, E, k_ij, seed_step, box_clamp=box_clamp,
                                verbose=verbose)

        stats.update({"cross_sing": nsing,
                      "residual_tris": mst["residual_triangles"],
                      "long_before": mst["long_before"],
                      "long_after": mst["long_after"],
                      "l1_change": mst["l1_change"],
                      "feasible": mst["feasible"]})
        if verbose:
            print(f"  mcf2[{seed},clamp={box_clamp}]: cross_sing={nsing} "
                  f"residual_tris={mst['residual_triangles']} "
                  f"long {mst['long_before']}->{mst['long_after']} "
                  f"|drep-seed|1={mst['l1_change']}")

        if subdiv:
            # subdivide the repaired field's long edges into unit-step chains so
            # the extractor keeps (not skips) those connections -> fewer deg-3.
            Smc = _mcf2_lattice(m, drep, k_ij, E)
            sub = subdivide_seamless(m, Smc, scale, project=True)
            faces, Vq = extract_seamless(m, sub, N, q, scale, verbose=verbose)
        else:
            graph = ie.extract_graph(m, o, N, q, scale, p3d=None, drep=drep,
                                     verbose=verbose)
            faces, Vq = ie.extract_faces(graph, verbose=verbose)
        if not faces:
            Vq, Q = np.zeros((0, 3)), np.zeros((0, 4), np.int64)
        else:
            Q = np.asarray(faces, np.int64)
            Vq = gp.project_to_surface(m, Vq)
            Q = gp._drop_degenerate_quads(Vq, Q)
            Vq, Q = gp._enforce_manifold(m, Vq, Q)
            Vq, Q, _ = gp._cap_border_holes(m, Vq, Q)
            Vq, Q = ie.clean_quad_valence(Vq, Q, iters=5,
                                          merge_max_val=merge_max_val)
            Vq, Q = gp._enforce_manifold(m, Vq, Q)
            Vq, Q, _ = gp._cap_border_holes(m, Vq, Q)
            Vq, Q = gp._enforce_manifold(m, Vq, Q)
            if len(Q):
                used = np.unique(Q); remap = -np.ones(len(Vq), np.int64)
                remap[used] = np.arange(len(used)); Vq = Vq[used]; Q = remap[Q]
            Vq = gp.relax_quads(m, Vq, Q, iters=6, lam=0.5)
        stats["runtime"] = time.time() - t_all
        if verbose:
            print(f"  total {stats['runtime']:.0f}s quads={len(Q)}")
        return (Vq, Q, stats) if return_stats else (Vq, Q)
    except Exception:
        if verbose:
            import traceback; traceback.print_exc()
        empty = (np.zeros((0, 3)), np.zeros((0, 4), np.int64))
        return (empty[0], empty[1], stats) if return_stats else empty


# ---------------------------------------------------------------------------
#  diagnostics
# ---------------------------------------------------------------------------
def triangle_residual(mesh, off_E, step, k_ij):
    """QuadriFlow Eq.3 residual per triangle: walk the 3 edges in the FIRST
    vertex's frame; the rotated step sum must be 0 for a regular triangle. Counts
    triangles with nonzero residual = position singularities the seed still carries.

    For triangle (a,b,c), build the 2D lattice offset of each corner relative to a
    by composing edge steps with rotations (same convention as lattice_coords),
    then the closing residual = offset(a->a around the loop). Equivalent: integrate
    around the triangle; residual != 0 iff singular OR seed inconsistent."""
    F = np.asarray(mesh.faces, np.int64)
    eidx = {(int(a), int(b)): i for i, (a, b) in enumerate(off_E)}

    def directed(a, b):
        """integer step a->b in a's frame, and rotation a-frame->b-frame."""
        if a < b:
            e = eidx[(a, b)]
            return step[e].copy(), int(k_ij[e])
        else:
            e = eidx[(b, a)]
            kab = (-int(k_ij[e])) % 4
            return -(_R2(kab) @ step[e]), kab

    nbad = 0
    for tri in F:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        # accumulate around a->b->c->a in a's frame
        pos = np.zeros(2, np.int64)
        krot = 0
        ok = True
        for (x, y) in ((a, b), (b, c), (c, a)):
            if (min(x, y), max(x, y)) not in eidx:
                ok = False
                break
            st, kab = directed(x, y)
            pos = pos + (_R2(krot) @ st)
            krot = (krot + kab) % 4
        if not ok:
            continue
        if int(np.abs(pos).sum()) != 0 or krot != 0:
            nbad += 1
    return nbad


if __name__ == "__main__":
    import trimesh, time
    print("=== seamless offset self-test: triangle residual + |seamless-local|1 ===")

    def dense_cylinder(radius=1.0, height=4.0, n_theta=48):
        circ = 2 * np.pi * radius / n_theta
        n_z = max(2, int(round(height / circ)))
        th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        zs = np.linspace(-height / 2, height / 2, n_z + 1)
        V = [[radius * np.cos(t), radius * np.sin(t), z] for z in zs for t in th]
        F = []
        for iz in range(n_z):
            for it in range(n_theta):
                a = iz * n_theta + it; b = iz * n_theta + (it + 1) % n_theta
                c = (iz + 1) * n_theta + it; dd = (iz + 1) * n_theta + (it + 1) % n_theta
                F.append([a, b, dd]); F.append([a, dd, c])
        return trimesh.Trimesh(np.array(V), np.array(F), process=True)

    cases = [("dense_cyl", dense_cylinder()),
             ("sphere", trimesh.creation.icosphere(subdivisions=3))]
    for nm, m0 in cases:
        m = qf.clean_mesh(m0)
        el = float(np.linalg.norm(
            m.vertices[m.edges_unique[:, 0]] - m.vertices[m.edges_unique[:, 1]],
            axis=1).mean())
        d = fq.smooth_field(m)
        o, N, q = qf.position_field(m, d, el, iters=20)
        t0 = time.time()
        S = seamless_offsets(m, o, N, q, el)
        # singularities of the cross field (Poincare-Hopf reference)
        idx = fq.singularities(m, d)
        nsing = int(np.sum(idx != 0))
        chi = int(m.euler_number)
        # residual of the LOCAL seed vs the SEAMLESS seed
        ld = ie._edge_lattice_data(m, o, N, q, el)
        res_local = triangle_residual(m, S["E"], ld["step"], S["k_ij"])
        res_seam = triangle_residual(m, S["E"], S["step"], S["k_ij"])
        l1 = int(np.abs(S["step"] - S["step_local"]).sum())
        print(f"  {nm:10}: V={len(m.vertices)} chi={chi} cross_sing={nsing} "
              f"(sum_idx={int(idx.sum())}=4chi?{int(idx.sum())==4*chi}) "
              f"| residual_tris LOCAL={res_local} SEAMLESS={res_seam} "
              f"|seamless-local|1={l1} ({time.time()-t0:.1f}s)")
