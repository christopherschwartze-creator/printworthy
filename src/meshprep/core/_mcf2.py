"""Convention-matched MIN-COST-FLOW regularity repair for the seamless field.

WHY THIS EXISTS (the crux, file-verified). `_quadflow.build_regularity_system`
computes a triangle's regularity residual in a SINGLE-REFERENCE-FRAME convention
(`tri_rotk = k(a,lo) - k(lo,a)`) that DISAGREES with the convention the quad
extractor (`_im_extract.extract_graph`) and the seamless diagnostic
(`_seamless.triangle_residual`) actually use -- a PATH-INTEGRAL around the
triangle's 3 edges, transporting the per-edge integer step through the cross-
field rotation cocycle.  On the seamless seed the extractor reads exactly the
cross-field singularity count (sphere 8), but the old MCF's `_residual_b` reports
~75 (67 false positives), so the MCF fights a residual the extractor doesn't
share.  This module rebuilds the residual in the PATH-INTEGRAL convention so the
MCF minimizes the SAME residual the extractor sees.

WHAT IT ROUTES (fix #2 -- spreading the winding).  Seeded on the LOCAL
position-field offsets (`_edge_lattice_data(...)["step"]`, which are already
SPREAD across the surface -- they carry no concentrated BFS winding) the flow
makes them PATH-INTEGRAL-consistent with `box_clamp` keeping every final edge
|step|inf <= 1.  So the unavoidable Poincare-Hopf winding is distributed across
many unit edges rather than concentrated into a few |step|inf~60 edges (the
seamless BFS seed's failure mode) that subdivide into irregular filaments.

THE RESIDUAL, AS A LINEAR FUNCTION OF THE EDGE STEPS.  For triangle (a,b,c),
accumulate the path-integral in a's frame:
    res_t = sum_{j in [ab,bc,ca]}  C_{t,j} @ step[e_j]
with, for the canonical edge variable step[e_j] (always lo->hi, lo frame):
    traversal canonical (x<y):  C = +R2(krot_before_j)
    traversal reversed  (x>y):  C = -R2(krot_before_j + (-k_ij[e_j]) % 4)
where krot_before_j is the cumulative cross-field rotation of the edges traversed
BEFORE edge j (krot starts 0 at a).  Each C is +-a 90-degree integer rotation, so
each scalar row of res_t couples ONE scalar component of step[e_j] with sign +-1
-- exactly the structure the network-simplex min-cost flow wants.  res_t == 0 iff
the triangle is regular in the convention the extractor reads; res_t != 0 only on
the few genuine field singularities (which the flow cannot remove -- Poincare-Hopf
-- and leaves as irregular vertices, the documented robust behaviour).

This is a clean reimplementation of `_quadflow.repair_regularity`'s flow reduction
with the corrected coefficients; the BFS dual-balancing + flow construction are
structurally the same.  Permissive only: numpy / scipy / networkx.  NO GPL.
"""
import numpy as np
from collections import defaultdict, deque

from ._quadflow import _R2


# ---------------------------------------------------------------------------
#  per-triangle path-integral coefficients (the convention fix)
# ---------------------------------------------------------------------------
def build_pathint_system(mesh, E, k_ij):
    """Build the per-triangle path-integral regularity coefficients.

    For every triangle (a,b,c) and its 3 traversal edges (a->b),(b->c),(c->a),
    store the global canonical edge index, the SIGN and the cumulative ROTATION
    of that edge's canonical step variable in the triangle's path-integral
    residual (accumulated in a's frame).  `res_t = sum_j sign_{t,j} *
    R2(rotk_{t,j}) @ step[eid_{t,j}]` (a 2-vector); res_t==0 iff regular.

    Returns dict:
      F        (nF,3)  the mesh faces
      tri_eid  (nF,3)  global canonical edge index per traversal edge
      tri_sign (nF,3)  +1 if traversal == canonical (x<y) else -1
      tri_rotk (nF,3)  cumulative cross-field rotation (path-integral transport)
      edge2tri dict edge-index -> list of incident triangle ids
      boundary_edges set of edge indices on exactly one triangle
    """
    F = np.asarray(mesh.faces, np.int64)
    eidx = {(int(a), int(b)): i for i, (a, b) in enumerate(E)}
    nF = len(F)
    tri_eid = np.empty((nF, 3), np.int64)
    tri_sign = np.empty((nF, 3), np.int64)
    tri_rotk = np.empty((nF, 3), np.int64)
    for ti, tri in enumerate(F):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        krot = 0
        for j, (x, y) in enumerate(((a, b), (b, c), (c, a))):
            key = (x, y) if x < y else (y, x)
            e = eidx[key]
            tri_eid[ti, j] = e
            kc = int(k_ij[e])                       # rotation lo-frame -> hi-frame
            if x < y:                               # canonical traversal
                tri_sign[ti, j] = 1
                tri_rotk[ti, j] = krot % 4
                kab = kc
            else:                                   # reversed traversal
                kab = (-kc) % 4
                tri_sign[ti, j] = -1
                tri_rotk[ti, j] = (krot + kab) % 4
            krot = (krot + kab) % 4
    edge2tri = defaultdict(list)
    for ti in range(nF):
        for j in range(3):
            edge2tri[int(tri_eid[ti, j])].append(ti)
    boundary_edges = set(e for e, ts in edge2tri.items() if len(ts) == 1)
    return {"F": F, "tri_eid": tri_eid, "tri_sign": tri_sign,
            "tri_rotk": tri_rotk, "edge2tri": dict(edge2tri),
            "boundary_edges": boundary_edges, "E": E, "k_ij": k_ij}


def _coeff(sys, t, j):
    """Signed 2x2 integer rotation multiplying step[e_j] in triangle t's
    path-integral residual: sign * R2(rotk)."""
    return int(sys["tri_sign"][t, j]) * _R2(int(sys["tri_rotk"][t, j]))


def residual_triangles(sys, step):
    """Per-triangle path-integral residual; returns (nF,2) int and the count of
    nonzero (singular) triangles.  MATCHES _seamless.triangle_residual exactly."""
    nF = len(sys["tri_eid"])
    res = np.zeros((nF, 2), np.int64)
    for t in range(nF):
        acc = np.zeros(2, np.int64)
        for j in range(3):
            e = int(sys["tri_eid"][t, j])
            acc = acc + _coeff(sys, t, j) @ step[e]
        res[t] = acc
    nbad = int(np.sum(np.any(res != 0, axis=1)))
    return res, nbad


# ---------------------------------------------------------------------------
#  BFS dual-balancing: make each interior tree edge appear +1 / -1
# ---------------------------------------------------------------------------
def _balance(sys):
    """BFS over the dual (triangle adjacency) graph assigning a per-triangle
    extra rotation `balrot[t]` so that for each tree edge the shared step
    variable's coefficient in the two incident triangles is exactly +C and -C
    (it cancels for a regular pair).  Off-tree interior edges that do NOT
    sign-balance under the assigned balrots are FROZEN (they sit on a singular
    cycle); balanced off-tree edges stay mutable.  Boundary edges are frozen.

    Returns balrot (nF,), frozen (set of edge ids), in_tree (set of edge ids)."""
    F = sys["F"]; nF = len(F)
    tri_eid = sys["tri_eid"]; tri_sign = sys["tri_sign"]; tri_rotk = sys["tri_rotk"]
    edge2tri = sys["edge2tri"]
    balrot = np.zeros(nF, np.int64)
    visited = np.zeros(nF, bool)
    in_tree = set()

    def required_b2(t, j, t2, j2, bt):
        # want sign_t2 * R2(rotk_t2 + b2) = - sign_t * R2(rotk_t + bt)
        s_t = int(tri_sign[t, j]); s_t2 = int(tri_sign[t2, j2])
        r_t = int(tri_rotk[t, j]); r_t2 = int(tri_rotk[t2, j2])
        sign_extra = 2 if (s_t * s_t2) < 0 else 0
        return (r_t + bt - r_t2 + 2 + sign_extra) % 4

    for root in range(nF):
        if visited[root]:
            continue
        visited[root] = True
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
                j2 = int(np.where(tri_eid[t2] == e)[0][0])
                balrot[t2] = required_b2(t, j, t2, j2, int(balrot[t]))
                in_tree.add(e)
                dq.append(t2)

    frozen = set(sys["boundary_edges"])
    interior = set(e for e, ts in edge2tri.items() if len(ts) == 2)
    for e in (interior - in_tree):
        t, t2 = edge2tri[e][0], edge2tri[e][1]
        j = int(np.where(tri_eid[t] == e)[0][0])
        j2 = int(np.where(tri_eid[t2] == e)[0][0])
        need = required_b2(t, j, t2, j2, int(balrot[t]))
        if int(balrot[t2]) != need:
            frozen.add(e)              # frustrated -> on a singular cycle
        else:
            in_tree.add(e)             # balanced -> routable
    return balrot, frozen, in_tree


def _bcoeff(sys, balrot, t, j):
    """Balanced signed 2x2 coefficient: sign * R2(rotk + balrot[t])."""
    s = int(sys["tri_sign"][t, j])
    k = (int(sys["tri_rotk"][t, j]) + int(balrot[t])) % 4
    return s * _R2(k)


# ---------------------------------------------------------------------------
#  the min-cost flow
# ---------------------------------------------------------------------------
def repair(mesh, E, k_ij, seed, max_H=6, box_clamp=True, spread=True,
           verbose=False):
    """Convention-matched MCF regularity repair.

    mesh   : the triangle mesh (faces define the regularity triangles)
    E      : (Ne,2) canonical edges (lo<hi), 1:1 with `seed` rows
    k_ij   : (Ne,)  cross rotation lo-frame -> hi-frame
    seed   : (Ne,2 int) per-edge canonical step to repair (the LOCAL spread
             position-field offsets, by default -- see module docstring)
    box_clamp : keep every FINAL step within [-1, 1] (so no long edge is ever
                manufactured; the winding is forced to SPREAD across unit edges)
    spread : (only meaningful without box_clamp) cap |delta| per edge at 1 too

    Returns drep (Ne,2 int), stats dict.  drep is the repaired per-edge step in
    the SAME canonical (lo->hi, lo-frame) convention `extract_graph(drep=...)`
    consumes.  The flow MINIMIZES sum |drep - seed|_1 subject to A*drep = 0 in
    the path-integral convention, routed with networkx max_flow_min_cost.
    """
    import networkx as nx
    sys = build_pathint_system(mesh, E, k_ij)
    seed = np.asarray(seed, np.int64).copy()
    d = seed.copy()
    nF = len(sys["tri_eid"])
    balrot, frozen, in_tree = _balance(sys)

    # ---- scalar variable map: each mutable edge has 2 scalar DOFs (e, c) ----
    var_id = {}
    def vid(e, c):
        key = (e, c)
        if key not in var_id:
            var_id[key] = len(var_id)
        return var_id[key]

    # each row (t, p) :  sum_j C[p, c] * step[e_j][c] = 0   (one scalar per edge)
    row_terms = [[] for _ in range(2 * nF)]
    row_const = np.zeros(2 * nF, np.int64)
    for t in range(nF):
        for j in range(3):
            e = int(sys["tri_eid"][t, j])
            C = _bcoeff(sys, balrot, t, j)
            for p in range(2):
                c = int(np.nonzero(C[p])[0][0])
                sgn = int(C[p, c])
                r = 2 * t + p
                if e in frozen:
                    row_const[r] += sgn * int(d[e][c])     # frozen -> rhs
                else:
                    row_terms[r].append((vid(e, c), sgn))
    nvar = len(var_id)
    inv_var = {vv: key for key, vv in var_id.items()}

    # x* per scalar var (the seed)
    xstar = np.zeros(nvar, np.int64)
    for (e, c), vv in var_id.items():
        xstar[vv] = seed[e][c]

    # any scalar var not appearing exactly once +1 and once -1 -> freeze to x*
    appear = defaultdict(list)
    for r in range(2 * nF):
        for (vv, sgn) in row_terms[r]:
            appear[vv].append((r, sgn))
    bad = []
    for vv, occ in appear.items():
        signs = sorted(s for _, s in occ)
        if not (len(occ) == 2 and signs == [-1, 1]):
            bad.append(vv)
    badset = set(bad)
    for vv in bad:
        for (r, sgn) in appear[vv]:
            row_const[r] += sgn * int(xstar[vv])
    for r in range(2 * nF):
        row_terms[r] = [(vv, s) for (vv, s) in row_terms[r] if vv not in badset]
    appear = defaultdict(list)
    for r in range(2 * nF):
        for (vv, sgn) in row_terms[r]:
            appear[vv].append((r, sgn))

    # A delta = bb,  bb = -(row_const + A x*)
    Axstar = np.zeros(2 * nF, np.int64)
    for vv, occ in appear.items():
        for (r, sgn) in occ:
            Axstar[r] += sgn * int(xstar[vv])
    bb = -(row_const + Axstar)

    # ---- feasibility: nudge knobs so each component sum (Bx,By) -> 0 ----------
    knob_rows = {}
    for t in range(nF):
        for j in range(3):
            e = int(sys["tri_eid"][t, j])
            C = _bcoeff(sys, balrot, t, j)
            for p in range(2):
                c = int(np.nonzero(C[p])[0][0])
                if e in frozen or (e, c) in set(inv_var[vv] for vv in bad):
                    sgn = int(C[p, c])
                    knob_rows.setdefault((e, c), []).append((2 * t + p, sgn))
    knobs = list(knob_rows.items())
    nudged = {}
    def comp_B():
        return int(bb[0::2].sum()), int(bb[1::2].sum())
    Bx, By = comp_B(); guard = 0
    while (Bx != 0 or By != 0) and knobs and guard < 200000:
        guard += 1; best = None
        for (ec, fp) in knobs:
            for direction in (+1, -1):
                dBx = sum(-s * direction for (r, s) in fp if r % 2 == 0)
                dBy = sum(-s * direction for (r, s) in fp if r % 2 == 1)
                if dBx == 0 and dBy == 0:
                    continue
                cost = abs(Bx + dBx) + abs(By + dBy)
                if cost < abs(Bx) + abs(By) and (best is None or cost < best[0]):
                    best = (cost, ec, direction, fp, dBx, dBy)
        if best is None:
            break
        _, ec, direction, fp, dBx, dBy = best
        for (r, s) in fp:
            bb[r] += -s * direction
        nudged[ec] = nudged.get(ec, 0) + direction
        Bx += dBx; By += dBy
    for (e, c), dv in nudged.items():
        d[e][c] += dv

    Bx, By = comp_B()
    if verbose:
        print(f"    [mcf2] vars={nvar - len(bad)} rows={2*nF} frozen={len(frozen)} "
              f"bad={len(bad)} nudges={sum(abs(v) for v in nudged.values())} "
              f"sum_b=({Bx},{By})")

    # ---- build & solve the flow (var arc rm->rp; saturate s->/->t = consistent)
    var_arcs = []
    okvars = True
    for vv, occ in appear.items():
        rminus = [r for (r, s) in occ if s == -1]
        rplus = [r for (r, s) in occ if s == +1]
        if len(rminus) != 1 or len(rplus) != 1:
            okvars = False
            continue
        var_arcs.append((vv, rminus[0], rplus[0]))

    delta = np.zeros(nvar, np.int64)
    feasible = False
    H = 2
    while H <= max_H:
        G = nx.DiGraph()
        s = ("s",); t = ("t",)
        for (vv, rm, rp) in var_arcs:
            if box_clamp:
                cap_up = max(0, 1 - int(xstar[vv]))     # final x in [-1,1]
                cap_dn = max(0, int(xstar[vv]) + 1)
            elif spread:
                cap_up = cap_dn = 1                     # |delta| <= 1 per edge
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
            feasible = (Csink == 0)
            break
        try:
            flow = nx.max_flow_min_cost(G, s, t)
            flowval = sum(flow.get(s, {}).values())
            feasible = (flowval == Csink)
        except (nx.NetworkXUnfeasible, nx.NetworkXError):
            H += 1
            continue
        for (vv, rm, rp) in var_arcs:
            fpos = flow.get(("r", rm), {}).get(("r", rp), 0)
            fneg = flow.get(("r", rp), {}).get(("r", rm), 0)
            delta[vv] = fpos - fneg
        if feasible or H >= max_H:
            break
        H += 1

    for (e, c), vv in var_id.items():
        if vv in badset:
            continue
        d[e][c] = xstar[vv] + int(delta[vv])

    res, nbad = residual_triangles(sys, d)
    l1 = int(np.abs(d - seed).sum())
    long_before = int((np.abs(seed).max(axis=1) > 1).sum())
    long_after = int((np.abs(d).max(axis=1) > 1).sum())
    stats = {"feasible": feasible, "H": H, "frozen": len(frozen),
             "bad_vars": len(bad), "residual_triangles": nbad,
             "l1_change": l1, "long_before": long_before,
             "long_after": long_after, "sys": sys}
    if verbose:
        print(f"    [mcf2] H={H} feasible={feasible} residual_tris={nbad} "
              f"|drep-seed|1={l1} long {long_before}->{long_after}")
    return d, stats
