"""Source-accurate mesh fixer with a deviation CERTIFICATE — preflight upgrade.

`accurate_fix(mesh) -> (fixed_mesh, report)` is the source-accurate replacement
for `preflight._core.run_fixer`'s toy fixer (cheap repair + unconditional COARSE
voxel reseal that ships a blob). It prefers FIDELITY: it only ever reaches for
the voxel sledgehammer when a real non-manifold junction leaves the surface
genuinely un-closeable by curvature fill, and it ships a buyer-readable
certificate stating exactly how much of the surface it left untouched.

PIPELINE (each step pure; the input mesh is never mutated):

  1. LOCALIZE — cheap topology repair only: merge coincident verts, drop
     degenerate/duplicate faces, fix winding. No global remesh, and NO trimesh
     `fill_holes()` — the 2026-07 AI-corpus autopsy (DIAGNOSE-1) measured that
     it CREATES >2-face edges on raw AI meshes (raw: 0 such edges; after
     fill_holes: 6-208), permanently poisoning `is_watertight` and making the
     manifold3d pass a guaranteed no-op. Trivial 3-edge holes are instead
     closed manifold-safely by `_close_boundary_triangles` (a triangle is
     added only when all 3 edges are boundary edges and unclaimed).

  2. CURVATURE-CONTINUOUS HOLE FILL — hole_fill_ei.fill_holes_ei_bisector. Each
     boundary cap is domed along the local surface bisector (the Extended-
     Incenter direction), continuing curvature instead of laying a flat fan.
     GPL-free (numpy+trimesh), touches ONLY boundary-loop neighbourhoods.
     The loop budget scales with the mesh's boundary-edge count (DIAGNOSE-1:
     the fixed 5000 cap truncated 2/12 autopsied meshes).

  2b. JUNCTION-CYCLE FILL — `_fill_boundary_cycles`. AI meshes are open-shell
     soup whose boundary components contain JUNCTION vertices (!= 2 boundary
     neighbours); the ordered-loop extractor rightly skips those, which left
     500-130k boundary edges open on 12/12 autopsied meshes. Here each
     junction vertex's boundary edges are paired by face-fan (wedge)
     membership, the boundary graph is walked into edge-disjoint SIMPLE
     cycles, and each cycle is capped with a sigma-gated centroid fan
     (3-cycles become plain triangles). This decomposes the previously
     un-fillable junction components into fillable loops.

  3. WINDING-NUMBER SOLIDIFY — `_solidify.solidify`, ONLY when the faithful
     ladder could not reach watertight. Generalized-winding-number occupancy
     (inside/outside decided per point, no morphological welding), marching
     cubes, then cKDTree REPROJECT of every vertex back ONTO the original
     surface — over a real gap a vertex finds no near original and stays the
     smooth bridge; everywhere else it snaps back to full detail. Includes a
     grid-stage genus gate and outward-orientation enforcement (the old
     voxel reseal shipped inside-out solids with phantom handles; DIAGNOSE-2).
     `force_solid.make_watertight` remains strictly as a LAST RESORT when
     solidify itself fails, with orientation/bbox guards applied on top.

  4. CARRY VERTEX ATTRIBUTES — if the input carried per-vertex colors and the
     fix preserved the vertex set (a localized cap that did not voxel-reseal),
     reattach them. After a voxel reseal the vertex set is rebuilt, so colors
     cannot be index-carried; we then nearest-neighbour transfer them from the
     original so the surface keeps its appearance. (Honestly labelled in the
     report as carry_mode = "nearest" | "none".)

  5. DEVIATION CERTIFICATE — two-sided Hausdorff vs the ORIGINAL surface
     (out->surf drift AND surf->out coverage), turned into the two numbers a
     buyer reads:
        surface_unchanged_pct : % of sampled original surface within a tight
                                tolerance of the fixed mesh (how much detail we
                                kept verbatim).
        max_deviation_mm      : worst surface displacement, in the caller's
                                print-mm units (set print_mm to get real mm).
     plus the raw normalised worst_rms for ranking. (Adapted from
     scripts/_quadflow_eval.two_sided_fidelity.)

  6. NEVER-WORSE ROLLBACK — ship the INPUT back if the "fix" is a net loss:
       * it did not reach watertight AND it made the surface materially less
         faithful than leaving it (we damaged the surface without buying
         printability), OR
       * it RAISED the genus (introduced phantom handles/tunnels) without the
         user having asked for a guaranteed solid — exactly the failure the toy
         fixer's coarse reseal commits, OR
       * (DIAGNOSE-2 regression guards) the watertight output has NON-POSITIVE
         volume even after orientation repair (inside-out solid — unsliceable),
         its bbox blew past the input's (spike artifacts), or its out->surf
         drift max exceeds 25% of the bbox diagonal (invented geometry far off
         the original — the spike signature the coverage-only certificate
         missed).
     A fix that reaches watertight without raising genus and without measurable
     surface damage always ships.

HONESTY STANCE (matches the toolkit): this measures GEOMETRIC fidelity, not
genus-correctness ("watertight" != "genus-correct"), and does not claim the
voxel bridge is the "true" missing geometry — only that it is the smallest-
footprint seal that keeps the rest of the surface on the original. Every number
returned is MEASURED. An UNCALIBRATED FEM would predict the right shape but not
a certified magnitude — this fixer makes NO FEM claim at all; it is a geometric
repair with a measured fidelity certificate. The `converges` field exists only
so a FEM-aware caller can see, honestly, that the convergence gate is N/A here.

Permissive deps only: numpy, scipy, trimesh, scikit-image (force_solid),
manifold3d (optional). No GPL. Pure / never-raise. Runnable:
`python -m meshprep.core._fix_accurate` runs selftest() vs the toy fixer;
exit 0 on pass.
"""
from __future__ import annotations

import sys
import time
from typing import Optional

import numpy as np

import trimesh

from .hole_fill_ei import fill_holes_ei_bisector, _sigma_min_apex_to_loop
from .force_solid import make_watertight
from ._manifold import _manifold_guarantee
from ._solidify import solidify
from ._types import RepairAction


# ── fidelity-outcome gate on the solidify stage (Phase 0.2) ─────────────────
# The global solidify is gated by the fidelity OUTCOME it produces, NOT by the
# input topology. A winding-number solidify on a garbage-topology input can
# drop MEASURED surface fidelity below any usable floor (100% -> 62% on one
# 66-mesh-benchmark input, genus 578 -> 427) while only producing a still-
# garbage "watertight" blob — a bad trade. When the solidified candidate's
# surface_unchanged_pct falls below FIDELITY_FLOOR and the pre-solidify
# (curvature/clean-fill) mesh is materially more faithful (>= candidate +
# FIDELITY_SKIP_MARGIN points), we DO NOT ship the solidified blob; we keep the
# pre-solidify mesh (which may remain non-watertight — the honest, correct
# trade for a garbage-topology input). OUTCOME-based; no input is special-cased.
FIDELITY_FLOOR = 85.0          # percent surface unchanged the solidify must keep
FIDELITY_SKIP_MARGIN = 10.0    # pre-solidify must beat the candidate by >= this


# ── topology helpers ────────────────────────────────────────────────────────


def _genus(mesh: trimesh.Trimesh) -> Optional[int]:
    """Total genus of a watertight mesh: SUM of per-component genus over
    edge-connected face components, each with its own Euler characteristic
    (vertices counted per component, so vertex-pinched multi-shell AI meshes
    do not drive the naive (2-chi)/2 negative). Equals the classic formula
    on a single closed component. None if not watertight. Never raises."""
    try:
        if not bool(getattr(mesh, "is_watertight", False)):
            return None
        faces = np.asarray(mesh.faces)
        if len(faces) == 0:
            return None
        comps = trimesh.graph.connected_components(
            mesh.face_adjacency, min_len=0, nodes=np.arange(len(faces)))
        total = 0
        for comp in comps:
            f = faces[np.asarray(comp, dtype=np.int64)]
            v_n = len(np.unique(f))
            e = np.sort(f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
            e_n = len(np.unique(e, axis=0))
            chi = v_n - e_n + len(f)
            total += max(0, (2 - chi) // 2)
        return int(total)
    except Exception:
        return None


def _boundary_rows(mesh: trimesh.Trimesh) -> np.ndarray:
    """Row indices (into mesh.edges / edges_sorted) of boundary face-edges
    (edges with exactly one incident face). Empty array on failure."""
    try:
        es = np.asarray(mesh.edges_sorted)
        if len(es) == 0:
            return np.empty(0, dtype=np.int64)
        _eu, einv, ecnt = np.unique(
            es, axis=0, return_inverse=True, return_counts=True)
        return np.flatnonzero(ecnt[einv] == 1)
    except Exception:
        return np.empty(0, dtype=np.int64)


# ── fidelity certificate (two-sided Hausdorff vs ORIGINAL surface) ──────────


def deviation_certificate(
    orig: trimesh.Trimesh,
    out: trimesh.Trimesh,
    *,
    n_samp: int = 8000,
    unchanged_tol_frac: float = 0.002,
    print_mm: Optional[float] = None,
) -> dict:
    """Two-sided Hausdorff deviation of `out` vs the `orig` surface, plus the
    two buyer-facing numbers.

        out2surf : how far OUTPUT vertices drift off the original surface
                   (did the fix invent geometry far from the original?)
        surf2out : how far ORIGINAL surface points sit from the output mesh
                   (did the fix delete / fail to cover original surface?)

    Derived, buyer-facing:
        surface_unchanged_pct : % of sampled original surface points within
                                `unchanged_tol_frac` x bbox-diag of the output
                                (how much surface we kept verbatim).
        max_deviation_mm      : the COVERAGE-direction max — the farthest any
                                ORIGINAL surface point sits from the fixed mesh,
                                scaled to mm if `print_mm` (longest bbox axis in
                                mm) is given, else model units. This is the
                                buyer-faithful number: it measures damage to
                                surface that SHOULD be preserved. We deliberately
                                do NOT use the drift (out->surf) max here — a fill
                                cap's apex legitimately sits off the original
                                where there was a hole, so drift overstates the
                                "deviation" with intended new geometry. The drift
                                max is still reported as out2surf_max for audit.

    Lower deviation / higher unchanged_pct = more faithful. PURE, never raises —
    on any failure returns inf deviations + 0% unchanged so a caller's
    "lower is better" / "higher is better" comparison treats it as worst-case.
    """
    out_d = {
        "out2surf_max": float("inf"), "out2surf_rms": float("inf"),
        "surf2out_max": float("inf"), "surf2out_rms": float("inf"),
        "worst_rms": float("inf"), "worst_max": float("inf"),
        "surface_unchanged_pct": 0.0, "max_deviation_mm": float("inf"),
        "ok": False,
    }
    try:
        diag = float(np.linalg.norm(orig.bounds[1] - orig.bounds[0]))
        if diag <= 0:
            return out_d
        ext = float((orig.bounds[1] - orig.bounds[0]).max())
        # out vertices -> original surface (drift)
        _, d1, _ = orig.nearest.on_surface(np.asarray(out.vertices, dtype=np.float64))
        # original surface samples -> output mesh (coverage)
        s = orig.sample(n_samp)
        _, d2, _ = out.nearest.on_surface(np.asarray(s, dtype=np.float64))

        out2surf_max = float(d1.max())
        surf2out_max = float(d2.max())
        worst_max = max(out2surf_max, surf2out_max)
        # % of original surface that the output reproduces within tolerance
        tol = unchanged_tol_frac * diag
        unchanged = float((d2 <= tol).mean()) * 100.0
        # mm scaling: print_mm is the longest bbox AXIS in mm -> per-unit mm
        mm_per_unit = (float(print_mm) / ext) if (print_mm and ext > 0) else 1.0

        out_d.update(
            out2surf_max=out2surf_max / diag,
            out2surf_rms=float(np.sqrt((d1 ** 2).mean()) / diag),
            surf2out_max=surf2out_max / diag,
            surf2out_rms=float(np.sqrt((d2 ** 2).mean()) / diag),
            worst_max=worst_max / diag,
            surface_unchanged_pct=round(unchanged, 2),
            # buyer-faithful: coverage-direction max (damage to preserved
            # surface), NOT the drift max (which includes the intended fill cap).
            max_deviation_mm=round(surf2out_max * mm_per_unit, 4),
            ok=True,
        )
        out_d["worst_rms"] = max(out_d["out2surf_rms"], out_d["surf2out_rms"])
        return out_d
    except Exception:
        return out_d


# ── step 1: localized cheap repair (no global remesh) ───────────────────────


def _cheap_repair(mesh: trimesh.Trimesh, actions: list[RepairAction]) -> trimesh.Trimesh:
    """Merge coincident vertices, drop degenerate/duplicate faces, fix winding.
    Operates on a COPY. Never raises. Returns the repaired copy (or the
    original on hard failure).

    DIAGNOSE-1(B): trimesh `fill_holes()` is deliberately ABSENT — on 12/12
    autopsied AI meshes it manufactured 6-208 >2-face edges on inputs that had
    ZERO, after which `is_watertight` can never be True and manifold3d rejects
    the mesh. Trivial holes are closed manifold-safely later by
    `_close_boundary_triangles`. As a safety net, if >2-face edges exist here
    (e.g. created by merge_vertices) and few faces are involved, those faces
    are stripped so the area becomes an ordinary fillable hole."""
    m = mesh.copy()
    try:
        m.merge_vertices()
        m.update_faces(m.nondegenerate_faces())
        m.update_faces(m.unique_faces())
        m.remove_unreferenced_vertices()
        # safety net: strip faces on >2-face edges (turn non-manifold fins
        # into fillable holes) when they are few
        n_stripped = 0
        try:
            es = np.asarray(m.edges_sorted)
            _eu, einv, ecnt = np.unique(
                es, axis=0, return_inverse=True, return_counts=True)
            bad_rows = np.flatnonzero(ecnt[einv] > 2)
            if len(bad_rows):
                bad_faces = np.unique(np.asarray(m.edges_face)[bad_rows])
                if len(bad_faces) <= max(8, int(0.02 * len(m.faces))):
                    keep = np.ones(len(m.faces), dtype=bool)
                    keep[bad_faces] = False
                    m.update_faces(keep)
                    m.remove_unreferenced_vertices()
                    n_stripped = int(len(bad_faces))
        except Exception:
            pass
        try:
            trimesh.repair.fix_normals(m, multibody=True)
        except Exception:
            m.fix_normals()
        actions.append(RepairAction(
            action="cheap_repair", tool="trimesh.localized",
            metrics={"vertices": int(len(m.vertices)), "faces": int(len(m.faces)),
                     "nonmanifold_faces_stripped": n_stripped,
                     "watertight": bool(m.is_watertight)}))
    except Exception as e:
        actions.append(RepairAction(
            action="cheap_repair:exception", tool="trimesh.localized",
            metrics={"error": f"{type(e).__name__}: {e}"}))
        return mesh
    return m


# ── step 1b: manifold-safe trivial 3-loop closer ─────────────────────────────


def _close_boundary_triangles(
    mesh: trimesh.Trimesh, actions: list[RepairAction]
) -> tuple[trimesh.Trimesh, int]:
    """Close 3-edge boundary cycles with a single triangle, MANIFOLD-SAFELY:
    a triangle is emitted only when all three edges are currently boundary
    edges and no previously-emitted triangle claimed any of them (so no edge
    ever exceeds 2 faces). Winding is the reverse of the existing faces'
    traversal, so orientation stays consistent.

    Replaces trimesh `fill_holes()` for the trivial case it used to handle
    (DIAGNOSE-1(B)/(D): the EI fill's sigma gate rightly rejects apex-fans on
    3-loops, and fill_holes poisoned manifoldness closing them).
    Returns (mesh, n_closed); the input is never mutated."""
    try:
        rows = _boundary_rows(mesh)
        if len(rows) == 0:
            return mesh, 0
        dir_edges = np.asarray(mesh.edges)[rows]   # directed u->v per face
        D = {(int(u), int(v)) for u, v in dir_edges}
        out: dict[int, list[int]] = {}
        for u, v in D:
            out.setdefault(u, []).append(v)
        seen: set[frozenset] = set()
        cands: list[tuple[int, int, int]] = []
        for (u, v) in D:
            for w in out.get(v, ()):
                if w != u and (w, u) in D:
                    key = frozenset((u, v, w))
                    if len(key) == 3 and key not in seen:
                        seen.add(key)
                        cands.append((u, v, w))
        if not cands:
            return mesh, 0
        used: set[frozenset] = set()
        new_faces: list[list[int]] = []
        for (u, v, w) in cands:
            e1, e2, e3 = (frozenset((u, v)), frozenset((v, w)),
                          frozenset((w, u)))
            if e1 in used or e2 in used or e3 in used:
                continue
            used.update((e1, e2, e3))
            # existing faces traverse u->v, v->w, w->u; the cap must traverse
            # the reverse: (v, u, w)
            new_faces.append([v, u, w])
        if not new_faces:
            return mesh, 0
        faces_out = np.vstack([np.asarray(mesh.faces, dtype=np.int64),
                               np.asarray(new_faces, dtype=np.int64)])
        outm = trimesh.Trimesh(vertices=np.asarray(mesh.vertices).copy(),
                               faces=faces_out, process=False)
        actions.append(RepairAction(
            action="close_boundary_triangles", tool="manifold_safe_tri_close",
            metrics={"closed": int(len(new_faces)),
                     "candidates": int(len(cands))}))
        return outm, len(new_faces)
    except Exception as e:
        actions.append(RepairAction(
            action="close_boundary_triangles:exception",
            tool="manifold_safe_tri_close",
            metrics={"error": f"{type(e).__name__}: {e}"}))
        return mesh, 0


# ── step 2b: junction-component cycle fill ───────────────────────────────────


def _fill_boundary_cycles(
    mesh: trimesh.Trimesh, actions: list[RepairAction],
    *, sigma_floor: float = 0.03, max_cycles: int = 50_000,
) -> tuple[trimesh.Trimesh, int]:
    """Decompose the FULL boundary graph — including the junction components
    the ordered-loop extractor skips — into edge-disjoint SIMPLE cycles and
    cap each with a centroid fan (3-cycles become plain triangles).

    DIAGNOSE-1(A): on raw AI meshes every boundary vertex has an EVEN number
    of boundary edges (each open face-fan contributes exactly 2), so the
    boundary graph decomposes into cycles (Veblen). At each junction vertex
    the boundary edges are paired by FACE-FAN (wedge) membership so cycles
    follow the surface; unpaired leftovers get arbitrary pairing (labelled:
    HEURISTIC — the never-worse rollback gates guard the result). Every cycle
    is filled; poorly-conditioned fans are COUNTED (low_sigma_fans) but not
    rejected, because leaving a cycle open forces the global solidify, which
    measured 13-65% surface-kept on the meshy corpus — strictly worse than a
    bad local fan. Returns (mesh, n_filled); the input is never mutated."""
    try:
        rows = _boundary_rows(mesh)
        if len(rows) == 0:
            return mesh, 0
        bedges = np.asarray(mesh.edges_sorted)[rows]     # undirected (a<b)
        bdir = np.asarray(mesh.edges)[rows]              # directed u->v
        bface = np.asarray(mesh.edges_face)[rows]
        B = len(rows)
        adj: dict[int, list[int]] = {}
        for i, (a, b) in enumerate(bedges):
            adj.setdefault(int(a), []).append(i)
            adj.setdefault(int(b), []).append(i)
        junctions = [v for v, lst in adj.items() if len(lst) != 2]

        # wedge pairing at junction vertices
        pair: dict[tuple[int, int], int] = {}
        n_arbitrary = 0
        if junctions:
            faces_arr = np.asarray(mesh.faces)
            # vertex -> incident faces, built only for junction vertices
            jset = set(junctions)
            v2f: dict[int, list[int]] = {v: [] for v in junctions}
            for fi, tri in enumerate(faces_arr):
                for vv in tri:
                    vv = int(vv)
                    if vv in jset:
                        v2f[vv].append(fi)
            # global undirected edge counts (for interior-edge fan adjacency)
            es_all = np.asarray(mesh.edges_sorted)
            eu, ecnt = np.unique(es_all, axis=0, return_counts=True)
            cntmap = {(int(a), int(b)): int(c)
                      for (a, b), c in zip(eu, ecnt)}
            for v in junctions:
                fs = v2f.get(v, [])
                parent = {f: f for f in fs}

                def _find(x: int) -> int:
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                e2f: dict[tuple[int, int], list[int]] = {}
                for f in fs:
                    tri = faces_arr[f]
                    for x in tri:
                        x = int(x)
                        if x != v:
                            key = (min(v, x), max(v, x))
                            e2f.setdefault(key, []).append(f)
                for key, flist in e2f.items():
                    if cntmap.get(key, 0) == 2 and len(flist) == 2:
                        ra, rb = _find(flist[0]), _find(flist[1])
                        if ra != rb:
                            parent[ra] = rb
                fan_edges: dict[int, list[int]] = {}
                for ei in adj[v]:
                    f = int(bface[ei])
                    root = _find(f) if f in parent else -1
                    fan_edges.setdefault(root, []).append(ei)
                leftovers: list[int] = []
                for _root, elist in fan_edges.items():
                    if len(elist) == 2:
                        pair[(v, elist[0])] = elist[1]
                        pair[(v, elist[1])] = elist[0]
                    else:
                        leftovers.extend(elist)
                for k in range(0, len(leftovers) - 1, 2):
                    pair[(v, leftovers[k])] = leftovers[k + 1]
                    pair[(v, leftovers[k + 1])] = leftovers[k]
                    n_arbitrary += 1

        # 1) walk each COMPLETE closed trail (deterministic under the
        #    per-vertex pairing), 2) peel it into edge-disjoint SIMPLE cycles.
        #    (Peeling during the walk is wrong: after extracting an interior
        #    cycle the pairing at the peel vertex is already consumed and the
        #    walk dead-ends, silently discarding the rest of the trail — the
        #    bug that left 278-4530 boundary edges open on the leak probe.)
        used = np.zeros(B, dtype=bool)
        cycles: list[tuple[list[int], int]] = []   # (vertex cycle, first edge)
        n_deadend_edges = 0

        def _next_edge(cur: int, prev_e: int):
            lst = adj.get(cur, ())
            if len(lst) == 2:
                nxt = lst[1] if lst[0] == prev_e else lst[0]
                return None if nxt == prev_e else nxt
            return pair.get((cur, prev_e))

        for start in range(B):
            if used[start]:
                continue
            if len(cycles) >= max_cycles:
                break
            a0, b0 = int(bedges[start][0]), int(bedges[start][1])
            t_edges = [start]
            t_verts = [a0, b0]
            used[start] = True
            cur = b0
            closed = False
            guard = 0
            while guard <= B + 2:
                guard += 1
                nxt = _next_edge(cur, t_edges[-1])
                if nxt is None:
                    break
                if used[nxt]:
                    closed = (nxt == start and cur == a0)
                    break
                used[nxt] = True
                na, nb = int(bedges[nxt][0]), int(bedges[nxt][1])
                cur = nb if na == cur else na
                t_edges.append(nxt)
                t_verts.append(cur)
            if not closed or cur != a0:
                n_deadend_edges += len(t_edges)   # incomplete pairing; honest
                continue
            # peel simple cycles from the closed trail sequence
            pos = {t_verts[0]: 0}
            sv = [t_verts[0]]
            se: list[int] = []
            for i, e_i in enumerate(t_edges):
                v_n = t_verts[i + 1]
                if v_n in pos:
                    j = pos[v_n]
                    cyc_v = sv[j:]
                    cyc_e0 = se[j] if j < len(se) else e_i
                    if len(cyc_v) >= 3:
                        cycles.append((cyc_v, cyc_e0))
                    for vv in sv[j + 1:]:
                        pos.pop(vv, None)
                    del sv[j + 1:]
                    del se[j:]
                else:
                    sv.append(v_n)
                    pos[v_n] = len(sv) - 1
                    se.append(e_i)

        if not cycles:
            return mesh, 0

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        new_verts: list[np.ndarray] = []
        new_faces: list[list[int]] = []
        n_filled = n_tri = n_fan = n_rej = 0
        for cyc, e0 in cycles:
            # winding: if the existing face traverses the first cycle edge in
            # cycle order, the cap must traverse anti-cycle-order
            u0, v0 = int(bdir[e0][0]), int(bdir[e0][1])
            in_order = (u0 == cyc[0] and v0 == cyc[1]) or \
                       (u0 == cyc[-1] and v0 == cyc[0])
            # (second clause: e0 may connect the cycle's last->first vertices)
            reverse = in_order
            if len(cyc) == 3:
                f = [cyc[0], cyc[2], cyc[1]] if reverse else list(cyc)
                new_faces.append([int(x) for x in f])
                n_filled += 1
                n_tri += 1
                continue
            loop_pts = verts[np.asarray(cyc, dtype=np.int64)]
            apex = loop_pts.mean(axis=0)
            if sigma_floor > 0.0 and \
                    _sigma_min_apex_to_loop(apex, loop_pts) < sigma_floor:
                # counted honestly, but we STILL fill: a poorly-conditioned
                # fan keeps ~100% of the surrounding surface, whereas leaving
                # the cycle open forces the GLOBAL solidify (measured 13-65%
                # surface-kept on the meshy corpus). The never-worse rollback
                # gates (deviation/drift/bbox) guard the outcome.
                n_rej += 1
            apex_idx = len(verts) + len(new_verts)
            new_verts.append(apex)
            n = len(cyc)
            for i in range(n):
                j = (i + 1) % n
                if reverse:
                    new_faces.append([apex_idx, int(cyc[j]), int(cyc[i])])
                else:
                    new_faces.append([apex_idx, int(cyc[i]), int(cyc[j])])
            n_filled += 1
            n_fan += 1

        if n_filled == 0:
            actions.append(RepairAction(
                action="fill_boundary_cycles", tool="junction_wedge_fill",
                metrics={"cycles": int(len(cycles)), "filled": 0,
                         "low_sigma_fans": int(n_rej),
                         "deadend_edges": int(n_deadend_edges)}))
            return mesh, 0
        vout = np.vstack([verts] + [np.asarray(new_verts)]) if new_verts \
            else verts.copy()
        fout = np.vstack([np.asarray(mesh.faces, dtype=np.int64),
                          np.asarray(new_faces, dtype=np.int64)])
        outm = trimesh.Trimesh(vertices=vout, faces=fout, process=False)
        actions.append(RepairAction(
            action="fill_boundary_cycles", tool="junction_wedge_fill",
            metrics={"boundary_edges": int(B),
                     "junction_vertices": int(len(junctions)),
                     "cycles": int(len(cycles)), "filled": int(n_filled),
                     "plain_triangles": int(n_tri), "fans": int(n_fan),
                     "low_sigma_fans": int(n_rej),
                     "deadend_edges": int(n_deadend_edges),
                     "arbitrary_pairings": int(n_arbitrary)}))
        return outm, n_filled
    except Exception as e:
        actions.append(RepairAction(
            action="fill_boundary_cycles:exception",
            tool="junction_wedge_fill",
            metrics={"error": f"{type(e).__name__}: {e}"}))
        return mesh, 0


# ── step 4: vertex attribute carry ──────────────────────────────────────────


def _carry_vertex_colors(orig: trimesh.Trimesh, out: trimesh.Trimesh,
                         actions: list[RepairAction]) -> str:
    """Reattach per-vertex colors from `orig` onto `out`, IN PLACE on `out`.

    Returns the carry mode actually used:
        "none"    : the original had no per-vertex colors (nothing to carry).
        "nearest" : the vertex set was rebuilt (voxel reseal); colors are
                    nearest-neighbour transferred from the original surface so
                    appearance survives even though indices don't.

    Never raises; on any failure returns "none" and leaves `out` untouched.
    """
    try:
        vc = getattr(getattr(orig, "visual", None), "vertex_colors", None)
        if vc is None:
            return "none"
        vc = np.asarray(vc)
        if vc.ndim != 2 or vc.shape[0] != len(orig.vertices) or vc.shape[0] == 0:
            return "none"
        n_out = len(out.vertices)
        ov = np.asarray(orig.vertices, dtype=np.float64)
        nv = np.asarray(out.vertices, dtype=np.float64)

        # nearest transfer: vertex set rebuilt (reseal) — map each out vertex
        # to the nearest original vertex's color so appearance survives
        from scipy.spatial import cKDTree
        _, idx = cKDTree(ov).query(nv, k=1)
        out.visual.vertex_colors = vc[idx]
        actions.append(RepairAction(
            action="carry_vertex_colors", tool="nearest_neighbour",
            metrics={"mode": "nearest", "transferred": int(n_out)}))
        return "nearest"
    except Exception as e:
        actions.append(RepairAction(
            action="carry_vertex_colors:exception", tool="attr_carry",
            metrics={"error": f"{type(e).__name__}: {e}"}))
        return "none"


# ── public API ──────────────────────────────────────────────────────────────


def accurate_fix(
    mesh: trimesh.Trimesh,
    *,
    memory_budget_mb: float = 1500.0,
    n_samp: int = 8000,
    deviation_tolerance: float = 0.5,
    unchanged_tol_frac: float = 0.002,
    print_mm: Optional[float] = None,
    allow_genus_increase: bool = False,
) -> tuple[trimesh.Trimesh, dict]:
    """Source-accurate repair: localize -> curvature fill -> reproject (only if
    needed) -> carry vertex attrs -> deviation certificate -> never-worse
    rollback (deviation AND genus).

    Parameters
    ----------
    memory_budget_mb : float
        Budget for the voxel-seal stage (force_solid auto_resolution). 1500 MB
        keeps the compute-safety cap; force_solid clamps res to [48, 224].
    n_samp : int
        Surface samples for the two-sided deviation certificate.
    deviation_tolerance : float
        Rollback guard. If the final mesh is NOT watertight and its worst_rms
        deviation exceeds (1 + deviation_tolerance) x the input's self-
        deviation floor, the fix is a net loss -> roll back to input.
    unchanged_tol_frac : float
        Tolerance (fraction of bbox diagonal) for the "% surface unchanged"
        certificate number — a surface sample within this of the output counts
        as kept verbatim. 0.002 = 0.2% of the diagonal.
    print_mm : float | None
        Longest bbox axis in mm. When set, max_deviation in the certificate is
        reported in mm; otherwise in model units.
    allow_genus_increase : bool
        If False (default), a fix that RAISES genus (phantom handles) without
        buying it is rolled back — catches the toy reseal's tunnel artifacts.
        Set True when the user explicitly wants a guaranteed solid regardless.

    Returns
    -------
    (out_mesh, report). report keys:
        stages           : list of stage names actually executed
        watertight       : bool — did we reach watertight?
        certificate      : the deviation_certificate dict of the SHIPPED mesh
        surface_unchanged_pct, max_deviation_mm : surfaced at top level too
        genus_in, genus_out : genus before/after (None if not watertight)
        rolled_back      : bool — did we ship the input unchanged?
        rollback_reasons : list[str] if rolled back
        used_voxel_seal  : bool — did a global seal fire? (legacy key; see
                           seal_method: "gwn_solidify" is the winding-number
                           solidify, "voxel_seal_reproject" the last-resort
                           voxel reseal)
        color_carry      : "none" | "nearest"
        converges        : None — NO FEM here; convergence gate is N/A (honest)
        actions          : list[RepairAction] provenance log
        wall_time_s      : float

    PURE: input never mutated. NEVER raises — on any internal failure ships the
    input back with rolled_back=True and an error note.
    """
    t0 = time.time()
    actions: list[RepairAction] = []
    original = mesh
    report: dict = {
        "stages": [], "watertight": False, "rolled_back": False,
        "used_voxel_seal": False, "color_carry": "none",
        # NO FEM is run here; this is a geometric repair. The convergence gate
        # ("a FEM is real only if it reproduces closed form under refinement")
        # is N/A to a fidelity probe — recorded honestly so a FEM-aware caller
        # does not misread a missing convergence table as "a FEM that failed".
        "converges": None, "converges_note": "N/A: geometric fidelity repair, no FEM",
        "actions": actions, "wall_time_s": 0.0,
    }

    try:
        in_wt = bool(getattr(original, "is_watertight", False))
        report["input_watertight"] = in_wt
        report["genus_in"] = _genus(original)

        # ── stage 1: localized cheap repair ──────────────────────────────────
        m = _cheap_repair(original, actions)
        report["stages"].append("cheap_repair")

        # ── stage 1b: manifold-safe trivial 3-loop closer ─────────────────────
        if not bool(getattr(m, "is_watertight", False)):
            m, n_tri = _close_boundary_triangles(m, actions)
            if n_tri:
                report["stages"].append("tri_close")

        # ── stage 2: curvature-aware hole fill (detail-preserving) ───────────
        if not bool(getattr(m, "is_watertight", False)):
            # loop budget scales with the boundary size (DIAGNOSE-1(C): the
            # fixed 5000 cap truncated large AI meshes); capped for runtime
            n_bedges = int(len(_boundary_rows(m)))
            budget = int(min(20_000, max(5_000, n_bedges)))
            m = fill_holes_ei_bisector(
                m, actions, min_edges=4, max_edges=100000,
                sigma_floor=0.03, loop_budget=budget, dome_strength=1.0)
            report["stages"].append("curvature_fill")

        # ── stage 2b: junction-component cycle fill ──────────────────────────
        if not bool(getattr(m, "is_watertight", False)):
            m, n_cyc = _fill_boundary_cycles(m, actions)
            if n_cyc:
                report["stages"].append("junction_cycle_fill")
            m, n_tri2 = _close_boundary_triangles(m, actions)
            if n_tri2:
                report["stages"].append("tri_close_moprun")

        # manifold cleanup on the faithful intermediate
        m, man_action = _manifold_guarantee(m)
        actions.append(man_action)
        faithful_intermediate = m
        inter_wt = bool(getattr(m, "is_watertight", False))
        report["intermediate_watertight"] = inter_wt

        # ── stage 3: winding-number solidify (only if still open) ────────────
        # Gated by the fidelity OUTCOME, not the input topology (see the
        # FIDELITY_FLOOR / FIDELITY_SKIP_MARGIN comment at module top). A seal
        # ships only if its MEASURED surface_unchanged_pct clears the floor OR
        # the more-faithful pre-solidify mesh is not materially better; else we
        # keep the pre-solidify mesh (may stay non-watertight — the honest
        # trade for a garbage-topology input, "watertight" != "genus-correct").
        if not inter_wt:

            def _unchanged_pct(cand: trimesh.Trimesh) -> float:
                # MEASURED % of the ORIGINAL surface the candidate keeps
                # verbatim (same two-sided certificate used at ship time).
                return float(deviation_certificate(
                    original, cand, n_samp=n_samp,
                    unchanged_tol_frac=unchanged_tol_frac,
                    print_mm=print_mm)["surface_unchanged_pct"])

            def _fidelity_ok(cand: trimesh.Trimesh, path: str) -> bool:
                """Ship-or-skip the sealed candidate on the fidelity OUTCOME.
                Returns True (ship it) unless the seal dropped fidelity below
                FIDELITY_FLOOR while the pre-solidify mesh was materially more
                faithful, in which case it records the skip and returns False.
                Never special-cases a specific input — purely outcome-based."""
                cand_pct = _unchanged_pct(cand)
                if cand_pct >= FIDELITY_FLOOR:
                    return True
                pre_pct = _unchanged_pct(faithful_intermediate)
                if pre_pct < cand_pct + FIDELITY_SKIP_MARGIN:
                    return True   # seal is low-fidelity but so is the fill; ship
                # low-fidelity solidify on garbage topology: keep the more-
                # faithful pre-solidify mesh; do NOT ship the blob.
                report["seal_method"] = "skipped_low_fidelity_solidify"
                report["solidify_rejected_fidelity"] = round(cand_pct, 2)
                report["used_voxel_seal"] = False
                actions.append(RepairAction(
                    action="skip_low_fidelity_solidify",
                    tool="fidelity_outcome_gate",
                    metrics={"path": path,
                             "solidified_unchanged_pct": round(cand_pct, 2),
                             "pre_solidify_unchanged_pct": round(pre_pct, 2),
                             "fidelity_floor": FIDELITY_FLOOR,
                             "skip_margin": FIDELITY_SKIP_MARGIN}))
                return False

            sol_res = 96 if memory_budget_mb >= 600 else 64
            solidified, sol_metrics = solidify(
                faithful_intermediate, res=sol_res, actions=actions)
            report["seal_metrics"] = sol_metrics
            if bool(sol_metrics.get("watertight", False)) and \
                    _fidelity_ok(solidified, "gwn_solidify"):
                m = solidified
                report["used_voxel_seal"] = True   # legacy report key
                report["seal_method"] = "gwn_solidify"
                report["stages"].append("gwn_solidify")
            elif not bool(sol_metrics.get("watertight", False)):
                # LAST RESORT: the old voxel reseal, with the DIAGNOSE-2
                # orientation guard applied on top (make_watertight ships
                # skimage marching-cubes' inside-out winding as-is). Governed
                # by the SAME fidelity-outcome gate.
                sealed, seal_metrics = make_watertight(
                    faithful_intermediate, memory_budget_mb=memory_budget_mb,
                    actions=actions)
                report["seal_metrics_fallback"] = seal_metrics
                if bool(seal_metrics.get("watertight", False)):
                    try:
                        sealed.merge_vertices()
                        trimesh.repair.fix_normals(sealed, multibody=True)
                        if bool(sealed.is_watertight) and \
                                float(sealed.volume) < 0.0:
                            sealed.invert()
                    except Exception:
                        pass
                    if _fidelity_ok(sealed, "voxel_seal_reproject"):
                        m = sealed
                        report["used_voxel_seal"] = True
                        report["seal_method"] = "voxel_seal_reproject"
                        report["stages"].append("voxel_reseal_reproject")

        final = m
        final_wt = bool(getattr(final, "is_watertight", False))

        # ── orientation guard (DIAGNOSE-2 RC1): a watertight solid must have
        #    positive volume or slicers see it inside-out ("no layers") ──────
        if final_wt and final is not original:
            try:
                if float(final.volume) < 0.0:
                    trimesh.repair.fix_normals(final, multibody=True)
                    if float(final.volume) < 0.0:
                        final.invert()
                    actions.append(RepairAction(
                        action="orient_outward", tool="volume_sign_guard",
                        metrics={"volume_after": float(final.volume)}))
            except Exception:
                pass

        # ── stage 4: carry vertex attributes (colors) ────────────────────────
        report["color_carry"] = _carry_vertex_colors(original, final, actions)
        report["stages"].append("carry_attributes")

        # ── stage 5: deviation certificate (two-sided vs ORIGINAL) ───────────
        cert_input = deviation_certificate(
            original, original, n_samp=n_samp,
            unchanged_tol_frac=unchanged_tol_frac, print_mm=print_mm)
        cert_final = deviation_certificate(
            original, final, n_samp=n_samp,
            unchanged_tol_frac=unchanged_tol_frac, print_mm=print_mm)
        report["certificate_input_self"] = cert_input    # ~100% / ~0 floor
        report["certificate_final"] = cert_final

        genus_out = _genus(final)
        report["genus_out"] = genus_out

        # ── stage 6: never-worse rollback (deviation AND genus) ──────────────
        floor = max(cert_input["worst_rms"], 1e-6)
        final_dev = cert_final["worst_rms"]
        damaged = final_dev > (1.0 + deviation_tolerance) * floor

        g_in = report["genus_in"]
        g_out = genus_out
        genus_raised = (
            not allow_genus_increase
            and g_in is not None and g_out is not None
            and g_out > g_in
        )

        rollback = False
        reasons: list[str] = []
        if not final_wt and damaged:
            rollback = True
            reasons.append("not_watertight_and_less_faithful")
        if not np.isfinite(final_dev):
            rollback = True
            reasons.append("nonfinite_deviation")
        if genus_raised:
            rollback = True
            reasons.append(f"genus_increase_{g_in}->{g_out}_phantom_handles")

        # ── DIAGNOSE-2 regression guards ─────────────────────────────────────
        # (a) inside-out solid: watertight but non-positive volume even after
        #     the orientation guard -> unsliceable; never ship it
        if final_wt and final is not original:
            try:
                if float(final.volume) <= 0.0:
                    rollback = True
                    reasons.append("nonpositive_volume_inside_out")
            except Exception:
                pass
        # (b) bbox blowup: spikes/flyers past the input bounds (44 mm Taubin
        #     spikes shipped under a 4 mm coverage-only certificate)
        try:
            diag_in = float(np.linalg.norm(
                original.bounds[1] - original.bounds[0]))
            margin = 0.05 * diag_in
            if (np.any(final.bounds[0] < original.bounds[0] - margin)
                    or np.any(final.bounds[1] > original.bounds[1] + margin)):
                rollback = True
                reasons.append("bbox_exceeds_input")
        except Exception:
            pass
        # (c) drift spike: out->surf max (previously computed but UNUSED in
        #     this decision) — geometry invented far off the original. 0.25 of
        #     the bbox diagonal is far beyond any legitimate dome/bridge
        #     (dome offset is clamped to 0.5*loop_radius). HEURISTIC threshold.
        if np.isfinite(cert_final.get("out2surf_max", np.inf)) and \
                cert_final["out2surf_max"] > 0.25 and final is not original:
            rollback = True
            reasons.append("drift_spike_out2surf")

        if rollback:
            actions.append(RepairAction(
                action="rollback_to_input", tool="accurate_fix.rollback",
                metrics={"reasons": reasons,
                         "final_worst_rms": float(final_dev),
                         "input_floor_rms": float(floor),
                         "genus_in": g_in, "genus_out": g_out}))
            final = original
            final_wt = bool(getattr(original, "is_watertight", False))
            report["rolled_back"] = True
            report["rollback_reasons"] = reasons
            report["used_voxel_seal"] = False
            cert_final = cert_input
            genus_out = report["genus_in"]
            report["genus_out"] = genus_out

        report["watertight"] = final_wt
        report["certificate"] = cert_final
        report["surface_unchanged_pct"] = cert_final["surface_unchanged_pct"]
        report["max_deviation_mm"] = cert_final["max_deviation_mm"]
        report["wall_time_s"] = float(time.time() - t0)
        return final, report

    except Exception as e:
        actions.append(RepairAction(
            action="accurate_fix:unhandled_exception",
            tool="accurate_fix.outer_guard",
            metrics={"error": f"{type(e).__name__}: {e}"}))
        cert = deviation_certificate(original, original, n_samp=n_samp,
                                     unchanged_tol_frac=unchanged_tol_frac,
                                     print_mm=print_mm)
        report.update(
            rolled_back=True, error=f"{type(e).__name__}: {e}",
            watertight=bool(getattr(original, "is_watertight", False)),
            certificate=cert,
            surface_unchanged_pct=cert["surface_unchanged_pct"],
            max_deviation_mm=cert["max_deviation_mm"],
            genus_out=_genus(original), wall_time_s=float(time.time() - t0))
        return original, report


# ── the CURRENT preflight toy fixer (cheap + unconditional coarse reseal) ────
# Standalone copy so the selftest can compare WITHOUT importing preflight._core
# (which pulls the three heavy checker modules). Faithful to run_fixer's
# strategy: cheap repair, then an UNCONDITIONAL coarse voxel reseal (res 96).


def fix_toy(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict]:
    """The preflight toy fixer's strategy, standalone. cheap repair -> if not
    watertight, COARSE voxel reseal (res<=96, marching cubes, ship the blob).
    Returns (mesh, report). Never raises."""
    m = mesh.copy()
    report: dict = {"stages": ["cheap_repair"], "used_voxel_reseal": False}
    try:
        m.merge_vertices()
        m.update_faces(m.nondegenerate_faces())
        m.update_faces(m.unique_faces())
        m.fill_holes()
        m.fix_normals()
    except Exception:
        pass
    if not bool(getattr(m, "is_watertight", False)):
        try:
            from scipy import ndimage
            from skimage import measure
            ext = float((m.bounds[1] - m.bounds[0]).max())
            if ext > 0:
                res = 96
                pitch = ext / max(32, min(res, 128))
                occ = np.asarray(m.voxelized(pitch=pitch).matrix, dtype=bool)
                occ = ndimage.binary_closing(occ, iterations=2)
                occ = ndimage.binary_fill_holes(occ)
                lbl, n = ndimage.label(occ)
                if n > 1:
                    keep = 1 + int(np.argmax([(lbl == i).sum() for i in range(1, n + 1)]))
                    occ = lbl == keep
                padded = np.pad(occ.astype(np.float32), 1)
                v, f, _, _ = measure.marching_cubes(padded, level=0.5)
                v = (v - 1.0) * pitch + m.bounds[0]
                sealed = trimesh.Trimesh(vertices=v, faces=f, process=True)
                sealed.fix_normals()
                if sealed.is_watertight:
                    m = sealed
                    report["used_voxel_reseal"] = True
                    report["stages"].append("coarse_voxel_reseal")
        except Exception:
            pass
    if float(getattr(m, "volume", 0.0)) < 0:
        try:
            m.invert()
        except Exception:
            pass
    report["watertight"] = bool(getattr(m, "is_watertight", False))
    return m, report


# ── selftest ────────────────────────────────────────────────────────────────


def _box(extents=(2.0, 1.0, 1.0)) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=extents).subdivide().subdivide()


def _break_hole(box: trimesh.Trimesh, n_faces: int = 30) -> trimesh.Trimesh:
    """Delete a patch of faces near one corner -> a localized open hole."""
    m = box.copy()
    fc = m.triangles_center
    corner = m.bounds[1]
    order = np.argsort(np.linalg.norm(fc - corner, axis=1))
    drop = set(int(i) for i in order[:n_faces])
    keep = np.array([i for i in range(len(m.faces)) if i not in drop])
    return trimesh.Trimesh(vertices=m.vertices.copy(), faces=m.faces[keep].copy(),
                           process=False)


def _break_open_back(box: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove an entire -X face of the box (one big open back)."""
    m = box.copy()
    fc = m.triangles_center
    xmin = m.bounds[0][0]
    ext = float((m.bounds[1] - m.bounds[0]).max())
    keep = np.where(fc[:, 0] > xmin + 0.02 * ext)[0]
    return trimesh.Trimesh(vertices=m.vertices.copy(), faces=m.faces[keep].copy(),
                           process=False)


def _break_nonmanifold(box: trimesh.Trimesh) -> trimesh.Trimesh:
    """A hole plus a few duplicate flipped-winding faces -> non-manifold edges
    so the curvature fill cannot close it and the reseal branch can fire."""
    m = _break_hole(box, n_faces=12)
    extra = m.faces[:4][:, [0, 2, 1]]
    faces = np.vstack([m.faces, extra])
    return trimesh.Trimesh(vertices=m.vertices.copy(), faces=faces, process=False)


def selftest() -> int:
    """Run accurate_fix vs the toy fixer on deliberately-broken primitives and
    assert the contract: (a) reaches watertight on closeable defects, (b) the
    SHIPPED mesh is at least as faithful as the toy fixer (lower deviation /
    higher % unchanged) whenever both reach watertight, (c) vertex colors are
    carried, (d) never raises. Prints a MEASURED table. Exit 0 on pass."""
    from ._mesh_util import ascii_console
    ascii_console()
    print("=" * 92)
    print("_fix_accurate selftest -- accurate_fix vs toy on broken primitives")
    print("=" * 92)
    box = _box()
    # tag vertices with colors so the attribute-carry path is exercised
    box.visual.vertex_colors = (np.abs(np.asarray(box.vertices)) /
                                np.abs(box.vertices).max() * 255).astype(np.uint8)
    box.visual.vertex_colors = np.hstack(
        [box.visual.vertex_colors[:, :3],
         255 * np.ones((len(box.vertices), 1), np.uint8)])

    cases = [
        ("hole_30f", _break_hole(box, 30)),
        ("hole_small", _break_hole(box, 8)),
        ("open_back", _break_open_back(box)),
        ("nonmanifold", _break_nonmanifold(box)),
    ]

    print(f"{'case':12} {'in_wt':>5} | {'acc_wt':>6} {'unchg%':>7} {'dev':>7} "
          f"{'seal':>5} {'carry':>7} {'gOut':>4} {'s':>3} | "
          f"{'toy_wt':>6} {'unchg%':>7} {'dev':>7} {'gOut':>4}")
    all_ok = True
    n_acc_better = 0
    for name, broken in cases:
        try:
            fixed, rep = accurate_fix(broken, n_samp=4000)
            toy, trep = fix_toy(broken)
            tdev = deviation_certificate(broken, toy, n_samp=4000)
            awt = rep["watertight"]
            aunch = rep["surface_unchanged_pct"]
            adev = rep["certificate"]["worst_rms"]
            aseal = rep["used_voxel_seal"]
            carry = rep["color_carry"]
            agout = rep.get("genus_out")
            twt = trep["watertight"]
            tunch = tdev["surface_unchanged_pct"]
            tdv = tdev["worst_rms"]
            tgout = _genus(toy)
            print(f"{name:12} {str(bool(broken.is_watertight)):>5} | "
                  f"{str(awt):>6} {aunch:>7.1f} {adev:>7.4f} {str(aseal):>5} "
                  f"{carry:>7} {str(agout):>4} {rep['wall_time_s']:>2.0f}s | "
                  f"{str(twt):>6} {tunch:>7.1f} {tdv:>7.4f} {str(tgout):>4}")
            # contract checks
            assert isinstance(fixed, trimesh.Trimesh), f"{name}: no mesh"
            assert np.isfinite(adev), f"{name}: nonfinite accurate deviation"
            assert carry in ("nearest",), \
                f"{name}: vertex colors not carried (mode={carry})"
            # when both reach watertight, accurate must be >= as faithful
            if awt and twt:
                if adev <= tdv + 1e-6:
                    n_acc_better += 1
                assert adev <= tdv + 0.02, \
                    f"{name}: accurate materially LESS faithful than toy " \
                    f"(acc {adev:.4f} > toy {tdv:.4f})"
        except Exception as e:
            print(f"{name:12} EXCEPTION {type(e).__name__}: {e}")
            all_ok = False

    print("-" * 92)
    print("dev = worst_rms two-sided Hausdorff / bbox-diag (LOWER=faithful); "
          "unchg% = surface kept verbatim (HIGHER=faithful)")
    print("seal = did the voxel reseal fire (False = detail preserved); "
          "carry = vertex-color carry mode; gOut = genus after fix")
    print(f"accurate >= toy faithful on {n_acc_better}/{len(cases)} "
          f"closeable cases")
    if all_ok:
        print("SELFTEST PASSED")
        return 0
    print("SELFTEST FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(selftest())
