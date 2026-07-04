"""pyQuadriFlow backend: a wrapper around the MIT-licensed pyQuadriFlow
(Satabol's ctypes binding of hjwdzh/QuadriFlow, the same QuadriFlow used in
Blender) that drops into this remesher as the `pyqf` mode.

WHY THIS EXISTS
---------------
Our from-scratch integer-grid extractor (`_mcf2`) hits the MIQ NP-hard wall:
organic shapes round ~half the lattice edges ambiguously -> ~52-63% val4 and,
worse, occasionally injects spurious genus HANDLES that watertight/manifold
checks miss. QuadriFlow's C++ min-cost-flow + integer-grid solver does the same
job far better (val4 ~94-97%, genus-clean) and is permissively licensed, so it
is the gold-standard backend here when GPL-free is the only hard constraint.

LICENSE (verified on this machine)
----------------------------------
- pyQuadriFlow wrapper: MIT (dist-info/licenses/LICENSE == "MIT").
- Underlying hjwdzh/QuadriFlow: MIT.
- Its deps: Boost (Boost Software License), Eigen (MPL2), pcg32 (Apache-2.0),
  parallel_stable_sort (MIT). All permissive.
  CAVEAT: Eigen's SimplicialCholesky path is LGPL; QuadriFlow ships a
  `BUILD_FREE_LICENSE` flag that swaps it for the MPL2 SparseLU. We link a
  PREBUILT binary (clib/ctypes_QuadriFlow.{dll,so}); we have NOT audited which
  Eigen path that prebuilt blob uses. LGPL (not GPL) is weak-copyleft and is
  generally acceptable for dynamic linking, but if the project's bar is
  "permissive ONLY, zero copyleft", rebuild QuadriFlow from source with
  -DBUILD_FREE_LICENSE=ON to be airtight. This is the only license footnote.

API
---
    remesh_pyqf(mesh, target_quads=1500) -> (V (N,3) float64, Q (M,4) int64)

Returns watertight, manifold quads. Never raises on the happy path used by
quad_remesh; raises only if pyQuadriFlow itself is missing or the C lib errors
(callers in quad_remesh catch nothing -- pyqf is a hard backend, not a fallback).
"""
import os
import sys
import numpy as np
import trimesh


def available():
    """True iff the pyQuadriFlow C library imports cleanly here."""
    try:
        from pyQuadriFlow.pyQuadriFlow import pyquadriflow  # noqa: F401
        return True
    except Exception:
        return False


def _genus(euler):
    return (2 - int(euler)) // 2


def remesh_pyqf(mesh, target_quads=1500, seed=0, preserve_sharp=False,
                preserve_boundary=False, adaptive_scale=False,
                aggressive_sat=False, min_cost_flow=True, return_stats=False,
                verbose=False):
    """Field-aligned quad remesh via pyQuadriFlow (QuadriFlow C++ core).

    mesh         : a trimesh.Trimesh (triangles). We feed its raw verts/faces;
                   QuadriFlow does its own internal cross-field + integer grid.
    target_quads : requested face count. QuadriFlow's `faces` arg is a TARGET;
                   it returns approximately this many quads (typically +-1%).
    min_cost_flow: QuadriFlow's MCF integer-grid solver (its highest-quality
                   mode). On by default -- it is the whole point of using it.

    The other flags map 1:1 to QuadriFlow's CLI flags (sharp/boundary preserve,
    adaptive scale, aggressive SAT). Defaults are the clean-organic preset.

    Returns (V, Q). V float64 (N,3), Q int64 (M,4). The output is QuadriFlow's
    own vertex set (NOT a subset of the input verts) and lies ON QuadriFlow's
    reconstructed surface, which tracks the input to <0.02 bbox RMS in practice
    -- we do NOT re-project (re-projection would fight QF's own smoothing and
    can re-introduce self-intersections). Watertight + manifold by construction.

    With return_stats=True also returns a dict with raw/clean counts, genus, and
    whether QF changed the topology vs the input (QF tends to *repair* genus,
    e.g. it cleans the genus-1 PSB octopus to genus 0).
    """
    from pyQuadriFlow.pyQuadriFlow import pyquadriflow

    mesh = _as_trimesh(mesh)
    in_euler = int(mesh.euler_number)
    verts = np.asarray(mesh.vertices, dtype=np.float64).tolist()
    faces = np.asarray(mesh.faces, dtype=np.int64).tolist()

    res = pyquadriflow(
        int(target_quads), int(seed), verts, faces,
        bool(preserve_sharp), bool(preserve_boundary), bool(adaptive_scale),
        bool(aggressive_sat), bool(min_cost_flow),
    )
    V = np.asarray(res["vertices"], dtype=np.float64)
    Q = np.asarray(res["faces"], dtype=np.int64)

    # QuadriFlow encodes a triangle (rare, at singular spots) as a quad with a
    # repeated last index. Keep the 4-tuple form (the eval/print pipeline already
    # dedups via dict.fromkeys), but record how many so we never hide them.
    n_tri_as_quad = int(sum(1 for q in Q if len(set(int(x) for x in q)) < 4))

    V, Q = _weld(V, Q)

    if return_stats:
        body = _quad_body(V, Q)
        out_euler = int(body.euler_number) if body is not None else None
        stats = {
            "n_verts": int(len(V)),
            "n_quads": int(len(Q)),
            "tri_as_quad": n_tri_as_quad,
            "input_genus": _genus(in_euler),
            "output_genus": _genus(out_euler) if out_euler is not None else None,
            "watertight": bool(body.is_watertight) if body is not None else False,
            "topology_changed_vs_input": (out_euler != in_euler)
            if out_euler is not None else None,
        }
        if verbose:
            print("[pyqf]", stats)
        return V, Q, stats
    return V, Q


def _as_trimesh(mesh):
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    # accept (V, F) too
    try:
        V, F = mesh
        return trimesh.Trimesh(np.asarray(V), np.asarray(F), process=False)
    except Exception as e:
        raise TypeError(f"remesh_pyqf needs a trimesh.Trimesh or (V,F); got "
                        f"{type(mesh)}") from e


def _weld(V, Q, tol=1e-8):
    """Merge numerically-coincident vertices so the quad graph is truly shared
    (watertightness/manifoldness depend on it). QuadriFlow already returns welded
    output in practice, but this makes the contract explicit and idempotent.
    Reindexes Q; drops orphan verts."""
    V = np.asarray(V, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.int64)
    if len(V) == 0 or len(Q) == 0:
        return V, Q
    # round-to-grid weld (stable, deterministic)
    key = np.round(V / tol).astype(np.int64)
    _, inv, first = _unique_rows(key)
    remap = inv
    Vw = V[first]
    Qw = remap[Q]
    # drop verts no quad references, reindex
    used = np.unique(Qw)
    if len(used) != len(Vw):
        relabel = -np.ones(len(Vw), dtype=np.int64)
        relabel[used] = np.arange(len(used))
        Vw = Vw[used]
        Qw = relabel[Qw]
    return Vw, Qw


def _unique_rows(a):
    """np.unique(axis=0) returning (unique, inverse, first_index)."""
    a = np.ascontiguousarray(a)
    view = a.view([("", a.dtype)] * a.shape[1])
    uniq, idx, inv = np.unique(view, return_index=True, return_inverse=True)
    return a[idx], inv.ravel(), idx


def _quad_body(V, Q):
    """Triangulate quads (consistent per-quad diagonal) into a welded trimesh for
    genus/watertight measurement. Matches _print3d.quads_to_trimesh convention."""
    tris = []
    for q in Q:
        u = list(dict.fromkeys(int(x) for x in q))
        if len(u) >= 3:
            tris.append([u[0], u[1], u[2]])
        if len(u) == 4:
            tris.append([u[0], u[2], u[3]])
    if not tris:
        return None
    try:
        body = trimesh.Trimesh(np.asarray(V), np.asarray(tris), process=True)
        body.merge_vertices()
        return body
    except Exception:
        return None
