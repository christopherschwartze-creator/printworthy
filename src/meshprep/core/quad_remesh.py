"""Unified field-aligned quad-remesh entry point.

A from-scratch, GPL-free (numpy/scipy/networkx/trimesh) quad remesher built on a
globally-optimal 4-RoSy cross-field. ONE function dispatches to the right backend
by intent. See README_retopo.md for the honest accuracy numbers + the ceiling.

    quad_remesh(mesh, target_quads=1500, mode="aligned") -> (V, Q)

modes
-----
"aligned"  (default) : clean + FIELD-ALIGNED (edge loops follow curvature/limbs).
                       Integer-grid extraction (mcf2). Organic val4 ~52-63%,
                       watertight, fidelity <0.006. Best general purpose.
"clean"              : cleanest ISOTROPIC quads (val4 ~85%) when flow doesn't
                       matter (global Blossom tri->quad). Not curvature-following.
"cad"                : "aligned" + a principal-curvature-locked field, for
                       MECHANICAL / developable inputs (cylinders, extrusions) where
                       the field would otherwise be degenerate. Hurts organic shapes.
"fast"               : the plain Instant-Meshes ring-trace (no MCF), a bit faster,
                       slightly noisier than "aligned".
"pyqf"               : OPTIONAL EXTRA. Wraps pyQuadriFlow (QuadriFlow C++ core,
                       same as Blender). val4 ~94-97%, genus-clean, watertight,
                       ~2-3s. NOT installed by default: the prebuilt wheel has
                       an LGPL-Eigen caveat (see THIRD_PARTY_NOTICES.md).
                       Install via the [retopo-pyqf] extra; the permissive
                       from-scratch backends above are the default.

All backends: watertight + manifold, surface-faithful, deterministic (seeded field).
"""


def quad_remesh(mesh, target_quads=1500, mode="aligned", verbose=False):
    """Quad-remesh a triangle mesh. Returns (V (N,3) float, Q (M,4) int). Never
    raises; returns empty arrays on degenerate input."""
    mode = mode.lower()
    if mode in ("aligned", "default"):
        from . import _seamless as sm
        return sm.remesh_mcf2(mesh, target_quads=target_quads, verbose=verbose)
    if mode == "cad":
        from . import _seamless as sm
        return sm.remesh_mcf2(mesh, target_quads=target_quads, curv_field=True,
                              verbose=verbose)
    if mode == "clean":
        from . import _blossom_remesh as br
        return br.blossom_remesh(mesh, target_quads=target_quads, verbose=verbose)
    if mode == "fast":
        from . import _quadflow as qf
        return qf.remesh_im(mesh, target_quads=target_quads, verbose=verbose)
    if mode == "pyqf":
        from . import _pyqf_backend as pq
        return pq.remesh_pyqf(mesh, target_quads=target_quads, verbose=verbose)
    raise ValueError(f"unknown mode {mode!r}; use aligned|clean|cad|fast|pyqf")


def quality(mesh, V, Q):
    """Honest quality report for a remesh result (the metrics that can't be gamed)."""
    from . import _quadflow_eval as ev
    man = ev.manifold_stats(Q)
    vs = ev.valence_stats(Q, len(V))
    fid = ev.two_sided_fidelity(mesh, V, Q)
    out = {"quads": int(len(Q)), "watertight": bool(man["manifold"]),
           "val4_pct": round(vs["pct_val4"], 1),
           "fidelity_rms": round(max(fid["surf2out_rms"], fid["out2surf_rms"]), 4)}
    try:
        out["flow_aligned"] = round(ev.flow_alignment(mesh, V, Q), 2)
    except Exception:
        pass
    return out
