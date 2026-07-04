"""Robust GLOBAL Blossom quad remesher (no per-part seams).

The per-PART Blossom was 'really bad' because of the seams between parts. Run on
the WHOLE mesh at once there are no seams -> a uniform, high-val4, faithful,
watertight quad-dominant mesh. This is the robust deliverable; field alignment
(via _field_quad) can be layered on top later for curvature-following flow.
"""
import numpy as np
from . import _blossom as bl
from . import _quadflow as qf
from . import _grid_place as gp


def blossom_remesh(mesh, target_quads=1200, smooth_iters=12, verbose=False):
    """Clean -> global Blossom tri-to-quad -> enforce manifold+watertight ->
    relax + project to surface. Returns (V, Q). Never raises."""
    try:
        tf = max(800, int(2.0 * target_quads))
        m = qf.clean_mesh(mesh, target_faces=tf)
        h = float(np.sqrt(m.area / max(target_quads, 1)))
        V, Q, info = bl.blossom_quad_patch(m, h)
        if len(Q) == 0:
            return np.zeros((0, 3)), np.zeros((0, 4), np.int64)
        V = np.asarray(V); Q = np.asarray(Q)
        # close to watertight: escalating tolerance-weld of border slits + cap +
        # manifold enforce, until no border edges (the input is closed, so any
        # border is a Blossom artifact, e.g. an odd leftover triangle).
        # close GENTLY: cap holes (no feature loss) + a SMALL-tolerance weld only
        # for tiny slits. The old aggressive 2.0*h weld collapsed thin features
        # (the vase's stepped rims melted to a spike).
        V, Q = gp._enforce_manifold(m, V, Q)
        for tol in (0.25, 0.5):
            if gp._quad_watertight(Q)[0]:
                break
            V, Q, _ = gp._tolerance_weld_boundary(V, Q, tol=tol * h)
            V, Q, _ = gp._cap_border_holes(m, V, Q)
            V, Q = gp._enforce_manifold(m, V, Q)
        V = gp.relax_quads(m, V, Q, iters=smooth_iters, lam=0.5)
        V = gp.project_to_surface(m, V)
        V, Q = gp._enforce_manifold(m, V, Q)
        return V, Q
    except Exception:
        if verbose:
            import traceback; traceback.print_exc()
        return np.zeros((0, 3)), np.zeros((0, 4), np.int64)
