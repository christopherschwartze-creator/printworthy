"""Force-Solid — guaranteed-watertight repair via voxel-seal + reproject.

The outcome modes "watertight" and "print-ready" use this when the normal
detail-preserving repair leaves residual openings that no boundary closer can
seal — chiefly the non-manifold boundary junctions (a boundary vertex on >2
boundary edges) that fan-fill, ear-clip, and manifold3d's merge all leave open.

Idea (a customer's framing): "force an oversized solid voxel, then use the
original frame to subtractively remove the excess." Concretely:

  1. Voxelize the mesh surface onto a grid, morphologically CLOSE small gaps,
     and fill enclosed interior → a binary volume that is watertight by
     construction.
  2. Marching-cubes the volume → a watertight manifold surface (blobby at the
     voxel resolution).
  3. Reproject every output vertex to its nearest point on the ORIGINAL surface
     (KD-tree on a dense surface sampling), but only within a distance
     threshold. Vertices over a real gap have no near original surface, so they
     are NOT snapped — they remain the smooth voxel bridge. Everywhere else the
     surface snaps back ONTO the original, recovering full detail. This is the
     "subtractive removal of excess."
  4. A final manifold-guarantee pass cleans any non-manifold edges the
     reprojection produced.

This rebuilds, with permissive deps only (trimesh + scipy + scikit-image),
what CGAL's (GPL) alpha-wrapping does. Memory is dominated by the voxel
rasterization, so the resolution is auto-tuned to a memory budget — bigger
meshes get a coarser grid (coarser bridges) to stay bounded. Proven on the
wineglass junction-boundary residual: res=112 → watertight, 91%+ of the surface
sits on the original (mean deviation ~0.24% of the bbox diagonal), ~2.4 GB peak.

NOT a default: it changes geometry near the gaps (voxel-smooth bridges), so it
is an OPT-IN outcome the customer chooses when they need a guaranteed solid.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import trimesh

from ._types import RepairAction

# Memory peak is dominated by trimesh.voxelized's rasterization. Empirical peaks
# on a ~200k-face mesh: res 112 → 2.37 GB, 128 → 2.65, 144 → 3.0, 160 → 3.3,
# 192 → 5.0. The auto-tuner is calibrated against these and the mesh's face
# count, then clamped to a safe band.
RES_MIN = 48
RES_MAX = 224
DEFAULT_MEMORY_BUDGET_MB = 2500
_CALIB_FACES = 200_000          # face count the budget→res mapping was fit at
REPROJECT_FRAC = 0.012          # snap radius as a fraction of the bbox diagonal
SEAL_ITERS = 2                  # morphological closing iterations (gap bridging)


def auto_resolution(mesh: trimesh.Trimesh, memory_budget_mb: float) -> int:
    """Pick a voxel resolution (cells along the longest bbox axis) whose
    estimated peak stays within `memory_budget_mb`, scaled down for meshes
    with more faces than the calibration point."""
    # budget → base res: 2000 MB ≈ 96, +32 res per +1000 MB (from the peaks above)
    base = 96.0 + (float(memory_budget_mb) - 2000.0) / 1000.0 * 32.0
    n_faces = max(1, int(getattr(mesh, "faces", np.empty((0, 3))).shape[0]))
    scale = min(1.0, (_CALIB_FACES / n_faces) ** 0.25)   # coarser for bigger meshes
    res = int(round(base * scale))
    return int(max(RES_MIN, min(RES_MAX, res)))


def make_watertight(
    mesh: trimesh.Trimesh,
    *,
    memory_budget_mb: float = DEFAULT_MEMORY_BUDGET_MB,
    reproject_frac: float = REPROJECT_FRAC,
    actions: Optional[list[RepairAction]] = None,
) -> tuple[trimesh.Trimesh, dict]:
    """Return a guaranteed-watertight version of `mesh` (best-effort) plus a
    metrics dict. PURE: the input is never mutated. NEVER raises — on any
    failure (missing optional dep, voxelization error, empty mesh) it returns
    the input unchanged with ``ok=False`` in the metrics and a logged action.

    Run this AFTER the normal detail-preserving repair, on the residual — it
    only needs to bridge what the faithful pass could not close.
    """
    metrics: dict = {"ok": False, "method": "voxel_seal_reproject"}

    def _log(tool: str, extra: dict) -> None:
        if actions is not None:
            m = dict(metrics); m.update(extra)
            actions.append(RepairAction(
                action="force_solid", tool=tool, metrics=m))

    if mesh is None or not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        _log("force_solid:empty", {})
        return mesh, metrics
    if bool(getattr(mesh, "is_watertight", False)):
        metrics.update(ok=True, already_watertight=True)
        _log("force_solid:already_watertight", {})
        return mesh, metrics

    try:
        from scipy import ndimage
        from scipy.spatial import cKDTree
        from skimage import measure
    except Exception as e:  # optional scientific deps
        metrics["error"] = f"missing dep: {type(e).__name__}: {e}"
        _log("force_solid:missing_dep", {})
        return mesh, metrics

    try:
        ext = float((mesh.bounds[1] - mesh.bounds[0]).max())
        diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        if ext <= 0:
            _log("force_solid:degenerate_bounds", {})
            return mesh, metrics
        res = auto_resolution(mesh, memory_budget_mb)
        pitch = ext / res

        # 1. surface voxelization → occupancy grid
        vox = mesh.voxelized(pitch=pitch)
        occ = np.asarray(vox.matrix, dtype=bool)
        # 2. seal small gaps + fill enclosed interior → watertight binary volume
        occ = ndimage.binary_closing(occ, iterations=SEAL_ITERS)
        occ = ndimage.binary_fill_holes(occ)
        # 3. marching cubes on a padded volume → watertight manifold surface
        padded = np.pad(occ.astype(np.float32), 1)
        mv, mf, _, _ = measure.marching_cubes(padded, level=0.5)
        origin = vox.transform[:3, 3] - 0.5 * pitch
        world = (mv - 1.0) * pitch + origin
        solid = trimesh.Trimesh(vertices=world, faces=mf, process=False)
        try:
            solid.update_faces(solid.nondegenerate_faces())
            solid.remove_unreferenced_vertices()
        except Exception:
            pass

        # 4. reproject to the original surface (KD-tree on a dense sampling;
        #    the trimesh BVH closest_point path peaks ~3× the memory).
        n_samples = int(min(400_000, max(50_000, 3 * len(solid.vertices))))
        samp, _ = trimesh.sample.sample_surface(mesh, n_samples)
        samp = np.asarray(samp)
        dist, idx = cKDTree(samp).query(solid.vertices, k=1)
        # Snap radius must cover the voxel quantization: the marching-cubes
        # surface inherently sits ~1 pitch off the original, so a fixed
        # fraction-of-bbox radius fails to snap on coarse grids (pitch >
        # frac*ext). Take the larger of the two so reprojection is
        # resolution-robust; gaps wider than ~2 pitch still won't snap (they
        # stay as the smooth bridge).
        snap_radius = max(reproject_frac * ext, 1.75 * pitch)
        snap = dist < snap_radius
        nv = solid.vertices.copy()
        nv[snap] = samp[idx][snap]
        solid = trimesh.Trimesh(vertices=nv, faces=solid.faces, process=False)
        # light Taubin on the bridge (non-snapped) verts to de-staircase
        try:
            import trimesh.smoothing as sm
            if (~snap).any():
                sm.filter_taubin(solid, lamb=0.5, nu=-0.53, iterations=6)
        except Exception:
            pass

        # 5. finishing manifold-guarantee (cleans non-manifold edges from
        #    marching-cubes + reprojection; res≈128 needs this to be watertight)
        from ._manifold import _manifold_guarantee
        solid, _ = _manifold_guarantee(solid)

        n_snapped = int(snap.sum())
        n_total = int(len(snap))
        mean_dev = float(dist[snap].mean() / diag) if n_snapped else 0.0
        # Genus of the sealed solid. `watertight=True` is NOT enough: the voxel
        # closing can bridge nearby surface sheets into PHANTOM HANDLES (tunnels)
        # that are watertight+manifold yet unprintable — the print lens caught
        # these where val4/watertight checks missed them. Report genus so the
        # defect is visible (we do NOT gate `ok` on it here: a genuinely-handled
        # input, e.g. a mug, is a legitimate genus>0 solid; `genus_clean` lets a
        # caller decide). Grid-stage genus repair is a documented follow-up.
        try:
            euler = int(solid.euler_number)
            genus = (2 - euler) // 2
        except Exception:
            euler = None
            genus = None
        metrics.update(
            ok=bool(solid.is_watertight),
            watertight=bool(solid.is_watertight),
            euler_number=euler,
            genus=genus,
            genus_clean=bool(genus is not None and genus <= 0),
            voxel_res=int(res),
            vertices_in=int(len(mesh.vertices)),
            vertices_out=int(len(solid.vertices)),
            faithful_fraction=round(n_snapped / max(1, n_total), 4),
            mean_surface_deviation=round(mean_dev, 6),
            n_bridge_vertices=int(n_total - n_snapped),
        )
        _log("force_solid:voxel_seal_reproject", {})
        # if still not watertight (extremely rare) we ship it anyway,
        # flagged via metrics["watertight"]/["ok"] = False
        return solid, metrics
    except Exception as e:  # never raise out
        metrics["error"] = f"{type(e).__name__}: {e}"
        _log("force_solid:exception", {})
        return mesh, metrics
