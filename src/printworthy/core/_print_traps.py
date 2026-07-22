"""Resin TRAPPED-VOLUME / suction-cup detection for a given build orientation.

For SLA / DLP / resin (and, more loosely, for any pour-in / drain-out process) the
failure mode is RESIN THAT CANNOT RUN OFF: it pools in upward-facing basins or stays
sealed inside hollow cavities, adding weight, curing into blobs, or creating a
suction cup that rips the part off the plate. This module finds those regions for a
chosen build/gravity direction and suggests where to put a VENT / DRAIN hole.

`find_traps(mesh, build_dir, *, hollow_wall_mm=None) -> dict` reports two kinds of
trap (both geometric, see HONESTY below):

  (a) EXTERNAL upward-facing CUPS  -- concave basins on the OUTER surface whose rim
      sits above their floor, so resin poured on the part collects and cannot drain
      off along -build_dir. Mechanism: surface-voxelize -> fill_holes to get the
      true solid (this leaves an OPEN bowl as air, unlike `.fill()` which seals it)
      -> flood AIR upward from the bottom (-build_dir) exterior; air that the bottom
      flood cannot reach yet sits OVER solid is pooled resin (a basin). This is the
      standard immersion / pour-and-pool test, decided per orientation.

  (b) INTERNAL undrained CAVITIES (only when `hollow_wall_mm` is given, OR the mesh
      already has sealed interior voids) -- enclosed air pockets with no path to the
      outside. Two sub-cases:
        * the mesh ALREADY has sealed voids (e.g. a hollow shell): detected with the
          un-filled surface occupancy + a morphological close (so a thin printed
          wall reads as a solid sheet) + fill_holes; a void that fill_holes closes is
          sealed -> trapped, regardless of orientation (it has no opening at all).
        * `hollow_wall_mm` given: we HOLLOW the solid on the voxel grid (erode by the
          wall thickness, exactly as `_print3d.hollow_shell` does) and report the
          resulting cavity. A freshly hollowed solid has NO vent, so its whole cavity
          is trapped -- that is the point: you must drill a drain. The suggested vent
          is the cavity's lowest point along -build_dir.

Returned per trap: {centroid, volume_cm3, type, kind, ...} + a `vents` list of
suggested DRAIN/VENT hole locations (a point on the part surface where a hole would
let that pocket drain: the rim breach for a cup, the lowest cavity wall point for an
internal void).

`best_drain_orientation(mesh, ...)` scans a handful of axis + hull-normal build
directions and returns the one with the least total trapped+pooled volume (external
cups only -- sealed internal voids are orientation-independent until vented).

`render_traps(mesh, result, out_png)` draws the part (translucent) with pooled /
trapped regions highlighted and the suggested vent markers.

HONESTY (this project has an inflated-claim history -- read literally):
  * This is GEOMETRIC drainage analysis on a COARSE voxel grid, NOT a fluid / CFD
    simulation. "Pooled" / "trapped" volumes are voxel-count volumes at the grid
    pitch (a few % off the true volume from discretization); they assume resin
    behaves as a rigid fill of the basin, ignoring surface tension, viscosity,
    drain-time, and tilt-during-drain. Treat them as where-and-how-much triage, not
    a guaranteed drained-residue.
  * "watertight != genus-correct": thin double walls and thin bores are SEALED by
    surface voxelization (a known trimesh behaviour), so a too-thin drain hole or a
    paper-thin wall can read closed. We require a vent radius >= ~2.5*pitch to be
    seen as open, and we close thin walls deliberately. Vent SUGGESTIONS are
    locations only -- this module does not cut the hole.
  * DOCUMENTED CAPABILITY GAP: any cavity with a wide-enough opening reads as NOT
    trapped, regardless of where the opening points -- an UPWARD-facing vent (which
    would not actually drain in that pose) is not distinguished from a bottom drain.
    Only the vent-down drain orientation is verified (self-test 3). Closing this
    would need a flood FROM the vent opening, not just the part exterior.
  * Volumes/positions are MEASURED off the grid; every heuristic is labelled.

Pure numpy / scipy / trimesh. NO GPL. Imports `_print3d` (for hollow_shell reuse and
the auto-pitch idea) but never edits it. Never raises: every entry point returns a
dict with ok=False + a reason on internal failure. Decimate big meshes (<=6000
faces) and keep the voxel grid coarse (<=~96^3) on a memory-tight box.

Run as a script for the self-tests:  python -m printworthy.core._print_traps
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import trimesh

from ._mesh_util import say

# ---------------------------------------------------------------------------
#  grid resolution budget (mirror force_solid.auto_resolution, but capped LOW
#  for the OOM-tight 16GB box: every heavy step here must stay <2GB / <1s).
# ---------------------------------------------------------------------------
RES_MIN = 40
RES_MAX = 110          # hard cap: a 110^3 bool grid + a few copies stays well <2GB
DEFAULT_RES = 72       # ~72^3 -> sub-second, <0.5GB; the validated working point

# Display cap for the matplotlib 3D backdrop in render_traps (DISPLAY ONLY:
# every mask, marker and number is computed on the full analysis mesh).
_RENDER_MAX_FACES = 6000


def _auto_res(mesh, memory_budget_mb=1200.0):
    """Voxel cells along the longest bbox axis, kept within a SMALL memory budget
    and clamped to [RES_MIN, RES_MAX]. Coarser for bigger meshes (more faces ->
    slower voxelization). This is intentionally lower than force_solid's band
    because trapped-volume detection runs several labelings per call."""
    base = 64.0 + (float(memory_budget_mb) - 1000.0) / 1000.0 * 24.0
    n_faces = max(1, int(getattr(mesh, "faces", np.empty((0, 3))).shape[0]))
    scale = min(1.0, (60000.0 / n_faces) ** 0.25)
    return int(max(RES_MIN, min(RES_MAX, round(base * scale))))


def _pitch_for(mesh, res=None, memory_budget_mb=1200.0):
    res = res or _auto_res(mesh, memory_budget_mb)
    ext = float(np.asarray(mesh.bounding_box.extents).max())
    return ext / max(res, 1), res


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


# ---------------------------------------------------------------------------
#  shared voxel-grid helper: occupancy + an index->world mapper
# ---------------------------------------------------------------------------
class _Grid:
    """Holds a voxelization and converts voxel indices <-> world points using the
    trimesh VoxelGrid transform (verified: indices_to_points = scale*idx + origin)."""

    def __init__(self, vox):
        self.vox = vox
        self.transform = np.asarray(vox.transform, float)
        self.pitch = float(self.transform[0, 0])  # isotropic axis-aligned pitch

    def idx_to_world(self, idx):
        """idx: (k,3) integer voxel indices -> (k,3) world centroids."""
        idx = np.atleast_2d(np.asarray(idx, float))
        return self.vox.indices_to_points(idx)

    def comp_world_centroid(self, mask):
        """World centroid of a boolean voxel mask."""
        ii = np.argwhere(mask)
        if len(ii) == 0:
            return None
        return self.idx_to_world(ii).mean(0)

    def vol_cm3(self, mask):
        return float(np.asarray(mask).sum()) * self.pitch ** 3 / 1000.0


def _drain_axis(build_dir):
    """For a build/up direction, return (axis_index, sign) of its dominant component.
    Resin drains toward -build_dir, i.e. the -axis*sign exterior face."""
    g = _unit(build_dir)
    ax = int(np.argmax(np.abs(g)))
    return ax, float(np.sign(g[ax]) or 1.0)


# ===========================================================================
#  (a) EXTERNAL upward-facing CUPS  (pour-and-pool immersion test)
# ===========================================================================
def external_pools(mesh, build_dir, res=None, memory_budget_mb=1200.0,
                   min_voxels=30, vox=None):
    """Find resin that POOLS in upward-facing basins on the outer surface for build
    direction `build_dir` (resin drains toward -build_dir). Returns
    (pooled_mask, grid, components). HEURISTIC: voxel immersion, not a fluid sim.

    Mechanism (verified): surface-voxelize (do NOT `.fill()` -- that seals an open
    bowl) -> fill_holes to recover the solid while leaving an OPEN basin as air ->
    flood air upward from the bottom (-build_dir) exterior slab -> a voxel of air the
    bottom flood cannot reach, but which has solid somewhere below it in its
    drain-column, is pooled resin held by gravity.

    `vox`: optional precomputed SURFACE voxelization (trimesh VoxelGrid) of
    `mesh` at the pitch `_pitch_for` would pick -- lets `find_traps` voxelize
    ONCE and share it across passes (voxelization dominated the measured cost;
    the masks and numbers are identical because the grid is identical)."""
    try:
        from scipy import ndimage
    except Exception:
        return None, None, []
    try:
        if vox is None:
            pitch, res = _pitch_for(mesh, res, memory_budget_mb)
            vox = mesh.voxelized(pitch=pitch)
        grid = _Grid(vox)
        surf = np.asarray(vox.matrix, bool)
        solid = ndimage.binary_fill_holes(surf)          # open bowl stays air
        air = ~solid

        ax, sign = _drain_axis(build_dir)
        # orient so build-up -> +Z (last axis), drain side at index 0
        solid_w = np.moveaxis(solid, ax, 2)
        air_w = np.moveaxis(air, ax, 2)
        flip = sign < 0
        if flip:
            solid_w = solid_w[:, :, ::-1]
            air_w = air_w[:, :, ::-1]

        # flood air from the bottom (z=0) exterior; reachable air drains away
        lab, _ = ndimage.label(air_w)
        bottom = np.unique(lab[:, :, 0])
        bottom = bottom[bottom > 0]
        drained = np.isin(lab, bottom)
        pooled_w = air_w & ~drained

        # keep only basins: air with SOLID somewhere below it (its drain column is
        # blocked) -- this drops open exterior air sitting above the rim.
        has_below = np.zeros_like(solid_w)
        acc = np.zeros(solid_w.shape[:2], bool)
        for k in range(solid_w.shape[2]):
            acc = acc | solid_w[:, :, k]
            has_below[:, :, k] = acc
        pooled_w = pooled_w & has_below

        # map back to the original index frame
        if flip:
            pooled_w = pooled_w[:, :, ::-1]
        pooled = np.moveaxis(pooled_w, 2, ax)

        comps = _components(pooled, grid, min_voxels, ndimage)
        return pooled, grid, comps
    except Exception:
        return None, None, []


# ===========================================================================
#  (b) INTERNAL sealed cavities
# ===========================================================================
def internal_sealed_voids(mesh, res=None, memory_budget_mb=1200.0, min_voxels=30,
                          close_iters=1, vox=None):
    """Detect SEALED interior air pockets (no opening to the outside) in `mesh` --
    e.g. a hollow shell with no drain hole. Orientation-INDEPENDENT (a sealed void
    traps resin no matter how you tilt it; it needs a vent). Returns
    (void_mask, grid, components).

    Mechanism (verified): un-filled surface occupancy -> binary_close (so a thin
    printed wall reads as a solid sheet, killing the air INSIDE the wall that thin
    surface voxelization would otherwise leave) -> fill_holes; a void that fill_holes
    closes is enclosed -> sealed/trapped. A void that opens to the surface (a real
    drain hole wider than ~2.5*pitch) is NOT closed by fill_holes -> not reported.

    `vox`: optional precomputed SURFACE voxelization shared by the caller
    (see `external_pools`); same grid -> identical masks and numbers."""
    try:
        from scipy import ndimage
    except Exception:
        return None, None, []
    try:
        if vox is None:
            pitch, res = _pitch_for(mesh, res, memory_budget_mb)
            vox = mesh.voxelized(pitch=pitch)
        grid = _Grid(vox)
        surf = np.asarray(vox.matrix, bool)
        closed = ndimage.binary_closing(surf, iterations=close_iters) if close_iters else surf
        filled = ndimage.binary_fill_holes(closed)
        void = filled & ~closed
        comps = _components(void, grid, min_voxels, ndimage)
        return void, grid, comps
    except Exception:
        return None, None, []


def hollowed_cavity(mesh, build_dir, wall_mm, res=None, memory_budget_mb=1200.0,
                    min_voxels=30):
    """HOLLOW `mesh` to a uniform `wall_mm` shell on the voxel grid (same erosion as
    `_print3d.hollow_shell`) and report the resulting interior cavity as a trapped
    internal volume. A freshly hollowed solid has NO vent, so the whole cavity is
    trapped -- the value is telling you where to drill the drain. Returns
    (cavity_mask, grid, components). build_dir only sets which cavity point is
    lowest (the suggested vent); the cavity itself is orientation-independent."""
    try:
        from scipy import ndimage
    except Exception:
        return None, None, []
    try:
        if not bool(getattr(mesh, "is_watertight", False)):
            return None, None, []
        pitch, res = _pitch_for(mesh, res, memory_budget_mb)
        vox = mesh.voxelized(pitch=pitch).fill()
        grid = _Grid(vox)
        solid = np.asarray(vox.matrix, bool)
        n_erode = max(1, int(round(float(wall_mm) / pitch)))
        cavity = ndimage.binary_erosion(solid, iterations=n_erode)
        comps = _components(cavity, grid, min_voxels, ndimage)
        return cavity, grid, comps
    except Exception:
        return None, None, []


# ===========================================================================
#  component extraction + vent suggestion
# ===========================================================================
def _components(mask, grid, min_voxels, ndimage):
    """Label a boolean voxel mask into connected components and summarise each
    (voxel count, world centroid, volume). Drops components below `min_voxels`
    (voxel noise). Returns a list of dicts sorted by volume desc."""
    if mask is None or not mask.any():
        return []
    lab, n = ndimage.label(mask)
    out = []
    for L in range(1, n + 1):
        m = lab == L
        nv = int(m.sum())
        if nv < min_voxels:
            continue
        ii = np.argwhere(m)
        ctr = grid.idx_to_world(ii).mean(0)
        out.append({"_idx": ii, "_mask_label": L, "n_voxels": nv,
                    "volume_cm3": round(grid.vol_cm3(m), 3),
                    "centroid": [round(float(x), 3) for x in ctr]})
    out.sort(key=lambda c: -c["n_voxels"])
    return out


def _vent_for_cup(comp, grid, build_dir):
    """Suggested rim-breach vent for an external pool: the HIGHEST point of the pool
    along build_dir (the rim level) -- drilling a slot there lets the basin overflow
    and drain. Returns {point, rationale}."""
    g = _unit(build_dir)
    pts = grid.idx_to_world(comp["_idx"])
    h = pts @ g
    rim = pts[int(np.argmax(h))]
    return {"point": [round(float(x), 3) for x in rim],
            "type": "rim_breach",
            "rationale": "highest pooled point along build_dir; a slot here lets the "
                         "basin overflow / drain"}


def _vent_for_cavity(comp, grid, build_dir, mesh):
    """Suggested drain vent for an internal cavity: project the cavity's LOWEST point
    along -build_dir straight out to the part surface; a hole there drains the
    cavity. Returns {point=cavity_low, surface_point, rationale}."""
    g = _unit(build_dir)
    pts = grid.idx_to_world(comp["_idx"])
    h = pts @ g
    low = pts[int(np.argmin(h))]
    # ray from the cavity low point along -build_dir to the outer surface
    surf_pt = low
    try:
        inter = (trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
                 if _embree() else mesh.ray)
        locs, _, _ = inter.intersects_location(low[None, :] - 1e-3 * g[None, :],
                                               -g[None, :], multiple_hits=False)
        if len(locs):
            surf_pt = locs[int(np.argmin((locs @ g)))]
    except Exception:
        pass
    return {"point": [round(float(x), 3) for x in low],
            "surface_point": [round(float(x), 3) for x in surf_pt],
            "type": "drain_hole",
            "rationale": "lowest cavity point along -build_dir, projected to the "
                         "surface; drill a drain hole here"}


def _embree():
    try:
        import trimesh.ray.ray_pyembree  # noqa: F401
        return True
    except Exception:
        return False


# ===========================================================================
#  top-level API
# ===========================================================================
def find_traps(mesh, build_dir, *, hollow_wall_mm=None, res=None,
               memory_budget_mb=1200.0, min_volume_cm3=0.02, verbose=False,
               keep_masks=False):
    """Detect resin traps in `mesh` for build/gravity direction `build_dir`.

    Parameters
    ----------
    mesh : trimesh.Trimesh   (set to mm; will be voxelized on a coarse grid)
    build_dir : (3,) up vector. Resin drains toward -build_dir.
    hollow_wall_mm : float or None. If given, ALSO hollow the solid to this wall
        thickness and report the (un-vented) interior cavity as an internal trap.
    res : optional voxel resolution override (cells along longest axis). Default
        auto (capped at RES_MAX=110 for memory).
    min_volume_cm3 : drop traps smaller than this (voxel noise).

    Returns a dict (never raises):
      ok, build_dir, voxel_res, pitch_mm,
      external_cups : [ {centroid, volume_cm3, type='external_cup', n_voxels, vent} ],
      internal_cavities : [ {centroid, volume_cm3, type='internal_cavity',
                             kind='sealed'|'hollowed', n_voxels, vent} ],
      n_traps, total_trapped_cm3, vents (flat list), notes.

    keep_masks : if True, the raw voxel masks + grids ride along under the
      PRIVATE key '_render_cache' so `render_traps` can draw without
      re-voxelizing (the re-voxelization used to double the trap cost). The
      key holds numpy arrays -- NOT JSON-serializable -- so callers that
      persist the result must leave keep_masks False (the default) or pop it.
    """
    result = {"ok": False, "build_dir": [round(float(x), 4) for x in _unit(build_dir)],
              "external_cups": [], "internal_cavities": [], "vents": [],
              "n_traps": 0, "total_trapped_cm3": 0.0, "notes": []}
    if mesh is None or not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        result["notes"].append("empty mesh")
        return result
    try:
        from scipy import ndimage  # noqa: F401
    except Exception:
        result["notes"].append("scipy.ndimage unavailable")
        return result

    min_vox_floor = 8  # absolute label-noise floor

    # ONE surface voxelization shared by both passes below (they used to each
    # voxelize the same mesh at the SAME pitch -- identical grids, so sharing
    # changes nothing but the wall time; voxelization was the measured
    # analyze-stage hotspot). On any failure fall back to per-pass voxelization.
    vox = None
    try:
        pitch, _ = _pitch_for(mesh, res, memory_budget_mb)
        vox = mesh.voxelized(pitch=pitch)
    except Exception:
        vox = None

    # ---- (a) external cups ----
    pooled, gridA, cupsA = external_pools(
        mesh, build_dir, res=res, memory_budget_mb=memory_budget_mb,
        min_voxels=min_vox_floor, vox=vox)
    if gridA is not None:
        result["voxel_res"] = int(round(float(mesh.bounding_box.extents.max()) / gridA.pitch))
        result["pitch_mm"] = round(gridA.pitch, 4)
        for c in cupsA:
            if c["volume_cm3"] < min_volume_cm3:
                continue
            vent = _vent_for_cup(c, gridA, build_dir)
            entry = {"type": "external_cup",
                     "centroid": c["centroid"], "volume_cm3": c["volume_cm3"],
                     "n_voxels": c["n_voxels"], "vent": vent}
            result["external_cups"].append(entry)
            result["vents"].append({"trap": "external_cup", **vent})
    else:
        result["notes"].append("external-cup pass failed")

    # ---- (b) internal cavities ----
    # b1: sealed voids already present (e.g. a hollow shell with no hole)
    voids, gridB, voidsC = internal_sealed_voids(
        mesh, res=res, memory_budget_mb=memory_budget_mb,
        min_voxels=min_vox_floor, vox=vox)
    if gridB is not None:
        if "pitch_mm" not in result:
            result["pitch_mm"] = round(gridB.pitch, 4)
            result["voxel_res"] = int(round(
                float(mesh.bounding_box.extents.max()) / gridB.pitch))
        for c in voidsC:
            if c["volume_cm3"] < min_volume_cm3:
                continue
            vent = _vent_for_cavity(c, gridB, build_dir, mesh)
            entry = {"type": "internal_cavity", "kind": "sealed",
                     "centroid": c["centroid"], "volume_cm3": c["volume_cm3"],
                     "n_voxels": c["n_voxels"], "vent": vent}
            result["internal_cavities"].append(entry)
            result["vents"].append({"trap": "internal_cavity_sealed", **vent})

    # b2: explicit hollowing request
    if hollow_wall_mm is not None:
        cav, gridC, cavC = hollowed_cavity(
            mesh, build_dir, hollow_wall_mm, res=res,
            memory_budget_mb=memory_budget_mb, min_voxels=min_vox_floor)
        if gridC is not None:
            for c in cavC:
                if c["volume_cm3"] < min_volume_cm3:
                    continue
                vent = _vent_for_cavity(c, gridC, build_dir, mesh)
                entry = {"type": "internal_cavity", "kind": "hollowed",
                         "centroid": c["centroid"], "volume_cm3": c["volume_cm3"],
                         "n_voxels": c["n_voxels"], "wall_mm": float(hollow_wall_mm),
                         "vent": vent}
                result["internal_cavities"].append(entry)
                result["vents"].append({"trap": "internal_cavity_hollowed", **vent})
            if not cavC:
                result["notes"].append(
                    "hollow_wall_mm given but no cavity left (wall too thick "
                    "or part too thin to hollow)")
        elif not bool(getattr(mesh, "is_watertight", False)):
            result["notes"].append("hollow_wall_mm requested but mesh not watertight")

    cups = result["external_cups"]
    cavs = result["internal_cavities"]
    result["n_traps"] = len(cups) + len(cavs)
    result["total_trapped_cm3"] = round(
        sum(c["volume_cm3"] for c in cups) + sum(c["volume_cm3"] for c in cavs), 3)
    result["external_cup_cm3"] = round(sum(c["volume_cm3"] for c in cups), 3)
    result["internal_cavity_cm3"] = round(sum(c["volume_cm3"] for c in cavs), 3)
    result["ok"] = True
    result["method"] = "geometric voxel drainage (immersion flood + sealed-void), NOT a fluid sim"
    if keep_masks:
        result["_render_cache"] = {"pooled": pooled, "grid_a": gridA,
                                   "voids": voids, "grid_b": gridB}
    if verbose:
        say(f"  traps: {len(cups)} external cup(s) {result['external_cup_cm3']}cm3, "
              f"{len(cavs)} internal cavity(ies) {result['internal_cavity_cm3']}cm3 "
              f"@ res {result.get('voxel_res')} pitch {result.get('pitch_mm')}mm")
    return result


def best_drain_orientation(mesh, candidates=None, hollow_wall_mm=None,
                           res=None, memory_budget_mb=1200.0, verbose=False):
    """Scan a few build directions and return the one with the LEAST external pooled
    volume (sealed internal voids are orientation-independent, so they are reported
    but do not drive the choice). Returns dict {best_dir, best_external_cm3, scan}.

    `candidates`: list of up-vectors; default = +/- the 3 axes + the convex-hull
    face normals (natural flat rest bases), deduped. Kept small for the budget."""
    if candidates is None:
        cands = [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
        try:
            hn = np.asarray(mesh.convex_hull.face_normals, float)
            # cluster hull normals coarsely so we test a handful of distinct rests
            keep = []
            for c in np.vstack([hn, -hn]):
                c = _unit(c)
                if all(abs(c @ _unit(k)) < 0.94 for k in keep):
                    keep.append(c)
                if len(keep) >= 8:
                    break
            cands = cands + [list(k) for k in keep]
        except Exception:
            pass
        candidates = cands

    scan = []
    best = None
    for bd in candidates:
        r = find_traps(mesh, bd, hollow_wall_mm=hollow_wall_mm, res=res,
                       memory_budget_mb=memory_budget_mb)
        ext_cm3 = r.get("external_cup_cm3", 0.0)
        rec = {"dir": [round(float(x), 3) for x in _unit(bd)],
               "external_cup_cm3": ext_cm3,
               "n_external": len(r.get("external_cups", [])),
               "internal_cavity_cm3": r.get("internal_cavity_cm3", 0.0)}
        scan.append(rec)
        key = (ext_cm3, rec["n_external"])
        if best is None or key < best[0]:
            best = (key, rec)
    scan.sort(key=lambda s: (s["external_cup_cm3"], s["n_external"]))
    out = {"best_dir": best[1]["dir"] if best else [0, 0, 1],
           "best_external_cup_cm3": best[1]["external_cup_cm3"] if best else 0.0,
           "n_candidates": len(candidates),
           "scan": scan[:12]}
    if verbose:
        say(f"  best drain dir {out['best_dir']} -> "
              f"{out['best_external_cup_cm3']}cm3 external pooling "
              f"(scanned {len(candidates)} orientations)")
    return out


# ===========================================================================
#  rendering
# ===========================================================================
def render_traps(mesh, result, out_png, title=None):
    """Draw `mesh` (translucent) laid flat in its build orientation with pooled /
    trapped regions highlighted and suggested vent markers. Reuses the voxel
    masks from `find_traps(..., keep_masks=True)` when present ('_render_cache');
    otherwise recomputes them for display (the summary fields alone are not
    enough to draw). Returns out_png or None (never raises)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except Exception as e:
        say(f"render skipped (matplotlib unavailable): {e}")
        return None
    try:
        bd = np.asarray(result.get("build_dir", [0, 0, 1]), float)
        R = trimesh.geometry.align_vectors(_unit(bd), [0, 0, 1.0])

        def _flat(pts):
            p = trimesh.transform_points(np.atleast_2d(np.asarray(pts, float)), R)
            return p

        b = mesh.copy()
        # DISPLAY-ONLY simplification: the two 3D panels each push every
        # backdrop triangle through matplotlib's painter sort; the backdrop is
        # a uniform translucent silhouette, so drawing it decimated loses no
        # information (all masks/markers/numbers come from the full analysis).
        if len(b.faces) > _RENDER_MAX_FACES:
            try:
                from ._mesh_util import decimate as _decimate
                b = _decimate(b, _RENDER_MAX_FACES)
            except Exception:
                pass
        b.apply_transform(R)
        zmin = float(b.vertices[:, 2].min())
        b.apply_translation([0, 0, -zmin])

        fig = plt.figure(figsize=(13, 6))

        # masks for the visualization: reuse find_traps' own (identical grid,
        # zero extra cost); recompute only when the cache is absent (legacy
        # callers) -- recomputing used to DOUBLE the trap-analysis cost.
        cache = result.get("_render_cache") or {}
        pooled, gridA = cache.get("pooled"), cache.get("grid_a")
        voids, gridB = cache.get("voids"), cache.get("grid_b")
        if pooled is None or gridA is None:
            pooled, gridA, _ = external_pools(mesh, bd)
        if voids is None or gridB is None:
            voids, gridB, _ = internal_sealed_voids(mesh)

        def _draw(ax):
            tris = b.triangles
            pc = Poly3DCollection(tris, facecolor="#bfcad6", edgecolor="none")
            pc.set_alpha(0.16)
            ax.add_collection3d(pc)
            # pooled resin voxels (external cups) as a point cloud
            if pooled is not None and pooled.any() and gridA is not None:
                ii = np.argwhere(pooled)
                if len(ii) > 4000:
                    ii = ii[np.random.default_rng(0).choice(len(ii), 4000, replace=False)]
                pw = _flat(gridA.idx_to_world(ii)); pw[:, 2] -= zmin
                ax.scatter(pw[:, 0], pw[:, 1], pw[:, 2], c="#1f77ff", s=3,
                           alpha=0.5, label="external pool")
            if voids is not None and voids.any() and gridB is not None:
                ii = np.argwhere(voids)
                if len(ii) > 4000:
                    ii = ii[np.random.default_rng(1).choice(len(ii), 4000, replace=False)]
                vw = _flat(gridB.idx_to_world(ii)); vw[:, 2] -= zmin
                ax.scatter(vw[:, 0], vw[:, 1], vw[:, 2], c="#d62728", s=3,
                           alpha=0.5, label="internal cavity")
            # vent markers
            for v in result.get("vents", []):
                p = v.get("surface_point", v.get("point"))
                if p is None:
                    continue
                pf = _flat(np.asarray(p)); pf[:, 2] -= zmin
                ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], marker="v", s=90,
                           c="#00b050", edgecolor="k", linewidths=0.6, zorder=5)
            V = b.vertices
            c = V.mean(0); rr = float(np.abs(V - c).max()) * 1.02
            ax.set_xlim(c[0] - rr, c[0] + rr)
            ax.set_ylim(c[1] - rr, c[1] + rr)
            ax.set_zlim(max(0, c[2] - rr), c[2] + rr)
            try:
                ax.set_box_aspect((1, 1, 1))
            except Exception:
                pass
            ax.set_axis_off()

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        _draw(ax1)
        ax1.view_init(elev=20, azim=-60)
        n_cup = len(result.get("external_cups", []))
        n_cav = len(result.get("internal_cavities", []))
        ax1.set_title(f"{title or 'traps'} (perspective)\n"
                      f"{n_cup} cup {result.get('external_cup_cm3',0)}cm3, "
                      f"{n_cav} cavity {result.get('internal_cavity_cm3',0)}cm3 | "
                      f"green v = suggested vent", fontsize=9)

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        _draw(ax2)
        ax2.view_init(elev=-25, azim=-60)   # look up at the underside
        ax2.set_title("underside (build plate up)\n"
                      f"res {result.get('voxel_res')} pitch {result.get('pitch_mm')}mm "
                      "| geometric drainage, not a fluid sim", fontsize=9)

        # NOTE: tight_layout() and bbox_inches="tight" each force a FULL extra
        # 3D draw of the figure (the projection sort is the render hotspot);
        # fixed margins produce the same picture for one draw instead of three.
        fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04,
                            wspace=0.06)
        plt.savefig(out_png, dpi=115)
        plt.close()
        return out_png
    except Exception as e:
        say(f"render failed: {e}")
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
        return None


# ===========================================================================
#  self-tests + demo
# ===========================================================================
def _make_cup():
    """Open cup = difference of two cylinders (closed floor, open top)."""
    outer = trimesh.creation.cylinder(radius=12, height=30, sections=48)
    inner = trimesh.creation.cylinder(radius=9, height=26, sections=48)
    inner.apply_translation([0, 0, 3])     # floor at bottom, open at top
    cup = outer.difference(inner)
    cup.fix_normals()
    return cup


def _make_hollow_sphere():
    """Sealed hollow sphere = difference of two icospheres (no opening)."""
    o = trimesh.creation.icosphere(radius=12, subdivisions=3)
    i = trimesh.creation.icosphere(radius=8, subdivisions=3)
    hs = o.difference(i)
    hs.fix_normals()
    return hs


def _make_drilled_sphere(radius=3.5):
    """Hollow sphere with a wide drain hole through the bottom shell into the cavity.
    The drill's TOP sits at z=0 (inside the r8 cavity) and runs far below, so the
    boolean produces a genuine through-vent (euler drops 4 -> 2, confirmed).
    NOTE: build the hollow shell WITHOUT an intermediate fix_normals -- doing it
    before the drill boolean makes manifold3d fail to cut the through-vent (the
    cavity stays sealed, euler 4). Fix normals once, at the very end."""
    o = trimesh.creation.icosphere(radius=12, subdivisions=3)
    i = trimesh.creation.icosphere(radius=8, subdivisions=3)
    hs = o.difference(i)                    # no fix_normals here (see note)
    drill = trimesh.creation.cylinder(radius=radius, height=40, sections=32)
    drill.apply_translation([0, 0, -20])   # top at z=0 (in cavity), bottom far below
    d = hs.difference(drill)
    d.fix_normals()
    return d


def run_self_tests(verbose=True):
    """The brief's required self-tests. Returns (all_pass, details)."""
    details = {}
    ok_all = True

    # 1. cup upright -> 1 external cup; flipped -> 0
    cup = _make_cup()
    up = find_traps(cup, [0, 0, 1])
    down = find_traps(cup, [0, 0, -1])
    n_up = len(up["external_cups"])
    n_down = len(down["external_cups"])
    t1 = (n_up == 1 and n_down == 0)
    details["cup_upright_external_cups"] = {"n": n_up, "cm3": up["external_cup_cm3"]}
    details["cup_flipped_external_cups"] = {"n": n_down, "cm3": down["external_cup_cm3"]}
    details["test1_cup_pass"] = bool(t1)
    ok_all &= t1

    # 2. sealed hollow sphere -> 1 internal trapped cavity (any orientation)
    hs = _make_hollow_sphere()
    sealed = find_traps(hs, [0, 0, 1])
    n_seal = len(sealed["internal_cavities"])
    t2 = (n_seal == 1)
    details["sealed_sphere_internal_cavities"] = {
        "n": n_seal, "cm3": sealed["internal_cavity_cm3"]}
    details["test2_sealed_pass"] = bool(t2)
    ok_all &= t2

    # 3. drilled sphere, hole at the DRAIN side (bottom, build +Z) -> 0 internal
    drilled = _make_drilled_sphere()
    drained = find_traps(drilled, [0, 0, 1])
    n_drain = len(drained["internal_cavities"])
    t3 = (n_drain == 0)
    details["drilled_sphere_drain_orientation_internal"] = {
        "n": n_drain, "cm3": drained["internal_cavity_cm3"]}
    details["test3_drilled_drains_pass"] = bool(t3)
    ok_all &= t3

    if verbose:
        print("SELF-TESTS")
        print(f"  1. cup upright cups={n_up} (want 1, {up['external_cup_cm3']}cm3), "
              f"flipped cups={n_down} (want 0)  -> {'PASS' if t1 else 'FAIL'}")
        print(f"  2. sealed hollow sphere internal cavities={n_seal} "
              f"(want 1, {sealed['internal_cavity_cm3']}cm3)  -> {'PASS' if t2 else 'FAIL'}")
        print(f"  3. drilled-bottom sphere @ drain orientation internal={n_drain} "
              f"(want 0)  -> {'PASS' if t3 else 'FAIL'}")
        print(f"  ALL: {'PASS' if ok_all else 'FAIL'}")
    return ok_all, details


if __name__ == "__main__":
    import sys as _sys
    from ._mesh_util import ascii_console
    ascii_console()
    ok, _details = run_self_tests(verbose=True)
    _sys.exit(0 if ok else 1)
