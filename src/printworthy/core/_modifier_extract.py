"""_modifier_extract.py -- turn an FEM stress field into a WATERTIGHT modifier
mesh (the dense-infill core) in WORLD coordinates.

PIPELINE
========
strength_analysis() gives, on a voxel-hex grid, per-element {fos, vm, centroids,
pitch}. We:
  1. rebuild the integer voxel lattice from the centroids (they sit at
     (i+0.5)*pitch + origin), recover origin + grid shape;
  2. form the RELATIVE importance field. Default = von Mises stress normalized to
     [0,1]; 1/FOS is offered as an alternative (both are RELATIVE -- "more material
     where MORE loaded" -- NOT a certified margin; FOS here is comparative);
  3. threshold at a percentile (default 70th) -> binary high-stress occupancy;
  4. keep only voxels that are also SOLID (inside the part) and, optionally, the
     largest connected component (the load path), so we don't scatter speckle;
  5. dilate by 1 voxel (so the marched surface fully ENCLOSES the flagged cells and
     the slicer's inside-test is unambiguous);
  6. marching_cubes on the (padded) binary field at iso=0.5 -> a closed surface,
     scaled by pitch and shifted by origin -> WORLD coords;
  7. clip/verify: the modifier mesh must be watertight and lie INSIDE the base bbox.

Everything here is permissive (numpy / scipy.ndimage / skimage.measure / trimesh).
"""
from __future__ import annotations

import numpy as np


def grid_from_centroids(centroids, pitch):
    """Recover (shape, origin, ijk) from element centroids on a voxel-hex grid.

    centroids : (N,3) world centers, each = (i+0.5)*pitch + origin.
    Returns (shape (3,), origin (3,), ijk (N,3) int) -- the integer voxel index
    of every element, and the dense grid shape that contains them.
    """
    c = np.asarray(centroids, float)
    p = float(pitch)
    cmin = c.min(axis=0)
    # origin = corner of voxel (0,0,0): centroid_min - 0.5*pitch
    origin = cmin - 0.5 * p
    ijk = np.rint((c - origin) / p - 0.5).astype(np.int64)
    ijk -= ijk.min(axis=0)  # shift so min index is 0
    shape = ijk.max(axis=0) + 1
    return tuple(int(s) for s in shape), origin, ijk


def importance_field(fos, vm, mode="vm"):
    """RELATIVE importance per element in [0,1]. mode='vm' (von Mises, default) or
    'inv_fos' (1/FOS). Both are RELATIVE load proxies, NOT certified margins."""
    if mode == "inv_fos":
        f = np.asarray(fos, float)
        f = np.where(np.isfinite(f) & (f > 0), f, np.nanmax(f[np.isfinite(f)]))
        imp = 1.0 / f
    else:
        imp = np.asarray(vm, float)
    imp = np.where(np.isfinite(imp), imp, 0.0)
    lo, hi = float(imp.min()), float(imp.max())
    if hi - lo < 1e-30:
        return np.zeros_like(imp)
    return (imp - lo) / (hi - lo)


def high_stress_occupancy(centroids, pitch, importance, *, percentile=70.0,
                          largest_component=True):
    """Binary voxel grid of the high-importance region.

    Returns (occ_solid (bool 3D), occ_hot (bool 3D), origin (3,), thr (float),
             frac_flagged (float)).
    occ_solid = all elements (the part on the grid); occ_hot = flagged subset.
    """
    from scipy import ndimage

    shape, origin, ijk = grid_from_centroids(centroids, pitch)
    imp = np.asarray(importance, float)
    occ_solid = np.zeros(shape, bool)
    field = np.zeros(shape, float)
    occ_solid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    field[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = imp

    thr = float(np.percentile(imp, percentile))
    occ_hot = occ_solid & (field >= thr)

    if largest_component and occ_hot.any():
        lbl, n = ndimage.label(occ_hot)
        if n > 1:
            sizes = ndimage.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
            keep = int(np.argmax(sizes)) + 1
            occ_hot = (lbl == keep)

    frac = float(occ_hot.sum()) / max(1, int(occ_solid.sum()))
    return occ_solid, occ_hot, origin, thr, frac


def occupancy_to_mesh(occ, pitch, origin, *, dilate=1):
    """Watertight surface of a binary voxel set in WORLD coords via marching cubes.

    dilate : grow the set by N voxels first so the marched iso-surface fully
             ENCLOSES every flagged cell (an unambiguous inside-test for the
             slicer). Returns a trimesh (watertight) or None.
    """
    import trimesh
    from scipy import ndimage
    from skimage import measure

    occ = np.asarray(occ, bool)
    if not occ.any():
        return None
    if dilate > 0:
        occ = ndimage.binary_dilation(occ, iterations=int(dilate))
    # pad so the surface closes on every side (marching cubes needs a 0-border)
    vol = np.pad(occ.astype(np.float32), 1, mode="constant", constant_values=0.0)
    try:
        verts, faces, _normals, _vals = measure.marching_cubes(
            vol, level=0.5, spacing=(pitch, pitch, pitch))
    except Exception:
        return None
    if len(verts) == 0 or len(faces) == 0:
        return None
    # undo the 1-voxel pad (in world units) and shift to world origin
    verts = verts - np.array([pitch, pitch, pitch])
    verts = verts + np.asarray(origin, float)
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    try:
        trimesh.repair.fix_normals(m)
        trimesh.repair.fix_winding(m)
    except Exception:
        pass
    return m


def clip_to_base(mod_mesh, base_mesh):
    """Clamp the modifier to lie strictly INSIDE the base part.

    The voxel marching surface can poke ~half a pitch past the part boundary where
    a flagged voxel touches the surface. A boolean INTERSECTION with the base
    (manifold3d, Apache-2.0) trims the modifier to exactly the part-interior dense
    region -- the slicer only needs the part region the modifier ENCLOSES, so this
    is lossless for targeting while guaranteeing containment. Falls back to a bbox
    clamp if the boolean backend is unavailable/degenerate. Returns a trimesh or
    the input unchanged. Never raises.
    """
    try:
        inter = mod_mesh.intersection(base_mesh)
        if (inter is not None and len(getattr(inter, "faces", [])) > 0
                and abs(inter.volume) > 0):
            return inter
    except Exception:
        pass
    # fallback: clamp vertices into the base bbox (keeps it inside, watertight-safe)
    try:
        bmin, bmax = base_mesh.bounds
        v = np.clip(mod_mesh.vertices, bmin, bmax)
        import trimesh
        return trimesh.Trimesh(vertices=v, faces=mod_mesh.faces, process=True)
    except Exception:
        return mod_mesh


def verify_modifier(mod_mesh, base_mesh):
    """Cheap structural checks: watertight + inside the base bbox + nonzero volume.
    Returns a dict; never raises."""
    out = {"watertight": False, "inside_base_bbox": False,
           "volume_mm3": 0.0, "n_faces": 0}
    try:
        out["watertight"] = bool(mod_mesh.is_watertight)
        out["n_faces"] = int(len(mod_mesh.faces))
        out["volume_mm3"] = float(abs(mod_mesh.volume))
        bmin, bmax = base_mesh.bounds
        mmin, mmax = mod_mesh.bounds
        eps = 1e-6 * float(np.linalg.norm(bmax - bmin) + 1.0)
        out["inside_base_bbox"] = bool(
            np.all(mmin >= bmin - eps) and np.all(mmax <= bmax + eps))
    except Exception:
        pass
    return out
