"""Winding-number solidification of arbitrary triangle soup (GPL-free).

`solidify(mesh, res=96)` turns an arbitrary (open, multi-shell, self-
overlapping) triangle soup into ONE watertight solid whose surface sits ON
the original wherever the original had surface, using the GENERALIZED
WINDING NUMBER (GWN) to decide inside/outside instead of the crude
morphological voxel closing that `force_solid.make_watertight` uses.

WHY (DIAGNOSE-1/2, 2026-07 AI-corpus autopsy):
  * `binary_closing(iterations=2)` welds nearby shells into phantom handles
    (measured genus up to 126 on resealed AI meshes) and its auto-resolution
    collapses coverage on large meshes (surface-kept 8-50%).
  * skimage marching-cubes emits INWARD winding under the trimesh/STL
    convention; the old reseal never re-oriented, so 12/12 resealed meshes
    shipped with NEGATIVE volume and 3/12 hard-failed the slicer.
  * trimesh `filter_taubin` on a reprojected (duplicate-vertex) mesh has
    rowsum-deficient Laplacian rows -> a pull toward the WORLD ORIGIN that
    manufactured 16-44 mm spikes. This module never calls it: bridge
    vertices are smoothed with a pinned, affine-correct neighbour average.

PIPELINE
  1. GWN on a regular grid (<= `res` cells on the longest bbox axis; grid
     memory is a few MB). Backend: gpytoolbox.fast_winding_number (MIT
     core) when importable, else a pure numpy/scipy Barnes-Hut octree GWN
     (`_gwn_numpy`) — both MEASURED against each other in selftest().
     Occupancy = |w| >= 0.5 (abs is robust to a globally-inverted shell).
  2. Speckle prune (< 4-voxel components; label: HEURISTIC, sub-0.2 mm dust
     at 60 mm print scale) and a GRID-STAGE GENUS GATE: when the extracted
     surface genus exceeds `genus_limit`, try binary_opening / closing /
     largest-component variants and keep the lowest-genus one that retains
     >= 55% of the inside voxels. Label: HEURISTIC — a genuinely handled
     object (mug, donut) keeps its handles because opening cannot remove a
     >= 2-voxel-thick topological handle without violating the retention
     floor; measured, not guaranteed.
  3. Marching cubes -> triangle surface, then ORIENT OUTWARD (fix_normals +
     invert while volume < 0) — the guard the old reseal lost.
  4. cKDTree REPROJECT of every output vertex onto a dense sampling of the
     ORIGINAL surface within snap_radius = max(0.012*ext, 1.75*pitch) — the
     proven force_solid recipe/constants. Vertices over a real gap stay the
     smooth voxel bridge; everything else snaps back onto the original.
  5. Pinned bridge smoothing (non-snapped verts only, neighbour mean, no
     origin pull), duplicate-vertex merge (kills the zero-length edges the
     snap creates), manifold3d pass, final outward-orientation enforce.

Returns (mesh, metrics) with metrics = {method, backend, on_surface_pct,
genus, watertight, volume, ...}. PURE (input never mutated), never raises.
`on_surface_pct` is MEASURED post-hoc: % of final vertices within
0.002 x bbox-diag of the sampled original surface.

Permissive deps only: numpy, scipy, trimesh, scikit-image, optionally
gpytoolbox (MIT) and manifold3d. `gpytoolbox.copyleft` is never imported.
Runnable: `python -m meshprep.core._solidify` -> selftest(), exit 0 on pass.
"""
from __future__ import annotations

import sys
import time
from typing import Optional

import numpy as np
import trimesh

from ._types import RepairAction

# force_solid's proven reprojection constants (kept in sync by value; the
# recipe is shared, the implementations differ in the smoothing + orientation
# fixes documented above).
REPROJECT_FRAC = 0.012      # snap radius as fraction of longest bbox axis
SNAP_PITCH_MULT = 1.75      # ...but at least this many voxel pitches
RES_MIN, RES_MAX = 32, 128
MAX_GRID_CELLS = 2_600_000  # hard cap on grid points (memory guard)
GWN_SOURCE_FACE_CAP = 150_000   # decimate the GWN source above this
SPECKLE_MIN_VOXELS = 4      # HEURISTIC: drop inside-components smaller than this
GENUS_RETENTION_FLOOR = 0.55


# ── generalized winding number backends ─────────────────────────────────────


def _gwn_gpytoolbox(Q: np.ndarray, V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """gpytoolbox (MIT) fast winding number. Raises if unavailable."""
    import gpytoolbox  # MIT core; .copyleft is never touched (license_guard)
    return np.asarray(gpytoolbox.fast_winding_number(
        np.ascontiguousarray(Q, dtype=np.float64),
        np.ascontiguousarray(V, dtype=np.float64),
        np.ascontiguousarray(F, dtype=np.int32)))


def _gwn_exact_chunk(q: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Exact GWN of query block `q` (m,3) vs triangles T (f,3,3).
    van Oosterom–Strackee solid angle, vectorized over the (m,f) block."""
    a = T[None, :, 0, :] - q[:, None, :]
    b = T[None, :, 1, :] - q[:, None, :]
    c = T[None, :, 2, :] - q[:, None, :]
    la = np.linalg.norm(a, axis=2)
    lb = np.linalg.norm(b, axis=2)
    lc = np.linalg.norm(c, axis=2)
    det = np.einsum("mfi,mfi->mf", a, np.cross(b, c))
    denom = (la * lb * lc
             + np.einsum("mfi,mfi->mf", a, b) * lc
             + np.einsum("mfi,mfi->mf", b, c) * la
             + np.einsum("mfi,mfi->mf", c, a) * lb)
    return np.arctan2(det, denom).sum(axis=1) / (2.0 * np.pi)


class _BHNode:
    __slots__ = ("com", "dipole", "radius", "children", "faces")

    def __init__(self):
        self.com = None       # area-weighted center of member faces
        self.dipole = None    # sum of face area-vectors (0.5 * cross)
        self.radius = 0.0     # bound: com -> farthest member-face vertex
        self.children = None  # list[_BHNode] | None
        self.faces = None     # np.ndarray face indices (leaf only)


def _bh_build(cent: np.ndarray, area_vec: np.ndarray, reach: np.ndarray,
              idx: np.ndarray, leaf: int = 64, depth: int = 0) -> _BHNode:
    node = _BHNode()
    a = np.linalg.norm(area_vec[idx], axis=1)
    wsum = a.sum()
    node.com = (cent[idx] * a[:, None]).sum(axis=0) / wsum if wsum > 0 \
        else cent[idx].mean(axis=0)
    node.dipole = area_vec[idx].sum(axis=0)
    node.radius = float((np.linalg.norm(cent[idx] - node.com, axis=1)
                         + reach[idx]).max())
    if len(idx) <= leaf or depth >= 12:
        node.faces = idx
        return node
    mid = cent[idx].mean(axis=0)
    node.children = []
    side = cent[idx] > mid  # (n,3) bool
    codes = side[:, 0].astype(np.int8) * 4 + side[:, 1] * 2 + side[:, 2]
    for o in range(8):
        sub = idx[codes == o]
        if len(sub):
            node.children.append(
                _bh_build(cent, area_vec, reach, sub, leaf, depth + 1))
    if len(node.children) == 1:       # degenerate split -> make it a leaf
        node.children = None
        node.faces = idx
    return node


def _bh_eval(node: _BHNode, Q: np.ndarray, qidx: np.ndarray, w: np.ndarray,
             T: np.ndarray, beta: float = 2.0) -> None:
    """Node-major Barnes-Hut traversal: dipole for far queries, exact leaves."""
    if len(qidx) == 0:
        return
    d = Q[qidx] - node.com[None, :]
    dist = np.linalg.norm(d, axis=1)
    far = dist > (beta * node.radius + 1e-12)
    if far.any():
        fi = qidx[far]
        dd = d[far]
        r3 = np.maximum(dist[far], 1e-30) ** 3
        # solid angle of the aggregated patch: A . (com - q) / |com - q|^3
        w[fi] += (-(dd @ node.dipole)) / (4.0 * np.pi * r3)
    near = qidx[~far]
    if len(near) == 0:
        return
    if node.children is None:
        # exact evaluation, chunked so the (m,f) block stays bounded
        Tf = T[node.faces]
        step = max(1, int(4_000_000 // max(1, len(node.faces))))
        for i0 in range(0, len(near), step):
            blk = near[i0:i0 + step]
            w[blk] += _gwn_exact_chunk(Q[blk], Tf)
    else:
        for ch in node.children:
            _bh_eval(ch, Q, near, w, T, beta)


def _gwn_numpy(Q: np.ndarray, V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Pure numpy Barnes-Hut generalized winding number (fallback backend).
    Accuracy: dipole far-field at beta=2 — selftest() measures it against
    gpytoolbox and requires >= 99% inside/outside agreement on a sphere."""
    T = V[F].astype(np.float64)
    cent = T.mean(axis=1)
    area_vec = 0.5 * np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    reach = np.linalg.norm(T - cent[:, None, :], axis=2).max(axis=1)
    root = _bh_build(cent, area_vec, reach, np.arange(len(F)))
    w = np.zeros(len(Q), dtype=np.float64)
    _bh_eval(root, np.asarray(Q, dtype=np.float64), np.arange(len(Q)), w, T)
    return w


def _winding_grid(mesh: trimesh.Trimesh, res: int
                  ) -> tuple[np.ndarray, tuple, np.ndarray, float, str]:
    """GWN sampled on a regular grid over the padded bbox.
    Returns (w flat, grid shape, origin, pitch, backend)."""
    lo, hi = mesh.bounds
    ext_axes = hi - lo
    ext = float(ext_axes.max())
    pitch = ext / float(res)
    pad = 3  # cells of empty margin so marching cubes closes cleanly
    n = np.maximum(2, np.ceil(ext_axes / pitch).astype(int) + 1 + 2 * pad)
    while int(n.prod()) > MAX_GRID_CELLS:
        pitch *= 1.26
        n = np.maximum(2, np.ceil(ext_axes / pitch).astype(int) + 1 + 2 * pad)
    origin = lo - pad * pitch
    ax = [origin[i] + np.arange(n[i]) * pitch for i in range(3)]
    Q = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces)
    try:
        w = _gwn_gpytoolbox(Q, V, F)
        backend = "gpytoolbox"
    except Exception:
        w = _gwn_numpy(Q, V, F)
        backend = "numpy_octree"
    return w, tuple(int(x) for x in n), origin, pitch, backend


# ── occupancy -> surface helpers ─────────────────────────────────────────────


def _mc_surface(occ: np.ndarray, origin: np.ndarray, pitch: float
                ) -> Optional[trimesh.Trimesh]:
    """Marching cubes on a padded occupancy grid -> trimesh in world coords.
    NOTE: skimage MC winding is INWARD under the trimesh convention; the
    caller must run _orient_outward before trusting volume/normals."""
    try:
        from skimage import measure
        padded = np.pad(occ.astype(np.float32), 1)
        mv, mf, _, _ = measure.marching_cubes(padded, level=0.5)
        world = (mv - 1.0) * pitch + origin
        m = trimesh.Trimesh(vertices=world, faces=mf, process=False)
        m.update_faces(m.nondegenerate_faces())
        m.remove_unreferenced_vertices()
        return m if len(m.faces) else None
    except Exception:
        return None


def _genus_component_aware(mesh: trimesh.Trimesh) -> Optional[int]:
    """Total genus: sum of per-component genus over edge-connected face
    components, each with its own Euler characteristic (robust to
    vertex-pinched multi-shell surfaces, where a global (2-chi)/2 or a
    body_count-based correction goes negative). None on failure."""
    try:
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


def _orient_outward(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Make winding outward: fix_normals (multibody) then invert while a
    watertight mesh reports negative volume. In place; returns m."""
    try:
        trimesh.repair.fix_normals(m, multibody=True)
    except Exception:
        pass
    try:
        if bool(m.is_watertight) and float(m.volume) < 0.0:
            m.invert()
    except Exception:
        pass
    return m


def _prune_speckle(occ: np.ndarray, min_vox: int = SPECKLE_MIN_VOXELS
                   ) -> np.ndarray:
    """Drop inside-components smaller than min_vox voxels (HEURISTIC:
    sub-0.2 mm dust at 60 mm / res 96 — unprintable at any setting)."""
    try:
        from scipy import ndimage
        lbl, ncomp = ndimage.label(occ)
        if ncomp <= 1:
            return occ
        sizes = np.bincount(lbl.ravel())
        keep = sizes >= min_vox
        keep[0] = False
        return keep[lbl]
    except Exception:
        return occ


def _genus_of_occ(occ: np.ndarray, origin: np.ndarray, pitch: float
                  ) -> tuple[Optional[int], Optional[trimesh.Trimesh]]:
    m = _mc_surface(occ, origin, pitch)
    if m is None:
        return None, None
    return _genus_component_aware(m), m


def _genus_gate_grid(occ: np.ndarray, origin: np.ndarray, pitch: float,
                     genus_limit: int) -> tuple[np.ndarray, dict]:
    """GRID-STAGE genus gate: when the marching-cubes surface genus exceeds
    `genus_limit`, try opening / closing / largest-component variants and
    keep the lowest-genus candidate that retains >= 55% of inside voxels.
    Returns (occupancy, info). HEURISTIC — labelled in the metrics."""
    info: dict = {"fired": False}
    g0, _ = _genus_of_occ(occ, origin, pitch)
    info["genus_raw_grid"] = g0
    if g0 is None or g0 <= genus_limit:
        return occ, info
    try:
        from scipy import ndimage
    except Exception:
        return occ, info
    n0 = int(occ.sum())
    if n0 == 0:
        return occ, info

    def largest(o: np.ndarray) -> np.ndarray:
        lbl, ncomp = ndimage.label(o)
        if ncomp <= 1:
            return o
        sizes = np.bincount(lbl.ravel()); sizes[0] = 0
        return lbl == int(sizes.argmax())

    candidates = [
        ("close1", ndimage.binary_closing(occ, iterations=1)),
        ("open1", ndimage.binary_opening(occ, iterations=1)),
        ("open1_largest", largest(ndimage.binary_opening(occ, iterations=1))),
    ]
    best = None  # (genus, name, occ, retention)
    for name, cand in candidates:
        ret = float(cand.sum()) / n0
        if ret < GENUS_RETENTION_FLOOR:
            continue
        g, _ = _genus_of_occ(cand, origin, pitch)
        if g is None:
            continue
        if best is None or g < best[0]:
            best = (g, name, cand, ret)
    if best is not None and best[0] < g0 and (
            best[0] <= genus_limit or best[0] <= g0 // 2):
        info.update(fired=True, variant=best[1], genus_before=int(g0),
                    genus_after=int(best[0]), retention=round(best[3], 3),
                    note="HEURISTIC grid-stage genus gate")
        return best[2], info
    info["note"] = "gate evaluated, no candidate accepted"
    return occ, info


# ── public API ───────────────────────────────────────────────────────────────


def solidify(
    mesh: trimesh.Trimesh,
    res: int = 96,
    *,
    reproject_frac: float = REPROJECT_FRAC,
    genus_limit: int = 2,
    smooth_bridge_iters: int = 6,
    actions: Optional[list[RepairAction]] = None,
) -> tuple[trimesh.Trimesh, dict]:
    """Robust inside/outside solidification of arbitrary triangle soup via
    the generalized winding number. See module docstring for the pipeline.

    Parameters
    ----------
    res : int
        Grid cells along the longest bbox axis (clamped to [32, 128]; total
        cells hard-capped at ~2.6M -> a few MB, never a memory risk).
    reproject_frac : float
        Snap radius as a fraction of the longest bbox axis (force_solid's
        0.012), floored at 1.75 x pitch so coarse grids still snap.
    genus_limit : int
        Grid-stage genus gate threshold (see _genus_gate_grid). A genuinely
        handled solid (mug=1, donut=1) sits below the default 2.
    smooth_bridge_iters : int
        Pinned neighbour-mean smoothing iterations on NON-snapped (bridge)
        vertices only. Snapped vertices never move (no shrink, no origin
        pull — replaces the filter_taubin call that made 44 mm spikes).

    Returns
    -------
    (mesh, metrics) — metrics always has {method, ok, watertight}; on
    success also {backend, res_effective, pitch, on_surface_pct, genus,
    volume, n_bridge_vertices, genus_gate, ...}. PURE, NEVER raises: on any
    failure returns the input unchanged with ok=False.
    """
    t0 = time.time()
    metrics: dict = {"method": "gwn_solidify", "ok": False, "watertight": False}

    def _log(tool: str) -> None:
        if actions is not None:
            actions.append(RepairAction(
                action="solidify", tool=tool, metrics=dict(metrics)))

    if mesh is None or not hasattr(mesh, "vertices") or len(mesh.vertices) == 0 \
            or len(getattr(mesh, "faces", [])) == 0:
        metrics["error"] = "empty mesh"
        _log("solidify:empty")
        return mesh, metrics
    try:
        lo, hi = mesh.bounds
        ext = float((hi - lo).max())
        diag = float(np.linalg.norm(hi - lo))
        if not np.isfinite(ext) or ext <= 0:
            metrics["error"] = "degenerate bounds"
            _log("solidify:degenerate_bounds")
            return mesh, metrics

        res = int(max(RES_MIN, min(RES_MAX, int(res))))

        # GWN source: decimate a COPY above the face cap (res never scales
        # down with face count — that was the old reseal's coverage killer).
        src = mesh
        if len(mesh.faces) > GWN_SOURCE_FACE_CAP:
            try:
                src = mesh.simplify_quadric_decimation(
                    face_count=GWN_SOURCE_FACE_CAP)
                metrics["gwn_source_decimated_to"] = int(len(src.faces))
            except Exception:
                src = mesh

        # 1. winding number grid -> occupancy
        w, shape, origin, pitch, backend = _winding_grid(src, res)
        metrics.update(backend=backend, res_effective=int(round(ext / pitch)),
                       pitch=float(pitch))
        occ = (np.abs(w) >= 0.5).reshape(shape)
        thr = 0.5
        if not occ.any():
            thr = 0.3   # heavily-open shell: fractional winding inside
            occ = (np.abs(w) >= thr).reshape(shape)
        metrics["occupancy_threshold"] = thr
        if not occ.any():
            metrics["error"] = "no enclosed volume at |w|>=0.3"
            _log("solidify:no_volume")
            return mesh, metrics
        occ = _prune_speckle(occ)

        # 2. grid-stage genus gate
        occ, gate = _genus_gate_grid(occ, origin, pitch, genus_limit)
        metrics["genus_gate"] = gate

        # 3. marching cubes + outward orientation
        solid = _mc_surface(occ, origin, pitch)
        if solid is None or len(solid.faces) == 0:
            metrics["error"] = "marching cubes produced no surface"
            _log("solidify:mc_empty")
            return mesh, metrics
        solid = _orient_outward(solid)

        # 4. reproject onto the ORIGINAL (full-res) surface
        from scipy.spatial import cKDTree
        n_samples = int(min(400_000, max(50_000, 3 * len(solid.vertices))))
        samp, _fid = trimesh.sample.sample_surface(mesh, n_samples)
        samp = np.asarray(samp, dtype=np.float64)
        tree = cKDTree(samp)
        dist, idx = tree.query(np.asarray(solid.vertices, dtype=np.float64), k=1)
        snap_radius = max(reproject_frac * ext, SNAP_PITCH_MULT * pitch)
        snap = dist < snap_radius
        nv = np.asarray(solid.vertices, dtype=np.float64).copy()
        nv[snap] = samp[idx[snap]]

        # 5. pinned bridge smoothing (affine-correct neighbour mean; snapped
        #    verts are pinned -> no shrink, no origin pull)
        bridge = ~snap
        if bridge.any() and smooth_bridge_iters > 0:
            e = np.asarray(solid.edges_unique)
            for _ in range(int(smooth_bridge_iters)):
                acc = np.zeros_like(nv)
                cnt = np.zeros(len(nv))
                np.add.at(acc, e[:, 0], nv[e[:, 1]])
                np.add.at(acc, e[:, 1], nv[e[:, 0]])
                np.add.at(cnt, e[:, 0], 1.0)
                np.add.at(cnt, e[:, 1], 1.0)
                good = cnt > 0
                mean = nv.copy()
                mean[good] = acc[good] / cnt[good, None]
                nv[bridge] = 0.5 * nv[bridge] + 0.5 * mean[bridge]

        solid = trimesh.Trimesh(vertices=nv, faces=solid.faces, process=False)
        # merge the duplicate positions the snap creates (they poisoned the
        # old reseal's smoother with zero-length edges)
        solid.merge_vertices()
        solid.update_faces(solid.nondegenerate_faces())
        solid.remove_unreferenced_vertices()

        # 6. manifold pass + final outward orientation
        try:
            from ._manifold import _manifold_guarantee
            solid, man_act = _manifold_guarantee(solid)
            if actions is not None:
                actions.append(man_act)
        except Exception:
            pass
        solid = _orient_outward(solid)

        wt = bool(solid.is_watertight)
        vol = float(getattr(solid, "volume", 0.0))
        genus = _genus_component_aware(solid) if wt else None
        # MEASURED on-surface: % of final verts within 0.002*diag of the
        # sampled original surface (same tolerance as the certificate)
        d_final, _ = tree.query(np.asarray(solid.vertices, dtype=np.float64), k=1)
        on_surface_pct = float((d_final <= 0.002 * diag).mean()) * 100.0

        metrics.update(
            ok=wt, watertight=wt, volume=vol,
            volume_positive=bool(vol > 0), genus=genus,
            on_surface_pct=round(on_surface_pct, 2),
            snapped_fraction=round(float(snap.mean()), 4),
            n_bridge_vertices=int(bridge.sum()),
            vertices_out=int(len(solid.vertices)),
            faces_out=int(len(solid.faces)),
            snap_radius=float(snap_radius),
            wall_time_s=round(time.time() - t0, 2),
        )
        _log("solidify:gwn")
        if not wt:
            # honest failure: return the input, not a half-broken surface
            metrics["error"] = "not watertight after manifold pass"
            return mesh, metrics
        return solid, metrics
    except Exception as e:
        metrics["error"] = f"{type(e).__name__}: {e}"
        _log("solidify:exception")
        return mesh, metrics


# ── selftest ─────────────────────────────────────────────────────────────────


def _naive_seal_occupancy(mesh: trimesh.Trimesh, res: int = 96) -> tuple:
    """The OLD reseal's occupancy recipe (surface voxelize + closing(2) +
    fill_holes) — reproduced here only so selftest can measure the cavity
    contrast. Returns (occ, origin, pitch)."""
    from scipy import ndimage
    ext = float((mesh.bounds[1] - mesh.bounds[0]).max())
    pitch = ext / res
    vox = mesh.voxelized(pitch=pitch)
    occ = np.asarray(vox.matrix, dtype=bool)
    occ = ndimage.binary_closing(occ, iterations=2)
    occ = ndimage.binary_fill_holes(occ)
    origin = vox.transform[:3, 3]
    return occ, origin, pitch


def _occ_at(occ: np.ndarray, origin: np.ndarray, pitch: float,
            pts: np.ndarray) -> np.ndarray:
    ijk = np.round((np.asarray(pts) - origin) / pitch).astype(int)
    ok = np.all((ijk >= 0) & (ijk < np.array(occ.shape)), axis=1)
    out = np.zeros(len(ijk), dtype=bool)
    out[ok] = occ[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    return out


def _make_bottle() -> Optional[trimesh.Trimesh]:
    """Hollow box with a NARROW neck channel to the outside (a 'mug with an
    open top' whose opening the old closing(2) seals shut). Built with
    manifold3d booleans; None if boolean support is unavailable."""
    try:
        outer = trimesh.creation.box(extents=(40, 40, 40))
        inner = trimesh.creation.box(extents=(34, 34, 34))
        # channel: 1.5 x 1.5 wide (~2-3 voxels at res 96 on a 40mm box),
        # long enough to pierce the top wall
        chan = trimesh.creation.box(extents=(1.5, 1.5, 12))
        chan.apply_translation((0, 0, 17))
        bottle = trimesh.boolean.difference([outer, inner], engine="manifold")
        bottle = trimesh.boolean.difference([bottle, chan], engine="manifold")
        if not bottle.is_watertight:
            return None
        return bottle
    except Exception:
        return None


def _shatter(m: trimesh.Trimesh, n_patches: int = 3, patch: int = 40,
             seed: int = 7) -> trimesh.Trimesh:
    """Delete a few random face patches -> open shell soup."""
    rng = np.random.default_rng(seed)
    fc = m.triangles_center
    drop: set[int] = set()
    for _ in range(n_patches):
        c = fc[rng.integers(len(fc))]
        order = np.argsort(np.linalg.norm(fc - c, axis=1))
        drop.update(int(i) for i in order[:patch])
    keep = np.array([i for i in range(len(m.faces)) if i not in drop])
    return trimesh.Trimesh(vertices=m.vertices.copy(),
                           faces=m.faces[keep].copy(), process=False)


def selftest() -> int:
    """MEASURED contract checks. Exit 0 on pass.
      1. backend agreement: numpy Barnes-Hut GWN vs gpytoolbox (>=99%
         inside/outside agreement on a sphere grid) — skipped with a note
         if gpytoolbox is missing.
      2. cavity preservation: narrow-neck hollow box — solidify keeps the
         cavity EMPTY while the old closing(2)+fill recipe fills it solid.
      3. shell soup: shattered sphere + offset partial shell -> watertight,
         volume>0, genus 0, on_surface >= 85%.
      4. corpus mesh (skipped if the AI corpus is absent): decimated sf3d
         axe.glb -> watertight + on_surface >= 85%.
    """
    from ._mesh_util import ascii_console
    ascii_console()
    print("=" * 78)
    print("_solidify selftest -- winding-number solidification")
    print("=" * 78)
    ok = True

    # 1. backend agreement
    sph = trimesh.creation.icosphere(3)
    rng = np.random.default_rng(0)
    Q = rng.uniform(-1.5, 1.5, size=(4000, 3))
    r = np.linalg.norm(Q, axis=1)
    Q = Q[np.abs(r - 1.0) > 0.08]     # keep away from the ambiguous surface
    w_np = _gwn_numpy(Q, np.asarray(sph.vertices), np.asarray(sph.faces))
    truth = (np.linalg.norm(Q, axis=1) < 1.0)
    agree_truth = float(((w_np >= 0.5) == truth).mean())
    print(f"[1] numpy-BH GWN vs analytic sphere: {agree_truth*100:.2f}% agree")
    ok &= agree_truth >= 0.99
    try:
        w_gp = _gwn_gpytoolbox(Q, np.asarray(sph.vertices), np.asarray(sph.faces))
        agree = float(((w_np >= 0.5) == (w_gp >= 0.5)).mean())
        maxerr = float(np.abs(w_np - w_gp).max())
        print(f"    vs gpytoolbox: {agree*100:.2f}% agree, max |dw|={maxerr:.4f}")
        ok &= agree >= 0.99
    except Exception as e:
        print(f"    gpytoolbox unavailable ({type(e).__name__}) -- skipped")

    # 2. cavity preservation vs naive seal
    bottle = _make_bottle()
    if bottle is None:
        print("[2] SKIP: manifold3d boolean unavailable, no bottle case")
    else:
        cav_pts = np.array([[0., 0., 0.], [5., 5., 0.], [-8., 0., 8.]])
        neck_pts = np.array([[0., 0., 18.], [0., 0., 19.5]])
        w, shape, origin, pitch, _b = _winding_grid(bottle, 96)
        occ_g = (np.abs(w) >= 0.5).reshape(shape)
        n_occ, n_origin, n_pitch = _naive_seal_occupancy(bottle, 96)
        cav_gwn = _occ_at(occ_g, origin, pitch, cav_pts)
        cav_naive = _occ_at(n_occ, n_origin, n_pitch, cav_pts)
        neck_gwn = _occ_at(occ_g, origin, pitch, neck_pts)
        print(f"[2] cavity voxels solid?  gwn={cav_gwn.tolist()} "
              f"naive={cav_naive.tolist()}  neck solid? gwn={neck_gwn.tolist()}")
        gwn_keeps = not cav_gwn.any() and not neck_gwn.any()
        naive_fills = bool(cav_naive.all())
        print(f"    gwn keeps cavity+neck open: {gwn_keeps}; "
              f"naive closing(2)+fill fills cavity: {naive_fills}")
        ok &= gwn_keeps and naive_fills

    # 3. shell soup
    base = trimesh.creation.icosphere(4)
    soup_a = _shatter(base, 3, 40)
    shell_b = base.copy()
    shell_b.apply_scale(1.02)
    soup_b = _shatter(shell_b, 5, 220, seed=3)
    soup = trimesh.util.concatenate([soup_a, soup_b])
    fixed, met = solidify(soup, res=96)
    print(f"[3] shell soup: wt={met['watertight']} vol>0={met.get('volume_positive')} "
          f"genus={met.get('genus')} on_surface={met.get('on_surface_pct')}% "
          f"backend={met.get('backend')} t={met.get('wall_time_s')}s")
    ok &= bool(met["watertight"]) and bool(met.get("volume_positive")) \
        and met.get("genus") == 0 and float(met.get("on_surface_pct", 0)) >= 85.0

    # 4. corpus mesh (optional)
    import os
    corpus = r"C:\Users\mecht\Project_EI\3DEI\Forge\corpus\sf3d\axe.glb"
    if os.path.exists(corpus):
        m = trimesh.load(corpus, force="mesh", process=False)
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(list(m.geometry.values()))
        if len(m.faces) > 60000:
            m = m.simplify_quadric_decimation(face_count=60000)
        fixed, met = solidify(m, res=96)
        print(f"[4] corpus axe.glb: wt={met['watertight']} "
              f"genus={met.get('genus')} on_surface={met.get('on_surface_pct')}% "
              f"t={met.get('wall_time_s')}s")
        ok &= bool(met["watertight"]) and \
            float(met.get("on_surface_pct", 0)) >= 85.0
    else:
        print("[4] SKIP: AI corpus not present on this machine")

    print("-" * 78)
    print("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(selftest())
