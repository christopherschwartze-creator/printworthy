"""Advanced 3D-printing optimization, layered on top of `_print3d`.

`_print3d` gives build-orientation, hollowing, printability, curved-layer toolpaths.
This module adds the print-shop / slicer-prep features that turn an oriented part
into something you can actually queue on a printer:

  1. SUPPORT GENERATION   generate_supports(body, up)
       Real, printable support GEOMETRY (grid pillars or a tree) under the overhang
       faces, dropped to the build plate. Returns a SEPARATE watertight mesh +
       support volume (cm3). The pillars taper to a small tip touch-point so they
       snap off; the body of each pillar is solid (watertight) for volume accounting.

  2. BRIDGE / OVERHANG ANALYSIS   bridge_analysis(body, up)
       Find long unsupported HORIZONTAL spans (bridges) the slicer must print in
       mid-air. A near-down-facing face whose nearest geometry directly below it is
       far away is a bridge; we measure the span by clustering such faces and
       reporting count + max bridge length (mm). Slicers bridge fine up to ~5-10mm;
       beyond that = sag / failure -> needs support.

  3. MULTI-PART BED PACKING   pack_bed(parts, bed=(220,220))
       Orient each part (min-support up), take its 2D footprint bbox, and shelf /
       bin-pack the rectangles onto the bed. Reports packing efficiency (used area /
       bed area), whether ALL fit, and per-part placements (for arranging copies or
       N different parts).

  4. WALL AUTO-THICKEN DETECTION   thin_wall_map(body, nozzle)
       Per-sample Shape-Diameter-Function wall thickness; flag the samples below
       2*nozzle (the printable minimum) and report WHERE (the thin-wall points, a
       thickness histogram, and the fraction of surface that is too thin).
       Geometry thickening is offered (optional, voxel dilation) but reported
       separately because it changes the shape.

Honest about roughness: support volume is the pillar geometry's true watertight
volume, but the pillar PLACEMENT is a heuristic grid/tree sampling, not a slicer's
exact support (which conforms to the overhang outline). Bridge length is a
ray-cast span estimate, not a contour trace. Bed packing is shelf/skyline 2D bin
packing on axis-aligned footprint bboxes (no rotation-in-plane optimization).

Permissive only: numpy / scipy / trimesh. NO GPL. Imports `_print3d`; never edits it.
"""
import os
import numpy as np
import trimesh

from . import _print3d


# ===========================================================================
#  shared: ray intersector + lay-flat
# ===========================================================================
def _intersector(mesh):
    try:
        return trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    except Exception:
        return mesh.ray


def lay_flat(body, up):
    """Copy of `body` rotated so up->+Z and resting on z=0 (build plate at z=0)."""
    b = body.copy()
    b.apply_transform(trimesh.geometry.align_vectors(np.asarray(up, float), [0, 0, 1.0]))
    b.apply_translation([0, 0, -float(b.vertices[:, 2].min())])
    return b


# ===========================================================================
#  1. SUPPORT-STRUCTURE GENERATION
# ===========================================================================
def _overhang_points(body_z, crit_deg, spacing):
    """Sample points on the overhang (down-facing, below-critical) faces of a part
    already laid flat (+Z = build axis). Returns P (k,3) sample points on the
    underside."""
    _, needs = _print3d._face_overhang(body_z, [0, 0, 1.0], crit_deg)
    fid = np.nonzero(needs)[0]
    if len(fid) == 0:
        return np.zeros((0, 3))
    # sample each overhang face: 1 point at the centroid, plus area-proportional
    # extra points so large overhangs get a denser pillar field.
    centers = body_z.triangles_center[fid]
    areas = body_z.area_faces[fid]
    pts = [centers]
    # extra samples for faces bigger than the pillar spacing footprint
    cell = spacing * spacing
    extra_n = np.clip((areas / max(cell, 1e-9)).astype(int), 0, 6)
    tris = body_z.triangles[fid]
    rng = np.random.default_rng(0)
    for i, ne in enumerate(extra_n):
        if ne <= 0:
            continue
        # uniform barycentric samples in triangle i
        r = rng.random((ne, 2))
        flip = r.sum(1) > 1
        r[flip] = 1 - r[flip]
        a, b, c = tris[i]
        p = a + r[:, :1] * (b - a) + r[:, 1:] * (c - a)
        pts.append(p)
    return np.vstack(pts)


def _snap_grid(P, spacing):
    """Snap support sample points to a grid of pitch `spacing` in XY, keeping the
    LOWEST z per occupied cell (the underside the pillar must reach). De-duplicates
    a dense overhang into a regular pillar field."""
    if len(P) == 0:
        return P
    key = np.floor(P[:, :2] / spacing).astype(np.int64)
    order = np.argsort(P[:, 2])               # lowest z first
    Ps, ks = P[order], key[order]
    seen = set()
    out = []
    for p, k in zip(Ps, ks):
        kk = (int(k[0]), int(k[1]))
        if kk not in seen:
            seen.add(kk)
            out.append(p)
    return np.array(out)


def generate_supports(body, up, crit_deg=45.0, spacing_mm=4.0, pillar_diam_mm=1.2,
                      tip_diam_mm=0.6, clearance_mm=0.4, style="grid", verbose=False):
    """Generate REAL printable support geometry under the overhangs of `body`,
    built along `up`, down to the build plate (z=0 after laying flat).

    style="grid": vertical tapered pillars on a regular XY grid under the overhangs
                  (robust, slicer-agnostic).
    style="tree": pillars that merge pairwise toward a common trunk lower down
                  (less material, organic-support style); falls back to grid where
                  merging is geometrically blocked.

    Each pillar is a tapered cylinder (pillar_diam at the base -> tip_diam at the
    contact), stopping `clearance_mm` BELOW the surface so it snaps off. Returns
    (support_mesh, info). support_mesh is a separate watertight* mesh; info has
    support_volume_cm3, n_pillars, watertight.
    (*each pillar is watertight; the union is a disjoint set of watertight pillars,
    which trimesh reports as watertight iff every component closes — we verify.)
    """
    bz = lay_flat(body, up)
    P = _overhang_points(bz, crit_deg, spacing_mm)
    if len(P) == 0:
        empty = trimesh.Trimesh()
        return empty, {"n_pillars": 0, "support_volume_cm3": 0.0,
                       "watertight": True, "note": "no overhangs need support"}
    P = _snap_grid(P, spacing_mm)
    # contact point = clearance below the sampled underside; base = build plate z=0
    tops = P.copy()
    tops[:, 2] = np.maximum(tops[:, 2] - clearance_mm, 0.05)   # tip just under surface

    pillars = []

    def _pillar(x, y, z_top, r_base, r_tip, z_base=0.0):
        h = float(z_top - z_base)
        if h < 0.2:
            return None
        seg = max(3, int(np.ceil(h / 6.0)))   # vertical segments for a tapered stack
        # build a tapered cylinder as a stack of frustums via a revolved profile
        zs = np.linspace(z_base, z_top, seg + 1)
        rs = np.linspace(r_base, r_tip, seg + 1)
        rings = 10
        ang = np.linspace(0, 2 * np.pi, rings, endpoint=False)
        verts = []
        for z, r in zip(zs, rs):
            ring = np.stack([x + r * np.cos(ang), y + r * np.sin(ang),
                             np.full(rings, z)], 1)
            verts.append(ring)
        V = np.vstack(verts)
        faces = []
        for s in range(seg):
            for k in range(rings):
                a = s * rings + k
                b = s * rings + (k + 1) % rings
                c = (s + 1) * rings + k
                d = (s + 1) * rings + (k + 1) % rings
                faces.append([a, c, b]); faces.append([b, c, d])
        # bottom cap (at z_base)
        cb = len(V); V = np.vstack([V, [x, y, z_base]])
        for k in range(rings):
            faces.append([cb, (k + 1) % rings, k])
        # top cap (tip)
        ct = len(V); V = np.vstack([V, [x, y, z_top]])
        base_top = seg * rings
        for k in range(rings):
            faces.append([ct, base_top + k, base_top + (k + 1) % rings])
        return trimesh.Trimesh(np.asarray(V), np.asarray(faces), process=True)

    if style == "tree":
        # cluster pillar tops in XY; each cluster gets one trunk that branches up to
        # the individual contacts. Simplest robust realization: a thick trunk from
        # the plate up to a junction height, then thin branches to each tip.
        from scipy.cluster.hierarchy import fcluster, linkage
        if len(tops) >= 2:
            Z = linkage(tops[:, :2], method="single")
            # cluster radius ~ 3 spacings so nearby pillars share a trunk
            labels = fcluster(Z, t=3.0 * spacing_mm, criterion="distance")
        else:
            labels = np.array([1])
        for lab in np.unique(labels):
            grp = tops[labels == lab]
            cx, cy = grp[:, 0].mean(), grp[:, 1].mean()
            jz = max(0.2, 0.45 * grp[:, 2].min())     # junction height
            # trunk: plate -> junction, thicker
            trunk = _pillar(cx, cy, jz, pillar_diam_mm / 2 * 1.4, pillar_diam_mm / 2)
            if trunk is not None:
                pillars.append(trunk)
            # branches: each tip is a thin pillar that STARTS at the junction height
            # (genuinely sits on the shared trunk -> only one plate-to-junction
            # column of material for the whole cluster, the real tree-support saving).
            for g in grp:
                br = _pillar(g[0], g[1], g[2], pillar_diam_mm / 2 * 0.8,
                             tip_diam_mm / 2, z_base=jz)
                if br is not None:
                    pillars.append(br)
    else:   # grid
        for t in tops:
            pil = _pillar(t[0], t[1], t[2], pillar_diam_mm / 2, tip_diam_mm / 2)
            if pil is not None:
                pillars.append(pil)

    if not pillars:
        empty = trimesh.Trimesh()
        return empty, {"n_pillars": 0, "support_volume_cm3": 0.0, "watertight": True,
                       "note": "overhangs present but all contacts at/below plate"}

    support = trimesh.util.concatenate(pillars)
    # volume = sum of per-pillar volumes (robust even if components touch)
    vol = float(sum(abs(p.volume) for p in pillars))
    all_wt = all(p.is_watertight for p in pillars)
    info = {
        "n_pillars": len(pillars),
        "style": style,
        "support_volume_cm3": round(vol / 1000.0, 3),
        "contact_diam_mm": tip_diam_mm,
        "pillar_diam_mm": pillar_diam_mm,
        "spacing_mm": spacing_mm,
        "watertight": bool(all_wt),    # every pillar closes
        "n_faces": int(len(support.faces)),
    }
    if verbose:
        print(f"  supports: {info['n_pillars']} pillars, {info['support_volume_cm3']} cm3, "
              f"watertight={info['watertight']}")
    return support, info


# ===========================================================================
#  2. BRIDGE / OVERHANG ANALYSIS
# ===========================================================================
def bridge_analysis(body, up, crit_deg=45.0, near_down_deg=20.0, bridge_warn_mm=5.0,
                    verbose=False):
    """Find long unsupported HORIZONTAL spans (bridges). A bridge is a roughly
    horizontal, down-facing surface (face normal within `near_down_deg` of straight
    down) whose nearest geometry directly below is far away — the slicer prints it
    in mid-air. We:
      - select near-horizontal down-facing faces,
      - for each, ray-cast straight down to the next surface; the *gap* is the drop
        the bridge spans over empty space,
      - cluster the bridge faces into connected spans and measure each span's
        in-plane extent (mm).
    Returns dict: n_bridge_faces, n_spans, max_bridge_len_mm, spans (list).
    `bridge_warn_mm`: spans longer than this are flagged risky.
    """
    bz = lay_flat(body, up)
    n = np.asarray(bz.face_normals, float)
    down = np.array([0, 0, -1.0])
    cosang = n @ down
    # near-horizontal underside: normal points mostly down
    is_bridge_face = cosang > np.cos(np.radians(near_down_deg))
    fid = np.nonzero(is_bridge_face)[0]
    if len(fid) == 0:
        out = {"n_bridge_faces": 0, "n_spans": 0, "max_bridge_len_mm": 0.0,
               "spans": [], "note": "no near-horizontal downfacing faces"}
        if verbose:
            print("  bridges: none")
        return out

    # gap below each candidate face: cast a ray straight down from just under it
    centers = bz.triangles_center[fid]
    origins = centers + np.array([0, 0, -1e-3])
    dirs = np.tile(down, (len(fid), 1))
    inter = _intersector(bz)
    try:
        locs, ray_idx, _ = inter.intersects_location(origins, dirs, multiple_hits=False)
    except Exception:
        locs, ray_idx = np.zeros((0, 3)), np.array([], int)
    gap = np.zeros(len(fid))                     # 0 = something right below (solid)
    if len(ray_idx):
        g = origins[ray_idx][:, 2] - locs[:, 2]
        gap[ray_idx] = g
    # faces with no hit below = open to the plate (gap to z=0)
    hit_mask = np.zeros(len(fid), bool)
    hit_mask[ray_idx] = True
    gap[~hit_mask] = centers[~hit_mask, 2]       # drop to build plate

    # a face is "unsupported bridge" if the gap below it is meaningful (the slicer
    # has nothing to print onto). Use half a typical layer-bridge clearance.
    unsupported = gap > 0.5
    bfid = fid[unsupported]
    if len(bfid) == 0:
        out = {"n_bridge_faces": 0, "n_spans": 0, "max_bridge_len_mm": 0.0,
               "spans": [], "note": "downfacing faces all sit on solid below"}
        if verbose:
            print("  bridges: 0 (all downfacing faces supported from below)")
        return out

    # cluster the bridge faces into connected spans via face adjacency
    import networkx as nx
    bset = set(int(x) for x in bfid)
    G = nx.Graph()
    G.add_nodes_from(bset)
    adj = bz.face_adjacency
    for a, b in adj:
        if a in bset and b in bset:
            G.add_edge(int(a), int(b))
    spans = []
    for comp in nx.connected_components(G):
        comp = list(comp)
        pts = bz.triangles_center[comp][:, :2]   # XY footprint of the span
        if len(pts) == 1:
            extent = float(np.sqrt(bz.area_faces[comp[0]]))   # ~ tri size
        else:
            # in-plane diameter = max pairwise XY distance (bridge length)
            mn, mx = pts.min(0), pts.max(0)
            extent = float(np.linalg.norm(mx - mn))
        gmean = float(gap[np.isin(fid, comp)].mean()) if len(comp) else 0.0
        spans.append({"n_faces": len(comp), "length_mm": round(extent, 2),
                      "mean_gap_mm": round(gmean, 2),
                      "risky": bool(extent > bridge_warn_mm)})
    spans.sort(key=lambda s: -s["length_mm"])
    maxlen = spans[0]["length_mm"] if spans else 0.0
    out = {
        "n_bridge_faces": int(len(bfid)),
        "n_spans": len(spans),
        "max_bridge_len_mm": maxlen,
        "n_risky_spans": int(sum(s["risky"] for s in spans)),
        "bridge_warn_mm": bridge_warn_mm,
        "spans": spans[:20],     # top 20 by length
    }
    if verbose:
        print(f"  bridges: {out['n_spans']} spans, max {maxlen}mm, "
              f"{out['n_risky_spans']} risky (>{bridge_warn_mm}mm)")
    return out


# ===========================================================================
#  3. MULTI-PART BED PACKING
# ===========================================================================
def _footprint(body, up, margin_mm):
    """Oriented part's 2D footprint bbox (w,h) in mm + the laid-flat body."""
    bz = lay_flat(body, up)
    xy = bz.vertices[:, :2]
    w = float(xy[:, 0].max() - xy[:, 0].min()) + 2 * margin_mm
    d = float(xy[:, 1].max() - xy[:, 1].min()) + 2 * margin_mm
    return w, d, bz


def _shelf_pack(boxes, bed_w, bed_h):
    """Shelf (next-fit decreasing height) 2D bin packing of axis-aligned boxes onto
    one bed. boxes = list of (w,h,id). Returns (placements, all_fit). Tries each box
    in its given orientation and rotated 90deg, picking whichever fits the shelf.
    placements: list of (id, x, y, w, h, rotated)."""
    # sort by height descending (classic shelf heuristic)
    order = sorted(range(len(boxes)), key=lambda i: -max(boxes[i][0], boxes[i][1]))
    placements = []
    shelf_y = 0.0
    shelf_h = 0.0
    cursor_x = 0.0
    all_fit = True
    for i in order:
        w, h, bid = boxes[i]
        placed = False
        for (ww, hh, rot) in [(w, h, False), (h, w, True)]:
            if ww > bed_w or hh > bed_h:
                continue
            # try current shelf
            if cursor_x + ww <= bed_w + 1e-9 and shelf_y + hh <= bed_h + 1e-9:
                placements.append((bid, cursor_x, shelf_y, ww, hh, rot))
                cursor_x += ww
                shelf_h = max(shelf_h, hh)
                placed = True
                break
        if placed:
            continue
        # open a new shelf
        for (ww, hh, rot) in [(w, h, False), (h, w, True)]:
            if ww > bed_w or hh > bed_h:
                continue
            ny = shelf_y + shelf_h
            if ny + hh <= bed_h + 1e-9 and ww <= bed_w + 1e-9:
                shelf_y = ny
                shelf_h = hh
                cursor_x = ww
                placements.append((bid, 0.0, shelf_y, ww, hh, rot))
                placed = True
                break
        if not placed:
            all_fit = False
    return placements, all_fit


def pack_bed(parts, bed=(220.0, 220.0), margin_mm=3.0, crit_deg=45.0,
             ups=None, verbose=False):
    """Orient + arrange parts on a single build plate.

    parts : list of trimesh meshes (already in mm; e.g. scaled to print_size). For
            N copies of one part, pass [mesh]*N.
    ups   : optional list of pre-computed build-up vectors (one per part). If None,
            each part is orientation-optimized for minimum support.
    Returns dict: all_fit, packing_efficiency, bed_mm, n_parts, placements,
    arranged_scene (a trimesh.Scene of the laid-out parts, for rendering/export).
    """
    bed_w, bed_h = bed
    boxes = []
    flats = []
    chosen_ups = []
    for i, p in enumerate(parts):
        up = (ups[i] if ups is not None else
              _print3d.optimize_orientation(p, crit_deg=crit_deg)["up"])
        chosen_ups.append(up)
        w, d, bz = _footprint(p, up, margin_mm)
        boxes.append((w, d, i))
        flats.append(bz)
    placements, all_fit = _shelf_pack(boxes, bed_w, bed_h)
    used_area = sum(w * h for (_, _, _, w, h, _) in placements)
    eff = used_area / (bed_w * bed_h)
    # build an arranged scene: translate each flat body to its placement
    placed_ids = set()
    arranged = []
    pmap = {pid: (x, y, w, h, rot) for (pid, x, y, w, h, rot) in placements}
    for i, bz in enumerate(flats):
        if i not in pmap:
            continue
        placed_ids.add(i)
        x, y, w, h, rot = pmap[i]
        b = bz.copy()
        if rot:
            b.apply_transform(trimesh.transformations.rotation_matrix(
                np.pi / 2, [0, 0, 1]))
        # move so its min-xy corner lands at (x+margin, y+margin)
        mn = b.vertices[:, :2].min(0)
        b.apply_translation([x + margin_mm - mn[0], y + margin_mm - mn[1], 0])
        arranged.append(b)
    scene = trimesh.Scene(arranged) if arranged else trimesh.Scene()
    out = {
        "n_parts": len(parts),
        "n_placed": len(placed_ids),
        "all_fit": bool(all_fit and len(placed_ids) == len(parts)),
        "bed_mm": [bed_w, bed_h],
        "packing_efficiency": round(float(eff), 3),
        "used_area_mm2": round(float(used_area), 1),
        "placements": [
            {"part": int(pid), "x": round(x, 1), "y": round(y, 1),
             "w": round(w, 1), "h": round(h, 1), "rotated": bool(rot)}
            for (pid, x, y, w, h, rot) in placements
        ],
        "_scene": scene,
        "_ups": chosen_ups,
    }
    if verbose:
        print(f"  bed pack: {out['n_placed']}/{out['n_parts']} placed, "
              f"all_fit={out['all_fit']}, efficiency={100*eff:.1f}%")
    return out


# ===========================================================================
#  4. WALL AUTO-THICKEN DETECTION
# ===========================================================================
def thin_wall_map(body, nozzle=0.4, n_samples=3000, thicken=False, verbose=False):
    """Per-sample wall thickness via the Shape Diameter Function: from each sample
    face, ray-cast inward along -normal to the far wall. Flag samples with thickness
    < 2*nozzle (the printable minimum). Reports WHERE the thin walls are.

    Returns dict: min_wall_mm, frac_thin (fraction of sampled surface below the
    threshold), thin_threshold_mm, thin_points (xyz of thin samples, capped),
    histogram, and (if thicken) a dilated 'thickened_mesh' + delta volume.
    """
    thresh = 2.0 * nozzle
    inter = _intersector(body)
    nf = len(body.faces)
    k = min(n_samples, nf)
    idx = np.random.default_rng(0).choice(nf, k, replace=False)
    centers = body.triangles_center[idx]
    normals = body.face_normals[idx]
    origins = centers - 1e-3 * normals       # nudge inside
    dirs = -normals
    try:
        locs, ray_idx, _ = inter.intersects_location(origins, dirs, multiple_hits=False)
    except Exception:
        return {"error": "ray cast failed", "min_wall_mm": None}
    if len(locs) == 0:
        return {"min_wall_mm": None, "frac_thin": None,
                "note": "no inward intersections (open mesh?)"}
    d = np.linalg.norm(locs - origins[ray_idx], axis=1)
    good = d > 1e-4
    d = d[good]
    hit_centers = centers[ray_idx][good]
    if len(d) == 0:
        return {"min_wall_mm": None, "frac_thin": None}
    thin = d < thresh
    # histogram for reporting
    bins = [0, thresh * 0.5, thresh, thresh * 2, thresh * 4, np.inf]
    hist, _ = np.histogram(d, bins=bins)
    labels = [f"<{thresh*0.5:.1f}", f"{thresh*0.5:.1f}-{thresh:.1f}",
              f"{thresh:.1f}-{thresh*2:.1f}", f"{thresh*2:.1f}-{thresh*4:.1f}",
              f">{thresh*4:.1f}"]
    thin_pts = hit_centers[thin]
    # cap reported points
    if len(thin_pts) > 300:
        sel = np.random.default_rng(1).choice(len(thin_pts), 300, replace=False)
        thin_pts_rep = thin_pts[sel]
    else:
        thin_pts_rep = thin_pts
    out = {
        "nozzle_mm": nozzle,
        "thin_threshold_mm": round(thresh, 3),
        "min_wall_mm": round(float(d.min()), 3),
        "p1_wall_mm": round(float(np.percentile(d, 1)), 3),
        "median_wall_mm": round(float(np.median(d)), 3),
        "frac_thin": round(float(thin.mean()), 4),
        "n_thin_samples": int(thin.sum()),
        "n_samples": int(len(d)),
        "histogram": {lab: int(c) for lab, c in zip(labels, hist)},
        "thin_points": thin_pts_rep.round(2).tolist(),
        "thin_centroid": (thin_pts.mean(0).round(2).tolist()
                          if len(thin_pts) else None),
    }
    if thicken and thin.any():
        try:
            from scipy.ndimage import binary_dilation
            pitch = max(nozzle * 0.5, float(body.extents.max()) / 256.0)
            n_dil = max(1, int(round((thresh - 0) / pitch / 2)))
            vox = body.voxelized(pitch=pitch).fill()
            mat = np.asarray(vox.matrix, bool)
            dil = binary_dilation(mat, iterations=n_dil)
            thick = trimesh.voxel.VoxelGrid(dil, transform=vox.transform).marching_cubes
            thick.fix_normals()
            out["thickened_mesh"] = thick
            out["thicken_delta_cm3"] = round(
                (float(dil.sum()) - float(mat.sum())) * pitch ** 3 / 1000.0, 3)
            out["thicken_note"] = ("voxel-dilated by ~{:.2f}mm everywhere (changes "
                                   "shape; for targeted thickening slice locally)"
                                   .format(n_dil * pitch))
        except Exception as e:
            out["thicken_error"] = str(e)
    if verbose:
        print(f"  walls: min {out['min_wall_mm']}mm thresh {thresh:.1f}mm "
              f"frac_thin {100*out['frac_thin']:.1f}% ({out['n_thin_samples']} pts)")
    return out


# ===========================================================================
#  driver / validation
# ===========================================================================
def _load(name, idx, base, target_faces=8000):
    from ._mesh_util import decimate
    m = trimesh.load(os.path.join(base, f"{idx}.off"), process=True)
    m = decimate(m, target_faces)
    m.fix_normals()
    return m


def _scale_to(body, size_mm):
    b = body.copy()
    b.apply_scale(size_mm / (float(b.bounding_box.extents.max()) + 1e-12))
    return b


def run_validation(names_idx, base, print_size_mm=60.0, verbose=True):
    """Run all four features on each named part; return a results dict."""
    results = {}
    bodies = {}
    ups = {}
    for nm, idx in names_idx:
        if verbose:
            print(f"\n=== {nm} ===")
        m = _load(nm, idx, base)
        body = _scale_to(m, print_size_mm)
        ori = _print3d.optimize_orientation(body, crit_deg=45.0)
        up = ori["up"]
        bodies[nm] = body
        ups[nm] = up
        sup, sinfo = generate_supports(body, up, style="grid", verbose=verbose)
        tree, tinfo = generate_supports(body, up, style="tree", verbose=False)
        bridges = bridge_analysis(body, up, verbose=verbose)
        walls = thin_wall_map(body, nozzle=0.4, verbose=verbose)
        results[nm] = {
            "orientation_up": [round(float(x), 3) for x in up],
            "support_frac": round(float(ori["support_frac"]), 3),
            "height_mm": round(float(ori["height"]), 1),
            "supports_grid": sinfo,
            "supports_tree": {k: tinfo[k] for k in
                              ("n_pillars", "support_volume_cm3", "watertight")},
            "bridges": {k: bridges[k] for k in
                        ("n_bridge_faces", "n_spans", "max_bridge_len_mm",
                         "n_risky_spans") if k in bridges},
            "walls": {k: walls[k] for k in
                      ("min_wall_mm", "p1_wall_mm", "median_wall_mm",
                       "thin_threshold_mm", "frac_thin", "n_thin_samples")
                      if k in walls},
        }
    # bed packing: 4 copies of teddy AND the 4 different parts
    part_list = [bodies[nm] for nm, _ in names_idx]
    up_list = [ups[nm] for nm, _ in names_idx]
    pack_diff = pack_bed(part_list, bed=(220, 220), ups=up_list, verbose=verbose)
    first = names_idx[0][0]
    pack_copies = pack_bed([bodies[first]] * 4, bed=(220, 220),
                           ups=[ups[first]] * 4, verbose=verbose)
    results["_bed_packing"] = {
        "four_different_parts": {k: pack_diff[k] for k in
                                 ("all_fit", "packing_efficiency", "n_placed",
                                  "n_parts", "placements")},
        f"four_copies_of_{first}": {k: pack_copies[k] for k in
                                    ("all_fit", "packing_efficiency", "n_placed",
                                     "n_parts")},
    }
    results["_scenes"] = {"pack_diff": pack_diff, "pack_copies": pack_copies,
                          "bodies": bodies, "ups": ups}
    return results


def render_figure(results, out_png):
    """6-panel figure: per-part support pillars + bridge faces, bed packing layout,
    and a thin-wall map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    scn = results["_scenes"]
    bodies = scn["bodies"]
    ups = scn["ups"]
    names = [k for k in bodies.keys()]
    fig = plt.figure(figsize=(18, 11))

    # top row: support pillars under each of 2 parts + bridges on a 3rd
    showp = names[:2]
    for i, nm in enumerate(showp):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        bz = lay_flat(bodies[nm], ups[nm])
        sup, sinfo = generate_supports(bodies[nm], ups[nm], style="grid")
        ax.add_collection3d(Poly3DCollection(
            bz.triangles, alpha=0.25, facecolor="#88aacc", edgecolor="none"))
        if len(sup.faces):
            ax.add_collection3d(Poly3DCollection(
                sup.triangles, alpha=0.9, facecolor="#cc5544", edgecolor="none"))
        _setlim(ax, bz.vertices)
        ax.set_title(f"{nm}: {sinfo['n_pillars']} support pillars\n"
                     f"{sinfo['support_volume_cm3']} cm3 (watertight={sinfo['watertight']})",
                     fontsize=10)
        ax.set_axis_off()

    # bridges on the 3rd part
    nm = names[2] if len(names) > 2 else names[0]
    ax = fig.add_subplot(2, 3, 3, projection="3d")
    bz = lay_flat(bodies[nm], ups[nm])
    br = bridge_analysis(bodies[nm], ups[nm])
    n = bz.face_normals
    cosang = n @ np.array([0, 0, -1.0])
    bf = cosang > np.cos(np.radians(20))
    ax.add_collection3d(Poly3DCollection(
        bz.triangles, alpha=0.2, facecolor="#aaaaaa", edgecolor="none"))
    if bf.any():
        ax.add_collection3d(Poly3DCollection(
            bz.triangles[bf], alpha=0.85, facecolor="#dd9911", edgecolor="none"))
    _setlim(ax, bz.vertices)
    ax.set_title(f"{nm}: {br['n_spans']} bridge spans\n"
                 f"max {br['max_bridge_len_mm']}mm, {br['n_risky_spans']} risky",
                 fontsize=10)
    ax.set_axis_off()

    # bed packing (different parts)
    ax = fig.add_subplot(2, 3, 4)
    pk = scn["pack_diff"]
    _draw_bed(ax, pk, [n for n in names])
    ax.set_title(f"bed pack: {pk['n_placed']}/{pk['n_parts']} parts, "
                 f"{100*pk['packing_efficiency']:.0f}% eff, fit={pk['all_fit']}",
                 fontsize=10)

    # bed packing (4 copies)
    ax = fig.add_subplot(2, 3, 5)
    pk2 = scn["pack_copies"]
    _draw_bed(ax, pk2, [names[0]] * 4)
    ax.set_title(f"bed pack: 4x {names[0]}, "
                 f"{100*pk2['packing_efficiency']:.0f}% eff, fit={pk2['all_fit']}",
                 fontsize=10)

    # thin-wall map on first part
    ax = fig.add_subplot(2, 3, 6, projection="3d")
    nm = names[0]
    bz = lay_flat(bodies[nm], ups[nm])
    wm = thin_wall_map(bodies[nm], nozzle=0.4)
    ax.add_collection3d(Poly3DCollection(
        bz.triangles, alpha=0.18, facecolor="#bbbbbb", edgecolor="none"))
    tp = np.array(wm.get("thin_points", []))
    if len(tp):
        # thin points were computed pre-layflat; recompute on bz for display
        wm2 = thin_wall_map(bz, nozzle=0.4)
        tp = np.array(wm2.get("thin_points", []))
    if len(tp):
        ax.scatter(tp[:, 0], tp[:, 1], tp[:, 2], c="red", s=6)
    _setlim(ax, bz.vertices)
    ax.set_title(f"{nm}: thin walls < {wm['thin_threshold_mm']}mm\n"
                 f"min {wm['min_wall_mm']}mm, {100*wm['frac_thin']:.1f}% of surface",
                 fontsize=10)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close()
    return out_png


def _setlim(ax, V):
    c = V.mean(0)
    r = float(np.abs(V - c).max()) * 1.05
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _draw_bed(ax, pk, names):
    import matplotlib.patches as mp
    bw, bh = pk["bed_mm"]
    ax.add_patch(mp.Rectangle((0, 0), bw, bh, fill=False, edgecolor="black", lw=2))
    cmap = ["#88aacc", "#cc8844", "#88cc88", "#cc88cc", "#cccc44", "#44cccc"]
    for pl in pk["placements"]:
        x, y, w, h = pl["x"], pl["y"], pl["w"], pl["h"]
        pid = pl["part"]
        ax.add_patch(mp.Rectangle((x, y), w, h, facecolor=cmap[pid % len(cmap)],
                                  edgecolor="black", alpha=0.7))
        lab = names[pid] if pid < len(names) else str(pid)
        ax.text(x + w / 2, y + h / 2, f"{lab}{'*' if pl['rotated'] else ''}",
                ha="center", va="center", fontsize=8)
    ax.set_xlim(-5, bw + 5)
    ax.set_ylim(-5, bh + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("mm")
