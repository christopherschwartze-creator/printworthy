"""3D-printing optimization on top of the field-aligned quad remesher.

For printing, "good" is not high val4 -- it is: WATERTIGHT + MANIFOLD (the remesher
guarantees this), SURFACE-ACCURATE (it does, <0.006), SUPPORT-MINIMAL (build
orientation), and FEATURE-PRINTABLE (no sub-nozzle-width walls). This module adds
the printing-specific optimizations:

  optimize_orientation : the build orientation that MINIMIZES support material
                         (overhang area x drop height) -- the #1 print optimization.
  overhang_analysis    : per-face overhang severity at a given orientation.
  printability_report  : watertight / manifold / bbox / min wall thickness / support%.
  curved_layers        : EI/field connection -- the cross-field's quad rows that run
                         ~perpendicular to the build axis are candidate NON-PLANAR
                         (curved-layer) toolpaths: stronger, smoother prints than flat
                         slicing, and the field-aligned remesh provides them directly.

Permissive: numpy / scipy / trimesh. NO GPL.
"""
import os
import numpy as np
import trimesh


# ---------------------------------------------------------------------------
#  overhang / support model
# ---------------------------------------------------------------------------
def _face_overhang(mesh, up, crit_deg=45.0):
    """Per-face overhang severity in [0,1] for build direction `up` (layers stack
    along +up; the build plate is the min-up face). A DOWNWARD-facing face is
    self-supporting if its angle from horizontal >= crit; below that it needs
    support. severity = how far below the critical angle (0 = ok, 1 = facing
    straight down). Returns (severity (F,), needs_support (F,) bool)."""
    up = np.asarray(up, float); up /= np.linalg.norm(up) + 1e-12
    n = np.asarray(mesh.face_normals, float)
    ndu = n @ up                                   # >0 up-facing, <0 down-facing
    crit = np.radians(crit_deg)
    # face angle from horizontal = 90 - angle(n, up); overhang if that < crit_deg
    # i.e. angle(n,up) > 90 - crit  AND down-facing (ndu<0)
    cos_thresh = -np.cos(crit)                     # ndu below this => overhang
    needs = ndu < cos_thresh
    # severity: 0 at the threshold, 1 at straight-down (ndu=-1)
    sev = np.clip((cos_thresh - ndu) / (1.0 + cos_thresh + 1e-12), 0, 1)
    sev[~needs] = 0.0
    return sev, needs


def support_cost(mesh, up, crit_deg=45.0):
    """Estimated support material for build direction `up` = sum over overhang
    faces of (face area x severity x drop-height-to-base). Lower = less support."""
    up = np.asarray(up, float); up /= np.linalg.norm(up) + 1e-12
    sev, needs = _face_overhang(mesh, up, crit_deg)
    if not needs.any():
        return 0.0
    area = np.asarray(mesh.area_faces, float)
    h = np.asarray(mesh.triangles_center, float) @ up
    base = float(h.min())
    drop = np.clip(h - base, 0, None)              # height above the build plate
    return float(np.sum(area[needs] * sev[needs] * drop[needs]))


def _candidate_ups(mesh, n_sphere=80):
    """Candidate build-up directions: convex-hull face normals (natural flat
    rest-bases, both signs) + a Fibonacci sphere sampling."""
    cands = []
    try:
        hull = mesh.convex_hull
        hn = np.asarray(hull.face_normals, float)
        cands.extend(list(hn)); cands.extend(list(-hn))
    except Exception:
        pass
    # Fibonacci sphere
    i = np.arange(n_sphere) + 0.5
    phi = np.arccos(1 - 2 * i / n_sphere)
    th = np.pi * (1 + 5 ** 0.5) * i
    fib = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    cands.extend(list(fib))
    C = np.array(cands)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    # dedup near-duplicates
    keep, seen = [], []
    for c in C:
        if all(abs(c @ s) < 0.997 for s in seen):
            seen.append(c); keep.append(c)
    return np.array(keep)


def _orient_score(mesh, up, crit_deg, area_tot, diag, w_height):
    """SUPPORT material is PRIMARY (support_frac in [0,1]); part height is a small
    secondary tiebreaker (lower = fewer layers = faster). Returns (score, sfrac,
    height)."""
    sev, needs = _face_overhang(mesh, up, crit_deg)
    sfrac = float(np.sum(np.asarray(mesh.area_faces)[needs]) / (area_tot + 1e-12))
    h = np.asarray(mesh.vertices, float) @ (up / (np.linalg.norm(up) + 1e-12))
    height = float(h.max() - h.min())
    score = sfrac + w_height * (height / (diag + 1e-9))   # sfrac dominates
    return score, sfrac, height


def optimize_orientation(mesh, crit_deg=45.0, w_height=0.04, w_strength=0.0,
                         field_dir=None, n_sphere=160, refine=True, verbose=False):
    """Build orientation (up vector) minimizing SUPPORT material (primary) + a
    `w_height` term (shorter part = fewer layers = faster). `refine`: local search
    around the best candidate for a continuous optimum. Returns dict {up,
    support_frac, height, transform}."""
    cands = _candidate_ups(mesh, n_sphere)
    area_tot = float(mesh.area); diag = float(np.linalg.norm(mesh.extents))
    best = None
    for up in cands:
        score, sfrac, height = _orient_score(mesh, up, crit_deg, area_tot, diag, w_height)
        if w_strength > 0 and field_dir is not None:
            fd = np.asarray(field_dir, float); fd /= np.linalg.norm(fd) + 1e-12
            score += w_strength * abs(fd @ up)
        if best is None or score < best[0]:
            best = (score, np.asarray(up, float), sfrac, height)
    # continuous refinement: jitter the best up over a shrinking cone
    if refine:
        rng = np.random.default_rng(0)
        up = best[1].copy()
        for step in (0.25, 0.12, 0.05, 0.02):
            improved = True
            while improved:
                improved = False
                for _ in range(24):
                    cand = up + step * rng.standard_normal(3)
                    cand /= np.linalg.norm(cand) + 1e-12
                    s, sf, ht = _orient_score(mesh, cand, crit_deg, area_tot, diag, w_height)
                    if s < best[0] - 1e-9:
                        best = (s, cand, sf, ht); up = cand; improved = True
    _, up, sfrac, height = best
    if verbose:
        print(f"  best up={up.round(3)} support_frac={100*sfrac:.1f}% height={height:.1f}")
    return {"up": up, "support_frac": float(sfrac), "height": float(height),
            "transform": trimesh.geometry.align_vectors(up, [0, 0, 1.0])}


def print_estimate(body, up, layer_mm=0.2, wall_mm=2.0, speed_mm_s=50.0,
                   infill=0.15, hollow=True):
    """Rough print time + material estimate for `body` built along `up`.
    Material = shell walls + infill (or solid). Time ~ extruded length / speed.
    Returns dict (cm3, grams@PLA, layers, hours)."""
    up = np.asarray(up, float); up /= np.linalg.norm(up) + 1e-12
    h = np.asarray(body.vertices, float) @ up
    height = float(h.max() - h.min())
    layers = max(1, int(np.ceil(height / layer_mm)))
    v_solid = float(body.volume) if body.is_watertight else float(body.bounding_box.volume) * 0.4
    if hollow:
        # wall shell + infill of the cavity
        sinfo = {}
        try:
            _, sinfo = hollow_shell(body, wall_mm=wall_mm)
        except Exception:
            pass
        saved = sinfo.get("material_saved_pct", 60.0) / 100.0
        v_wall = v_solid * (1 - saved)
        v_used = v_wall + v_solid * saved * infill
    else:
        v_used = v_solid
    cm3 = v_used / 1000.0
    grams = cm3 * 1.24                          # PLA density ~1.24 g/cm3
    # FDM time heuristic: extrusion time + per-layer overhead. ~0.12 h/cm3 of
    # extruded material (perimeters+infill at typical 50mm/s, 0.2mm) + layer accel.
    hours = cm3 * 0.12 + layers * 0.0025
    return {"layers": layers, "height_mm": round(height, 1),
            "material_cm3": round(cm3, 2), "grams_PLA": round(grams, 1),
            "est_hours": round(hours, 2)}


# ---------------------------------------------------------------------------
#  printability report
# ---------------------------------------------------------------------------
def min_wall_thickness(mesh, n_samples=2000):
    """Estimate the minimum wall thickness via the Shape Diameter Function: cast a
    ray inward from sample points along -normal, distance to the far side. The min
    over samples ~ thinnest wall (sub-nozzle-width walls fail to print)."""
    try:
        rmi = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    except Exception:
        rmi = mesh.ray
    idx = np.random.default_rng(0).choice(len(mesh.faces), min(n_samples, len(mesh.faces)), replace=False)
    origins = mesh.triangles_center[idx] - 1e-4 * mesh.face_normals[idx]
    dirs = -mesh.face_normals[idx]
    try:
        locs, ray_idx, _ = rmi.intersects_location(origins, dirs, multiple_hits=False)
    except Exception:
        return float("nan")
    if len(locs) == 0:
        return float("nan")
    d = np.linalg.norm(locs - origins[ray_idx], axis=1)
    d = d[d > 1e-5]
    return float(np.percentile(d, 1)) if len(d) else float("nan")


def printability_report(mesh, nozzle=0.4, oriented_up=None, crit_deg=45.0):
    """Watertight/manifold/feature checks + support estimate. `nozzle` = nozzle
    width in the mesh's units (set the mesh to mm). Returns a dict."""
    ext = mesh.bounding_box.extents
    rep = {
        "watertight": bool(mesh.is_watertight),
        "manifold": bool(mesh.is_winding_consistent and mesh.is_watertight),
        "n_faces": int(len(mesh.faces)),
        "bbox_mm": [round(float(x), 2) for x in ext],
        "volume_mm3": round(float(mesh.volume), 2) if mesh.is_watertight else None,
    }
    mw = min_wall_thickness(mesh)
    rep["min_wall_mm"] = round(mw, 3) if np.isfinite(mw) else None
    rep["thin_walls"] = (mw < 2 * nozzle) if np.isfinite(mw) else None
    if oriented_up is not None:
        _, needs = _face_overhang(mesh, oriented_up, crit_deg)
        rep["support_frac"] = round(float(np.sum(mesh.area_faces[needs]) / mesh.area), 3)
    return rep


# ---------------------------------------------------------------------------
#  EI/field connection: curved-layer (non-planar) toolpaths
# ---------------------------------------------------------------------------
def curved_layer_rows(V, Q, up):
    """The field-aligned quad ROWS that run ~perpendicular to the build axis `up`
    are candidate NON-PLANAR print layers (curved slicing -> stronger, smoother
    than flat layers, and follows the surface). Returns the quad edges grouped as
    'layer' (perp to up) vs 'riser' (along up), for visualization / toolpath export.
    This is exactly what the field-aligned remesh provides for free."""
    up = np.asarray(up, float); up /= np.linalg.norm(up) + 1e-12
    V = np.asarray(V)
    layer, riser = [], []
    seen = set()
    for q in Q:
        qq = [int(x) for x in q]
        for a, b in zip(qq, qq[1:] + qq[:1]):
            if a == b:
                continue
            e = (a, b) if a < b else (b, a)
            if e in seen:
                continue
            seen.add(e)
            ev = V[b] - V[a]; n = np.linalg.norm(ev)
            if n < 1e-9:
                continue
            (riser if abs((ev / n) @ up) > 0.5 else layer).append(e)
    return {"layer_edges": layer, "riser_edges": riser,
            "n_layer": len(layer), "n_riser": len(riser)}


# ---------------------------------------------------------------------------
#  hollowing / uniform-wall shell (material savings)
# ---------------------------------------------------------------------------
def hollow_shell(body, wall_mm=2.0, pitch_mm=None, drain_dir=None, drain_diam_mm=0.0):
    """Hollow a watertight solid to a UNIFORM-thickness shell (the #1 material/time
    saver in printing). Voxel-erode the solid by `wall_mm` and keep solid MINUS
    eroded -> a closed shell with an inner cavity. Optional drain hole (resin
    prints) along -drain_dir. Returns (shell_mesh, info). Robust (voxel-based, so
    no self-intersection like naive vertex offset on thin/concave parts)."""
    if not body.is_watertight:
        return body, {"ok": False, "reason": "input not watertight"}
    pitch = pitch_mm or max(wall_mm / 3.0, float(body.extents.max()) / 220.0)
    n_erode = max(1, int(round(wall_mm / pitch)))
    try:
        from scipy.ndimage import binary_erosion
        vox = body.voxelized(pitch=pitch).fill()
        mat = np.asarray(vox.matrix, bool)
        eroded = binary_erosion(mat, iterations=n_erode)
        if not eroded.any():
            return body, {"ok": False, "reason": "wall too thick (nothing left to hollow)"}
        inner = trimesh.voxel.VoxelGrid(eroded, transform=vox.transform).marching_cubes
        inner.fix_normals()
        inner_in = inner.copy(); inner_in.invert()           # cavity wall faces inward
        shell = trimesh.util.concatenate([body, inner_in])
        v_solid = float(body.volume)
        # material savings from the robust VOXEL-COUNT ratio (mesh volume of the
        # fine inner marching-cubes is unreliable; voxel counts are exact).
        saved_frac = float(eroded.sum()) / max(float(mat.sum()), 1.0)
        info = {"ok": True, "wall_mm": wall_mm,
                "solid_cm3": round(v_solid / 1000, 2),
                "shell_cm3": round(v_solid * (1 - saved_frac) / 1000, 2),
                "material_saved_pct": round(100 * saved_frac, 1)}
    except Exception as e:
        return body, {"ok": False, "reason": str(e)}
    if drain_diam_mm > 0 and drain_dir is not None:
        try:
            g = np.asarray(drain_dir, float); g /= np.linalg.norm(g) + 1e-12
            ctr = body.centroid - g * (body.extents.max())
            cyl = trimesh.creation.cylinder(radius=drain_diam_mm / 2,
                                            height=2 * body.extents.max())
            cyl.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], g))
            cyl.apply_translation(ctr)
            shell = shell.difference(cyl)
            info["drain_hole_mm"] = drain_diam_mm
        except Exception:
            pass
    return shell, info


# ---------------------------------------------------------------------------
#  full print-optimized pipeline
# ---------------------------------------------------------------------------
def quads_to_trimesh(V, Q):
    """Triangulate a quad mesh into a WATERTIGHT trimesh (merge shared vertices so
    watertightness is preserved; consistent per-quad diagonal)."""
    tris = []
    for q in Q:
        u = list(dict.fromkeys(int(x) for x in q))
        if len(u) >= 3:
            tris.append([u[0], u[1], u[2]])
        if len(u) == 4:
            tris.append([u[0], u[2], u[3]])
    body = trimesh.Trimesh(np.asarray(V), np.array(tris), process=True)  # merges verts
    try:
        body.merge_vertices()
        body.update_faces(body.nondegenerate_faces())   # drop zero-area tris
        body.update_faces(body.unique_faces())          # drop duplicate/coincident tris
        body.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(body)
        if not body.is_watertight:
            trimesh.repair.fill_holes(body)
            trimesh.repair.fix_normals(body)
    except Exception:
        pass
    return body


def print_optimize(mesh, target_quads=1500, mode="aligned", nozzle=0.4,
                   print_size_mm=60.0, crit_deg=45.0, verbose=False):
    """Quad-remesh -> scale to a print size -> orientation-optimize -> printability
    report. `print_size_mm` = longest bbox dim of the printed part (so wall/feature
    checks are in real mm vs the nozzle). Returns (V, Q, body, report, orientation)."""
    from . import quad_remesh as qr
    V, Q = qr.quad_remesh(mesh, target_quads=target_quads, mode=mode, verbose=verbose)
    remesh_body = quads_to_trimesh(V, Q)
    in_euler = int(mesh.euler_number)
    remesh_euler = int(remesh_body.euler_number)
    # PRINTABILITY GATE: spurious handles (genus defect) = phantom tunnels =
    # unprintable. The mcf2 extraction can inject them despite being "watertight
    # manifold" (a defect the printing lens exposes that val4 checks miss). If the
    # remesh genus != input genus, print the GENUS-CORRECT cleaned input instead;
    # the field-aligned remesh still provides the curved-layer toolpaths.
    genus_ok = (remesh_euler == in_euler)
    if genus_ok and remesh_body.is_watertight:
        body = remesh_body
    else:
        body = mesh.copy()
        try:
            body.process(); body.merge_vertices()
            trimesh.repair.fix_normals(body)
            if not body.is_watertight:
                trimesh.repair.fill_holes(body); trimesh.repair.fix_normals(body)
        except Exception:
            pass
    # scale so the longest dimension = print_size_mm (units now mm)
    s = print_size_mm / (float(body.bounding_box.extents.max()) + 1e-12)
    body.apply_scale(s); Vs = np.asarray(V) * s
    ori = optimize_orientation(body, crit_deg=crit_deg, verbose=verbose)
    rep = printability_report(body, nozzle=nozzle, oriented_up=ori["up"], crit_deg=crit_deg)
    rep["support_frac_optimized"] = ori["support_frac"]
    # naive baseline (build along +Z) for comparison
    rep["support_frac_z_up"] = round(
        float(np.sum(body.area_faces[_face_overhang(body, [0, 0, 1], crit_deg)[1]]) / body.area), 3)
    rep["input_genus"] = (2 - in_euler) // 2
    rep["remesh_genus"] = (2 - remesh_euler) // 2
    rep["remesh_genus_ok"] = bool(genus_ok and remesh_body.is_watertight)
    rep["print_body"] = "remesh" if rep["remesh_genus_ok"] else "genus-corrected input (remesh had phantom handles)"
    rep["orientation_height_mm"] = round(ori.get("height", 0.0), 1)
    rep["print_estimate"] = print_estimate(body, ori["up"])
    rep["curved_layers"] = curved_layer_rows(Vs, Q, ori["up"])
    return Vs, Q, body, rep, ori


def _lay_flat(body, up):
    """Return a copy of `body` rotated so `up`->+Z and resting on z=0 (print-ready)."""
    b = body.copy()
    b.apply_transform(trimesh.geometry.align_vectors(np.asarray(up, float), [0, 0, 1.0]))
    b.apply_translation([0, 0, -float(b.vertices[:, 2].min())])
    return b


def curved_layer_paths(V, Q, up, n_layers=40):
    """Extract NON-PLANAR (curved) print layers from the field-aligned quad rows:
    bin by height (vertex.up), and within each band collect the 'layer' edges (the
    field rows perpendicular to the build axis) as 3D toolpath segments that FOLLOW
    THE SURFACE rather than a flat plane. Returns list of (height, segments)."""
    cl = curved_layer_rows(V, Q, up)
    V = np.asarray(V); up = np.asarray(up, float); up /= np.linalg.norm(up) + 1e-12
    h = V @ up
    lo, hi = float(h.min()), float(h.max())
    edges = cl["layer_edges"]
    bins = np.linspace(lo, hi, n_layers + 1)
    out = []
    for li in range(n_layers):
        segs = [(int(a), int(b)) for (a, b) in edges
                if bins[li] <= 0.5 * (h[a] + h[b]) < bins[li + 1]]
        if segs:
            out.append((0.5 * (bins[li] + bins[li + 1]), segs))
    return out


def export_print_ready(body, ori, V, Q, outdir, name="part", wall_mm=2.0):
    """Write a print-ready bundle: oriented solid STL, oriented hollow-shell STL,
    curved-layer toolpaths (OBJ polylines), and a JSON report."""
    import json
    os.makedirs(outdir, exist_ok=True)
    up = ori["up"]
    solid = _lay_flat(body, up)
    solid.export(os.path.join(outdir, f"{name}_oriented.stl"))
    files = [f"{name}_oriented.stl"]
    shell, sinfo = hollow_shell(body, wall_mm=wall_mm, drain_dir=-np.asarray(up),
                                drain_diam_mm=3.0)
    if sinfo.get("ok"):
        _lay_flat(shell, up).export(os.path.join(outdir, f"{name}_hollow.stl"))
        files.append(f"{name}_hollow.stl")
    # curved-layer toolpaths -> OBJ polylines (laid flat)
    R = trimesh.geometry.align_vectors(np.asarray(up, float), [0, 0, 1.0])
    Vf = trimesh.transform_points(np.asarray(V), R)
    Vf[:, 2] -= Vf[:, 2].min()
    layers = curved_layer_paths(V, Q, up)
    with open(os.path.join(outdir, f"{name}_curved_layers.obj"), "w", encoding="utf-8") as f:
        for p in Vf:
            f.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        for _, segs in layers:
            for a, b in segs:
                f.write(f"l {a+1} {b+1}\n")
    files.append(f"{name}_curved_layers.obj")
    rep = {"oriented_up": [round(float(x), 4) for x in up],
           "files": files, "hollow": sinfo, "n_curved_layers": len(layers)}
    with open(os.path.join(outdir, f"{name}_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    return rep
