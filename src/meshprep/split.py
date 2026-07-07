"""Bed-fit check + smart CoACD bed-split with hidden seams and reassembly connectors.

Public interface (stable):

    split_for_bed(mesh, profile, *, connectors=True, out_dir=None,
                  smart=False, load_case=None, fit_mm=0.4)
        -> {"ok": bool, "parts": [ {...}, ... ], "note": str[, "seams", "connectors"]}

    plan_seams(mesh, parts, *, load_case=None)      -> scored seam planes
    plan_connectors(parts, seams, *, fit_mm=0.4, style="peg")
                                                    -> parts with peg/socket features
    fit_coupon(out_path=None, *, clearances=..., peg_d=4.0)
                                                    -> printable clearance-sweep coupon

Two paths, both never-raise:

  * CHEAP path (default, ``smart=False``) — axis-aligned bounding-box fit test
    vs. bed volume (best axis PERMUTATION, a heuristic: no free-rotation packing,
    no diagonal placement). If oversized and ``coacd`` is importable, run an
    approximate convex decomposition and REPORT the parts (loose convex chunks,
    not rejoined). Missing ``coacd`` -> honest stub note, never installs anything.

  * SMART path (``smart=True``) — chains: CoACD topology proposal -> seam-plane
    SCORING/refinement -> EXACT boolean cut on the ORIGINAL mesh -> reassembly
    CONNECTORS -> per-part printability gate + bed-fit. Degrades down a labeled
    FALLBACK LADDER (boolean failure -> plane cut without connectors -> plain
    CoACD report), never crashes.

Never-raise discipline: any internal failure degrades to
``{"ok": False, "parts": [], "note": "..."}``.

WHAT IS REAL vs FUTURE (this module)
====================================
REAL now:
  * plan_seams  — candidate planes from adjacent CoACD part interfaces (+/-10%
    normal sweep) AND from concave neck-cut waists on the original mesh; scored
    by a RELATIVE/COMPARATIVE heuristic (concavity - load - area, weights below).
  * exact boolean cut on the original mesh via manifold3d, with a manifold-safety
    FIXED-POINT repair loop and a trimesh plane-cut fallback (the kill-criterion
    degrade — repaired AI meshes can defeat manifold3d's strict validator).
  * plan_connectors — cylindrical peg (d=4 mm) / socket (d=4+fit mm) pairs at
    maximally-spread interior seam points, via manifold3d union/difference.
  * fit_coupon — a one-print clearance-sweep test coupon (same one-coupon
    philosophy as warp calibration).
FUTURE (documented, NOT implemented):
  * print-in-place DOVETAILS / snap-fits (pegs are the simplest thing that works
    and ship first).
  * free-rotation bed packing (the fit test is AABB-permutation only).

NEG-CUT MACHINERY: the concave-waist scoring (min-cut waist length + tightness
floor 1/(2*pi)) is ADAPTED/inlined below from Forge/scripts/_neck_cut.py; it is
kept in this file (rather than a separate core module) to hold this change inside
the module this lane owns. scipy (a meshprep dependency) backs the min-cut. The
curvature primitives are imported as-is from core.curvature_decompose.

SCORING HONESTY: every seam score is a RELATIVE/heuristic ranking, NOT a
certified geometric quality. min_fos from the load term is a COMPARATIVE
screening metric (see _print_fem). Nothing here asserts a calibrated number.

VALIDATION GATES (exercised by ``python -m`` self-check at the bottom):
  G1 volume conservation: a boolean cut of a box conserves volume within 1%.
  G3 seam-quality control: on a dumbbell the top seam lands on the NECK, not
     through a sphere (exercises the exact failure a naive mid-bbox cut makes).
KILL CRITERION (honored): if manifold3d booleans are unreliable on the repaired
geometry, fall back to plane cuts WITHOUT connectors and say so per part.
"""
from __future__ import annotations

import os
import time

import numpy as np

# Generic FDM fallback bed (mm) when the profile carries no bed dimensions.
_GENERIC_BED_MM = (220.0, 220.0, 250.0)

# Compute safety (16 GB shared box): decompose on a decimated proxy.
_MAX_FACES = 6000
# FEM element cap for the load term (heavy-job memory discipline).
_MAX_FEM_ELEM = 2500

# Seam score weights (RELATIVE / COMPARATIVE heuristic; documented, tunable).
#   score = _W_CONCAVITY*concavity - _W_LOAD*load - _W_AREA*area_norm
# concavity : mean concave dihedral along the plane/surface loop (hidden seam).
# load      : mean normalized inv-FOS of FEM cells in the cut slab (avoid the
#             load path); 0 and skipped when no load case is given.
# area_norm : cross-section area / bbox reference (smaller glue face = better).
_W_CONCAVITY = 0.5
_W_LOAD = 0.35
_W_AREA = 0.15

# Connector geometry (mm). Peg nominal d=4; socket d = peg + fit_mm.
_PEG_D = 4.0
_WALL_MM = 0.8              # ~2 perimeters at a 0.4 nozzle
_PEG_PROTRUDE = 4.0        # peg protrusion length each side of the seam
_MAX_CONNECTORS = 3        # 2-3 alignment features per seam

# Greedy smart-split cut budget (memory / scope cap).
_MAX_SEAM_CUTS = 4

# Smart-split wall-clock budget (s): a large pathological mesh (repeated
# curvature + boolean cuts) can otherwise run unbounded; on overrun the pipeline
# stops and returns a labeled best-effort partial split (never hangs).
_SMART_BUDGET_S = 90.0
# Cap on scored seam candidates (bounds O(pairs^2) section work on big meshes).
_MAX_SEAM_CANDS = 120

# Neck-cut tightness floor: hemisphere-equator isoperimetric reference.
_TIGHT_FLOOR = 1.0 / (2.0 * np.pi)


# ======================================================================
# Bed / IO helpers (unchanged CHEAP-path core)
# ======================================================================
def _bed_mm(profile):
    """Extract (x, y, z) bed dimensions in mm from whatever `profile` is.

    Accepts a profile NAME (resolved via meshprep.profiles, lazily), a dict,
    an object with attributes, or a bare 2/3-sequence of mm. Returns
    ``((x, y, z), label)``; falls back to the generic bed, never raises.
    """
    label = "profile"
    try:
        if isinstance(profile, str):
            label = profile
            try:
                from meshprep.profiles import get_profile  # product layer, lazy
                profile = get_profile(profile)
            except Exception:
                return _GENERIC_BED_MM, f"{label} (profiles unavailable -> generic bed)"
        if profile is None:
            return _GENERIC_BED_MM, "generic bed (no profile)"
        if isinstance(profile, (tuple, list)) and len(profile) in (2, 3):
            dims = [float(v) for v in profile]
            if len(dims) == 2:
                dims.append(_GENERIC_BED_MM[2])
            return tuple(dims), "explicit bed"

        def pick(container, key):
            if isinstance(container, dict):
                return container.get(key)
            return getattr(container, key, None)

        for key in ("bed_mm", "bed_size_mm", "bed"):
            v = pick(profile, key)
            if v is not None:
                dims = [float(x) for x in v]
                if len(dims) == 2:
                    dims.append(_GENERIC_BED_MM[2])
                return tuple(dims[:3]), pick(profile, "name") or label
        xyz = [pick(profile, k) for k in ("bed_x", "bed_y", "bed_z")]
        if all(v is not None for v in xyz):
            return tuple(float(v) for v in xyz), pick(profile, "name") or label
    except Exception:
        pass
    return _GENERIC_BED_MM, "generic bed (profile unreadable)"


def _fits(extents, bed):
    """Axis-permutation AABB fit: sorted extents <= sorted bed, elementwise.
    Exact for axis-aligned placements; a heuristic overall (no free rotation)."""
    return all(e <= b + 1e-9 for e, b in zip(sorted(extents), sorted(bed)))


def _load(mesh):
    """Accept a trimesh.Trimesh (anything with .faces/.vertices) or a path."""
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        return mesh
    import trimesh  # lazy (heavy)
    m = trimesh.load(str(mesh), force="mesh")
    if not hasattr(m, "faces"):
        raise ValueError(f"not a triangle mesh: {mesh}")
    return m


def _unit(v):
    v = np.asarray(v, float)
    nn = float(np.linalg.norm(v))
    return v / nn if nn > 1e-12 else np.array([0.0, 0.0, 1.0])


# ======================================================================
# manifold3d bridge + manifold-safety fixed-point repair
# ======================================================================
def _to_manifold(mesh):
    """Build a manifold3d.Manifold from a trimesh. Returns (man|None, err|None).
    man is None if construction failed or produced an empty (rejected) Manifold.
    manifold3d's constructor is a STRICT oriented-2-manifold validator: a
    trimesh-'watertight' mesh with reversed-winding duplicate faces is rejected
    (returns empty) rather than repaired -- that is the documented AI-mesh KILL."""
    try:
        import manifold3d as m3
        v = np.asarray(mesh.vertices, dtype=np.float32)
        f = np.asarray(mesh.faces, dtype=np.uint32)
        if len(f) == 0:
            return None, "empty mesh"
        mgl = m3.Mesh(vert_properties=v, tri_verts=f)
        man = m3.Manifold(mgl)
        try:
            st = man.status()
            ok = (st == m3.Error.NoError)
            st_name = str(st)
        except Exception:
            ok, st_name = True, "unknown"
        if man.is_empty() or man.num_tri() == 0 or not ok:
            return None, st_name
        return man, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _manifold_to_trimesh(man):
    import trimesh
    mesh = man.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]
    f = np.asarray(mesh.tri_verts, dtype=np.int64)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def _manifold_safety_repair(mesh, *, max_iters=4):
    """Fixed-point cleanup so manifold3d's strict validator accepts the geometry.

    Root cause (documented on repaired AI meshes): the mesh is trimesh-watertight
    but is_winding_consistent=False because ~1-2% of faces are exact-position
    DUPLICATES with reversed winding, plus a few near-zero-area degenerates. A
    SINGLE cleanup pass fixes winding but BREAKS watertightness (the duplicates
    masked real holes) -- so this iterates dedupe -> refill -> re-check to a fixed
    point. Returns a NEW trimesh; returns the input on total failure (never raises).
    """
    import trimesh
    try:
        work = mesh.copy()
    except Exception:
        return mesh
    try:
        for _ in range(max_iters):
            V = np.asarray(work.vertices, float)
            F = np.asarray(work.faces, np.int64)
            if len(F) == 0:
                break
            already = bool(work.is_winding_consistent) and bool(work.is_watertight)
            if already:
                break
            # drop near-zero-area degenerate triangles
            tri = V[F]
            areas = 0.5 * np.linalg.norm(
                np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
            pos_areas = areas[areas > 0]
            amed = float(np.median(pos_areas)) if len(pos_areas) else 1.0
            keep = areas > max(1e-12, 1e-6 * amed)
            # drop exact-duplicate AND reversed-winding duplicate faces: keying on
            # the SORTED vertex triple collapses (a,b,c) and (a,c,b) to one key.
            sortedF = np.sort(F, axis=1)
            _, first_idx = np.unique(sortedF, axis=0, return_index=True)
            dup_keep = np.zeros(len(F), bool)
            dup_keep[first_idx] = True
            keep = keep & dup_keep
            newF = F[keep]
            w2 = trimesh.Trimesh(vertices=V, faces=newF, process=False)
            try:
                w2.remove_unreferenced_vertices()
            except Exception:
                pass
            # refill the holes the removed faces unmasked, then re-orient
            try:
                w2.fill_holes()
            except Exception:
                pass
            try:
                trimesh.repair.fix_normals(w2)
            except Exception:
                pass
            # no progress guard: if nothing changed and still bad, stop
            if len(newF) == len(F) and not (
                    bool(w2.is_winding_consistent) and bool(w2.is_watertight)):
                work = w2
                break
            work = w2
            if bool(work.is_winding_consistent) and bool(work.is_watertight):
                break
        return work
    except Exception:
        return mesh


def _build_manifold_or_repair(mesh):
    """L1 (direct) then L2 (manifold-safety repair + retry).
    Returns (man|None, repaired_mesh, note)."""
    man, err = _to_manifold(mesh)
    if man is not None:
        return man, mesh, "manifold3d accepted the mesh directly"
    fixed = _manifold_safety_repair(mesh)
    man2, err2 = _to_manifold(fixed)
    if man2 is not None:
        return man2, fixed, "manifold3d accepted after manifold-safety repair"
    return None, fixed, (f"manifold3d rejected geometry (L1={err}; "
                         f"after repair={err2})")


# ======================================================================
# Neck-cut waist machinery (adapted/inlined from Forge/scripts/_neck_cut.py)
# ======================================================================
def _waist_mincut(nf, fadj, elen, sideA, sideB):
    """Length of the shortest separating loop (the WAIST) between two face sets,
    via a min-cut on the face-dual graph (edge capacity = shared-edge length).
    scipy-backed. Returns None on failure."""
    try:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import maximum_flow
        SCALE, BIG = 1000.0, 10 ** 9
        rows, cols, caps = [], [], []
        for (a, b), L in zip(fadj, elen):
            c = max(int(round(float(L) * SCALE)), 1)
            rows += [int(a), int(b)]
            cols += [int(b), int(a)]
            caps += [c, c]
        S, T = nf, nf + 1
        for f in sideA:
            rows += [S]
            cols += [int(f)]
            caps += [BIG]
        for f in sideB:
            rows += [int(f)]
            cols += [T]
            caps += [BIG]
        g = sp.csr_matrix((caps, (rows, cols)), shape=(nf + 2, nf + 2),
                          dtype=np.int64)
        return maximum_flow(g, S, T).flow_value / SCALE
    except Exception:
        return None


def _face_components(face_ids, fadj_pairs):
    """Connected components of a SUBSET of faces under face adjacency."""
    from collections import defaultdict
    s = set(int(f) for f in face_ids)
    parent = {f: f for f in s}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in fadj_pairs:
        a, b = int(a), int(b)
        if a in s and b in s:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    comps = defaultdict(list)
    for f in s:
        comps[find(f)].append(f)
    return list(comps.values())


def _neck_cuts(mesh, *, conc_rel=0.05, tight_thr=_TIGHT_FLOOR,
               min_part_frac=0.01, min_band_frac=0.003, smooth_iters=2):
    """Accepted concave necks (each = dict band/sides/tightness/perim...).
    Adapted from Forge _neck_cut.neck_cuts; returns [] on any failure."""
    try:
        from .core.curvature_decompose import (principal_curvatures,
                                                smooth_curvature_field)
        F = np.asarray(mesh.faces, dtype=np.int64)
        nf = len(F)
        if nf == 0:
            return []
        fa = np.asarray(mesh.area_faces, dtype=np.float64)
        total = float(fa.sum())
        fadj = np.asarray(mesh.face_adjacency, dtype=np.int64)
        fae = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
        V = np.asarray(mesh.vertices, dtype=np.float64)
        if len(fadj) == 0:
            return []
        elen = np.linalg.norm(V[fae[:, 0]] - V[fae[:, 1]], axis=1)

        k1, k2, _ = principal_curvatures(mesh)
        k1, k2 = smooth_curvature_field(mesh, k1, k2, smooth_iters)
        scale = float(np.percentile(np.abs(k1), 75)) or 1.0
        neck_v = k2 < (-conc_rel * scale)
        neck_f = neck_v[F].any(axis=1)
        band_ids = np.flatnonzero(neck_f)
        if len(band_ids) == 0:
            return []

        bands = _face_components(band_ids, fadj)
        all_faces = set(range(nf))
        accepted = []
        for band in bands:
            if len(band) < min_band_frac * nf:
                continue
            band_set = set(band)
            rest = list(all_faces - band_set)
            comps = _face_components(rest, fadj)
            if len(comps) < 2:
                continue
            comp_area = [float(fa[c].sum()) for c in comps]
            order = np.argsort(comp_area)
            small_comp = set(comps[order[0]])
            big_comp = set(comps[order[-1]])
            min_side = comp_area[order[0]]
            perim = _waist_mincut(nf, fadj, elen, big_comp, small_comp)
            if perim is None or perim <= 1e-9:
                continue
            tight = min_side / (perim ** 2)
            if tight >= tight_thr and min_side >= min_part_frac * total:
                accepted.append(dict(band=band, sides=comps, tightness=tight,
                                     min_side_frac=min_side / total, perim=perim))
        return accepted
    except Exception:
        return []


def _neck_plane(mesh, neck):
    """Cut plane for a neck: origin at the band centroid, normal along the
    axis separating the two largest sides. Returns (origin, normal) or (None,None)."""
    try:
        F = np.asarray(mesh.faces, np.int64)
        V = np.asarray(mesh.vertices, float)
        fc = V[F].mean(axis=1)
        band = neck.get("band", [])
        sides = neck.get("sides", [])
        if not band or len(sides) < 2:
            return None, None
        order = np.argsort([len(s) for s in sides])
        small = np.asarray(sides[order[0]], int)
        big = np.asarray(sides[order[-1]], int)
        cs = fc[small].mean(axis=0)
        cb = fc[big].mean(axis=0)
        n = _unit(cb - cs)
        o = fc[np.asarray(band, int)].mean(axis=0)
        return o, n
    except Exception:
        return None, None


# ======================================================================
# Plane geometry: cross-section, concavity, load, intersection
# ======================================================================
def _section_polygon(mesh, origin, normal):
    """Largest cross-section polygon (shapely) of `mesh` at the plane, plus the
    planar->3D 4x4 transform. Returns (polygon|None, to_3D|None). Never raises."""
    try:
        sec = mesh.section(plane_origin=np.asarray(origin, float),
                           plane_normal=np.asarray(normal, float))
        if sec is None:
            return None, None
        try:
            p2, to3d = sec.to_2D()
        except Exception:
            p2, to3d = sec.to_planar()
        polys = list(p2.polygons_full)
        if not polys:
            return None, None
        poly = max(polys, key=lambda p: p.area)
        return poly, np.asarray(to3d, float)
    except Exception:
        return None, None


def _plane_cross_area(mesh, origin, normal):
    poly, _ = _section_polygon(mesh, origin, normal)
    if poly is None:
        return float("nan")
    return float(poly.area)


def _plane_concavity(mesh, origin, normal):
    """Mean SIGNED dihedral along the plane/surface intersection loop:
    +angle where the loop crosses a concave edge (hidden seam), -angle where
    convex. Higher = more hidden. RELATIVE heuristic. Never raises."""
    try:
        V = np.asarray(mesh.vertices, float)
        F = np.asarray(mesh.faces, np.int64)
        fadj = np.asarray(mesh.face_adjacency, np.int64)
        if len(fadj) == 0:
            return 0.0
        fc = V[F].mean(axis=1)
        nn = _unit(normal)
        o = np.asarray(origin, float)
        d = (fc - o) @ nn
        sa = np.sign(d[fadj[:, 0]])
        sb = np.sign(d[fadj[:, 1]])
        cross = sa != sb
        if not np.any(cross):
            return 0.0
        ang = np.asarray(mesh.face_adjacency_angles, float)[cross]
        convex = np.asarray(mesh.face_adjacency_convex, bool)[cross]
        signed = np.where(convex, -ang, ang)
        return float(np.mean(signed))
    except Exception:
        return 0.0


def _plane_intersects(mesh, origin, normal):
    """True if the plane actually separates the mesh (vertices on both sides)."""
    try:
        V = np.asarray(mesh.vertices, float)
        d = (V - np.asarray(origin, float)) @ _unit(normal)
        return bool(d.min() < -1e-9 and d.max() > 1e-9)
    except Exception:
        return False


def _load_importance(mesh, load_case):
    """FEM inv-FOS importance field for the load term. Returns
    {centroids, inv, pitch, thr} or None (FEM unavailable/failed)."""
    try:
        from .core import _print_fem
        res = _print_fem.strength_analysis(mesh, load_case=load_case,
                                           max_elem=_MAX_FEM_ELEM)
        if not res.get("ok"):
            return None
        cents = np.asarray(res["centroids"], float)
        fos = np.asarray(res["fos"], float)
        if len(cents) == 0:
            return None
        inv = 1.0 / np.clip(fos, 1e-6, None)
        lo, hi = np.percentile(inv, 5), np.percentile(inv, 95)
        invn = np.clip((inv - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        try:
            pitch = float(res["pitch"])
        except Exception:
            pitch = float(np.cbrt(max(np.prod(mesh.extents), 1e-9)
                                  / max(len(cents), 1)))
        return {"centroids": cents, "inv": invn, "pitch": pitch,
                "thr": float(np.percentile(invn, 70))}
    except Exception:
        return None


def _plane_load(mesh, origin, normal, field):
    """(mean_inv_fos_in_slab, crosses_load_path) for cells near the plane."""
    try:
        c = field["centroids"]
        invn = field["inv"]
        pitch = field["pitch"]
        d = (c - np.asarray(origin, float)) @ _unit(normal)
        band = np.abs(d) <= max(pitch, 1e-6)
        if not np.any(band):
            return 0.0, False
        ml = float(invn[band].mean())
        return ml, bool(ml > field["thr"])
    except Exception:
        return 0.0, False


# ======================================================================
# Part-interface helpers
# ======================================================================
def _as_part_meshes(parts):
    """Coerce CoACD parts (trimesh | (verts,faces) | dict-with-file) to trimesh."""
    import trimesh
    out = []
    for p in parts or []:
        try:
            if hasattr(p, "vertices") and hasattr(p, "faces"):
                out.append(p)
            elif isinstance(p, (tuple, list)) and len(p) == 2:
                v, f = p
                out.append(trimesh.Trimesh(vertices=np.asarray(v, float),
                                           faces=np.asarray(f), process=False))
            elif isinstance(p, dict) and p.get("file"):
                out.append(trimesh.load(p["file"], force="mesh"))
        except Exception:
            continue
    return out


def _parts_adjacent(a, b, *, tol_frac=0.05):
    """Cheap adjacency: AABB overlap (with a small tolerance) of two parts."""
    try:
        amin, amax = a.bounds
        bmin, bmax = b.bounds
        diag = float(np.linalg.norm(amax - amin) + np.linalg.norm(bmax - bmin))
        tol = tol_frac * diag
        return bool(np.all(amin - tol <= bmax) and np.all(bmin - tol <= amax))
    except Exception:
        return True


def _dedup_best(seams):
    """Collapse the +/-10% sweep offsets of one interface to its best-scoring one."""
    best = {}
    for s in seams:
        k = s.get("source", "")
        if k not in best or s["score"] > best[k]["score"]:
            best[k] = s
    return list(best.values())


# ======================================================================
# plan_seams (REAL)
# ======================================================================
def plan_seams(mesh, parts, *, load_case=None):
    """Score and rank seam planes for a smart split.

    CANDIDATES: for each adjacent CoACD part pair, the interface plane swept
    +/-10% along its normal (5 offsets); plus concave neck-cut WAIST planes on
    the ORIGINAL mesh (where a seam is invisible).

    SCORE (all RELATIVE / heuristic):
      (a) CONCAVITY: mean concave dihedral along the plane/surface loop.
      (b) LOAD: if `load_case` given, penalize planes whose cut slab overlaps
          high inv-FOS (weak) cells; sets crosses_load_path. No load case ->
          term skipped with a plain note.
      (c) AREA: smaller cross-section = less glue face = better.
      score = _W_CONCAVITY*concavity - _W_LOAD*load - _W_AREA*area_norm  (COMPARATIVE)

    Returns {"ok", "seams": [{"plane": (origin, normal), "concavity_score",
    "crosses_load_path", "area_mm2", "score", "source"}], "note"}.
    Honest refusal (ok=False) if no scorable interface. Never raises.
    """
    try:
        m = _load(mesh)
        # Bound scoring cost on large meshes: score on a decimated proxy (the
        # score is a RELATIVE ranking heuristic, so the proxy ranks the same
        # seam planes; cut planes are geometry-space and stay valid). Prevents
        # the repeated section/concavity + curvature work from running unbounded.
        try:
            if len(m.faces) > _MAX_FACES:
                from .core._mesh_util import decimate
                m = decimate(m, max_faces=_MAX_FACES)
        except Exception:
            pass
        pmeshes = _as_part_meshes(parts)
        cands = []  # (origin, normal, source)

        # (A) adjacent CoACD part-pair interfaces, swept +/-10% along the normal
        cents = [np.asarray(pm.vertices, float).mean(axis=0) for pm in pmeshes]
        for i in range(len(pmeshes)):
            for j in range(i + 1, len(pmeshes)):
                if not _parts_adjacent(pmeshes[i], pmeshes[j]):
                    continue
                ci, cj = cents[i], cents[j]
                sep = float(np.linalg.norm(cj - ci))
                if sep <= 1e-9:
                    continue
                n = _unit(cj - ci)
                mid = 0.5 * (ci + cj)
                for t in np.linspace(-0.10, 0.10, 5):
                    cands.append((mid + t * sep * n, n, f"coacd_pair_{i}_{j}"))

        # (B) concave neck-cut waist seeds on the ORIGINAL mesh
        for kk, nk in enumerate(_neck_cuts(m)):
            o, n = _neck_plane(m, nk)
            if o is not None:
                cands.append((o, n, f"neck_waist_{kk}"))

        if not cands:
            return {"ok": False, "seams": [],
                    "note": "no scorable interface: no adjacent CoACD part pairs "
                            "and no concave neck waist found (never raises)."}
        if len(cands) > _MAX_SEAM_CANDS:
            # bound section work on big configs, but never drop hidden neck seams
            necks = [c for c in cands if c[2].startswith("neck_")]
            pairs = [c for c in cands if not c[2].startswith("neck_")]
            cands = pairs[:max(_MAX_SEAM_CANDS - len(necks), 0)] + necks

        note_bits = []
        load_field = None
        if load_case is not None:
            load_field = _load_importance(m, load_case)
            if load_field is None:
                note_bits.append("load-path avoidance: not evaluated "
                                 "(FEM unavailable/failed for this mesh).")
        else:
            note_bits.append("load-path avoidance: not evaluated (no load case).")

        bbox = np.asarray(m.extents, float)
        area_ref = float(np.median(bbox)) ** 2 or 1.0

        seams = []
        for (o, n, src) in cands:
            area = _plane_cross_area(m, o, n)
            if not (area == area) or area <= 0.0:
                continue  # plane does not actually cut the surface
            conc = _plane_concavity(m, o, n)
            crosses, load = False, 0.0
            if load_field is not None:
                load, crosses = _plane_load(m, o, n, load_field)
            area_norm = min(area / area_ref, 1.0)
            score = (_W_CONCAVITY * conc - _W_LOAD * load - _W_AREA * area_norm)
            seams.append({
                "plane": (tuple(float(x) for x in o), tuple(float(x) for x in n)),
                "score": float(score),
                "concavity_score": float(conc),
                "crosses_load_path": bool(crosses),
                "area_mm2": float(area),
                "source": src,
            })

        if not seams:
            return {"ok": False, "seams": [],
                    "note": "no candidate plane produced a valid cross-section "
                            "(no scorable interface); never raises."}

        seams = _dedup_best(seams)
        seams.sort(key=lambda s: s["score"], reverse=True)
        note_bits.insert(
            0, f"{len(seams)} seam candidate(s) scored (RELATIVE/COMPARATIVE "
               f"heuristic: {_W_CONCAVITY}*concavity - {_W_LOAD}*load - "
               f"{_W_AREA}*area_norm).")
        return {"ok": True, "seams": seams, "note": " ".join(note_bits)}
    except Exception as e:
        return {"ok": False, "seams": [], "note": f"seam planning failed: {e}"}


# ======================================================================
# Boolean cut ladder
# ======================================================================
def _cut_by_plane(mesh, origin, normal):
    """Cut `mesh` by a plane into (pos, neg) halves down the fallback ladder.

    L1 manifold3d boolean split on the ORIGINAL geometry ->
    L2 manifold-safety repair + retry (inside _build_manifold_or_repair) ->
    L3 trimesh plane slice WITHOUT connectors (the labeled KILL degrade).

    Returns {"ok", "pos", "neg", "method": "boolean"|"plane",
    "connectors_ok": bool, "note"}. Never raises."""
    try:
        n = _unit(normal)
        o = np.asarray(origin, float)
        offset = float(np.dot(n, o))
        man, _fixed, mnote = _build_manifold_or_repair(mesh)
        if man is not None:
            try:
                pos, neg = man.split_by_plane([float(x) for x in n], offset)
                if pos.num_tri() > 0 and neg.num_tri() > 0:
                    return {"ok": True, "pos": _manifold_to_trimesh(pos),
                            "neg": _manifold_to_trimesh(neg), "method": "boolean",
                            "connectors_ok": True, "note": mnote}
            except Exception as e:
                mnote = f"{mnote}; boolean split raised: {e}"
        # L3 fallback: trimesh plane slice, connectors unavailable
        try:
            pos = mesh.slice_plane(o, n, cap=True)
            neg = mesh.slice_plane(o, -n, cap=True)
            if (pos is not None and neg is not None
                    and len(pos.faces) > 0 and len(neg.faces) > 0):
                return {"ok": True, "pos": pos, "neg": neg, "method": "plane",
                        "connectors_ok": False,
                        "note": ("connectors unavailable for this mesh (boolean "
                                 "backend rejected the repaired geometry); parts "
                                 f"are loose plane cuts. [{mnote}]")}
        except Exception as e:
            return {"ok": False, "note": f"boolean and plane-slice both failed: "
                                         f"{e}; [{mnote}]"}
        return {"ok": False, "note": f"plane-slice produced an empty half; [{mnote}]"}
    except Exception as e:
        return {"ok": False, "note": f"cut failed: {e}"}


# ======================================================================
# Connectors (REAL)
# ======================================================================
def _spread_interior_points(poly, to3d, inset, n_points):
    """Up to n_points maximally-spread interior points of the seam polygon,
    inset from the boundary, mapped to 3D. Returns a list of 3-vectors."""
    from shapely.geometry import Point
    core = poly.buffer(-inset)
    if core.is_empty or core.area <= 0:
        core = poly.buffer(-0.1 * (poly.area ** 0.5))  # gentler inset
    if core.is_empty or core.area <= 0:
        return []
    if core.geom_type == "MultiPolygon":
        core = max(core.geoms, key=lambda p: p.area)
    ext = np.asarray(core.exterior.coords)[:-1]
    if len(ext) == 0:
        return []
    c = np.asarray(core.representative_point().coords)[0]
    pts2d = [c]
    if n_points >= 2:
        d = np.linalg.norm(ext - c, axis=1)
        pts2d.append(ext[int(np.argmax(d))])
    if n_points >= 3:
        d2 = np.linalg.norm(ext - pts2d[-1], axis=1)
        pts2d.append(ext[int(np.argmax(d2))])
    out3d = []
    for p in pts2d[:n_points]:
        pp = c + 0.7 * (np.asarray(p, float) - c)  # pull inward -> stay interior
        if not core.contains(Point(pp)):
            pp = c
        world = to3d @ np.array([pp[0], pp[1], 0.0, 1.0])
        out3d.append(world[:3])
    return out3d


def _cylinder_at(point, normal, radius, height):
    """Watertight trimesh cylinder centered at `point`, axis along `normal`."""
    import trimesh
    T = np.asarray(trimesh.geometry.align_vectors([0, 0, 1], _unit(normal)),
                   float).copy()
    T[:3, 3] = np.asarray(point, float)
    return trimesh.creation.cylinder(radius=float(radius), height=float(height),
                                     sections=32, transform=T)


def _seam_section(part, o, n):
    """Cross-section of `part` at the seam plane, nudged OFF the coincident
    cut-cap plane when needed. trimesh.section returns None when the plane is
    exactly a cap face (the seam plane is the boolean cut's own cap), so if the
    on-plane section is empty this steps a small epsilon INTO the body along
    +/-normal and retries. Returns (polygon|None, to_3D|None). Never raises."""
    poly, to3d = _section_polygon(part, o, n)
    if poly is not None:
        return poly, to3d
    try:
        o = np.asarray(o, float)
        nn = _unit(n)
        ext = float(np.max(np.asarray(part.extents, float)))
        step = max(1e-4, 5e-3 * ext)
        for s in (1.0, -1.0, 2.0, -2.0, 4.0, -4.0):
            poly, to3d = _section_polygon(part, o + s * step * nn, nn)
            if poly is not None:
                return poly, to3d
    except Exception:
        pass
    return None, None


def plan_connectors(parts, seams, *, fit_mm=0.4, style="peg"):
    """Add cylindrical peg/socket alignment features across a seam so the two
    parts reassemble without a jig.

    parts : [partA, partB] (trimesh); seams : [seam] (uses seams[0]["plane"]).
    Peg (d=_PEG_D) UNION on the +normal-side part; socket (d=_PEG_D+fit_mm)
    DIFFERENCE on the other -- both exact manifold3d booleans. 2-3 features at
    maximally-spread interior seam points, inset >= 2*wall from the boundary.
    If the boolean backend is degenerate on these parts, REFUSE (ok=False) with a
    labeled note rather than ship mesh that will not reassemble. Never raises.

    Returns {"ok", "parts": [meshes], "n_connectors", "fit_mm", "note"}.
    (Print-in-place dovetails are FUTURE; pegs ship first.)
    """
    try:
        import manifold3d as m3
        if not parts or len(parts) < 2 or not seams:
            return {"ok": False, "parts": list(parts or []), "n_connectors": 0,
                    "fit_mm": float(fit_mm),
                    "note": "connectors need exactly two parts and one seam."}
        partA, partB = parts[0], parts[1]
        o, n = seams[0]["plane"]
        o = np.asarray(o, float)
        n = _unit(n)

        # Orient: A = the part on the +normal side (peg protrudes from A into B).
        try:
            if float((np.asarray(partA.vertices, float).mean(0) - o) @ n) < 0:
                partA, partB = partB, partA
        except Exception:
            pass

        manA, _fa, _na = _build_manifold_or_repair(partA)
        manB, _fb, _nb = _build_manifold_or_repair(partB)
        if manA is None or manB is None:
            return {"ok": False, "parts": list(parts), "n_connectors": 0,
                    "fit_mm": float(fit_mm),
                    "note": ("connectors unavailable: boolean backend rejected a "
                             "part even after manifold-safety repair (RELATIVE); "
                             "parts remain loose plane cuts.")}

        poly, to3d = _seam_section(partA, o, n)
        if poly is None:
            return {"ok": False, "parts": list(parts), "n_connectors": 0,
                    "fit_mm": float(fit_mm),
                    "note": "connectors unavailable: no seam cross-section polygon."}

        peg_r = _PEG_D / 2.0
        sock_r = (_PEG_D + float(fit_mm)) / 2.0
        inset = peg_r + 2.0 * _WALL_MM
        pts = _spread_interior_points(poly, to3d, inset, _MAX_CONNECTORS)
        if len(pts) < 2:
            return {"ok": False, "parts": list(parts), "n_connectors": 0,
                    "fit_mm": float(fit_mm),
                    "note": ("connectors unavailable: seam cross-section too small "
                             "to inset a peg (< 2 spread interior points).")}

        h = 2.0 * _PEG_PROTRUDE
        n_ok = 0
        for pt in pts:
            try:
                peg = _cylinder_at(pt, n, peg_r, h)
                sock = _cylinder_at(pt, n, sock_r, h + 0.4)
                mpeg, _e1 = _to_manifold(peg)
                msock, _e2 = _to_manifold(sock)
                if mpeg is None or msock is None:
                    continue
                newA = manA + mpeg
                newB = manB - msock
                if newA.num_tri() == 0 or newB.num_tri() == 0:
                    continue
                manA, manB = newA, newB
                n_ok += 1
            except Exception:
                continue

        if n_ok < 2:
            return {"ok": False, "parts": list(parts), "n_connectors": n_ok,
                    "fit_mm": float(fit_mm),
                    "note": ("connectors unavailable: fewer than 2 peg/socket "
                             "booleans succeeded on this geometry (RELATIVE).")}

        outA = _manifold_to_trimesh(manA)
        outB = _manifold_to_trimesh(manB)
        return {"ok": True, "parts": [outA, outB], "n_connectors": n_ok,
                "fit_mm": float(fit_mm),
                "note": (f"{n_ok} peg(d={_PEG_D})/socket(d={_PEG_D + fit_mm:.1f}) "
                         f"pair(s) added via manifold3d union/difference. Fit "
                         f"clearance {fit_mm} mm is PRINTER-DEPENDENT -- run "
                         f"`fit_coupon()` once to find your working clearance.")}
    except Exception as e:
        return {"ok": False, "parts": list(parts or []), "n_connectors": 0,
                "fit_mm": float(fit_mm), "note": f"connector generation failed: {e}"}


# ======================================================================
# Fit coupon (pure geometry, one-print calibration)
# ======================================================================
def fit_coupon(out_path=None, *, clearances=(0.2, 0.3, 0.4, 0.5), peg_d=_PEG_D):
    """Emit a small clearance-sweep test coupon the user prints ONCE and then
    records the working peg/socket clearance (same one-coupon philosophy as warp
    calibration). Pure geometry, NO analysis. Never raises.

    Returns {"ok", "mesh"|None, "clearances", "note"[, "file"]}.
    """
    try:
        import manifold3d as m3
        clr = [float(c) for c in clearances]
        pitch = 16.0
        plate_t = 3.0
        n = len(clr)
        Wx = n * pitch + 8.0
        Wy = 30.0
        plate = m3.Manifold.cube([Wx, Wy, plate_t], True).translate(
            [0, 0, plate_t / 2.0])
        acc = plate
        for i, c in enumerate(clr):
            x = -Wx / 2.0 + 8.0 + i * pitch
            # standing peg (union onto the plate)
            peg = m3.Manifold.cylinder(6.0, peg_d / 2.0, peg_d / 2.0, 48,
                                       False).translate([x, -7.0, plate_t])
            acc = acc + peg
            # socket block with a through-ish hole (block minus oversized cylinder)
            block = m3.Manifold.cube([10.0, 10.0, 6.0], True).translate(
                [x, 8.0, plate_t + 3.0])
            hole = m3.Manifold.cylinder(7.0, (peg_d + c) / 2.0, (peg_d + c) / 2.0,
                                        48, False).translate(
                [x, 8.0, plate_t + 0.5])
            acc = acc + (block - hole)
        mesh = _manifold_to_trimesh(acc)
        out = {"ok": True, "mesh": mesh, "clearances": clr,
               "note": (f"fit coupon: {n} peg(d={peg_d})/socket pairs at clearances "
                        f"{clr} mm. Print once; the smallest clearance whose socket "
                        f"accepts the peg with a snug slide is your fit_mm. "
                        f"PRINTER-DEPENDENT calibration, no analysis performed.")}
        if out_path is not None:
            try:
                mesh.export(str(out_path))
                out["file"] = str(out_path)
            except Exception as e:
                out["note"] += f" (export failed: {e})"
        return out
    except Exception as e:
        return {"ok": False, "mesh": None, "clearances": list(clearances),
                "note": f"fit coupon build failed: {e}"}


# ======================================================================
# CoACD proposal + printability helpers
# ======================================================================
def _coacd_parts(mesh):
    """CoACD convex decomposition on a decimated proxy. Returns
    ``(parts, proxy_n_faces)`` where parts is a list of trimesh parts (or None if
    CoACD is unavailable/failed) and proxy_n_faces is the decimated-proxy face
    count (or None) -- exposed so the CHEAP path can report the compute-cap note.
    Never raises."""
    try:
        import coacd
    except Exception:
        return None, None
    try:
        import trimesh
        from .core._mesh_util import decimate
        proxy = decimate(mesh, max_faces=_MAX_FACES)
        proxy_nf = int(len(proxy.faces))
        try:
            coacd.set_log_level("error")
        except Exception:
            pass
        raw = coacd.run_coacd(
            coacd.Mesh(np.asarray(proxy.vertices, dtype=np.float64),
                       np.asarray(proxy.faces, dtype=np.int64)),
            threshold=0.05,
        )
        parts = []
        for (pv, pf) in raw:
            parts.append(trimesh.Trimesh(vertices=np.asarray(pv, float),
                                         faces=np.asarray(pf), process=False))
        return parts, proxy_nf
    except Exception:
        return None, None


def _printable(mesh):
    """assess_printability verdict as a bool (never raises)."""
    try:
        from .core._printability import assess_printability
        return bool(assess_printability(mesh).get("printable", False))
    except Exception:
        return False


def _watertight_as_exported(mesh):
    """Watertightness of the part AS THE USER RECEIVES IT: round-trip through an
    STL (face-soup, no shared vertices) and reload exactly as an exported part
    reloads for printing. This is the honest check -- manifold3d/boolean output
    is watertight only in its indexed in-memory form; a peg/socket boolean can
    reload NON-watertight. assess_printability does NOT test this, so the per-part
    gate must. Returns bool. Never raises."""
    import io
    import trimesh
    try:
        buf = io.BytesIO()
        mesh.export(buf, file_type="stl")
        buf.seek(0)
        rt = trimesh.load(buf, file_type="stl", force="mesh")
        return bool(rt.is_watertight)
    except Exception:
        try:
            return bool(mesh.is_watertight)
        except Exception:
            return False


def _finalize_watertight(mesh):
    """Best-effort: make a part watertight AS EXPORTED before the gate judges it.
    Weld coincident vertices, drop duplicate faces, fill small holes, re-orient,
    then verify via the STL round-trip. Returns (mesh_out, watertight_bool). If
    repair cannot close it, returns the (repaired) mesh with False so the gate
    fails it honestly rather than shipping a non-watertight part as PASS."""
    import trimesh
    if _watertight_as_exported(mesh):
        return mesh, True
    try:
        w = mesh.copy()
        try:
            w.merge_vertices()
        except Exception:
            pass
        try:
            w.update_faces(w.unique_faces())
            w.update_faces(w.nondegenerate_faces())
            w.remove_unreferenced_vertices()
        except Exception:
            pass
        try:
            trimesh.repair.fill_holes(w)
        except Exception:
            pass
        try:
            trimesh.repair.fix_normals(w)
        except Exception:
            pass
        return w, _watertight_as_exported(w)
    except Exception:
        return mesh, False


# ======================================================================
# split_for_bed (CHEAP + SMART)
# ======================================================================
def split_for_bed(mesh, profile=None, *, connectors=True, out_dir=None,
                  smart=False, load_case=None, fit_mm=0.4):
    """Check bed fit; if oversized, split into printable parts.

    Parameters
    ----------
    mesh       : trimesh.Trimesh or path to a mesh file.
    profile    : printer profile (name / dict / object / (x,y,z) mm); the bed
                 dimensions are read from it (generic 220x220x250 fallback).
    connectors : add reassembly peg/socket features (SMART path only). When a
                 boolean cut is unavailable the parts stay loose and the note says so.
    out_dir    : if given and a split happens, each part is saved there as
                 ``part_000.stl`` etc. (best-effort).
    smart      : opt-in SMART pipeline (seam scoring -> boolean cut on the
                 original mesh -> connectors -> per-part gate). Default False
                 keeps the CHEAP CoACD-report behavior byte-compatible.
    load_case  : optional {load_face, anchor_face, load_axis, total_force_N} for
                 the seam LOAD term (avoid putting a glue joint on the load path).
    fit_mm     : peg/socket clearance (SMART connectors); exposed for the CLI
                 ``--fit`` param. Printer-dependent; calibrate via ``fit_coupon``.

    CHEAP return: ``{"ok", "parts", "note"}`` where parts entries are
    ``{"index", "n_faces", "extents_mm", "fits_bed"[, "file"]}``.
    SMART return: each part entry additionally carries ``"printable"`` and
    ``"watertight"`` (watertightness measured AS EXPORTED via an STL round-trip,
    because a boolean/connector part is watertight only in indexed in-memory form
    and assess_printability does not test it). It also carries ``"seams"``
    (executed seam records) and ``"connectors": {"n_connectors", "fit_mm",
    "fallback"}``. ``ok`` = every part fits the bed AND is printable AND is
    watertight-as-exported. On a large pathological mesh the pipeline honors an
    internal wall-clock budget and returns a labeled best-effort PARTIAL split.
    """
    try:
        try:
            m = _load(mesh)
        except Exception:
            return {"ok": False, "parts": [],
                    "note": "could not read a triangle mesh from the input -- "
                            "not a valid mesh file"}
        if (getattr(m, "faces", None) is None or len(m.faces) == 0
                or getattr(m, "extents", None) is None):
            return {"ok": False, "parts": [],
                    "note": "input is not a triangle mesh (no faces) -- "
                            "cannot check bed fit"}
        bed, bed_label = _bed_mm(profile)
        ext = [float(v) for v in m.extents]
        if not all(v == v and abs(v) != float("inf") for v in ext):
            return {"ok": False, "parts": [],
                    "note": "mesh bounding box is not finite (NaN/inf "
                            "coordinates) -- cannot check bed fit"}
        ext_s = "x".join(f"{v:.1f}" for v in ext)
        bed_s = "x".join(f"{v:.0f}" for v in bed)

        if _fits(ext, bed):
            return {
                "ok": True,
                "parts": [{"index": 0, "n_faces": int(len(m.faces)),
                           "extents_mm": ext, "fits_bed": True}],
                "note": (f"no split needed: {ext_s} mm fits {bed_s} mm bed "
                         f"({bed_label}; AABB-permutation fit, heuristic)"),
            }

        if smart:
            return _smart_split(m, bed, bed_label, ext_s, bed_s,
                                connectors=connectors, out_dir=out_dir,
                                load_case=load_case, fit_mm=fit_mm)

        # ---- CHEAP path: CoACD report (loose convex chunks) -----------------
        parts_topo, proxy_nf = _coacd_parts(m)
        if parts_topo is None:
            try:
                import coacd  # noqa: F401  (distinguish "missing" from "failed")
                reason = ("CoACD ran but produced no parts")
            except Exception:
                reason = ("CoACD is not installed - install the extra "
                          "`meshprep[split]` to enable bed-splitting")
            return {"ok": False, "parts": [],
                    "note": (f"mesh {ext_s} mm exceeds {bed_s} mm bed ({bed_label}) "
                             f"and {reason}")}

        parts, saved = [], 0
        for i, pm in enumerate(parts_topo):
            pext = [float(v) for v in pm.extents]
            entry = {"index": i, "n_faces": int(len(pm.faces)),
                     "extents_mm": pext, "fits_bed": _fits(pext, bed)}
            if out_dir is not None:
                try:
                    os.makedirs(out_dir, exist_ok=True)
                    fp = os.path.join(str(out_dir), f"part_{i:03d}.stl")
                    pm.export(fp)
                    entry["file"] = fp
                    saved += 1
                except Exception:
                    pass
            parts.append(entry)

        n_fit = sum(p["fits_bed"] for p in parts)
        bits = [f"mesh {ext_s} mm exceeds {bed_s} mm bed ({bed_label}); "
                f"CoACD split into {len(parts)} convex parts, {n_fit} fit the bed"]
        if proxy_nf is not None and proxy_nf < len(m.faces):
            bits.append(f"decomposed on a {proxy_nf}-face proxy (compute cap)")
        if connectors:
            bits.append("connectors: planned (parts are loose, not joined)")
        if saved:
            bits.append(f"{saved} part STLs saved to {out_dir}")
        return {"ok": bool(parts) and n_fit == len(parts),
                "parts": parts, "note": "; ".join(bits)}
    except Exception as e:
        return {"ok": False, "parts": [], "note": f"split failed: {e}"}


def _geom_bisect_seam(piece):
    """A VISIBLE bisecting seam through the piece centroid, normal along its
    longest axis (halving the largest dimension so it moves toward bed fit)."""
    ext = np.asarray(piece.extents, float)
    axis = int(np.argmax(ext))
    n = np.zeros(3)
    n[axis] = 1.0
    o = np.asarray(piece.bounds, float).mean(axis=0)
    return {"plane": (tuple(float(x) for x in o), tuple(float(x) for x in n)),
            "score": float("nan"), "concavity_score": 0.0,
            "crosses_load_path": False, "area_mm2": float("nan"),
            "source": "geom_bisect"}


def _apply_cut(pieces, target, seam, connectors, fit_mm):
    """Cut pieces[target] by seam, add connectors, return
    (new_pieces, delta_connectors, fallback_bool, method_note|None, method)
    or None if the cut failed. Never raises (callers already wrap)."""
    o, n = seam["plane"]
    cut = _cut_by_plane(pieces[target], o, n)
    if not cut.get("ok"):
        return None
    pos, neg = cut["pos"], cut["neg"]
    fallback = (cut["method"] == "plane")
    mnote = cut["note"] if fallback else None
    dconn = 0
    if connectors and cut["connectors_ok"]:
        cres = plan_connectors([pos, neg], [seam], fit_mm=fit_mm)
        if cres.get("ok"):
            pos, neg = cres["parts"]
            dconn = cres["n_connectors"]
        else:
            fallback = True
            mnote = cres["note"]
    new_pieces = pieces[:target] + [pos, neg] + pieces[target + 1:]
    return new_pieces, dconn, fallback, mnote, cut["method"]


def _smart_split(m, bed, bed_label, ext_s, bed_s, *, connectors, out_dir,
                 load_case, fit_mm):
    """SMART pipeline: CoACD topology -> seam scoring -> boolean cut on the
    ORIGINAL mesh -> connectors -> per-part gate. Fallback ladder, never raises."""
    try:
        notes = [f"mesh {ext_s} mm exceeds {bed_s} mm bed ({bed_label})"]
        timed_out = False

        parts_topo, _proxy_nf = _coacd_parts(m)
        if parts_topo is None:
            return {"ok": False, "parts": [], "seams": [],
                    "connectors": {"n_connectors": 0, "fit_mm": float(fit_mm),
                                   "fallback": True},
                    "note": "; ".join(notes + [
                        "smart split: CoACD unavailable -- install the extra "
                        "`meshprep[split]` (no fallback split performed)"])}

        sp = plan_seams(m, parts_topo, load_case=load_case)
        seams = sp["seams"] if sp.get("ok") else []
        if seams:
            notes.append(sp["note"])
        else:
            notes.append(f"no scorable hidden seam ({sp.get('note')})")

        # ---- TIER 1: greedy cut along ranked (hidden) seams ------------------
        # Wall-clock budget for the CUTTING loops only (the geometric-bisection +
        # boolean work that can run unbounded on a large pathological mesh). CoACD
        # and plan_seams run before this; their cost is bounded separately (CoACD
        # decimates to a proxy; plan_seams scores on a decimated proxy).
        deadline = time.monotonic() + _SMART_BUDGET_S
        import trimesh  # noqa: F401
        pieces = [m]
        executed = []
        n_conn_total = 0
        any_fallback = False
        method_notes = []
        used = 0
        for seam in seams:
            if used >= _MAX_SEAM_CUTS:
                break
            if time.monotonic() > deadline:
                timed_out = True
                break
            if all(_fits(p.extents, bed) for p in pieces):
                break
            o, n = seam["plane"]
            target = None
            for idx, p in enumerate(pieces):
                if _fits(p.extents, bed):
                    continue
                if _plane_intersects(p, o, n):
                    target = idx
                    break
            if target is None:
                continue
            res = _apply_cut(pieces, target, seam, connectors, fit_mm)
            if res is None:
                continue
            pieces, dconn, fb, mnote, _method = res
            n_conn_total += dconn
            any_fallback = any_fallback or fb
            if mnote:
                method_notes.append(mnote)
            executed.append(seam)
            used += 1

        # ---- TIER 2: geometric bisection to fit the bed (VISIBLE seam) -------
        # A convex/over-bed piece with no hidden seam still has to fit; bisect its
        # LONGEST axis. Honest label: this seam is NOT hidden.
        geom_used = 0
        for _ in range(_MAX_SEAM_CUTS + 2):
            if time.monotonic() > deadline:
                timed_out = True
                break
            over = [i for i, p in enumerate(pieces) if not _fits(p.extents, bed)]
            if not over:
                break
            # largest non-fitting piece first
            target = max(over, key=lambda i: float(np.max(pieces[i].extents)))
            seam = _geom_bisect_seam(pieces[target])
            res = _apply_cut(pieces, target, seam, connectors, fit_mm)
            if res is None:
                break
            pieces, dconn, fb, mnote, _method = res
            n_conn_total += dconn
            any_fallback = any_fallback or fb
            if mnote:
                method_notes.append(mnote)
            executed.append(seam)
            geom_used += 1
        if geom_used:
            method_notes.append(f"{geom_used} geometric bisection cut(s) applied "
                                f"(no hidden seam available -- these seams are "
                                f"VISIBLE; place them where finishing is easy).")

        # ---- per-part gate + bed fit ----------------------------------------
        # Watertightness is checked AS EXPORTED (STL round-trip), because a
        # boolean/connector part is watertight only in indexed in-memory form and
        # can reload non-watertight -- and assess_printability does not test it.
        out_parts = []
        all_ok = bool(pieces)
        n_nonwt = 0
        final_pieces = []
        for i, p in enumerate(pieces):
            p, wt = _finalize_watertight(p)
            final_pieces.append(p)
            pext = [float(v) for v in p.extents]
            fits = _fits(pext, bed)
            pr = _printable(p)
            out_parts.append({"index": i, "n_faces": int(len(p.faces)),
                              "extents_mm": pext, "fits_bed": fits,
                              "printable": pr, "watertight": bool(wt)})
            if not wt:
                n_nonwt += 1
            if not (fits and pr and wt):
                all_ok = False
        pieces = final_pieces
        _save_parts(pieces, out_parts, out_dir)

        if not executed:
            notes.append("smart split: no seam could be executed on an oversized "
                         "piece (parts unchanged)")
            all_ok = False
        else:
            notes.append(f"cut into {len(pieces)} part(s) along {len(executed)} "
                         f"seam(s) ({len(executed) - geom_used} hidden, "
                         f"{geom_used} geometric)")
        if not connectors:
            notes.append("connectors: disabled by caller")
        elif n_conn_total:
            notes.append(f"{n_conn_total} peg/socket connector(s) added")
        if n_nonwt:
            notes.append(f"{n_nonwt} part(s) reload NON-WATERTIGHT as exported STL "
                         f"(boolean/connector artifact the weld+fill pass could not "
                         f"close) -- gate FAILS them; slice with mesh-repair on or "
                         f"re-split without connectors")
        if timed_out:
            notes.append(f"smart split hit its internal {_SMART_BUDGET_S:.0f}s time "
                         f"budget and stopped early -- this is a best-effort PARTIAL "
                         f"split; any part still over-bed is reported fits_bed=False")
        notes.extend(method_notes)

        return {"ok": bool(all_ok), "parts": out_parts, "seams": executed,
                "connectors": {"n_connectors": n_conn_total,
                               "fit_mm": float(fit_mm),
                               "fallback": bool(any_fallback or not connectors)},
                "note": "; ".join(notes)}
    except Exception as e:
        return {"ok": False, "parts": [], "seams": [],
                "connectors": {"n_connectors": 0, "fit_mm": float(fit_mm),
                               "fallback": True},
                "note": f"smart split failed: {e}"}


def _save_parts(meshes, entries, out_dir):
    """Best-effort STL export of each part; annotates entries with 'file'."""
    if out_dir is None:
        return
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        return
    for i, (pm, entry) in enumerate(zip(meshes, entries)):
        try:
            fp = os.path.join(str(out_dir), f"part_{i:03d}.stl")
            pm.export(fp)
            entry["file"] = fp
        except Exception:
            pass


# ======================================================================
# Inline self-check (VALIDATION GATES G1 volume conservation, G3 seam control)
# ======================================================================
def _selfcheck():
    import trimesh
    ok_all = True

    # G1 -- volume conservation on a boolean box cut (<= 1%)
    box = trimesh.creation.box(extents=(30.0, 20.0, 40.0))
    cut = _cut_by_plane(box, (0.0, 0.0, 3.0), (0.0, 0.0, 1.0))
    if cut.get("ok") and cut["method"] == "boolean":
        v0 = float(box.volume)
        vsum = float(cut["pos"].volume) + float(cut["neg"].volume)
        rel = abs(vsum - v0) / v0
        g1 = rel <= 0.01
        print(f"G1 volume conservation: sum={vsum:.1f} orig={v0:.1f} "
              f"rel_err={rel * 100:.3f}%  -> {'PASS' if g1 else 'FAIL'}")
        ok_all &= g1
    else:
        print(f"G1 volume conservation: FAIL (no boolean cut: {cut.get('note')})")
        ok_all = False

    # G3 -- seam control: a dumbbell's top seam must land on the NECK, not a lobe.
    try:
        s1 = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
        s1.apply_translation([-14.0, 0, 0])
        s2 = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
        s2.apply_translation([14.0, 0, 0])
        neck = trimesh.creation.cylinder(radius=3.0, height=28.0)
        neck.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [1, 0, 0]))
        db = trimesh.util.concatenate([s1, s2, neck])
        mdb, _f, _n = _build_manifold_or_repair(db)
        if mdb is not None:
            db = _manifold_to_trimesh(mdb)
        # parts proposal = the two lobes (centroids straddle x=0)
        left = db.slice_plane([0, 0, 0], [-1, 0, 0], cap=True)
        right = db.slice_plane([0, 0, 0], [1, 0, 0], cap=True)
        sp = plan_seams(db, [left, right])
        if sp.get("ok") and sp["seams"]:
            top = sp["seams"][0]
            (ox, oy, oz), (nx, ny, nz) = top["plane"]
            on_neck = abs(ox) <= 5.0
            axis_aligned = abs(nx) >= 0.8
            g3 = on_neck and axis_aligned
            print(f"G3 seam control: top seam origin x={ox:.2f} (|x|<=5? {on_neck}), "
                  f"normal x={nx:.2f} (aligned? {axis_aligned}), "
                  f"concavity={top['concavity_score']:.3f}  "
                  f"-> {'PASS' if g3 else 'FAIL'}")
            ok_all &= g3
        else:
            print(f"G3 seam control: FAIL (no seam: {sp.get('note')})")
            ok_all = False
    except Exception as e:
        print(f"G3 seam control: ERROR {e}")
        ok_all = False

    # Connector smoke test on the box cut halves (fit logic real, non-fatal)
    try:
        if cut.get("ok") and cut["method"] == "boolean":
            seam = {"plane": ((0.0, 0.0, 3.0), (0.0, 0.0, 1.0))}
            cr = plan_connectors([cut["pos"], cut["neg"]], [seam], fit_mm=0.4)
            print(f"connectors smoke: ok={cr['ok']} n={cr['n_connectors']} "
                  f"note={cr['note'][:80]}")
    except Exception as e:
        print(f"connectors smoke: ERROR {e}")

    # Fit coupon builder smoke test
    try:
        fc = fit_coupon()
        print(f"fit_coupon: ok={fc['ok']} "
              f"faces={len(fc['mesh'].faces) if fc.get('mesh') is not None else 0} "
              f"clearances={fc['clearances']}")
    except Exception as e:
        print(f"fit_coupon: ERROR {e}")

    print(f"\nSELF-CHECK {'PASS' if ok_all else 'FAIL'}")
    return ok_all


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selfcheck() else 1)
