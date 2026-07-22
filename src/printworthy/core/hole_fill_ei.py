"""EI-bisector curvature-aware hole reconstruction (build-plan "Move 2").

Replaces the GPL `pymeshfix` residual-hole closer with a dependency-free
(numpy + trimesh) fill that *follows the local surface curvature* instead
of laying a flat planar fan.

THE GEOMETRY (why this exists)
------------------------------
Pass-1's `fill_medium_holes` closes a boundary loop by placing one apex
vertex at the loop's planar centroid and fanning triangles to it. That is
correct topologically and cheap, but on a curved surface — the open end of
a sphere cap, the rim of a bowl, a missing patch on an organic Meshy/SF3D
model — a flat centroid fan leaves a visible crease: the fan sits in the
loop plane while the surface around it bulges out. Downstream the crease
shows up as a ring of low-σ / high-ρ_deficit vertices (the fan vertices
disagree in normal with the curved neighbours) and as a dent in the
rendered cap.

The fix: place the apex NOT in the loop plane but offset along the local
surface *bisector* direction — the area-weighted mean of the incident face
normals at the loop, exactly Pass-1's `_estimate_loop_outward_normal`.
This is the "Extended Incenter" intuition the build plan calls "Move 2":
the apex is pulled toward where the surface is already heading, so the cap
domes to continue the surrounding curvature rather than flattening it.

HOW MUCH TO DOME
----------------
We estimate the local curvature from how far the incident-face normals
deviate from the loop-plane normal:

    curvature_proxy = mean_f ( 1 - |n_f · n_plane| )

where n_plane is the best-fit normal of the loop's vertex positions and
n_f ranges over the faces incident to the loop's boundary vertices.

    * curvature_proxy ≈ 0  → the rim is essentially flat; a dome would
      invent geometry. We fall back to the flat centroid (offset 0) and
      record the loop as "flat".
    * curvature_proxy large but the loop is wildly non-planar (the
      best-fit plane is meaningless, e.g. a saddle-shaped or spiral rim)
      → the dome direction is unreliable; fall back to flat as well.
    * otherwise → dome. The raw offset is

          offset = dome_strength * curvature_proxy * loop_radius

      clamped to `MAX_OFFSET_FRAC * loop_radius` so the apex can never
      spike out past ~half the loop radius (no needles).

The apex moves along `+bisector` if the bisector points away from the
loop centroid's "inside" and `-bisector` otherwise; we pick the sign that
moves the apex to the convex (outward) side, matching the surrounding
surface. The fan-triangle winding is then orientation-matched to the local
surface normal exactly as Pass-1 does, so manifold_guarantee never has to
flip it.

HONESTY ABOUT THE METRIC
------------------------
The action metrics record `n_domed` vs `n_flat` so a reader can see how
many caps actually got the curvature treatment versus fell back to a flat
fan. The σ_min sanity gate at the apex (`_sigma_min_apex_to_loop`, the
same tangent-projected, valence-normalised σ as Pass-1's
`_sigma_min_centroid_to_loop`) is evaluated at the *domed* apex position,
so a dome that produces a degenerate fan is rejected just like a flat one.

This module imports only numpy + trimesh. No GPL, no optional heavy deps.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import trimesh

from ._types import RepairAction

# ── tuning constants ──────────────────────────────────────────────────────

# Maximum apex offset as a fraction of the loop radius. Caps spikes: even
# at dome_strength → ∞ the apex never sits past half a radius above the
# loop plane. A hemisphere cap on a circular loop of radius r has its pole
# exactly r above the plane; 0.5·r keeps us comfortably shallower than a
# full hemisphere, which is the right call for *residual* holes (we are
# patching a gap in an existing surface, not modelling a new dome).
MAX_OFFSET_FRAC = 0.5

# Below this curvature proxy the rim is treated as flat — doming would
# invent geometry. Chosen to be just above float-noise on a planar loop.
CURVATURE_FLOOR = 0.02

# Above this planarity residual (normalised RMS distance of loop vertices
# from their best-fit plane, relative to loop radius) the best-fit plane —
# and therefore the dome direction — is judged unreliable, so we fall back
# to a flat fan. Saddle / spiral / self-folded rims land here.
PLANARITY_CEIL = 0.35

# Loops with more than this many edges optionally get one concentric ring
# inserted between the rim and the apex, for better triangle *aspect ratio*
# on big caps.
#
# DISABLED by default (set above any realistic loop size). Measured on the
# Meshy wineglass corpus mesh: the ring-subdivided cap regresses σ_min_p10
# from 0.648 → 0.585 (the ring vertices have poor frame conditioning),
# which trips the σ-regression rollback veto and ships the input back as
# FAIL. The plain single-apex fan (domed or flat) instead *improves*
# σ_min_p10 to 0.651 while closing the same loops. Ring subdivision
# optimises triangle aspect — a metric `mesh_health` does not weight —
# at the cost of vertex σ, which it weights at 0.30. So the ring path is
# retained (for a future variant that preserves σ) but gated off. See the
# isolation table in the 2026-06-03 validation run.
RING_SUBDIVIDE_MIN_EDGES = 10 ** 9



def fill_holes_ei_bisector(
    mesh: trimesh.Trimesh,
    actions: list[RepairAction],
    *,
    min_edges: int = 51,
    max_edges: int = 500,
    sigma_floor: float = 0.03,
    loop_budget: int = 5000,  # novice-first: close ALL medium loops (was 200)
    dome_strength: float = 1.0,
) -> trimesh.Trimesh:
    """Curvature-aware residual-hole closer (Pass-3 op).

    Pass-1-op signature: ``(mesh, actions, **kw) -> mesh``; appends one
    ``RepairAction`` describing the fill. Pure numpy + trimesh.

    For each ordered boundary loop with ``min_edges <= |loop| <= max_edges``
    (cheapest first, capped at ``loop_budget`` loops), this places one apex
    vertex offset along the local surface bisector by a curvature-derived
    amount (clamped to ``MAX_OFFSET_FRAC * loop_radius``) and fans the loop
    to it, orientation-matched to the local surface normal. When the
    curvature estimate is unreliable (near-flat rim or wildly non-planar
    loop) the apex falls back to the flat planar centroid; the action
    metric ``n_flat`` / ``n_domed`` records the split honestly.

    Parameters
    ----------
    min_edges, max_edges : int
        Inclusive boundary-loop edge-count window. Loops outside are left
        for other passes (small loops → Pass-1 `fill_small_holes`; very
        large open-back loops → Pass-2 `repair_open_back`).
    sigma_floor : float
        Reject a candidate fill if the apex vertex's tangent-projected
        σ_min against its new 1-ring falls below this (degenerate fan on a
        spiral / highly non-convex loop). The σ is evaluated at the *domed*
        apex, so doming that worsens the fan is caught.
    loop_budget : int
        Per-mesh cap on number of loops filled. Runaway-defect protection.
    dome_strength : float
        Scales the curvature-derived offset. 0.0 reproduces a flat fan;
        1.0 is the calibrated default. The result is still clamped by
        ``MAX_OFFSET_FRAC`` regardless of this value.

    Returns
    -------
    trimesh.Trimesh
        A new mesh with the accepted caps appended. The caller's input is
        never mutated (we build fresh vertex/face arrays). On any failure
        or no-op the input mesh is returned unchanged.
    """
    loops_before = _count_boundary_loops(mesh)
    if loops_before == 0:
        actions.append(RepairAction(
            action="fill_holes_ei_bisector",
            tool="ei_bisector_fan_fill:no_loops",
            metrics={"loops_before": 0, "loops_after": 0, "filled": 0,
                     "n_domed": 0, "n_flat": 0},
        ))
        return mesh

    try:
        ordered = _extract_boundary_loops_ordered(mesh)
    except Exception as e:
        actions.append(RepairAction(
            action="fill_holes_ei_bisector",
            tool="ei_bisector_fan_fill:extraction_failed",
            metrics={"error": f"{type(e).__name__}: {e}",
                     "loops_before": int(loops_before)},
        ))
        return mesh

    target_loops = [loop for loop in ordered
                    if min_edges <= len(loop) <= max_edges]
    skipped_too_small = sum(1 for loop in ordered if len(loop) < min_edges)
    skipped_too_large = sum(1 for loop in ordered if len(loop) > max_edges)
    target_loops.sort(key=len)  # cheapest (smallest) first
    if loop_budget is not None:
        target_loops = target_loops[:int(loop_budget)]

    if not target_loops:
        actions.append(RepairAction(
            action="fill_holes_ei_bisector",
            tool="ei_bisector_fan_fill:no_loops_in_range",
            metrics={
                "loops_before": int(loops_before),
                "min_edges": int(min_edges),
                "max_edges": int(max_edges),
                "skipped_too_small": int(skipped_too_small),
                "skipped_too_large": int(skipped_too_large),
                "n_domed": 0, "n_flat": 0,
            },
        ))
        return mesh

    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()

    # vertex → incident-face indices, built once. Drives both the outward
    # normal estimate (orientation match) and the bisector direction.
    v2f: list[list[int]] = [[] for _ in range(len(verts))]
    for fi, face in enumerate(faces):
        for vi in face:
            v2f[int(vi)].append(fi)

    new_verts: list[np.ndarray] = []
    new_faces: list[np.ndarray] = []
    n_filled = 0
    n_rejected_sigma = 0
    n_flipped = 0
    n_domed = 0
    n_flat = 0
    n_ring_subdivided = 0
    offsets_recorded: list[float] = []

    for loop in target_loops:
        loop_arr = np.asarray(loop, dtype=np.int64)
        loop_pts = verts[loop_arr]
        centroid = loop_pts.mean(axis=0)

        # --- loop radius + best-fit plane (for curvature + clamp) --------
        rel = loop_pts - centroid
        loop_radius = float(np.linalg.norm(rel, axis=1).mean())
        if loop_radius <= 0.0:
            # All loop vertices coincident — degenerate, skip.
            n_rejected_sigma += 1
            continue
        plane_normal, planarity = _loop_plane_fit(rel, loop_radius)

        # --- bisector direction (area-weighted incident face normal) -----
        bisector = _estimate_loop_outward_normal(loop_arr, verts, faces, v2f)

        # --- curvature proxy: incident-normal deviation from loop plane --
        curvature = _loop_curvature_proxy(
            loop_arr, verts, faces, v2f, plane_normal,
        )

        # --- decide domed vs flat ----------------------------------------
        apex = centroid.copy()
        domed = False
        offset_mag = 0.0
        if (
            bisector is not None
            and plane_normal is not None
            and curvature >= CURVATURE_FLOOR
            and planarity <= PLANARITY_CEIL
            and dome_strength > 0.0
        ):
            raw = float(dome_strength) * curvature * loop_radius
            offset_mag = min(raw, MAX_OFFSET_FRAC * loop_radius)
            # Dome along the bisector directly. The bisector is the
            # area-weighted mean of the incident face normals, so on a
            # correctly-wound mesh (Pass-1 `reorient_normals_coherently`
            # has already run) it points OUTWARD by construction — that is
            # the EI direction the build plan asks the apex to follow.
            #
            # We deliberately do NOT re-sign it against `plane_normal`: the
            # best-fit plane normal is `vh[-1]` from an SVD, whose sign is
            # arbitrary, so aligning the bisector to it would flip a correct
            # outward bisector to inward roughly half the time. The plane
            # normal is used only for the flatness / planarity decision
            # above, never as the dome axis. If a mesh reaches Pass 3 with
            # inverted winding the bisector points inward, but the σ gate's
            # flat retry and the downstream rollback contract both protect
            # against shipping that damage.
            apex = centroid + offset_mag * bisector
            domed = True

        # --- σ_min sanity gate at the (possibly domed) apex --------------
        if sigma_floor > 0.0:
            sigma_apex = _sigma_min_apex_to_loop(apex, loop_pts)
            if sigma_apex < sigma_floor:
                # If a domed apex fails, retry once flat — a flat fan is
                # often acceptable where a dome over-reaches. Only count a
                # rejection if even the flat fan fails.
                if domed:
                    sigma_flat = _sigma_min_apex_to_loop(centroid, loop_pts)
                    if sigma_flat >= sigma_floor:
                        apex = centroid
                        domed = False
                        offset_mag = 0.0
                    else:
                        n_rejected_sigma += 1
                        continue
                else:
                    n_rejected_sigma += 1
                    continue

        # --- orientation: match fan winding to local surface normal ------
        # (the bisector above IS the area-weighted incident-normal estimate;
        # nothing it depends on has changed, so reuse it instead of recomputing)
        local_normal = bisector
        e0 = loop_pts[0] - apex
        e1 = loop_pts[1] - apex
        fan_normal = np.cross(e0, e1)
        flip = (local_normal is not None
                and float(np.dot(fan_normal, local_normal)) < 0.0)
        if flip:
            n_flipped += 1

        # --- emit geometry (single apex fan, or apex + one ring) ---------
        use_ring = len(loop_arr) > RING_SUBDIVIDE_MIN_EDGES
        if use_ring:
            added = _emit_ringed_cap(
                loop_arr, loop_pts, centroid, apex, offset_mag,
                len(verts), new_verts, new_faces, flip,
            )
            if added:
                n_ring_subdivided += 1
            else:
                # Ring construction declined (degenerate); fall back to a
                # plain single-apex fan so the loop still closes.
                _emit_single_apex_fan(
                    loop_arr, apex, len(verts), new_verts, new_faces, flip,
                )
        else:
            _emit_single_apex_fan(
                loop_arr, apex, len(verts), new_verts, new_faces, flip,
            )

        n_filled += 1
        if domed:
            n_domed += 1
            offsets_recorded.append(float(offset_mag))
        else:
            n_flat += 1

    if n_filled == 0:
        actions.append(RepairAction(
            action="fill_holes_ei_bisector",
            tool="ei_bisector_fan_fill:all_rejected",
            metrics={
                "loops_before": int(loops_before),
                "targets": int(len(target_loops)),
                "rejected_sigma_floor": int(n_rejected_sigma),
                "sigma_floor": float(sigma_floor),
                "n_domed": 0, "n_flat": 0,
            },
        ))
        return mesh

    verts_out = np.vstack([verts, np.array(new_verts, dtype=np.float64)])
    faces_out = np.vstack([faces, np.array(new_faces, dtype=np.int64)])
    out = trimesh.Trimesh(vertices=verts_out, faces=faces_out, process=False)
    loops_after = _count_boundary_loops(out)

    mean_offset = (
        float(np.mean(offsets_recorded)) if offsets_recorded else 0.0
    )
    actions.append(RepairAction(
        action="fill_holes_ei_bisector",
        tool="ei_bisector_fan_fill",
        metrics={
            "loops_before": int(loops_before),
            "loops_after": int(loops_after),
            "filled": int(n_filled),
            "fans_flipped": int(n_flipped),
            "rejected_sigma_floor": int(n_rejected_sigma),
            "skipped_too_small": int(skipped_too_small),
            "skipped_too_large": int(skipped_too_large),
            # Honest dome accounting (module docstring "HONESTY").
            "n_domed": int(n_domed),
            "n_flat": int(n_flat),
            "n_ring_subdivided": int(n_ring_subdivided),
            "mean_dome_offset": mean_offset,
            "min_edges": int(min_edges),
            "max_edges": int(max_edges),
            "sigma_floor": float(sigma_floor),
            "dome_strength": float(dome_strength),
            "loop_budget": int(loop_budget) if loop_budget else -1,
        },
    ))
    return out


# ── geometry emit helpers ─────────────────────────────────────────────────


def _emit_single_apex_fan(
    loop_arr: np.ndarray,
    apex: np.ndarray,
    base_n_verts: int,
    new_verts: list[np.ndarray],
    new_faces: list[np.ndarray],
    flip: bool,
) -> None:
    """Append one apex vertex and a triangle fan wrapping the loop.

    Winding is chosen by `flip` (matched upstream to the local surface
    normal). Mutates `new_verts` / `new_faces` in place.
    """
    apex_idx = base_n_verts + len(new_verts)
    new_verts.append(np.asarray(apex, dtype=np.float64))
    n = len(loop_arr)
    for i in range(n):
        j = (i + 1) % n
        if flip:
            new_faces.append(np.array(
                [apex_idx, int(loop_arr[j]), int(loop_arr[i])], dtype=np.int64))
        else:
            new_faces.append(np.array(
                [apex_idx, int(loop_arr[i]), int(loop_arr[j])], dtype=np.int64))


def _emit_ringed_cap(
    loop_arr: np.ndarray,
    loop_pts: np.ndarray,
    centroid: np.ndarray,
    apex: np.ndarray,
    offset_mag: float,
    base_n_verts: int,
    new_verts: list[np.ndarray],
    new_faces: list[np.ndarray],
    flip: bool,
) -> bool:
    """Append a single concentric ring (halfway in) plus a central apex,
    quad-stripping the rim→ring band and fanning the ring→apex.

    The ring sits at the midpoint between each rim vertex and the centroid,
    lifted half the apex offset along the same dome direction — so the cap
    is two short bands instead of one long sliver fan, improving triangle
    aspect ratio on large loops. Returns True on success, False if the loop
    is too small/degenerate to ring cleanly (caller falls back to a plain
    fan).

    Triangle winding mirrors the single-apex fan via `flip`.
    """
    n = len(loop_arr)
    if n < 6:
        return False

    # Dome direction: from centroid to apex (zero-length if flat fan).
    dome_vec = apex - centroid
    half_lift = 0.5 * dome_vec

    # Ring vertices: halfway from each rim vertex toward the centroid,
    # raised by half the dome lift so the band curves rather than creasing.
    ring_pts = 0.5 * (loop_pts + centroid) + half_lift
    ring_start = base_n_verts + len(new_verts)
    for p in ring_pts:
        new_verts.append(np.asarray(p, dtype=np.float64))
    apex_idx = base_n_verts + len(new_verts)
    new_verts.append(np.asarray(apex, dtype=np.float64))

    def _ring_idx(k: int) -> int:
        return ring_start + (k % n)

    # Band 1: rim (loop_arr) ↔ ring. Two triangles per quad
    #   (rim_i, rim_j, ring_j) and (rim_i, ring_j, ring_i).
    for i in range(n):
        j = (i + 1) % n
        ri, rj = int(loop_arr[i]), int(loop_arr[j])
        gi, gj = _ring_idx(i), _ring_idx(j)
        if flip:
            new_faces.append(np.array([ri, gj, rj], dtype=np.int64))
            new_faces.append(np.array([ri, gi, gj], dtype=np.int64))
        else:
            new_faces.append(np.array([ri, rj, gj], dtype=np.int64))
            new_faces.append(np.array([ri, gj, gi], dtype=np.int64))

    # Band 2: ring → apex fan.
    for i in range(n):
        j = (i + 1) % n
        gi, gj = _ring_idx(i), _ring_idx(j)
        if flip:
            new_faces.append(np.array([apex_idx, gj, gi], dtype=np.int64))
        else:
            new_faces.append(np.array([apex_idx, gi, gj], dtype=np.int64))
    return True


# ── curvature / plane estimation ──────────────────────────────────────────


def _loop_plane_fit(
    rel: np.ndarray,
    loop_radius: float,
) -> tuple[Optional[np.ndarray], float]:
    """Best-fit plane normal of a loop (relative-to-centroid positions) and
    a normalised planarity residual.

    Returns ``(plane_normal, planarity)`` where:
        plane_normal : unit normal (smallest-variance PCA axis), or None
                       if the loop is too degenerate to fit a plane.
        planarity    : RMS distance of loop vertices from the best-fit
                       plane, normalised by ``loop_radius``. 0 = perfectly
                       planar; large = saddle / spiral / folded rim.
    """
    if rel.shape[0] < 3 or loop_radius <= 0.0:
        return None, float("inf")
    # PCA via SVD of the centred point cloud. Smallest singular direction
    # is the plane normal.
    try:
        _u, s, vh = np.linalg.svd(rel, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, float("inf")
    plane_normal = vh[-1]
    nrm = float(np.linalg.norm(plane_normal))
    if nrm < 1e-12:
        return None, float("inf")
    plane_normal = plane_normal / nrm
    # Out-of-plane residual: projection of each point onto the normal.
    resid = rel @ plane_normal
    rms = float(np.sqrt(np.mean(resid * resid)))
    planarity = rms / loop_radius
    return plane_normal, planarity


def _loop_curvature_proxy(
    loop_arr: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    v2f: list[list[int]],
    plane_normal: Optional[np.ndarray],
) -> float:
    """Mean deviation of loop-incident face normals from the loop plane.

        curvature = mean_f ( 1 - |n_f · n_plane| )

    A flat rim (all incident normals parallel to the plane normal) gives
    ≈ 0; a strongly curved rim (incident normals tilted away from the
    plane) gives a larger value. Returns 0.0 when the plane normal or the
    incident faces are unavailable (caller then treats the loop as flat).
    """
    if plane_normal is None:
        return 0.0
    incident: set[int] = set()
    for v in loop_arr:
        incident.update(v2f[int(v)])
    if not incident:
        return 0.0
    devs: list[float] = []
    for fi in incident:
        a, b, c = faces[fi]
        nf = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        nrm = float(np.linalg.norm(nf))
        if nrm < 1e-12:
            continue
        nf = nf / nrm
        devs.append(1.0 - abs(float(np.dot(nf, plane_normal))))
    if not devs:
        return 0.0
    return float(np.mean(devs))


# ── reused-pattern helpers (re-derived to keep Pass-3 self-contained) ──────
#
# These mirror the Pass-1 helpers of the same role. They are re-derived here
# rather than imported so this module has no dependency on pass1 internals
# (which the constraints forbid editing) and can move/refactor independently.


def _estimate_loop_outward_normal(
    loop_arr: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    v2f: list[list[int]],
) -> Optional[np.ndarray]:
    """Area-weighted mean normal of the existing faces incident to a loop's
    boundary vertices — the EI "bisector" direction.

    The raw cross product (b−a)×(c−a) has magnitude 2·area, so summing
    un-normalised cross products already area-weights. Returns a unit
    vector, or None when the loop has no incident faces or they cancel.
    (Behaviourally identical to ``forge.pass1._estimate_loop_outward_normal``.)
    """
    incident: set[int] = set()
    for v in loop_arr:
        incident.update(v2f[int(v)])
    if not incident:
        return None
    acc = np.zeros(3, dtype=np.float64)
    for fi in incident:
        a, b, c = faces[fi]
        acc += np.cross(verts[b] - verts[a], verts[c] - verts[a])
    norm = float(np.linalg.norm(acc))
    if norm < 1e-12:
        return None
    return acc / norm


def _sigma_min_apex_to_loop(
    apex: np.ndarray,
    loop_pts: np.ndarray,
) -> float:
    """Tangent-projected, valence-normalised σ_min at the apex vertex
    against the loop 1-ring (Sigma FINDINGS Tier 2.6 mixed-element
    normalisation — held in the 2026-06-03 recalibration).

    Identical in form to ``forge.pass1._sigma_min_centroid_to_loop`` but
    evaluated at the *domed* apex so that an over-reaching dome producing a
    degenerate fan is rejected. Returns σ ∈ [0, 1]; 0 on degeneracy.
    """
    edges = loop_pts - apex
    rs = np.linalg.norm(edges, axis=1)
    keep = rs > 0
    if int(keep.sum()) < 3:
        return 0.0
    units = edges[keep] / rs[keep, None]
    U = units.T                                       # (3, k)
    try:
        U_left, _s, _ = np.linalg.svd(U, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    T = U_left[:, :2]                                 # tangent plane
    U_2d = T.T @ U                                    # (2, k)
    s2 = np.linalg.svd(U_2d, compute_uv=False)
    k = float(units.shape[0])
    if k < 2:
        return 0.0
    return float(min(1.0, s2.min() / float(np.sqrt(k / 2.0))))


# ── boundary-loop topology (re-derived from pass1) ────────────────────────


def _extract_boundary_loops_ordered(mesh: trimesh.Trimesh) -> list[list[int]]:
    """Return each boundary loop as an ordered cycle of vertex indices.

    Walks the boundary adjacency graph in cycle order. Junctions (vertices
    with ≠ 2 boundary neighbours) make a component unsafe for fan-fill and
    are skipped. Mirrors ``forge.pass1._extract_boundary_loops_ordered``.
    """
    groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=None)
    boundary_edges = [
        tuple(int(x) for x in mesh.edges_sorted[g[0]])
        for g in groups if len(g) == 1
    ]
    if not boundary_edges:
        return []
    adj: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    seen: set[int] = set()
    loops: list[list[int]] = []
    for start in list(adj):
        if start in seen:
            continue
        comp = _component(start, adj)
        if any(len(adj.get(v, [])) != 2 for v in comp):
            seen |= comp
            continue
        cycle: list[int] = [start]
        seen.add(start)
        prev = -1
        current = start
        while True:
            nbrs = adj.get(current, [])
            nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
            if nxt == start:
                break
            cycle.append(nxt)
            seen.add(nxt)
            prev = current
            current = nxt
            if len(cycle) > len(adj):
                cycle = []
                break
        if len(cycle) >= 3:
            loops.append(cycle)
    return loops


def _component(start: int, adj: dict[int, list[int]]) -> set[int]:
    seen: set[int] = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def _count_boundary_loops(mesh: trimesh.Trimesh) -> int:
    """Number of boundary loops (components of the boundary-edge graph).

    Mirrors the counting half of ``forge.pass1._boundary_loop_sizes`` —
    counts boundary-graph components, NOT boundary edges, so it agrees with
    the loop-level metrics in the action log.
    """
    try:
        groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=None)
        boundary_edges = [
            tuple(mesh.edges_sorted[g[0]]) for g in groups if len(g) == 1
        ]
        if not boundary_edges:
            return 0
        adj: dict[int, list[int]] = {}
        for a, b in boundary_edges:
            adj.setdefault(int(a), []).append(int(b))
            adj.setdefault(int(b), []).append(int(a))
        seen: set[int] = set()
        n_loops = 0
        for v in list(adj):
            if v in seen:
                continue
            stack = [v]
            comp: set[int] = set()
            while stack:
                u = stack.pop()
                if u in comp:
                    continue
                comp.add(u)
                stack.extend(adj.get(u, ()))
            seen |= comp
            n_edges = sum(len(adj.get(u, ())) for u in comp) // 2
            if n_edges > 0:
                n_loops += 1
        return n_loops
    except Exception:
        return 0
