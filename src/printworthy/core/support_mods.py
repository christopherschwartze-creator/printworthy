"""support_mods.py -- physics/risk-driven SUPPORT BLOCKERS + ENFORCERS.

WHAT
====
Slicers place supports by a blind overhang angle. We already compute a
per-face RISK field (core/_print_premortem.premortem: overhang, thin-wall,
warp-proxy, scar channels) and we already write modifier volumes a slicer
honors (core/threemf_writer, proven applied by PrusaSlicer for fill_density).
Combine them: emit a 3MF whose extra volumes ENFORCE supports where the part
actually needs them and BLOCK them where they would only scar.

    support_mods(mesh, *, out_3mf, profile=None, enforce_pctile=85.0,
                 block_pctile=30.0) -> {"ok", "out_3mf", "n_enforcers",
                                        "n_blockers", "risk_top", "note"}

WHY (the wedge)
===============
No free tool auto-generates per-region support enforcer/blocker volumes from
an analysis. Users hand-paint them in the slicer today. Same proven writer
tech as reinforce; a genuinely new output.

SCHEMA (source-verified, corrects this module's own earlier guess)
====================================================================
PrusaSlicer's ModelVolume::type_to_string (Model.cpp) emits the volume_type
metadata value as CamelCase SINGULAR: "SupportEnforcer" / "SupportBlocker" --
NOT the snake_case plural this stub originally guessed ("support_enforcers" /
"support_blockers"). threemf_writer.VT_SUPPORT_ENFORCER / VT_SUPPORT_BLOCKER
carry the corrected, verified strings. Enforcer/blocker volumes are their own
ModelVolumeType (not PARAMETER_MODIFIER), so they are written with
volume_type=<the type> and NO modifier=1 flag (see threemf_writer._prusa_
config_xml). Orca/Bambu's equivalent schema was NOT source-verified, so this
module writes with include_orca=False and says so in the note -- it does not
invent an Orca support key.

BUILT (see BUILD PLAN below for the shipped mechanics; deviations noted)
=========================================================================
1. RISK FIELD. premortem(mesh, build_dir=[0,0,1]) -> per-face risk in [0,1] +
   channel breakdown. The build_dir is FIXED at the mesh's own +Z (the part
   as already oriented by an earlier pipeline stage), not premortem's default
   build_dir=None (which re-runs its own min-support orientation SEARCH and
   would silently second-guess that upstream decision -- see the BUILT
   section's step 1 note for why).
   ENFORCE candidates: faces where the OVERHANG channel dominates and the
   COMBINED risk >= enforce_pctile percentile (default 85th, conservative).
   "Dominates" merges overhang+scar into ONE physical signal first (scar
   fires on ~the same down-facing faces as overhang -- premortem's own
   summary.combined.NOTE_scar_overhang_coupled says so), so the two never
   split a face's "vote" and double-suppress a real enforcer candidate.
   BLOCK candidates: faces PAST the slicer's blind-angle test (the same
   `_print3d._face_overhang` boolean premortem's own overhang channel is
   built from) whose OVERHANG-CHANNEL value (already height/severity gated)
   is LOW -- < block_pctile percentile (default 30th) AND < 0.15 (the stub's
   conservative hard cap). This is the "geometrically an overhang, but so
   close to the plate / so mild the height-gate says it is not a real
   failure risk" case a blind angle-only slicer support-generator cannot see.
2. VOLUMES. Voxelize the XY footprint of the candidate faces (in a build-
   plate-aligned frame: rotate `up` -> +Z, exactly the `lay_flat` convention
   `_print_advanced.generate_supports` already uses, then inverse-transform
   the result back into the mesh's own world frame so it shares a coordinate
   system with the base part), keep the largest connected footprint
   component (speckle cleanup, same idea as `_modifier_extract.
   high_stress_occupancy`'s largest_component=True), then EXTRUDE in Z:
     ENFORCER: from the bed (z=0) up to a few mm PAST the candidate faces'
       highest point (INTO the solid above the down-facing surface).
     BLOCKER:  a thin shell from a little below the faces' lowest point up to
       a few mm past their highest point (NOT a full column to the bed).
   `_modifier_extract.occupancy_to_mesh` (marching cubes -> watertight mesh,
   reused verbatim) turns the occupancy into geometry.
   DEVIATION FROM THE ORIGINAL PLAN (measured, see CLI proof below): the
   original plan said "...-> clip_to_base -> verify_modifier", mirroring
   reinforce's dense-infill modifier pipeline. reinforce's modifier must sit
   STRICTLY INSIDE the solid, so a boolean INTERSECTION with the base
   (`_modifier_extract.clip_to_base`) is correct there. An enforcer/blocker
   is different: the CLI proof (below) measured that a volume merely
   TOUCHING the model surface (zero overlap) produced a BYTE-IDENTICAL,
   functionally inert gcode -- while a volume overlapping the solid by ~4mm
   worked. So the volume must extend OUTSIDE the solid too (down through the
   open air where support material actually prints), which a full boolean
   intersection with the base would strip away entirely. Containment is
   therefore a light AABB clamp into the base's own bounding box (the same
   fallback `clip_to_base` itself uses when its boolean path fails) rather
   than the boolean intersection, plus a reported `overlap_with_solid_mm3`
   sanity check (boolean intersection volume, informational only) so a
   degenerate "touching-only" volume is visible in the result, not silently
   shipped as if it were as good as a real overlap.
3. WRITE. threemf_writer.Volume(..., is_modifier=False,
   volume_type="SupportEnforcer"|"SupportBlocker") via the writer's additive
   volume_type kwarg. include_orca=False (Orca support schema unverified;
   see SCHEMA note above).
4. SLICER FLAGS (measured, not asserted -- see the CLI proof transcript this
   module's implementation was built and checked against):
     Enforcer: support_material=1, support_material_auto=0 (supports ONLY
       inside enforcer volumes) -> ADDS support filament vs. no-enforcer
       baseline (measured +1.27 cm^3 / +66 support-gcode segments on the
       control L-bracket overhang part).
     Blocker: support_material=1, support_material_auto=1 (auto overhang
       detection ON, so there is something to suppress) -> REMOVES support
       back to the no-support baseline (measured -1.09 cm^3 / -66 segments).
     Round-trip: PrusaSlicer's own --export-3mf re-save reproduced
       `<metadata type="volume" key="volume_type" value="SupportEnforcer"/>`
       verbatim and its own --info excluded the enforcer volume from the
       reported printable volume (recognized as non-model geometry, not
       silently coerced to a ModelPart).
5. WIRE-IN: OUT OF SCOPE for this lane/repo-boundary run (only
   core/support_mods.py, core/threemf_writer.py, and the README roadmap rows
   are sanctioned edits here). `printworthy prep --supports auto` /
   `report line` wiring belongs in cli.py / pipeline.py and is NOT done by
   this module -- call `support_mods()` directly until that lands.

VALIDATION GATES
================
G1 schema: PASSED -- threemf_writer's VT_SUPPORT_ENFORCER/VT_SUPPORT_BLOCKER
   are the exact Model.cpp::type_to_string strings (source recon), and the
   writer's byte-identical gate confirms adding volume_type does not disturb
   any existing (reinforce) output byte.
G2 headless: PASSED, but externally (a hand-built proof harness, not this
   module) -- see BUILT step 4 above for the measured deltas. This module's
   own responsibility ends at emitting a schema-correct, geometrically-
   overlapping 3MF; re-confirm "prints correctly" by opening it in a slicer
   whenever the analysis pipeline (premortem/thresholds) changes.
G3 false-positive control: a face only clears the enforce threshold if its
   COMBINED risk is BOTH >= the enforce_pctile percentile AND > 0.05 (the
   codebase's standing "any risk" floor) -- a supportless-printable part
   (nothing exceeds premortem's height-gated overhang risk) yields
   n_enforcers=0 and the plain note "no supports needed" rather than
   manufacturing an enforcer out of percentile noise on an all-safe part.
G4 containment: every volume is checked (not enforced by clipping) for
   watertightness + AABB containment in the base's bbox
   (`_modifier_extract.verify_modifier`) and reports `overlap_with_solid_mm3`
   / `overlap_ok` so a degenerate near-zero-overlap volume is visible in the
   result rather than silently shipped.

HONEST LABELS
=============
Risk is a heuristic triage field (premortem's own label), NOT a guarantee a
support is necessary/sufficient. Blockers trade failure risk for cosmetics --
default conservative (risk < 0.15) and say so in the report. The slicer
EFFECT (enforcer adds / blocker removes support material) is confirmed by
MEASURED filament/support-segment deltas from a CLI proof run, not asserted;
opening the 3MF in your own slicer is the final confirmation on your part and
profile. Orca/Bambu: support-volume schema not source-verified -- this module
omits the Orca sibling config rather than guess at its keys.

COMPUTE SAFETY: premortem's own face cap applies (decimate <=6000 faces
before calling). The support-column voxel grid is built at a pitch chosen
from the part's own extent (not the premortem field, which has no voxel grid
of its own -- premortem is a pure per-face heuristic; the ORIGINAL build plan
loosely called this "the premortem pitch", but there is no such value to
reuse, so this module picks one, capped so the column's total voxel count
cannot explode on a tall/thin part (adaptive coarsening, see
`_column_occupancy`). Never raises: every stage is wrapped; a failed geometry
build for one candidate set (enforcer or blocker) just yields zero volumes of
that kind, not a crash.

Run as a script for a quick self-check: python -m printworthy.core.support_mods
"""
from __future__ import annotations

import os

# single-thread numerics before heavy imports (shared 16 GB box)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from . import _modifier_extract as mx
from . import _print3d
from . import _print_premortem as pmortem
from . import threemf_writer as tw
from ._mesh_util import decimate as _decimate

# compute-safety ceiling for the extruded support-column voxel grid (total
# cells). At this ceiling a full nx*ny*nz bool array is a few MB -- generous
# headroom under the 16 GB box's constraints even before considering that
# marching cubes runs once per candidate set (enforcer, blocker), not per part.
_MAX_COLUMN_CELLS = 1_500_000


def _consult_printability(mesh):
    """Consult the shared printability/solidity gate (core._printability).
    Returns the gate dict, or None if the module is unavailable. The gate is
    pure and never raises; this wrapper is defensive regardless. (Same helper
    reinforce.py uses -- kept local per this module's existing convention of
    not sharing private helpers across sibling files.)"""
    try:
        from ._printability import assess_printability
    except Exception:
        return None
    try:
        return assess_printability(mesh)
    except Exception:
        return None


def _pick_pitch(mesh, target_cells=36.0, min_pitch=0.15):
    """A reasonable isotropic voxel pitch for THIS part: its longest extent
    divided into `target_cells` cells, floored at `min_pitch` so a tiny/thin
    mesh doesn't produce a degenerate (near-zero or huge) grid. There is no
    premortem voxel grid to borrow a pitch from (premortem is a pure per-face
    heuristic, not a voxelized field) -- this is a fresh, independent choice."""
    try:
        ext = float(np.asarray(mesh.extents, float).max())
    except Exception:
        ext = 0.0
    if not np.isfinite(ext) or ext <= 0:
        return max(min_pitch, 1.0)
    return max(ext / float(target_cells), min_pitch)


def _lay_flat_transforms(mesh, up):
    """Rigid transform so `up` -> +Z and the mesh's lowest point sits at z=0
    (the build plate) -- the same convention `_print_advanced.lay_flat` uses
    for support-pillar generation. Returns (mesh_rot, M, Minv): M maps the
    ORIGINAL (world) frame -> the lay-flat frame; Minv is its inverse, used to
    place generated support geometry back into the base mesh's own coordinate
    frame (threemf_writer concatenates a part's volumes into one shared mesh,
    so every volume must share the base's frame)."""
    import trimesh
    R = trimesh.geometry.align_vectors(np.asarray(up, float), [0.0, 0.0, 1.0])
    mesh_rot = mesh.copy()
    mesh_rot.apply_transform(R)
    z0 = float(mesh_rot.vertices[:, 2].min())
    T = trimesh.transformations.translation_matrix([0.0, 0.0, -z0])
    mesh_rot.apply_transform(T)
    M = T @ R
    Minv = np.linalg.inv(M)
    return mesh_rot, M, Minv


def _transform_points(pts, M):
    pts = np.asarray(pts, float)
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (h @ M.T)[:, :3]


def _largest_2d_component(footprint):
    """Keep only the largest connected component of a 2D boolean footprint
    (speckle cleanup -- the same idea as _modifier_extract.high_stress_
    occupancy's largest_component=True, applied to the XY footprint before
    it is extruded in Z, since Z is always contiguous by construction here)."""
    try:
        from scipy import ndimage
        lbl, n = ndimage.label(footprint)
        if n > 1:
            sizes = ndimage.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
            keep = int(np.argmax(sizes)) + 1
            return lbl == keep
    except Exception:
        pass
    return footprint


def _column_occupancy(mesh_rot, face_idx, pitch0, *, to_bed,
                      margin_solid_mm=4.0, shell_below_mm=2.0, pad_cells=2,
                      max_cells=_MAX_COLUMN_CELLS, max_tries=6):
    """Voxel occupancy (nx,ny,nz bool) + world origin, in the lay-flat frame
    (+Z=up, bed at z=0), for a support column/shell under `face_idx`.

    to_bed=True  (ENFORCER): fills every voxel from the bed (z=0) up to
        `margin_solid_mm` PAST the candidate faces' highest point -- the full
        open-air support column PLUS a few mm poking into the solid above.
        MEASURED requirement (CLI proof, module docstring): a volume only
        touching the surface (zero overlap) is functionally inert; ~4mm of
        overlap is what the proof used and measured working.
    to_bed=False (BLOCKER): a thin shell from `shell_below_mm` below the
        candidate faces' lowest point up to `margin_solid_mm` above their
        highest point -- NOT a full column to the bed.

    Adaptive pitch: if the requested pitch would need more than `max_cells`
    voxels for this candidate set's footprint + Z-span, the pitch is coarsened
    (x1.6 per try, up to max_tries) until it fits -- compute safety on a
    tall/thin part, never a crash.

    Returns (occ, origin, pitch_used) or (None, None, None) if face_idx is
    empty or the geometry never fits the cell budget."""
    face_idx = np.asarray(face_idx, dtype=int)
    if len(face_idx) == 0:
        return None, None, None
    cen = mesh_rot.triangles_center[face_idx]
    xy = cen[:, :2]
    z_lo = float(cen[:, 2].min())
    z_hi = float(cen[:, 2].max())
    z_top_cap = float(mesh_rot.vertices[:, 2].max())
    xy_min = xy.min(axis=0)
    xy_span = xy.max(axis=0) - xy_min

    pitch = float(pitch0)
    for _ in range(max_tries):
        nx = int(np.ceil(xy_span[0] / pitch)) + 1 + 2 * pad_cells
        ny = int(np.ceil(xy_span[1] / pitch)) + 1 + 2 * pad_cells
        margin_cells = max(1, int(np.ceil(margin_solid_mm / pitch)))
        z_top_index = min(int(np.ceil(z_hi / pitch)) + margin_cells,
                          int(np.floor(z_top_cap / pitch)))
        if to_bed:
            z_bot_index = 0
        else:
            below_cells = max(1, int(np.ceil(shell_below_mm / pitch)))
            z_bot_index = max(0, int(np.floor(z_lo / pitch)) - below_cells)
        z_top_index = max(z_top_index, z_bot_index + 1)
        nz = z_top_index + 1
        if nx * ny * nz <= max_cells:
            break
        pitch *= 1.6
    else:
        return None, None, None

    origin_xy = xy_min - pad_cells * pitch
    ix = np.rint((xy[:, 0] - origin_xy[0]) / pitch).astype(np.int64)
    iy = np.rint((xy[:, 1] - origin_xy[1]) / pitch).astype(np.int64)
    footprint = np.zeros((nx, ny), bool)
    footprint[ix, iy] = True
    footprint = _largest_2d_component(footprint)
    if not footprint.any():
        return None, None, None

    occ = np.zeros((nx, ny, nz), bool)
    occ[footprint, z_bot_index:z_top_index + 1] = True
    origin = np.array([origin_xy[0], origin_xy[1], -0.5 * pitch])
    return occ, origin, pitch


def _build_support_volume(base, base_rot, Minv, face_idx, pitch, *, to_bed):
    """The full pipeline for one candidate set: voxelize+extrude (lay-flat
    frame) -> marching-cubes mesh (_modifier_extract.occupancy_to_mesh,
    reused verbatim) -> back into the base's world frame -> light bbox
    containment clamp -> verify + overlap report.

    Returns (mesh_or_None, info_dict). Never raises."""
    import trimesh
    try:
        occ, origin, pitch_used = _column_occupancy(base_rot, face_idx, pitch,
                                                     to_bed=to_bed)
        if occ is None or not occ.any():
            return None, {"ok": False, "reason": "empty/oversized occupancy"}
        mod = mx.occupancy_to_mesh(occ, pitch_used, origin, dilate=1)
        if mod is None or len(mod.faces) < 4:
            return None, {"ok": False, "reason": "marching cubes empty"}

        verts_world = _transform_points(mod.vertices, Minv)
        try:
            vol_world = trimesh.Trimesh(vertices=verts_world, faces=mod.faces,
                                        process=True)
            trimesh.repair.fix_normals(vol_world)
            trimesh.repair.fix_winding(vol_world)
        except Exception:
            vol_world = trimesh.Trimesh(vertices=verts_world, faces=mod.faces)

        # Light AABB clamp into the base's bbox (NOT a boolean intersection --
        # see module docstring "DEVIATION FROM THE ORIGINAL PLAN": a full
        # solid intersection would strip the open-air part of the column the
        # enforcer/blocker needs).
        try:
            bmin, bmax = base.bounds
            clamped = np.clip(vol_world.vertices, bmin, bmax)
            vol_world = trimesh.Trimesh(vertices=clamped, faces=vol_world.faces,
                                        process=True)
        except Exception:
            pass

        chk = mx.verify_modifier(vol_world, base)
        overlap_mm3 = 0.0
        try:
            inter = vol_world.intersection(base)
            if inter is not None and len(getattr(inter, "faces", [])) > 0:
                overlap_mm3 = float(abs(inter.volume))
        except Exception:
            pass
        info = {
            "ok": True,
            "n_faces_flagged": int(len(face_idx)),
            "pitch_mm": round(float(pitch_used), 4),
            "watertight": chk["watertight"],
            "inside_base_bbox": chk["inside_base_bbox"],
            "volume_mm3": round(chk["volume_mm3"], 3),
            "n_faces": chk["n_faces"],
            "overlap_with_solid_mm3": round(overlap_mm3, 3),
            "overlap_ok": bool(overlap_mm3 > 0.5),
        }
        return vol_world, info
    except Exception as e:
        return None, {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def support_mods(mesh, *, out_3mf="supports.3mf", profile=None,
                 enforce_pctile=85.0, block_pctile=30.0):
    """Emit a slicer-ready 3MF with risk-driven SUPPORT ENFORCER/BLOCKER
    volumes (PrusaSlicer SupportEnforcer/SupportBlocker schema).

    mesh            : trimesh solid (mm).
    out_3mf         : output path.
    profile         : RESERVED for a future slicer-profile hint (e.g. a
                      different self-support angle); not used yet -- passing
                      one has no effect (no silent behavior change).
    enforce_pctile  : combined-risk percentile (of overhang/scar-dominant
                      faces) above which a region gets an ENFORCER. Default
                      85th -- conservative (top ~15% of risk).
    block_pctile    : overhang-channel percentile below which a face PAST the
                      slicer's blind angle gets a BLOCKER. Default 30th, and
                      always additionally capped at risk < 0.15 (the stub's
                      conservative default).

    Returns:
      {"ok": True, "out_3mf": path|None, "n_enforcers": int, "n_blockers": int,
       "risk_top": float, "enforcers": [...], "blockers": [...],
       "premortem_summary": {...}, "uncalibrated": True, "note": str}
      or {"ok": False, "out_3mf": None, "reason"|"note": str}. Never raises.
    """
    try:
        import trimesh  # noqa: F401
    except Exception as e:
        return {"ok": False, "implemented": True, "out_3mf": None,
                "note": f"trimesh unavailable: {e}"}

    try:
        # -- PRINTABILITY / SOLIDITY GATE (same gate reinforce.py consults) --
        gate = _consult_printability(mesh)
        if isinstance(gate, dict) and gate.get("verdict") == "FAIL":
            return {"ok": False, "implemented": True, "out_3mf": None,
                    "reason": (gate.get("plain_summary")
                               or "input is not a printable closed solid -- "
                                  "cannot place supports."),
                    "printability": gate}

        base = _decimate(mesh, 6000)
        if base is None or len(base.faces) < 4:
            return {"ok": False, "implemented": True, "out_3mf": None,
                    "reason": "degenerate/empty base mesh"}

        # 1) RISK FIELD (reused verbatim) ----------------------------------- #
        # Fixed +Z build_dir, NOT premortem's own build_dir=None (which
        # re-runs optimize_orientation's min-support SEARCH): support_mods
        # assumes `mesh` already sits in its final print orientation (the
        # convention every other print-stage module in this codebase uses --
        # _print_fem.warp_analysis/_print_advanced.generate_supports/lay_flat
        # all take an explicit/default +Z "as printed" axis; orientation
        # SELECTION is a separate, earlier pipeline stage). Re-optimizing here
        # would silently second-guess that upstream decision and, on simple
        # test geometry, can rotate a real overhang away entirely (observed
        # in this module's own self-check).
        pm_res = pmortem.premortem(base, build_dir=[0.0, 0.0, 1.0])
        risk_face = np.asarray(pm_res["risk_face"], float)
        channels = pm_res["channels"]
        up = np.asarray(pm_res["build_dir"], float)

        if risk_face.size == 0 or len(risk_face) != len(base.faces):
            return {"ok": False, "implemented": True, "out_3mf": None,
                    "reason": "premortem produced no usable risk field"}

        zeros = np.zeros_like(risk_face)
        overhang_ch = np.asarray(channels.get("overhang", zeros), float)
        scar_ch = np.asarray(channels.get("scar", zeros), float)
        thin_ch = np.asarray(channels.get("thin", zeros), float)
        warp_ch = np.asarray(channels.get("warp", zeros), float)

        # overhang + scar are ONE physical signal (premortem's own summary
        # says scar fires on ~the same down-facing faces) -- merge before
        # computing which channel "dominates" so they never split a face's
        # vote against thin/warp.
        overhang_signal = np.maximum(overhang_ch, scar_ch)
        dominant = np.argmax(np.vstack([overhang_signal, thin_ch, warp_ch]), axis=0)
        is_overhang_dominant = dominant == 0

        # -- ENFORCE candidates --------------------------------------------- #
        thr_enf = float(np.percentile(risk_face, float(enforce_pctile)))
        enforce_mask = (is_overhang_dominant & (risk_face >= thr_enf)
                        & (risk_face > 0.05))

        # -- BLOCK candidates ------------------------------------------------ #
        # faces past the slicer's blind angle test (the same boolean
        # premortem's own overhang channel gates), but LOW on the (height/
        # severity-gated) overhang channel itself -- i.e. geometrically an
        # overhang, physically a non-issue.
        _sev, needs_blind = _print3d._face_overhang(base, up, crit_deg=45.0)
        thr_blk = float(np.percentile(overhang_ch, float(block_pctile)))
        block_mask = needs_blind & (overhang_ch < thr_blk) & (overhang_ch < 0.15)

        risk_top = (float(risk_face[enforce_mask].max()) if enforce_mask.any()
                   else float(risk_face.max()))

        n_flagged_enf = int(enforce_mask.sum())
        n_flagged_blk = int(block_mask.sum())
        if n_flagged_enf == 0 and n_flagged_blk == 0:
            return {"ok": True, "implemented": True, "out_3mf": None,
                    "n_enforcers": 0, "n_blockers": 0,
                    "risk_top": round(risk_top, 4),
                    "premortem_summary": pm_res.get("summary"),
                    "uncalibrated": True,
                    "note": ("no supports needed: no face cleared the "
                             f"enforce_pctile={float(enforce_pctile):.0f} "
                             "overhang/scar-dominant risk threshold, and no "
                             "low-risk-but-past-blind-angle face qualified as "
                             "a blocker. Risk is a HEURISTIC triage field "
                             "(premortem), not a guarantee any support was "
                             "actually required here.")}

        # 2) VOLUMES (build-plate-aligned voxelize + extrude + marching cubes) #
        base_rot, _M, Minv = _lay_flat_transforms(base, up)
        pitch = _pick_pitch(base)

        volumes = []
        enf_report = []
        blk_report = []

        if n_flagged_enf > 0:
            enf_idx = np.nonzero(enforce_mask)[0]
            vmesh, vinfo = _build_support_volume(base, base_rot, Minv, enf_idx,
                                                 pitch, to_bed=True)
            if vmesh is not None and vinfo.get("ok"):
                volumes.append(tw.Volume(vmesh.vertices, vmesh.faces,
                                         name="support_enforcer",
                                         is_modifier=False,
                                         volume_type=tw.VT_SUPPORT_ENFORCER))
                enf_report.append(vinfo)

        if n_flagged_blk > 0:
            blk_idx = np.nonzero(block_mask)[0]
            vmesh, vinfo = _build_support_volume(base, base_rot, Minv, blk_idx,
                                                 pitch, to_bed=False)
            if vmesh is not None and vinfo.get("ok"):
                volumes.append(tw.Volume(vmesh.vertices, vmesh.faces,
                                         name="support_blocker",
                                         is_modifier=False,
                                         volume_type=tw.VT_SUPPORT_BLOCKER))
                blk_report.append(vinfo)

        n_enforcers = len(enf_report)
        n_blockers = len(blk_report)
        if n_enforcers == 0 and n_blockers == 0:
            return {"ok": True, "implemented": True, "out_3mf": None,
                    "n_enforcers": 0, "n_blockers": 0,
                    "risk_top": round(risk_top, 4),
                    "premortem_summary": pm_res.get("summary"),
                    "uncalibrated": True,
                    "note": ("candidate faces were flagged but no watertight "
                             "support volume could be built from them "
                             "(degenerate geometry after voxelization) -- no "
                             "3MF written.")}

        # 3) WRITE (base ModelPart + enforcer/blocker volumes) -------------- #
        base_vol = tw.Volume(base.vertices, base.faces, name="base",
                             is_modifier=False)
        parts = [{"name": "support_mods_part", "volumes": [base_vol] + volumes}]
        out_path = tw.write_3mf(parts, out_3mf, include_orca=False)

        note = (
            "Risk is a HEURISTIC triage field (premortem's own label), NOT a "
            "guarantee a support is necessary/sufficient. Blockers trade "
            "failure risk for cosmetics and default conservative (risk<0.15). "
            "The 3MF is SCHEMA-CORRECT against PrusaSlicer's documented "
            "Model.cpp/3mf.cpp SupportEnforcer/SupportBlocker volume types. "
            "The SLICING EFFECT is MEASURED, not asserted: a CLI proof on a "
            "controlled overhang part showed an enforcer volume ADDING "
            "support filament (+1.27 cm^3 / +66 support-gcode segments, "
            "support_material_auto=0) and a blocker volume REMOVING support "
            "back to the no-support baseline (-1.09 cm^3 / -66 segments, "
            "support_material_auto=1) -- confirm on your own part by opening "
            "this 3MF in a slicer. Orca/Bambu support-volume schema was NOT "
            "source-verified -- this 3MF omits the Orca sibling config "
            "(include_orca=False); enforcer/blocker volumes would degrade to "
            "plain model parts there.")

        return {
            "ok": True,
            "implemented": True,
            "out_3mf": os.path.abspath(out_path),
            "n_enforcers": n_enforcers,
            "n_blockers": n_blockers,
            "risk_top": round(risk_top, 4),
            "enforcers": enf_report,
            "blockers": blk_report,
            "premortem_summary": pm_res.get("summary"),
            "uncalibrated": True,
            "note": note,
        }
    except Exception as e:
        return {"ok": False, "implemented": True, "out_3mf": None,
                "note": f"{type(e).__name__}: {e}"}


# =========================================================================== #
#  self-check                                                                  #
# =========================================================================== #
def _demo_overhang_bracket():
    """L-bracket: a stem + a cantilevered cap forming one clean 20mm overhang
    shelf underside (a real down-facing, height-gated overhang -- an
    ENFORCER-candidate control), boolean-unioned so it's one watertight solid.
    Mirrors the geometry class used in the schema recon's CLI proof."""
    import trimesh
    stem = trimesh.creation.box(extents=[10.0, 10.0, 20.0])
    stem.apply_translation([5.0, 5.0, 10.0])
    cap = trimesh.creation.box(extents=[30.0, 10.0, 5.0])
    cap.apply_translation([15.0, 5.0, 22.5])
    try:
        m = stem.union(cap)
    except Exception:
        m = trimesh.util.concatenate([stem, cap])
    return m


def _demo_safe_cone():
    """A 60deg-half-angle cone (well inside the 45deg self-support limit) --
    a G3 false-positive control: should yield n_enforcers=0."""
    import trimesh
    return trimesh.creation.cone(radius=20.0, height=15.0)


def _run_self_check():
    from ._mesh_util import ascii_console, say
    ascii_console()
    say("support_mods self-check")
    say("=" * 60)

    bracket = _demo_overhang_bracket()
    r1 = support_mods(bracket, out_3mf="_selftest_support_bracket.3mf")
    say("L-bracket overhang control:")
    say(f"  ok={r1['ok']} n_enforcers={r1.get('n_enforcers')} "
        f"n_blockers={r1.get('n_blockers')} risk_top={r1.get('risk_top')}")
    for e in r1.get("enforcers", []) or []:
        say(f"    enforcer: {e}")
    say(f"  note: {r1.get('note')}")

    cone = _demo_safe_cone()
    r2 = support_mods(cone, out_3mf="_selftest_support_cone.3mf")
    say("60deg cone (G3 false-positive control):")
    say(f"  ok={r2['ok']} n_enforcers={r2.get('n_enforcers')} "
        f"n_blockers={r2.get('n_blockers')}")
    say(f"  note: {r2.get('note')}")

    ok = bool(r1.get("ok")) and int(r1.get("n_enforcers") or 0) >= 1
    ok = ok and bool(r2.get("ok")) and int(r2.get("n_enforcers") or 0) == 0
    say("=" * 60)
    say("SELF-CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_run_self_check())
