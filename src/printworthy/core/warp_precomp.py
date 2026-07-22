"""warp_precomp.py -- STUB: warp PRE-COMPENSATION (the flagship roadmap item).

WHAT
====
We already PREDICT warp (fem_warp_probe: per-layer eigenstrain -> one linear
elastic solve -> corner lift, validated against the Timoshenko bimetal closed
form). This feature INVERTS it: pre-deform the input mesh by the NEGATIVE of
the predicted displacement field so the physical part warps INTO spec and
comes off the bed straight. This is a premium metal-AM capability (inverse
distortion compensation) that no free FDM tool ships.

    precompensate(mesh, material="PLA", *, scale=None, iterations=1, ...)
        -> {"ok", "mesh", "applied_scale", "predicted_residual_mm",
            "uncalibrated", "claim", "note"}

WHY (the wedge)
===============
The demo IS the marketing: photo of the same part printed twice, warped vs
straight. It also forces the one validation this product still lacks -- a
physical print -- because pre-comp is only quantitative after the one-coupon
calibration (core/_warp_calibrate.py) fixes the per-printer/filament scale.

BUILD PLAN (step by step, anchored to code that exists)
=======================================================
1. EXPOSE THE FIELD. `_print_fem.warp_analysis` currently returns only corner
   lifts. Add `keep_field=True`: thread out the solver's full DOF vector `u`
   (fem_warp_probe solves K u = f; reshape to (n_nodes, 3)) together with
   `grid.node_coords` (n_nodes, 3) and the voxel pitch. Zero new math -- the
   field is already computed and discarded.
2. INTERPOLATE TO VERTICES. For each input-mesh vertex, trilinear-interpolate
   u within its containing voxel (fall back to nearest solid node within
   2*pitch for vertices that sit just outside the voxelization; beyond that,
   inverse-distance over the 8 nearest nodes). Vectorize with a cKDTree over
   node_coords. Produces u_v (n_verts, 3).
3. APPLY THE INVERSE. v' = v - k * u_v, where k = calibration scale from
   `_warp_calibrate.get_scale(label)` (default 1.0 + uncalibrated=True label).
   Clamp |k * u_v| to a safety cap (default 5% of part height) so a bad scale
   can never grossly distort the part.
4. FIXED-POINT ITERATION (iterations=2 recommended). The pre-deformed shape
   has a slightly different warp field; re-run warp_analysis on the deformed
   mesh and apply the residual correction. Two iterations suffice for a
   near-linear problem (the solve IS linear; geometry feedback is the only
   nonlinearity). Hard cap iterations<=3 -- this is not an optimizer.
5. GATE THE OUTPUT. Run core._printability.assess_printability on the result
   (pre-comp must never ship an inverted/degenerate mesh) and report
   max_predeform_mm + predicted_residual_mm (the corner lift the FEM predicts
   for the DEFORMED shape -- should drop ~5-10x vs the original).
6. WIRE-IN. `prep(..., warp_precomp=True)` opt-in stage after orient, before
   package; CLI flag `printworthy prep --warp-precomp`; report line states scale
   provenance (calibrated label vs uncalibrated default).

VALIDATION GATES (control exercises the exact failure mode)
============================================================
G1 closed-form: pre-compensate the Timoshenko bimetal strip; the FEM residual
   lift of the deformed strip must drop monotonically with iterations and be
   <20% of the uncompensated lift at the validated grid.
G2 sign control: flip the eigenstrain sign -> pre-deform must flip direction
   (catches an inverted-field bug, the failure mode that matters).
G3 no-op control: a material with zero CTE mismatch -> u ~ 0 -> mesh returned
   essentially unchanged (max vertex move < 1e-6 * height).
G4 printability: gate PASS on the deformed output for the whole rebench
   corpus subset; never-worse rollback if the gate FAILs.
G5 PHYSICAL (the one that counts, needs a printer): print coupon + compensated
   coupon; measured flatness must improve. Ships as "calibrated for your
   printer" only after this.

HONEST LABELS
=============
Uncalibrated by default (claim string mandatory). The compensation is exactly
as good as the warp model: coarse Q1 voxel FEM, inherent-strain, conservative
lower-bound field -- expect UNDER-compensation before calibration; say so.
Not thermal simulation; not applicable to warp modes the model lacks
(e.g. delamination mid-print).

COMPUTE SAFETY (16 GB box)
==========================
Reuses warp_analysis caps (max_elem<=6000). Interpolation is O(n_verts).
iterations<=3 hard cap. No new memory class; batching rules of
_print_premortem (THIN_PAIR_BUDGET) are the precedent if any ray work is
added later.

KILL CRITERIA
=============
K1 G1 residual does not drop under iteration (model too coarse to invert).
K2 physical coupon shows no measurable flatness improvement at calibrated
   scale -> record the negative, keep the PREDICTOR, kill the COMPENSATOR.
"""
from __future__ import annotations

_NOT_BUILT = (
    "warp pre-compensation is a documented stub -- the full build plan is in "
    "this module's docstring (printworthy/core/warp_precomp.py). Nothing was "
    "modified.")


def precompensate(mesh, material="PLA", *, scale=None, iterations=1,
                  calibration_label=None, max_predeform_frac=0.05):
    """STUB. Pre-deform `mesh` against its predicted warp field.

    Planned return:
      {"ok": True, "mesh": <trimesh>, "applied_scale": float,
       "iterations": int, "max_predeform_mm": float,
       "predicted_residual_mm": float, "uncalibrated": bool,
       "claim": <honesty string>, "note": str}

    Current behavior: honest refusal (never raises).
    """
    return {"ok": False, "implemented": False, "mesh": None,
            "note": _NOT_BUILT}
