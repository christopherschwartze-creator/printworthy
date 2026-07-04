"""support_mods.py -- STUB: physics/risk-driven SUPPORT BLOCKERS + ENFORCERS.

WHAT
====
Slicers place supports by a blind overhang angle. We already compute a
per-face RISK field (core/_print_premortem.premortem: overhang, thin-wall,
warp-proxy, scar channels) and we already write modifier volumes a slicer
honors (core/threemf_writer, proven applied by PrusaSlicer for fill_density).
Combine them: emit a 3MF whose modifier volumes ENFORCE supports where the
part actually needs them and BLOCK them where they would only scar.

    support_mods(mesh, *, out_3mf, profile=None, enforce_pctile=85.0,
                 block_pctile=30.0) -> {"ok", "out_3mf", "n_enforcers",
                                        "n_blockers", "note"}

WHY (the wedge)
===============
No free tool auto-generates per-region support enforcer/blocker volumes from
an analysis. Users hand-paint them in the slicer today. Same proven writer
tech as reinforce; a genuinely new output.

BUILD PLAN
==========
1. RISK FIELD. premortem(mesh) -> per-face risk in [0,1] + channel breakdown.
   ENFORCE candidates: faces where the OVERHANG channel dominates and risk >=
   enforce_pctile percentile. BLOCK candidates: downward-ish faces whose
   overhang risk is LOW (< block_pctile) but which the slicer's blind angle
   test would still support -- i.e. faces with angle just past the threshold
   where the scar channel says the cosmetic cost outweighs the (low) failure
   risk. Start conservative: blockers only where risk < 0.15.
2. VOLUMES. Reuse core/_modifier_extract exactly as reinforce does: face set
   -> voxel occupancy (grid_from_centroids on face centroids at the premortem
   pitch) -> largest-component + dilate 1 -> occupancy_to_mesh -> clip. For
   ENFORCERS the volume must extend DOWNWARD to the bed / nearest surface
   below (extrude the occupancy column down; supports live under the face,
   not on it). For BLOCKERS the volume is a thin shell just under the surface.
3. WRITE. threemf_writer.Volume(..., is_modifier=True) with the support
   metadata keys. SOURCE-VERIFY against PrusaSlicer's 3mf.cpp/Model.cpp the
   exact keys (the fill_density verification is the template): expected
   `support_enforcers` / `support_blockers` volume_type values used by the
   PrusaSlicer project loader. DO NOT ship until the bytes are diffed against
   a slicer-saved project containing hand-painted enforcers (save one in the
   GUI, unzip, copy the schema -- same method that proved fill_density).
4. HEADLESS PROOF. Slice via slicer.estimate three ways: no supports forced,
   enforcer 3MF, blocker 3MF, on a part with a known overhang. Enforcer run
   must ADD support material vs baseline; blocker run must REMOVE it
   (filament_mm3 comparison -- the same 1.85x methodology that proved the
   graded modifier).
5. WIRE-IN: `meshprep prep --supports auto`; report line: "supports enforced
   on N regions (top risk 0.93), blocked on M cosmetic faces".

VALIDATION GATES
================
G1 schema: byte-diff vs a GUI-authored enforcer/blocker project (exact keys).
G2 headless: enforcer adds / blocker removes support filament on the control
   part (the T-overhang from premortem's selftest).
G3 false-positive control: a supportless-printable part (45-deg chamfer cone)
   must produce ZERO enforcers and the note "no supports needed".
G4 containment: every volume watertight + inside/under the part bbox
   (verify_modifier, as in reinforce).

HONEST LABELS
=============
Risk is a heuristic triage field (premortem's own label), NOT a guarantee a
support is necessary/sufficient. Blockers trade failure risk for cosmetics --
default conservative and say so in the report.

COMPUTE SAFETY: premortem pitch caps apply; volumes on the analysis grid
(<=6k-face proxy). KILL: if G2 shows PrusaSlicer's CLI ignores the volumes
(unlike fill_density), stop -- GUI-only value is not worth the support burden.
"""
from __future__ import annotations

_NOT_BUILT = (
    "support blockers/enforcers are a documented stub -- build plan in "
    "meshprep/core/support_mods.py. Nothing was modified.")


def support_mods(mesh, *, out_3mf="supports.3mf", profile=None,
                 enforce_pctile=85.0, block_pctile=30.0):
    """STUB. Emit a 3MF with risk-driven support enforcer/blocker volumes.

    Planned return: {"ok": True, "out_3mf": path, "n_enforcers": int,
                     "n_blockers": int, "risk_top": float, "note": str}
    Current behavior: honest refusal (never raises).
    """
    return {"ok": False, "implemented": False, "out_3mf": None,
            "note": _NOT_BUILT}
