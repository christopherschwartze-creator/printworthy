"""multimaterial.py -- STUB: stress-mapped MULTI-MATERIAL region assignment.

WHAT
====
The reinforce pipeline already turns a validated FEM stress field into
modifier volumes that PrusaSlicer honors (fill_density, proven 1.85x). The
same per-volume config mechanism carries an `extruder` assignment. So: map
LOAD to MATERIAL -- rigid filament (extruder 1) on the load path, flexible/
cheap filament (extruder 2) elsewhere -- in one slicer-ready 3MF.

    material_map(mesh, load_case, *, out_3mf, rigid_extruder=1,
                 soft_extruder=2, percentile=70.0)
        -> {"ok", "out_3mf", "n_regions", "frac_rigid", "note"}

WHY (the wedge)
===============
Multi-material owners (MMU/AMS/IDEX) hand-paint material regions today.
Nobody assigns them from physics. Cheap to build: ~90% of the machinery is
reinforce.py verbatim.

BUILD PLAN
==========
1. FIELD. Reuse reinforce's stress path AS-IS: _print_fem.strength_analysis
   -> _modifier_extract.importance_field -> high_stress_occupancy(percentile)
   -> occupancy_to_mesh -> clip_to_base -> verify_modifier. (Identical code
   path; factor the shared block out of reinforce() rather than copying it.)
2. WRITE. threemf_writer.Volume settings: {"extruder": "1"} on the hot-region
   modifier (and optionally the base). SOURCE-VERIFY the key the same way
   fill_density was verified: PrusaSlicer 3mf.cpp routes unknown volume
   config keys through config.set_deserialize, and `extruder` is a standard
   per-volume option -- but CONFIRM by saving a GUI project with a per-part
   extruder override and diffing Metadata/Slic3r_PE_model.config. Orca/Bambu
   sibling (model_settings.config) uses a different key -- best-effort, label
   it "not source-diffed" exactly as reinforce does.
3. HEADLESS PROOF. Config-free CLI slicing has ONE extruder, so the graded
   1.85x trick does not transfer directly. Proof strategy: slice with a
   2-extruder .ini bundle (ship a minimal one in tests/assets) and assert the
   gcode contains tool changes (T0/T1) whose Z-span matches the modifier
   region's Z-span. If CLI multi-extruder proves flaky, the honest gate is
   GUI confirmation -- label it "user-confirmed in slicer", as reinforce did
   before its CLI proof.
4. INTERFACE PARITY: same load_case dict as reinforce; same passport sidecar
   (add "extruder_map"); CLI `meshprep reinforce --materials rigid:1,soft:2`
   (extends the existing subcommand rather than adding a new one).
5. HONEST MECHANICS NOTE: a rigid core inside a soft shell changes the load
   path (stiffness contrast attracts load to the rigid region) -- the field
   is computed on a HOMOGENEOUS solid, so label the assignment RELATIVE
   ("more rigid where more loaded"), not an engineered composite. The wedge
   harness's stiffness-contrast caveat (wedge_pipeline PROXY block) is the
   template for this disclaimer.

VALIDATION GATES
================
G1 schema byte-diff (GUI-authored per-volume extruder project).
G2 gcode tool-change proof on the cantilever demo (T1 appears; Z-span of T1
   moves matches the core region within one layer).
G3 degenerate control: uniform stress (pure tension bar) -> gradient
   pre-screen (reinforce.recommend_grading) says "don't bother" -> refuse
   with "stress too uniform for a material split to matter".
G4 containment + watertight per volume (verify_modifier).

COMPUTE SAFETY: identical to reinforce (max_elem<=3000, decimate<=6000).
KILL: if no per-volume extruder override exists in the 3MF schema (G1 fails),
fall back to exporting per-region STLs for manual slicer assembly -- still
useful, much weaker wedge; say so.
"""
from __future__ import annotations

_NOT_BUILT = (
    "stress-mapped multi-material is a documented stub -- build plan in "
    "meshprep/core/multimaterial.py. Nothing was modified.")


def material_map(mesh, load_case, *, out_3mf="materials.3mf",
                 rigid_extruder=1, soft_extruder=2, percentile=70.0):
    """STUB. Emit a 3MF whose high-stress region is assigned the rigid
    extruder and the remainder the soft/cheap extruder.

    Planned return: {"ok": True, "out_3mf": path, "n_regions": int,
                     "frac_rigid": float, "uncalibrated": True,
                     "claim": <honesty string>, "note": str}
    Current behavior: honest refusal (never raises).
    """
    return {"ok": False, "implemented": False, "out_3mf": None,
            "note": _NOT_BUILT}
