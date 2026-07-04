"""reinforce.py -- the "structural passport": carry STRUCTURE forward to the
printer as a single slicer-ready 3MF with GRADED infill.

WHAT IT DOES
============
Input : a mesh + a load case (+ material).
Output: ONE PrusaSlicer-compatible 3MF in which the high-stress region is tagged
        as a MODIFIER VOLUME with HIGHER infill density (dense on the load path,
        the part's global/base density elsewhere). PrusaSlicer / OrcaSlicer /
        Bambu Studio open it and print the graded infill AS-IS -- no FEM and no
        optimization at the slicer level.

It REUSES the validated FEM solver (scripts/_print_fem.strength_analysis) for the
stress field and does NOT modify preflight/ or scripts/. This is a standalone
sibling package.

HONESTY (this stack carries an inflated-claim history -- label everything)
==========================================================================
* The importance field grades by RELATIVE structural load (von Mises stress, or
  1/FOS), normalized. It puts MORE material where the part is MORE stressed. That
  relative ordering is exactly what graded infill wants. It is NOT a certified
  safety factor and the dense % is a design choice, not a proven margin.
* "Schema-correct" is shown against PrusaSlicer's DOCUMENTED source format
  (3mf.cpp / Model.cpp), not asserted. See threemf_writer.py.
* We CANNOT run a slicer headless here. Strongest headless validation: the 3MF is
  a valid OPC zip that re-opens; it contains base + N modifier volumes; the
  modifier carries the documented infill-override metadata; each modifier mesh is
  watertight and inside the base and covers the high-stress region (vs a random
  control). "Prints graded infill" is finally confirmed by opening it in a slicer.

COMPUTE SAFETY (16 GB, OOM-sensitive): FEM grid capped (max_elem<=~3000), input
decimated <=6000 faces, marching cubes on the coarse analysis sub-grid, 1 thread.
"""
from __future__ import annotations

import json
import os
import sys

# single-thread numerics before heavy imports (shared 16 GB box)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

# REUSE the validated FEM solver (import is the only coupling).
from . import _print_fem
from ._mesh_util import decimate as _decimate, fmt_fos as _fmt_fos
from . import _modifier_extract as mx
from . import threemf_writer as tw

# demo geometries _demo_mesh understands; any other `mesh` arg is a real file
# and MUST go through the shared ingress guard (size/type/NaN/face-count).
_DEMO_NAMES = {"cantilever", "lbracket", "psb"}


def _consult_printability(mesh):
    """Consult the shared printability/solidity gate (core._printability).
    Returns the gate dict, or None if the module is unavailable. The gate is
    pure and never raises; this wrapper is defensive regardless."""
    try:
        from ._printability import assess_printability
    except Exception:
        return None
    try:
        return assess_printability(mesh)
    except Exception:
        return None


_DEFAULT_TIERS = [
    # (percentile, dense_fill_density_percent, tier_name)
    (70.0, 80, "dense_core"),
]

# CORE/BASE strength ratio of the default recipe (dense_core 80% / base 15%).
# Graded infill only pays off when the part's stress gradient EXCEEDS this.
_RECIPE_RATIO = 80.0 / 15.0


def _grading_gradient_stats(fos, vm, centroids, pitch, *,
                            percentile=70.0, recipe_ratio=_RECIPE_RATIO):
    """Stress-gradient PRE-SCREEN: does GRADED infill beat an equal-strength
    UNIFORM infill on this part? Replicates the validated wedge harness EXACTLY:
      importance inv_fos -> high_stress_occupancy(p70, largest_component=True)
      -> binary_dilation 1 (intersect solid) -> element core mask; then
          gradient_ratio = fos_out_min / fos_global_min
    where fos_out_min = min FOS OUTSIDE the core, fos_global_min = min FOS anywhere.

    Graded infill (dense core + sparse base) only wins when the stress gradient
    EXCEEDS the recipe strength ratio CORE/BASE (default 80/15 ~= 5.33): the core
    must carry enough more load than the surround to justify the surround being
    sparse. On the 10-part wedge set this cleanly separated the 2 winners
    (10.8x, 14.7x) from the 8 losers (all <= 3.6x) at threshold 5.33.

    RELATIVE/uncalibrated: gradient_ratio is built from COMPARATIVE FOS (force
    magnitude cancels); the yes/no is a screening recommendation, NOT a certified
    structural margin. Never raises; ok=False on a degenerate (all/none) core mask.
    """
    from scipy import ndimage
    try:
        fos = np.asarray(fos, float)
        vm = np.asarray(vm, float)
        centroids = np.asarray(centroids, float)
        pitch = float(pitch)
        imp = mx.importance_field(fos, vm, mode="inv_fos")
        occ_solid, occ_hot, origin, thr, frac = mx.high_stress_occupancy(
            centroids, pitch, imp, percentile=float(percentile),
            largest_component=True)
        if not occ_hot.any():
            return {"ok": False, "reason": "empty core after threshold"}
        occ_dil = ndimage.binary_dilation(occ_hot, iterations=1) & occ_solid
        _shape, _origin2, ijk = mx.grid_from_centroids(centroids, pitch)
        mask = occ_dil[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        if not mask.any() or mask.all():
            return {"ok": False,
                    "reason": f"degenerate core mask (frac={float(mask.mean()):.3f})"}
        fos_out_min = float(fos[~mask].min())
        fos_global_min = float(fos.min())
        if not np.isfinite(fos_global_min) or fos_global_min <= 0:
            return {"ok": False, "reason": "non-positive global min FOS"}
        gradient_ratio = fos_out_min / fos_global_min
        recommend = bool(gradient_ratio >= recipe_ratio)
        reason = (
            f"stress gradient {gradient_ratio:.1f}x >= recipe ratio "
            f"{recipe_ratio:.2f}x: the core is much more loaded than the surround, "
            f"so grading (dense core + sparse base) beats an equal-strength uniform "
            f"infill -- GRADE."
            if recommend else
            f"stress gradient {gradient_ratio:.1f}x < recipe ratio "
            f"{recipe_ratio:.2f}x: load is too uniform across the part to justify a "
            f"sparse base -- an equal-strength UNIFORM infill is lighter; print "
            f"solid uniform infill instead.")
        return {
            "ok": True,
            "recommend_grading": recommend,
            "gradient_ratio": float(gradient_ratio),
            "fos_out_min": fos_out_min,
            "fos_global_min": fos_global_min,
            "threshold": float(recipe_ratio),
            "frac_core": float(mask.mean()),
            "percentile": float(percentile),
            "reason": reason,
            "uncalibrated": True,
        }
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def recommend_grading(mesh, load_case, material="PLA", *,
                      recipe_ratio=_RECIPE_RATIO, max_elem=2500):
    """Standalone gradient PRE-SCREEN: should this part get GRADED infill, or is a
    uniform infill lighter for the same strength?

    Runs the SOLID FEM (_print_fem.strength_analysis, now Pa-correct) then computes
    the wedge-validated stress-gradient ratio (see _grading_gradient_stats). Returns
    {ok, recommend_grading, gradient_ratio, fos_out_min, fos_global_min, threshold,
     frac_core, reason, uncalibrated:True} or {ok:False, reason}. Never raises.

    HONESTY: gradient_ratio uses COMPARATIVE/uncalibrated FOS; the yes/no is a
    screening recommendation, not a certified structural claim.
    """
    try:
        base = _decimate(mesh, 6000)
        if base is None or len(base.faces) < 4:
            return {"ok": False, "reason": "degenerate/empty base mesh"}
        sres = _print_fem.strength_analysis(
            base, material=material, load_case=load_case, max_elem=max_elem)
        if not sres.ok:
            return {"ok": False, "reason": f"FEM: {sres.get('reason')}"}
        return _grading_gradient_stats(
            sres["fos"], sres["vm"], sres["centroids"], sres["pitch"],
            recipe_ratio=recipe_ratio)
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def reinforce(mesh, load_case, material="PLA", out_3mf="reinforced.3mf", *,
              base_fill_density=15, tiers=None, importance_mode="vm",
              max_elem=3000, largest_component=True, dilate=1,
              include_orca=True, verbose=False):
    """Produce a slicer-ready 3MF with graded infill from a mesh + load case.

    mesh              : trimesh solid (mm). The base printed part.
    load_case         : {"load_face","anchor_face","load_axis","total_force_N"}.
    material          : 'PLA'|'PETG'|'ABS'|... (fem_materials) or a Material.
    out_3mf           : output path.
    base_fill_density : the part's GLOBAL infill % (the slicer default for the base
                        volume; recorded on the base for reference, used by the
                        Orca config). The modifier raises density ONLY on the load
                        path.
    tiers             : list of (percentile, dense_density_%, name). Default: one
                        tier, >=70th-percentile stress -> 80% infill (a dense core).
                        Provide e.g. [(85,100,"core"),(60,50,"shell")] for 2 tiers.
    importance_mode   : 'vm' (von Mises stress, default) or 'inv_fos'.
    max_elem          : FEM grid cap (compute safety; <=~3000).

    Returns a dict:
      {ok, out_3mf, n_modifiers, tiers:[...], min_fos, peak_vm_Pa, base_faces,
       importance_mode, uncalibrated:True, claim, ...} or {ok:False, reason}.
    Never raises.
    """
    try:
        import trimesh  # noqa: F401
    except Exception as e:
        return {"ok": False, "reason": f"trimesh unavailable: {e}"}

    if tiers is None:
        tiers = _DEFAULT_TIERS

    try:
        # -- PRINTABILITY / SOLIDITY GATE (before any FOS is reported) --------
        # A non-watertight shell, an inside-out/zero-volume solid, or a
        # degenerate part must be REFUSED here -- never dressed up as an "OK ->
        # reinforced.3mf" with a comparative FOS. The gate is authoritative.
        gate = _consult_printability(mesh)
        if isinstance(gate, dict) and gate.get("verdict") == "FAIL":
            return {"ok": False,
                    "reason": (gate.get("plain_summary")
                               or "input is not a printable closed solid -- "
                                  "cannot reinforce."),
                    "printability": gate}

        base = _decimate(mesh, 6000)
        if base is None or len(base.faces) < 4:
            return {"ok": False, "reason": "degenerate/empty base mesh"}

        # 1) FEM stress field (REUSED, validated solver) ------------------- #
        sres = _print_fem.strength_analysis(
            base, material=material, load_case=load_case, max_elem=max_elem)
        if not sres.ok:
            return {"ok": False, "reason": f"FEM: {sres.get('reason')}"}
        fos = np.asarray(sres["fos"], float)
        vm = np.asarray(sres["vm"], float)
        centroids = np.asarray(sres["centroids"], float)
        pitch = float(sres["pitch"])

        # -- FEM-COMPUTED-NOTHING guard: a run that yields no usable stress
        # field (min_fos=n/a, peak stress 0, empty/degenerate elements) is NOT
        # a valid reinforcement -- refuse rather than ship min_fos=n/a as "OK".
        _min_fos = sres.get("min_fos")
        _peak_vm = sres.get("peak_vm_Pa")
        if (vm.size == 0 or not np.isfinite(vm).any()
                or _peak_vm is None or not np.isfinite(float(_peak_vm))
                or float(_peak_vm) <= 0.0
                or _min_fos is None or not np.isfinite(float(_min_fos))):
            return {"ok": False,
                    "reason": ("the stress analysis computed nothing usable "
                               "(min_fos=n/a / peak stress 0) -- the part is "
                               "degenerate or below printable thickness; "
                               "cannot reinforce.")}
        if verbose:
            print(f"   FEM: nelem={sres['nelem']} pitch={pitch:.3f}mm "
                  f"min_fos={_fmt_fos(sres['min_fos'])} peak_vm={sres['peak_vm_Pa']:.2e}Pa")

        imp = mx.importance_field(fos, vm, mode=importance_mode)

        # graded-vs-uniform PRE-SCREEN from the SAME solved field (free, no 2nd
        # FEM). Uses inv_fos + p70 core regardless of importance_mode, matching
        # the validated wedge harness. Advisory only -- does NOT block building.
        grec = _grading_gradient_stats(fos, vm, centroids, pitch)
        grading_note = None
        if grec.get("ok") and not grec.get("recommend_grading"):
            grading_note = (
                "GRADING NOT RECOMMENDED for this load case: " + grec["reason"]
                + " The graded 3MF was still written (not blocked), but an "
                "equal-strength uniform infill would likely be lighter.")

        # 2) one modifier volume per tier --------------------------------- #
        base_vol = tw.Volume(base.vertices, base.faces, name="base",
                             is_modifier=False,
                             settings={"fill_density": f"{int(base_fill_density)}%"})
        volumes = [base_vol]
        tier_report = []
        for (pctile, dens, name) in sorted(tiers, key=lambda t: t[0]):
            occ_solid, occ_hot, origin, thr, frac = mx.high_stress_occupancy(
                centroids, pitch, imp, percentile=float(pctile),
                largest_component=largest_component)
            if not occ_hot.any():
                tier_report.append({"name": name, "percentile": pctile,
                                    "ok": False, "reason": "empty after threshold"})
                continue
            mod = mx.occupancy_to_mesh(occ_hot, pitch, origin, dilate=dilate)
            if mod is None:
                tier_report.append({"name": name, "percentile": pctile,
                                    "ok": False, "reason": "marching cubes empty"})
                continue
            # trim to strictly inside the base part (the slicer applies the modifier
            # to the part region it encloses; clipping guarantees containment)
            mod = mx.clip_to_base(mod, base)
            if mod is None or len(mod.faces) < 4:
                tier_report.append({"name": name, "percentile": pctile,
                                    "ok": False, "reason": "empty after clip"})
                continue
            chk = mx.verify_modifier(mod, base)
            volumes.append(tw.Volume(
                mod.vertices, mod.faces, name=name, is_modifier=True,
                settings={"fill_density": f"{int(dens)}%"}))
            tier_report.append({
                "name": name, "percentile": pctile, "dense_density_pct": int(dens),
                "ok": True, "frac_voxels_flagged": round(frac, 4),
                "threshold": thr, "watertight": chk["watertight"],
                "inside_base_bbox": chk["inside_base_bbox"],
                "volume_mm3": round(chk["volume_mm3"], 3),
                "n_faces": chk["n_faces"],
            })

        n_mod = sum(1 for t in tier_report if t.get("ok"))
        if n_mod == 0:
            return {"ok": False, "reason": "no modifier volume produced",
                    "tiers": tier_report}

        # 3) write the 3MF ------------------------------------------------- #
        parts = [{"name": "reinforced_part", "volumes": volumes}]
        out_path = tw.write_3mf(parts, out_3mf, include_orca=include_orca)

        # the modifier summary the slicer will apply (one entry per written volume)
        modifiers = [{
            "name": t["name"],
            "stress_percentile": t["percentile"],
            "infill_density_pct": t["dense_density_pct"],
            "fill_density": f"{t['dense_density_pct']}%",
            "volume_type": "ParameterModifier",
            "watertight": t["watertight"],
            "inside_base": t["inside_base_bbox"],
            "volume_mm3": t["volume_mm3"],
            "n_faces": t["n_faces"],
            "frac_voxels_flagged": t.get("frac_voxels_flagged"),
        } for t in tier_report if t.get("ok")]

        claim = (
            "RELATIVE structural grading: the modifier marks the higher-stress "
            "region for denser infill (more material where MORE loaded). The "
            "stress field reuses a closed-form-validated FEM solver but the FOS "
            "is COMPARATIVE/uncalibrated, so the dense % is a design choice, NOT "
            "a certified safety margin. The 3MF is SCHEMA-CORRECT against "
            "PrusaSlicer's documented Slic3r_PE_model.config modifier+fill_density "
            "format (3mf.cpp/Model.cpp); confirm graded printing by opening it in "
            "a slicer.")

        # 4) sidecar JSON -- the "structural passport" --------------------- #
        passport = {
            "tool": "EI-Forge reinforce (structural passport)",
            "schema_version": 1,
            "out_3mf": os.path.abspath(out_path),
            "load_case": load_case,
            "material": material if isinstance(material, str) else material.name,
            "base_fill_density_pct": int(base_fill_density),
            "importance_mode": importance_mode,
            "importance_field": (
                "von Mises stress (RELATIVE), normalized to [0,1]"
                if importance_mode == "vm"
                else "1/FOS (RELATIVE), normalized to [0,1]"),
            "modifiers": modifiers,
            "fem": {
                "nelem": int(sres["nelem"]),
                "pitch_mm": pitch,
                "min_fos": float(sres["min_fos"]),
                "peak_vm_Pa": float(sres["peak_vm_Pa"]),
                "min_fos_kind": sres.get(
                    "min_fos_kind",
                    "COMPARATIVE screening metric (not a certified SF)"),
            },
            "slicer_schema": {
                "target": "PrusaSlicer (verified vs 3mf.cpp/Model.cpp)",
                "config_part": "Metadata/Slic3r_PE_model.config",
                "modifier_flag": "volume_type=ParameterModifier + modifier=1",
                "infill_key": "fill_density (percent string, e.g. '80%')",
                "orca_bambu_note": (
                    "Metadata/model_settings.config with subtype=modifier_part + "
                    "sparse_infill_density (best-effort sibling; not source-diffed)"
                    if include_orca else "omitted"),
            },
            "uncalibrated": True,
            "fos_disclaimer": (
                "FOS / importance is RELATIVE and uncalibrated. It grades by "
                "comparative structural load (more material where MORE stressed), "
                "NOT a certified safety factor. The dense infill % is a design "
                "choice. 'Prints graded infill' is finally confirmed by opening "
                "the 3MF in a slicer."),
            "claim": claim,
        }
        passport_path = os.path.splitext(os.path.abspath(out_path))[0] + ".passport.json"
        try:
            with open(passport_path, "w", encoding="utf-8") as fh:
                json.dump(passport, fh, indent=2)
        except Exception:
            passport_path = None

        return {
            "ok": True,
            "out_3mf": os.path.abspath(out_path),
            "passport_json": passport_path,
            "n_modifiers": n_mod,
            "modifiers": modifiers,
            "tiers": tier_report,
            "report": passport,
            "base_fill_density_pct": int(base_fill_density),
            "base_faces": int(len(base.faces)),
            "importance_mode": importance_mode,
            "min_fos": float(sres["min_fos"]),
            "peak_vm_Pa": float(sres["peak_vm_Pa"]),
            "load_case": load_case,
            "material": material if isinstance(material, str) else material.name,
            "grading_recommendation": grec,
            "note": grading_note,
            "uncalibrated": True,
            "claim": claim,
        }
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


# =========================================================================== #
#  CLI                                                                         #
# =========================================================================== #
def _build_cli():
    import argparse
    p = argparse.ArgumentParser(
        prog="reinforce",
        description="Mesh + load case -> slicer-ready 3MF with graded (modifier) "
                    "infill on the load path. PrusaSlicer-compatible.")
    p.add_argument("mesh", help="input mesh (STL/OBJ/PLY/3MF/... anything trimesh "
                                 "loads); or 'cantilever' / 'lbracket' / 'psb' for "
                                 "a built-in demo geometry")
    p.add_argument("-o", "--out", default="reinforced.3mf", help="output 3MF path")
    p.add_argument("--material", default="PLA")
    p.add_argument("--load-face", default="zmax")
    p.add_argument("--anchor-face", default="zmin")
    p.add_argument("--load-axis", type=int, default=2)
    p.add_argument("--force", type=float, default=100.0, help="total force (N)")
    p.add_argument("--base-density", type=int, default=15, help="global infill %%")
    p.add_argument("--dense-density", type=int, default=80,
                   help="modifier (load-path) infill %%")
    p.add_argument("--percentile", type=float, default=70.0,
                   help="stress percentile -> dense core threshold")
    p.add_argument("--tiers", default=None,
                   help="multi-tier nesting spec 'pct:density:name,...' e.g. "
                        "'60:50:mid,85:100:core' (overrides --percentile/--dense-density). "
                        "Hotter tiers nest inside cooler ones; the slicer applies the "
                        "innermost (highest-percentile) modifier last, so it wins the overlap.")
    p.add_argument("--mode", default="vm", choices=["vm", "inv_fos"])
    p.add_argument("--max-elem", type=int, default=3000)
    p.add_argument("--no-orca", action="store_true",
                   help="omit the Orca/Bambu compatibility config")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _demo_mesh(name):
    import trimesh
    if name == "cantilever" or name == "lbracket":
        # a slender beam: clear bending load path -> clear high-stress root
        m = trimesh.creation.box(extents=[80.0, 12.0, 12.0])
        m.apply_translation([40.0, 6.0, 6.0])
        return m
    if name == "psb":
        # organic demo: a smooth capsule (variable section). The dev demo
        # loaded a PSB benchmark mesh; datasets are not shipped in meshprep.
        return trimesh.creation.capsule(height=60.0, radius=12.0, count=[16, 16])
    return trimesh.load(name, force="mesh")


def main(argv=None, mesh=None):
    # argparse handles -h / bad args with its own SystemExit -- let that pass.
    args = _build_cli().parse_args(argv)
    try:
        # -- SAME ingress guard prep uses (F1): size(25MB)/type/0-face/NaN are
        # rejected in plain English BEFORE any heavy load -- never a traceback.
        if mesh is None:
            if args.mesh in _DEMO_NAMES:
                mesh = _demo_mesh(args.mesh)
            else:
                from .preflight_core import GuardReject, load_and_guard
                try:
                    mesh, gnotes = load_and_guard(args.mesh)
                except GuardReject as g:
                    print(f"REFUSED: {g}")
                    return 2
                for n in gnotes:
                    print("  ", n)
        load_case = {
            "load_face": args.load_face, "anchor_face": args.anchor_face,
            "load_axis": args.load_axis, "total_force_N": args.force,
        }
        if args.tiers:
            tiers = []
            for spec in args.tiers.split(","):
                f = spec.split(":")
                pct, dens = float(f[0]), int(f[1])
                tiers.append((pct, dens,
                              f[2] if len(f) > 2 else f"tier_{int(pct)}"))
        else:
            tiers = [(args.percentile, args.dense_density, "dense_core")]
        res = reinforce(
            mesh, load_case, material=args.material, out_3mf=args.out,
            base_fill_density=args.base_density, tiers=tiers,
            importance_mode=args.mode, max_elem=args.max_elem,
            include_orca=not args.no_orca, verbose=args.verbose)
    except Exception as e:                          # never-raise to the user
        print(f"REFUSED: could not reinforce ({type(e).__name__}: {e}). "
              "Check the path and that the file is a closed solid mesh.")
        return 2
    if not res.get("ok"):
        # honest refusal (not a crash): degenerate / non-solid / FEM-nothing
        print("REFUSED:", res.get("reason"))
        return 2
    print(f"OK -> {res['out_3mf']}")
    if res.get("passport_json"):
        print(f"   passport  : {res['passport_json']}")
    print(f"   modifiers : {res['n_modifiers']}")
    for t in res["tiers"]:
        if t.get("ok"):
            print(f"   tier '{t['name']}': p>={t['percentile']:.0f} "
                  f"-> {t['dense_density_pct']}% infill, "
                  f"{t['frac_voxels_flagged']*100:.1f}% of voxels, "
                  f"watertight={t['watertight']} inside={t['inside_base_bbox']} "
                  f"vol={t['volume_mm3']:.1f}mm3")
    print(f"   base infill: {res['base_fill_density_pct']}%  "
          f"min_fos={_fmt_fos(res['min_fos'])} (RELATIVE) peak_vm={res['peak_vm_Pa']:.2e}Pa")
    grec = res.get("grading_recommendation") or {}
    if grec.get("ok"):
        yn = "yes" if grec.get("recommend_grading") else "no"
        print(f"   grading recommended: {yn} "
              f"(gradient {grec['gradient_ratio']:.1f}x vs "
              f"{grec['threshold']:.1f}x threshold)")
    else:
        print(f"   grading recommended: n/a ({grec.get('reason', 'unavailable')})")
    if res.get("note"):
        print("   ADVISORY:", res["note"])
    print("   NOTE:", res["claim"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
