"""meshprep.pipeline — the ONE entry point.

    prep(path_or_mesh, ...) -> PrepResult dict

Orchestrates the vendored, validated engines in `meshprep.core`:

    load -> guard -> analyze (printability / traps / topology
                              [+ warp FEM opt-in, + resin report in resin mode])
         -> fix (source-accurate repair w/ deviation certificate)
         -> orient (support-minimising build orientation)
         -> strengthen (graded-infill 3MF, when a load case is given)
         -> bed-fit advisor -> slicer savings (when a slicer is installed)
         -> package (prep STL + report.md + report.json + renders)

Discipline (house rules, carried through):
  * NEVER-RAISE: every stage degrades with a note; `prep` always returns a dict.
  * HONEST LABELS: geometric heuristics say so; the warp FEM says "uncalibrated"
    until a coupon is fitted; strength numbers are COMPARATIVE screening.
  * COMPUTE-CAPPED: FEM max_elem <= 2500, retopo/reinforce run on <= 6000-face
    decimated proxies, single-threaded numerics (set at package import).

PrepResult (dict) keys:
  ok, verdict ('PASS'|'WARN'|'FAIL'|'REJECTED'|'ERROR'), headline,
  report {markdown, json, md_path, json_path}, report_json (alias for the json),
  files {input, fixed_glb, fixed_stl, prep_stl, prep_3mf?, quad_obj?, gcode?,
         report_md, report_json},
  renders [png paths], savings (slicer numbers or None), channels, cert,
  fix (fidelity/deviation certificate info or None), stages, notes,
  profile, mode, out_dir.
"""
from __future__ import annotations

import os
import tempfile
import time

DEFAULT_PRINT_MM = 60.0

_AXIS_FACES = {0: ("xmax", "xmin"), 1: ("ymax", "ymin"), 2: ("zmax", "zmin")}


# ---------------------------------------------------------------------------
#  small helpers
# ---------------------------------------------------------------------------
def _load_case(spec, force_n=100.0):
    """Normalise a reinforce/strength load spec. Accepts None, a full dict
    ({"load_face","anchor_face","load_axis","total_force_N"} — missing fields
    defaulted), or an axis shorthand 'x'|'y'|'z'|0|1|2 (load on the +face,
    anchored at the -face, `force_n` newtons)."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        d = {"load_face": "zmax", "anchor_face": "zmin", "load_axis": 2,
             "total_force_N": float(force_n)}
        d.update(spec)
        return d
    ax = {"x": 0, "y": 1, "z": 2}.get(str(spec).lower())
    if ax is None:
        try:
            ax = int(spec)
        except (TypeError, ValueError):
            ax = 2
    if ax not in (0, 1, 2):
        ax = 2
    lf, af = _AXIS_FACES[ax]
    return {"load_face": lf, "anchor_face": af, "load_axis": ax,
            "total_force_N": float(force_n)}


class _Stages:
    """Timed, never-raise stage runner; rows feed the report's provenance table."""

    def __init__(self):
        self.rows: list[dict] = []

    def run(self, name, fn, label=""):
        t0 = time.perf_counter()
        row = {"stage": name, "ok": False, "seconds": None, "label": label,
               "note": ""}
        try:
            out = fn()
            row["ok"] = True
            return out
        except Exception as e:                      # never-raise discipline
            row["note"] = f"{type(e).__name__}: {e}"
            return None
        finally:
            row["seconds"] = round(time.perf_counter() - t0, 2)
            self.rows.append(row)

    def skip(self, name, why, label=""):
        self.rows.append({"stage": name, "ok": None, "seconds": 0.0,
                          "label": label, "note": f"skipped: {why}"})


def _write_quad_obj(V, Q, path):
    """Write a quad mesh as OBJ (v + 4-index f lines)."""
    with open(path, "w", encoding="utf-8") as fh:
        for v in V:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for q in Q:
            fh.write("f " + " ".join(str(int(i) + 1) for i in q) + "\n")


# ---------------------------------------------------------------------------
#  UNIT-INGEST CONTRACT  (never raises)
# ---------------------------------------------------------------------------
#  STL/OBJ carry NO units. We assume the mm-domain, but a part exported in
#  cm / m / inches lands at the wrong extent. We CANNOT infer units with
#  certainty, so the contract is DISCLOSURE + NUDGE, never a silent rescale.
_UNIT_FACTOR_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "inch": 25.4}


def scan_units(mesh) -> dict:
    """scan_units(mesh) -> {max_extent, likely_unit, warning}.

    Heuristic on the longest bbox side, assuming the mm-domain (a plausible
    desktop print is ~5-300 mm):

        < 2 mm   -> "m?"    (looks like metres, x1000 too small)
        < 5 mm   -> "inch?" (inches/cm scale; still rescaled by the else-branch)
        < 10 mm  -> "cm?"   (lands in the assume-mm band but suspiciously small)
        > 300 mm -> "mm"    (kept, but unusually large -> nudge)
        else     -> "mm"    (plausible)

    Returns a plain-English `warning` (or None) that the pipeline elevates from
    a footnote to a headline-visible WARN. NEVER raises, NEVER rescales.
    """
    try:
        if (mesh is None or not hasattr(mesh, "vertices")
                or len(mesh.vertices) == 0):
            return {"max_extent": 0.0, "likely_unit": "unknown", "warning": None}
        ext = float(max(mesh.bounds[1] - mesh.bounds[0]))
    except Exception:
        return {"max_extent": 0.0, "likely_unit": "unknown", "warning": None}
    # non-finite guard (NaN/inf bbox) -- honest refusal, no crash
    if not (ext == ext) or ext in (float("inf"), float("-inf")):
        return {"max_extent": ext, "likely_unit": "unknown",
                "warning": "the mesh size is not a finite number -- it may be "
                           "corrupt; cannot judge units."}
    warning = None
    if ext <= 0:
        unit = "unknown"
    elif ext < 2.0:
        unit = "m?"
        warning = (f"this part's longest side is only {ext:.3g} in file units -- "
                   f"that looks like METRES (about 1000x too small for mm). A "
                   f"real print is ~5-300 mm, so it will be rescaled to a "
                   f"{DEFAULT_PRINT_MM:g} mm default GUESS. Pass print_mm (or "
                   f"assume_unit='m') to set the true size.")
    elif ext < 5.0:
        unit = "inch?"
        warning = (f"this part's longest side is only {ext:.3g} in file units -- "
                   f"that looks like INCHES or cm (too small for mm). A real "
                   f"print is ~5-300 mm, so it will be rescaled to a "
                   f"{DEFAULT_PRINT_MM:g} mm default GUESS. Pass print_mm (or "
                   f"assume_unit) to set the true size.")
    elif ext < 10.0:
        unit = "cm?"
        warning = (f"this part is only {ext:.3g} mm on its longest side -- "
                   f"unusually small; did you export in cm or inches? If so, "
                   f"pass assume_unit='cm' (or --print-mm) to rescale. It is "
                   f"kept at {ext:.3g} mm for now, not silently changed.")
    elif ext > 300.0:
        unit = "mm"
        warning = (f"this part is {ext:.3g} mm on its longest side -- unusually "
                   f"large for a desktop printer; if you meant a smaller unit "
                   f"pass --print-mm (or assume_unit) to rescale. Kept as-is.")
    else:
        unit = "mm"
    return {"max_extent": ext, "likely_unit": unit, "warning": warning}


def _consult_gate(mesh, *, assume_unit="mm", context=None) -> dict | None:
    """Consult the printability/solidity gate (core._printability). Returns the
    gate dict, or None if the gate module is unavailable (documented residual).
    The gate is PURE and never raises; this wrapper is defensive regardless."""
    try:
        from .core._printability import assess_printability
    except Exception:
        return None
    try:
        return assess_printability(mesh, assume_unit=assume_unit, context=context)
    except Exception as e:
        # The gate MUST NOT raise per its contract; if it ever does, refuse
        # honestly rather than let a bad result ship under a green headline.
        return {"printable": False, "verdict": "FAIL",
                "issues": [{"code": "gate_error", "severity": "fail",
                            "message": f"could not verify printability "
                                       f"({type(e).__name__})"}],
                "checks": {},
                "plain_summary": "Could not verify this is printable -- refusing "
                                 "to claim success."}


# ---------------------------------------------------------------------------
#  the one call
# ---------------------------------------------------------------------------
def prep(path_or_mesh, *, profile="generic_fdm", material=None, mode=None,
         fix=True, orient=True, reinforce_load=None, reinforce_force_n=100.0,
         fem_warp=False, fem_strength=False, retopo=False,
         out_dir=None, print_mm=None, nozzle_mm=None, max_faces=None,
         assume_unit=None, check_only=False, slicer_savings=True,
         verbose=False) -> dict:
    """Drop a mesh -> print-ready package + plain-English report. Never raises.

    path_or_mesh     : mesh file path (GLB/OBJ/STL/PLY/OFF/3MF) or a trimesh.
    profile          : printer preset name / dict (meshprep.profiles).
    material         : 'PLA'|'PETG'|'ABS'|'resin'|... (default: profile's).
    mode             : 'fdm' | 'resin' (default: the profile's technology).
    fix              : source-accurate repair with a deviation certificate.
    orient           : reorient to the support-minimising build direction.
    reinforce_load   : None, axis 'x'|'y'|'z', or a full load-case dict ->
                       graded-infill 3MF (comparative FEM, uncalibrated).
    fem_warp         : opt-in inherent-strain warp FEM (uncalibrated unless a
                       coupon was fitted via `meshprep calibrate`).
    fem_strength     : opt-in orientation-strength screening (slow; 4 FEMs).
    retopo           : opt-in quad remesh of the prepped part (permissive
                       from-scratch backend) -> prep_quads.obj.
    out_dir          : where the package lands (default: a fresh temp dir).
    print_mm         : longest side of the printed part, mm. Default: keep the
                       mesh's own size if it looks like mm (5-500), else 60 mm.
    assume_unit      : explicit unit of the FILE ('mm'|'cm'|'m'|'inch'). When
                       given (and print_mm is not), the mesh extent is read in
                       that unit and converted to mm -- this is the honest way
                       to fix a mis-scaled export instead of the mm guess. It
                       clears the unit-sanity nudge (you told us the unit).
    max_faces        : decimate the working mesh above this many faces
                       (default: the validated ingress cap, 60k). Lower it
                       (e.g. 6000) for fast triage on a shared box — analysis
                       then runs on the simplified mesh (noted in the report).
    check_only       : analysis + report only; no fix/orient/files beyond the
                       report and renders (the API stub uses this).
    slicer_savings   : before/after print-time+filament via an installed
                       PrusaSlicer/Orca (degrades to a note when absent).
    """
    try:
        return _prep(path_or_mesh, profile=profile, material=material,
                     mode=mode, fix=fix, orient=orient,
                     reinforce_load=reinforce_load,
                     reinforce_force_n=reinforce_force_n, fem_warp=fem_warp,
                     fem_strength=fem_strength, retopo=retopo, out_dir=out_dir,
                     print_mm=print_mm, nozzle_mm=nozzle_mm,
                     max_faces=max_faces, assume_unit=assume_unit,
                     check_only=check_only,
                     slicer_savings=slicer_savings, verbose=verbose)
    except Exception as e:                          # never-raise, period
        msg = f"{type(e).__name__}: {e}"
        plain = ("Sorry -- meshprep hit an internal error and could not finish. "
                 "Your file was not changed. Nothing was produced, so treat this "
                 "as FAIL (do not print).")
        return {"ok": False, "verdict": "ERROR",
                "headline": plain, "plain_summary": plain,
                "error": msg, "channels": {},
                "files": {}, "renders": [], "stages": [], "notes": [],
                "report": {"markdown": f"# meshprep report -- ERROR\n\n{plain}\n\n"
                                       f"_details: {msg}_\n",
                           "json": {"verdict": "ERROR", "ok": False,
                                    "headline": plain, "error": msg},
                           "md_path": None, "json_path": None}}


def _prep(path_or_mesh, *, profile, material, mode, fix, orient,
          reinforce_load, reinforce_force_n, fem_warp, fem_strength, retopo,
          out_dir, print_mm, nozzle_mm, max_faces, assume_unit, check_only,
          slicer_savings, verbose):
    import numpy as np

    from . import report as report_mod
    from .core import preflight_core as pc
    from .core._mesh_util import say
    from .profiles import get_profile, scale_to_fit

    stages = _Stages()
    notes: list[str] = []

    # -- resolve profile / mode / material ---------------------------------
    prof = get_profile(profile)
    if "fallback_note" in prof:
        notes.append(prof["fallback_note"])
    mode = (mode or prof.get("technology") or "fdm").lower()
    if mode not in ("fdm", "resin"):
        notes.append(f"unknown mode {mode!r} -> fdm")
        mode = "fdm"
    material = str(material or ("resin" if mode == "resin"
                                else prof.get("material", "PLA")))
    nozzle = float(nozzle_mm or prof.get("nozzle_mm")
                   or prof.get("pixel_mm") or 0.4)
    if check_only:
        fix = orient = retopo = slicer_savings = False
        reinforce_load = None

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="meshprep_")
        notes.append(f"no out_dir given -> {out_dir}")
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # -- out_dir WRITABILITY probe: fail fast and plainly. (Foolproofing find:
    #    an ACL-denied out_dir used to yield PASS with zero deliverables.) -----
    try:
        _probe = os.path.join(out_dir, ".meshprep_write_probe")
        with open(_probe, "w") as _pf:
            _pf.write("ok")
        os.remove(_probe)
    except Exception as _we:
        return {"ok": False, "verdict": "ERROR",
                "headline": (f"Can't save results into the output folder "
                             f"({out_dir}) -- {type(_we).__name__}. Pick a "
                             "writable folder (out_dir=...) and run again."),
                "out_dir": out_dir, "files": {}, "channels": {},
                "renders": [], "savings": None, "fix": None, "cert": None,
                "stages": stages.rows,
                "notes": notes + ["out_dir writability probe failed: "
                                  f"{type(_we).__name__}: {_we}"]}

    result: dict = {"ok": False, "verdict": "ERROR", "headline": "",
                    "profile": prof.get("name"), "mode": mode,
                    "material": material, "out_dir": out_dir,
                    "channels": {}, "files": {}, "renders": [],
                    "savings": None, "fix": None, "cert": None,
                    "stages": stages.rows, "notes": notes}

    # -- 1. load + guard ----------------------------------------------------
    src_path = None
    cap = int(max_faces) if max_faces else pc.DECIMATE_FACES
    if isinstance(path_or_mesh, (str, os.PathLike)):
        src_path = str(path_or_mesh)
        result["files"]["input"] = src_path
        try:
            mesh, lnotes = pc.load_and_guard(src_path, decimate_faces=cap)
            notes.extend(lnotes)
            stages.rows.append({"stage": "load+guard", "ok": True,
                                "seconds": None, "label": "ingress guards",
                                "note": "; ".join(lnotes)})
        except pc.GuardReject as g:
            result.update({"verdict": "REJECTED", "rejected": str(g),
                           "headline": f"Upload refused -- {g}"})
            md, js = report_mod.build_report(result)
            result["report"] = report_mod.attach(result, md, js, out_dir)
            return result
    else:
        mesh = path_or_mesh
        if mesh is None or not hasattr(mesh, "faces") or len(mesh.faces) == 0:
            result.update({"verdict": "REJECTED",
                           "rejected": "not a triangle mesh",
                           "headline": "Input is not a triangle mesh."})
            md, js = report_mod.build_report(result)
            result["report"] = report_mod.attach(result, md, js, out_dir)
            return result
        mesh = mesh.copy()
        if len(mesh.faces) > cap:
            from .core._mesh_util import decimate
            n_in = len(mesh.faces)
            mesh = decimate(mesh, cap)
            notes.append(f"decimated {n_in:,} -> {len(mesh.faces):,} faces "
                         "(analysis runs on the simplified mesh)")
        stages.rows.append({"stage": "load+guard", "ok": True, "seconds": None,
                            "label": "in-memory mesh", "note": ""})

    # -- 2. units / print size (UNIT-INGEST CONTRACT; never silently rescale) --
    unit_scan = scan_units(mesh)
    result["unit_scan"] = unit_scan
    ext = float(unit_scan.get("max_extent") or 0.0)
    au = str(assume_unit).lower() if assume_unit else None
    if print_mm:
        eff = float(print_mm)
        notes.append(f"scaled so longest side = {eff:g} mm (explicit print_mm)")
        unit_scan["warning"] = None            # user gave the size outright
    elif au in _UNIT_FACTOR_MM:
        # explicit unit override -> convert honestly (the RIGHT fix for a
        # mis-scaled export; not a guess). Clears the unit-sanity nudge.
        eff = ext * _UNIT_FACTOR_MM[au] if ext > 0 else DEFAULT_PRINT_MM
        notes.append(f"interpreted the file as {au} (explicit assume_unit) -> "
                     f"longest side {eff:g} mm")
        unit_scan["warning"] = None
    elif 5.0 <= ext <= 500.0:
        # the assume-mm band: keep the mesh's own size, but a suspicious size
        # (unit_scan.warning) is elevated to a headline WARN below -- NOT a
        # silent green PASS.
        eff = ext
        notes.append(f"units assumed mm (longest side {ext:.1f} mm); pass "
                     "print_mm or assume_unit to rescale")
    else:
        # out of the plausible mm range -> a GUESS at the default size. Kept
        # explicitly labelled a guess (a 200 mm-in-metres part is ALSO forced
        # to the default here; that is why it stays override-able).
        eff = DEFAULT_PRINT_MM
        notes.append(f"mesh extent {ext:.3g} looks unitless/mis-scaled -> GUESS: "
                     f"scaled longest side to {eff:g} mm (pass print_mm or "
                     f"assume_unit to override)")
    if not (eff == eff) or eff <= 0:               # NaN/inf/zero -> safe default
        eff = DEFAULT_PRINT_MM
    mesh_mm = pc._scaled_to_mm(mesh, eff)
    result["print_mm"] = eff

    # -- 3. analyze -----------------------------------------------------------
    lc = _load_case(reinforce_load, reinforce_force_n)
    strength_lc = lc or (_load_case("z", reinforce_force_n) if fem_strength else None)
    checks = stages.run(
        "analyze",
        lambda: pc.run_checks(mesh_mm, print_mm=eff, nozzle_mm=nozzle,
                              work_dir=out_dir, check_fem=fem_warp,
                              material=material, check_strength=fem_strength,
                              load_case=strength_lc),
        label="printability + traps: geometric heuristics; warp: inherent-strain "
              "FEM; strength: comparative screening")
    if checks is None:
        checks = {"facts": {}, "channels": {}, "renders": {},
                  "print_mm": eff, "nozzle_mm": nozzle}
    result["channels"] = checks.get("channels", {})
    result["facts"] = checks.get("facts", {})
    cert = stages.run("verdict", lambda: pc.build_cert(checks),
                      label="ranked issues + fixes") or \
        {"verdict": "WARN", "headline": "analysis incomplete", "issues": [],
         "good": []}
    result["cert"] = cert

    # -- 3b. resin mode -------------------------------------------------------
    if mode == "resin":
        from . import resin as resin_mod
        rr = stages.run("resin",
                        lambda: resin_mod.resin_report(mesh_mm, profile=prof,
                                                       verbose=verbose),
                        label="voxel triage (res<=96), not slicer-grade")
        if rr:
            result["channels"]["resin"] = rr

    # -- 4. fix ---------------------------------------------------------------
    work_mesh = mesh_mm
    after_cert = None
    if fix:
        r = stages.run("fix",
                       lambda: pc.run_fixer(mesh_mm, work_dir=out_dir,
                                            print_mm=eff),
                       label="source-accurate repair + deviation certificate")
        if r is not None:
            glb, stl, fnote, fixed_m, dev_cert = r
            work_mesh = fixed_m
            result["files"]["fixed_glb"] = glb
            result["files"]["fixed_stl"] = stl
            after = stages.run(
                "re-check",
                lambda: pc.run_checks(fixed_m, print_mm=eff, nozzle_mm=nozzle,
                                      work_dir=os.path.join(out_dir, "after")),
                label="verdict flip (before -> after)")
            after_cert = pc.build_cert(after) if after else None
            result["fix"] = {"note": fnote,
                             "deviation_certificate": dev_cert,
                             "fidelity_line": pc._fidelity_line(dev_cert),
                             "before_verdict": cert["verdict"],
                             "after_verdict": (after_cert or {}).get("verdict"),
                             "after_headline": (after_cert or {}).get("headline")}
    else:
        stages.skip("fix", "disabled" if not check_only else "check-only")

    # -- 4b. internal-shell (cavity) census ----------------------------------
    # GEOMETRIC estimate: the faithful fixer keeps internal shells as CAVITIES,
    # so a faithful print uses less material than a cavity-filling auto-repair.
    # Runs on the (fixed, if fix ran) working mesh; additive to the report.
    def _shells():
        from .core._internal_shells import analyze_internal_shells
        return analyze_internal_shells(work_mesh)
    ish = stages.run("internal-shells", _shells,
                     label="geometric cavity census (component split + bbox "
                           "nesting); NOT a slicer measurement")
    if isinstance(ish, dict):
        result["internal_shells"] = ish
        if isinstance(result.get("fix"), dict):
            result["fix"]["internal_shells"] = ish

    # -- 5. orient ------------------------------------------------------------
    if orient:
        from .core import _print3d as p3

        def _orient():
            base_frac = None
            try:
                sev, needs = p3._face_overhang(work_mesh, [0.0, 0.0, 1.0])
                base_frac = float(np.sum(np.asarray(
                    work_mesh.area_faces)[needs]) / (work_mesh.area + 1e-12))
            except Exception:
                pass
            ori = p3.optimize_orientation(work_mesh, verbose=False)
            m = work_mesh.copy()
            m.apply_transform(ori["transform"])
            m.apply_translation([0.0, 0.0, -float(m.bounds[0][2])])  # onto bed
            return ori, m, base_frac

        r = stages.run("orient", _orient,
                       label="support-minimising build direction (geometric)")
        if r is not None:
            ori, oriented, base_frac = r
            work_mesh = oriented
            result["channels"]["orientation"] = {
                "up": [round(float(x), 3) for x in ori["up"]],
                "support_area_frac": round(float(ori["support_frac"]), 4),
                "support_area_frac_as_loaded": (None if base_frac is None
                                                else round(base_frac, 4)),
                "height_mm": round(float(ori["height"]), 2),
                "method": "geometric overhang minimisation (45 deg rule)"}
    else:
        stages.skip("orient", "disabled" if not check_only else "check-only")

    # -- the deliverable mesh ---------------------------------------------------
    prep_stl = None
    if not check_only:
        def _export():
            p = os.path.join(out_dir, "prep.stl")
            work_mesh.export(p)
            return p
        prep_stl = stages.run("package", _export, label="prep.stl (mm, oriented)")
        if prep_stl:
            result["files"]["prep_stl"] = prep_stl

    # -- 6. strengthen ----------------------------------------------------------
    if lc is not None and not check_only:
        from .core import reinforce as rf
        out3mf = os.path.join(out_dir, "reinforced.3mf")
        rr = stages.run(
            "reinforce",
            lambda: rf.reinforce(work_mesh, lc, material=material,
                                 out_3mf=out3mf, max_elem=2500,
                                 verbose=verbose),
            label="graded-infill 3MF; min_fos is COMPARATIVE, uncalibrated")
        if rr:
            from .core._mesh_util import fmt_fos
            result["channels"]["reinforce"] = {
                "ok": rr.get("ok"),
                "reason": rr.get("reason"),
                "out_3mf": rr.get("out_3mf"),
                "n_modifiers": rr.get("n_modifiers"),
                "min_fos": rr.get("min_fos"),
                "min_fos_display": fmt_fos(rr.get("min_fos")),
                "peak_vm_Pa": rr.get("peak_vm_Pa"),
                "claim": rr.get("claim"),
                "load_case": lc}
            if rr.get("ok") and rr.get("out_3mf"):
                result["files"]["prep_3mf"] = rr["out_3mf"]

    # -- 7. bed fit ---------------------------------------------------------------
    fit = stages.run("bed-fit", lambda: scale_to_fit(work_mesh, prof),
                     label="AABB 6-permutation triage, not packing")
    if fit:
        result["channels"]["fit"] = fit
        if fit.get("fits") is False:
            notes.append("part exceeds the bed -- scale by "
                         f"{fit.get('suggested_scale')} or split it "
                         "(meshprep.split.split_for_bed)")

    # -- 8. retopo (opt-in) --------------------------------------------------------
    if retopo and not check_only:
        def _retopo():
            from .core import quad_remesh as qr
            from .core._mesh_util import decimate
            base = decimate(work_mesh.copy(), 6000)
            V, Q = qr.quad_remesh(base, target_quads=1500)
            if len(Q) == 0:
                return {"ok": False, "reason": "remesh returned no quads"}
            p = os.path.join(out_dir, "prep_quads.obj")
            _write_quad_obj(V, Q, p)
            qual = qr.quality(base, V, Q)
            return {"ok": True, "quad_obj": p, **qual}
        rt = stages.run("retopo", _retopo,
                        label="from-scratch permissive quad remesh (aligned)")
        if rt:
            result["channels"]["retopo"] = rt
            if rt.get("quad_obj"):
                result["files"]["quad_obj"] = rt["quad_obj"]

    # -- 9. slicer savings -----------------------------------------------------------
    if slicer_savings and prep_stl and not check_only:
        def _savings():
            from . import slicer as sl
            if sl.find_slicer() is None:
                return {"ok": False,
                        "note": "no slicer installed -- savings unavailable "
                                "(install PrusaSlicer for before/after numbers)"}
            before = os.path.join(out_dir, "original_for_slice.stl")
            mesh_mm.export(before)
            return sl.compare(before, prep_stl, profile=prof,
                              label_a="as uploaded", label_b="prepped")
        result["savings"] = stages.run(
            "slicer", _savings,
            label="the slicer's own estimates under this profile")
    elif slicer_savings and not check_only:
        stages.skip("slicer", "no prepped file to compare")

    # -- verdict + report ------------------------------------------------------------
    final_cert = after_cert or cert
    result["ok"] = True
    base_verdict = final_cert.get("verdict", "WARN")
    base_headline = final_cert.get("headline", "")
    result["renders"] = [p for p in checks.get("renders", {}).values()
                         if p and os.path.exists(p)]

    # -- PRINTABILITY GATE (authoritative; can only DOWNGRADE, never upgrade) --
    # Consult the solidity/printability gate on the FINAL deliverable mesh
    # (fixed+oriented in full mode; the analysed input in check-only). A
    # FAIL-printable output can NEVER be reported under a PASS/success headline.
    gate = _consult_gate(work_mesh, assume_unit="mm",
                         context={"mode": mode, "check_only": check_only})
    result["printability_gate"] = gate
    _order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    if isinstance(gate, dict):
        gv = gate.get("verdict")
        gate_issues = [i for i in (gate.get("issues") or [])
                       if isinstance(i, dict)]
        result["gate_issues"] = gate_issues
        for i in gate_issues:                      # surface plainly in notes too
            sev = str(i.get("severity", "")).upper()
            notes.append(f"[{sev or 'GATE'}] {i.get('message', '')}")
        if _order.get(gv, 1) > _order.get(base_verdict, 1):
            base_verdict = gv                      # gate is stricter -> adopt it
    elif gate is None:
        notes.append("printability gate unavailable -- solidity not independently "
                     "re-checked (residual: install/verify core._printability).")

    # -- UNIT-SANITY nudge (can only downgrade PASS -> WARN; never a false PASS) --
    uwarn = (unit_scan or {}).get("warning")
    result["unit_warning"] = uwarn
    if uwarn:
        notes.append("UNIT CHECK: " + uwarn)
        if base_verdict == "PASS":
            base_verdict = "WARN"

    # a FAIL-printable (or errored gate) output is NOT a success
    if base_verdict == "FAIL":
        result["ok"] = False

    # -- deliverable-exists guard: never declare success without the file on
    #    disk (belt to the ingress writability probe: catches mid-run write
    #    failures -- permissions lost, disk full, antivirus quarantine) --------
    deliver_fail = None
    if not check_only and base_verdict in ("PASS", "WARN"):
        _ship = (result.get("files") or {}).get("prep_stl")
        if not (_ship and os.path.isfile(_ship)):
            base_verdict = "ERROR"
            result["ok"] = False
            deliver_fail = (f"The prepared mesh could not be saved into "
                            f"{out_dir} -- nothing was delivered, so this is "
                            "not a success. Check the folder is writable and "
                            "has space, or pick another (out_dir=...).")
            notes.append("DELIVERABLE MISSING: prep.stl absent at package "
                         "time; verdict downgraded to ERROR.")

    # headline: put the plain refusal/nudge FIRST so a novice sees it up top
    head_bits: list[str] = []
    if deliver_fail:
        head_bits.append(deliver_fail)
    if isinstance(gate, dict) and gate.get("verdict") == "FAIL":
        head_bits.append(str(gate.get("plain_summary")
                             or "This is not a printable closed solid."))
    if uwarn:
        head_bits.append(uwarn)
    if base_headline:
        head_bits.append(base_headline)
    result["verdict"] = base_verdict
    result["headline"] = " ".join(head_bits).strip() or base_headline

    md, js = report_mod.build_report(result)
    # additive report keys: internal-shell census (GEOMETRIC estimate, not a
    # slicer measurement). fix.internal_shells already rides along inside "fix".
    ish = result.get("internal_shells")
    if isinstance(ish, dict):
        js["n_internal_shells"] = ish.get("n_internal_shells")
        js["solid_volume_ratio"] = ish.get("solid_volume_ratio")
    result["report"] = report_mod.attach(result, md, js, out_dir)
    result["report_json"] = js
    if result["report"].get("md_path"):
        result["files"]["report_md"] = result["report"]["md_path"]
    if result["report"].get("json_path"):
        result["files"]["report_json"] = result["report"]["json_path"]
    if isinstance(result.get("savings"), dict):
        g = (result["savings"].get("b") or {}).get("gcode_path")
        if g:
            result["files"]["gcode"] = g
    if verbose:
        say(md)
    return result
