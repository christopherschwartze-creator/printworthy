"""EI Forge — Pre-Flight core pipeline (host-agnostic).

The value proposition, made concrete: drop a mesh, get a *verdict you can trust*
— PASS / WARN / FAIL — with the ONE reason it'll fail, WHERE, and the single fix,
plus the visual heatmap and the numbers a buyer believes. If they hit "Fix", they
get a before -> after PASS flip and a downloadable file. That flip is the value.

This module is the engine; the Gradio app is the shell around it. The checker
modules live alongside it in printworthy.core (vendored, package-relative).

Honesty stance (matches the rest of the toolkit): every print/strength signal is a
labelled GEOMETRIC heuristic, not a slicer/FEM sim. "watertight" never means
"genus-correct". The cert says what it measured and what it did not.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

import numpy as np

import trimesh

from ._mesh_util import fmt_fos as _ffos

# --- ingress guards (the only non-negotiables once strangers can upload) ----------
MAX_BYTES = 25 * 1024 * 1024        # reject uploads over 25 MB
HARD_MAX_FACES = 3_000_000          # reject absurd meshes outright
DECIMATE_FACES = 60_000             # decimate anything above this (speed/OOM)
ACCEPTED_EXT = {".glb", ".gltf", ".obj", ".stl", ".ply", ".off", ".3mf"}

# --- defaults --------------------------------------------------------------------
DEFAULT_PRINT_MM = 60.0             # longest bbox axis, mm (meshes are unitless)
DEFAULT_NOZZLE_MM = 0.4
WARP_WARN_MM = 0.3                  # predicted corner-lift (mm) above which we WARN


class GuardReject(Exception):
    """Raised when an upload is rejected at ingress (size / shape / NaN)."""


# =================================================================================
#  load + guard
# =================================================================================
def load_and_guard(path: str, *, max_bytes: int = MAX_BYTES,
                   hard_max_faces: int = HARD_MAX_FACES,
                   decimate_faces: int = DECIMATE_FACES):
    """Load an uploaded mesh, rejecting anything dangerous. Returns
    (mesh, notes:list[str]). Raises GuardReject on a hard reject."""
    notes: list[str] = []
    ext = os.path.splitext(path)[1].lower()
    if ext not in ACCEPTED_EXT:
        raise GuardReject(f"Unsupported file type '{ext}'. "
                          f"Accepted: {', '.join(sorted(ACCEPTED_EXT))}.")
    try:
        size = os.path.getsize(path)
    except OSError:
        raise GuardReject("Could not read the uploaded file.")
    if size > max_bytes:
        raise GuardReject(f"File is {size/1e6:.1f} MB — over the {max_bytes/1e6:.0f} MB "
                          "test limit. Decimate it first or use a smaller asset.")
    try:
        loaded = trimesh.load(path, force="mesh", process=False)
    except Exception as e:
        raise GuardReject(f"Could not parse the mesh ({type(e).__name__}).")
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if hasattr(g, "faces")]
        if not geoms:
            raise GuardReject("File contains no triangle geometry.")
        loaded = trimesh.util.concatenate(geoms)
    if not hasattr(loaded, "faces") or len(loaded.faces) == 0:
        raise GuardReject("File contains no triangle faces.")
    if not np.isfinite(np.asarray(loaded.vertices)).all():
        raise GuardReject("Mesh has non-finite (NaN/Inf) vertex coordinates — refusing.")
    if len(loaded.faces) > hard_max_faces:
        raise GuardReject(f"{len(loaded.faces):,} faces — over the {hard_max_faces:,} "
                          "hard cap for this preview.")
    n_in = len(loaded.faces)
    if n_in > decimate_faces:
        from ._mesh_util import decimate
        loaded = decimate(loaded, decimate_faces)
        if len(loaded.faces) < n_in:
            notes.append(f"Decimated {n_in:,} -> {len(loaded.faces):,} faces for a fast preview "
                         "(analysis runs on the simplified mesh).")
        else:
            notes.append(f"Large mesh ({n_in:,} faces); analysis may be slow.")
    try:
        loaded.process(validate=True)
        loaded.remove_unreferenced_vertices()
    except Exception:
        pass
    return loaded, notes


def _scaled_to_mm(mesh: trimesh.Trimesh, print_mm: float) -> trimesh.Trimesh:
    """A COPY scaled so the longest bbox axis == print_mm (so wall/nozzle are in mm)."""
    m = mesh.copy()
    ext = float((m.bounds[1] - m.bounds[0]).max())
    if ext > 0:
        m.apply_scale(print_mm / ext)
    return m


# =================================================================================
#  topology facts (the trustworthy numbers)
# =================================================================================
def topology_facts(mesh: trimesh.Trimesh) -> dict:
    wt = bool(getattr(mesh, "is_watertight", False))
    wind = bool(getattr(mesh, "is_winding_consistent", False))
    out = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": wt,
        "winding_consistent": wind,
        "bbox_mm": [round(float(x), 2) for x in (mesh.bounds[1] - mesh.bounds[0])],
    }
    try:
        euler = int(mesh.euler_number)
        out["euler"] = euler
        out["genus"] = (2 - euler) // 2 if wt else None
    except Exception:
        out["euler"] = None
        out["genus"] = None
    try:
        out["volume_cm3"] = round(float(abs(mesh.volume)) / 1000.0, 2) if wt else None
    except Exception:
        out["volume_cm3"] = None
    return out


# =================================================================================
#  run the three checkers (each defensive; a failed channel degrades, never raises)
# =================================================================================
def run_checks(mesh: trimesh.Trimesh, *, print_mm: float, nozzle_mm: float,
               work_dir: str, check_animation: bool = False,
               check_fem: bool = False, material: str = "PLA",
               check_strength: bool = False, load_case: dict | None = None) -> dict:
    os.makedirs(work_dir, exist_ok=True)
    m = _scaled_to_mm(mesh, print_mm)
    report: dict = {"facts": topology_facts(m), "print_mm": print_mm,
                    "nozzle_mm": nozzle_mm, "renders": {}, "channels": {}}

    # --- #1 printability heatmap -------------------------------------------------
    build_dir = None
    try:
        from . import _print_premortem as pm
        pr = pm.premortem(m, nozzle=nozzle_mm)
        comb = pr["summary"]["combined"]
        thin = pr["summary"].get("thin", {})
        report["channels"]["printability"] = {
            "max_risk": comb.get("max_risk"),
            "high_risk_area_frac": comb.get("high_risk_area_frac"),
            "n_high_risk_faces": comb.get("n_high_risk_faces(>0.5)"),
            "dominant": comb.get("dominant_channel_of_risk_faces", {}),
            "min_wall_mm": thin.get("min_wall_mm"),
        }
        build_dir = pr.get("build_dir")
        png = os.path.join(work_dir, "premortem.png")
        try:
            pm.render_premortem(m, png, build_dir=build_dir, result=pr,
                                title="Printability risk (oriented to minimise supports)",
                                nozzle=nozzle_mm)
            report["renders"]["printability"] = png
        except Exception:
            pass
    except Exception as e:
        report["channels"]["printability"] = {"error": f"{type(e).__name__}: {e}"}

    # --- #5 resin traps ----------------------------------------------------------
    try:
        from . import _print_traps as tp
        bd = build_dir if build_dir is not None else np.array([0.0, 0.0, 1.0])
        # keep_masks: lets render_traps reuse the voxel masks instead of
        # re-voxelizing (which used to double the trap cost); the private
        # cache is popped right after rendering, so nothing non-serializable
        # leaves this function.
        tr = tp.find_traps(m, np.asarray(bd, float), keep_masks=True)
        report["channels"]["resin_traps"] = {
            "n_traps": tr.get("n_traps"),
            "external_cup_cm3": tr.get("external_cup_cm3"),
            "internal_cavity_cm3": tr.get("internal_cavity_cm3"),
            "total_trapped_cm3": tr.get("total_trapped_cm3"),
            "n_external_cups": len(tr.get("external_cups", [])),
            "n_internal_cavities": len(tr.get("internal_cavities", [])),
            "vents": tr.get("vents", [])[:5],
        }
        png = os.path.join(work_dir, "traps.png")
        try:
            tp.render_traps(m, tr, png, title="Resin pooling / trapped volume")
            report["renders"]["resin_traps"] = png
        except Exception:
            pass
        finally:
            tr.pop("_render_cache", None)
    except Exception as e:
        report["channels"]["resin_traps"] = {"error": f"{type(e).__name__}: {e}"}

    # --- #7 deformation readiness (optional; animation lens) ---------------------
    if check_animation:
        report["channels"]["deformation"] = {
            "note": "animation / rig-readiness checks ship in the separate "
                    "animation build (autorig); not bundled in printworthy"}

    # --- FEM warp (inherent-strain distortion; opt-in, heavier: voxel + solve) ---
    if check_fem:
        try:
            bd = build_dir   # reuse the support-minimal orientation from premortem
            try:
                from . import _warp_calibrate as wc   # calibration-aware (auto-upgrades)
                wr = wc.calibrated_warp(m, material=material, build_dir=bd)
            except Exception:
                from . import _print_fem as pf        # uncalibrated fallback
                wr = pf.warp_analysis(m, material=material, build_dir=bd)
            if wr.get("ok"):
                report["channels"]["warp"] = {
                    "max_corner_lift_mm": wr.get("max_corner_lift_mm"),
                    "sign": wr.get("sign"),
                    "material": wr.get("material"),
                    "calibrated": bool(wr.get("calibrated", False)),
                    "cal_scale": wr.get("cal_scale"),
                    "cal_label": wr.get("cal_label"),
                    "uncalibrated": wr.get("uncalibrated", True),
                    "magnitude_kind": wr.get("magnitude_kind"),
                    "warning": wr.get("warning"),
                }
                png = os.path.join(work_dir, "warp.png")
                try:
                    from . import _print_fem as pf
                    pf.render_warp(wr, png,
                                   title="Predicted print warp (inherent-strain FEM)")
                    report["renders"]["warp"] = png
                except Exception:
                    pass
            else:
                report["channels"]["warp"] = {
                    "note": wr.get("reason", "warp analysis unavailable")}
        except Exception as e:
            report["channels"]["warp"] = {"error": f"{type(e).__name__}: {e}"}

    # --- FEM strength + orient-for-strength (opt-in; needs a load case) -------
    if check_strength:
        try:
            from . import _print_fem as pf
            orr = pf.orient_for_strength(m, load_case=load_case, material=material,
                                         max_elem=2500, n_candidates=4, keep_results=True)
            if orr.get("ok"):
                report["channels"]["strength"] = {
                    "best_dir": orr.get("best_dir"),
                    "worst_dir": orr.get("worst_dir"),
                    "best_min_fos": orr.get("best_min_fos"),
                    "worst_min_fos": orr.get("worst_min_fos"),
                    "best_min_fos_display": _ffos(orr.get("best_min_fos")),
                    "worst_min_fos_display": _ffos(orr.get("worst_min_fos")),
                    "improvement_ratio": orr.get("improvement_ratio"),
                    "n_evaluated": orr.get("n_evaluated"),
                    "material": orr.get("material"),
                    "min_fos_kind": "COMPARATIVE screening (not a certified safety factor)",
                }
                try:
                    best, worst = orr.get("best_result"), orr.get("worst_result")
                    if best is not None:
                        png = os.path.join(work_dir, "strength.png")
                        pf.render_strength(best, png,
                                           title="Stress / weakest region (best orientation)")
                        report["renders"]["strength"] = png
                    if best is not None and worst is not None:
                        cmp_png = os.path.join(work_dir, "orient.png")
                        pf.render_orient_compare(best, worst, cmp_png,
                                                 label_strong="strongest orientation",
                                                 label_weak="weakest orientation")
                        report["renders"]["orient"] = cmp_png
                except Exception:
                    pass
            else:
                report["channels"]["strength"] = {
                    "note": orr.get("reason", "strength analysis unavailable")}
        except Exception as e:
            report["channels"]["strength"] = {"error": f"{type(e).__name__}: {e}"}

    return report


# =================================================================================
#  the cert — turn channels into a verdict + ranked plain-English issues + fixes
# =================================================================================
def build_cert(report: dict) -> dict:
    facts = report["facts"]
    ch = report["channels"]
    nozzle = report["nozzle_mm"]
    issues: list[dict] = []       # FAIL/WARN/INFO items, each {sev,title,detail,fix}
    good: list[str] = []          # the "what's already fine" list (value-feel)

    def add(sev, title, detail, fix):
        issues.append({"sev": sev, "title": title, "detail": detail, "fix": fix})

    # topology
    if not facts["watertight"]:
        add("FAIL", "Not watertight",
            "There are open holes/boundaries — a slicer can misread inside vs outside "
            "and produce a broken or hollow print.",
            "Run the free Fix (below): it seals the mesh into a single solid.")
    else:
        good.append("Watertight (a single sealed solid)")
    _genus = facts.get("genus")
    if facts["watertight"] and isinstance(_genus, int) and _genus > 0:
        add("WARN", f"{_genus} unexpected tunnel/handle(s)",
            "The surface has handles (genus > 0). Often these are phantom tunnels from "
            "the source geometry, not real holes you intended.",
            "Inspect the heatmap; the Fix can also remove thin phantom bridges.")
    elif facts["watertight"] and isinstance(_genus, int) and _genus < 0:
        # negative genus = the Euler count is inconsistent (many disjoint
        # pieces / non-manifold joins). The raw number is meaningless to a
        # user — never show it.
        add("WARN", "Surface connects to itself in unexpected ways",
            "The shell count does not add up to a simple solid — typical of a model "
            "made of many separate or interpenetrating pieces.",
            "Inspect the render; if the model looks whole, this is usually harmless.")
    if facts["watertight"] and not facts["winding_consistent"]:
        add("WARN", "Inconsistent face winding",
            "Some faces point inward — normals are not coherent.",
            "The free Fix re-orients all faces outward.")

    # printability
    p = ch.get("printability", {})
    mw = p.get("min_wall_mm")
    if isinstance(mw, (int, float)) and mw < nozzle:
        add("FAIL", f"Thinnest wall {mw:.2f} mm < {nozzle:.2f} mm nozzle",
            "There are walls thinner than a single extrusion — they will print as gaps "
            "or snap off.",
            f"Thicken those regions, or scale the model up ~{nozzle/max(mw,1e-3):.1f}×.")
    elif isinstance(mw, (int, float)):
        good.append(f"Walls clear the nozzle (thinnest {mw:.2f} mm ≥ {nozzle:.2f} mm)")
    af = p.get("high_risk_area_frac")
    if isinstance(af, (int, float)) and af >= 0.03:
        add("WARN", f"{af*100:.0f}% of the surface is overhang/support-risk",
            "Steep overhangs past the ~45° self-support angle need support material and "
            "will be scarred where supports touch.",
            "Analysed at the support-minimising orientation (`prep` applies it; `check` "
            "only reports it) — the heatmap shows the remaining support zones "
            "(red = supports + scarring).")
    elif isinstance(af, (int, float)):
        good.append("Few overhangs — minimal supports needed")

    # resin traps
    t = ch.get("resin_traps", {})
    icc = t.get("internal_cavity_cm3") or 0
    ncav = t.get("n_internal_cavities") or 0
    if ncav and icc > 0:
        loc = ""
        vents = t.get("vents", [])
        if vents and isinstance(vents[0], dict) and vents[0].get("pos") is not None:
            pos = vents[0]["pos"]
            loc = f" near ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}) mm"
        add("FAIL", f"Sealed cavity traps {icc:.1f} cm³ (resin/powder)",
            "An enclosed void will trap uncured resin or powder — a known resin-print "
            "failure and a cleaning nightmare.",
            f"Add a drain/vent hole{loc} (marked on the trap render).")
    ext = t.get("external_cup_cm3") or 0
    if (t.get("n_external_cups") or 0) and ext > 0:
        add("WARN", f"Top surface pools {ext:.1f} cm³ of resin",
            "An upward-facing cup holds liquid in this orientation.",
            "Flip or tilt the part, or add a relief channel.")

    # deformation channel: only ever a "ships separately (autorig)" note now —
    # no cert item (the rig-readiness checks are not bundled in printworthy).

    # FEM warp (inherent-strain distortion prediction)
    w = ch.get("warp")
    if w and isinstance(w.get("max_corner_lift_mm"), (int, float)):
        lift = w["max_corner_lift_mm"]
        cal = bool(w.get("calibrated"))
        if w.get("warning"):
            # e.g. resin: thermal-eigenstrain warp does not apply
            pass
        elif lift >= WARP_WARN_MM:
            calnote = (" — calibrated for your printer/filament" if cal
                       else " (lower-bound estimate, uncalibrated)")
            add("WARN", f"Predicted corner warp ≈ {lift:.2f} mm{calnote}",
                f"Inherent-strain FEM predicts the edges curl up off the bed by "
                f"≈ {lift:.2f} mm as the part cools — risk of corner lift / bed "
                "detachment, worst on large flat footprints and high-CTE materials "
                f"({w.get('material','PLA')}).",
                "Add a brim or raft, cut the flat-base area, enclose the chamber (ABS), "
                + ("" if cal else "and print a 1-bar calibration coupon for a certified mm."))
        else:
            good.append(f"Low predicted warp (≈ {lift:.2f} mm{'' if cal else ', est.'})")

    # FEM strength / orientation (orthotropic, comparative-only)
    s = ch.get("strength")
    if s and isinstance(s.get("improvement_ratio"), (int, float)):
        ratio = s["improvement_ratio"]

        def _d(v):
            return "[" + ", ".join(f"{x:.0f}" for x in v) + "]" if v else "?"
        if ratio >= 1.3:
            add("WARN", f"Build orientation changes strength ×{ratio:.1f}",
                "FDM is weakest across the print layers. Under this load, printing with the "
                f"layer-normal along {_d(s.get('best_dir'))} is ×{ratio:.1f} stronger (higher "
                f"minimum factor-of-safety) than the worst orientation {_d(s.get('worst_dir'))} "
                "— a comparative screening across orientations, not a certified margin.",
                f"Print in the recommended orientation {_d(s.get('best_dir'))} so the load runs "
                "along the layers, not across them.")
        else:
            good.append(f"Orientation barely affects strength here (×{ratio:.1f})")

    # verdict
    sevs = [i["sev"] for i in issues]
    if "FAIL" in sevs:
        verdict = "FAIL"
    elif "WARN" in sevs:
        verdict = "WARN"
    else:
        verdict = "PASS"

    n_fail = sevs.count("FAIL")
    n_warn = sevs.count("WARN")
    if verdict == "PASS":
        headline = "Ready to print. No blocking issues found."
    elif verdict == "WARN":
        headline = (f"Printable, with {n_warn} thing(s) to be aware of "
                    "(supports / orientation / minor topology).")
    else:
        first = next(i for i in issues if i["sev"] == "FAIL")
        headline = f"Won't print as-is — {first['title'].lower()}. " \
                   f"{n_fail} blocking issue(s); the free Fix resolves most."
    # rank: FAIL, WARN, INFO
    order = {"FAIL": 0, "WARN": 1, "INFO": 2}
    issues.sort(key=lambda i: order[i["sev"]])
    return {"verdict": verdict, "headline": headline, "issues": issues, "good": good}


# =================================================================================
#  the free fixer (commodity OSS) — make it printable, show before -> after
# =================================================================================
def run_fixer(mesh: trimesh.Trimesh, *, work_dir: str, print_mm: float):
    """Make-it-printable fix. Returns (out_glb_path, out_stl_path, note, mesh, cert).

    Primary path is the SOURCE-ACCURATE fixer (_fix_accurate.accurate_fix):
    localized cheap repair -> curvature-continuous EI-bisector hole fill ->
    voxel-seal+reproject ONLY for catastrophic non-manifold junctions -> carry
    vertex attributes -> two-sided Hausdorff deviation CERTIFICATE -> never-worse
    rollback (deviation AND genus). It prefers FIDELITY over the toy fixer's
    unconditional coarse voxel reseal: a hole gets a domed cap and the rest of
    the surface stays on the original, so the buyer sees a fix that preserved
    detail (high "% surface unchanged", low max deviation) and topology
    (no phantom handles) instead of a blobby genus-N reseal.

    The deviation certificate is returned as a 5th element so the cert UI can
    show "fix preserved 100% of the surface (max dev 0.06 mm) and topology
    (genus 0)". `cert` is {} when the accurate path is unavailable.

    Fallback: if the accurate fixer fails, we degrade to the ONE toy path
    (_fix_accurate.fix_toy: cheap repair + coarse voxel reseal) so the fix
    still ships."""
    note: list[str] = []
    cert: dict = {}

    from . import _fix_accurate as fa

    # ── primary: source-accurate fixer with deviation certificate ────────────
    try:
        fixed, rep = fa.accurate_fix(mesh, print_mm=print_mm, n_samp=6000,
                                     memory_budget_mb=1500.0)
        m = fixed
        c = rep.get("certificate", {})
        cert = {
            "surface_unchanged_pct": rep.get("surface_unchanged_pct"),
            "max_deviation_mm": rep.get("max_deviation_mm"),
            "worst_rms": c.get("worst_rms"),
            "genus_in": rep.get("genus_in"),
            "genus_out": rep.get("genus_out"),
            "watertight": rep.get("watertight"),
            "used_voxel_seal": rep.get("used_voxel_seal"),
            "rolled_back": rep.get("rolled_back"),
            "color_carry": rep.get("color_carry"),
            "stages": rep.get("stages"),
        }
        # plain-English provenance for the note line
        if rep.get("rolled_back"):
            note.append("kept the original mesh (a 'fix' would have made it worse — "
                        f"{', '.join(rep.get('rollback_reasons', []) or ['no net gain'])})")
        else:
            unchg = rep.get("surface_unchanged_pct")
            dev = rep.get("max_deviation_mm")
            gout = rep.get("genus_out")
            note.append("source-accurate fix: curvature-continuous hole fill, "
                        "detail preserved")
            if isinstance(unchg, (int, float)):
                note.append(f"{unchg:.0f}% of the surface kept verbatim")
            if isinstance(dev, (int, float)) and dev != float("inf"):
                note.append(f"max surface deviation {dev:.3f} mm")
            if gout is not None:
                note.append(f"topology preserved (genus {gout})")
            if rep.get("used_voxel_seal"):
                note.append("a real non-manifold junction needed a voxel bridge "
                            "(smallest-footprint seal; rest of surface untouched)")
            if rep.get("color_carry") == "nearest":
                note.append(f"vertex colours carried ({rep['color_carry']})")
    except Exception as e:
        # ── fallback: the ONE toy path (fix_toy = cheap repair + coarse reseal) ──
        note.append(f"source-accurate fixer failed ({type(e).__name__}); "
                    "using the toy fallback (cheap repair + coarse voxel reseal)")
        m, trep = fa.fix_toy(mesh)
        if trep.get("used_voxel_reseal"):
            note.append("voxel re-seal applied (guaranteed solid; fine detail simplified)")

    base = os.path.join(work_dir, "fixed")
    glb = base + ".glb"
    stl = base + ".stl"
    try:
        m.export(glb)
    except Exception:
        glb = None
    try:
        m.export(stl)
    except Exception:
        stl = None
    return glb, stl, "; ".join(note), m, cert


def _fidelity_line(cert: dict) -> str:
    """One-line buyer-facing summary of the deviation certificate. Empty string
    when no certificate is available (fallback toy path). Never raises."""
    if not cert:
        return ""
    try:
        if cert.get("rolled_back"):
            return ("No change shipped — a fix would have made the mesh worse, "
                    "so we kept your original.")
        unchg = cert.get("surface_unchanged_pct")
        dev = cert.get("max_deviation_mm")
        gout = cert.get("genus_out")
        parts: list[str] = []
        if isinstance(unchg, (int, float)):
            parts.append(f"preserved {unchg:.0f}% of the original surface")
        if isinstance(dev, (int, float)) and dev != float("inf"):
            parts.append(f"max deviation {dev:.3f} mm")
        if gout is not None:
            parts.append(f"topology intact (genus {gout})")
        if cert.get("used_voxel_seal"):
            parts.append("one non-manifold junction bridged")
        if not parts:
            return ""
        return "Fix " + ", ".join(parts) + "."
    except Exception:
        return ""


# =================================================================================
#  top-level orchestration
# =================================================================================
def preflight(path: str, *, print_mm: float = DEFAULT_PRINT_MM,
              nozzle_mm: float = DEFAULT_NOZZLE_MM, check_animation: bool = False,
              check_fem: bool = False, material: str = "PLA",
              check_strength: bool = False, load_case: dict | None = None,
              do_fix: bool = False, work_dir: str | None = None) -> dict:
    """Full pre-flight on an uploaded file. Never raises GuardReject upward without a
    friendly message; other exceptions are wrapped. Returns a dict the UI renders."""
    work_dir = work_dir or tempfile.mkdtemp(prefix="preflight_")
    out: dict = {"ok": False, "work_dir": work_dir}
    try:
        mesh, notes = load_and_guard(path)
    except GuardReject as g:
        out["rejected"] = str(g)
        return out
    out["ingress_notes"] = notes
    report = run_checks(mesh, print_mm=print_mm, nozzle_mm=nozzle_mm,
                        work_dir=work_dir, check_animation=check_animation,
                        check_fem=check_fem, material=material,
                        check_strength=check_strength, load_case=load_case)
    cert = build_cert(report)
    out.update({"ok": True, "report": report, "cert": cert})

    if do_fix:
        glb, stl, fnote, fixed, dev_cert = run_fixer(
            mesh, work_dir=work_dir, print_mm=print_mm)
        after = run_checks(fixed, print_mm=print_mm, nozzle_mm=nozzle_mm,
                           work_dir=os.path.join(work_dir, "after"),
                           check_animation=False)
        # ensure the 'after' render dir exists
        out["fixed"] = {
            "glb": glb, "stl": stl, "note": fnote,
            "before_verdict": cert["verdict"],
            "after_cert": build_cert(after),
            "after_report": after,
            # source-accurate deviation certificate: what the fix preserved.
            # The buyer reads "% surface unchanged" + "max deviation mm" +
            # genus to see the fix kept detail and topology (vs a toy blob).
            "deviation_certificate": dev_cert,
            "fidelity_line": _fidelity_line(dev_cert),
        }
    return out


# =================================================================================
#  CLI (for subprocess hosting / quick tests): prints JSON, writes renders to --out
# =================================================================================
def _main(argv):
    import argparse
    from ._mesh_util import ascii_console
    ascii_console()                     # cp1252-safe console (house pattern)
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--print-mm", type=float, default=DEFAULT_PRINT_MM)
    ap.add_argument("--nozzle", type=float, default=DEFAULT_NOZZLE_MM)
    ap.add_argument("--animation", action="store_true")
    ap.add_argument("--fem", action="store_true", help="inherent-strain warp FEM (slower)")
    ap.add_argument("--material", default="PLA", help="PLA|PETG|ABS|resin")
    ap.add_argument("--strength", action="store_true",
                    help="orthotropic strength + orient-for-strength FEM (slow)")
    ap.add_argument("--load-axis", type=int, default=2, choices=(0, 1, 2),
                    help="load direction for --strength: 0=X 1=Y 2=Z")
    ap.add_argument("--force", type=float, default=100.0, help="load (N) for --strength")
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args(argv)
    lc = None
    if a.strength:
        ax = a.load_axis
        face = {0: ("xmax", "xmin"), 1: ("ymax", "ymin"), 2: ("zmax", "zmin")}[ax]
        lc = {"load_face": face[0], "anchor_face": face[1], "load_axis": ax,
              "total_force_N": a.force}
    res = preflight(a.path, print_mm=a.print_mm, nozzle_mm=a.nozzle,
                    check_animation=a.animation, check_fem=a.fem, material=a.material,
                    check_strength=a.strength, load_case=lc, do_fix=a.fix, work_dir=a.out)
    # renders are file paths; keep them, drop big arrays already excluded
    print(json.dumps(res, default=str, indent=2))


if __name__ == "__main__":
    try:
        _main(sys.argv[1:])
    except Exception:
        traceback.print_exc()
        sys.exit(1)
