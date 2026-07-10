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
  files {input, fixed_glb, fixed_stl, prep_stl, prep_3mf?, supports_3mf?,
         quad_obj?, gcode?, report_md, report_json},
  renders [png paths], render_captions {stem -> plain caption}, savings (slicer
  numbers or None), channels, cert, fix (fidelity/deviation certificate info or
  None), stages, notes, profile, mode, out_dir,
  review (ADDITIVE plain-words data block for the browser review document --
  every field the hosted UI needs, see _review_base for the full key list;
  always present, unknown values are explicit None).

Presets (`preset=`): behaviour bundles for a HOSTING context, resolved via
meshprep.profiles.get_preset. `preset="space"` is the hosted-demo bundle:
slicer savings / reinforce / retopo off, 20k-face analysis proxy, opt-in warp
FEM capped at 1500 elements, and a ~120 s SOFT time budget past which the
remaining OPTIONAL stages are skipped with an honest note (the fix, the
re-check and the verdict always run).
"""
from __future__ import annotations

import math
import os
import re
import tempfile
import time

DEFAULT_PRINT_MM = 60.0

# a part whose smallest bbox extent is under this is called "flat" (plates,
# coasters, lithophanes) -- drives the review's flat_object flag only.
FLAT_MIN_MM = 3.0

# -- everyday-object anchors for the review layer. These are COMPARISONS for
#    scale (plain-words orientation for a novice), never measurements.
_DEV_ANCHORS = (
    (0.001, "no measurable change"),
    (0.02, "a quarter of a human hair"),
    (0.08, "about the width of a human hair"),
    (0.10, "thinner than a sheet of paper"),
    (0.30, "a few sheets of paper"),
    (0.80, "about a credit card's thickness"),
    (2.00, "about a coin's thickness"),
)

_SIZE_ANCHORS = (
    (15.0, "about the size of a coin"),
    (30.0, "about the size of a wine cork"),
    (45.0, "about the size of a golf ball"),
    (70.0, "about the size of an egg"),
    (100.0, "about the size of a computer mouse"),
    (135.0, "about the height of a soda can"),
    (200.0, "about the size of a large coffee mug"),
    (320.0, "about the size of a shoebox"),
)


def _deviation_anchor(dev_mm):
    """Plain-words physical comparison for a max-deviation figure. None in ->
    None out."""
    if not isinstance(dev_mm, (int, float)) or dev_mm != dev_mm:
        return None
    for cap, label in _DEV_ANCHORS:
        if dev_mm <= cap:
            return label
    return "several millimetres -- clearly visible; inspect the renders"


def _size_anchor(longest_mm):
    """Plain-words everyday-object comparison for the longest dimension."""
    if not isinstance(longest_mm, (int, float)) or longest_mm != longest_mm:
        return None
    for cap, label in _SIZE_ANCHORS:
        if longest_mm <= cap:
            return label
    return "bigger than a shoebox"

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
    """Timed, never-raise stage runner; rows feed the report's provenance table.

    `budget_s` (optional) is a SOFT total-time budget: `over_budget()` flips
    True once elapsed wall time exceeds it. The pipeline consults it before
    each OPTIONAL stage only -- required stages (analyze, verdict, fix,
    re-check, package, report) always run."""

    def __init__(self, budget_s=None):
        self.rows: list[dict] = []
        self.budget_s = float(budget_s) if budget_s else None
        self.budget_hit = False
        self._t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def over_budget(self) -> bool:
        if self.budget_s is None:
            return False
        if self.elapsed() > self.budget_s:
            self.budget_hit = True
            return True
        return False

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
                   f"{DEFAULT_PRINT_MM:g} mm default GUESS. Set the print size "
                   f"(or assume_unit='m') to set the true size.")
    elif ext < 5.0:
        unit = "inch?"
        warning = (f"this part's longest side is only {ext:.3g} in file units -- "
                   f"that looks like INCHES or cm (too small for mm). A real "
                   f"print is ~5-300 mm, so it will be rescaled to a "
                   f"{DEFAULT_PRINT_MM:g} mm default GUESS. Set the print size "
                   f"(or assume_unit) to set the true size.")
    elif ext < 10.0:
        unit = "cm?"
        warning = (f"this part is only {ext:.3g} mm on its longest side -- "
                   f"unusually small; did you export in cm or inches? If so, "
                   f"pass assume_unit='cm' (or set the print size) to rescale. "
                   f"It is kept at {ext:.3g} mm for now, not silently changed.")
    elif ext > 300.0:
        unit = "mm"
        warning = (f"this part is {ext:.3g} mm on its longest side -- unusually "
                   f"large for a desktop printer; if you meant a smaller unit, "
                   f"set the print size (or assume_unit) to rescale. Kept as-is.")
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
#  seam weld + hole census  (never raise)
# ---------------------------------------------------------------------------
def _open_edge_count(mesh):
    """Number of boundary (open) edges, or -1 if it can't be measured."""
    try:
        import trimesh.grouping as tg
        return int(len(tg.group_rows(mesh.edges_sorted, require_count=1)))
    except Exception:
        return -1


def _weld_seams(mesh, notes):
    """Positional vertex weld for exporter seam-splits. GLB/OBJ exporters
    duplicate vertices wherever UV/normal attributes change (every hard edge);
    loaded raw, those duplicates read as open boundary edges, and a genuinely
    CLOSED solid gets misreported as 'not watertight' (plus phantom paper-thin
    'walls' between the coincident sheets). That was the false-FAIL a clean
    watertight upload used to hit. The weld merges vertices at identical
    POSITIONS only -- no vertex moves, no face is added or removed -- and the
    welded copy is adopted only if it strictly reduces open edges. Never
    raises; returns the (possibly welded) mesh."""
    try:
        if mesh is None or getattr(mesh, "is_watertight", False):
            return mesh
        b0 = _open_edge_count(mesh)
        if b0 <= 0:
            return mesh
        m = mesh.copy()
        try:
            m.merge_vertices(merge_tex=True, merge_norm=True)
        except TypeError:                    # older trimesh signature
            m.merge_vertices()
        m.remove_unreferenced_vertices()
        b1 = _open_edge_count(m)
        if 0 <= b1 < b0:
            notes.append(f"welded exporter seam-splits ({b0:,} -> {b1:,} open "
                         "edges; vertex positions untouched) -- attribute "
                         "seams, not real holes")
            return m
    except Exception:
        pass
    return mesh


def _boundary_loops(mesh):
    """Count open boundary loops (candidate holes). None when unmeasurable."""
    try:
        if mesh is None or getattr(mesh, "is_watertight", False):
            return 0
        out = mesh.outline()
        return 0 if out is None else int(len(out.entities))
    except Exception:
        return None


def _source_watertight_probe(src_path):
    """Reload the ORIGINAL file once (welded, no decimation) purely to ask:
    was the source a sealed solid? Used only when the decimated analysis proxy
    shows open edges -- it distinguishes 'your file is broken' from 'our
    simplification broke it', the difference between an honest FAIL and a
    false accusation. Returns True / False / None (couldn't tell)."""
    try:
        import trimesh
        m = trimesh.load(src_path, force="mesh")
        if isinstance(m, trimesh.Scene):
            geoms = [g for g in m.geometry.values() if hasattr(g, "faces")]
            if not geoms:
                return None
            m = trimesh.util.concatenate(geoms)
        if not hasattr(m, "faces") or len(m.faces) == 0:
            return None
        try:
            m.merge_vertices(merge_tex=True, merge_norm=True)
        except Exception:
            pass
        return bool(m.is_watertight)
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  fast post-fix re-check (space preset)
# ---------------------------------------------------------------------------
def _fast_recheck(fixed_m, *, print_mm, nozzle_mm, work_dir, before_checks):
    """Space-preset re-check: topology + printability are RE-MEASURED on the
    fixed mesh; the voxel resin-trap census (the slowest channel -- it used to
    double job time) is CARRIED FORWARD from the pre-fix analysis and labelled
    as such (the faithful fixer keeps cavities, it does not carve new ones).
    Same shape as preflight_core.run_checks; feeds build_cert unchanged."""
    from .core import preflight_core as pc
    m = pc._scaled_to_mm(fixed_m, print_mm)
    rep = {"facts": pc.topology_facts(m), "print_mm": print_mm,
           "nozzle_mm": nozzle_mm, "renders": {}, "channels": {}}
    try:
        from .core import _print_premortem as pm
        pr = pm.premortem(m, nozzle=nozzle_mm)
        comb = pr["summary"]["combined"]
        thin = pr["summary"].get("thin", {})
        rep["channels"]["printability"] = {
            "max_risk": comb.get("max_risk"),
            "high_risk_area_frac": comb.get("high_risk_area_frac"),
            "n_high_risk_faces": comb.get("n_high_risk_faces(>0.5)"),
            "dominant": comb.get("dominant_channel_of_risk_faces", {}),
            "min_wall_mm": thin.get("min_wall_mm"),
        }
        try:
            os.makedirs(work_dir, exist_ok=True)
            png = os.path.join(work_dir, "premortem.png")
            pm.render_premortem(m, png, build_dir=pr.get("build_dir"),
                                result=pr, nozzle=nozzle_mm,
                                title="After the fix -- printability risk")
            rep["renders"]["printability"] = png
        except Exception:
            pass
    except Exception as e:
        rep["channels"]["printability"] = {"error": f"{type(e).__name__}: {e}"}
    tr = (before_checks or {}).get("channels", {}).get("resin_traps")
    if isinstance(tr, dict) and "error" not in tr:
        rep["channels"]["resin_traps"] = dict(
            tr, carried_forward=True,
            note="carried from the pre-fix analysis (the faithful fix keeps "
                 "cavities; it does not carve new ones); not re-measured")
    return rep


# ---------------------------------------------------------------------------
#  the review layer -- plain-words data block for the hosted review document.
#  ADDITIVE: result['review'] never renames or replaces an existing key.
#  HONESTY: every physics/geometry number keeps its label (estimate /
#  comparative / uncalibrated); language may be simple, claims are never
#  upgraded; a refusal stays a refusal.
# ---------------------------------------------------------------------------
def _plainify(text):
    """Strip CLI jargon (backticks, flags) from a user-facing string. The
    browser review has no flags to drop and no terminal to type into."""
    t = str(text or "").replace("`", "")
    for a, b in (
        ("Run the full prep (drop the --check flag)", "Run the full prep"),
        ("(drop the --check flag)", ""),
        ("drop the --check flag", "use the full prep"),
        ("--print-mm", "the print-size setting"),
        ("--check", "check-only mode"),
        ("(prep applies it; check only reports it)", ""),
    ):
        t = t.replace(a, b)
    return " ".join(t.split())


def _support_zone_plain(mesh):
    """Plain-language location of the support zones at the CURRENT (+z build)
    orientation. Geometric (45 deg rule); never raises -- None on failure."""
    try:
        import numpy as np
        from .core import _print3d as p3
        sev, needs = p3._face_overhang(mesh, [0.0, 0.0, 1.0])
        needs = np.asarray(needs, bool)
        if not needs.any():
            return "no meaningful support zones in this orientation"
        areas = np.asarray(mesh.area_faces)[needs]
        total = float(mesh.area) + 1e-12
        if float(areas.sum()) / total < 0.02:
            return "no meaningful support zones in this orientation"
        cz = np.asarray(mesh.triangles_center)[needs][:, 2]
        zc = float((areas * cz).sum() / (areas.sum() + 1e-12))
        z0, z1 = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
        f = (zc - z0) / max(z1 - z0, 1e-9)
        if f < 0.34:
            return "mostly on the underside, near the build plate"
        if f < 0.67:
            return "on downward-facing surfaces around the middle of the part"
        return "under the upper overhangs, near the top of the part"
    except Exception:
        return None


def _review_base() -> dict:
    """Every review field, explicitly present (None until known). This is the
    contract with the hosted UI: keys are stable, values may be None."""
    d = {k: None for k in (
        "verdict", "surface_kept_pct", "max_deviation_mm", "deviation_anchor",
        "holes_filled_count", "top_issue_plain", "residual_issue_plain",
        "cavities_count", "material_savings_pct", "material_savings_note",
        "extents_x_mm", "extents_y_mm", "extents_z_mm", "extents_mm",
        "longest_dim_mm", "min_extent_mm", "size_anchor", "units_source",
        "user_target_size_mm", "render_before", "render_after",
        "render_support", "support_pct_original", "support_pct_oriented",
        "support_zone_plain", "thin_wall_min_mm", "printable_min_mm",
        "scale_factor_suggested", "components_count", "watertight_after",
        "normals_consistent", "output_filename", "output_format",
        "output_size_mb", "process_suggestion", "known_risks_shop_phrasing",
        "tri_count_original", "tri_count_analysis", "technical_report_url",
        "source_watertight")}
    d.update({"fix_ran": False, "blocking_count": 0, "warnings": [],
              "flat_object": False, "orientation_applied": False,
              "decimation_used": False,
              "warp": {"ran": False, "hotspot_plain": None,
                       "ratio_vs_rest": None, "ratio_note": None,
                       "suggested_shop_phrase": None}})
    return d


def _minimal_review(verdict_plain, top_plain) -> dict:
    """Review block for early exits (REJECTED / hard ERROR)."""
    d = _review_base()
    d["verdict"] = verdict_plain
    d["top_issue_plain"] = _plainify(top_plain) or None
    return d


_SEV_TO_REVIEW = {"FAIL": "blocking", "WARN": "check", "INFO": "info"}


def _review_warnings(final_cert, gate_issues, unit_warning, *, fix_ran,
                     support_pct_original, support_pct_oriented,
                     thin_wall_min_mm, orientation_applied,
                     source_watertight=None):
    """cert/gate/unit findings -> ranked plain-words warning rows. Every row
    has a NON-EMPTY action_plain. Known degenerate findings are rewritten
    (0.00 mm walls != 'scale it up'; 'run the Fix' is nonsense when the fix
    already ran; the ONLY supports figure shown is the oriented one)."""
    warns: list[dict] = []
    zero_thin = (isinstance(thin_wall_min_mm, (int, float))
                 and thin_wall_min_mm <= 0.005)

    for it in (final_cert or {}).get("issues", []) or []:
        if not isinstance(it, dict):
            continue
        sev = _SEV_TO_REVIEW.get(str(it.get("sev", "INFO")).upper(), "info")
        fact = _plainify(it.get("title"))
        consequence = _plainify(it.get("detail"))
        action = _plainify(it.get("fix"))
        part = None
        low = fact.lower()
        if low.startswith("thinnest wall") and zero_thin:
            # a 0.00 reading is shattered/sheet geometry, not a scaling problem
            fact = ("Zero-thickness geometry detected -- parts of this model "
                    "are infinitely thin sheets, not solid walls.")
            consequence = ("A printer cannot make material with no thickness; "
                           "those regions would simply be missing from the "
                           "print. Scaling the model up cannot help -- zero "
                           "stays zero at any size.")
            action = ("Give the sheet regions real thickness in a modelling "
                      "tool (or re-export the model as a solid).")
            part = "zero-thickness shell regions"
        elif "not watertight" in low and fix_ran:
            # the fix already ran in this very job -- never point back at it
            action = ("The automatic fix already ran and could not fully seal "
                      "this part; it needs manual repair in a mesh editor "
                      "before printing.")
            if source_watertight is True:
                consequence += (" Note: your ORIGINAL file is a sealed solid; "
                                "these open edges came from the simplified "
                                "analysis copy, not from your model.")
        elif "overhang" in low or "support-risk" in low:
            # ONE supports figure only: the oriented one (plus its provenance)
            if isinstance(support_pct_oriented, (int, float)):
                where = ("applied to the downloaded file" if orientation_applied
                         else "suggested in this report")
                fact = (f"About {support_pct_oriented:.0f}% of the surface "
                        f"needs support material in the print orientation "
                        f"{where}")
                if isinstance(support_pct_original, (int, float)):
                    fact += (f" (down from {support_pct_original:.0f}% as "
                             f"uploaded)")
                fact += "."
            action = ("Print it in the "
                      + ("applied" if orientation_applied else "suggested")
                      + " orientation; the support render shows where "
                        "supports will touch and leave small marks.")
        # never point the user back at a fix that ALREADY ran in this job
        if fix_ran and re.search(r"\bfree fix\b|\brun the (free )?fix\b"
                                 r"|\bfix \(below\)", action, re.I):
            action = ("The automatic fix already ran in this job and could "
                      "not resolve this; it needs manual repair in a mesh "
                      "editor.")
        if not action:
            action = "No action needed -- listed so you know it was checked."
        warns.append({"severity": sev, "fact_plain": fact,
                      "consequence_plain": consequence,
                      "action_plain": action, "part_name": part})

    seen = " | ".join(w["fact_plain"].lower() for w in warns)
    for gi in gate_issues or []:
        if not isinstance(gi, dict):
            continue
        code = str(gi.get("code", ""))
        if code == "not_watertight" and "watertight" in seen:
            continue                          # already covered by the cert row
        sev = {"fail": "blocking", "warn": "check"}.get(
            str(gi.get("severity", "")).lower(), "info")
        msg = _plainify(gi.get("message"))
        action = {
            "not_watertight": ("The automatic fix could not fully seal this "
                               "part; it needs manual repair in a mesh "
                               "editor." if fix_ran else
                               "Run the fix; if it cannot seal the part, "
                               "repair it in a mesh editor."),
            "inside_out": "Flip the face normals in a modelling tool.",
        }.get(code, "Fix this in a modelling tool, or re-export the model "
                    "from its source program.")
        warns.append({"severity": sev, "fact_plain": msg,
                      "consequence_plain": "It can make the print unreliable "
                                           "or wrong.",
                      "action_plain": action, "part_name": None})

    if unit_warning:
        warns.append({
            "severity": "check",
            "fact_plain": _plainify(unit_warning),
            "consequence_plain": "The part could print at the wrong physical "
                                 "size.",
            "action_plain": "Check the printed size shown in this report and "
                            "set the print size explicitly if it is wrong.",
            "part_name": None})

    order = {"blocking": 0, "check": 1, "info": 2}
    warns.sort(key=lambda w: order.get(w["severity"], 2))
    return warns


def _shop_risks(warns, review):
    """Shop-vocabulary one-liner for the print-shop block. None when clean."""
    bits: list[str] = []
    for w in warns:
        if w["severity"] not in ("blocking", "check"):
            continue
        f = w["fact_plain"].lower()
        if "zero-thickness" in f:
            bits.append("zero-thickness shell regions (needs CAD repair)")
        elif "thinnest wall" in f or "thin wall" in f:
            pmin = review.get("printable_min_mm")
            bits.append(f"walls under the {pmin or 0.4} mm single-extrusion "
                        f"width")
        elif "watertight" in f or "open holes" in f or "sealed" in f:
            bits.append("shell not fully sealed")
        elif "support" in f or "overhang" in f:
            sp = review.get("support_pct_oriented")
            bits.append("supports on ~{:.0f}% of the surface".format(sp)
                        if isinstance(sp, (int, float)) else
                        "supports required on overhangs")
        elif "cavity" in f or "traps" in f:
            bits.append("enclosed internal cavity kept (vent it before resin "
                        "printing)")
        elif "warp" in f or "curl" in f:
            bits.append("corner-lift risk on the first layers; brim advised")
    if not bits:
        return None
    dedup: list[str] = []
    for b in bits:
        if b not in dedup:
            dedup.append(b)
    return "; ".join(dedup)


def _build_review(result, *, work_mesh, checks_before, checks_after,
                  final_cert, nozzle_mm, prof, material, mode, units_source,
                  user_target_mm, src_path, tri_original, tri_analysis,
                  decimation_used, holes_before, holes_after,
                  orientation_applied, support_render, out_dir) -> dict:
    """Assemble the review data block. Never raises; unknown values stay None."""
    d = _review_base()
    try:
        d["verdict"] = {"PASS": "ready", "WARN": "ready", "FAIL": "not_ready",
                        "REJECTED": "unreadable", "ERROR": "unreadable"
                        }.get(result.get("verdict"), "not_ready")

        # -- the trust line: surface kept + max deviation -------------------
        fx = result.get("fix") if isinstance(result.get("fix"), dict) else None
        d["fix_ran"] = bool(fx)
        cert = (fx or {}).get("deviation_certificate") or {}
        if fx:
            if cert.get("rolled_back"):
                d["surface_kept_pct"] = 100.0     # nothing shipped = untouched
                d["max_deviation_mm"] = 0.0
            else:
                skp = cert.get("surface_unchanged_pct")
                if isinstance(skp, (int, float)) and skp == skp:
                    d["surface_kept_pct"] = round(float(skp), 1)
                dev = cert.get("max_deviation_mm")
                if (isinstance(dev, (int, float)) and dev == dev
                        and dev != float("inf")):
                    d["max_deviation_mm"] = round(float(dev), 3)
        d["deviation_anchor"] = _deviation_anchor(d["max_deviation_mm"])
        if isinstance(holes_before, int) and isinstance(holes_after, int):
            d["holes_filled_count"] = max(0, holes_before - holes_after)

        # -- size / units ----------------------------------------------------
        ext = None
        try:
            ext = [float(x) for x in (work_mesh.bounds[1] - work_mesh.bounds[0])]
        except Exception:
            bb = ((checks_after or checks_before or {}).get("facts")
                  or {}).get("bbox_mm")
            if bb and len(bb) == 3:
                ext = [float(x) for x in bb]
        if ext and all(e == e for e in ext):
            d["extents_x_mm"], d["extents_y_mm"], d["extents_z_mm"] = (
                round(ext[0], 2), round(ext[1], 2), round(ext[2], 2))
            d["extents_mm"] = [round(e, 2) for e in ext]
            d["longest_dim_mm"] = round(max(ext), 2)
            d["min_extent_mm"] = round(min(ext), 3)
            d["flat_object"] = bool(min(ext) < FLAT_MIN_MM)
            d["size_anchor"] = _size_anchor(max(ext))
        d["units_source"] = units_source
        d["user_target_size_mm"] = (float(user_target_mm)
                                    if isinstance(user_target_mm, (int, float))
                                    else None)

        # -- supports / orientation -----------------------------------------
        oc = (result.get("channels") or {}).get("orientation") or {}
        spo = oc.get("support_area_frac_as_loaded")
        spn = oc.get("support_area_frac")
        if isinstance(spo, (int, float)):
            d["support_pct_original"] = round(float(spo) * 100.0, 1)
        if isinstance(spn, (int, float)):
            d["support_pct_oriented"] = round(float(spn) * 100.0, 1)
        d["orientation_applied"] = bool(orientation_applied)
        if work_mesh is not None:
            d["support_zone_plain"] = _support_zone_plain(work_mesh)

        # -- thin walls -------------------------------------------------------
        pr_after = ((checks_after or {}).get("channels") or {}).get(
            "printability") or {}
        pr_before = ((checks_before or {}).get("channels") or {}).get(
            "printability") or {}
        mw = pr_after.get("min_wall_mm")
        if not isinstance(mw, (int, float)):
            mw = pr_before.get("min_wall_mm")
        if isinstance(mw, (int, float)) and mw == mw:
            d["thin_wall_min_mm"] = round(float(mw), 3)
        d["printable_min_mm"] = float(nozzle_mm)
        tw = d["thin_wall_min_mm"]
        # MUST stay None for zero-thickness geometry: scaling can't fix zero
        if (isinstance(tw, (int, float)) and tw > 0.005
                and tw < d["printable_min_mm"]):
            d["scale_factor_suggested"] = math.ceil(
                (d["printable_min_mm"] / tw) * 10.0) / 10.0

        # -- solidity for the shop block --------------------------------------
        gate = result.get("printability_gate") or {}
        gchecks = gate.get("checks") or {}
        nc = gchecks.get("n_components")
        if not isinstance(nc, int):
            try:
                nc = int(work_mesh.body_count)
            except Exception:
                nc = None
        d["components_count"] = nc
        wt = gchecks.get("watertight")
        if wt is None:
            wt = ((checks_after or checks_before or {}).get("facts")
                  or {}).get("watertight")
        d["watertight_after"] = bool(wt) if wt is not None else None
        wind = ((checks_after or checks_before or {}).get("facts")
                or {}).get("winding_consistent")
        d["normals_consistent"] = bool(wind) if wind is not None else None
        d["source_watertight"] = result.get("source_watertight")

        # -- cavities kept + material estimate --------------------------------
        ish = result.get("internal_shells")
        if isinstance(ish, dict):
            d["cavities_count"] = int(ish.get("n_internal_shells") or 0)
            ratio = ish.get("solid_volume_ratio")
            if isinstance(ratio, (int, float)) and 0.0 <= ratio <= 1.0:
                d["material_savings_pct"] = round((1.0 - float(ratio)) * 100.0, 1)
                d["material_savings_note"] = (
                    "geometric ESTIMATE: upper bound vs a cavity-filling "
                    "repair at 100% infill; real savings at normal infill "
                    "are smaller")
        else:
            tr = (result.get("channels") or {}).get("resin_traps") or {}
            ncav = tr.get("n_internal_cavities")
            if isinstance(ncav, int):
                d["cavities_count"] = ncav

        # -- renders -----------------------------------------------------------
        rb = ((checks_before or {}).get("renders") or {}).get("printability")
        ra = ((checks_after or {}).get("renders") or {}).get("printability")
        d["render_before"] = rb
        d["render_after"] = ra or rb
        # the support slot carries ONLY a true support render: captioning the
        # after-render as "support zones" would mislabel the picture, so the
        # slot stays empty when the support render was not produced.
        d["render_support"] = support_render

        # -- warnings (ranked; every row has an action) ------------------------
        warns = _review_warnings(
            final_cert, result.get("gate_issues"), result.get("unit_warning"),
            fix_ran=d["fix_ran"],
            support_pct_original=d["support_pct_original"],
            support_pct_oriented=d["support_pct_oriented"],
            thin_wall_min_mm=d["thin_wall_min_mm"],
            orientation_applied=d["orientation_applied"],
            source_watertight=d["source_watertight"])

        # warp advisory (opt-in; ESTIMATE, uncalibrated unless a coupon fitted)
        wch = (result.get("channels") or {}).get("warp") or {}
        lift = wch.get("max_corner_lift_mm")
        if isinstance(lift, (int, float)) and lift == lift:
            from .core import preflight_core as pc
            cal = bool(wch.get("calibrated"))
            d["warp"] = {
                "ran": True,
                "hotspot_plain": "the outer corners of the first layers, "
                                 "where the part meets the bed",
                # COMPARATIVE tendency, not a certified mm: predicted corner
                # lift relative to the attention threshold (1.0 = at the
                # level where we'd flag it). Uncalibrated unless stated.
                "ratio_vs_rest": round(float(lift) / pc.WARP_WARN_MM, 2),
                "ratio_note": "warp tendency vs the attention threshold "
                              "(comparative"
                              + ("" if cal else ", uncalibrated") + ")",
                "suggested_shop_phrase": (
                    "Please print this with a brim (or raft) -- the base "
                    "corners tend to lift." if lift >= pc.WARP_WARN_MM else
                    "Standard first-layer settings should be fine; a brim "
                    "is cheap insurance."),
            }
            if lift >= pc.WARP_WARN_MM:
                warns.append({
                    "severity": "check",
                    "fact_plain": "The base of this part tends to curl up as "
                                  "it cools (estimate"
                                  + ("" if cal else ", uncalibrated") + ").",
                    "consequence_plain": "Corners can lift off the build "
                                         "plate mid-print.",
                    "action_plain": "Ask for a brim or raft under the first "
                                    "layer.",
                    "part_name": "the base corners"})

        d["warnings"] = warns
        d["blocking_count"] = sum(1 for w in warns
                                  if w["severity"] == "blocking")
        d["top_issue_plain"] = (warns[0]["fact_plain"] if warns else
                                "No issues found -- this part looks ready "
                                "to print.")
        if d["fix_ran"] and d["verdict"] == "not_ready":
            # the BARE fact only: the review page supplies its own framing
            # ("the repair did NOT fix ...") — a baked-in prefix here produced
            # broken double-prefixed English on the rendered page.
            blk = next((w for w in warns if w["severity"] == "blocking"), None)
            d["residual_issue_plain"] = (
                blk["fact_plain"] if blk
                else "a problem that needs your decision remains.")

        # -- output / shop block ------------------------------------------------
        files = result.get("files") or {}
        stem = "model"
        if src_path:
            raw = os.path.splitext(os.path.basename(str(src_path)))[0]
            clean = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
            if clean:
                stem = clean
        d["output_filename"] = f"{stem}_print_ready.stl"
        d["output_format"] = "binary STL"
        pstl = files.get("prep_stl")
        try:
            if pstl and os.path.isfile(pstl):
                d["output_size_mb"] = round(os.path.getsize(pstl) / 1e6, 2)
        except Exception:
            pass
        layer = (prof or {}).get("layer_mm", 0.2)
        disp = (prof or {}).get("display") or (prof or {}).get("name") or ""
        if mode == "resin":
            d["process_suggestion"] = (f"MSLA resin, {material}, "
                                       f"{layer:g} mm layers ({disp})")
        else:
            d["process_suggestion"] = (f"FDM, {material}, {nozzle_mm:g} mm "
                                       f"nozzle, {layer:g} mm layers ({disp})")
        d["known_risks_shop_phrasing"] = _shop_risks(warns, d)

        # -- fine-print disclosure ------------------------------------------------
        d["decimation_used"] = bool(decimation_used)
        d["tri_count_original"] = (int(tri_original)
                                   if isinstance(tri_original, int)
                                   else (int(tri_analysis)
                                         if isinstance(tri_analysis, int)
                                         else None))
        d["tri_count_analysis"] = (int(tri_analysis)
                                   if isinstance(tri_analysis, int) else None)
        # never a server-local path: the review renders this as a dead link in
        # a browser and leaks the temp dir into the shop-facing review.md.
        # The bare filename names the artifact that rides in the downloads.
        d["technical_report_url"] = "report.md" if out_dir else None
    except Exception as e:                        # never-raise discipline
        d["review_error"] = f"{type(e).__name__}: {e}"
    return d


# ---------------------------------------------------------------------------
#  the one call
# ---------------------------------------------------------------------------
def prep(path_or_mesh, *, profile="generic_fdm", material=None, mode=None,
         preset=None,
         fix=True, orient=True, reinforce_load=None, reinforce_force_n=100.0,
         fem_warp=False, fem_strength=False, retopo=False, supports=False,
         out_dir=None, print_mm=None, nozzle_mm=None, max_faces=None,
         assume_unit=None, check_only=False, slicer_savings=True,
         verbose=False) -> dict:
    """Drop a mesh -> print-ready package + plain-English report. Never raises.

    path_or_mesh     : mesh file path (GLB/OBJ/STL/PLY/OFF/3MF) or a trimesh.
    profile          : printer preset name / dict (meshprep.profiles).
    preset           : hosting-context behaviour bundle (profiles.PREP_PRESETS).
                       "space" = the hosted-demo bundle: slicer savings /
                       reinforce / retopo off, 20k-face analysis proxy, warp
                       FEM (still opt-in via fem_warp) capped at 1500
                       elements, ~120 s soft time budget (optional stages are
                       skipped with an honest note past it; the fix, the
                       re-check and the verdict always run).
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
    supports         : opt-in risk-driven SupportEnforcer/SupportBlocker 3MF
                       (core.support_mods), built from the same premortem risk
                       field, at the orientation that ships -> supports.3mf.
                       Heuristic triage, not a guarantee any support was
                       required; see the 'supports' channel note. Never blocks
                       the verdict.
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
                     mode=mode, preset=preset, fix=fix, orient=orient,
                     reinforce_load=reinforce_load,
                     reinforce_force_n=reinforce_force_n, fem_warp=fem_warp,
                     fem_strength=fem_strength, retopo=retopo,
                     supports=supports, out_dir=out_dir,
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
                "review": _minimal_review("unreadable", plain),
                "files": {}, "renders": [], "stages": [], "notes": [],
                "report": {"markdown": f"# meshprep report -- ERROR\n\n{plain}\n\n"
                                       f"_details: {msg}_\n",
                           "json": {"verdict": "ERROR", "ok": False,
                                    "headline": plain, "error": msg},
                           "md_path": None, "json_path": None}}


def _prep(path_or_mesh, *, profile, material, mode, preset, fix, orient,
          reinforce_load, reinforce_force_n, fem_warp, fem_strength, retopo,
          supports=False,
          out_dir, print_mm, nozzle_mm, max_faces, assume_unit, check_only,
          slicer_savings, verbose):
    import numpy as np

    from . import report as report_mod
    from .core import preflight_core as pc
    from .core._mesh_util import say
    from .profiles import get_preset, get_profile, scale_to_fit

    notes: list[str] = []

    # -- resolve preset (hosting-context bundle; may trim heavy stages) ------
    ps = get_preset(preset)
    if ps.get("fallback_note"):
        notes.append(ps["fallback_note"])
        ps = {}
    if ps:
        notes.append(f"preset '{ps.get('name')}' active")
        if ps.get("slicer_savings") is False and slicer_savings:
            slicer_savings = False
            notes.append("slicer savings are not available under this preset "
                         "(no slicer installed on the host)")
        if ps.get("reinforce") is False and reinforce_load is not None:
            reinforce_load = None
            notes.append("reinforce (graded infill) is not part of this "
                         "preset -- skipped")
        if ps.get("retopo") is False and retopo:
            retopo = False
            notes.append("retopo is not part of this preset -- skipped")
        if ps.get("fem_strength") is False and fem_strength:
            fem_strength = False
            notes.append("orientation-strength FEM is not part of this "
                         "preset -- skipped")
        if max_faces is None and ps.get("max_faces"):
            max_faces = int(ps["max_faces"])
    budget_s = ps.get("stage_budget_s")
    warp_max_elem = ps.get("fem_warp_max_elem")   # set -> warp runs as its own
    #                                              capped, budget-gated stage
    hide_paths = bool(ps.get("hide_local_paths"))
    use_fast_recheck = bool(ps.get("fast_recheck"))

    stages = _Stages(budget_s=budget_s)
    _BUDGET_WHY = (f"soft time budget (~{budget_s:.0f} s) reached -- optional "
                   "stage skipped; the fix and the verdict are unaffected"
                   if budget_s else "")

    def _over_budget(stage_name, label=""):
        """True (and an honest skip row) when the soft budget is spent."""
        if stages.over_budget():
            stages.skip(stage_name, _BUDGET_WHY, label)
            return True
        return False

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
        fix = orient = retopo = slicer_savings = supports = False
        reinforce_load = None

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="meshprep_")
        # the raw path stays in result["out_dir"] for programmatic use; the
        # NOTE (which reaches the user-facing report) never embeds a
        # server-local temp path (meaningless + path-leaking when hosted).
        notes.append("no out_dir given -> results in a temporary working "
                     "folder")
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
        _where = "" if hide_paths else f" ({out_dir})"
        _head = (f"Can't save results into the output folder{_where} -- "
                 f"{type(_we).__name__}. Pick a writable folder "
                 "(out_dir=...) and run again.")
        return {"ok": False, "verdict": "ERROR",
                "headline": _head,
                "review": _minimal_review("unreadable", _head),
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
    tri_original = None                 # facet count before any decimation
    decimation_used = False
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
            for ln in lnotes:           # fine-print disclosure numbers
                mm_ = re.search(r"[Dd]ecimated ([\d,]+)\s*->\s*([\d,]+)", ln)
                if mm_:
                    tri_original = int(mm_.group(1).replace(",", ""))
                    decimation_used = True
        except pc.GuardReject as g:
            result.update({"verdict": "REJECTED", "rejected": str(g),
                           "headline": f"Upload refused -- {g}",
                           "review": _minimal_review("unreadable",
                                                     f"Upload refused -- {g}")})
            md, js = report_mod.build_report(result)
            js["review"] = result["review"]
            result["report"] = report_mod.attach(result, md, js, out_dir)
            result["report_json"] = js
            return result
    else:
        mesh = path_or_mesh
        if mesh is None or not hasattr(mesh, "faces") or len(mesh.faces) == 0:
            result.update({"verdict": "REJECTED",
                           "rejected": "not a triangle mesh",
                           "headline": "Input is not a triangle mesh.",
                           "review": _minimal_review(
                               "unreadable", "Input is not a triangle mesh.")})
            md, js = report_mod.build_report(result)
            js["review"] = result["review"]
            result["report"] = report_mod.attach(result, md, js, out_dir)
            result["report_json"] = js
            return result
        mesh = mesh.copy()
        if len(mesh.faces) > cap:
            from .core._mesh_util import decimate
            n_in = len(mesh.faces)
            tri_original = int(n_in)
            decimation_used = True
            mesh = decimate(mesh, cap)
            notes.append(f"decimated {n_in:,} -> {len(mesh.faces):,} faces "
                         "(analysis runs on the simplified mesh)")
        stages.rows.append({"stage": "load+guard", "ok": True, "seconds": None,
                            "label": "in-memory mesh", "note": ""})

    # -- 1b. seam weld (false-FAIL guard) -------------------------------------
    # GLB/OBJ exporters split vertices at UV/normal seams; loaded raw those
    # splits read as open edges and a genuinely watertight upload FAILed as
    # 'not watertight' with phantom hair-thin walls. Weld coincident vertices
    # (positions untouched) before ANY analysis.
    mesh = _weld_seams(mesh, notes)
    tri_analysis = int(len(mesh.faces))
    if tri_original is None:
        tri_original = tri_analysis

    # -- 1c. if the analysis PROXY has open edges, ask the ORIGINAL file
    #    whether it was sealed: never accuse the user's model of holes that
    #    our own processing (decimation, seam weld, loader concatenation)
    #    introduced. Runs only when the proxy is open, so the extra load is
    #    paid exactly when the distinction matters.
    source_watertight = None
    if src_path and not getattr(mesh, "is_watertight", False):
        source_watertight = stages.run(
            "source-probe", lambda: _source_watertight_probe(src_path),
            label="was the ORIGINAL file sealed? (analysis proxy has open "
                  "edges)")
        if source_watertight is True:
            notes.append("the ORIGINAL file is a sealed solid; any open "
                         "edges reported below were introduced by the "
                         "analysis processing, not by your model")
    result["source_watertight"] = source_watertight

    # -- 2. units / print size (UNIT-INGEST CONTRACT; never silently rescale) --
    unit_scan = scan_units(mesh)
    result["unit_scan"] = unit_scan
    ext = float(unit_scan.get("max_extent") or 0.0)
    au = str(assume_unit).lower() if assume_unit else None
    units_source = "assumed_mm"                # review-layer provenance enum
    user_target_mm = None
    if print_mm:
        eff = float(print_mm)
        units_source, user_target_mm = "user_target", eff
        notes.append(f"scaled so longest side = {eff:g} mm (explicit print_mm)")
        unit_scan["warning"] = None            # user gave the size outright
    elif au in _UNIT_FACTOR_MM:
        # explicit unit override -> convert honestly (the RIGHT fix for a
        # mis-scaled export; not a guess). Clears the unit-sanity nudge.
        eff = ext * _UNIT_FACTOR_MM[au] if ext > 0 else DEFAULT_PRINT_MM
        units_source = "file"                  # the user told us the file unit
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
    result["units_source"] = units_source
    result["user_target_size_mm"] = user_target_mm

    # -- 3. analyze -----------------------------------------------------------
    lc = _load_case(reinforce_load, reinforce_force_n)
    strength_lc = lc or (_load_case("z", reinforce_force_n) if fem_strength else None)
    # under a preset with fem_warp_max_elem, the (opt-in) warp FEM runs later
    # as its own budget-gated stage with the element cap -- not inside analyze
    warp_in_analyze = bool(fem_warp) and not warp_max_elem
    checks = stages.run(
        "analyze",
        lambda: pc.run_checks(mesh_mm, print_mm=eff, nozzle_mm=nozzle,
                              work_dir=out_dir, check_fem=warp_in_analyze,
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
    if mode == "resin" and not _over_budget("resin"):
        from . import resin as resin_mod
        rr = stages.run("resin",
                        lambda: resin_mod.resin_report(mesh_mm, profile=prof,
                                                       verbose=verbose),
                        label="voxel triage (res<=96), not slicer-grade")
        if rr:
            result["channels"]["resin"] = rr

    # -- 4. fix ---------------------------------------------------------------
    # NEVER budget-gated: the fix, its re-check and the verdict always run.
    work_mesh = mesh_mm
    after_cert = None
    after_checks = None
    holes_before = _boundary_loops(mesh_mm)
    holes_after = holes_before
    if fix:
        r = stages.run("fix",
                       lambda: pc.run_fixer(mesh_mm, work_dir=out_dir,
                                            print_mm=eff),
                       label="source-accurate repair + deviation certificate")
        if r is not None:
            glb, stl, fnote, fixed_m, dev_cert = r
            work_mesh = fixed_m
            holes_after = _boundary_loops(fixed_m)
            result["files"]["fixed_glb"] = glb
            result["files"]["fixed_stl"] = stl
            if use_fast_recheck:
                after_checks = stages.run(
                    "re-check",
                    lambda: _fast_recheck(
                        fixed_m, print_mm=eff, nozzle_mm=nozzle,
                        work_dir=os.path.join(out_dir, "after"),
                        before_checks=checks),
                    label="verdict flip (before -> after); topology + "
                          "printability re-measured, trap census carried "
                          "forward (labelled)")
            else:
                after_checks = stages.run(
                    "re-check",
                    lambda: pc.run_checks(fixed_m, print_mm=eff,
                                          nozzle_mm=nozzle,
                                          work_dir=os.path.join(out_dir,
                                                                "after")),
                    label="verdict flip (before -> after)")
            after_cert = pc.build_cert(after_checks) if after_checks else None
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
    ish = None
    if not _over_budget("internal-shells"):
        ish = stages.run("internal-shells", _shells,
                         label="geometric cavity census (component split + "
                               "bbox nesting); NOT a slicer measurement")
    if isinstance(ish, dict):
        result["internal_shells"] = ish
        if isinstance(result.get("fix"), dict):
            result["fix"]["internal_shells"] = ish

    # -- 5. orient ------------------------------------------------------------
    orientation_applied = False
    orient_budget_skipped = False
    if orient and _over_budget("orient",
                               "support-minimising build direction"):
        orient = False
        orient_budget_skipped = True
        notes.append("orientation was skipped by the time budget -- the "
                     "downloaded file keeps the as-fixed orientation")
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
            orientation_applied = not check_only
            result["channels"]["orientation"] = {
                "up": [round(float(x), 3) for x in ori["up"]],
                "support_area_frac": round(float(ori["support_frac"]), 4),
                "support_area_frac_as_loaded": (None if base_frac is None
                                                else round(base_frac, 4)),
                "height_mm": round(float(ori["height"]), 2),
                "method": "geometric overhang minimisation (45 deg rule)"}
    elif not orient_budget_skipped:
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

    # -- 5b. support render (preset-only; the review's third image) -----------
    # COST-AWARE gate: this stage costs about as much as the re-check (both
    # are premortem-dominated), so skip it when starting it would OVERRUN the
    # budget, not merely when the budget is already spent.
    support_render = None
    _est_render_s = 0.0
    if budget_s:
        for _row in stages.rows:
            if _row["stage"] in ("re-check", "analyze") and _row.get("seconds"):
                _est_render_s = max(_est_render_s, float(_row["seconds"]))
    _render_would_overrun = bool(
        budget_s and stages.elapsed() + _est_render_s > budget_s)
    if ps and orientation_applied and _render_would_overrun:
        stages.skip("support-render",
                    f"would overrun the soft time budget (~{budget_s:.0f} s); "
                    "optional render skipped -- the support NUMBERS above are "
                    "unaffected")
    elif ps and orientation_applied and not _over_budget(
            "support-render", "support zones at the applied orientation"):
        def _support_png():
            from .core import _print_premortem as pm
            pr = pm.premortem(work_mesh, nozzle=nozzle)
            png = os.path.join(out_dir, "support.png")
            pm.render_premortem(work_mesh, png, build_dir=pr.get("build_dir"),
                                result=pr, nozzle=nozzle,
                                title="Support zones (applied orientation)")
            return png
        support_render = stages.run(
            "support-render", _support_png,
            label="where supports will touch, at the shipped orientation")
        if support_render and not os.path.exists(support_render):
            support_render = None

    # -- 5c. warp FEM under a preset element cap (opt-in; ESTIMATE) -----------
    # Runs AFTER orient so the prediction is at the orientation that ships.
    # Never upgrades the verdict machinery -- it only ADDS a labelled channel.
    if fem_warp and warp_max_elem and not check_only and not _over_budget(
            "warp", "opt-in inherent-strain FEM"):
        def _warp_capped():
            m = pc._scaled_to_mm(work_mesh, eff)
            try:
                from .core import _warp_calibrate as wc
                wr = wc.calibrated_warp(m, material=material, build_dir=None,
                                        max_elem=int(warp_max_elem))
            except Exception:
                from .core import _print_fem as pf
                wr = pf.warp_analysis(m, material=material, build_dir=None,
                                      max_elem=int(warp_max_elem))
            return wr
        wr = stages.run(
            "warp", _warp_capped,
            label=f"inherent-strain warp FEM, max_elem={int(warp_max_elem)} "
                  "(preset cap); uncalibrated unless a coupon was fitted")
        # wr is a PrintFEMResult (dict-like .get), not a dict -- duck-type it
        if wr is not None and wr.get("ok"):
            result["channels"]["warp"] = {
                "max_corner_lift_mm": wr.get("max_corner_lift_mm"),
                "sign": wr.get("sign"),
                "material": wr.get("material"),
                "calibrated": bool(wr.get("calibrated", False)),
                "cal_scale": wr.get("cal_scale"),
                "cal_label": wr.get("cal_label"),
                "uncalibrated": wr.get("uncalibrated", True),
                "magnitude_kind": wr.get("magnitude_kind"),
                "warning": wr.get("warning"),
                "max_elem": int(warp_max_elem),
                "at_orientation": "as shipped (post-orient)"}
            try:
                from .core import _print_fem as pf
                wpng = os.path.join(out_dir, "warp.png")
                pf.render_warp(wr, wpng,
                               title="Predicted print warp "
                                     "(inherent-strain FEM, estimate)")
                if os.path.exists(wpng):
                    checks.setdefault("renders", {})["warp"] = wpng
            except Exception:
                pass
        elif wr is not None:
            result["channels"]["warp"] = {
                "note": wr.get("reason", "warp analysis unavailable")}

    # -- 5d. support mods (opt-in; risk-driven SupportEnforcer/SupportBlocker
    #        3MF). Runs on work_mesh AFTER orient (core.support_mods fixes its
    #        own build_dir at the mesh's own +Z -- it assumes an already-
    #        oriented part, same convention as warp/reinforce above). Additive
    #        channel + file only; never touches the verdict.
    if supports and not check_only and not _over_budget(
            "supports", "risk-driven support enforcer/blocker 3MF"):
        from .core.support_mods import support_mods as _support_mods
        out_supports_3mf = os.path.join(out_dir, "supports.3mf")
        sres = stages.run(
            "supports",
            lambda: _support_mods(work_mesh, out_3mf=out_supports_3mf),
            label="risk-driven SupportEnforcer/SupportBlocker 3MF "
                  "(PrusaSlicer schema); heuristic triage, uncalibrated")
        if isinstance(sres, dict):
            result["channels"]["supports"] = {
                "ok": sres.get("ok"),
                "out_3mf": sres.get("out_3mf"),
                "n_enforcers": sres.get("n_enforcers"),
                "n_blockers": sres.get("n_blockers"),
                "risk_top": sres.get("risk_top"),
                "uncalibrated": sres.get("uncalibrated", True),
                "note": sres.get("note")}
            if sres.get("ok") and sres.get("out_3mf"):
                result["files"]["supports_3mf"] = sres["out_3mf"]
                n_e = int(sres.get("n_enforcers") or 0)
                n_b = int(sres.get("n_blockers") or 0)
                notes.append(
                    f"supports enforced on {n_e} risk region"
                    f"{'s' if n_e != 1 else ''} / blocked on {n_b} region"
                    f"{'s' if n_b != 1 else ''} -- open supports.3mf and set "
                    "support_material_auto=0 in the slicer for the enforcer "
                    "regions to take effect (support_material_auto=1 for the "
                    "blocker regions); risk is a heuristic triage field, not "
                    "a guarantee any support was actually required.")
            elif sres.get("ok"):
                notes.append("supports: " + (sres.get("note")
                             or "no risk region cleared the enforce/block "
                                "thresholds -- no supports.3mf written."))
            else:
                notes.append("supports: " + (sres.get("reason")
                             or sres.get("note") or "support analysis failed"))

    # -- 6. strengthen ----------------------------------------------------------
    if lc is not None and not check_only and not _over_budget(
            "reinforce", "graded-infill 3MF"):
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
    fit = None
    if not _over_budget("bed-fit"):
        fit = stages.run("bed-fit", lambda: scale_to_fit(work_mesh, prof),
                         label="AABB 6-permutation triage, not packing")
    if fit:
        result["channels"]["fit"] = fit
        if fit.get("fits") is False:
            notes.append("part exceeds the bed -- scale by "
                         f"{fit.get('suggested_scale')} or split it "
                         "(meshprep.split.split_for_bed)")

    # -- 8. retopo (opt-in) --------------------------------------------------------
    if retopo and not check_only and not _over_budget("retopo"):
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
    if (slicer_savings and prep_stl and not check_only
            and not _over_budget("slicer")):
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

    # -- cavity cross-check (false-FAIL guard) --------------------------------
    # The coarse voxel trap screen can flag the interior of a bumpy SOLID as a
    # 'sealed cavity'. Geometrically, a cavity inside a watertight mesh
    # REQUIRES a second closed shell -- so when the exact census (component
    # split + nesting) finds a single sealed body and zero internal shells,
    # the voxel claim is a resolution artifact, demoted to a note. This is a
    # false-measurement CORRECTION by a strictly more reliable exact check,
    # not a claim upgrade; it never touches any other issue.
    try:
        _ish = result.get("internal_shells")
        _facts_now = ((after_checks or checks or {}).get("facts") or {})
        _ncomp = None
        try:
            _ncomp = int(work_mesh.body_count)
        except Exception:
            pass
        if (isinstance(_ish, dict) and _ish.get("n_internal_shells") == 0
                and _facts_now.get("watertight") and _ncomp == 1):
            _iss = [i for i in (final_cert.get("issues") or [])
                    if isinstance(i, dict)]
            _cav = [i for i in _iss
                    if str(i.get("title", "")).startswith("Sealed cavity")]
            if _cav:
                _keep = [i for i in _iss if i not in _cav]
                _keep.append({
                    "sev": "INFO",
                    "title": "Voxel cavity screen overruled by the exact "
                             "census",
                    "detail": "The coarse voxel screen flagged a possible "
                              "sealed cavity, but the exact shell census "
                              "found a single sealed body with no internal "
                              "shell -- a cavity there is geometrically "
                              "impossible, so this was a voxel-resolution "
                              "artifact.",
                    "fix": "Nothing to do."})
                _sevs = [i["sev"] for i in _keep]
                _v = ("FAIL" if "FAIL" in _sevs
                      else "WARN" if "WARN" in _sevs else "PASS")
                _nf, _nw = _sevs.count("FAIL"), _sevs.count("WARN")
                if _v == "PASS":
                    _h = "Ready to print. No blocking issues found."
                elif _v == "WARN":
                    _h = (f"Printable, with {_nw} thing(s) to be aware of "
                          "(supports / orientation / minor topology).")
                else:
                    _first = next(i for i in _keep if i["sev"] == "FAIL")
                    _h = (f"Won't print as-is -- {_first['title'].lower()}. "
                          f"{_nf} blocking issue(s); the free Fix resolves "
                          "most.")
                _order = {"FAIL": 0, "WARN": 1, "INFO": 2}
                _keep.sort(key=lambda i: _order.get(i["sev"], 2))
                final_cert = dict(final_cert, issues=_keep, verdict=_v,
                                  headline=_h)
                # keep the fix's before->after line coherent with the demotion
                if isinstance(result.get("fix"), dict) and after_cert is not None:
                    result["fix"]["after_verdict"] = _v
                    result["fix"]["after_headline"] = _h
                if after_cert is None:
                    result["cert"] = final_cert
                notes.append("voxel trap screen flagged a sealed cavity, but "
                             "the exact shell census found none (single "
                             "sealed body) -- demoted as a voxel artifact")
    except Exception:
        pass

    # additive: the POST-FIX ranked cert the review/report should render
    # (result['cert'] stays the pre-fix cert, unrenamed)
    result["final_cert"] = final_cert

    result["ok"] = True
    base_verdict = final_cert.get("verdict", "WARN")
    base_headline = final_cert.get("headline", "")
    _rend = [p for p in checks.get("renders", {}).values()
             if p and os.path.exists(p)]
    for extra in ((after_checks or {}).get("renders", {}).values()
                  if isinstance(after_checks, dict) else ()):
        if extra and os.path.exists(extra) and extra not in _rend:
            _rend.append(extra)
    if support_render and support_render not in _rend:
        _rend.append(support_render)
    result["renders"] = _rend
    # plain captions keyed by render-file stem (the UI shows these instead of
    # raw filename jargon like 'premortem'/'traps')
    result["render_captions"] = {
        "premortem": "Where this print is at risk (red = needs attention)",
        "traps": "Trapped liquid and enclosed hollows",
        "warp": "Predicted warp as it cools (estimate, uncalibrated)",
        "strength": "Weakest region under load (comparative screening)",
        "orient": "Strongest vs weakest build orientation (comparative)",
        "support": "Where supports will touch (applied orientation)",
    }

    # honest one-line note when the soft budget clipped optional stages
    if stages.budget_hit and budget_s:
        notes.append(f"time budget (~{budget_s:.0f} s) reached -- some "
                     "optional extras were skipped (each is marked in the "
                     "stage table); the fix and the verdict above ran in "
                     "full")

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

    # -- review data block (ADDITIVE; plain-words contract for the hosted UI).
    #    Built AFTER the verdict so it can never influence it, BEFORE the
    #    report so report.py can render from it.
    review = _build_review(
        result, work_mesh=work_mesh, checks_before=checks,
        checks_after=after_checks, final_cert=final_cert, nozzle_mm=nozzle,
        prof=prof, material=material, mode=mode, units_source=units_source,
        user_target_mm=user_target_mm, src_path=src_path,
        tri_original=tri_original, tri_analysis=tri_analysis,
        decimation_used=decimation_used, holes_before=holes_before,
        holes_after=holes_after, orientation_applied=orientation_applied,
        support_render=support_render, out_dir=out_dir)
    result["review"] = review
    result["extents_mm"] = review.get("extents_mm")     # additive top-level

    md, js = report_mod.build_report(result)
    js["review"] = review
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
