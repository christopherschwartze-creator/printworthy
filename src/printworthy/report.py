"""printworthy.report — the ONE plain-English report renderer (markdown + json).

`build_report(result)` turns a PrepResult (see printworthy.pipeline) into
(markdown_str, json_dict); `attach(result, md, js, out_dir)` writes
``report.md`` / ``report.json`` and returns the report sub-dict.

Rendering rules (house discipline):
  * report.md is UTF-8 (issue strings from the engines may carry °/—/≥); anything
    printed to a CONSOLE goes through core/_mesh_util.ascii_console, so cp1252
    terminals never crash. Do not claim the markdown itself is ASCII — it isn't.
  * HONEST LABELS travel with every number: geometric heuristics say so, the
    warp FEM says calibrated/uncalibrated, strength is COMPARATIVE screening,
    slicer savings are the slicer's own estimates.
  * Never raises: a malformed result still yields a (short) report.
"""
from __future__ import annotations

import json
import os

_REPORT_SCHEMA = 1

_SEV_MARK = {"FAIL": "[FAIL]", "WARN": "[WARN]", "INFO": "[info]"}

# per-channel honesty labels, rendered in the provenance section
_HONESTY = {
    "printability": "geometric heuristic (45-deg overhang + Shape-Diameter "
                    "thin-wall), not a slicer simulation",
    "resin_traps": "voxel drainage triage; upward-facing vents read as drained "
                   "(documented capability gap); volumes are a few % off",
    "warp": "inherent-strain FEM; UNCALIBRATED unless a coupon was fitted "
            "(right sign/shape/trend, not a certified mm)",
    "strength": "orthotropic FEM, COMPARATIVE screening across orientations -- "
                "not a certified safety factor",
    "reinforce": "graded-infill 3MF from the same comparative FEM; min_fos is "
                 "RELATIVE, uncalibrated",
    "resin": "voxel triage (res<=96), not slicer-grade; hollow threshold is a "
             "heuristic label",
    "fit": "axis-aligned bbox over 6 axis permutations (triage, not packing)",
    "orientation": "geometric overhang minimisation (45-deg rule)",
    "retopo": "from-scratch permissive quad remesh; honest quality metrics "
              "attached (val4 %, fidelity rms)",
    "savings": "the slicer's own estimates under the chosen profile",
    "fix": "two-sided Hausdorff deviation certificate vs the source mesh",
    "deformation": "animation / rig-readiness ships separately (autorig build)",
    "trust": "geometric visibility analysis (back-face + occlusion + "
             "frustum test) vs the source camera; requires the source image "
             "and its camera pose -- absent without them, never inferred",
}


# ---------------------------------------------------------------------------
#  json sanitising
# ---------------------------------------------------------------------------
def _jsonable(x, depth=0):
    """Recursively convert numpy scalars/arrays etc. into plain JSON types."""
    if depth > 8:
        return str(x)
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v, depth + 1) for v in x]
    item = getattr(x, "item", None)          # numpy scalar
    if callable(item):
        try:
            return _jsonable(x.item(), depth + 1)
        except Exception:
            pass
    tolist = getattr(x, "tolist", None)      # numpy array
    if callable(tolist):
        try:
            return _jsonable(x.tolist(), depth + 1)
        except Exception:
            pass
    return str(x)


def _num(x, fmt="{:.2f}", dash="--"):
    return fmt.format(x) if isinstance(x, (int, float)) else dash


# ---------------------------------------------------------------------------
#  markdown sections (each never raises: guarded by build_report's try)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#  novice-plain verdict: one line a beginner reads without any jargon.
#  The printability GATE (core/_printability) is authoritative when present --
#  it can only refuse or warn, never upgrade a claim, so a FAIL from the gate
#  overrides an otherwise-green pipeline verdict in the headline a novice sees.
# ---------------------------------------------------------------------------
def _gate(result):
    """The gate dict wired in by prep() (result['gate']), if any. Never raises."""
    g = result.get("gate")
    if isinstance(g, dict) and g.get("verdict") in ("PASS", "WARN", "FAIL"):
        return g
    return None


def _plain_headline(result):
    """(headline_str, next_action_str_or_None). ASCII, no jargon, console-safe."""
    if result.get("rejected"):
        return ("Couldn't read the file.",
                "This couldn't be opened as a 3D model -- check it's a real "
                "STL/OBJ/3MF/PLY/GLB and try again.")
    g = _gate(result)
    gv = g["verdict"] if g else str(result.get("verdict", "")).upper()
    summary = g["plain_summary"] if g else None
    if gv == "FAIL":
        nxt = None
        if g:
            for it in g.get("issues", []):
                if it.get("severity") == "fail":
                    nxt = "Do this next: " + _first_action(it.get("code"))
                    break
        return ("Not ready to print yet -- one thing needs fixing.",
                nxt or "Run the full prep (drop the --check flag) to try an "
                       "automatic repair.")
    if gv == "WARN":
        return ("Probably print-ready, but please double-check one thing.",
                summary and ("Heads up: " + summary))
    if gv == "PASS":
        return ("Print-ready.", None)
    # no gate and no clear verdict: fall back to the raw pipeline verdict text
    raw = str(result.get("verdict", "?"))
    return (f"Result: {raw}.", None)


def _first_action(code):
    """Map a gate issue code to the single plainest next step. Never raises."""
    return {
        "not_watertight": "run the full prep to close the holes and make it a "
                          "solid, then re-check.",
        "inside_out": "run the full prep -- it flips the inside-out surfaces "
                      "for you automatically.",
        "zero_volume": "this shape is flat/collapsed; give it real thickness in "
                       "your modelling tool before printing.",
        "sub_nozzle_extent": "make the part thicker than your nozzle (0.4 mm), "
                             "or scale it up -- it's too thin to print as is.",
        "no_faces": "this file has no geometry; re-export the model.",
        "all_degenerate": "the mesh is broken (all flat triangles); re-export "
                          "or re-model it.",
        "bbox_nonfinite": "the file has broken coordinates; re-export it "
                          "cleanly.",
        "fem_no_result": "the part is too degenerate to reinforce; fix its "
                         "shape first, then try again.",
    }.get(code, "run the full prep to attempt an automatic repair, then "
                "re-check.")


def _gate_section(result):
    """Prominent plain-English printability verdict, driven by the gate."""
    g = _gate(result)
    if not g:
        return []
    lines = ["## Can I print this?", "", g.get("plain_summary", ""), ""]
    fails = [i for i in g.get("issues", []) if i.get("severity") == "fail"]
    warns = [i for i in g.get("issues", []) if i.get("severity") == "warn"]
    if fails:
        lines += ["**Blocking problems (must fix before printing):**", ""]
        lines += [f"- {i.get('message')}" for i in fails]
        lines.append("")
    if warns:
        lines += ["**Cautions (please check):**", ""]
        lines += [f"- {i.get('message')}" for i in warns]
        lines.append("")
    return lines


def _banner(result):
    head, nxt = _plain_headline(result)
    lines = [f"# printworthy report -- {head}", ""]
    if nxt:
        lines += [f"**{nxt}**", ""]
    # the machine-readable verdict stays visible for anyone who wants it
    lines += [f"_verdict: {result.get('verdict', '?')}_", ""]
    if result.get("headline"):
        lines += [f"{result['headline']}", ""]
    if result.get("rejected"):
        lines += [f"Input rejected: {result['rejected']}", ""]
    fx = result.get("fix") or {}
    if fx.get("after_verdict"):
        b, a = fx.get("before_verdict"), fx.get("after_verdict")
        flip = f"{b} -> {a}" if a != b else f"{a} (unchanged)"
        lines += [f"Free fix applied: **{flip}**", ""]
    meta = (f"profile `{result.get('profile')}` | mode `{result.get('mode')}` "
            f"| material `{result.get('material')}`")
    if result.get("print_mm"):
        meta += f" | print size {result['print_mm']:g} mm (longest side)"
    lines += [meta, ""]
    return lines


def _issues(result):
    cert = result.get("cert") or {}
    issues = cert.get("issues") or []
    lines = ["## What to fix (most important first)", ""]
    if not issues:
        lines.append("Nothing to flag.")
    for i in issues:
        lines.append(f"- {_SEV_MARK.get(i.get('sev'), '[?]')} "
                     f"**{i.get('title')}** -- {i.get('detail')}")
        lines.append(f"  - fix: {i.get('fix')}")
    lines.append("")
    good = cert.get("good") or []
    if good:
        lines += ["## Already fine", ""]
        lines += [f"- {g}" for g in good]
        lines.append("")
    return lines


def _numbers(result):
    f = result.get("facts") or {}
    ch = result.get("channels") or {}
    p = ch.get("printability") or {}
    t = ch.get("resin_traps") or {}
    # normals row: winding-consistency alone does NOT mean the normals are
    # CORRECT -- a uniformly-inverted (inside-out) solid is winding-consistent
    # yet points inward (negative volume). Consult the gate's signed-volume
    # signal so check_only never asserts "Consistent normals: yes" on an
    # inside-out part; instead it says so plainly and notes prep will flip it.
    g = _gate(result)
    inside_out = bool(g and g.get("checks", {}).get("positive_volume") is False
                      and f.get("winding_consistent"))
    if inside_out:
        normals_val = ("consistent, but pointing INWARD (negative volume) -- "
                       "full prep will flip them")
    elif f.get("winding_consistent"):
        normals_val = "yes (consistent)"
    else:
        normals_val = "NO"
    rows = []
    if f:
        rows += [("Triangles", f"{f.get('faces', 0):,}"),
                 ("Watertight", "yes" if f.get("watertight") else "NO"),
                 ("Consistent normals", normals_val),
                 ("Handles/tunnels (genus)", "--" if f.get("genus") is None
                  else str(f["genus"])),
                 ("Bounding box (mm)", " x ".join(str(x) for x in
                                                  f.get("bbox_mm", [])))]
    if p and "error" not in p:
        rows.append(("Thinnest wall (mm)", _num(p.get("min_wall_mm"))))
        af = p.get("high_risk_area_frac")
        rows.append(("Surface needing supports",
                     _num(af, "{:.0%}") if isinstance(af, (int, float)) else "--"))
    if t and "error" not in t:
        rows.append(("Trapped resin/powder (cm3)",
                     _num(t.get("total_trapped_cm3"), "{:.1f}")))
    w = ch.get("warp") or {}
    if isinstance(w.get("max_corner_lift_mm"), (int, float)):
        tag = "" if w.get("calibrated") else " (est., uncalibrated)"
        rows.append(("Predicted corner warp (mm)",
                     f"{w['max_corner_lift_mm']:.2f}{tag}"))
    s = ch.get("strength") or {}
    if isinstance(s.get("improvement_ratio"), (int, float)):
        bd = s.get("best_dir")
        bds = "[" + ",".join(f"{x:.0f}" for x in bd) + "]" if bd else "?"
        rows.append(("Orientation strength gain (comparative)",
                     f"x{s['improvement_ratio']:.1f} -> print {bds}"))
    o = ch.get("orientation") or {}
    if isinstance(o.get("support_area_frac"), (int, float)):
        before = o.get("support_area_frac_as_loaded")
        v = f"{o['support_area_frac']:.1%}"
        if isinstance(before, (int, float)):
            v = f"{before:.1%} -> {v} (reoriented)"
        rows.append(("Overhang area (support risk)", v))
    r = ch.get("reinforce") or {}
    if r.get("ok"):
        rows.append(("Reinforce min FOS (RELATIVE)",
                     str(r.get("min_fos_display"))))
        rows.append(("Dense-infill modifiers", str(r.get("n_modifiers"))))
    fit = ch.get("fit") or {}
    if fit.get("ok"):
        v = "yes" if fit.get("fits") else \
            f"NO -- scale by {fit.get('suggested_scale')}"
        rows.append((f"Fits the bed ({result.get('profile')})", v))
    rt = ch.get("retopo") or {}
    if rt.get("ok"):
        rows.append(("Quad remesh", f"{rt.get('quads')} quads, "
                                    f"val4 {rt.get('val4_pct')}%, fidelity rms "
                                    f"{rt.get('fidelity_rms')}"))
    if not rows:
        return []
    lines = ["## The numbers", "", "| Measurement | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines.append("")
    return lines


def _fix_section(result):
    fx = result.get("fix") or {}
    if not fx:
        return []
    lines = ["## Repair", ""]
    if fx.get("fidelity_line"):
        lines.append(f"**{fx['fidelity_line']}**")
        lines.append("")
    if fx.get("note"):
        lines.append(f"_{fx['note']}_")
        lines.append("")
    if fx.get("after_headline"):
        lines.append(f"After the fix: {fx['after_headline']}")
        lines.append("")
    # internal-shell (cavity) certificate line -- additive, only when present.
    # GEOMETRIC estimate (component split + bbox nesting), not a slicer number;
    # the honesty label travels with it.
    ish = fx.get("internal_shells") or {}
    n = ish.get("n_internal_shells")
    if isinstance(n, int) and n > 0:
        void = ish.get("internal_void_mm3")
        ratio = ish.get("solid_volume_ratio")
        void_s = f"{void:.0f}" if isinstance(void, (int, float)) else "--"
        pct_s = (f"{ratio * 100:.0f}" if isinstance(ratio, (int, float))
                 else "--")
        lines.append(
            f"Kept {n} closed internal shell(s) as cavities "
            f"(void ~{void_s} mm^3); a faithful print uses ~{pct_s}% of the "
            "material a cavity-filling auto-repair would (geometric estimate; "
            "the gap shrinks at low infill).")
        lines.append("")
    return lines


def _trust_section(result):
    """Trust map (opt-in; requires an explicit source camera). ADDITIVE and
    entirely OMITTED when the feature did not run -- an absent camera must
    never render a fabricated section (mirrors the cavities-triple pattern:
    computed flag + conditional prose, honesty label always attached)."""
    tc = (result.get("channels") or {}).get("trust")
    if not isinstance(tc, dict) or not tc.get("ok"):
        return []
    pct = tc.get("frac_invented_pct")
    pct_s = f"{pct:.1f}" if isinstance(pct, (int, float)) else "?"
    seen_s = f"{100.0 - pct:.1f}" if isinstance(pct, (int, float)) else "?"
    lines = ["## Trust map -- what the source camera actually saw", "",
             f"**{pct_s}% of the surface area is AI-invented** (back-facing, "
             f"occluded, or outside the source camera's frame); the "
             f"remaining {seen_s}% is evidence-backed (visible to the "
             "source photo). Pure geometric visibility analysis (back-face "
             "+ occlusion + frustum test), not a learned prior; requires "
             "the source image and its camera pose.", ""]
    lines.append(
        f"Trust map: the front of this model is faithful to your source "
        f"image; the back and interior ({pct_s}% by surface area) were "
        "invented by the AI generator and made printable, but their "
        "accuracy cannot be verified against anything. For decorative "
        "prints this is usually fine. For functional parts, replicas, "
        "measurement, or any structural/foolproof claim, review the "
        "invented (orange) regions before relying on them.")
    lines.append("")
    if tc.get("carry_mode"):
        lines.append(f"Label carry through the fix: {tc['carry_mode']} "
                     "(disclosed; report['trust_carry'] mirrors this -- "
                     "exact for index-preserving repairs, approximate "
                     "nearest-face after a global reseal).")
        lines.append("")
    n_rf = tc.get("n_repair_filled")
    if isinstance(n_rf, int) and n_rf > 0:
        lines.append(f"{n_rf} face(s) were added by the repair step and are "
                     "labelled repair_filled -- not evidence, and not the "
                     "AI's original invention, but new geometry from the "
                     "fixer.")
        lines.append("")
    _disc = None
    for _cn in ("strength", "reinforce"):
        _ch = (result.get("channels") or {}).get(_cn) or {}
        if isinstance(_ch, dict) and _ch.get("trust_disclosure"):
            _disc = _ch["trust_disclosure"]
            break
    if _disc:
        lines.append(f"**{_disc}**")
        lines.append("")
    return lines


def _resin_section(result):
    rr = (result.get("channels") or {}).get("resin") or {}
    if not rr:
        return []
    lines = ["## Resin checks", ""]
    for fl in rr.get("flags") or []:
        lines.append(f"- [WARN] {fl}")
    if not (rr.get("flags") or []):
        lines.append("- no resin-specific flags")
    h = rr.get("hollow") or {}
    if h.get("suggested"):
        lines.append(f"- hollowing suggested "
                     f"(~{h.get('material_saved_pct')}% resin saved; "
                     "needs a drain hole)")
    lines.append("")
    return lines


def _savings_section(result):
    sv = result.get("savings")
    if not isinstance(sv, dict):
        return []
    lines = ["## Print savings (slicer estimates)", ""]
    if not sv.get("ok"):
        lines.append(f"_{sv.get('note', 'savings unavailable')}_")
    else:
        bits = []
        if isinstance(sv.get("saved_min"), (int, float)):
            bits.append(f"{sv['saved_min']:.0f} min print time")
        if isinstance(sv.get("saved_g"), (int, float)):
            bits.append(f"{sv['saved_g']:.1f} g filament")
        pct = sv.get("saved_pct")
        pct_s = f" ({pct:+.0f}%)" if isinstance(pct, (int, float)) else ""
        lines.append("Prepped vs as-uploaded: "
                     + (", ".join(bits) if bits else "no change")
                     + pct_s + f" -- {sv.get('slicer', 'slicer')} estimates "
                               "under this profile.")
        if sv.get("note"):
            lines.append(f"_{sv['note']}_")
    lines.append("")
    return lines


def _provenance(result):
    lines = ["## Provenance (what each number is, honestly)", ""]
    ch = result.get("channels") or {}
    seen = []
    for key in ch:
        lab = _HONESTY.get(key)
        if lab:
            seen.append(f"- **{key}**: {lab}")
    if result.get("fix"):
        seen.append(f"- **fix**: {_HONESTY['fix']}")
    if isinstance(result.get("savings"), dict):
        seen.append(f"- **savings**: {_HONESTY['savings']}")
    seen.append(f"- **animation**: {_HONESTY['deformation']}")
    lines += seen
    lines.append("")
    st = result.get("stages") or []
    if st:
        lines += ["### Stages", "", "| stage | ok | s | note |", "|---|---|---|---|"]
        for r in st:
            ok = {True: "yes", False: "FAILED", None: "skip"}.get(r.get("ok"), "?")
            lines.append(f"| {r.get('stage')} | {ok} | "
                         f"{r.get('seconds') if r.get('seconds') is not None else '--'} | "
                         f"{r.get('note', '')} |")
        lines.append("")
    return lines


def _files_and_notes(result):
    lines = []
    files = {k: v for k, v in (result.get("files") or {}).items() if v}
    if files:
        lines += ["## Files", ""]
        lines += [f"- {k}: `{v}`" for k, v in files.items()]
        lines.append("")
    notes = result.get("notes") or []
    if notes:
        lines += ["## Notes", ""]
        lines += [f"- {n}" for n in notes]
        lines.append("")
    return lines


# ---------------------------------------------------------------------------
#  public API
# ---------------------------------------------------------------------------
def build_report(result) -> tuple[str, dict]:
    """PrepResult -> (markdown, json_dict). Never raises."""
    try:
        lines: list[str] = []
        for section in (_banner, _gate_section, _issues, _numbers, _fix_section,
                        _trust_section, _resin_section, _savings_section,
                        _provenance, _files_and_notes):
            try:
                lines += section(result)
            except Exception as e:
                lines += [f"_({section.__name__} unavailable: "
                          f"{type(e).__name__})_", ""]
        md = "\n".join(lines).rstrip() + "\n"
    except Exception as e:
        md = f"# printworthy report\n\nreport rendering failed: {e}\n"
    try:
        cert = result.get("cert") or {}
        js = _jsonable({
            "schema": _REPORT_SCHEMA,
            "verdict": result.get("verdict"),
            "headline": result.get("headline"),
            "rejected": result.get("rejected"),
            "profile": result.get("profile"),
            "mode": result.get("mode"),
            "material": result.get("material"),
            "print_mm": result.get("print_mm"),
            "issues": cert.get("issues", []),
            "good": cert.get("good", []),
            "facts": result.get("facts", {}),
            "channels": result.get("channels", {}),
            "fix": result.get("fix"),
            "trust_carry": result.get("trust_carry"),
            "savings": result.get("savings"),
            "files": result.get("files", {}),
            "renders": result.get("renders", []),
            "stages": result.get("stages", []),
            "notes": result.get("notes", []),
            "honesty": {k: _HONESTY[k] for k in _HONESTY
                        if k in (result.get("channels") or {})
                        or k in ("deformation",)},
        })
    except Exception as e:
        js = {"verdict": result.get("verdict"), "error": f"json build: {e}"}
    return md, js


def attach(result, md, js, out_dir=None) -> dict:
    """Write report.md / report.json into out_dir (best-effort) and return the
    report sub-dict {markdown, json, md_path, json_path}. Never raises."""
    md_path = json_path = None
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            md_path = os.path.join(out_dir, "report.md")
            with open(md_path, "w", encoding="utf-8") as fh:
                fh.write(md)
        except Exception:
            md_path = None
        try:
            json_path = os.path.join(out_dir, "report.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(js, fh, indent=2, default=str)
        except Exception:
            json_path = None
    return {"markdown": md, "json": js, "md_path": md_path,
            "json_path": json_path}


# =============================================================================
#  REVIEW DOCUMENT — the Space-facing novice page (ADDITIVE layer)
# =============================================================================
# `build_review(result)` renders the browser-tab review page per the final
# synthesized spec (verdict -> what-changed -> size -> pictures -> ranked
# cards -> download+shop note -> optional warp -> fine print).
# `review_selfcheck(md)` mechanically tests the spec's acceptance criteria
# and returns a list of violations (empty = clean) for validators / CI.
#
# House rules honoured here:
#   * additive: build_report/attach above are untouched; this layer READS the
#     same result dict; missing keys degrade to omitted content, never raise.
#   * honesty: numbers keep their labels (measured / estimate / uncalibrated);
#     language may simplify, never upgrade; a refusal is a refusal.
#   * the review string is UTF-8 (em-dashes, degree sign, multiplication sign);
#     console printing goes through core/_mesh_util.ascii_console (see the
#     __main__ smoke harness at the bottom).
#   * Space runtime pin (verified installed on this box): gradio==6.19.0,
#     gradio_client==2.5.0 — these exact versions belong in the space
#     requirements file.
import math
import re

# --- everyday-object anchors (lookup tables; plain words, no jargon) ---------
_RV_SIZE_ANCHORS = [
    (10.0,   "the size of a grain of rice"),
    (20.0,   "the size of a fingernail"),
    (35.0,   "the length of a wine cork"),
    (55.0,   "the size of a golf ball"),
    (80.0,   "the width of a credit card"),
    (110.0,  "the height of a chess king"),
    (160.0,  "the height of a smartphone"),
    (230.0,  "the height of a sheet of paper"),
    (350.0,  "the size of a shoebox"),
]
_RV_SIZE_ANCHOR_BIG = "bigger than a basketball"

_RV_DEV_ANCHORS = [
    (0.0,   "no measurable change at all"),
    (0.03,  "about a quarter of a human hair"),
    (0.08,  "about the width of a human hair"),
    (0.15,  "thinner than a sheet of paper"),
    (0.35,  "about the thickness of a business card"),
    (0.80,  "about the thickness of a credit card"),
    (1.50,  "about the thickness of a coin"),
]
_RV_DEV_ANCHOR_BIG = ("thicker than a coin — look closely at the pictures "
                      "below before printing")


def _rv_size_anchor(mm):
    try:
        mm = float(mm)
    except (TypeError, ValueError):
        return "an everyday object"
    for lim, word in _RV_SIZE_ANCHORS:
        if mm < lim:
            return word
    return _RV_SIZE_ANCHOR_BIG


def _rv_dev_anchor(mm):
    try:
        mm = float(mm)
    except (TypeError, ValueError):
        return "a very small distance"
    for lim, word in _RV_DEV_ANCHORS:
        if mm <= lim:
            return word
    return _RV_DEV_ANCHOR_BIG


def _rv_fmt(x, nd=2):
    """Format a number compactly ('97.4', '100', '0.06'); str(x) fallback."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if xf != xf or xf in (float("inf"), float("-inf")):   # NaN / inf
        return "?"
    if 0 < abs(xf) < 0.01:
        nd = max(nd, 3)
    s = f"{xf:.{nd}f}"
    # strip trailing zeros only AFTER a decimal point; f"{60.0:.0f}" == "60" has none,
    # and a blind rstrip("0") would turn 60 -> "6" and 100 -> "1" (exact multiples of
    # ten at nd=0 are common: support %, savings %, invented %).
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _rv_num(x):
    """float(x) or None (never raises)."""
    try:
        v = float(x)
        return v if v == v and v not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


# --- render sanitising: NEVER a server-local path on the page -----------------
_RV_RENDER_MAX_BYTES = 4 * 1024 * 1024


def _rv_render_uri(p):
    """A render reference safe to embed in the review: pass through data:/http(s)
    URIs; inline a local image file as a base64 data URI (self-contained in the
    browser AND in the downloaded review.md); anything else (dead/leaky local
    path) -> None so the picture is omitted rather than broken. Never raises."""
    if not isinstance(p, str) or not p.strip():
        return None
    if p.startswith(("data:", "http://", "https://")):
        return p
    try:
        if os.path.isfile(p) and os.path.getsize(p) <= _RV_RENDER_MAX_BYTES:
            ext = os.path.splitext(p)[1].lower().lstrip(".") or "png"
            if ext == "jpg":
                ext = "jpeg"
            if ext in ("png", "jpeg", "gif", "webp"):
                import base64
                with open(p, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode("ascii")
                return f"data:image/{ext};base64,{b64}"
    except Exception:
        pass
    return None


# --- plain-language scrubber: raw engine strings NEVER reach the novice page --
# Ordered (pattern, replacement); case-insensitive unless the pattern says so.
# This may only SIMPLIFY language — every number and claim passes through
# unchanged; only vocabulary is translated.
_RV_PLAIN_SUBS = [
    (r"Set the print size \(or assume_unit='?m'?\)",
     "Set the print size (or set the file-units to metres)"),
    (r"assume_unit(='?\w+'?)?", "the file-units setting"),
    (r"\bprint_mm\b", "the print-size setting"),
    # engine reject phrasings, translated whole BEFORE the generic word
    # substitutions below would render them clumsy ("surface detail faces")
    (r"contains no triangle faces", "has no printable 3D surface in it"),
    (r"is not a triangle mesh", "is not a readable 3D model"),
    (r"\(?`?--[a-z][\w-]*`?\)?", "the matching option"),
    (r"[A-Za-z]:[\\/][^\s)\"'`]+", "(your download package)"),
    (r"(/tmp|/home)/[^\s)\"'`]*", "(your download package)"),
    (r"mesh editor", "3D repair program"),
    (r"\bmesh(es)?\b", "model"),
    (r"hausdorff", "direct distance measurement"),
    (r"\(genus\s*>\s*0\)\.?", "."),
    (r"\bgenus\b", "tunnel count"),
    (r"\bwatertight\b", "fully sealed"),
    (r"\bmanifold\b", "sealed"),
    (r"face normals|\bnormals?\b", "surface direction"),
    (r"overhang/support-risk", "steep and will need scaffolding"),
    (r"\boverhangs?\b", "steep areas"),
    (r"single extrusion", "single printer line"),
    (r"\bextrusions?\b", "printer line"),
    (r"\bvoxel\w*", "3D grid"),
    (r"\binfill\b", "inner filling"),
    (r"\bbrim\b( or raft)?|\braft\b", "wider, stickier first layer"),
    (r"bounding box|\bbbox\b", "overall size"),
    (r"tessellat\w+|triangulat\w+|\btriangles?\b|\bvertex\b|\bvertices\b",
     "surface detail"),
    (r"\bdegenerate\w*", "broken"),
    (r"self.intersect\w*", "surfaces passing through each other"),
    (r"\bdecimat\w+", "simplified"),
    (r"shape.diameter( function)?|\bsdf\b", "thickness measurement"),
    (r"\bheuristics?\b", "rule of thumb"),
    (r"\bremesh\w*|\bretopo\w*", "surface rebuild"),
    (r"\bboolean\b", "merge"),
    (r"topolog\w+", "shape structure"),
    (r"\bslicer\b", "print software"),
    (r"\bsimulation\b", "software check"),
    (r"blocking issues", "must-fix problems"),   # plural first: keeps grammar
    (r"blocking issue(s)?", "must-fix problem"),
    (r"\bpipeline\b", "process"),
    (r"\bstages?\b", "step"),
    (r"\btriage\b", "quick check"),
    (r"\bre-check\w*", "second check"),
    (r"\bprovenance\b", "where each number comes from"),
    (r"\bguaranteed\b", "expected"),
    (r"\bcertified\b", "verified"),
    (r"\bfoolproof\b|\boptimal\b|\bperfect\b", "good"),
]
_RV_PLAIN_SUBS_CS = [                    # case-sensitive verdict/process tokens
    (r"\bFDM\b", "standard filament printing"),
    (r"\bSLA\b", "resin printing"),
    (r"\bPLA\b", "standard plastic"),
    (r"\bFAIL\b", "not printable yet"),
    (r"\bPASS\b", "printable"),
]


def _rv_plain_text(t):
    """Translate an engine-facing string into shop-window words. Numbers and
    meaning survive; jargon does not. Never raises."""
    try:
        s = str(t or "")
        for pat, rep in _RV_PLAIN_SUBS:
            s = re.sub(pat, rep, s, flags=re.I)
        for pat, rep in _RV_PLAIN_SUBS_CS:
            s = re.sub(pat, rep, s)
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"\s+([.,;:!?])", r"\1", s)
        return s
    except Exception:
        return str(t or "")


# --- topic classifier: known findings render as CANONICAL plain cards ---------
def _rv_card_topic(fact_lower):
    """Map a raw finding string to a canonical card topic (or None)."""
    fl = fact_lower
    if "zero-thickness" in fl or "zero thickness" in fl:
        return None                      # already plain (pipeline rewrite)
    if "thinnest wall" in fl or ("thin" in fl and
                                 ("wall" in fl or "nozzle" in fl
                                  or "extrusion" in fl)):
        return "thin"
    if ("overhang" in fl or "support-risk" in fl or "support material" in fl
            or "scaffolding" in fl):
        return "supports"
    if "watertight" in fl or "open holes" in fl or "fully seal" in fl \
            or "open boundaries" in fl:
        return "sealed"
    if "tunnel/handle" in fl or "genus" in fl or "handles" in fl:
        return "tunnels"
    if "winding" in fl or "point inward" in fl or "inside-out" in fl \
            or "face wrong way" in fl:
        return "inside_out"
    return None


# numberless one-liners for restating a residual problem (so the same decimal
# never repeats across banner / receipt / download sections)
_RV_TOPIC_SHORT = {
    "thin": "some parts are still thinner than the printer can draw",
    "supports": "large steep areas will need scaffolding",
    "sealed": "the shape still isn't fully sealed",
    "tunnels": "the shape has unexpected tunnels through it",
    "inside_out": "parts of the surface still face the wrong way",
}


def _rv_thin_card(f):
    """Canonical thin-wall blocking card (numbers from the measured fields)."""
    thin, pmin = f.get("thin_wall_min_mm"), f.get("printable_min_mm")
    if thin is None or pmin is None:
        return None
    if thin <= 0:
        return {"severity": "blocking",
                "fact_plain": "Some walls have zero thickness — there is no "
                              "material there at all.",
                "consequence_plain": "They would print as gaps no matter the "
                                     "size — making the model bigger will NOT "
                                     "fix a zero-thickness wall.",
                "action_plain": "Regenerate the model with a solid/printable "
                                "option, or ask the print shop to thicken it "
                                "for you (a small paid job).",
                "part_name": f.get("part_name"), "topic": "thin",
                "short_plain": _RV_TOPIC_SHORT["thin"]}
    if thin >= pmin:
        return None
    part = f.get("part_name") or "thinnest part of your model"
    sf = f.get("scale_factor_suggested")
    if sf is None:
        sf = math.ceil((pmin / thin) * 10.0) / 10.0
        f["scale_factor_suggested"] = sf
    return {"severity": "blocking",
            "fact_plain": f"The {part} is {_rv_fmt(thin)} mm thick — thinner "
                          "than the finest line a standard printer can draw "
                          f"({_rv_fmt(pmin)} mm).",
            "consequence_plain": "It may come out with gaps, or snap off in "
                                 "your hands.",
            "action_plain": "Either ask the shop to print the whole model "
                            f"{_rv_fmt(sf, 1)}× bigger, or accept that this "
                            "part will be fragile — your call. (A resin "
                            "print handles fine detail; the shop note below "
                            "mentions it.)",
            "part_name": part, "topic": "thin",
            "short_plain": _RV_TOPIC_SHORT["thin"]}


def _rv_supports_card(f):
    """Canonical scaffolding info card (the ONLY supports % on the page body
    besides the pictures pair)."""
    sup = f.get("support_pct_oriented")
    if sup is None or sup < 0.5:
        return None            # "about 0% needs scaffolding" is just noise
    zone = str(f.get("support_zone_plain") or "on the underside")
    # upstream zone strings may carry their own "mostly ..." — never render
    # "mostly mostly on the underside"
    zone = re.sub(r"^\s*mostly\s+", "", zone, flags=re.I)
    where = "" if "no meaningful" in zone.lower() else f", mostly {zone}"
    return {"severity": "info",
            "fact_plain": f"About {_rv_fmt(sup, 0)}% of the surface will need "
                          f"temporary scaffolding{where}.",
            "consequence_plain": "Small rough patches where the scaffolding "
                                 "touched — the red zones in the picture "
                                 "above.",
            "action_plain": "Nothing; the shop handles this. Just don't "
                            "expect those spots to be glass-smooth.",
            "part_name": None, "topic": "supports",
            "short_plain": _RV_TOPIC_SHORT["supports"]}


def _rv_canonical_card(topic, sev, fact, consequence, action, f):
    """Replace a known raw finding with its canonical plain card. Falls back
    to the scrubbed original text when the measured fields are missing."""
    if topic == "thin":
        c = _rv_thin_card(f)
        if c:
            c["severity"] = "blocking" if sev == "blocking" else c["severity"]
            return c
    elif topic == "supports":
        c = _rv_supports_card(f)
        if c:
            return c
    elif topic == "sealed":
        act = _rv_plain_text(action) or (
            "Ask your print shop — sealing a shape is a routine repair.")
        return {"severity": sev,
                "fact_plain": "The shape isn't fully sealed — its surface "
                              "has open gaps.",
                "consequence_plain": "Printers need a sealed shape to tell "
                                     "inside from outside; gaps can produce "
                                     "a broken or hollow print.",
                "action_plain": act, "part_name": None, "topic": "sealed",
                "short_plain": _RV_TOPIC_SHORT["sealed"]}
    elif topic == "tunnels":
        return {"severity": "info",
                "fact_plain": "The shape has hidden tunnels or handles "
                              "running through it (like the hole through a "
                              "mug handle).",
                "consequence_plain": "These often come with the file rather "
                                     "than being designed on purpose; they "
                                     "usually print fine.",
                "action_plain": "Look at the pictures — if the shape looks "
                                "right to you, there is nothing to do.",
                "part_name": None, "topic": "tunnels",
                "short_plain": _RV_TOPIC_SHORT["tunnels"]}
    elif topic == "inside_out":
        act = _rv_plain_text(action) or (
            "The automatic fix turns these right-side out.")
        return {"severity": sev,
                "fact_plain": "Parts of the surface face the wrong way "
                              "(inside-out).",
                "consequence_plain": "The printer can misjudge what is solid "
                                     "and what is empty space.",
                "action_plain": act, "part_name": None, "topic": "inside_out",
                "short_plain": _RV_TOPIC_SHORT["inside_out"]}
    # unknown topic: scrub every string, keep the meaning
    return {"severity": sev,
            "fact_plain": _rv_plain_text(fact),
            "consequence_plain": _rv_plain_text(consequence)
            or "It may affect how the print comes out.",
            "action_plain": _rv_plain_text(action)
            or "Ask your print shop — they handle this routinely.",
            "part_name": None, "topic": None, "short_plain": None}


# --- field extraction: flat spec keys first, then derived from the pipeline --
def _rv_get(result, key, default=None):
    rv = result.get("review")
    if isinstance(rv, dict) and key in rv:
        return rv[key]
    if key in result:
        return result[key]
    return default


_RV_SEV_MAP = {"fail": "blocking", "warn": "check", "info": "info",
               "blocking": "blocking", "check": "check"}
_RV_SEV_RANK = {"blocking": 0, "check": 1, "info": 2}


def _rv_warnings(result, f):
    """Normalise / synthesise the ranked warning cards. Every card gets a
    non-empty action; every raw engine string is either replaced by its
    CANONICAL plain card (known topics) or scrubbed of jargon. Never raises."""
    cards = []
    seen_topics = set()

    def _take(sev, fact, consequence, action, part=None):
        fact = str(fact or "").strip()
        if not fact:
            return
        topic = _rv_card_topic(fact.lower())
        if topic in seen_topics:
            return                        # one card per topic, first wins
        c = _rv_canonical_card(topic, sev, fact, consequence, action, f)
        if c is None:
            return
        if topic is None and part:
            c["part_name"] = part
        if c.get("topic"):
            seen_topics.add(c["topic"])
        cards.append(c)

    raw = _rv_get(result, "warnings")
    if isinstance(raw, list):
        for w in raw:
            if not isinstance(w, dict):
                continue
            _take(_RV_SEV_MAP.get(str(w.get("severity", "info")).lower(),
                                  "info"),
                  w.get("fact_plain") or w.get("fact"),
                  w.get("consequence_plain") or w.get("consequence"),
                  w.get("action_plain") or w.get("action"),
                  w.get("part_name"))
    else:
        # graceful fallback: map the cert's ranked issues
        cert = result.get("cert") or {}
        for i in cert.get("issues") or []:
            if not isinstance(i, dict):
                continue
            _take(_RV_SEV_MAP.get(str(i.get("sev", "INFO")).lower(), "info"),
                  i.get("title"), i.get("detail"), i.get("fix"))

    joined = " ".join(c["fact_plain"].lower() for c in cards)

    # --- standard detection: thin walls (blocking) ---------------------------
    if "thin" not in seen_topics and "thin" not in joined:
        c = _rv_thin_card(f)
        if c:
            cards.append(c)
            seen_topics.add("thin")

    # --- standard detection: multi-piece (check) ------------------------------
    ncomp = f.get("components_count")
    if isinstance(ncomp, int) and ncomp > 1 and "pieces" not in joined:
        cards.append({
            "severity": "check",
            "fact_plain": f"Your model is made of {ncomp} separate pieces. "
                          "Sometimes that's on purpose (a character holding a "
                          "separate sword); sometimes it means the file is "
                          "shattered.",
            "consequence_plain": "Shattered pieces can print as loose "
                                 "fragments.",
            "action_plain": "Look at the 'after' picture above — if the model "
                            "looks whole and solid, you're fine; if bits are "
                            "floating apart, re-export the model and upload "
                            "again.",
            "part_name": None, "topic": "pieces",
            "short_plain": "the model is split into separate pieces",
        })

    # --- standard detection: supports (info) ----------------------------------
    if "supports" not in seen_topics and "scaffolding" not in joined:
        c = _rv_supports_card(f)
        if c:
            cards.append(c)
            seen_topics.add("supports")

    cards.sort(key=lambda c: _RV_SEV_RANK.get(c["severity"], 3))
    return cards


def _rv_fields(result):
    """Normalise a prep() result (or a flat spec-shaped dict) into the review
    field set. Missing data -> None (sections omit themselves). Never raises."""
    if not isinstance(result, dict):
        result = {}
    facts = result.get("facts") or {}
    fix = result.get("fix") if isinstance(result.get("fix"), dict) else {}
    dcert = fix.get("deviation_certificate") or {}
    gate = result.get("gate") if isinstance(result.get("gate"), dict) else {}
    ch = result.get("channels") or {}
    pr = ch.get("printability") if isinstance(ch.get("printability"), dict) else {}
    orient = ch.get("orientation") if isinstance(ch.get("orientation"), dict) else {}
    ish = (result.get("internal_shells")
           if isinstance(result.get("internal_shells"), dict)
           else (fix.get("internal_shells") or {}))
    files = result.get("files") or {}

    f = {}
    f["fix_ran"] = bool(_rv_get(result, "fix_ran", bool(fix)))
    f["surface_kept_pct"] = _rv_num(_rv_get(result, "surface_kept_pct",
                                            dcert.get("surface_unchanged_pct")))
    f["max_deviation_mm"] = _rv_num(_rv_get(result, "max_deviation_mm",
                                            dcert.get("max_deviation_mm")))
    f["deviation_anchor"] = (_rv_get(result, "deviation_anchor")
                             or _rv_dev_anchor(f["max_deviation_mm"]))
    hc = _rv_get(result, "holes_filled_count")
    f["holes_filled_count"] = int(hc) if isinstance(hc, (int, float)) else None

    # cavities
    nc = _rv_get(result, "cavities_count", ish.get("n_internal_shells"))
    f["cavities_count"] = int(nc) if isinstance(nc, (int, float)) else 0
    ms = _rv_get(result, "material_savings_pct")
    if ms is None and _rv_num(ish.get("solid_volume_ratio")) is not None:
        ms = float(ish["solid_volume_ratio"]) * 100.0
    f["material_savings_pct"] = _rv_num(ms)

    # dimensions
    bbox = _rv_get(result, "bbox_mm", facts.get("bbox_mm"))
    ext = [_rv_num(_rv_get(result, "extents_x_mm")),
           _rv_num(_rv_get(result, "extents_y_mm")),
           _rv_num(_rv_get(result, "extents_z_mm"))]
    if any(e is None for e in ext) and isinstance(bbox, (list, tuple)) \
            and len(bbox) == 3:
        ext = [_rv_num(b) for b in bbox]
    f["extents_mm"] = ext if all(e is not None for e in ext) else None
    if f["extents_mm"]:
        f["longest_dim_mm"] = _rv_num(_rv_get(result, "longest_dim_mm",
                                              max(f["extents_mm"])))
        f["min_extent_mm"] = _rv_num(_rv_get(result, "min_extent_mm",
                                             min(f["extents_mm"])))
    else:
        f["longest_dim_mm"] = _rv_num(_rv_get(result, "longest_dim_mm"))
        f["min_extent_mm"] = _rv_num(_rv_get(result, "min_extent_mm"))
    fo = _rv_get(result, "flat_object")
    if fo is None and f["longest_dim_mm"] and f["min_extent_mm"] is not None:
        fo = (f["min_extent_mm"] < 2.0
              and f["longest_dim_mm"] / max(f["min_extent_mm"], 1e-9) > 15.0)
    f["flat_object"] = bool(fo)
    sa = (_rv_get(result, "size_anchor")
          or _rv_size_anchor(f["longest_dim_mm"]))
    # upstream anchors may already start with "about ..." while the size
    # sentence supplies its own "about" — strip it so the page never reads
    # "about about the size of an egg"
    f["size_anchor"] = re.sub(r"^\s*about\s+", "", str(sa), flags=re.I)

    us = _rv_get(result, "units_source")
    if us not in ("file", "assumed_mm", "user_target"):
        us = "user_target" if result.get("print_mm") else "assumed_mm"
    f["units_source"] = us
    f["user_target_size_mm"] = _rv_num(_rv_get(result, "user_target_size_mm",
                                               result.get("print_mm")))

    # renders: whatever the caller supplies is SANITISED here — a local image
    # file becomes an embedded data URI (works in the browser and in the
    # downloaded review.md); a server path that can't be embedded is dropped
    # rather than rendered as a dead, leaky link.
    f["render_before"] = _rv_render_uri(_rv_get(result, "render_before"))
    f["render_after"] = _rv_render_uri(_rv_get(result, "render_after"))
    f["render_support"] = _rv_render_uri(_rv_get(result, "render_support"))

    # orientation / supports
    spo = _rv_get(result, "support_pct_original")
    spn = _rv_get(result, "support_pct_oriented")
    if spn is None and _rv_num(orient.get("support_area_frac")) is not None:
        spn = float(orient["support_area_frac"]) * 100.0
    if spo is None and _rv_num(orient.get("support_area_frac_as_loaded")) is not None:
        spo = float(orient["support_area_frac_as_loaded"]) * 100.0
    f["support_pct_original"] = _rv_num(spo)
    f["support_pct_oriented"] = _rv_num(spn)
    oa = _rv_get(result, "orientation_applied")
    if oa is None:
        oa = (f["support_pct_original"] is not None
              and f["support_pct_oriented"] is not None)
    f["orientation_applied"] = bool(oa)
    f["support_zone_plain"] = _rv_get(result, "support_zone_plain")

    # thin walls
    f["thin_wall_min_mm"] = _rv_num(_rv_get(result, "thin_wall_min_mm",
                                            pr.get("min_wall_mm")))
    f["printable_min_mm"] = _rv_num(_rv_get(result, "printable_min_mm",
                                            result.get("nozzle_mm") or 0.4))
    sf = _rv_num(_rv_get(result, "scale_factor_suggested"))
    if f["thin_wall_min_mm"] is not None and f["thin_wall_min_mm"] <= 0:
        sf = None            # HARD RULE: never suggest scaling a zero wall
    f["scale_factor_suggested"] = sf
    f["part_name"] = _rv_get(result, "part_name")

    cc = _rv_get(result, "components_count")
    f["components_count"] = int(cc) if isinstance(cc, (int, float)) else None

    wt = _rv_get(result, "watertight_after")
    if wt is None:
        wt = dcert.get("watertight")
    if wt is None:
        wt = facts.get("watertight")
    f["watertight_after"] = bool(wt) if wt is not None else None
    nrm = _rv_get(result, "normals_consistent", facts.get("winding_consistent"))
    f["normals_consistent"] = bool(nrm) if nrm is not None else None

    # output file
    ofn = _rv_get(result, "output_filename")
    stl = files.get("fixed_stl") or files.get("prepped_stl")
    if not ofn and stl:
        ofn = os.path.basename(str(stl))
    f["output_filename"] = ofn
    f["output_format"] = _rv_get(result, "output_format", "binary STL")
    mb = _rv_num(_rv_get(result, "output_size_mb"))
    if mb is None and stl:
        try:
            mb = os.path.getsize(str(stl)) / (1024.0 * 1024.0)
        except OSError:
            mb = None
    f["output_size_mb"] = mb
    f["process_suggestion"] = _rv_get(result, "process_suggestion",
                                      "FDM, PLA, 0.4 mm nozzle")
    f["known_risks_shop_phrasing"] = _rv_get(result, "known_risks_shop_phrasing")

    # warp (opt-in; only spec-shaped input renders a result)
    w = _rv_get(result, "warp")
    if not isinstance(w, dict):
        w = {}
    f["warp"] = {
        "ran": bool(w.get("ran")),
        "hotspot_plain": w.get("hotspot_plain"),
        "ratio_vs_rest": _rv_num(w.get("ratio_vs_rest")),
        "suggested_shop_phrase": w.get("suggested_shop_phrase"),
    }
    if f["warp"]["ran"] and (f["warp"]["hotspot_plain"] is None
                             or f["warp"]["ratio_vs_rest"] is None):
        f["warp"]["ran"] = False       # incomplete data: treat as not run

    # trust map (opt-in; GEOMETRIC VISIBILITY vs the source camera). ABSENT
    # (ran=False) unless a camera was actually supplied -- never fabricated.
    tch = _rv_get(result, "trust")
    if not isinstance(tch, dict):
        tch = (result.get("channels") or {}).get("trust")
    if not isinstance(tch, dict):
        tch = {}
    f["trust"] = {
        "ran": bool(tch.get("ran", tch.get("ok"))),
        "frac_invented_pct": _rv_num(tch.get("frac_invented_pct")),
        "carry_mode": tch.get("carry_mode"),
        # never a server-local path on the page: sanitise exactly like the
        # before/after/support renders below (embed or drop, never a dead link)
        "render_trust": _rv_render_uri(tch.get("render_trust")
                                       or tch.get("render_path")),
        "fem_disclosed": bool(tch.get("fem_disclosed")),
    }
    if f["trust"]["ran"] and f["trust"]["frac_invented_pct"] is None:
        f["trust"]["ran"] = False      # incomplete data: treat as not run

    du = _rv_get(result, "decimation_used")
    f["decimation_used"] = bool(du)
    f["tri_count_original"] = _rv_get(result, "tri_count_original")
    f["tri_count_analysis"] = _rv_get(result, "tri_count_analysis")
    # only a real web URL may render as a hyperlink; a server-local path (or
    # bare filename) is mentioned in words instead — never a dead/leaky link
    turl = _rv_get(result, "technical_report_url", "report.md")
    f["technical_report_url"] = (turl if isinstance(turl, str)
                                 and turl.startswith(("http://", "https://"))
                                 else None)
    sw = _rv_get(result, "source_watertight")
    f["source_watertight"] = bool(sw) if sw is not None else None

    # --- trust-line DISPLAY strings (shared by the receipt and the shop echo).
    # Rounding may only DOWNGRADE the claim: if anything was repaired, the page
    # never shows "100% ... exactly as uploaded" or "0 mm" — a rounded-up 100%
    # next to "we sealed N gaps" reads as a contradiction (claim upgrade).
    kept, dev = f["surface_kept_pct"], f["max_deviation_mm"]
    hc0 = f["holes_filled_count"]
    changed = bool(f["fix_ran"] and ((isinstance(hc0, int) and hc0 > 0)
                                     or (dev is not None and dev > 0)
                                     or (kept is not None and kept < 100.0)))
    f["repairs_made"] = changed
    kept_s = _rv_fmt(kept, 1) if kept is not None else None
    if changed and kept_s == "100":
        kept_s = "99.9"                     # display floor, never a claim up
    dev_s = None
    if dev is not None:
        if changed and dev < 0.001:
            dev_s = "0.001"                 # ceiling display: 0 <= 0.001 holds
            f["deviation_anchor"] = "too small for us to measure"
        else:
            dev_s = _rv_fmt(math.ceil(float(dev) * 1000.0) / 1000.0, 3)
    f["kept_s"], f["dev_s"] = kept_s, dev_s

    # verdict (three states; blocking cards force not_ready; refusal wins)
    if result.get("rejected"):
        verdict = "unreadable"
    else:
        raw = str(_rv_get(result, "verdict", "")).strip()
        low = raw.lower()
        if low in ("ready", "not_ready", "unreadable"):
            verdict = low
        else:
            eff = str(fix.get("after_verdict") or gate.get("verdict")
                      or raw).upper()
            if eff in ("PASS", "WARN"):
                verdict = "ready"
            elif eff == "FAIL":
                verdict = "not_ready"
            else:
                verdict = "unreadable" if not facts else "not_ready"
    f["verdict"] = verdict

    f["warnings"] = _rv_warnings(result, f) if verdict != "unreadable" else []
    f["blocking_count"] = sum(1 for c in f["warnings"]
                              if c["severity"] == "blocking")
    if f["verdict"] == "ready" and f["blocking_count"] > 0:
        f["verdict"] = "not_ready"     # never a green banner over a blocker

    # top issue: the CLASSIFIED blocking card is authoritative (already plain,
    # numbers appear exactly once more in its own card); a caller-supplied
    # string is a scrubbed fallback only.
    blk = next((c for c in f["warnings"] if c["severity"] == "blocking"), None)
    top = blk["fact_plain"] if blk else None
    if not top:
        raw_top = _rv_get(result, "top_issue_plain")
        top = _rv_plain_text(raw_top) if raw_top else None
    f["top_issue_plain"] = top or ("One part of your model needs a decision "
                                   "before printing.")
    # residual restatements are NUMBERLESS one-liners so the same decimal
    # never repeats across the banner, the receipt and the download button
    short = (blk or {}).get("short_plain")
    if not short:
        raw_res = str(_rv_get(result, "residual_issue_plain") or "")
        raw_res = re.sub(r"^\s*even after the automatic fix:\s*", "",
                         raw_res, flags=re.I)
        short = re.sub(r"\s*\d+(\.\d+)?\s*(mm|%|×|x)?\s*", " ",
                       _rv_plain_text(raw_res)).strip(" .,;—-") or None
    f["residual_short"] = short or "one problem that needs your decision"
    f["residual_issue_plain"] = f["residual_short"]
    return f


# --- section renderers --------------------------------------------------------
def _rv_banner(f):
    v = f["verdict"]
    if v == "ready":
        hc = f.get("holes_filled_count")
        mid = (f"The fix sealed {hc} small gap{'s' if hc != 1 else ''}; "
               "everything else is exactly as you uploaded it."
               if isinstance(hc, int) and hc > 0 else
               "Nothing needed to change — your file was already "
               "sound.")
        return ["> ✅ **READY TO PRINT** — Good news: your model is ready.",
                f"> {mid}",
                "> Your print-ready file is at the bottom of this page.", ""]
    if v == "not_ready":
        n = f["blocking_count"]
        need = ("one thing needs your decision." if n <= 1
                else f"{n} things need your decision.")
        did = ("Everything that could safely be repaired was fixed — the "
               "receipt is in the next section — but this one is a choice "
               "only you can make. See item 1 below for your options."
               if f["fix_ran"] else
               "Everything was checked and nothing needed changing — this "
               "one is a choice only you can make. See item 1 below for "
               "your options.")
        return [f"> ⚠️ **NOT READY YET** — {need}",
                f"> {f['top_issue_plain']}",
                f"> {did}", ""]
    return ["> ❌ **THIS FILE COULDN'T BE READ** — nothing was changed, and "
            "there is no fixed file to download.",
            "> This usually means the export went wrong, not that your model "
            "is bad.",
            "> One thing to try: re-export from Meshy as .glb or .stl (the "
            "file formats print shops use) and upload it again.", ""]


def _rv_changed(f):
    lines = ["## What changed — and what didn't", ""]
    if f["fix_ran"] and f.get("kept_s") is not None \
            and f.get("dev_s") is not None and f.get("repairs_made"):
        kept_s, dev_s = f["kept_s"], f["dev_s"]
        lines.append(
            f"**{kept_s}% of your surface is exactly as you uploaded it.** "
            "Where repairs were made, the new surface is never more than "
            f"**{dev_s} mm** from your original — {f['deviation_anchor']}.")
        lines.append("")
        hc = f.get("holes_filled_count")
        if isinstance(hc, int) and hc > 0:
            one = hc == 1
            gap_line = (f"Your model had {hc} small gap{'' if one else 's'} "
                        "in its surface — like "
                        + ("a pinhole" if one else "pinholes")
                        + " in a balloon. Printers need a fully sealed "
                          "shape, so "
                        + ("it was sealed." if one else "they were sealed."))
            if f.get("source_watertight") is True:
                gap_line += (" (The gaps appeared in the simplified working "
                             "copy that gets analysed — your original file "
                             "was already fully sealed.)")
            lines.append(gap_line)
        lines.append("What did **not** happen: no smoothing, no reshaping, "
                     "no detail reduction, no rescaling beyond the size you "
                     "chose. The overall shape and details are unchanged — "
                     "this is verified by measuring, not by eye.")
        lines.append("")
    else:
        tail = (" — it was already fully sealed."
                if f.get("watertight_after") is not False else ".")
        lines.append("Nothing was changed: **100% of your "
                     "surface is exactly as you uploaded it (0 mm "
                     "deviation)**" + tail)
        lines.append("")
    if f["cavities_count"] > 0:
        n = f["cavities_count"]
        many = n > 1
        sav = ""
        if f["material_savings_pct"] is not None:
            sav = (" and uses roughly "
                   f"{_rv_fmt(f['material_savings_pct'], 0)}% of the material "
                   "a fill-everything repair would (a geometric estimate, "
                   "not a print-shop quote)")
        lines.append(
            f"Your model has **{n} hollow space{'s' if many else ''} inside, "
            f"kept hollow instead of filled solid** — that "
            f"keeps the print faithful to your design{sav}.")
        lines.append("")
    if f["fix_ran"] and f["verdict"] == "not_ready":
        did = ("the repair sealed the gaps but" if f.get("repairs_made")
               else "the automatic check ran, but it")
        lines.append(
            f"One honest note: {did} did **NOT** fix everything: "
            f"{f['residual_short']}. That needs a decision from you "
            "(see item 1 below) — your model's shape is not changed "
            "automatically.")
        lines.append("")
    lines.append("Your original upload is never modified; everything "
                 "produced is a new file.")
    lines.append("")
    return lines


def _rv_size(f):
    if not f["extents_mm"] or f["longest_dim_mm"] is None:
        return []
    x_s, y_s, z_s = (_rv_fmt(v, 1) for v in f["extents_mm"])
    L_s = _rv_fmt(f["longest_dim_mm"], 1)
    lines = ["## Size check — is this the size you meant?", "",
             f"Your model will print **{L_s} mm** at its longest — about "
             f"{f['size_anchor']}.",
             f"Full size: **{x_s} × {y_s} × {z_s} mm** (all measurements in "
             "millimetres).", ""]
    if f["units_source"] == "assumed_mm":
        lines += ["Heads-up: your file didn't say what units it uses (most "
                  "AI-generated files don't) — assumed millimetres, the "
                  "safe, standard guess.", ""]
    elif f["units_source"] == "user_target":
        lines += ["It was scaled so its longest side matches the size you "
                  "picked on the upload screen.", ""]
    if f["flat_object"] and f["min_extent_mm"] is not None:
        lines += [f"Note the {_rv_fmt(f['min_extent_mm'], 1)} mm thickness: "
                  "this model is nearly flat, like a cardboard cutout. If you "
                  "expected a full 3D object, look at the pictures below "
                  "before paying for a print.", ""]
    lines += [f"If \"{f['size_anchor']}\" sounds wrong, don't worry — nothing "
              "is broken, and every warning below depends on the size, so fix "
              "this FIRST. One thing to do: tell the print shop the real size "
              "you want (for example \"make it 15 cm tall\") — resizing is "
              "normal, safe, and takes them seconds.", ""]
    return lines


def _rv_pictures(f):
    rb, ra, rs = f["render_before"], f["render_after"], f["render_support"]
    lines = ["## See it — before and after (pictures)", ""]
    if rb and ra:
        imgs = f"![before]({rb}) ![after]({ra})"
        if rs:
            imgs += f" ![support zones]({rs})"
        lines += [imgs, ""]
        pos = (", shown in the printing position already applied"
               if f["orientation_applied"] else "")
        ident = ("They should look identical — the repairs are smaller than "
                 "your screen can show (the receipt above is the measurement "
                 "behind that)." if f.get("repairs_made") else
                 "They should look identical — nothing was changed.")
        lines += [f"**Left:** exactly what you uploaded. **Right:** the file "
                  f"you'll print{pos}. {ident}", ""]
        if rs:
            lines += ["**Third picture:** the red areas are where the "
                      "printer will build temporary scaffolding to hold up "
                      "steep parts while printing. The scaffolding is "
                      "removed afterwards but can leave small rough patches "
                      "in those red spots — don't expect them to be "
                      "glass-smooth.", ""]
    else:
        # never a broken image or a server path: say plainly there are none
        lines += ["Pictures couldn't be produced for this file — every "
                  "measurement above still stands; it's only the preview "
                  "images that are missing.", ""]
    if f["orientation_applied"] and f["support_pct_original"] is not None \
            and f["support_pct_oriented"] is not None:
        lines += ["The model was already rotated to the position that needs "
                  "the least scaffolding — from "
                  f"{_rv_fmt(f['support_pct_original'], 0)}% of the surface "
                  f"down to {_rv_fmt(f['support_pct_oriented'], 0)}% "
                  "(estimated with the standard 45° steepness rule of thumb, "
                  "not a run through real printer software). That rotation is "
                  "saved into your download.", ""]
    return lines


_RV_SEV_TAG = {"blocking": "(needs your decision)", "check": "(please check)",
               "info": "(good to know)"}


def _rv_warn_section(f):
    lines = ["## Things to know before you print (most important first)", ""]
    if not f["warnings"]:
        lines += ["Good news: nothing you need to worry about — no "
                  "problems worth flagging.", ""]
        return lines
    for i, c in enumerate(f["warnings"], 1):
        tag = _RV_SEV_TAG.get(c["severity"], "(good to know)")
        lines.append(f"**{i}. {tag} {c['fact_plain']}**")
        lines.append(f"*What this means for your print:* "
                     f"{c['consequence_plain']}")
        lines.append(f"*Do one thing:* {c['action_plain']}")
        lines.append("")
    return lines


def _rv_shop(f):
    lines = ["## Your file + a note for the print shop", ""]
    name = f["output_filename"] or "your file"
    mb_s = (f" ({_rv_fmt(f['output_size_mb'], 1)} MB)"
            if f["output_size_mb"] is not None else "")
    label = ("print-ready file" if f["verdict"] == "ready"
             else "repaired file")
    lines += [f"**[ ⬇ Download your {label} — {name}{mb_s} ]**", ""]
    if f["verdict"] == "not_ready":
        lines += ["Because the verdict above is \"Not ready yet\", the shop "
                  "will hit the same problem found "
                  f"({f['residual_issue_plain']}) — settle item 1 first, or "
                  "expect the shop to call you.", ""]
    lines += ["Send this note along with the file — copy-paste it into the "
              "order form; you don't need to understand it, the shop does:",
              "", "```"]
    lines.append(f"File: {name} — {f['output_format']}, units: millimetres")
    if f["extents_mm"]:
        x_s, y_s, z_s = (_rv_fmt(v, 1) for v in f["extents_mm"])
        lines.append(f"Dimensions: {x_s} × {y_s} × {z_s} mm. Scale is "
                     "intentional — please print at this size, do not "
                     "rescale.")
    if f["watertight_after"] is not None:
        wt = ("YES — single manifold solid, verified after repair"
              if f["watertight_after"] else "NO — see known issues")
        nrm = ("yes" if f["normals_consistent"] else "no") \
            if f["normals_consistent"] is not None else "unknown"
        lines.append(f"Watertight: {wt}. Consistent normals: {nrm}.")
    if f["fix_ran"] and f.get("kept_s") is not None \
            and f.get("dev_s") is not None and f.get("repairs_made"):
        kept_s, dev_s = f["kept_s"], f["dev_s"]
        try:
            mod_s = _rv_fmt(max(0.0, 100.0 - float(kept_s)), 1)
        except ValueError:
            mod_s = "--"
        hc = f.get("holes_filled_count")
        if isinstance(hc, int):
            hole_s = (f"{hc} boundary hole{'' if hc == 1 else 's'} filled"
                      if hc > 0 else "no boundary holes to fill")
            if hc > 0 and f.get("source_watertight") is True:
                hole_s += (" (source file was watertight; holes arose in "
                           "the simplified analysis copy)")
        else:
            hole_s = "boundary holes filled"
        lines.append(f"Repairs performed: {hole_s}; {kept_s}% of surface "
                     f"unmodified ({mod_s}% modified), max deviation {dev_s} "
                     "mm vs. source (two-sided Hausdorff); original detail "
                     "otherwise untouched.")
    else:
        lines.append("Repairs performed: none — 100% of surface unmodified "
                     "(0% modified), max deviation 0 mm vs. source "
                     "(two-sided Hausdorff).")
    if f["cavities_count"] > 0:
        nc = f["cavities_count"]
        voids = ("sealed internal voids are" if nc > 1
                 else "sealed internal void is")
        lines.append(f"Internal cavities: {nc} {voids} INTENTIONAL — please "
                     "do NOT run a make-solid / cavity-fill auto-repair.")
    if f["orientation_applied"]:
        lines.append("Orientation: file is saved in a support-minimising "
                     "orientation — please print as-oriented.")
    lines.append(f"Suggested process: {f['process_suggestion']}")
    if f["known_risks_shop_phrasing"]:
        lines.append(f"Known risks: {f['known_risks_shop_phrasing']}")
    lines.append("Prepared by printworthy — checks are geometric (45° overhang "
                 "rule, direct thickness measurement), not a slicer "
                 "simulation.")
    lines += ["```", "**[ Copy to clipboard ]**", ""]
    return lines


def _rv_warp(f):
    lines = ["## Optional: will it warp? (takes about a minute)", ""]
    w = f["warp"]
    if not w["ran"]:
        lines += ["**[ Run warp check — about 1 minute ]**",
                  "This runs a rough physics simulation of your print "
                  "cooling down. What you'll get back is a **comparison, not "
                  "a promise**: it ranks which parts of your model are more "
                  "likely to curl or lift than others. It cannot predict "
                  "exact millimetres — the numbers are uncalibrated "
                  "estimates.", ""]
        return lines
    ratio = w["ratio_vs_rest"]
    ratio_s = _rv_fmt(ratio, 1)
    # the ratio compares the predicted lift to the ATTENTION THRESHOLD (the
    # level at which flagging starts) — say exactly that; "0.1× more likely
    # than the rest" was both confusing and wrong.
    if isinstance(ratio, (int, float)) and ratio < 1.0:
        lines += ["**Warp check result (uncalibrated estimate):** good news "
                  "— the tendency to curl or lift off the print bed came in "
                  "below the flagging threshold, including at "
                  f"{w['hotspot_plain']}; that is an uncalibrated estimate — "
                  "a comparison, not a promise.", ""]
    else:
        lines += [f"**Warp check result (uncalibrated estimate):** "
                  f"{w['hotspot_plain']} tends to curl and lift off the "
                  f"print bed — about **{ratio_s}× the flagging threshold** "
                  "— an uncalibrated estimate, useful for "
                  "spotting the risky area, not for predicting exact "
                  "millimetres; trust the comparison, not the raw number.",
                  ""]
    if w["suggested_shop_phrase"]:
        lines += ["If it matters, the one thing to do is tell the shop: "
                  f"\"{w['suggested_shop_phrase']}\" — they'll know what "
                  "that means.", ""]
    return lines


def _rv_trust(f):
    """Trust map (opt-in; requires the source camera). ABSENT (no section at
    all, not even a call-to-action) unless a camera was actually supplied —
    unlike the warp button, this needs data a novice cannot click into
    existence, so nothing here is ever a fake or an invitation to fake it."""
    t = f.get("trust") or {}
    if not t.get("ran"):
        return []
    pct = t.get("frac_invented_pct")
    pct_s = _rv_fmt(pct, 0) if isinstance(pct, (int, float)) else "some"
    lines = ["## Optional: what the source photo actually saw", "",
             f"Your file was compared against the photo it was generated "
             f"from. About **{pct_s}%** of the surface (by area) — mostly "
             "the back and the inside — was never in that photo; the AI "
             "invented it to make the shape whole and printable. Invented "
             "areas can't be checked against anything real, so treat them "
             "as a careful guess, not a fact.", ""]
    lines += ["For a decorative print, that's usually fine — nobody sees "
              "the inside. For a functional part, a replica, or anything "
              "you plan to measure or depend on, look at the invented "
              "(orange) areas in the picture below before you rely on "
              "them.", ""]
    img = t.get("render_trust")
    if img:
        lines += [f"![trust map]({img})", ""]
    if t.get("fem_disclosed"):
        lines += ["Heads-up: the strength estimate elsewhere in this report "
                  "also covers some invented area — it is not physically "
                  "grounded there; treat it as indicative only.", ""]
    return lines


def _rv_fine_print(f):
    lines = ["<details>",
             "<summary><strong>The fine print</strong> (tap to expand)"
             "</summary>", ""]
    if f["verdict"] == "unreadable":
        lines += ["**How honest are these numbers?** There are none — the "
                  "file couldn't be read, so nothing was changed and "
                  "nothing was measured.", ""]
    else:
        lines.append("**How honest are these numbers?**")
        if f["fix_ran"]:
            lines.append("**Measured:** the receipt in \"What changed\" — "
                         "the percentage of your surface that was kept and "
                         "the maximum deviation — is computed directly on your "
                         "repaired file against your original — it is not an "
                         "estimate.")
        hollow = (" The hollow-space material figure is a rough geometric "
                  "estimate." if f["cavities_count"] > 0 else "")
        lines.append("**Estimates:** wall thickness, scaffolding coverage, "
                     "and the printing angle come from standard geometry "
                     "rules of thumb (the 45° steepness rule, direct "
                     "thickness measurement), not from a run through real "
                     "printer software — your shop's slicer software (the "
                     "program that prepares files for their printer) has the "
                     f"final word.{hollow}")
        lines.append("**Uncalibrated:** the optional warp number — trust its "
                     "ranking of risky spots, not its exact values.")
        if f["units_source"] == "assumed_mm":
            lines.append("The size check assumed millimetres because your "
                         "file didn't specify.")
        if f["decimation_used"] and f["tri_count_original"] \
                and f["tri_count_analysis"]:
            lines.append("For speed, a lightly simplified copy of your model "
                         f"was analysed ({f['tri_count_original']:,} → "
                         f"{f['tri_count_analysis']:,} facets); the file you "
                         "download is repaired at full quality.")
        lines.append("")
    lines += ["Nothing on this page is a guarantee your print succeeds — it "
              "is the honest best reading of your file, and every place it "
              "still needs a decision is stated plainly."]
    # the full technical report: a real web URL renders as a link; otherwise
    # it is named in words (report.md rides in the downloads) — NEVER a dead
    # or server-local link, and never offered at all when nothing was read.
    if f["verdict"] != "unreadable":
        url = f.get("technical_report_url")
        if url:
            lines.append("Want every raw number, timing, and measurement "
                         "method? **[ Download the full technical report ]"
                         f"({url})**")
        else:
            lines.append("Want every raw number, timing, and measurement "
                         "method? The full technical report (report.md) is "
                         "included with your downloads.")
    lines += ["", "</details>", ""]
    return lines


def build_review(result, audience="novice") -> str:
    """PrepResult (or flat spec-shaped dict) -> the Space review page
    (markdown, UTF-8). Never raises. `audience` is fixed to 'novice' for now
    (reserved for a future expert view); any value renders the novice page."""
    try:
        f = _rv_fields(result)
        lines = _rv_banner(f)
        lines.append("---")
        lines.append("")
        if f["verdict"] != "unreadable":
            for sec in (_rv_changed, _rv_size, _rv_pictures, _rv_warn_section,
                        _rv_shop, _rv_warp, _rv_trust):
                try:
                    lines += sec(f)
                except Exception as e:      # a broken section degrades quietly
                    lines += [f"_(section unavailable: {type(e).__name__})_",
                              ""]
        lines += _rv_fine_print(f)
        return "\n".join(lines).rstrip() + "\n"
    except Exception as e:
        return ("> ❌ **THIS FILE COULDN'T BE READ** — nothing was changed, "
                "and there is no fixed file to download.\n> This usually "
                "means the export went wrong, not that your model is bad.\n"
                "> One thing to try: re-export from Meshy as .glb or .stl "
                "(the file formats print shops use) and upload it again.\n\n"
                f"<!-- review rendering failed: {type(e).__name__}: {e} -->\n")


# --- mechanical acceptance-criteria checker ------------------------------------
_RV_VERDICT_PHRASES = ("READY TO PRINT", "NOT READY YET",
                       "THIS FILE COULDN'T BE READ")

_RV_HEADER_ORDER = ["## What changed", "## Size check", "## See it",
                    "## Things to know before you print",
                    "## Your file + a note for the print shop",
                    "## Optional: will it warp", "The fine print"]

# banned outside the shop block (case-insensitive unless noted)
_RV_BANNED = [
    (r"\bmanifold\b", "manifold"), (r"\bwatertight\b", "watertight"),
    (r"\bgenus\b", "genus"), (r"topolog", "topology"),
    (r"hausdorff", "Hausdorff"), (r"\bnormals\b", "normals"),
    (r"\bmesh(es)?\b", "mesh"), (r"\bdecimat", "decimation"),
    (r"\bvoxel", "voxel"), (r"\bboolean\b", "boolean"),
    (r"\bremesh|\bretopo", "remesh/retopo"),
    (r"self.intersect", "self-intersection"), (r"\bdegenerate", "degenerate"),
    (r"tessellat|triangulat|\btriangles?\b|\bvertex\b|\bvertices\b",
     "tessellation/triangle/vertex"),
    (r"bounding box|\bbbox\b", "bounding box"),
    (r"extrusion", "extrusion"), (r"\binfill\b", "infill"),
    (r"\boverhang", "overhang"), (r"\bheuristic|shape.diameter|\bsdf\b",
                                  "heuristic/SDF"),
    (r"\bprovenance\b", "provenance"),
    (r"\bguaranteed\b|\bcertified\b", "guaranteed/certified"),
    (r"\bfoolproof\b|\boptimal\b|\bperfect\b", "foolproof/optimal/perfect"),
    (r"blocking issue", "blocking issue(s)"),
    (r"\bpipeline\b|\bstage\b|\btriage\b|\bre-check\b",
     "pipeline/stage/re-check/triage"),
]
_RV_BANNED_CS = [           # case-sensitive tokens
    (r"\bFDM\b|\bSLA\b|\bPLA\b", "FDM/SLA/PLA"),
    (r"\bFAIL\b|\bPASS\b", "FAIL/PASS verdict token"),
]


def _rv_find_shop_span(md):
    for m in re.finditer(r"```(.*?)```", md, re.S):
        if "units: millimetres" in m.group(1):
            return m.span()
    return None


def _rv_find_region(md, start_marker, end_markers):
    i = md.find(start_marker)
    if i < 0:
        return None
    end = len(md)
    for em in end_markers:
        j = md.find(em, i + len(start_marker))
        if 0 <= j < end:
            end = j
    return (i, end)


def review_selfcheck(md) -> list:
    """Mechanically test the review page against the spec's acceptance
    criteria. Returns a list of violation strings (empty = clean).
    Never raises."""
    v: list[str] = []
    try:
        if not isinstance(md, str) or not md.strip():
            return ["empty review document"]

        # embedded images are legitimate content, not text: mask their base64
        # payloads before every textual scan (a random payload can otherwise
        # spell a banned word or a fake number). The ![tag](...) structure
        # survives, so the picture checks still see the images.
        md = re.sub(r"\(data:image/[a-z]+;base64,[^)]*\)",
                    "(embedded-image)", md)

        # -- 1. banner first; exactly one verdict phrase ----------------------
        first = next((ln for ln in md.splitlines() if ln.strip()), "")
        if not first.startswith("> ") or not any(p in first
                                                 for p in _RV_VERDICT_PHRASES):
            v.append("banner is not the first rendered element")
        hits = [(p, md.count(p)) for p in _RV_VERDICT_PHRASES if p in md]
        if len(hits) != 1:
            v.append("page must contain exactly one verdict phrase "
                     f"(found {[p for p, _ in hits]})")
        elif hits[0][1] != 1:
            v.append(f"verdict phrase '{hits[0][0]}' appears {hits[0][1]} "
                     "times (must be once)")
        unreadable = "THIS FILE COULDN'T BE READ" in md
        not_ready = "NOT READY YET" in md

        shop_span = _rv_find_shop_span(md)
        shop_txt = md[shop_span[0]:shop_span[1]] if shop_span else ""
        masked = (md[:shop_span[0]] + md[shop_span[1]:]) if shop_span else md
        warp_reg = _rv_find_region(md, "## Optional: will it warp",
                                   ["<details"])
        warp_txt = md[warp_reg[0]:warp_reg[1]] if warp_reg else ""

        # -- 2. unreadable: refusal is a refusal ------------------------------
        if unreadable:
            if "nothing was changed" not in md:
                v.append("unreadable banner must say 'nothing was changed'")
            if md.count("One thing to try:") != 1:
                v.append("unreadable page must offer exactly one recovery "
                         "action")
            if "⬇" in md or shop_span:
                v.append("unreadable page must not offer a download or shop "
                         "block")
            for h in _RV_HEADER_ORDER[:-1]:
                if h in md:
                    v.append(f"unreadable page must not render '{h}'")
        else:
            # -- 3. section headers present, once each, in order --------------
            pos = -1
            for h in _RV_HEADER_ORDER:
                c = md.count(h)
                if c == 0:
                    v.append(f"missing section header '{h}'")
                    continue
                if c > 1:
                    v.append(f"section header '{h}' appears {c} times")
                p = md.find(h)
                if p < pos:
                    v.append(f"section '{h}' out of spec order")
                pos = p

            # -- 4. trust line + shop echo -------------------------------------
            kept_s = dev_s = None
            m = re.search(r"\*\*([\d.]+)% of your surface is exactly as you "
                          r"uploaded it\.\*\*.{0,200}?never more than "
                          r"\*\*([\d.]+) mm\*\* from your original — (.{3,})",
                          md, re.S)
            if m:
                kept_s, dev_s = m.group(1), m.group(2)
            elif "100% of your surface is exactly as you uploaded it (0 mm " \
                 "deviation)" in md:
                kept_s, dev_s = "100", "0"
            else:
                v.append("Section 2 trust line missing (neither repair form "
                         "nor 100%/0 mm form found)")
            if not shop_span:
                v.append("shop block missing (no fenced block with "
                         "'units: millimetres')")
            if kept_s and shop_txt:
                rep = next((ln for ln in shop_txt.splitlines()
                            if ln.startswith("Repairs performed:")), "")
                if not rep:
                    v.append("shop block missing 'Repairs performed:' line")
                else:
                    if f"{kept_s}% of surface" not in rep:
                        v.append("shop repairs line does not echo the trust "
                                 f"line's kept % ({kept_s}%)")
                    if f"{dev_s} mm" not in rep:
                        v.append("shop repairs line does not echo the trust "
                                 f"line's max deviation ({dev_s} mm)")

            # -- 5. residual-failure discipline --------------------------------
            fix_ran = m is not None
            if not_ready:
                if "✅" in md or "✔" in md:
                    v.append("success glyph on a not-ready page")
                if fix_ran and "did **NOT** fix" not in md:
                    v.append("fix ran + not ready: mandatory residual-failure "
                             "sentence missing in Section 2")
                dl_reg = _rv_find_region(
                    md, "## Your file + a note for the print shop",
                    ["## Optional: will it warp"])
                if dl_reg and "settle item 1 first" not in \
                        md[dl_reg[0]:dl_reg[1]]:
                    v.append("not ready: residual issue not restated beside "
                             "the download button")
                if re.search(r"print-ready", masked, re.I):
                    v.append("'print-ready' used on a non-ready page")

            # -- 6. cavities: kept + INTENTIONAL travel together ---------------
            has_hollow = "hollow space" in masked
            has_intent = "INTENTIONAL" in shop_txt
            if has_hollow != has_intent:
                v.append("cavity statements inconsistent (Section 2 'hollow "
                         "space' and shop 'INTENTIONAL' must appear "
                         "together)")
            if has_intent and "do NOT run a make-solid / cavity-fill" \
                    not in shop_txt:
                v.append("shop block cavity line missing the do-NOT-auto-fill "
                         "instruction")

            # -- 6b. trust map: header + disclosure prose travel together -----
            # (mirrors the cavities check above -- a partially-rendered
            # section is worse than none: it must be all-or-nothing)
            has_trust_header = ("## Optional: what the source photo actually "
                                "saw" in md)
            has_trust_prose = "was never in that photo" in masked
            if has_trust_header != has_trust_prose:
                v.append("trust section header and its disclosure prose "
                         "must appear together")
            if has_trust_header and not re.search(
                    r"About \*\*[\d.]+%\*\* of the surface", masked):
                v.append("trust section missing its invented-percentage "
                         "figure")

            # -- 7. material savings always labelled ---------------------------
            for para in masked.split("\n\n"):
                if "% of the material" in para and \
                        "(a geometric estimate" not in para:
                    v.append("material-savings % rendered without its "
                             "'(a geometric estimate' label")

            # -- 8/9. size section ----------------------------------------------
            sz = _rv_find_region(md, "## Size check", ["## See it"])
            if sz:
                s = md[sz[0]:sz[1]]
                if " mm** at its longest — about " not in s:
                    v.append("size section missing longest-dimension anchor "
                             "sentence")
                if not re.search(r"\*\*[\d.]+ × [\d.]+ × [\d.]+ mm\*\*", s):
                    v.append("size section missing the three explicit mm "
                             "dimensions")
                if "(all measurements in millimetres)" not in s:
                    v.append("size section missing explicit millimetre units")
                if s.count("One thing to do:") != 1:
                    v.append("size section must end with exactly one 'One "
                             "thing to do:' action")
                if "didn't say what units" in s and \
                        "assumed millimetres" not in s:
                    v.append("assumed-units disclosure incomplete")

            # -- 10. renders ------------------------------------------------------
            pic = _rv_find_region(md, "## See it",
                                  ["## Things to know before you print"])
            if pic:
                p = md[pic[0]:pic[1]]
                has_imgs = "![" in p
                if has_imgs:
                    for tagname in ("![before](", "![after]("):
                        if tagname not in p:
                            v.append(f"pictures section missing {tagname}...)"
                                     " render")
                    has_support = ("![support zones](" in p
                                   or "![support](" in p)
                    if has_support:
                        if "temporary scaffolding" not in p:
                            v.append("support-zone caption must gloss "
                                     "supports as 'temporary scaffolding'")
                        if "rough patches" not in p:
                            v.append("support-zone caption must warn about "
                                     "'rough patches'")
                elif "couldn't produce pictures" not in p:
                    v.append("pictures section has neither renders nor the "
                             "honest no-pictures note")
                if "down to" in p and not ("rule of thumb" in p
                                           and "real printer software" in p):
                    v.append("orientation support pair missing its 45°-rule "
                             "estimate label")

            # -- 11-15. warning cards ---------------------------------------------
            wr = _rv_find_region(md, "## Things to know before you print",
                                 ["## Your file + a note for the print shop"])
            if wr:
                w = md[wr[0]:wr[1]]
                facts_ln = re.findall(
                    r"^\*\*\d+\. \((needs your decision|please check|"
                    r"good to know)\) (.+)$", w, re.M)
                n_mean = w.count("*What this means for your print:*")
                n_act = w.count("*Do one thing:*")
                if not (len(facts_ln) == n_mean == n_act):
                    v.append(f"warning cards malformed: {len(facts_ln)} fact "
                             f"lines, {n_mean} consequence lines, {n_act} "
                             "action lines")
                rank = {"needs your decision": 0, "please check": 1,
                        "good to know": 2}
                seq = [rank[t] for t, _ in facts_ln]
                if seq != sorted(seq):
                    v.append("warning cards not ordered blocking → check → "
                             "info")
                # zero-thickness: no scaling suggestion in ITS action
                blocks = re.split(r"\n(?=\*\*\d+\. \()", w)
                for b in blocks:
                    if "zero thickness" in b:
                        act = re.search(r"\*Do one thing:\* (.+)", b)
                        if act and re.search(r"bigger|×|\d+\s*%|scale",
                                             act.group(1), re.I):
                            v.append("zero-thickness card suggests scaling "
                                     "(hard rule violation)")
                sup_hits = re.findall(r"[\d.]+% of the surface", w)
                if len(sup_hits) > 1:
                    v.append("more than one supports percentage in the "
                             "warnings section")

            # -- 16. orientation strings paired -------------------------------------
            saved = "saved into your download" in md
            as_or = "please print as-oriented" in shop_txt
            if saved != as_or:
                v.append("orientation strings must appear together ('saved "
                         "into your download' + shop 'print as-oriented')")

            # -- 17. shop block minimum content --------------------------------------
            if shop_txt:
                for req in ("units: millimetres", "do not rescale",
                            "Watertight: ", "not a slicer simulation"):
                    if req not in shop_txt:
                        v.append(f"shop block missing required '{req}'")
                if "[ Copy to clipboard ]" not in md:
                    v.append("shop block missing the copy button")

            # -- 18. warp discipline ---------------------------------------------------
            if warp_txt:
                ran = "Warp check result" in warp_txt
                btn = "Run warp check" in warp_txt
                if ran and btn:
                    v.append("warp section shows both the button and a "
                             "result")
                if ran:
                    for sent in re.split(r"(?<=[.!?])\s+",
                                         warp_txt.replace("\n", " ")):
                        if re.search(r"[\d.]+×", sent) and \
                                "uncalibrated" not in sent:
                            v.append("warp ratio not in the same sentence as "
                                     "'uncalibrated'")
                if btn and "a **comparison, not a promise**" not in warp_txt \
                        and "a comparison, not a promise" not in warp_txt:
                    v.append("warp pre-click framing missing 'a comparison, "
                             "not a promise'")

        # -- 19/20. fine print ------------------------------------------------------
        if "<details" not in md:
            v.append("fine print must be a collapsed <details> block")
        if "Nothing on this page is a guarantee your print succeeds" not in md:
            v.append("missing the no-guarantee line")
        # a dead or placeholder link is worse than no link, on ANY page
        if re.search(r"\]\(\s*(None|null|)\s*\)", md):
            v.append("dead link on the page (empty/None target)")
        if unreadable:
            if "full technical report" in md:
                v.append("unreadable page must not offer a technical report "
                         "(nothing was measured)")
        elif "full technical report" not in md:
            v.append("missing the technical-report link/mention")
        fp = _rv_find_region(md, "<details", ["</details>"])
        if fp:
            fpt = md[fp[0]:fp[1]]
            if "facets" in fpt and "repaired at full quality" not in fpt:
                v.append("decimation disclosure missing 'repaired at full "
                         "quality'")

        # -- 21. banned jargon outside its allowed context ----------------------------
        scan = masked
        for pat, name in _RV_BANNED:
            if re.search(pat, scan, re.I):
                v.append(f"banned word outside shop block: {name}")
        for pat, name in _RV_BANNED_CS:
            if re.search(pat, scan):
                v.append(f"banned token outside shop block: {name}")
        no_warp = scan.replace(warp_txt, "") if warp_txt else scan
        if re.search(r"\bbrim\b", no_warp, re.I):
            v.append("'brim' outside the warp result")
        if re.search(r"\bsimulation\b", no_warp, re.I):
            v.append("'simulation' used for a non-warp estimate")
        for sm in re.finditer(r"\bslicer\b", scan, re.I):
            tail = scan[sm.end():sm.end() + 60]
            if not tail.startswith(" software (the program"):
                v.append("bare 'slicer' outside the shop block")
                break

        # -- 22. no paths / CLI / telemetry --------------------------------------------
        if re.search(r"[A-Za-z]:[\\/][^\s)]+|/tmp/|/home/", md):
            v.append("absolute or temp file path on the page")
        if re.search(r"(?<![\w—-])--[a-z]\w+", md):
            v.append("CLI flag on the page")

        # -- 23. numeric duplication (decimals; dims allowed 3×, rest 2×) --------------
        dims_ok = set()
        dm = re.search(r"\*\*([\d.]+) × ([\d.]+) × ([\d.]+) mm\*\*", masked)
        if dm:
            dims_ok = set(dm.groups())
        lm = re.search(r"\*\*([\d.]+) mm\*\* at its longest", masked)
        if lm:
            dims_ok.add(lm.group(1))
        counts: dict = {}
        for tok in re.findall(r"\d+\.\d+", masked):
            counts[tok] = counts.get(tok, 0) + 1
        for tok, c in counts.items():
            limit = 3 if tok in dims_ok else 2
            if c > limit:
                v.append(f"numeric value {tok} appears {c} times (limit "
                         f"{limit})")
    except Exception as e:
        v.append(f"selfcheck internal error: {type(e).__name__}: {e}")
    return v
