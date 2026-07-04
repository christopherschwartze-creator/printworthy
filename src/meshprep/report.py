"""meshprep.report — the ONE plain-English report renderer (markdown + json).

`build_report(result)` turns a PrepResult (see meshprep.pipeline) into
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
                "We couldn't open this as a 3D model -- check it's a real "
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
    lines = [f"# meshprep report -- {head}", ""]
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
                        _resin_section, _savings_section, _provenance,
                        _files_and_notes):
            try:
                lines += section(result)
            except Exception as e:
                lines += [f"_({section.__name__} unavailable: "
                          f"{type(e).__name__})_", ""]
        md = "\n".join(lines).rstrip() + "\n"
    except Exception as e:
        md = f"# meshprep report\n\nreport rendering failed: {e}\n"
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
