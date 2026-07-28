"""printworthy Gradio front door — the Hugging Face Space slice.

One drop-zone for a complete novice: drop a mesh -> plain-words verdict badge
(with the surface-kept trust line + a size sanity line) -> the review document
-> risk renders -> download prep.stl + review.md.  Opt-in warp physics
(~1 min, estimate, uncalibrated).

Deliberately EXCLUDED from this UI (the code paths stay in the package, use
the CLI / API for them): reinforce / graded-infill 3MF, quad retopo, slicer
savings, resin mode, autorig.

Gradio is an optional extra (``pip install gradio``, or the ``[app]`` extra
if printworthy is installed from PyPI); this module imports clean without it and
:func:`build_demo` raises clear guidance.

Run:  printworthy app            (or python -m printworthy.app)
"""
from __future__ import annotations

import html
import os
import re

_COLOR = {"PASS": ("#1b7f3b", "#e7f6ec"),
          "WARN": ("#9a6a00", "#fdf4e0"),
          "FAIL": ("#b3261e", "#fbe9e7"),
          "REJECTED": ("#b3261e", "#fbe9e7"),
          "ERROR": ("#b3261e", "#fbe9e7"),
          "WORKING": ("#1a5fb4", "#e7f0fb")}

# Plain-words verdict framing for a novice. NEVER upgrades a claim:
# FAIL stays a refusal, REJECTED stays "can't read it".
_PLAIN = {"PASS": "Ready to print",
          "WARN": "Printable — read the warnings",
          "FAIL": "Not printable yet",
          "REJECTED": "Can't read this file",
          "ERROR": "Something went wrong",
          "WORKING": "Working on it"}

# Caption map keyed on the render-file stems the pipeline actually emits
# (premortem/traps/warp/...); legacy stems kept so older packages still map.
_GALLERY_CAP = {
    "premortem": "Where printing is risky (red = needs supports, may scar, "
                 "or is too thin)",
    "traps": "Pockets that could trap liquid or loose material",
    "warp": "Predicted print warp — experimental, untested",
    "strength": "Stronger vs weaker regions — comparative estimate",
    "orient": "Suggested printing direction",
    "support": "Where supports will touch (in the applied printing "
               "position)",
    "trust": "What the source photo actually saw — green was visible to "
             "your camera, orange the AI invented (unverifiable); requires "
             "your source image + pose",
    # legacy stems (older pipeline versions)
    "printability": "Where printing is risky (red = needs supports, may "
                    "scar, or is too thin)",
    "resin_traps": "Pockets that could trap liquid or loose material",
}

# UI unit choices -> pipeline assume_unit codes (None = let the file speak).
_UNITS = {"Auto-detect": None,
          "Millimetres (mm)": "mm",
          "Centimetres (cm)": "cm",
          "Metres (m)": "m",
          "Inches": "inch"}

INTRO = """# printworthy
**Drop a 3D model. Get back a print-ready file and a summary report.**

Fixes only what blocks printing, and reports exactly how much was altered.
Also sanity-checks the size, and flags thin walls, trapped pockets, and
support-heavy angles — each with a fix and a next step.

<small>Use at your own risk. Checks are geometric estimates. Optional warp
prediction is an experimental physics simulation — **untested**. Not a
slicer; results are advisory only.</small>"""

PRIVACY = ("<small>**Privacy:** your file is processed in memory and is not "
           "stored, logged, shared, or used for training — nothing is "
           "uploaded anywhere except this Space. Limits: **25&nbsp;MB** / "
           "3M triangles; very detailed meshes are simplified for analysis "
           "(the review says so when that happens).</small>")

FOOTER = ("<small>Open source — "
          "[printworthy on GitHub]"
          "(https://github.com/christopherschwartze-creator/printworthy)"
          ".</small>")


# ---------------------------------------------------------------------------
#  badge (verdict + trust line + size line)
# ---------------------------------------------------------------------------
def _plain_headline(text):
    """Route badge headlines through the report module's plain-language
    scrubber so raw engine strings (assume_unit=..., 'blocking issue(s)',
    engine reject messages) never reach the novice — the same contract the
    review pages already honour. Simplifies vocabulary only; every number
    and verdict passes through unchanged. Never raises."""
    try:
        from .report import _rv_plain_text
        return _rv_plain_text(text)
    except Exception:
        return str(text or "")


def _badge(verdict, headline, extras=()):
    """Colored banner: plain-words verdict, the technical verdict code,
    headline, and extra lines (trust line, size sanity). Headlines are
    scrubbed to plain language before display."""
    headline = _plain_headline(headline)
    fg, bg = _COLOR.get(verdict, _COLOR["WARN"])
    plain = _PLAIN.get(verdict, verdict)
    rows = "".join(
        f"<div style='margin-top:6px;font-size:1.0em;color:#222'>"
        f"{html.escape(str(x))}</div>" for x in extras if x)
    code = ("" if verdict == "WORKING" else
            f" <span style='font-size:0.85em;font-weight:400;color:{fg}'>"
            f"({html.escape(verdict)})</span>")
    return (f"<div style='background:{bg};border-left:8px solid {fg};"
            f"padding:14px 18px;border-radius:8px'>"
            f"<span style='font-size:1.5em;font-weight:700;color:{fg}'>"
            f"{html.escape(plain)}</span>{code}"
            f"<div style='margin-top:4px;font-size:1.05em;color:#222'>"
            f"{html.escape(str(headline))}</div>{rows}</div>")


def _trust_line(result, fix_requested):
    """THE trust line: how much surface the fix touched, max deviation.
    Composed from the deviation certificate; falls back to the pipeline's
    own fidelity line; never invents numbers."""
    fx = result.get("fix")
    if not fix_requested:
        return "Your model was not modified (Fix was switched off)."
    if not isinstance(fx, dict):
        return ""
    cert = fx.get("deviation_certificate")
    if isinstance(cert, dict) and not cert.get("rolled_back"):
        unchg = cert.get("surface_unchanged_pct")
        dev = cert.get("max_deviation_mm")
        if isinstance(unchg, (int, float)) and isinstance(dev, (int, float)) \
                and dev == dev and dev != float("inf"):
            touched = max(0.0, min(100.0, 100.0 - float(unchg)))
            return (f"The fix touched {touched:.0f}% of your surface "
                    f"(max deviation {dev:.3f} mm) — the rest is exactly "
                    f"your original.")
    return str(fx.get("fidelity_line") or "")


def _size_line(result, resized, unit_code):
    """Size sanity, in plain words: what size the part will print at and
    whether we kept, converted, or guessed it."""
    eff = result.get("print_mm")
    if not isinstance(eff, (int, float)):
        return ""
    scan = result.get("unit_scan") or {}
    ext = scan.get("max_extent")
    if resized:
        return (f"Size: scaled so the longest side is {eff:g} mm — "
                f"you asked for that size.")
    if unit_code:
        return (f"Size: read in {unit_code} as you set — longest side "
                f"{eff:g} mm.")
    if isinstance(ext, (int, float)) and 5.0 <= float(ext) <= 500.0:
        line = (f"Size check: your file's longest side reads {eff:.1f} mm — "
                f"your model was KEPT at its original size. Use 'Resize' "
                f"below if that's wrong.")
    else:
        line = (f"Size check: the file didn't say its units and its size "
                f"looked implausible, so {eff:g} mm was GUESSED for the "
                f"longest side. Set 'File units' or 'Resize' if that's "
                f"wrong.")
    warn = scan.get("warning")
    # the pipeline already lifts the unit warning into the headline; repeating
    # it in the size line printed the same warning twice on the badge
    if warn and warn not in str(result.get("headline") or ""):
        line += f" Heads-up: {warn}"
    return line


# ---------------------------------------------------------------------------
#  review document (build_review when available; scrubbed legacy fallback)
# ---------------------------------------------------------------------------
def _scrub(md, result):
    """Conservative UI-side cleanup of the LEGACY engineer report: redact
    server-local temp paths and CLI-flag advice that mean nothing in a
    browser. Simplifies wording only — never changes a verdict or number."""
    try:
        out_dir = result.get("out_dir")
        if out_dir:
            for v in (str(out_dir), str(out_dir).replace("\\", "/"),
                      str(out_dir).replace("\\", "\\\\")):
                md = md.replace(v, "(your download package)")
        # any leftover server temp paths (Windows or POSIX)
        md = re.sub(r"[A-Za-z]:[\\/][^\s`'\"<>|)]*printworthy_[^\s`'\"<>|)]*",
                    "(your download package)", md)
        md = re.sub(r"/[^\s`'\"<>|)]*printworthy_[^\s`'\"<>|)]*",
                    "(your download package)", md)
        # CLI advice that has no meaning in the browser
        md = md.replace("Run the full prep (drop the `--check` flag)",
                        "Run the fix (tick 'Fix problems' and prep again)")
        md = md.replace("(drop the `--check` flag)", "")
        md = md.replace("(`prep` applies it; `check` only reports it)",
                        "(this prep already applied it)")
        # features that ship separately / are not in this UI
        md = "\n".join(ln for ln in md.splitlines()
                       if "autorig" not in ln.lower())
    except Exception:
        pass
    return md


def _review_markdown(result):
    """The review document for the UI. Prefers report.build_review(result)
    (the novice-facing review; may not exist until the report branch merges)
    and falls back to the existing report markdown, scrubbed."""
    try:
        from . import report as report_mod
        if hasattr(report_mod, "build_review"):
            rv = report_mod.build_review(result)
            if isinstance(rv, str) and rv.strip():
                return rv
            if isinstance(rv, dict):
                m = rv.get("markdown") or rv.get("md")
                if isinstance(m, str) and m.strip():
                    return m
    except Exception:
        pass
    md = (result.get("report") or {}).get("markdown", "")
    return _scrub(md, result)


def _write_review(md, result):
    """Persist the review as review.md next to the package so it can be
    downloaded and handed to a print shop. Returns the path or None."""
    out_dir = result.get("out_dir")
    if not (md and out_dir):
        return None
    try:
        p = os.path.join(str(out_dir), "review.md")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(md)
        return p
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  gallery + downloads
# ---------------------------------------------------------------------------
def _gallery(result):
    out = []
    for p in result.get("renders") or []:
        if p and os.path.exists(p):
            stem = os.path.splitext(os.path.basename(p))[0]
            cap = _GALLERY_CAP.get(stem)
            if cap is None:  # unknown stem -> humanize, never raw jargon
                cap = stem.replace("_", " ").replace("-", " ").capitalize()
            out.append((p, cap))
    # the pipeline can emit the same render for before AND after the repair
    # (same file stem, different folder) — identical captions read as
    # accidental duplicates, so say which is which.
    caps = [c for _, c in out]
    for i, (p, cap) in enumerate(out):
        if caps.count(cap) > 1:
            low = p.replace("\\", "/").lower()
            if "/before/" in low or low.endswith("before.png"):
                out[i] = (p, cap + " — your upload, as analysed")
            elif "/after/" in low or low.endswith("after.png"):
                out[i] = (p, cap + " — after the repair")
            else:
                out[i] = (p, cap + (" — as uploaded" if caps.index(cap) == i
                                    else " — after the repair"))
    return out


def _downloads(result, review_path):
    """The Space slice ships: the fixed model (prep.stl), the review
    (review.md), and the full technical report (report.md — the review's
    fine print points the user at it, so it must actually be present)."""
    files = result.get("files") or {}
    out = []
    stl = files.get("prep_stl")
    if stl and os.path.exists(str(stl)):
        out.append(str(stl))
    if review_path and os.path.exists(str(review_path)):
        out.append(str(review_path))
    rep = files.get("report_md")
    if rep and os.path.exists(str(rep)) and str(rep) not in out:
        out.append(str(rep))
    return out


# ---------------------------------------------------------------------------
#  handler (generator: immediate 'working' feedback, then the result)
# ---------------------------------------------------------------------------
def run(file, printer, resize, print_mm, unit, nozzle, material,
        fix, orient, fem):
    """Yields: badge_html, report_md, gallery, downloads."""
    empty = ("", [], [])
    if not file:
        yield (_badge("WARN", "Drop a 3D model (GLB / OBJ / STL / PLY / "
                              "OFF / 3MF) to start."), *empty)
        return
    wait = ("Checking your file now. Big files can take a few minutes."
            + (" The warp physics pass adds about a minute more."
               if fem else ""))
    yield (_badge("WORKING", wait), *empty)

    from .pipeline import prep
    kw = {}
    try:  # optional Space-side face cap (keeps free-CPU jobs inside timeout)
        mf = int(os.environ.get("PRINTWORTHY_MAX_FACES", "0"))
        if mf > 0:
            kw["max_faces"] = mf
    except (TypeError, ValueError):
        pass
    # hosting-context bundle (PRINTWORTHY_PRESET=space on the hosted Space):
    # activates the purpose-built budgeted pipeline — 120 s soft stage budget,
    # warp FEM capped at 1500 elements, fast post-fix re-check, support
    # render. Without this the Space ran every job at full desktop settings.
    preset = (os.environ.get("PRINTWORTHY_PRESET") or "").strip() or None
    if preset:
        kw["preset"] = preset
    # prep() is never-raise, but keep the UI itself foolproof: any surprise
    # becomes a clean FAIL badge, never a Gradio traceback.
    try:
        unit_code = _UNITS.get(unit)
        res = prep(file, profile=printer,
                   print_mm=float(print_mm) if resize else None,
                   assume_unit=unit_code,
                   nozzle_mm=float(nozzle) if nozzle else None,
                   material=str(material) if material else None,
                   fix=bool(fix), orient=bool(orient),
                   fem_warp=bool(fem),
                   slicer_savings=False,       # no slicer on the Space
                   **kw)
    except Exception as e:
        yield (_badge("FAIL", f"Could not process this file "
                              f"({type(e).__name__}). Your model was not "
                              f"changed; nothing was produced — do not "
                              f"print."), *empty)
        return
    verdict = res.get("verdict", "ERROR")
    if verdict in ("REJECTED", "ERROR"):
        head = (res.get("rejected") or res.get("error")
                or res.get("headline") or "Could not analyse this file.")
        yield (_badge(verdict, str(head)), *empty)
        return
    md = _review_markdown(res)
    review_path = _write_review(md, res)
    badge = _badge(verdict, res.get("headline", ""),
                   extras=(_trust_line(res, bool(fix)),
                           _size_line(res, bool(resize), _UNITS.get(unit))))
    yield badge, md, _gallery(res), _downloads(res, review_path)


# ---------------------------------------------------------------------------
#  UI
# ---------------------------------------------------------------------------
def _fdm_profiles(profiles):
    """Only FDM printers in this UI (resin mode is CLI/API-only for now)."""
    try:
        keep = [k for k, v in profiles.items()
                if not (isinstance(v, dict)
                        and str(v.get("technology", "fdm")).lower() == "resin")]
        return keep or list(profiles)
    except Exception:
        return list(profiles)


def build_demo():
    """Construct (not launch) the Gradio Blocks app. Raises RuntimeError with
    install guidance if gradio is absent."""
    try:
        import gradio as gr
    except ImportError as e:
        raise RuntimeError(
            "The printworthy app needs gradio. Install it: "
            "`pip install gradio`.") from e

    from .profiles import PRINTER_PROFILES
    printers = _fdm_profiles(PRINTER_PROFILES)
    default_printer = "generic_fdm" if "generic_fdm" in printers else printers[0]

    with gr.Blocks(title="printworthy") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                file = gr.File(label="Your 3D model", type="filepath",
                               file_types=[".glb", ".gltf", ".obj", ".stl",
                                           ".ply", ".off", ".3mf"])
                printer = gr.Dropdown(printers, value=default_printer,
                                      label="Printer")
                with gr.Row():
                    fix = gr.Checkbox(
                        label="Fix problems (keeps your shape — reports "
                              "exactly how much was touched)", value=True)
                    orient = gr.Checkbox(
                        label="Suggest the best printing orientation",
                        value=True)
                fem = gr.Checkbox(
                    label="Predict warp — ~1 min, experimental "
                          "(untested)", value=False)
                with gr.Accordion("Size & units (your file's own size is "
                                  "kept unless you change this)",
                                  open=False):
                    resize = gr.Checkbox(
                        label="Resize for printing", value=False)
                    print_mm = gr.Slider(
                        10, 250, value=60, step=5,
                        label="If resizing: longest side (mm)")
                    unit = gr.Dropdown(
                        list(_UNITS), value="Auto-detect",
                        label="File units (set this if the size reads wrong)")
                with gr.Accordion("Advanced", open=False):
                    nozzle = gr.Slider(
                        0.1, 1.0, value=0.4, step=0.05,
                        label="Nozzle size (mm) — sets the thin-wall check")
                    material = gr.Dropdown(
                        ["PLA", "PETG", "ABS"], value="PLA",
                        label="Material — used by the warp estimate")
                go = gr.Button("Prep it", variant="primary")
                gr.Markdown(PRIVACY)
            with gr.Column(scale=2):
                badge = gr.HTML()
                gallery = gr.Gallery(label="Visual checks", columns=2,
                                     height=380, object_fit="contain")
                downloads = gr.Files(
                    label="Downloads — fixed model (prep.stl) + review for "
                          "the print shop (review.md)")
                report = gr.Markdown()
        gr.Markdown(FOOTER)
        go.click(run,
                 inputs=[file, printer, resize, print_mm, unit, nozzle,
                         material, fix, orient, fem],
                 outputs=[badge, report, gallery, downloads])
    # one worker: free Space CPUs choke on parallel FEM/analysis jobs
    try:
        demo.queue(default_concurrency_limit=1)
    except TypeError:                       # older gradio signature
        demo.queue(concurrency_count=1)
    return demo


def main(port=7860, share=False):
    from .core._mesh_util import ascii_console
    ascii_console()
    demo = build_demo()
    try:
        demo.launch(server_port=port, share=share, max_file_size="25mb")
    except TypeError:                       # older gradio without the cap arg
        demo.launch(server_port=port, share=share)
    return 0


if __name__ == "__main__":
    main()
