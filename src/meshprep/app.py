"""meshprep Gradio front door — one drop-zone driving the FULL pipeline.

Port of the validated Pre-Flight shell (Forge preflight/app.py) onto
:func:`meshprep.pipeline.prep`: drop a mesh -> verdict badge + the unified
plain-English report + risk-heatmap gallery + downloads (prepped STL, graded
3MF, report.md/json) + the slicer savings line, with the FEM / resin /
reinforce toggles.

Gradio is an optional extra (``pip install meshprep[app]``); this module
imports clean without it and :func:`build_demo` raises clear guidance.

Run:  meshprep app            (or python -m meshprep.app)
"""
from __future__ import annotations

import os

_COLOR = {"PASS": ("#1b7f3b", "#e7f6ec"),
          "WARN": ("#9a6a00", "#fdf4e0"),
          "FAIL": ("#b3261e", "#fbe9e7"),
          "REJECTED": ("#b3261e", "#fbe9e7"),
          "ERROR": ("#b3261e", "#fbe9e7")}

_LOADS = {"Vertical (Z)": "z", "Horizontal (X)": "x", "Horizontal (Y)": "y"}

_GALLERY_CAP = {"printability": "Printability risk (red = supports/scarring/thin)",
                "resin_traps": "Resin pooling / trapped volume",
                "warp": "Predicted print warp (inherent-strain FEM)",
                "strength": "Stress / weakest region (best orientation)",
                "orient": "Strongest vs weakest build direction"}

INTRO = """# meshprep
**Drop a 3D model. Get a print-ready package + a plain-English report you can trust.**
Checker -> source-accurate fix -> best orientation -> (optional) graded-infill reinforcement -> report.

<small>Research preview. Fast checks are **geometric heuristics**; the optional warp/strength passes are a real
reduced-order **FEM** — *uncalibrated* (right sign/shape/trend, not certified numbers) until you fit one coupon
(`meshprep calibrate`). Not a slicer. "Watertight" does not guarantee "genus-correct". Verdicts are advisory.</small>"""

PRIVACY = ("<small>**Privacy:** your mesh is processed locally/in-memory and not stored, logged, "
           "or used for training. Files over 25 MB / 3M triangles are rejected; large meshes are "
           "decimated for a fast preview.</small>")


def _badge(verdict, headline):
    fg, bg = _COLOR.get(verdict, _COLOR["WARN"])
    return (f"<div style='background:{bg};border-left:8px solid {fg};"
            f"padding:14px 18px;border-radius:8px'>"
            f"<span style='font-size:1.5em;font-weight:700;color:{fg}'>{verdict}</span>"
            f"<div style='margin-top:4px;font-size:1.05em;color:#222'>{headline}</div></div>")


def _gallery(result):
    out = []
    # renders is a list of paths; recover captions from the filename stem
    for p in result.get("renders") or []:
        if p and os.path.exists(p):
            stem = os.path.splitext(os.path.basename(p))[0]
            out.append((p, _GALLERY_CAP.get(stem, stem)))
    return out


def _downloads(result):
    files = result.get("files") or {}
    order = ("prep_stl", "prep_3mf", "fixed_glb", "quad_obj", "gcode",
             "report_md", "report_json")
    return [files[k] for k in order
            if files.get(k) and os.path.exists(str(files[k]))]


def _savings_line(result):
    sv = result.get("savings")
    if not isinstance(sv, dict):
        return ""
    if not sv.get("ok"):
        return f"_{sv.get('note', '')}_"
    bits = []
    if isinstance(sv.get("saved_min"), (int, float)):
        bits.append(f"**{sv['saved_min']:.0f} min** print time")
    if isinstance(sv.get("saved_g"), (int, float)):
        bits.append(f"**{sv['saved_g']:.1f} g** filament")
    pct = sv.get("saved_pct")
    pct_s = f" ({pct:+.0f}%)" if isinstance(pct, (int, float)) else ""
    return ("### Savings\nPrepped vs as-uploaded: "
            + (", ".join(bits) if bits else "no change") + pct_s
            + f" — {sv.get('slicer', 'slicer')} estimates.")


# ---------------------------------------------------------------------------
#  handler
# ---------------------------------------------------------------------------
def run(file, printer, print_mm, nozzle, material, fem, resin_mode,
        reinforce, load_dir, fix, orient):
    """Returns: badge_html, report_md, gallery, downloads, savings_md."""
    empty = ("", [], [], "")
    if not file:
        return (_badge("WARN", "Upload a mesh (GLB / OBJ / STL / PLY / OFF / "
                               "3MF) to start."), *empty)
    from .pipeline import prep
    # prep() is never-raise, but keep the UI itself foolproof: any surprise
    # becomes a clean FAIL badge, never a Gradio traceback.
    try:
        res = prep(file, profile=printer, print_mm=float(print_mm),
                   nozzle_mm=float(nozzle) if nozzle else None,
                   material=str(material) if material else None,
                   mode="resin" if resin_mode else None,
                   fix=bool(fix), orient=bool(orient),
                   fem_warp=bool(fem),
                   reinforce_load=_LOADS.get(load_dir, "z") if reinforce else None)
    except Exception as e:
        return (_badge("FAIL", f"Could not process this file "
                               f"({type(e).__name__}). It was not changed; "
                               f"do not print."), *empty)
    verdict = res.get("verdict", "ERROR")
    if verdict in ("REJECTED", "ERROR"):
        head = res.get("rejected") or res.get("error") or "Could not analyse."
        return (_badge(verdict, str(head)), *empty)
    badge = _badge(verdict, res.get("headline", ""))
    md = (res.get("report") or {}).get("markdown", "")
    return badge, md, _gallery(res), _downloads(res), _savings_line(res)


# ---------------------------------------------------------------------------
#  UI
# ---------------------------------------------------------------------------
def build_demo():
    """Construct (not launch) the Gradio Blocks app. Raises RuntimeError with
    install guidance if gradio is absent."""
    try:
        import gradio as gr
    except ImportError as e:
        raise RuntimeError(
            "The meshprep app needs gradio. Install the extra: "
            "`pip install meshprep[app]` (or `pip install gradio`).") from e

    from .profiles import PRINTER_PROFILES

    with gr.Blocks(title="meshprep") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                file = gr.File(label="Mesh file", type="filepath",
                               file_types=[".glb", ".gltf", ".obj", ".stl",
                                           ".ply", ".off", ".3mf"])
                printer = gr.Dropdown(list(PRINTER_PROFILES),
                                      value="generic_fdm", label="Printer")
                print_mm = gr.Slider(10, 250, value=60, step=5,
                                     label="Print size — longest side (mm)")
                nozzle = gr.Slider(0.1, 1.0, value=0.4, step=0.05,
                                   label="Nozzle / min feature (mm)")
                material = gr.Dropdown(["PLA", "PETG", "ABS", "resin"],
                                       value="PLA",
                                       label="Material (for the FEM passes)")
                with gr.Row():
                    fix = gr.Checkbox(label="Fix (source-accurate)", value=True)
                    orient = gr.Checkbox(label="Best orientation", value=True)
                with gr.Row():
                    fem = gr.Checkbox(label="Predict warp (FEM, slower)",
                                      value=False)
                    resin_mode = gr.Checkbox(label="Resin mode", value=False)
                with gr.Row():
                    reinforce = gr.Checkbox(
                        label="Reinforce (graded-infill 3MF, FEM)", value=False)
                    load_dir = gr.Dropdown(list(_LOADS), value="Vertical (Z)",
                                           label="Load direction")
                go = gr.Button("Prep it", variant="primary")
                gr.Markdown(PRIVACY)
            with gr.Column(scale=2):
                badge = gr.HTML()
                savings = gr.Markdown()
                gallery = gr.Gallery(label="Risk heatmaps", columns=2,
                                     height=380, object_fit="contain")
                downloads = gr.Files(label="Download the package "
                                           "(prepped STL / 3MF / report)")
                report = gr.Markdown()
        go.click(run,
                 inputs=[file, printer, print_mm, nozzle, material, fem,
                         resin_mode, reinforce, load_dir, fix, orient],
                 outputs=[badge, report, gallery, downloads, savings])
    return demo


def main(port=7860, share=False):
    from .core._mesh_util import ascii_console
    ascii_console()
    build_demo().launch(server_port=port, share=share)
    return 0


if __name__ == "__main__":
    main()
