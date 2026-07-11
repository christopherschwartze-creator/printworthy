"""meshprep CLI — ``meshprep <check|fix|prep|reinforce|calibrate|supports|app>
...``.

One console entry (`meshprep = meshprep.cli:main` in pyproject). The mesh
subcommands are thin wrappers over :func:`meshprep.pipeline.prep`; `reinforce`,
`calibrate` and `supports` delegate to their own standalone arg surface (own
``-h``); `app` launches the Gradio front door. `prep --supports` runs the same
support-mods stage inline, as part of the full pipeline.

ASCII-safe output (cp1252 consoles) via core._mesh_util.ascii_console/say.
Exit codes: 0 = PASS/WARN, 1 = FAIL, 2 = rejected/error.
"""
from __future__ import annotations

import argparse
import sys


def _add_common(p, *, full=False):
    p.add_argument("path", help="mesh file (GLB/OBJ/STL/PLY/OFF/3MF)")
    p.add_argument("--profile", default="generic_fdm",
                   help="printer preset (ender3|prusa_mk4|bambu_a1|bambu_x1c|"
                        "generic_fdm|generic_resin_msla or an alias)")
    p.add_argument("--print-mm", type=float, default=None,
                   help="longest side of the printed part in mm "
                        "(default: keep size if it looks like mm, else 60)")
    p.add_argument("--assume-unit", default=None,
                   choices=("mm", "cm", "m", "inch", "in"),
                   help="explicit unit of the FILE (STL/OBJ carry none). Use "
                        "this to honestly fix a mis-scaled export, e.g. a part "
                        "exported in cm -> --assume-unit cm.")
    p.add_argument("--nozzle", type=float, default=None,
                   help="nozzle / min feature mm (default: the profile's)")
    p.add_argument("--material", default=None, help="PLA|PETG|ABS|resin|...")
    p.add_argument("--out", default=None,
                   help="output folder (default: a fresh temp dir)")
    p.add_argument("--max-faces", type=int, default=None,
                   help="decimate above this many faces before analysis "
                        "(default 60000; lower = faster triage)")
    p.add_argument("--json", action="store_true",
                   help="print the report JSON instead of markdown")
    if full:
        p.add_argument("--mode", default=None, choices=("fdm", "resin"),
                       help="default: the profile's technology")
        p.add_argument("--fem", action="store_true",
                       help="inherent-strain warp FEM (slower; uncalibrated "
                            "unless a coupon was fitted)")
        p.add_argument("--strength", action="store_true",
                       help="orientation-strength FEM screening (slow)")
        p.add_argument("--reinforce", default=None, metavar="AXIS",
                       help="graded-infill 3MF for a load along x|y|z")
        p.add_argument("--force", type=float, default=100.0,
                       help="load (N) for --reinforce/--strength")
        p.add_argument("--retopo", action="store_true",
                       help="also quad-remesh the prepped part (permissive "
                            "backend) -> prep_quads.obj")
        p.add_argument("--supports", action="store_true",
                       help="also emit a risk-driven SupportEnforcer/"
                            "SupportBlocker 3MF (PrusaSlicer schema) -> "
                            "supports.3mf, at the orientation that ships "
                            "(run `meshprep supports -h` for the standalone "
                            "form with tunable percentiles)")
        p.add_argument("--no-fix", action="store_true")
        p.add_argument("--no-orient", action="store_true")
        p.add_argument("--no-slicer", action="store_true",
                       help="skip the before/after slicer estimates")


def _emit(res, as_json):
    import json as _json

    from .core._mesh_util import say
    if as_json:
        say(_json.dumps(res.get("report_json")
                        or (res.get("report") or {}).get("json")
                        or {"verdict": res.get("verdict"),
                            "error": res.get("error")},
                        indent=2, default=str))
    else:
        say((res.get("report") or {}).get("markdown")
            or f"{res.get('verdict')}: {res.get('error', '')}")
    v = res.get("verdict")
    if v in ("PASS", "WARN"):
        return 0
    if v == "FAIL":
        return 1
    return 2


def _build_supports_cli():
    p = argparse.ArgumentParser(
        prog="meshprep supports",
        description="Mesh -> risk-driven support ENFORCER/BLOCKER 3MF "
                    "(PrusaSlicer SupportEnforcer/SupportBlocker schema). "
                    "Reuses the printability risk field (overhang/scar/thin/"
                    "warp) already computed for check/prep to mark regions "
                    "the slicer's blind angle test misses.")
    p.add_argument("path", help="mesh file (GLB/OBJ/STL/PLY/OFF/3MF)")
    p.add_argument("-o", "--out", default="supports.3mf", help="output 3MF path")
    p.add_argument("--enforce-pctile", type=float, default=85.0,
                   help="combined-risk percentile (of overhang/scar-dominant "
                        "faces) above which a region gets a SupportEnforcer "
                        "(default 85th -- conservative, top ~15%% of risk)")
    p.add_argument("--block-pctile", type=float, default=30.0,
                   help="overhang-channel percentile below which a face past "
                        "the slicer's blind-angle test gets a SupportBlocker "
                        "(default 30th; also always capped at risk<0.15)")
    return p


def _run_supports(argv):
    """``meshprep supports`` -- standalone dispatch (own arg surface, like
    reinforce/calibrate). Loads through the same ingress guard as reinforce,
    calls core.support_mods.support_mods() directly, and prints plain-language
    output including the module's own honest note (which carries the measured
    slicer flags verbatim -- see support_mods.py's docstring step 4) rather
    than re-deriving/duplicating that text here."""
    from .core._mesh_util import say

    args = _build_supports_cli().parse_args(argv)  # -h / bad args -> SystemExit
    try:
        from .core.preflight_core import GuardReject, load_and_guard
        from .core.support_mods import support_mods
        try:
            mesh, gnotes = load_and_guard(args.path)
        except GuardReject as g:
            say(f"REFUSED: {g}")
            return 2
        for n in gnotes:
            say(f"   {n}")
        res = support_mods(mesh, out_3mf=args.out,
                           enforce_pctile=args.enforce_pctile,
                           block_pctile=args.block_pctile)
    except SystemExit:
        raise
    except Exception as e:                          # never-raise to the user
        say(f"REFUSED: could not compute supports ({type(e).__name__}: {e}). "
            "Check the path and that the file is a closed solid mesh.")
        return 2

    if not res.get("ok"):
        # honest refusal (not a crash): degenerate / non-solid input
        say("REFUSED: " + str(res.get("reason") or res.get("note")
                              or "unknown error"))
        return 2
    if not res.get("out_3mf"):
        # ok, but nothing cleared the enforce/block thresholds -- no 3MF
        say("OK -- no supports.3mf written.")
        say("   " + str(res.get("note") or ""))
        return 0

    say(f"OK -> {res['out_3mf']}")
    say(f"   enforcers: {res.get('n_enforcers', 0)}   "
        f"blockers: {res.get('n_blockers', 0)}   "
        f"risk_top: {res.get('risk_top')}")
    say("   NOTE: " + str(res.get("note") or ""))
    return 0


def main(argv=None):
    from .core._mesh_util import ascii_console
    ascii_console()
    argv = list(sys.argv[1:] if argv is None else argv)

    # engine CLIs keep their own validated arg surface -- delegate, but NEVER
    # let a raw traceback reach the user. The reinforce path is the one entry
    # that bypasses pipeline.prep's ingress guard, so guard + wrap it here.
    if argv and argv[0] == "reinforce":
        from .core._mesh_util import say
        from .core import reinforce as rf
        try:
            return rf.main(argv[1:])
        except SystemExit:                          # argparse -h / bad args
            raise
        except Exception as e:                      # never-raise to the user
            say(f"REFUSED: could not reinforce ({type(e).__name__}: {e}). "
                "Check the file path and that it is a closed solid mesh.")
            return 2
    if argv and argv[0] == "calibrate":
        from .core._mesh_util import say
        from .core import _warp_calibrate as wc
        try:
            return wc._main(argv[1:])
        except SystemExit:
            raise
        except Exception as e:
            say(f"REFUSED: calibration failed ({type(e).__name__}: {e}).")
            return 2
    if argv and argv[0] == "supports":
        return _run_supports(argv[1:])

    ap = argparse.ArgumentParser(
        prog="meshprep",
        description="Drop a mesh -> print-ready package + plain-English report.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="analysis + report only (no files "
                                           "changed)")
    _add_common(p_check, full=False)
    p_check.add_argument("--fem", action="store_true",
                         help="inherent-strain warp FEM (slower)")
    p_check.add_argument("--strength", action="store_true",
                         help="orientation-strength FEM screening (slow)")
    p_check.add_argument("--mode", default=None, choices=("fdm", "resin"))

    p_fix = sub.add_parser("fix", help="source-accurate repair + deviation "
                                       "certificate (no reorientation)")
    _add_common(p_fix, full=False)

    p_prep = sub.add_parser("prep", help="the full pipeline: check + fix + "
                                         "orient + package (+ opt-in extras)")
    _add_common(p_prep, full=True)

    sub.add_parser("reinforce", add_help=False,
                   help="mesh + load case -> graded-infill 3MF "
                        "(run `meshprep reinforce -h`)")
    sub.add_parser("calibrate", add_help=False,
                   help="fit the warp scale from one printed test bar "
                        "(run `meshprep calibrate -h`)")
    sub.add_parser("supports", add_help=False,
                   help="mesh -> risk-driven support ENFORCER/BLOCKER 3MF "
                        "(run `meshprep supports -h`)")

    p_app = sub.add_parser("app", help="launch the Gradio front door")
    p_app.add_argument("--port", type=int, default=7860)
    p_app.add_argument("--share", action="store_true")

    a = ap.parse_args(argv)

    if a.cmd == "app":
        from .core._mesh_util import say
        try:
            from . import app as app_mod
            return app_mod.main(port=a.port, share=a.share)
        except Exception as e:                      # never-raise to the user
            say(f"Could not launch the app ({type(e).__name__}: {e}). "
                "Install gradio: pip install gradio.")
            return 2

    from .pipeline import prep
    common = dict(profile=a.profile, print_mm=a.print_mm, nozzle_mm=a.nozzle,
                  material=a.material, out_dir=a.out, max_faces=a.max_faces,
                  assume_unit=getattr(a, "assume_unit", None))
    # prep() is itself never-raise, but wrap the dispatch so nothing (arg glue,
    # emit) can leak a traceback to a novice at the console.
    try:
        if a.cmd == "check":
            res = prep(a.path, check_only=True, fem_warp=a.fem,
                       fem_strength=a.strength, mode=a.mode, **common)
        elif a.cmd == "fix":
            res = prep(a.path, fix=True, orient=False, slicer_savings=False,
                       **common)
        else:  # prep
            res = prep(a.path, mode=a.mode, fix=not a.no_fix,
                       orient=not a.no_orient, fem_warp=a.fem,
                       fem_strength=a.strength, reinforce_load=a.reinforce,
                       reinforce_force_n=a.force, retopo=a.retopo,
                       supports=a.supports,
                       slicer_savings=not a.no_slicer, **common)
        return _emit(res, a.json)
    except Exception as e:
        from .core._mesh_util import say
        say(f"FAIL: meshprep could not process this file "
            f"({type(e).__name__}: {e}). It was not changed; do not print.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
