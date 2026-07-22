"""printworthy.core — the vendored, validated engines.

Vendored from the development monorepo (Forge) per RELEASE_MANIFEST.md:
preflight checker+fixer, graded-infill 3MF reinforce, print/FEM stack, and the
from-scratch permissive quad-retopo stack. Modules keep their original names
(`_print_fem`, `_fix_accurate`, ...) so the validated code reads unchanged;
public callables are exposed lazily below (nothing heavy imports at package
import time).

License note: the default retopo backends (mcf2 / blossom) are permissive and
from scratch; pyQuadriFlow is ONLY reachable via `quad_remesh(mode="pyqf")`
with the `[retopo-pyqf]` extra installed (LGPL caveat — THIRD_PARTY_NOTICES).
"""

_PUBLIC = {
    # preflight (checker + cert + fixer)
    "preflight": ("preflight_core", "preflight"),
    "load_and_guard": ("preflight_core", "load_and_guard"),
    "run_checks": ("preflight_core", "run_checks"),
    "build_cert": ("preflight_core", "build_cert"),
    "run_fixer": ("preflight_core", "run_fixer"),
    # source-accurate fixer
    "accurate_fix": ("_fix_accurate", "accurate_fix"),
    "fix_toy": ("_fix_accurate", "fix_toy"),
    "deviation_certificate": ("_fix_accurate", "deviation_certificate"),
    # printability checkers
    "premortem": ("_print_premortem", "premortem"),
    "find_traps": ("_print_traps", "find_traps"),
    "best_drain_orientation": ("_print_traps", "best_drain_orientation"),
    "run_traps_self_tests": ("_print_traps", "run_self_tests"),
    # FEM (warp / strength / orientation)
    "warp_analysis": ("_print_fem", "warp_analysis"),
    "strength_analysis": ("_print_fem", "strength_analysis"),
    "orient_for_strength": ("_print_fem", "orient_for_strength"),
    "calibrated_warp": ("_warp_calibrate", "calibrated_warp"),
    "calibrate": ("_warp_calibrate", "calibrate"),
    # watertight / hole fill / manifold
    "make_watertight": ("force_solid", "make_watertight"),
    "fill_holes_ei_bisector": ("hole_fill_ei", "fill_holes_ei_bisector"),
    # retopo (function is quad_remesh.quad_remesh; key avoids the module name)
    "remesh": ("quad_remesh", "quad_remesh"),
    "remesh_quality": ("quad_remesh", "quality"),
    "quad_genus": ("_genus", "quad_genus"),
    "genus_repair": ("_genus", "genus_repair"),
    # shared utilities
    "decimate": ("_mesh_util", "decimate"),
    "fmt_fos": ("_mesh_util", "fmt_fos"),
    "say": ("_mesh_util", "say"),
    "ascii_console": ("_mesh_util", "ascii_console"),
}

__all__ = sorted(_PUBLIC)


def __getattr__(name):
    import importlib
    try:
        mod_name, attr = _PUBLIC[name]
    except KeyError:
        raise AttributeError(f"module 'printworthy.core' has no attribute {name!r}")
    return getattr(importlib.import_module("." + mod_name, __name__), attr)


def __dir__():
    return sorted(set(list(globals()) + __all__))
