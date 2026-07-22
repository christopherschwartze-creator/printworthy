"""Shared small utilities for printworthy.core.

Single home for helpers that were duplicated across the source tree (the audit
found ~4 copies of the decimation guard) plus the ascii-safe console helpers
every selftest / CLI entry uses on cp1252 Windows consoles.
"""
from __future__ import annotations


def decimate(mesh, max_faces=6000):
    """Quadric-decimate to <= max_faces (compute safety). Handles both trimesh
    APIs (`face_count=` kwarg vs ratio positional). Never raises; returns the
    input unchanged on any failure or if already under the cap.

    NOTE (port): some source copies called `simplify_quadric_decimation(n)`
    positionally, which RAISES on trimesh >= 4 (positional arg is `percent`)
    and silently no-op'd via their try/except. This canonical version actually
    decimates; selftest inputs are all under the cap, so validated numbers are
    unchanged."""
    try:
        if len(mesh.faces) <= max_faces:
            return mesh
        try:
            m = mesh.simplify_quadric_decimation(face_count=max_faces)
        except TypeError:
            m = mesh.simplify_quadric_decimation(1 - max_faces / len(mesh.faces))
        if m is None or len(m.faces) == 0:
            return mesh
        return m
    except Exception:
        return mesh


def fmt_fos(x, prec=2):
    """Display form of a factor-of-safety. The FEM clips FOS to a 1e6 ceiling
    for unloaded elements (validated numerics, untouched); at the DISPLAY layer
    that sentinel means the load there was negligible — never show it raw."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if x != x or x >= 1e6:        # NaN, inf, or the unloaded-element ceiling
        return "n/a (load tiny)"
    return f"{x:.{prec}f}"


def ascii_console():
    """Make stdout/stderr survive cp1252 consoles (errors='replace'). Call at
    the top of every selftest / CLI entry point. Never raises."""
    import sys
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass


def say(*args, sep=" ", end="\n", file=None):
    """print() that never dies with UnicodeEncodeError on a cp1252 console."""
    import sys
    f = file if file is not None else sys.stdout
    text = sep.join(str(a) for a in args) + end
    try:
        f.write(text)
    except UnicodeEncodeError:
        enc = getattr(f, "encoding", None) or "ascii"
        f.write(text.encode(enc, errors="replace").decode(enc, errors="replace"))
