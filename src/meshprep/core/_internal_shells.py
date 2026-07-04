"""meshprep.core._internal_shells — internal-shell (cavity) census.

Why this exists
---------------
The source-accurate fixer preserves internal shells as CAVITIES. A faithfully
repaired mesh therefore prints with LESS filament than a chunky slicer
auto-repair that tends to flood those cavities solid. Historically this
surprised users ("your fix made a lighter print than the slicer's"). This
module turns that surprise into one honest, reported line.

`analyze_internal_shells(mesh)` is a PURE, never-raises geometric probe. It
splits the mesh into connected components, calls the biggest one the OUTER
shell, and counts every other CLOSED component whose bounding box sits strictly
inside the outer bbox as an INTERNAL shell (a cavity).

HONESTY (this stack has an inflated-claim history)
--------------------------------------------------
Every number here is a GEOMETRIC estimate, NOT a slicer measurement.
`solid_volume_ratio` = (outer_solid_volume - internal_void_volume) / outer_solid
is the 100%-infill material ratio of a faithful print vs a naive cavity-filling
fill — i.e. a DEFENSIBLE UPPER BOUND on how much material the faithful print
saves. At realistic low infill the actual percentage gap is SMALLER (only the
wall/infill fraction of the void is ever saved). The returned `note` says so.
Never upgrade this to a slicer/filament claim.
"""
from __future__ import annotations

_EPS = 1e-9


def _bbox_inside(inner_bounds, outer_bounds, tol) -> bool:
    """True iff `inner_bounds` AABB lies strictly inside `outer_bounds` AABB
    (with a small tolerance so a shell sharing a face is not counted). Bounds
    are ((min_xyz), (max_xyz)) array-likes."""
    (imin, imax), (omin, omax) = inner_bounds, outer_bounds
    for k in range(3):
        if not (imin[k] > omin[k] + tol and imax[k] < omax[k] - tol):
            return False
    return True


def analyze_internal_shells(mesh) -> dict:
    """Geometric internal-shell (cavity) census. PURE, never raises.

    Returns a dict with:
      n_internal_shells : int   -- closed components nested inside the outer one
      internal_void_mm3 : float -- summed |enclosed volume| of those cavities
      outer_solid_mm3   : float -- |enclosed volume| of the outer shell as solid
      solid_volume_ratio: float -- max(0,(outer-void))/outer, in [0,1]; the
                                    100%-infill material ratio of a faithful
                                    print vs a cavity-filling auto-repair
                                    (a DEFENSIBLE UPPER BOUND on the gap)
      note              : str   -- honest, ascii-safe label

    Degenerate / non-watertight / single-shell inputs return
    n_internal_shells=0 and solid_volume_ratio=1.0 (no reportable cavity).
    """
    fail = {"n_internal_shells": 0, "internal_void_mm3": 0.0,
            "outer_solid_mm3": 0.0, "solid_volume_ratio": 1.0,
            "note": "geometric internal-shell estimate; no internal shell "
                    "detected (single shell or non-watertight input)."}
    try:
        if mesh is None or not hasattr(mesh, "faces") or len(mesh.faces) == 0:
            return dict(fail)

        # Split into connected components (do NOT require watertight — we want
        # every piece, then test closedness ourselves).
        try:
            comps = mesh.split(only_watertight=False)
        except Exception:
            comps = None
        if comps is None or len(comps) < 2:
            return dict(fail)

        # Characterise each component: bbox, bbox-diagonal volume, closedness,
        # and signed/enclosed volume magnitude.
        parts = []
        for c in comps:
            try:
                if c is None or not hasattr(c, "faces") or len(c.faces) == 0:
                    continue
                b = c.bounds  # ((minx,miny,minz),(maxx,maxy,maxz))
                if b is None:
                    continue
                ext = [float(b[1][k] - b[0][k]) for k in range(3)]
                bbox_vol = ext[0] * ext[1] * ext[2]
                closed = bool(getattr(c, "is_watertight", False))
                try:
                    encl = abs(float(c.volume)) if closed else 0.0
                except Exception:
                    encl = 0.0
                parts.append({"comp": c, "bounds": b, "bbox_vol": bbox_vol,
                              "closed": closed, "encl": encl})
            except Exception:
                continue
        if len(parts) < 2:
            return dict(fail)

        # OUTER shell = the component whose AABB contains the others; use the
        # largest bbox volume as the selector (ties broken by enclosed volume).
        outer = max(parts, key=lambda p: (p["bbox_vol"], p["encl"]))

        # Enclosed volume of the outer shell treated as SOLID. If the outer
        # shell isn't closed we fall back to its bbox volume so the ratio stays
        # well-defined (labelled honestly in the note either way).
        if outer["closed"] and outer["encl"] > _EPS:
            outer_solid = outer["encl"]
        else:
            outer_solid = outer["bbox_vol"]
        if outer_solid <= _EPS:
            return dict(fail)

        # Tolerance scaled to the outer shell so "strictly inside" is robust to
        # float noise (1e-6 of the longest bbox side).
        omin, omax = outer["bounds"]
        longest = max(float(omax[k] - omin[k]) for k in range(3))
        tol = max(_EPS, 1e-6 * longest)

        internal_void = 0.0
        n_internal = 0
        for p in parts:
            if p is outer:
                continue
            if not p["closed"] or p["encl"] <= _EPS:
                continue  # open flap / degenerate piece is not a sealed cavity
            if _bbox_inside(p["bounds"], outer["bounds"], tol):
                internal_void += p["encl"]
                n_internal += 1

        if n_internal == 0:
            return dict(fail)

        ratio = max(0.0, (outer_solid - internal_void)) / outer_solid
        # Clamp for safety (nested/overlapping cavities could overshoot).
        if ratio > 1.0:
            ratio = 1.0

        note = ("geometric internal-shell estimate (component split + bbox "
                "nesting), NOT a slicer measurement. solid_volume_ratio is the "
                "100%-infill material ratio of a faithful print vs a "
                "cavity-filling auto-repair -- a defensible UPPER BOUND on the "
                "gap; at realistic low infill the actual percentage gap is "
                "smaller.")
        return {"n_internal_shells": int(n_internal),
                "internal_void_mm3": float(internal_void),
                "outer_solid_mm3": float(outer_solid),
                "solid_volume_ratio": float(ratio),
                "note": note}
    except Exception:
        # never-raise: any surprise degrades to "no reportable cavity"
        return dict(fail)
