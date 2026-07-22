"""printworthy.core._printability -- the shared PRINTABILITY / SOLIDITY GATE.

One pure function, `assess_printability(mesh)`, that every entry point (prep,
check, fix, reinforce) can consult BEFORE it emits a green "PASS" headline or a
structural factor-of-safety.  It answers a single blunt question a novice cares
about: *can this actually be printed, or is the tool about to lie to me?*

Design contract (do not weaken):
  * PURE.  It reads a mesh, computes cheap geometric facts, and returns a dict.
    It NEVER mutates the mesh and NEVER raises -- any internal error is caught
    and turned into an honest verdict="FAIL" with a plain-language issue.  A
    tool that refuses honestly is foolproof; a tool that ships a bad result
    silently is not.
  * It catches the exact "silent-wrong" modes the audit found:
      - inside-out solid (signed volume <= 0 while winding is consistent),
      - not a closed solid (not watertight / open shell with holes),
      - zero / degenerate faces that block slicing,
      - a whole-part wall thinner than the nozzle (the 60x0.02x0.02 sliver),
      - non-finite bounding box (NaN / inf coordinates),
      - a structural FEM that computed nothing (min_fos n/a, peak stress 0),
      - implausible scale (a model in metres pretending to be millimetres),
      - runaway / undocumented component or handle counts.
  * It NEVER upgrades a claim.  It can only refuse or warn.  Every honesty
    label the callers carry ("RELATIVE", "COMPARATIVE", "uncalibrated",
    "geometric") is left untouched -- the gate blocks *before* those numbers
    are shown, it does not restate or launder them.

Return schema (GATE CONTRACT):
  {
    "printable": bool,                 # False on FAIL; True on PASS/WARN
    "verdict":   "PASS"|"WARN"|"FAIL",
    "issues":    [ {code, severity: "fail"|"warn", message} ],  # plain English
    "checks":    { watertight, positive_volume, degenerate_faces, genus,
                   bbox_finite, max_extent_mm, scale_plausible_mm,
                   n_components, n_faces, ... },
    "plain_summary": str,              # ONE sentence a beginner understands
  }
"""
from __future__ import annotations

# Default fused-filament nozzle diameter (mm).  A wall thinner than this cannot
# be extruded -- prep/check/fix already FAIL on it; the gate makes reinforce
# honor the same floor.
_DEFAULT_NOZZLE_MM = 0.40

# Plausible longest-side extent for a desktop print, in mm.  Below the small
# bound a model is almost certainly in metres (or otherwise mis-scaled); above
# the large bound it will not fit any hobby printer -- both are WARN, not FAIL.
_SCALE_MIN_MM = 2.0
_SCALE_MAX_MM = 1000.0

# Above this many separate pieces, a mesh is more likely shattered/broken than
# an intentional multi-part model -- WARN so a novice looks.
_MANY_COMPONENTS = 24

# A print with more than this many handles/tunnels is almost always a topology
# artifact rather than intent -- WARN (non-blocking).
_RUNAWAY_GENUS = 50


def _finite(x) -> bool:
    try:
        x = float(x)
        return x == x and x not in (float("inf"), float("-inf"))
    except Exception:
        return False


def _plain(issues, verdict, checks) -> str:
    """ONE beginner sentence.  Leads with the blocking reason on FAIL, the top
    caution on WARN, else a reassuring PASS line.  ASCII only (console-safe)."""
    if verdict == "FAIL":
        for it in issues:
            if it.get("severity") == "fail":
                return "Not printable yet: " + it["message"]
        return "Not printable yet: the file could not be read as a solid part."
    if verdict == "WARN":
        for it in issues:
            if it.get("severity") == "warn":
                return "Probably printable, but check this first: " + it["message"]
    ext = checks.get("max_extent_mm")
    ext_s = f"{ext:.0f} mm" if isinstance(ext, (int, float)) else "its measured size"
    return ("Looks print-ready: it is one closed, solid shape ("
            + ext_s + " on its longest side) with no blocking problems.")


def assess_printability(mesh, *, assume_unit="mm", context=None) -> dict:
    """Assess whether `mesh` can be 3-D printed.  PURE; never raises.

    Parameters
    ----------
    mesh : anything exposing the trimesh.Trimesh API (is_watertight, volume,
           bounds, area_faces, euler_number, body_count).
    assume_unit : the unit the caller believes the mesh is in.  Only "mm" (the
           default domain) triggers the metres/scale warning; any explicit
           non-"mm" override means the caller has declared their units, so the
           scale caution is suppressed (units honored, never silently rescaled).
    context : optional dict.  Recognised keys, all optional:
           "nozzle_mm"  -> override the min-printable wall (default 0.40).
           "fem"        -> {min_fos, peak_vm, degenerate_core_frac, ...} from a
                           structural run; the gate flags a FEM that computed
                           nothing (min_fos n/a, peak stress 0, all-degenerate
                           core) so a bogus factor-of-safety is never shipped.

    Returns the GATE CONTRACT dict (see module docstring).
    """
    context = context if isinstance(context, dict) else {}
    issues: list[dict] = []
    checks: dict = {
        "watertight": None, "positive_volume": None, "degenerate_faces": None,
        "genus": None, "bbox_finite": None, "max_extent_mm": None,
        "min_extent_mm": None, "scale_plausible_mm": None,
        "n_components": None, "n_faces": None,
    }

    def fail(code, msg):
        issues.append({"code": code, "severity": "fail", "message": msg})

    def warn(code, msg):
        issues.append({"code": code, "severity": "warn", "message": msg})

    try:
        import numpy as _np
    except Exception:
        _np = None

    try:
        nozzle_mm = float(context.get("nozzle_mm", _DEFAULT_NOZZLE_MM))
        if not _finite(nozzle_mm) or nozzle_mm <= 0:
            nozzle_mm = _DEFAULT_NOZZLE_MM

        # ---- face count -----------------------------------------------------
        try:
            n_faces = int(len(mesh.faces))
        except Exception:
            n_faces = 0
        checks["n_faces"] = n_faces
        if n_faces == 0:
            checks["degenerate_faces"] = 0
            fail("no_faces",
                 "the file has no triangles at all, so there is nothing to "
                 "slice or print.")

        # ---- bounding box / extents ----------------------------------------
        bbox_finite = False
        max_extent = None
        min_extent = None
        try:
            bounds = mesh.bounds  # (2,3) min/max
            lo = [float(v) for v in bounds[0]]
            hi = [float(v) for v in bounds[1]]
            allc = lo + hi
            bbox_finite = all(_finite(c) for c in allc)
            if bbox_finite:
                ext = [hi[i] - lo[i] for i in range(len(hi))]
                max_extent = max(ext) if ext else 0.0
                min_extent = min(ext) if ext else 0.0
        except Exception:
            bbox_finite = False
        checks["bbox_finite"] = bool(bbox_finite)
        checks["max_extent_mm"] = max_extent
        checks["min_extent_mm"] = min_extent
        if not bbox_finite:
            fail("bbox_nonfinite",
                 "the model has broken (not-a-number or infinite) coordinates, "
                 "so its size and shape cannot be trusted.")

        # ---- degenerate faces ----------------------------------------------
        deg = 0
        try:
            areas = mesh.area_faces
            if _np is not None:
                areas = _np.asarray(areas, dtype=float)
                scale = max_extent if isinstance(max_extent, (int, float)) and max_extent > 0 else 1.0
                tiny = max(1e-15, (scale * scale) * 1e-12)
                deg = int((areas <= tiny).sum())
            else:
                deg = sum(1 for a in areas if float(a) <= 1e-15)
        except Exception:
            deg = 0
        checks["degenerate_faces"] = deg
        if n_faces > 0 and deg >= n_faces:
            fail("all_degenerate",
                 "every triangle in the model is flat/zero-area, so it has no "
                 "real surface a printer can build.")
        elif n_faces > 0 and deg > 0 and deg >= 0.5 * n_faces:
            warn("many_degenerate",
                 f"{deg} of {n_faces} triangles are zero-area (flat slivers); a "
                 "slicer may skip them -- have the mesh cleaned up.")

        # ---- watertight (closed solid) -------------------------------------
        try:
            watertight = bool(getattr(mesh, "is_watertight", False))
        except Exception:
            watertight = False
        checks["watertight"] = watertight
        if not watertight:
            fail("not_watertight",
                 "this is not a closed solid -- the surface has gaps or holes, "
                 "so a slicer cannot tell the inside from the outside.")

        # ---- signed volume / inside-out ------------------------------------
        try:
            winding = bool(getattr(mesh, "is_winding_consistent", True))
        except Exception:
            winding = True
        signed_vol = None
        try:
            signed_vol = float(mesh.volume)
            if not _finite(signed_vol):
                signed_vol = None
        except Exception:
            signed_vol = None
        checks["signed_volume_mm3"] = signed_vol
        checks["positive_volume"] = (signed_vol is not None and signed_vol > 0)
        # Inside-out is only diagnosable on a closed solid with a defined volume.
        if watertight and signed_vol is not None:
            if signed_vol < 0 and winding:
                fail("inside_out",
                     "the shape is inside-out -- its surfaces face inward "
                     "(negative volume); the walls all point the wrong way. "
                     "Full prep flips them automatically.")
            elif abs(signed_vol) <= 1e-9:
                fail("zero_volume",
                     "the solid encloses no measurable volume (it is flat or "
                     "collapsed), so there is nothing solid to print.")

        # ---- sub-nozzle whole-part wall (the sliver) -----------------------
        if isinstance(min_extent, (int, float)) and bbox_finite and n_faces > 0:
            if min_extent < nozzle_mm:
                fail("sub_nozzle_extent",
                     f"the part's thinnest dimension is {min_extent:.3g} mm, "
                     f"below the {nozzle_mm:.2f} mm nozzle -- the printer "
                     "cannot make a wall this thin, so it cannot be printed or "
                     "reinforced.")

        # ---- genus (handles / tunnels) -------------------------------------
        genus = None
        try:
            if watertight:
                euler = int(mesh.euler_number)
                genus = (2 - euler) // 2
        except Exception:
            genus = None
        checks["genus"] = genus
        if isinstance(genus, int) and genus > _RUNAWAY_GENUS:
            warn("runaway_genus",
                 f"the model reports {genus} handles/tunnels, which usually "
                 "means broken topology rather than an intended shape -- worth "
                 "a look before printing.")

        # ---- components ----------------------------------------------------
        n_comp = None
        try:
            n_comp = int(getattr(mesh, "body_count"))
        except Exception:
            try:
                n_comp = int(len(mesh.split(only_watertight=False)))
            except Exception:
                n_comp = None
        checks["n_components"] = n_comp
        if isinstance(n_comp, int) and n_comp > _MANY_COMPONENTS:
            warn("many_components",
                 f"the model is made of {n_comp} separate pieces; if you did "
                 "not intend a multi-part model this often means the mesh is "
                 "shattered -- check before printing.")

        # ---- scale plausibility (units) ------------------------------------
        scale_ok = True
        if isinstance(max_extent, (int, float)) and bbox_finite:
            if str(assume_unit).lower() == "mm":
                if max_extent < _SCALE_MIN_MM:
                    scale_ok = False
                    warn("scale_too_small",
                         f"the whole model is only {max_extent:.3g} mm across; "
                         "that is smaller than a grain of rice, so it was very "
                         "likely modelled in metres -- scale it up (x1000?) "
                         "before printing. (Nothing was rescaled for you.)")
                elif max_extent > _SCALE_MAX_MM:
                    scale_ok = False
                    warn("scale_too_large",
                         f"the model is {max_extent:.0f} mm across -- larger "
                         "than any desktop printer bed; check whether the units "
                         "are right before printing.")
        checks["scale_plausible_mm"] = bool(scale_ok)

        # ---- FEM sanity (reinforce path) -----------------------------------
        fem = context.get("fem")
        if isinstance(fem, dict):
            mf = fem.get("min_fos")
            pv = fem.get("peak_vm", fem.get("peak_vm_Pa"))
            dcf = fem.get("degenerate_core_frac")
            bad_mf = (mf is None) or (isinstance(mf, str)) or \
                     (isinstance(mf, float) and mf != mf)   # None / "n/a" / NaN
            bad_pv = isinstance(pv, (int, float)) and _finite(pv) and pv == 0
            bad_core = isinstance(dcf, (int, float)) and dcf >= 0.99
            if bad_mf or bad_pv or bad_core:
                fail("fem_no_result",
                     "the structural check produced no usable result (the part "
                     "was too degenerate to simulate), so any factor-of-safety "
                     "here would be meaningless -- it will not be reported.")

    except Exception as e:  # absolute backstop: the gate itself must never raise
        issues.append({
            "code": "gate_error", "severity": "fail",
            "message": ("the printability check could not finish ("
                        + type(e).__name__ + "), so this model is treated as "
                        "not-yet-printable to be safe."),
        })

    # ---- fold issues into a verdict ---------------------------------------
    has_fail = any(i.get("severity") == "fail" for i in issues)
    has_warn = any(i.get("severity") == "warn" for i in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")
    printable = (verdict != "FAIL")

    return {
        "printable": printable,
        "verdict": verdict,
        "issues": issues,
        "checks": checks,
        "plain_summary": _plain(issues, verdict, checks),
    }


# ---------------------------------------------------------------------------
#  selftest -- proves the gate FAILs an inside-out solid and a holed shell and
#  PASSes a clean cube.  Run:  python -m printworthy.core._printability
# ---------------------------------------------------------------------------
def selftest() -> bool:
    from ._mesh_util import ascii_console, say
    ascii_console()
    ok = True

    try:
        import trimesh
        import numpy as np
    except Exception as e:
        say(f"[selftest] trimesh/numpy unavailable ({e}); cannot run.")
        return False

    def check(label, cond):
        nonlocal ok
        ok = ok and bool(cond)
        say(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    # 1) clean cube -> PASS
    cube = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    r = assess_printability(cube)
    say("clean cube        ->", r["verdict"], "|", r["plain_summary"])
    check("clean cube is PASS", r["verdict"] == "PASS")
    check("clean cube printable", r["printable"] is True)
    check("clean cube positive volume", r["checks"]["positive_volume"] is True)

    # 2) inside-out cube (uniformly flipped winding, negative volume) -> FAIL
    inv = cube.copy()
    inv.invert()
    r = assess_printability(inv)
    say("inside-out cube   ->", r["verdict"], "|", r["plain_summary"])
    check("inside-out cube is FAIL", r["verdict"] == "FAIL")
    check("inside-out flagged inside_out",
          any(i["code"] == "inside_out" for i in r["issues"]))
    check("inside-out positive_volume False", r["checks"]["positive_volume"] is False)

    # 3) holed shell (a face removed) -> FAIL (not a closed solid)
    holed = cube.copy()
    holed.update_faces(np.arange(len(holed.faces)) != 0)
    r = assess_printability(holed)
    say("holed shell       ->", r["verdict"], "|", r["plain_summary"])
    check("holed shell is FAIL", r["verdict"] == "FAIL")
    check("holed shell flagged not_watertight",
          any(i["code"] == "not_watertight" for i in r["issues"]))

    # 4) extreme-aspect sliver 60 x 0.02 x 0.02 -> FAIL (sub-nozzle wall)
    sliver = trimesh.creation.box(extents=[60.0, 0.02, 0.02])
    r = assess_printability(sliver)
    say("sliver 60x.02x.02 ->", r["verdict"], "|", r["plain_summary"])
    check("sliver is FAIL", r["verdict"] == "FAIL")
    check("sliver flagged sub_nozzle_extent",
          any(i["code"] == "sub_nozzle_extent" for i in r["issues"]))

    # 5) FEM-computed-nothing via context -> FAIL, even on an otherwise fine cube
    r = assess_printability(cube, context={"fem": {"min_fos": None, "peak_vm": 0.0}})
    say("cube + dead FEM   ->", r["verdict"], "|", r["plain_summary"])
    check("dead-FEM is FAIL", r["verdict"] == "FAIL")
    check("dead-FEM flagged fem_no_result",
          any(i["code"] == "fem_no_result" for i in r["issues"]))

    # 6) metres-scale cube: a 1.5 m part read as 1.5 mm -- too small to be a
    #    real mm print, but each wall (1.5 mm) is still above the nozzle, so the
    #    only signal is the scale WARN (not a sub-nozzle FAIL).
    tiny = trimesh.creation.box(extents=[1.5, 1.5, 1.5])
    r = assess_printability(tiny)
    say("metres-scale cube ->", r["verdict"], "|", r["plain_summary"])
    check("metres-scale is WARN", r["verdict"] == "WARN")
    check("metres-scale flagged scale_too_small",
          any(i["code"] == "scale_too_small" for i in r["issues"]))
    # explicit unit override suppresses the scale warning (units honored)
    r2 = assess_printability(tiny, assume_unit="m")
    check("explicit assume_unit='m' clears scale warn",
          not any(i["code"].startswith("scale_") for i in r2["issues"]))

    say("")
    say("[selftest] OVERALL:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
