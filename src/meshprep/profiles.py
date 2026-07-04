"""Printer presets + bed-fit / scale advisor.

`PRINTER_PROFILES` maps a printer name to its physical envelope and sane
defaults: `bed_mm` (x, y, z build volume), `nozzle_mm` (FDM) or `pixel_mm`
(resin XY pixel), default `layer_mm`, default `material`, and `technology`
('fdm' | 'resin'). Volumes are the manufacturers' PUBLISHED build volumes; the
really usable area is a little smaller (skirt/brim, clips, purge), which is why
`scale_to_fit` keeps a margin.

* `get_profile(name)` NEVER raises: unknown / None / garbage -> `generic_fdm`
  (a note in the returned dict says the fallback fired).
* `scale_to_fit(mesh, profile)` -> does-it-fit + suggested uniform scale.
  HONESTY: it tests the axis-aligned bbox in all 6 axis permutations (i.e.
  90-degree re-orientations only). A diagonal placement can fit slightly more;
  this is bed-fit TRIAGE, not an optimal-packing solver. It also assumes the
  mesh is in mm (the house unit convention everywhere in meshprep).

Pure stdlib + (lazy) numpy. Never raises from any public entry point.
Run as a script for the self-tests:  python -m meshprep.profiles
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
#  presets
# ---------------------------------------------------------------------------
PRINTER_PROFILES = {
    "ender3": {
        "display": "Creality Ender 3 (V2/Neo class)",
        "technology": "fdm", "bed_mm": (220.0, 220.0, 250.0),
        "nozzle_mm": 0.4, "layer_mm": 0.2, "material": "PLA",
        "notes": "open frame -- PLA/PETG; ABS warps without an enclosure",
    },
    "prusa_mk4": {
        "display": "Prusa MK4/MK4S",
        "technology": "fdm", "bed_mm": (250.0, 210.0, 220.0),
        "nozzle_mm": 0.4, "layer_mm": 0.2, "material": "PLA",
        "notes": "open frame; input-shaper firmware",
    },
    "bambu_a1": {
        "display": "Bambu Lab A1",
        "technology": "fdm", "bed_mm": (256.0, 256.0, 256.0),
        "nozzle_mm": 0.4, "layer_mm": 0.2, "material": "PLA",
        "notes": "open frame -- PLA/PETG",
    },
    "bambu_x1c": {
        "display": "Bambu Lab X1 Carbon",
        "technology": "fdm", "bed_mm": (256.0, 256.0, 256.0),
        "nozzle_mm": 0.4, "layer_mm": 0.2, "material": "PLA",
        "notes": "enclosed -- ABS/ASA/PA-CF capable; default kept PLA",
    },
    "generic_fdm": {
        "display": "Generic FDM (200 mm cube)",
        "technology": "fdm", "bed_mm": (200.0, 200.0, 200.0),
        "nozzle_mm": 0.4, "layer_mm": 0.2, "material": "PLA",
        "notes": "conservative default when the printer is unknown",
    },
    "generic_resin_msla": {
        "display": "Generic MSLA resin (mid-size 4K mono LCD class)",
        "technology": "resin", "bed_mm": (130.0, 80.0, 160.0),
        "pixel_mm": 0.05, "layer_mm": 0.05, "material": "standard resin",
        "notes": "Mars/Photon-class envelope; pixel = XY voxel of the LCD",
    },
}

# common spellings -> canonical key (lookup is lowercased + [-_ .] stripped)
_ALIASES = {
    "ender": "ender3", "ender3v2": "ender3", "ender3neo": "ender3",
    "mk4": "prusa_mk4", "prusamk4": "prusa_mk4", "prusamk4s": "prusa_mk4",
    "prusa": "prusa_mk4",
    "a1": "bambu_a1", "bambua1": "bambu_a1",
    "x1c": "bambu_x1c", "x1carbon": "bambu_x1c", "bambux1c": "bambu_x1c",
    "bambux1carbon": "bambu_x1c",
    "fdm": "generic_fdm", "generic": "generic_fdm",
    "resin": "generic_resin_msla", "sla": "generic_resin_msla",
    "msla": "generic_resin_msla", "genericresin": "generic_resin_msla",
    "genericresinmsla": "generic_resin_msla",
}

DEFAULT_PROFILE = "generic_fdm"


def _canon(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def get_profile(name=None):
    """Return a COPY of the named profile dict with `name` filled in.
    Never raises: unknown / None -> generic_fdm (+ a fallback note).
    A dict passed in is normalized (missing fields defaulted) and passed through,
    so callers can hand-roll a custom printer."""
    if isinstance(name, dict):                       # custom profile passthrough
        p = dict(PRINTER_PROFILES[DEFAULT_PROFILE])
        p.update(name)
        p["name"] = str(name.get("name", "custom"))
        return p
    key = _canon(name) if name is not None else DEFAULT_PROFILE
    key = _ALIASES.get(key, key)
    if key not in PRINTER_PROFILES:
        for k in PRINTER_PROFILES:                   # canon-form direct keys
            if _canon(k) == key:
                key = k
                break
        else:
            p = dict(PRINTER_PROFILES[DEFAULT_PROFILE])
            p["name"] = DEFAULT_PROFILE
            p["fallback_note"] = f"unknown printer {name!r} -> {DEFAULT_PROFILE}"
            return p
    p = dict(PRINTER_PROFILES[key])
    p["name"] = key
    return p


# ---------------------------------------------------------------------------
#  fit check + scale advisor
# ---------------------------------------------------------------------------
_PERMS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


def scale_to_fit(mesh, profile=None, margin_mm=1.0):
    """Does `mesh` (mm) fit the profile's bed, and what uniform scale would make
    it fit best? Tries all 6 axis permutations (90-degree re-orientations).

    Returns a dict (never raises):
      ok, fits, bbox_mm, bed_mm, usable_mm, max_scale (largest uniform scale
      that still fits, >1 means headroom), suggested_scale (1.0 if it fits,
      else the shrink factor), best_permutation (mesh axes -> bed x,y,z),
      limiting_axis ('x'|'y'|'z' of the BED), profile, notes.
    """
    out = {"ok": False, "fits": None, "notes": []}
    try:
        prof = get_profile(profile)
        out["profile"] = prof["name"]
        if "fallback_note" in prof:
            out["notes"].append(prof["fallback_note"])
        bed = [float(v) for v in prof["bed_mm"]]
        m = float(margin_mm)
        # margin both sides in x/y (part is centered), top only in z
        usable = [max(bed[0] - 2 * m, 1e-6), max(bed[1] - 2 * m, 1e-6),
                  max(bed[2] - m, 1e-6)]
        ext = [float(v) for v in mesh.bounding_box.extents]
        if min(ext) <= 0 or not all(e == e for e in ext):
            out["notes"].append("degenerate bbox")
            return out
        best_s, best_p, best_ax = -1.0, _PERMS[0], 2
        for p in _PERMS:
            ratios = [usable[i] / ext[p[i]] for i in range(3)]
            s = min(ratios)
            if s > best_s:
                best_s, best_p, best_ax = s, p, int(ratios.index(s))
        fits = best_s >= 1.0
        out.update({
            "ok": True, "fits": bool(fits),
            "bbox_mm": [round(e, 2) for e in ext],
            "bed_mm": [round(b, 1) for b in bed],
            "usable_mm": [round(u, 1) for u in usable],
            "margin_mm": m,
            "max_scale": round(best_s, 4),
            "suggested_scale": 1.0 if fits else round(best_s, 4),
            "best_permutation": "".join("xyz"[i] for i in best_p),
            "limiting_axis": "xyz"[best_ax],
            "method": "axis-aligned bbox over 6 axis permutations "
                      "(90-degree re-orientations only; triage, not packing)",
        })
        if not fits:
            out["notes"].append(
                f"too big: scale by {out['suggested_scale']} "
                f"(mesh axes '{out['best_permutation']}' onto bed x,y,z; "
                f"bed {out['limiting_axis']}-axis limits)")
    except Exception as e:                            # never-raise discipline
        out["notes"].append(f"scale_to_fit failed: {e}")
    return out


# ---------------------------------------------------------------------------
#  self-tests
# ---------------------------------------------------------------------------
def run_self_tests(verbose=True):
    """Deterministic checks on primitives (no external data). (all_pass, details)."""
    from .core._mesh_util import say
    import trimesh
    details = {}
    ok_all = True

    # 1. fallback never raises
    g = get_profile("no_such_printer_9000")
    t1 = g["name"] == DEFAULT_PROFILE and "fallback_note" in g
    t1 &= get_profile(None)["name"] == DEFAULT_PROFILE
    t1 &= get_profile("X1C")["name"] == "bambu_x1c"        # alias, case
    t1 &= get_profile({"name": "big", "bed_mm": (500, 500, 500)})["bed_mm"][0] == 500
    details["test1_get_profile"] = bool(t1)
    ok_all &= t1

    # 2. 100 mm cube fits ender3 with headroom
    cube = trimesh.creation.box(extents=(100, 100, 100))
    r2 = scale_to_fit(cube, "ender3")
    t2 = r2["ok"] and r2["fits"] and r2["max_scale"] > 2.0
    details["test2_cube_fits"] = {"pass": bool(t2), "max_scale": r2.get("max_scale")}
    ok_all &= t2

    # 3. 240x10x10 bar: exceeds ender3 x/y (218 usable) but fits standing in z
    #    (249 usable) -> permutation logic must find it
    bar = trimesh.creation.box(extents=(240, 10, 10))
    r3 = scale_to_fit(bar, "ender3")
    t3 = r3["ok"] and r3["fits"] and r3["best_permutation"][2] == "x"
    details["test3_bar_permutation"] = {"pass": bool(t3),
                                        "perm": r3.get("best_permutation")}
    ok_all &= t3

    # 4. 300 mm bar does NOT fit; suggested scale puts it exactly at the cap
    bar2 = trimesh.creation.box(extents=(300, 10, 10))
    r4 = scale_to_fit(bar2, "ender3")
    s = r4.get("suggested_scale", 0)
    t4 = r4["ok"] and not r4["fits"] and 0 < s < 1 and abs(s * 300 - 249) < 1.0
    details["test4_bar_too_big"] = {"pass": bool(t4), "suggested_scale": s}
    ok_all &= t4

    # 5. never-raise on garbage mesh
    r5 = scale_to_fit(None, "ender3")
    t5 = r5["ok"] is False and r5["notes"]
    details["test5_never_raise"] = bool(t5)
    ok_all &= bool(t5)

    if verbose:
        say("PROFILES SELF-TESTS")
        say(f"  1. get_profile fallback/alias/custom      -> {'PASS' if t1 else 'FAIL'}")
        say(f"  2. 100mm cube fits ender3 (max_scale {r2.get('max_scale')}) "
            f"-> {'PASS' if t2 else 'FAIL'}")
        say(f"  3. 240mm bar fits via permutation ({r3.get('best_permutation')}) "
            f"-> {'PASS' if t3 else 'FAIL'}")
        say(f"  4. 300mm bar suggested_scale {s} -> {'PASS' if t4 else 'FAIL'}")
        say(f"  5. never-raise on None mesh -> {'PASS' if t5 else 'FAIL'}")
        say(f"  ALL: {'PASS' if ok_all else 'FAIL'}")
    return bool(ok_all), details


if __name__ == "__main__":
    import sys as _sys
    from .core._mesh_util import ascii_console
    ascii_console()
    _ok, _d = run_self_tests(verbose=True)
    _sys.exit(0 if _ok else 1)
