"""Warp calibration — turn the UNCALIBRATED inherent-strain warp estimate into a
CERTIFIED millimetre for ONE printer + filament + profile, from a single test bar.

WHY THIS IS A ONE-SCALAR FIT (validated in selftest, not asserted):
Inherent-strain warp is LINEAR in the eigenstrain eps* = -alpha * dT_lock (linear
elasticity), so for any part:  corner_lift = scale * FEM_lift,  with a single
dimensionless `scale` per material/process. That `scale` absorbs ALL the lumped
magnitude error — CTE uncertainty, the dT_lock choice, and Q1-hex shear-locking —
while the FEM keeps the GEOMETRY dependence (which is exactly what lets the one
scale fitted on a coupon bar transfer to your real part). Calibrate and predict at
the SAME max_elem.

REAL-WORLD WORKFLOW (the single coupon that certifies a material):
  1. Print one flat bar (default 100 x 30 x 6 mm) in your material/printer/profile.
  2. Measure the max corner lift off the bed with calipers (mm).
  3. calibrate(measured_lift_mm, bar_dims_mm=(100,30,6), material="PLA",
              label="Ender3-PLA-Hatchbox-0.2")        # stores the scale to JSON
  4. Thereafter calibrated_warp(mesh, "PLA", label=...)  — or preflight's warp
     channel — reports a CERTIFIED mm for that exact profile.

CLI:
  python -m meshprep.core._warp_calibrate --measured 0.8 --bar 100x30x6 --material PLA --label Ender3-PLA
  python -m meshprep.core._warp_calibrate --selftest
  python -m meshprep.core._warp_calibrate --list

HONEST SCOPE: one scale per (printer, filament, profile). The selftest proves the
machinery (a) is justified by warp's LINEARITY in dT_lock, (b) recovers a known
scale, and (c) transfers that scale across GEOMETRY *within the model*. It does NOT
prove a single scale captures every real-world geometry-dependent effect (warp
onset, bed adhesion, contact) — a SECOND coupon geometry would test that, and is
recommended for a high-stakes guarantee. Permissive deps only.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

import numpy as np
import trimesh

from . import _print_fem as pf
from . import fem_materials as fm

# The WRITABLE calibration store lives in the user's data dir (~/.meshprep/),
# never inside site-packages. The bundled (read-only, usually empty) package
# data file serves only as a seed/fallback for reads.
CALIB_PATH = os.path.join(os.path.expanduser("~"), ".meshprep",
                          "warp_calibration.json")
_BUNDLED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "warp_calibration.json")
DEFAULT_BAR_MM = (100.0, 30.0, 6.0)


# --- store -----------------------------------------------------------------------
def _load_store() -> dict:
    for path in (CALIB_PATH, _BUNDLED_PATH):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return {}


def _save_store(d: dict) -> bool:
    try:
        os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
        with open(CALIB_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        return True
    except Exception:
        return False


def _key(material: str, label) -> str:
    return f"{str(material).lower()}::{label or 'default'}"


# --- FEM on a test bar -----------------------------------------------------------
def _bar(dims_mm) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=[float(x) for x in dims_mm])


def predict_bar_lift(bar_dims_mm=DEFAULT_BAR_MM, material="PLA", *,
                     max_elem=4000, build_dir=None):
    """FEM-predicted max corner lift (mm) for a flat bar. Returns (lift_mm, result)."""
    r = pf.warp_analysis(_bar(bar_dims_mm), material=material,
                         build_dir=build_dir, max_elem=max_elem)
    if not r.get("ok"):
        return None, r
    return float(r.get("max_corner_lift_mm")), r


# --- calibrate from a measured coupon -------------------------------------------
def calibrate(measured_lift_mm, *, bar_dims_mm=DEFAULT_BAR_MM, material="PLA",
              label=None, max_elem=4000, store=True) -> dict:
    """Fit the one warp scale from a measured test-bar corner lift. Stores it
    keyed by (material, label) so future predictions for that profile are certified."""
    measured = float(measured_lift_mm)
    pred, r = predict_bar_lift(bar_dims_mm, material, max_elem=max_elem)
    if pred is None or pred <= 0:
        return {"ok": False, "reason": "FEM predicted ~0 lift on the bar "
                "(check bar geometry / material; resin has no thermal warp)"}
    scale = measured / pred
    rec = {
        "ok": True, "material": material, "label": label or "default",
        "scale": scale, "measured_lift_mm": measured, "fem_lift_mm": pred,
        "bar_dims_mm": [float(x) for x in bar_dims_mm], "max_elem": int(max_elem),
        "dT_lock_default_K": r.get("dT_lock_K"),
        "dT_lock_effective_K": (r.get("dT_lock_K") or 0.0) * scale,
        "note": ("scale multiplies the FEM corner-lift to a certified mm for THIS "
                 "printer+filament+profile; calibrate and predict at the same max_elem."),
    }
    if store:
        d = _load_store()
        d[_key(material, label)] = rec
        rec["saved_to"] = CALIB_PATH if _save_store(d) else None
    return rec


def get_scale(material="PLA", label=None):
    """Return (scale, label) for a profile, or (1.0, None) if uncalibrated.
    Falls back from a specific label to the material-level 'default' entry."""
    d = _load_store()
    r = d.get(_key(material, label)) or d.get(_key(material, None))
    if r and r.get("ok", True):
        return float(r["scale"]), r.get("label")
    return 1.0, None


# --- calibration-aware warp ------------------------------------------------------
def calibrated_warp(mesh, material="PLA", build_dir=None, label=None, **kw):
    """warp_analysis, but if a calibration exists for (material, label) the corner
    lift is scaled to a CERTIFIED mm and the result is re-flagged calibrated.
    Returns a PrintFEMResult (same shape as warp_analysis)."""
    r = pf.warp_analysis(mesh, material=material, build_dir=build_dir, **kw)
    if not r.get("ok"):
        return r
    scale, lab = get_scale(material, label)
    d = r.data
    if scale != 1.0:
        d["max_corner_lift_mm"] = float(d["max_corner_lift_mm"]) * scale
        d["corner_lift_mm"] = {k: float(v) * scale
                               for k, v in d.get("corner_lift_mm", {}).items()}
        if isinstance(d.get("max_uplift_mm"), (int, float)):
            d["max_uplift_mm"] = float(d["max_uplift_mm"]) * scale
        d.update(calibrated=True, cal_scale=scale, cal_label=lab, uncalibrated=False,
                 magnitude_kind=f"CALIBRATED for '{lab}' (scale {scale:.3f})",
                 claim=(f"CALIBRATED magnitude for profile '{lab}': corner lift fitted "
                        "to a measured coupon. Sign/shape from the validated FEM, "
                        "magnitude from your 1-bar fit. Assumes one scale transfers "
                        "across geometry (a 2nd coupon would confirm)."))
    else:
        d["calibrated"] = False
    return r


# --- validation (the honest gate, runnable without a printer) -------------------
def selftest(max_elem=3000, verbose=True):
    """Validate the calibration MACHINERY against the FEM (no printer needed):
      1. LINEARITY  — warp ∝ dT_lock (justifies a single scale).
      2. ROUND-TRIP — calibrate on a synthetic measurement, recover the scale.
      3. CROSS-GEOM — a bar-fitted scale recovers a DIFFERENT geometry's warp.
    """
    out = {}
    PASS = True
    base, _ = fm.get_material("PLA")
    barA = DEFAULT_BAR_MM

    # 1) LINEARITY: lift(dT)/dT constant  (K identical, RHS ∝ dT -> u ∝ dT)
    lifts = {}
    for dT in (5.0, 10.0, 20.0):
        mat = dataclasses.replace(base, Tg_C=base.T_bed_C + dT)   # locked_delta_T() == dT
        lifts[dT] = float(pf.warp_analysis(_bar(barA), material=mat,
                                           max_elem=max_elem).get("max_corner_lift_mm"))
    ratios = [lifts[dT] / dT for dT in lifts]
    spread = (max(ratios) - min(ratios)) / (np.mean(ratios) + 1e-30)
    lin_ok = spread < 0.02
    out["linearity"] = {"lifts_mm": lifts, "lift_per_K": ratios,
                        "spread": float(spread), "ratio_10_over_5": lifts[10.0] / lifts[5.0],
                        "ok": bool(lin_ok)}
    PASS &= lin_ok

    # 2) ROUND-TRIP: synthetic 'measurement' = FEM * true_scale -> recover true_scale
    pred, _ = predict_bar_lift(barA, "PLA", max_elem=max_elem)
    true_scale = 1.8
    rec = calibrate(pred * true_scale, bar_dims_mm=barA, material="PLA",
                    label="_selftest", max_elem=max_elem, store=True)
    rt_ok = abs(rec["scale"] - true_scale) < 1e-6
    out["round_trip"] = {"true_scale": true_scale, "recovered": rec["scale"], "ok": bool(rt_ok)}
    PASS &= rt_ok

    # 3) CROSS-GEOMETRY: apply the bar-fitted scale to a DIFFERENT geometry.
    #    "truth" for B = the same FEM run with dT_lock raised by true_scale.
    base_dT = base.locked_delta_T()
    matB_true = dataclasses.replace(base, Tg_C=base.T_bed_C + base_dT * true_scale)
    barB = (120.0, 40.0, 8.0)
    true_B = float(pf.warp_analysis(_bar(barB), material=matB_true,
                                    max_elem=max_elem).get("max_corner_lift_mm"))
    def_B = float(pf.warp_analysis(_bar(barB), material="PLA",
                                   max_elem=max_elem).get("max_corner_lift_mm"))
    cal_B = def_B * rec["scale"]
    err_def = abs(def_B - true_B) / (true_B + 1e-30)
    err_cal = abs(cal_B - true_B) / (true_B + 1e-30)
    cross_ok = (err_cal < err_def) and (err_cal < 0.02)
    out["cross_geometry"] = {"true_mm": true_B, "default_uncal_mm": def_B,
                             "calibrated_mm": cal_B, "err_default": float(err_def),
                             "err_calibrated": float(err_cal), "ok": bool(cross_ok)}
    PASS &= cross_ok

    # clean up the selftest store entry
    d = _load_store(); d.pop(_key("PLA", "_selftest"), None); _save_store(d)

    out["ALL_PASS"] = bool(PASS)
    if verbose:
        print("warp-calibration selftest")
        print(f"  1. LINEARITY  lift/dT spread={spread*100:.2f}% (lift prop-to dT_lock) "
              f"ratio(10/5)={lifts[10.0]/lifts[5.0]:.3f}  -> {'PASS' if lin_ok else 'FAIL'}")
        print(f"  2. ROUND-TRIP true_scale={true_scale} recovered={rec['scale']:.4f}  "
              f"-> {'PASS' if rt_ok else 'FAIL'}")
        print(f"  3. CROSS-GEOM (different bar) true={true_B:.3f}mm | "
              f"uncalibrated={def_B:.3f}mm (err {err_def*100:.0f}%) -> "
              f"calibrated={cal_B:.3f}mm (err {err_cal*100:.2f}%)  "
              f"-> {'PASS' if cross_ok else 'FAIL'}")
        print("  ALL_PASS" if PASS else "  FAILED")
    return out


def _parse_bar(s):
    try:
        return tuple(float(x) for x in s.lower().replace("mm", "").split("x"))
    except Exception:
        return DEFAULT_BAR_MM


def _main(argv):
    import argparse
    from ._mesh_util import ascii_console
    ascii_console()
    ap = argparse.ArgumentParser(description="Warp calibration from one test-bar print.")
    ap.add_argument("--measured", type=float, help="measured corner lift (mm) from your printed bar")
    ap.add_argument("--bar", default="100x30x6", help="bar dims mm, e.g. 100x30x6")
    ap.add_argument("--material", default="PLA")
    ap.add_argument("--label", default=None, help="printer+filament+profile tag")
    ap.add_argument("--max-elem", type=int, default=4000)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest(); return
    if a.list:
        print(json.dumps(_load_store(), indent=2)); return
    if a.measured is not None:
        rec = calibrate(a.measured, bar_dims_mm=_parse_bar(a.bar), material=a.material,
                        label=a.label, max_elem=a.max_elem)
        print(json.dumps(rec, indent=2))
        return
    ap.print_help()


if __name__ == "__main__":
    _main(sys.argv[1:])
