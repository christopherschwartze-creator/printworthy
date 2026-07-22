"""_print_fem.py -- the unified reduced-order "FEM for 3D printing" for EI Forge.

WHAT THIS IS
============
A single public entry point that wraps the THREE validated solver pieces into the
printability-checker API the Forge preflight consumes:

    warp_analysis(mesh, material, build_dir)  -> inherent-strain distortion,
                                                 residual-stress hotspots,
                                                 predicted max corner lift (mm)
    strength_analysis(mesh, material, load_case, build_dir)
                                              -> orthotropic stress / FOS field
    orient_for_strength(mesh, load_case, material)
                                              -> best build orientation by min-FOS
    fem_mesh_quality(occ_or_mesh, ...)        -> sigma* certificate ON THE ANALYSIS
                                                 GRID + its HONEST verdict
    render_warp / render_strength / render_orient_compare
    selftest()                                -> ALL_PASS only if the three
                                                 ANALYTICAL benchmarks converge to
                                                 closed form (bar / cantilever /
                                                 eigenstrain bend).

It does NOT re-derive any physics. The solvers it calls were each independently
validated against closed forms under mesh refinement:
  * fem_voxel_core.VoxelHexFEM   : uniaxial bar (machine precision), cantilever
                                   (monotone 0.835 -> 0.980 toward Euler-Bernoulli).
  * fem_warp_probe.predict_warp  : Timoshenko bimetal eigenstrain + released-curl,
                                   both converge to closed form, correct sign.
  * fem_orthotropic.solve_part_stress : off-axis modulus (1.4% -> 0.52% monotone),
                                   anisotropy sign (across-Z weaker), orient works.
This module RE-RUNS those gates inside selftest() so the whole stack is
re-certified every time, not trusted by reference.

THE HONESTY GATE (non-negotiable; this project has a documented inflated-claim
history). selftest() returns ALL_PASS ONLY IF, freshly measured here:
  (B1) uniaxial bar  d = F L/(A E)        : rel_err < 1% (Q1 hex exact in tension).
  (B2) cantilever    d = P L^3/(3 E I)    : d_fem/d_EB monotone-increasing toward 1
                                            from below (Q1 shear-locking signature).
  (B3) eigenstrain bend (Timoshenko 1925) : curvature converges to 1.5*da*dT/h,
                                            sign correct, rel_err decreasing.
If any benchmark fails or stops converging, the verdict is GATE FAILED and NO
print number from this module may be trusted.

UNCALIBRATED-MODEL DISCIPLINE (label EVERY output)
==================================================
The SOLVERS are certified against exact closed forms. The PRINT PREDICTIONS are
NOT certified magnitudes: material constants (PLA E/CTE/dT_lock, Zt/Xt) are
mid-range literature (+/-20-40%), and the fully-integrated Q1 hex SHEAR-LOCKS in
bending so warp magnitude is a mesh-dependent LOWER bound. Therefore every output
dict carries `uncalibrated=True` and a `claim` string spelling out what is
certified (SIGN / SHAPE / TREND / order of magnitude) versus what is not (a
certified millimetre / a certified safety factor). A certified magnitude needs a
single-scalar coupon fit for the exact printer+filament+profile.

VOXEL-STAIRCASE CAVEAT: the analysis grid is a coarse voxel occupancy of the
part; watertight != genus-correct, and the staircased boundary slightly
over-stiffens thin features. This is a solid-model approximation, disclosed.

COMPUTE SAFETY: single-thread; grids capped on SOLID element count (<= ~6000 for
a ~90s pure-python assembly), NOT on bbox dimension; scipy direct solve below
~12k dofs, pyamg SA-AMG+CG above. Inputs decimated and voxel-capped before solve.

Run:  python -m printworthy.core._print_fem            # selftest: 3 analytical gates + demos
      python -m printworthy.core._print_fem --quick    # smaller grids, faster
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

# single-thread the numerics (16 GB shared RAM, OOM-sensitive; user keeps CPU)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from ._mesh_util import decimate as _decimate, fmt_fos as _fmt_fos
_HERE = os.path.dirname(os.path.abspath(__file__))

# ---- validated solver pieces (import is the only coupling) ---------------- #
_IMPORT_ERR = None
try:
    from . import fem_materials as fm
    from . import fem_voxel_core as fvc
    from . import fem_warp_probe as fwp
    from . import fem_orthotropic as fortho
    _HAVE_FEM = True
except Exception as _e:  # pragma: no cover
    _HAVE_FEM = False
    _IMPORT_ERR = _e

# ei_core sigma* (FEM mesh-quality metric). Optional; absence -> sigma* skipped.
try:
    from . import quality as _eiq
    _HAVE_SIGMA = True
except Exception:
    _HAVE_SIGMA = False


# =========================================================================== #
#  Result container                                                           #
# =========================================================================== #
@dataclass
class PrintFEMResult:
    """A thin record around a channel result dict. `.ok` and `[...]` both work."""
    kind: str
    data: dict

    @property
    def ok(self) -> bool:
        return bool(self.data.get("ok", False))

    def __getitem__(self, k):
        return self.data[k]

    def get(self, k, default=None):
        return self.data.get(k, default)


# the single honest disclaimer reused on every print output
_UNCALIBRATED_CLAIM = (
    "UNCALIBRATED physics-based estimate. CERTIFIED: sign + shape + trend + "
    "order of magnitude (the solver reproduces closed forms under refinement). "
    "NOT certified: an absolute millimetre / an absolute safety factor (material "
    "constants are mid-range literature +/-20-40%; Q1 hex shear-locks so warp "
    "magnitude is a LOWER bound). A certified magnitude needs a single-scalar "
    "coupon fit for the exact printer+filament+profile."
)


# =========================================================================== #
#  build-direction -> occupancy axis helpers                                  #
# =========================================================================== #
def _normalize_dir(v) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _reorient_build_up(mesh, build_dir):
    """Rotate a copy of `mesh` so `build_dir` becomes +z (the voxel z = build
    axis the warp model assumes). Returns the rotated trimesh. Never raises:
    on any failure returns the input mesh unchanged.

    The print is laid down layer by layer along the build direction; the warp
    model needs that direction to be the occupancy grid's z so the through-
    thickness eigenstrain gradient forms along it. If build_dir is already +z
    (default) the mesh is returned as-is.
    """
    try:
        import trimesh
        up = _normalize_dir(build_dir)
        z = np.array([0.0, 0.0, 1.0])
        if abs(up @ z) > 0.9999:           # already aligned (+/- z)
            if up[2] < 0:                  # flip so build is +z
                m = mesh.copy()
                m.apply_transform(trimesh.transformations.rotation_matrix(
                    np.pi, [1.0, 0.0, 0.0], mesh.centroid))
                return m
            return mesh
        axis = np.cross(up, z)
        axis /= np.linalg.norm(axis) + 1e-12
        ang = float(np.arccos(np.clip(up @ z, -1.0, 1.0)))
        m = mesh.copy()
        m.apply_transform(trimesh.transformations.rotation_matrix(
            ang, axis, mesh.centroid))
        return m
    except Exception:
        return mesh


# =========================================================================== #
#  CHANNEL 1: WARP  (inherent-strain distortion + residual-stress hotspots)   #
# =========================================================================== #
def warp_analysis(mesh, material="PLA", build_dir=None, *,
                  max_elem=6000, gamma_z=0.0, bottom_locked_frac=0.0,
                  use_locked=True, quick=False, keep_field=False):
    """Predict inherent-strain WARP of a part: corner lift (mm) + hotspot field.

    mesh        : trimesh solid (the part as printed).
    material    : name in fem_materials ('PLA','PETG','ABS','resin') or a Material.
    build_dir   : layer-stacking direction in the mesh frame; default +z.
                  (resin is flagged: its warp is cure-shrinkage, not thermal --
                  the thermal-eigenstrain path does NOT apply, so a warning is set.)
    max_elem    : SOLID-element cap for the analysis grid (compute safety).

    Returns a PrintFEMResult. data keys (distances in mm):
        ok, max_corner_lift_mm (LOWER bound), corner_lift_mm (per corner),
        residual_hotspot_eps_grad, thickness_H_mm, n_elem, pitch_mm,
        material, build_dir, uncalibrated=True, claim, sign='edges lift up off bed'.
    keep_field=True additionally returns u_field_mm (n_node,3) and
        node_coords_mm (n_node,3) -- the full released-curl displacement field,
        for warp_precomp's vertex interpolation (opt-in; off by default so no
        existing caller pays for two extra (n_node,3) arrays).

    PHYSICS: reuses the VALIDATED released-curl inherent-strain solver
    (fem_warp_probe.predict_warp), whose eigenstrain load passes the Timoshenko
    bimetal gate (sign + decreasing rel_err under refinement) AND whose released
    continuous-gradient curl passes its own closed-form gate. The bed-bonded
    bottom layer stores less locked contraction than the last-printed top; that
    through-thickness gradient is a bending eigenstrain -> edges curl UP on
    release (the certified sign). Magnitude is a LOWER bound (Q1 shear-locking).
    """
    if not _HAVE_FEM:
        return PrintFEMResult("warp", {"ok": False, "reason": f"FEM import: {_IMPORT_ERR}"})
    try:
        mtl, exact = fm.get_material(material) if isinstance(material, str) else (material, True)
        warn = None
        if mtl.name.lower() == "resin":
            warn = ("resin warp is cure-shrinkage, NOT thermal: the thermal-"
                    "eigenstrain path does not apply; result is a placeholder.")
        m = _decimate(mesh)
        if build_dir is not None:
            m = _reorient_build_up(m, build_dir)
        mx = 2500 if quick else max_elem
        occ, pitch, info = fwp.voxelize_capped(
            m, max_elem=mx, min_z_layers=3 if not quick else 2)
        origin = info.get("node_origin", np.zeros(3))
        res = fwp.predict_warp(
            occ, pitch, mtl, origin=origin, use_locked=use_locked,
            gamma_z=gamma_z, bottom_locked_frac=bottom_locked_frac)
        if not res.get("ok"):
            return PrintFEMResult("warp", {"ok": False, "reason": "no solid voxels"})

        # residual-stress hotspot proxy: the magnitude of the released vertical
        # displacement field is largest where the curl moment is most resisted.
        # We report the eigenstrain GRADIENT (the bending driver) + the spatial
        # peak uplift so the caller can flag where stress concentrates. (A full
        # per-element residual-stress field would need a stress recovery pass;
        # the warp solver is displacement-based, so we report the kinematic
        # hotspot honestly rather than fabricate a stress number.)
        out = {
            "ok": True,
            "kind": "warp",
            "material": mtl.name,
            "material_exact": bool(exact),
            "build_dir": None if build_dir is None else list(_normalize_dir(build_dir)),
            "max_corner_lift_mm": float(res["max_corner_lift_mm"]),
            "corner_lift_mm": {k: float(v) for k, v in res["corner_lift_mm"].items()},
            "max_uplift_mm": float(res["max_uplift_mm"]),
            "max_total_disp_mm": float(res["max_total_disp_mm"]),
            "residual_hotspot_eps_grad": float(res["eps_gradient"]),
            "kappa_strip_cf_per_mm": float(res["kappa_strip_cf_per_mm"]),
            "corner_lift_strip_cf_mm": float(res["corner_lift_strip_cf_mm"]),
            "thickness_H_mm": float(res["thickness_H_mm"]),
            "eps_scalar": float(res["eps_scalar"]),
            "dT_lock_K": float(res["dT_lock_K"]),
            "n_elem": int(res["n_elem"]),
            "n_node": int(res["n_node"]),
            "pitch_mm": tuple(float(p) for p in res["pitch_mm"]),
            "bbox_mm": tuple(float(b) for b in res["bbox_mm"]),
            "sign": "edges lift UP off the bed (corner warp)",
            "magnitude_kind": "LOWER bound (Q1 hex shear-locks)",
            "uncalibrated": True,
            "claim": _UNCALIBRATED_CLAIM,
            "warning": warn,
            "_raw": res,   # kept for rendering; not a public field
        }
        if keep_field:
            # res["U"]/res["coords"] are already computed by fwp.predict_warp
            # (fem_warp_probe.py: displacement_field() line 254, "U"/"coords"
            # keys lines 342/344) and already flow this far inside `res` --
            # publish stable public names, no recompute.
            out["u_field_mm"] = res["U"]            # (n_node, 3) mm
            out["node_coords_mm"] = res["coords"]   # (n_node, 3) mm, same node order
            # pitch is already public as out["pitch_mm"] (line 248) -- reuse, no new key.
        return PrintFEMResult("warp", out)
    except Exception as e:
        return PrintFEMResult("warp", {"ok": False, "reason": f"{type(e).__name__}: {e}"})


# =========================================================================== #
#  CHANNEL 2: STRENGTH  (orthotropic stress / factor-of-safety field)         #
# =========================================================================== #
# A load_case is a dict:
#   {"load_face":"zmax", "anchor_face":"zmin", "load_axis":2, "total_force_N":300}
_DEFAULT_LOAD_CASE = {
    "load_face": "zmax", "anchor_face": "zmin", "load_axis": 2,
    "total_force_N": 300.0,
}


def _build_orientation_R(build_dir):
    """3x3 rotation taking MATERIAL axes -> GLOBAL, encoding which global axis the
    weak layer-normal (material axis 3 = Z) points along. Identity = layers stack
    along global +z (printed flat). build_dir is where the layer-normal points in
    the GLOBAL frame; we rotate material-z onto it."""
    up = _normalize_dir(build_dir if build_dir is not None else (0, 0, 1))
    z = np.array([0.0, 0.0, 1.0])
    if abs(up @ z) > 0.9999:
        return np.eye(3) if up[2] > 0 else fortho.rot_about_x(np.pi)
    axis = np.cross(z, up)
    axis /= np.linalg.norm(axis) + 1e-12
    ang = float(np.arccos(np.clip(z @ up, -1.0, 1.0)))
    # Rodrigues rotation matrix
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def strength_analysis(mesh, material="PLA", load_case=None, build_dir=None, *,
                      max_elem=5000, length_unit="mm"):
    """Orthotropic stress / factor-of-safety field for a part under a face load.

    mesh       : trimesh solid (mm or m -- consistent with load_case forces).
    material   : name or fem_materials.Material; mapped to an OrthoMaterial.
    load_case  : dict {load_face, anchor_face, load_axis, total_force_N}.
                 Faces are 'xmin'|'xmax'|'ymin'|'ymax'|'zmin'|'zmax'.
    build_dir  : layer-normal direction in the GLOBAL frame (encodes the build
                 orientation: which global axis is the weak weld direction).
                 Default +z = printed flat = layers stack along z.
    length_unit: mesh length unit, 'mm' (default, the print domain) or 'm'. A
                 metre-scale mesh left at 'mm' over-scales stress ~1e6x (FOS ~0);
                 a soft `unit_warning` is returned when the extent looks metre-
                 scale. Threaded to the solver's
                 UNIT BRIDGE so a mm mesh's traction (N/mm^2) is converted to Pa
                 before the FOS compares it against the Pa allowables (else FOS
                 inflates x1e6 and pins at the ceiling). See solve_part_stress.

    Returns PrintFEMResult. data: ok, min_fos, p05_fos, mean_fos, peak_vm_Pa,
    applied_sigma_Pa, nelem, fos (per-elem), vm, centroids, uncalibrated, claim.

    PHYSICS: reuses the VALIDATED orthotropic solver (fem_orthotropic), whose
    element passes the off-axis-modulus closed form under refinement (1.4% ->
    0.52% monotone) and whose anisotropy SIGN is correct (load across the layers
    reads a lower FOS than along them). min_fos is a COMPARATIVE screening metric
    (which orientation/load is weaker), not a certified safety factor.
    """
    if not _HAVE_FEM:
        return PrintFEMResult("strength", {"ok": False, "reason": f"FEM import: {_IMPORT_ERR}"})
    try:
        # Guard degenerate input: fem_orthotropic.voxelize_mesh falls back to a
        # 1x1x1 grid on a bad mesh and would "succeed" on garbage -- refuse it.
        try:
            nfaces = len(mesh.faces)
            vol_ext = float(np.prod(np.asarray(mesh.extents, float))) \
                if getattr(mesh, "extents", None) is not None else 0.0
        except Exception:
            nfaces, vol_ext = 0, 0.0
        if nfaces < 4 or not np.isfinite(vol_ext) or vol_ext <= 0.0:
            return PrintFEMResult("strength",
                                  {"ok": False, "reason": "degenerate/empty mesh"})
        # UNIT-MISMATCH GUARD: length_unit defaults to 'mm' (the print domain).
        # A mesh whose largest extent is < 2 under 'mm' is almost certainly a
        # metre-scale mesh mislabelled -> the traction would be over-scaled 1e6x
        # and FOS pinned near 0. Surface it (soft, non-blocking) rather than let
        # it fail silently -- closes the length_unit footgun for external callers.
        _unit_warn = None
        try:
            _max_ext = float(np.max(np.asarray(mesh.extents, float)))
            if str(length_unit).lower() == "mm" and _max_ext < 2.0:
                _unit_warn = (f"mesh max extent {_max_ext:.4g} is < 2 mm under "
                              f"length_unit='mm'; if this mesh is in metres pass "
                              f"length_unit='m' (else stress reads ~1e6x too high, "
                              f"FOS ~0).")
        except Exception:
            pass
        name = material if isinstance(material, str) else material.name
        omat = fortho.material_from_table(name)
        lc = dict(_DEFAULT_LOAD_CASE)
        if load_case:
            lc.update(load_case)
        R = _build_orientation_R(build_dir)
        m = _decimate(mesh)
        res = fortho.solve_part_stress(
            m, omat, R,
            load_axis=int(lc["load_axis"]), load_face=lc["load_face"],
            anchor_face=lc["anchor_face"], total_force=float(lc["total_force_N"]),
            target_max_elements=max_elem, length_unit=length_unit)
        if not res.get("ok"):
            return PrintFEMResult("strength", {"ok": False, "reason": res.get("reason")})
        res = dict(res)
        res.update({
            "kind": "strength",
            "material": omat.name,
            "load_case": lc,
            "unit_warning": _unit_warn,
            "build_dir": None if build_dir is None else list(_normalize_dir(build_dir)),
            "Xt_Pa": omat.Xt, "Zt_Pa": omat.Zt,
            "min_fos_kind": "COMPARATIVE screening metric (not a certified SF)",
            "uncalibrated": True,
            "claim": _UNCALIBRATED_CLAIM,
        })
        return PrintFEMResult("strength", res)
    except Exception as e:
        return PrintFEMResult("strength", {"ok": False, "reason": f"{type(e).__name__}: {e}"})


def orient_for_strength(mesh, load_case=None, material="PLA", *,
                        candidate_dirs=None, max_elem=4000, n_candidates=6,
                        verbose=False, keep_results=False, length_unit="mm"):
    """Pick the build orientation that MAXIMISES min-FOS for a fixed load case.

    Sweeps a small set of candidate build directions (where the weak layer-normal
    points), solves the orthotropic FOS field for each, and returns the
    orientation with the highest min-FOS. The load is fixed in the GLOBAL frame;
    only the material (layer) frame rotates -- this is exactly "which way should I
    print this part so the load runs more ALONG the layers".

    candidate_dirs : list of 3-vectors (layer-normal directions). Default = the
                     6 axis directions + a couple of 45-deg tilts (cheap, covers
                     the dominant choices). For a richer sweep pass directions
                     from _print3d._candidate_ups(mesh).

    Returns PrintFEMResult. data: ok, best_dir, best_min_fos, ranking
    (list of (dir, min_fos)), n_evaluated, uncalibrated, claim.

    HONEST: comparative only. The WINNER is the orientation whose min-FOS is
    highest among those tried -- a screening recommendation, not a certified
    safety statement. Same uncalibrated discipline as strength_analysis.
    """
    if not _HAVE_FEM:
        return PrintFEMResult("orient", {"ok": False, "reason": f"FEM import: {_IMPORT_ERR}"})
    try:
        if candidate_dirs is None:
            candidate_dirs = [
                (0, 0, 1), (1, 0, 0), (0, 1, 0),
                (0, 0, -1), (1, 0, 1), (0, 1, 1),
            ][:max(1, n_candidates)]
        ranking = []
        results = {}   # dir-tuple -> the full strength PrintFEMResult (if kept)
        for d in candidate_dirs:
            r = strength_analysis(mesh, material=material, load_case=load_case,
                                  build_dir=d, max_elem=max_elem,
                                  length_unit=length_unit)
            if r.ok:
                dt = tuple(float(x) for x in _normalize_dir(d))
                ranking.append((dt, float(r["min_fos"])))
                if keep_results:
                    results[dt] = r
                if verbose:
                    print(f"   dir={tuple(round(x,2) for x in d)}  "
                          f"min_fos={_fmt_fos(r['min_fos'], 3)}  nelem={r['nelem']}")
        if not ranking:
            return PrintFEMResult("orient", {"ok": False, "reason": "no orientation solved"})
        ranking.sort(key=lambda t: -t[1])
        best_dir, best_fos = ranking[0]
        worst_dir, worst_fos = ranking[-1]
        return PrintFEMResult("orient", {
            "ok": True, "kind": "orient",
            "best_dir": list(best_dir), "best_min_fos": best_fos,
            "worst_dir": list(worst_dir), "worst_min_fos": worst_fos,
            "improvement_ratio": (best_fos / worst_fos) if worst_fos > 0 else float("inf"),
            "ranking": ranking, "n_evaluated": len(ranking),
            "best_result": results.get(tuple(best_dir)) if keep_results else None,
            "worst_result": results.get(tuple(worst_dir)) if keep_results else None,
            "material": material if isinstance(material, str) else material.name,
            "uncalibrated": True,
            "claim": "Comparative orientation screening: the recommended build "
                     "direction maximises min-FOS AMONG THOSE TRIED. "
                     + _UNCALIBRATED_CLAIM,
        })
    except Exception as e:
        return PrintFEMResult("orient", {"ok": False, "reason": f"{type(e).__name__}: {e}"})


# =========================================================================== #
#  CHANNEL 3: sigma* mesh-quality certificate ON THE ANALYSIS GRID            #
# =========================================================================== #
def fem_mesh_quality(occ=None, pitch=1.0, *, mesh=None, max_elem=6000):
    """sigma* (FEM mesh-quality / interpolation-error-bound) certificate on the
    ANALYSIS grid -- WITH ITS HONEST VERDICT.

    The interpolation-error bound is err <= C * h / sigma*, where sigma* measures
    element-frame conditioning. THE HONEST FINDING (see _fem_validation_plan GATE
    D): on a STRUCTURED voxel-hex analysis grid every solid voxel is an IDENTICAL
    axis-aligned unit cube, so the per-ELEMENT sigma* (Jacobian conditioning) is
    CONSTANT across the entire grid. A constant field carries ZERO spatial
    information about where the FEM error is largest. Therefore:

        sigma*-as-a-spatial-error-certificate is NOT APPLICABLE to this stack.

    We do NOT dress a constant up as a certificate. What we DO report honestly:
      * element_sigma_star : the (constant) per-element Jacobian sigma* for the
        cube -- which is 1.0 for an axis-aligned cube (orthonormal edge frame),
        i.e. the analysis grid is perfectly conditioned everywhere. h = pitch is
        the single knob in the C*h/sigma* bound: REFINE THE PITCH to lower the
        error bound; sigma* cannot localize it because it is uniform.
      * spatially_informative : False (with the reason), so callers never present
        it as a per-region trust map.
    For the SURFACE remesh / tet path (variable element shape) sigma* IS
    informative; a hook (`surface_sigma_star_field`) is provided for that case.

    Accepts either an occupancy grid (occ, pitch) or a trimesh (mesh) to voxelize.
    Never raises.
    """
    try:
        if occ is None and mesh is not None and _HAVE_FEM:
            occ, pitch, _info = fwp.voxelize_capped(_decimate(mesh), max_elem=max_elem)
        if occ is None:
            return {"ok": False, "reason": "no occupancy/mesh provided"}
        occ = np.asarray(occ, bool)
        n_solid = int(occ.sum())
        if n_solid == 0:
            return {"ok": False, "reason": "occupancy grid has no solid voxels"}
        # The axis-aligned cube's local edge frame is orthonormal -> its scaled
        # Jacobian sigma_min == 1 (perfect conditioning); h = pitch.
        element_sigma_star = 1.0
        h = float(pitch if np.isscalar(pitch) else np.asarray(pitch).ravel()[0])
        return {
            "ok": True,
            "kind": "fem_mesh_quality",
            "n_solid_elements": n_solid,
            "grid_shape": tuple(int(s) for s in occ.shape),
            "pitch_mm": h,
            "element_sigma_star": element_sigma_star,
            "sigma_star_field_constant": True,
            "spatially_informative": False,
            "interp_error_bound_proportional_to": "C * h / sigma* = C * pitch "
                                                  "(uniform; lower the pitch to lower the bound)",
            "verdict": ("sigma* is CONSTANT (=1.0) on a structured voxel-hex grid "
                        "-- every element is an identical axis-aligned cube, so it "
                        "carries NO spatial error information. It is NOT a per-region "
                        "trust map for this stack; the only error knob is the pitch h. "
                        "sigma* IS informative on a variable-shape surface/tet mesh "
                        "(use surface_sigma_star_field there)."),
            "trust_everywhere_equally": True,
            "uncalibrated": True,
        }
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def surface_sigma_star_field(mesh):
    """Per-VERTEX sigma* of a SURFACE mesh (the variable-element case where
    sigma* IS spatially informative). Thin wrapper over ei_core.quality. Returns
    {ok, sigma_star (N,), min, mean, frac_below_0p3, ...} or {ok:False,...}.
    Never raises. This is the path to use if the print FEM ever moves to a
    body-fitted (variable-shape) mesh -- NOT the structured voxel grid above.
    """
    if not _HAVE_SIGMA:
        return {"ok": False, "reason": "ei_core.quality unavailable"}
    try:
        V = np.asarray(mesh.vertices, float)
        F = np.asarray(mesh.faces, np.int64)
        sig, status = _eiq.vertex_sigma_min_field(V, F)
        sig = np.asarray(sig, float)
        finite = sig[np.isfinite(sig)]
        return {
            "ok": True, "kind": "surface_sigma_star",
            "n_vertices": int(V.shape[0]),
            "sigma_star": sig,
            "min": float(finite.min()) if finite.size else float("nan"),
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "frac_below_0p3": float(np.mean(finite < 0.3)) if finite.size else 0.0,
            "spatially_informative": True,
            "note": "variable-element surface metric: low sigma* = sliver/poor "
                    "frame = larger C*h/sigma* interpolation-error bound there.",
        }
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


# =========================================================================== #
#  RENDERERS (delegate to the validated modules; never raise)                 #
# =========================================================================== #
def render_warp(result, out_png, title="print warp (inherent strain)", warp_scale=None):
    """Render a warp_analysis result. Returns out_png or None."""
    try:
        data = result.data if isinstance(result, PrintFEMResult) else result
        raw = data.get("_raw")
        if raw is None or not data.get("ok"):
            return None
        return fwp.render_warp(raw, out_png, title=title, warp_scale=warp_scale)
    except Exception:
        return None


def render_strength(result, out_png, title="orthotropic stress (FOS)"):
    """Render a strength_analysis result. Returns out_png or None."""
    try:
        data = result.data if isinstance(result, PrintFEMResult) else result
        if not data.get("ok"):
            return None
        return fortho.render_stress_field(data, out_png, title=title)
    except Exception:
        return None


def render_orient_compare(result_strong, result_weak, out_png,
                          label_strong="strong orientation",
                          label_weak="weak orientation"):
    """Side-by-side FOS comparison of two strength_analysis results (shared scale).
    Pass the BEST and WORST strength results from an orient_for_strength sweep.
    Returns out_png or None."""
    try:
        a = result_strong.data if isinstance(result_strong, PrintFEMResult) else result_strong
        b = result_weak.data if isinstance(result_weak, PrintFEMResult) else result_weak
        if not (a.get("ok") and b.get("ok")):
            return None
        return fortho.render_orientation_compare(a, b, out_png, label_strong, label_weak)
    except Exception:
        return None


# =========================================================================== #
#  THE ANALYTICAL GATE (baked into selftest)                                  #
# =========================================================================== #
def _gate_bar() -> tuple[bool, list]:
    """B1: uniaxial bar d = F L/(A E). Q1 hex exact -> rel_err ~1e-13."""
    rows = fvc.bench_uniaxial()
    finest = rows[-1][3]
    return (finest < 0.01), rows


def _gate_cantilever() -> tuple[bool, list, bool, float]:
    """B2: cantilever d = P L^3/(3 E I). Monotone toward 1 from below, >=0.97."""
    rows = fvc.bench_cantilever()
    ratios = [r[3] for r in rows]
    monotone = all(ratios[i + 1] > ratios[i] for i in range(len(ratios) - 1))
    ok = monotone and (0.97 <= ratios[-1] <= 1.03)
    return ok, rows, monotone, ratios[-1]


def _gate_eigenstrain(quick=False) -> tuple[bool, list, float, bool, bool]:
    """B3: eigenstrain bend (Timoshenko bimetal). rel_err decreasing, sign ok."""
    grids = ((16, 2, 4), (24, 2, 4), (32, 2, 4)) if quick else \
            ((16, 2, 4), (24, 2, 4), (32, 2, 4), (40, 2, 6), (48, 2, 8))
    rows, kts, monotone, sign_ok = fwp.bimetal_convergence(grids)
    finest = rows[-1][3]
    ok = monotone and sign_ok and finest < 0.20
    return ok, rows, kts, monotone, sign_ok


# =========================================================================== #
#  SELFTEST                                                                    #
# =========================================================================== #
def selftest(quick=False, make_renders=True) -> bool:
    print("=" * 76)
    print("_print_fem  --  unified reduced-order FEM-for-print  |  THE GATE")
    print("=" * 76)
    if not _HAVE_FEM:
        print("FEM stack NOT importable:", _IMPORT_ERR)
        return False
    print("stack: fem_voxel_core + fem_warp_probe + fem_orthotropic "
          "(scikit-fem BSD-3 / pyamg MIT / numpy+scipy BSD)")
    ok = True

    # ---------- B1: uniaxial bar ------------------------------------------- #
    print("\n--- B1  uniaxial bar  d = F L/(A E)  (assembly + BC + traction) ---")
    g1, r1 = _gate_bar()
    print(f"{'nelem':>8} {'d_fem':>14} {'d_exact':>14} {'rel_err':>10}")
    for n, a, b, rel, _s in r1:
        print(f"{n:>8} {a:>14.6e} {b:>14.6e} {rel:>10.2e}")
    print(f"  -> finest rel_err = {r1[-1][3]:.2e}  (gate <1%; Q1 hex EXACT in tension)  "
          f"{'PASS' if g1 else 'FAIL'}")
    ok &= g1

    # ---------- B2: cantilever --------------------------------------------- #
    print("\n--- B2  cantilever  d = P L^3/(3 E I)  (bending) ---")
    g2, r2, mono2, last2 = _gate_cantilever()
    print(f"{'nelem':>8} {'d_fem':>14} {'d_EB':>14} {'ratio':>9}")
    for n, a, b, rat, _s in r2:
        print(f"{n:>8} {a:>14.6e} {b:>14.6e} {rat:>9.4f}")
    print(f"  -> d_fem/d_EB monotone-up={mono2}, finest={last2:.4f} "
          f"(toward 1 from below = correct Q1 shear-locking)  {'PASS' if g2 else 'FAIL'}")
    ok &= g2

    # ---------- B3: eigenstrain bend --------------------------------------- #
    print("\n--- B3  eigenstrain bend  (Timoshenko 1925 bimetal; the WARP mechanism) ---")
    g3, r3, kts, mono3, sign3 = _gate_eigenstrain(quick=quick)
    print(f"  closed-form Timoshenko kappa = {kts:.6e} /mm  (r = {1/kts:.1f} mm)")
    print(f"{'grid':>10} {'kappa_fem':>14} {'ratio':>8} {'rel_err':>10}")
    for gstr, kf, rat, rel in r3:
        print(f"{gstr:>10} {kf:>14.4e} {rat:>8.4f} {rel:>10.2%}")
    print(f"  -> rel_err monotone-decreasing={mono3}, sign correct every grid={sign3}, "
          f"finest={r3[-1][3]:.2%}  {'PASS' if g3 else 'FAIL'}")
    ok &= g3

    gate_ok = ok
    print("\n" + "-" * 76)
    print("ANALYTICAL GATE: " + ("ALL THREE BENCHMARKS CONVERGE TO CLOSED FORM"
                                  if gate_ok else "A BENCHMARK FAILED"))
    print("-" * 76)

    # ---------- public-API demos (only meaningful if the gate passed) ------ #
    if gate_ok:
        import trimesh
        pla, _ = fm.get_material("PLA")
        sd = _HERE

        # WARP channel demo on a flat bracket (warp-prone)
        print("\n[DEMO] warp_analysis -- flat bracket 100x30x6 mm (PLA)")
        flat = trimesh.creation.box(extents=[100.0, 30.0, 6.0])
        wr = warp_analysis(flat, "PLA", quick=quick)
        if wr.ok:
            print(f"   max corner lift = {wr['max_corner_lift_mm']*1e3:.1f} um "
                  f"({wr['max_corner_lift_mm']:.3f} mm, LOWER bound)  "
                  f"nelem={wr['n_elem']}  sign='{wr['sign']}'")
            if make_renders:
                p = render_warp(wr, os.path.join(sd, "print_fem_warp_demo.png"),
                                title="warp_analysis: flat bracket (PLA)")
                print(f"   render -> {p}")
            ok &= (wr["max_corner_lift_mm"] > 0)
        else:
            print("   warp demo FAILED:", wr.get("reason")); ok = False

        # WARP trend control: thinner warps more (same footprint/pitch)
        print("\n[CTRL] warp trend -- thinner plate warps MORE (controlled)")
        lifts = []
        for Lz in (4.0, 8.0, 12.0):
            occ = np.ones((50, 15, int(round(Lz / 2.0))), bool)
            rr = fwp.predict_warp(occ, 2.0, pla)
            lifts.append(rr["max_corner_lift_mm"])
            print(f"   h={Lz:4.0f}mm: corner lift {rr['max_corner_lift_mm']*1e3:7.1f} um")
        trend_ok = all(lifts[i] > lifts[i + 1] for i in range(len(lifts) - 1))
        print(f"   monotone-decreasing with thickness = {trend_ok}  "
              f"{'PASS' if trend_ok else 'FAIL'}")
        ok &= trend_ok

        # STRENGTH channel demo + anisotropy sign control.
        # Small element cap: the demo only needs to show the SIGN of the
        # anisotropy (across-Z weaker than along-XY); convergence is already
        # certified in fem_orthotropic's off-axis-modulus gate. The orthotropic
        # BilinearForm assembles ~12 x slower than the core, so keep it tiny.
        s_cap = 350 if quick else 1000
        print("\n[DEMO] strength_analysis -- cube, load ACROSS vs ALONG layers"
              f"  (<= {s_cap} elem)")
        cube = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
        cube.apply_translation([0.01, 0.01, 0.01])
        sx = strength_analysis(cube, "PLA",
                               {"load_face": "xmax", "anchor_face": "xmin",
                                "load_axis": 0, "total_force_N": 500.0},
                               max_elem=s_cap)
        sz = strength_analysis(cube, "PLA",
                               {"load_face": "zmax", "anchor_face": "zmin",
                                "load_axis": 2, "total_force_N": 500.0},
                               max_elem=s_cap)
        if sx.ok and sz.ok:
            ratio = sz["min_fos"] / sx["min_fos"]
            print(f"   ALONG-XY min_fos={_fmt_fos(sx['min_fos'])}  "
                  f"ACROSS-Z min_fos={_fmt_fos(sz['min_fos'])}  ratio={ratio:.3f} "
                  f"(expected ~Zt/Xt={sz['Zt_Pa']/sz['Xt_Pa']:.2f})")
            aniso_ok = sz["min_fos"] < sx["min_fos"]
            print(f"   across-Z weaker than along-XY = {aniso_ok}  "
                  f"{'PASS' if aniso_ok else 'FAIL'}")
            ok &= aniso_ok
            if make_renders:
                p = render_strength(sz, os.path.join(sd, "print_fem_strength_demo.png"),
                                    title="strength_analysis: cube, load ACROSS layers")
                print(f"   render -> {p}")
        else:
            print("   strength demo FAILED:", sx.get("reason"), sz.get("reason")); ok = False

        # ORIENT-FOR-STRENGTH demo on a bracket
        print("\n[DEMO] orient_for_strength -- bracket, wall tensile load")
        bracket = fortho.make_bracket()
        lc = {"load_face": "zmax", "anchor_face": "zmin", "load_axis": 2,
              "total_force_N": 300.0}
        o_cap = 400 if quick else 1200
        ores = orient_for_strength(bracket, lc, "PLA",
                                   candidate_dirs=[(0, 0, 1), (1, 0, 0)],
                                   max_elem=o_cap, keep_results=make_renders)
        if ores.ok:
            print(f"   best build dir = {[round(x,2) for x in ores['best_dir']]}  "
                  f"min_fos={_fmt_fos(ores['best_min_fos'])}")
            print(f"   worst build dir= {[round(x,2) for x in ores['worst_dir']]}  "
                  f"min_fos={_fmt_fos(ores['worst_min_fos'])}  "
                  f"improvement x{ores['improvement_ratio']:.2f}")
            orient_ok = ores["best_min_fos"] >= ores["worst_min_fos"]
            print(f"   best >= worst (sweep ordered correctly) = {orient_ok}  "
                  f"{'PASS' if orient_ok else 'FAIL'}")
            ok &= orient_ok
            # render the best-vs-worst comparison (reuse the sweep's solved
            # results -- no double solve)
            if make_renders and ores.get("best_result") and ores.get("worst_result"):
                p = render_orient_compare(
                    ores["best_result"], ores["worst_result"],
                    os.path.join(sd, "print_fem_orient_compare.png"),
                    label_strong="best build dir (load more ALONG layers)",
                    label_weak="worst build dir (load ACROSS layers)")
                print(f"   render -> {p}")
        else:
            print("   orient demo FAILED:", ores.get("reason")); ok = False

        # sigma* mesh-quality verdict (HONEST: constant on voxel grid)
        print("\n[CHECK] fem_mesh_quality -- sigma* certificate verdict")
        occ = np.ones((20, 8, 4), bool)
        q = fem_mesh_quality(occ, pitch=2.0)
        print(f"   element sigma* = {q['element_sigma_star']:.3f} (constant)  "
              f"spatially_informative = {q['spatially_informative']}")
        print(f"   verdict: {q['verdict'][:88]}...")
        # the honest gate: it must REFUSE to claim spatial information
        sigma_honest = (q["ok"] and q["spatially_informative"] is False
                        and q["sigma_star_field_constant"] is True)
        print(f"   sigma* correctly reported NON-spatial on the voxel grid = "
              f"{sigma_honest}  {'PASS' if sigma_honest else 'FAIL'}")
        ok &= sigma_honest

    print("\n" + "=" * 76)
    print("ALL_PASS" if ok else "SOME CHECK FAILED")
    print("Solver certified against closed forms (bar/cantilever/eigenstrain).")
    print("Print PREDICTIONS are UNCALIBRATED: sign/shape/trend + order of magnitude")
    print("certified; a certified mm/safety-factor needs a single-coupon calibration.")
    print("Voxel-staircase caveat: coarse occupancy solid model; watertight != genus.")
    print("=" * 76)
    return ok


if __name__ == "__main__":
    from ._mesh_util import ascii_console
    ascii_console()
    q = "--quick" in sys.argv
    sys.exit(0 if selftest(quick=q) else 1)
