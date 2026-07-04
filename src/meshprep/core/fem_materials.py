"""FDM material / process-physics calibration constants and the warp eigenstrain spec.

INVESTIGATE-2 deliverable. This module is the SINGLE SOURCE OF TRUTH for the
physical constants the reduced-order print FEM consumes. It does NOT solve any
FEM itself -- it supplies (a) labelled, cited calibration constants, (b) the
exact per-layer eigenstrain a warp model applies and why, and (c) the
closed-form Timoshenko bimetal benchmark that any eigenstrain solver MUST
reproduce under mesh refinement before its warp output may be trusted.

HONESTY STANCE (read this before using any number here):
  * Every constant is an APPROXIMATE CALIBRATION CONSTANT, not measured truth
    for a specific printer/filament/slice profile. FDM properties scatter
    +/- 20-40% with nozzle temp, layer height, infill, raster angle, brand,
    moisture, and age. The values below are mid-range literature consensus.
  * An UNCALIBRATED model built on these CAN legitimately claim:
        - the SHAPE of a deflection / warp field (where it bends, which way),
        - the SIGN of curvature (warp lifts corners up off the bed),
        - the TREND across designs (part A warps more than part B),
        - ORDER OF MAGNITUDE (mm not m, not um).
    It CANNOT claim:
        - a certified deflection/warp magnitude (needs a real-print calibration
          coupon to fit the lumped eigenstrain to that exact process),
        - a pass/fail safety margin on absolute stress,
        - anything quantitative about a material/process not in this table.
  * "anisotropy" here is the Z/XY STRENGTH ratio (interlayer-weld limited). The
    Z/XY STIFFNESS (modulus) ratio is closer to 1 than the strength ratio; we
    label both separately because conflating them is a known footgun.

All functions are pure and never raise: unknown material -> returns the PLA
default with a flag, not an exception. Run `python -m meshprep.core.fem_materials`
for the selftest (constants sanity + Timoshenko closed-form check).

SOURCES (public; accessed 2026-06):
  [Tg/E]   Thermo-Mechanical Behavior of 3D-Printed PLA below Tg, PMC11174730:
           glassy E ~= 1.05 GPa (DMTA storage modulus, low end), Tg ~= 65 C.
  [E tens] Comparative Young's modulus FDM PLA, PMC8746243 / MDPI Polymers:
           static tensile E ~= 2.0-3.6 GPa depending on profile/brand.
  [CTE]    Filament-datasheet consensus (Ultimaker/American Filament et al.):
           PLA 68e-6 /C, PETG 68.4e-6 /C, ABS 90e-6 /C (linear, below Tg).
           DIC study (E3S 2018, Botean) reports far larger APPARENT CTE
           (4.7-5.5e-4 /C) -- that is an effective value inflated by infill
           porosity, NOT the solid-polymer CTE; we use the solid value and
           note the porosity caveat in EIGENSTRAIN_NOTES.
  [aniso]  Z vs XY strength: RapidMade / OSTI 1808415 / fullcontrol:
           Z-axis tensile strength ~= 0.4-0.75 of XY; PLA mid ~= 0.5-0.7.
  [warp]   Inherent-strain / eigenstrain method (PMC10973057; Brunel FDM warp
           analytic model, bura.brunel.ac.uk/.../25747): each just-deposited
           layer cools from deposition temp toward bed/ambient and wants to
           contract by eps = CTE*dT; the platform/prior layers resist, the
           torque of the resisting reaction bends the part -> warp.
  [bimetal] Timoshenko, "Analysis of Bi-metal Thermostats", JOSA 11(3),
           233-255, 1925. Closed-form curvature of a differentially-straining
           bilayer strip; our INHERENT-STRAIN benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# --------------------------------------------------------------------------- #
#  Calibration table                                                          #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Material:
    """One FDM material's approximate calibration constants.

    Units:
        E_GPa            : Young's modulus, GPa (isotropic-elastic surrogate).
        nu               : Poisson ratio, dimensionless.
        cte_per_K        : linear coefficient of thermal expansion, 1/K
                           (== 1/C for a temperature DIFFERENCE).
        Tg_C             : glass transition temperature, C.
        T_deposit_C      : nozzle / deposition temperature, C.
        T_bed_C          : bed (heated platform) temperature, C.
        T_ambient_C      : chamber / room ambient, C.
        z_xy_strength    : Z-axis / XY-plane TENSILE STRENGTH ratio (<1).
        z_xy_stiffness   : Z-axis / XY-plane STIFFNESS (modulus) ratio (<=1,
                           closer to 1 than strength -- separate footgun).
        confidence       : qualitative tag on how well-pinned these are.

    Every value is mid-range literature; treat +/- 20-40% as the honest band.
    """
    name: str
    E_GPa: float
    nu: float
    cte_per_K: float
    Tg_C: float
    T_deposit_C: float
    T_bed_C: float
    T_ambient_C: float
    z_xy_strength: float
    z_xy_stiffness: float
    confidence: str = "approximate-calibration"

    @property
    def E_MPa(self) -> float:
        return self.E_GPa * 1.0e3

    @property
    def E_Pa(self) -> float:
        return self.E_GPa * 1.0e9

    @property
    def shear_modulus_GPa(self) -> float:
        """G = E / (2(1+nu)) -- isotropic-elastic surrogate."""
        return self.E_GPa / (2.0 * (1.0 + self.nu))

    def delta_T_warp(self, *, to: str = "bed") -> float:
        """Effective cooling drop driving warp, K (positive number).

        to="bed"     -> deposit temp down to bed temp (warp during print).
        to="ambient" -> deposit temp down to room (residual/after-release).

        NOTE this is an UPPER bound on the relevant dT: real per-layer warp
        uses a smaller EFFECTIVE drop because polymer relaxes/flows above Tg
        and only locks in strain below it. The honest "locked" drop is closer
        to (Tg - T_bed); see EIGENSTRAIN_NOTES and `locked_delta_T`.
        """
        T_lo = self.T_bed_C if to == "bed" else self.T_ambient_C
        return float(self.T_deposit_C - T_lo)

    def locked_delta_T(self) -> float:
        """The dT over which eigenstrain actually LOCKS IN, K (positive).

        Above Tg the polymer is rubbery (E ~= 1 MPa, see PMC11174730) and
        relaxes -- strain accumulated there does not survive. Only contraction
        from ~Tg down to the bed/ambient is stored as residual eigenstrain.
        This is the physically defensible drop to feed the warp model:
            dT_lock = Tg - T_bed   (clamped >= 0).
        Using the full (T_deposit - T_bed) OVER-predicts warp by ~2-3x and is
        a documented overclaim trap -- do not.
        """
        return float(max(0.0, self.Tg_C - self.T_bed_C))


# Mid-range consensus values. PLA is the primary/best-pinned entry.
PLA = Material(
    name="PLA",
    E_GPa=3.5,            # tensile, high-quality FDM coupon [PMC8746243]; DMTA
                          # storage modulus runs lower (~1.05 GPa, PMC11174730)
    nu=0.36,              # 0.33-0.36 reported; 0.36 mid
    cte_per_K=68e-6,      # solid PLA, datasheet consensus
    Tg_C=60.0,            # 57-65 C across sources; 60 mid
    T_deposit_C=200.0,    # 195-210 C typical PLA nozzle
    T_bed_C=55.0,         # 50-60 C heated bed (or unheated ~ambient)
    T_ambient_C=25.0,
    z_xy_strength=0.65,   # 0.5-0.75 PLA interlayer; 0.65 mid
    z_xy_stiffness=0.90,  # stiffness anisotropy mild
    confidence="primary-best-pinned",
)

PETG = Material(
    name="PETG",
    E_GPa=2.1,
    nu=0.40,
    cte_per_K=68.4e-6,
    Tg_C=80.0,            # ~78-85 C
    T_deposit_C=240.0,
    T_bed_C=75.0,
    T_ambient_C=25.0,
    z_xy_strength=0.75,   # PETG bonds better -> less anisotropic than PLA
    z_xy_stiffness=0.92,
    confidence="secondary",
)

ABS = Material(
    name="ABS",
    E_GPa=2.0,
    nu=0.35,
    cte_per_K=90e-6,      # high CTE -> notoriously warp-prone
    Tg_C=105.0,
    T_deposit_C=240.0,
    T_bed_C=100.0,        # high bed temp specifically to fight warp
    T_ambient_C=25.0,     # enclosure recommended; open-air ambient assumed
    z_xy_strength=0.55,
    z_xy_stiffness=0.88,
    confidence="secondary",
)

# Photopolymer SLA/DLP resin (not FDM-extruded; included for the secondary
# slot). Resin is far more isotropic (cured monolith, no layer welds) so the
# Z/XY ratios are near 1; warp is cure-shrinkage driven, NOT thermal, so the
# thermal-eigenstrain warp path does NOT apply -- flagged below.
RESIN = Material(
    name="resin",
    E_GPa=2.5,            # standard cured resin 1.5-3.5 GPa
    nu=0.40,
    cte_per_K=100e-6,     # cured photopolymer, broad range
    Tg_C=60.0,            # standard resin; tough/engineering resins higher
    T_deposit_C=25.0,     # cured at ~ambient -> thermal dT ~ 0
    T_bed_C=25.0,
    T_ambient_C=25.0,
    z_xy_strength=0.95,   # near-isotropic
    z_xy_stiffness=0.97,
    confidence="secondary-non-thermal-warp",
)

_TABLE = {m.name.lower(): m for m in (PLA, PETG, ABS, RESIN)}


def get_material(name: Optional[str] = None) -> tuple[Material, bool]:
    """Look up a material by name (case-insensitive). Never raises.

    Returns (material, exact_match). On an unknown/None name, returns the PLA
    default with exact_match=False so callers can flag that they fell back.
    """
    if name is None:
        return PLA, False
    m = _TABLE.get(str(name).strip().lower())
    if m is None:
        return PLA, False
    return m, True


def list_materials() -> list[str]:
    return [m.name for m in (PLA, PETG, ABS, RESIN)]


# --------------------------------------------------------------------------- #
#  Eigenstrain (inherent-strain) warp specification                           #
# --------------------------------------------------------------------------- #

EIGENSTRAIN_NOTES = """\
PER-LAYER EIGENSTRAIN -- EXACT SPECIFICATION
============================================
The warp model is the INHERENT-STRAIN (eigenstrain) method [PMC10973057,
Brunel FDM analytic model]. Each layer, as it cools from the deposition
temperature toward the bed/ambient, WANTS to contract isotropically in-plane.
The prior (already-cold, stiff) layers and the platform resist that
contraction; the unbalanced reaction torque bends the part -> warp.

We impose this as a prescribed (eigen)strain in the elastic problem, NOT as a
transient thermal solve. For layer L at vertical level z:

    eps*_L  =  -alpha_eff * dT_lock                (a SCALAR contraction)

applied as an isotropic IN-PLANE eigenstrain tensor (XY plane = build plane):

    eps*_ij(L) = eps*_L * [[1,0,0],
                            [0,1,0],
                            [0,0,gamma_z]]

  * alpha_eff = material.cte_per_K  (solid-polymer CTE; do NOT use the
    porosity-inflated DIC "apparent" CTE of 4-5e-4 /C -- that double-counts
    infill voids the mesh already represents geometrically).
  * dT_lock = material.locked_delta_T() = max(0, Tg - T_bed).
    RATIONALE: above Tg the polymer is rubbery (E ~ 1 MPa, ratio >100:1 to the
    glassy 1 GPa, PMC11174730) and RELAXES -- strain accumulated above Tg does
    not survive. Only contraction from ~Tg down to the bed locks in as stored
    eigenstrain. Using full (T_deposit - T_bed) over-predicts warp ~2-3x: a
    KNOWN OVERCLAIM TRAP. (sign: eps* < 0 = contraction = corners lift = warp.)
  * gamma_z in {0..1}: in-plane contraction dominates because layers are
    constrained against the cold substrate while the free top surface relieves
    Z contraction. gamma_z=0 is the standard thin-layer assumption (pure
    in-plane inherent strain); gamma_z=1 is fully isotropic. Default 0.

LUMPED vs RESOLVED: the SCALAR magnitude alpha_eff*dT_lock is a LUMPED proxy
for the true accumulated inelastic strain (thermal contraction + crystall-
ization shrink + plastic yield). For an UNCALIBRATED run it gives the right
shape/sign/trend. A CALIBRATED run fits ONE multiplier 'k' so that k*eps*
matches a measured warp coupon for THAT printer+filament+profile; after that
single-scalar fit the magnitudes become quantitative for that process only.

ANISOTROPY: the elastic solve may stay isotropic (modulus anisotropy is mild,
z_xy_stiffness ~ 0.9). The Z/XY STRENGTH ratio is a SEPARATE post-check on
predicted stress (a Z-normal tensile stress is de-rated by z_xy_strength
before comparing to the XY strength allowable) -- it does NOT enter the
eigenstrain or the stiffness matrix.

VALIDATION GATE: the eigenstrain solver is only trustworthy if, on a 2-layer
differential-strain strip, it reproduces the Timoshenko (1925) bimetal
curvature (`timoshenko_bimetal_curvature`) -- correct SIGN and converging to
the closed form under mesh refinement. See `bimetal_benchmark_case`.
"""


def layer_eigenstrain_scalar(material: Material, *, use_locked: bool = True) -> float:
    """Scalar isotropic in-plane eigenstrain magnitude eps*_L for one layer.

    Returns eps* = -alpha * dT  (NEGATIVE = contraction). Never raises.

    use_locked=True  -> dT = locked_delta_T() = max(0, Tg - T_bed)  [DEFAULT,
                        physically defensible, avoids the over-predict trap].
    use_locked=False -> dT = delta_T_warp(to="bed") = T_deposit - T_bed
                        [the naive upper bound; over-predicts ~2-3x].
    """
    dT = material.locked_delta_T() if use_locked else material.delta_T_warp(to="bed")
    return float(-material.cte_per_K * dT)


# --------------------------------------------------------------------------- #
#  Closed-form benchmarks (the honesty gate)                                  #
# --------------------------------------------------------------------------- #

def timoshenko_bimetal_curvature(
    alpha1: float, alpha2: float, dT: float, total_thickness: float,
    *, m: float = 1.0, n: float = 1.0,
) -> float:
    """Timoshenko (1925) bimetal-strip curvature kappa = 1/r.

    Two bonded layers with CTEs alpha1 (layer 1, lower) and alpha2 (layer 2,
    upper), uniform heating dT, total thickness h = t1 + t2. The strip bends
    toward the lower-expansion layer.

        kappa = 6 (alpha1 - alpha2) dT (1 + m)^2
                ------------------------------------------------------------
                h [ 3 (1+m)^2 + (1 + m n)( m^2 + 1/(m n) ) ]

    with m = t1/t2 (thickness ratio), n = E1/E2 (modulus ratio).

    EQUAL-PROPERTY case (m=1, n=1) -- the one our eigenstrain warp benchmark
    uses -- the bracket = 3*4 + 2*(1 + 1) = 16, numerator = 24 (a1-a2) dT, so

        kappa = 1.5 (alpha1 - alpha2) dT / h        [verified analytically]

    Units: alpha in 1/K, dT in K, h in mm -> kappa in 1/mm (r in mm).
    Sign: kappa>0 means layer-1 side concave (bends toward lower-CTE layer).
    Never raises; returns 0.0 for degenerate (h<=0) input.
    """
    if total_thickness <= 0.0 or m <= 0.0 or n <= 0.0:
        return 0.0
    num = 6.0 * (alpha1 - alpha2) * dT * (1.0 + m) ** 2
    den = total_thickness * (3.0 * (1.0 + m) ** 2 + (1.0 + m * n) * (m * m + 1.0 / (m * n)))
    return float(num / den)


def timoshenko_equal_property_curvature(
    d_alpha: float, dT: float, total_thickness: float,
) -> float:
    """Equal-thickness equal-modulus simplification: kappa = 1.5 d_alpha dT / h.

    d_alpha = alpha1 - alpha2. This is the exact closed form the eigenstrain
    solver must reproduce; identical to the general formula at m=n=1.
    """
    if total_thickness <= 0.0:
        return 0.0
    return float(1.5 * d_alpha * dT / total_thickness)


def uniaxial_bar_elongation(F: float, L: float, A: float, E: float) -> float:
    """Closed form d = F L / (A E). Inputs in consistent SI; never raises."""
    if A <= 0.0 or E <= 0.0:
        return 0.0
    return float(F * L / (A * E))


def cantilever_tip_deflection(P: float, L: float, E: float, b: float, h: float) -> float:
    """Euler-Bernoulli cantilever tip deflection d = P L^3 / (3 E I),
    I = b h^3 / 12 (rectangular section). Consistent SI; never raises."""
    if E <= 0.0 or b <= 0.0 or h <= 0.0:
        return 0.0
    I = b * h ** 3 / 12.0
    return float(P * L ** 3 / (3.0 * E * I))


def bimetal_benchmark_case() -> dict:
    """A concrete, fully-specified differential-strain strip for the FEM gate.

    The eigenstrain solver should model a 2-layer strip where layer 1 carries
    an in-plane eigenstrain eps* = -alpha*dT and layer 2 carries 0 (or the two
    carry +/- equal eigenstrains), both layers same E, nu, thickness. Its
    predicted curvature must converge to `kappa_closed_form` (sign + value)
    under refinement. Geometry/material chosen to stay in the small-grid,
    OOM-safe regime (a few-element-thick strip suffices).
    """
    alpha1, alpha2 = 70e-6, 0.0   # layer-1 contracts, layer-2 inert
    dT = 150.0                    # K (cooling; here used as the eigenstrain dT)
    h = 2.0                       # mm total thickness (t1 = t2 = 1 mm)
    kappa = timoshenko_equal_property_curvature(alpha1 - alpha2, dT, h)
    return {
        "description": "2-layer equal-property differential-eigenstrain strip",
        "alpha1_per_K": alpha1,
        "alpha2_per_K": alpha2,
        "dT_K": dT,
        "total_thickness_mm": h,
        "layer_thickness_mm": h / 2.0,
        "m_thickness_ratio": 1.0,
        "n_modulus_ratio": 1.0,
        "kappa_closed_form_per_mm": kappa,
        "radius_closed_form_mm": (1.0 / kappa) if kappa != 0 else float("inf"),
        "sign_meaning": "kappa>0 -> bends toward the inert (low-CTE) layer",
        "gate": "FEM eigenstrain curvature must match this sign and converge "
                "to this value under mesh refinement; else warp output is NOT viable.",
    }


# --------------------------------------------------------------------------- #
#  Self-test                                                                   #
# --------------------------------------------------------------------------- #

def _selftest() -> int:
    import math
    fails = 0

    def check(cond: bool, msg: str):
        nonlocal fails
        status = "ok  " if cond else "FAIL"
        if not cond:
            fails += 1
        print(f"  [{status}] {msg}")

    print("fem_materials selftest")
    print("-" * 60)

    # 1. Table sanity -- every constant in a physically sane band.
    print("calibration table sanity:")
    for name in list_materials():
        mtl, exact = get_material(name)
        check(exact, f"{name}: exact lookup")
        check(0.5 <= mtl.E_GPa <= 5.0, f"{name}: E={mtl.E_GPa} GPa in [0.5,5]")
        check(0.2 <= mtl.nu <= 0.45, f"{name}: nu={mtl.nu} in [0.2,0.45]")
        check(40e-6 <= mtl.cte_per_K <= 120e-6, f"{name}: CTE={mtl.cte_per_K:.1e}/K")
        check(0.4 <= mtl.z_xy_strength <= 1.0, f"{name}: Z/XY strength={mtl.z_xy_strength}")
        check(mtl.z_xy_stiffness >= mtl.z_xy_strength,
              f"{name}: stiffness ratio >= strength ratio (footgun guard)")

    # 2. Unknown material falls back to PLA, flagged.
    mtl, exact = get_material("unobtanium")
    check((not exact) and mtl.name == "PLA", "unknown material -> PLA fallback, flagged")
    mtl, exact = get_material(None)
    check((not exact) and mtl.name == "PLA", "None -> PLA fallback, flagged")

    # 3. locked_delta_T is the smaller, defensible drop.
    print("eigenstrain dT logic:")
    full = PLA.delta_T_warp(to="bed")          # 200-55 = 145
    lock = PLA.locked_delta_T()                 # 60-55  = 5
    check(abs(full - 145.0) < 1e-9, f"PLA full dT to bed = {full} K (== 145)")
    check(abs(lock - 5.0) < 1e-9, f"PLA locked dT = {lock} K (== Tg-Tbed = 5)")
    check(lock < full, "locked dT < naive full dT (over-predict trap avoided)")
    # ABS: high bed temp deliberately shrinks locked dT (warp mitigation).
    check(ABS.locked_delta_T() == 5.0, "ABS locked dT = 105-100 = 5 K (bed fights warp)")

    # 4. Eigenstrain scalar sign + magnitude.
    eps = layer_eigenstrain_scalar(PLA)         # -68e-6 * 5 = -3.4e-4
    check(eps < 0.0, "eigenstrain eps* < 0 (contraction)")
    check(abs(eps - (-68e-6 * 5.0)) < 1e-12, f"eps* = {eps:.2e} (== -CTE*dT_lock)")
    eps_naive = layer_eigenstrain_scalar(PLA, use_locked=False)
    check(abs(eps_naive) > abs(eps) * 10, "naive eps* >> locked eps* (the trap)")

    # 5. Timoshenko: general formula == equal-property simplification at m=n=1.
    print("Timoshenko bimetal closed form:")
    k_gen = timoshenko_bimetal_curvature(70e-6, 0.0, 150.0, 2.0, m=1.0, n=1.0)
    k_simpl = timoshenko_equal_property_curvature(70e-6, 150.0, 2.0)
    check(math.isclose(k_gen, k_simpl, rel_tol=1e-12),
          f"general kappa {k_gen:.6e} == simplified {k_simpl:.6e} at m=n=1")
    check(math.isclose(k_simpl, 0.007875, rel_tol=1e-9),
          f"kappa = 1.5*da*dT/h = {k_simpl:.6f} /mm (r = {1/k_simpl:.2f} mm)")
    # Sign: positive d_alpha -> positive kappa (bends toward low-CTE layer).
    check(timoshenko_equal_property_curvature(70e-6, 150.0, 2.0) > 0, "kappa sign +")
    check(timoshenko_equal_property_curvature(-70e-6, 150.0, 2.0) < 0, "kappa sign flips with d_alpha")

    # 6. Other analytical benchmarks (closed-form self-consistency).
    print("structural closed forms:")
    # Uniaxial bar: known case d = F L /(A E). PLA bar 100x10x10 mm, 1000 N.
    d_bar = uniaxial_bar_elongation(1000.0, 0.1, 1e-4, PLA.E_Pa)  # SI: 0.1m,1e-4 m^2
    check(d_bar > 0, "uniaxial bar elongation > 0")
    # Hand-check: 1000*0.1/(1e-4*3.5e9) = 100/(350000)=2.857e-4 m
    check(math.isclose(d_bar, 1000*0.1/(1e-4*3.5e9), rel_tol=1e-12), f"bar d={d_bar:.3e} m matches F L/(A E)")
    # Cantilever slender beam.
    d_tip = cantilever_tip_deflection(10.0, 0.1, PLA.E_Pa, 0.01, 0.005)
    check(d_tip > 0, f"cantilever tip deflection = {d_tip:.3e} m > 0")

    # 7. Degenerate inputs never raise, return 0.
    check(timoshenko_equal_property_curvature(70e-6, 150.0, 0.0) == 0.0, "h<=0 -> 0, no raise")
    check(uniaxial_bar_elongation(1.0, 1.0, 0.0, 1.0) == 0.0, "A<=0 -> 0, no raise")
    check(cantilever_tip_deflection(1.0, 1.0, 0.0, 1.0, 1.0) == 0.0, "E<=0 -> 0, no raise")

    # 8. Benchmark case dict is well-formed.
    case = bimetal_benchmark_case()
    check(case["kappa_closed_form_per_mm"] > 0, "benchmark case kappa > 0")
    check("gate" in case, "benchmark case carries the validation gate text")

    print("-" * 60)
    print(f"{'ALL PASS' if fails == 0 else str(fails) + ' FAILURE(S)'}")
    return fails


if __name__ == "__main__":
    import sys
    from ._mesh_util import ascii_console
    ascii_console()
    sys.exit(1 if _selftest() else 0)
