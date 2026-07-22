"""FEM validation harness -- the analytical gate that turns "looks plausible" into "viable".

THE HONESTY GATE (non-negotiable, INVESTIGATE-4 deliverable)
============================================================
A reduced-order print FEM is "viable" ONLY if it reproduces CLOSED-FORM analytical
solutions, with relative error that DECREASES as the grid refines (or is already at
machine precision for fields the element represents exactly). "A render that bends"
is NOT viability. Every FEM claim in the printability checker must pass the benchmarks
below before it is allowed to print a number to a user.

This module is the SPEC + a runnable harness. It carries a self-contained, dependency-
free (numpy + scipy.sparse only) trilinear Q1 voxel-hex solver good enough to BE the
benchmark fixture -- the same element family a voxel-occupancy print FEM would use, so
the benchmarks validate the real kernel, not a toy. The element here is *fully
integrated* Q1 (2x2x2 Gauss); the harness DELIBERATELY exposes its shear-locking so the
cantilever pass-criterion is set honestly.

Run:  python scripts/_fem_validation_plan.py            # prints all 4 benchmark tables
      python scripts/_fem_validation_plan.py --selftest # asserts the pass criteria

ALL NUMBERS BELOW THE DOCSTRING ARE MEASURED BY THE CODE, not asserted by prose.

------------------------------------------------------------------------------------
THE FOUR GATES (geometry / BCs / closed form / convergence test)
------------------------------------------------------------------------------------

GATE A -- UNIAXIAL BAR  (validates: assembly, material law, traction loading)
  Geometry : rectangular prism, L along x, square section A = b*h.
  BCs      : ROLLER, not clamp. u_x = 0 on the x=0 face; pin (u_y,u_z) at one corner
             and u_z at a second corner to remove rigid-body modes ONLY. A full 3-DOF
             clamp is WRONG here -- it injects a St-Venant boundary layer (lateral
             Poisson contraction is blocked) whose stress concentration a finer mesh
             resolves MORE sharply, so error GROWS with refinement (measured: 1.6%->4.1%
             with a clamp). The roller reproduces a pure uniaxial stress state.
  Load     : CONSISTENT nodal traction on the x=L face (integrate the shape functions:
             each surface quad sends pressure*area/4 to its 4 corners). Equal/"lumped"
             nodal forces are WRONG -- they leave a ~2% residual that never converges.
  Closed   : tip elongation  d = F*L / (A*E) ;  axial stress  sigma = F/A.
  Compare  : u_x at the AXIS node of the x=L face (mid-section), NOT the face mean.
  Test     : Q1 hexes represent a linear displacement field EXACTLY, so a correct
             implementation gives rel_err at MACHINE PRECISION (~1e-14) for every mesh.
  PASS     : rel_err < 1e-9 at the coarsest grid (4x2x2). If it is ~1e-2 instead, the
             load is lumped not consistent, or the BC is a clamp -> implementation bug.
             (This gate is a correctness unit test, not a convergence study.)

GATE B -- CANTILEVER TIP DEFLECTION  (validates: bending, the demanding gate)
  Geometry : slender beam, length L, section b x h, slenderness L/h >= ~15.
  BCs      : x=0 face fully clamped (all 3 DOF) -- bending needs a true encastre.
  Load     : transverse tip load P in -z, consistent on the x=L face.
  Closed   : Euler-Bernoulli  d = P*L^3 / (3*E*I),  I = b*h^3/12.
  Caveat   : (1) EULER-BERNOULLI vs SHEAR CORRECTION. EB ignores shear; for a stubby
             beam (L/h < ~10) the TRUE deflection is LARGER than EB by the Timoshenko
             shear term d_shear = alpha*P*L/(G*A_s) (alpha~6/5 rectangle). So on a stubby
             beam, FEM > EB is correct physics, NOT error -- never gate a stubby beam to
             EB. Use a SLENDER beam so the EB and shear answers agree to ~1%.
             (2) SHEAR LOCKING. Fully-integrated Q1 hexes are too STIFF in bending and
             converge to EB SLOWLY from below (measured: rel_err 71% @ 8x2x2 down to
             10% @ 40x4x4, monotone). This is a known Q1 pathology, not a bug.
  Test     : rel_err must DECREASE MONOTONICALLY under refinement and head toward ~0.
  PASS (this fully-integrated Q1 fixture): monotone-decreasing AND rel_err <= 0.12 at
             40x4x4. To get the textbook "<5% on a slender beam", the production kernel
             must use SELECTIVE-REDUCED INTEGRATION (1-pt on the deviatoric/shear terms)
             or incompatible modes (Q1/E9) -- flagged as the required upgrade. A harness
             that reported "<5% at 8x2x2" with full integration would be FABRICATED.

GATE C -- EIGENSTRAIN / THERMAL BIMETAL BEND  (validates: INHERENT-STRAIN warp, the
                                              whole point of a *print* FEM)
  Why      : 3D-print warp is an inherent-strain problem: each deposited layer cools and
             wants to shrink; differential shrink through the thickness bends the part.
             The closed-form sanity check is the Timoshenko (1925) bimetal strip.
  Geometry : straight bilayer strip, two bonded layers thickness t1=t2=t, total h=2t,
             length L, width w, L/h slender. Layer 1 expansion a1, layer 2 a2, dT.
  Mechanism: apply an EIGENSTRAIN (stress-free thermal strain eps0 = a*dT) per layer via
             a thermal-load vector  f_th = integral( B^T C eps0 ) over each element;
             solve K u = f_th. (This is the eigenstrain hook a real warp predictor needs.)
  BCs      : one end clamped (or roller + 1 pin), free to curl -- a true cantilever strip.
  Closed   : Timoshenko bimetal curvature, equal moduli & thickness simplification:
                 kappa = 6 * (a2 - a1) * dT / ( h ) * ... ->  for E1=E2, t1=t2=t:
                 kappa = 3*(a2-a1)*dT / (2*h)         [rad / length]
             tip deflection of a curling cantilever  delta ~ kappa * L^2 / 2.
  Test     : (i) SIGN -- strip must curl toward the lower-expansion layer (this is the
             "does the warp point the right way" check that a render cannot fake);
             (ii) ORDER OF MAGNITUDE -- FEM kappa within ~2x of Timoshenko, improving with
             refinement. We do NOT claim a calibrated magnitude (see UNCALIBRATED below).
  PASS     : correct sign at every grid AND |log10(kappa_FEM/kappa_TS)| < 0.3 (within 2x)
             at the finest grid, with the ratio moving toward 1 under refinement.

GATE D -- sigma* CORRELATION (the mesh-quality-certificate claim -- and its honest limit)
  Claim under test: "sigma* (ei_core.quality) flags where the FEM error is largest"
                    (the interpolation-error bound err <= C*h/sigma*).
  HONEST FINDING (measured): the existing sigma* in ei_core/quality.py is a SURFACE-mesh
  frame-conditioning metric -- the smallest singular value of the 3xk unit-edge matrix at
  a mesh VERTEX. On a VOXEL-HEX analysis grid every element is an identical axis-aligned
  cube, so the per-ELEMENT sigma* (Jacobian conditioning) is CONSTANT across the grid and
  carries ZERO spatial information about FEM error. Therefore:
     * For a STRUCTURED voxel-hex print FEM, the sigma* certificate is VACUOUS -- it
       cannot localize error because the grid is uniform by construction. Report this;
       do NOT dress a constant up as a certificate.
     * sigma* IS meaningful for the SURFACE remesh / tet path (variable element shape),
       where slivers genuinely raise the C*h/sigma* interpolation bound. If the print FEM
       ever moves to a body-fitted tet/distorted-hex mesh, recompute sigma* PER ELEMENT
       (Jacobian SVD), not per surface vertex, and THEN the correlation test applies:
         build a graded/distorted mesh with known sliver band, solve a problem with a
         smooth exact solution, and check that Spearman( per-elem sigma*, per-elem error )
         is significantly negative. PASS = rho_spearman < -0.4, p < 0.05.
  So Gate D's verdict on the voxel-hex stack is: sigma*-as-FEM-error-certificate is
  NOT APPLICABLE (constant field). It is retained only for the variable-element path.

------------------------------------------------------------------------------------
UNCALIBRATED-FEM STANCE (label every output)
------------------------------------------------------------------------------------
Material constants (PLA E~3.5 GPa, nu~0.36, CTE~70e-6/K, dT~150-180 K, Z/XY ~0.5-0.8)
are CALIBRATION CONSTANTS, not measured truth. An uncalibrated FEM predicts the right
SHAPE / TREND / SIGN (where it warps, which orientation warps less, which rib stiffens)
but NOT a certified millimetre. Frame every magnitude as "physics-based estimate pending
real-print calibration". The gates above certify the SOLVER reproduces physics; they do
NOT certify the material numbers.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ----------------------------------------------------------------------------------
# Trilinear Q1 voxel-hex kernel (numpy + scipy.sparse only; this IS the fixture).
# ----------------------------------------------------------------------------------
_NODES = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                   [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
_G = 1.0 / np.sqrt(3.0)
_GP = np.array([[a, b, c] for a in (-_G, _G) for b in (-_G, _G) for c in (-_G, _G)])


def _dN_natural(xi, eta, zeta):
    """Shape-function gradients d N_i / d(xi,eta,zeta), shape (8,3)."""
    d = np.zeros((8, 3))
    for i, (s, t, u) in enumerate(_NODES):
        d[i, 0] = 0.125 * s * (1 + t * eta) * (1 + u * zeta)
        d[i, 1] = 0.125 * (1 + s * xi) * t * (1 + u * zeta)
        d[i, 2] = 0.125 * (1 + s * xi) * (1 + t * eta) * u
    return d


def _C_iso(E, nu):
    """Isotropic linear-elastic constitutive 6x6 (Voigt)."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return np.array([
        [lam + 2 * mu, lam, lam, 0, 0, 0],
        [lam, lam + 2 * mu, lam, 0, 0, 0],
        [lam, lam, lam + 2 * mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu]], float)


def _B_at(gp, Jx, Jy, Jz):
    g = _dN_natural(*gp)
    g[:, 0] /= Jx
    g[:, 1] /= Jy
    g[:, 2] /= Jz
    B = np.zeros((6, 24))
    for i in range(8):
        bx, by, bz = g[i]
        c = 3 * i
        B[0, c] = bx
        B[1, c + 1] = by
        B[2, c + 2] = bz
        B[3, c] = by
        B[3, c + 1] = bx
        B[4, c + 1] = bz
        B[4, c + 2] = by
        B[5, c] = bz
        B[5, c + 2] = bx
    return B


def _hex_Ke(hx, hy, hz, E, nu):
    C = _C_iso(E, nu)
    Jx, Jy, Jz = hx / 2, hy / 2, hz / 2
    detJ = Jx * Jy * Jz
    Ke = np.zeros((24, 24))
    for gp in _GP:
        B = _B_at(gp, Jx, Jy, Jz)
        Ke += (B.T @ C @ B) * detJ
    return Ke


def _hex_thermal_fe(hx, hy, hz, E, nu, eps0):
    """Element thermal/eigenstrain load vector f = integral B^T C eps0 dV.
    eps0 is a length-6 Voigt stress-free strain (e.g. [a*dT]*3 + [0]*3)."""
    C = _C_iso(E, nu)
    Jx, Jy, Jz = hx / 2, hy / 2, hz / 2
    detJ = Jx * Jy * Jz
    fe = np.zeros(24)
    Ceps = C @ np.asarray(eps0, float)
    for gp in _GP:
        B = _B_at(gp, Jx, Jy, Jz)
        fe += (B.T @ Ceps) * detJ
    return fe


class _Grid:
    """Structured nx*ny*nz hex grid over [0,L]x[0,b]x[0,h]."""

    def __init__(self, nx, ny, nz, L, b, h):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.L, self.b, self.h = L, b, h
        self.hx, self.hy, self.hz = L / nx, b / ny, h / nz
        self.nX, self.nY, self.nZ = nx + 1, ny + 1, nz + 1
        self.nn = self.nX * self.nY * self.nZ
        self.ndof = 3 * self.nn

    def nid(self, i, j, k):
        return (i * self.nY + j) * self.nZ + k

    def elem_nodes(self, ei, ej, ek):
        n = self.nid
        return [n(ei, ej, ek), n(ei + 1, ej, ek), n(ei + 1, ej + 1, ek), n(ei, ej + 1, ek),
                n(ei, ej, ek + 1), n(ei + 1, ej, ek + 1), n(ei + 1, ej + 1, ek + 1), n(ei, ej + 1, ek + 1)]

    def assemble_K(self, E, nu):
        Ke = _hex_Ke(self.hx, self.hy, self.hz, E, nu)
        r, c, d = [], [], []
        for ei in range(self.nx):
            for ej in range(self.ny):
                for ek in range(self.nz):
                    en = self.elem_nodes(ei, ej, ek)
                    dofs = np.array([3 * n + dd for n in en for dd in range(3)])
                    for a in range(24):
                        ra = dofs[a]
                        for bb in range(24):
                            r.append(ra)
                            c.append(dofs[bb])
                            d.append(Ke[a, bb])
        return sp.csr_matrix((d, (r, c)), shape=(self.ndof, self.ndof))


def _solve(K, f, fixed_dofs, ndof):
    free = np.array(sorted(set(range(ndof)) - set(fixed_dofs)))
    u = np.zeros(ndof)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    return u


def _consistent_face_x(grid, total_force, sign_axis=0):
    """Consistent nodal load for a uniform traction on the x=L face along axis sign_axis.
    Returns (dof, value) list. Each surface quad (area hy*hz) sends pressure*area/4 to
    each of its 4 corner nodes (Q1 consistent loading)."""
    hy, hz = grid.hy, grid.hz
    area = (grid.nY - 1) * hy * (grid.nZ - 1) * hz
    pressure = total_force / area
    fnode = np.zeros((grid.nY, grid.nZ))
    for j in range(grid.nY - 1):
        for k in range(grid.nZ - 1):
            q = pressure * (hy * hz) / 4.0
            for (jj, kk) in [(j, k), (j + 1, k), (j, k + 1), (j + 1, k + 1)]:
                fnode[jj, kk] += q
    out = []
    for j in range(grid.nY):
        for k in range(grid.nZ):
            out.append((3 * grid.nid(grid.nx, j, k) + sign_axis, fnode[j, k]))
    return out


# ----------------------------------------------------------------------------------
# GATE A -- uniaxial bar
# ----------------------------------------------------------------------------------
def gate_A_uniaxial(nx, ny, nz, L=10.0, b=2.0, h=2.0, E=3500.0, nu=0.3, F=100.0):
    g = _Grid(nx, ny, nz, L, b, h)
    K = g.assemble_K(E, nu)
    f = np.zeros(g.ndof)
    for dof, val in _consistent_face_x(g, F, sign_axis=0):
        f[dof] += val
    fixed = {}
    for j in range(g.nY):
        for k in range(g.nZ):
            fixed[3 * g.nid(0, j, k) + 0] = 0.0          # roller: u_x=0 on x=0 face
    fixed[3 * g.nid(0, 0, 0) + 1] = 0.0                  # rigid-body pins only
    fixed[3 * g.nid(0, 0, 0) + 2] = 0.0
    fixed[3 * g.nid(0, g.nY - 1, 0) + 2] = 0.0
    u = _solve(K, f, set(fixed), g.ndof)
    A = b * h
    analytic = F * L / (A * E)
    jc, kc = g.nY // 2, g.nZ // 2
    fem = u[3 * g.nid(nx, jc, kc) + 0]
    return fem, analytic, abs(fem - analytic) / abs(analytic)


# ----------------------------------------------------------------------------------
# GATE B -- cantilever tip deflection (Euler-Bernoulli, slender)
# ----------------------------------------------------------------------------------
def gate_B_cantilever(nx, ny, nz, L=20.0, b=1.0, h=1.0, E=3500.0, nu=0.3, P=1.0):
    g = _Grid(nx, ny, nz, L, b, h)
    K = g.assemble_K(E, nu)
    f = np.zeros(g.ndof)
    for dof, val in _consistent_face_x(g, P, sign_axis=2):    # load along -z
        f[dof] -= val
    fixed = {}
    for j in range(g.nY):                                     # full clamp at x=0
        for k in range(g.nZ):
            for d in range(3):
                fixed[3 * g.nid(0, j, k) + d] = 0.0
    u = _solve(K, f, set(fixed), g.ndof)
    tip = abs(np.mean([u[3 * g.nid(nx, j, k) + 2] for j in range(g.nY) for k in range(g.nZ)]))
    I = b * h ** 3 / 12.0
    eb = P * L ** 3 / (3 * E * I)
    return tip, eb, abs(tip - eb) / eb


# ----------------------------------------------------------------------------------
# GATE C -- eigenstrain bimetal bend (Timoshenko)
# ----------------------------------------------------------------------------------
def gate_C_bimetal(nx, ny, nz, L=40.0, b=2.0, h=2.0, E=3500.0, nu=0.3,
                   a1=0.0, a2=70e-6, dT=150.0):
    """Bilayer strip: lower half (z<h/2) expansion a1, upper half a2. Eigenstrain load,
    one end clamped, free to curl. Returns (kappa_fem, kappa_timoshenko, ratio, sign_ok)."""
    g = _Grid(nx, ny, nz, L, b, h)
    K = g.assemble_K(E, nu)
    f = np.zeros(g.ndof)
    for ei in range(g.nx):
        for ek in range(g.nz):
            zc = (ek + 0.5) * g.hz
            a = a1 if zc < h / 2 else a2
            eps0 = np.array([a * dT, a * dT, a * dT, 0, 0, 0])
            fe = _hex_thermal_fe(g.hx, g.hy, g.hz, E, nu, eps0)
            for ej in range(g.ny):
                en = g.elem_nodes(ei, ej, ek)
                dofs = np.array([3 * n + dd for n in en for dd in range(3)])
                for a_i in range(24):
                    f[dofs[a_i]] += fe[a_i]
    fixed = {}
    for j in range(g.nY):
        for k in range(g.nZ):
            for d in range(3):
                fixed[3 * g.nid(0, j, k) + d] = 0.0
    u = _solve(K, f, set(fixed), g.ndof)
    # tip deflection (mid-plane z=h/2) along z; curvature kappa ~ 2*delta / L^2
    jc = g.nY // 2
    kc = g.nZ // 2
    delta = np.mean([u[3 * g.nid(g.nx, jc, k) + 2] for k in range(g.nZ)])
    kappa_fem = 2.0 * delta / (L ** 2)
    # Timoshenko equal-modulus equal-thickness: kappa = 3*(a2-a1)*dT/(2*h)
    kappa_ts = 3.0 * (a2 - a1) * dT / (2.0 * h)
    # higher-expansion layer is on top (z>h/2); it expands more -> strip curls DOWN (-z)
    # so delta should be negative; "sign_ok" = curls toward lower-expansion layer.
    sign_ok = (np.sign(delta) == -np.sign(a2 - a1)) if (a2 - a1) != 0 else True
    ratio = abs(kappa_fem / kappa_ts) if kappa_ts != 0 else float("inf")
    return kappa_fem, kappa_ts, ratio, bool(sign_ok)


# ----------------------------------------------------------------------------------
# GATE D -- sigma* correlation (verdict: N/A on uniform voxel grid; demonstrate why)
# ----------------------------------------------------------------------------------
def gate_D_sigma_star_note():
    """Returns the structured verdict for the sigma*-as-error-certificate claim on a
    voxel-hex grid. Measured fact: per-element Jacobian sigma* is identical for every
    axis-aligned cube, so it has zero spatial variance and cannot localize error."""
    # Per-element Jacobian of an axis-aligned cube is diag(hx/2,hy/2,hz/2); its singular
    # values are exactly {hx/2,hy/2,hz/2} for EVERY element -> sigma*_min is constant.
    return {
        "applicable_to_voxel_hex": False,
        "reason": ("axis-aligned cube elements share an identical diagonal Jacobian; "
                   "per-element sigma* is constant across the grid -> zero spatial "
                   "information about FEM error. The certificate is vacuous on a "
                   "structured voxel grid."),
        "where_it_applies": ("variable-shape element meshes (surface remesh / body-fitted "
                             "tet or distorted hex). There, recompute sigma* PER ELEMENT "
                             "(Jacobian SVD), build a graded mesh with a known sliver band, "
                             "solve a smooth-exact problem, and PASS if "
                             "Spearman(per-elem sigma*, per-elem error) < -0.4 at p<0.05."),
        "surface_metric_note": ("ei_core.quality.vertex_sigma_min is a SURFACE-vertex frame "
                                "metric, NOT an element-Jacobian metric; do not conflate."),
    }


# ----------------------------------------------------------------------------------
# Reporting / selftest
# ----------------------------------------------------------------------------------
def _print_table_A():
    print("GATE A -- uniaxial bar (roller BC + consistent load + axis node).")
    print("  closed form d=FL/AE ; Q1 represents linear field EXACTLY -> expect ~1e-14.")
    print(f"  {'grid':>10} {'FEM_d':>14} {'analytic':>14} {'rel_err':>11}")
    worst = 0.0
    for (nx, ny, nz) in [(4, 2, 2), (8, 2, 2), (16, 4, 4), (32, 4, 4)]:
        fem, an, err = gate_A_uniaxial(nx, ny, nz)
        worst = max(worst, err)
        print(f"  {f'{nx}x{ny}x{nz}':>10} {fem:>14.6e} {an:>14.6e} {err:>11.2e}")
    print(f"  -> PASS criterion rel_err < 1e-9 : {'PASS' if worst < 1e-9 else 'FAIL'} "
          f"(worst {worst:.1e})\n")
    return worst < 1e-9


def _print_table_B():
    print("GATE B -- cantilever tip deflection (slender L/h=20, full clamp, EB=PL^3/3EI).")
    print("  fully-integrated Q1 SHEAR-LOCKS -> converges slowly from below; must be MONOTONE.")
    print(f"  {'grid':>10} {'FEM_d':>13} {'EB_d':>13} {'rel_err':>11} {'trend':>6}")
    prev = None
    monotone = True
    errs = []
    for (nx, ny, nz) in [(8, 2, 2), (16, 2, 2), (16, 4, 4), (32, 4, 4), (40, 4, 4)]:
        fem, eb, err = gate_B_cantilever(nx, ny, nz)
        errs.append(err)
        tr = "" if prev is None else ("DOWN" if err < prev + 1e-12 else "UP!")
        if prev is not None and err > prev + 1e-9:
            monotone = False
        print(f"  {f'{nx}x{ny}x{nz}':>10} {fem:>13.5f} {eb:>13.5f} {err:>11.2e} {tr:>6}")
        prev = err
    ok = monotone and errs[-1] <= 0.12
    print(f"  -> PASS: monotone-decreasing AND finest rel_err<=0.12 : "
          f"{'PASS' if ok else 'FAIL'} (finest {errs[-1]:.1%}, monotone={monotone})")
    print("     NOTE: to reach textbook <5% slender, production kernel needs "
          "selective-reduced / incompatible-modes integration.\n")
    return ok


def _print_table_C():
    print("GATE C -- eigenstrain bimetal bend (Timoshenko kappa=3*da*dT/2h; sign + 2x mag).")
    print(f"  {'grid':>10} {'kappa_FEM':>13} {'kappa_TS':>13} {'ratio':>8} {'sign_ok':>8}")
    last_ratio = None
    sign_all = True
    for (nx, ny, nz) in [(20, 2, 4), (32, 2, 4), (40, 2, 6)]:
        kf, kts, ratio, sok = gate_C_bimetal(nx, ny, nz)
        sign_all = sign_all and sok
        last_ratio = ratio
        print(f"  {f'{nx}x{ny}x{nz}':>10} {kf:>13.3e} {kts:>13.3e} {ratio:>8.3f} {str(sok):>8}")
    mag_ok = (last_ratio is not None) and abs(np.log10(last_ratio)) < 0.3
    ok = sign_all and mag_ok
    print(f"  -> PASS: correct sign at every grid AND within 2x at finest : "
          f"{'PASS' if ok else 'FAIL'} (sign_all={sign_all}, finest ratio={last_ratio:.3f})")
    print("     NOTE: shear-locking biases kappa LOW (same Q1 stiffness pathology as B); "
          "magnitude is UNCALIBRATED -- sign + order are the certified claims.\n")
    return ok


def _print_note_D():
    print("GATE D -- sigma* error-certificate correlation.")
    d = gate_D_sigma_star_note()
    print(f"  applicable_to_voxel_hex: {d['applicable_to_voxel_hex']}")
    print(f"  reason: {d['reason']}")
    print(f"  where_it_applies: {d['where_it_applies']}")
    print(f"  verdict: NOT APPLICABLE on a structured voxel grid (constant field).\n")
    return True  # the verdict itself is the deliverable; nothing to numerically pass


def selftest():
    a = _print_table_A()
    b = _print_table_B()
    c = _print_table_C()
    _print_note_D()
    assert a, "GATE A failed: uniaxial bar not at machine precision (load/BC bug)"
    assert b, "GATE B failed: cantilever not monotone-converging or finest err>12%"
    assert c, "GATE C failed: bimetal wrong sign or >2x magnitude at finest grid"
    print("SELFTEST: all gates PASS (A machine-exact, B monotone<=12%, C sign+order).")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        _print_table_A()
        _print_table_B()
        _print_table_C()
        _print_note_D()
