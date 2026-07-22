"""PROBE-1: voxel/hex linear-elastic FEM CORE for the EI Forge print-FEM.

THE GATE. This module is the foundation the entire reduced-order print-FEM
rests on. It is only "real" if it REPRODUCES CLOSED-FORM ANALYTICAL SOLUTIONS
under mesh refinement -- relative error must DECREASE as the grid refines.
Everything downstream (warp / inherent-strain, strength de-rating) is built on
the solver validated here; if this gate fails, none of the print-FEM is real.

WHAT THIS IS
------------
A thin, deps-light WRAP of scikit-fem (BSD-3) providing:
  * VoxelHexFEM : a structured trilinear-hex linear-elasticity solver that can
    be built from EITHER a structured box (for analytical validation) OR an
    arbitrary occupancy grid (keep only SOLID voxels -> MeshHex elements), the
    path the real print-FEM uses. No tet mesher, no CGAL/TetGen GPL trap --
    MeshHex is structured, so the stack stays fully PERMISSIVE
    (scikit-fem BSD-3, pyamg MIT, scipy/numpy BSD).
  * face-based Dirichlet (clamp) + Neumann (traction) BC tagging,
  * scipy.sparse direct solve below ~12k dofs, pyamg SA-AMG + CG above,
  * the two structural benchmarks with MEASURED convergence tables:
        B1 uniaxial bar       d = F L /(A E)            (validates assembly+BC)
        B2 cantilever tip     d = P L^3 /(3 E I)        (validates bending)
  * a cantilever-deflection render (PNG).

HONESTY / VIABILITY GATE (what `selftest` actually checks)
----------------------------------------------------------
  * BAR: trilinear hex captures the linear axial field EXACTLY -> rel_err must
    be < 1% at every grid (in practice ~1e-13). This proves the assembly,
    material law, traction load, and clamp BC are all correct.
  * CANTILEVER: a coarse trilinear hex SHEAR-LOCKS in bending (too stiff), so
    d_fem < d_EB and the ratio d_fem/d_EB rises toward 1 FROM BELOW as the mesh
    refines. This is the correct, documented hex artifact -- NOT a bug. The
    viability evidence is the convergence DIRECTION and that the gap CLOSES
    under refinement. We assert: ratio monotone increasing AND finest ratio
    within a few % of 1 (>= 0.97 at the validation grid). The residual gap is
    the FEM still being slightly too stiff (locking), decreasing with h.
    [Physical aside: the true shear-corrected (Timoshenko) deflection for an
     L/h=10 beam is only +0.78% above EB, so the FEM converging up toward ~1.0
     from below is converging toward the physical answer, not overshooting.]

UNCALIBRATED-MODEL DISCIPLINE: this core solves linear elasticity exactly for
its inputs; magnitudes are only as good as the (material) constants fed in.
Downstream print outputs remain physics-based ESTIMATES of shape/sign/trend
pending real-print calibration -- same labelled discipline as the rest of the
toolkit. This module itself, however, is validated against EXACT closed forms,
so the SOLVER is certified even though a print PREDICTION is not.

Run:  python scripts/fem_voxel_core.py        (prints tables, writes render,
                                               asserts the gate, exit 0/1)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

# single-thread the numerics (compute-safety: 16GB shared RAM, user keeps CPU)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    import skfem
    from skfem import (
        MeshHex, ElementHex1, ElementVector, Basis, FacetBasis,
        LinearForm, asm, condense,
    )
    from skfem.models.elasticity import linear_elasticity, lame_parameters
    from skfem.helpers import sym_grad
    _HAVE_SKFEM = True
    _IMPORT_ERR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    _HAVE_SKFEM = False
    _IMPORT_ERR = _e

try:
    import pyamg
    _HAVE_PYAMG = True
except Exception:  # pragma: no cover
    _HAVE_PYAMG = False


# --------------------------------------------------------------------------- #
#  small helpers
# --------------------------------------------------------------------------- #
def lame(E: float, nu: float) -> tuple[float, float]:
    """(lambda, mu) Lame parameters from (E, nu)."""
    return lame_parameters(E, nu)


def _solve_sparse(K: sp.spmatrix, f: np.ndarray, *, amg_dof_threshold: int = 12000):
    """Solve K u = f. scipy direct below threshold, pyamg SA-AMG+CG above.

    Returns (u, info) where info records which solver ran. Never raises on a
    convergence stall: falls back to spsolve if CG does not converge.
    The SOLVE is cheap (1-2 s even at ~15k dofs); assembly dominates runtime.
    """
    n = f.shape[0]
    Kc = K.tocsr()
    if n < amg_dof_threshold or not _HAVE_PYAMG:
        u = spla.spsolve(Kc, f)
        return np.asarray(u).ravel(), {"solver": "spsolve", "dofs": n}
    # large: smoothed-aggregation AMG as a CG preconditioner
    ml = pyamg.smoothed_aggregation_solver(Kc)
    M = ml.aspreconditioner()
    u, conv = spla.cg(Kc, f, M=M, rtol=1e-10, maxiter=2000)
    if conv != 0:  # stalled -> exact fallback
        u = spla.spsolve(Kc, f)
        return np.asarray(u).ravel(), {"solver": "spsolve(amg-fallback)", "dofs": n}
    return np.asarray(u).ravel(), {"solver": "pyamg-cg", "dofs": n}


# --------------------------------------------------------------------------- #
#  the core solver
# --------------------------------------------------------------------------- #
@dataclass
class FEMResult:
    """Output of a VoxelHexFEM solve."""
    u: np.ndarray                 # full displacement dof vector
    basis: "skfem.Basis"          # the vector basis (for post-processing)
    mesh: "skfem.MeshHex"
    n_elements: int
    n_dofs: int
    solver: str
    assemble_s: float
    solve_s: float

    def nodal_displacement(self) -> np.ndarray:
        """(n_nodes, 3) array of nodal displacements (x,y,z).

        skfem orders ElementVector(ElementHex1) nodal dofs as
        [node0_x, node0_y, node0_z, node1_x, ...], so a reshape recovers them.
        """
        nnodes = self.mesh.p.shape[1]
        if self.u.shape[0] == 3 * nnodes:
            return self.u.reshape(nnodes, 3)
        return np.zeros((nnodes, 3))  # should not happen for vector ElementHex1


class VoxelHexFEM:
    """Structured trilinear-hex linear-elasticity solver (scikit-fem wrap).

    Build from a structured box (validation) or an occupancy grid (real parts).
    Pure-ish: methods return new objects; the instance holds the assembled mesh
    + basis so multiple load cases can reuse one assembly.
    """

    def __init__(self, mesh: "MeshHex", E: float, nu: float):
        if not _HAVE_SKFEM:
            raise RuntimeError(f"scikit-fem unavailable: {_IMPORT_ERR}")
        self.mesh = mesh
        self.E = float(E)
        self.nu = float(nu)
        self.basis = Basis(mesh, ElementVector(ElementHex1()))
        lam, mu = lame(E, nu)
        self._lam, self._mu = lam, mu
        t0 = time.time()
        self.K = asm(linear_elasticity(lam, mu), self.basis)
        self.assemble_s = time.time() - t0

    # ---- builders -------------------------------------------------------- #
    @classmethod
    def from_box(cls, L, W, H, nx, ny, nz, E, nu) -> "VoxelHexFEM":
        """Structured hex box [0,L]x[0,W]x[0,H], nx*ny*nz elements."""
        xs = np.linspace(0.0, L, nx + 1)
        ys = np.linspace(0.0, W, ny + 1)
        zs = np.linspace(0.0, H, nz + 1)
        return cls(MeshHex.init_tensor(xs, ys, zs), E, nu)

    @classmethod
    def from_occupancy(cls, occ: np.ndarray, pitch: float, E: float, nu: float,
                       origin=(0.0, 0.0, 0.0)) -> "VoxelHexFEM":
        """Build a hex mesh from a boolean occupancy grid (True = solid voxel).

        Each SOLID voxel becomes one unit hex element (a cube of side `pitch`).
        Only solid voxels are meshed -> sparse parts cost only their solid
        element count (the measured budget gate: keep solid elements <= ~6000
        for a ~90 s pure-python skfem assembly). Returns a VoxelHexFEM whose
        mesh has exactly occ.sum() elements. Shared faces are deduplicated via
        a node-coordinate hash so adjacent voxels are properly connected.

        occ shape = (nx, ny, nz); voxel (i,j,k) spans
            [origin + (i,j,k)*pitch, origin + (i+1,j+1,k+1)*pitch].
        """
        occ = np.asarray(occ, dtype=bool)
        ox, oy, oz = origin
        idx = np.argwhere(occ)  # (m,3) solid voxel indices
        if idx.shape[0] == 0:
            raise ValueError("occupancy grid has no solid voxels")
        # unit-cube corner offsets in skfem MeshHex's EXACT node order (learned
        # from MeshHex.init_tensor on a single element): the local corner order
        # is NOT binary-counting -- it is the sequence below. Getting this wrong
        # scrambles the element geometry (degenerate Jacobian, empty facet sets).
        #   [0,0,0],[0,1,0],[1,0,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1]
        corner_off = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1],
                               [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]],
                              dtype=np.int64)
        m = idx.shape[0]
        all_corners = (idx[:, None, :] + corner_off[None, :, :]).reshape(-1, 3)
        # unique integer corners -> node ids
        uniq, inv = np.unique(all_corners, axis=0, return_inverse=True)
        # node coordinates (physical)
        p = np.empty((3, uniq.shape[0]))
        p[0] = ox + uniq[:, 0] * pitch
        p[1] = oy + uniq[:, 1] * pitch
        p[2] = oz + uniq[:, 2] * pitch
        # element connectivity (8, m) in skfem's expected corner ordering.
        t = inv.reshape(m, 8).T.astype(np.int64)
        mesh = MeshHex(p, t)
        return cls(mesh, E, nu)

    # ---- BC selectors ---------------------------------------------------- #
    def clamp_face(self, predicate: Callable[[np.ndarray], np.ndarray]):
        """Return the dof set on the face where `predicate(p)` is True (clamp)."""
        return self.basis.get_dofs(predicate)

    def traction(self, predicate: Callable[[np.ndarray], np.ndarray],
                 vec: tuple[float, float, float]) -> np.ndarray:
        """Assemble a uniform surface traction `vec` (force/area) on a face.

        Returns the load vector contribution f. Integrating a constant traction
        over the face area gives the correct total force = traction * area.
        """
        fb = FacetBasis(self.mesh, self.basis.elem,
                        facets=self.mesh.facets_satisfying(predicate))
        tx, ty, tz = vec

        @LinearForm
        def _trac(v, w):
            return tx * v[0] + ty * v[1] + tz * v[2]

        return asm(_trac, fb)

    # ---- eigenstrain body load (used by the warp channel; validated in the
    #      smoke module's bimetal benchmark, exposed here for reuse) -------- #
    def eigenstrain_load(self, eps_field: Callable[[np.ndarray], tuple]) -> np.ndarray:
        """Body load from a prescribed eigenstrain field eps*(x) (3 diag comps).

        eps_field(x) -> (exx, eyy, ezz) at quadrature points x (shape (3,...)).
        Builds f = integral( sigma* : eps(v) ), sigma* = C:eps* (diagonal here).
        Provided so the inherent-strain warp channel reuses the SAME solver.
        """
        lam, mu = self._lam, self._mu

        @LinearForm
        def _eig(v, w):
            exx, eyy, ezz = eps_field(w.x)
            tr = exx + eyy + ezz
            s_xx = 2.0 * mu * exx + lam * tr
            s_yy = 2.0 * mu * eyy + lam * tr
            s_zz = 2.0 * mu * ezz + lam * tr
            ev = sym_grad(v)
            return s_xx * ev[0, 0] + s_yy * ev[1, 1] + s_zz * ev[2, 2]

        return asm(_eig, self.basis)

    # ---- solve ----------------------------------------------------------- #
    def solve(self, f: np.ndarray, clamped) -> FEMResult:
        """Solve K u = f with `clamped` dofs fixed to zero. Returns FEMResult."""
        t0 = time.time()
        # condense returns (Kc, fc, x, I); solve the reduced system then expand.
        Kc, fc, x, I = condense(self.K, f, D=clamped)
        uc, info = _solve_sparse(Kc, fc)
        u = x.copy()
        u[I] = uc
        solve_s = time.time() - t0
        return FEMResult(
            u=u, basis=self.basis, mesh=self.mesh,
            n_elements=self.mesh.t.shape[1], n_dofs=self.K.shape[0],
            solver=info["solver"], assemble_s=self.assemble_s, solve_s=solve_s,
        )


# --------------------------------------------------------------------------- #
#  BENCHMARK 1 -- uniaxial bar  (validates assembly + BC + traction)
# --------------------------------------------------------------------------- #
def bench_uniaxial(E=3.5e9, nu=0.0, L=0.10, W=0.01, H=0.01, F=100.0,
                   grids=((4, 1, 1), (8, 2, 2), (16, 4, 4))):
    """Bar along x, clamped at x=0, axial traction (total force F) at x=L.

    Closed form (nu=0 -> no Poisson lateral coupling): d = F L /(A E), A = W H.
    Returns list of (nelem, d_fem, d_exact, rel_err, solver).
    """
    A = W * H
    d_exact = F * L / (A * E)
    sigma = F / A
    rows = []
    for (nx, ny, nz) in grids:
        fem = VoxelHexFEM.from_box(L, W, H, nx, ny, nz, E, nu)
        f = fem.traction(lambda p: np.isclose(p[0], L), (sigma, 0.0, 0.0))
        clamped = fem.clamp_face(lambda p: np.isclose(p[0], 0.0))
        res = fem.solve(f, clamped)
        tip = fem.basis.get_dofs(lambda p: np.isclose(p[0], L)).nodal["u^1"]
        d_fem = float(np.mean(res.u[tip]))
        rel = abs(d_fem - d_exact) / abs(d_exact)
        rows.append((nx * ny * nz, d_fem, d_exact, rel, res.solver))
    return rows


# --------------------------------------------------------------------------- #
#  BENCHMARK 2 -- cantilever tip deflection (validates bending)
# --------------------------------------------------------------------------- #
def bench_cantilever(E=3.5e9, nu=0.3, L=0.10, b=0.01, h=0.01, P=10.0,
                     grids=((16, 2, 2), (32, 3, 3), (48, 4, 4), (64, 4, 4))):
    """Cantilever clamped at x=0, downward (-z) shear traction (total P) at x=L.

    Euler-Bernoulli: d = P L^3 /(3 E I), I = b h^3 /12. Trilinear hex shear-
    locks coarse -> d_fem < d_EB; ratio rises toward 1 from below under refine-
    ment. Returns (nelem, d_fem, d_EB, ratio, solver).
    """
    I = b * h ** 3 / 12.0
    d_eb = P * L ** 3 / (3.0 * E * I)
    A = b * h
    shear = P / A
    rows = []
    for (nx, ny, nz) in grids:
        fem = VoxelHexFEM.from_box(L, b, h, nx, ny, nz, E, nu)
        f = fem.traction(lambda p: np.isclose(p[0], L), (0.0, 0.0, -shear))
        clamped = fem.clamp_face(lambda p: np.isclose(p[0], 0.0))
        res = fem.solve(f, clamped)
        tip = fem.basis.get_dofs(lambda p: np.isclose(p[0], L)).nodal["u^3"]
        d_fem = abs(float(np.mean(res.u[tip])))
        rows.append((nx * ny * nz, d_fem, d_eb, d_fem / d_eb, res.solver))
    return rows


# --------------------------------------------------------------------------- #
#  render
# --------------------------------------------------------------------------- #
def render_cantilever(out_png: str, E=3.5e9, nu=0.3, L=0.10, b=0.01, h=0.01,
                      P=10.0, nx=48, ny=4, nz=4, warp_scale=None,
                      conv_grids=((16, 2, 2), (32, 3, 3), (48, 4, 4),
                                  (64, 4, 4))) -> dict:
    """Solve the cantilever once and render the result to a PNG (3 panels):

      (1) deflection PROFILE: undeflected axis vs the deflected beam axis,
      (2) side ELEVATION (x-z) of the full deflected mesh, nodes colored by |u|,
      (3) CONVERGENCE: d_fem/d_EB vs element count -> the viability evidence
          (monotone toward 1 = the solver converging to the closed form).

    All panels annotated with MEASURED numbers, not decoration. Uses robust 2D
    axes (matplotlib 3D box-aspect is unreliable for a 100:10:10 sliver).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import collections

    I = b * h ** 3 / 12.0
    d_eb = P * L ** 3 / (3.0 * E * I)
    A = b * h
    shear = P / A
    fem = VoxelHexFEM.from_box(L, b, h, nx, ny, nz, E, nu)
    f = fem.traction(lambda p: np.isclose(p[0], L), (0.0, 0.0, -shear))
    clamped = fem.clamp_face(lambda p: np.isclose(p[0], 0.0))
    res = fem.solve(f, clamped)
    disp = res.u.reshape(-1, 3)
    p0 = fem.mesh.p.T  # (nnodes,3) reference coords
    tipsel = fem.basis.get_dofs(lambda p: np.isclose(p[0], L)).nodal["u^3"]
    d_fem = abs(float(np.mean(res.u[tipsel])))
    ratio = d_fem / d_eb

    if warp_scale is None:  # scale so deflection is visible at print scale
        warp_scale = max(1.0, 0.25 * L / max(d_fem, 1e-12))
    pdef = p0 + warp_scale * disp
    mag = np.linalg.norm(disp, axis=1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))

    # --- (1) deflection profile (y-midplane axis) -------------------------
    ymid = b / 2.0
    band = np.isclose(p0[:, 1], ymid, atol=b / (2 * ny) + 1e-9)
    prof = collections.defaultdict(list)
    for xv, zr, zd in zip(p0[band, 0], p0[band, 2], pdef[band, 2]):
        prof[round(xv, 9)].append((zr, zd))
    xa = sorted(prof.keys())
    zr_a = np.array([np.mean([t[0] for t in prof[x]]) for x in xa])
    zd_a = np.array([np.mean([t[1] for t in prof[x]]) for x in xa])
    ax1.plot(np.array(xa) * 1e3, zr_a * 1e3, "k--", lw=1.2, label="undeflected")
    ax1.plot(np.array(xa) * 1e3, zd_a * 1e3, "C0-", lw=2.4,
             label=f"deflected (x{warp_scale:.0f})")
    ax1.set_xlabel("x along beam [mm]")
    ax1.set_ylabel("z [mm]")
    ax1.set_title("Deflection profile (beam axis)")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(alpha=0.3)

    # --- (2) side elevation of full deflected mesh, colored by |u| --------
    sc = ax2.scatter(pdef[:, 0] * 1e3, pdef[:, 2] * 1e3, c=mag * 1e3,
                     cmap="viridis", s=6, edgecolors="none")
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("z [mm]")
    ax2.set_title(f"Deflected mesh, side view (warp x{warp_scale:.0f})")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(alpha=0.3)
    cb = fig.colorbar(sc, ax=ax2, shrink=0.85, pad=0.02)
    cb.set_label("|u| [mm]")

    # --- (3) convergence: ratio -> 1 under refinement ---------------------
    conv = bench_cantilever(E=E, nu=nu, L=L, b=b, h=h, P=P, grids=conv_grids)
    ns = [r[0] for r in conv]
    rs = [r[3] for r in conv]
    ax3.axhline(1.0, color="k", ls="--", lw=1, label="Euler-Bernoulli (=1)")
    ax3.plot(ns, rs, "o-", color="C3", lw=2, label="d_fem / d_EB")
    for n_, r_ in zip(ns, rs):
        ax3.annotate(f"{r_:.3f}", (n_, r_), textcoords="offset points",
                     xytext=(0, -12), fontsize=7, ha="center")
    ax3.set_xlabel("number of hex elements")
    ax3.set_ylabel("d_fem / d_EB")
    ax3.set_title("Convergence (monotone -> 1 = viable)")
    ax3.set_ylim(min(rs) - 0.04, 1.04)
    ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(alpha=0.3)

    fig.suptitle(
        f"PROBE-1  VoxelHex FEM cantilever  |  {nx}x{ny}x{nz} hex "
        f"({res.n_elements} elem, {res.n_dofs} dofs, {res.solver})   "
        f"d_fem={d_fem:.4e} m  d_EB={d_eb:.4e} m  ratio={ratio:.4f} "
        f"(converging from below; coarse-hex shear-locking residual)",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return {"d_fem": d_fem, "d_EB": d_eb, "ratio": ratio,
            "n_elements": res.n_elements, "n_dofs": res.n_dofs,
            "assemble_s": res.assemble_s, "solve_s": res.solve_s,
            "solver": res.solver, "png": out_png}


# --------------------------------------------------------------------------- #
#  occupancy round-trip sanity (the real-part path)
# --------------------------------------------------------------------------- #
def check_occupancy_bar(E=3.5e9, nu=0.0, L=0.10, W=0.01, H=0.01, F=100.0,
                        nx=16, ny=4, nz=4):
    """Build the SAME bar from an OCCUPANCY GRID (all-solid) and verify it gives
    the identical closed-form answer as the structured box. This proves the
    from_occupancy builder (the real-part path) assembles a correct mesh.
    """
    occ = np.ones((nx, ny, nz), dtype=bool)
    pitch_x = L / nx  # note: from_occupancy uses one pitch; use cubic voxels
    # use cubic voxels so the occupancy bar matches a box of equal pitch:
    pitch = L / nx
    # adjust W,H to the cubic grid so the closed form stays exact
    Wg, Hg = pitch * ny, pitch * nz
    A = Wg * Hg
    d_exact = F * L / (A * E)
    sigma = F / A
    fem = VoxelHexFEM.from_occupancy(occ, pitch, E, nu)
    Lg = pitch * nx
    f = fem.traction(lambda p: np.isclose(p[0], Lg), (sigma, 0.0, 0.0))
    clamped = fem.clamp_face(lambda p: np.isclose(p[0], 0.0))
    res = fem.solve(f, clamped)
    tip = fem.basis.get_dofs(lambda p: np.isclose(p[0], Lg)).nodal["u^1"]
    d_fem = float(np.mean(res.u[tip]))
    rel = abs(d_fem - d_exact) / abs(d_exact)
    return {"n_elements": res.n_elements, "d_fem": d_fem, "d_exact": d_exact,
            "rel_err": rel}


# --------------------------------------------------------------------------- #
#  pretty printer + selftest (THE GATE)
# --------------------------------------------------------------------------- #
def _print_table(name, rows, ratio_mode=False):
    print(f"\n=== {name} ===")
    if ratio_mode:
        print(f"{'nelem':>8} {'d_fem':>14} {'d_theory':>14} {'ratio':>9} {'solver':>16}")
        for n, a, b, r, s in rows:
            print(f"{n:>8} {a:>14.6e} {b:>14.6e} {r:>9.4f} {s:>16}")
    else:
        print(f"{'nelem':>8} {'d_fem':>14} {'d_exact':>14} {'rel_err':>9} {'solver':>16}")
        for n, a, b, r, s in rows:
            print(f"{n:>8} {a:>14.6e} {b:>14.6e} {r:>9.3%} {s:>16}")


def selftest(make_render=True) -> bool:
    """THE GATE. Returns True iff the solver reproduces the closed forms with
    the correct convergence behaviour. Prints MEASURED convergence tables."""
    if not _HAVE_SKFEM:
        print("scikit-fem NOT importable:", _IMPORT_ERR)
        return False
    ok = True
    print("=" * 74)
    print("PROBE-1  VoxelHex linear-elastic FEM CORE  --  THE VIABILITY GATE")
    print("=" * 74)
    print(f"stack: scikit-fem {skfem.__version__} (BSD-3), "
          f"pyamg {'present' if _HAVE_PYAMG else 'ABSENT'} (MIT), scipy/numpy (BSD)")

    # ---- B1: uniaxial bar -------------------------------------------------
    r1 = bench_uniaxial()
    _print_table("Benchmark 1: uniaxial bar  d = F*L/(A*E)  [assembly+BC]", r1)
    last1 = r1[-1][3]
    print(f"  -> finest rel_err = {last1:.3e}  (gate: < 1%); "
          "trilinear hex captures the linear axial field EXACTLY.")
    ok &= last1 < 0.01

    # ---- occupancy round-trip --------------------------------------------
    occ = check_occupancy_bar()
    print(f"\n  [from_occupancy] all-solid bar via occupancy grid "
          f"({occ['n_elements']} elem): rel_err = {occ['rel_err']:.3e} "
          f"(gate: < 1%, proves the real-part mesh builder).")
    ok &= occ["rel_err"] < 0.01

    # ---- B2: cantilever ---------------------------------------------------
    r2 = bench_cantilever()
    _print_table("Benchmark 2: cantilever  d_EB = P*L^3/(3EI)  [bending]", r2,
                 ratio_mode=True)
    ratios = [row[3] for row in r2]
    monotone_up = all(ratios[i + 1] > ratios[i] for i in range(len(ratios) - 1))
    print(f"  -> ratio d_fem/d_EB: {[round(x,4) for x in ratios]}")
    print(f"     monotone_increasing = {monotone_up};  finest ratio = {ratios[-1]:.4f}")
    print("     (ratio<1 rising toward 1 = correct trilinear-hex shear-locking:")
    print("      coarse hex too stiff, gap CLOSES under refinement -> convergence")
    print("      to the closed form. Residual gap is locking, not error in the law.)")
    # GATE: must be monotone toward 1 AND reach within a few % of EB.
    ok &= monotone_up and (ratios[-1] >= 0.97) and (ratios[-1] <= 1.03)

    # ---- render -----------------------------------------------------------
    if make_render:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fem_cantilever_deflection.png")
        meta = render_cantilever(out)
        print(f"\n  [render] cantilever deflection -> {meta['png']}")
        print(f"           {meta['n_elements']} elem, {meta['n_dofs']} dofs, "
              f"solver={meta['solver']}, assemble={meta['assemble_s']:.1f}s, "
              f"solve={meta['solve_s']:.2f}s")
        print(f"           d_fem={meta['d_fem']:.4e} m  d_EB={meta['d_EB']:.4e} m  "
              f"ratio={meta['ratio']:.4f}")

    print("\n" + "=" * 74)
    verdict = "GATE PASSED -- solver reproduces closed forms under refinement" if ok \
        else "GATE FAILED -- solver does NOT reproduce closed forms"
    print(verdict)
    print("=" * 74)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
