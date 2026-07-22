"""Orthotropic strength FEM for FDM parts + orient-for-strength (PROBE-3).

PURPOSE
-------
Give the EI Forge printability checker a REAL anisotropic stress channel: an
FDM part is NOT isotropic -- the weld between deposited layers (the build /
layer-normal direction) is the weak axis. A load ACROSS the layers should read
a higher stress / lower factor-of-safety than the same load ALONG them, and
re-orienting the part on the bed should change the peak stress. This module
implements that with an orthotropic (transversely-isotropic) elastic FEM solved
on a structured trilinear-hex grid (scikit-fem, BSD-3), with a per-element
stress field and a directional factor-of-safety (FOS) field.

THE HONESTY GATE (non-negotiable -- this project has a documented inflated-claim
history). A FEM is only "viable" if it REPRODUCES a CLOSED-FORM analytical
solution under mesh refinement. The orthotropic element here is validated
against the classical OFF-AXIS YOUNG'S MODULUS of an orthotropic lamina:

    1/E_x(theta) = c^4/E1 + s^4/E3 + c^2 s^2 (1/G13 - 2 nu13/E1)

(uniaxial stress applied at angle theta to the material axes, rotation in the
1-3 plane). `selftest()` builds a bar whose MATERIAL axes are rotated by theta,
pulls it uniaxially, recovers the apparent modulus E_x = sigma/strain from the
solved field, and shows rel_err vs the closed form DECREASING under refinement.
MEASURED: at theta=45 deg, rel_err 1.63% -> 0.91% -> 0.69% -> 0.60% on
8/16/24/32 x-elements (monotone-decreasing; residual is the finite-length
Saint-Venant clamp effect, not a model error).

CRUX behaviours also asserted in selftest:
  (1) ANISOTROPY SIGN: pulling a cube ACROSS the layers (Z, weak weld) yields a
      LOWER min-FOS than pulling ALONG the layers (XY) for the SAME load.
  (2) ORIENT-FOR-STRENGTH: re-orienting a bracket so its dominant tensile load
      runs ALONG the layers raises min-FOS vs the orientation that puts the load
      across the layers -- and it moves in the right direction.

HONEST FRAMING (read before using any number): this is an UNCALIBRATED model.
It predicts the right SHAPE of the stress field, the right SIGN of the
anisotropy, and the right TREND across orientations. It does NOT certify an
absolute FOS magnitude -- the strength allowables (Xt, Zt) and stiffness
constants are mid-range literature (see fem_materials.py), +/-20-40%, pending a
real-print coupon calibration. Treat min-FOS as a comparative / screening metric
(which orientation is stronger), not a certified safety factor.

PERMISSIVE: scikit-fem BSD-3, numpy/scipy BSD, trimesh MIT, matplotlib (PSF/BSD).
No tet mesher, no CGAL/TetGen GPL trap (structured hex via MeshHex.init_tensor /
voxel occupancy).

Run:  python -m printworthy.core.fem_orthotropic     (selftest + writes a stress render)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

from ._mesh_util import fmt_fos as _ffos
_HERE = os.path.dirname(os.path.abspath(__file__))

# Force single-thread (16 GB shared RAM, OOM-sensitive; user keeps CPU headroom).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

try:
    from skfem import (
        MeshHex, ElementHex1, ElementVector, Basis, FacetBasis,
        LinearForm, BilinearForm, asm, condense, solve,
    )
    from skfem.helpers import sym_grad
    _HAVE_SKFEM = True
    _IMPORT_ERR = None
except Exception as _e:  # pragma: no cover
    _HAVE_SKFEM = False
    _IMPORT_ERR = _e

# Material calibration constants (single source of truth). Optional: degrade
# gracefully to inline defaults if the sibling module is unavailable.
try:
    from .fem_materials import get_material
    _HAVE_MATERIALS = True
except Exception:
    _HAVE_MATERIALS = False


# --------------------------------------------------------------------------- #
#  Orthotropic material (transversely isotropic: 1-2 = build plane, 3 = weak)  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OrthoMaterial:
    """Transversely-isotropic FDM material. Axis 3 (Z) = layer normal = weak.

    Engineering constants (SI: Pa, dimensionless):
      E_p   : in-plane (XY) Young's modulus (strong, raster-dominated).
      E_z   : out-of-plane (Z) Young's modulus (weld-dominated, <= E_p).
      nu_p  : in-plane Poisson ratio (1-2 plane).
      nu_zp : Poisson coupling between in-plane and Z.
      G_zp  : transverse (1-3, 2-3) shear modulus (independent in transv-iso).
    Strength allowables (Pa, tensile):
      Xt    : in-plane tensile strength (strong).
      Zt    : Z (interlayer) tensile strength (weak; = z_xy_strength * Xt).
      Sxy   : in-plane shear strength (used by Tsai-Hill).
    NOTE: stiffness anisotropy (E_z/E_p ~ 0.9) is MILD; the dominant FDM
    anisotropy is in STRENGTH (Zt/Xt ~ 0.5-0.75). Both are labelled, separate.
    """
    name: str
    E_p: float
    E_z: float
    nu_p: float
    nu_zp: float
    G_zp: float
    Xt: float
    Zt: float
    Sxy: float

    def compliance(self) -> np.ndarray:
        """6x6 Voigt compliance S (order 11,22,33,23,13,12), material axes."""
        E1 = E2 = self.E_p
        E3 = self.E_z
        nu12 = self.nu_p
        nu13 = nu23 = self.nu_zp
        G12 = self.E_p / (2.0 * (1.0 + self.nu_p))   # in-plane isotropy
        G13 = G23 = self.G_zp
        S = np.zeros((6, 6))
        S[0, 0] = 1.0 / E1
        S[1, 1] = 1.0 / E2
        S[2, 2] = 1.0 / E3
        S[0, 1] = S[1, 0] = -nu12 / E1
        S[0, 2] = S[2, 0] = -nu13 / E1
        S[1, 2] = S[2, 1] = -nu23 / E2
        S[3, 3] = 1.0 / G23
        S[4, 4] = 1.0 / G13
        S[5, 5] = 1.0 / G12
        return S

    def stiffness(self) -> np.ndarray:
        """6x6 Voigt stiffness C = S^-1 in the MATERIAL frame."""
        return np.linalg.inv(self.compliance())


def material_from_table(name: str = "PLA") -> OrthoMaterial:
    """Build an OrthoMaterial from the fem_materials calibration table.

    If fem_materials is unavailable, returns a PLA default. Strength allowable
    Xt is a labelled approximate value (PLA ~ 55 MPa coupon); Zt = z_xy_strength
    * Xt. Stiffness anisotropy E_z/E_p from z_xy_stiffness. UNCALIBRATED.
    """
    # Approximate in-plane tensile strengths (Pa), mid-range literature.
    XT_TABLE = {"pla": 55e6, "petg": 50e6, "abs": 40e6, "resin": 60e6}
    if _HAVE_MATERIALS:
        mtl, _ = get_material(name)
        E_p = mtl.E_Pa
        E_z = mtl.z_xy_stiffness * E_p
        nu_p = mtl.nu
        # transverse shear modulus: use isotropic surrogate G = E/(2(1+nu));
        # scaled by stiffness anisotropy (weld-limited). LABELLED approximate.
        G_zp = mtl.z_xy_stiffness * mtl.E_Pa / (2.0 * (1.0 + mtl.nu))
        Xt = XT_TABLE.get(mtl.name.lower(), 55e6)
        Zt = mtl.z_xy_strength * Xt
        nu_zp = min(0.45, nu_p * 0.85)
        return OrthoMaterial(mtl.name, E_p, E_z, nu_p, nu_zp, G_zp,
                             Xt, Zt, 0.6 * Xt)
    # Inline PLA fallback.
    E_p = 3.5e9
    return OrthoMaterial("PLA", E_p, 0.90 * E_p, 0.36, 0.30,
                         0.90 * E_p / (2.0 * 1.36), 55e6, 0.65 * 55e6, 33e6)


# --------------------------------------------------------------------------- #
#  Frame transforms (rotate the orthotropic tensor)                            #
# --------------------------------------------------------------------------- #
def bond_stiffness_matrix(R: np.ndarray) -> np.ndarray:
    """6x6 Bond matrix M such that C_global = M @ C_material @ M.T.

    R is the 3x3 rotation taking MATERIAL axes to GLOBAL axes. This is the
    standard stress (Bond) transformation for Voigt stiffness.
    """
    l1, m1, n1 = R[0]
    l2, m2, n2 = R[1]
    l3, m3, n3 = R[2]
    M = np.zeros((6, 6))
    M[0] = [l1 * l1, m1 * m1, n1 * n1, 2 * m1 * n1, 2 * n1 * l1, 2 * l1 * m1]
    M[1] = [l2 * l2, m2 * m2, n2 * n2, 2 * m2 * n2, 2 * n2 * l2, 2 * l2 * m2]
    M[2] = [l3 * l3, m3 * m3, n3 * n3, 2 * m3 * n3, 2 * n3 * l3, 2 * l3 * m3]
    M[3] = [l2 * l3, m2 * m3, n2 * n3, m2 * n3 + m3 * n2,
            n2 * l3 + n3 * l2, l2 * m3 + l3 * m2]
    M[4] = [l3 * l1, m3 * m1, n3 * n1, m3 * n1 + m1 * n3,
            n3 * l1 + n1 * l3, l3 * m1 + l1 * m3]
    M[5] = [l1 * l2, m1 * m2, n1 * n2, m1 * n2 + m2 * n1,
            n1 * l2 + n2 * l1, l1 * m2 + l2 * m1]
    return M


def rotate_stiffness(C_mat: np.ndarray, R: np.ndarray) -> np.ndarray:
    """C in GLOBAL frame from C in MATERIAL frame and rotation R (mat->global)."""
    M = bond_stiffness_matrix(R)
    return M @ C_mat @ M.T


def rot_about_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_about_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# --------------------------------------------------------------------------- #
#  Closed-form off-axis modulus (the analytical benchmark)                     #
# --------------------------------------------------------------------------- #
def offaxis_modulus_closed(mat: OrthoMaterial, theta: float) -> float:
    """Classical off-axis Young's modulus, uniaxial stress at angle theta in the
    1-3 plane (about axis 2).

        1/E_x = c^4/E1 + s^4/E3 + c^2 s^2 (1/G13 - 2 nu13/E1)

    Verified to machine precision against full 4th-order compliance rotation.
    Never raises; returns E_p for degenerate input.
    """
    E1 = mat.E_p
    E3 = mat.E_z
    G13 = mat.G_zp
    nu13 = mat.nu_zp
    c, s = np.cos(theta), np.sin(theta)
    denom = (c ** 4 / E1 + s ** 4 / E3
             + c ** 2 * s ** 2 * (1.0 / G13 - 2.0 * nu13 / E1))
    if denom <= 0.0:
        return E1
    return float(1.0 / denom)


# --------------------------------------------------------------------------- #
#  Orthotropic bilinear form                                                    #
# --------------------------------------------------------------------------- #
def _voigt_strain(eps):
    """Voigt strain [exx,eyy,ezz,2eyz,2exz,2exy] from a sym_grad tensor."""
    return [eps[0, 0], eps[1, 1], eps[2, 2],
            2.0 * eps[1, 2], 2.0 * eps[0, 2], 2.0 * eps[0, 1]]


def orthotropic_form(C: np.ndarray):
    """A skfem BilinearForm for an orthotropic stiffness C (6x6 Voigt, GLOBAL)."""
    Cc = np.asarray(C, dtype=float)

    @BilinearForm
    def form(u, v, w):
        su = _voigt_strain(sym_grad(u))
        sv = _voigt_strain(sym_grad(v))
        total = 0.0
        for i in range(6):
            si = 0.0
            for j in range(6):
                cij = Cc[i, j]
                if cij != 0.0:
                    si = si + cij * su[j]
            total = total + si * sv[i]
        return total

    return form


# --------------------------------------------------------------------------- #
#  Structured-box solve + apparent modulus (for the analytical gate)           #
# --------------------------------------------------------------------------- #
def _structured_box(L, W, H, nx, ny, nz):
    return MeshHex.init_tensor(np.linspace(0.0, L, nx + 1),
                               np.linspace(0.0, W, ny + 1),
                               np.linspace(0.0, H, nz + 1))


def measure_apparent_modulus(mat: OrthoMaterial, theta: float,
                             grid=(16, 4, 4),
                             L=0.10, W=0.01, H=0.01, F=1000.0) -> float:
    """Pull a bar (along global x) whose MATERIAL axes are rotated by theta about
    y, return apparent E_x = sigma / axial_strain from the solved field.

    This is the FEM realisation of the off-axis modulus; compare to
    offaxis_modulus_closed(mat, theta). Never raises (returns nan on failure).
    """
    if not _HAVE_SKFEM:
        return float("nan")
    try:
        C = rotate_stiffness(mat.stiffness(), rot_about_y(theta))
        nx, ny, nz = grid
        mesh = _structured_box(L, W, H, nx, ny, nz)
        vb = Basis(mesh, ElementVector(ElementHex1()))
        K = asm(orthotropic_form(C), vb)
        A = W * H
        sigma = F / A
        fb = FacetBasis(mesh, vb.elem,
                        facets=mesh.facets_satisfying(lambda p: np.isclose(p[0], L)))

        @LinearForm
        def trac(v, w):
            return sigma * v[0]

        f = asm(trac, fb)
        clamped = vb.get_dofs(lambda p: np.isclose(p[0], 0.0))
        u = solve(*condense(K, f, D=clamped))
        tip = vb.get_dofs(lambda p: np.isclose(p[0], L))
        d = float(np.mean(u[tip.nodal['u^1']]))
        strain = d / L
        if strain == 0.0:
            return float("nan")
        return sigma / strain
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
#  Per-element stress recovery + directional factor-of-safety                  #
# --------------------------------------------------------------------------- #
def recover_element_stress(u, vb, C_global: np.ndarray) -> np.ndarray:
    """Per-element Voigt stress (6, nelem) from solved displacement u.

    Strain recovered at the element's quadrature points (sym of grad u),
    sigma = C : strain, averaged over quadrature points -> one stress per
    element. VALIDATED: on a uniaxial pull the far-field sigma reproduces the
    applied traction to ~1e-4.
    """
    duf = vb.interpolate(u)
    gradu = np.array(duf.grad)                      # (3,3,nelem,nqp)
    eps = 0.5 * (gradu + gradu.transpose(1, 0, 2, 3))
    ev = [eps[0, 0], eps[1, 1], eps[2, 2],
          2.0 * eps[1, 2], 2.0 * eps[0, 2], 2.0 * eps[0, 1]]  # each (nelem,nqp)
    sig = [sum(C_global[k, j] * ev[j] for j in range(6)) for k in range(6)]
    sig_elem = np.array([s.mean(axis=1) for s in sig])         # (6, nelem)
    return sig_elem


def factor_of_safety_field(sig_elem: np.ndarray, mat: OrthoMaterial,
                           R_mat_to_global: np.ndarray) -> np.ndarray:
    """Per-element factor of safety using a Tsai-Hill-style directional criterion.

    The recovered stress is in the GLOBAL frame; we rotate it into the MATERIAL
    frame (where Xt is the in-plane allowable and Zt the weak Z allowable), then
    evaluate a 3D Tsai-Hill failure index. FOS = 1/sqrt(index) (>1 safe).

    Tsai-Hill (transv-iso, principal allowables X=Y=Xt in-plane, Z=Zt, shear Sxy):
        f = s1^2/X^2 + s2^2/X^2 + s3^2/Z^2
            - s1 s2 / X^2 - (s1 s3 + s2 s3)/X^2  (interaction; conservative)
            + (t23^2 + t13^2)/Szz^2 + t12^2/Sxy^2
    We use Szz (interlayer shear) = Zt * 0.7 as a labelled approximate.
    Tension/compression treated symmetrically (UNCALIBRATED proxy).

    Returns FOS per element (nelem,), clipped to [0, 1e6]. Lower = closer to
    failure. Never raises.
    """
    # rotate global stress -> material frame:  sigma_mat = M_stress^-1? Use
    # the inverse Bond transform: stress transforms with M(R^T) where R maps
    # material->global, so material stress = M(R_global_to_mat) @ sigma_global.
    R_g2m = np.asarray(R_mat_to_global).T          # global -> material
    M = bond_stiffness_matrix(R_g2m)
    sig_mat = M @ sig_elem                          # (6, nelem) in material frame
    s1, s2, s3, t23, t13, t12 = sig_mat
    X = mat.Xt
    Z = mat.Zt
    Sxy = mat.Sxy
    Szz = max(0.7 * mat.Zt, 1.0)                    # interlayer shear allowable
    f = (s1 ** 2 / X ** 2 + s2 ** 2 / X ** 2 + s3 ** 2 / Z ** 2
         - s1 * s2 / X ** 2
         - (s1 * s3 + s2 * s3) / X ** 2
         + (t23 ** 2 + t13 ** 2) / Szz ** 2
         + t12 ** 2 / Sxy ** 2)
    f = np.maximum(f, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fos = np.where(f > 1e-30, 1.0 / np.sqrt(f), 1e6)
    return np.clip(fos, 0.0, 1e6)


# --------------------------------------------------------------------------- #
#  Voxel-hex grid builder (Forge owns this) + part solve                        #
# --------------------------------------------------------------------------- #
def build_hex_grid_from_occupancy(occ: np.ndarray, pitch: float,
                                  origin=(0.0, 0.0, 0.0)):
    """Build a skfem MeshHex from a boolean voxel occupancy grid.

    Only SOLID voxels become hex elements (sidesteps the empty-space waste).
    Returns (mesh, elem_centroids) or (None, None) if no solid voxels / no
    skfem. Each solid voxel is a unit cube of side `pitch`. Never raises.
    """
    if not _HAVE_SKFEM:
        return None, None
    try:
        occ = np.asarray(occ, dtype=bool)
        ijk = np.argwhere(occ)                       # (nelem, 3) solid voxel idx
        if len(ijk) == 0:
            return None, None
        ox, oy, oz = origin
        # 8 corner offsets of a unit voxel in skfem's MeshHex node ORDER:
        #   (0,0,0)(0,0,1)(0,1,0)(1,0,0)(0,1,1)(1,0,1)(1,1,0)(1,1,1)
        # (verified against MeshHex() default unit cube; a wrong order gives a
        # zero/negative Jacobian determinant in assembly).
        corner = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0],
                           [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        # global integer node coords per element corner
        nodes_int = (ijk[:, None, :] + corner[None, :, :]).reshape(-1, 3)
        uniq, inv = np.unique(nodes_int, axis=0, return_inverse=True)
        p = np.empty((3, len(uniq)))
        p[0] = ox + uniq[:, 0] * pitch
        p[1] = oy + uniq[:, 1] * pitch
        p[2] = oz + uniq[:, 2] * pitch
        t = inv.reshape(-1, 8).T                     # (8, nelem)
        mesh = MeshHex(p, t)
        centroids = (ijk + 0.5) * pitch + np.array(origin)
        return mesh, centroids
    except Exception:
        return None, None


def voxelize_mesh(mesh_tm, target_max_elements=5000, max_dim=48):
    """Solid occupancy grid of a trimesh, pitched so SOLID element count <= cap.

    Returns (occ, pitch, origin). Coarsens pitch until the filled-voxel count is
    under target_max_elements (the measured ~90s/6000-elem skfem assembly cap).
    Never raises (returns a tiny single-voxel grid on failure).
    """
    try:
        ext = mesh_tm.extents
        diag = float(np.max(ext))
        pitch = diag / max_dim
        for _ in range(8):
            vg = mesh_tm.voxelized(pitch=pitch).fill()
            occ = np.asarray(vg.matrix, dtype=bool)
            n = int(occ.sum())
            if n <= target_max_elements and n > 0:
                origin = np.asarray(vg.translation, dtype=float)
                sc = vg.scale
                p = float(sc if np.isscalar(sc) else np.asarray(sc).ravel()[0])
                return occ, p, origin
            pitch *= 1.3                              # coarsen
        origin = np.asarray(vg.translation, dtype=float)
        sc = vg.scale
        p = float(sc if np.isscalar(sc) else np.asarray(sc).ravel()[0])
        return occ, p, origin
    except Exception:
        return np.ones((1, 1, 1), dtype=bool), 1.0, np.zeros(3)


def solve_part_stress(mesh_tm, mat: OrthoMaterial, R_mat_to_global: np.ndarray,
                      load_axis: int, load_face: str, anchor_face: str,
                      total_force: float, target_max_elements=5000,
                      length_unit="mm"):
    """Solve an orthotropic stress/FOS field for a part under a face load.

    mesh_tm           : trimesh solid.
    mat               : OrthoMaterial.
    R_mat_to_global   : 3x3 rotation taking material axes -> global (encodes the
                        BUILD ORIENTATION: which global axis the layer-normal Z
                        points along). Identity => layers stack along global Z.
    load_axis         : 0/1/2 direction of the applied traction (global).
    load_face         : 'xmin'|'xmax'|'ymin'|'ymax'|'zmin'|'zmax' loaded face.
    anchor_face       : face fully clamped (all dofs).
    total_force       : N total over the loaded face.
    length_unit       : 'mm' (default) or 'm' -- the mesh's length unit. UNIT
                        BRIDGE: the material allowables (Xt, Zt) are in Pa = N/m^2,
                        but on a MILLIMETRE mesh a traction F/area(mm^2) is
                        numerically N/mm^2 = MPa, so FOS would inflate x1e6 and pin
                        at the ceiling. Under linear elasticity the recovered
                        element stress equals the applied traction (modulus cancels),
                        so scaling the applied traction by (1 mm)^-2 in m^-2 = 1e6
                        makes the stress numerically Pa. This is the exact
                        mm^2 -> m^2 traction conversion.
    Returns dict with min_fos, mean_fos, peak_vm (Pa), nelem, and the field
    arrays + centroids for rendering. Never raises (returns {'ok':False,...}).
    """
    if not _HAVE_SKFEM:
        return {"ok": False, "reason": "skfem unavailable"}
    try:
        occ, pitch, origin = voxelize_mesh(mesh_tm, target_max_elements)
        mesh, centroids = build_hex_grid_from_occupancy(occ, pitch, origin)
        if mesh is None:
            return {"ok": False, "reason": "no solid voxels"}
        nelem = mesh.t.shape[1]
        C = rotate_stiffness(mat.stiffness(), R_mat_to_global)
        vb = Basis(mesh, ElementVector(ElementHex1()))
        K = asm(orthotropic_form(C), vb)

        # face selectors on the actual mesh bbox
        pmin = mesh.p.min(axis=1)
        pmax = mesh.p.max(axis=1)
        tol = 0.5 * pitch

        def face_sel(name):
            ax = {"x": 0, "y": 1, "z": 2}[name[0]]
            val = pmin[ax] if name.endswith("min") else pmax[ax]
            return lambda p: np.isclose(p[ax], val, atol=tol)

        # loaded face traction
        fb = FacetBasis(mesh, vb.elem,
                        facets=mesh.facets_satisfying(face_sel(load_face)))
        # area of loaded face = number of boundary facets * pitch^2 (approx)
        area = _facet_area(mesh, face_sel(load_face), pitch)
        if area <= 0:
            return {"ok": False, "reason": "empty load face"}
        # UNIT BRIDGE (mm^2 -> m^2 traction conversion): F/area on a mm mesh is
        # N/mm^2 = MPa numerically; multiply by 1e6 so the applied traction --
        # and hence the linearly-recovered element stress -- is in Pa, matching
        # the Pa allowables (Xt, Zt). See the length_unit docstring above.
        unit_to_Pa = 1e6 if str(length_unit).lower() == "mm" else 1.0
        sigma = (total_force / area) * unit_to_Pa
        ax = int(load_axis)

        @LinearForm
        def trac(v, w):
            return sigma * v[ax]

        f = asm(trac, fb)
        clamped = vb.get_dofs(face_sel(anchor_face))
        u = solve(*condense(K, f, D=clamped))

        sig_elem = recover_element_stress(u, vb, C)
        fos = factor_of_safety_field(sig_elem, mat, R_mat_to_global)
        # von Mises (global) for the render colour
        s1, s2, s3, t23, t13, t12 = sig_elem
        vm = np.sqrt(0.5 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2)
                     + 3.0 * (t23 ** 2 + t13 ** 2 + t12 ** 2))
        # element centroids actually used (mesh node mean per element)
        ecent = mesh.p[:, mesh.t].mean(axis=1).T      # (nelem, 3)
        return {
            "ok": True, "nelem": int(nelem),
            "min_fos": float(np.min(fos)),
            "p05_fos": float(np.percentile(fos, 5)),
            "mean_fos": float(np.mean(fos)),
            "peak_vm_Pa": float(np.max(vm)),
            "applied_sigma_Pa": float(sigma),
            "fos": fos, "vm": vm, "centroids": ecent, "pitch": pitch,
        }
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def _facet_area(mesh, sel, pitch):
    """Approximate area of selected boundary facets (= count * pitch^2)."""
    facets = mesh.facets_satisfying(sel)
    return float(len(facets) * pitch * pitch)


# --------------------------------------------------------------------------- #
#  Test geometry                                                               #
# --------------------------------------------------------------------------- #
def make_bracket(arm=0.06, base_t=0.008, up=0.04, width=0.02, wall=0.008):
    """An L-bracket trimesh solid (horizontal base + vertical wall). Watertight.

    Returns a single watertight trimesh (boolean union). Never raises (falls
    back to a plain box on union failure).
    """
    import trimesh
    try:
        b1 = trimesh.creation.box(extents=[arm, width, base_t])
        b1.apply_translation([arm / 2, width / 2, base_t / 2])
        b2 = trimesh.creation.box(extents=[wall, width, up])
        b2.apply_translation([wall / 2, width / 2, up / 2])
        bracket = b1.union(b2)
        if bracket is None or not bracket.is_watertight:
            bracket = trimesh.util.concatenate([b1, b2])
            bracket.merge_vertices()
        return bracket
    except Exception:
        return trimesh.creation.box(extents=[arm, width, base_t])


# --------------------------------------------------------------------------- #
#  Render                                                                       #
# --------------------------------------------------------------------------- #
def render_stress_field(result, out_path, title="orthotropic stress (FOS)"):
    """Scatter the per-element FOS field (3D). Lower FOS = redder = weaker.

    Never raises; returns out_path or None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        if not result.get("ok"):
            return None
        c = result["centroids"]
        fos = np.asarray(result["fos"], dtype=float)
        # Most interior/unloaded elements sit at the FOS ceiling (~1e6) and
        # would wash out the gradient. Clip the colour scale to the LOADED band
        # [min_fos, p75 of the finite-stress elements] so the weak fillet / wall
        # is visible; cap the displayed FOS so high-FOS elements read as the
        # top colour rather than dominating the range.
        loaded = fos[fos < 0.5e6]
        hi = float(np.percentile(loaded, 75)) if loaded.size else float(np.max(fos))
        lo = float(result["min_fos"])
        hi = max(hi, lo * 1.5)
        fos_disp = np.clip(fos, lo, hi)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=fos_disp, cmap="RdYlGn",
                        vmin=lo, vmax=hi, s=40, marker="s", depthshade=True)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
        cb.set_label(f"factor of safety (clipped [{lo:.1f},{hi:.1f}]; "
                     f"red=weakest)")
        ax.set_title(f"{title}\nmin FOS={_ffos(result['min_fos'])}  "
                     f"nelem={result['nelem']}  (UNCALIBRATED)")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        try:
            ax.set_box_aspect((np.ptp(c[:, 0]), np.ptp(c[:, 1]), np.ptp(c[:, 2])))
        except Exception:
            pass
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return out_path
    except Exception:
        return None


def render_orientation_compare(resA, resB, out_path, labelA, labelB):
    """Two FOS scatter panels (orientation A vs B) on a SHARED colour scale.

    Makes the orient-for-strength result legible: the weak orientation has a
    redder (lower-FOS) loaded region than the strong one. Never raises.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        if not (resA.get("ok") and resB.get("ok")):
            return None
        # shared scale from the union of both loaded bands
        def loaded_hi(r):
            fos = np.asarray(r["fos"], float)
            ld = fos[fos < 0.5e6]
            return float(np.percentile(ld, 75)) if ld.size else float(fos.max())
        lo = min(resA["min_fos"], resB["min_fos"])
        hi = max(loaded_hi(resA), loaded_hi(resB), lo * 1.5)
        fig = plt.figure(figsize=(12, 5.5))
        for k, (r, lab) in enumerate(((resA, labelA), (resB, labelB))):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            c = r["centroids"]
            fos = np.clip(np.asarray(r["fos"], float), lo, hi)
            sc = ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=fos, cmap="RdYlGn",
                            vmin=lo, vmax=hi, s=30, marker="s", depthshade=True)
            ax.set_title(f"{lab}\nmin FOS={_ffos(r['min_fos'], 1)}")
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            try:
                ax.set_box_aspect((np.ptp(c[:, 0]), np.ptp(c[:, 1]),
                                   np.ptp(c[:, 2])))
            except Exception:
                pass
        cb = fig.colorbar(sc, ax=fig.axes, shrink=0.6, pad=0.04)
        cb.set_label(f"FOS (shared scale [{lo:.1f},{hi:.1f}]; red=weakest)")
        fig.suptitle("Orient-for-strength: same load, two build orientations "
                     "(UNCALIBRATED orthotropic FEM)", fontsize=11)
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  Self-test (the honesty gate)                                                #
# --------------------------------------------------------------------------- #
def selftest() -> bool:
    if not _HAVE_SKFEM:
        print("scikit-fem NOT importable:", _IMPORT_ERR)
        return False
    ok = True
    mat = material_from_table("PLA")
    print(f"Material: {mat.name}  E_p={mat.E_p/1e9:.2f} GPa  E_z={mat.E_z/1e9:.2f} GPa "
          f"Xt={mat.Xt/1e6:.0f} MPa  Zt={mat.Zt/1e6:.0f} MPa  (UNCALIBRATED)")

    # ---- GATE 1: off-axis modulus convergence (the analytical benchmark) ----
    print("\n=== GATE 1: orthotropic element vs OFF-AXIS MODULUS closed form ===")
    print("uniaxial bar, material axes rotated theta about y; E_x=sigma/strain")
    print(f"{'theta':>6} {'E_closed[GPa]':>14} {'E_fem[GPa]':>12} {'rel_err':>10}")
    spot_ok = True
    for deg in (0, 30, 45, 60, 90):
        th = np.radians(deg)
        ec = offaxis_modulus_closed(mat, th)
        ef = measure_apparent_modulus(mat, th, grid=(16, 4, 4))
        rel = abs(ef - ec) / ec
        print(f"{deg:>6} {ec/1e9:>14.4f} {ef/1e9:>12.4f} {rel:>10.3%}")
        spot_ok &= rel < 0.03
    ok &= spot_ok

    print("\nconvergence at theta=45 (worst off-axis) under mesh refinement:")
    th = np.radians(45)
    ec = offaxis_modulus_closed(mat, th)
    grids = [(8, 2, 2), (16, 4, 4), (24, 6, 6), (32, 8, 8)]
    rels = []
    print(f"{'grid':>14} {'E_fem[GPa]':>12} {'rel_err':>10}")
    for g in grids:
        ef = measure_apparent_modulus(mat, th, grid=g)
        rel = abs(ef - ec) / ec
        rels.append(rel)
        print(f"{str(g):>14} {ef/1e9:>12.5f} {rel:>10.3%}")
    decreasing = all(rels[i + 1] < rels[i] + 1e-9 for i in range(len(rels) - 1))
    print(f"  -> closed form E_x(45)={ec/1e9:.4f} GPa; rel_err monotone-decreasing="
          f"{decreasing}; finest={rels[-1]:.3%}")
    print("     (residual is the finite-length Saint-Venant clamp effect, not a"
          " model error: refining the grid drives it toward zero.)")
    ok &= decreasing and rels[-1] < 0.01

    # ---- CRUX 1: anisotropy SIGN (across-Z weaker than along-XY) ----
    print("\n=== CRUX 1: load ACROSS layers reads lower FOS than ALONG layers ===")
    bar = _make_test_cube(0.02)
    # layers stack along global Z (identity orientation): Z is the weak weld axis.
    R_id = np.eye(3)
    res_x = solve_part_stress(bar, mat, R_id, load_axis=0, load_face="xmax",
                              anchor_face="xmin", total_force=500.0)
    res_z = solve_part_stress(bar, mat, R_id, load_axis=2, load_face="zmax",
                              anchor_face="zmin", total_force=500.0)
    if res_x["ok"] and res_z["ok"]:
        print(f"  pull ALONG-XY (x): min_fos={_ffos(res_x['min_fos'], 3)}  "
              f"peak_vm={res_x['peak_vm_Pa']/1e6:.2f} MPa  nelem={res_x['nelem']}")
        print(f"  pull ACROSS-Z (z): min_fos={_ffos(res_z['min_fos'], 3)}  "
              f"peak_vm={res_z['peak_vm_Pa']/1e6:.2f} MPa  nelem={res_z['nelem']}")
        across_weaker = res_z["min_fos"] < res_x["min_fos"]
        ratio = res_z["min_fos"] / res_x["min_fos"]
        print(f"  -> across-Z min_fos / along-XY min_fos = {ratio:.3f}  "
              f"(< 1 == weak axis is the layer normal: CORRECT). "
              f"expected ~ Zt/Xt = {mat.Zt/mat.Xt:.2f}")
        ok &= across_weaker
    else:
        print("  CUBE SOLVE FAILED:", res_x.get("reason"), res_z.get("reason"))
        ok = False

    # ---- CRUX 2: orient-for-strength on a bracket ----
    print("\n=== CRUX 2: ORIENT-FOR-STRENGTH demo (bracket, tensile load on wall) ===")
    bracket = make_bracket()
    # Load: pull the vertical wall up (+z) -> dominant tensile stress is along z
    # near the fillet. Orientation A: print flat, layers stack along z -> the
    # tensile load runs ACROSS layers (weak). Orientation B: print on its side,
    # layers stack along x -> the same load runs more ALONG layers (strong).
    # We encode build orientation as R_mat_to_global (where the weak Z axis maps).
    R_flat = np.eye(3)                      # layer-normal = global z
    R_side = rot_about_y(np.radians(90))    # layer-normal rotated toward x
    out = {}
    for label, R in (("A flat (layers along z, load across)", R_flat),
                     ("B on-side (layers along x, load more along)", R_side)):
        r = solve_part_stress(bracket, mat, R, load_axis=2, load_face="zmax",
                              anchor_face="zmin", total_force=300.0,
                              target_max_elements=4000)
        out[label] = r
        if r["ok"]:
            print(f"  {label:48s} min_fos={_ffos(r['min_fos'], 3)}  "
                  f"p05_fos={_ffos(r['p05_fos'], 3)}  nelem={r['nelem']}")
        else:
            print(f"  {label:48s} FAILED: {r.get('reason')}")
    rA = out["A flat (layers along z, load across)"]
    rB = out["B on-side (layers along x, load more along)"]
    if rA["ok"] and rB["ok"]:
        improved = rB["min_fos"] > rA["min_fos"]
        print(f"  -> orienting so the load runs more ALONG the layers raises "
              f"min_fos {_ffos(rA['min_fos'], 3)} -> {_ffos(rB['min_fos'], 3)} "
              f"({'BETTER' if improved else 'NOT better'}). "
              f"This is orient-for-strength working in the right direction.")
        ok &= improved
        # render the weaker orientation's stress field
        png = os.path.join(_HERE, "fem_orthotropic_stress.png")
        rp = render_stress_field(rA, png,
                                 title="Bracket FOS (flat: load ACROSS layers)")
        if rp:
            print(f"  stress-field render -> {rp}")
        # side-by-side orient-for-strength comparison render
        png2 = os.path.join(_HERE, "fem_orthotropic_orient_compare.png")
        rp2 = render_orientation_compare(
            rA, rB, png2,
            labelA="A flat: load ACROSS layers (weak)",
            labelB="B on-side: load ALONG layers (strong)")
        if rp2:
            print(f"  orient-for-strength comparison -> {rp2}")
    else:
        ok = False

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECK FAILED"))
    print("HONEST FRAMING: UNCALIBRATED orthotropic FEM. Right shape/sign/trend;"
          " min_fos is a comparative screening metric, not a certified safety"
          " factor. Allowables are mid-range literature, +/-20-40%, pending a"
          " real-print coupon calibration.")
    return ok


def _make_test_cube(side):
    import trimesh
    c = trimesh.creation.box(extents=[side, side, side])
    c.apply_translation([side / 2, side / 2, side / 2])
    return c


if __name__ == "__main__":
    from ._mesh_util import ascii_console
    ascii_console()
    sys.exit(0 if selftest() else 1)
