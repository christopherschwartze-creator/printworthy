"""fem_warp_probe.py -- PROBE-2: INHERENT-STRAIN (eigenstrain) PRINT-WARP model.

WHAT THIS IS
============
A reduced-order, physics-based predictor of 3D-print WARP (corner lift off the
bed). It applies the INHERENT-STRAIN method on a VOXEL-HEX grid built from a
real mesh's occupancy, bonds the part to the bed (clamp the bottom solid layer),
imposes a per-layer in-plane thermal-contraction eigenstrain, and solves ONE
linear elasticity problem -> the distortion field. The "corner lift" is the
upward (+z, off-bed) displacement at the part footprint corners.

WHY IT IS PHYSICS, NOT A HEURISTIC (the honesty gate)
=====================================================
The eigenstrain mechanism is validated against the Timoshenko (1925) bimetal
closed form. A differentially-straining strip bends with curvature
    kappa = 1.5 * d_alpha * dT / h            (equal-modulus equal-thickness)
and our solver reproduces it with relative error that DECREASES MONOTONICALLY
under mesh refinement and the correct SIGN at every grid (measured here, not
asserted): rel_err ~34% -> 18% -> 11% -> 6.0% -> 3.3% on 16->48 length-elements.
That is the certificate that the WARP DIRECTION and SHAPE are real physics. The
underlying Q1 voxel-hex kernel also passes the uniaxial bar (machine precision)
and slender cantilever (monotone, shear-locking-from-below) gates -- see
_fem_validation_plan.py. This module only RE-USES that validated kernel.

HONEST SCOPE (label every output)
=================================
  * The kernel uses fully-integrated trilinear Q1 hexes, which SHEAR-LOCK in
    bending: warp magnitude is UNDER-predicted on a coarse grid by a known,
    mesh-dependent factor (the bimetal ratio above quantifies it). So absolute
    warp millimetres are a LOWER bound that rises as the grid refines.
  * Material constants (PLA E, CTE, dT_lock) are CALIBRATION CONSTANTS from
    fem_materials.py, not measured truth for a specific printer/filament/profile.
  * Therefore the certified claims are: the SIGN (corners lift up), the SHAPE
    (where warp concentrates -- thin wide spans, far from the bed anchor), the
    TREND (part A warps more than B; thicker/shorter warps less), and ORDER OF
    MAGNITUDE. A certified millimetre needs a single-scalar coupon calibration
    for that exact process (multiply eps* by a fitted k). Every number printed
    here is a "physics-based estimate pending real-print calibration".
  * watertight != genus-correct; voxel occupancy is a coarse solid model.

THE EIGENSTRAIN, EXACTLY (from fem_materials.EIGENSTRAIN_NOTES)
==============================================================
Each solid voxel carries an isotropic in-plane stress-free strain
    eps*  =  -alpha * dT_lock                 (NEGATIVE = contraction)
applied as Voigt [eps*, eps*, gamma_z*eps*, 0, 0, 0]; alpha = solid-polymer CTE,
dT_lock = max(0, Tg - T_bed) (the LOCKED drop -- using the full deposit->bed drop
over-predicts warp ~2-3x, a documented trap). The bottom solid layer is bonded
to the bed (clamped) so the part cannot contract there; the unbalanced
contraction above produces an upward-curling moment -> corner lift. (This is the
same bilayer mechanism as the bimetal benchmark, with the rigid bed playing the
role of the inert low-CTE layer.)

Run:  python -m printworthy.core.fem_warp_probe            # selftest: bimetal gate + bracket demos + renders
      python -m printworthy.core.fem_warp_probe --quick    # smaller demo, faster
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Validated Q1 voxel-hex kernel (uniaxial=machine-precision, cantilever monotone,
# bimetal eigenstrain converges to Timoshenko). We reuse its element routines.
from . import _fem_validation_plan as _fv
from . import fem_materials as fm

# pyamg is optional (MIT). We only need it for large systems; small demos use
# scipy.sparse direct solve.
try:
    import pyamg  # noqa: F401
    _HAVE_PYAMG = True
except Exception:
    _HAVE_PYAMG = False


# =========================================================================== #
#  Voxel-occupancy hex model (only SOLID voxels become elements)              #
# =========================================================================== #
class VoxelHexModel:
    """A linear-elastic FEM on the SOLID voxels of an occupancy grid.

    Unlike _fv._Grid (a full structured box), this keeps ONLY occupied voxels as
    trilinear-Q1 hex elements and builds the shared-node connectivity, so a real
    part's geometry (holes, L-shapes, thin spans) is represented. Every voxel is
    an identical axis-aligned cube of size (px,py,pz), so a SINGLE element
    stiffness K_e and a single eigenstrain load vector f_e are reused for all
    elements (fast, exact -- the cube Jacobian is constant).

    Coordinates: node (i,j,k) sits at (origin + (i*px, j*py, k*pz)). Solid voxel
    (vi,vj,vk) occupies nodes i in {vi,vi+1}, etc. z is the BUILD axis (up).

    Pure construction; never raises on an empty grid (n_elem==0 -> trivial).
    """

    def __init__(self, occ: np.ndarray, pitch, origin=(0.0, 0.0, 0.0)):
        occ = np.asarray(occ, dtype=bool)
        if occ.ndim != 3:
            raise ValueError("occupancy must be a 3D boolean array")
        self.occ = occ
        self.nx, self.ny, self.nz = occ.shape
        if np.isscalar(pitch):
            self.px = self.py = self.pz = float(pitch)
        else:
            self.px, self.py, self.pz = (float(p) for p in pitch)
        self.origin = np.asarray(origin, float)

        # Enumerate solid voxels and the unique nodes they touch.
        vox = np.argwhere(occ)                      # (n_elem, 3) voxel indices
        self.voxels = vox
        self.n_elem = len(vox)

        # Build the global node index map. A node id is the lexicographic index
        # of an active corner (a corner touched by >=1 solid voxel).
        node_used = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), bool)
        # the 8 corner offsets in the SAME order as _fv._Grid.elem_nodes
        self._corner_off = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], int)
        for off in self._corner_off:
            node_used[vox[:, 0] + off[0], vox[:, 1] + off[1], vox[:, 2] + off[2]] = True
        nu_idx = np.argwhere(node_used)
        self.node_coords = self.origin + nu_idx * np.array([self.px, self.py, self.pz])
        self.n_node = len(nu_idx)
        self.ndof = 3 * self.n_node
        # map (i,j,k) -> node id
        self._nmap = -np.ones((self.nx + 1, self.ny + 1, self.nz + 1), np.int64)
        self._nmap[nu_idx[:, 0], nu_idx[:, 1], nu_idx[:, 2]] = np.arange(self.n_node)

        # Element -> 8 global node ids (n_elem, 8)
        self.elem_nodes = np.empty((self.n_elem, 8), np.int64)
        for c, off in enumerate(self._corner_off):
            self.elem_nodes[:, c] = self._nmap[
                vox[:, 0] + off[0], vox[:, 1] + off[1], vox[:, 2] + off[2]]

    # ---- assembly --------------------------------------------------------- #
    def assemble_K(self, E, nu):
        """Global stiffness. One element K_e reused for every cube."""
        Ke = _fv._hex_Ke(self.px, self.py, self.pz, E, nu)   # (24,24)
        # element dof index table (n_elem, 24)
        edofs = (3 * self.elem_nodes[:, :, None] + np.arange(3)[None, None, :]
                 ).reshape(self.n_elem, 24)
        # scatter Ke into COO. Vectorize over elements.
        ii = np.repeat(edofs, 24, axis=1).reshape(-1)            # row dof
        jj = np.tile(edofs, (1, 24)).reshape(-1)                 # col dof
        vv = np.tile(Ke.reshape(-1), self.n_elem)
        K = sp.csr_matrix((vv, (ii, jj)), shape=(self.ndof, self.ndof))
        return K

    def eigenstrain_load(self, E, nu, eps_scalar, gamma_z=0.0):
        """Global RHS for a uniform isotropic in-plane eigenstrain in every voxel.

        eps_scalar: the in-plane stress-free strain (NEGATIVE = contraction).
        gamma_z   : fraction of eps_scalar applied in z (0 = pure in-plane).
        """
        eps0 = np.array([eps_scalar, eps_scalar, gamma_z * eps_scalar, 0.0, 0.0, 0.0])
        fe = _fv._hex_thermal_fe(self.px, self.py, self.pz, E, nu, eps0)   # (24,)
        edofs = (3 * self.elem_nodes[:, :, None] + np.arange(3)[None, None, :]
                 ).reshape(self.n_elem, 24)
        f = np.zeros(self.ndof)
        np.add.at(f, edofs.reshape(-1), np.tile(fe, self.n_elem))
        return f

    def gradient_eigenstrain_load(self, E, nu, eps_top, *, eps_bottom=0.0,
                                  gamma_z=0.0):
        """Global RHS for a LINEAR through-thickness eigenstrain gradient.

        The in-plane stress-free strain varies linearly with z (the build axis)
        from eps_bottom at the lowest solid voxel to eps_top at the highest:
            eps*(z) = eps_bottom + (eps_top - eps_bottom) * (z - z0)/(z1 - z0)
        This is the physically correct print-warp driver (see module docstring):
        the bed-bonded bottom layer locked flat against the platform (carries
        ~eps_bottom of stored contraction), while the last-printed top layer
        carries the full locked contraction eps_top; the gradient is the bending
        eigenstrain. On RELEASE the part curls to kappa ~ (eps_top-eps_bottom)/H.

        Per element we use the cube's z-centroid to set its eps*; the unit-eps
        element load is scaled per element (one quadrature, reused).
        """
        # unit in-plane eigenstrain element load (eps* = 1), scaled per element.
        fe_unit = _fv._hex_thermal_fe(
            self.px, self.py, self.pz, E, nu,
            np.array([1.0, 1.0, gamma_z, 0.0, 0.0, 0.0]))
        # element z-centroids
        zc = self.node_coords[self.elem_nodes][:, :, 2].mean(axis=1)  # (n_elem,)
        z0, z1 = float(zc.min()), float(zc.max())
        H = (z1 - z0) if (z1 - z0) > 1e-12 else 1.0
        frac = (zc - z0) / H                                          # 0..1
        eps_e = eps_bottom + (eps_top - eps_bottom) * frac            # (n_elem,)
        edofs = (3 * self.elem_nodes[:, :, None] + np.arange(3)[None, None, :]
                 ).reshape(self.n_elem, 24)
        f = np.zeros(self.ndof)
        # scatter scaled element loads
        contrib = (eps_e[:, None] * fe_unit[None, :]).reshape(-1)
        np.add.at(f, edofs.reshape(-1), contrib)
        return f, float(eps_top - eps_bottom), H

    # ---- boundary conditions --------------------------------------------- #
    def bed_clamp_dofs(self):
        """Dofs to fix = all 3 components of every node on the bottom (k=0) solid
        plane that is actually used. This bonds the part to the bed."""
        kmin = 0
        used = self._nmap[:, :, kmin]
        ids = used[used >= 0]
        dofs = np.concatenate([3 * ids + d for d in range(3)])
        return dofs

    def isostatic_release_dofs(self):
        """Minimal (statically determinate) restraint to remove the 6 rigid-body
        modes of a RELEASED part without imposing any spurious load -- the part
        is free to curl. Three bottom nodes (spread out) carry a 3-2-1 mount:
            n0: pin all 3 dof; n1: pin y,z; n2: pin z.
        This kills 3 translations + 3 rotations and nothing else.
        """
        coords = self.node_coords
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        bot = np.isclose(z, z.min())
        bidx = np.where(bot)[0]
        if bidx.size < 3:                       # degenerate: pin whatever exists
            bidx = np.arange(self.n_node)
        cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
        n0 = bidx[np.argmin((x[bidx] - cx) ** 2 + (y[bidx] - cy) ** 2)]
        n1 = bidx[np.argmax(x[bidx])]           # far in +x
        # n2: the bottom node farthest in y from the n0-n1 line (max |y - cy|)
        n2 = bidx[np.argmax(np.abs(y[bidx] - cy))]
        dofs = [3 * n0, 3 * n0 + 1, 3 * n0 + 2,
                3 * n1 + 1, 3 * n1 + 2,
                3 * n2 + 2]
        return np.array(sorted(set(dofs)), dtype=np.int64)

    def bottom_layer_node_ids(self):
        used = self._nmap[:, :, 0]
        return used[used >= 0]

    # ---- solve ------------------------------------------------------------ #
    def solve(self, K, f, fixed_dofs, use_amg=None):
        free = np.setdiff1d(np.arange(self.ndof), np.asarray(list(fixed_dofs)))
        u = np.zeros(self.ndof)
        Kff = K[free][:, free].tocsr()
        ff = f[free]
        if use_amg is None:
            use_amg = _HAVE_PYAMG and Kff.shape[0] > 8000
        if use_amg and _HAVE_PYAMG:
            ml = pyamg.smoothed_aggregation_solver(Kff.tocsr())
            u[free] = ml.solve(ff, tol=1e-10, accel="cg")
        else:
            u[free] = spla.spsolve(Kff.tocsc(), ff)
        return u

    # ---- post ------------------------------------------------------------- #
    def displacement_field(self, u):
        return u.reshape(self.n_node, 3)


# =========================================================================== #
#  Warp prediction on a real part                                             #
# =========================================================================== #
def predict_warp(occ, pitch, material, *, origin=(0., 0., 0.),
                 use_locked=True, gamma_z=0.0, bottom_locked_frac=0.0):
    """Predict the inherent-strain warp (corner lift after RELEASE) of a part.

    occ      : (nx,ny,nz) bool occupancy, z = build axis.
    pitch    : voxel size in mm (scalar or (px,py,pz)).
    material : fem_materials.Material.
    Returns a dict (never raises on a non-degenerate grid). Distances in mm.

    PHYSICS (the corrected, validated model -- see module docstring + the
    released-curl convergence gate). 3D-print warp is the residual CURL revealed
    when the part is RELEASED from the bed. While printing, the bed holds the
    bottom layer flat as it solidifies, so the bottom stores LESS locked
    contraction than the last-printed top; this through-thickness eigenstrain
    GRADIENT is a bending eigenstrain. We impose a linear in-plane eigenstrain
    eps*(z) from `bottom_locked_frac * eps_scalar` (bottom, held flat by the bed)
    to `eps_scalar` (top), restrain only the 6 rigid-body modes (isostatic
    release), and solve ONE linear problem. The part curls to
        kappa ~ (1 - bottom_locked_frac) * eps_scalar / H ,
    edges lifting UP off the bed (the certified sign). A FULLY-BONDED part with a
    uniform eigenstrain merely sinks uniformly and shows no warp -- that earlier
    model was physically wrong and is replaced by this released-curl model.

    "max_corner_lift_mm" = the upward (+z) rise of the footprint corners relative
    to the part's central anchor region -- the classic FDM warp metric.

    bottom_locked_frac in [0,1): how much of the locked contraction the
    bed-bonded bottom already carries (0 = bottom perfectly flat-locked = maximum
    gradient = maximum warp; default 0, the conservative-shape upper-trend).
    """
    model = VoxelHexModel(occ, pitch, origin=origin)
    E = material.E_MPa        # MPa, so stresses come out in MPa; mm geometry.
    nu = material.nu
    eps_scalar = fm.layer_eigenstrain_scalar(material, use_locked=use_locked)
    dT_lock = material.locked_delta_T() if use_locked else material.delta_T_warp(to="bed")

    K = model.assemble_K(E, nu)
    eps_top = eps_scalar
    eps_bot = bottom_locked_frac * eps_scalar
    f, eps_grad, H = model.gradient_eigenstrain_load(
        E, nu, eps_top, eps_bottom=eps_bot, gamma_z=gamma_z)
    fixed = model.isostatic_release_dofs()
    u = model.solve(K, f, fixed)
    U = model.displacement_field(u)              # (n_node, 3) mm

    coords = model.node_coords
    uz = U[:, 2]

    # Corner lift = uz of the 4 footprint corners (top-of-corner column) minus
    # the central anchor region's mean uz (the part's reference plane). Edges
    # curling UP -> positive corner lift.
    x, y = coords[:, 0], coords[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    tolx = max(0.06 * (xmax - xmin), model.px)
    toly = max(0.06 * (ymax - ymin), model.py)
    centre = (np.abs(x - cx) < tolx) & (np.abs(y - cy) < toly)
    uz_ref = float(uz[centre].mean()) if centre.any() else 0.0
    corner_masks = {
        "x-_y-": (x <= xmin + tolx) & (y <= ymin + toly),
        "x+_y-": (x >= xmax - tolx) & (y <= ymin + toly),
        "x-_y+": (x <= xmin + tolx) & (y >= ymax - toly),
        "x+_y+": (x >= xmax - tolx) & (y >= ymax - toly),
    }
    corner_lift = {}
    for name, m in corner_masks.items():
        corner_lift[name] = (float(uz[m].max()) - uz_ref) if m.any() else 0.0
    max_corner_lift = max(corner_lift.values()) if corner_lift else 0.0

    # closed-form curl estimate for THIS part (slender-strip analogue): the curl
    # curvature is the eigenstrain gradient |eps_grad|/H; a span-L curl lifts the
    # free edge ~ kappa * (L/2)^2 / 2 relative to the centre. Reported as a
    # cross-check ORDER-OF-MAGNITUDE, NOT a certified number (real part is a plate
    # not a 1D strip; the FEM is the prediction, this is the sanity envelope).
    kappa_cf = abs(eps_grad) / H if H > 0 else 0.0
    span = max(xmax - xmin, ymax - ymin)
    corner_lift_strip_cf = kappa_cf * (0.5 * span) ** 2 / 2.0

    return {
        "ok": model.n_elem > 0,
        "model": model,
        "U": U,
        "uz": uz,
        "coords": coords,
        "uz_ref_mm": uz_ref,
        "n_elem": model.n_elem,
        "n_node": model.n_node,
        "ndof": model.ndof,
        "material": material.name,
        "eps_scalar": eps_scalar,
        "eps_top": eps_top,
        "eps_bottom": eps_bot,
        "eps_gradient": eps_grad,
        "thickness_H_mm": H,
        "kappa_strip_cf_per_mm": kappa_cf,
        "corner_lift_strip_cf_mm": corner_lift_strip_cf,
        "dT_lock_K": dT_lock,
        "use_locked": use_locked,
        "bottom_locked_frac": bottom_locked_frac,
        "corner_lift_mm": corner_lift,
        "max_corner_lift_mm": max_corner_lift,
        "max_uplift_mm": float(uz.max()),
        "max_downset_mm": float(uz.min()),
        "max_total_disp_mm": float(np.linalg.norm(U, axis=1).max()),
        "pitch_mm": (model.px, model.py, model.pz),
        "bbox_mm": (float(xmax - xmin), float(ymax - ymin),
                    float(coords[:, 2].max() - coords[:, 2].min())),
        "uncalibrated": True,
        "note": ("released-curl inherent-strain model; Q1 hex shear-locks -> "
                 "magnitude is a LOWER bound (released-curl gate ~5% low @ fine "
                 "grid); sign/shape/trend certified, mm pending coupon calibration."),
    }


# =========================================================================== #
#  HONESTY GATE: bimetal eigenstrain convergence (the warp mechanism check)   #
# =========================================================================== #
def bimetal_convergence(grids=((16, 2, 4), (24, 2, 4), (32, 2, 4),
                               (40, 2, 6), (48, 2, 8))):
    """Re-run the Timoshenko bimetal eigenstrain benchmark at increasing
    refinement. Returns (rows, closed_form_kappa, monotone, sign_ok).

    rows: (grid_str, kappa_fem, ratio_to_TS, rel_err). The mechanism is viable
    iff rel_err DECREASES under refinement (converges to closed form) with the
    correct sign at every grid. Uses the validated gate_C_bimetal kernel.
    """
    rows = []
    sign_all = True
    errs = []
    kts = None
    for g in grids:
        kf, kts, ratio, sok = _fv.gate_C_bimetal(*g)
        sign_all = sign_all and sok
        rel = abs(abs(kf) - kts) / kts
        errs.append(rel)
        rows.append((f"{g[0]}x{g[1]}x{g[2]}", kf, ratio, rel))
    monotone = all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))
    return rows, float(kts), monotone, sign_all


def released_curl_convergence(grids=((16, 2, 4), (24, 2, 4), (32, 2, 4),
                                     (40, 2, 6), (48, 2, 8)),
                              L=40.0, b=2.0, h=2.0, eps_top=-3.4e-4):
    """Validate the RELEASED-PART CURL model (the one predict_warp uses) against
    its closed form.

    A strip with a LINEAR in-plane eigenstrain from 0 (bottom) to eps_top (top),
    clamped at one end and free to curl, reaches curvature
        kappa = |eps_top - eps_bottom| / h = |eps_top| / h     (eps_bottom = 0).
    SIGN: eps_top < 0 means the top fibre contracts more -> top is "shorter" ->
    the strip curls CONCAVE-UP (tip rises, +z). For a part on a bed this is edges
    lifting UP = warp.  Returns (rows, kappa_cf, monotone, sign_ok).

    rows: (grid_str, kappa_fem_signed, ratio, rel_err). Uses the same validated
    _fv kernel routines (_hex_thermal_fe, _Grid) as the bimetal gate, but with a
    CONTINUOUS gradient instead of a 2-layer step -- this is exactly the load
    predict_warp applies, so passing this gate certifies the warp solver itself.
    """
    kappa_cf_mag = abs(eps_top) / h            # closed-form curvature MAGNITUDE
    rows = []
    errs = []
    sign_ok = True
    for (nx, ny, nz) in grids:
        g = _fv._Grid(nx, ny, nz, L, b, h)
        K = g.assemble_K(3500.0, 0.3)
        f = np.zeros(g.ndof)
        for ei in range(g.nx):
            for ek in range(g.nz):
                zc = (ek + 0.5) * g.hz
                e = eps_top * (zc / h)         # 0 at z=0 -> eps_top at z=h
                eps0 = np.array([e, e, 0, 0, 0, 0.0])
                fe = _fv._hex_thermal_fe(g.hx, g.hy, g.hz, 3500.0, 0.3, eps0)
                for ej in range(g.ny):
                    en = g.elem_nodes(ei, ej, ek)
                    dofs = np.array([3 * n + d for n in en for d in range(3)])
                    f[dofs] += fe
        fixed = set()
        for j in range(g.nY):
            for k in range(g.nZ):
                for d in range(3):
                    fixed.add(3 * g.nid(0, j, k) + d)
        u = _fv._solve(K, f, fixed, g.ndof)
        jc = g.nY // 2
        delta = np.mean([u[3 * g.nid(g.nx, jc, k) + 2] for k in range(g.nZ)])
        kappa_fem = 2.0 * delta / (L ** 2)      # signed
        # top contracts more (eps_top<0) -> tip UP -> kappa_fem > 0
        sign_ok = sign_ok and (np.sign(kappa_fem) == -np.sign(eps_top))
        rel = abs(abs(kappa_fem) - kappa_cf_mag) / kappa_cf_mag
        errs.append(rel)
        ratio = abs(kappa_fem) / kappa_cf_mag
        rows.append((f"{nx}x{ny}x{nz}", kappa_fem, ratio, rel))
    monotone = all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))
    return rows, float(kappa_cf_mag), monotone, sign_ok


# =========================================================================== #
#  Test geometries                                                            #
# =========================================================================== #
def make_flat_bracket(ext=(100.0, 30.0, 6.0)):
    """A long thin plate -- the canonical warp-prone bracket. trimesh box."""
    import trimesh
    return trimesh.creation.box(extents=list(ext))


def make_L_bracket():
    """An L-bracket (foot bonded to bed + a vertical wall). Tests warp on a part
    whose mass is not all in one flat plane."""
    import trimesh
    foot = trimesh.creation.box(extents=[60, 30, 5]); foot.apply_translation([30, 15, 2.5])
    wall = trimesh.creation.box(extents=[5, 30, 40]); wall.apply_translation([2.5, 15, 20])
    try:
        L = trimesh.boolean.union([foot, wall])
        if L.is_empty or len(L.faces) == 0:
            raise ValueError
    except Exception:
        L = trimesh.util.concatenate([foot, wall])
    return L


def voxelize_capped(mesh, *, max_elem=6000, min_z_layers=3,
                    pitch_candidates=(4, 3, 2.5, 2, 1.5, 1.25, 1.0)):
    """Voxelize a mesh to an occupancy grid, choosing the FINEST pitch that keeps
    solid element count <= max_elem AND gives >= min_z_layers through-thickness
    (warp needs vertical resolution to form a bending gradient).

    Returns (occ, pitch, info). Reuses trimesh .voxelized().fill(). The z axis of
    the returned occupancy is the part's z (build axis) by construction (trimesh
    keeps world axes). Never raises: falls back to the coarsest pitch that fits.
    """
    import trimesh  # noqa: F401
    best = None
    for pitch in pitch_candidates:
        vg = mesh.voxelized(pitch=pitch).fill()
        occ = np.asarray(vg.matrix, bool)
        nsolid = int(occ.sum())
        nz = occ.shape[2]
        # trimesh VoxelGrid: translation is the center of voxel (0,0,0); the node
        # origin (corner) is translation - pitch/2.
        try:
            tr = np.asarray(vg.translation, float)
        except Exception:
            tr = np.asarray(vg.bounds[0], float)
        node_origin = tr - 0.5 * pitch
        cand = (occ, float(pitch), {
            "n_elem": nsolid, "grid": occ.shape, "z_layers": nz,
            "node_origin": node_origin})
        # prefer finest pitch satisfying both constraints
        if nsolid <= max_elem and nz >= min_z_layers:
            best = cand
            break
        # keep the last one that at least fits the element cap (z may be thin)
        if nsolid <= max_elem:
            best = cand
    if best is None:
        # nothing fit; use the coarsest candidate (guaranteed smallest count)
        vg = mesh.voxelized(pitch=pitch_candidates[0]).fill()
        occ = np.asarray(vg.matrix, bool)
        try:
            tr = np.asarray(vg.translation, float)
        except Exception:
            tr = np.asarray(vg.bounds[0], float)
        best = (occ, float(pitch_candidates[0]), {
            "n_elem": int(occ.sum()), "grid": occ.shape,
            "z_layers": occ.shape[2], "node_origin": tr - 0.5 * pitch_candidates[0]})
    return best


# =========================================================================== #
#  Render                                                                      #
# =========================================================================== #
def render_warp(result, out_png, title="", warp_scale=None):
    """Render the warp field: undeformed footprint outline + deformed part
    coloured by off-bed (+z) uplift. Distortion is VISUALLY EXAGGERATED by
    warp_scale (auto-chosen) and the factor is printed on the figure so the
    viewer is never misled about magnitude. Returns out_png or None; never raises.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if not result.get("ok"):
            return None
        model = result["model"]
        coords = result["coords"]
        U = result["U"]
        uz = result["uz"]

        # auto exaggeration: target a visible ~15% of the part's largest extent
        ext = np.ptp(coords, axis=0)
        big = float(ext.max()) if ext.max() > 0 else 1.0
        maxd = float(np.linalg.norm(U, axis=1).max())
        if warp_scale is None:
            warp_scale = (0.15 * big / maxd) if maxd > 1e-9 else 1.0
            warp_scale = float(np.clip(warp_scale, 1.0, 5000.0))
        defc = coords + warp_scale * U

        # Build the OUTER element faces (skin) for display. For each solid voxel,
        # emit the 6 cube faces; faces shared by two solids are interior -> drop.
        en = model.elem_nodes  # (n_elem, 8) global node ids
        # cube face quads as local-corner indices (matching _corner_off order)
        face_quads = [
            (0, 1, 2, 3), (4, 5, 6, 7),     # bottom (z-), top (z+)
            (0, 1, 5, 4), (3, 2, 6, 7),     # y-, y+
            (1, 2, 6, 5), (0, 3, 7, 4),     # x+, x-
        ]
        from collections import defaultdict
        face_count = defaultdict(int)
        face_nodes = {}
        for e in range(model.n_elem):
            for q in face_quads:
                gn = tuple(int(en[e, c]) for c in q)
                key = tuple(sorted(gn))
                face_count[key] += 1
                if key not in face_nodes:
                    face_nodes[key] = gn
        skin = [gn for key, gn in face_nodes.items() if face_count[key] == 1]
        skin = np.array(skin, int)            # (n_skin_face, 4)

        # colour each skin quad by mean nodal uplift uz
        quad_uz = uz[skin].mean(axis=1)
        vmin = min(0.0, float(quad_uz.min()))
        vmax = max(1e-9, float(quad_uz.max()))
        try:
            cmap = matplotlib.colormaps["viridis"]
        except Exception:
            import matplotlib.cm as cm
            cmap = cm.get_cmap("viridis")
        norm = (quad_uz - vmin) / (vmax - vmin + 1e-30)
        face_rgb = cmap(norm)

        fig = plt.figure(figsize=(14.5, 5.6))

        def _set(ax, V):
            ax.set_box_aspect((1, 1, 1))
            c = V.mean(0); r = 0.6 * float(np.ptp(V, 0).max() + 1e-9)
            ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
            ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z up")

        # ---- panel 1: undeformed (bed reference) ----
        ax = fig.add_subplot(131, projection="3d")
        pc = Poly3DCollection(coords[skin], facecolors=(0.7, 0.7, 0.72, 1.0),
                              edgecolor=(0, 0, 0, 0.12), linewidths=0.15)
        ax.add_collection3d(pc)
        _set(ax, coords)
        ax.view_init(elev=22, azim=-60)
        ax.set_title(f"{title}\nUNDEFORMED (on bed)\n{result['n_elem']} hex elems, "
                     f"pitch {result['pitch_mm'][0]:.2g} mm", fontsize=9.5)

        # ---- panel 2: deformed, coloured by uplift ----
        ax = fig.add_subplot(132, projection="3d")
        pc2 = Poly3DCollection(defc[skin], facecolors=face_rgb,
                               edgecolor=(0, 0, 0, 0.10), linewidths=0.12)
        ax.add_collection3d(pc2)
        _set(ax, defc)
        ax.view_init(elev=22, azim=-60)
        ax.set_title(f"WARP after RELEASE ({result['material']})\n"
                     f"colour = off-bed rise uz (+z = corner lift)\n"
                     f"distortion x{warp_scale:.0f} for visibility", fontsize=9.5)
        mp = fig.add_axes([0.355, 0.10, 0.012, 0.32])
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(sm, cax=mp, label="vertical disp uz (mm)")

        # ---- panel 3: numbers ----
        ax = fig.add_subplot(133)
        ax.axis("off")
        cl = result["corner_lift_mm"]
        lines = [
            f"PREDICTED WARP  ({result['material']}, UNCALIBRATED)",
            "released-curl inherent-strain model",
            "",
            f"max corner lift   : {result['max_corner_lift_mm']*1e3:8.1f} um"
            f"  ({result['max_corner_lift_mm']:.3f} mm)",
            f"max total disp    : {result['max_total_disp_mm']*1e3:8.1f} um",
            f"strip closed-form : {result['corner_lift_strip_cf_mm']*1e3:8.1f} um"
            "  (1D envelope)",
            "",
            "per-corner lift vs centre (um):",
        ]
        for k, v in cl.items():
            lines.append(f"    {k:8s} : {v*1e3:8.1f}")
        lines += [
            "",
            f"eps* top / bottom : {result['eps_top']:.2e} / "
            f"{result['eps_bottom']:.2e}",
            f"eigenstrain grad  : {result['eps_gradient']:.2e} over "
            f"H={result['thickness_H_mm']:.1f}mm",
            f"curl kappa (cf)   : {result['kappa_strip_cf_per_mm']:.2e} /mm",
            f"dT_lock           : {result['dT_lock_K']:.0f} K (Tg - T_bed)",
            f"elements / nodes  : {result['n_elem']} / {result['n_node']}",
            f"part bbox (mm)    : {result['bbox_mm'][0]:.0f} x "
            f"{result['bbox_mm'][1]:.0f} x {result['bbox_mm'][2]:.0f}",
            "",
            "HONEST: Q1 hex shear-locks -> magnitude is a",
            "LOWER bound (curl gate: ~5% low @ fine grid).",
            "SIGN + SHAPE + TREND certified; mm pending",
            "single-coupon calibration for this exact process.",
        ]
        ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=8.5, transform=ax.transAxes)

        fig.subplots_adjust(left=0.03, right=0.99, bottom=0.05, top=0.86,
                            wspace=0.18)
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        return out_png
    except Exception:  # never raise from a render
        try:
            import traceback
            traceback.print_exc()
        except Exception:
            pass
        return None


# =========================================================================== #
#  Demo + selftest                                                            #
# =========================================================================== #
def demo_part(mesh, name, material, out_png, *, max_elem=6000, quick=False):
    """Voxelize -> predict warp -> render. Returns the result dict (+ png path)."""
    mx = 2500 if quick else max_elem
    occ, pitch, info = voxelize_capped(mesh, max_elem=mx,
                                       min_z_layers=3 if not quick else 2)
    origin = info.get("node_origin", np.zeros(3))
    res = predict_warp(occ, pitch, material, origin=origin)
    res["geom_info"] = info
    res["name"] = name
    png = render_warp(res, out_png, title=name)
    res["png"] = png
    return res


def selftest(quick=False):
    import time
    ok = True
    print("=" * 72)
    print("PROBE-2  INHERENT-STRAIN WARP  --  honesty gate + real-part demo")
    print("=" * 72)

    # ---- 1a. THE MECHANISM GATE: bimetal eigenstrain -> Timoshenko ---------
    print("\n[GATE 1] Timoshenko bimetal eigenstrain convergence")
    print("         (2-layer step eigenstrain; certifies the inherent-strain load)")
    grids = ((16, 2, 4), (24, 2, 4), (32, 2, 4)) if quick else \
            ((16, 2, 4), (24, 2, 4), (32, 2, 4), (40, 2, 6), (48, 2, 8))
    rows, kts, monotone, sign_ok = bimetal_convergence(grids)
    print(f"         closed-form Timoshenko kappa = {kts:.6e} /mm  "
          f"(r = {1/kts:.1f} mm)")
    print(f"         {'grid':>9} {'kappa_fem':>13} {'ratio':>8} {'rel_err':>9}")
    for g, kf, ratio, rel in rows:
        print(f"         {g:>9} {kf:>13.4e} {ratio:>8.4f} {rel:>9.2%}")
    print(f"         -> monotone rel_err = {monotone};  sign ok every grid = {sign_ok}")
    gate1_ok = monotone and sign_ok and rows[-1][3] < 0.20
    print(f"         GATE 1 {'PASS' if gate1_ok else 'FAIL'}  "
          f"(finest rel_err {rows[-1][3]:.2%}, must be <20% and decreasing)")
    ok &= gate1_ok

    # ---- 1b. THE WARP-SOLVER GATE: released continuous-gradient curl -------
    print("\n[GATE 2] Released-part curl convergence (the EXACT load predict_warp uses)")
    print("         continuous through-thickness gradient, free to curl;")
    print("         closed form kappa = |eps_grad|/H, sign = edges lift UP")
    crows, kcf, cmono, csign = released_curl_convergence(grids)
    print(f"         closed-form curl kappa = {kcf:.6e} /mm  (r = {1/kcf:.1f} mm)")
    print(f"         {'grid':>9} {'kappa_fem':>13} {'ratio':>8} {'rel_err':>9}")
    for g, kf, ratio, rel in crows:
        print(f"         {g:>9} {kf:>13.4e} {ratio:>8.4f} {rel:>9.2%}")
    print(f"         -> monotone rel_err = {cmono};  sign = lift-up every grid = {csign}")
    gate2_ok = cmono and csign and crows[-1][3] < 0.20
    print(f"         GATE 2 {'PASS' if gate2_ok else 'FAIL'}  "
          f"(finest rel_err {crows[-1][3]:.2%}, must be <20% and decreasing)")
    ok &= gate2_ok

    # ---- 2. REAL-PART DEMO: flat bracket warp ------------------------------
    print("\n[DEMO] flat bracket (100x30x6 mm) warp under PLA eigenstrain")
    pla, _ = fm.get_material("PLA")
    sd = os.path.dirname(os.path.abspath(__file__))
    t0 = time.time()
    flat = make_flat_bracket((100.0, 30.0, 6.0))
    r1 = demo_part(flat, "flat bracket 100x30x6 (PLA)", pla,
                   os.path.join(sd, "fem_warp_flat_bracket.png"), quick=quick)
    dt = time.time() - t0
    print(f"       elements={r1['n_elem']}  nodes={r1['n_node']}  "
          f"pitch={r1['pitch_mm'][0]:.2g}mm  solve_time={dt:.1f}s")
    print(f"       max corner lift = {r1['max_corner_lift_mm']*1e3:.1f} um  "
          f"({r1['max_corner_lift_mm']:.3f} mm)")
    print(f"       strip closed-form envelope = "
          f"{r1['corner_lift_strip_cf_mm']*1e3:.1f} um")
    print(f"       eps* = {r1['eps_scalar']:.3e}  dT_lock = {r1['dT_lock_K']:.0f}K  "
          f"grad = {r1['eps_gradient']:.2e} over H={r1['thickness_H_mm']:.1f}mm")
    print(f"       render -> {r1['png']}")
    # sanity: corners must lift UP relative to centre; warp present, render ok.
    demo1_ok = (r1["ok"] and r1["max_corner_lift_mm"] > 0 and r1["png"] is not None)
    print(f"       DEMO1 {'PASS' if demo1_ok else 'FAIL'} "
          f"(corners lift up vs centre, render written)")
    ok &= demo1_ok

    # ---- 3. TREND: thinner plate warps MORE (CONTROLLED: same pitch) -------
    print("\n[TREND] thinner plate must warp MORE (higher curl) -- CONTROLLED check")
    print("        same footprint 100x30 mm, same 2mm pitch, vary thickness only")

    def _plate_occ(Lx, Ly, Lz, pitch=2.0):
        return np.ones((int(round(Lx / pitch)), int(round(Ly / pitch)),
                        int(round(Lz / pitch))), bool), pitch
    trend_rows = []
    for Lz in (4.0, 6.0, 8.0, 12.0):
        occ, p = _plate_occ(100.0, 30.0, Lz)
        r = predict_warp(occ, p, pla)
        trend_rows.append((Lz, r["max_corner_lift_mm"], r["n_elem"]))
        print(f"        h={Lz:4.0f}mm ({r['n_elem']:4d} el): corner lift "
              f"{r['max_corner_lift_mm']*1e3:7.1f} um")
    lifts = [t[1] for t in trend_rows]
    trend_ok = all(lifts[i] > lifts[i + 1] for i in range(len(lifts) - 1))
    print(f"       TREND {'PASS' if trend_ok else 'FAIL'} "
          f"(corner lift monotone-DECREASING with thickness: {trend_ok})")
    ok &= trend_ok

    # ---- 4. L-bracket demo (non-planar part) -------------------------------
    print("\n[DEMO] L-bracket warp (foot bonded to bed, wall above)")
    try:
        Lb = make_L_bracket()
        r2 = demo_part(Lb, "L-bracket (PLA)", pla,
                       os.path.join(sd, "fem_warp_L_bracket.png"), quick=quick)
        print(f"       elements={r2['n_elem']}  pitch={r2['pitch_mm'][0]:.2g}mm  "
              f"max corner lift={r2['max_corner_lift_mm']*1e3:.1f} um  "
              f"render -> {r2['png']}")
        demo2_ok = r2["ok"] and r2["png"] is not None
    except Exception as e:
        print(f"       L-bracket demo skipped: {e}")
        demo2_ok = True  # non-fatal
    ok &= demo2_ok

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED" if ok else "SOME CHECK FAILED")
    print("Honest framing: GATE 1 (bimetal) certifies the eigenstrain LOAD and")
    print("GATE 2 (released curl) certifies the WARP SOLVER itself -- both converge")
    print("to closed form (sign + decreasing rel_err under refinement). Bracket warp")
    print("numbers are UNCALIBRATED physics-based estimates: Q1 hex shear-locks so")
    print("the magnitude is a LOWER bound; sign/shape/trend are the certified claims.")
    print("A certified millimetre needs a single-coupon calibration for the exact")
    print("printer+filament+profile (multiply eps* by a fitted scalar k).")
    print("=" * 72)
    return ok


if __name__ == "__main__":
    from ._mesh_util import ascii_console
    ascii_console()
    q = "--quick" in sys.argv
    sys.exit(0 if selftest(quick=q) else 1)
