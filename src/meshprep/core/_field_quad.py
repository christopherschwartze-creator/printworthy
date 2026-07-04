"""Global field-aligned quad remeshing (clean rebuild).

The per-part/segment-then-weld approach was abandoned: it missed ~14% of the
surface (blobby) and had incoherent edge flow because every part seam fought the
natural flow. GOOD quad retopo comes from a GLOBAL smooth cross-field that follows
curvature, extracted faithfully on the surface (the Instant-Meshes / field-aligned
family). This module is phase 1: the 4-RoSy cross-field.

Permissive only: numpy / scipy / trimesh. NO GPL.
"""
import numpy as np
import trimesh
from collections import defaultdict


# ---------------------------------------------------------------------------
#  tangent frames + curvature-based initialisation
# ---------------------------------------------------------------------------
def tangent_frames(mesh):
    """Per-vertex orthonormal frame (n, t1, t2). n = vertex normal."""
    N = np.asarray(mesh.vertex_normals, float)
    N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    # arbitrary t1 perpendicular to n (stable: cross with the least-aligned axis)
    ref = np.tile(np.array([1.0, 0, 0]), (len(N), 1))
    mask = np.abs(N[:, 0]) > 0.9
    ref[mask] = np.array([0, 1.0, 0])
    t1 = ref - (np.sum(ref * N, axis=1, keepdims=True)) * N
    t1 = t1 / (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12)
    t2 = np.cross(N, t1)
    return N, t1, t2


def principal_curv_data(mesh):
    """Principal-curvature cross data for curvature-aligned fielding.

    Returns
    -------
    pdir : (V,3) unit principal direction (max |kappa|) projected into the
           tangent plane; t1 fallback where curvature is degenerate.
    aniso : (V,) curvature ANISOTROPY in [0,1]:
              |k1 - k2| / (|k1| + |k2|)
            ~1 on a developable/cylinder body (one curvature ~0, the other not):
            the principal cross is WELL-DEFINED and should constrain the field.
            ~0 at an umbilic/sphere vertex (k1==k2): the principal direction is
            arbitrary, so the alignment penalty must switch OFF there (let the
            free smoothest field win). Zero where curvature is undefined.
    """
    from .curvature_decompose import principal_curvatures
    N, t1, t2 = tangent_frames(mesh)
    try:
        k1, k2, dirs = principal_curvatures(mesh)
    except Exception:
        return t1.copy(), np.zeros(len(N))
    dirs = np.asarray(dirs, float)
    pdir = t1.copy()
    if dirs.shape == (len(N), 3):
        d = dirs - np.sum(dirs * N, axis=1, keepdims=True) * N
        nrm = np.linalg.norm(d, axis=1)
        good = nrm > 1e-6
        pdir[good] = d[good] / nrm[good, None]
    k1 = np.asarray(k1, float); k2 = np.asarray(k2, float)
    denom = np.abs(k1) + np.abs(k2)
    aniso = np.where(denom > 1e-12, np.abs(k1 - k2) / (denom + 1e-12), 0.0)
    aniso = np.clip(aniso, 0.0, 1.0)
    return pdir, aniso


def principal_dir_init(mesh):
    """Initial cross direction from the principal curvature (max |kappa|) direction.
    Falls back to t1 where curvature is umbilic/degenerate. Returns unit (V,3)
    vectors in the tangent plane."""
    pdir, _ = principal_curv_data(mesh)
    return pdir


# ---------------------------------------------------------------------------
#  4-RoSy smoothing
# ---------------------------------------------------------------------------
def _best_rosy(acc, b, n):
    """Return the rotation of b (about n, by k*90 deg) best aligned with acc.
    The 4-RoSy representatives of b are {b, n x b, -b, -n x b}."""
    c = np.cross(n, b)
    cands = np.stack([b, c, -b, -c])
    dots = cands @ acc
    return cands[int(np.argmax(dots))]


def connection_laplacian(mesh, N, t1, t2, n_rosy=4):
    """Complex Hermitian connection Laplacian for an n-RoSy field (Knoppel et al.
    2013). L_ij = -w_ij e^{i n rho_ij}, where rho_ij is the Levi-Civita transport
    angle from frame j to frame i across the edge (measured via the shared edge
    direction in each tangent plane). Cotangent weights w_ij. Returns csr complex."""
    import scipy.sparse as sp
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, np.int64)
    n = len(V)
    # cotangent weights
    w = defaultdict(float)
    for tri in F:
        for a in range(3):
            i, j, k = int(tri[a]), int(tri[(a + 1) % 3]), int(tri[(a + 2) % 3])
            u1 = V[i] - V[k]; u2 = V[j] - V[k]
            cr = np.linalg.norm(np.cross(u1, u2))
            cot = (u1 @ u2) / cr if cr > 1e-18 else 0.0
            e = (i, j) if i < j else (j, i)
            w[e] += 0.5 * np.clip(cot, -1e4, 1e4)
    rows, cols, vals = [], [], []
    deg = np.zeros(n)
    for (i, j), wij in w.items():
        wij = max(wij, 1e-4)
        e = V[j] - V[i]
        phi_i = np.arctan2(e @ t2[i], e @ t1[i])         # edge i->j in i's frame
        phi_j = np.arctan2((-e) @ t2[j], (-e) @ t1[j])   # edge j->i in j's frame
        rho = phi_i - phi_j + np.pi                       # transport j -> i
        ph = np.exp(1j * n_rosy * rho)
        rows += [i, j]; cols += [j, i]; vals += [-wij * ph, -wij * np.conj(ph)]
        deg[i] += wij; deg[j] += wij
    L = sp.csr_matrix((vals, (rows, cols)), shape=(n, n)) + sp.diags(deg).astype(complex)
    return L


def smooth_field(mesh, dirs=None, iters=None, align_curv=0.0, curv=False,
                 curv_alpha=4.0, verbose=False):
    """GLOBALLY-OPTIMAL 4-RoSy field = smallest eigenvector of the complex
    connection Laplacian (Knoppel 2013). Gives the minimal-singularity smooth
    field directly (no iteration tuning).

    curv=True (OPT-IN, default OFF): solve the CURVATURE-ALIGNED field instead (a
    soft principal-curvature penalty, anisotropy-gated -- see `smooth_field_curv`).
    This breaks the degeneracy of the free smoothest field on intrinsically FLAT
    regions: a cylinder body has zero Gaussian curvature, so every constant-angle
    4-RoSy field is equally smooth and the free eigensolve lands on an ARBITRARY
    (sometimes diagonal) constant -> a sheared grid + nondeterministic output.
    Measured effect (remesh_mcf2, val4): cylinder +5..7, and the field becomes
    DETERMINISTIC; sphere safe (umbilic gate -> free field); BUT organic shapes
    REGRESS -3..-7 (the noisy principal direction on tube-like limbs trades grid
    regularity for curvature-following). Per-vertex anisotropy does NOT separate
    cylinders from organic tubes, so this is left OPT-IN, not the default. Use it
    for mechanical / developable inputs (cylinders, extrusions, CAD).

    align_curv in [0,1] (legacy): post-hoc BLEND toward the principal-curvature
    field on the free eigensolve. Superseded by curv=True (the in-solve penalty is
    smoother and umbilic-safe); kept for the A/B. If align_curv>0 the legacy blend
    path is used and curv is ignored. Returns per-vertex unit directions."""
    if curv and align_curv <= 0:
        try:
            return smooth_field_curv(mesh, alpha=curv_alpha, verbose=verbose)
        except Exception:
            pass  # fall through to the free eigensolve
    import scipy.sparse.linalg as spla
    N, t1, t2 = tangent_frames(mesh)
    L = connection_laplacian(mesh, N, t1, t2, n_rosy=4)
    n = L.shape[0]
    try:
        # smallest-magnitude eigenpair via shift-invert near 0 (robust for PSD L).
        # v0 = deterministic start (ones) -> reproducible field (ARPACK otherwise
        # uses a random start, injecting +-3-4% run-to-run variance into val4/flow;
        # also fixes the cylinder nondeterminism where the flat-region null mode is
        # picked at a random constant angle). Code-review finding.
        v0 = np.ones(n, dtype=complex)
        vals, vecs = spla.eigsh(L, k=1, sigma=1e-8, which="LM", v0=v0)
        u = vecs[:, 0]
    except Exception:
        # fallback: inverse power iteration
        import scipy.sparse as sp
        A = (L + 1e-6 * sp.identity(n)).tocsc()
        u = np.exp(1j * np.random.default_rng(0).uniform(0, 2 * np.pi, n))
        for _ in range(50):
            u = spla.spsolve(A, u); u = u / (np.linalg.norm(u) + 1e-12)
    if align_curv > 0:
        d0 = principal_dir_init(mesh)
        th0 = np.arctan2(np.einsum("ij,ij->i", d0, t2), np.einsum("ij,ij->i", d0, t1))
        u = (1 - align_curv) * (u / (np.abs(u) + 1e-12)) + align_curv * np.exp(1j * 4 * th0)
    theta = np.angle(u) / 4.0
    d = np.cos(theta)[:, None] * t1 + np.sin(theta)[:, None] * t2
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    return d / (nrm + 1e-12)


# ---------------------------------------------------------------------------
#  crease-aware field (constrain the field to follow sharp features)
# ---------------------------------------------------------------------------
def detect_creases(mesh, angle_thresh=0.7):
    """Crease edges = high dihedral angle (sharp features). Returns a per-vertex
    (mask, tangent dir): at a crease vertex the field should align to the crease
    direction. Prevents the field (and so the quads) from rounding off sharp
    edges -- the vase's stepped rims, mechanical creases, etc."""
    N, t1, t2 = tangent_frames(mesh)
    V = np.asarray(mesh.vertices, float)
    fae = np.asarray(mesh.face_adjacency_edges, np.int64)
    ang = np.asarray(mesh.face_adjacency_angles, float)
    mask = np.zeros(len(V), bool)
    acc = np.zeros((len(V), 3))
    for (a, b), an in zip(fae, ang):
        if an > angle_thresh:
            e = V[b] - V[a]; ne = np.linalg.norm(e)
            if ne < 1e-12:
                continue
            e = e / ne
            for v in (a, b):
                mask[v] = True
                acc[v] += e if acc[v] @ e >= 0 else -e   # consistent sign
    nrm = np.linalg.norm(acc, axis=1)
    good = mask & (nrm > 1e-9)
    cdir = np.zeros((len(V), 3)); cdir[good] = acc[good] / nrm[good, None]
    # project onto tangent plane
    cdir = cdir - np.einsum("ij,ij->i", cdir, N)[:, None] * N
    nn = np.linalg.norm(cdir, axis=1)
    good = good & (nn > 1e-9)
    cdir[good] = cdir[good] / nn[good, None]
    return good, cdir


def smooth_field_creases(mesh, alpha=20.0, angle_thresh=0.7):
    """Globally-optimal 4-RoSy field CONSTRAINED to follow creases. Solves the
    connection Laplacian with a soft alignment penalty at crease vertices (a linear
    solve, not the free eigensolve). Returns (d, crease_mask)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    N, t1, t2 = tangent_frames(mesh)
    mask, cdir = detect_creases(mesh, angle_thresh)
    if not mask.any():
        return smooth_field(mesh), mask
    L = connection_laplacian(mesh, N, t1, t2, n_rosy=4)
    th = np.arctan2(np.einsum("ij,ij->i", cdir, t2), np.einsum("ij,ij->i", cdir, t1))
    q = np.exp(1j * 4 * th) * mask                       # constraint targets
    P = sp.diags(mask.astype(float))
    A = (L + alpha * P).tocsc()
    try:
        x = spla.spsolve(A, alpha * q)
    except Exception:
        return smooth_field(mesh), mask
    theta = np.angle(x) / 4.0
    d = np.cos(theta)[:, None] * t1 + np.sin(theta)[:, None] * t2
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    return d / (nrm + 1e-12), mask


# ---------------------------------------------------------------------------
#  curvature-aligned field (break the flat-region degeneracy)
# ---------------------------------------------------------------------------
def smooth_field_curv(mesh, alpha=4.0, aniso_thresh=0.35, gamma=2.0,
                      with_creases=True, crease_alpha=20.0, crease_angle=0.7,
                      verbose=False):
    """Globally-optimal 4-RoSy field CONSTRAINED toward principal-curvature
    directions, to break the degeneracy of the free smoothest field on
    INTRINSICALLY FLAT regions (Knoppel 2013 + a per-vertex soft alignment
    penalty, same machinery as `smooth_field_creases`).

    WHY: a cylinder body has zero Gaussian curvature, so EVERY constant-angle
    4-RoSy field is equally smooth. The free eigensolve therefore lands on an
    ARBITRARY global constant angle -- often a DIAGONAL (45deg) field -> a sheared
    grid -> ~50% val4. The principal-curvature cross (circumferential = max|k|,
    axial = min|k|) is the geometrically-correct frame and IS well defined on a
    developable region, so we add a soft penalty pulling the field toward it.

    The penalty WEIGHT is the curvature ANISOTROPY |k1-k2|/(|k1|+|k2|), gated by
    `aniso_thresh` and sharpened by `gamma`:
        w_v = alpha * clip((aniso_v - thresh)/(1 - thresh), 0, 1) ** gamma
    -> on a cylinder body (aniso~1) every vertex is constrained -> the field
       LOCKS to axial+circumferential.
    -> on a sphere / umbilic vertex (aniso~0) the weight is ~0 -> the penalty is
       effectively OFF -> the free smoothest field is recovered (no regression).

    Solves  (L + diag(w)) x = (w * e^{i 4 theta_curv})  -- a linear solve, not
    the free eigensolve. Optionally folds in the crease constraint at the SAME
    time (creases get the strong `crease_alpha`, dominating curvature there).

    Returns per-vertex unit directions (V,3). Falls back to `smooth_field` on any
    failure or if NO vertex is anisotropic enough (fully-umbilic mesh).
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    N, t1, t2 = tangent_frames(mesh)
    n = len(N)
    try:
        pdir, aniso = principal_curv_data(mesh)
    except Exception:
        return smooth_field(mesh)
    # per-vertex penalty weight from anisotropy (gated + sharpened)
    span = max(1.0 - aniso_thresh, 1e-6)
    w = alpha * np.clip((aniso - aniso_thresh) / span, 0.0, 1.0) ** gamma
    # crease vertices (optional): align to the crease tangent with a strong weight
    cw = np.zeros(n)
    cdir = np.zeros((n, 3))
    if with_creases:
        cmask, cd = detect_creases(mesh, crease_angle)
        cw[cmask] = crease_alpha
        cdir[cmask] = cd[cmask]
    # combined target angle per vertex: crease dominates where present, else curv
    use_crease = cw > 0
    tgt_dir = np.where(use_crease[:, None], cdir, pdir)
    tot_w = np.where(use_crease, cw, w)
    if not np.any(tot_w > 1e-9):
        if verbose:
            print("    [field-curv] no anisotropic/crease vertices -> free field")
        return smooth_field(mesh)
    theta_t = np.arctan2(np.einsum("ij,ij->i", tgt_dir, t2),
                         np.einsum("ij,ij->i", tgt_dir, t1))
    q = np.exp(1j * 4 * theta_t) * tot_w                  # rhs = w * e^{i4theta}
    L = connection_laplacian(mesh, N, t1, t2, n_rosy=4)
    A = (L + sp.diags(tot_w.astype(float))).tocsc()
    try:
        x = spla.spsolve(A, q)
    except Exception:
        return smooth_field(mesh)
    if not np.all(np.isfinite(x)):
        return smooth_field(mesh)
    if verbose:
        nc = int(np.sum(w > 1e-9)); ncr = int(np.sum(cw > 0))
        print(f"    [field-curv] alpha={alpha} constrained={nc}/{n} "
              f"(crease={ncr}) mean_w={tot_w[tot_w>0].mean():.2f}")
    theta = np.angle(x) / 4.0
    d = np.cos(theta)[:, None] * t1 + np.sin(theta)[:, None] * t2
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    return d / (nrm + 1e-12)


# ---------------------------------------------------------------------------
#  singularities (per-triangle index of the cross-field)
# ---------------------------------------------------------------------------
def singularities(mesh, d):
    """Per-face cross-field index (0 regular; +/-1 = +/-1/4 turn singular).
    Sum over faces = 4*chi (Poincare-Hopf for a 4-RoSy field)."""
    N, _, _ = tangent_frames(mesh)
    F = np.asarray(mesh.faces, np.int64)
    idx = np.zeros(len(F), np.int64)
    for fi, tri in enumerate(F):
        total = 0
        for a in range(3):
            i, j = int(tri[a]), int(tri[(a + 1) % 3])
            # rotation k taking cross(i) to best-match cross(j) in a common plane
            navg = N[i] + N[j]; navg /= (np.linalg.norm(navg) + 1e-12)
            di = d[i] - (d[i] @ navg) * navg; di /= (np.linalg.norm(di) + 1e-12)
            dj = d[j] - (d[j] @ navg) * navg; dj /= (np.linalg.norm(dj) + 1e-12)
            c = np.cross(navg, di)
            ang = np.arctan2(dj @ c, dj @ di)              # in (-pi, pi]
            k = int(np.round(ang / (np.pi / 2))) % 4
            total += k if k <= 2 else k - 4
        total = -total                                  # orient so sum = +4*chi
        m = ((total % 4) + 4) % 4
        idx[fi] = -1 if m == 3 else m
    return idx


if __name__ == "__main__":
    import time
    def report(name, mesh, expect):
        t0 = time.time()
        d = smooth_field(mesh, iters=25)
        idx = singularities(mesh, d)
        nsing = int(np.sum(idx != 0))
        ssum = int(np.sum(idx))
        chi = int(mesh.euler_number)
        print(f"{name:14} V={len(mesh.vertices):5d} sing={nsing:3d} "
              f"index_sum={ssum:+d} (expect 4*chi={4*chi:+d}) {'OK' if ssum==4*chi else 'CHECK'} "
              f"({time.time()-t0:.1f}s)")

    print("=== 4-RoSy cross-field validation (index sum must = 4*euler) ===")
    report("sphere", trimesh.creation.icosphere(subdivisions=3), None)
    cyl = trimesh.creation.cylinder(radius=1, height=3, sections=40)
    report("cylinder", cyl, None)
    report("torus", trimesh.creation.torus(2, 0.7, major_sections=40, minor_sections=20), None)
    report("box", trimesh.creation.box((1, 1, 1)), None)
