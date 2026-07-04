"""Independent validation harness for the QuadriFlow remesher. Grades fidelity
(TWO-SIDED), manifoldness, flow quality vs the old per-part result."""
import numpy as np
import trimesh
from collections import defaultdict


def quads_to_tris(V, Q):
    tris = []
    for q in Q:
        u = list(dict.fromkeys(int(x) for x in q))
        if len(u) >= 3:
            tris.append([u[0], u[1], u[2]])
        if len(u) == 4:
            tris.append([u[0], u[2], u[3]])
    return trimesh.Trimesh(np.asarray(V), np.asarray(tris), process=False)


def two_sided_fidelity(orig, V, Q, n_samp=8000):
    """Both directions of Hausdorff / bbox: output->surface (drift) AND
    surface->output (coverage/blobbing). The second is what exposed the old 14%."""
    scale = float(np.linalg.norm(orig.vertices.max(0) - orig.vertices.min(0)))
    qm = quads_to_tris(V, Q)
    _, d1, _ = orig.nearest.on_surface(np.asarray(V))           # out verts -> surface
    s = orig.sample(n_samp)
    _, d2, _ = qm.nearest.on_surface(s)                         # surface -> out mesh
    return dict(out2surf_max=d1.max() / scale, out2surf_rms=np.sqrt((d1**2).mean()) / scale,
                surf2out_max=d2.max() / scale, surf2out_rms=np.sqrt((d2**2).mean()) / scale)


def manifold_stats(Q):
    cnt = defaultdict(int)
    for q in Q:
        qq = [int(x) for x in q]
        for a, b in zip(qq, qq[1:] + qq[:1]):
            if a != b:
                e = (a, b) if a < b else (b, a); cnt[e] += 1
    inc = list(cnt.values())
    border = sum(1 for c in inc if c == 1)
    nonman = sum(1 for c in inc if c > 2)
    return dict(border=border, nonmanifold=nonman, max_inc=max(inc) if inc else 0,
                manifold=(nonman == 0 and border == 0))


def valence_stats(Q, n):
    val = np.zeros(n, np.int64)
    adj = defaultdict(set)
    for q in Q:
        qq = [int(x) for x in q]
        for a, b in zip(qq, qq[1:] + qq[:1]):
            if a != b:
                adj[a].add(b); adj[b].add(a)
    for i in range(n):
        val[i] = len(adj[i])
    used = val > 0
    return dict(pct_val4=100 * np.mean(val[used] == 4) if used.any() else 0,
                n_irregular=int(np.sum((val != 4) & used)))


def flow_alignment(orig, V, Q):
    """How well do quad EDGES follow the cross-field, penalizing DIAGONAL flow.
    For each quad edge let theta = angle to the nearer cross arm {d, n x d}; score
    = 1 - |sin(2 theta)| (== 1 axis-aligned, 0 at 45deg diagonal). The OLD metric
    max(|cos|) had a ~0.85 floor (any tangent edge scores >=1/sqrt2), so a RANDOM
    mesh scored 0.90 -- blind. This sin(2theta) form actually separates field-
    aligned from isotropic (code-review finding #4). Mean in [0,1]."""
    from . import _field_quad as fq
    from scipy.spatial import cKDTree
    d = fq.smooth_field(orig)
    N, _, _ = fq.tangent_frames(orig)
    tree = cKDTree(np.asarray(orig.vertices))
    V = np.asarray(V)
    seen, scores = set(), []
    for q in Q:
        qq = [int(x) for x in q]
        for a, b in zip(qq, qq[1:] + qq[:1]):
            if a == b:
                continue
            e = (a, b) if a < b else (b, a)
            if e in seen:
                continue
            seen.add(e)
            ev = V[b] - V[a]; nev = np.linalg.norm(ev)
            if nev < 1e-9:
                continue
            ev = ev / nev
            mid = 0.5 * (V[a] + V[b])
            _, k = tree.query(mid)
            di = d[k]; ni = N[k]
            c = np.cross(ni, di)
            cd = abs(ev @ di); cc = abs(ev @ c)        # |cos| to the two arms
            sin2 = 2.0 * cd * cc                         # |sin(2 theta)| (theta to arm di)
            scores.append(1.0 - min(sin2, 1.0))
    return float(np.mean(scores)) if scores else float("nan")


def excess_irregular(orig, V, Q):
    """The PRIMARY quality number (code-review #4): (#irregular - |topological
    floor|) / #quads. 0 = only the forced cones; high = manufactured irregulars.
    Directly measures the val4 defect, unlike val4% (which a deg-6-injecting
    cleanup can inflate by shrinking the denominator)."""
    from . import _field_quad as fq
    adj = defaultdict(set)
    for q in Q:
        qq = [int(x) for x in q]
        for a, b in zip(qq, qq[1:] + qq[:1]):
            if a != b:
                adj[a].add(b); adj[b].add(a)
    val = np.array([len(adj[i]) for i in range(len(V))])
    used = val > 0
    n_irr = int(np.sum(val[used] != 4))
    try:
        floor = abs(int(np.sum(fq.singularities(orig, fq.smooth_field(orig)))))
    except Exception:
        floor = 0
    nq = max(len(Q), 1)
    return (n_irr - floor) / nq, n_irr, floor


def render_gallery(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    n = len(results)
    fig = plt.figure(figsize=(4 * n, 4))
    for i, (nm, V, Q, info) in enumerate(results):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        polys = [V[[x for x in dict.fromkeys(int(c) for c in q)]] for q in Q]
        ax.add_collection3d(Poly3DCollection(polys, facecolor="#9db8d8",
                            edgecolor="#22364f", linewidths=0.3, alpha=0.95))
        V = np.asarray(V); mn, mx = V.min(0), V.max(0); ct = (mn + mx) / 2; r = (mx - mn).max() / 2
        for s, c in zip("xyz", ct):
            getattr(ax, f"set_{s}lim")(c - r, c + r)
        ax.set_axis_off(); ax.view_init(18, 30)
        ax.set_title(f"{nm}\n{info}", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight")
    print("saved", path)
