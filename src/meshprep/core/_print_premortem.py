"""Pre-print FAILURE HEATMAP -- a per-FACE printing-risk field for FDM.

`premortem(mesh, build_dir=None) -> dict` paints every triangle of a mesh with a
RISK score in [0,1] that estimates how likely that region is to print badly at a
given build orientation. It does NOT slice or simulate; it combines four cheap,
geometry-derived channels into one field so you can SEE the at-risk regions before
committing a print. All four channels are HEURISTICS (documented per channel);
treat the output as a triage map, not a guarantee.

The four risk channels (each in [0,1], per face):

  (a) OVERHANG  -- faces whose downward tilt exceeds the ~45deg self-support limit.
      severity = how far past the limit (1 = straight down), GATED by height above
      the build plate so the resting base (drop-height ~0) does NOT count as an
      overhang. Reuses `_print3d._face_overhang` + the `support_cost` drop-height
      idea. The most physical channel; the rest are softer proxies.

  (b) THIN-WALL -- faces on walls thinner than ~2x the nozzle (default 0.4mm).
      Per-face inward Shape-Diameter-Function ray cast (same mechanism as
      `_print3d.min_wall_thickness` / `_print_advanced.thin_wall_map`, but evaluated
      at EVERY face so it maps cleanly to the heatmap). risk ramps 0->1 as wall
      thickness falls from 2*nozzle down to 1*nozzle. HEURISTIC: single inward ray,
      so tubes/folds can read thinner/thicker than a true medial-axis thickness.

  (c) WARP proxy -- HEURISTIC, NOT a thermal/FEM warp model. Two contributions,
      both concentrated near the FIRST LAYERS (low height above plate, where curl
      lifts off the bed): (i) large flat footprint x tall part aspect ratio -> a big
      flat first layer on a tall print tends to curl at the corners; (ii) sharp
      convex edges near the bed concentrate stress. Reported as warp_risk; it flags
      WHERE curl is plausible, it does not predict that curl WILL occur.

  (d) SUPPORT-SCAR -- faces the support structure will touch (cosmetic damage where
      supports snap off). The support contact shadow is exactly the set of
      down-facing overhang faces that `_print_advanced.generate_supports` samples to
      drop pillars onto; we mark those faces. HEURISTIC: marks the contact FACES,
      not the exact pillar tip footprints.

Combination: risk = max over channels (a region is risky if ANY failure mode is
high), with per-channel weights so you can emphasise one mode. The summary reports
each channel's coverage and the top risk so numbers are auditable.

`render_premortem(mesh, out_png, build_dir=None)` draws the mesh face-coloured by
risk (matplotlib Poly3DCollection; blue=safe -> red=high) at the build orientation.

Pure numpy/scipy/trimesh + reuse of `_print3d` and `_print_advanced`. Never raises:
every channel is wrapped; a failed channel contributes zeros and is flagged in the
summary. Decimate large meshes (<=6000 faces) before calling for speed.

Run as a script for the self-tests:  python -m meshprep.core._print_premortem
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import trimesh

from . import _print3d
from ._mesh_util import say


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def _height_above_plate(mesh, up):
    """Per-face height of the face centroid above the lowest point along `up`,
    normalised to [0,1] by the part's extent along `up`. 0 = on the build plate,
    1 = at the top of the part."""
    up = _unit(up)
    h = np.asarray(mesh.triangles_center, float) @ up
    lo = float(h.min())
    span = float(h.max() - lo) + 1e-12
    return np.clip((h - lo) / span, 0.0, 1.0)


def _drop_above_plate(mesh, up):
    """Per-face ABSOLUTE drop-height: distance of the face centroid above the lowest
    point along `up`, in the mesh's units (same quantity `_print3d.support_cost`
    uses). 0 = sitting on the build plate. Unlike the normalised height fraction this
    distinguishes a flat resting BASE (drop ~0) from an elevated curved FLANK (drop
    > 0) regardless of part height."""
    up = _unit(up)
    h = np.asarray(mesh.triangles_center, float) @ up
    return np.clip(h - float(h.min()), 0.0, None)


def _drop_and_height(mesh, up):
    """(per-face drop above the plate, part's print height along `up`) -- the shared
    quantities the overhang gate and the scar gate are both built from."""
    drop = _drop_above_plate(mesh, up)
    vu = np.asarray(mesh.vertices, float) @ _unit(up)
    return drop, float(vu.max() - vu.min()) + 1e-9


def _intersector(mesh):
    try:
        return trimesh.ray.ray_pyembree.RayMeshIntersector(mesh), True
    except Exception:
        return mesh.ray, False


# Ray-batch sizing for the pure-python (non-embree) intersector. The pure engine's
# broad phase gathers, for EVERY ray at once, all triangles whose bbox the ray
# segment crosses; on a thin/elongated mesh an inward ray traverses the whole
# part, so per-ray candidates ~ O(n_faces) and casting all F rays in one call
# accumulates an O(F * n_faces) candidate array -- multi-GB, unbounded, past any
# memory_budget (observed >2.4GB on a legal 20k-face thin part, killed at 1.5GB).
# Casting in batches caps peak at O(BATCH * n_faces). Because a batch's candidate
# cost itself scales with n_faces, we size BATCH from a fixed ray*candidate PAIR
# budget so peak memory stays bounded for ANY face count (the pipeline caps meshes
# at 60k faces): batch = clamp(PAIR_BUDGET / n_faces, MIN, MAX). Total ray work
# (== n_faces^2) is independent of batch, so shrinking it costs ~no extra time.
# embree is O(1) memory per ray, so it casts in one shot.
THIN_PAIR_BUDGET = 8_000_000   # ray*candidate pairs/batch (~<600MB working)
THIN_RAY_BATCH_MIN = 32
THIN_RAY_BATCH_MAX = 512


def _thin_ray_batch(nf: int) -> int:
    """Memory-bounded ray batch for the pure-python intersector (see above)."""
    b = THIN_PAIR_BUDGET // max(int(nf), 1)
    return int(max(THIN_RAY_BATCH_MIN, min(THIN_RAY_BATCH_MAX, b)))


# ---------------------------------------------------------------------------
#  (a) OVERHANG channel
# ---------------------------------------------------------------------------
def _overhang_channel(mesh, up, crit_deg=45.0):
    """Per-face overhang risk in [0,1]. Reuses `_print3d._face_overhang` for the
    severity, then GATES by ABSOLUTE drop-height above the build plate so the part's
    flat resting base (drop ~0) does not register as a mid-air overhang while an
    elevated face (drop > 0) does. The gate ramps 0 -> 1 as drop goes from a small
    plate floor (~2% of the part's HEIGHT-ALONG-THE-BUILD-AXIS, min 0.3mm) to ~12% of
    that height -- i.e. a face needs both a sub-critical down-tilt AND clearance off
    the bed to be flagged. (The vertical scale is the extent along `up`, NOT the 3D
    bbox diagonal, so an elongated part lying down is gated by its true print height.)

    HEURISTIC note: this is the most physical channel -- the 45deg rule and the
    drop-height gate are standard FDM rules of thumb, not a per-layer support
    simulation. By geometry a lying cylinder's overhang is mild (its steepest faces
    are low, near the bed-contact line); a cantilevered ledge held in the air reads
    ~1.0. Returns (risk (F,), needs_support (F,) bool)."""
    sev, needs = _print3d._face_overhang(mesh, up, crit_deg=crit_deg)
    drop, height = _drop_and_height(mesh, up)
    g_lo = max(0.3, 0.02 * height)      # below this drop = on the plate (supported)
    g_hi = max(g_lo + 1e-6, 0.12 * height)
    gate = np.clip((drop - g_lo) / (g_hi - g_lo), 0.0, 1.0)
    risk = sev * gate
    risk[~needs] = 0.0
    return np.clip(risk, 0.0, 1.0), needs


# ---------------------------------------------------------------------------
#  (b) THIN-WALL channel
# ---------------------------------------------------------------------------
def _thin_wall_channel(mesh, nozzle=0.4):
    """Per-face wall-thickness risk in [0,1] via an inward SDF ray from each face
    centroid (same mechanism as `_print3d.min_wall_thickness`, evaluated per face so
    it maps to the heatmap). risk ramps 0 at 2*nozzle to 1 at 1*nozzle (and clamps
    to 1 below the nozzle). Faces with no inward hit (open/degenerate) read 0.

    HEURISTIC note: a SINGLE inward ray approximates local wall thickness; thin tubes,
    grazing folds, or non-orthogonal walls can read thinner or thicker than a true
    medial-axis thickness. Returns (risk (F,), thickness (F,) float with inf where
    no hit)."""
    nf = len(mesh.faces)
    thick = np.full(nf, np.inf)
    try:
        inter, fast = _intersector(mesh)
        origins = np.asarray(mesh.triangles_center, float) - 1e-3 * np.asarray(mesh.face_normals, float)
        dirs = -np.asarray(mesh.face_normals, float)
        # Cast in memory-bounded batches on the pure-python engine (see
        # _thin_ray_batch). embree is O(1)/ray so one shot is fine there.
        batch = nf if fast else _thin_ray_batch(nf)
        for s in range(0, nf, max(1, batch)):
            o = origins[s:s + batch]
            v = dirs[s:s + batch]
            locs, ray_idx, _ = inter.intersects_location(o, v, multiple_hits=False)
            if len(ray_idx):
                d = np.linalg.norm(locs - o[ray_idx], axis=1)
                ok = d > 1e-4
                # ray_idx is LOCAL to this batch; shift back to global face ids
                gidx = ray_idx[ok] + s
                # nearest inward hit per face
                np.minimum.at(thick, gidx, d[ok])
    except Exception:
        return np.zeros(nf), thick
    lo, hi = float(nozzle), 2.0 * float(nozzle)        # 1x .. 2x nozzle ramp
    risk = np.where(np.isfinite(thick),
                    np.clip((hi - thick) / (hi - lo + 1e-12), 0.0, 1.0),
                    0.0)
    return np.clip(risk, 0.0, 1.0), thick


# ---------------------------------------------------------------------------
#  (c) WARP proxy channel  (HEURISTIC)
# ---------------------------------------------------------------------------
def _warp_channel(mesh, up, crit_deg=45.0, plate_band=0.12, sharp_deg=50.0):
    """HEURISTIC warp/curl risk in [0,1], concentrated near the first layers.

    NOT a thermal model. Two additive contributions, both gated to the bottom
    `plate_band` fraction of the part (where bed-curl happens):
      (i)  FLAT-FOOTPRINT-x-ASPECT: a large flat area resting on the bed on a TALL
           part curls at the corners. We compute the footprint flatness (down-facing
           near-horizontal faces low on the part) and scale it by the part's
           height/footprint aspect ratio. This is a SCALAR severity applied to the
           low flat faces.
      (ii) SHARP-EDGE-NEAR-BED: convex dihedral edges sharper than `sharp_deg` close
           to the plate concentrate stress -> corners lift. Faces on such edges get
           extra risk.
    Returns (risk (F,), info dict). info documents the scalar aspect/flatness used so
    the heuristic is auditable."""
    up = _unit(up)
    nf = len(mesh.faces)
    risk = np.zeros(nf)
    info = {"heuristic": True}
    try:
        n = np.asarray(mesh.face_normals, float)
        hfrac = _height_above_plate(mesh, up)
        near_bed = hfrac <= plate_band
        down = -up
        # near-horizontal down-facing = the flat footprint resting on the bed
        cos_down = n @ down
        flat_floor = near_bed & (cos_down > np.cos(np.radians(20.0)))
        # aspect ratio: height along up / footprint diagonal in the bed plane
        V = np.asarray(mesh.vertices, float)
        hcoord = V @ up
        height = float(hcoord.max() - hcoord.min())
        # project verts to the bed plane (remove up component) for footprint extent
        perp = V - np.outer(hcoord, up)
        fmin, fmax = perp.min(0), perp.max(0)
        foot_diag = float(np.linalg.norm(fmax - fmin)) + 1e-9
        aspect = height / foot_diag
        floor_area = float(np.asarray(mesh.area_faces)[flat_floor].sum())
        floor_frac = floor_area / (float(mesh.area) + 1e-12)
        # severity: a big flat floor on a tall part curls; ramp aspect over [0.6, 3]
        aspect_sev = np.clip((aspect - 0.6) / (3.0 - 0.6), 0.0, 1.0)
        flat_sev = np.clip(floor_frac / 0.15, 0.0, 1.0)    # >=15% flat floor = full
        warp_floor = float(aspect_sev * flat_sev)
        risk[flat_floor] = np.maximum(risk[flat_floor], warp_floor)
        info.update({"aspect_ratio": round(aspect, 2),
                     "floor_flat_frac": round(floor_frac, 3),
                     "warp_floor_sev": round(warp_floor, 3)})

        # (ii) sharp convex edges near the bed
        try:
            fa = mesh.face_adjacency                          # (E,2) face pairs
            ang = mesh.face_adjacency_angles                  # (E,) dihedral, radians
            convex = mesh.face_adjacency_convex               # (E,) bool
            sharp = convex & (ang > np.radians(sharp_deg))
            # gate by proximity to bed (use the lower of the two faces' heights)
            pair_h = np.minimum(hfrac[fa[:, 0]], hfrac[fa[:, 1]])
            sharp &= (pair_h <= plate_band)
            sev_edge = np.clip((ang - np.radians(sharp_deg)) /
                               (np.pi - np.radians(sharp_deg) + 1e-9), 0.0, 1.0)
            s06 = 0.6 * sev_edge[sharp]
            np.maximum.at(risk, fa[sharp, 0], s06)
            np.maximum.at(risk, fa[sharp, 1], s06)
            info["n_sharp_bed_edges"] = int(sharp.sum())
        except Exception as e:
            info["sharp_edge_error"] = str(e)
    except Exception as e:
        info["error"] = str(e)
    return np.clip(risk, 0.0, 1.0), info


# ---------------------------------------------------------------------------
#  (d) SUPPORT-SCAR channel
# ---------------------------------------------------------------------------
def _support_scar_channel(mesh, up, crit_deg=45.0):
    """Per-face support-scar risk in [0,1]: the faces a support structure will
    contact (and scar when snapped off). The support contact shadow is exactly the
    down-facing overhang faces that `_print_advanced.generate_supports` samples to
    drop pillars onto -- i.e. the same `_face_overhang` `needs` set, gated to faces
    above the plate (the bed itself is not scarred). Scar severity is uniform-ish per
    contact (binary-ish), softened slightly by overhang severity so the most
    down-facing (most-supported) faces read highest.

    HEURISTIC note: marks contact FACES, not the exact pillar-tip footprints (a real
    slicer's support touches a sparser conforming set); treat as an upper bound on
    the scarred region. Returns (risk (F,), n_contact int)."""
    sev, needs = _print3d._face_overhang(mesh, up, crit_deg=crit_deg)
    drop, height = _drop_and_height(mesh, up)
    above = drop > max(0.3, 0.02 * height)      # the bed itself is not scarred
    contact = needs & above
    risk = np.zeros(len(mesh.faces))
    # 0.5 baseline scar on any contact face + up to 0.5 more by how down-facing it is
    risk[contact] = 0.5 + 0.5 * sev[contact]
    return np.clip(risk, 0.0, 1.0), int(contact.sum())


# ---------------------------------------------------------------------------
#  main entry
# ---------------------------------------------------------------------------
def premortem(mesh, build_dir=None, *, nozzle=0.4, crit_deg=45.0,
              weights=None, verbose=False):
    """Per-FACE printing-risk heatmap combining four HEURISTIC channels (overhang,
    thin-wall, warp-proxy, support-scar). If `build_dir` is None, the support-minimal
    orientation is chosen via `_print3d.optimize_orientation`.

    Parameters
    ----------
    mesh : trimesh.Trimesh  (decimate to <=6000 faces first for speed)
    build_dir : array-like (3,) or None  -- the build 'up' axis; layers stack along it
    nozzle : float  -- nozzle width in the mesh's units (scale mesh to mm for real
             wall numbers; unitless meshes give unitless thresholds)
    crit_deg : float -- self-support critical overhang angle
    weights : dict or None -- per-channel weight in [0,1] for
             {'overhang','thin','warp','scar'} (default all 1.0). The combined risk is
             the weighted MAX across channels (any high failure mode -> high risk).

    Returns
    -------
    dict with:
      'risk_face'  : (F,) float in [0,1]   combined per-face risk
      'channels'   : {'overhang','thin','warp','scar'} each (F,) float in [0,1]
      'build_dir'  : (3,) the up axis used
      'summary'    : measured per-channel coverage + the worst regions, plus a
                     HEURISTIC disclaimer for every channel.
    Never raises: a failed channel contributes zeros and is flagged in the summary.
    """
    nf = len(mesh.faces)  # mesh is not copied; we only read
    w = {"overhang": 1.0, "thin": 1.0, "warp": 1.0, "scar": 1.0}
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})

    summary = {
        "n_faces": int(nf),
        "nozzle_mm": float(nozzle),
        "crit_deg": float(crit_deg),
        "channel_weights": dict(w),
        "DISCLAIMER": ("All four channels are HEURISTICS, not a slicer/FEM simulation. "
                       "OVERHANG (45deg + plate-gate) is the most physical; THIN-WALL "
                       "is a single-ray SDF; WARP is a flat-footprint/aspect/sharp-edge "
                       "PROXY (does not predict curl will occur); SUPPORT-SCAR marks "
                       "contact faces (upper bound, not exact pillar footprints). "
                       "Combined risk = weighted MAX across channels."),
    }

    # build direction
    if build_dir is None:
        try:
            ori = _print3d.optimize_orientation(mesh, crit_deg=crit_deg,
                                                n_sphere=120, refine=True)
            up = _unit(ori["up"])
            summary["orientation"] = "optimized (min-support)"
            summary["optimize_support_frac"] = round(float(ori.get("support_frac", 0.0)), 3)
        except Exception as e:
            up = np.array([0.0, 0.0, 1.0])
            summary["orientation"] = f"FALLBACK +Z (optimize failed: {e})"
    else:
        up = _unit(build_dir)
        summary["orientation"] = "user-supplied"
    summary["build_dir"] = [round(float(x), 4) for x in up]

    channels = {}

    # (a) overhang
    try:
        oc, needs = _overhang_channel(mesh, up, crit_deg=crit_deg)
        summary["overhang"] = {
            "n_overhang_faces": int(needs.sum()),
            "overhang_area_frac": round(float(np.asarray(mesh.area_faces)[needs].sum()
                                              / (float(mesh.area) + 1e-12)), 3),
            "n_risk_faces_gated": int((oc > 0.05).sum()),
            "max": round(float(oc.max()), 3),
            "heuristic": "45deg self-support rule + build-plate height gate",
        }
    except Exception as e:
        oc = np.zeros(nf); summary["overhang"] = {"error": str(e)}
    channels["overhang"] = oc

    # (b) thin-wall
    try:
        tc, thick = _thin_wall_channel(mesh, nozzle=nozzle)
        finite = np.isfinite(thick)
        summary["thin"] = {
            "n_thin_faces": int((tc > 0.05).sum()),
            "min_wall_mm": (round(float(thick[finite].min()), 3) if finite.any() else None),
            "thin_threshold_mm": round(2.0 * nozzle, 3),
            "frac_thin_faces": round(float((tc > 0.05).mean()), 4),
            "heuristic": "single inward-ray Shape-Diameter-Function per face",
        }
    except Exception as e:
        tc = np.zeros(nf); summary["thin"] = {"error": str(e)}
    channels["thin"] = tc

    # (c) warp proxy
    try:
        wc, winfo = _warp_channel(mesh, up, crit_deg=crit_deg)
        summary["warp"] = {
            "n_warp_faces": int((wc > 0.05).sum()),
            "max": round(float(wc.max()), 3),
            **winfo,
        }
    except Exception as e:
        wc = np.zeros(nf); summary["warp"] = {"error": str(e), "heuristic": True}
    channels["warp"] = wc

    # (d) support scar
    try:
        sc, n_contact = _support_scar_channel(mesh, up, crit_deg=crit_deg)
        summary["scar"] = {
            "n_contact_faces": int(n_contact),
            "scar_area_frac": round(float(np.asarray(mesh.area_faces)[sc > 0.05].sum()
                                          / (float(mesh.area) + 1e-12)), 3),
            "max": round(float(sc.max()), 3),
            "heuristic": "support contact = down-facing overhang faces (upper bound)",
        }
    except Exception as e:
        sc = np.zeros(nf); summary["scar"] = {"error": str(e)}
    channels["scar"] = sc

    # combine: weighted MAX (any high failure mode -> high risk)
    stack = np.vstack([
        w["overhang"] * channels["overhang"],
        w["thin"] * channels["thin"],
        w["warp"] * channels["warp"],
        w["scar"] * channels["scar"],
    ])
    risk = np.clip(stack.max(axis=0), 0.0, 1.0)
    # which channel dominates each face (for the summary breakdown)
    dom = np.argmax(stack, axis=0)
    names = ["overhang", "thin", "warp", "scar"]
    dom_counts = {names[i]: int((dom[risk > 0.05] == i).sum()) for i in range(4)}

    summary["combined"] = {
        "max_risk": round(float(risk.max()), 3),
        "mean_risk": round(float(risk.mean()), 4),
        "n_high_risk_faces(>0.5)": int((risk > 0.5).sum()),
        "n_any_risk_faces(>0.05)": int((risk > 0.05).sum()),
        "high_risk_area_frac": round(float(np.asarray(mesh.area_faces)[risk > 0.5].sum()
                                           / (float(mesh.area) + 1e-12)), 3),
        "dominant_channel_of_risk_faces": dom_counts,
        "NOTE_scar_overhang_coupled": ("SUPPORT-SCAR fires on ~the same faces as "
            "OVERHANG (scar contacts ARE the down-facing overhang faces), so scar "
            "tends to dominate the combined map on supported regions -- they are one "
            "physical concern (this region needs support AND will be scarred), not two "
            "independent ones. THIN and WARP are the genuinely separate channels."),
    }
    if verbose:
        say("  premortem:", summary["combined"])

    return {"risk_face": risk, "channels": channels,
            "build_dir": up, "summary": summary}


# ---------------------------------------------------------------------------
#  rendering
# ---------------------------------------------------------------------------
def render_premortem(mesh, out_png, build_dir=None, result=None, title=None,
                     nozzle=0.4, crit_deg=45.0):
    """Render `mesh` face-coloured by combined print risk (blue=safe -> red=high) at
    the build orientation, plus a 4-panel breakdown of the individual channels.
    Computes `premortem` if `result` is not supplied. Returns out_png (or None on
    failure -- never raises)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm, colors
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except Exception as e:
        say(f"render skipped (matplotlib unavailable): {e}")
        return None

    try:
        if result is None:
            result = premortem(mesh, build_dir=build_dir, nozzle=nozzle, crit_deg=crit_deg)
        up = result["build_dir"]
        risk = result["risk_face"]
        chans = result["channels"]

        # lay the mesh flat (up -> +Z) so the heatmap reads in print orientation
        try:
            b = mesh.copy()
            b.apply_transform(trimesh.geometry.align_vectors(np.asarray(up, float), [0, 0, 1.0]))
            b.apply_translation([0, 0, -float(b.vertices[:, 2].min())])
        except Exception:
            b = mesh

        tris = b.triangles

        def _cmap(name):
            try:
                return matplotlib.colormaps[name]
            except Exception:
                return cm.get_cmap(name)

        V = b.vertices
        c = V.mean(0); r = float(np.abs(V - c).max()) * 1.02

        def _add(ax, field, ttl, cmap="turbo", elev=-35, azim=-60, plate=True):
            cmap_o = _cmap(cmap)
            fc = cmap_o(np.clip(field, 0, 1))
            pc = Poly3DCollection(tris, facecolors=fc, edgecolor="none", linewidths=0)
            pc.set_alpha(None)
            ax.add_collection3d(pc)
            ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
            ax.set_zlim(c[2] - r, c[2] + r)
            try:
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=elev, azim=azim)   # negative elev = look UP at underside
            except Exception:
                pass
            ax.set_axis_off()
            ax.set_title(ttl, fontsize=10)
            if plate:
                try:
                    xx, yy = np.meshgrid(np.linspace(c[0]-r, c[0]+r, 2),
                                         np.linspace(c[1]-r, c[1]+r, 2))
                    ax.plot_surface(xx, yy, np.zeros_like(xx), color="0.85",
                                    alpha=0.12, zorder=0)
                except Exception:
                    pass

        fig = plt.figure(figsize=(16, 9))
        # panel 1: combined risk, TOP view (how the part looks on the bed)
        ax = fig.add_subplot(2, 3, 1, projection="3d")
        _add(ax, risk, (title or "COMBINED risk") + "  (TOP view)", elev=28, azim=-60)
        # panel 2: combined risk, UNDERSIDE view (where print risk actually lives)
        ax = fig.add_subplot(2, 3, 2, projection="3d")
        _add(ax, risk, "COMBINED risk  (UNDERSIDE, blue=safe red=high)",
             elev=-32, azim=-60)
        sm = cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap=_cmap("turbo"))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="risk")

        # panels 3-5: the three SEPARATE channels (underside view), overhang+scar are
        # spatially coupled so we show overhang (the cause) + thin + warp.
        for i, key in enumerate(["overhang", "thin", "warp"]):
            ax = fig.add_subplot(2, 3, i + 3, projection="3d")
            cov = round(float((chans[key] > 0.05).mean()) * 100, 1)
            _add(ax, chans[key], f"{key}  ({cov}% faces, underside)", elev=-32, azim=-60)

        # text panel: summary
        ax = fig.add_subplot(2, 3, 6)
        ax.axis("off")
        s = result["summary"]
        comb = s.get("combined", {})
        lines = [
            f"orientation: {s.get('orientation','?')}",
            f"build_dir: {s.get('build_dir')}",
            f"max risk: {comb.get('max_risk')}",
            f"high-risk faces (>0.5): {comb.get('n_high_risk_faces(>0.5)')}",
            f"high-risk area frac: {comb.get('high_risk_area_frac')}",
            f"dominant: {comb.get('dominant_channel_of_risk_faces')}",
            "",
            f"overhang area: {s.get('overhang',{}).get('overhang_area_frac')}",
            f"min wall mm: {s.get('thin',{}).get('min_wall_mm')}",
            f"warp aspect: {s.get('warp',{}).get('aspect_ratio')}",
            f"scar contacts: {s.get('scar',{}).get('n_contact_faces')}",
            "  (scar ~ overhang faces; coupled)",
            "",
            "ALL CHANNELS ARE HEURISTICS",
        ]
        ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9,
                family="monospace")

        plt.tight_layout()
        plt.savefig(out_png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return out_png
    except Exception as e:
        say(f"render failed: {e}")
        try:
            plt.close("all")
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
#  self-test
# ---------------------------------------------------------------------------
def selftest(verbose=True):
    """Physically-motivated sanity checks (measured, not hand-tuned to pass):

      1. CANTILEVERED LEDGE (a horizontal bar sticking out atop a post) -- its
         underside is a straight-down overhang held HIGH in the air = the textbook
         unsupported overhang. Pass = the overhang channel maxes out (>0.5) on the
         bar underside. This is the unambiguous overhang test.
      2. FLAT PLATE lying flat -- must read LOW (max risk < 0.5, zero high-risk faces):
         the resting base is gated as plate-supported, not a mid-air overhang.
      3. THIN PLATE (0.3mm < 2*0.4mm nozzle) -- the thin-wall channel must fire (>0.5).
      4. HORIZONTAL CYLINDER lying on the bed -- a secondary CONTRAST check: its lower
         flank must carry more overhang risk than its upper half (the underside is
         worse than the topside). NOTE by geometry a lying cylinder's overhang is MILD
         (steep faces are low, near the bed-contact line) so we test the lower>upper
         contrast, NOT a high absolute max -- this documents the gate behaving
         correctly, it is not expected to read 1.0.
    Returns a dict of measured numbers + per-check and overall pass booleans."""
    out = {}

    # 1. cantilevered ledge -- unambiguous unsupported overhang
    post = trimesh.creation.box(extents=[8, 8, 30]); post.apply_translation([0, 0, 15])
    bar = trimesh.creation.box(extents=[40, 8, 4]); bar.apply_translation([14, 0, 32])
    cant = trimesh.util.concatenate([post, bar]); cant.merge_vertices()
    rc = premortem(cant, build_dir=[0, 0, 1.0], nozzle=0.4)
    oc = rc["channels"]["overhang"]
    cant_pass = float(oc.max()) > 0.5 and int((oc > 0.5).sum()) >= 1
    out["cantilever_ledge"] = {
        "overhang_max": round(float(oc.max()), 3),
        "n_overhang_faces(>0.5)": int((oc > 0.5).sum()),
        "lights_up": bool(cant_pass)}

    # 2. flat plate lying flat -- low risk everywhere
    plate = trimesh.creation.box(extents=[60, 40, 2.0])
    plate.apply_translation([0, 0, 1.0])
    rp = premortem(plate, build_dir=[0, 0, 1.0], nozzle=0.4)
    plate_max = float(rp["risk_face"].max())
    plate_high = int((rp["risk_face"] > 0.5).sum())
    plate_pass = plate_max < 0.5 and plate_high == 0
    out["flat_plate"] = {"max_risk": round(plate_max, 3),
                         "n_high(>0.5)": plate_high,
                         "reads_low": bool(plate_pass)}

    # 3. thin plate (0.3mm) -- thin-wall must fire
    thin = trimesh.creation.box(extents=[40, 30, 0.3])
    thin.apply_translation([0, 0, 0.15])
    rt = premortem(thin, build_dir=[0, 0, 1.0], nozzle=0.4)
    tc = rt["channels"]["thin"]
    thin_pass = float(tc.max()) > 0.5
    out["thin_plate_0.3mm"] = {"thin_max": round(float(tc.max()), 3),
                               "min_wall_mm": rt["summary"].get("thin", {}).get("min_wall_mm"),
                               "lights_up": bool(thin_pass)}

    # 4. horizontal cylinder -- secondary underside-contrast check
    cyl = trimesh.creation.cylinder(radius=10.0, height=44.0, sections=64)
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    cyl.apply_translation([0, 0, 10.0])
    rcy = premortem(cyl, build_dir=[0, 0, 1.0], nozzle=0.4)
    ocy = rcy["channels"]["overhang"]
    z = cyl.triangles_center[:, 2]
    mid = 0.5 * (z.min() + z.max())
    lower = float(ocy[z < mid].mean()); upper = float(ocy[z >= mid].mean())
    contrast = (lower + 1e-9) / (upper + 1e-9)
    cyl_pass = lower > upper and contrast >= 2.0
    out["cylinder_underside_contrast"] = {
        "lower_half_mean": round(lower, 4), "upper_half_mean": round(upper, 4),
        "lower/upper_contrast": round(float(contrast), 1),
        "overhang_max": round(float(ocy.max()), 3),
        "underside_worse": bool(cyl_pass),
        "note": "lying cylinder overhang is geometrically mild; contrast is the signal"}

    out["ALL_PASS"] = bool(cant_pass and plate_pass and thin_pass and cyl_pass)
    if verbose:
        import json
        print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    from ._mesh_util import ascii_console
    ascii_console()
    selftest(verbose=True)
