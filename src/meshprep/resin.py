"""Resin (SLA/MSLA) mode: per-layer island triage + traps + hollow suggestion.

Two public entry points, both never-raise:

`islands_per_layer(mesh, layer_h_mm=0.05, build_dir=None)`
    Voxel-slice the part along `build_dir` and count NEW unsupported ISLANDS
    per layer: an in-plane connected region with no solid within one voxel
    below its footprint (the one-voxel dilation grants a ~45-degree overhang
    growth allowance at the grid pitch). The bottom-most occupied slice is
    plate contact, hence supported. Every island cures loose in the vat unless
    a support touches it first -- the #1 resin failure after adhesion.

`resin_report(mesh, profile)`
    The resin-mode bundle: islands + resin traps (cups / sealed cavities via
    `core._print_traps.find_traps`) + a hollowing suggestion with resin-saved %
    (via `core._print3d.hollow_shell`) + minimum feature size vs the printer's
    LCD pixel + bed fit. Each channel degrades independently (never crashes).

HONESTY (read literally):
  * Island detection is GEOMETRIC voxel triage at a coarse pitch (grid capped
    at ~96 cells, so one voxel layer spans MANY 0.05 mm print layers), not a
    Lychee/Chitubox-grade per-layer analysis. It finds where islands appear
    and how many, not sub-pitch ones; a feature thinner than the pitch can be
    missed entirely. The requested `layer_h_mm` is reported for context only.
  * Traps / hollowing inherit the documented gaps of `_print_traps` and
    `_print3d.hollow_shell` (voxel drainage, not a fluid sim; vent SUGGESTED,
    not cut). The hollow threshold (>= 25 % saved) is a heuristic label.
  * min-feature uses the Shape Diameter min-wall estimate; "printable" means
    >= 2 LCD pixels (rule of thumb, uncalibrated).

Compute-capped for the 16 GB box: meshes decimated to <= 6000 faces, voxel
grids <= 96 cells along the longest axis, single-threaded numerics.

Run as a script for the self-tests:  python -m meshprep.resin
"""
from __future__ import annotations

RES_MIN, RES_MAX, DEFAULT_RES = 32, 96, 80


def _unit(v, np):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def _degenerate_geometry(mesh):
    """Cheap, never-raise degeneracy test that gates the ray/voxel-heavy
    channels. Returns True when the mesh has no printable extent -- so a
    channel that would build an embree ray caster or a voxel grid over it can
    be honestly SKIPPED instead of hanging.

    True if: no faces; non-finite vertices; a zero/negative or non-finite
    bounding box (all-coincident geometry); or a face count so large that the
    <=6000 compute-cap decimation clearly no-op'd -- decimate() cannot reduce
    fully-coincident geometry, so a 560k coincident-triangle mesh reaches the
    ray caster unbounded and hangs (F3). All checks are O(#vertices) and never
    voxelize or cast a ray, so this itself cannot hang."""
    try:
        import numpy as np
        if mesh is None or not hasattr(mesh, "faces") or len(mesh.faces) == 0:
            return True
        v = np.asarray(getattr(mesh, "vertices", None), float)
        if v.size == 0 or not np.all(np.isfinite(v)):
            return True
        e = getattr(mesh, "extents", None)
        if e is None:
            return True
        ext = np.asarray(e, float)
        if ext.size == 0 or not np.all(np.isfinite(ext)) or float(ext.max()) <= 1e-9:
            return True
        # decimate() no-ops on coincident geometry -> face count stays huge;
        # never feed that to the embree ray caster (documented hang).
        if len(mesh.faces) > 50000:
            return True
        return False
    except Exception:
        return True


# ---------------------------------------------------------------------------
#  islands per layer
# ---------------------------------------------------------------------------
def islands_per_layer(mesh, layer_h_mm=0.05, build_dir=None, res=None,
                      max_report=8, verbose=False):
    """Count NEW unsupported islands per voxel layer along `build_dir` (default
    +Z). Returns a dict (never raises):

      ok, pitch_mm, voxel_res, n_voxel_layers, layer_h_mm (requested, context
      only), layers_per_voxel_layer, total_new_islands, n_layers_with_islands,
      first_island_z_mm (world z of the first island, None if clean),
      worst_layers: [{layer, z_mm, new_islands}] (top `max_report` by count),
      method, notes.

    A layer-0 (bottom-most occupied slice) component is plate contact ->
    supported. Support test = overlap with the PREVIOUS slice dilated by one
    voxel (~45-degree growth allowance at the pitch)."""
    out = {"ok": False, "layer_h_mm": float(layer_h_mm),
           "total_new_islands": 0, "n_layers_with_islands": 0,
           "first_island_z_mm": None, "worst_layers": [], "notes": []}
    try:
        import numpy as np
        import trimesh
        from scipy import ndimage
        from .core._mesh_util import decimate, say

        if mesh is None or not hasattr(mesh, "faces") or len(mesh.faces) == 0:
            out["notes"].append("empty mesh")
            return out

        m = mesh.copy()
        bd = _unit(build_dir if build_dir is not None else [0.0, 0.0, 1.0], np)
        if abs(float(bd @ [0, 0, 1])) < 1.0 - 1e-9:      # rotate build_dir -> +Z
            m.apply_transform(trimesh.geometry.align_vectors(bd, [0, 0, 1]))
        elif float(bd[2]) < 0:
            m.apply_transform(trimesh.transformations.rotation_matrix(
                np.pi, [1, 0, 0]))
        m = decimate(m, 6000)

        r = int(res) if res else DEFAULT_RES
        r = max(RES_MIN, min(RES_MAX, r))
        ext = float(np.asarray(m.bounding_box.extents).max())
        if not np.isfinite(ext) or ext <= 0:
            out["notes"].append("degenerate bbox")
            return out
        pitch = ext / r
        vox = m.voxelized(pitch=pitch)
        try:
            vox = vox.fill()                             # interior counts as support
        except Exception:
            out["notes"].append("voxel fill failed; using surface occupancy")
        mat = np.asarray(vox.matrix, bool)
        z0 = float(vox.transform[2, 3])                  # world z of index 0
        nz = int(mat.shape[2])

        out.update({"pitch_mm": round(pitch, 4), "voxel_res": r,
                    "n_voxel_layers": nz,
                    "layers_per_voxel_layer": round(pitch / max(layer_h_mm, 1e-6), 1)})

        struct8 = ndimage.generate_binary_structure(2, 2)
        occupied = [k for k in range(nz) if mat[:, :, k].any()]
        if not occupied:
            out["notes"].append("empty voxelization")
            return out
        k0 = occupied[0]                                 # plate-contact slice

        layers, total = [], 0
        prev_dil = None
        for k in range(k0, nz):
            sl = mat[:, :, k]
            if not sl.any():
                prev_dil = None                          # gap: nothing supports above
                continue
            new = 0
            if k > k0:
                lab, n = ndimage.label(sl, structure=struct8)
                for i in range(1, n + 1):
                    comp = lab == i
                    if prev_dil is None or not np.logical_and(comp, prev_dil).any():
                        new += 1
            if new:
                total += new
                layers.append({"layer": int(k - k0),
                               "z_mm": round(z0 + k * pitch, 2),  # voxel center
                               "new_islands": int(new)})
            prev_dil = ndimage.binary_dilation(sl, structure=struct8)

        layers_sorted = sorted(layers, key=lambda d: -d["new_islands"])
        out.update({
            "ok": True,
            "total_new_islands": int(total),
            "n_layers_with_islands": len(layers),
            "first_island_z_mm": layers[0]["z_mm"] if layers else None,
            "worst_layers": layers_sorted[:int(max_report)],
            "method": ("geometric voxel-slice island triage at pitch "
                       f"{round(pitch, 3)} mm (~{out['layers_per_voxel_layer']} print "
                       "layers per voxel layer); one-voxel growth allowance; "
                       "NOT slicer-grade"),
        })
        if verbose:
            say(f"  islands: {total} new island(s) across {len(layers)} layer(s), "
                f"first at z={out['first_island_z_mm']}mm "
                f"@ res {r} pitch {round(pitch, 3)}mm")
    except Exception as e:                               # never-raise discipline
        out["notes"].append(f"islands_per_layer failed: {e}")
    return out


# ---------------------------------------------------------------------------
#  resin-mode report bundle
# ---------------------------------------------------------------------------
def resin_report(mesh, profile=None, build_dir=None, hollow_wall_mm=2.0,
                 res=None, verbose=False):
    """Resin-mode triage bundle. Returns a dict (never raises):

      ok, profile, build_dir,
      fit      : profiles.scale_to_fit result,
      islands  : islands_per_layer result,
      traps    : core.find_traps result (+ hollowed-cavity vent suggestion when
                 hollowing is suggested),
      hollow   : hollow_shell info + suggested flag (>= 25 % saved heuristic),
      min_feature : {min_wall_mm, pixel_mm, min_printable_mm, below_pixel},
      flags    : plain-English warning strings,
      notes    : per-channel degradation reasons.

    Every channel is independently guarded: a failing channel leaves a note and
    an ok=False sub-dict, the report itself always comes back."""
    out = {"ok": False, "flags": [], "notes": []}
    try:
        import numpy as np  # noqa: F401
        from .core._mesh_util import decimate, say
        from .profiles import get_profile, scale_to_fit

        prof = get_profile(profile)
        out["profile"] = prof["name"]
        out["technology"] = prof.get("technology")
        if prof.get("technology") != "resin":
            out["notes"].append(
                f"profile '{prof['name']}' is {prof.get('technology')}, not resin; "
                "running resin analysis anyway")
        bd = [0.0, 0.0, 1.0] if build_dir is None else list(map(float, build_dir))
        out["build_dir"] = bd

        if mesh is None or not hasattr(mesh, "faces") or len(mesh.faces) == 0:
            out["notes"].append("empty mesh")
            return out
        mdec = decimate(mesh.copy(), 6000)

        # Degenerate (zero-extent / non-finite / coincident) input reaches the
        # ray/voxel channels unbounded (decimate cannot cap coincident geometry)
        # and would HANG the report. Detect once, cheaply, and skip those
        # channels honestly rather than hanging (F3). The cheap fit check below
        # still runs.
        degenerate = _degenerate_geometry(mdec)
        if degenerate:
            out["notes"].append(
                "degenerate geometry (zero extent / non-finite / coincident "
                "vertices, or uncapped face count) -- ray/voxel feature checks "
                "skipped to avoid a hang")
            out["flags"].append(
                "degenerate geometry -- this mesh has no printable volume (all "
                "points coincident, zero-size, or non-finite); fix the mesh "
                "before a resin check is meaningful")

        # -- fit (cheap, bbox only) --
        out["fit"] = scale_to_fit(mesh, prof)
        if out["fit"].get("fits") is False:
            out["flags"].append(
                f"does not fit the {prof['name']} build volume -- scale by "
                f"{out['fit'].get('suggested_scale')} "
                f"(orientation {out['fit'].get('best_permutation')})")

        # -- islands --
        try:
            out["islands"] = islands_per_layer(
                mdec, layer_h_mm=prof.get("layer_mm", 0.05),
                build_dir=bd, res=res, verbose=verbose)
        except Exception as e:      # islands_per_layer never raises; belt+braces
            out["islands"] = {"ok": False, "notes": [str(e)]}
        isl = out["islands"]
        if isl.get("total_new_islands", 0) > 0:
            out["flags"].append(
                f"{isl['total_new_islands']} unsupported island(s), first at "
                f"z={isl['first_island_z_mm']} mm -- add supports or reorient "
                "(voxel triage, not slicer-grade)")
        elif not isl.get("ok"):
            out["notes"].append("island channel degraded")

        # -- hollow suggestion (needs watertight) --
        hollow = {"ok": False, "suggested": False}
        try:
            if bool(getattr(mdec, "is_watertight", False)):
                from .core import _print3d
                _shell, info = _print3d.hollow_shell(mdec, wall_mm=float(hollow_wall_mm))
                hollow = dict(info)
                hollow["suggested"] = bool(
                    info.get("ok") and info.get("material_saved_pct", 0.0) >= 25.0)
                hollow["criterion"] = ">=25% resin saved (heuristic threshold)"
                if hollow["suggested"]:
                    out["flags"].append(
                        f"hollowing to a {hollow_wall_mm} mm wall saves "
                        f"~{info['material_saved_pct']}% resin -- REQUIRES a drain "
                        "hole (see traps.vents for a suggested location)")
            else:
                hollow["reason"] = "mesh not watertight (run the fixer first)"
                out["notes"].append("hollow channel skipped: not watertight")
        except Exception as e:
            hollow = {"ok": False, "suggested": False, "reason": str(e)}
            out["notes"].append("hollow channel degraded")
        out["hollow"] = hollow

        # -- traps (cups + sealed cavities; include the would-be hollowed cavity
        #    so the drain suggestion lands in the same report) --
        try:
            if degenerate:
                out["traps"] = {"ok": False,
                                "notes": ["degenerate geometry -- trap check skipped"]}
                out["notes"].append("traps channel skipped: degenerate geometry")
            else:
                from .core import _print_traps
                out["traps"] = _print_traps.find_traps(
                    mdec, bd,
                    hollow_wall_mm=float(hollow_wall_mm) if hollow.get("suggested") else None,
                    verbose=verbose)
        except Exception as e:
            out["traps"] = {"ok": False, "notes": [str(e)]}
            out["notes"].append("traps channel degraded")
        tr = out["traps"]
        n_cups = len(tr.get("external_cups", []))
        n_seal = len([c for c in tr.get("internal_cavities", [])
                      if c.get("kind") == "sealed"])
        if n_cups:
            out["flags"].append(
                f"{n_cups} resin cup(s) pool {tr.get('external_cup_cm3')} cm3 in this "
                "orientation -- tilt/reorient or add the suggested vent(s)")
        if n_seal:
            out["flags"].append(
                f"{n_seal} SEALED internal cavity(ies) -- uncured resin stays inside; "
                "drill the suggested vent(s)")

        # -- min feature vs LCD pixel --
        # PRIMARY F3 fix: min_wall_thickness builds an embree RayMeshIntersector
        # and casts thousands of rays; on coincident zero-area / nan-normal
        # triangles it neither raises nor returns (>20 s isolated), hanging the
        # whole report past the wall. Skip it honestly on degenerate input.
        feat = {"ok": False}
        if degenerate:
            feat = {"ok": False,
                    "reason": "degenerate geometry -- feature check skipped"}
            out["notes"].append("min-feature channel skipped: degenerate geometry")
        else:
            try:
                from .core import _print3d
                mw = _print3d.min_wall_thickness(mdec)
                px = float(prof.get("pixel_mm", prof.get("nozzle_mm", 0.05)))
                mw_ok = mw == mw and mw not in (float("inf"),)   # finite
                feat = {"ok": True,
                        "min_wall_mm": round(float(mw), 3) if mw_ok else None,
                        "pixel_mm": px,
                        "min_printable_mm": round(2 * px, 3),
                        "below_pixel": bool(mw < 2 * px) if mw_ok else None,
                        "method": "Shape Diameter min-wall vs 2-pixel rule of thumb "
                                  "(uncalibrated)"}
                if feat["below_pixel"]:
                    out["flags"].append(
                        f"thinnest feature ~{feat['min_wall_mm']} mm is under 2 LCD "
                        f"pixels ({feat['min_printable_mm']} mm) -- may not resolve")
            except Exception as e:
                feat = {"ok": False, "reason": str(e)}
                out["notes"].append("min-feature channel degraded")
        out["min_feature"] = feat

        out["ok"] = True
        out["method"] = ("resin-mode geometric triage bundle; every channel "
                         "labelled, none slicer- or fluid-sim-grade")
        if verbose:
            say(f"  resin report [{prof['name']}]: {len(out['flags'])} flag(s)")
            for f in out["flags"]:
                say(f"    - {f}")
    except Exception as e:                               # never-raise discipline
        out["notes"].append(f"resin_report failed: {e}")
    return out


# ---------------------------------------------------------------------------
#  self-tests
# ---------------------------------------------------------------------------
_PSB_DIR = r"C:\Users\mecht\Project_EI\Applied\MeshDecomp\datasets\psb\MeshsegBenchmark-1.0\data\off"


def _load_psb(idx, size_mm=60.0):
    """Test-only PSB loader (returns None if the dataset is not on this box)."""
    import os
    import trimesh
    import numpy as np
    path = os.path.join(os.environ.get("MESHPREP_PSB", _PSB_DIR), f"{idx}.off")
    if not os.path.isfile(path):
        return None
    m = trimesh.load(path, force="mesh", process=True)
    m.apply_translation(-m.bounding_box.centroid)
    m.apply_scale(size_mm / float(np.max(m.bounding_box.extents)))
    return m


def _make_floating_sphere_scene():
    """Plate on the bed + a sphere floating ABOVE it (gap) -- the sphere's first
    layer has nothing below it, so it MUST read as an island."""
    import trimesh
    plate = trimesh.creation.box(extents=(40, 40, 2))
    plate.apply_translation([0, 0, 1])                   # sits on z=0
    sphere = trimesh.creation.icosphere(radius=6, subdivisions=3)
    sphere.apply_translation([0, 0, 15])                 # bottom at z=9, gap 2..9
    return trimesh.util.concatenate([plate, sphere])


def run_self_tests(verbose=True):
    """Brief-required tests: a primitive with a KNOWN island + a real PSB mesh.
    Returns (all_pass, details)."""
    from .core._mesh_util import say
    import trimesh
    details = {}
    ok_all = True

    # 1. floating sphere above a plate -> >=1 island, first at z ~= 9 mm
    scene = _make_floating_sphere_scene()
    r1 = islands_per_layer(scene, build_dir=[0, 0, 1])
    z1 = r1.get("first_island_z_mm")
    pitch = r1.get("pitch_mm", 0.3)
    t1 = (r1["ok"] and r1["total_new_islands"] >= 1
          and z1 is not None and abs(z1 - 9.0) <= 2 * pitch + 0.5)
    details["test1_floating_sphere"] = {
        "pass": bool(t1), "islands": r1.get("total_new_islands"),
        "first_z_mm": z1, "pitch_mm": pitch}
    ok_all &= t1

    # 2. solid box alone -> 0 islands (control: exercises the exact failure
    #    mode of over-flagging plate-supported geometry)
    box = trimesh.creation.box(extents=(20, 20, 30))
    r2 = islands_per_layer(box, build_dir=[0, 0, 1])
    t2 = r2["ok"] and r2["total_new_islands"] == 0
    details["test2_box_clean"] = {"pass": bool(t2),
                                  "islands": r2.get("total_new_islands")}
    ok_all &= t2

    # 3. never-raise on garbage
    r3 = islands_per_layer(None)
    r3b = resin_report(None)
    t3 = (r3["ok"] is False and r3["notes"]
          and r3b["ok"] is False and r3b["notes"])
    details["test3_never_raise"] = bool(t3)
    ok_all &= bool(t3)

    # 4. PSB mesh end-to-end resin_report (skipped-with-note if dataset absent)
    teddy = _load_psb(161)
    if teddy is None:
        details["test4_psb_teddy"] = "SKIP (PSB dataset not found)"
        t4 = True
    else:
        r4 = resin_report(teddy, "generic_resin_msla", verbose=False)
        t4 = (r4["ok"] and r4["islands"].get("ok")
              and r4["fit"].get("ok") and "hollow" in r4 and "traps" in r4)
        details["test4_psb_teddy"] = {
            "pass": bool(t4), "flags": r4.get("flags"),
            "islands": r4["islands"].get("total_new_islands"),
            "hollow_saved_pct": r4["hollow"].get("material_saved_pct"),
            "n_traps": r4["traps"].get("n_traps"),
            "min_wall_mm": r4["min_feature"].get("min_wall_mm")}
    ok_all &= t4

    if verbose:
        say("RESIN SELF-TESTS")
        say(f"  1. floating sphere -> {r1.get('total_new_islands')} island(s), "
            f"first z={z1}mm (want ~9)  -> {'PASS' if t1 else 'FAIL'}")
        say(f"  2. solid box -> {r2.get('total_new_islands')} islands (want 0)  "
            f"-> {'PASS' if t2 else 'FAIL'}")
        say(f"  3. never-raise on None -> {'PASS' if t3 else 'FAIL'}")
        say(f"  4. PSB teddy resin_report -> "
            f"{details['test4_psb_teddy'] if isinstance(details['test4_psb_teddy'], str) else ('PASS' if t4 else 'FAIL')}")
        if isinstance(details["test4_psb_teddy"], dict):
            d = details["test4_psb_teddy"]
            say(f"     islands={d['islands']} hollow_saved={d['hollow_saved_pct']}% "
                f"traps={d['n_traps']} min_wall={d['min_wall_mm']}mm")
            for f in d["flags"] or []:
                say(f"     flag: {f}")
        say(f"  ALL: {'PASS' if ok_all else 'FAIL'}")
    return bool(ok_all), details


if __name__ == "__main__":
    import sys as _sys
    from .core._mesh_util import ascii_console
    ascii_console()
    _ok, _d = run_self_tests(verbose=True)
    _sys.exit(0 if _ok else 1)
