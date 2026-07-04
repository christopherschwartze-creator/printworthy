"""Bed-fit check + CoACD bed-split (minimal implementation of a documented stub).

Public interface (stable):

    split_for_bed(mesh, profile, *, connectors=True, out_dir=None)
        -> {"ok": bool, "parts": [ {...}, ... ], "note": str}

Implemented (the CHEAP path only):
  * fit test  — axis-aligned bounding box vs. bed volume, best axis
    PERMUTATION (sorted-extents comparison). Honest label: this is a
    heuristic — no free-rotation packing search, no diagonal placement.
  * split     — if the mesh does not fit and `coacd` is importable
    (optional extra ``meshprep[split]``), run an approximate convex
    decomposition and report the parts (with per-part fit). Parts are NOT
    joined back: connector pegs/dovetails are planned, not implemented.
  * degrade   — if `coacd` is missing, return an honest stub note instead
    of raising. This module never installs anything.

Never-raise discipline: any internal failure degrades to
``{"ok": False, "parts": [], "note": "split failed: ..."}``.
"""
from __future__ import annotations

import os

# Generic FDM fallback bed (mm) when the profile carries no bed dimensions.
_GENERIC_BED_MM = (220.0, 220.0, 250.0)

# Compute safety (16 GB shared box): decompose on a decimated proxy.
_MAX_FACES = 6000


def _bed_mm(profile):
    """Extract (x, y, z) bed dimensions in mm from whatever `profile` is.

    Accepts a profile NAME (resolved via meshprep.profiles, lazily), a dict,
    an object with attributes, or a bare 2/3-sequence of mm. Returns
    ``((x, y, z), label)``; falls back to the generic bed, never raises.
    """
    label = "profile"
    try:
        if isinstance(profile, str):
            label = profile
            try:
                from meshprep.profiles import get_profile  # product layer, lazy
                profile = get_profile(profile)
            except Exception:
                return _GENERIC_BED_MM, f"{label} (profiles unavailable -> generic bed)"
        if profile is None:
            return _GENERIC_BED_MM, "generic bed (no profile)"
        if isinstance(profile, (tuple, list)) and len(profile) in (2, 3):
            dims = [float(v) for v in profile]
            if len(dims) == 2:
                dims.append(_GENERIC_BED_MM[2])
            return tuple(dims), "explicit bed"

        def pick(container, key):
            if isinstance(container, dict):
                return container.get(key)
            return getattr(container, key, None)

        for key in ("bed_mm", "bed_size_mm", "bed"):
            v = pick(profile, key)
            if v is not None:
                dims = [float(x) for x in v]
                if len(dims) == 2:
                    dims.append(_GENERIC_BED_MM[2])
                return tuple(dims[:3]), pick(profile, "name") or label
        xyz = [pick(profile, k) for k in ("bed_x", "bed_y", "bed_z")]
        if all(v is not None for v in xyz):
            return tuple(float(v) for v in xyz), pick(profile, "name") or label
    except Exception:
        pass
    return _GENERIC_BED_MM, "generic bed (profile unreadable)"


def _fits(extents, bed):
    """Axis-permutation AABB fit: sorted extents <= sorted bed, elementwise.
    Exact for axis-aligned placements; a heuristic overall (no free rotation)."""
    return all(e <= b + 1e-9 for e, b in zip(sorted(extents), sorted(bed)))


def _load(mesh):
    """Accept a trimesh.Trimesh (anything with .faces/.vertices) or a path."""
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        return mesh
    import trimesh  # lazy (heavy)
    m = trimesh.load(str(mesh), force="mesh")
    if not hasattr(m, "faces"):
        raise ValueError(f"not a triangle mesh: {mesh}")
    return m


def split_for_bed(mesh, profile=None, *, connectors=True, out_dir=None):
    """Check bed fit; if oversized, CoACD-decompose into printable parts.

    Parameters
    ----------
    mesh       : trimesh.Trimesh or path to a mesh file.
    profile    : printer profile (name / dict / object / (x,y,z) mm); the bed
                 dimensions are read from it (generic 220x220x250 fallback).
    connectors : requested joining connectors. NOT implemented yet — when a
                 split happens the note says "connectors: planned".
    out_dir    : if given and a split happens, each part is saved there as
                 ``part_000.stl`` etc. (best-effort).

    Returns ``{"ok", "parts", "note"}``. ``parts`` entries:
    ``{"index", "n_faces", "extents_mm", "fits_bed"[, "file"]}``.
    """
    try:
        try:
            m = _load(mesh)
        except Exception:
            # unreadable / truncated / non-mesh file -> plain refusal, not a
            # leaked Python exception string (foolproof dim 4: novice-readable).
            return {"ok": False, "parts": [],
                    "note": "could not read a triangle mesh from the input -- "
                            "not a valid mesh file"}
        # 0-face / point-cloud / non-mesh loads yield extents == None; guard up
        # front so we never do [float(v) for v in None] (was leaked to the user
        # as "split failed: 'NoneType' object is not iterable").
        if (getattr(m, "faces", None) is None or len(m.faces) == 0
                or getattr(m, "extents", None) is None):
            return {"ok": False, "parts": [],
                    "note": "input is not a triangle mesh (no faces) -- "
                            "cannot check bed fit"}
        bed, bed_label = _bed_mm(profile)
        ext = [float(v) for v in m.extents]
        if not all(v == v and abs(v) != float("inf") for v in ext):
            # NaN/inf bounding box: a fit comparison would be meaningless.
            return {"ok": False, "parts": [],
                    "note": "mesh bounding box is not finite (NaN/inf "
                            "coordinates) -- cannot check bed fit"}
        ext_s = "x".join(f"{v:.1f}" for v in ext)
        bed_s = "x".join(f"{v:.0f}" for v in bed)

        if _fits(ext, bed):
            return {
                "ok": True,
                "parts": [{"index": 0, "n_faces": int(len(m.faces)),
                           "extents_mm": ext, "fits_bed": True}],
                "note": (f"no split needed: {ext_s} mm fits {bed_s} mm bed "
                         f"({bed_label}; AABB-permutation fit, heuristic)"),
            }

        try:
            import coacd  # optional extra: meshprep[split]
        except ImportError:
            return {
                "ok": False,
                "parts": [],
                "note": (f"mesh {ext_s} mm exceeds {bed_s} mm bed ({bed_label}) "
                         "and CoACD is not installed - install the extra "
                         "`meshprep[split]` to enable bed-splitting"),
            }

        import numpy as np  # lazy
        from .core._mesh_util import decimate
        proxy = decimate(m, max_faces=_MAX_FACES)   # compute cap; never raises
        try:
            coacd.set_log_level("error")
        except Exception:
            pass
        raw = coacd.run_coacd(
            coacd.Mesh(np.asarray(proxy.vertices, dtype=np.float64),
                       np.asarray(proxy.faces, dtype=np.int64)),
            threshold=0.05,
        )

        parts, saved = [], 0
        for i, (pv, pf) in enumerate(raw):
            pv = np.asarray(pv, dtype=float)
            pext = [float(v) for v in (pv.max(axis=0) - pv.min(axis=0))]
            entry = {"index": i, "n_faces": int(len(pf)),
                     "extents_mm": pext, "fits_bed": _fits(pext, bed)}
            if out_dir is not None:
                try:
                    import trimesh
                    os.makedirs(out_dir, exist_ok=True)
                    fp = os.path.join(str(out_dir), f"part_{i:03d}.stl")
                    trimesh.Trimesh(vertices=pv, faces=np.asarray(pf)).export(fp)
                    entry["file"] = fp
                    saved += 1
                except Exception:
                    pass
            parts.append(entry)

        n_fit = sum(p["fits_bed"] for p in parts)
        bits = [f"mesh {ext_s} mm exceeds {bed_s} mm bed ({bed_label}); "
                f"CoACD split into {len(parts)} convex parts, {n_fit} fit the bed"]
        if len(proxy.faces) < len(m.faces):
            bits.append(f"decomposed on a {len(proxy.faces)}-face proxy (compute cap)")
        if connectors:
            bits.append("connectors: planned (parts are loose, not joined)")
        if saved:
            bits.append(f"{saved} part STLs saved to {out_dir}")
        return {"ok": bool(parts) and n_fit == len(parts),
                "parts": parts, "note": "; ".join(bits)}
    except Exception as e:  # never-raise discipline
        return {"ok": False, "parts": [], "note": f"split failed: {e}"}
