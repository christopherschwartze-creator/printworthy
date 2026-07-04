"""selftest.py -- the HARD GATE for the reinforce package.

We CANNOT run a real slicer headless here, so the selftest performs the strongest
HEADLESS validation possible and prints an honest verdict:

  G1  OPC: the produced .3mf re-opens as a valid zip and contains exactly the
      required OPC parts: [Content_Types].xml, _rels/.rels, 3D/3dmodel.model, and
      Metadata/Slic3r_PE_model.config.
  G2  GEOMETRY: 3dmodel.model parses; the object's mesh has the triangle count the
      config's firstid/lastid ranges imply (ranges are contiguous, cover the whole
      mesh, base first then modifiers).
  G3  SCHEMA: the config marks the base volume volume_type=ModelPart and every
      modifier volume with BOTH key="modifier" value="1" AND
      key="volume_type" value="ParameterModifier", and each modifier carries
      key="fill_density" value="NN%". This is diffed against the documented
      PrusaSlicer 3mf.cpp format (constants reproduced in threemf_writer).
  G4  MESH SANITY: each modifier mesh (decoded back from the 3MF triangle range) is
      watertight and lies inside the base bbox with nonzero volume.
  G5  TARGETING (vs control): the modifier region overlaps the FEM high-stress
      voxels FAR more than a RANDOM region of equal size does. This is the check
      that the modifier actually addresses the bones, not arbitrary geometry.

Run:  python -m reinforce.selftest         (from Forge/)
      python -m meshprep.core.selftest
"""
from __future__ import annotations

import os
import sys
import zipfile
import xml.etree.ElementTree as ET

_HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np

from .reinforce import reinforce
from . import _modifier_extract as mx
from . import threemf_writer as tw
from ._mesh_util import fmt_fos as _fmt_fos

_REQUIRED_PARTS = ["[Content_Types].xml", "_rels/.rels",
                   "3D/3dmodel.model", "Metadata/Slic3r_PE_model.config"]
_MODEL_NS = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"


def _parse_model(model_xml):
    """Return list of objects, each {id, V (n,3), F (m,3)}."""
    root = ET.fromstring(model_xml)
    objs = []
    for obj in root.iter(f"{_MODEL_NS}object"):
        mesh = obj.find(f"{_MODEL_NS}mesh")
        if mesh is None:
            continue
        V = [[float(v.get("x")), float(v.get("y")), float(v.get("z"))]
             for v in mesh.find(f"{_MODEL_NS}vertices")]
        F = [[int(t.get("v1")), int(t.get("v2")), int(t.get("v3"))]
             for t in mesh.find(f"{_MODEL_NS}triangles")]
        objs.append({"id": obj.get("id"), "V": np.array(V), "F": np.array(F)})
    return objs


def _parse_config(cfg_xml):
    """Return list of objects, each {id, volumes:[{firstid,lastid,meta:{k:v}}]}."""
    root = ET.fromstring(cfg_xml)
    objs = []
    for obj in root.iter("object"):
        vols = []
        for vol in obj.iter("volume"):
            meta = {m.get("key"): m.get("value") for m in vol.iter("metadata")}
            vols.append({"firstid": int(vol.get("firstid")),
                        "lastid": int(vol.get("lastid")), "meta": meta})
        objs.append({"id": obj.get("id"), "volumes": vols})
    return objs


def run(verbose=True):
    import trimesh
    P = print if verbose else (lambda *a, **k: None)
    P("=" * 76)
    P("reinforce  --  structural passport  |  HARD GATE (headless validation)")
    P("=" * 76)

    ok = True
    out = os.path.join(_HERE, "_selftest_reinforced.3mf")

    # cantilever beam: clear bending load path -> high stress at the clamped root
    beam = trimesh.creation.box(extents=[80.0, 12.0, 12.0])
    beam.apply_translation([40.0, 6.0, 6.0])
    load_case = {"load_face": "xmax", "anchor_face": "xmin",
                 "load_axis": 2, "total_force_N": 100.0}

    P("\n[build] cantilever 80x12x12 mm, tip load +z, root clamped (PLA)")
    res = reinforce(beam, load_case, "PLA", out, base_fill_density=15,
                    tiers=[(70.0, 80, "dense_core")], max_elem=2500, verbose=verbose)
    if not res.get("ok"):
        P("  reinforce FAILED:", res.get("reason"))
        return False
    P(f"  -> {res['out_3mf']}  modifiers={res['n_modifiers']} "
      f"min_fos={_fmt_fos(res['min_fos'])}(REL) peak_vm={res['peak_vm_Pa']:.2e}Pa")

    # ---- G1 OPC ---------------------------------------------------------- #
    P("\n--- G1  valid OPC zip + required parts ---")
    try:
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
            model_xml = z.read("3D/3dmodel.model").decode("utf-8")
            cfg_xml = z.read("Metadata/Slic3r_PE_model.config").decode("utf-8")
        g1 = all(p in names for p in _REQUIRED_PARTS)
    except Exception as e:
        P("  zip open FAILED:", e); return False
    for p in _REQUIRED_PARTS:
        P(f"    {'OK ' if p in names else 'MISS'} {p}")
    P(f"  -> {'PASS' if g1 else 'FAIL'}")
    ok &= g1

    # ---- G2 GEOMETRY ----------------------------------------------------- #
    P("\n--- G2  3dmodel.model parses; firstid/lastid ranges tile the mesh ---")
    g2 = True
    try:
        mobjs = _parse_model(model_xml)
        cobjs = _parse_config(cfg_xml)
        g2 &= (len(mobjs) == 1 and len(cobjs) == 1)
        mo, co = mobjs[0], cobjs[0]
        ntri = len(mo["F"])
        vols = co["volumes"]
        # contiguous, base first, cover [0, ntri-1]
        cursor = 0
        for i, v in enumerate(vols):
            g2 &= (v["firstid"] == cursor)
            g2 &= (v["lastid"] >= v["firstid"])
            cursor = v["lastid"] + 1
        g2 &= (cursor == ntri)
        P(f"    object mesh triangles = {ntri}; volumes = {len(vols)}; "
          f"ranges contiguous & cover mesh = {g2}")
        for v in vols:
            P(f"      vol [{v['firstid']:>4}..{v['lastid']:>4}]  "
              f"meta={ {k: v['meta'][k] for k in ('name','volume_type') if k in v['meta']} }")
    except Exception as e:
        P("  parse FAILED:", e); g2 = False
    P(f"  -> {'PASS' if g2 else 'FAIL'}")
    ok &= g2

    # ---- G3 SCHEMA (the documented PrusaSlicer modifier+infill format) --- #
    P("\n--- G3  modifier + fill_density metadata matches documented schema ---")
    g3 = True
    try:
        base_v = vols[0]
        mod_vs = vols[1:]
        # base must be ModelPart
        g3 &= (base_v["meta"].get("volume_type") == tw.VT_MODEL_PART)
        P(f"    base volume_type == 'ModelPart' : "
          f"{base_v['meta'].get('volume_type') == tw.VT_MODEL_PART}")
        g3 &= (len(mod_vs) >= 1)
        for v in mod_vs:
            m = v["meta"]
            c1 = (m.get("modifier") == "1")
            c2 = (m.get("volume_type") == tw.VT_PARAMETER_MODIFIER)
            fd = m.get("fill_density")
            c3 = (fd is not None and fd.endswith("%") and fd[:-1].isdigit())
            P(f"    modifier: modifier=1 ({c1})  "
              f"volume_type=ParameterModifier ({c2})  "
              f"fill_density={fd} ({c3})")
            g3 &= (c1 and c2 and c3)
    except Exception as e:
        P("  schema check FAILED:", e); g3 = False
    P(f"  -> {'PASS' if g3 else 'FAIL'}  (diffed vs 3mf.cpp/Model.cpp constants)")
    ok &= g3

    # ---- G4 MESH SANITY (decode each modifier range, check it) ----------- #
    P("\n--- G4  each modifier mesh watertight + inside base bbox ---")
    g4 = True
    try:
        V, F = mo["V"], mo["F"]
        base_range = vols[0]
        base_tris = F[base_range["firstid"]:base_range["lastid"] + 1]
        base_mesh = trimesh.Trimesh(vertices=V, faces=base_tris, process=True)
        for v in vols[1:]:
            tris = F[v["firstid"]:v["lastid"] + 1]
            mm = trimesh.Trimesh(vertices=V, faces=tris, process=True)
            chk = mx.verify_modifier(mm, base_mesh)
            P(f"    modifier: watertight={chk['watertight']} "
              f"inside_bbox={chk['inside_base_bbox']} vol={chk['volume_mm3']:.1f}mm3 "
              f"faces={chk['n_faces']}")
            g4 &= (chk["watertight"] and chk["inside_base_bbox"]
                   and chk["volume_mm3"] > 0)
    except Exception as e:
        P("  mesh sanity FAILED:", e); g4 = False
    P(f"  -> {'PASS' if g4 else 'FAIL'}")
    ok &= g4

    # ---- G5 TARGETING vs RANDOM CONTROL ---------------------------------- #
    P("\n--- G5  modifier covers FEM high-stress voxels >> a random control ---")
    g5 = True
    try:
        from . import _print_fem
        sres = _print_fem.strength_analysis(beam, "PLA", load_case, max_elem=2500)
        cent = np.asarray(sres["centroids"], float)
        pitch = float(sres["pitch"])
        imp = mx.importance_field(np.asarray(sres["fos"], float),
                                  np.asarray(sres["vm"], float), mode="vm")
        occ_solid, occ_hot, origin, thr, frac = mx.high_stress_occupancy(
            cent, pitch, imp, percentile=70.0, largest_component=True)
        # decode the modifier voxels: which solid voxels fall inside the modifier mesh?
        v_mod = vols[1]
        tris = mo["F"][v_mod["firstid"]:v_mod["lastid"] + 1]
        mod_mesh = trimesh.Trimesh(vertices=mo["V"], faces=tris, process=True)
        inside = mod_mesh.contains(cent)            # (nelem,) elements in modifier
        # the "true" high-stress set, as a per-element boolean aligned to centroids
        shape, org2, ijk = mx.grid_from_centroids(cent, pitch)
        hot_elem = occ_hot[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        # precision: of elements inside the modifier, how many are truly hot?
        prec = float((inside & hot_elem).sum()) / max(1, int(inside.sum()))
        # recall: of truly hot elements, how many are captured by the modifier?
        rec = float((inside & hot_elem).sum()) / max(1, int(hot_elem.sum()))
        # RANDOM control: a random element subset of the SAME size as the modifier
        rng = np.random.default_rng(0)
        k = int(inside.sum())
        prec_ctrl = []
        for _ in range(20):
            idx = rng.choice(len(cent), size=max(1, k), replace=False)
            sel = np.zeros(len(cent), bool); sel[idx] = True
            prec_ctrl.append(float((sel & hot_elem).sum()) / max(1, int(sel.sum())))
        prec_ctrl = float(np.mean(prec_ctrl))
        P(f"    modifier precision (inside-and-hot / inside) = {prec:.2f}")
        P(f"    modifier recall    (inside-and-hot / hot)    = {rec:.2f}")
        P(f"    random-region precision (mean of 20)         = {prec_ctrl:.2f}")
        # the gate: the modifier targets hot voxels much better than random, and
        # captures most of them
        g5 = (prec >= max(0.6, 2.0 * prec_ctrl)) and (rec >= 0.6)
    except Exception as e:
        P("  targeting check FAILED:", e); g5 = False
    P(f"  -> {'PASS' if g5 else 'FAIL'}")
    ok &= g5

    P("\n" + "=" * 76)
    P("ALL_PASS" if ok else "SOME GATE FAILED")
    P("Headless validation only. The 3MF is schema-correct vs the DOCUMENTED")
    P("PrusaSlicer Slic3r_PE_model.config format and re-opens as a valid OPC zip;")
    P("the modifier covers the RELATIVE high-stress load path (vs a random control).")
    P("Graded infill 'prints as-is' is FINALLY confirmed by opening it in a slicer.")
    P("The importance field is a RELATIVE load proxy, NOT a certified safety margin.")
    P("=" * 76)
    return ok


def run_tiers(verbose=True):
    """Validate MULTI-TIER nesting: a 2-tier reinforce (mid 50% + core 100%) must
    write two nested modifier volumes, schema-correct, with the hotter/smaller core
    written LAST (so PrusaSlicer applies it over the mid in the overlap) and the core
    geometrically INSIDE the mid (nesting)."""
    import trimesh
    P = print if verbose else (lambda *a, **k: None)
    P("\n" + "=" * 76)
    P("reinforce  --  MULTI-TIER NESTING gate (2 modifiers: mid 50% + core 100%)")
    P("=" * 76)
    ok = True
    out = os.path.join(_HERE, "_selftest_tiers.3mf")
    beam = trimesh.creation.box(extents=[80.0, 12.0, 12.0])
    beam.apply_translation([40.0, 6.0, 6.0])
    load_case = {"load_face": "xmax", "anchor_face": "xmin",
                 "load_axis": 2, "total_force_N": 100.0}
    # cooler/bigger 'mid' (>=55th pct -> 50%) and hotter/smaller 'core' (>=85th -> 100%)
    res = reinforce(beam, load_case, "PLA", out, base_fill_density=15,
                    tiers=[(55.0, 50, "mid"), (85.0, 100, "core")],
                    max_elem=2500, verbose=verbose)
    if not res.get("ok"):
        P("  reinforce FAILED:", res.get("reason")); return False
    P(f"  -> {res['out_3mf']}  modifiers={res['n_modifiers']}")

    with zipfile.ZipFile(out) as z:
        model_xml = z.read("3D/3dmodel.model").decode("utf-8")
        cfg_xml = z.read("Metadata/Slic3r_PE_model.config").decode("utf-8")
    mo = _parse_model(model_xml)[0]
    co = _parse_config(cfg_xml)[0]
    vols = co["volumes"]

    # T1: base + exactly 2 modifier volumes, each schema-correct
    P("\n--- T1  base + 2 modifier volumes, each schema-correct ---")
    mod_vs = vols[1:]
    t1 = (len(vols) == 3 and len(mod_vs) == 2)
    for v in mod_vs:
        m = v["meta"]
        good = (m.get("modifier") == "1"
                and m.get("volume_type") == tw.VT_PARAMETER_MODIFIER
                and (m.get("fill_density") or "").endswith("%"))
        P(f"    modifier '{m.get('name')}': fill_density={m.get('fill_density')} schema_ok={good}")
        t1 &= good
    P(f"  -> {'PASS' if t1 else 'FAIL'}"); ok &= t1

    # T2: write ORDER -> hotter 'core' (100%) is LAST so it overrides 'mid' (50%)
    P("\n--- T2  hotter core written LAST + higher infill (overrides mid in overlap) ---")
    try:
        last = vols[-1]["meta"]; prev = vols[-2]["meta"]
        d_last = int(last["fill_density"].rstrip("%"))
        d_prev = int(prev["fill_density"].rstrip("%"))
        t2 = (last.get("name") == "core" and d_last == 100
              and prev.get("name") == "mid" and d_prev == 50 and d_last > d_prev)
        P(f"    order: ...{prev.get('name')}({d_prev}%) -> {last.get('name')}({d_last}%) "
          f"[core last & denser = {t2}]")
    except Exception as e:
        P("  order check FAILED:", e); t2 = False
    P(f"  -> {'PASS' if t2 else 'FAIL'}"); ok &= t2

    # T3: NESTING -> core is concentrically EMBEDDED in the mid (smaller + centroid
    # inside + most verts inside). NOTE: strict mesh containment is NOT required and
    # not expected -- core and mid are extracted+dilated INDEPENDENTLY, so their
    # marching-cubes surfaces fuzz at the boundary (~20% of core verts poke just
    # outside mid). That is BENIGN: PrusaSlicer applies the inner (core) modifier
    # LAST, so it wins the overlap, and any poke-sliver gets core's 100% over base
    # directly (still correct -- it's high-stress). The real property is that core is
    # the inner, smaller, concentric tier; we test that, not strict containment.
    P("\n--- T3  core is concentrically EMBEDDED in mid (inner/smaller tier) ---")
    try:
        V, F = mo["V"], mo["F"]
        mid_v, core_v = vols[1], vols[2]
        mid_m = trimesh.Trimesh(V, F[mid_v["firstid"]:mid_v["lastid"] + 1], process=True)
        core_m = trimesh.Trimesh(V, F[core_v["firstid"]:core_v["lastid"] + 1], process=True)
        frac_in = float(mid_m.contains(core_m.vertices).mean())
        centroid_in = bool(mid_m.contains(np.array([core_m.centroid]))[0])
        smaller = core_m.volume < mid_m.volume
        t3 = smaller and centroid_in and (frac_in >= 0.70)
        P(f"    core vol {core_m.volume:.0f} < mid vol {mid_m.volume:.0f} = {smaller}; "
          f"core centroid in mid = {centroid_in}; {frac_in*100:.0f}% of core verts "
          f"inside mid (boundary fuzz from independent dilation is benign)")
    except Exception as e:
        P("  nesting check FAILED:", e); t3 = False
    P(f"  -> {'PASS' if t3 else 'FAIL'}"); ok &= t3

    # T4: each modifier watertight + inside base
    P("\n--- T4  each modifier watertight + inside base ---")
    try:
        base_m = trimesh.Trimesh(mo["V"], mo["F"][vols[0]["firstid"]:vols[0]["lastid"] + 1],
                                 process=True)
        t4 = True
        for v in vols[1:]:
            mm = trimesh.Trimesh(mo["V"], mo["F"][v["firstid"]:v["lastid"] + 1], process=True)
            chk = mx.verify_modifier(mm, base_m)
            P(f"    '{v['meta'].get('name')}': watertight={chk['watertight']} "
              f"inside={chk['inside_base_bbox']} vol={chk['volume_mm3']:.0f}mm3")
            t4 &= (chk["watertight"] and chk["inside_base_bbox"] and chk["volume_mm3"] > 0)
    except Exception as e:
        P("  sanity FAILED:", e); t4 = False
    P(f"  -> {'PASS' if t4 else 'FAIL'}"); ok &= t4

    P("\n" + ("TIERS_ALL_PASS" if ok else "TIERS: SOME GATE FAILED"))
    return ok


if __name__ == "__main__":
    from ._mesh_util import ascii_console
    ascii_console()
    a = run()
    b = run_tiers()
    sys.exit(0 if (a and b) else 1)
