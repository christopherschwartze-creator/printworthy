"""threemf_writer.py -- a PERMISSIVE, hand-rolled OPC/3MF writer that emits a
SLICER-READY 3MF with per-volume MODIFIER infill overrides.

WHY HAND-ROLLED (license + control)
====================================
A 3MF is an OPC zip: [Content_Types].xml + _rels/.rels + 3D/3dmodel.model, plus
(for a slicer project) the per-volume settings under Metadata/. The lightest
PERMISSIVE path that can write multiple objects/volumes AND the slicer config is
pure stdlib `zipfile` + hand-written XML -- ZERO new dependencies, fully under our
own (permissive) copyright, and total control over the documented slicer schema.
lib3mf (BSD-2) and trimesh's 3MF (MIT) are both permissive but neither writes the
PrusaSlicer Slic3r_PE_model.config modifier/infill metadata, so we'd hand-write
that file regardless; doing the whole archive by hand keeps it dependency-free.

THE SCHEMA WE EMIT (verified against PrusaSlicer source: src/libslic3r/Format/3mf.cpp
and src/libslic3r/Model.cpp on github.com/prusa3d/PrusaSlicer)
=====================================================================================
PrusaSlicer represents an object as ONE <object> in 3D/3dmodel.model whose <mesh>
is the CONCATENATION of all its volumes' triangles. Each volume is the contiguous
triangle index range [firstid, lastid] into that single mesh. The per-volume
settings (incl. which range is a MODIFIER and its infill override) live in
Metadata/Slic3r_PE_model.config:

    <?xml version="1.0" encoding="UTF-8"?>
    <config>
     <object id="1" instances_count="1">
      <metadata type="object" key="name" value="part"/>
      <volume firstid="0" lastid="11">
       <metadata type="volume" key="name" value="base"/>
       <metadata type="volume" key="volume_type" value="ModelPart"/>
      </volume>
      <volume firstid="12" lastid="…">
       <metadata type="volume" key="name" value="dense_core"/>
       <metadata type="volume" key="modifier" value="1"/>
       <metadata type="volume" key="volume_type" value="ParameterModifier"/>
       <metadata type="volume" key="fill_density" value="80%"/>
      </volume>
     </object>
    </config>

Exact constants taken verbatim from 3mf.cpp:
  OBJECT_TAG="object" VOLUME_TAG="volume" METADATA_TAG="metadata" CONFIG_TAG="config"
  ID_ATTR="id" TYPE_ATTR="type" KEY_ATTR="key" VALUE_ATTR="value"
  FIRST_TRIANGLE_ID_ATTR="firstid" LAST_TRIANGLE_ID_ATTR="lastid"
  INSTANCESCOUNT_ATTR="instances_count"  (3mf.cpp:139; importer ignores it)
  MODIFIER_KEY="modifier"  (value "1")          -> importer sets PARAMETER_MODIFIER
  VOLUME_TYPE_KEY="volume_type"                 -> value via ModelVolume::type_to_string
  type_to_string(PARAMETER_MODIFIER) == "ParameterModifier"  (Model.cpp)
  type_to_string(MODEL_PART)         == "ModelPart"
The per-volume infill override is written by 3mf.cpp as
  <metadata type="volume" key="<opt>" value="<config.opt_serialize(opt)>"/>
and `fill_density` is a ConfigOptionPercent, so opt_serialize yields a trailing
'%', i.e. value="80%". The importer applies any unrecognised volume metadata key
straight into volume->config, so key="fill_density" value="80%" becomes the
per-modifier infill density PrusaSlicer slices with. We emit BOTH the modifier=1
flag AND volume_type=ParameterModifier (the importer accepts either) for maximum
slicer-version compatibility.

ORCA/BAMBU VARIANT (noted, not the v1 target): OrcaSlicer/BambuStudio store the
same modifier-volume GEOMETRY but a different config under Metadata/model_settings.config
using <part ... subtype="modifier_part"> and a different infill key
(sparse_infill_density). See orca_config_xml() for a best-effort sibling config we
also drop in so the archive degrades gracefully there; PrusaSlicer remains the
verified v1 target.

LICENSE: this writer is original code (permissive). No GPL/AGPL dependency.
"""
from __future__ import annotations

import io
import zipfile
from typing import Sequence

import numpy as np

# ---- verbatim PrusaSlicer 3mf.cpp constants (the contract) ----------------- #
CONFIG_TAG = "config"
OBJECT_TAG = "object"
VOLUME_TAG = "volume"
METADATA_TAG = "metadata"
ID_ATTR = "id"
TYPE_ATTR = "type"
KEY_ATTR = "key"
VALUE_ATTR = "value"
FIRST_TRIANGLE_ID_ATTR = "firstid"
LAST_TRIANGLE_ID_ATTR = "lastid"
INSTANCESCOUNT_ATTR = "instances_count"  # 3mf.cpp:139 (importer ignores it; decorative)
OBJECT_TYPE = "object"          # the type= attr value for object-level metadata
VOLUME_TYPE = "volume"          # the type= attr value for volume-level metadata
MODIFIER_KEY = "modifier"
VOLUME_TYPE_KEY = "volume_type"
NAME_KEY = "name"
# ModelVolume::type_to_string (Model.cpp)
VT_MODEL_PART = "ModelPart"
VT_PARAMETER_MODIFIER = "ParameterModifier"
# verbatim from PrusaSlicer Model.cpp::type_to_string (source recon, CamelCase
# SINGULAR -- see support_mods.py for the schema evidence and the CLI proof
# that these two values are honored by PrusaSlicer's support generator).
VT_SUPPORT_ENFORCER = "SupportEnforcer"
VT_SUPPORT_BLOCKER = "SupportBlocker"
# the OPC part paths PrusaSlicer uses
MODEL_FILE = "3D/3dmodel.model"
PRUSA_CONFIG_FILE = "Metadata/Slic3r_PE_model.config"
ORCA_CONFIG_FILE = "Metadata/model_settings.config"


def _xml_escape(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


class Volume:
    """One volume = a triangle range inside the parent object's single mesh.

    is_modifier  : True -> PARAMETER_MODIFIER (the slicer applies `settings`
                   only to the model region this volume's geometry encloses).
    settings     : dict of {config_key: serialized_value}. For an infill override
                   pass {"fill_density": "80%"} (value MUST be the opt_serialize
                   form, i.e. percent options carry a trailing '%').
    volume_type  : None (default) -> behavior unchanged, type is derived from
                   is_modifier exactly as before. Or an explicit, source-verified
                   ModelVolumeType STRING (see VT_* constants above), e.g.
                   VT_SUPPORT_ENFORCER / VT_SUPPORT_BLOCKER. Enforcer/blocker
                   volumes are NOT ParameterModifiers, so passing volume_type
                   suppresses the modifier=1 flag (see _prusa_config_xml).
    """

    def __init__(self, vertices, faces, name="volume", is_modifier=False,
                 settings=None, volume_type=None):
        self.vertices = np.asarray(vertices, float)
        self.faces = np.asarray(faces, np.int64)
        self.name = name
        self.is_modifier = bool(is_modifier)
        self.settings = dict(settings or {})
        self.volume_type = volume_type


def _model_xml(objects: Sequence) -> str:
    """3D/3dmodel.model: one <object> per part; its <mesh> is the concatenation
    of all its volumes' triangles (this is exactly how PrusaSlicer stores
    multi-volume objects, so the firstid/lastid ranges in the config line up)."""
    out = ['<?xml version="1.0" encoding="UTF-8"?>']
    out.append('<model unit="millimeter" xml:lang="en-US" '
               'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
               'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">')
    out.append(' <metadata name="Application">EI-Forge-reinforce</metadata>')
    out.append(' <resources>')
    for obj in objects:
        out.append(f'  <object id="{obj["id"]}" type="model">')
        out.append('   <mesh>')
        out.append('    <vertices>')
        for v in obj["vertices"]:
            out.append(f'     <vertex x="{v[0]:.6f}" y="{v[1]:.6f}" z="{v[2]:.6f}"/>')
        out.append('    </vertices>')
        out.append('    <triangles>')
        for t in obj["faces"]:
            out.append(f'     <triangle v1="{int(t[0])}" v2="{int(t[1])}" v3="{int(t[2])}"/>')
        out.append('    </triangles>')
        out.append('   </mesh>')
        out.append('  </object>')
    out.append(' </resources>')
    out.append(' <build>')
    for obj in objects:
        out.append(f'  <item objectid="{obj["id"]}"/>')
    out.append(' </build>')
    out.append('</model>')
    return "\n".join(out) + "\n"


def _meta(indent, type_value, key, value) -> str:
    return (f'{" " * indent}<{METADATA_TAG} {TYPE_ATTR}="{type_value}" '
            f'{KEY_ATTR}="{_xml_escape(key)}" '
            f'{VALUE_ATTR}="{_xml_escape(value)}"/>')


def _prusa_config_xml(objects: Sequence) -> str:
    """Metadata/Slic3r_PE_model.config -- the per-volume modifier + infill schema,
    matching 3mf.cpp's _add_model_config_file_to_archive exactly."""
    out = ['<?xml version="1.0" encoding="UTF-8"?>', f'<{CONFIG_TAG}>']
    for obj in objects:
        out.append(f' <{OBJECT_TAG} {ID_ATTR}="{obj["id"]}" '
                   f'{INSTANCESCOUNT_ATTR}="1">')
        out.append(_meta(2, OBJECT_TYPE, NAME_KEY, obj["name"]))
        for vol in obj["volumes"]:
            out.append(f'  <{VOLUME_TAG} '
                       f'{FIRST_TRIANGLE_ID_ATTR}="{vol["firstid"]}" '
                       f'{LAST_TRIANGLE_ID_ATTR}="{vol["lastid"]}">')
            out.append(_meta(3, VOLUME_TYPE, NAME_KEY, vol["name"]))
            vt = vol.get("volume_type")
            if vt is not None:
                # explicit source-verified type (SupportEnforcer/SupportBlocker/
                # etc.); enforcer/blocker are NOT ParameterModifiers, so do NOT
                # emit modifier=1 for them.
                out.append(_meta(3, VOLUME_TYPE, VOLUME_TYPE_KEY, vt))
            elif vol["is_modifier"]:
                # emit BOTH flags (importer accepts either; cross-version safe)
                out.append(_meta(3, VOLUME_TYPE, MODIFIER_KEY, "1"))
                out.append(_meta(3, VOLUME_TYPE, VOLUME_TYPE_KEY,
                                 VT_PARAMETER_MODIFIER))
            else:
                out.append(_meta(3, VOLUME_TYPE, VOLUME_TYPE_KEY, VT_MODEL_PART))
            for k, v in vol["settings"].items():
                out.append(_meta(3, VOLUME_TYPE, k, v))
            out.append(f'  </{VOLUME_TAG}>')
        out.append(f' </{OBJECT_TAG}>')
    out.append(f'</{CONFIG_TAG}>')
    return "\n".join(out) + "\n"


# OrcaSlicer / BambuStudio key for infill density (sparse_infill_density).
_ORCA_INFILL_KEY = "sparse_infill_density"


def _orca_config_xml(objects: Sequence) -> str:
    """Best-effort Metadata/model_settings.config for Orca/Bambu. Same modifier
    GEOMETRY; Orca marks the volume <part ... subtype="modifier_part"> and reads
    sparse_infill_density. This is a COMPATIBILITY sibling, not the verified
    target -- PrusaSlicer (above) is what we validate against."""
    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<config>']
    for obj in objects:
        out.append(f' <object id="{obj["id"]}">')
        out.append(f'  <metadata key="name" value="{_xml_escape(obj["name"])}"/>')
        for vol in obj["volumes"]:
            subtype = "modifier_part" if vol["is_modifier"] else "normal_part"
            out.append(f'  <part id="{vol["part_id"]}" subtype="{subtype}" '
                       f'firstid="{vol["firstid"]}" lastid="{vol["lastid"]}">')
            out.append(f'   <metadata key="name" value="{_xml_escape(vol["name"])}"/>')
            # translate fill_density -> sparse_infill_density (strip the % for Orca's
            # value convention; Orca stores the bare number)
            fd = vol["settings"].get("fill_density")
            if fd is not None:
                out.append(f'   <metadata key="{_ORCA_INFILL_KEY}" '
                           f'value="{_xml_escape(str(fd).rstrip("%"))}"/>')
            out.append('  </part>')
        out.append(' </object>')
    out.append('</config>')
    return "\n".join(out) + "\n"


_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
    ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
    ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
    '</Types>\n'
)

_RELS = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
    ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" '
    'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
    '</Relationships>\n'
)


def write_3mf(parts: Sequence, out_path: str, *, include_orca=True) -> str:
    """Write a slicer-ready 3MF.

    parts : list of dicts, one per printed object, each:
        {"name": str, "volumes": [Volume, ...]}  -- the FIRST volume should be the
        base ModelPart; subsequent modifier Volumes carry the infill overrides.
        All volumes of a part are concatenated into that object's single mesh, and
        each volume's [firstid,lastid] triangle range is recorded in the config.

    Returns out_path. Raises only on bad input (caller wraps).
    """
    model_objects = []   # for 3dmodel.model
    cfg_objects = []      # for the slicer config
    next_obj_id = 1
    for part in parts:
        vol_list = part["volumes"]
        all_v = []
        all_f = []
        cfg_vols = []
        tri_cursor = 0
        v_offset = 0
        part_id = 1
        for vol in vol_list:
            nv = len(vol.vertices)
            nf = len(vol.faces)
            all_v.append(vol.vertices)
            all_f.append(vol.faces + v_offset)
            firstid = tri_cursor
            lastid = tri_cursor + nf - 1
            cfg_vols.append({
                "name": vol.name,
                "firstid": firstid,
                "lastid": lastid,
                "is_modifier": vol.is_modifier,
                "settings": vol.settings,
                "part_id": part_id,
                "volume_type": vol.volume_type,
            })
            tri_cursor += nf
            v_offset += nv
            part_id += 1
        V = np.vstack(all_v) if all_v else np.zeros((0, 3))
        F = np.vstack(all_f) if all_f else np.zeros((0, 3), np.int64)
        model_objects.append({"id": next_obj_id, "vertices": V, "faces": F})
        cfg_objects.append({"id": next_obj_id, "name": part["name"],
                            "volumes": cfg_vols})
        next_obj_id += 1

    model_xml = _model_xml(model_objects)
    prusa_xml = _prusa_config_xml(cfg_objects)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr(MODEL_FILE, model_xml)
        z.writestr(PRUSA_CONFIG_FILE, prusa_xml)
        if include_orca:
            z.writestr(ORCA_CONFIG_FILE, _orca_config_xml(cfg_objects))
    with open(out_path, "wb") as fh:
        fh.write(buf.getvalue())
    return out_path
