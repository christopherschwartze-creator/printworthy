"""Manifold-guarantee step (manifold3d merge), extracted so nothing in
printworthy imports the old product's forge.repair pipeline.

Verbatim extraction of forge/repair.py::_manifold_guarantee — the function
is self-contained (numpy + trimesh + RepairAction only; it calls no other
repair.py helper). Do not edit the numerics. (Only the metrics note string
was ASCII-ified for cp1252 consoles.)
"""
from __future__ import annotations

import numpy as np
import trimesh

from ._types import RepairAction


def _manifold_guarantee(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, RepairAction]:
    """Final Pass 3 guarantee step: manifold3d.Merge over the result.

    Skipped (no-op + warning action) if manifold3d isn't installed.
    """
    try:
        import manifold3d  # type: ignore
    except ImportError:
        return mesh, RepairAction(
            action="manifold_guarantee",
            tool="missing:manifold3d",
            metrics={"note": "skipped -- manifold3d optional dep not installed"},
        )

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    n_v_before, n_f_before = int(len(verts)), int(len(faces))
    try:
        # manifold3d rejects non-manifold inputs and returns an empty
        # Manifold rather than trying to repair. We must NOT propagate
        # that empty mesh as our output — Pass 2 hasn't run yet and the
        # input may still have residual non-manifold edges. Keep the
        # original mesh when manifold3d would empty it.
        mgl = manifold3d.Mesh(vert_properties=verts, tri_verts=faces)
        man = manifold3d.Manifold(mgl)
        out_mesh = man.to_mesh()
        new_verts = np.asarray(out_mesh.vert_properties)[:, :3]
        new_faces = np.asarray(out_mesh.tri_verts, dtype=np.int64)
        if new_verts.shape[0] == 0 or new_faces.shape[0] == 0:
            return mesh, RepairAction(
                action="manifold_guarantee",
                tool="manifold3d.Manifold:rejected_non_manifold_input",
                metrics={
                    "vertices_before": n_v_before,
                    "faces_before": n_f_before,
                    "note": "input not manifold; preserved unmodified. "
                            "Resolve via Pass 2 ops before this step.",
                },
            )
        result = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
        return result, RepairAction(
            action="manifold_guarantee",
            tool="manifold3d.Manifold",
            metrics={
                "vertices_before": n_v_before,
                "vertices_after": int(len(new_verts)),
                "faces_before": n_f_before,
                "faces_after": int(len(new_faces)),
            },
        )
    except Exception as e:
        return mesh, RepairAction(
            action="manifold_guarantee",
            tool="manifold3d:failed",
            metrics={"error": f"{type(e).__name__}: {e}",
                     "vertices_before": n_v_before, "faces_before": n_f_before},
        )
