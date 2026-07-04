"""Shared types vendored for meshprep.core.

`RepairAction` — one row of the repair provenance log (verbatim from the
old product's forge/types.py; RepairState and ForgeReport had no consumer
in this closure and were trimmed).
`FrameStatus` — per-vertex frame-conditioning category used by quality.py
(verbatim from ei_core/types.py; the other ei_core types were trimmed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


@dataclass
class RepairAction:
    """One row of the provenance log.

    Logged by every Pass 1 / Pass 2 / Pass 3 op as it runs. Used by
    Phase 3 corpus-test and by per-mesh debugging on customer pipelines.

    `action`   : short identifier (e.g. "fill_holes", "polish_ei_coherence").
    `tool`     : library / function actually invoked.
    `severity` : 0–1 derived from how far the trigger metric exceeded
                 threshold. Used to order ops within the same defect class.
    `metrics`  : free-form dict; conventionally records before/after
                 counts ("input_holes", "output_holes"), affected vertex
                 counts, σ_min shifts. Schema is intentionally loose so
                 ops can record what's specifically meaningful for them.
    """

    action: str
    tool: str
    severity: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)


class FrameStatus(IntEnum):
    """Per-vertex frame-conditioning state.

    OK              — full 3-D frame (valence ≥ 3 unique-direction edges)
                      σ* is the SVD-σ_min of the 3×k unit-edge matrix.
                      For ρ fields, ≥ 2 incident faces with an aggregate.
    RANK_DEFICIENT  — valence = 2 (σ field) or all incident edges collapse
                      to a plane. σ* uses the 2-D PCA closed form
                      √(1 − |cos θ|). Meaningful but rank-2 in 3-D.
    BOUNDARY_ONLY   — for ρ fields only (CODE_REVIEW H2R-2): exactly one
                      incident face, so the per-vertex aggregate has no
                      pair to compare or normalise. Value is conventionally
                      0 but the cell carries no information; consumers
                      should mask out from p10/p90 distributions.
    UNDEFINED       — valence ≤ 1 / no incident face. σ* = 1.0 placeholder.
    """

    OK = 0
    RANK_DEFICIENT = 1
    BOUNDARY_ONLY = 3       # new code per H2R-2; appended so existing
                            # OK / RANK_DEFICIENT / UNDEFINED values
                            # remain stable in serialized topo.json sidecars.
    UNDEFINED = 2
