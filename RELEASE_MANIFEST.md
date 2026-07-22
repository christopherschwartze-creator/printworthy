# printworthy — Release Manifest

The clean OSS perimeter (LICENSE, README, THIRD_PARTY_NOTICES, pyproject, license_guard,
CHANGELOG) is complete in this directory. This manifest specifies **exactly which source
modules vendor into the package** (and which must be kept out), plus the remaining mechanical
steps and the conscious caveats. Source lives in the development monorepo at
`C:\Users\mecht\Project_EI\3DEI\Forge`.

## Target layout (after vendoring)
```
printworthy/
├── LICENSE  README.md  THIRD_PARTY_NOTICES.md  pyproject.toml  license_guard.py  CHANGELOG.md
└── src/printworthy/
    ├── preflight/     ← Forge/preflight/        (checker + source-accurate fixer)
    ├── reinforce/     ← Forge/reinforce/        (graded-infill 3MF)
    ├── autorig/       ← Forge/autorig/          (mesh -> posable glTF rig)
    └── core/          ← the reused Forge/scripts + a few Forge/{forge,ei_core} modules
```

## IN — vendor these (the import closure of the three tools)

**Tools (whole packages):** `preflight/`, `reinforce/`, `autorig/`.

**Print + FEM (`Forge/scripts/`):** `_print_premortem.py`, `_print_traps.py`, `_print3d.py`,
`_print_advanced.py`, `_print_fem.py`, `fem_warp_probe.py`, `fem_orthotropic.py`,
`fem_materials.py`, `_warp_calibrate.py`, `_fix_accurate.py`.

**Retopology (`Forge/scripts/`):** `quad_remesh.py`, `_seamless.py`, `_mcf2.py`,
`_im_extract.py`, `_field_quad.py`, `_blossom.py`, `_blossom_remesh.py`, `_quadflow.py`,
`_quadflow_eval.py`. Optional (LGPL-flagged extra): `_pyqf_backend.py`.

**Rig (`Forge/scripts/`):** `_rigready.py`, `_deform_score.py`, `_hybrid_cut.py`, `_genus.py`.

**Reused from the old product (vendor ONLY these files, not the packages):**
`Forge/forge/oversized/force_solid.py`, `Forge/forge/pass3/hole_fill_ei.py`,
`Forge/ei_core/quality.py` (σ*/ρ), `Forge/ei_core/_util.py`.
(NOT vendored after all: `ei_core/self_intersect.py` — the scout traced zero consumers in
the print pipeline's import closure; it stays in Forge with the animation build, which uses
it via `_deform_score`.)

## OUT — must NOT ship (landmines + bloat)

- **`Forge/sf3d_repo/`** — Stability AI **Community License (non-commercial, registration,
  attribution)**. *The_ critical exclusion. It must never be in a published MIT artifact.
- **`Forge/ei_api/`** — FastAPI REST surface (carries the unaddressed SSRF / malicious-mesh /
  rate-limit / billing blockers). The OSS suite is library+CLI only, not a hosted service.
- **`Forge/ei_cli/`, `Forge/ei_glb/`** — old-product CLI / GLB writer; not used by the tools.
- **The rest of `Forge/forge/`** (the repair pipeline, oversized modes, report, health, …) and
  **the rest of `Forge/ei_core/`** — only the five files above are reused.
- **`viz/`, `cloud/`, `corpus/`, `audit/`, `instructions/`, `results/`**, the `CODE_REVIEW_*`
  / `BRIEF_*` / `RAPID_RESUME` dev docs.
- **Experimental / superseded scripts** (NOT in the IN list): e.g. `_aniso_strength.py` and
  `_lightweight.py` (v0, retired), the `_diag_*`, `_tune_*`, `_sweep_*`, old per-part grid
  (`_grid_*`), `_acd_*`, etc.
- **The old product's optional extras** — `classical` (libigl / gpytoolbox) and `appearance`
  (torch / transformers / lpips). The suite does not import them.

## Remaining mechanical steps (the vendor pass — a separate, low-risk chunk)

1. **Break the `forge.repair` reach.** `_fix_accurate.py` calls
   `forge.repair._manifold_guarantee` and the σ-rollback veto, and `force_solid.py` imports
   `..repair._manifold_guarantee`. Extract just `_manifold_guarantee` (+ its tiny helpers)
   into `core/_manifold.py` so the package does **not** pull the whole repair pipeline. (~1 h.)
2. **Rewrite imports.** The dev modules use `sys.path.insert` + bare `import _print_fem`. In
   the package, convert to intra-package imports (`from printworthy.core import _print_fem`), or
   keep a single `core/__init__.py` that puts `core/` on the path. Mechanical, test after.
3. **Isolated install test.** `pip install -e .` in a clean venv → run each tool's
   `selftest` + the three CLIs on a sample mesh; confirm no import reaches outside `printworthy`.
4. **Re-run `license_guard.py src/`** on the vendored tree (must stay CLEAN).

## Conscious caveats (decide before `twine upload`)

- **Name `printworthy` is a placeholder** — `pip index versions printworthy` / check PyPI, and run a
  trademark clearance (avoid the `forge`-was-taken + Autodesk-Forge problems). A distinctive
  coined mark is safer than a generic one.
- **Copyright holder** — `LICENSE` and `pyproject` say *Christopher Schwartze*; set your exact
  legal name/entity.
- **MIT publication forfeits the patent option.** Open-sourcing the σ*/ρ + FEM code starts the
  clock and gives away any provisional. That is the deliberate trade for the "free credibility"
  path (chosen over the paper/patent route). If you ever want to protect σ*/ρ, do **not**
  publish this first.
- **Per-tool in-tool validation is still pending** (slicer print, Blender pose, warp coupon) —
  the README states this honestly; don't market past "schema-correct / re-loads / poses under
  our LBS — confirm in-tool".

## Premium exclusions

The real implementations behind three roadmap stubs in this repo — FEM warp pre-compensation,
the bureau/B2B print-quote engine (`quote.py`), and a QuadriFlow-backed retopo-pro tier — live
in a **separate, proprietary package** (`meshprep_pro`, its own private git repository, not this
one). The stubs in this public repo stay exactly as documentation of the build plan; they are
not implemented here and this repo does not depend on `meshprep_pro`. `meshprep_pro` is not
published anywhere as of this note.
