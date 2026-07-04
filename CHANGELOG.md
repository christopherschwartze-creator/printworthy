# Changelog

All notable changes to **meshprep** are recorded here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow SemVer.

## [0.1.0.dev0] — unreleased (staging)

First open-source perimeter. The tool code is built and validated in the development
monorepo; this is the clean, permissive, publishable wrapper around it.

### Added
- **`preflight`** — printability PASS/WARN/FAIL verdict with ranked plain-English issues +
  a free **source-accurate repair** (curvature-continuous hole fill; ~100 % of the surface
  kept verbatim, deviation certificate, never-worse rollback). Optional inherent-strain
  **warp FEM** channel.
- **`reinforce`** — mesh + load case → **graded-infill 3MF** (FEM load-path dense core as a
  PrusaSlicer-schema-correct modifier volume; multi-tier nesting; Orca/Bambu sibling config).
- **Print-FEM core** (`scikit-fem` + `pyamg`): inherent-strain warp (Timoshenko-validated) +
  orthotropic strength / orient-for-strength + a one-coupon **warp calibration**.
- **Product layer (integration)** — ONE entry `meshprep.prep()` (`pipeline.py`): load → guard
  → analyze → fix → orient → strengthen → bed-fit → slicer savings → package (PrepResult dict);
  every stage never-raise/degrade, honest labels carried through. Unified plain-English
  **report** (`report.md` + `report.json`), **CLI** (`meshprep check|fix|prep|reinforce|
  calibrate|app`), **Gradio front door** (`meshprep app`, `[app]` extra), printer **profiles**
  + bed-fit advisor, **resin** mode (islands/traps/hollow), **slicer** savings wrapper,
  **batch** folder mode, CoACD bed-**split**, and the (non-hardened) FastAPI farm-endpoint
  **stub** (`[api]` extra).
- **`license_guard.py`** — AST scan proving zero copyleft / non-commercial imports.
- Clean MIT perimeter: `LICENSE`, `README`, `THIRD_PARTY_NOTICES`, `pyproject`, this file.

### Notes
- **`autorig` (animation build) ships separately** — not bundled in meshprep; the report says so.
- Permissive-only (MIT/BSD/Apache/MPL-2); GPL/CGAL/non-commercial deliberately unreachable.
- `pyQuadriFlow` is an **LGPL-flagged opt-in extra only** — rebuild from source; not bundled.
- Geometric/FEM, not ML; physics predictions are **uncalibrated estimates** unless a coupon
  is fitted. Per-tool **in-tool validation** (slicer / Blender / printed coupon) is pending.
- Distribution name is a **placeholder** pending PyPI + trademark clearance.

### Remaining before publish
See `RELEASE_MANIFEST.md` → "Remaining mechanical steps" (vendor the listed modules, break the
`forge.repair` reach, rewrite imports, isolated-install test) and "Conscious caveats" (name,
copyright, the MIT-forfeits-patent trade).
