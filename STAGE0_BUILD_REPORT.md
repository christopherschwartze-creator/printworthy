# Stage-0 Build Report — meshprep

Date: 2026-07-01
Location: `C:\Users\mecht\Project_EI\3DEI\meshprep`
Source of truth for provenance: `RELEASE_MANIFEST.md` (one known discrepancy, see §6.3)
Verdict from the independent verify pass: **ship-ready** (with three minor honesty issues logged in §6).

---

## 1. What the product now IS

One installable package, one pipeline, three front doors. Drop a mesh in, get a print-ready
package plus a plain-English report out.

- **One-call API:** `meshprep.prep(path_or_mesh, profile="ender3", ...)` → `PrepResult` dict
  `{ok, verdict, headline, report{markdown,json,md_path,json_path}, files{...}, renders,
  savings, channels, cert, fix, stages, notes}`. Outer try/except: `prep()` can never raise
  (missing file / `prep(None)` → verdict `REJECTED` with a friendly headline, no exception).
- **CLI:** `meshprep check|fix|prep|reinforce|calibrate|app` (installed console script;
  exit 0 = PASS/WARN, 1 = FAIL, 2 = rejected/error). `--max-faces` gives fast triage
  (teddy: 136 s full → 13 s at 6000 faces, decimation noted in the report).
- **App:** `meshprep app` — Gradio front door (port of `preflight/app.py`) driving the FULL
  pipeline: drop-zone → badge + unified report + heatmap gallery + downloads
  (prep STL / 3MF / report.md/json) + savings line; fem/resin/reinforce/fix/orient toggles.
  Gradio is import-guarded behind the `[app]` extra.

Pipeline stages (in `src/meshprep/pipeline.py`):
`load → guard → analyze (premortem/traps/topology; warp-FEM opt-in; resin_report in resin
mode) → fix (run_fixer + deviation certificate + before→after re-check) → orient
(overhang-minimising, dropped onto bed) → reinforce (graded-infill 3MF, max_elem=2500) →
bed-fit (scale_to_fit) → retopo opt-in (quad OBJ) → slicer savings (degrades to a note) →
package (report.md + report.json + STL/3MF + renders)`.

Exact commands:

```
pip install -e C:\Users\mecht\Project_EI\3DEI\meshprep
meshprep check <mesh> [--max-faces 6000]
meshprep prep  <mesh> --profile ender3
meshprep reinforce <mesh> ...      # delegates to the vendored, schema-verified engine
meshprep calibrate ...             # warp calibration store at ~/.meshprep/
meshprep app                       # requires pip install meshprep[app]
```

Vendored engine bulk: 40 files, 18,451 lines under `src/meshprep/core/`, plus the product
layer (`pipeline.py`, `report.py`, `cli.py`, `app.py`, `profiles.py`, `resin.py`,
`slicer.py`, `batch.py`, `split.py`, `api.py`). The Forge source tree was never modified.

---

## 2. Feature matrix

| Feature | Status | Measured status |
|---|---|---|
| Preflight check (premortem + traps + topology) | **Integrated** (vendored `preflight_core` + print stack) | e2e on PSB teddy-161: verdict WARN (7% overhang), both checker channels ok, renders written |
| Fixer + deviation certificate | **Integrated** | Broken icosphere: "Free fix applied: FAIL → WARN", certificate "100% surface kept, 0.000 mm max deviation, genus 0"; `_fix_accurate` accurate columns bit-identical to pre-port Forge baseline |
| Orientation | **Integrated** | Teddy: overhang area 25.0% → 7.4% after orient |
| Graded-infill reinforce (3MF) | **Integrated** | Gates G1–G5 ALL_PASS (G5 precision 0.69 / recall 0.86 / random control 0.15), tiers T1–T4 PASS (nelem=1862, pitch=2.167 mm, peak_vm=1.08e+01 Pa); output 3MF is a valid OPC zip incl. Slic3r/model_settings metadata |
| FEM (warp probe, orthotropic strength, materials, calibrate) | **Integrated** (opt-in flags) | `_print_fem --quick` ALL_PASS: bar rel_err 8.4e-15, cantilever ratio 0.9799 monotone, Timoshenko eigenstrain converging, warp-trend control, strength anisotropy 0.635≈Zt/Xt; calibration store relocated to `~/.meshprep/warp_calibration.json` (bundled JSON as read-only seed) |
| Quad retopo (mcf2 / blossom / field, permissive default) | **Integrated** (opt-in `retopo=` flag) | Imports clean; pyQuadriFlow strictly behind `[retopo-pyqf]` extra with try/except (license_guard: only the expected 4 opt-in WARNs, guarded in `_pyqf_backend`) |
| Printer profiles + bed-fit | **Built new** (`profiles.py`) | 5/5 selftests PASS: 100 mm cube max_scale 2.18 on ender3; 240 mm bar fits only via z-permutation "yzx"; 300 mm bar suggested_scale 0.83 = exactly 249/300 usable; `None` never raises |
| Resin mode (islands/layer + traps + hollow/drain) | **Built new** (`resin.py`) | 4/4 selftests PASS in 4.5 s: floating-sphere control finds exactly 1 island at z=9.0 mm (true bottom 9 mm); solid box 0 islands (over-flag control); PSB teddy full resin_report: 3 islands, hollow saves 62.2%, 1 trap w/ vent suggestion, min_wall 3.586 mm |
| Slicer savings (PrusaSlicer/Orca CLI wrapper) | **Built new** (`slicer.py`), **end-to-end UNTESTED** | No slicer installed on this box; detection+degrade validated, parser unit-tested 31/31 PASS (PrusaSlicer footer, Orca/Bambu totals, multi-extruder sums, Cura header, garbage→None); `estimate()/compare()` return `ok:False` with friendly notes. First real slice = remaining validation |
| Unified report (md + json) | **Built new** (`report.py`) | One renderer; verdict banner + fix flip, ranked issues, numbers table, provenance with honest labels per channel; numpy-sanitizing JSON |
| Batch folder mode | **Built new** (`batch.py`, thin) | `batch_prep` end-to-end previously blocked on pipeline, now runs through it |
| Bed-split (CoACD) | **STUB** (`split.py`) | Documented interface only |
| Farm API (FastAPI) | **STUB** (`api.py`) | Import-guarded behind `[api]` extra; not a served product; `check_only=True` contract honored by `prep()` |
| Autorig / animation | **EXCLUDED by design** | Ships separately; noted in report provenance and CHANGELOG moved from "Added" to "ships separately" |

Vendor completeness: all 24 mapped scripts modules plus the scout's 3 missed hard deps
(`fem_voxel_core`, `_fem_validation_plan`, `_grid_place`), `curvature_decompose.py` vendored
FLAT (kills the `forge/__init__` → `forge.repair` import reach), `RepairAction`/`FrameStatus`
extracted with dead types trimmed, `_manifold_guarantee` extracted verbatim as
`core/_manifold.py` (confirmed self-contained; nothing imports `forge.repair`),
`warp_calibration.json` pinned by exact name as package data.

---

## 3. Code-review ledger

### Known-debt items (mission list) — all 7 closed
1. **cp1252 UnicodeEncodeError** — `ascii_console()`/`say()` in `core/_mesh_util.py`, used at
   every selftest/CLI entry and in never-raise render paths; FEM gates pass under forced
   cp1252 (A 1.7e-13, B 10.0% monotone, C ratio 0.940). Residual: report.md itself is UTF-8,
   not ASCII (see §6.1).
2. **`_decimate` × 4** — deduped to `_mesh_util.decimate` (handles both trimesh APIs).
3. **`run_fixer` dual fallback** — ONE fallback path (`fa.fix_toy`); `_voxel_reseal` +
   inline cheap-repair deleted (~70 lines).
4. **`_fix_accurate` dead 'exact' color-carry branch** — removed per audit.
5. **min_fos 1e6 sentinel** — `fmt_fos` everywhere; user-facing string is
   `"n/a (load tiny)"` (verified in the reinforce+warp e2e).
6. **Never-raise discipline** — `render_premortem` body now guarded like `render_traps`
   (returns None, still writes PNG on happy path); `prep()` outer guard; channels degrade.
7. **Dead code** — dead imports, dead `_intersector` embree build in `generate_supports`,
   dead `rng`, always-True `show_internal` param, unused unpacks, duplicated
   drop-height/print-height gate factored into `_drop_and_height()`, warp per-edge Python
   loop → `np.maximum.at` (byte-identical selftest).

### Review outcomes per area
- **Print stack** (`_print_premortem/_print_traps/_print3d/_print_advanced`): LOC 2499 → 2500
  (net +1: ~30 lines dead code removed, offset by the never-raise wrapper, a documented-gap
  docstring bullet, and one helper). Post-edit selftests + deterministic smokes
  **byte-identical** to pre-edit baselines (diff empty). The upward-vent capability gap was
  claimed documented but was NOT in the original docstring — now genuinely documented.
- **FEM stack**: calibrate store moved out of site-packages to `~/.meshprep/` (schema
  unchanged, selftest output byte-identical); remaining raw min_fos f-strings wrapped with
  `fmt_fos` (identical numeric string for finite values).

### Deferred (explicitly not done, on purpose)
- `hollowed_cavity(build_dir=...)` unused param kept (interface decision for resin wrapper).
- Actually CLOSING the upward-vent gap (flood-from-vent per orientation) — a feature, not a
  port fix; the stated most-valuable next step for the traps channel.
- `_print_advanced.render_figure/run_validation` still raise on missing matplotlib (no
  never-raise claim there; guard if pipeline adopts them).
- `bridge_analysis`'s networkx dep (already a hard dep via `_blossom`; no action).

**Guardrail honored:** zero changes inside validated numerics (FEM/fixer/3MF algorithms) —
all refactoring was around them, and the byte-identical diffs prove it.

---

## 4. Verification results (independent verify pass)

- **install_clean: true** — `pip install -e` + `meshprep --help` OK; all runs from a cwd
  outside both repos.
- **no_outside_imports: true** — 37/37 modules import in a clean subprocess with only
  `src/` on path; zero `sys.path.insert`, zero forge/ei_core imports, zero hardcoded PSB
  paths; pyflakes zero undefined names; compileall clean.
- **Selftests 4/4 PASS, numbers_match_known_good: true**
  - `_print_traps`: cup 5.926 cm³ / 1 cup, flipped 0, sealed sphere 1.953 cm³ / 1 cavity,
    drilled 0 — **exact match** to pre-port numbers.
  - reinforce `core.selftest`: ALL_PASS + TIERS_ALL_PASS (G1–G5, T1–T4, same numbers).
  - `_fix_accurate`: accurate ≥ toy on 4/4; accurate columns **bit-identical** to Forge
    baseline (dev 0.0098/0.0082/0.0215/0.0074, 100% unchanged, genus 0). The toy `dev`
    column differs in the 3rd decimal — re-running the ORIGINAL varies run-to-run, so it is
    stochastic, not a port change.
  - `_print_fem --quick`: ALL_PASS (numbers in §2).
- **e2e:** teddy-161 25.6 s WARN with full artifact set; broken icosphere 18.5 s with
  verdict flip FAIL→WARN, valid OPC 3MF, sentinel rendered correctly; never-raise confirmed;
  CLI exit codes correct; cp1252 console survives.
- **license_clean: true** — `license_guard src` → CLEAN; only the expected 4 opt-in
  pyQuadriFlow WARNs (guarded, `[retopo-pyqf]` extra, never default). Default retopo path is
  the permissive from-scratch backends. Dead dep `pygltflib` dropped from pyproject.
- **app_constructs: true** (Gradio UI builds; served-session click-through not part of this
  pass).

---

## 5. Stage-0 demo readiness

### Demo recipe (all local, ~30 s per mesh)
1. `pip install -e C:\Users\mecht\Project_EI\3DEI\meshprep`
2. Hero mesh — PSB teddy:
   `meshprep prep "C:\Users\mecht\Project_EI\Applied\MeshDecomp\datasets\psb\MeshsegBenchmark-1.0\data\off\161.off" --profile ender3 --max-faces 6000`
   → 26.4 s; WARN; fix certificate "100% surface, 0.000 mm deviation"; orientation cuts
   overhang 25.0% → 7.4%; `prep.stl` + `report.md/json` + 2 renders.
   (Alternates: hand=181.off, fourleg=361.off.)
3. Verdict-flip demo — broken icosphere (2 hole patches): report shows
   "Free fix applied: FAIL → WARN" plus `reinforced.3mf`.
4. Resin demo — `--profile generic_resin_msla` on the teddy: 3 islands, hollow saves 62.2%,
   1 trap with vent suggestion, plain-English flags.
5. GUI — `pip install meshprep[app]` then `meshprep app`: drop the same mesh, show badge +
   report + heatmaps + downloads.

### Where the savings number comes from
`slicer.compare(before, after)` slices both meshes with the detected slicer CLI
(PrusaSlicer preferred — slices config-free) and diffs the **slicer's own** print-time and
filament estimates (`saved_g`, `saved_min`, `saved_pct`). On this box no slicer is
installed, so every demo run degrades honestly to "no slicer installed" in the savings line
— the pipeline treats `ok:False` as "estimates unavailable", not an error.

### Still needs the USER
- **Install PrusaSlicer and run one real `prep --slicer-savings`** — the only unexercised
  end-to-end path (parser is unit-tested against real footer formats, but no real slice has
  run here). Orca/Bambu additionally need machine/process JSON profiles.
- **Physical in-slicer confirm**: open a produced `reinforced.3mf` in PrusaSlicer/Orca and
  confirm the graded-infill modifiers render as intended (schema-verified, not eyeballed).
- **Community post / distribution** — outside Claude's scope.

---

## 6. Honest gaps and limits that survived (no overclaim)

All honest labels survive into the shipped report: *geometric heuristic* (premortem
channels), *UNCALIBRATED* (warp FEM until `meshprep calibrate` is run against real prints),
*COMPARATIVE/RELATIVE* (min_fos and strength — rankings, not absolute safety factors),
*"triage, not packing"* (bed-fit), *"autorig ships separately"*.

Known issues, in order of importance:

1. **"ASCII-safe markdown" claim is false as stated** (report.py docstring line 8 and build
   summary): report.md passes through U+00B0/U+2014/U+2265 from vendored-core issue
   strings. No crash (UTF-8 write + ascii_console guard) — but fix the docstring or
   transliterate.
2. **Check-only mode misleading line**: the overhang fix line still reads "We already
   oriented it to minimise this" while the orient stage is skipped in `check_only`.
3. **Manifest/tree disagreement**: `RELEASE_MANIFEST.md` line 36 lists
   `ei_core/self_intersect.py` as vendored; it was never copied (scout found zero consumers
   in the closure, so nothing breaks) — reconcile the manifest.
4. **Slicer path end-to-end untested** (§5); `filament_g` never fabricated from volume when
   the slicer doesn't report grams (Cura gives time+mm³ only).
5. **Upward-vent gap in the traps channel**: a top-vented cavity is not flood-tested from
   the vent per orientation — now honestly documented as a capability gap in the shipped
   docstring/report; closing it is the top follow-up feature.
6. **Warp numbers uncalibrated by default** (0.026 mm on the cantilever box, flagged as
   such); calibration store is per-user at `~/.meshprep/`.
7. **Stubs are stubs**: `split.py` (CoACD bed-split) and `api.py` (farm endpoint) are
   documented interfaces, not products.
8. **Compute envelope respected, not stress-tested**: selftests/e2e ran within the mission
   caps (≤6000 faces, FEM max_elem≤2500, voxel≤96, 1 thread); full-resolution large-mesh
   behavior on this 16 GB box is characterized only by the 136 s full-teddy check run.
