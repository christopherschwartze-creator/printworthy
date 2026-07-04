# meshprep — Phase 0 + Phase 1 Report

_Date: 2026-07-02. Synthesized from the implementation + validation records for the four changes. All numbers below are as measured by the validation harness; no runs were repeated for this report._

## 1. Outcome

Four changes were built and all four passed independent validation:

- **Phase 0.1 — FEM Pa unit bridge** (`fem_orthotropic.solve_part_stress` + `_print_fem`): **LANDED, PASS.** `min_fos` on mm meshes is now numerically in Pascals and compares correctly against Pa allowables instead of pinning at the 1e6 ceiling.
- **Phase 0.2 — solidify fidelity outcome-regate** (`_fix_accurate.py`): **LANDED, PASS.** The winding-number seal now ships only when its measured fidelity clears a floor, killing the crane 100%→62% bad trade without special-casing any input.
- **Phase 0.3 — internal-shell (cavity) certificate** (`_internal_shells.py` + report/pipeline wiring): **LANDED, PASS.** A pure, never-raises cavity census now surfaces one honest report line and two additive JSON keys.
- **Phase 1 — graded-infill gradient pre-screen** (`recommend_grading` / `_grading_gradient_stats` in `reinforce.py`): **LANDED, PASS.** Grading is now recommended only when a stress-gradient pre-screen predicts it will actually save material; the flag matched ground truth on every part checked (9/9 offline + 3/3 live).

A package-wide regression + license guard sweep across all nine edited files also passed: license CLEAN (only the 4 pre-existing pyQuadriFlow opt-in warnings), all pre-existing selftests green, never-raise contract intact.

No change was rolled back and no change failed.

## 2. Change summary

| Change | Files | What it fixes | Verdict | Key measured number(s) |
|---|---|---|---|---|
| 0.1 FEM Pa unit bridge | `core/fem_orthotropic.py`, `core/_print_fem.py`, `core/reinforce.py` | mm-mesh stress was 1e6× too high → FOS pinned at 1e6 ceiling; now Pa-correct | PASS | mm bracket F=100N: `min_fos=116.67`, `peak_vm=0.47 MPa`; `m`-control reproduces old bug (`min_fos=1e6`, 100% at ceiling); inv_fos importance now non-flat (std 0.153, ptp 1.0 vs old std 0) |
| 0.2 solidify outcome-regate | `core/_fix_accurate.py` | Crane winding-number seal dropped fidelity 100%→62% making a genus-578 blob; now skipped on low fidelity | PASS | Crane surface_unchanged now `100.0%` (`solidify_rejected_fidelity=62.2`, `seal_method=skipped_low_fidelity_solidify`); cohort mean `100.0%`, 8/8 watertight, no regression |
| 0.3 internal-shell certificate | `core/_internal_shells.py` (new), `report.py`, `pipeline.py` | Kept cavities were invisible to the user; now reported with an upper-bound material estimate | PASS | Cavity test box: `n_internal_shells=1`, `void=8000 mm³`, `solid_volume_ratio=0.875` = (40³−20³)/40³; solid box `n=0, ratio=1.0`; additive-only (identical JSON key sets) |
| Phase 1 gradient pre-screen | `core/reinforce.py` | Graded infill offered blindly; now offered only when predicted to win | PASS | 9/9 offline + 3/3 live match; threshold `5.33×` separates winner `57254` (14.65×) from 8 losers (≤3.62×) |

## 3. Phase 1 pre-screen — gradient_ratio vs saved% (wedge cross-check)

`gradient_ratio = fos_out_min / fos_global_min`. `actual_win` = best material_saving across infill exponents ≥ 15%. Threshold = 80/15 = **5.33×**.

| Part | gradient_ratio | best_save% | pre-screen | actual win | match |
|---|---|---|---|---|---|
| 120755 | 1.588 | −20.5 | False | False | ✓ |
| 78661 | 1.000 | −5.7 | False | False | ✓ |
| 48342 | 3.618 | +3.5 | False | False | ✓ |
| 917937 | 1.000 | −14.2 | False | False | ✓ |
| 46263 | 1.225 | −41.5 | False | False | ✓ |
| 186560 | 1.000 | −12.0 | False | False | ✓ |
| 45616 | 1.000 | −3.2 | False | False | ✓ |
| 61762 | 2.414 | −4.8 | False | False | ✓ |
| **57254** | **14.651** | **+34.5** | **True** | **True** | ✓ |

The `5.33×` threshold sits cleanly in the gap: the single winner is at 14.65×, every loser is at ≤3.62×. Live `recommend_grading` runs (post unit-fix, 100 N) confirmed the second documented winner and a loser: **56979** True (ratio 10.80), **57254** True (ratio 10.44), **45616** False (ratio 1.00) — all `fos_global_min` finite in the 6–84 range, not clipped, confirming the unit bridge flows through at nominal load.

**Honest limit.** The pre-screen is validated on **10 parts total** (9 in `wedge_rows.json` + `56979` verified live; `56979` is absent from the JSON file, so the "10-part" separation is 9-in-file + 1-live, not 10-in-one-file). The metric is a **comparative / uncalibrated FOS** ratio, not a calibrated material-savings prediction — it says "grading is likely to win / not win," not "you will save X%." `57254`'s live ratio (10.44) differs from its stored JSON ratio (14.65) because `recommend_grading` decimates and remeshes independently of the wedge harness; both are far above threshold, so the binary recommendation is robust but the exact ratio is not portable. The pre-screen is labeled `uncalibrated:True` and does not block building.

## 4. Regressions

**No functional regressions were found.** What was checked in the package-wide sweep:

- **License guard** on `src`: CLEAN, 0 banned copyleft imports; exactly the 4 pre-existing pyQuadriFlow opt-in warnings in `_pyqf_backend.py`.
- **Selftests, all green:** reinforce G1–G5 + tiers T1–T4 (TIERS_ALL_PASS), `_solidify` (GWN 100% agree), `_fix_accurate` (dev bit-identical to the STAGE0 Forge baseline, accurate ≥ toy 4/4), `fem_orthotropic` (ALL CHECKS PASSED, anisotropy ratio 0.648 vs Zt/Xt 0.65), `_print_fem` (ALL_PASS, strength ratio 0.635), `fem_materials`, `fem_voxel_core` (cantilever 0.9723), `_warp_calibrate`, `fem_warp_probe`, `_fem_validation_plan`.
- **Smoke:** `meshprep.prep` callable; `prep()` on a broken primitive returned a dict without raising (never-raise contract intact through the edited `report.py` + `pipeline.py`).
- **py_compile** on all 9 edited files: exit 0.

**One latent footgun flagged (not a regression within scope), worth an explicit note:**

- `length_unit` defaults to `"mm"`, but both FEM modules' own selftest/demo geometry is in **meters** (cube 0.02 m, bar 0.10 m). With the default, a meter-scale mesh over-scales traction by 1e6 (selftest shows `peak_vm=1.3e12 Pa`, `min_fos` displaying 0.000). The selftests stay green **only because every gate is ratio/sign-based (scale-invariant)**. Within the stated "Forge is mm" scope this is the intended tradeoff, but any external caller passing a meter-scale mesh without `length_unit="m"` will silently get stress 1e6× too high. Recommend an explicit note in the API docs or a mesh-extent heuristic guard. This is a documentation/guard gap, not a broken gate — all four requested gates and the full regression suite pass.

Also noted as intended (not defects): `min_fos` prints `0.000` in the tiny-demo FEM cruxes because the demo loads drive stress far above yield (the sentinel/comparative regime); the pass gates key off ratios/monotonicity, which hold. `fem_orthotropic` selftest genuinely takes ~4.5 min (grid refinement) — not a hang; needs a timeout > 2 min.

## 5. What is now true that wasn't

- **`min_fos` is Pa-correct on mm meshes.** The same part declared in mm@mm and m@m now converges to identical `min_fos=116.666` and identical best/worst separation `1.5719` — the signature of a correct unit bridge. FOS values are now real numbers against real Pa allowables, not a pinned 1e6 ceiling; the inv_fos importance field is no longer flat, so orientation and reinforcement decisions see real stress structure.
- **Graded infill is offered only when it wins.** A stress-gradient pre-screen runs on the same FEM solve (no second solve) and recommends grading only above the 5.33× threshold; on the validated parts it recommended grading for exactly the parts that actually saved ≥15% material and stayed silent for the rest. Building is never blocked — it's an advisory.
- **The fixer no longer makes the crane bad trade.** The winding-number solidify seal is gated on a measured two-sided fidelity outcome (floor 85%, skip-margin 10%), not on input genus. Low-fidelity seals (crane: 62.2%) are skipped in favor of the more-faithful pre-solidify mesh; on the 8-mesh cohort all reached 100% surface fidelity and watertight via the faithful ladder, so skipping the sledgehammer cost nothing.
- **Users are told about kept cavities.** A closed internal shell is now reported as an intentional cavity with a geometric upper-bound on the material a cavity-filling auto-repair would have added (e.g. "~88% of the material … the gap shrinks at low infill"), labeled a geometric estimate, not a slicer number.

## 6. Next — Phase 2 hooks

From `project_meshprep_wireframe_fixer_plan`:

- **One-stop flow polish.** Wire the four Phase 0/1 signals (Pa-correct FOS, cavity census, grading advisory, fidelity-gated seal) into a single coherent prep summary so a user gets one honest verdict line + the advisories without reading JSON. Close the `length_unit` footgun (doc note or mesh-extent guard) as part of this.
- **GUI eyeball.** A visual review step — show the shipped mesh, flag kept cavities and the low-fidelity-skip decisions, and let the user see the graded-infill recommendation with the gradient map, rather than trusting the numbers blind.
- **Functional-print GTM.** The FEM + grading pre-screen now produce defensible comparative signals; the go-to-market motion is the functional-print guarantee (orient-for-strength + graded infill only where it wins), which needs the calibration work to turn the comparative FOS into a stated physical claim before it can be sold as a number rather than a ranking.
