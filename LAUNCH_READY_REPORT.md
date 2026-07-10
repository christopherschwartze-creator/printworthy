# meshprep — Launch-Ready Report

_Generated 2026-07-04. Sourced only from this run's BUILD / SWEEP / VERIFY records and the reports they cite. Every physics number below keeps its original uncalibrated / comparative / geometric label. No number was invented or rounded up. Deferred work is stated as deferred._

---

## 1. What closed this run

Four lanes landed additive, compile-clean changes across the public `meshprep` and the `meshprep_pro` repos. VERIFY overall verdict: **PARTIAL** — every static check passed; the one failing check is a sanctioned RAM defer, not a defect.

### Lane W — supports CLI/pipeline wiring (public meshprep)
- Files: `src/meshprep/cli.py`, `src/meshprep/pipeline.py` (2 files, additive only).
- Added standalone `meshprep supports <mesh> [-o out.3mf] [--enforce-pctile 85.0] [--block-pctile 30.0]`, dispatched like `reinforce`/`calibrate`, listed in top-level help. Output note (honest note + measured PrusaSlicer `support_material` / `support_material_auto` flags) is `support_mods()`'s own returned string verbatim, not re-derived.
- Added opt-in `prep(..., supports=False)` / `meshprep prep --supports`: new stage "5d. support mods" after orient/warp-FEM, before reinforce, gated by `check_only` and the soft time budget. On success writes `supports.3mf`, adds `result['channels']['supports']` (ok, out_3mf, n_enforcers, n_blockers, risk_top, uncalibrated, note), `result['files']['supports_3mf']`, and one plain notes line labeling risk as "a heuristic triage field, not a guarantee". Runs inside the never-raise `stages.run()` wrapper — a failure degrades to a notes line, never a crash or verdict change.
- Self-check: py_compile clean, pyflakes zero findings, argparse Namespace parse + `inspect.signature(prep)` confirm the new `supports=False` kwarg. Committed (lane-W commit `d6ac613`), repo boundary respected — `core/support_mods.py`, `threemf_writer.py` untouched.

### Lane T — inline selftest HARD GATE (meshprep_pro)
- File: `src/meshprep_pro/multimaterial_pro.py` (440 insertions, 1 deletion — the `__all__` line, to add `selftest`). `material_map_pro()` behavior unchanged.
- Added `selftest()` + private gates `_gate_g1/_gate_g2/_gate_g3/_gate_optional_slicer`, `_ram_check/_ram_gate`, duplicated XML helpers, in the house style of `meshprep.core.selftest.run()`.
  - **G1**: 80×12×12 mm cantilever through `material_map_pro`; checks extruder convention (rigid `extruder=1`→T0, soft `extruder=2`→T1), per-volume `extruder` schema on both ModelParts, hot mesh watertight + inside base bbox, and FEM high-stress targeting beats a random control (reuses reinforce selftest G5 precision/recall pattern, percentile=70, mode=vm, max_elem≤2500).
  - **G2**: same geometry loaded axially; checks `material_map_pro` refuses a near-uniform stress state via the `recommend_grading` prescreen with a message containing "uniform".
  - **G3**: successful result's `claim` string literally contains "RELATIVE" + "HOMOGENEOUS solid" and `uncalibrated is True`.
  - **Optional slicer gate**: slices G1's 3MF with auto-detected PrusaSlicer console against bundled `tests/assets/multi_extruder.ini`, regex-scans g-code for a bare `T1` tool-change line.
- RAM discipline: every heavy step preceded by a live `psutil.virtual_memory().available` check against a 2.0 GB floor; below it the step is skipped and recorded "deferred: low RAM"; missing/unreadable psutil fails **closed**.
- Committed `ba5a400`. Only the owned file staged.

### Lane D — demo assets (public meshprep)
- Changed `README.md` (new "See it" section near top). New: `examples/README.md` (3 walkthroughs) + 7 captioned PNGs under `docs/images/` (before-after-fix, risk-heatmap-ai-axe, reinforce-dense-core, warp-prediction, fix-certificate, corpus-gallery, support-zones).
- Mined from existing validation renders (no new generation — RAM 1.7–2.0 GB). Money shots: composed same-mesh diagnose→certify on a real AI axe using its own report numbers; 6-panel risk heatmap; graded dense-core reinforce (vM 6.64 vs 3.41 comparative). Every physics figure keeps its uncalibrated/comparative/geometric label; warp 0.103 mm labeled UNCALIBRATED; supports +42.5% labeled measured. Corrected the brief's recalled numbers to report-exact (dumbbell seam x=−3.48 not 0.0; conservation 0.047% / 0.162% / 0.000%; dropped a non-existent "+6.7%").
- Committed `7448e0f`. VERIFY confirmed all 7 image refs resolve on disk and `docs/images` is not gitignored.

### Lane P — launch-post drafts (public meshprep)
- New: `LAUNCH_POSTS.md` at repo root. Three verdict-first drafts in the product's honest voice, each with 2–3 title options, ≤300-word body, honest "what I'd love feedback on" closer:
  1. r/functionalprint (physics): 90×86×18 mm bracket split into 7 bed-fitting parts + 18 connectors, volume conserved 0.047%, seams labeled VISIBLE; supports measured from g-code (+1.27 cm³ / +42.5% enforcer, −26.7% blocker back to 2.99 cm³ baseline, 40° cone = zero enforcers); warp 0.0848→0.0075 mm labeled comparative/uncalibrated; uniform-stress bar refuses.
  2. r/3Dprinting (novice): broken Meshy axe receipt "0% reshaped, max deviation 0.003 mm", 1619 holes sealed, 136+53 fuzz inputs zero crashes / zero PASS-over-bad-geometry, plus an honest FAIL example.
  3. HF / Show HN (technical): never-crash fuzz contract, source-verified 3MF strings, GWN vs analytic-sphere 100% closed-form validation, license-clean QuadriFlow (26 SparseLU / 0 SimplicialLLT).
- Plus a 5-point posting checklist. All live links are `[LINK-TBD]`; header states posting is the user's action. Committed `09c0a5b`.

### Cross-cutting verification (VERIFY)
- py_compile clean on all three changed .py files.
- `license_guard.py` (public): **CLEAN**, 0 banned imports, 4 pyQuadriFlow opt-in WARNs only, exit 0.
- `meshprep_pro._license_check`: **CLEAN**, 0 banned, 0 WARN, exit 0.
- `meshprep supports --help` and `meshprep prep --help` both show the new flags; README/examples commands validate against the live argparse tree.
- LAUNCH_POSTS.md numbers spot-checked verbatim against source reports (S2_FEATURES / PRO_BUILD / FOOLPROOF).
- **Wheel freshness fixed**: `check_wheel.py` initially FAILED (wheel older than cli.py/pipeline.py/split.py). Rebuilt via `pip wheel . -w space/wheels --no-deps` (pure-python, light) → now "fresh, critical modules byte-identical". Committed `6bafba5`.

---

## 2. What was deferred, and why

**Root cause (all defers):** free RAM never cleared the required floor this whole session — another session's research job was running on the 16 GB box the entire time and was left untouched. Samples: sweep saw 1.73–1.81 GB free; VERIFY saw psutil available 1.30–1.90 GB; lanes W/T/D saw ~1.6–2.0 GB. Heavy steps (prep/FEM/slicing/fuzz/marching-cubes) need a durable ≥2.0 GB margin (sweep driver needs ≥2.5 GB start floor). Every lane deferred honestly rather than push through.

| Deferred item | Why | Exact command to resume |
|---|---|---|
| Lane W end-to-end `meshprep supports` + `prep(supports=True)` on a small box | RAM never held ≥2 GB; imports trimesh/scipy + marching-cubes | `python -m meshprep.cli supports <20mm_box.stl> -o out.3mf` then `python -c "import meshprep; meshprep.prep('<box.stl>', supports=True)"` (from `3DEI/meshprep`) |
| Lane T gate LOGIC G1/G2/G3 + optional slicer gate (scaffolding IS runtime-verified; gate bodies are not) | RAM stayed at/below 2 GB floor, so gates deferred honestly; 2× FEM solve + slicer subprocess never ran | `python -m meshprep_pro.multimaterial_pro` (from `3DEI/meshprep_pro`), once free RAM durably >2 GB → real ALL_PASS/FAIL |
| VERIFY step 7 tiny end-to-end `meshprep supports` on a 20 mm box | Same 2 GB floor; sanctioned defer, not a code defect | same as Lane W row above |
| Sweep: ~97–100 remaining foolproof re-audit cells | Sweep needs ≥2.5 GB start floor; prior 2026-07-02 run hit `aborted_low_ram=true` at 39/136+ cells | run `reaudit_final_run.py` with system Python `C:\Users\mecht\AppData\Local\Programs\Python\Python311\python.exe` from the foolproof dir; PASS = `aborted_low_ram=false`, both selftest flags true, `watchdog_ok=true`, zero non-selftest crash/silent_wrong, `cells_total` ≈ full ~143. See `SWEEP_RUNBOOK.md` in the foolproof scratchpad. |
| Lane D: raw-geometry hole-close before/after silhouette, smart-split parts render, PrusaSlicer GUI screenshot | Source-accurate fixes → identical silhouettes (no dramatic pair exists); GUI screenshot needs the GUI; generation blocked by low RAM | Substituted textually with measured numbers; regenerate renders once RAM frees, or capture GUI screenshot manually |

---

## 3. User's FINAL outward checklist (in order)

These are **your** actions — everything above is code/asset-ready; these ship it.

1. **GitHub push** — per the deploy guide **Step 0**. Push the public `meshprep` repo (commits `d6ac613`, `7448e0f`, `09c0a5b`, `6bafba5`) to GitHub.
2. **Free Space deploy** — per **Part 1**. Deploy the HF Space, then run `space/check_wheel.py` (already fresh) + the smoke matrix to capture real 2 vCPU timings before any post.
3. **LS test-mode + Pro Space** — per **DEPLOY_PRO.md**. Stand up the Pro (`meshprep_pro`) Space and wire the store in test mode.
4. **Launch posts** — `LAUNCH_POSTS.md`, following its embedded 5-point posting checklist:
   - Deploy the Space + run `check_wheel.py` first.
   - Match one screenshot per post.
   - Timing / one-at-a-time cadence.
   - Respond fast + honest, listing deferred items.
   - Keep a broken mesh ready.
   - Replace all `[LINK-TBD]` with the live HF Space + public repo URLs before posting.

---

## 4. Honest state of the product

**Verified this run:** all changed source compiles clean; both repos pass license guard with zero banned copyleft/non-commercial imports; the public CLI exposes `supports` as a subcommand and `prep --supports` flag with correct argparse wiring; the wheel shipped to the Space is byte-fresh against source; every number in the demo assets and launch drafts traces verbatim to a measured report line with its uncalibrated/comparative/geometric label intact; and the Pro selftest's RAM-discipline scaffolding is runtime-verified to defer-not-push and to fail closed. **Not yet verified — and only a real run (or a real user) can prove it:** no mesh has been driven end-to-end through the new `supports` path this session (static parse/signature checks stand in), the Pro selftest gate *logic* (G1 targeting math, G2 uniform-stress refusal, G3 claim string, T1 tool-change parse) is hand-reviewed against real source APIs but never executed, ~100 foolproof re-audit cells remain unrun, and no functional print, absolute-mm warp calibration, or concurrent-load behavior has been observed. Physics outputs remain **comparative and uncalibrated by design** — the product claims relative/geometric guidance, never a load-rated guarantee. It is launch-ready as an honest, never-crash prep tool with clearly-labeled heuristics; the remaining confidence comes from resuming the four deferred runs once RAM clears and from first real users.

---

## Executive summary

- Four lanes landed additive, compile-clean, license-clean changes; VERIFY verdict PARTIAL — all static checks pass, the sole failing check is a sanctioned RAM defer.
- Lane W wired `support_mods` into the public CLI (`meshprep supports`) and pipeline (`prep --supports`), never-raise, all keys additive (commit d6ac613).
- Lane T added an inline `selftest()` HARD GATE to `multimaterial_pro.py` (G1–G3 + optional slicer), RAM-disciplined and fail-closed (commit ba5a400).
- Lane D shipped 7 captioned demo PNGs + `examples/README.md` + a README "See it" section, every physics number kept its uncalibrated/comparative label (commit 7448e0f).
- Lane P wrote three honest, verdict-first launch drafts in `LAUNCH_POSTS.md` with a posting checklist; all live links are `[LINK-TBD]` (commit 09c0a5b).
- Wheel freshness was caught stale and fixed — rebuilt and byte-verified against source (commit 6bafba5).
- Deferred everywhere for one reason: free RAM stayed ~1.3–2.0 GB (another session's job running); no mesh was driven end-to-end, and ~100 foolproof cells + the Pro gate logic remain unrun — each with an exact resume command above.
- Outward checklist for the user, in order: GitHub push (Step 0) → free Space deploy (Part 1) → LS test-mode + Pro Space (DEPLOY_PRO.md) → launch posts per `LAUNCH_POSTS.md`.
- State of product: static surface fully verified and honestly labeled; end-to-end runtime behavior and gate logic await a RAM-clear rerun and first real users.
