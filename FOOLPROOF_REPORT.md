# meshprep — Foolproofing Campaign Report

_Definitive record of the meshprep hardening effort: the original harden pass and this closeout._
_Date: 2026-07-02._

**What "foolproof" means here** (the bar this campaign was measured against):

1. **Never crashes** — no raw Python traceback ever reaches a user, on any input.
2. **Never silently ships a bad part** — it will not hand you a non-printable mesh while claiming success.
3. **Refuses or warns in plain language** when it can't help — no jargon, no empty verdicts.
4. **Novice-readable verdicts** — a beginner can read the headline and know what to do next.

Honest refusals and documented residual limits are **correct outcomes of this bar, not failures.** A tool that says "I can't print this, and here's why, in one sentence" is behaving exactly as designed.

---

## 1. The promise — and whether the evidence backs it

> **"Drop anything in, and you get either a print-ready mesh or a plain-language explanation of why not."**

**The evidence now backs this promise for the geometry inputs we tested, with two named caveats (a read-only output directory, and slow-not-hung analysis on large detailed meshes).** The numbers that back it:

- **Original harden sweep:** 136 cells (24 pathological inputs × applicable entry points) → **all category `ok`** on the final run; residual product-facing failures = 0. Self-tests proved the harness itself can catch a deliberately planted crash and a deliberately planted silent-wrong.
- **Regression + false-positive re-audit (r2): verdict PASS.** A clean watertight cube stays `PASS` / `claims_success=True` (not downgraded by the new gate); a broken icosphere (icosphere with ~35% of faces removed) returns an **honest `FAIL`, `raised=False`, plain reason** ("thinnest wall 0.01 mm < 0.40 mm nozzle") — the "fixed-with-certificate or honest FAIL, never a raise" contract, demonstrated live.
- **Completeness critic (53 novel inputs not in the baseline corpus): 0 new crashes, 0 hangs, 0 memhogs.** Every geometry adversary — self-intersecting solids, genus-50 slabs, nested cavities, NaN vertices, sub-nanometre and 1e9-scale cubes, unicode/long paths, a true 26.23 MiB over-guard file — was met with `REJECTED` / `ERROR` / `WARN` / `FAIL`, **never a PASS over bad geometry.**
- **License:** `license_guard src` → CLEAN, 0 banned copyleft/non-commercial imports (only the 4 expected opt-in pyQuadriFlow WARNs).
- **Full regression:** all per-module selftests green; clean box and cylinder → `PASS`, clean icosphere → `WARN` (advisory, not a false-FAIL). Zero clean meshes were false-failed.

**The one genuine open item against the promise** (found by the critic): writing into a **read-only / unwritable output directory** returns `PASS` "Ready to print" while producing no deliverable and silently skipping its own post-fix re-check. See §5. This is an I/O edge case, not a geometry-honesty failure, but it is a real gap in the promise and is documented, not hidden.

---

## 2. Timeline, honestly told

**Baseline.** A watchdog-free fuzz harness drove 10 distinct product-facing failures across 241 cells. The prep-family (`check`/`fix`/`prep`) was already foolproof — 0 crashes, 0 silent-wrong, 0 hangs across all 24 input classes. **Every** failure lived in two structural gaps:

- The `reinforce` subcommand **bypassed** the ingress guard and never-raise wrapper (dispatched straight into `core/reinforce.py`).
- There was **no shared printability/solidity gate** consulted before a PASS or a structural FOS was emitted, so degenerate / non-solid inputs got green "OK" headlines.

**Triage → 6 root causes** (after dedup of the 10 failures):

- **F1** — `reinforce` entry unguarded → raw traceback on a missing/unloadable path (crash, high); jargon refusal on non-mesh.
- **F2** — `reinforce` reports "OK → reinforced.3mf" + a comparative FOS on a zero-volume sliver and on non-watertight shells (silent-wrong, medium).
- **F3** — `resin_report` **hangs** on a degenerate 560k-face all-coincident mesh (the embree ray channel never returns; `decimate()` no-ops on coincident geometry so the "≤6000 face" cap doesn't hold).
- **F4** — cm-band unit mis-scale: a 60 mm part exported in cm reads 6.0, lands in the `[5,500]` "assume mm" band, ships at 6 mm under a green PASS (confusing, medium).
- **F5** — inside-out cube passes `check_only` while the report asserts "Consistent normals: yes" (confusing, low).
- **F6** — `split_for_bed` leaks a raw exception string ("`'NoneType' object is not iterable`") on 0-face inputs (confusing, low).

**Hardened** (three territories, in parallel): H1 wired the never-raise wrappers + ingress guard onto `reinforce`, added `scan_units()` + the cm-band WARN, and made the printability gate **authoritative and downgrade-only** in `prep()`. H2 built the new `core/_printability.py` gate and rewrote `report.py` to a plain-English, gate-driven headline. H3 fixed the resin hang and the split note leak.

**Then the first re-audit OOM'd the box — and we own it: the validation run was itself the memory bug.** The re-audit driver, backgrounded with a shell `&`, drove the product and RSS climbed monotonically past 5.8 GB with no cap. Leak-hunt forensics (§4) proved this was **not** a probe artifact — it was a genuine product balloon in `premortem`'s thin-wall channel. Two things changed as a result:

1. **A parent-side memory watchdog** (`run_watched`): every child is spawned via `Popen`, stdout/stderr drained on daemon threads (no pipe-deadlock), whole-tree RSS polled every 0.5 s; on RSS > cap (default 1500 MB) the entire process tree is killed and the cell is categorized `memhog` with peak RSS recorded. Backend is psutil (tree-aware) with a **verified** ctypes/psapi fallback that is smoke-tested at init so a silently-zero probe is never trusted. Proof: a 100 MB/tick hog child was killed at peak 1595 MB (95 MB over the cap, within one poll interval); a clean cube ran to `PASS` at 186.7 MB; free RAM held 5.6–6.1 GB.
2. **Strict serialization**: `run_all` is a single serial for-loop (grep for `multiprocessing`/`Pool`/`.map`/`concurrent.futures` = 0 hits); one child at a time; free RAM checked before every spawn and the run halts with `aborted_low_ram` if < 2 GB.

**This closeout's results:** r2 regression/false-positive sweep = **PASS** (14/14 checks green); critic = **PARTIAL** (0 crashes, one read-only-dir silent-wrong); leak-hunt = **fixed and verified**; the six baseline root causes = all fixed and re-verified (recheck PASS). The r1 re-audit remains **PARTIAL/INCOMPLETE** — the process was reaped by a Bash tool-call timeout after 21 clean cells, a process-supervision artifact (free RAM was a healthy 6.6 GB, **not** an OOM). See §6.

---

## 3. Guarantee mechanisms now in place

Each mechanism below is paired with the evidence that verifies it.

**Never-raise wrappers on every entry point.** `prep()`'s ERROR path emits a plain sentence, not a stack trace; the CLI `reinforce`/`calibrate`/`prep` dispatch and the Gradio `app.run()` are wrapped so no traceback reaches a user (`-h` SystemExit still passes).
_Evidence:_ F1 fix verified — `reinforce <missing path>` → "REFUSED: Could not read the uploaded file." rc=2, no traceback. Regression smoke: 0 crashes after.

**Ingress guard on `reinforce` (the one path that used to bypass it).** Real files now route through the same `load_and_guard` as `prep`: 25 MB size reject **before** `trimesh.load`, plus non-mesh / 0-face / NaN rejects, all with plain messages.
_Evidence:_ pointcloud.ply → "REFUSED: File contains no triangle faces." rc=2. The 28 MB over-guard file is rejected at ~76.7 MB peak RSS with **no mesh allocation** (size checked before load).

**Printability gate wired into `prep()` — no PASS over bad geometry.** New pure module `core/_printability.py::assess_printability()` never raises (any internal error → `FAIL`, "treated as not-yet-printable to be safe"). It FAILs: not-watertight, signed volume ≤ 0 (inside-out, even when winding is consistent), zero-volume collapse, 0-faces / all-degenerate, sub-nozzle thinnest extent (< 0.40 mm), non-finite bbox, and FEM-computed-nothing. It WARNs: implausible mm scale, > 24 components, genus > 50. In `prep()` the gate is **authoritative and downgrade-only** — it can turn PASS into WARN/FAIL, never the reverse.
_Evidence:_ selftest 6/6 green (clean cube PASS; inside-out, holed shell, sliver, dead-FEM all FAIL; metres-scale WARN). False-positive check: clean cube **not** downgraded; broken icosphere → honest `FAIL`, `raised=False`. F2 fix: `reinforce` on the 0.02 mm sliver and on non-watertight shells now **refuses** ("REFUSED: Not printable yet: …") instead of shipping "OK → reinforced.3mf".

**Unit-sanity on ingest.** `scan_units()` flags metre/inch/cm mis-scale; a suspiciously small part in the `[5,500]` assume-mm band is elevated to a **headline WARN** (PASS→WARN, no silent auto-rescale). An explicit `--assume-unit` / `assume_unit=` override converts honestly and clears the guess; the sub-5 mm rescale-to-60 mm branch keeps its "this is a guess, pass print_mm to override" disclosure.
_Evidence (F4):_ 60 mm-in-cm (ext 6) → assume-mm eff 6 **+ WARN**; metres/inches → guess eff 60 + WARN with rescale disclosure preserved; `assume_unit=cm` → honest eff 60, warning cleared.

**Fidelity-gated sealing.** `accurate_fix` seals real broken meshes to watertight positive-volume solids but **rolls back honestly** (`claims_success=False`, reasons like `nonfinite_deviation`) on un-closeable / degenerate input — it never claims a fix it didn't make. GWN-based solidify keeps genuine cavities and necks open rather than naively filling.
_Evidence:_ `_fix_accurate` selftest — accurate ≥ toy faithful on 4/4 closeable cases; `_solidify` selftest — GWN vs analytic sphere 100% agree, cavity/neck preserved, corpus axe.glb watertight/genus-0.

**Plain, novice-readable verdicts.** `report.py`'s headline is now one beginner sentence (Print-ready / Probably print-ready, check one thing / Not ready — one thing needs fixing / Couldn't read the file) + one next action, driven by the gate. A "## Can I print this?" section lists blocking problems and cautions in plain words. The "Consistent normals" row no longer asserts "yes" on an inside-out solid (F5 fix: it reads "consistent, but pointing INWARD (negative volume) — full prep will flip them").
_Evidence:_ CLI check on clean cube → rc=0, headline "Ready to print. No blocking issues found." F6 fix: `split_for_bed` on 0-face input → "input is not a triangle mesh (no faces) — cannot check bed fit", no leaked exception string.

**All honesty labels preserved.** Every `RELATIVE` / `COMPARATIVE` / `uncalibrated` / `geometric` / `heuristic` label survived verbatim across all fixes; FEM stayed capped ≤ 2500 elements; no slicing is performed (printability is judged geometrically, per scope). No claim was upgraded anywhere.
_Evidence:_ r2 checks confirm RELATIVE/uncalibrated labels intact in reinforce output, `fem_orthotropic` and `_print_fem` selftests report "UNCALIBRATED framing preserved", `sigma*` honestly reported non-spatial.

---

## 4. Leak-hunt verdict on the 5.8 GB runaway

**Verdict: `product_bug` — found and fixed.** The balloon was **inside the product**, not the probe: `prof.py` decimated its geometry to ≤ 60000 faces before calling the product, so it never materialized a huge mesh.

**Root cause:** `core/_print_premortem.py::_thin_wall_channel` cast `intersects_location(origins, dirs)` on **all faces at once**. With pyembree/embreex **not installed** here, trimesh falls back to its pure-Python ray engine; on a thin/elongated part each inward ray crosses ~O(n_faces) candidate triangles, so casting all F rays in one call accumulates an O(F·n_faces) candidate array — unbounded, monotonic, no guard. Reachable from `prep()`: the pipeline caps faces at 60000 but then calls `premortem` with **no further decimation**, despite the docstring advising "decimate to ≤6000 first."

**Fix:** memory-bounded ray batching (`THIN_PAIR_BUDGET = 8e6`, batch = clamp(budget/nf, 32, 512)) so peak memory is independent of face count; the embree fast path still casts in one shot; batch-local ray indices are shifted to global face ids. Total ray work (nf²) is unchanged → no time cost.

**Verification (under the watchdog, one process at a time):**
- Batched thin-wall == single-shot **exactly** (max thickness/risk diff = 0.0).
- `premortem` n=10: 2.4 GB (killed) → 0.36 GB. n=30 (60000-face worst case): killed → 633 MB, `memhog=False`.
- End-to-end via real `prep()`: thin-chain repro bounded to 671 MB (was heading to 5.8 GB); over-guard rejected cheaply at 90 MB; largest corpus mesh bounded ~1 GB. Free RAM held 5.2–6.7 GB throughout.

`find_traps` was **never** the culprit — it is already bounded by its `RES_MAX=110` voxel cap. Channel isolation confirmed only `_thin_wall_channel` ballooned.

---

## 5. Honest residual limits (designed behavior, not bugs)

These are things the tool **refuses or warns on rather than fixes**, stated as intended behavior.

- **Read-only / unwritable output directory (the one genuine open item).** `prep()` into a directory it can't write returns `PASS` "Ready to print" while writing no deliverable and silently skipping its post-fix re-check; the only trace is two "FAILED" rows in a provenance table. Reproduced twice. The same mode would hit on a full disk, a locked path, or antivirus lock. **Recommended fix (not yet applied): when the package/re-check stage fails to write, downgrade to WARN/ERROR or add a headline note — never report PASS.** This is the top of the Phase-2 fix list.
- **Time, not memory, on large detailed meshes.** The pure-Python thin-wall ray cast is O(n_faces²) in time → ~180–260 s on 60000-face meshes; `prep()` has no internal wall-timeout, so a big detailed mesh runs for minutes (memory-bounded and progressing, but reads as a "hang"). **Mitigation:** install `embreex` (O(1)/ray) or evaluate thin-wall on a ≤6000-face proxy as the docstring advises. This surfaced as `key.glb` / thin-chain hitting the harness 260 s wall (`memhog=False`).
- **Degenerate coincident geometry (resin).** A 560k-face all-coincident mesh has no printable volume; `resin_report` now **honestly skips** the ray/voxel channels with a plain flag ("degenerate geometry — this mesh has no printable volume … fix the mesh before a resin check is meaningful") rather than hanging. `decimate()` still no-ops on coincident geometry — neutralized at the caller, not in `_mesh_util`.
- **cm-band unit ambiguity.** STL/OBJ carry no units; the tool **cannot infer** them. It nudges (headline WARN + `--assume-unit` override) rather than silently guessing. Correct-by-design disclosure, not certainty.
- **Geometric scope, not a slicer.** Self-intersecting solids and genus-50 slabs are **WARNed, not hard-failed** — appropriate, since meshprep is explicitly a geometric pre-flight and slicers union positive regions. Nested cavities → FAIL (right call for resin).
- **Private-engine threshold artifacts (not product-reachable).** `accurate_fix` reports `watertight=True` on a zero-volume all-coincident mesh and on a 1e-9 mm cube (both below its fixed degeneracy threshold). Both are **guard-bypassed** private-engine calls; the product `prep()`/CLI wrap them safely (ERROR / WARN via the unit-scan nudge). Documents a scale-threshold blind spot in the raw engine only.

---

## 6. What is still untested or unproven

Be specific — these are honest gaps, not claims of coverage:

- **The r1 same-corpus re-audit never completed.** The driver was reaped by a Bash tool-call timeout after 21 clean cells (all `ok`, plausible RSS). Two source-diff-confirmed fixes remain **runtime-unverified by that sweep**: the resin degenerate-geometry gate on `huge_over_guard.stl` (though the leak-hunt separately verified it: rc=0, 694.7 MB, no hang), and four `reinforce` cells re-tested only in the fix round, not re-swept. **Recommendation:** relaunch `reaudit_final_run.py` via native `run_in_background=true` (not shell `&`) and poll to the "REAUDIT FINAL SUMMARY" line; the script is idempotent and incrementally flushed.
- **The Gradio GUI (`app.py`) was never launched.** Its `run()` prep call is wrapped so a surprise becomes a clean FAIL badge, and it wraps `prep()` which is covered — but the actual browser/upload path, file-picker behavior, and rendering were not exercised. This is a Phase-2 "eyeball" item.
- **Concurrent / multi-user use is unproven.** The whole campaign ran **strictly one subprocess at a time** by design. Nothing tested two `prep()` calls racing on shared resources (only `double_prep` back-to-back on the same out_dir, which passed). Server-mode concurrency is untested.
- **`calibrate` CLI was never fuzzed.**
- **Formats:** tested on STL / OBJ / PLY / GLB. 3MF **output** is schema-checked; 3MF/STEP/AMF as **input** were not fuzzed.
- **`_print_advanced.py` has its own separate `_intersector`** (unchanged by the leak fix). It stayed ~100 MB during `optimize_orientation` on the thin mesh but was **not** exhaustively probed on large thin inputs — warrants a similar batching audit if ever driven on 60000-face thin parts.
- **Fix coverage is not an exhaustive shape sweep.** The leak fix was validated on the tori-chain family (n=10/30) + corpus meshes + an exact batched-vs-single-shot equality check. `THIN_PAIR_BUDGET=8e6` is tuned for the 60000-face cap; raising `DECIMATE_FACES` would need it re-checked.
- **No real functional print** has been run — printability is judged geometrically, never by slicing or printing a part.

---

## 7. Next

Per **`project_meshprep_wireframe_fixer_plan`** (Phase 2):

1. **Close the one open gap first:** make the read-only / unwritable-output case downgrade to WARN/ERROR with a plain headline note (the §5 top item) — small, high-value honesty fix.
2. **One-stop summary polish:** finish the resumable r1 re-audit sweep (properly supervised background run) so the resin + reinforce fixes are runtime-confirmed on the full manifest, not just source-diffed.
3. **GUI eyeball:** actually launch `app.py`, walk the novice upload → verdict → download path, confirm the plain headlines render as intended.
4. **Functional-print GTM:** the vacant SmartSlice/Markforged physics-preflight seat, at the decided $200/mo bar — carry every `uncalibrated` / `RELATIVE` label into the product copy verbatim.

---

_Honesty discipline held throughout: no HEURISTIC / geometric / uncalibrated / RELATIVE / COMPARATIVE label was touched or upgraded; the tool refuses to balloon (bounded batches) rather than raising any cap; documented residual limits and honest refusals are recorded as correct outcomes._

---

## Closeout addendum (2026-07-03, main session, independently verified)

1. **Full-sweep status:** the resumed native sweep ran 36 product cells (all clean) then
   honored its low-RAM floor and stopped — the box was concurrently running a ~20-process
   qc_watermark research campaign. ~100 manifest cells remain to be swept when the box is free
   (driver: `scratchpad/foolproof/reaudit_final_run.py`; it is safe to re-run any time).
2. **The 4 recorded "hangs" are NOT product bugs.** All four cells (core/cli reinforce on
   duplicated_faces.stl, prep_check on nonmanifold_soup.obj, resin on open_shell_holes.stl)
   were re-run directly on a quiet interpreter: each completed in **0.7–2.1 s** with honest
   verdicts (REFUSED / FAIL / flags). The sweep-time "hangs" were cold-import stalls under
   the concurrent campaign's RAM pressure (children died at 70–95 MB RSS with zero output =
   still importing). Environmental artifact; adjudicated closed.
3. **Read-only output dir — FIXED + verified** (was this report's top open item). Two guards
   in `pipeline.py`: an ingress writability probe (sentinel write; plain ERROR "pick a
   writable folder" on failure) and a deliverable-exists check (PASS/WARN downgrades to
   ERROR if prep.stl is not on disk at package time — catches mid-run permission loss/disk
   full). Verified with a real NTFS ACL write-deny: ERROR + plain headline, no raise;
   writable control still PASS with 8 artifacts. py_compile clean.
