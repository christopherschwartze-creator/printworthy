# meshprep Adjustment Report — 75-mesh AI benchmark follow-up

Date: 2026-07-02. Corpus: 75 real AI-generated meshes (sf3d / meshy / hunyuan, `3DEI/Forge/corpus`).
Every number below is measured; evidence files are listed at the end. Negatives are stated as plainly as positives.

---

## 1. What was wrong -> what we changed -> what it measures now

| | OLD (measured) | Mechanism found (named, not a vibe) | Change made | NEW (measured) | Target hit? |
|---|---|---|---|---|---|
| **F1 — fixer fell back to crude voxel reseal on 50/58 broken meshes; output unfaithful** | Surface kept mean **60.4%** (median 61%), max-deviation mean **4.5 mm**, genus-0 only **5/58 (8.6%)**; worst-10 mean surface kept 25.7% | (A) AI meshes are open-shell soup; boundary loops containing *junction vertices* were silently skipped by the loop extractor on 12/12 autopsied meshes, so holes stayed open. (B) trimesh `fill_holes()` inside `_cheap_repair` *created* 6–208 non-manifold edges on inputs that had zero, permanently poisoning the watertight check and disabling manifold3d. (C) A fixed loop budget (5000) truncated 2/12 huge meshes — amplified by a harness bug: `simplify_quadric_decimation(60000)` positional arg bound to `percent` in trimesh 4.11.5, so decimation was a silent no-op on 49/75 meshes. When the reseal then fired, its voxel grid was too coarse (pitch 5.7–6.6x the certificate tolerance) and `binary_closing` welded shells into phantom handles (genus 17 blobs); no gate could catch it because the genus gate required a watertight *input*. | Removed `fill_holes()`; added a manifold-safe 3-cycle closer; added a junction-vertex wedge splitter + cycle peeler (`_fill_boundary_cycles`, leaves 0 boundary edges on all 12 autopsy meshes); loop budget now scales with boundary-edge count; new `core/_solidify.py` (generalized-winding-number solidify, no morphological closing) replaces the crude reseal; `make_watertight` demoted to strict last resort; genus formula fixed (per-component); harness decimation fixed (keyword arg). | Same-58 subset: surface kept mean **99.3%** (median 100%), max-dev mean **0.096 mm**, genus-0 **51/58 (87.9%)**, watertight 58/58, global seal fired **1/58** (was 50/58), 0 rollbacks, runtime 20.3 s -> 6.7 s mean. All-66 broken (real decimation breaks 8 more hunyuan meshes): 99.4% / 100% / 0.090 mm / 78.8% genus-0. Worst-10: 25.7% -> **100.0%** surface kept. | **PASS** (target: detail >= 85% mean; watertight 100%) |
| **F2 — 3/10 fixed meshes failed to slice where the raw broken mesh sliced fine** | Fixed hard-failed in PrusaSlicer 2/3 ("no layers" / "print is empty"), and 7/9 of the survivors sliced *silently* at 8–72% of the raw filament volume | **Real product bug, not (only) a harness artifact.** RC1: skimage marching-cubes output is inside-out under trimesh/STL convention; the reseal never re-oriented it (the guard existed in `fix_toy` but was lost), and no gate checks signed volume — inverted "watertight" solids shipped. RC2: trimesh `filter_taubin` smoothing pulled bridge vertices toward the origin, producing spike/bbox blowout (chair1 fix bbox grew ~15 mm past the input) -> "Floating object part / Collapsing overhang / Loose extrusions". Separately, the old harness sliced at unit scale (a genuine artifact that made its stored slice numbers useless — both things were true). | Outward-orientation enforce (fix_normals + invert while volume < 0) after marching cubes and again before shipping; `filter_taubin` removed (pinned bridge-only neighbour-mean smoothing instead); three new shipping gates: nonpositive-volume rollback, bbox-exceeds-input rollback, out-to-surface drift-spike rollback (this metric was previously computed but unused). Rebench slices at 60 mm + bed-drop *before* fix and export. | The 3 named regression meshes slice **6/6** (raw ok AND fixed ok). 12-mesh slice sample: raw **12/12**, fixed **12/12**. All shipped volumes positive. 0 rollbacks needed in the 66-mesh run. | **PASS** (target: fixed slice-success >= raw) |
| **F3 — PrusaSlicer auto-repairs raw broken meshes 10/10, so "we make it printable" is not a differentiator** | Candidate replacement pitch: "the slicer repairs *silently and wrongly*; we report what changed" | Tested head-to-head (Section 2): on this corpus the slicer's silent inside/outside guess is **not demonstrably wrong** — it matches a principled winding-number ground truth within ~2% filament on 15/16 meshes. | No code change needed for this finding; the deviation certificate stays (it is now honest — 99.4% surface kept means it reports near-zero change on most meshes). | Pitch verdict: **transparency-as-"the-slicer-is-wrong" FAILS its test.** See Section 2 for the honest reframe. | **Clean negative** (a valid outcome; the pitch as originally framed is dead on this evidence) |

**Regressions introduced by the new fixer (all of them):**
- 1 material: `hunyuan/sdxl_087_origami_crane_v3` — the (labelled-heuristic) genus gate fired `gwn_solidify` on a genus-578 garbage-topology input: surface kept 100 -> 62.0%, dev 0.002 -> 4.29 mm, genus 578 -> 427 (still garbage — a bad trade; the only solidify firing in the whole 66-mesh run).
- Minor, on inputs changed by now-real decimation: `hunyuan/raccoon_wizard` genus 8 -> 11 (+0.17 mm), `hunyuan/sdxl_062_old_key_v2` genus 5 -> 10, `hunyuan/tree` dev +0.29 mm.
- Old negative-genus values (−1, −3, −74) in the baseline were artifacts of the old genus formula, not real geometry.
- Known limit, visible and **not** fixed: fixed meshes print at 0.33–1.08x (median ~0.56x) the raw slicer-auto-repair filament — the thin-shell / internal-cavity divergence discussed in Section 2.

Caveats on the headline numbers: *watertight is not the same as genus-correct* — the remaining non-zero-genus outputs are mostly faithful topology (donut = 1, jar-with-handle = 1) or hunyuan garbage-topology inputs. `license_guard.py src` re-run after the change: **CLEAN** (0 banned imports; 4 pre-existing opt-in WARNs for pyQuadriFlow, unrelated).

---

## 2. The transparency pitch verdict

**The question:** the pitch "PrusaSlicer repairs silently with an arbitrary inside/outside guess; we tell you exactly what changed" only survives if the slicer's silent guess is *demonstrably wrong* on real meshes.

**The test:** 16 mesh-cases (13 broken + 3 controls) across all three generators. For each mesh we built two candidate "true solids": a blind voxel fill (what a naive reseal assumes) and a generalized-winding-number (GWN) solid (`|w| >= 0.5` — the principled inside/outside). We then sliced the raw broken mesh (letting PrusaSlicer repair it silently) and both ground truths, and compared filament volumes.

**The results:**
- **The inside/outside question is not actually ambiguous on this corpus.** GWN ambiguous-cell fraction was 0.0% on 15/16 meshes (0.1% on the 16th). AI meshes are open-shell soup, but their orientation is locally consistent — there is essentially one right answer, and
- **PrusaSlicer finds it.** Slicer-repair-of-raw vs GWN-truth filament differed by **0–2.4% on 15/16 meshes**. One real divergence: `meshy/sdxl_047_stapler_v3` (236 components, 73 open edges) — slicer 670 mm3 vs GWN 400 mm3, **+67%**. That is 1 in 16.
- The actor that IS demonstrably wrong is the **blind voxel fill**: +5% to +73% filament over GWN on ordinary meshes and **36x** on the stapler (14,650 vs 400 mm3). That validates removing our own crude reseal (F1) — it does not indict the slicer.
- Sensitivity caveat: 100%-infill slicing failed config-free ("fill pattern not supposed to work at 100% density"), so comparisons used the default profile (~15% infill + perimeters), which dilutes internal-volume differences. It was still sensitive enough to show the 67% stapler divergence clearly, but small internal errors could hide below ~2%.

**Verdict: the "slicer guesses wrong" framing is dead on this evidence.** Say so and move on.

**What survives, honestly:**
1. **Transparency is a trust/audit feature, not a correctness rescue.** "Here is a certificate: 99.4% of your surface untouched, max deviation 0.09 mm, here is exactly what was filled" is a *supporting* feature that the slicer genuinely does not offer — but it is not a headline differentiator built on slicer mistakes.
2. **The one real, visible divergence is intent, not error:** our faithful fix preserves thin shells and internal structure, so it prints at median ~0.56x the filament of the slicer's chunky auto-repair (range 0.33–1.08). Neither is provably "wrong" without knowing what the user wants — a certificate line that says "N internal shells kept as cavities; expect ~X% of the slicer's material estimate" turns this from a surprise into a feature. Whether users value that is **untested**.
3. Stapler-class meshes (hundreds of components, heavy open soup) are the only class where the slicer's guess measurably diverges. Prevalence of that class in the wild is unknown — one case here is an anecdote, not a market.

---

## 3. The functional-parts wedge verdict (graded infill)

**The question:** does FEM-driven graded infill (base 15% / core 80% at the p70 stress region) save material vs the cheapest *uniform* infill with the same comparative safety factor?

**The test:** 10 real functional thingi10k parts (brackets/mounts/clamps), 90 mm, documented load case each, solid FEM (comparative Tsai-Hill FOS), then: find the lowest uniform density in {20,30,40,60,80}% whose FOS >= the graded design's FOS under two strength-density exponents (n=1.0 and n=1.5); the verdict uses whichever is *worse* for us per part. Both designs sliced in PrusaSlicer 2.9.6 (per-volume 3MF modifiers verified honored). FOS here is comparative only — it ranks designs under one solver, it does not certify parts.

| part | what it is | stress gradient (out/min FOS) | matched uniform | material saving | time saving |
|---|---|---|---|---|---|
| 57254 | angled saddle mount | 14.7x | 80% | **+34.5%** | **+124 min** |
| 56979 | gusseted shelf bracket | 10.8x | 80% | **+19.9%** | **+11.5 min** |
| 45616 | rail mount plate | 1.0x | 20% | −3.2% | −1.0 min |
| 78661 | U-channel bolt bracket | 1.0x | 20% | −5.7% | −11.1 min |
| 186560 | tall pivot mount | 1.0x | 20% | −12.0% | −23.1 min |
| 917937 | U-clamp plate with posts | 1.0x | 20% | −14.2% | −9.3 min |
| 61762 | X-braced frame plate | 2.4x | 30% | −14.9% | −16.9 min |
| 48342 | clamp mount w/ tube boss | 3.6x | 40% | −15.5% | −56.1 min |
| 120755 | corner/T bracket | 1.6x | 30% | −20.5% | −24.8 min |
| 46263 | block bracket | 1.2x | 20% | **−41.5%** | −63.6 min |

**Median: −13.1% material, −14 min. Parts saving >= 15%: 2/10. Verdict: `wedge-marginal`.**

Plain reading: on 8 of 10 real parts, the graded design is *heavier* than the uniform infill that already matches its safety factor. The mechanism is clean: graded wins only when the part's stress gradient exceeds the strength ratio of the recipe (0.80/0.15 ≈ 5.3x) — i.e. steep-gradient cantilever-like parts, where it wins big (+20% to +35%, up to 2 hours of print time). Half the corpus matched a mere 20% uniform: typical printed brackets are simply overbuilt, the kill scenario the brief warned about — it happened.

**The market number for graded infill is therefore: not a blanket saving; a ~1-in-5 targeted feature worth +20–35% when a cheap pre-screen says the gradient is steep enough.** Sold as universal, it is a regression machine.

Side-finding (product-relevant bug, found by this test): `reinforce`/FEM on mm-unit meshes compares mm-traction (numerically MPa) against Pa allowables — without an explicit 1e6 unit bridge every FOS clips at the ceiling and the importance field goes flat. This needs a fix in the product, not just in the test harness.

---

## 4. What's next — one action per surviving thread, and what to stop

**Fixer (F1/F2, healthy):** Gate `gwn_solidify` by *outcome*, not input genus — accept its result only if surface-kept stays above the fidelity floor; otherwise ship the clean-fill result. That deletes the run's only bad trade (origami crane, 100% -> 62%) and costs nothing elsewhere. Second (cheap, high leverage): add the internal-shell line to the certificate ("N closed internal shells kept as cavities; expected material ~X% of slicer auto-repair") — it converts the largest remaining user-visible surprise (median 0.56x filament) into reported behavior.

**Transparency pitch (reframed, not dead):** Reposition the certificate as audit/trust ("we prove we didn't touch your surface"), not slicer-error correction. Before ever reviving the error-based pitch, measure the prevalence of stapler-class meshes (100+ component soup) in a larger corpus — one divergent mesh in 16 is an anecdote.

**Graded infill (marginal, salvageable):** Build the pre-screen: run the solid FEM, compute the gradient ratio (fos_out_min / fos_global_min), and only offer grading when it exceeds ~5.3 (the recipe's strength ratio) — recommend plain uniform otherwise. Fix the FEM unit bridge in `reinforce` while in there. Re-run the wedge on pre-screen-passing parts only; if the win-rate-when-offered isn't near 100%, kill the feature.

**Stop doing:**
- Selling "we make it printable" — PrusaSlicer does it silently, free, 10/10 on raw broken meshes.
- Claiming the slicer's repair is wrong — measured false on 15/16 here.
- Offering graded infill without the gradient pre-screen — median outcome is negative.
- The crude voxel reseal as a routine path (already demoted to last resort; it fired once in 66 meshes).
- Trusting positional args into trimesh APIs in harnesses (`simplify_quadric_decimation` bound 60000 to `percent` and silently no-opped decimation on 49/75 meshes — use keywords).

---

## Evidence

All under `C:\Users\mecht\AppData\Local\Temp\claude\c--Users-mecht-Project-EI\5dfbd365-cabf-4007-a311-3f2d4a932bd4\scratchpad\` unless noted:
- F1 autopsy: `adjust\autopsy_f1.py`, `adjust\autopsy_f1.log`, `adjust\autopsy_f1_results.json`
- F2 repro: `adjust\diag2_repro.py`, `adjust\diag2\diag2_results.json` (+ STL/gcode pairs)
- F3 slicer-truth test: `adjust\d3_run.py`, `adjust\d3\d3_results.json`, `adjust\d3\d3_100.json`, `adjust\d3\d3_ambiguity.json`, `adjust\d3\d3.log`
- Re-benchmark: `adjust\rebench.py`, `adjust\rebench_results.json`, `adjust\rebench_comparison.md`, `adjust\rebench.log`; old baseline `ai_bench_results.json`
- Wedge: `adjust\wedge_pipeline.py`, `adjust\wedge_results.json`, `adjust\wedge_log.jsonl`
- Code: `C:\Users\mecht\Project_EI\3DEI\meshprep\src\meshprep\core\_fix_accurate.py` (upgraded, API unchanged), `...\core\_solidify.py` (new); `license_guard.py src` = CLEAN.
