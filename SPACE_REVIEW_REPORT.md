# SPACE_REVIEW_REPORT — Packaging the Hugging Face Space around the novice review document

Date: 2026-07-05. Scope: `3DEI/meshprep/` — `src/meshprep/{report,app,pipeline,profiles}.py` + new `space/` package. Record of: first-ever app launch (recon), design spec for the novice review page, three-territory implementation, two independent verification passes (both PARTIAL), and two fix rounds (second round closed all open items on this box; Space hardware itself remains unmeasured).

Honesty discipline throughout: every physics number is labeled (uncalibrated / estimate / comparative), the review document simplifies language but never upgrades a claim, and refusals are stated as refusals. The fixer's trust line — **"we touched X% of your surface, max deviation Y mm"** — is the central promise and appears in the verdict badge, Section 2 of the review, and the shop block.

---

## 1. What the Space now is

A single-purpose Gradio page (gradio 6.19.0, pinned): a novice drops in a 3D model (.glb/.stl/.obj/...), optionally picks a printer and print size, and presses one button. The app keeps the file at its own size by default (resizing is opt-in, with a unit-guess disclosure when the file doesn't declare units), runs the meshprep pipeline under a purpose-built `space` preset (20k-face analysis proxy, fast re-check, ~120 s soft budget, warp FEM capped at 1500 elements and opt-in only), and returns three things: a plain-words verdict banner whose first words are one of READY TO PRINT / NOT READY YET / WE COULDN'T READ THIS FILE, with the trust line ("We touched X% of your surface, max deviation Y mm") directly in the badge; an 8-section novice review page (verdict → what-we-changed receipt → size check with everyday-object anchor → before/after/support renders → ranked "Do one thing" warning cards → download + copy-pasteable print-shop note → opt-in warp check → collapsed fine print with the no-guarantee line); and exactly two downloads, the repaired STL and the review itself. Garbage input is refused in under a second with "we changed nothing," no downloads, and no traceback. Nothing on the page claims to be a slicer simulation; the 45° rule and thickness checks are labeled geometry rules of thumb, and the optional warp number is welded to the word "uncalibrated" in the same sentence.

---

## 2. The review document — one full real sample

Below is the **verbatim render for the novice scenario** (broken Meshy `axe.glb`, 166k faces, non-watertight, 63+ components), taken during the independent verification pass (`scratchpad/space_review/verify/review_1_novice_axe.md`). It is pasted as-is because it is the honest verification-time record. **It predates the two fix rounds**, so it still shows the defects the verifier flagged and the fix rounds subsequently repaired — annotated after the sample.

~~~markdown
> ⚠️ **NOT READY YET** — one thing needs your decision.
> Zero-thickness geometry detected -- parts of this model are infinitely thin sheets, not solid walls.
> We repaired everything we safely could — your receipt is in the next section — but this one is a choice only you can make. See item 1 below for your options.

---

## What we changed — and what we didn't

**100% of your surface is exactly as you uploaded it.** Where we made repairs, the new surface is never more than **0.003 mm** from your original — a quarter of a human hair.

Your model had 1619 small gaps in its surface — like pinholes in a balloon. Printers need a fully sealed shape, so we sealed them.
What we did **not** do: no smoothing, no reshaping, no detail reduction, no rescaling beyond the size you chose. The overall shape and details are unchanged — we verify this by measuring, not by eye.

One honest note: the repair sealed the gaps but did **NOT** fix Even after the automatic fix: Zero-thickness geometry detected -- parts of this model are infinitely thin sheets, not solid walls. — that needs a decision from you (see item 1 below). We don't change your model's shape automatically.

Your original upload is never modified; everything we produce is a new file.

## Size check — is this the size you meant?

Your model will print **54.4 mm** at its longest — about about the size of an egg.
Full size: **3.8 × 48.6 × 54.4 mm** (all measurements in millimetres).

Heads-up: your file didn't say what units it uses (most AI-generated files don't), so we assumed millimetres — the safe, standard guess.

If "about the size of an egg" sounds wrong, don't worry — nothing is broken, and every warning below depends on the size, so fix this FIRST. One thing to do: tell the print shop the real size you want (for example "make it 15 cm tall") — resizing is normal, safe, and takes them seconds.

## See it — before and after (pictures)

![before](C:\Users\mecht\AppData\Local\Temp\meshprep_ya_r3qk4\premortem.png) ![after](C:\Users\mecht\AppData\Local\Temp\meshprep_ya_r3qk4\after\premortem.png) ![support zones](C:\Users\mecht\AppData\Local\Temp\meshprep_ya_r3qk4\after\premortem.png)

**Left:** exactly what you uploaded. **Right:** the file you'll print, shown in the printing position we've already applied. They should look identical — the repairs are smaller than your screen can show (the receipt above is the measurement behind that).

**Third picture:** the red areas are where the printer will build temporary scaffolding to hold up steep parts while printing. The scaffolding is removed afterwards but can leave small rough patches in those red spots — don't expect them to be glass-smooth.

We already rotated the model to the position that needs the least scaffolding — from 34% of the surface down to 4% (estimated with the standard 45° steepness rule of thumb, not a run through real printer software). That rotation is saved into your download.

## Things to know before you print (most important first)

**1. (needs your decision) Zero-thickness geometry detected -- parts of this model are infinitely thin sheets, not solid walls.**
*What this means for your print:* A printer cannot make material with no thickness; those regions would simply be missing from the print. Scaling the model up cannot help -- zero stays zero at any size.
*Do one thing:* Give the sheet regions real thickness in a modelling tool (or re-export the model as a solid).

**2. (please check) -696 unexpected tunnel/handle(s)**
*What this means for your print:* The surface has handles (genus > 0). Often these are phantom tunnels from the source geometry, not real holes you intended.
*Do one thing:* Inspect the heatmap; the Fix can also remove thin phantom bridges.

**3. (please check) About 4% of the surface needs support material in the print orientation applied to the downloaded file (down from 34% as uploaded).**
*What this means for your print:* Steep overhangs past the ~45° self-support angle need support material and will be scarred where supports touch.
*Do one thing:* Print it in the applied orientation; the support render shows where supports will touch and leave small marks.

**4. (please check) the model is made of 547 separate pieces; if you did not intend a multi-part model this often means the mesh is shattered -- check before printing.**
*What this means for your print:* It can make the print unreliable or wrong.
*Do one thing:* Fix this in a modelling tool, or re-export the model from its source program.

**5. (please check) this part's longest side is only 1.9 in file units -- that looks like METRES (about 1000x too small for mm). A real print is ~5-300 mm, so it will be rescaled to a 60 mm default GUESS. Set the print size (or assume_unit='m') to set the true size.**
*What this means for your print:* The part could print at the wrong physical size.
*Do one thing:* Check the printed size shown in this report and set the print size explicitly if it is wrong.

**6. (good to know) About 4% of the surface will need temporary scaffolding, mostly on downward-facing surfaces around the middle of the part.**
*What this means for your print:* Small rough patches where the scaffolding touched — the red zones in the picture above.
*Do one thing:* Nothing; the shop handles this. Just don't expect those spots to be glass-smooth.

## Your file + a note for the print shop

**[ ⬇ Download your repaired file — axe_print_ready.stl (1.4 MB) ]**

Because the verdict above is "Not ready yet", the shop will hit the same problem we found (Even after the automatic fix: Zero-thickness geometry detected -- parts of this model are infinitely thin sheets, not solid walls.) — settle item 1 first, or expect the shop to call you.

Send this note along with the file — copy-paste it into the order form; you don't need to understand it, the shop does:

```
File: axe_print_ready.stl — binary STL, units: millimetres
Dimensions: 3.8 × 48.6 × 54.4 mm. Scale is intentional — please print at this size, do not rescale.
Watertight: YES — single manifold solid, verified after repair. Consistent normals: yes.
Repairs performed: 1619 boundary holes filled; 100% of surface unmodified (0% modified), max deviation 0.003 mm vs. source (two-sided Hausdorff); original detail otherwise untouched.
Orientation: file is saved in a support-minimising orientation — please print as-oriented.
Suggested process: FDM, PLA, 0.4 mm nozzle, 0.2 mm layers (Generic FDM (200 mm cube))
Known risks: zero-thickness shell regions (needs CAD repair); supports on ~4% of the surface
Prepared by meshprep — checks are geometric (45° overhang rule, direct thickness measurement), not a slicer simulation.
```
**[ Copy to clipboard ]**

## Optional: will it warp? (takes about a minute)

**[ Run warp check — about 1 minute ]**
This runs a rough physics simulation of your print cooling down. What you'll get back is a **comparison, not a promise**: it ranks which parts of your model are more likely to curl or lift than others. It cannot predict exact millimetres — the numbers are uncalibrated estimates.

<details>
<summary><strong>The fine print</strong> (tap to expand)</summary>

**How honest are these numbers?**
**Measured:** the receipt in "What we changed" — the percentage of your surface we kept and the maximum deviation — is computed directly on your repaired file against your original — it is not an estimate.
**Estimates:** wall thickness, scaffolding coverage, and the printing angle come from standard geometry rules of thumb (the 45° steepness rule, direct thickness measurement), not from a run through real printer software — your shop's slicer software (the program that prepares files for their printer) has the final word.
**Uncalibrated:** the optional warp number — trust its ranking of risky spots, not its exact values.
The size check assumed millimetres because your file didn't specify.
For speed, we analysed a lightly simplified copy of your model (166,563 → 19,981 facets); the file you download is repaired at full quality.

Nothing on this page is a guarantee your print succeeds — it is our honest best reading of your file, and we've told you plainly where it still needs a decision.
Want every raw number, timing, and measurement method? **[ Download the full technical report ](C:\Users\mecht\AppData\Local\Temp\meshprep_ya_r3qk4\report.md)**

</details>
~~~

**What this sample gets right** (verified): the verdict banner comes first with plain words and no failure counts; the trust line and its shop-block echo carry identical numbers; the deviation has a physical anchor ("a quarter of a human hair"); the residual failure is stated, not hidden behind a success mark; every card has exactly one action; the zero-thickness card correctly says scaling cannot help (no "print a 24-metre axe" advice); the shop block includes do-not-rescale, watertight YES/NO, and the "not a slicer simulation" close; warp is "a comparison, not a promise... uncalibrated"; the fine print carries the no-guarantee line and the facet-simplification disclosure.

**Defects visible in this sample, all fixed in fix rounds 1–2**: (a) trust-line contradiction — "100% of your surface is exactly as you uploaded it" beside "1619 small gaps... we sealed them" (rounding now floors at 99.9% / 0.001 mm so a repaired model never shows the unrepaired form); (b) the double-prefixed residual sentence "did **NOT** fix Even after the automatic fix:..." (numberless restatement now); (c) server temp paths as image and technical-report links (renders now embedded as data URIs; dead links dropped); (d) "-696 unexpected tunnel/handle(s)" (negative genus now renders a plain no-number card, and the upstream count was fixed); (e) raw engine jargon in cards 2/4/5 — "genus", "mesh is shattered", `assume_unit='m'` (canonical plain cards + jargon scrubber); (f) "about about the size of an egg"; (g) the support-zones picture being the same file as the after render (disambiguated).

---

## 3. First-launch recon: what we found, what was fixed

This was the **first time the app was ever launched**. It launched successfully: server ready 10.7 s after spawn, four novice-scenario jobs driven headless via gradio_client, strictly sequential, peak server-tree RSS 1124 MB, clean teardown (port empty, pid dead). Garbage input was already rejected cleanly in 0.2 s, and the FEM warp opt-in already held its ~1 min budget with honest "(est., uncalibrated)" labels.

Everything else needed work. The recon blockers, and their dispositions:

| # | Recon finding | Disposition |
|---|---|---|
| 1 | **Clean parts never PASS** — a verified-watertight control reported "[FAIL] Not watertight". Root cause found by experiment: GLB UV/normal seam-split vertices read as open edges under `process=False`. | FIXED (pipeline): positional seam weld, adopted only when open edges strictly decrease, positions untouched — a measurement correction, not a repair. Plus a source-watertight re-probe so simplification-introduced holes are never blamed on the user. Verify then exposed a second false-positive layer (thin-wall gate reading edge artifacts as 0.02–0.30 mm walls on real CAD parts); fixed in round 1 with an exit-face-alignment ray filter and area-supported minimum wall. Post-fix: real-corpus clean exemplar 104421.glb renders READY TO PRINT; a 40 mm cube PASSes in 4.1 s. The 0.3 mm-plate must-fire selftest still fires. |
| 2 | **11.3-minute novice job** (analyze 238.9 s + full re-check 394.8 s, zero progress feedback). | FIXED across all three rounds: `space` preset (20k-face analysis proxy, fast re-check, ~120 s soft budget, cost-aware render gating), generator handler for immediate progress, embreex 4.4.0 pinned for ray casting (analyze 202→105 s on the worst part), then round 2 removed mechanical waste (trap census voxelized 4× per job, each figure drawn 3×) — worst-part UI time 421.4 s → 71.5 s, verdict byte-identical. |
| 3 | **Silent rescale of every upload to the 60 mm slider** — the novice's "secret changes" fear built in as default. | FIXED (app): keep-original-size is now the default (`print_mm=None`); Resize is an opt-in checkbox with a File-units override; the badge states KEPT vs GUESSED vs as-you-set, and the review's Size check section discloses any assumed-millimetres guess. |
| 4 | Trust line buried mid-report next to a contradictory "Free fix applied: FAIL (unchanged)". | FIXED: trust line is in the verdict badge and Section 2; the contradiction was replaced by the mandatory residual-failure sentence. |
| 5 | Engineer-facing report (Hausdorff, genus, CLI flags, temp paths, stage table) shown to novices; FAIL/PASS codes; "drop the --check flag" advice in a UI with no flags; "install PrusaSlicer" instruction to a browser user; jargon gallery captions ("premortem", "traps"). | FIXED: new `build_review()` renders the 8-section novice page; jargon confined to the shop block by a mechanical `review_selfcheck()` (banned-word regex with shop-block/warp exclusion masks); slicer-savings stage skipped entirely under the preset; captions humanized; badge headlines routed through the same scrubber (round 2). |
| 6 | "Scale up ~400×" advice from a 0.00 mm wall reading. | FIXED: zero-thickness branch — "scaling cannot help, zero stays zero"; `scale_factor_suggested` hard-forced to None. |
| 7 | Support-number self-contradiction (93% vs 34.2%→4.6%). | FIXED: warnings show only the post-orientation figure with "down from X% as uploaded" provenance; the 93% was a combined-risk channel, no longer used for supports. |
| 8 | Phantom sealed-cavity readings (new false-FAIL found during work: ~130 cm³ "cavity" on a bumpy star solid). | FIXED: demoted only when three independent exact checks agree (watertight + single body + zero internal shells), with a self-explaining note — a false-measurement correction by strictly more reliable checks, never a claim upgrade. |

Recon artifacts: `scratchpad/space_review/{results.json, baseline_report_a.md, report_b.md, report_d.md, server.log, drive.py}`. No fixes were applied during recon, per instructions.

---

## 4. Verification (two independent passes, both PARTIAL at the time; fix rounds followed)

Budget context: target hardware is a free HF Space, ~2 vCPU / 16 GB; local verification ran on a faster 8-core box, so local timings are a **lower bound** on Space timings.

### Pass 1 — live UI end-to-end (gradio_client against the real Space app)

| Check | Pass | Evidence |
|---|---|---|
| Env + pins | YES | gradio 6.19.0 / gradio_client 2.5.0 installed = exactly the pins in `space/requirements.txt`; system Python 3.11 |
| Memory discipline / teardown | YES | One server at a time, jobs sequential, tree-killed in finally; psutil: no port holders, pid dead; peak server-tree RSS 745 MB max across 11 jobs (budget 2500 MB); largest input 8.3 MB (< 25 MB cap) |
| Novice broken GLB (axe) | NO (then) | Review rendered banner-first, trust line central in badge AND review AND shop block; downloaded prep.stl re-loads watertight in trimesh (26,974 faces). But 5 selfcheck violations (temp-path links, jargon, duplicate numbers) and 190.4 s > 120 s budget — both fixed in rounds 1–2 |
| review_selfcheck on rendered pages | NO (then) | Every real render failed its own checker (temp paths, genus/mesh/overhang jargon, numeric repetition) — root cause was the PrepResult→review field mapping, not the template; fixed round 1, all four real-corpus cases selfcheck-clean after |
| Clean part → PASS | NO (then) | 6 real thingi10k parts: 0 PASS; sub-0.1 mm "thinnest wall" readings on watertight CAD parts = edge-artifact false positives; fixed round 1 (104421 → READY TO PRINT; 8-part scan 0.58–110.7 mm readings, one honest FAIL at 0.278 mm, zero false thin-wall FAILs) |
| Garbage .stl refusal | YES | 0.2 s; plain "We can't read this file" badge; no downloads; zero traceback text in any output |
| Unit-suspicious file (metres) | YES | Size sanity is the badge headline plus a dedicated Size check section with exactly one action (cosmetics fixed later) |
| Opt-in warp physics | YES | 19.7 s, peak 609 MB; labeled "uncalibrated estimate... trust the comparison between regions, do not trust the raw number"; no claim upgrades |
| FEM cap ≤ 2500 elements | NO (then) | UI path never passed `preset='space'`, so warp ran at default max_elem=6000 — the root cause of both this and the budget miss; fixed round 1 (`MESHPREP_PRESET=space`, cap 1500 active) |
| Per-job wall < 120 s | NO (then) | garbage 0.2 s, clean 17.5 s, unit 17.9 s, warp 19.7 s OK; axe 190.4 s and 1013014.glb 421.4 s breached; after rounds 1–2: worst part 71.5 s UI end-to-end, axe-class within budget, exemplar 16.7 s — **on this box**; 2 vCPU Space unmeasured |
| Honesty discipline | YES | All physics numbers stayed labeled; trust line central; refusals plain; savings/reinforce absent; no claim upgraded anywhere |

### Pass 2 — packaging, review quality, consumer regression

| Check | Pass | Evidence |
|---|---|---|
| Wheel matches source | NO (then) | CRITICAL: shipped wheel was stale — built before the space work, missing the entire `space` preset; rebuilt and byte-verified during the pass; round 1 added `space/check_wheel.py` freshness guard (verified to fail on the stale wheel) wired into DEPLOY.md |
| Clean-venv requirements complete | YES | Fresh venv from `space/requirements.txt` only: zero missing deps; `prep(preset='space')` + `build_review` ran end-to-end; pins match the live environment |
| Real-corpus review selfcheck | NO (then) | a:4 / b:5 / d:5 violations (rejected case: 0); raw engine strings leaking through the field mapping; fixed round 1 — all cases clean, and the checker itself verified to FIRE on 6 adversarial controls |
| Adversarial jargon audit | NO (then) | Verbatim leaks quoted (genus, "mesh is shattered", unglossed overhang, "single extrusion", `assume_unit` API advice); fixed round 1 (page) + round 2 (badge headlines) |
| Honesty and coherence audit | NO (then) | Trust-line 100%/0 mm shown for repaired models; broken residual sentence; "-696 tunnels"; duplicate support image; dead `(None)` report link on refusals — all fixed round 1 with downgrade-safe rounding floors and selfcheck rules for each |
| Regression: CLI `check` | YES | HEAD-vs-working on 104421.glb: report.md identical except two timing cells; report.json keys strictly additive (`review.*` only); `build_report` byte-path untouched |
| Regression: license guard | YES | 0 banned copyleft/non-commercial imports; 4 pre-existing opt-in pyQuadriFlow WARNs, unchanged |
| Regression: py_compile | YES | All four src modules + `space/app.py` compile |

### Resource summary vs budget

| Metric | Measured (local, 8-core, 16 GB) | Budget |
|---|---|---|
| Peak server-tree RSS, any job | 745 MB (verify) / 1124 MB (recon, pre-preset) | 2500 MB (16 GB box; Space has 16 GB) |
| Worst job wall, post-fix, UI path | 71.5 s (1013014.glb); exemplar 16.7 s; PASS cube 4.1 s | 120 s soft budget — held locally; **2 vCPU Space not measured, expect a multiple** |
| Refusal latency | 0.2 s | — |
| Warp opt-in job | 19.7 s, 609 MB, max_elem 1500 under preset | ≤ 2500 elements — held |
| Largest input tested | 8.3 MB | 25 MB launch cap |

---

## 5. What is NOT in the Space, and why

Trimmed or excluded deliberately — the code paths remain in the package; only the Space surface is cut:

- **Slicer savings**: on a hosted Space it would permanently read "no slicer installed — install PrusaSlicer", an operator instruction a browser user cannot act on. Stage skipped entirely under the preset; Savings panel removed.
- **Reinforce / graded-infill 3MF + Load direction**: FEM strength work is outside the fixed slice; checkbox, dropdown, and the 3MF/gcode download slots removed.
- **Retopo remnant** (`quad_obj` download slot): feature not in the slice; removed.
- **Autorig mention** ("animation / rig-readiness ships separately"): a roadmap line, not a capability of this page; removed.
- **Resin mode**: explicit trim decision — the resin workflow is not in the fixed slice (cavities-kept honesty rows remain); resin printer profiles filtered from the dropdown.
- **Stage/timing tables, temp paths, CLI flags, FAIL/PASS codes, Hausdorff/genus vocabulary**: engineer-facing material moved to the downloadable technical report or the shop block; banned from the novice page by the mechanical selfchecker.
- **PDF/PNG download row**: dropped to keep one terminal action (the STL + the review); renders are already on the page.

---

## 6. Honest residuals, and the deploy steps that belong to the user

### Residuals (nothing here is claimed fixed)

1. **The Space's actual hardware is unmeasured.** All timings are from an 8-core local box. DEPLOY.md states the measured local numbers (71.5 s worst UI job, 16.7 s exemplar) and explicitly says 2 vCPU Space hardware is NOT yet measured — expect a multiple. The 120 s soft budget will skip optional stages honestly (with per-stage notes) if the Space is slow, but the mandatory analyze/fix/re-check path could still exceed a request timeout on the worst uploads. First deploy should be treated as a measurement run.
2. **The thin-wall gate is improved, not proven.** The edge-artifact filter eliminated the observed false positives on an 8-part scan and the must-fire selftest still fires, but the corpus is small; direct thickness measurement remains a geometric estimate, labeled as such on the page.
3. **Physics labels are load-bearing.** Warp is uncalibrated (trust the ranking, not the number); support coverage and orientation are 45°-rule estimates; material-savings figures are geometric estimates and the slicer-quote comparison is excluded from the Space entirely. None of these were upgraded, and the selfchecker enforces the labels mechanically — but the selfchecker is our own tool, not an external audit.
4. **Local sample pages in this report predate the fix rounds** (Section 2 annotations list exactly what changed). Post-fix pages passed `review_selfcheck` on all four real-corpus cases plus adversarial controls; they were not re-audited by an independent verifier after round 2.

### Deploy steps — user-owned (publishing is outward-facing)

Claude does not publish. The Space directory is ready at `C:\Users\mecht\Project_EI\3DEI\meshprep\space\` (README with HF frontmatter pinned to gradio 6.19.0, thin `app.py` entry, fully pinned `requirements.txt` including embreex 4.4.0 and the local wheel, `DEPLOY.md`, `wheels/meshprep-0.1.0.dev0-py3-none-any.whl`). Before publishing:

1. Run `space/check_wheel.py` — the wheel-vs-source freshness guard. It was proven to fail on the stale wheel that nearly shipped; never deploy without it passing.
2. Follow `DEPLOY.md` for the HF publish steps (create the Space under your account, push the `space/` contents, free CPU tier).
3. On first deploy, run the smoke matrix named in DEPLOY.md yourself: the 40 mm cube PASS exemplar, the 104421.glb real-corpus clean exemplar (READY TO PRINT, caution cards only), the broken axe (NOT READY YET with residual line), and a garbage file (plain refusal). Record the Space-hardware timings — they are the missing measurement.
4. Fill the GitHub footer placeholder in the app if/when the repo is public.
5. The 25 MB / 20k-face caps are set for the free tier; raising them is a hardware decision, not a code change.

### Artifact index

- Source: `C:\Users\mecht\Project_EI\3DEI\meshprep\src\meshprep\{report,app,pipeline,profiles}.py`
- Space package: `C:\Users\mecht\Project_EI\3DEI\meshprep\space\{README.md, app.py, requirements.txt, DEPLOY.md, check_wheel.py, wheels\meshprep-0.1.0.dev0-py3-none-any.whl}`
- Recon + verification evidence: `scratchpad\space_review\` (results.json, verify\results_verify.json, verify\review_*.md, rv2_*.md, rv2_summary.json, reg_head/reg_cur report.json pair, server logs)
