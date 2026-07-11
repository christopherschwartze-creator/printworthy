# Trust Map — Ship Report

**Feature:** per-face "seen vs invented" trust map for the wireframe→print fixer
**Date:** 2026-07-10
**Source of record:** handoff brief (`C:\Research\Paper\Briefs\meshprep_trust_map_handoff.md`), provenance PoC (`Applied/Quasi/provenance/trackA`), and the meshprep build/verify/fix cycle (commits `01cbb1c`, `716f017`, `59b8215` + two fix rounds).
**Honesty rule (the law):** the **occlusion anchor** is a proven geometric principle — it ships. The **geometry-only head** is unproven — it does **not** ship (research only). Labels are about *visibility*, not correctness of the visible parts; near-silhouette faces are borderline and expose a soft confidence, never a crisp overclaim. Without a source image + camera pose the feature is **ABSENT**, never faked.

---

## 1. What shipped — the trust map as a user experiences it

A user who uploads an AI-generated mesh **and supplies the source photo's camera** gets a fourth, opt-in output channel alongside the repaired mesh, the print report, and the FEM/strength estimate: a green/orange/blue overlay and a percentage that say which parts of the model the camera actually saw versus which parts the generator invented. The proven **occlusion anchor** drives it — a single embree z-buffer/raycast pass on the delivered mesh from the supplied camera marks a face `ai_invented` if it is back-facing, occluded behind other geometry, or outside the view frustum, and `evidence_backed` otherwise. Faces added by the fixer are disclosed as a third label, `repair_filled`, by construction. Grazing/silhouette faces carry a **soft confidence** and are flagged rather than asserted, so the rim of the model never reads as crisp certainty. The feature is **FREE and public** (it is the honesty-brand headline — "we tell you which parts the AI invented"), it runs sub-second on real corpus meshes (a 166k-face mesh in 0.74s), and it is **entirely absent** — no faked pose, no channel, an honest "requires the source image plus its camera pose" note — whenever a camera is not supplied. Because FEM stress on invented geometry is meaningless, the strength/reinforce channels gain a never-silent caveat ("Structural estimate rests on invented geometry: N% of the surface area is AI-invented") without any change to the FOS numerics.

**Exact novice-review wording shipped** (report `review.md`, jargon-clean per the review page's banned-word list):

> **Optional: what the source photo actually saw**
> About **60%** of the surface was invented by the AI (the camera never saw the back, undersides, or hidden parts — the generator filled them in from its guess). We made those parts printable and sound, but they can't be checked against your photo. If accuracy matters (a functional part, a replica, anything measured), review the invented regions. For a purely decorative print this usually doesn't matter.

The technical `report.md` carries a parallel `## Trust map` section (green = evidence-backed, orange = AI-invented, blue = repair-filled) plus the carry-mode disclosure (see §4).

---

## 2. Evidence table — real Meshy meshes

Independent oracle = a fresh embree `RayMeshIntersector` (centroid single-ray first-hit + front-facing + in-view), built in a separate harness, not the module's own code. "Agreement" = render-as-source: the anchor's label vs the oracle's on every face. Gate: **≥99% agreement AND all disagreements confined to the silhouette band.**

| Mesh (faces) | Agreement — pre-fix | Agreement — shipped (post fix rd.1) | Off-silhouette disagreements — shipped | frac_invented |
|---|---|---|---|---|
| sf3d / axe (5,876) | 99.95% | ≥99.9% | 0 | 0.50 |
| hunyuan / axe (54,672) | 99.93% | ≥99.9% | 0 | 0.50 |
| meshy / coffee_mug (465,645) | 99.78% | ≥99.9% | 0 | 0.78 |
| meshy / chess_knight (275,969) | **98.50% (FAIL)** | **99.96%** | 0 | 0.67 |
| meshy / teapot (235,217) | **97.29% (FAIL)** | **99.96%** | 0 | 0.59 |

**Root cause of the pre-fix failures (found, proven, fixed):** the first anchor reused a thin-wall *grazing-angle* guard (`_GRAZE_HIT_MIN = 0.34`) that only counted an occluder if its hit-normal was steep enough. On smoothly-curved high-poly rims viewed edge-on, true occlusions by the rim were discarded, so genuinely hidden faces were mislabeled `evidence_backed` **at confidence 1.0** — the exact over-claiming direction the honesty discipline forbids (100% of the 2,546 teapot disagreements were module=evidence / oracle=invented, 0 reverse; 2,504/2,546 stable under jitter = real occlusions, not oracle noise). **Fix (round 1):** replaced the angle-dependent guard with an angle-independent z-buffer **depth** test (nearest-hit `intersects_location`), and extended the silhouette/soft-confidence band to cover the previously-missing scene-silhouette region. Every remaining disagreement now falls **inside** `silhouette[]` and reads soft.

**Pose-perturbation sensitivity (the anchor's precision near silhouettes).** A 2°/5°/10° camera-orbit sweep was implemented (it did not exist before verification). Shipped mode uses an **explicit calibrated camera** so pose is exact and this is a robustness bound, not an operating error. Pre-fix, small numbers of label flips fell outside the silhouette band at full confidence. Round-1 fix cut this 13–47×; **round 2** added an opt-in, RAM-gated `pose_jitter` closure that re-measures a ±X° orbit and folds every actually-flipping face into the soft silhouette band, driving the 2° residual to **zero** on the meshes that had failed (teapot 9→0, knight 250→0 flips) *within the certified cone*. Larger-orbit flips remain a **disclosed, quantified residual** (e.g. knight 0.073% at 2° without the closure), reported honestly rather than force-passed.

**Controls (all green):** absent (no camera → `ok=False`, labels `None`, honest note); false-positive open sheet 0.0% invented; backside sphere 58.6% (one-view ~half hidden); **inter-object occlusion** — a front-facing face fully hidden behind another component is correctly `ai_invented` (the branch the PoC dataset never exercised, now a mandatory selftest control, 2/2); out-of-frustum all invented; label-transfer index-exact + reseal full-coverage; overlay caption present.

---

## 3. The head experiment (Lane C) — the brief's §5 question, answered

**Question (§5):** does the geometry-only head keep working on **unmarked** meshes, or was it secretly keying on the PoC's test-pattern texture? If it holds, mesh-only (text→3D, no source image) support is real; if it collapses, the source-image-required occlusion anchor is the true ceiling.

**Verdict: transfer HOLDS — with an honest reconstruction-quality caveat.** Eight paired marked/unmarked scenes were built (identical geometry/camera/RNG per seed, verified byte-identical ground truth; unmarked = a single flat fill replacing the QC fiducial texture). The head was retrained identically on marked scenes only and scored on unmarked held-out:

- Marked reproduction AUC = **0.9823** (byte-identical to `results/trackA.json` — pipeline fidelity confirmed).
- Unmarked-transfer AUC = **0.9507**, gap = **0.0316**. Both clear the pre-declared "transfer holds" bar (unmarked ≥ 0.80, gap ≤ 0.10) and are nowhere near "cheating" (≤ 0.60 or gap ≥ 0.25). Thresholds were declared **before** the run, not tuned to the result.
- Feature importances stay top-heavy on the same two *geometry-only* features (dist_to_border, depth_rel) in both marked- and unmarked-trained heads — consistent with a genuine height-field signal, not mark-detection.
- **Honest caveat (surfaced, not folded into the AUC):** removing texture measurably degrades the *underlying monocular reconstruction* — observable evidence rate drops 0.886 → 0.571, distorted vertices rise ~9× — a real Depth-Anything effect orthogonal to the transfer question. If anything it strengthens the verdict (the head tracked a harder, shifted distribution).

**What it means for the product.** The §5 gate is passed *in the lab*, so mesh-only support is a live future option rather than a dead end. **But it is not sufficient to ship the head**, and the brief's law is unchanged: only 8 synthetic scenes / 2 held-out; the head still Goodharts partly on a layout prior (dist_to_border); it was trained on the weak PoC depth-grid reconstructor, not Meshy output. Track A's status stays **SCOPED (audit-only)**. A future paid mesh-only tier is *contingent on re-validating this transfer on real Meshy meshes first* — until then the **source image is required** and the free occlusion anchor is the true ceiling. Findings appended as a dated §11 to `FINDINGS_provenance.md`; the existing verdict table was left unmodified.

---

## 4. Descoped / deferred — honestly

**Label transfer through the fixer (shipped, three regimes, honestly bounded):** labels {`evidence_backed`, `ai_invented`, `repair_filled`} carry from input faces to output faces. Transfer is **exact** for index-preserving local-patch repairs (labels carry by direct index; new faces = `repair_filled`) and for the cheap-repair subsequence (exact via captured masks). After a **global reseal** (winding-number/voxel — index-destroying) transfer is **approximate**: nearest-original-face inheritance, conservatively biased toward `ai_invented`/`repair_filled` near the evidence/invented seam and past a voxel pitch — it **never upgrades** invented→evidence via a reseal. The report names which carry mode ran (`report['trust_carry']`), so a resealed part never silently presents approximate labels as exact.

**Deferred / not shipped (with reasons):**
- **Pose ESTIMATION (render-and-match / PnP)** — the rung-b path — is **not built**; it ships gated behind `experimental_estimate=True` (default off) and currently returns ABSENT rather than a faked pose. It stays experimental until its perturbation bound is certified. This is a **maturity gate, not a paywall**.
- **Image-upload / pose input surface** into `prep()`, `cli.py`, and the Gradio Space UI (paired `gr.Image` slot) is **not wired** — v1 exposes the trust map as an opt-in stage that requires an explicit `trust_camera`, absent otherwise. This is the main plumbing residual.
- **Multi-view** consolidation: not attempted (single canonical source view only).
- **Volume-weighted** invented fraction: FEM disclosure says "% of surface **area**" because only the area-weighted fraction is computed (voxel-occupancy volume fraction deferred).
- **Minor UI gap:** the Space gallery thumbnail caption (`_GALLERY_CAP`) still shows an auto-humanized "Trust" label; the full honest disclosure prose reaches the user via the review panel on the same page.

**The production gap (state plainly to any user/stakeholder):** the shipped anchor is production-grade **only with an explicit calibrated camera**. Real users must supply their **source image + camera pose**; Meshy's canonical image-mode camera is **not embedded in any GLB/manifest/sidecar** (verified — no metadata exists), so until pose estimation is built and certified, the trust map is present only when the caller can hand meshprep a known camera. Value scales with accuracy-sensitivity: functional parts, replicas, measurement, and any structural/foolproof claim benefit; purely decorative prints may not care — and the copy says so.

---

## 5. Launch relevance

The evidence supports **one** honest launch one-liner for `LAUNCH_POSTS` (anchor-only; nothing about the head or mesh-only support):

> **New: a Trust Map for AI-generated models.** Upload your model and the source photo — we color every surface by whether the camera *actually saw it* or the AI *invented* it (the back, undersides, hidden parts). We still make the whole thing printable and sound; we just stop pretending the invented half was ever real. Free. Because "foolproof" should mean *never silently wrong*.

Do **not** claim mesh-only / no-photo support in launch copy — that is gated on re-validating the head transfer on real Meshy meshes.

---

## 6. Next steps

1. **Wire the input surface** — thread `trust_image` + optional explicit-pose args through `prep()`, `cli.py`, and a paired image slot in the Space UI; add the `_GALLERY_CAP['trust']` honest caption. (Unblocks real user adoption; largest remaining gap.)
2. **Build + certify pose estimation** (render-and-match / PnP), report its residual, and only then lift `experimental_estimate`. This is what makes the anchor usable when the caller can't hand over a camera — including for the Meshy canonical-camera unknown.
3. **Corpus-scale continuous validation** — promote the render-as-source oracle + pose-perturbation sweep from a one-off verify harness into a repo check gating ≥99.9% agreement + all-disagreements-in-silhouette on real Meshy meshes (the synthetic icosphere selftest missed the high-poly failures).
4. **Volume-weighted invented fraction** for the FEM disclosure line (currently area-weighted).
5. **Head, mesh-only tier (research → maybe):** re-run the §5 transfer test on real Meshy reconstructions (not the PoC depth-grid) with N ≫ 8 before any mesh-only claim; keep it out of the product until then.

---

## Executive summary

1. Shipped the **Trust Map**: a free, opt-in per-face overlay + % that labels an AI mesh as evidence_backed / ai_invented / repair_filled, driven by the **proven occlusion anchor** (embree z-buffer visibility from a supplied camera) — no AI, no training, no watermark.
2. Honesty law held throughout: the anchor ships; the unproven **geometry-only head does NOT ship**; labels are about *visibility*, not correctness; near-silhouette faces carry a **soft confidence**, never a crisp overclaim.
3. Without a source image + camera pose the feature is **ABSENT** (honest note), never faked — verified on the no-camera path (zero trust footprint anywhere).
4. Verification on **real high-poly Meshy meshes** caught a genuine over-claiming bug (a grazing-angle occlusion guard discarded true rim occlusions → hidden faces mislabeled "seen" at confidence 1.0; teapot 97.3%, knight 98.5%).
5. **Fixed at root** (angle-independent depth z-buffer + extended silhouette band): agreement rose to **≥99.9%** on all five corpus meshes, every remaining disagreement now inside the soft silhouette band.
6. Implemented the previously-absent **pose-perturbation sweep**; shipped mode uses an exact explicit camera, and an opt-in jitter closure drives the 2° residual to zero within the certified cone, larger orbits disclosed as a quantified residual.
7. FEM/strength channels gained a **never-silent caveat** ("estimate rests on N% AI-invented surface area") with **zero change to FOS numerics**.
8. Label transfer through the fixer is **exact** for index-preserving repairs and **approximate-but-conservatively-biased** (never invented→evidence) after a global reseal, with the carry mode disclosed in the report.
9. **Head experiment (brief §5) answered: transfer HOLDS** (unmarked AUC 0.9507, gap 0.0316, pre-declared bars) — mesh-only support is a live *future* option, but the head stays research-only (N=8 synthetic, layout-prior Goodhart, wrong reconstructor); source image still required.
10. Descoped honestly: pose **estimation** stays experimental-gated (maturity, not paywall), image-upload UI + multi-view not wired, volume-weighted fraction deferred; **production gap = users must supply source image + camera, and Meshy's canonical camera is unknown/unembedded.**
11. **Launch:** one honest anchor-only one-liner approved for LAUNCH_POSTS ("we tell you which parts the AI invented — free"); no mesh-only claim until the head is re-validated on real Meshy meshes.
12. **Next:** wire the image/pose input surface (biggest gap), build + certify pose estimation, promote the corpus oracle into a CI gate, then revisit mesh-only only if the head re-validates.
