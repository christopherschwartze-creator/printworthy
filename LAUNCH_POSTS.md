# Launch Posts — meshprep

> **These are DRAFTS. Posting is the user's action, not the tool's.** Nothing here goes live until
> you (a) deploy the Hugging Face Space, (b) capture the screenshots called out in the checklist, and
> (c) paste the drafts yourself. Every number below is copied verbatim from a measured report
> (`meshprep_pro/S2_FEATURES_REPORT.md`, `meshprep_pro/PRO_BUILD_REPORT.md`,
> `meshprep/FOOLPROOF_REPORT.md`, `meshprep/SPACE_REVIEW_REPORT.md`) with its honesty label intact.
> Do not round up, do not strip a label, do not add a benchmark win we did not measure. Replace every
> `[LINK-TBD]` before posting.

---

## Draft 1 — r/functionalprint (the physics angle)

**Title options:**
1. I built a free pre-flight that predicts warp, adds supports only where the risk model flags them, and grades material only when it actually changes the answer
2. Warp pre-compensation + risk-driven supports + rigid-where-loaded material split — and I can show you the measured deltas
3. A free, local, open-source print pre-flight that refuses to guess: every physics number is labeled uncalibrated or comparative

**Body (~285 words):**

I kept losing functional parts to warp and to supports in the wrong places, so I built a tool that
does the geometry-and-physics reading before the slicer. It is free, runs locally, and is
open-source. A hosted demo is at [LINK-TBD] and the repo is [LINK-TBD].

One concrete part: a 90 x 86 x 18 mm bracket that would not fit a 60 mm bed. The tool split it into
**7 printable, bed-fitting parts joined by 18 peg/socket connectors, volume conserved to 0.047%**,
and scored the seams to land in the concave necks (on a dumbbell control the seam lands at x = -3.48,
versus 12.5 mm to the nearest lobe). Honest caveat it prints on the page: those bisection seams are
labeled **VISIBLE**, not hidden-in-a-crease.

Supports are placed by a risk model, and the filament delta is measured from gcode, not asserted. An
enforcer under a real overhang added **+1.27 cm3 (+42.5%)** of support where the model flagged risk
(auto-detect was OFF the whole time); a blocker over a false-positive overhang removed it back to the
**exact 2.99 cm3 no-support baseline (-26.7%)**. A 40-degree chamfer cone correctly got **zero
enforcers** ("no supports needed").

Warp is a paid feature and stays honest about it: on a 60 x 8 x 3 mm bimetal PLA strip the *predicted*
released-part drift dropped from **0.0848 mm to 0.0075 mm after one inverse-warp pass** (monotone
toward zero after that). Every one of those millimetres is labeled **comparative / uncalibrated** —
it ranks and pre-compensates, it does not promise you an absolute number. A uniform-stress bar
(gradient 1.0x) is *refused*: "stress too uniform for a material split to matter."

**What I'd love feedback on:** does risk-driven support placement earn trust for you if the delta is
measured, or do you want to see it against your own worst overhang part? And is a warp number that is
honestly *comparative* useful to you, or worthless without a calibrated mm? Send me a part that warps
and I will show you the before/after prediction.

---

## Draft 2 — r/3Dprinting (the novice angle)

**Title options:**
1. Drop a Meshy/AI mesh in, get a plain-English verdict and a fixed file — and it tells you exactly what it touched
2. A free tool that reads your 3D file like a person and hands back "here's what I changed, here's the one thing you still have to decide"
3. I made the boring "is this file printable?" step foolproof for beginners — no jargon, never crashes

**Body (~280 words):**

If you have ever exported something from Meshy, Luma, or a scan and had no idea whether it would
actually print, this is for you. You drop the file in, pick a printer/size if you want, press one
button, and get back two things: a plain-words verdict (**READY TO PRINT / NOT READY YET / WE
COULDN'T READ THIS FILE**) and a repaired file to download. Free, in the browser, demo at [LINK-TBD].

The hook for me is the trust line. It tells you **exactly what it touched**. On a broken Meshy axe
(166k faces, full of pinholes, 63+ pieces) the receipt read: **0% of your surface reshaped, max
deviation 0.003 mm** — a quarter of a human hair — while it sealed **1619 tiny holes** so a printer
could read it as a solid. No smoothing, no secret reshaping, no silent rescaling. Your original file
is never modified; everything it makes is a new file.

It also refuses to lie to you. When part of that axe was an infinitely thin sheet, it did not pretend
to fix it — it said so in one sentence and told you it is a decision only you can make ("scaling
cannot help, zero stays zero at any size"). It writes you a copy-paste note for the print shop, too.

Under the hood it is built to never crash and never hand you a bad part while claiming success: across
**136 pathological inputs plus 53 more nasty ones, zero crashes and zero cases where it passed bad
geometry**. A broken shell comes back as an honest FAIL ("thinnest wall 0.01 mm < 0.40 mm nozzle"),
not a traceback.

**What I'd love feedback on:** is the "here's what I touched, to the 0.001 mm" receipt the thing that
would make you trust an automatic fixer? And what is the weirdest AI-generated mesh you can throw at
it before it says something confusing instead of something clear? Break it for me.

---

## Draft 3 — HF community / Show-HN style (the technical angle)

**Title options:**
1. Show HN: meshprep — an open-core 3D-print pre-flight with a fuzz-hardened never-crash contract and source-verified 3MF tricks
2. Open-core print pre-flight: FEM validated against a closed-form case, license-clean remesher, and a "never ships a bad part" gate
3. I source-read PrusaSlicer to get the 3MF support/multi-material schema right, then fuzzed the whole thing until it stopped crashing

**Body (~295 words):**

meshprep is an open-core pre-flight for 3D printing. The free core (MIT) does repair, printability
gating, oriented supports, and oversized-part splitting; a separate paid package adds warp
pre-compensation, quoting, and a license-clean remesher. Repo: [LINK-TBD]. Design notes below.

**Never-crash contract, fuzz-verified.** A watchdog harness drove **136 pathological cells plus 53
novel adversaries** (self-intersecting solids, genus-50 slabs, NaN vertices, 1e9-scale cubes): **zero
crashes, zero hangs, zero cases of PASS-over-bad-geometry**. The printability gate is authoritative and
downgrade-only — it can turn a PASS into a FAIL, never the reverse. A memory watchdog caught a real
O(F * n_faces) balloon in the thin-wall ray channel (it hit 5.8 GB in testing) and bounded it to
batches; batched output is bit-identical to single-shot.

**Source-verified 3MF, not guessed.** Before writing the support/multi-material path I read
PrusaSlicer 2.9.6 source: the volume types are `SupportEnforcer` / `SupportBlocker` (CamelCase,
singular — `Model.cpp:1399-1432`), and a per-volume `extruder` key rides through
`config.set_deserialize` (`3mf.cpp:2687-2714`). Measured pitfall baked into the code: an enforcer box
*exactly coincident* with the overhang produces byte-identical, functionally inert gcode — it must
intersect the solid by a few mm. All slicer effects are measured from gcode deltas, never asserted.

**FEM validated against a closed-form case.** The solidifier's generalized-winding-number field agrees
**100% with an analytic sphere**; warp pre-compensation drops predicted released-part drift **0.0848 ->
0.0075 mm** in one pass (labeled comparative / uncalibrated — no absolute-mm claim).

**License-clean by construction.** The paid remesher ships a QuadriFlow rebuilt with
`BUILD_FREE_LICENSE=ON`; `nm -C` on the shipped binary shows **26 MPL2 SparseLU symbols and 0 LGPL
SimplicialLLT** — the exact check that flags the LGPL blob.

**What I'd love feedback on:** the open-core line (free removes print blockers, paid monetizes the
FEM/manufacturing value-add) — does that split read as fair? And where would you attack the never-crash
contract next: 3MF/STEP as *input* (fuzzed only as output so far), or concurrent/server use (untested)?

---

## Posting checklist (user-owned)

1. **Deploy the Space FIRST.** Nothing posts with a dead `[LINK-TBD]`. Run `space/check_wheel.py`
   until it passes (it was proven to catch a stale wheel), then follow `space/DEPLOY.md` to publish on
   the free CPU tier. Treat first deploy as a *measurement run* — the 2 vCPU Space timing is unmeasured
   (local worst-job was 71.5 s; expect a multiple). Do not post links until the smoke matrix (cube PASS,
   104421.glb READY, broken axe NOT READY, garbage refusal) renders correctly on Space hardware.
2. **Screenshot to attach, per post:**
   - r/functionalprint (Draft 1): the **support before/after render** (red support zones) and the
     **split result** (7 parts + connectors). The measured filament deltas are the story — a gcode/verdict
     screenshot showing +1.27 cm3 / -26.7% lands harder than a beauty render.
   - r/3Dprinting (Draft 2): the **verdict banner + "What we changed" receipt** showing the trust line
     ("0% reshaped, max deviation 0.003 mm") on the broken axe. That receipt IS the hook — make it the
     first image.
   - HF / Show HN (Draft 3): the **`nm -C` license output** (26 SparseLU / 0 SimplicialLLT) and the
     source-string table; a technical audience wants the receipts, not renders.
3. **When to post:** functional-print and maker subs skew US daytime/evening; Show HN skews weekday
   US morning. Post one at a time, not all three same day — you have to respond fast (see below) and
   can only babysit one thread well.
4. **Respond fast, honestly.** The whole pitch is honesty; the fastest way to lose it is an
   overclaim in a reply. Keep every label ("comparative", "uncalibrated", "measured not asserted") in
   your answers too. If someone asks for a benchmark you did not run, say you did not run it — the
   reports list exactly what is deferred (2 vCPU Space timing, 3MF-as-input fuzzing, concurrent use,
   real functional print, absolute-mm warp calibration). Point to those as roadmap, not as done.
5. **Have a broken mesh ready.** Every closer asks people to send a part. Be ready to actually run
   one on stream/in-thread — the tool refuses honestly, which is the demo.
