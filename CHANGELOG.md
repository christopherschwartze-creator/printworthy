# Changelog

All notable changes to **meshprep** are recorded here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow SemVer.

## [0.1.0a1] — unreleased (launch-ready; alpha, not yet published)

> Version note: bumped off the `0.1.0.dev0` development marker to the alpha
> pre-release `0.1.0a1`. Still **not published to PyPI** — install from source
> (see README). "Alpha" (not `rc`/`1.0`) is deliberate: the code is
> feature-complete and fuzz-hardened, but the first real-hardware Space deploy
> and the physical print validations (warp coupon, load test) are still
> unwalked. Promote to `0.1.0` at the actual public release if desired.


The complete free tier: one-call prep pipeline, hardened, packaged for
Hugging Face Spaces, with demo assets and launch collateral. A separate
proprietary pro package (warp pre-compensation, retopo-pro, quotes,
multi-material, credit-gated Pro Space) exists outside this repo; its public
stubs here carry the full build plans.

### Added
- **`meshprep.prep()` one-call pipeline** (`pipeline.py`): load → ingress
  guards → analyze (premortem / traps / topology / unit scan) → source-accurate
  **fix** (deviation certificate, never-worse rollback, fidelity-gated
  sealing) → orient → optional warp/strength FEM → graded-infill 3MF →
  bed-fit → slicer savings → plain-language report. Never raises; never
  reports success over a non-printable result.
- **Printability gate** (`core/_printability.py`) — a PASS verdict is
  structurally impossible over inside-out / non-watertight / degenerate
  output; out_dir writability probed at ingress; deliverable-exists check
  before any success verdict.
- **Print-FEM** (`scikit-fem` + `pyamg`): inherent-strain warp
  (Timoshenko-validated), orthotropic strength / orient-for-strength,
  explicit `length_unit` Pa bridge + metre-scale `unit_warning`, one-coupon
  **calibration** (`meshprep calibrate`).
- **`reinforce`** — graded-infill 3MF (schema source-verified against
  PrusaSlicer `3mf.cpp`; CLI-proven 1.85× modifier application) + the
  **gradient pre-screen** (`recommend_grading`): grading is offered only when
  the stress gradient beats the recipe ratio (~5.3×) — measured to separate
  the parts where grading wins (+20–35 % material) from the majority where
  uniform infill is the better deal.
- **Risk-driven support enforcers/blockers** (`core/support_mods.py` +
  `meshprep supports`, `prep --supports`): premortem risk field →
  `SupportEnforcer`/`SupportBlocker` volumes (strings source-verified against
  PrusaSlicer `Model.cpp`; slicing effect measured, not asserted:
  enforcer +42.5 % / blocker −26.7 % support material on the control part).
- **Smart bed-split** (`split.py`, `split_for_bed(..., smart=True)`): seams
  scored into concave creases (dumbbell control: seam lands dead-center on
  the neck), exact `manifold3d` cuts on the original mesh, peg/socket
  connectors + printable `fit_coupon()`, labeled fallback ladder; CoACD
  isolated in a wall-clock-bounded subprocess (a native crash can no longer
  take down the host process).
- **Novice review document** (`report.build_review`) — one page,
  verdict-first (READY / NOT READY YET / WE COULDN'T READ THIS FILE), the
  trust line ("we touched X % of your surface, max deviation Y mm"), a
  copy-pasteable "for your print shop" block, every warning with exactly one
  next action; `review_selfcheck()` enforces ~23 acceptance criteria
  mechanically.
- **Hugging Face Space packaging** (`space/`): pinned requirements, wheel
  delivery + `check_wheel.py` staleness guard (it has caught two stale wheels
  to date), `preset="space"` hosted bundle (soft time budget, capped FEM,
  keep-original-size default), `DEPLOY.md` + the full first-timer guide
  `PUBLISH_TO_HUGGINGFACE.md` (free tier + paid-offering architecture).
- **Demo assets**: `docs/images/` (7 captioned renders from real corpus
  meshes), `examples/` walkthroughs with measured numbers, `LAUNCH_POSTS.md`
  drafts, `LAUNCH_READY_REPORT.md`.
- **`license_guard.py`** — AST scan proving zero copyleft / non-commercial
  imports (CLEAN at every commit).

### Fixed (highlights from the hardening campaigns — full records in *_REPORT.md)
- Fixer: junction-vertex boundary loops silently skipped; `fill_holes()`
  creating non-manifold edges; inside-out marching-cubes output shipping as
  "watertight" (surface fidelity on real broken AI meshes: 60.4 % → **99.4 %
  mean, median 100 %**).
- FEM stress compared MPa-numeric against Pa allowables (FOS pinned at the
  1e6 ceiling) — explicit unit bridge added.
- Unbounded memory in premortem's thin-wall ray channel (2.4 GB → 0.36 GB via
  ray batching) — found the hard way, when the validation harness OOM'd the
  build machine.
- Read-only output directory produced a hollow PASS with zero deliverables.
- False "not watertight" FAIL on clean parts; an 11.3-minute hosted job with
  no feedback (→ 71.5 s worst-case under the space preset); silent rescaling
  of every upload (→ keep-original-size + explicit unit nudge).

### Verified
- **Foolproofing: the full 143-cell matrix is complete** — every entry point
  × every pathological input class on shipped code: **141/141 product cells
  clean** (0 crashes, 0 silent-wrongs, 0 hangs, 0 memory hogs); the only two
  failures are deliberately planted tripwires proving the detection works.
  See `FOOLPROOF_REPORT.md`.
- All selftests green: fixer, solidify, printability, reinforce (+tiers),
  FEM (closed-form gates), warp calibration.

### Notes
- **`autorig` (animation build) ships separately** — not bundled.
- Permissive-only (MIT/BSD/Apache/MPL-2); GPL/CGAL/non-commercial
  deliberately unreachable; `pyQuadriFlow` remains an LGPL-flagged opt-in
  extra (its prebuilt blob provably links Eigen's LGPL solver — the pro
  package ships a license-clean rebuild instead).
- Physics predictions are **uncalibrated estimates** unless a coupon is
  fitted; verdicts are advisory. The remaining unwalked gates are physical:
  print one warp coupon, load one graded bracket, measure real Space timings.

### Remaining before publish
Owner actions only — see `START_HERE_LAUNCH_CHECKLIST.md`: trademark check on
the name, set author/homepage, GitHub push, Space deploy, Lemon Squeezy
test-mode purchase, launch posts.
