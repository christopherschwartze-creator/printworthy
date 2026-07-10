# START HERE — your launch checklist

Everything the machine could build and verify is done. Every remaining step
is yours because it is outward-facing: it creates accounts, publishes code,
or takes money. Do them in this order; each links to its detailed guide.

**Total time to a live free tier: ~1–2 hours. To a sellable pro tier: +2–4 hours.**

---

## Phase A — the free tier goes live (~1–2 h)

- [ ] **A1. Decide the name & clear it** (15 min): `meshprep` is free on PyPI
      (verified 2026-07-03) — do a quick trademark sanity search (USPTO/EUIPO
      web search for "meshprep" in software). Also set your legal name and
      homepage in `pyproject.toml` (two marked placeholders).
- [ ] **A2. GitHub push** (15 min): create the public repo, then Step 0 of
      `PUBLISH_TO_HUGGINGFACE.md` — replace the two `REPLACE-ME` links,
      rebuild the wheel, run `space/check_wheel.py` (must PASS).
- [ ] **A3. Deploy the free Space PRIVATE** (30 min): Steps 2–5 of
      `PUBLISH_TO_HUGGINGFACE.md` (account → create Space → upload `space/`
      → watch build).
- [ ] **A4. Smoke-test on real hardware** (15 min): the 4-drop matrix in
      Step 6. **Record the timings** — first real 2-vCPU numbers; if a job
      exceeds ~5 min, pause and report back before going public.
- [ ] **A5. Flip public** (30 s): Step 7. The free tier is now live.

## Phase B — the paid tier (~2–4 h, can be days later)

- [ ] **B1. License terms** (the one real-world dependency): have the
      placeholder LICENSE in `meshprep_pro/` reviewed/replaced before any
      sale. Until then everything can run in test mode.
- [ ] **B2. Lemon Squeezy store** (1 h): Part 2 of
      `PUBLISH_TO_HUGGINGFACE.md` — 3 products (seat ~$79 / credits ~$10 /
      bureau sub ~$49) in **test mode**; buy each with the test card.
- [ ] **B3. Private wheel repo + secrets** (30 min): upload the pro wheel +
      vendor binary to a *private* HF repo; set `HF_TOKEN` + `LS_API_KEY`
      secrets. **Never put pro files in a public Space repo.**
- [ ] **B4. Deploy the Pro Space PRIVATE + full test-mode purchase** (1 h):
      `meshprep_pro/DEPLOY_PRO.md` — buy → key → job → credit decrements →
      certificate downloads. This is the one thing the local mock could not
      prove.
- [ ] **B5. Go live**: LS out of test mode, Pro Space public.

## Phase C — announce (an evening, then respond fast)

- [ ] **C1.** Fill the `[LINK-TBD]` placeholders in `LAUNCH_POSTS.md`,
      take the screenshots named in its posting checklist.
- [ ] **C2.** Post draft 2 (r/3Dprinting, novice angle) first — biggest
      audience, gentlest crowd. A day later: draft 1 (r/functionalprint).
      Draft 3 (technical) when you feel like it.
- [ ] **C3.** Respond to every comment fast for 48 h. Refund instantly on
      any pro complaint — the first ten customers are the reputation.

## Phase D — the physical validations (whenever, with a printer)

- [ ] **D1.** Print the warp coupon → `meshprep calibrate` (turns estimates
      into calibrated mm — and unlocks honest warp pre-comp claims).
- [ ] **D2.** Print a graded bracket + its uniform twin; load both.
- [ ] **D3.** Print `split.fit_coupon()` to find your printer's clearance.

## Phase E — the money that doesn't need a crowd (parallel to everything)

- [ ] **E1.** Email 3–5 print bureaus: the quote-engine pitch
      ("broken uploads caught before a human opens them; reproducible
      quotes"). One yes ≈ the whole $200/mo bar.

---

*Where things stand behind this checklist: free tier verified across a
143-cell pathological-input matrix (141/141 clean); money path verified
row-by-row against a faithful Lemon Squeezy mock (credit consumed only on
delivered artifact); demo images + example walkthroughs shipped; all
reports in `*_REPORT.md` files in this repo and `meshprep_pro/`.*
