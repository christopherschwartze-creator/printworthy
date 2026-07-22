# START HERE — your launch checklist

Everything the machine could build and verify is done. Every remaining step
is yours because it is outward-facing: it creates accounts, publishes code,
or takes money. Do them in this order; each links to its detailed guide.

**Time to a live FREE tier: ~1–2 hours of hands-on work.** The PAID tier is
**not** a fixed time box: it is gated on **counsel-reviewed license terms** — a
real-world dependency whose lead time is counsel's, not yours (could be days to
weeks) — and on the pro build being deploy-ready. Budget the free tier as an
afternoon; treat the paid tier as "whenever counsel signs off," not "+N hours."

## ⚠️ Repo layout — read this first (it has already confused one search)

**Naming note (2026-07-11):** the product/package/repo name is **`printworthy`**
(renamed from meshprep after name research). The LOCAL FOLDERS still carry the
old names — `3DEI\meshprep\` holds the printworthy repo, `3DEI\meshprep_pro\`
holds the pro repo — because an open editor held file locks during the rename.
Folder names are cosmetic; every command below uses the real folder paths.

There are **TWO separate repos in sibling folders**:

```
C:\Users\mecht\Project_EI\3DEI\meshprep\       <- PUBLIC (MIT). This one gets pushed to GitHub.
C:\Users\mecht\Project_EI\3DEI\meshprep_pro\   <- PROPRIETARY. Its OWN git repo. NEVER pushed.
```

`meshprep_pro` is **deliberately absent** from the public folder and from the
public repo's git history — if it were inside `printworthy/`, the Phase-A GitHub
push would open-source the paid product. So: looking for it inside this repo
correctly finds nothing. Every `meshprep_pro/...` reference below means the
**absolute sibling path** `C:\Users\mecht\Project_EI\3DEI\meshprep_pro\...`.

**Backup warning:** `meshprep_pro` currently exists ONLY on this machine
(local git, no remote). Before or during Phase A, give it an off-machine
copy: a **private** GitHub repo is fine (private ≠ published), or include
`3DEI\meshprep_pro` in the rclone backup set. The public repo gets a remote
in step A2 anyway; the pro repo should not be the only unbacked asset.

---

## Phase A — the free tier goes live (~1–2 h)

- [ ] **A1. Name + author metadata** (15 min, no repo needed yet):
      `printworthy` is free on PyPI (verified 2026-07-03; re-check at
      https://pypi.org/project/printworthy/ — a 404 means still free). Preliminary
      name research (PyPI / GitHub / HF / web + descriptiveness flag) is written
      up in **`NAME_RESEARCH.md`** — this is research to hand to counsel, **NOT
      a clearance**. The name is **not cleared** until counsel says so (same
      counsel review as B1 — you can bundle the two questions). Set your exact
      legal name/entity in `pyproject.toml` (`authors = [...]`, one marked
      placeholder). **Do not set the Homepage URL yet** — it points at the
      GitHub repo that does not exist until A2, so it's set in A2 (this was the
      old ordering bug: A1 used to set a Homepage for a repo A2 hadn't created).
- [ ] **A2. GitHub push + wire the URLs** (15 min): create the public repo
      (github.com → New repository → `printworthy`, public, no README — we have
      one), then:
      ```
      git remote add origin https://github.com/<your-username>/printworthy.git
      git push -u origin main
      ```
      NOW that the repo URL exists, set it in `pyproject.toml`
      (`[project.urls] Homepage = https://github.com/<your-username>/printworthy`)
      and do Step 0 of `PUBLISH_TO_HUGGINGFACE.md` — replace the two
      `REPLACE-ME` links (in `space/README.md` and `src/printworthy/app.py`) **and**
      the `<your-username>` placeholder in this repo's `README.md` install line
      with your username. Because you touched `app.py` and `pyproject.toml`,
      **rebuild the wheel** (`pip wheel . -w space/wheels --no-deps`, delete the
      stale `.whl`, update the filename in `space/requirements.txt`) and run
      `space/check_wheel.py` (must PASS) before deploying.
- [ ] **A3. Deploy the free Space PRIVATE** (30 min): Steps 2–5 of
      `PUBLISH_TO_HUGGINGFACE.md` (account → create Space → upload `space/`
      → watch build). **Privacy/retention:** the Space ingests user-uploaded
      mesh files. The current posture (stated in `space/README.md` and enforced
      by ephemeral storage) is: files processed **in memory, not stored, not
      logged, not used for training**; results vanish on container restart;
      25 MB / 3M-triangle cap. Keep that promise true when you flip public —
      do not add analytics/logging that would retain uploads, and if you ever
      do, update the privacy line first.
- [ ] **A4. Smoke-test on real hardware** (15 min): the 4-drop matrix in
      Step 6. **Record the timings** — first real 2-vCPU numbers; if a job
      exceeds ~5 min, pause and report back before going public.
- [ ] **A5. Flip public** (30 s): Step 7. The free tier is now live.

## Phase B — the paid tier (counsel-gated; runs in test mode until B5, can be days later)

> Everything below runs in **test mode** (no real money) until B5, which is
> gated on B1 (counsel-reviewed license terms). Do not flip to live until then.

- [ ] **B1. License terms** (the one real-world dependency): have the
      placeholder LICENSE at
      `C:\Users\mecht\Project_EI\3DEI\meshprep_pro\LICENSE` reviewed/replaced
      by counsel before any sale. This is the gating item for the whole paid
      tier and its lead time is counsel's. Bundle the name-clearance question
      (A1 / `NAME_RESEARCH.md`) into the same review.
- [ ] **B2. Lemon Squeezy store** (~1 h hands-on): Part 2 of
      `PUBLISH_TO_HUGGINGFACE.md` — 3 products (seat ~$79 / credits ~$10 /
      bureau sub ~$49) in **test mode**; buy each with the test card and
      confirm each key validates via the License API.
- [ ] **B3. Private wheel repo + secrets** (30 min): build the pro wheel
      (`pip wheel C:\Users\mecht\Project_EI\3DEI\meshprep_pro -w <dir> --no-deps`)
      and upload it + the vendor binary
      (`C:\Users\mecht\Project_EI\3DEI\meshprep_pro\vendor\quadriflow`) to a
      *private* HF repo; set `HF_TOKEN` + `LS_API_KEY` secrets (details in
      DEPLOY_PRO.md, B4). **Never put pro files in a public Space repo.**
- [ ] **B4. Deploy the Pro Space PRIVATE + full test-mode purchase** (1 h):
      guide at `C:\Users\mecht\Project_EI\3DEI\meshprep_pro\DEPLOY_PRO.md`
      (Space files in `...\meshprep_pro\space_pro\`) — buy → key → job →
      credit decrements → certificate downloads. This is the one thing the
      local mock could not prove.
- [ ] **B5. Go live** (do NOT do this until B1 counsel sign-off is in hand):
      1. Flip **Lemon Squeezy out of test mode**. ⚠️ **Test mode and live mode
         use SEPARATE API keys.** The `LS_API_KEY` secret you set in B3 is the
         **test** key — after flipping to live you must generate the **live**
         API key and update the `LS_API_KEY` secret in the Pro Space to the live
         value, or every real purchase will fail to validate.
      2. **Rotate the webhook signing secret too.** LS issues a distinct
         **webhook secret** for live mode; regenerate it and update the
         corresponding secret in the Pro Space, or live purchase webhooks will
         fail signature verification (and any test-mode webhook secret left in
         place is a leaked-in-test-history credential — rotate it regardless).
      3. Re-run one real (small, real-card) purchase end-to-end to confirm the
         live key + live webhook path works before announcing.
      4. Flip the Pro Space **public**.

## Phase C — announce (an evening, then respond fast)

- [ ] **C1.** Fill the `[LINK-TBD]` placeholders in `LAUNCH_POSTS.md`,
      take the screenshots named in its posting checklist.
- [ ] **C2.** Post draft 2 (r/3Dprinting, novice angle) first — biggest
      audience, gentlest crowd. A day later: draft 1 (r/functionalprint).
      Draft 3 (technical) when you feel like it.
      - ⚠️ **Warp-claim caveat.** Draft 1's warp paragraph is a **paid**
        (`meshprep_pro`) feature, so it can only be *demoed* once the Pro Space
        (Phase B) is live — don't promise warp in a post before B is up. And its
        numbers are labeled **comparative / uncalibrated** on purpose: the
        **calibrated-mm** warp story is only earned after **Phase D1** (the
        printed warp coupon). Until D1, keep every warp number labeled
        comparative/uncalibrated in posts and replies — do **not** upgrade it to
        an absolute-mm claim. (This is the review's "Phase C announces a warp
        story Phase D hasn't measured yet" — the fix is: keep the label, or hold
        the warp claim until D1.)
- [ ] **C3.** Respond to every comment fast for 48 h. Refund instantly on
      any pro complaint — the first ten customers are the reputation.

## Phase D — the physical validations (whenever, with a printer)

- [ ] **D1.** Print the warp coupon → `printworthy calibrate` (turns estimates
      into calibrated mm — and unlocks honest **calibrated-mm** warp claims;
      until this is done, warp stays *comparative/uncalibrated* everywhere,
      including the Phase C posts).
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
