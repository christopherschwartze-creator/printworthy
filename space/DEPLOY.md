# Deploying the meshprep Space

> First time publishing to Hugging Face? Use the full step-by-step
> walkthrough instead: `../PUBLISH_TO_HUGGINGFACE.md` (account creation →
> private smoke test → flip public). This file is the terse operator
> checklist for repeat deploys.

**Deploying is YOUR outward-facing action.** Nothing here publishes
anything by itself; the Space only goes live when you create it and push
this folder under your own Hugging Face account.

## What ships

This `space/` folder is the complete Space repo:

```
space/
  README.md          <- HF Space config (YAML frontmatter) + landing text
  app.py             <- entry point; imports meshprep.app.build_demo()
  requirements.txt   <- PINNED deps + the local meshprep wheel
  wheels/
    meshprep-0.1.0.dev0-py3-none-any.whl
```

meshprep is not on PyPI; it rides along as the wheel in `wheels/`,
referenced from `requirements.txt` as a relative path. After ANY change to
the package, rebuild it before pushing:

```
pip wheel C:/Users/mecht/Project_EI/3DEI/meshprep -w C:/Users/mecht/Project_EI/3DEI/meshprep/space/wheels --no-deps
```

If the version in `pyproject.toml` changed, update the wheel filename in
`requirements.txt` and delete the stale wheel.

**Mandatory freshness guard — run before EVERY push:**

```
python C:/Users/mecht/Project_EI/3DEI/meshprep/space/check_wheel.py
```

It fails (non-zero) when the wheel is older than any source file or when
the wheel's copies of profiles/pipeline/report/app differ from
`src/meshprep` — exactly the stale-wheel failure that once nearly shipped a
Space whose backend lacked the `space` preset. Do not push on a FAIL.

Before pushing, replace the two `REPLACE-ME` GitHub placeholder links
(in `README.md` here and in `src/meshprep/app.py` FOOTER).

## Option A — web upload (no git needed)

1. Log in at https://huggingface.co and go to https://huggingface.co/new-space
2. Name: `meshprep` (or your choice). License: MIT. SDK: **Gradio**.
   Hardware: **CPU basic (free)**. Visibility: your call (private is fine
   for a first smoke test; flip to public later).
3. On the new Space's **Files** tab, click **Add file -> Upload files** and
   upload the CONTENTS of this folder (`README.md`, `app.py`,
   `requirements.txt`, and the `wheels/` folder with the .whl inside).
4. The Space builds automatically (a few minutes: it installs the pinned
   requirements). Watch the **Logs** tab; when it says Running, open the
   **App** tab and drop a test mesh.

## Option B — git push

```
git clone https://huggingface.co/spaces/<your-username>/meshprep
# copy the contents of this space/ folder into the clone (including wheels/)
cd meshprep
git add -A
git commit -m "meshprep space"
git push
```

(If the wheel ever exceeds 10 MB, `git lfs track "*.whl"` first — the
current wheel is well under that.)

## What to expect on the free tier

- **Hardware:** 2 vCPU, 16 GB RAM, no GPU. One job at a time (the app
  queues with a single worker on purpose).
- **Speed:** a typical mesh preps in seconds to ~2 minutes. `app.py` sets
  `MESHPREP_PRESET=space`, which activates the hosted bundle: 20k-face
  analysis proxy, ~120 s soft stage budget (optional stages skip with an
  honest note; fix/re-check/verdict always run), fast post-fix re-check,
  and the support render. The review states when a mesh was simplified.
  The opt-in warp physics pass adds ~0.5-1 min and is capped at 1,500 FEM
  elements (hosting cap is <= 2,500).
  Measured 2026-07-05 on a local 8-core box, end-to-end THROUGH THE UI
  (gradio_client -> upload -> badge + downloads; embreex installed):
  worst corpus part `thingi10k/1013014.glb` **71.5 s** (72.4 s calling
  `prep()` directly — the UI adds ~nothing), clean exemplar
  `104421.glb` **16.7 s**. Before this measurement the same worst part
  took 204 s through the UI; the shave was mechanical (voxelize the trap
  grid once instead of four times, stop matplotlib drawing each 3D
  figure three times, draw pictures on a display-decimated proxy) — no
  check, threshold, or reported number changed, and the worst part now
  finishes with ALL optional stages run instead of budget-skipped.
  These are LOCAL 8-core numbers: a free Space has 2 slower vCPUs, so
  expect a multiple of them there (not yet measured on Space hardware);
  the ~120 s budget is SOFT — past it, optional extras skip with an
  honest note while the fix, re-check and verdict always run to
  completion.
- **Sleep:** free Spaces go to sleep after ~48 h without traffic; the
  first visit afterwards takes ~1 min to wake. Nothing is lost.
- **Storage:** ephemeral. Uploaded files and results live in the
  container's temp space and vanish on restart — consistent with the
  privacy note (nothing stored).
- **Upload cap:** 25 MB, enforced in the app and stated in the UI.

## Smoke test after deploy

1. Drop a known-good simple solid (e.g. a 40 mm calibration cube) ->
   expect a green PASS ("Ready to print"), the trust line ("we touched X%
   of your surface..."), a size-check line, renders with human captions,
   and the downloads (prep.stl + review.md + report.md). Verified locally:
   a 40 mm cube preps to PASS in ~4 s under the space preset.
2. Drop the real-corpus clean exemplar
   `3DEI/Forge/corpus/thingi10k/104421.glb` -> expect "Printable — read
   the warnings" (WARN) and a review headed READY TO PRINT with
   caution-level cards only — real organic parts almost always carry an
   honest supports/topology caution, so literal PASS is reserved for
   parts with nothing to flag. Verified locally: ~11-20 s, no false
   thin-wall FAIL (the gate now reports area-supported wall thickness).
3. Rename a .txt to .stl and drop it -> expect a plain refusal ("We can't
   read this file"), no traceback.
4. Tick "Predict warp" -> expect the warp render and numbers labeled
   estimate/uncalibrated (FEM capped at 1,500 elements under the space
   preset — verified locally: +~30 s on a 163 KB part).
