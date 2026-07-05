# Publishing meshprep to Hugging Face — complete first-timer walkthrough

This takes you from "I have never used Hugging Face" to a live, working Space,
in about 30–45 minutes. **Cost: $0** — the free CPU tier is genuinely free, for
you and for everyone who uses it.

> Quick operator checklist (for later deploys, once you know the ropes):
> `space/DEPLOY.md`. This guide is the long-form version of the same process.

The strategy in one line: **create the Space PRIVATE first, smoke-test it on
real hardware, then flip it public.** Private = your test bench; public = the
publish button. Nothing is visible to anyone until you flip it.

---

## Step 0 — Decide about the source link (5 min, one-time)

Two files contain a placeholder link `https://github.com/REPLACE-ME/meshprep`:

- `space/README.md` (the Space landing text)
- `src/meshprep/app.py` (the app footer — note: this one lives INSIDE the
  package, so changing it requires rebuilding the wheel, Step 1)

The GitHub repo **does not exist yet** (the git repo is local-only so far).
Pick one:

- **Option 1 — create the GitHub repo first** (recommended if you want the
  open-core funnel working from day one): on github.com → New repository →
  name `meshprep`, public, no README (we have one) → then from
  `C:\Users\mecht\Project_EI\3DEI\meshprep`:
  ```
  git remote add origin https://github.com/<your-username>/meshprep.git
  git push -u origin main
  ```
  Then replace `REPLACE-ME` with your username in both files.
- **Option 2 — ship without the link for now**: edit both files and delete the
  source-link line (or write "source coming soon"). You can add it back in a
  later update.

Either way, edit the two files BEFORE Step 1, because Step 1 bakes `app.py`
into the wheel.

## Step 1 — Rebuild + verify the wheel (2 min, EVERY deploy)

The Space doesn't install meshprep from the internet — it ships as a wheel
file inside the `space/wheels/` folder. If you changed ANY package file since
the wheel was built (including the footer edit above), rebuild:

```
pip wheel C:/Users/mecht/Project_EI/3DEI/meshprep -w C:/Users/mecht/Project_EI/3DEI/meshprep/space/wheels --no-deps
```

Then run the freshness guard — **mandatory before every deploy**:

```
python C:/Users/mecht/Project_EI/3DEI/meshprep/space/check_wheel.py
```

- **PASS** → continue.
- **FAIL** → the wheel is older than the source. Rebuild (command above),
  delete any old `.whl` left in `space/wheels/` (there must be exactly one),
  and re-run the guard. This check exists because a stale wheel once nearly
  shipped a Space whose backend was missing the hosted-mode preset.

## Step 2 — Create your Hugging Face account (5 min, one-time)

1. Go to https://huggingface.co/join — sign up with your email (free).
2. Verify the email. That's it. No payment method is ever requested for the
   free tier.

## Step 3 — Create the Space (3 min)

1. Go to https://huggingface.co/new-space
2. Fill the form exactly:

   | Field | Value |
   |---|---|
   | Owner | you |
   | Space name | `meshprep` (or another — this becomes the URL: `huggingface.co/spaces/<you>/meshprep`) |
   | License | MIT |
   | Select the SDK | **Gradio** (leave the template blank/default) |
   | Space hardware | **CPU basic · 2 vCPU · 16 GB · FREE** |
   | Visibility | **Private** ← test bench first |

3. Click **Create Space**. You land on an empty Space page.

## Step 4 — Upload the files (5 min, no git needed)

You are uploading the **contents** of `C:\Users\mecht\Project_EI\3DEI\meshprep\space\`:

```
README.md
app.py
requirements.txt
wheels/meshprep-0.1.0.dev0-py3-none-any.whl
```

(NOT `DEPLOY.md`/`check_wheel.py` — harmless if included, but they're operator
docs, not app files. Never upload anything from outside the `space/` folder.)

1. On your new Space, open the **Files** tab.
2. **Add file → Upload files.**
3. Drag in `app.py` and `requirements.txt`.
   For the wheel: drag the whole `wheels` FOLDER into the drop zone (modern
   browsers upload folders with their structure). If your browser won't take
   a folder, upload the `.whl` alone and type `wheels/` in front of its name
   in the path box so it lands at `wheels/meshprep-...whl`.
4. `README.md` is special: the Space was created with a default README. Open
   **Files → README.md → edit** (pencil icon), select-all, delete, and paste
   the full contents of `space/README.md` — **including the `---` YAML block
   at the top** (that block IS the Space configuration: SDK, gradio version,
   entry file). Commit the change.
5. Each upload asks for a commit message — anything ("initial deploy") is fine.

## Step 5 — Watch it build (3–8 min, hands off)

The Space builds automatically after every commit.

1. Open the **Logs** tab (top of the Space page).
2. Healthy build: `pip install` lines for the pinned packages, then the wheel,
   then Gradio startup. First build takes a few minutes (it compiles nothing —
   just downloads).
3. When the status pill says **Running**, open the **App** tab. You should see
   the drag-and-drop page.

**If the build fails**, see Troubleshooting at the bottom — 90% of failures
are a wrong wheel filename in `requirements.txt` or a mangled README
frontmatter.

## Step 6 — Smoke test on real hardware (10 min — this run IS the measurement)

All local timings were measured on an 8-core desktop; the free Space has
2 slower vCPUs, so **expect a multiple**. This first session produces the real
numbers — note them down.

| # | Drop this | Expect | Record |
|---|---|---|---|
| 1 | A simple solid (40 mm calibration cube STL — export one from any slicer or use a known-good file) | Green **READY TO PRINT**, the trust line ("we touched X% of your surface…"), size line, renders, downloads work | seconds |
| 2 | A real AI mesh (any `.glb` from your Meshy account, or `3DEI/Forge/corpus/meshy/*.glb`) | Honest verdict (usually **NOT READY YET** with a receipt of what was fixed, or WARN), review passes the eyeball test | seconds |
| 3 | A `.txt` file renamed to `.stl` | Plain refusal: "We can't read this file" — **no error wall** | — |
| 4 | Re-run #2 with **Predict warp** ticked | Completes, numbers labeled *estimate/uncalibrated* | added seconds |

Also confirm: the first line of the review is the verdict; the downloads
(`prep.stl`, `review.md`) open; nothing in the page shows a Windows path or a
Python traceback.

**If a job exceeds ~5 minutes on the Space**, that's worth knowing before
going public — bring the timing back and we'll tighten the hosted preset
(it has a soft budget designed for exactly this).

## Step 7 — Flip it public (30 seconds, the actual publish)

Space page → **Settings** → **Change visibility** → Public.

What public means: anyone with the link can use it; it appears in HF search;
the Spaces gallery may index it. This is the outward-facing moment — after
this, the tool is live and shareable (e.g., in a Reddit post or a Meshy
community thread).

## Updating the Space later

1. Change code in `src/meshprep/...` locally.
2. **Step 1 again** (rebuild wheel + `check_wheel.py` — never skip).
3. Upload the changed files on the **Files** tab (same drag-and-drop; the new
   wheel replaces the old one — delete the stale `.whl` in the Files tab if
   the filename changed).
4. The Space rebuilds automatically. ~2–5 min of downtime; the queue resumes.

## What to expect in daily operation (free tier)

- **Sleep:** after ~48 h with no visitors the Space sleeps; the next visitor
  waits ~1 min while it wakes. Nothing is lost.
- **One job at a time:** simultaneous users wait in a visible queue (by
  design — 2 vCPUs).
- **Storage is ephemeral:** uploads and results vanish on restart — which is
  exactly what the privacy note promises.
- **Upload cap:** 25 MB / 3M triangles, enforced and stated in the UI.
- **Your cost: $0, permanently**, unless you deliberately upgrade hardware in
  Settings (not needed).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Build fails: `ERROR: Could not find ... wheels/meshprep-...` | Wheel filename in `requirements.txt` doesn't match the uploaded file, or the wheel landed at repo root instead of `wheels/` | Files tab → check the path is `wheels/<exact-name>.whl`; fix the name in `requirements.txt` |
| Build fails on a pinned package | PyPI hiccup or platform wheel missing | Retry (Settings → Factory rebuild). If persistent, report the exact log line back to a session |
| "Configuration error" before any build | README YAML frontmatter mangled (missing `---` fences or `sdk_version`) | Re-paste `space/README.md` exactly, fences included |
| App builds but shows a blank page / error | `sdk_version` in README doesn't match `gradio==` in requirements | Both must say the same version (currently 6.19.0) |
| Runtime error after weeks of fine operation | HF base image moved | Settings → **Factory rebuild** first; if still broken, report back |
| First visit very slow | The Space was asleep | Normal; ~1 min wake |
| A job seems stuck | 2 vCPU is slow + soft budget skips extras with a note; hard cap will end it | Wait it out once, note the timing, report back if > ~5 min |
| You changed code but the Space behaves old | Stale wheel shipped | Step 1: rebuild + `check_wheel.py`, re-upload the wheel |

## Optional: git instead of web uploads (for later)

```
# one-time: create a WRITE token at huggingface.co/settings/tokens
git clone https://huggingface.co/spaces/<you>/meshprep
# copy the contents of space/ into the clone
cd meshprep
git add -A && git commit -m "deploy" && git push
# username = your HF username, password = the token
```

If the wheel ever exceeds 10 MB (it's far under today):
`git lfs track "*.whl"` before committing.
