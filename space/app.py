"""Hugging Face Space entry point — thin wrapper around printworthy's app.

The real UI lives in printworthy.app (installed from ./wheels/*.whl via
requirements.txt). This file only sets Space-appropriate limits and
launches.
"""
import os

# THE Space behaviour bundle: preset="space" (printworthy.profiles.PREP_PRESETS)
# is the purpose-built hosted-demo configuration — 20k-face analysis proxy,
# ~120 s soft stage budget (optional stages skip with an honest note; the fix,
# re-check and verdict always run), opt-in warp FEM capped at 1500 elements
# (well under the 2500-element hosting cap), fast post-fix re-check, support
# render, and no server-local paths surfaced to users. printworthy.app.run()
# reads this env var and passes preset= to prep(). Without it, jobs ran at
# full desktop settings and blew the per-job time budget.
os.environ.setdefault("PRINTWORTHY_PRESET", "space")

# Belt to the preset's own max_faces: cap the working mesh so one job
# (analysis + fix + re-check) stays inside the request window. The review
# document states when a mesh was simplified for analysis, so nothing is
# hidden.
os.environ.setdefault("PRINTWORTHY_MAX_FACES", "20000")

from printworthy.app import build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch(max_file_size="25mb")
