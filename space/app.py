"""Hugging Face Space entry point — thin wrapper around printworthy's app.

The real UI lives in printworthy.app. printworthy is NOT installed via
requirements.txt: HF's build system installs requirements.txt in an ISOLATED
stage where only that file is mounted into the container -- the rest of the
repo (including wheels/) is not present yet, so a local relative wheel path
in requirements.txt fails at build time with a bare "No such file or
directory" (hit in production 2026-07-25). Fixed by installing the wheel
here, at RUNTIME, the moment this file executes -- by then the full repo
(wheels/ included) has been copied into the container.
"""
import os
import subprocess
import sys


def _ensure_printworthy():
    """Install printworthy from the bundled local wheel if not already
    importable. --no-deps: every dependency printworthy needs is already
    installed from requirements.txt in the build stage; this only adds our
    own package on top, so it is fast and makes no network/version-resolver
    calls of its own. Never raises a confusing import error -- if the wheel
    is missing or broken, the real pip error is allowed to surface plainly
    rather than being swallowed, since a broken Space should fail loudly in
    the logs rather than serve a mysterious blank page."""
    try:
        import printworthy  # noqa: F401
        return
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    wheels_dir = os.path.join(here, "wheels")
    candidates = [f for f in os.listdir(wheels_dir) if f.endswith(".whl")] \
        if os.path.isdir(wheels_dir) else []
    if not candidates:
        raise RuntimeError(
            f"printworthy is not installed and no wheel was found in "
            f"{wheels_dir!r} to install it from. Check that wheels/*.whl "
            f"was actually uploaded to this Space.")
    wheel_path = os.path.join(wheels_dir, sorted(candidates)[-1])
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir",
         "--no-deps", wheel_path])


_ensure_printworthy()

# Free Hugging Face accounts can now only create Gradio Spaces on ZeroGPU
# hardware (CPU-basic creation requires a PRO subscription) -- confirmed in
# production 2026-07-26: "Runtime error: No @spaces.GPU function detected
# during startup". printworthy is a pure NumPy/SciPy/scikit-fem CPU pipeline
# with zero GPU need; the platform's startup check is a static scan for the
# PRESENCE of at least one @spaces.GPU-decorated function, not proof that one
# is ever called (this is the documented "dummy placeholder" pattern other
# non-GPU Gradio Spaces use to satisfy the same check). `spaces` is baked
# into every free Space by the platform itself, NOT a real printworthy
# dependency -- importing it defensively so this file still runs unchanged
# for local dev / `printworthy app` outside of Hugging Face, where the
# package is absent and no such check exists.
try:
    import spaces as _hf_spaces

    @_hf_spaces.GPU
    def _zerogpu_startup_placeholder():
        """Never called. Exists only so Hugging Face's ZeroGPU platform
        detects a @spaces.GPU function at startup; printworthy does no GPU
        work at all."""
        return None
except ImportError:
    pass

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
