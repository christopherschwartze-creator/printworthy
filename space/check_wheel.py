"""Deploy guard: refuse to ship a stale printworthy wheel.

The Space installs printworthy from ./wheels/*.whl. A wheel built before the
latest source change silently ships an old backend (this actually happened:
a wheel predating the whole 'space' preset feature sat in wheels/ ready to
deploy). Run this BEFORE every push; it exits non-zero when the wheel must
be rebuilt.

Checks
  1. exactly one wheel in wheels/, its filename version matches the package's
     actual __version__, and requirements.txt does NOT reference any local
     .whl (regression guard: HF's build system installs requirements.txt in
     an ISOLATED stage where the rest of the repo, including wheels/, is not
     yet present -- a local wheel path there fails at build time with a bare
     OSError, hit in production 2026-07-25. The wheel is installed at RUNTIME
     instead, by app.py's _ensure_printworthy(), once the full repo exists);
  2. the wheel is newer than every tracked source file in src/printworthy
     (build/ artifacts and caches ignored);
  3. the wheel's own copies of the feature-critical modules byte-match the
     source tree (profiles.py / pipeline.py / report.py / app.py) — mtime
     alone can lie across clones.

Usage:  python space/check_wheel.py
"""
from __future__ import annotations

import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)                       # .../printworthy
SRC = os.path.join(PKG_ROOT, "src", "printworthy")
WHEELS = os.path.join(HERE, "wheels")
REQS = os.path.join(HERE, "requirements.txt")

# modules whose staleness would ship a silently-wrong backend
CRITICAL = ["profiles.py", "pipeline.py", "report.py", "app.py"]


def fail(msg):
    print(f"[check_wheel] FAIL: {msg}")
    print("[check_wheel] rebuild with:\n  pip wheel "
          f"{PKG_ROOT} -w {WHEELS} --no-deps")
    return 1


def main() -> int:
    wheels = sorted(f for f in os.listdir(WHEELS) if f.endswith(".whl")) \
        if os.path.isdir(WHEELS) else []
    if len(wheels) != 1:
        return fail(f"expected exactly one wheel in wheels/, found {wheels}")
    whl = os.path.join(WHEELS, wheels[0])

    # 1a. requirements.txt must NOT reference a local wheel (regression guard
    #     for the build-stage-isolation bug — see module docstring)
    try:
        with open(REQS, encoding="utf-8") as fh:
            reqs = fh.read()
    except OSError as e:
        return fail(f"cannot read requirements.txt: {e}")
    active_lines = [ln for ln in reqs.splitlines()
                    if ln.strip() and not ln.strip().startswith("#")]
    if any(".whl" in ln for ln in active_lines):
        return fail(
            "requirements.txt has an active (non-comment) line referencing "
            "a local .whl again — this WILL fail at build time (HF installs "
            "requirements.txt before the rest of the repo exists). The "
            "wheel must be installed at runtime by app.py's "
            "_ensure_printworthy(), not listed here.")

    # 1b. the wheel's filename version must match the package's real version
    try:
        sys.path.insert(0, os.path.join(PKG_ROOT, "src"))
        import printworthy as _pw
        real_version = _pw.__version__
    except Exception as e:
        return fail(f"could not import printworthy to check its version: {e}")
    if real_version not in wheels[0]:
        return fail(f"wheel filename {wheels[0]!r} does not match "
                    f"printworthy.__version__ {real_version!r}")

    # 2. wheel newer than every source file
    wt = os.path.getmtime(whl)
    stale = []
    for root, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith((".pyc", ".pyo")):
                continue
            p = os.path.join(root, f)
            if os.path.getmtime(p) > wt:
                stale.append(os.path.relpath(p, PKG_ROOT))
    if stale:
        return fail("wheel is OLDER than source file(s): "
                    + ", ".join(stale[:8])
                    + ("..." if len(stale) > 8 else ""))

    # 3. byte-compare the feature-critical modules
    try:
        with zipfile.ZipFile(whl) as z:
            names = set(z.namelist())
            for mod in CRITICAL:
                arc = f"printworthy/{mod}"
                if arc not in names:
                    return fail(f"wheel is missing {arc}")
                src_path = os.path.join(SRC, mod)
                with open(src_path, "rb") as fh:
                    if z.read(arc) != fh.read():
                        return fail(f"wheel copy of {mod} differs from "
                                    "src/printworthy — stale build")
    except zipfile.BadZipFile:
        return fail("wheel is not a valid zip archive")

    print(f"[check_wheel] OK: {wheels[0]} is fresh "
          "(newer than src, critical modules byte-identical).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
