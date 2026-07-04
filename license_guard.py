"""license_guard.py — PROVE the source imports no copyleft / non-commercial code.

AST-scans the given source trees for banned imports (GPL / LGPL-static / CGAL /
non-commercial) and a warn-list (license-flagged optional). Exit 0 = clean, 1 = a banned
import was found. Wire it into CI so the permissive-only guarantee can never silently rot.

    python license_guard.py src/                       # scan the vendored package
    python license_guard.py ../Forge/preflight ../Forge/reinforce ../Forge/autorig ...

Substring match on the dotted import path, so `from igl import copyleft`,
`import gpytoolbox.copyleft`, etc. are all caught. The libigl/​gpytoolbox *cores* (MPL-2 /
MIT) are NOT banned — only their `copyleft` submodules and the hard-GPL/CGAL/non-commercial
libraries.
"""
from __future__ import annotations

import ast
import os
import sys

# Hard-banned: GPL / CGAL / non-commercial. A match fails the scan.
BANNED = {
    "pymeshfix", "pymeshlab", "skeletor", "pinocchio", "tetgen", "pytetwild",
    "wildmeshing", "cgal", "comiso", "pycomiso", "rignet", "unirig", "meshlib",
    "igl.copyleft", "gpytoolbox.copyleft",
}
# License-flagged: allowed ONLY via an explicit, documented opt-in extra (not core).
WARN = {
    "pyquadriflow",   # prebuilt wheel statically links LGPL Eigen + no valid license
}


def _imports(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except Exception:
        return []
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out.extend(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            out.append(mod)
            out.extend((mod + "." + a.name) if mod else a.name for a in n.names)
    return out


def scan(roots):
    banned_hits, warn_hits, nfiles = [], [], 0
    for root in roots:
        if os.path.isfile(root) and root.endswith(".py"):
            walk = [(os.path.dirname(root), [], [os.path.basename(root)])]
        else:
            walk = os.walk(root)
        for dirpath, _dirs, files in walk:
            if "__pycache__" in dirpath or (os.sep + "tests") in dirpath:
                continue
            for f in files:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dirpath, f)
                nfiles += 1
                for imp in _imports(p):
                    low = imp.lower()
                    for b in BANNED:
                        if b in low:
                            banned_hits.append((p, imp, b))
                    for w in WARN:
                        if w in low:
                            warn_hits.append((p, imp, w))
    return nfiles, banned_hits, warn_hits


def main(argv):
    roots = argv or ["src"]
    nfiles, banned, warn = scan(roots)
    print(f"license_guard: scanned {nfiles} .py files under {roots}")
    for p, imp, w in warn:
        print(f"  WARN   {imp:32} license-flagged ({w}) — must be an opt-in extra: {p}")
    if banned:
        for p, imp, b in banned:
            print(f"  BANNED {imp:32} copyleft/non-commercial ({b}): {p}")
        print(f"FAIL: {len(banned)} banned import(s) — NOT permissive-clean.")
        return 1
    print(f"CLEAN: 0 banned copyleft/non-commercial imports "
          f"({len(warn)} license-flagged WARN, opt-in only).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
