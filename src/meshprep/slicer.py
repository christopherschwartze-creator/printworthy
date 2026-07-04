"""meshprep.slicer — external slicer CLI wrapper: the "saves X g / Y min" number.

Drives an INSTALLED slicer (PrusaSlicer / SuperSlicer / OrcaSlicer / Bambu
Studio) as a subprocess to slice a model and parse print time + filament use
out of the g-code footer comments. That turns meshprep's geometry work
(repair, orientation, graded infill) into a number the user cares about:
``compare(before, after)`` -> "saves 12.3 g / 41 min (18.5 %)".

License hygiene (AGPL-CLEAN): the slicers are AGPL. This module only ever
*executes their CLI* in a separate process and reads the text files they
write. Nothing is linked, imported, or vendored from them. Installing a
slicer is the user's choice; without one every call degrades to
``{"ok": False, "note": ...}`` and the pipeline treats the numbers as
unavailable-but-optional.

Never-raise discipline: every public function returns a dict; failures are
``ok: False`` plus a plain-English ``note``. Subprocesses get a hard timeout.

Honest labels: the numbers are the SLICER'S estimates under the profile used
(its defaults unless you pass overrides/ini). They are comparable between two
meshes sliced the same way — which is exactly the before/after use — but they
are not a promise about a specific printer's clock.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

__all__ = ["find_slicer", "estimate", "compare", "parse_gcode_stats"]

_TIMEOUT_S = 300          # hard wall per slice subprocess
_FIL_DIAMETER_MM = 1.75   # only used to derive mm3 from a length-only footer

# ------------------------------------------------------------------ detection

# (name, executable stems). Order = preference: PrusaSlicer's CLI is the most
# reliable for a plain --export-gcode; Orca/Bambu share a quirkier CLI.
_CANDIDATES = (
    ("prusa", ("prusa-slicer-console", "prusa-slicer")),
    ("superslicer", ("superslicer_console", "superslicer")),
    ("orca", ("orca-slicer", "orcaslicer")),
    ("bambu", ("bambu-studio", "bambustudio")),
)


def _windows_install_dirs():
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    lap = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")
    out = []
    for base in (pf, pf86, lap):
        if not base:
            continue
        out += [
            ("prusa", os.path.join(base, "Prusa3D", "PrusaSlicer",
                                   "prusa-slicer-console.exe")),
            ("prusa", os.path.join(base, "PrusaSlicer",
                                   "prusa-slicer-console.exe")),
            ("orca", os.path.join(base, "OrcaSlicer", "orca-slicer.exe")),
            ("bambu", os.path.join(base, "Bambu Studio", "bambu-studio.exe")),
            ("superslicer", os.path.join(base, "SuperSlicer",
                                         "superslicer_console.exe")),
        ]
    return out


def find_slicer(prefer=None):
    """Locate an installed slicer CLI.

    Checks PATH first, then the common Windows install directories.
    Returns ``{"name": "prusa"|"superslicer"|"orca"|"bambu", "path": exe}``
    or ``None`` if nothing is installed. ``prefer`` (a name) is tried first.
    Never raises."""
    try:
        import shutil
        order = list(_CANDIDATES)
        if prefer:
            order.sort(key=lambda c: c[0] != prefer)
        for name, stems in order:
            for stem in stems:
                path = shutil.which(stem)
                if path:
                    return {"name": name, "path": path}
        if sys.platform == "win32":
            hits = [(n, p) for n, p in _windows_install_dirs()
                    if os.path.isfile(p)]
            if prefer:
                hits.sort(key=lambda c: c[0] != prefer)
            if hits:
                return {"name": hits[0][0], "path": hits[0][1]}
    except Exception:
        pass
    return None


# ------------------------------------------------------- g-code footer parse

# Duration like "2d 11h 32m 4s" (any subset of fields, PrusaSlicer/Orca form).
_DUR_RE = re.compile(
    r"(?:(\d+)\s*d)?\s*(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?\s*$",
    re.IGNORECASE)

_TIME_LINE_RES = (
    # PrusaSlicer/SuperSlicer: "; estimated printing time (normal mode) = 1h 32m 4s"
    re.compile(r"^;\s*estimated printing time(?:\s*\([^)]*\))?\s*[=:]\s*(.+?)\s*$",
               re.IGNORECASE),
    # OrcaSlicer / Bambu Studio: "; total estimated time: 1h 12m 30s"
    re.compile(r"^;\s*total estimated time\s*[=:]\s*(.+?)\s*$", re.IGNORECASE),
    # Orca variant: "; model printing time: 27m 5s; total estimated time: 32m 1s"
    re.compile(r";\s*total estimated time\s*[=:]\s*([^;]+?)\s*$", re.IGNORECASE),
)

# Cura puts ";TIME:5416" (seconds) in the HEADER.
_CURA_TIME_RE = re.compile(r"^;\s*TIME\s*:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
# Cura ";Filament used: 1.23456m" (meters; may list several extruders).
_CURA_FIL_RE = re.compile(r"^;\s*Filament used\s*:\s*(.+?)\s*$",
                          re.IGNORECASE | re.MULTILINE)

def _fil_line_re(unit):
    # Prusa: "; filament used [g] = 3.70"   Orca/Bambu: "; total filament used [g] = 3.70"
    # Multi-extruder prints emit comma-separated values -> summed by the caller.
    return re.compile(
        r"^;\s*(?:total\s+)?filament used\s*\[%s\]\s*[=:]\s*([\d.,\s]+?)\s*$"
        % re.escape(unit), re.IGNORECASE | re.MULTILINE)

_FIL_G_RE = _fil_line_re("g")
_FIL_CM3_RE = _fil_line_re("cm3")
_FIL_MM_RE = _fil_line_re("mm")


def _parse_duration_s(text):
    """'2d 1h 3m 5s' (any subset) -> seconds, or None."""
    try:
        m = _DUR_RE.match(text.strip())
        if not m or not any(m.groups()):
            return None
        d, h, mi, s = (int(g) if g else 0 for g in m.groups())
        return d * 86400 + h * 3600 + mi * 60 + s
    except Exception:
        return None


def _sum_csv_floats(text):
    """'3.50, 0.00' -> 3.5 (multi-extruder footers list one value per tool)."""
    try:
        vals = [float(v) for v in text.split(",") if v.strip()]
        return sum(vals) if vals else None
    except ValueError:
        return None


def parse_gcode_stats(text):
    """Parse slicer estimate comments out of g-code text.

    Understands PrusaSlicer/SuperSlicer footers, OrcaSlicer/Bambu Studio
    footers, and Cura headers (see docs/slicer.md for the exact lines).
    Returns ``{print_time_s, filament_g, filament_mm3, filament_mm}`` with
    ``None`` for anything absent. Never raises."""
    out = {"print_time_s": None, "filament_g": None,
           "filament_mm3": None, "filament_mm": None}
    try:
        for line in text.splitlines():
            if out["print_time_s"] is None and line.lstrip().startswith(";"):
                for rx in _TIME_LINE_RES:
                    m = rx.match(line.strip()) or rx.search(line)
                    if m:
                        sec = _parse_duration_s(m.group(1))
                        if sec is not None:
                            out["print_time_s"] = sec
                            break
        for key, rx in (("filament_g", _FIL_G_RE),
                        ("filament_mm3", _FIL_CM3_RE),
                        ("filament_mm", _FIL_MM_RE)):
            m = rx.search(text)
            if m:
                v = _sum_csv_floats(m.group(1))
                if v is not None:
                    out[key] = v * 1000.0 if key == "filament_mm3" else v  # cm3->mm3
        # Cura fallbacks (header comments)
        if out["print_time_s"] is None:
            m = _CURA_TIME_RE.search(text)
            if m:
                out["print_time_s"] = int(m.group(1))
        if out["filament_mm"] is None:
            m = _CURA_FIL_RE.search(text)
            if m:
                meters = _sum_csv_floats(m.group(1).replace("m", ""))
                if meters is not None:
                    out["filament_mm"] = meters * 1000.0
        # derive mm3 from length if the footer only gave a length
        if out["filament_mm3"] is None and out["filament_mm"] is not None:
            import math
            area = math.pi * (_FIL_DIAMETER_MM / 2.0) ** 2
            out["filament_mm3"] = out["filament_mm"] * area
    except Exception:
        pass
    return out


# ------------------------------------------------------------------ estimate

def _profile_flags(name, profile):
    """Map the (optional) meshprep profile dict to slicer CLI flags.

    Only settings every target CLI understands are mapped; anything richer
    goes through ``profile["slicer_ini"]`` (Prusa ``--load``) or
    ``profile["slicer_settings"]`` (Orca/Bambu ``--load-settings``)."""
    p = profile or {}
    flags = []
    if name in ("prusa", "superslicer"):
        ini = p.get("slicer_ini")
        if ini:
            flags += ["--load", str(ini)]
        if p.get("layer_height"):
            flags += ["--layer-height", str(p["layer_height"])]
        if p.get("infill_pct") is not None:
            flags += ["--fill-density", "%s%%" % p["infill_pct"]]
        if p.get("nozzle_mm"):
            flags += ["--nozzle-diameter", str(p["nozzle_mm"])]
    elif name in ("orca", "bambu"):
        st = p.get("slicer_settings")
        if st:
            if isinstance(st, (list, tuple)):
                st = ";".join(str(s) for s in st)
            flags += ["--load-settings", st]
        fl = p.get("slicer_filaments")
        if fl:
            if isinstance(fl, (list, tuple)):
                fl = ";".join(str(s) for s in fl)
            flags += ["--load-filaments", fl]
    return flags


def _read_gcode_text(gcode_path, cap_bytes=1_500_000):
    """Read enough of a g-code file to see header+footer comments (Prusa puts
    estimates at the END; Cura at the START). Reads head+tail on huge files."""
    size = os.path.getsize(gcode_path)
    with open(gcode_path, "r", encoding="utf-8", errors="replace") as f:
        if size <= cap_bytes:
            return f.read()
        head = f.read(cap_bytes // 3)
        f.seek(max(0, size - cap_bytes * 2 // 3))
        return head + "\n" + f.read()


def _gcode_from_3mf(path_3mf, out_dir):
    """Orca/Bambu export sliced plates as *.gcode.3mf (a zip with the g-code
    inside, e.g. Metadata/plate_1.gcode). Extract the first g-code member."""
    import zipfile
    with zipfile.ZipFile(path_3mf) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".gcode")]
        if not names:
            return None
        dst = os.path.join(out_dir, os.path.basename(names[0]))
        with z.open(names[0]) as src, open(dst, "wb") as f:
            f.write(src.read())
        return dst


def _run(cmd, timeout_s):
    kw = {}
    if sys.platform == "win32":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout_s, **kw)   # list argv, shell=False:
    # safe with spaces in paths on Windows, no quoting games.


def estimate(model_path, profile=None, slicer=None,
             timeout_s=_TIMEOUT_S, out_dir=None):
    """Slice ``model_path`` with an installed slicer CLI and return its own
    print-time / filament estimates.

    Returns ``{ok, print_time_s, filament_g, filament_mm3, gcode_path,
    slicer, note}``. ``ok: False`` (with a friendly ``note``) when no slicer
    is installed, the slice fails, or nothing parseable comes back — the
    pipeline treats that as "estimates unavailable", never an error.
    Never raises."""
    res = {"ok": False, "print_time_s": None, "filament_g": None,
           "filament_mm3": None, "gcode_path": None, "slicer": None,
           "note": ""}
    try:
        # -- resolve the slicer -------------------------------------------
        if isinstance(slicer, str):
            name = next((n for n, _ in _CANDIDATES
                         if n.split("-")[0] in os.path.basename(slicer).lower()
                         or n in os.path.basename(slicer).lower()), None)
            slicer = {"name": name or "prusa", "path": slicer}
        if slicer is None:
            slicer = find_slicer((profile or {}).get("slicer_prefer"))
        if not slicer:
            res["note"] = ("No slicer CLI found (looked for PrusaSlicer, "
                           "SuperSlicer, OrcaSlicer, Bambu Studio on PATH and "
                           "in Program Files). Print-time/filament estimates "
                           "are optional; install any of them to enable the "
                           "'saves X g / Y min' numbers.")
            return res
        res["slicer"] = slicer
        model_path = os.path.abspath(str(model_path))
        if not os.path.isfile(model_path):
            res["note"] = "Model file not found: %s" % model_path
            return res

        # -- slice ---------------------------------------------------------
        work = out_dir or tempfile.mkdtemp(prefix="meshprep_slice_")
        os.makedirs(work, exist_ok=True)
        stem = os.path.splitext(os.path.basename(model_path))[0]
        name, exe = slicer["name"], slicer["path"]
        flags = _profile_flags(name, profile)
        if name in ("prusa", "superslicer"):
            gcode = os.path.join(work, stem + ".gcode")
            # --center drops the model onto the bed regardless of its source
            # coordinates. A 3MF from reinforce keeps the mesh's ORIGINAL x/y/z,
            # which the CLI (unlike the GUI) does NOT auto-arrange -> otherwise
            # "objects outside print volume". 110,110 is on-bed for the common
            # 220x220 and 250x210 default beds; override via profile['bed_center'].
            bc = str((profile or {}).get("bed_center", "110,110"))
            cmd = ([exe, "--export-gcode", "--center", bc, "--output", gcode]
                   + flags + [model_path])
        else:  # orca / bambu: --slice 0 = all plates; writes *.gcode(.3mf)
            gcode = None
            cmd = [exe, "--slice", "0", "--outputdir", work] + flags + [model_path]
        try:
            proc = _run(cmd, timeout_s)
        except subprocess.TimeoutExpired:
            res["note"] = ("Slicer timed out after %ds (%s). The model may be "
                           "very large; estimates skipped." % (timeout_s, name))
            return res
        except OSError as e:
            res["note"] = "Could not launch slicer %s: %s" % (exe, e)
            return res

        # -- locate the g-code ----------------------------------------------
        if gcode is None or not os.path.isfile(gcode):
            produced = sorted(
                (os.path.join(work, f) for f in os.listdir(work)
                 if f.lower().endswith((".gcode", ".gcode.3mf"))),
                key=os.path.getmtime, reverse=True)
            gcode = produced[0] if produced else None
        if gcode and gcode.lower().endswith(".3mf"):
            gcode = _gcode_from_3mf(gcode, work)
        if not gcode or not os.path.isfile(gcode):
            tail = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()
            res["note"] = ("Slicer (%s) produced no g-code (exit %s). %s" %
                           (name, proc.returncode, tail[-400:].strip()))
            if name in ("orca", "bambu") and "--load-settings" not in cmd:
                res["note"] += (" Orca/Bambu CLIs usually need machine/process "
                                "settings: pass profile['slicer_settings'] "
                                "(JSON paths) or install PrusaSlicer for "
                                "config-free estimates.")
            return res

        # -- parse ---------------------------------------------------------
        stats = parse_gcode_stats(_read_gcode_text(gcode))
        res.update({k: stats[k] for k in
                    ("print_time_s", "filament_g", "filament_mm3")})
        res["gcode_path"] = gcode
        if res["print_time_s"] is None and res["filament_mm3"] is None:
            res["note"] = ("Sliced OK but no estimate comments recognized in "
                           "the g-code footer (unknown slicer format?). "
                           "G-code kept at %s." % gcode)
            return res
        res["ok"] = True
        res["note"] = ("Estimates are %s's own numbers for the profile used "
                       "(slicer defaults unless overridden)." % name)
        return res
    except Exception as e:  # never-raise
        res["note"] = "Slicer estimate failed: %s: %s" % (type(e).__name__, e)
        return res


# ------------------------------------------------------------------- compare

def compare(path_a, path_b, profile=None, slicer=None, timeout_s=_TIMEOUT_S,
            label_a="before", label_b="after"):
    """Slice two models the SAME way and report the savings of b over a —
    the headline "saves X g / Y min" for before/after repair or
    normal-vs-graded-infill.

    Returns ``{ok, a, b, saved_g, saved_min, saved_pct, note}`` where ``a``
    and ``b`` are the two ``estimate()`` dicts. ``saved_pct`` is the material
    (grams) saving when both grams are known, else the time saving; positive
    means ``path_b`` is cheaper. Never raises."""
    out = {"ok": False, "a": None, "b": None, "saved_g": None,
           "saved_cm3": None, "saved_min": None, "saved_pct": None, "note": ""}
    try:
        if slicer is None:
            slicer = find_slicer((profile or {}).get("slicer_prefer"))
            if not slicer:
                out["note"] = ("No slicer CLI installed - before/after "
                               "savings unavailable (optional).")
                return out
        a = estimate(path_a, profile=profile, slicer=slicer, timeout_s=timeout_s)
        b = estimate(path_b, profile=profile, slicer=slicer, timeout_s=timeout_s)
        out["a"], out["b"] = a, b
        if not (a["ok"] and b["ok"]):
            out["note"] = ("Could not slice both models: %s=%s | %s=%s" %
                           (label_a, a["note"], label_b, b["note"]))
            return out
        # Grams need a filament DENSITY in the profile; the config-free default
        # profile reports 0 g but ALWAYS reports cm3 -> prefer volume when grams
        # are absent or zero, so the saving is never a misleading "0.0 g".
        ga, gb = a["filament_g"], b["filament_g"]
        va, vb = a["filament_mm3"], b["filament_mm3"]
        if ga and gb:                       # both > 0 (density was set)
            out["saved_g"] = round(ga - gb, 2)
        if va is not None and vb is not None:
            out["saved_cm3"] = round((va - vb) / 1000.0, 2)
        if a["print_time_s"] is not None and b["print_time_s"] is not None:
            out["saved_min"] = round((a["print_time_s"] - b["print_time_s"]) / 60.0, 1)
        # saved_pct: grams if we have them, else volume, else time
        if out["saved_g"] is not None and ga:
            out["saved_pct"] = round(100.0 * out["saved_g"] / ga, 1)
        elif out["saved_cm3"] is not None and va:
            out["saved_pct"] = round(100.0 * (va - vb) / va, 1)
        elif out["saved_min"] is not None and a["print_time_s"]:
            out["saved_pct"] = round(100.0 * (a["print_time_s"] - b["print_time_s"])
                                     / a["print_time_s"], 1)
        out["ok"] = any(out[k] is not None
                        for k in ("saved_g", "saved_cm3", "saved_min"))
        if out["ok"]:
            if out["saved_g"] is not None:
                mat = "%s g" % out["saved_g"]
            elif out["saved_cm3"] is not None:
                mat = "%s cm3" % out["saved_cm3"]
            else:
                mat = "? material"
            m = ("%s min" % out["saved_min"]
                 if out["saved_min"] is not None else "? min")
            out["note"] = ("%s saves %s / %s vs %s (slicer estimates, same profile "
                           "both sides; set a filament density for grams)."
                           % (label_b, mat, m, label_a))
        else:
            out["note"] = "Sliced both, but the footers carried no comparable numbers."
        return out
    except Exception as e:
        out["note"] = "Comparison failed: %s: %s" % (type(e).__name__, e)
        return out
