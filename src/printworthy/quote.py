"""quote.py -- STUB: instant PRINT-QUOTE engine (the bureau/B2B channel).

WHAT
====
A print bureau's intake problem: a customer uploads a mesh; someone must
decide "will this print, what will it cost, what should we charge" -- today a
human opens it in a slicer. We already have every ingredient headless:
ingress guards + printability gate (will it print), the fixer (make it
printable + what changed), slicer.estimate (real filament mm3 + print time
from the PrusaSlicer CLI), profiles (machine + material), batch.py (many
files). This module composes them into one deterministic quote document.

    quote(path_or_mesh, *, profile="generic_fdm", material="PLA",
          rates=None) -> {"ok", "verdict", "printable_as_is", "fix_summary",
                          "filament_g", "filament_cm3", "print_time_h",
                          "machine_cost", "material_cost", "labor_flag",
                          "total", "currency", "assumptions", "note"}
    quote_batch(paths, **kw) -> list of the above + a CSV/JSON manifest.

WHY (the wedge)
===============
This is the channel where ONE customer covers the entire $200/mo bar. The
pitch to a bureau: "pipe your intake through this; broken uploads get caught
(and fixed, with a certificate) before a human touches them, and every quote
is reproducible." Nobody sells this small; the incumbents sell platforms.

BUILD PLAN
==========
1. RATE CARD. `rates` dict (or rates.json next to the profile):
   {"currency": "EUR", "machine_per_h": 2.50, "material_per_kg": 25.0,
    "labor_flat": 5.0, "margin_pct": 30, "min_price": 8.0}. Ship a
   commented default; NEVER invent prices silently -- every number appears
   under "assumptions" in the output.
2. PIPELINE. prep(check_only=False, slicer_savings=False) for verdict + fix
   + certificate; then slicer.estimate on the PREPPED stl with the requested
   profile/material for filament_mm3 + print_time_s. grams via material
   density when the config-free profile reports 0 g (slicer.py precedent:
   prefer cm3, note the density source).
3. PRICE. material_cost = kg * material_per_kg; machine_cost = h *
   machine_per_h; labor_flag = True when the fixer had to do more than
   trivial repair (fix_summary severity) -- bureaus price human attention,
   so surface it as a FLAG + suggested flat fee, not a hidden number.
   total = (material + machine + labor) * (1 + margin) clamped to min_price.
4. DETERMINISM. Same file + same rates -> byte-identical quote JSON
   (timestamps in a sidecar, not the quote body) so bureaus can cache and
   audit. Include printworthy version + profile hash in "assumptions".
5. BATCH + API. quote_batch over a folder -> quotes.csv (one row per file:
   name, verdict, g, h, total, flags). The api.py stub's first real
   endpoint: POST /quote (multipart STL -> quote JSON). Rate-limit + size
   guards are already ingress-guard patterns.
6. WIRE-IN: CLI `printworthy quote part.stl --rates rates.json` and
   `printworthy quote ./uploads/ --csv quotes.csv`.

VALIDATION GATES
================
G1 reproducibility: two runs, byte-identical quote body.
G2 rate transparency control: empty rates -> quote refuses with "no rate
   card" (never invents prices -- exercises the exact trust failure).
G3 broken-upload path: a rebench broken mesh -> quote includes the fix
   certificate + labor_flag=True (the intake value prop, demonstrated).
G4 sanity: quoted grams within 2% of slicer gcode analysis on the 3-part
   control set (box, bracket, organic).

HONEST LABELS
=============
A quote is an ESTIMATE from a config-free slice unless the bureau supplies
their real printer profile .ini -- say which one was used. Print time from
the slicer's own estimator (its accuracy is the slicer's, not ours).

COMPUTE SAFETY: one part at a time (batch is sequential, 6-worker cap max if
parallelized later); slicer runs already serialized in slicer.py.
KILL: if bureaus in outreach (Phase 3) say intake triage isn't their pain,
park it -- the module costs nothing to keep as a stub.
"""
from __future__ import annotations

_NOT_BUILT = (
    "print quoting is a documented stub -- build plan in printworthy/quote.py. "
    "Nothing was modified.")


def quote(path_or_mesh, *, profile="generic_fdm", material="PLA", rates=None):
    """STUB. One mesh -> one reproducible quote document.

    Planned return: see module docstring step 2-3.
    Current behavior: honest refusal (never raises)."""
    return {"ok": False, "implemented": False, "total": None,
            "note": _NOT_BUILT}


def quote_batch(paths, **kw):
    """STUB. Many meshes -> list of quotes + CSV manifest.

    Current behavior: honest refusal (never raises)."""
    return {"ok": False, "implemented": False, "quotes": [],
            "note": _NOT_BUILT}
