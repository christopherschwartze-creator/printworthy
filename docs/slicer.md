# `printworthy.slicer` — slicer-CLI estimates ("saves X g / Y min")

Wraps an **installed** slicer's command line to slice a model and read the
slicer's own print-time / filament estimates out of the g-code comments.
This is what turns a repair or a graded-infill pass into a headline number:
`compare(before, after)` → *"after saves 12.3 g / 41.0 min (18.5 %)"*.

**AGPL-clean:** PrusaSlicer / SuperSlicer / OrcaSlicer / Bambu Studio are
AGPL. printworthy only *executes* their CLI as a subprocess and reads the text
files they write — nothing is linked, imported, or vendored. No slicer is
bundled; installing one is the user's choice.

**Never raises.** No slicer installed → `{"ok": False, "note": ...}` and the
pipeline simply reports estimates as unavailable (optional feature).

## API

```python
from printworthy import slicer

slicer.find_slicer()                # -> {"name": "prusa", "path": exe} | None
slicer.estimate("part.stl",         # -> {ok, print_time_s, filament_g,
                profile=None,       #     filament_mm3, gcode_path, slicer, note}
                slicer=None, timeout_s=300, out_dir=None)
slicer.compare("before.stl", "after.stl")
                                    # -> {ok, a, b, saved_g, saved_min,
                                    #     saved_pct, note}
slicer.parse_gcode_stats(text)      # footer parser (exposed for testing)
```

* Detection: PATH first (`prusa-slicer-console`, `prusa-slicer`,
  `superslicer_console`, `orca-slicer`, `bambu-studio`, ...), then common
  Windows install dirs (`Program Files\Prusa3D\PrusaSlicer`, `\OrcaSlicer`,
  `\Bambu Studio`, `%LOCALAPPDATA%\Programs\...`). Preference order
  prusa → superslicer → orca → bambu (Prusa's CLI slices config-free).
* Slice commands: Prusa/SuperSlicer `--export-gcode --output <g> <model>`;
  Orca/Bambu `--slice 0 --outputdir <dir> <model>` (their `*.gcode.3mf`
  output is unzipped and the embedded plate g-code parsed). Orca/Bambu CLIs
  usually **need** machine/process JSON via `profile["slicer_settings"]` /
  `profile["slicer_filaments"]`; without them the note says so.
* Profile keys mapped to CLI flags (Prusa path): `layer_height`,
  `infill_pct`, `nozzle_mm`, `slicer_ini` (a full `--load` ini wins for
  anything richer). Unknown keys are ignored.
* `saved_pct` in `compare()` is the **material** (grams) saving when both
  sides report grams, else the time saving; positive = second model cheaper.

## Comment formats parsed (`parse_gcode_stats`)

| Slicer | Lines parsed (footer unless noted) |
|---|---|
| PrusaSlicer / SuperSlicer | `; estimated printing time (normal mode) = 1h 32m 4s` · `; filament used [g] = 3.64` · `; filament used [cm3] = 2.93` · `; filament used [mm] = 1219.90` |
| OrcaSlicer / Bambu Studio | `; total estimated time: 32m 1s` (also when appended after `; model printing time: ...`) · `; total filament used [g] = 3.70` · `; filament used [cm3] =` · `; filament used [mm] =` |
| Cura (header) | `;TIME:5416` (seconds) · `;Filament used: 0.84m` |

Details: durations accept any subset of `Nd Nh Nm Ns`. Multi-extruder
comma lists (`3.50, 1.25`) are **summed**. `filament_mm3` comes from the
`[cm3]` line ×1000, or is derived from length assuming 1.75 mm filament when
only a length is present. Grams are never fabricated from volume (density
unknown) — `filament_g` stays `None` if the slicer didn't say.

## Honest label

The numbers are the *slicer's* estimates under the profile used (its
defaults unless you override). They are directly comparable between two
meshes sliced identically — the before/after use — but they are not a
promise about a particular printer's wall clock.

## Test status (2026-07-01, no slicer installed on the dev box)

31/31 unit checks pass: duration parser (6), Prusa footer (4), Orca/Bambu
footers (5), multi-extruder summing (2), Cura header (4), junk-input
never-raise (3), detection + graceful degrade of `estimate`/`compare` (5),
launch-failure and missing-model notes (2). A real end-to-end estimate still
needs a box with a slicer installed — first run on such a box is the
remaining validation step.
