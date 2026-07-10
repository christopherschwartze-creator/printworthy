# meshprep examples

Three copy-paste walkthroughs with the **exact commands** and the **real measured
outcomes** from the validation runs. Every number below is pulled from a report in this
repo (`FOOLPROOF_REPORT.md`, `ADJUSTMENT_REPORT.md`) or the pro `S2_FEATURES_REPORT.md` —
none are invented, and every physics estimate keeps its honesty label.

> Bring your own mesh: any `.glb / .stl / .obj / .ply`. The renders shown in the top-level
> [`README`](../README.md#see-it) come from these same code paths on a real AI-generated
> axe (54,672 faces from an image-to-3D model) and a set of CAD brackets.

---

## 1. Check + fix a broken AI mesh

AI image-to-3D meshes arrive as open-shell soup: unclosed boundary loops, phantom
tunnels, walls thinner than a nozzle. `check` tells you what's wrong in one sentence;
`fix` repairs it **source-accurately** and hands back a deviation certificate.

```bash
# analysis only — nothing is modified
meshprep check axe.glb

# the source-accurate repair alone (writes a fixed mesh + certificate)
meshprep fix axe.glb -o axe_fixed.stl
```

**What you get (measured on the real axe, `foolproof/results/ai_axe`):**

- Verdict **FAIL** — *"Won't print as-is — thinnest wall 0.23 mm < 0.40 mm nozzle. 1
  blocking issue; the free Fix resolves most."* Plus a WARN for **1 phantom tunnel**
  (genus 1) — the kind of artifact the source generator left behind.
- The fix keeps **100.0 % of the surface** verbatim, **0.000 mm** max deviation, stays
  **watertight**, and **preserves topology** (genus 1 in → 1 out) — in **0.92 s**.
- The mesh had no units (extent 1.99), so it is flagged and scaled longest-side to 60 mm
  with a plain WARN — never a silent guess.

**Across the whole broken-mesh corpus (66 AI meshes):** mean **99.4 % surface kept**,
**0.090 mm** mean max-deviation, **100 % watertight**, and the worst-10 (most-shattered
inputs) went from 25.7 % → **100.0 %** surface kept after the repair rewrite
(`ADJUSTMENT_REPORT.md`, F1). The fix **rolls back honestly** on un-closeable garbage
(`claims_success=False`) rather than shipping a mangled solid.

> **Honest scope:** the verdict is geometric heuristic triage (45° overhang +
> Shape-Diameter thin-wall), not a slicer simulation. "Watertight" is not the same as
> "genus-correct". See the [risk heatmap](../docs/images/risk-heatmap-ai-axe.png) and the
> [fix certificate](../docs/images/fix-certificate.png).

---

## 2. Reinforce a loaded part — graded-infill 3MF

Give `reinforce` a mesh **and a load case**; it runs the FEM, then writes a slicer-ready
**3MF** with a dense-infill modifier volume on the high-stress load path. Open the `.3mf`
in PrusaSlicer / Orca / Bambu and it prints as-is.

```bash
# load along +Z (axis 2), 200 N; two graded tiers: 50% then 100% infill toward the core
meshprep reinforce bracket.stl -o reinforced.3mf \
    --load-axis 2 --force 200 --tiers "55:50:mid,85:100:core"
```

**What you get:**

- A 3MF whose dense modifier follows the FEM stress field. In the validation cantilever,
  mean von Mises **inside the reinforced core is 6.64 vs 3.41** for a random-placement
  control — the reinforcement lands where the load actually concentrates
  (see [the render](../docs/images/reinforce-dense-core.png)).
- A **gradient pre-screen**: if the stress field is too uniform for a graded split to
  matter, it tells you plainly to use flat uniform infill instead of pretending a split
  helps (the pro multi-material path refuses the same way — *"stress too uniform for a
  material split to matter"*, `S2_FEATURES_REPORT.md`).

> **Honest scope:** the importance field is a **relative / comparative, uncalibrated**
> load path — it is *not* a certified factor-of-safety. Run `meshprep calibrate` on one
> printed coupon to turn the physics estimates into calibrated millimetres for your
> printer + filament.

---

## 3. Risk-driven supports (SupportEnforcer / SupportBlocker)

The premortem risk field can emit PrusaSlicer support-modifier volumes: **enforcers** where
overhang risk is real, **blockers** over false-positive overhangs. The volume types are
schema-verified against PrusaSlicer's `Model.cpp` / `3mf.cpp`, and the slicing effect is
**measured from gcode**, not asserted.

```bash
# standalone: mesh -> risk-driven SupportEnforcer/Blocker 3MF
meshprep supports l_bracket.stl -o bracket_supported.3mf

# or fold it into the full pipeline at the shipped orientation
meshprep prep l_bracket.stl --supports
```

Prefer the Python API? It returns the enforcer/blocker counts and an honesty note:

```python
import trimesh
from meshprep.core.support_mods import support_mods

mesh = trimesh.load("l_bracket.stl")
out = support_mods(mesh, out_3mf="bracket_supported.3mf")
print(out["ok"], out["n_enforcers"], out["n_blockers"], out["note"])
# open bracket_supported.3mf in PrusaSlicer; result carries "uncalibrated": True
```

**What was measured** (PrusaSlicer 2.9.6 CLI, auto-detect OFF, `S2_FEATURES_REPORT.md`):

- On a watertight L-bracket with one 20 mm overhang shelf: baseline **0 support segments /
  2.99 cm³** → with the enforcer volume, **66 segments / 4.26 cm³** = **+1.27 cm³ (+42.5 %)**
  of support material, attributable solely to the enforcer.
- A **blocker** over a false-positive overhang removes it **back to the exact 2.99 cm³
  no-support baseline (−26.7 %)**.
- A purpose-built **40° chamfer cone** correctly yields **zero enforcers** — *"no supports
  needed"* — the false-positive control.

> **Honest scope:** verified on **PrusaSlicer 2.9.6** only; Orca/Bambu use a different
> `model_settings.config` schema (noted, unverified). Enforcer geometry must volumetrically
> overlap the solid by a few mm — a merely-touching box slices to byte-identical gcode
> (a measured pitfall, handled by the module).

---

## Bonus — smart split for an oversized part

When a part won't fit the bed, `split_for_bed(..., smart=True)` scores seams into concave
creases (EI neck-cut) or CoACD interfaces, makes an exact `manifold3d` boolean cut, and
adds peg/socket connectors with a printable fit coupon.

```python
from meshprep.split import split_for_bed
result = split_for_bed(mesh, profile="generic_fdm", smart=True)   # opt-in
```

Measured (`S2_FEATURES_REPORT.md`): a 90×86×18 mm bracket that won't fit a 60 mm bed →
**7 printable, bed-fitting parts joined by 18 peg/socket connectors**, volume conserved to
**0.047 %** (pure boolean neck-cut: **0.000 %**). On a dumbbell control the seam lands **on
the neck** (origin x = −3.48; 3.48 mm to the neck ≪ 12.5 mm to the nearest lobe), defeating
a naive mid-bbox cut.

> **Honest scope:** geometric-bisection seams are labeled **VISIBLE** (they land in a
> crease, but are not hidden-in-crease quality). CoACD runs in a timeout-guarded child
> process; on failure the split degrades to a labeled loose plane cut rather than crashing.
