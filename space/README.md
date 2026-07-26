---
title: printworthy
emoji: 🖨️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.19.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Drop a 3D model, get a print-ready file + a plain review
---

# printworthy — get your 3D model ready for a print shop

Drop a 3D model (GLB / OBJ / STL / PLY / OFF / 3MF). You get back:

- **One verdict** in plain words: ready to print / not printable yet /
  we can't read this file.
- **A faithful fix** with a trust line: *we touched X% of your surface,
  max deviation Y mm* — the rest is exactly your original.
- **A size sanity check**: we keep your file's own size and tell you what
  it reads as; you only rescale if you ask to.
- **A design review**: thin walls, trapped pockets, support-heavy angles,
  a suggested printing orientation — each warning with what to do next,
  plus visual risk previews.
- **Downloads**: the fixed model (`prep.stl`) and a shop-ready review
  (`review.md`).
- **Optional warp prediction** (~1 min): a real physics simulation,
  clearly labeled *uncalibrated* — it shows where and roughly how much,
  not certified numbers.

**Honesty rules baked in:** every physics number stays labeled
(estimate / uncalibrated / comparative); refusals are stated plainly,
never dressed up as success; verdicts are advisory — the final word
belongs to your printer or print shop.

**Privacy:** files are processed in memory, not stored, logged, or used
for training — nothing is uploaded anywhere except this Space.
Limits: 25 MB / 3M triangles.

Source: https://github.com/christopherschwartze-creator/printworthy
