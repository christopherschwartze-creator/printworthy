# Name research — "meshprep"

> **This is preliminary name-availability research to hand to counsel. It is NOT
> a trademark clearance, NOT legal advice, and NOT an opinion that the name is
> available, cleared, or safe to use.** Where a source shows nothing, that means
> "no evidence found at that source on that date," not "clear." The authoritative
> trademark registers (USPTO, EUIPO) were reachable but could not be searched
> programmatically here and **must be searched by a qualified attorney.**

**Subject:** `meshprep` — proposed name for the free/OSS 3D-print mesh pre-flight
tool (repair + printability + print-FEM warp/strength + graded-infill 3MF).
**Research dates:** package/registry checks 2026-07-03 (PyPI, prior) and
2026-07-11 (this pass). **Prepared by:** automated research pass +
hands-on re-verification of the two load-bearing collisions.

---

## Bottom line for the naming decision (observations, not a clearance)

Two things counsel and Ted should weigh before committing the name:

1. **Two pre-existing open-source projects are already named `meshprep` /
   `MeshPrep`, in the *exact same* 3D-print-mesh-repair niche** (details in §2,
   both re-verified by direct GitHub API fetch on 2026-07-11). Neither is
   high-traction (0–1 stars) and neither owns the bare `meshprep` package/handle
   namespaces — but same-name-same-domain overlap is real and is the single most
   material finding here (open-source community confusion, potential trademark
   priority questions).
2. **"mesh prep" is a common descriptive term of art** in this field (SOLIDWORKS
   "Mesh Prep Wizard", Fusion "Mesh preparation", plus generic screen-printing
   "mesh prep"). A descriptive name is inherently **weaker/harder to protect** as
   a trademark absent acquired distinctiveness (§5).

The *registry namespaces* (PyPI, npm, the GitHub `meshprep` org handle, Hugging
Face) all appear **open/unclaimed** as of the dates below. So the availability
question is not "can I register the package name" (yes) but "is the *brand*
defensible and non-confusing" (the two items above — a counsel question).

---

## Collisions summary

| Source | Exact `meshprep`? | Result (date) |
|---|---|---|
| PyPI package `meshprep` | — | **Unregistered** — JSON API 404 (2026-07-03 and 2026-07-11) |
| PyPI variants `mesh-prep` / `meshprepare` / `meshprepper` | — | All unregistered (404), 2026-07-11 |
| npm `meshprep` / `mesh-prep` | — | Both unregistered (404), 2026-07-11 |
| GitHub org/user handle `github.com/meshprep` | — | Does not exist (404) — unclaimed |
| **GitHub repo `doccaz/meshprep`** | **Yes (exact)** | **COLLISION** — same domain; 0★ (2026-07-11) |
| **GitHub repo `DragonAceNL/MeshPrep`** | **Yes (camelCase)** | **COLLISION** — same domain; 1★ (2026-07-11) |
| Hugging Face user/org/model `meshprep` | — | None found (404 / no results), 2026-07-11 |
| Registered trademark "meshprep" (web-indexed) | — | No filing surfaced; **USPTO/EUIPO not searched — counsel must run** |
| Domain names `meshprep.{com,io,dev,org}` | — | **Inconclusive** — local DNS hijacks NXDOMAIN; needs proper WHOIS |
| Generic phrase "mesh prep" | n/a | Widely used descriptively → **descriptiveness/strength concern** |

---

## 1. PyPI (package name)

- `https://pypi.org/pypi/meshprep/json` → **HTTP 404 — not registered** (the
  authoritative signal). Also checked 2026-07-03 (prior pass): free.
- ⚠️ Note for whoever re-checks: `https://pypi.org/project/meshprep/` currently
  returns HTTP **200 with a Cloudflare "Client Challenge" interstitial** — that
  200 is the bot-wall, **not** a real package page. Use the **`/pypi/<name>/json`
  endpoint** (404 = free) as the reliable check, not the HTML 200.
- Variants `mesh-prep`, `meshprepare`, `meshprepper` → all 404 (free).

**Finding:** the PyPI package name is unclaimed. (This repo does **not** publish
to PyPI regardless — see README/CHANGELOG; it installs from source.)

## 2. GitHub — the two real collisions (re-verified 2026-07-11)

Direct GitHub REST API fetches, this pass:

- **`doccaz/meshprep`** — status 200 (exists). Description (verbatim):
  *"Browser-based OBJ mesh editor and 3D print optimizer for FDM/SLA printing."*
  Language TypeScript, **0 stars**, created 2026-06-17, last push 2026-06-17.
- **`DragonAceNL/MeshPrep`** — status 200 (exists). Description (verbatim):
  *"Automatically fix broken 3D models for printing. Repairs holes, errors, and
  bad geometry — then verifies with your slicer. No 3D modeling skills needed."*
  Language C#, **1 star**, created 2025-12-31, last push 2026-01-10.

Both are the **same name in the same product category** as this tool (the second
one's positioning — auto-repair broken print models, verify with a slicer — is
strikingly close). Low traction, but a genuine overlap. The bare org handle
`github.com/meshprep` is **unclaimed** (404).

Other, non-colliding matches (different domain, listed for completeness):
`simofoti/MeshPreprocessing` (ML mesh registration, 12★),
`Vinc3r/MeshPrepAndExportForBlender` (Blender export add-on),
`pronoypatra/meshpreprocess`, `pratishtha3105/meshpreproc-mixer`,
`kiranhegde/MeshPreprocessorUG3` (generic preprocessing utilities).

## 3. Hugging Face

- User/org `https://huggingface.co/meshprep` → 404 (does not exist).
- Model search `?search=meshprep` → no results.

**Finding:** no HF Space/model/org/user named `meshprep` found (2026-07-11).

## 4. Trademark / general web

Searches run 2026-07-11: `"meshprep"`; `"mesh prep" software 3d printing`;
`"meshprep" 3D print mesh preparation preflight`;
`meshprep trademark registered software`; `meshprep trademark USPTO justia`;
`meshprep github pypi 3d printing mesh repair tool`.

- **"mesh prep" as a generic/descriptive phrase — heavily used:** screen-printing
  chemicals ("Universal Mesh Prep" — MacDermid/Grimco/Ulano/Fujifilm/AlbaChem);
  CAD features (SOLIDWORKS "Mesh Prep Wizard" / ScanTo3D, Autodesk Fusion "Mesh
  preparation"); Meshmixer/MeshLab colloquially called "mesh prep" tools.
- **"meshprep" as a distinctive one-word brand:** outside the two GitHub repos in
  §2, **no evidence found** of a company/commercial product/brand using the single
  word "meshprep" in software/3D/CAD/manufacturing (2026-07-11).
- **Registries:**
  - **USPTO** `https://tmsearch.uspto.gov/` — reachable (200) but JS/session-driven;
    a real wordmark search **could not be completed programmatically** and must be
    run by counsel in the live system.
  - **EUIPO / TMview** — reachable (200) but the public search API returned no
    usable scripted response. **Not completed** — counsel must run.
  - **Justia Trademarks** `.../search?q=meshprep` — bot-blocked (403), unreadable.
    Web searches surfaced no "meshprep" filing; they did surface an unrelated
    adjacent mark **"MESHTECH" (Mesh Systems, LLC)** — noted only as an
    adjacent-space example, not a "meshprep" hit.
- **npm:** `meshprep` and `mesh-prep` both 404 (unregistered).
- **Domains:** inconclusive — the local resolver hijacks NXDOMAIN (a control lookup
  of a deliberately-fake domain returned the same parking IP), so DNS gives no
  reliable read; `whois` unavailable here. **Domain status needs a proper WHOIS.**

## 5. Descriptiveness flag (observation for counsel, not a legal conclusion)

"meshprep" reads as compressed "**mesh prep**[aration]", and "mesh preparation"
is an established term of art in exactly this field (§4). For mesh-preparation
software the name looks **likely descriptive** of what the tool does. Descriptive
marks are weaker and harder to register/protect absent acquired distinctiveness.
Flagging only — counsel should assess whether one-word compression, stylization,
or a logo adds enough distinctiveness, and whether a more arbitrary/suggestive
brand would be more defensible.

---

## What counsel still needs to do (this research does NOT substitute)

1. Run authoritative **USPTO** (TESS/live search) and **EUIPO/TMview** wordmark
   searches for "meshprep" and "mesh prep" in the relevant Nice classes
   (software / 3D / manufacturing) — reachable but not executable here.
2. Assess the **two same-name same-niche GitHub projects** (§2) for
   confusion/priority risk.
3. Assess **descriptiveness / registrability** (§5).
4. Proper **WHOIS** on `meshprep.{com,io,dev,…}` (DNS was unreliable here).
5. Only after that: a go/no-go on the name. **Until then the name is NOT cleared.**
