# Third-Party Notices

`meshprep` is MIT-licensed and depends only on **permissive** third-party components.
Each is the property of its respective authors under the license shown. This file is the
attribution record; it is not required by MIT but is provided for transparency and to make
the permissive-only guarantee auditable (see `license_guard.py`).

## Runtime dependencies (core)

| Component | License | Use in meshprep |
|---|---|---|
| NumPy | BSD-3-Clause | all array math |
| SciPy | BSD-3-Clause | sparse linear algebra (FEM solves), ndimage, cKDTree, csgraph |
| trimesh | MIT | mesh I/O, geometry, voxelization, sampling |
| scikit-image | BSD-3-Clause | voxel ops + medial `skeletonize` |
| scikit-fem | BSD-3-Clause | the reduced-order hex/elasticity FEM |
| pyamg | MIT | algebraic-multigrid preconditioner for large FEM systems |
| manifold3d | Apache-2.0 | watertight booleans, genus/Euler |
| networkx | BSD-3-Clause | skeleton graph (tree extraction) |
| pygltflib | MIT | glTF-2.0 skin export |
| matplotlib | Matplotlib License (BSD-style, PSF-based) | heatmap / preview renders |

## Optional dependencies

| Component | License | Status |
|---|---|---|
| pyfqmr | MIT | `[fast]` extra — faster quadric decimation |
| **pyQuadriFlow** | **see caveat** | `[retopo-pyqf]` extra — **NOT enabled by default** |

### pyQuadriFlow caveat (the one license landmine — read before enabling)
The **prebuilt `pyQuadriFlow` wheel** ships a binary that **statically links LGPL Eigen
(SimplicialCholesky)** and carries **no valid license file** (a 3-byte placeholder). Do **not**
redistribute or depend on that wheel from an MIT package. If you want QuadriFlow-grade quad
retopology, **rebuild it yourself from `hjwdzh/QuadriFlow` source with
`-DBUILD_FREE_LICENSE=ON`** (which swaps in MPL-2 `SparseLU`), verify
`strings build/quadriflow* | grep -q SimplicialCholesky` is empty, and vendor the real
MIT + MPL-2 + Boost + Apache(pcg32) license texts. The **default** retopology path
(`mcf2` / `blossom`, pure NumPy/SciPy/networkx) needs none of this and is fully permissive.

## Explicitly AVOIDED (copyleft / non-commercial) — and why

These are common in this domain but are **deliberately not used or depended on**; the bundled
`license_guard.py` fails the build if any are imported:

| Component | License | Permissive substitute used instead |
|---|---|---|
| pymeshfix / MeshFix | GPL-3 | our source-accurate `hole_fill` (curvature dome) |
| PyMeshLab | GPL-3 | trimesh + manifold3d + scikit-image |
| libigl `igl.copyleft.*`, TetGen, fTetWild/pytetwild (CGAL) | GPL / CGAL | surface-only cotangent + bone-heat skinning (no tet mesher) |
| `gpytoolbox.copyleft` | GPL | (MIT `gpytoolbox` core only, if used at all) |
| skeletor | GPL-3 | scikit-image `skeletonize` (medial axis) |
| Pinocchio (auto-rig) | LGPL | self-contained bone-heat reimplementation (algorithm, not code) |
| RigNet / UniRig | GPL-3 / research-only ML | geometric medial rig (no ML) |
| MeshLib | proprietary (free non-commercial) | trimesh + scipy + scikit-image |

> The point of the suite is precisely that none of the above is reachable: a competent user
> could not accidentally pull copyleft or non-commercial code into a derived work.
