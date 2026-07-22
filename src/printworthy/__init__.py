"""printworthy — drop a mesh, get a print-ready package + a plain-English report.

Product layer (pipeline/report/profiles/cli/app) sits on top of the vendored,
validated engines in `printworthy.core`.
"""
import os as _os

# Single-thread the numerics at package import (16 GB shared box, OOM-sensitive;
# every heavy path — FEM, voxel, field solves — inherits this).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

__version__ = "0.1.0a1"


def prep(*args, **kwargs):
    """One-call pipeline: load -> guard -> analyze -> fix -> orient ->
    strengthen -> package. Returns a PrepResult dict; never raises.
    See :func:`printworthy.pipeline.prep` for the full signature (lazy import so
    `import printworthy` stays instant)."""
    from .pipeline import prep as _prep
    return _prep(*args, **kwargs)
