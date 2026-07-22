"""Folder mode: run the one-call pipeline over every mesh in a directory.

    batch_prep(folder_in, folder_out, **prep_kwargs)
        -> {"ok", "n_total", "n_ok", "results": [...], "summary_csv", "note"}

Design constraints (deliberate):
  * SEQUENTIAL — one mesh at a time (16 GB shared box; the pipeline itself is
    already compute-capped). No pools, no threads.
  * NEVER-RAISE per file — a bad mesh yields an error row, not a crash.
  * `printworthy.pipeline.prep` is imported LAZILY inside the function, so this
    module imports clean standalone and degrades with a clear note if the
    pipeline layer is not built/installed yet.

Outputs per run: ``summary.csv`` in `folder_out` (one row per input, scalar
fields of each PrepResult flattened into columns) plus, per file, the full
PrepResult as ``<stem>.result.json`` (best-effort, `default=str`).
"""
from __future__ import annotations

import csv
import json
import os
import time

MESH_EXTS = (".stl", ".obj", ".ply", ".off", ".3mf", ".glb", ".gltf")


def _scalars(d, prefix="", out=None):
    """Flatten top-level (and one level nested) scalar fields of a PrepResult
    dict for the CSV. Schema-agnostic on purpose — the PrepResult shape may
    evolve; anything non-scalar is skipped."""
    out = {} if out is None else out
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v
        elif isinstance(v, dict) and not prefix:      # one nested level only
            _scalars(v, prefix=f"{k}.", out=out)
    return out


def batch_prep(folder_in, folder_out, **prep_kwargs):
    """Run ``printworthy.prep`` on every mesh file directly inside `folder_in`
    (non-recursive, sorted, sequential); write per-file JSON results and a
    ``summary.csv`` to `folder_out`. Never raises."""
    try:
        try:
            from printworthy.pipeline import prep  # lazy: may not exist yet
        except Exception as e:
            return {"ok": False, "n_total": 0, "n_ok": 0, "results": [],
                    "summary_csv": None,
                    "note": ("printworthy.pipeline is not available "
                             f"({type(e).__name__}: {e}) - batch mode needs the "
                             "pipeline layer; nothing was processed")}

        folder_in, folder_out = str(folder_in), str(folder_out)
        if not os.path.isdir(folder_in):
            return {"ok": False, "n_total": 0, "n_ok": 0, "results": [],
                    "summary_csv": None,
                    "note": f"input folder not found: {folder_in}"}
        os.makedirs(folder_out, exist_ok=True)

        files = sorted(f for f in os.listdir(folder_in)
                       if f.lower().endswith(MESH_EXTS))
        results = []
        for name in files:                       # SEQUENTIAL, one at a time
            path = os.path.join(folder_in, name)
            stem = os.path.splitext(name)[0]
            row = {"file": name, "ok": False, "seconds": None, "error": None}
            t0 = time.perf_counter()
            try:
                res = prep(path, out_dir=os.path.join(folder_out, stem),
                           **prep_kwargs)
                row["seconds"] = round(time.perf_counter() - t0, 2)
                if isinstance(res, dict):
                    row["ok"] = bool(res.get("ok", True))
                    row.update({k: v for k, v in _scalars(res).items()
                                if k not in row})
                    try:                          # full result, best-effort
                        with open(os.path.join(folder_out, f"{stem}.result.json"),
                                  "w", encoding="utf-8") as fh:
                            json.dump(res, fh, indent=2, default=str)
                    except Exception:
                        pass
                else:
                    row["ok"] = res is not None
            except Exception as e:               # never-raise per file
                row["seconds"] = round(time.perf_counter() - t0, 2)
                row["error"] = f"{type(e).__name__}: {e}"
            results.append(row)

        summary_csv = os.path.join(folder_out, "summary.csv")
        try:
            cols = ["file", "ok", "seconds", "error"]
            cols += sorted({k for r in results for k in r} - set(cols))
            with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(results)
        except Exception as e:
            summary_csv = None
            for r in results:
                r.setdefault("error", f"summary.csv write failed: {e}")

        n_ok = sum(1 for r in results if r["ok"])
        return {"ok": n_ok == len(results) and bool(results),
                "n_total": len(results), "n_ok": n_ok, "results": results,
                "summary_csv": summary_csv,
                "note": (f"{n_ok}/{len(results)} meshes prepped from {folder_in}"
                         if results else f"no mesh files ({', '.join(MESH_EXTS)}) "
                                         f"in {folder_in}")}
    except Exception as e:                        # never-raise, period
        return {"ok": False, "n_total": 0, "n_ok": 0, "results": [],
                "summary_csv": None, "note": f"batch failed: {e}"}
