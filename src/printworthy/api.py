"""FastAPI farm-endpoint STUB — a documented interface, NOT a served product.

STATUS: STUB / NOT HARDENED. The old product's REST surface (`ei_api`) was
excluded from this release precisely because its operational blockers were
never addressed (see RELEASE_MANIFEST.md, "OUT" section): no defense against
malicious/zip-bomb meshes, no SSRF protections, no rate limiting, no
auth/billing, no upload size ceilings beyond the crude one below. This stub
exists so a print-farm integrator can see the intended contract and wire it
into THEIR hardened gateway. Do not expose it to the internet as-is.

FastAPI is intentionally NOT a dependency of printworthy (default install stays
library+CLI). Install it yourself (``pip install fastapi uvicorn``) to use
this stub; :func:`create_app` raises a clear error otherwise.

Contract:
    POST /check   multipart file upload -> the pipeline's plain-English
                  report as JSON (check/analysis only; the full prep artifact
                  package is a local/CLI concern, not an upload-response).
    GET  /health  liveness + version.

Usage:
    uvicorn "printworthy.api:create_app" --factory --host 127.0.0.1
"""
# NOTE: no `from __future__ import annotations` here — PEP 563 stringizes the
# endpoint's `UploadFile` annotation, which pydantic cannot resolve for a
# function defined inside the create_app() factory (local scope).
import os
import tempfile

# Crude request ceiling (bytes) — a stub-level guard, not real hardening.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


def create_app():
    """App factory. Raises RuntimeError with install guidance if FastAPI is
    absent (import-guarded: `import printworthy.api` never needs fastapi)."""
    try:
        from fastapi import FastAPI, File, UploadFile
        from fastapi.responses import JSONResponse
    except ImportError as e:
        raise RuntimeError(
            "printworthy.api is an optional stub and FastAPI is not installed. "
            "Run `pip install fastapi uvicorn` (it is deliberately not part "
            "of the default printworthy install)."
        ) from e

    from printworthy import __version__

    app = FastAPI(
        title="printworthy farm endpoint (STUB — NOT hardened)",
        version=__version__,
        description=__doc__,
    )

    @app.get("/health")
    def health():
        return {"ok": True, "version": __version__,
                "status": "stub — not hardened, see printworthy/api.py docstring"}

    @app.post("/check")
    async def check(file: UploadFile = File(...)):
        """Run the pipeline's check/analysis on an uploaded mesh; return the
        report JSON. Never raises — failures come back as {"ok": false, ...}."""
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            return JSONResponse(status_code=413, content={
                "ok": False,
                "error": f"upload exceeds {MAX_UPLOAD_BYTES // 2**20} MiB stub ceiling"})
        suffix = os.path.splitext(file.filename or "mesh.stl")[1] or ".stl"
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
                fh.write(data)
                tmp = fh.name
            try:
                from printworthy.pipeline import prep  # lazy: product layer
            except Exception as e:
                return JSONResponse(status_code=503, content={
                    "ok": False,
                    "error": f"pipeline layer unavailable ({type(e).__name__}: {e})"})
            res = prep(tmp, check_only=True)
            # Return the report JSON (the pipeline's plain-English report),
            # falling back to the whole PrepResult if the shape differs.
            if isinstance(res, dict):
                payload = res.get("report_json") or res.get("report") or res
            else:
                payload = {"result": str(res)}
            import json as _json
            return JSONResponse(content=_json.loads(
                _json.dumps({"ok": True, "filename": file.filename,
                             "report": payload}, default=str)))
        except Exception as e:
            return JSONResponse(status_code=500, content={
                "ok": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    return app
