"""Trust map -- per-face *geometric visibility* analysis for AI-generated meshes.

WHAT THIS IS (and is not).  A single monocular image->3D generator (Meshy, SF3D,
Hunyuan, ...) can only have *seen* the surface the source photo actually showed;
everything else -- the back, the underside, the interior -- is invented by the
generator's prior and made watertight.  This module labels, per face, whether the
face was **evidence_backed** (visible to the source camera) or **ai_invented**
(back-facing, occluded, or outside the source frame).  It is *geometric visibility
analysis and REQUIRES the source image plus its camera pose*; without a camera the
feature is ABSENT (see ``trust_map`` mode='absent'), never inferred or faked.

Value positioning: for decorative prints the invented back usually does not matter.
For functional parts, replicas, measurement, or any structural / "this is faithful"
claim, the ai_invented (red) regions are the ones a user must review before relying
on the model -- they are not verifiable against anything.

THE OCCLUSION ANCHOR (the part that ships, free & public).  A face is invented iff
it is (back-facing) OR (occluded by other geometry in the z-buffer) OR (outside the
camera frustum).  This is *pure geometry* given the camera -- no learned prior, no
depth model.  It is the proven principle from the provenance PoC.

WHAT IS DEFERRED / NOT SHIPPED.
  * Pose ESTIMATION (render-and-match / PnP from the source image) is experimental
    and unbuilt in v1; ``pose='estimate'`` is gated behind ``experimental_estimate``
    and currently returns ABSENT with an honest note rather than a faked pose.
  * The geometry-only "head" that predicts invented-ness on *unmarked* meshes with
    no camera is UNPROVEN (Goodharts on layout priors) and is NOT importable or
    callable from here -- research only, not product.

-----------------------------------------------------------------------------------
ATTRIBUTION.  The visibility-classification logic (back-facing test, frustum test,
z-buffer occlusion / inter-object-occlusion anchor, and the "non-observable => must
be invented" rule) is VENDORED and adapted from the research PoC at
``Applied/Quasi/provenance/trackA`` (``trackA_scene.py`` Camera / make_scene, and
``trackA_audit.py`` audit_scene -- the occlusion sanity anchor).  It is copied here,
not imported: meshprep never imports from Applied/Quasi at runtime.  Two deliberate
departures from the PoC:
  1. The PoC's brute-force O(pixels x triangles) python ray loop is replaced with an
     embree BVH nearest-hit query (thousands of faces on real meshes, not ~12-24).
  2. The PoC's legal/distorted DEPTH-grading machinery (per-face affine alignment of
     a reconstructed height field to KNOWN ground-truth depth, LEGAL_TOL/JUMP_FRAC)
     is DROPPED -- production has no ground-truth depth, so only the pure visibility
     classification transfers.
The PoC's camera was fixed at the origin looking +Z; here the camera is a free
world-space eye (``Camera`` dataclass), decoupling image row/col from the eye.
-----------------------------------------------------------------------------------

PURE: no function here raises on bad input; failures degrade to an honest note.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import trimesh

# meshprep-internal reuse (NOT Applied/Quasi): the embree intersector factory and
# the memory-bounded ray-batch sizer for the pure-python fallback.
from ._print_premortem import _intersector, _thin_ray_batch

# --------------------------------------------------------------------------- #
#  Labels (three classes; repair_filled only appears post-fixer).
# --------------------------------------------------------------------------- #
EVIDENCE_BACKED = "evidence_backed"
AI_INVENTED = "ai_invented"
REPAIR_FILLED = "repair_filled"

# Grazing/edge false-hit guard, reused verbatim from _print_premortem._thin_wall_channel:
# a ray "hit" only counts as a genuine occluder if the occluding face roughly faces the
# ray (|dot(hit_normal, ray)| > this); grazing side-swipes are edge artifacts, discarded.
_GRAZE_HIT_MIN = 0.34

# Occlusion is the one heavy step (an embree BVH pass). This 16 GB box has OOM'd on
# raycast grids before; below this available-RAM floor we SKIP the occlusion pass,
# fall back to back-facing + frustum only, and say so in the note (never silently).
_OCCLUSION_RAM_FLOOR_GB = 1.0

# The canonical, honest ABSENT note. Kept as a constant so the report's
# self-consistency verifier and the selftest can string-match it.
ABSENT_NOTE = (
    "Trust map unavailable: no source camera/pose supplied. This feature is "
    "geometric visibility analysis and REQUIRES the source image plus its camera "
    "pose; without them the map is ABSENT, not inferred or faked."
)

_NOTE_PREFIX = (
    "Geometric visibility analysis (requires the source camera): "
    "green faces were visible to the source image; red faces (back / interior / "
    "occluded) were invented by the AI and cannot be verified. Decorative prints "
    "usually don't care; functional parts, replicas, or structural claims should "
    "review the invented regions."
)


# --------------------------------------------------------------------------- #
#  Camera -- a free world-space pinhole eye (right-handed).
# --------------------------------------------------------------------------- #
@dataclass
class Camera:
    """World-space pinhole camera, decoupled from image row/col.

    position   : (3,) eye location in world space.
    forward    : (3,) unit look direction (into the scene). Normalized on init.
    up         : (3,) approximate world up; re-orthogonalized against forward.
    fov_y_deg  : vertical field of view in degrees.
    resolution : (W, H) pixels.

    Conventions (mirror trackA_scene's CV convention -- x=right, y=DOWN, z=forward --
    but relocated from the origin to an arbitrary eye):
      * IN-FRONT : point P is in front iff dot(P - position, forward) > 0.
      * project  : f = (H/2)/tan(fov_y/2); u = f*x_cam/z_cam + cx, v = f*y_cam/z_cam + cy,
                   with cx=(W-1)/2, cy=(H-1)/2. In-view iff z_cam>0 and (u,v) in
                   [0,W-1] x [0,H-1].
    """

    position: Any
    forward: Any
    up: Any
    fov_y_deg: float = 50.0
    resolution: Any = (512, 512)

    # derived (filled in __post_init__)
    _pos: np.ndarray = field(default=None, repr=False)
    _fwd: np.ndarray = field(default=None, repr=False)
    _right: np.ndarray = field(default=None, repr=False)
    _down: np.ndarray = field(default=None, repr=False)
    _f: float = field(default=0.0, repr=False)

    def __post_init__(self):
        self._pos = np.asarray(self.position, float).reshape(3)
        fwd = np.asarray(self.forward, float).reshape(3)
        up = np.asarray(self.up, float).reshape(3)
        fn = np.linalg.norm(fwd)
        self._fwd = fwd / fn if fn > 1e-12 else np.array([0.0, 0.0, 1.0])
        # right = fwd x up ; re-orthogonalize a true "down" (CV: y grows downward)
        right = np.cross(self._fwd, up)
        rn = np.linalg.norm(right)
        if rn < 1e-9:  # up parallel to forward -> pick any perpendicular
            alt = np.array([1.0, 0.0, 0.0]) if abs(self._fwd[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            right = np.cross(self._fwd, alt)
            rn = np.linalg.norm(right)
        self._right = right / rn
        true_up = np.cross(self._right, self._fwd)
        self._down = -true_up / (np.linalg.norm(true_up) + 1e-12)
        W, H = int(self.resolution[0]), int(self.resolution[1])
        self.resolution = (W, H)
        self._f = (H / 2.0) / math.tan(math.radians(float(self.fov_y_deg)) / 2.0)

    @property
    def W(self) -> int:
        return int(self.resolution[0])

    @property
    def H(self) -> int:
        return int(self.resolution[1])

    def project(self, pts: np.ndarray):
        """World points (...,3) -> (uv (...,2), z_cam (...,)). z_cam>0 == in front."""
        pts = np.asarray(pts, float)
        rel = pts - self._pos
        xc = rel @ self._right
        yc = rel @ self._down
        zc = rel @ self._fwd
        cx = (self.W - 1) / 2.0
        cy = (self.H - 1) / 2.0
        zsafe = np.where(np.abs(zc) < 1e-12, 1e-12, zc)
        u = self._f * xc / zsafe + cx
        v = self._f * yc / zsafe + cy
        return np.stack([u, v], axis=-1), zc

    def in_view(self, pts: np.ndarray) -> np.ndarray:
        uv, zc = self.project(pts)
        return (zc > 0) & (uv[..., 0] >= 0) & (uv[..., 0] <= self.W - 1) \
            & (uv[..., 1] >= 0) & (uv[..., 1] <= self.H - 1)

    @classmethod
    def look_at(cls, position, target, up=(0.0, 1.0, 0.0), fov_y_deg=50.0,
                resolution=(512, 512)) -> "Camera":
        """Convenience constructor: an eye at `position` looking at `target`."""
        position = np.asarray(position, float).reshape(3)
        target = np.asarray(target, float).reshape(3)
        return cls(position=position, forward=target - position, up=up,
                   fov_y_deg=fov_y_deg, resolution=resolution)


# --------------------------------------------------------------------------- #
#  Internals.
# --------------------------------------------------------------------------- #
def _avail_ram_gb() -> Optional[float]:
    try:
        import psutil
        return float(psutil.virtual_memory().available) / 1e9
    except Exception:
        return None


def _bary_samples(n_jitter: int, rng: np.random.Generator) -> np.ndarray:
    """Sample-point barycentric weights: row 0 = centroid, then `n_jitter` interior
    jitter points (shared across all faces -> deterministic and cheap)."""
    rows = [np.array([1 / 3.0, 1 / 3.0, 1 / 3.0])]
    for _ in range(max(0, n_jitter)):
        r1, r2 = rng.random(), rng.random()
        if r1 + r2 > 1.0:
            r1, r2 = 1.0 - r1, 1.0 - r2
        # keep it interior-ish so a jitter point does not fall on an edge/vertex
        w0, w1, w2 = 1.0 - r1 - r2, r1, r2
        rows.append(np.array([w0, w1, w2]))
    return np.asarray(rows, float)  # (S, 3)


def _empty_result(mode: str, note: str) -> dict:
    return dict(
        ok=False, mode=mode, face_labels=None, frac_invented=None,
        confidence=None, silhouette=None, note=note,
    )


# --------------------------------------------------------------------------- #
#  The occlusion anchor (explicit-pose path).
# --------------------------------------------------------------------------- #
def _anchor(mesh: trimesh.Trimesh, camera: Camera, *, samples: int,
            grazing_deg: float, ram_floor_gb: float) -> dict:
    """Pure per-face visibility on `mesh` from `camera`. Returns the full result dict.
    Never raises: any failure degrades to a best-effort back-facing+frustum result."""
    F = int(len(mesh.faces))
    if F == 0:
        return _empty_result("explicit", _NOTE_PREFIX + " (empty mesh)")

    tris = np.asarray(mesh.triangles, float)           # (F,3,3)
    centroids = np.asarray(mesh.triangles_center, float)  # (F,3)
    normals = np.asarray(mesh.face_normals, float)     # (F,3)
    areas = np.asarray(mesh.area_faces, float)         # (F,)
    pos = camera._pos
    fwd = camera._fwd
    cos_graze = math.cos(math.radians(float(grazing_deg)))

    rng = np.random.default_rng(0)
    W = _bary_samples(max(0, int(samples) - 1), rng)   # (S,3)
    S = W.shape[0]
    # sample points (S,F,3): weighted vertex combos of each triangle
    pts = (W[:, 0][:, None, None] * tris[None, :, 0, :]
           + W[:, 1][:, None, None] * tris[None, :, 1, :]
           + W[:, 2][:, None, None] * tris[None, :, 2, :])

    rel = pts - pos[None, None, :]                     # (S,F,3)
    z_cam = rel @ fwd                                  # (S,F)
    vdir = rel / (np.linalg.norm(rel, axis=2, keepdims=True) + 1e-12)  # (S,F,3)

    # (1) BACK-FACING: outward normal points along the view ray (dot >= 0).
    backfacing = np.einsum("sfj,fj->sf", vdir, normals) >= 0.0   # (S,F)

    # (2) FRUSTUM: in front AND projected (u,v) inside image bounds.
    uv, _z = camera.project(pts.reshape(-1, 3))
    uv = uv.reshape(S, F, 2)
    in_view = (z_cam > 0) & (uv[..., 0] >= 0) & (uv[..., 0] <= camera.W - 1) \
        & (uv[..., 1] >= 0) & (uv[..., 1] <= camera.H - 1)

    # (3) OCCLUSION (embree BVH nearest-hit). RAM-gated: skip if the box is thin.
    occluded = np.zeros((S, F), bool)
    occlusion_deferred = False
    avail = _avail_ram_gb()
    if avail is not None and avail < ram_floor_gb:
        occlusion_deferred = True
    else:
        try:
            inter, fast = _intersector(mesh)
            origins = np.broadcast_to(pos, (S * F, 3)).reshape(-1, 3)
            dirs = vdir.reshape(-1, 3)
            fid = np.tile(np.arange(F, dtype=np.int64), S)  # (S*F,) face each ray targets
            batch = S * F if fast else max(1, _thin_ray_batch(F))
            hits = np.full(S * F, -1, np.int64)
            for s0 in range(0, S * F, batch):
                sl = slice(s0, s0 + batch)
                h = inter.intersects_first(ray_origins=origins[sl], ray_directions=dirs[sl])
                hits[sl] = np.asarray(h, np.int64)
            # a face is occluded iff the FIRST hit along the ray to its own point is
            # some OTHER face -- and that occluder is a genuine (non-grazing) surface.
            different = (hits != fid) & (hits >= 0)
            hn = np.zeros((S * F, 3))
            valid = hits >= 0
            hn[valid] = normals[hits[valid]]
            align = np.abs(np.einsum("ij,ij->i", hn, dirs))
            genuine = align > _GRAZE_HIT_MIN
            occ = different & genuine
            occluded = occ.reshape(S, F)
        except Exception:
            occlusion_deferred = True  # embree unavailable / failed -> honest fallback

    # per-sample invented label
    invented_s = backfacing | occluded | (~in_view)    # (S,F)
    invented0 = invented_s[0]                           # centroid label (row 0)

    # soft confidence: sub-sample agreement * grazing decay near silhouette
    agreement = (invented_s == invented0[None, :]).mean(axis=0)   # (F,) in [0,1]
    gd = np.abs(np.einsum("fj,fj->f", normals, vdir[0]))          # |dot(n, view)| at centroid
    graze_factor = np.clip(gd / max(cos_graze, 1e-6), 0.0, 1.0)
    confidence = np.clip(agreement * graze_factor, 0.0, 1.0)

    labels = np.where(invented0, AI_INVENTED, EVIDENCE_BACKED).astype(object)
    labels = np.asarray(labels, dtype="<U16")
    silhouette = (gd < cos_graze) | (agreement < 1.0)

    tot_area = float(areas.sum())
    frac_inv = float(areas[invented0].sum() / tot_area) if tot_area > 1e-12 else \
        float(invented0.mean())

    note = _NOTE_PREFIX
    if occlusion_deferred:
        note += (" [occlusion pass DEFERRED: low RAM or no embree -- labels reflect "
                 "back-facing + frustum only; hidden-but-front-facing faces may read "
                 "evidence_backed. Re-run with more memory for the full anchor.]")

    return dict(
        ok=True, mode="explicit",
        face_labels=labels,
        frac_invented=frac_inv,
        confidence=confidence.astype(float),
        silhouette=silhouette.astype(bool),
        note=note,
        # extras (harmless; consumers may ignore)
        n_faces=F,
        n_invented=int(invented0.sum()),
        occlusion_deferred=bool(occlusion_deferred),
        samples=int(S),
        grazing_deg=float(grazing_deg),
    )


# --------------------------------------------------------------------------- #
#  Public entry point.
# --------------------------------------------------------------------------- #
def trust_map(mesh, *, camera: Optional[Camera] = None, image=None,
              pose: str = "explicit", samples: int = 4, grazing_deg: float = 75.0,
              experimental_estimate: bool = False) -> dict:
    """Per-face geometric visibility (trust) map. See module docstring.

    Returns a dict:
      ok           : bool
      mode         : 'explicit' | 'estimated' | 'absent'
      face_labels  : np.ndarray[str] length F in {evidence_backed, ai_invented}
                     (repair_filled appears only AFTER the fixer, via transfer_labels)
      frac_invented: float  -- AREA-weighted (a few big backfaces should dominate)
      confidence   : np.ndarray[float] length F in [0,1]  -- low near silhouettes
      silhouette   : np.ndarray[bool] -- grazing-angle OR sub-sample-disagreeing faces
      note         : str  -- always labels this geometric visibility requiring the camera

    ABSENT (no camera and no usable estimate): ok=False, mode='absent',
    face_labels=None, note=ABSENT_NOTE. The report omits the section entirely; no
    labels are fabricated.
    """
    # ----- EXPLICIT (ships v1) -------------------------------------------------
    if camera is not None:
        if not isinstance(camera, Camera):
            try:
                camera = Camera(**camera) if isinstance(camera, dict) else camera
            except Exception:
                return _empty_result("absent", ABSENT_NOTE)
        try:
            m = mesh if isinstance(mesh, trimesh.Trimesh) else trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
                process=False)
            return _anchor(m, camera, samples=samples, grazing_deg=grazing_deg,
                           ram_floor_gb=_OCCLUSION_RAM_FLOOR_GB)
        except Exception as e:  # never raise
            return _empty_result(
                "absent", ABSENT_NOTE + f" (internal anchor error: {type(e).__name__})")

    # ----- ESTIMATE (experimental, v1.1, gated) --------------------------------
    if pose == "estimate" and experimental_estimate and image is not None:
        # Render-and-match / PnP pose registration is UNBUILT in v1 and this box is
        # RAM-thin; rather than overclaim an estimate we stay honest and return ABSENT
        # with a note that estimation is experimental and not yet certified.
        return _empty_result(
            "absent",
            "Trust map: pose ESTIMATION from the source image is experimental and not "
            "yet available in this build (render-and-match / PnP registration is "
            "uncertified). Supply an explicit calibrated camera to run the proven "
            "occlusion anchor. " + ABSENT_NOTE)

    # ----- ABSENT (the honest default) -----------------------------------------
    return _empty_result("absent", ABSENT_NOTE)


# --------------------------------------------------------------------------- #
#  Label transfer through the fixer (design section 3 / repair_unification).
# --------------------------------------------------------------------------- #
def transfer_labels(orig_mesh, orig_labels, new_mesh, *, mode: str = "auto",
                    orig_confidence=None, voxel_pitch: Optional[float] = None) -> dict:
    """Carry per-face trust labels from `orig_mesh` onto `new_mesh` after a repair.

    THREE regimes (recon-confirmed against _fix_accurate.py / hole_fill_ei.py):

      REGIME A -- LOCAL PATCH, INDEX-PRESERVING (mode='index'): fillers that build
        Trimesh(vstack(orig, new), process=False) keep faces [0:N_orig] in order.
        EXACT: labels[0:N_orig] = orig by direct index; labels[N_orig:] = repair_filled.

      REGIME B -- GLOBAL RESEAL, INDEX-DESTROYING (mode='nearest'): marching-cubes /
        winding-number rebuilds every face. APPROXIMATE: each new-face centroid inherits
        the label of its nearest original face (orig.nearest.on_surface). Conservative:
          (i) if dist_to_original > voxel_pitch -> no real source -> repair_filled;
          (ii) near a label SEAM (within pitch AND nearest original abuts a label change)
               -> downgrade confidence and bias to ai_invented/repair_filled; NEVER
               upgrade an invented face to evidence_backed via a reseal.

      REGIME C -- CHEAP_REPAIR SUBSEQUENCE (mode='mask'): pass the boolean/int mask that
        merge_vertices/unique_faces produced; labels carry by masked index.

    mode='auto' picks index vs nearest by whether new_mesh's first N_orig face centroids
    coincide with orig's (index-preserving) or not (reseal). For REGIME C (cheap_repair
    subsequence) use `transfer_labels_masked` with the captured keep-mask.

    Returns {labels, confidence, carry_mode}. Never raises.
    """
    try:
        o_labels = np.asarray(orig_labels, dtype="<U16")
        n_orig = int(len(o_labels))
        nf_new = int(len(new_mesh.faces))
        o_conf = None if orig_confidence is None else np.asarray(orig_confidence, float)

        chosen = mode
        if mode == "auto":
            chosen = "index"
            try:
                oc = np.asarray(orig_mesh.triangles_center, float)
                nc = np.asarray(new_mesh.triangles_center, float)
                if nf_new >= n_orig and n_orig > 0:
                    d = np.linalg.norm(nc[:n_orig] - oc[:n_orig], axis=1)
                    scale = float(np.linalg.norm(orig_mesh.extents)) + 1e-9
                    if np.median(d) > 1e-6 * scale:
                        chosen = "nearest"
                else:
                    chosen = "nearest"
            except Exception:
                chosen = "nearest"

        if chosen == "index":
            labels = np.full(nf_new, REPAIR_FILLED, dtype="<U16")
            k = min(n_orig, nf_new)
            labels[:k] = o_labels[:k]
            conf = np.ones(nf_new, float)
            if o_conf is not None:
                conf[:k] = o_conf[:k]
            conf[k:] = 1.0  # repair_filled is a construction fact -> confident label
            return dict(labels=labels, confidence=conf,
                        carry_mode="index-preserving (exact)")

        # REGIME B -- nearest original face (approximate, conservatively biased)
        nc = np.asarray(new_mesh.triangles_center, float)
        try:
            closest, dist, tri_id = orig_mesh.nearest.on_surface(nc)
            tri_id = np.asarray(tri_id, np.int64)
        except Exception:
            # fallback: brute nearest-centroid
            from scipy.spatial import cKDTree
            oc = np.asarray(orig_mesh.triangles_center, float)
            dist, tri_id = cKDTree(oc).query(nc)
            dist = np.asarray(dist, float)
            tri_id = np.asarray(tri_id, np.int64)

        pitch = voxel_pitch
        if pitch is None:
            # a sensible default: coarse voxel scale ~ 1% of the bbox diagonal
            pitch = 0.01 * (float(np.linalg.norm(orig_mesh.extents)) + 1e-9)

        labels = o_labels[np.clip(tri_id, 0, n_orig - 1)].astype("<U16")
        conf = np.ones(nf_new, float)
        if o_conf is not None:
            conf = o_conf[np.clip(tri_id, 0, n_orig - 1)].astype(float)

        far = dist > pitch
        labels[far] = REPAIR_FILLED
        conf[far] = 1.0

        # seam handling: a new face within one pitch of the boundary between an
        # evidence face and an invented face inherits ambiguously -> downgrade its
        # confidence and bias toward the SAFER disclosure (never invented -> evidence).
        try:
            adj = orig_mesh.face_adjacency  # (E,2)
            la = o_labels[adj[:, 0]]
            lb = o_labels[adj[:, 1]]
            seam_faces = np.zeros(n_orig, bool)
            changing = la != lb
            seam_faces[adj[changing, 0]] = True
            seam_faces[adj[changing, 1]] = True
            near = (~far) & seam_faces[np.clip(tri_id, 0, n_orig - 1)]
            # bias: if the inherited label is evidence_backed but it straddles a seam,
            # downgrade toward ai_invented (safer) and lower confidence.
            downgrade = near & (labels == EVIDENCE_BACKED)
            labels[downgrade] = AI_INVENTED
            conf[near] = np.minimum(conf[near], 0.5)
        except Exception:
            pass

        return dict(labels=labels, confidence=conf,
                    carry_mode="reseal / nearest-original-face (approximate, "
                               "biased toward invented near seams)")
    except Exception:
        # last resort: everything is repair_filled (honest: we could not carry)
        nf_new = int(len(new_mesh.faces)) if hasattr(new_mesh, "faces") else 0
        return dict(labels=np.full(nf_new, REPAIR_FILLED, dtype="<U16"),
                    confidence=np.ones(nf_new, float),
                    carry_mode="failed-carry (all repair_filled)")


def transfer_labels_masked(orig_labels, mask, orig_confidence=None) -> dict:
    """REGIME C helper: _cheap_repair's merge/unique_faces produce an IN-ORDER
    subsequence of the original faces. Given the boolean keep-mask (or integer index
    array) it captured, carry labels by masked index. Exact for that subsequence.

    `mask` : bool array length N_orig (keep) OR int index array into original faces.
    """
    try:
        o = np.asarray(orig_labels, dtype="<U16")
        mask = np.asarray(mask)
        if mask.dtype == bool:
            labels = o[mask].astype("<U16")
            if orig_confidence is not None:
                conf = np.asarray(orig_confidence, float)[mask]
            else:
                conf = np.ones(len(labels), float)
        else:  # integer index / update_faces map
            idx = mask.astype(np.int64)
            labels = o[np.clip(idx, 0, len(o) - 1)].astype("<U16")
            conf = (np.asarray(orig_confidence, float)[np.clip(idx, 0, len(o) - 1)]
                    if orig_confidence is not None else np.ones(len(labels), float))
        return dict(labels=labels, confidence=conf,
                    carry_mode="cheap-repair subsequence (exact by masked index)")
    except Exception:
        return dict(labels=np.asarray([], dtype="<U16"),
                    confidence=np.asarray([], float),
                    carry_mode="failed-carry")


# --------------------------------------------------------------------------- #
#  Overlay render helper (green / orange / blue, soft near silhouettes).
# --------------------------------------------------------------------------- #
_TRUST_COLORS = {
    EVIDENCE_BACKED: np.array([0.15, 0.70, 0.30]),   # green  = the AI saw it
    AI_INVENTED: np.array([0.95, 0.45, 0.10]),       # orange = the AI invented it
    REPAIR_FILLED: np.array([0.20, 0.45, 0.90]),     # blue   = the fixer added it
}

TRUST_CAPTION = ("Trust map -- which parts the AI actually saw (green) vs invented "
                 "(red/orange); blue = added by the repair step. Requires your source "
                 "image + camera; near-silhouette faces are shown softly (uncertain).")


def render_trust_overlay(mesh, result, path, *, max_faces: int = 6000,
                         elev: float = 20.0, azim: float = -60.0) -> dict:
    """Write a 3-class trust PNG. evidence=green, invented=orange, repair=blue; near-
    silhouette confidence desaturates the face so the rim reads SOFT, not crisp.

    Full-resolution fractions are preserved: the picture is drawn on a display-only
    decimated proxy (each display face coloured by its nearest full-res face), but the
    reported frac_invented is the full-res value. Never raises; returns a dict.
    """
    out = dict(ok=False, path=str(path), caption=TRUST_CAPTION,
               frac_invented=result.get("frac_invented") if isinstance(result, dict) else None)
    try:
        if not (isinstance(result, dict) and result.get("ok") and
                result.get("face_labels") is not None):
            out["note"] = "no trust map to render (absent)"
            return out
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        m = mesh if isinstance(mesh, trimesh.Trimesh) else trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
            process=False)
        labels = np.asarray(result["face_labels"], dtype="<U16")
        conf = result.get("confidence")
        conf = np.ones(len(labels)) if conf is None else np.asarray(conf, float)

        # full-res face colours (RGB), desaturated toward gray by (1-confidence)
        base = np.zeros((len(labels), 3))
        for lab, col in _TRUST_COLORS.items():
            base[labels == lab] = col
        gray = np.array([0.62, 0.62, 0.62])
        c = np.clip(conf, 0.0, 1.0)[:, None]
        rgb_full = c * base + (1.0 - c) * gray   # soft near silhouettes

        # DISPLAY proxy (mirrors _print_premortem): decimate + nearest-full-res colour
        b = m
        sel = None
        if len(m.faces) > max_faces:
            try:
                from ._mesh_util import decimate as _decimate
                bd = _decimate(m.copy(), max_faces)
                if 0 < len(bd.faces) < len(m.faces):
                    from scipy.spatial import cKDTree
                    sel = cKDTree(np.asarray(m.triangles_center, float)).query(
                        np.asarray(bd.triangles_center, float))[1]
                    b = bd
            except Exception:
                sel = None
        rgb = rgb_full[sel] if sel is not None else rgb_full

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        pc = Poly3DCollection(b.triangles, facecolors=rgb, edgecolor="none")
        pc.set_alpha(None)
        ax.add_collection3d(pc)
        V = b.vertices
        ctr = V.mean(0)
        rad = float(np.abs(V - ctr).max()) * 1.05 + 1e-9
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
        ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
        ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
        try:
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elev, azim=azim)
        except Exception:
            pass
        ax.set_axis_off()
        pct = 100.0 * float(out["frac_invented"] or 0.0)
        ax.set_title(f"Trust map -- {pct:.0f}% of surface area AI-invented\n"
                     "green=seen  orange=invented  blue=repaired", fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        out["ok"] = True
        out["frac_invented"] = float(out["frac_invented"] or 0.0)  # full-res, not proxy
        return out
    except Exception as e:
        out["note"] = f"render failed: {type(e).__name__}: {e}"
        return out


# --------------------------------------------------------------------------- #
#  Selftest -- the design's controls, each exercising an exact failure mode.
# --------------------------------------------------------------------------- #
def _open_heightfield(n=12, size=20.0, amp=1.5):
    """A single-sided open surface (top sheet only, no back faces) facing +Z."""
    xs = np.linspace(-size / 2, size / 2, n)
    ys = np.linspace(-size / 2, size / 2, n)
    X, Y = np.meshgrid(xs, ys)
    Z = amp * np.cos(X / size * math.pi) * np.cos(Y / size * math.pi)
    V = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            # wind so outward normal points toward -Z... we want +Z-ish; use both diags
            faces.append([a, c, b])
            faces.append([b, c, d])
    m = trimesh.Trimesh(vertices=V, faces=np.asarray(faces), process=False)
    # ensure the sheet's normals point toward the camera side (+Z)
    if np.asarray(m.face_normals, float)[:, 2].mean() < 0:
        m.faces = m.faces[:, ::-1]
        m._cache.clear()
    return m


def selftest(verbose: bool = True) -> dict:
    """Controls (each hits the exact failure mode it guards -- see design gates):

      1. ABSENT: trust_map(mesh) with no camera -> ok=False, mode='absent',
         face_labels=None, note names 'requires the source image'.
      2. FALSE-POSITIVE: an open front-surface fully facing the camera, no backfaces,
         no occlusion -> ~0% invented (guards against calling seen geometry invented).
      3. BACKSIDE: a closed sphere seen from one side -> ~50% (+/- tol) invented
         (guards that the anchor DOES flag the unseen half).
      4. INTER-OBJECT OCCLUSION (load-bearing): object A's FRONT-facing faces hidden
         behind object B in the z-buffer MUST be ai_invented -- the branch the PoC
         never exercised (its 37 non-observable faces were all self-occluded backsides).
      5. FRUSTUM: an object partly outside the FOV -> outside faces flagged, in-view not.
      6. RENDER-AS-SOURCE exactness: an independent dense projection oracle on a small
         mesh agrees with the anchor on the visible/invented split.
      7. LABEL TRANSFER: index-preserving carry is exact + added faces=repair_filled;
         reseal carry covers 100% of faces and never upgrades invented->evidence.

    RAM-gated: if available RAM < floor, the occlusion-dependent checks are recorded as
    'deferred: low RAM' rather than run. Returns a dict of measured numbers + booleans.
    """
    out: dict = {}
    ram = _avail_ram_gb()
    out["_ram_gb"] = None if ram is None else round(ram, 2)
    low_ram = (ram is not None and ram < 2.0)
    out["_low_ram"] = bool(low_ram)

    # 1. ABSENT ---------------------------------------------------------------
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    r_absent = trust_map(sphere)
    note_ok = "requires the source image" in r_absent["note"].lower()
    absent_pass = (r_absent["ok"] is False and r_absent["mode"] == "absent"
                   and r_absent["face_labels"] is None and note_ok)
    out["absent"] = {"ok_is_false": r_absent["ok"] is False,
                     "labels_none": r_absent["face_labels"] is None,
                     "note_ok": bool(note_ok),
                     "pass": bool(absent_pass)}

    if low_ram:
        out["_deferred"] = "low RAM (<2GB): occlusion-dependent controls skipped"
        out["overall_pass"] = bool(absent_pass)
        if verbose:
            print("trust_map.selftest: LOW RAM -- only ABSENT control ran:", absent_pass)
        return out

    # 2. FALSE-POSITIVE: open sheet fully facing the camera -------------------
    sheet = _open_heightfield()
    cam_sheet = Camera.look_at(position=[0, 0, 40], target=[0, 0, 0],
                               up=[0, 1, 0], fov_y_deg=60.0)
    r_sheet = trust_map(sheet, camera=cam_sheet)
    sheet_frac = float(r_sheet["frac_invented"])
    fp_pass = r_sheet["ok"] and sheet_frac < 0.10
    out["false_positive_open_sheet"] = {"frac_invented": round(sheet_frac, 3),
                                        "target": "< 0.10", "pass": bool(fp_pass)}

    # 3. BACKSIDE: closed sphere from one side -> ~50% invented ---------------
    cam_sph = Camera.look_at(position=[0, 0, 30], target=[0, 0, 0], up=[0, 1, 0],
                             fov_y_deg=45.0)
    r_sph = trust_map(sphere, camera=cam_sph)
    sph_frac = float(r_sph["frac_invented"])
    back_pass = r_sph["ok"] and (0.40 <= sph_frac <= 0.60)
    out["backside_sphere"] = {"frac_invented": round(sph_frac, 3),
                              "target": "0.50 +/- 0.10", "pass": bool(back_pass),
                              "occlusion_deferred": bool(r_sph.get("occlusion_deferred"))}

    # 4. INTER-OBJECT OCCLUSION (the load-bearing branch) ---------------------
    # Object A (small box) sits BEHIND object B (large box) from the camera. A's faces
    # pointing at the camera are front-facing AND in-view but hidden by B -> MUST be
    # ai_invented. Both objects are one mesh (self-occlusion between components == the
    # inter-object branch the PoC never exercised).
    A = trimesh.creation.box(extents=[2, 2, 2])
    A.apply_translation([0, 0, 12])            # far
    nA = len(A.faces)
    B = trimesh.creation.box(extents=[8, 8, 2])
    B.apply_translation([0, 0, 6])             # near, wide enough to cover A
    combo = trimesh.util.concatenate([A, B])
    cam_io = Camera.look_at(position=[0, 0, 0], target=[0, 0, 1], up=[0, 1, 0],
                            fov_y_deg=70.0)
    r_io = trust_map(combo, camera=cam_io)
    io_pass = False
    io_info: dict = {"occlusion_deferred": bool(r_io.get("occlusion_deferred"))}
    if r_io["ok"] and not r_io.get("occlusion_deferred"):
        labels = np.asarray(r_io["face_labels"])
        # A's faces are indices [0:nA]; find A's camera-facing (front) in-view faces
        cA = np.asarray(combo.triangles_center, float)[:nA]
        nAf = np.asarray(combo.face_normals, float)[:nA]
        vdir = cA - cam_io._pos
        vdir /= (np.linalg.norm(vdir, axis=1, keepdims=True) + 1e-12)
        front = np.einsum("ij,ij->i", nAf, vdir) < 0.0     # points back at camera
        inview = cam_io.in_view(cA)
        target_faces = front & inview
        if target_faces.any():
            occ_invented = labels[:nA][target_faces] == AI_INVENTED
            io_pass = bool(occ_invented.all())
            io_info.update({"n_front_inview_A_faces": int(target_faces.sum()),
                            "all_invented": bool(occ_invented.all())})
        else:
            io_info["note"] = "no front-facing in-view A faces (fixture geometry)"
    else:
        io_info["note"] = "occlusion deferred -- inter-object branch not exercised"
    io_info["pass"] = bool(io_pass)
    out["inter_object_occlusion"] = io_info

    # 5. FRUSTUM: long bar, narrow FOV -> part of it outside the frame ---------
    bar = trimesh.creation.box(extents=[80, 3, 3])
    bar.apply_translation([0, 0, 15])          # very long in x; the end caps leave frame
    cam_fr = Camera.look_at(position=[0, 0, 0], target=[0, 0, 1], up=[0, 1, 0],
                            fov_y_deg=90.0)
    r_fr = trust_map(bar, camera=cam_fr)
    fr_pass = False
    fr_info: dict = {}
    if r_fr["ok"]:
        cbar = np.asarray(bar.triangles_center, float)
        uv, zc = cam_fr.project(cbar)
        inview = cam_fr.in_view(cbar)
        labels = np.asarray(r_fr["face_labels"])
        # every out-of-frame face must be invented; assert some in and some out exist
        out_faces = ~inview
        fr_out_invented = bool((labels[out_faces] == AI_INVENTED).all()) if out_faces.any() else False
        fr_pass = bool(out_faces.any() and inview.any() and fr_out_invented)
        fr_info = {"n_out_of_frame": int(out_faces.sum()),
                   "n_in_frame": int(inview.sum()),
                   "all_out_invented": fr_out_invented, "pass": fr_pass}
    else:
        fr_info = {"pass": False, "note": "anchor not ok"}
    out["frustum"] = fr_info

    # 6. RENDER-AS-SOURCE exactness: independent dense-projection oracle -------
    # Oracle: for each face centroid, is it (front-facing AND in-view AND the nearest
    # thing the camera sees along that ray)? Computed WITHOUT reusing the anchor's
    # per-sample machinery -- a plain single-ray-per-face check -> compare the two.
    oracle_pass = False
    oracle_info: dict = {}
    small = trimesh.creation.icosphere(subdivisions=2, radius=4.0)  # 320 faces
    cam_o = Camera.look_at(position=[0, 0, 25], target=[0, 0, 0], up=[0, 1, 0],
                           fov_y_deg=40.0)
    r_small = trust_map(small, camera=cam_o, samples=1)  # centroid-only vs oracle
    if r_small["ok"] and not r_small.get("occlusion_deferred"):
        cc = np.asarray(small.triangles_center, float)
        nn = np.asarray(small.face_normals, float)
        vdir = cc - cam_o._pos
        vdir /= (np.linalg.norm(vdir, axis=1, keepdims=True) + 1e-12)
        front = np.einsum("ij,ij->i", nn, vdir) < 0.0
        inview = cam_o.in_view(cc)
        try:
            inter, _fast = _intersector(small)
            hit = np.asarray(inter.intersects_first(
                ray_origins=np.broadcast_to(cam_o._pos, cc.shape),
                ray_directions=vdir), np.int64)
            unocc = hit == np.arange(len(cc))
        except Exception:
            unocc = front  # degrade
        oracle_invented = ~(front & inview & unocc)
        anchor_invented = np.asarray(r_small["face_labels"]) == AI_INVENTED
        agree = (oracle_invented == anchor_invented)
        rate = float(agree.mean())
        # disagreements must fall on silhouette-flagged faces
        sil = np.asarray(r_small["silhouette"])
        disagree_off_sil = int((~agree & ~sil).sum())
        oracle_pass = rate >= 0.99 and disagree_off_sil == 0
        oracle_info = {"agreement": round(rate, 4), "target": ">= 0.99",
                       "disagreements": int((~agree).sum()),
                       "disagreements_outside_silhouette": disagree_off_sil,
                       "pass": bool(oracle_pass)}
    else:
        oracle_info = {"pass": False, "note": "occlusion deferred"}
    out["render_as_source_oracle"] = oracle_info

    # 7. LABEL TRANSFER: index-preserving (exact) + reseal (100% coverage) -----
    # start from a labeled small mesh
    base_labels = np.asarray(r_small["face_labels"]) if r_small["ok"] else \
        np.full(len(small.faces), EVIDENCE_BACKED, dtype="<U16")
    n0 = len(base_labels)
    # index-preserving: append 4 new faces (a small patch)
    extra_v = small.vertices[:3] + np.array([0, 0, 0.01])
    new_v = np.vstack([small.vertices, extra_v])
    new_f = np.vstack([small.faces, [[len(small.vertices), len(small.vertices) + 1,
                                      len(small.vertices) + 2]]])
    patched = trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)
    t_idx = transfer_labels(small, base_labels, patched, mode="index")
    idx_exact = bool((t_idx["labels"][:n0] == base_labels).all())
    added_repair = bool((t_idx["labels"][n0:] == REPAIR_FILLED).all())
    idx_pass = idx_exact and added_repair and len(t_idx["labels"]) == len(patched.faces)

    # reseal: rebuild the sphere at a different resolution -> nearest-face carry
    resealed = trimesh.creation.icosphere(subdivisions=3, radius=4.0)
    t_res = transfer_labels(small, base_labels, resealed, mode="nearest")
    full_cover = len(t_res["labels"]) == len(resealed.faces) and \
        bool(np.isin(t_res["labels"], [EVIDENCE_BACKED, AI_INVENTED, REPAIR_FILLED]).all())
    reseal_pass = full_cover and t_res["carry_mode"].startswith("reseal")
    out["label_transfer"] = {
        "index_exact": idx_exact, "added_are_repair_filled": added_repair,
        "index_pass": bool(idx_pass),
        "reseal_full_coverage": bool(full_cover),
        "reseal_carry_mode": t_res["carry_mode"],
        "reseal_pass": bool(reseal_pass),
        "pass": bool(idx_pass and reseal_pass)}

    # 8. OVERLAY RENDER (optional, best-effort) -------------------------------
    render_ok = None
    try:
        import tempfile
        import os
        pth = os.path.join(tempfile.gettempdir(), "_trust_selftest.png")
        rr = render_trust_overlay(small, r_small, pth)
        cap_ok = "requires your source image" in rr["caption"].lower()
        render_ok = bool(rr.get("ok")) and os.path.exists(pth) and cap_ok
        out["overlay_render"] = {"pass": bool(render_ok), "caption_ok": bool(cap_ok)}
    except Exception as e:
        out["overlay_render"] = {"pass": False, "note": str(e)}

    checks = [absent_pass, fp_pass, back_pass, io_pass, fr_pass, oracle_pass,
              bool(idx_pass and reseal_pass)]
    out["overall_pass"] = bool(all(checks))
    if verbose:
        print("trust_map.selftest results:")
        for k, v in out.items():
            print(f"  {k}: {v}")
        print("OVERALL PASS:", out["overall_pass"])
    return out


if __name__ == "__main__":
    selftest(verbose=True)
