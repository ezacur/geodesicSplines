# SPDX-License-Identifier: Apache-2.0
"""
geodesics.py — Geodesic algorithms for triangulated meshes.

Design philosophy
-----------------
All distances are measured as **polyline arc-length** (sum of segment lengths),
exploiting the fact that a geodesic on a triangle mesh is a piecewise-linear
polyline.  This is exact — there is no discretization error in the length.

Two complementary algorithms are provided:

  * **Shooting** (`compute_shoot`): traces a geodesic from a point in a given
    tangent direction for a prescribed arc-length.  Uses iterative ray–edge
    intersection with optional parallel transport across edges.

  * **Endpoint** (`compute_endpoint`, `compute_endpoint_from_origin`): finds
    the shortest geodesic between two arbitrary surface points via the
    Edge-Flip solver (potpourri3d).  Both endpoints are inserted into a
    temporary copy of the mesh topology so the solver operates on exact
    vertex positions — no snapping to pre-existing vertices (unless the point
    is within 1e-9 barycentric tolerance of one).

Topology insertion (``_add_point_buf``) handles three cases:
  - Interior of a face → 1-to-3 subdivision.
  - On an edge (barycentric coord ≈ 0) → edge split (both adjacent faces).
  - On a vertex (barycentric coord ≈ 1) → reuse existing vertex index.

A single `vtkStaticCellLocator` is built once and reused for all spatial
queries: ray-mesh intersection (picking), closest-point projection (cursor),
and face lookup.

Performance conventions
-----------------------
The shooting inner loop and projection kernel are the hottest paths in the
application.  Four ``@njit(cache=True, fastmath=True)`` kernels compile to
native machine code via Numba on first call (~1-2 s, cached to disk across
sessions).  When Numba is absent, the ``@njit`` decorator is a transparent
no-op and the functions execute as regular Python — identical semantics.

  * ``_parallel_transport`` — dihedral rotation across a shared edge.
  * ``_ray_edge_jit`` — ray–edge intersection for one face (replaces
    ``_ray_edge_crossing``; sentinel return instead of ``None``).
  * ``_shoot_loop`` — full inner loop of ``compute_shoot`` (phases 1–7).
  * ``_project_batch_kernel`` — analytical face-plane projection +
    barycentric clamping for ``project_smooth_batch``.

See also: the editor and gizmo modules ship four additional ``@njit``
kernels for screen-space and rendering work — ``_to_screen_kernel``,
``_hover_argmin_sq``, ``_closest_seg_on_polyline_2d`` (in
``geo_shoot.py``) and ``_rotation_x_to_jit`` (in ``gizmo.py``).  They
follow the same scalar-inlined conventions as the ones below.  The
README's "Numba JIT Kernels" table lists all eight with measured
speedups.

All four kernels follow the same conventions — the scalar-inlined style
that was originally motivated by Python interpreter overhead remains
load-bearing under Numba because it maps directly to efficient LLVM IR:

  * **No numpy in the inner loop.**  All vector math (dot, cross, norm) is
    inlined as scalar ``float`` operations.  This avoids Numba boxing
    overhead and generates tight machine code.  Do NOT refactor to
    "clean" numpy — it will be slower in both JIT and interpreter mode.
  * **Pre-allocated path buffer.**  A fixed ``(max_steps+1, 3)`` array is
    sliced at the end — no list appends, no final ``np.array()`` conversion.
  * **Face adjacency matrix** ``_face_adj[fi, edge_i]`` gives O(1) lookup
    of the adjacent face across edge *i*.
  * **Sentinel returns** from ``_ray_edge_jit``: returns a 6-tuple
    ``(found, t, hx, hy, hz, edge_idx)`` where ``found=0`` replaces
    ``None`` (Numba cannot return ``None``).
  * ``_parallel_transport`` is fully inlined to ``math.sqrt`` and scalar
    arithmetic for the same reason.

The topology-insertion path (``prepare_origin`` / ``compute_endpoint_from_origin``)
uses a different strategy — oversized pre-allocated buffers:

  * **V buffer**: frontier overwrite.  The origin is inserted once; each
    endpoint call writes at ``V_buf[nv_cached]`` without copying 120K vertices.
  * **F buffer**: local copy per call (~0.4 ms).  Topology insertion modifies
    *existing* faces (subdivides them), so the cached F must stay clean for
    the next endpoint.  A full frontier overwrite was attempted but requires
    fragile per-face save/restore that isn't worth the complexity.
  * **Robust face lookup** (``_find_face_buf``): unconditionally includes all
    faces created by prior insertions, not just those adjacent to the nearest
    original vertex.  Handles the case where newly inserted vertices are
    invisible to the original-mesh KDTree.
  * **Near-edge nudge** (``_add_point_buf``): when a point's barycentric
    coordinates place it very close to an edge (min coord < 1e-3), the
    point is shifted ~0.1% toward the face centroid before subdivision.
    This prevents sliver triangles with near-zero area that cause NaNs
    in the solver's cotan/area computations.
  * **Retry with nudge** (``compute_endpoint``): if the solver rejects
    the modified mesh on the first attempt, both endpoints are nudged
    toward their face centroids and the insertion is retried.  Only
    falls back to vertex-snap as a last resort.

All these choices are deliberate and load-bearing.  If you are tempted to
"clean up" the inner loop with numpy or simplify the buffer strategy,
**measure first** — the benchmarks are in this project's git history.

Normal field smoothing
----------------------
Real-world meshes often contain nearly-degenerate triangles whose face
normals introduce noise into vertex-normal interpolation, causing visual
jitter in the surface cursor and instability in geodesic shooting
directions.

The smoothing pipeline:

  1. ``_face_normals`` — raw, geometric face normals (cross product).
     Used by ``compute_shoot``'s inner loop for exact ray–edge math.
  2. ``_smooth_face_normals`` — Laplacian-smoothed face normals (N
     iterations, default 5).  Two weighting strategies are available,
     selected by the class variable ``COTANGENT_WEIGHTS``:

       - **Uniform** (default, ``COTANGENT_WEIGHTS = False``): each
         neighbor has equal weight.  Fast; assumes roughly equilateral
         triangles.
       - **Cotangent** (``COTANGENT_WEIGHTS = True``): weights by the
         cotangent of the dihedral angle at the shared edge.  Invariant
         to triangulation quality — better for photogrammetry / scanned
         meshes with long thin triangles, where uniform weights bias the
         smoothed normals toward densely-tessellated regions.

  3. ``_vertex_normals`` — area-weighted averages of *smooth* face
     normals, not raw ones.  Clean by construction.

``get_interpolated_normal`` selects the appropriate source:
  - **Interior** point (all bary > 0.05): returns raw ``_face_normals``
    — exact for the triangle, no noise since it's a single plane.
  - **Near edge/vertex** (any bary < 0.05): barycentric interpolation
    of ``_vertex_normals`` — smooth transitions between faces.
  - **Invalid bary** (locator face-assignment error): falls back to
    the raw face normal as a safe default.

Robustness: face assignment
---------------------------
VTK's ``vtkStaticCellLocator`` can return a face that does not actually
contain the query point (barycentric coords far outside [0, 1]).  This
happens on irregular meshes where buckets straddle many small faces.

``find_face()`` defends with a two-level fallback:
  1. ``FindClosestPoint`` via the VTK locator.  If bary coords validate,
     return immediately.
  2. KDTree nearest-vertex + ``_outside_score`` across all adjacent faces.
     Always finds a geometrically correct face.

``compute_shoot`` applies the same validation before its first step:
if the starting point's bary coords are invalid for the given face,
it calls ``find_face`` and snaps the point to the correct face.

Geodesic spline helpers
-----------------------
Methods for the hybrid geodesic/Euclidean Bézier curves used by
``geo_splines.py``:

  * ``compute_path_lengths(path)`` — pre-computes cumulative segment
    lengths once so that multiple ``geodesic_lerp`` calls on the same
    path avoid redundant ``np.diff`` + ``np.sqrt`` work.
  * ``geodesic_lerp(path, t, _cum, _total)`` — walks a precomputed
    geodesic polyline by arc-length.  Accepts optional pre-computed
    cumulative lengths to skip per-call recomputation.  Uses
    ``np.searchsorted`` instead of a Python scan loop.
  * ``geodesic_lerp_batch(path, t_vals, cum, total)`` — fully vectorized
    multi-*t* interpolation.  One ``searchsorted`` pass + one vectorized
    lerp replaces N individual ``geodesic_lerp`` calls.
  * ``hybrid_de_casteljau_curve(ctrl, path_out, path_in, n, fast)`` —
    cubic de Casteljau where level-1 lerps use geodesic paths and
    levels 2–3 use Euclidean + surface re-projection.  All three levels
    are vectorized across samples — no per-sample Python loop.  Surface
    projections are batched per level (4 batch calls instead of 4N
    individual calls).
  * ``adaptive_samples`` — sample count from control-polygon length.
  * ``project_to_surface`` / ``project_smooth_batch`` — single and
    batch point projection via the VTK locator.

Init-time optimizations
-----------------------
  * ``_smooth_face_normals_laplacian`` reuses the pre-built
    ``_edge_to_face`` dict instead of re-scanning all faces — ~50% faster.
  * ``_compute_vertex_normals`` uses ``np.bincount`` instead of
    ``np.add.at`` for ~10x faster scatter-add (``add.at`` disables SIMD).

Rejected optimizations (deliberate decisions)
----------------------------------------------
These were evaluated and intentionally not implemented:

  * **Threading the debounce computation** (``concurrent.futures``).
    potpourri3d's ``EdgeFlipGeodesicSolver`` has no thread-safety
    guarantees, and segment attributes (numpy arrays) would be written
    from a background thread while the main thread reads them for
    rendering — a data race.  The 340 ms stutter on consolidation is
    acceptable because it only happens once when the mouse stops, not
    during continuous drag.

  * **Rollback pattern for F_buf** instead of copying per endpoint call.
    The 0.4 ms copy is negligible compared to the solver's ~300 ms.
    Rollback would require tracking per-face modifications and restoring
    them — fragile and not worth the complexity for a 0.1% speedup.

  * **``x**2`` → ``x*x`` in the shoot inner loop**.  Measured at < 1%
    improvement.  Kept ``**2`` for readability since the loop is already
    heavily optimized.

  * **NumPy in the shoot fallback path** (vertex/edge case).
    ``cand_d = curr_d - cd_dot * cand_n`` creates temporaries, but this
    path executes ~1% of iterations.  Unrolling it to scalars would add
    15 lines for negligible gain.

Known limitations
-----------------
  * **Degenerate triangles**: ``_ray_edge_crossing`` returns None when all
    three edge determinants are near-zero (triangle area → 0).  The
    vertex/edge fallback in ``compute_shoot`` handles this, but the
    geodesic may lose a few microns of arc-length at the skip.
  * **Geodesic sensitivity**: shooting is inherently sensitive to initial
    conditions on curved surfaces.  Two nearby starting points with
    slightly different directions can produce divergent paths over long
    distances.  This is not a bug — it's a property of geodesics.
  * **VTK locator precision**: ``find_face`` and ``_pick`` can receive
    inconsistent (point, face_id) pairs from the VTK locator on irregular
    meshes.  Mitigated by barycentric validation + KDTree fallback, but
    not eliminated for all mesh configurations.
  * **Memory**: VTK interactor observers and segment actors accumulate
    during long sessions.  Call ``MidpointShooterApp.cleanup()`` to release
    them explicitly.

Next steps
----------
  - Fully geodesic de Casteljau (recursive geodesic lerp at every level
    instead of Euclidean + projection at levels 2–3).
  - Geodesic offset curves (equidistant from a spline, on surface).
  - [DONE] Cotangent-weight Laplacian for normal smoothing
    (``COTANGENT_WEIGHTS = True``).  Off by default.
  - [DONE] Numba JIT compilation of ``compute_shoot`` inner loop,
    ``_ray_edge_crossing``, ``_parallel_transport``, and
    ``project_smooth_batch`` projection kernel.  Falls back to pure
    Python when Numba is not installed.
"""
from __future__ import annotations

import logging
from math import sqrt as _math_sqrt
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import vtk
from scipy.spatial import KDTree
import potpourri3d as pp3d


# Module-level logger for solver diagnostics.  Stays at WARNING by
# default so a normal session is silent; set the parent logger to
# DEBUG (e.g. via ``GEO_SPLINES_DEBUG=1`` from the editor) to surface
# pp3d / VTK fallback chatter.  No handler is attached here — callers
# (geo_splines, spline_export) configure formatting on their own
# loggers, and Python's default propagation routes our records there.
log = logging.getLogger("geodesics")

# Common ndarray type aliases — using numpy.typing for IDE autocompletion
# and static type checkers.  Shape isn't encoded in these (numpy typing
# doesn't support static shapes yet), but the element dtype is.
F64Array = npt.NDArray[np.float64]   # e.g. vertex coords, path points
I32Array = npt.NDArray[np.int32]     # e.g. face indices


class OriginCache(TypedDict):
    """Cache dict returned by ``GeodesicMesh.prepare_origin``.

    Stores everything needed by ``compute_endpoint_from_origin`` to
    compute geodesics from a pre-inserted origin to arbitrary endpoints
    without rebuilding the mesh topology or the solver each time.

    Fields
    ------
    V_buf, F_buf : (N+k, 3) buffers with the origin pre-inserted.
        Oversized by a few slots for in-place endpoint insertion.
    nv, nf : int
        Vertex/face counts after origin insertion.  Use these as
        slice upper bounds (``V_buf[:nv]`` etc.), not ``len(V_buf)``.
    idx : int
        Vertex index of the inserted origin in the modified topology.
    p : (3,) ndarray
        Original (un-snapped) 3D position of the origin.  Used as
        fallback when vertex-snap path returns a direct straight line.
    solver : pp3d.EdgeFlipGeodesicSolver
        Pre-built solver on the origin-inserted mesh.  Reused for
        vertex-snap fast paths; non-snap paths delegate to
        ``compute_endpoint_local`` which builds its own submesh solver.
    kdtree : scipy.spatial.KDTree
        Global-mesh KDTree reference (for callers that snap endpoints).
    """
    V_buf: F64Array
    F_buf: I32Array
    nv: int
    nf: int
    idx: int
    p: F64Array
    solver: object  # pp3d.EdgeFlipGeodesicSolver (no public typing stub)
    kdtree: KDTree

try:
    from numba import njit
    HAS_NUMBA: bool = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """Transparent no-op when Numba is unavailable."""
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


# =====================================================================
#  Numba JIT kernels
#
#  Pure-scalar functions decorated with @njit(cache=True, fastmath=True).  When Numba
#  is installed these compile to native machine code on first call
#  (~1-2 s, cached across sessions).  When Numba is absent the no-op
#  decorator leaves them as regular Python — identical behaviour to the
#  previous hand-inlined code.
#
#  All kernels follow the same conventions:
#    - No Python objects (None, dicts, lists of mixed types).
#    - Arrays in, arrays/scalars out — no intermediate numpy allocations.
#    - Return sentinels instead of None (e.g. found=0).
# =====================================================================

@njit(cache=True, fastmath=True)
def _parallel_transport(d, n1, n2, e):
    """Parallel-transports direction *d* across a shared mesh edge *e*.

    Rotates *d* from the tangent plane of face with normal *n1* to the
    tangent plane of the adjacent face with normal *n2*, preserving the
    component along the edge and rotating the perpendicular component
    through the dihedral angle.

    Modifies *d* **in-place** — zero allocations.  Fully inlined scalar math.
    """
    sqrt = _math_sqrt

    d0, d1, d2 = float(d[0]), float(d[1]), float(d[2])
    e0, e1, e2 = float(e[0]), float(e[1]), float(e[2])

    de = d0*e0 + d1*e1 + d2*e2
    ae0, ae1, ae2 = de*e0, de*e1, de*e2
    dp0, dp1, dp2 = d0 - ae0, d1 - ae1, d2 - ae2

    n1a, n1b, n1c = float(n1[0]), float(n1[1]), float(n1[2])
    n2a, n2b, n2c = float(n2[0]), float(n2[1]), float(n2[2])

    p1x = n1b*e2 - n1c*e1;  p1y = n1c*e0 - n1a*e2;  p1z = n1a*e1 - n1b*e0
    p2x = n2b*e2 - n2c*e1;  p2y = n2c*e0 - n2a*e2;  p2z = n2a*e1 - n2b*e0

    len1 = sqrt(p1x*p1x + p1y*p1y + p1z*p1z)
    len2 = sqrt(p2x*p2x + p2y*p2y + p2z*p2z)

    if len1 < 1e-10 or len2 < 1e-10:
        dn2 = d0*n2a + d1*n2b + d2*n2c
        r0, r1, r2 = d0 - dn2*n2a, d1 - dn2*n2b, d2 - dn2*n2c
    else:
        scale = (dp0*p1x + dp1*p1y + dp2*p1z) / len1
        inv2 = scale / len2
        r0 = ae0 + inv2*p2x;  r1 = ae1 + inv2*p2y;  r2 = ae2 + inv2*p2z

    rn = sqrt(r0*r0 + r1*r1 + r2*r2)
    if rn > 1e-12:
        inv = 1.0 / rn
        d[0] = r0*inv; d[1] = r1*inv; d[2] = r2*inv
    else:
        d[0] = r0; d[1] = r1; d[2] = r2


@njit(cache=True, fastmath=True)
def _ray_edge_jit(fverts, fedges, fedge_len2, fid, px, py, pz, dx, dy, dz, nx, ny, nz):
    """Intersect ray (p, d) with edges of face *fid*.

    Returns ``(found, t, hx, hy, hz, edge_idx)`` where *found* is 0 or 1.
    ``found=0`` replaces the ``None`` return of ``_ray_edge_crossing``
    (Numba cannot return ``None``).

    Numerical robustness
    ~~~~~~~~~~~~~~~~~~~~
    Three thresholds control edge-case behaviour:

      - **det_tol** (``1e-10 * edge_len²``): the determinant
        ``(d × edge) · n`` is near-zero when the ray is almost parallel
        to the edge.  Scaling by ``edge_len²`` makes the tolerance
        invariant to mesh scale.
      - **s_tol** (``1e-4``): edge parametric bounds ``s ∈ [-s_tol, 1+s_tol]``
        accept intersections slightly outside the edge due to float
        rounding.  The hit point is clamped to ``[0, 1]``.
      - **t_min** (``-1e-8``): ``t > -t_min`` instead of ``t > 0`` avoids
        rejecting intersections at the current position (common after
        an edge-to-edge advance of 1e-7).

    On extremely degenerate triangles (area → 0), all three determinants
    may be near-zero, returning ``found=0``.  The vertex/edge fallback in
    ``_shoot_loop`` handles this.
    """
    best_t = 1e30
    best_i = -1
    best_s = 0.0
    s_tol = 1e-4
    t_min = -1e-8

    for i in range(3):
        e0 = fedges[fid, i, 0]
        e1 = fedges[fid, i, 1]
        e2 = fedges[fid, i, 2]

        cx = dy * e2 - dz * e1
        cy = dz * e0 - dx * e2
        cz = dx * e1 - dy * e0
        det = cx * nx + cy * ny + cz * nz
        if abs(det) < 1e-10 * fedge_len2[fid, i]:
            continue

        dfx = fverts[fid, i, 0] - px
        dfy = fverts[fid, i, 1] - py
        dfz = fverts[fid, i, 2] - pz

        # Precompute inverse-det: one division, then two multiplies.
        # LLVM usually applies this optimization automatically, but the
        # explicit form is self-documenting and independent of future
        # compiler changes.
        inv_det = 1.0 / det

        t_val = ((dfy * e2 - dfz * e1) * nx +
                 (dfz * e0 - dfx * e2) * ny +
                 (dfx * e1 - dfy * e0) * nz) * inv_det
        if t_val < t_min:
            continue

        s_val = ((dfy * dz - dfz * dy) * nx +
                 (dfz * dx - dfx * dz) * ny +
                 (dfx * dy - dfy * dx) * nz) * inv_det
        if s_val < -s_tol or s_val > 1.0 + s_tol:
            continue

        if t_val < best_t:
            best_t = t_val
            best_i = i
            best_s = max(0.0, min(1.0, s_val))

    if best_i < 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0)

    hx = fverts[fid, best_i, 0] + best_s * fedges[fid, best_i, 0]
    hy = fverts[fid, best_i, 1] + best_s * fedges[fid, best_i, 1]
    hz = fverts[fid, best_i, 2] + best_s * fedges[fid, best_i, 2]
    return (1, best_t, hx, hy, hz, best_i)


@njit(cache=True, fastmath=True)
def _shoot_loop(curr_p, curr_d, curr_fid, rem, max_steps, fast_mode,
                fnormals, fadj, fverts, fedges, fedge_len2,
                V, F, vf_data, vf_off, path_buf):
    """Inner loop of ``compute_shoot`` — JIT-compiled when Numba is available.

    Validation (VTK locator, barycentric check) is handled by the Python
    wrapper ``compute_shoot``.  This function receives pre-validated
    ``curr_p``, ``curr_d``, ``curr_fid`` and executes phases 1–7 of the
    geodesic tracing algorithm.

    Returns *path_n* (number of points written to *path_buf*).
    A return value ≤ 1 means the shoot failed (no valid path produced).

    Vertex/edge fallback (Phase 2b)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When ``_ray_edge_jit`` fails to find an edge crossing (~1% of
    iterations, typically at degenerate triangles or vertex/edge
    boundaries), the original Python code used ``KDTree.query`` to find
    the nearest mesh vertex.  Since scipy's KDTree is opaque C code that
    Numba cannot compile, the JIT version replaces it with a **local
    nearest-vertex search** over the 3 vertices of the current face.

    This is sufficient because the failure case occurs at a vertex/edge
    boundary — the nearest vertex is always one of the current face's
    vertices.  From that vertex, the CSR arrays ``(vf_data, vf_off)``
    give O(1) access to all adjacent faces, which are tested as
    candidate continuations for the geodesic.
    """
    sqrt = _math_sqrt
    path_buf[0, 0] = curr_p[0]
    path_buf[0, 1] = curr_p[1]
    path_buf[0, 2] = curr_p[2]
    path_n = 1
    edge_buf = np.empty(3)

    for _ in range(max_steps):
        if rem < 1e-12:
            break

        # -- Phase 1: project direction onto face tangent plane --------
        n0 = fnormals[curr_fid, 0]
        n1 = fnormals[curr_fid, 1]
        n2 = fnormals[curr_fid, 2]
        dn_proj = curr_d[0]*n0 + curr_d[1]*n1 + curr_d[2]*n2
        curr_d[0] -= dn_proj * n0
        curr_d[1] -= dn_proj * n1
        curr_d[2] -= dn_proj * n2
        dn = sqrt(curr_d[0]**2 + curr_d[1]**2 + curr_d[2]**2)
        if dn < 1e-12:
            break
        inv_dn = 1.0 / dn
        curr_d[0] *= inv_dn
        curr_d[1] *= inv_dn
        curr_d[2] *= inv_dn

        # -- Phase 2: intersect ray with face edges --------------------
        found, t, hx, hy, hz, edge_idx = _ray_edge_jit(
            fverts, fedges, fedge_len2, curr_fid,
            curr_p[0], curr_p[1], curr_p[2],
            curr_d[0], curr_d[1], curr_d[2],
            n0, n1, n2)

        if found == 0:
            # -- Phase 2b: vertex/edge fallback -------------------------
            # Find nearest vertex of current face (replaces KDTree.query)
            best_vd2 = 1e30
            vi = int(F[curr_fid, 0])
            for k in range(3):
                vk = int(F[curr_fid, k])
                dvx = V[vk, 0] - curr_p[0]
                dvy = V[vk, 1] - curr_p[1]
                dvz = V[vk, 2] - curr_p[2]
                vd2 = dvx*dvx + dvy*dvy + dvz*dvz
                if vd2 < best_vd2:
                    best_vd2 = vd2
                    vi = vk

            best_found = 0
            best_t = -1.0
            best_fi = -1
            best_ei = 0
            best_hx = 0.0
            best_hy = 0.0
            best_hz = 0.0
            cand_d = np.empty(3)
            best_d0 = curr_d[0]
            best_d1 = curr_d[1]
            best_d2 = curr_d[2]

            for _j in range(vf_off[vi], vf_off[vi + 1]):
                cand_fi = int(vf_data[_j])
                if cand_fi == curr_fid:
                    continue
                cn0 = fnormals[cand_fi, 0]
                cn1 = fnormals[cand_fi, 1]
                cn2 = fnormals[cand_fi, 2]
                cd_dot = curr_d[0]*cn0 + curr_d[1]*cn1 + curr_d[2]*cn2
                cand_d[0] = curr_d[0] - cd_dot * cn0
                cand_d[1] = curr_d[1] - cd_dot * cn1
                cand_d[2] = curr_d[2] - cd_dot * cn2
                cdn = sqrt(cand_d[0]**2 + cand_d[1]**2 + cand_d[2]**2)
                if cdn < 1e-12:
                    continue
                inv_cdn = 1.0 / cdn
                cand_d[0] *= inv_cdn
                cand_d[1] *= inv_cdn
                cand_d[2] *= inv_cdn
                if (cand_d[0]*curr_d[0] + cand_d[1]*curr_d[1]
                        + cand_d[2]*curr_d[2]) < 0:
                    continue
                cf, ct, chx, chy, chz, cei = _ray_edge_jit(
                    fverts, fedges, fedge_len2, cand_fi,
                    curr_p[0], curr_p[1], curr_p[2],
                    cand_d[0], cand_d[1], cand_d[2],
                    cn0, cn1, cn2)
                if cf == 1 and (best_found == 0 or ct > best_t):
                    best_found = 1
                    best_fi = cand_fi
                    best_t = ct
                    best_hx = chx
                    best_hy = chy
                    best_hz = chz
                    best_ei = cei
                    best_d0 = cand_d[0]
                    best_d1 = cand_d[1]
                    best_d2 = cand_d[2]

            if best_found == 1:
                curr_fid = best_fi
                n0 = fnormals[curr_fid, 0]
                n1 = fnormals[curr_fid, 1]
                n2 = fnormals[curr_fid, 2]
                curr_d[0] = best_d0
                curr_d[1] = best_d1
                curr_d[2] = best_d2
                found = 1
                t = best_t
                hx = best_hx
                hy = best_hy
                hz = best_hz
                edge_idx = best_ei

        if found == 0:
            break

        # -- Phase 3: arc-length check — exact final point -------------
        if t >= rem:
            path_buf[path_n, 0] = curr_p[0] + rem * curr_d[0]
            path_buf[path_n, 1] = curr_p[1] + rem * curr_d[1]
            path_buf[path_n, 2] = curr_p[2] + rem * curr_d[2]
            path_n += 1
            break

        # -- Phase 4: record edge crossing, advance --------------------
        path_buf[path_n, 0] = hx
        path_buf[path_n, 1] = hy
        path_buf[path_n, 2] = hz
        path_n += 1
        rem -= t

        # -- Phase 5: cross to adjacent face ---------------------------
        adj = int(fadj[curr_fid, edge_idx])
        if adj == -1:
            break

        # -- Phase 6: parallel transport direction across the edge -----
        if fast_mode:
            dn2 = (curr_d[0]*fnormals[adj, 0] + curr_d[1]*fnormals[adj, 1]
                    + curr_d[2]*fnormals[adj, 2])
            curr_d[0] -= dn2 * fnormals[adj, 0]
            curr_d[1] -= dn2 * fnormals[adj, 1]
            curr_d[2] -= dn2 * fnormals[adj, 2]
        else:
            via = int(F[curr_fid, edge_idx])
            vib = int(F[curr_fid, (edge_idx + 1) % 3])
            e0 = V[vib, 0] - V[via, 0]
            e1 = V[vib, 1] - V[via, 1]
            e2 = V[vib, 2] - V[via, 2]
            el2 = e0*e0 + e1*e1 + e2*e2
            if el2 > 1e-24:
                inv_el = 1.0 / sqrt(el2)
                e0 *= inv_el; e1 *= inv_el; e2 *= inv_el
            edge_buf[0] = e0; edge_buf[1] = e1; edge_buf[2] = e2
            _parallel_transport(curr_d, fnormals[curr_fid], fnormals[adj],
                                edge_buf)

        curr_fid = adj

        # -- Phase 7: nudge past edge boundary -------------------------
        curr_p[0] = hx + 1e-7 * curr_d[0]
        curr_p[1] = hy + 1e-7 * curr_d[1]
        curr_p[2] = hz + 1e-7 * curr_d[2]

    return path_n


@njit(cache=True, fastmath=True)
def _project_batch_kernel(pts, nearest_verts, vf_data, vf_off,
                          fverts, fnormals, out, out_faces):
    """Inner projection loop for ``project_smooth_batch``.

    Analytical face-plane projection + barycentric clamping for each
    point against all faces adjacent to its k nearest vertices.
    KDTree query is done in the Python wrapper before calling this kernel.

    *out_faces* (``int32[N]``) receives the index of the face each point
    projected onto (``-1`` if no valid face was found).  Callers that
    don't need the face indices can pass a 1-element dummy buffer.
    """
    n = pts.shape[0]
    for i in range(n):
        px = pts[i, 0]; py = pts[i, 1]; pz = pts[i, 2]
        best_d2 = 1e30
        rx = px; ry = py; rz = pz
        best_fi = -1

        for ki in range(nearest_verts.shape[1]):
            vi = int(nearest_verts[i, ki])
            for _j in range(vf_off[vi], vf_off[vi + 1]):
                fi = int(vf_data[_j])
                ax = fverts[fi, 0, 0]; ay = fverts[fi, 0, 1]; az = fverts[fi, 0, 2]
                nx = fnormals[fi, 0]; ny = fnormals[fi, 1]; nz = fnormals[fi, 2]

                ddx = px - ax; ddy = py - ay; ddz = pz - az
                dot_n = ddx * nx + ddy * ny + ddz * nz
                prx = px - dot_n * nx; pry = py - dot_n * ny; prz = pz - dot_n * nz

                e1x = fverts[fi, 1, 0] - ax; e1y = fverts[fi, 1, 1] - ay
                e1z = fverts[fi, 1, 2] - az
                e2x = fverts[fi, 2, 0] - ax; e2y = fverts[fi, 2, 1] - ay
                e2z = fverts[fi, 2, 2] - az
                v2x = prx - ax; v2y = pry - ay; v2z = prz - az

                d00 = e1x * e1x + e1y * e1y + e1z * e1z
                d01 = e1x * e2x + e1y * e2y + e1z * e2z
                d11 = e2x * e2x + e2y * e2y + e2z * e2z
                d20 = v2x * e1x + v2y * e1y + v2z * e1z
                d21 = v2x * e2x + v2y * e2y + v2z * e2z
                denom = d00 * d11 - d01 * d01
                if abs(denom) < 1e-15:
                    continue

                inv_d = 1.0 / denom
                bv = (d11 * d20 - d01 * d21) * inv_d
                bw = (d00 * d21 - d01 * d20) * inv_d
                bu = 1.0 - bv - bw

                if bu < 0.0: bu = 0.0
                if bv < 0.0: bv = 0.0
                if bw < 0.0: bw = 0.0
                s = bu + bv + bw
                if s > 1e-12:
                    inv_s = 1.0 / s
                    bu *= inv_s; bv *= inv_s; bw *= inv_s

                qx = bu * ax + bv * fverts[fi, 1, 0] + bw * fverts[fi, 2, 0]
                qy = bu * ay + bv * fverts[fi, 1, 1] + bw * fverts[fi, 2, 1]
                qz = bu * az + bv * fverts[fi, 1, 2] + bw * fverts[fi, 2, 2]

                d2 = (px - qx)**2 + (py - qy)**2 + (pz - qz)**2
                if d2 < best_d2:
                    best_d2 = d2
                    rx = qx; ry = qy; rz = qz
                    best_fi = fi

        out[i, 0] = rx; out[i, 1] = ry; out[i, 2] = rz
        out_faces[i] = best_fi


class GeodesicMesh:
    """Geodesic solver engine for 3D triangular meshes.

    Public API
    ----------
    compute_shoot(p, d, length, face_idx, ...)
        Trace a geodesic ray from *p* in tangent direction *d* for exactly
        *length* surface distance.  Returns an Nx3 polyline whose arc-length
        equals *length*.

    compute_endpoint(p_start, p_end)
        Shortest geodesic between two arbitrary surface points.  Both points
        are inserted into a temporary mesh copy so the Edge-Flip solver
        operates on exact positions.

    prepare_origin(p) / compute_endpoint_from_origin(cache, p_end)
        Two-step variant used during handle drag: the origin is pre-inserted
        into a cached solver for vertex-snap fast paths; non-snap endpoints
        delegate to ``compute_endpoint_local``.

    compute_endpoint_local(p_start, p_end)
        Fast geodesic via sphere pre-filter + bidirectional BFS ring growth
        + submesh extraction.  ~10× faster than ``compute_endpoint`` for
        close endpoints (typical span handles, de Casteljau levels, drag
        targets).  Automatic fallback to ``compute_endpoint`` on failure.

    find_face(p)
        Locate the mesh face containing (or nearest to) a 3D point.

    get_interpolated_normal(p, face_id)
        Smooth vertex-normal interpolation via barycentric coordinates.

    Attributes
    ----------
    V : ndarray (N, 3)       — vertex positions.
    F : ndarray (M, 3)       — triangle face indices.
    locator                  — vtkStaticCellLocator for spatial queries.
    _face_verts : (M, 3, 3)  — pre-indexed vertex coords per face.
    _face_edges : (M, 3, 3)  — edge vectors per face (cyclic).
    _face_adj : (M, 3) int32 — adjacent face per edge, -1 = boundary.
    """

    # Enable surface/midpoint distance checks and shoot truncation warnings.
    DIAGNOSE_PATHS = False

    # Normal smoothing strategy.  When False, uses uniform-weight Laplacian
    # (fast, assumes roughly equilateral triangles).  When True, uses
    # cotangent-weight Laplacian (invariant to triangulation quality —
    # better for photogrammetry / scanned meshes with long thin triangles).
    COTANGENT_WEIGHTS = True

    # One-shot spatial reordering of V and F by 3D Morton code (Z-order
    # curve) at construction time.  Every downstream structure
    # (``_face_verts``, ``_face_adj``, ``_face_normals``, ``_vf_data``,
    # ``KDTree``, ``EdgeFlipGeodesicSolver``, ``vtkStaticCellLocator``)
    # is built AFTER the permutation, so the hot-path code is unchanged —
    # the permutation just propagates naturally.
    #
    # Why it helps: when ``_shoot_loop`` steps from face ``fi`` to the
    # adjacent face ``_face_adj[fi, e]``, Morton ordering makes the
    # neighbour's entries in ``_face_verts`` / ``_face_normals`` /
    # ``_face_adj`` sit a few cache lines away instead of potentially
    # megabytes apart (the original mesh file order is usually
    # semi-random w.r.t. 3D position).  Same benefit for the bidirectional
    # BFS in ``compute_endpoint_local``.
    #
    # When it matters: mostly on meshes whose working set exceeds L3
    # (~16-64 MB on modern CPUs → roughly 1M+ faces).  On smaller meshes
    # everything fits in cache after warm-up and the gain is 5-10%.
    # On multi-million-face scans the speedup can reach 20-40% on the
    # traversal loops.
    #
    # Safety: splines are saved as 3D positions + tangents, never as
    # vertex indices, so reordering does not break JSON save/load.  The
    # flag is there purely for A/B benchmarking — leave it ON by default,
    # it is essentially free on small meshes and real on large ones.
    MORTON_REORDER = True

    def __init__(self, V: np.ndarray | object, F: np.ndarray | None = None):
        if hasattr(V, 'points') and hasattr(V, 'faces'):
            self._pv_mesh = V
            self.V = np.asarray(V.points, dtype=float)
            f = np.asarray(V.faces, dtype=int).reshape(-1, 4); self.F = f[:, 1:]
        else:
            self._pv_mesh = None
            self.V = np.asarray(V, dtype=float); self.F = np.asarray(F, dtype=int)

        # One-shot Morton reorder BEFORE any downstream structure is built.
        # All later arrays (_face_*, _vf_*, KDTree, solver, VTK locator)
        # naturally inherit the improved spatial locality.  See the
        # ``MORTON_REORDER`` class-level docstring for the rationale.
        if self.MORTON_REORDER:
            self._morton_reorder_inplace()

        self._kdtree         = KDTree(self.V)
        self._face_normals   = self._compute_face_normals()
        self._edge_to_face   = self._build_edge_adjacency()

        # Pre-computed face geometry (avoids double-indexing V[F[i]] in hot loops)
        self._face_verts = self.V[self.F]                        # (N_faces, 3, 3)
        self._face_edges = np.roll(self._face_verts, -1, axis=1) - self._face_verts
        # Pre-computed squared edge lengths — avoids 3 muls + 2 adds per
        # edge in the ray-edge intersection inner loop.
        self._face_edge_len2 = np.sum(self._face_edges ** 2, axis=2)  # (N_faces, 3)
        self._face_centroids = self._face_verts.mean(axis=1)        # (N_faces, 3)

        # Static face adjacency matrix — built before smooth normals so
        # _smooth_face_normals_laplacian can use the vectorized adjacency.
        self._face_adj = self._build_face_adjacency_matrix()
        self._face_components = self._compute_face_components()
        # Smoothing strategy selected by COTANGENT_WEIGHTS class variable.
        # See module docstring 'Normal field smoothing' for rationale.
        if self.COTANGENT_WEIGHTS:
            self._smooth_face_normals = self._smooth_face_normals_cotangent(iterations=5)
        else:
            self._smooth_face_normals = self._smooth_face_normals_laplacian(iterations=5)
        self._vertex_normals = self._compute_vertex_normals()
        self._vf_data, self._vf_offsets = self._build_vertex_faces()
        self._solver         = pp3d.EdgeFlipGeodesicSolver(self.V, self.F)

        # Central VTK locator — used for ALL surface queries (pick, project, find_face).
        self.locator = self._build_locator()

        # Pre-allocated VTK refs (avoids per-call object creation)
        self._vtk_cp = [0.0, 0.0, 0.0]
        self._vtk_cell_id = vtk.reference(0)
        self._vtk_sub_id = vtk.reference(0)
        self._vtk_dist2 = vtk.reference(0.0)

        # Set to True by ``compute_endpoint`` / ``compute_endpoint_local``
        # when the solver could not produce a true geodesic and fell back
        # to the 2-point straight-line stub.  Callers read this flag right
        # after the call to paint degraded spans red.  Not thread-safe, but
        # workers each have their own mesh instance so there is no sharing.
        self._last_was_fallback: bool = False

    # --- Morton / Z-order reordering -----------------------------------

    @staticmethod
    def _morton3_encode(qx: np.ndarray, qy: np.ndarray,
                        qz: np.ndarray) -> np.ndarray:
        """Interleaves the low 21 bits of *qx*, *qy*, *qz* into a uint64
        3D Morton code (Z-order curve).

        Bit layout of the result: ``z21 y21 x21 ... z0 y0 x0`` — each
        axis contributes one bit per group of 3, most significant first.
        Two points that are close in 3D end up with close Morton codes,
        so ``np.argsort`` of the codes produces a vertex/face order that
        traverses the mesh coherently in memory.

        Uses the classic "magic number" bit-spreading trick (a.k.a.
        "Dilated integers") instead of a per-bit loop — ~10× faster on
        numpy arrays of millions of elements.  The input is masked to
        21 bits so the final 63-bit code fits comfortably in uint64.
        """
        def _spread21(v: np.ndarray) -> np.ndarray:
            v = v.astype(np.uint64) & np.uint64(0x1FFFFF)
            v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
            v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
            v = (v | (v << np.uint64(8)))  & np.uint64(0x100F00F00F00F00F)
            v = (v | (v << np.uint64(4)))  & np.uint64(0x10C30C30C30C30C3)
            v = (v | (v << np.uint64(2)))  & np.uint64(0x1249249249249249)
            return v
        return (_spread21(qx)
                | (_spread21(qy) << np.uint64(1))
                | (_spread21(qz) << np.uint64(2)))

    @classmethod
    def _morton_codes_for_points(cls, pts: np.ndarray) -> np.ndarray:
        """Computes a 3D Morton code per row of *pts* (shape ``(N, 3)``).

        Quantizes each coordinate to 21 bits inside the mesh's axis-aligned
        bounding box.  21 bits per axis = 2^21 ≈ 2 million buckets per
        axis, far finer than any practical mesh — two distinct vertices
        get distinct codes unless they coincide numerically.
        """
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        # Add a tiny epsilon so the top-right corner doesn't overflow to
        # 2^21.  Scale then cast to uint32 (21 bits fits comfortably).
        extent = np.maximum(bbox_max - bbox_min, 1e-30)
        scale = (1 << 21) - 1
        q = ((pts - bbox_min) / extent * scale).astype(np.uint32)
        return cls._morton3_encode(q[:, 0], q[:, 1], q[:, 2])

    def _morton_reorder_inplace(self) -> None:
        """Permutes ``self.V`` and ``self.F`` by 3D Morton code.

        Two-pass reorder:
          1. **Vertices**: sort V by the Morton code of each vertex
             position.  Build the inverse permutation ``inv_perm_v`` to
             remap face indices from old to new V.
          2. **Faces**: sort F by the Morton code of each face centroid
             (computed from the new V + old F after step 1).  Faces that
             share an edge tend to have similar centroids → they end up
             near each other in memory.

        Both permutations are pure numpy fancy-indexing ops — runs in a
        few ms even on million-face meshes.  After this method returns,
        ``self.V`` and ``self.F`` are in their final layout and every
        downstream structure is built on top.

        No cross-file invariants are broken: splines are persisted as
        3D positions, not vertex indices, so save/load works unchanged.
        """
        # --- Step 1: reorder vertices ---
        perm_v = np.argsort(
            self._morton_codes_for_points(self.V), kind='stable')
        inv_perm_v = np.empty_like(perm_v)
        inv_perm_v[perm_v] = np.arange(len(perm_v), dtype=perm_v.dtype)
        self.V = np.ascontiguousarray(self.V[perm_v])
        # Remap face vertex indices into the new V ordering.
        self.F = inv_perm_v[self.F].astype(self.F.dtype, copy=False)

        # --- Step 2: reorder faces by centroid Morton code ---
        centroids = self.V[self.F].mean(axis=1)
        perm_f = np.argsort(
            self._morton_codes_for_points(centroids), kind='stable')
        self.F = np.ascontiguousarray(self.F[perm_f])

    def _build_face_adjacency_matrix(self) -> np.ndarray:
        """Static (M, 3) int32 matrix for O(1) face-neighbor lookup.

        ``_face_adj[fi, i]`` = index of the face sharing edge *i* of face
        *fi*, or -1 if boundary.  Edge *i* connects ``F[fi, i]`` →
        ``F[fi, (i+1)%3]``.

        Fully vectorized via edge-key sorting — no Python loops over faces.
        """
        F = self.F
        nf = len(F)
        nv = len(self.V)

        # All directed half-edges: 3 per face
        i0 = np.column_stack([F[:, 0], F[:, 1], F[:, 2]]).ravel()
        i1 = np.column_stack([F[:, 1], F[:, 2], F[:, 0]]).ravel()
        face_ids = np.repeat(np.arange(nf, dtype=np.int32), 3)
        edge_local = np.tile(np.arange(3, dtype=np.int32), nf)

        # Canonical edge key: (min, max) packed as single int64
        lo = np.minimum(i0, i1).astype(np.int64)
        hi = np.maximum(i0, i1).astype(np.int64)
        keys = lo * nv + hi

        # Sort to group matching edges — stable sort keeps insertion order
        order = np.argsort(keys, kind='mergesort')
        keys_s = keys[order]
        fids_s = face_ids[order]
        elocal_s = edge_local[order]

        # Adjacent entries with same key share an edge
        adj = np.full((nf, 3), -1, dtype=np.int32)
        same = keys_s[:-1] == keys_s[1:]
        idx = np.where(same)[0]
        fi_a, ei_a = fids_s[idx], elocal_s[idx]
        fi_b, ei_b = fids_s[idx + 1], elocal_s[idx + 1]
        adj[fi_a, ei_a] = fi_b
        adj[fi_b, ei_b] = fi_a
        return adj

    def _compute_face_components(self) -> np.ndarray:
        """Labels each face with its connected component index.

        Uses BFS on ``_face_adj``.  Returns an int32 array of length
        ``len(F)`` where ``labels[fi]`` is the component id (0-based).
        """
        nf = len(self.F)
        labels = np.full(nf, -1, dtype=np.int32)
        adj = self._face_adj
        comp = 0
        for seed in range(nf):
            if labels[seed] >= 0:
                continue
            # BFS from seed
            queue = [seed]
            labels[seed] = comp
            head = 0
            while head < len(queue):
                fi = queue[head]; head += 1
                for nb in adj[fi]:
                    if nb >= 0 and labels[nb] < 0:
                        labels[nb] = comp
                        queue.append(nb)
            comp += 1
        return labels

    def same_component(self, face_a: int, face_b: int) -> bool:
        """Returns True if *face_a* and *face_b* are in the same connected component.

        Returns True (optimistic) if either index is out of range — lets
        the caller attempt the geodesic rather than silently rejecting it.
        """
        nf = len(self._face_components)
        if not (0 <= face_a < nf and 0 <= face_b < nf):
            return True
        return int(self._face_components[face_a]) == int(self._face_components[face_b])

    def _build_locator(self) -> vtk.vtkStaticCellLocator | None:
        """Builds a tuned vtkStaticCellLocator for fast spatial queries.

        ``vtkStaticCellLocator`` is the optimal choice for static meshes:
        its uniform-grid bucket structure gives O(1) bucket access for
        ``FindClosestPoint`` — faster than octree (``vtkCellLocator``) or
        BSP (``vtkModifiedBSPTree``) for point-proximity queries on fixed
        geometry.  ``vtkCellTreeLocator`` has comparable ray-intersection
        performance but no advantage for the ``FindClosestPoint`` calls
        that dominate this application's projection workload.
        """
        if self._pv_mesh is None:
            return None
        loc = vtk.vtkStaticCellLocator()
        loc.SetDataSet(self._pv_mesh)
        loc.SetNumberOfCellsPerNode(8)
        loc.SetMaxNumberOfBuckets(max(len(self.F) // 4, 1000))
        loc.BuildLocator()
        return loc

    def _compute_face_normals(self) -> np.ndarray:
        A, B, C = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        cross = np.cross(B - A, C - A); norms = np.linalg.norm(cross, axis=1, keepdims=True)
        return cross / np.where(norms < 1e-15, 1.0, norms)

    def _smooth_face_normals_laplacian(self, iterations: int = 5) -> np.ndarray:
        """Uniform-weight Laplacian smoothing of face normals.

        Each face normal is averaged with its edge-adjacent neighbors with
        equal weight.  Fast and effective when triangles are roughly
        equilateral.  For meshes with irregular triangulation (long thin
        triangles), see ``_smooth_face_normals_cotangent`` which uses
        dihedral-angle cotangent weights.

        Builds the adjacency matrix from the pre-computed ``_face_adj`` array
        (vectorized, no dict iteration).  Each iteration is a single sparse
        matmul ``normals = A @ normals`` followed by row-wise re-normalization.
        """
        from scipy.sparse import coo_matrix, diags

        nf = len(self.F)
        adj = self._face_adj

        # Build sparse adjacency from _face_adj: fully vectorized
        fi_arr = np.repeat(np.arange(nf, dtype=np.int32), 3)
        fj_arr = adj.ravel()
        mask = fj_arr >= 0
        A = coo_matrix(
            (np.ones(int(mask.sum()), dtype=float),
             (fi_arr[mask], fj_arr[mask])),
            shape=(nf, nf)).tocsr()

        # Normalize rows → each row sums to 1 (average of neighbors)
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums[row_sums < 1e-15] = 1.0
        A = diags(1.0 / row_sums) @ A

        normals = self._face_normals.copy()
        for _ in range(iterations):
            normals = A @ normals
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.where(norms < 1e-15, 1.0, norms)
        return normals

    def _smooth_face_normals_cotangent(self, iterations: int = 5) -> np.ndarray:
        """Cotangent-weight Laplacian smoothing of face normals.

        Uses the cotangent of the dihedral angle at each shared edge as the
        weight for averaging adjacent face normals.  This makes the smoothing
        **invariant to triangulation quality** — long, thin triangles
        (common in photogrammetry / scanned meshes) get appropriately
        lower influence than well-shaped neighbors.

        The dihedral angle at an edge is the angle between the two face
        normals sharing that edge.  The cotangent weight is
        ``cot(θ) = cos(θ) / sin(θ)``, clamped to ``[-10, 10]`` to avoid
        blow-up at near-flat or near-folded edges.

        Activated by setting ``COTANGENT_WEIGHTS = True`` on the class.

        Builds the weighted adjacency matrix from ``_face_adj`` and
        ``_face_normals``, then iterates sparse matmul + re-normalization
        identically to the uniform-weight version.
        """
        from scipy.sparse import coo_matrix, diags

        nf = len(self.F)
        adj = self._face_adj
        fn = self._face_normals

        # Build row/col indices for all valid adjacencies
        fi_arr = np.repeat(np.arange(nf, dtype=np.int32), 3)
        fj_arr = adj.ravel()
        mask = fj_arr >= 0
        rows = fi_arr[mask]
        cols = fj_arr[mask]

        # Cotangent weights from dihedral angles between adjacent face normals
        ni = fn[rows]           # (K, 3) normals of face i
        nj = fn[cols]           # (K, 3) normals of face j
        cos_theta = np.sum(ni * nj, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
        sin_theta = np.maximum(sin_theta, 1e-10)  # avoid division by zero
        cot_w = cos_theta / sin_theta
        cot_w = np.clip(cot_w, -10.0, 10.0)

        # Shift to positive weights: w = cot + 10.001 (ensures all > 0)
        weights = cot_w + 10.001

        A = coo_matrix((weights, (rows, cols)), shape=(nf, nf)).tocsr()

        # Normalize rows → each row sums to 1 (weighted average)
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums[row_sums < 1e-15] = 1.0
        A = diags(1.0 / row_sums) @ A

        normals = self._face_normals.copy()
        for _ in range(iterations):
            normals = A @ normals
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.where(norms < 1e-15, 1.0, norms)
        return normals

    def _compute_vertex_normals(self) -> np.ndarray:
        """Angle-weighted vertex normals from smoothed face normals.

        Uses the **angle subtended at each vertex** (Baerentzen & Aanaes
        pseudo-normals, 2005) as the weight when accumulating face
        normals at vertices.  This is mathematically correct for normal
        interpolation and robust on obtuse or degenerate triangles —
        where pure area-weighting gives the wrong answer (a huge obtuse
        triangle contributes disproportionately to its 30° vertex).

        For each face with vertices A, B, C, the contribution to each
        vertex's normal is ``face_normal × angle_at_vertex``, where the
        angles sum to π and capture the visual "coverage" of the face
        at each vertex.

        Boundary-robust: isolated vertices (no incident faces) get the
        default normal (0,0,0) → normalized to (0,0,0), safe for
        downstream code that checks ``norm > 0``.

        Uses ``np.bincount`` instead of ``np.add.at`` for ~10x faster
        scatter-add on large meshes (``add.at`` disables SIMD vectorization).
        Per axis, the three per-corner scatters are concatenated into a
        single ``bincount`` call for better L1 cache reuse.
        """
        A = self.V[self.F[:, 0]]
        B = self.V[self.F[:, 1]]
        C = self.V[self.F[:, 2]]

        # Compute interior angles at A, B, C (per triangle)
        eAB = B - A; eAC = C - A; eBC = C - B
        lAB = np.linalg.norm(eAB, axis=1); lAC = np.linalg.norm(eAC, axis=1)
        lBC = np.linalg.norm(eBC, axis=1)
        lAB = np.maximum(lAB, 1e-15); lAC = np.maximum(lAC, 1e-15)
        lBC = np.maximum(lBC, 1e-15)

        # Angle at A: arccos(dot(AB, AC) / (|AB||AC|))
        cos_a = np.sum(eAB * eAC, axis=1) / (lAB * lAC)
        cos_b = np.sum(-eAB * eBC, axis=1) / (lAB * lBC)
        cos_c = np.sum(-eAC * -eBC, axis=1) / (lAC * lBC)
        ang_a = np.arccos(np.clip(cos_a, -1.0, 1.0))
        ang_b = np.arccos(np.clip(cos_b, -1.0, 1.0))
        ang_c = np.arccos(np.clip(cos_c, -1.0, 1.0))

        fn = self._smooth_face_normals
        nv = len(self.V)
        vn = np.zeros((nv, 3), dtype=float)
        # Flatten corner indices once — one bincount per axis covers all
        # three corners (A, B, C) of every face in a single pass.
        idx_all = np.concatenate((self.F[:, 0], self.F[:, 1], self.F[:, 2]))
        for c in range(3):
            w_all = np.concatenate((ang_a * fn[:, c],
                                    ang_b * fn[:, c],
                                    ang_c * fn[:, c]))
            vn[:, c] = np.bincount(idx_all, weights=w_all, minlength=nv)

        norms = np.linalg.norm(vn, axis=1, keepdims=True)
        return vn / np.where(norms < 1e-15, 1.0, norms)

    def _build_edge_adjacency(self) -> dict:
        """Undirected edge → face list mapping.

        Vectorized construction: all edges are computed and sorted in NumPy,
        then grouped into the dict in a single pass over sorted arrays.
        """
        F = self.F
        nf = len(F)
        nv = len(self.V)

        v0 = np.column_stack([F[:, 0], F[:, 1], F[:, 2]]).ravel()
        v1 = np.column_stack([F[:, 1], F[:, 2], F[:, 0]]).ravel()
        face_ids = np.repeat(np.arange(nf, dtype=np.int64), 3)
        lo = np.minimum(v0, v1).astype(np.int64)
        hi = np.maximum(v0, v1).astype(np.int64)
        keys = lo * nv + hi

        order = np.argsort(keys, kind='mergesort')
        keys_s = keys[order]
        fids_s = face_ids[order]

        breaks = np.concatenate([[0], np.where(np.diff(keys_s))[0] + 1,
                                 [len(keys_s)]])
        emap = {}
        for i in range(len(breaks) - 1):
            s, e = int(breaks[i]), int(breaks[i + 1])
            k = int(keys_s[s])
            emap[(k // nv, k % nv)] = fids_s[s:e].tolist()
        return emap

    def _build_vertex_faces(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-vertex face adjacency in CSR format (cache-friendly, Numba-ready).

        Returns ``(data, offsets)`` where faces adjacent to vertex *v* are
        ``data[offsets[v]:offsets[v+1]]``.

        Vectorized via ``argsort`` + ``searchsorted``: all (vertex, face) pairs
        are sorted by vertex id in NumPy, then split via offset array.
        """
        nv = len(self.V)
        nf = len(self.F)
        vertex_ids = self.F.ravel().astype(np.int32)
        face_ids = np.repeat(np.arange(nf, dtype=np.int32), 3)
        order = np.argsort(vertex_ids, kind='mergesort')
        data = face_ids[order].astype(np.int32)
        offsets = np.searchsorted(vertex_ids[order], np.arange(nv + 1)).astype(np.int32)
        return data, offsets

    def face_normal(self, face_id: int) -> np.ndarray:
        return self._face_normals[int(face_id)].copy()

    @staticmethod
    def _barycentric(p: np.ndarray, A: np.ndarray, B: np.ndarray,
                     C: np.ndarray) -> tuple[float, float, float]:
        """Barycentric coordinates of *p* w.r.t. triangle (A, B, C).

        Single canonical implementation shared by ``get_barycentric``,
        ``_bary_buf``, and ``get_barycentric``.
        """
        v0, v1, v2 = B - A, C - A, p - A
        d00 = np.dot(v0, v0); d01 = np.dot(v0, v1); d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0); d21 = np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-15:
            return 1/3, 1/3, 1/3
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        return 1.0 - v - w, v, w

    def get_barycentric(self, p: np.ndarray, face_id: int) -> tuple[float, float, float]:
        A, B, C = self.V[self.F[face_id]]
        return self._barycentric(p, A, B, C)

    def get_interpolated_normal(self, p: F64Array, face_id: int) -> F64Array:
        """Returns surface normal at point p on face face_id.

        Strategy based on barycentric validity:
          - Invalid (any coord outside [-0.1, 1.1]): fall back to face normal.
            Catches locator face-assignment errors.
          - Interior (all coords > 0.05): raw face normal (exact, no noise).
          - Near edge/vertex: barycentric interpolation of smooth vertex normals.
        """
        u, v, w = self.get_barycentric(p, face_id)
        # Guard: bary coords wildly off → wrong face assignment
        if max(u, v, w) > 1.1 or min(u, v, w) < -0.1:
            return self._face_normals[int(face_id)].copy()
        if min(u, v, w) > 0.05:
            return self._face_normals[int(face_id)].copy()
        f = self.F[face_id]
        # Clamp to [0,1] for safety near edges
        u, v, w = max(0, u), max(0, v), max(0, w)
        s = u + v + w
        if s > 1e-12:
            u, v, w = u/s, v/s, w/s
        n = u * self._vertex_normals[f[0]] + v * self._vertex_normals[f[1]] + w * self._vertex_normals[f[2]]
        nn = np.linalg.norm(n)
        return n / nn if nn > 1e-12 else self.face_normal(face_id)

    def find_face(self, p: F64Array) -> int:
        """Locates the face containing (or nearest to) a 3D point.

        Uses the VTK locator first; if the result has invalid barycentric
        coords (locator precision issue on irregular meshes), falls back
        to KDTree nearest-vertex + barycentric scoring.
        """
        if self.locator is not None:
            self.locator.FindClosestPoint(
                p, self._vtk_cp, self._vtk_cell_id, self._vtk_sub_id, self._vtk_dist2)
            cid = int(self._vtk_cell_id)
            u, v, w = self.get_barycentric(p, cid)
            if min(u, v, w) >= -0.1 and max(u, v, w) <= 1.1:
                return cid
            # Locator gave bad face — fall through to KDTree
        _, vi = self._kdtree.query(p)
        vi = int(vi)
        cands = self._vf_data[self._vf_offsets[vi]:self._vf_offsets[vi + 1]]
        return int(min(cands, key=lambda fi: self._outside_score(p, int(fi))))

    def _outside_score(self, p: np.ndarray, fi: int) -> float:
        u, v, w = self.get_barycentric(p, fi)
        return max(0.0, -u) + max(0.0, -v) + max(0.0, -w)

    def compute_shoot(self, p_start: F64Array, d_vec: F64Array, length: float,
                    face_idx: int = None, max_steps: int = 400,
                    fast_mode: bool = False) -> F64Array | None:
        """Shoot a geodesic from p_start in direction d_vec for arc-length *length*.

        Returns an Nx3 polyline (or None if the shoot fails immediately).

        Validation (VTK locator / barycentric check) stays in Python;
        the hot inner loop dispatches to ``_shoot_loop`` which is
        ``@njit``-compiled when Numba is available.

        Parameters
        ----------
        fast_mode : bool
            If True, skips parallel transport (direction maintains global
            orientation across edges).  Faster but less accurate on curved
            surfaces.  Used for cursor crosshair and drag previews.
        """
        # ``_vtk_cp`` is only meaningful when ``find_face`` took the VTK
        # locator branch — that branch is what populates it via
        # ``locator.FindClosestPoint``.  When the mesh was built from
        # raw (V, F) arrays (no ``pv.PolyData``), ``self.locator`` is
        # None, ``find_face`` falls through to the KDTree path, and
        # ``_vtk_cp`` keeps its zero-initialised value — historically
        # this silently zeroed ``p_start``, so the shoot started at the
        # world origin and the inner loop returned None for every node.
        # Only snap to the locator's closest-point when we actually have
        # a locator; otherwise trust the input ``p_start``.
        if face_idx is None:
            face_idx = self.find_face(p_start)
            if self.locator is not None:
                p_start = np.array(self._vtk_cp, dtype=float)
        else:
            u, v, w = self.get_barycentric(p_start, face_idx)
            if min(u, v, w) < -0.1 or max(u, v, w) > 1.1:
                face_idx = self.find_face(p_start)
                if self.locator is not None:
                    p_start = np.array(self._vtk_cp, dtype=float)

        curr_p = np.empty(3, dtype=float)
        curr_p[0] = p_start[0]; curr_p[1] = p_start[1]; curr_p[2] = p_start[2]
        curr_d = np.empty(3, dtype=float)
        curr_d[0] = d_vec[0]; curr_d[1] = d_vec[1]; curr_d[2] = d_vec[2]

        path_buf = np.empty((max_steps + 1, 3), dtype=float)
        path_n = _shoot_loop(
            curr_p, curr_d, int(face_idx), float(length), max_steps,
            fast_mode, self._face_normals, self._face_adj,
            self._face_verts, self._face_edges, self._face_edge_len2,
            self.V, self.F, self._vf_data, self._vf_offsets, path_buf)

        return path_buf[:path_n] if path_n > 1 else None

    def _ray_edge_crossing(self, p, d, face_id, n):
        """Intersect ray (p, d) with edges of face_id.

        Thin wrapper around the ``@njit`` kernel ``_ray_edge_jit``.
        Returns ``(t, hx, hy, hz, edge_idx)`` or ``None``.
        """
        found, t, hx, hy, hz, ei = _ray_edge_jit(
            self._face_verts, self._face_edges, self._face_edge_len2,
            int(face_id),
            float(p[0]), float(p[1]), float(p[2]),
            float(d[0]), float(d[1]), float(d[2]),
            float(n[0]), float(n[1]), float(n[2]))
        if found == 0:
            return None
        return (t, hx, hy, hz, int(ei))

    # --- ROBUST DYNAMIC RECONSTRUCTION ---

    def diagnose_path(self, path: np.ndarray, label: str) -> None:
        """Checks whether path points and segment midpoints lie on the mesh surface."""
        if not self.DIAGNOSE_PATHS:
            return
        if path is None or len(path) < 2:
            print(f"  [diag:{label}] path is None or degenerate")
            return
        dists, _ = self._kdtree.query(path)
        max_d, mean_d = float(dists.max()), float(dists.mean())
        geo_len = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        euclid  = float(np.linalg.norm(path[-1] - path[0]))
        ratio   = geo_len / euclid if euclid > 1e-12 else float('inf')
        mesh_scale = float(np.linalg.norm(self.V.max(axis=0) - self.V.min(axis=0)))
        surface_ok = max_d < 0.05 * mesh_scale

        # Check midpoints of each path segment
        midpoints = 0.5 * (path[:-1] + path[1:])
        mid_dists, _ = self._kdtree.query(midpoints)
        max_md, mean_md = float(mid_dists.max()), float(mid_dists.mean())
        mid_ok = max_md < 0.05 * mesh_scale

        if not surface_ok:
            print(f"  [diag:{label}] OFF-SURFACE  pts={len(path)}  geo_len={geo_len:.4f}  "
                  f"ratio={ratio:.2f}  max_dist={max_d:.4e}  mean={mean_d:.4e}")
        if not mid_ok:
            print(f"  [diag:{label}] MIDPOINTS OFF-SURFACE  segs={len(midpoints)}  "
                  f"max_dist={max_md:.4e}  mean={mean_md:.4e}")


    def project_smooth_batch(self, pts: F64Array) -> F64Array:
        """Batch projection onto nearest triangle surfaces.

        Phase 1 (Python): batch KDTree query for k=7 nearest vertices.
        The kernel then searches faces incident to ANY of those vertices
        (via ``_vf_data`` / ``_vf_offsets``) and picks the closest.

        Phase 2 (JIT kernel): analytical face-plane projection +
        barycentric clamping via ``_project_batch_kernel``.

        Why k=7 (was k=3)
        ~~~~~~~~~~~~~~~~~
        On sliver triangles (long, thin — common in photogrammetry),
        the closest face to a 3D point may have NONE of its vertices
        among the 3 nearest: the triangle's "long axis" aligns away
        from the query point, so its vertices are further than those
        of smaller, well-shaped neighbors that happen to be on the
        wrong side of the face.  With k=7 the correct face is
        virtually always in the candidate set.  The extra ~4 vertices
        add <5% to the query cost.
        """
        pts = np.ascontiguousarray(pts, dtype=np.float64)
        _, nearest_verts = self._kdtree.query(pts, k=7)
        nearest_verts = np.asarray(nearest_verts, dtype=np.int64)
        if nearest_verts.ndim == 1:
            nearest_verts = nearest_verts.reshape(-1, 1)

        out = np.empty((len(pts), 3), dtype=np.float64)
        out_faces = np.empty(len(pts), dtype=np.int32)  # discarded
        _project_batch_kernel(pts, nearest_verts,
                              self._vf_data, self._vf_offsets,
                              self._face_verts, self._face_normals,
                              out, out_faces)
        return out

    def project_smooth_batch_with_faces(
            self, pts: F64Array) -> tuple[F64Array, I32Array]:
        """Same as ``project_smooth_batch`` but also returns, for each
        input point, the index of the face it landed on (``-1`` if no
        valid face was found — should not happen on a clean mesh).

        Used by ``compute_endpoint_local`` to seed its submesh region
        from the projection of the straight A→B line.
        """
        pts = np.ascontiguousarray(pts, dtype=np.float64)
        _, nearest_verts = self._kdtree.query(pts, k=7)
        nearest_verts = np.asarray(nearest_verts, dtype=np.int64)
        if nearest_verts.ndim == 1:
            nearest_verts = nearest_verts.reshape(-1, 1)

        out = np.empty((len(pts), 3), dtype=np.float64)
        out_faces = np.empty(len(pts), dtype=np.int32)
        _project_batch_kernel(pts, nearest_verts,
                              self._vf_data, self._vf_offsets,
                              self._face_verts, self._face_normals,
                              out, out_faces)
        return out, out_faces

    def subdivide_secant_chords(self, pts: F64Array,
                                tol: float | None = None,
                                max_depth: int = 6) -> F64Array:
        """Recursively subdivide polyline segments that cut through the mesh.

        When two consecutive points of a surface polyline sit on opposite
        sides of a mesh feature (ridge, crease), the straight chord between
        them passes *below* the surface — producing a visible artifact where
        the line disappears behind the mesh.

        This method detects such segments by projecting the chord midpoint
        onto the surface and measuring its deviation from the Euclidean
        midpoint.  When the deviation exceeds *tol*, the segment is split
        at the projected midpoint and both halves are checked (up to
        *max_depth* iterations).

        Implementation: **level-synchronous batched processing**.  At each
        iteration, all current chord midpoints are computed at once,
        projected together via the vectorized ``project_smooth_batch``
        (JIT-compiled), and then the polyline is re-built by interleaving
        the original points with the selected midpoints — all in pure
        NumPy, no Python loop over segments.

        This is ~5-10× faster than the previous per-segment
        ``project_to_surface`` approach because it replaces N
        Python↔VTK round-trips per depth with a single batched call.

        Parameters
        ----------
        pts : (N, 3) surface polyline — should already be projected.
        tol : maximum allowed deviation (Euclidean distance between
              chord midpoint and its surface projection).  Defaults to
              ``mean_edge_length * 0.01`` — adaptive to mesh density.
        max_depth : iteration cap.  On each iteration, segments exceeding
              the tolerance are halved; already-refined segments are
              left alone.  6 iterations → up to 64× local refinement.

        Returns
        -------
        (M, 3) refined polyline with M >= N.  Unchanged if no segment
        exceeds the tolerance.
        """
        if len(pts) < 2:
            return pts
        if tol is None:
            mean_edge = float(np.sqrt(self._face_edge_len2.mean()))
            tol = mean_edge * 0.01
        tol_sq = tol * tol

        pts = np.asarray(pts, dtype=float)
        for _ in range(max_depth):
            if len(pts) < 2:
                break
            # Batch: chord midpoints for all segments
            midpoints = (pts[:-1] + pts[1:]) * 0.5
            # Batch project onto surface (single JIT call, no VTK round-trips)
            projected = self.project_smooth_batch(midpoints)
            # Per-segment deviation
            diffs = projected - midpoints
            dists_sq = np.sum(diffs * diffs, axis=1)
            needs_split = dists_sq > tol_sq
            if not needs_split.any():
                break

            # Vectorized interleave: original points + selected midpoints
            n_old = len(pts)
            n_new = int(needs_split.sum())
            out = np.empty((n_old + n_new, 3), dtype=float)
            cumsplit = np.concatenate(
                [[0], np.cumsum(needs_split.astype(np.int64))])
            # Original point i goes to index i + cumsplit[i]
            out[np.arange(n_old) + cumsplit] = pts
            # Midpoint of segment i (if split) goes right after original i
            seg_idx = np.nonzero(needs_split)[0]
            out[seg_idx + cumsplit[:-1][needs_split] + 1] = projected[needs_split]
            pts = out
        return pts

    def _make_work_buffers(self, extra_verts: int = 2, extra_faces: int = 6):
        """Create mutable working copies of V and F as pre-allocated numpy arrays.

        Returns (V_buf, F_buf, n_verts, n_faces) where V_buf/F_buf have room
        for ``extra_verts``/``extra_faces`` beyond the original mesh.  Avoids
        the costly ``V.copy()`` + ``list-based F_work`` round-trip
        (~123 ms → ~1 ms on 240K-face meshes).
        """
        nv, nf = len(self.V), len(self.F)
        V_buf = np.empty((nv + extra_verts, 3), dtype=float)
        V_buf[:nv] = self.V
        F_buf = np.empty((nf + extra_faces, 3), dtype=np.int32)
        F_buf[:nf] = self.F
        return V_buf, F_buf, nv, nf

    def prepare_origin(self, p_origin: F64Array) -> OriginCache:
        """Pre-insert origin into mesh topology, build a solver, and cache both.

        Returns an ``OriginCache`` TypedDict used by
        ``compute_endpoint_from_origin`` (see its field docstrings).

        The V buffer is oversized (+3 slots) so that endpoint insertion can
        write at ``V_buf[nv]`` without copying the entire vertex array.
        F buffer has +10 slots for the same reason but F is copied per-call
        (see ``compute_endpoint_from_origin`` docstring for why).

        If topology insertion produces a degenerate mesh (self-edges from
        nearly-degenerate triangles in the original mesh, or non-manifold
        edges from edge-boundary insertion), the solver construction will
        fail.  In that case, falls back to the pre-built solver with a
        vertex-snapped origin — slightly less exact but functional.
        """
        V_buf, F_buf, nv, nf = self._make_work_buffers(extra_verts=3, extra_faces=10)
        idx_o, nv, nf = self._add_point_buf(p_origin, V_buf, F_buf, nv, nf)
        nf = self._remove_degenerate_faces(F_buf, nf)
        # pp3d's pybind11 wrapper can raise RuntimeError (manifold check
        # fails on degenerate input) or ValueError (bad array dtype/shape).
        # We deliberately do NOT catch broader Exception — KeyboardInterrupt
        # and MemoryError must propagate so the user can interrupt long
        # session loads or surface OOM cleanly.
        try:
            solver = pp3d.EdgeFlipGeodesicSolver(V_buf[:nv], F_buf[:nf])
        except (RuntimeError, ValueError, TypeError) as exc:
            # Topology insertion produced a degenerate mesh.  Fall back to
            # the pre-built solver with vertex-snapped origin.
            log.debug("EdgeFlipGeodesicSolver failed in prepare_origin: %s", exc)
            _, idx_o = self._kdtree.query(p_origin)
            idx_o = int(idx_o)
            solver = self._solver
            V_buf, F_buf = self.V, self.F
            nv, nf = len(self.V), len(self.F)
        return {'V_buf': V_buf, 'F_buf': F_buf, 'nv': nv, 'nf': nf,
                'idx': idx_o, 'p': np.array(p_origin),
                'solver': solver, 'kdtree': self._kdtree}

    def compute_endpoint_from_origin(self, origin_cache: OriginCache,
                                     p_end: F64Array) -> F64Array | None:
        """Geodesic path from a pre-inserted origin to an arbitrary endpoint.

        Two-tier strategy:

          1. **Vertex-snap fast path** (~1 ms): if *p_end* snaps to an
             existing vertex (within 1e-9 barycentric tolerance), reuses
             the cached solver directly — no rebuild at all.
          2. **Local submesh** (~25 ms): delegates to ``compute_endpoint_local``
             which uses sphere pre-filter + bidirectional BFS + local
             solver construction (~10× faster than the global solver).

        Called during handle drag (A/B markers in ``GeodesicSegment``)
        after the debounce fires.  The vertex-snap path is rare in
        practice (user positions rarely coincide exactly with vertices),
        so most calls take the ~25 ms local submesh path.
        """
        try:
            idx_s = origin_cache['idx']
            V_buf = origin_cache['V_buf']
            F_buf = origin_cache['F_buf']
            nv_cached = origin_cache['nv']
            nf_cached = origin_cache['nf']

            # Tier 1: vertex-snap fast path via cached solver
            face_idx = self._find_face_buf(p_end, V_buf, F_buf, nv_cached, nf_cached)
            u, v, w = self._bary_buf(p_end, face_idx, V_buf, F_buf)
            fa, fb, fc = int(F_buf[face_idx, 0]), int(F_buf[face_idx, 1]), int(F_buf[face_idx, 2])
            eps = 1e-9
            snap_idx = None
            if u > 1 - eps: snap_idx = fa
            elif v > 1 - eps: snap_idx = fb
            elif w > 1 - eps: snap_idx = fc

            if snap_idx is not None:
                if idx_s == snap_idx:
                    return np.array([origin_cache['p'], p_end])
                path = origin_cache['solver'].find_geodesic_path(idx_s, snap_idx)
                self.diagnose_path(path, "endpoint-cached-snap")
                return path

            # Tier 2: local submesh solver (~10× faster than global)
            return self.compute_endpoint_local(origin_cache['p'], p_end)
        except (RuntimeError, ValueError, TypeError, IndexError, KeyError) as exc:
            # Tier 1 failure modes:
            #   - RuntimeError / ValueError  → solver rejected the snap.
            #   - IndexError                 → bary helpers got a stale F_buf row.
            #   - KeyError                   → caller passed a malformed cache dict.
            # All recoverable; degrade to Tier 2.  KeyboardInterrupt and
            # MemoryError still propagate.
            log.debug("compute_endpoint_from_origin tier-1 failed: %s", exc)
            return self.compute_endpoint_local(origin_cache['p'], p_end)

    def _try_endpoint_insertion(self, p_start, p_end):
        """Single attempt at topology insertion + solver construction.

        Returns ``(path, success)`` where *path* is the geodesic polyline
        and *success* is True if the solver accepted the modified mesh.
        """
        V_buf, F_buf, nv, nf = self._make_work_buffers(extra_verts=2, extra_faces=6)
        idx_s, nv, nf = self._add_point_buf(p_start, V_buf, F_buf, nv, nf)
        idx_e, nv, nf = self._add_point_buf(p_end,   V_buf, F_buf, nv, nf)
        nf = self._remove_degenerate_faces(F_buf, nf)
        if idx_s == idx_e:
            return np.array([p_start, p_end]), True
        solver = pp3d.EdgeFlipGeodesicSolver(V_buf[:nv], F_buf[:nf])
        path = solver.find_geodesic_path(idx_s, idx_e)
        return path, True

    # --- Local submesh geodesic solver ---

    @staticmethod
    def _extract_submesh(V: np.ndarray, F: np.ndarray,
                         face_indices: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts a submesh from *face_indices*.

        Returns ``(V_sub, F_sub, vmap)`` where *vmap* maps submesh vertex
        indices back to global indices (``global_idx = vmap[local_idx]``).
        """
        F_sub_global = F[face_indices]
        unique_verts, inverse = np.unique(F_sub_global.ravel(), return_inverse=True)
        V_sub = V[unique_verts]
        F_sub = inverse.reshape(-1, 3).astype(np.int32)
        return V_sub, F_sub, unique_verts

    def _faces_for_point(self, p: np.ndarray) -> set[int]:
        """Returns the face(s) that conservatively contain *p*.

        Uses ``find_face`` for the primary face, then adds all faces
        sharing any vertex of that face (1-ring) — covers the case
        where *p* sits exactly on an edge or vertex.
        """
        fi = self.find_face(p)
        result = {fi}
        for vi in self.F[fi]:
            start = self._vf_offsets[vi]
            end = self._vf_offsets[vi + 1]
            for adj_fi in self._vf_data[start:end]:
                result.add(int(adj_fi))
        return result

    def _expand_face_region(self, seed_faces,
                            k_rings: int) -> np.ndarray:
        """BFS expansion of a face seed set by *k_rings* topological rings.

        Plain uni-directional BFS (simpler and faster than the old
        bidirectional variant since the seed is already a dense "tube"
        along the expected geodesic path — the two fronts are already
        connected).  Returns a sorted int32 array of face indices.
        """
        adj = self._face_adj
        visited = {int(f) for f in seed_faces}
        frontier = set(visited)
        for _ in range(k_rings):
            next_f = set()
            for fi in frontier:
                for nb in adj[fi]:
                    nb_i = int(nb)
                    if nb_i >= 0 and nb_i not in visited:
                        visited.add(nb_i)
                        next_f.add(nb_i)
            if not next_f:
                break
            frontier = next_f
        return np.array(sorted(visited), dtype=np.int32)

    def _try_solve_on_region(self, p_start: np.ndarray,
                             p_end: np.ndarray,
                             face_region: np.ndarray):
        """Attempts ``EdgeFlipGeodesicSolver`` on the submesh induced by
        *face_region*.

        Returns one of:
          - ``('ok', path)``          — solver succeeded, no boundary touch.
          - ``('boundary', None)``   — boundary check failed (retry with
                                        bigger region may succeed).
          - ``('error', None)``      — solver exception or degenerate
                                        insertion (retry may or may not help).
          - ``('trivial', path)``    — the two endpoints resolved to the
                                        same inserted vertex; not an error,
                                        caller can use the 2-point stub.
        """
        V_sub, F_sub, vmap = self._extract_submesh(
            self.V, self.F, face_region)
        nv_sub, nf_sub = len(V_sub), len(F_sub)

        # Boundary faces of the submesh: any face with at least one
        # neighbour that is NOT in the submesh.  A geodesic whose endpoint
        # lies on such a face may have been truncated.
        adj = self._face_adj
        region_set = set(face_region.tolist())
        boundary_faces_global: set[int] = set()
        for fi in face_region:
            for nb in adj[fi]:
                if nb < 0 or int(nb) not in region_set:
                    boundary_faces_global.add(int(fi))
                    break

        # Topology-insertion buffers (oversize so _add_point_local can
        # subdivide without reallocation).
        extra = 4
        V_buf = np.empty((nv_sub + extra, 3), dtype=float)
        V_buf[:nv_sub] = V_sub
        F_buf = np.empty((nf_sub + 2 * extra, 3), dtype=np.int32)
        F_buf[:nf_sub] = F_sub

        try:
            _, vi_global_s = self._kdtree.query(p_start)
            _, vi_global_e = self._kdtree.query(p_end)
            vi_global_s = int(vi_global_s)
            vi_global_e = int(vi_global_e)

            def _to_local(vi_global, p):
                pos = int(np.searchsorted(vmap, vi_global))
                if pos < len(vmap) and vmap[pos] == vi_global:
                    return pos
                # Global nearest vertex is outside the submesh
                # (rare — seed was too tight).  Local KDTree for this
                # point only.
                from scipy.spatial import KDTree as _KDTree
                kd = _KDTree(V_sub)
                _, vi_local = kd.query(p)
                return int(vi_local)

            vi_local_s = _to_local(vi_global_s, p_start)
            vi_local_e = _to_local(vi_global_e, p_end)

            nv, nf = nv_sub, nf_sub
            idx_s, nv, nf = self._add_point_local(
                p_start, V_buf, F_buf, nv, nf, vi_local_s, nv_sub)
            idx_e, nv, nf = self._add_point_local(
                p_end, V_buf, F_buf, nv, nf, vi_local_e, nv_sub)

            nf = self._remove_degenerate_faces(F_buf, nf)

            if idx_s == idx_e:
                return ('trivial', np.array([p_start, p_end]))

            solver = pp3d.EdgeFlipGeodesicSolver(V_buf[:nv], F_buf[:nf])
            path = solver.find_geodesic_path(idx_s, idx_e)
        except (RuntimeError, ValueError, TypeError, IndexError) as exc:
            # pp3d / topology-insertion failure on this submesh region.
            # Caller (compute_endpoint_local) treats ('error', None) as a
            # signal to widen the seed and retry.
            log.debug("local submesh solver failed: %s", exc)
            return ('error', None)

        if path is None or len(path) < 2:
            return ('error', None)

        # Boundary check: if any path point falls on a boundary face of
        # the submesh, the solver may have been forced against the edge
        # of the region and the real geodesic goes further out.
        for pt in path:
            fi_global = self.find_face(pt)
            if fi_global in boundary_faces_global:
                return ('boundary', None)

        return ('ok', path)

    def compute_endpoint_local(self, p_start: F64Array,
                               p_end: F64Array,
                               n_line_samples: int = 100,
                               ) -> F64Array | None:
        """Geodesic path using a **projected-line** submesh pre-filter.

        Pre-filter strategy
        ~~~~~~~~~~~~~~~~~~~
        Instead of a spherical / bounding-box filter (which over-includes
        faces on the sides perpendicular to the A→B axis and under-
        includes when the geodesic has to go around a ridge), this method
        uses the **projection of the straight A→B line onto the mesh**
        as the seed for the submesh:

          1. Sample the euclidean segment ``[A, B]`` with *n_line_samples*
             points (default 100).
          2. ``project_smooth_batch_with_faces`` projects each point onto
             its closest triangle — returning the face index per sample.
          3. The set of hit faces forms the initial seed.  It is a narrow
             "tube" that follows the real terrain: on a ridge, the tube
             climbs and descends; in a flat region, it is a straight strip.
          4. Add 1-ring of topological neighbours to bridge any gaps
             between consecutive samples that landed in non-adjacent faces.

        Why this beats the sphere filter:

          * **Ridges / valleys**: a sphere centred on the euclidean
            midpoint cuts through the mountain — the solver then has to
            reach around, often triggering the boundary-check fallback.
            The projected line naturally hugs the surface where the real
            geodesic will want to go, so the ridge faces are in the seed.
          * **Tight tube**: typically captures ~100-300 faces vs
            ~500-2000 for the sphere, so the ``EdgeFlipGeodesicSolver``
            construction is faster.

        Adaptive retry
        ~~~~~~~~~~~~~~
        The submesh is expanded by an increasing number of rings on each
        failure:

              attempt     k_rings     submesh size (approx.)
                 1           3         seed + 3 rings   (tight)
                 2           7         seed + 7 rings
                 3          15
                 4          30
                 5          60         (last local attempt)

        After all 5 attempts fail (boundary touch, solver error, ...) the
        method falls back to ``compute_endpoint`` on the full mesh.  A
        ``'trivial'`` result (two endpoints collapsed to one vertex) is
        returned immediately — it is degenerate but correct.

        Side effects
        ~~~~~~~~~~~~
        Sets ``self._last_was_fallback = True`` if the final returned
        path is a 2-point straight-line stub (so callers can colour the
        span red).
        """
        self._last_was_fallback = False
        p_start = np.asarray(p_start, dtype=float)
        p_end = np.asarray(p_end, dtype=float)

        # --- Build the initial seed from the projected straight line ---
        try:
            line_pts = np.linspace(p_start, p_end, n_line_samples)
            _, seed_face_arr = self.project_smooth_batch_with_faces(line_pts)
            seed_face_arr = seed_face_arr[seed_face_arr >= 0]
            seed_set: set[int] = {int(f) for f in np.unique(seed_face_arr)}

            # Belt-and-suspenders: guarantee the endpoint faces are
            # present even if the projection kernel missed them at
            # t=0 / t=1 (possible on very sliver triangles).
            seed_set.update(self._faces_for_point(p_start))
            seed_set.update(self._faces_for_point(p_end))
        except (ValueError, IndexError, RuntimeError) as exc:
            # Projection kernel or face lookup failed on degenerate seed
            # input.  Bail out to the global solver — slower but robust.
            log.debug("seed construction failed for compute_endpoint_local: %s", exc)
            return self.compute_endpoint(p_start, p_end)

        if not seed_set:
            return self.compute_endpoint(p_start, p_end)

        # --- Adaptive retry loop ---
        for k_rings in (3, 7, 15, 30, 60):
            face_region = self._expand_face_region(seed_set, k_rings)
            status, path = self._try_solve_on_region(
                p_start, p_end, face_region)
            if status == 'ok':
                return path
            if status == 'trivial':
                # Endpoints collapsed to the same inserted vertex —
                # correct result, cannot be improved by retrying.
                return path
            # 'boundary' or 'error' → expand and try again

        # All local attempts exhausted — global solver as last resort.
        return self.compute_endpoint(p_start, p_end)

    def _add_point_local(self, p, V_buf, F_buf, nv, nf,
                         vi_local, nv_original):
        """Insert a point into submesh topology with 1-to-3 subdivision.

        *vi_local* is the submesh-local index of the nearest vertex to *p*
        (precomputed by the caller via the global KDTree + searchsorted).
        Avoids building a local KDTree per call.

        Performs the same nudge and post-subdivision area-check logic as
        the global ``_add_point_buf``.  Returns ``(vertex_idx, nv, nf)``.
        """
        vi = int(vi_local)

        # Candidate faces: all containing vi + any from prior insertions
        candidates = []
        for fi in range(nf):
            f = F_buf[fi]
            if vi in (int(f[0]), int(f[1]), int(f[2])):
                candidates.append(fi)
        for fi in range(nv_original, nf):
            if fi not in candidates:
                candidates.append(fi)
        if not candidates:
            candidates = list(range(nf))

        face_idx = min(candidates,
                       key=lambda i: self._outside_score_buf(p, i, V_buf, F_buf))

        u, v, w = self._bary_buf(p, face_idx, V_buf, F_buf)
        fa = int(F_buf[face_idx, 0])
        fb = int(F_buf[face_idx, 1])
        fc = int(F_buf[face_idx, 2])
        snap_eps = 1e-4

        if u > 1 - snap_eps: return fa, nv, nf
        if v > 1 - snap_eps: return fb, nv, nf
        if w > 1 - snap_eps: return fc, nv, nf

        edge_eps = 1e-3
        if min(u, v, w) < edge_eps:
            Va, Vb, Vc = V_buf[fa], V_buf[fb], V_buf[fc]
            centroid = (Va + Vb + Vc) / 3.0
            e0 = np.linalg.norm(Vb - Va)
            e1 = np.linalg.norm(Vc - Vb)
            e2 = np.linalg.norm(Va - Vc)
            min_edge = min(e0, e1, e2)
            nudge = max(1e-6, min(1e-2, min_edge * 0.01))
            p = p * (1.0 - nudge) + centroid * nudge

        p_idx = nv
        V_buf[p_idx] = p
        nv += 1

        saved_face = F_buf[face_idx].copy()
        saved_nf0 = F_buf[nf].copy()
        saved_nf1 = F_buf[nf + 1].copy()

        F_buf[face_idx] = [p_idx, fa, fb]
        F_buf[nf]       = [p_idx, fb, fc]
        F_buf[nf + 1]   = [p_idx, fc, fa]

        for fi in (face_idx, nf, nf + 1):
            tri = V_buf[F_buf[fi]]
            area = 0.5 * np.linalg.norm(
                np.cross(tri[1] - tri[0], tri[2] - tri[0]))
            if area < 1e-15:
                F_buf[face_idx] = saved_face
                F_buf[nf] = saved_nf0
                F_buf[nf + 1] = saved_nf1
                dists = [np.linalg.norm(p - V_buf[fa]),
                         np.linalg.norm(p - V_buf[fb]),
                         np.linalg.norm(p - V_buf[fc])]
                return [fa, fb, fc][np.argmin(dists)], nv - 1, nf

        return p_idx, nv, nf + 2

    def compute_endpoint(self, p_start: F64Array, p_end: F64Array) -> F64Array | None:
        """Geodesic path between two exact 3D points via buffer-based mesh insertion.

        If the first attempt fails (solver rejects the modified mesh),
        retries with points nudged toward their face centroids.  Only
        falls back to vertex-snap as a last resort.

        Returns a straight-line fallback if the two points lie on
        disconnected mesh components (no geodesic path exists).

        Side effect: ``self._last_was_fallback`` is set to True whenever
        this function returns a 2-point straight-line stub (cross-component
        or solver failure) so callers can flag the span as degraded.
        """
        self._last_was_fallback = False

        # Reject cross-component queries early — no geodesic can exist
        fi_s = self.find_face(p_start)
        fi_e = self.find_face(p_end)
        if not self.same_component(fi_s, fi_e):
            self._last_was_fallback = True
            return np.array([p_start, p_end])

        # Attempt 1: exact positions
        try:
            path, ok = self._try_endpoint_insertion(p_start, p_end)
            if ok:
                self.diagnose_path(path, "endpoint")
                return path
        except (RuntimeError, ValueError, TypeError, IndexError) as exc:
            log.debug("compute_endpoint attempt-1 failed: %s", exc)

        # Attempt 2: nudge both points toward their face centroids.
        # Nudge fraction is relative to the shortest edge of each face
        # — safe on both coarse and very dense meshes.
        try:
            verts_s = self.V[self.F[fi_s]]
            verts_e = self.V[self.F[fi_e]]
            A_s = verts_s.mean(axis=0)
            A_e = verts_e.mean(axis=0)
            min_edge_s = min(np.linalg.norm(verts_s[i] - verts_s[(i + 1) % 3])
                             for i in range(3))
            min_edge_e = min(np.linalg.norm(verts_e[i] - verts_e[(i + 1) % 3])
                             for i in range(3))
            nudge_s = max(1e-6, min(1e-2, min_edge_s * 0.01))
            nudge_e = max(1e-6, min(1e-2, min_edge_e * 0.01))
            p_s2 = p_start * (1.0 - nudge_s) + A_s * nudge_s
            p_e2 = p_end * (1.0 - nudge_e) + A_e * nudge_e
            path, ok = self._try_endpoint_insertion(p_s2, p_e2)
            if ok:
                self.diagnose_path(path, "endpoint-nudged")
                return path
        except (RuntimeError, ValueError, TypeError, IndexError) as exc:
            log.debug("compute_endpoint attempt-2 (nudged) failed: %s", exc)

        # Last resort: snap to nearest vertices and use pre-built solver
        if self.locator is not None:
            log.warning("endpoint insertion failed after retry; falling back to vertex snap")
        _, idx_s = self._kdtree.query(p_start)
        _, idx_e = self._kdtree.query(p_end)
        idx_s, idx_e = int(idx_s), int(idx_e)
        if idx_s == idx_e:
            self._last_was_fallback = True
            return np.array([p_start, p_end])
        try:
            path = self._solver.find_geodesic_path(idx_s, idx_e)
            self.diagnose_path(path, "endpoint-snapped")
            return path
        except (RuntimeError, ValueError, TypeError) as exc:
            log.debug("vertex-snap solver failed: %s", exc)
            self._last_was_fallback = True
            return np.array([p_start, p_end])

    @staticmethod
    def _remove_degenerate_faces(F_buf: np.ndarray, nf: int) -> int:
        """Removes faces with self-edges (repeated vertices) from *F_buf*.

        A face like ``[A, A, B]`` has a self-edge and will be rejected by
        geometry-central's manifold mesh constructor.  These arise from
        topology insertion near degenerate triangles in the original mesh.

        Operates in-place on *F_buf*.  Returns the new *nf*.
        """
        if nf == 0:
            return nf
        F = F_buf[:nf]
        valid = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        n_valid = int(valid.sum())
        if n_valid < nf:
            F_buf[:n_valid] = F[valid]
        return n_valid

    def _add_point_buf(self, p, V_buf, F_buf, nv, nf):
        """Insert a point into mesh topology using pre-allocated buffers.

        Operates on numpy arrays in-place — no list/tuple conversion.
        Returns (vertex_idx, new_nv, new_nf).

        Uses a two-tier threshold:
          - **snap_eps** (``1e-4``): if a barycentric coord is within this
            of 1.0, snap to that vertex.  Prevents edge splits that create
            near-zero-length edges.
          - **edge_eps** (``1e-4``): if a barycentric coord is below this,
            the point is on an edge — use edge-split instead of interior
            subdivision that would create a sliver triangle.

        The previous ``eps=1e-9`` threshold was too conservative — points
        within 1e-5 of an edge were classified as "interior" and the
        resulting 1-to-3 subdivision produced degenerate triangles rejected
        by geometry-central.
        """
        face_idx = self._find_face_buf(p, V_buf, F_buf, nv, nf)
        u, v, w = self._bary_buf(p, face_idx, V_buf, F_buf)
        fa, fb, fc = int(F_buf[face_idx, 0]), int(F_buf[face_idx, 1]), int(F_buf[face_idx, 2])
        snap_eps = 1e-4

        # Case 1: snap to nearest vertex
        if u > 1 - snap_eps: return fa, nv, nf
        if v > 1 - snap_eps: return fb, nv, nf
        if w > 1 - snap_eps: return fc, nv, nf

        # Nudge toward centroid if any bary coord is near zero (on an
        # edge).  Without this, 1-to-3 subdivision creates a sliver
        # triangle with near-zero area that can cause NaNs in the
        # solver's cotan/area computations.  The nudge fraction is
        # relative to the shortest edge of the face — safe on both
        # coarse and very dense meshes.
        edge_eps = 1e-3
        if min(u, v, w) < edge_eps:
            Va, Vb, Vc = V_buf[fa], V_buf[fb], V_buf[fc]
            centroid = (Va + Vb + Vc) / 3.0
            e0 = np.linalg.norm(Vb - Va)
            e1 = np.linalg.norm(Vc - Vb)
            e2 = np.linalg.norm(Va - Vc)
            min_edge = min(e0, e1, e2)
            # Scale nudge: ~1% of shortest edge (clamped to [1e-6, 1e-2])
            nudge = max(1e-6, min(1e-2, min_edge * 0.01))
            p = p * (1.0 - nudge) + centroid * nudge

        p_idx = nv
        V_buf[p_idx] = p
        nv += 1

        # Save slots that will be overwritten so we can undo if needed.
        saved_face = F_buf[face_idx].copy()
        saved_nf0 = F_buf[nf].copy()
        saved_nf1 = F_buf[nf + 1].copy()

        # 1-to-3 subdivision — always manifold by construction.
        F_buf[face_idx] = [p_idx, fa, fb]
        F_buf[nf]       = [p_idx, fb, fc]
        F_buf[nf + 1]   = [p_idx, fc, fa]

        # Post-subdivision area check: verify no degenerate triangle was
        # created.  If any sub-triangle has near-zero area, undo the
        # insertion completely and snap to the nearest original vertex.
        for fi in (face_idx, nf, nf + 1):
            tri = V_buf[F_buf[fi]]
            area = 0.5 * np.linalg.norm(
                np.cross(tri[1] - tri[0], tri[2] - tri[0]))
            if area < 1e-15:
                # Undo: restore all three F_buf slots + vertex count
                F_buf[face_idx] = saved_face
                F_buf[nf] = saved_nf0
                F_buf[nf + 1] = saved_nf1
                dists = [np.linalg.norm(p - V_buf[fa]),
                         np.linalg.norm(p - V_buf[fb]),
                         np.linalg.norm(p - V_buf[fc])]
                return [fa, fb, fc][np.argmin(dists)], nv - 1, nf

        return p_idx, nv, nf + 2

    @staticmethod
    def _split_face_buf(fi, v1, v2, p, F_buf, nf):
        """Splits face *fi* at edge (v1, v2), writes result in-place into *F_buf*.

        Returns updated *nf*.  Extracted as a static method so it is
        created once at class definition time, not per call.
        """
        f = F_buf[fi]
        for i in range(3):
            if {int(f[i]), int(f[(i+1) % 3])} == {int(v1), int(v2)}:
                c = int(f[(i + 2) % 3])
                F_buf[fi]  = [int(f[i]), p, c]
                F_buf[nf]  = [p, int(f[(i+1) % 3]), c]
                return nf + 1
        return nf

    @staticmethod
    def _find_reverse_halfedge(F_buf, nf, iv1, iv2, exclude_fi):
        """Finds the face containing the reverse half-edge (iv2→iv1).

        In a manifold mesh, each directed half-edge (A→B) has exactly
        one partner (B→A) in an adjacent face.  Searching for the
        REVERSE half-edge instead of just "contains both vertices"
        avoids false matches with sub-faces created by prior
        subdivisions that share the same vertex pair but different
        winding.
        """
        for fi in range(nf):
            if fi == exclude_fi:
                continue
            f = F_buf[fi]
            f0, f1, f2 = int(f[0]), int(f[1]), int(f[2])
            # Check all 3 directed half-edges for (iv2→iv1)
            if (f0 == iv2 and f1 == iv1) or \
               (f1 == iv2 and f2 == iv1) or \
               (f2 == iv2 and f0 == iv1):
                return fi
        return None

    def _split_edge_buf(self, p_idx, v1, v2, main_idx, F_buf, nf):
        """Edge split on pre-allocated F_buf. Returns (p_idx, nv_unchanged, new_nf).

        Finds the adjacent face by searching for the **reverse half-edge**
        (v2→v1) rather than just vertex membership.  This is critical for
        correctness after prior subdivisions: multiple sub-faces may
        contain both v1 and v2, but only ONE has the reverse half-edge.
        Using set-based lookup ``{v1, v2} in face`` would find the wrong
        face and produce non-manifold duplicate edges.
        """
        iv1, iv2 = int(v1), int(v2)
        key = (min(iv1, iv2), max(iv1, iv2))
        candidates = self._edge_to_face.get(key, [])
        adj_idx = next((fi for fi in candidates if fi != main_idx), None)

        # Validate: the adjacent face must contain the REVERSE half-edge.
        if adj_idx is not None:
            f = F_buf[adj_idx]
            f0, f1, f2 = int(f[0]), int(f[1]), int(f[2])
            has_reverse = ((f0 == iv2 and f1 == iv1) or
                           (f1 == iv2 and f2 == iv1) or
                           (f2 == iv2 and f0 == iv1))
            if not has_reverse:
                adj_idx = None

        # Fallback: scan for the face with the reverse half-edge.
        if adj_idx is None:
            adj_idx = self._find_reverse_halfedge(
                F_buf, nf, iv1, iv2, main_idx)

        nf = self._split_face_buf(main_idx, v1, v2, p_idx, F_buf, nf)
        if adj_idx is not None:
            nf = self._split_face_buf(adj_idx, v1, v2, p_idx, F_buf, nf)

        # nv is not changed here (already incremented by caller)
        return p_idx, p_idx + 1, nf

    # --- Helpers for buffer-based insertion ---
    def _find_face_buf(self, p, V_buf, F_buf, nv, nf):
        """Locate face containing *p* in the buffer topology.

        Uses the original-mesh KDTree to seed the candidate set with faces
        adjacent to the nearest original vertex, then **unconditionally**
        adds all faces created by prior insertions (indices ≥ n_original).
        Prior code filtered new faces by whether they contained ``vi``,
        missing faces that used the newly inserted origin vertex.

        Also includes original faces that were *modified* in-place by a
        prior subdivision (their index is < n_original but their vertices
        may have changed).  Since modified faces are already in the CSR
        candidate list by index, they are covered — but their new vertex
        set might not include ``vi`` anymore.  Adding all new-range faces
        ensures coverage regardless.
        """
        _, vi = self._kdtree.query(p)
        vi = int(vi)
        n_original = len(self.F)

        nv_orig = len(self._vf_offsets) - 1
        if vi < nv_orig:
            start, end = self._vf_offsets[vi], self._vf_offsets[vi + 1]
            candidates = [int(i) for i in self._vf_data[start:end] if i < nf]
        else:
            candidates = []

        # Include ALL faces created by prior insertions — not filtered by
        # vi.  After origin insertion there are at most ~4 new faces, so
        # the extra scoring cost is negligible.
        for i in range(n_original, nf):
            candidates.append(i)

        if not candidates:
            return 0
        return min(candidates, key=lambda i: self._outside_score_buf(p, i, V_buf, F_buf))

    def _outside_score_buf(self, p, i, V_buf, F_buf):
        u, v, w = self._bary_buf(p, i, V_buf, F_buf)
        return max(0.0, -u) + max(0.0, -v) + max(0.0, -w)

    def _bary_buf(self, p, fi, V_buf, F_buf):
        f = F_buf[fi]
        return self._barycentric(p, V_buf[int(f[0])], V_buf[int(f[1])], V_buf[int(f[2])])

    # ------------------------------------------------------------------
    # Geodesic spline helpers
    # ------------------------------------------------------------------

    def project_to_surface(self, pt: F64Array) -> F64Array:
        """Project a single 3D point onto the nearest triangle surface."""
        if self.locator is not None:
            self.locator.FindClosestPoint(
                pt, self._vtk_cp, self._vtk_cell_id, self._vtk_sub_id, self._vtk_dist2)
            return np.array(self._vtk_cp, dtype=float)
        _, idx = self._kdtree.query(pt)
        return self.V[int(idx)].copy()

    @staticmethod
    def compute_path_lengths(path: F64Array) -> tuple[F64Array, float]:
        """Pre-compute cumulative segment lengths for a geodesic polyline.

        Returns ``(cum_lengths, total)`` where *cum_lengths* is a 1-D array
        of cumulative arc-lengths (one per segment, length N-1 for N points)
        and *total* is the full polyline length.

        Pass these to ``geodesic_lerp`` or ``geodesic_lerp_batch`` to avoid
        redundant recomputation when interpolating the same path at multiple
        *t* values.  The result is invalidated when the path changes —
        callers must recompute after any path modification.
        """
        diffs = path[1:] - path[:-1]
        seg_lens = np.sqrt(np.sum(diffs * diffs, axis=1))
        cum = np.cumsum(seg_lens)
        return cum, float(cum[-1]) if len(cum) > 0 else 0.0

    @staticmethod
    def geodesic_lerp(path: F64Array, t: float,
                      _cum: F64Array = None,
                      _total: float = None) -> F64Array:
        """Interpolate along a precomputed geodesic polyline at parameter *t* in [0,1].

        Walks the polyline by arc-length, Euclidean lerp on the final
        sub-segment.  Exact on the discrete surface because geodesics on
        triangle meshes are piecewise-linear.

        Parameters
        ----------
        _cum, _total : optional
            Pre-computed cumulative lengths from ``compute_path_lengths``.
            When provided, skips the per-call length computation — essential
            when interpolating the same path at many *t* values (e.g. inside
            ``hybrid_de_casteljau_curve``).
        """
        if path is None or len(path) < 2:
            return path[0].copy() if path is not None and len(path) else np.zeros(3)
        if t <= 0.0:
            return path[0].copy()
        if t >= 1.0:
            return path[-1].copy()

        if _cum is not None and _total is not None:
            cum, total = _cum, _total
        else:
            diffs = path[1:] - path[:-1]
            seg_lens = np.sqrt(np.sum(diffs * diffs, axis=1))
            cum = np.cumsum(seg_lens)
            total = float(cum[-1]) if len(cum) > 0 else 0.0

        if total < 1e-15:
            return path[0].copy()

        target = t * total
        idx = int(np.searchsorted(cum, target))
        if idx >= len(cum):
            return path[-1].copy()
        prev_cum = float(cum[idx - 1]) if idx > 0 else 0.0
        sl = float(cum[idx]) - prev_cum
        frac = (target - prev_cum) / sl if sl > 1e-15 else 0.0
        return path[idx] * (1.0 - frac) + path[idx + 1] * frac

    @staticmethod
    def geodesic_lerp_batch(path: F64Array, t_vals: F64Array,
                            cum: F64Array, total: float) -> F64Array:
        """Vectorized interpolation along a geodesic polyline at multiple *t* values.

        Equivalent to calling ``geodesic_lerp`` for each *t*, but finds all
        target segments in one ``np.searchsorted`` pass and performs the
        final lerp as a single vectorized operation.

        Parameters
        ----------
        path : (N, 3)  polyline points.
        t_vals : (M,)  parameter values in [0, 1].
        cum : cumulative segment lengths from ``compute_path_lengths``.
        total : total path length from ``compute_path_lengths``.

        Returns
        -------
        (M, 3) interpolated points.
        """
        n = len(t_vals)
        if total < 1e-15:
            out = np.empty((n, 3), dtype=float)
            out[:] = path[0]
            return out

        targets = np.clip(t_vals * total, 0.0, total)
        indices = np.searchsorted(cum, targets)
        indices = np.clip(indices, 0, len(cum) - 1)

        prev_cum = np.where(indices > 0, cum[indices - 1], 0.0)
        sl = cum[indices] - prev_cum
        # ``np.where(cond, a/b, 0.0)`` would still evaluate ``a/b`` on
        # the masked elements (numpy is eager) and emit a "divide by
        # zero" RuntimeWarning for any zero-length segment that happens
        # to fall on a duplicate point in the polyline.  ``np.divide``
        # with ``where=`` actually skips the division on those indices,
        # so the warning never fires and the result for those samples
        # comes from the pre-zeroed output buffer.
        frac = np.zeros_like(sl)
        np.divide(targets - prev_cum, sl, out=frac, where=sl > 1e-15)

        frac_col = frac[:, np.newaxis]
        return path[indices] * (1.0 - frac_col) + path[indices + 1] * frac_col

    def adaptive_samples(self, ctrl_pts, resolution: float,
                         min_n: int, max_n: int) -> int:
        """Determine sample count for a Bézier span from control-polygon length.

        Uses inlined scalar math to avoid ``np.linalg.norm`` / ``np.asarray``
        overhead for the 3-4 control-point segments.
        """
        from math import sqrt
        poly_len = 0.0
        for i in range(len(ctrl_pts) - 1):
            a, b = ctrl_pts[i], ctrl_pts[i + 1]
            if a is not None and b is not None:
                dx = float(b[0]) - float(a[0])
                dy = float(b[1]) - float(a[1])
                dz = float(b[2]) - float(a[2])
                poly_len += sqrt(dx * dx + dy * dy + dz * dz)
        if poly_len < 1e-12:
            return min_n
        n = int(poly_len / resolution) + 1
        return max(min_n, min(max_n, n))

    @staticmethod
    def curvature_adaptive_t_vals(ctrl, n: int) -> F64Array:
        """Generate non-uniform parameter values concentrated at high-curvature regions.

        Analyses the two interior angles of the cubic Bézier control polygon
        ``[P0, H_out, H_in, P1]``.  Sharp angles predict high curvature at
        approximately t≈1/3 (angle at H_out) and t≈2/3 (angle at H_in).

        The density is modelled as::

            ρ(t) = 1 + k₁·G(t, 1/3, σ) + k₂·G(t, 2/3, σ)

        where G is a Gaussian bump and k₁, k₂ are proportional to the
        turning angles.  The CDF is inverted numerically to produce *n*
        sample values in [0, 1].

        Falls back to ``np.linspace(0, 1, n)`` when both angles are near
        zero (straight control polygon).
        """
        if n < 3:
            return np.linspace(0.0, 1.0, max(n, 1))

        P0, H_out, H_in, P1 = [np.asarray(p, dtype=float) for p in ctrl]

        # Compute turning angles at H_out and H_in
        d01 = H_out - P0
        d12 = H_in - H_out
        d23 = P1 - H_in
        n01 = np.linalg.norm(d01)
        n12 = np.linalg.norm(d12)
        n23 = np.linalg.norm(d23)

        theta1 = 0.0
        if n01 > 1e-12 and n12 > 1e-12:
            cos1 = np.dot(d01, d12) / (n01 * n12)
            cos1 = np.clip(cos1, -1.0, 1.0)
            theta1 = np.arccos(cos1)  # 0 = straight, π = reversal

        theta2 = 0.0
        if n12 > 1e-12 and n23 > 1e-12:
            cos2 = np.dot(d12, d23) / (n12 * n23)
            cos2 = np.clip(cos2, -1.0, 1.0)
            theta2 = np.arccos(cos2)

        # If both angles are small (< ~5°), uniform is fine
        if theta1 + theta2 < 0.09:
            return np.linspace(0.0, 1.0, n)

        # Gaussian bump parameters
        sigma = 0.18
        inv_2s2 = 1.0 / (2.0 * sigma * sigma)
        # k proportional to turning angle (0 at straight, ~π at reversal)
        k1 = theta1 * 2.0
        k2 = theta2 * 2.0

        # Build density on a fine uniform grid and invert the CDF
        m = max(n * 8, 256)
        t_fine = np.linspace(0.0, 1.0, m)
        g1 = np.exp(-((t_fine - 1.0 / 3.0) ** 2) * inv_2s2)
        g2 = np.exp(-((t_fine - 2.0 / 3.0) ** 2) * inv_2s2)
        rho = 1.0 + k1 * g1 + k2 * g2

        # Cumulative distribution (trapezoidal integration)
        cdf = np.empty(m, dtype=float)
        cdf[0] = 0.0
        dt = 1.0 / (m - 1)
        np.cumsum(0.5 * (rho[:-1] + rho[1:]) * dt, out=cdf[1:])
        if cdf[-1] <= 0:
            return np.linspace(0.0, 1.0, n)
        cdf /= cdf[-1]  # normalize to [0, 1]

        # Invert CDF: uniform quantiles → non-uniform t values
        quantiles = np.linspace(0.0, 1.0, n)
        t_vals = np.interp(quantiles, cdf, t_fine)
        # Force exact endpoints
        t_vals[0] = 0.0
        t_vals[-1] = 1.0
        return t_vals

    @staticmethod
    def refine_t_vals_by_curvature(curve_pts: F64Array,
                                   t_vals: F64Array,
                                   max_angle: float = 0.15) -> F64Array:
        """Phase-2 refinement: insert midpoints where the polyline bends sharply.

        Measures the turning angle between consecutive chord segments of
        *curve_pts*.  Where ``angle > max_angle`` (radians, ~8.6°), the
        parametric midpoint of that interval is inserted.

        Parameters
        ----------
        curve_pts : (N, 3) evaluated curve points (from Phase 1).
        t_vals : (N,) parameter values used to produce *curve_pts*.
        max_angle : threshold in radians above which a midpoint is inserted.

        Returns
        -------
        Sorted array of t values with extra samples inserted.  Returns
        *t_vals* unchanged if no refinement is needed.
        """
        if len(curve_pts) < 3:
            return t_vals

        # Chord vectors and their lengths
        d = np.diff(curve_pts, axis=0)                   # (N-1, 3)
        lens = np.linalg.norm(d, axis=1, keepdims=True)  # (N-1, 1)
        lens = np.maximum(lens, 1e-15)
        d_hat = d / lens                                  # unit vectors

        # Turning angle between consecutive chords
        dots = np.sum(d_hat[:-1] * d_hat[1:], axis=1)    # (N-2,)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)                          # (N-2,)

        # Find segments where angle exceeds threshold
        sharp = np.nonzero(angles > max_angle)[0]         # indices into [0..N-3]
        if len(sharp) == 0:
            return t_vals

        # Insert midpoint of the t-interval for each sharp bend.
        # The sharp angle at index j is between chords (j, j+1) and (j+1, j+2),
        # so we bisect intervals [t[j], t[j+1]] and [t[j+1], t[j+2]].
        new_t = set()
        for j in sharp:
            new_t.add(0.5 * (t_vals[j] + t_vals[j + 1]))
            new_t.add(0.5 * (t_vals[j + 1] + t_vals[j + 2]))

        if not new_t:
            return t_vals

        merged = np.union1d(t_vals, np.array(sorted(new_t)))
        merged[0] = 0.0
        merged[-1] = 1.0
        return merged

    def hybrid_de_casteljau_curve(self, ctrl, path_out: F64Array | None,
                                  path_in: F64Array | None,
                                  n_samples: int, fast: bool = False,
                                  t_vals: F64Array | None = None,
                                  path_12: F64Array | None = None) -> F64Array:
        """Evaluate a hybrid geodesic/Euclidean cubic Bézier curve on the surface.

        ctrl : [P0, H_out, H_in, P1]
        path_out : geodesic polyline P0 -> H_out  (node0.path_b)
        path_in  : geodesic polyline P1 -> H_in   (node1.path_a)
        path_12  : optional geodesic polyline H_out -> H_in (from
                   ``compute_endpoint_local``).  When provided, level-1
                   uses geodesic_lerp on ALL three segments
                   (semi-geodesic Bézier).  When None, uses Euclidean
                   lerp + projection for the middle segment (plain
                   hybrid Bézier).

        When *t_vals* is provided, those parameter values are used directly
        (ignoring *n_samples*).  Otherwise falls back to uniform
        ``linspace(0, 1, n_samples)``.

        At de Casteljau level 1:
          - P0->H_out  : geodesic_lerp along path_out  (exact on surface)
          - H_in->P1   : geodesic_lerp along reversed path_in (exact)
          - H_out->H_in: geodesic_lerp along path_12 (if provided) OR
                         Euclidean lerp + surface projection
        Levels 2-3: Euclidean lerp + surface projection.

        Performance
        ~~~~~~~~~~~
        All three de Casteljau levels are fully vectorized across samples
        (no per-sample Python loop).  Geodesic lerps use
        ``geodesic_lerp_batch`` with pre-computed cumulative lengths.
        Surface projections are batched per level — one
        ``project_smooth_batch`` call per level instead of per sample,
        reducing Python↔VTK overhead from 4N calls to 4 batch calls.
        """
        P0, H_out, H_in, P1 = [np.asarray(p, dtype=float) for p in ctrl]
        path_in_rev = path_in[::-1] if path_in is not None and len(path_in) > 1 else None
        do_proj = not fast

        # Pre-compute cumulative lengths (once per path, reused for all t)
        has_path_out = path_out is not None and len(path_out) > 1
        cum_out = total_out = None
        if has_path_out:
            cum_out, total_out = self.compute_path_lengths(path_out)

        has_path_in = path_in_rev is not None and len(path_in_rev) > 1
        cum_in = total_in = None
        if has_path_in:
            cum_in, total_in = self.compute_path_lengths(path_in_rev)

        has_path_12 = path_12 is not None and len(path_12) > 1
        cum_12 = total_12 = None
        if has_path_12:
            cum_12, total_12 = self.compute_path_lengths(path_12)

        # Parameter values — shared across all levels
        if t_vals is None:
            t_vals = np.linspace(0.0, 1.0, n_samples) if n_samples > 1 else np.array([0.0])
        t_col = t_vals[:, np.newaxis]       # (n, 1) for broadcasting
        one_minus_t = 1.0 - t_col

        # --- Level 1: 4 control points → 3 ---
        if has_path_out:
            b01 = self.geodesic_lerp_batch(path_out, t_vals, cum_out, total_out)
        else:
            b01 = P0 * one_minus_t + H_out * t_col

        if has_path_12:
            b12 = self.geodesic_lerp_batch(path_12, t_vals, cum_12, total_12)
        else:
            b12 = H_out * one_minus_t + H_in * t_col

        if has_path_in:
            b23 = self.geodesic_lerp_batch(path_in_rev, t_vals, cum_in, total_in)
        else:
            b23 = H_in * one_minus_t + P1 * t_col

        if do_proj and not has_path_12:
            # Only project b12 when computed via Euclidean lerp;
            # the geodesic path_12 is already on the surface.
            b12 = self.project_smooth_batch(b12)

        # --- Level 2: 3 → 2 (vectorized) ---
        c0 = b01 * one_minus_t + b12 * t_col
        c1 = b12 * one_minus_t + b23 * t_col

        if do_proj:
            c0 = self.project_smooth_batch(c0)
            c1 = self.project_smooth_batch(c1)

        # --- Level 3: 2 → 1 (vectorized) ---
        out = c0 * one_minus_t + c1 * t_col

        if do_proj:
            out = self.project_smooth_batch(out)

        return out

