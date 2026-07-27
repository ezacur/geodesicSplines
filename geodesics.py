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
"Numba JIT Kernels" table in ``docs/ARCHITECTURE.md`` lists all eight
with measured speedups.

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

  * **Origin buffers** (``prepare_origin``): the origin is inserted *once*
    into oversized ``V_buf`` / ``F_buf`` copies (+3 verts / +10 faces), and
    both are cached in the ``OriginCache`` alongside the pre-built solver.
    ``compute_endpoint_from_origin`` then treats these buffers as read-only:
    Tier 1 (vertex-snap) only *reads* them (``_find_face_buf`` / ``_bary_buf``)
    to test whether the endpoint coincides with an existing vertex, and Tier 2
    delegates to ``compute_endpoint_local`` (a fresh local submesh + solver).
    No endpoint call writes into the cached buffers, so no per-call F copy is
    needed here — the earlier "frontier overwrite" / "per-call F copy" scheme
    was retired when the endpoint path moved to the local-submesh solver.
    HAZARD: if the origin insertion produces a degenerate mesh, the
    ``prepare_origin`` fallback sets ``V_buf, F_buf = self.V, self.F`` —
    i.e. the cache then *aliases* the live mesh arrays rather than owning
    private copies.  Tier 1 only reads them, so this is currently safe, but
    any future code that mutates ``origin_cache['V_buf']`` / ``['F_buf']``
    would corrupt the live mesh on the fallback path.
  * **Robust face lookup** (``_find_face_buf``): unconditionally includes all
    faces created by prior insertions, not just those adjacent to the nearest
    original vertex.  Handles the case where newly inserted vertices are
    invisible to the original-mesh KDTree.
  * **Near-edge nudge** (``_add_point_buf``): when a point's barycentric
    coordinates place it very close to an edge (min coord < 1e-7), the
    point is shifted ~1% of the shortest edge toward the face centroid
    before subdivision.
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
  2. ``_smooth_face_normals`` — Laplacian-smoothed face normals.  Two
     weighting strategies are available, selected by the class variable
     ``COTANGENT_WEIGHTS`` (default cotangent at 2 iterations; the
     uniform variant runs 5):

       - **Uniform** (``COTANGENT_WEIGHTS = False``): each
         neighbor has equal weight.  Fast; assumes roughly equilateral
         triangles.
       - **Cotangent** (default, ``COTANGENT_WEIGHTS = True``): classical
         Pinkall-Polthier weights — for each shared edge, the dual-edge
         weight is ``½ · (cot α + cot β)`` where α and β are the
         angles **opposite** to the shared edge in each triangle.
         This is the canonical discrete Laplace-Beltrami discretisation:
         it depends only on intrinsic triangle geometry, so the
         smoother is genuinely invariant to triangulation quality.
         Better for photogrammetry / scanned meshes with long thin
         triangles where uniform weights bias the smoothed normals
         toward densely-tessellated regions.

  3. ``_vertex_normals`` — angle-weighted averages of *smooth* face
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
  * ``_smooth_face_normals_laplacian`` builds its sparse adjacency directly
    from the vectorized ``_face_adj`` matrix (no dict iteration) instead of
    re-scanning all faces — ~50% faster.
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
    (``COTANGENT_WEIGHTS = True``).  On by default.
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
import potpourri3d as pp3d
import vtk
from scipy.spatial import KDTree

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
    from numba import njit, prange
    HAS_NUMBA: bool = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """Transparent no-op when Numba is unavailable."""
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

    # ``prange`` falls back to plain ``range`` when Numba is absent —
    # the @njit shim above already neutralises ``parallel=True`` so the
    # kernel runs serially in pure Python with identical semantics.
    prange = range


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

      - **det_tol** (``1e-10 * edge_len``): the determinant
        ``(d × edge) · n`` is near-zero when the ray is almost parallel
        to the edge.  With unit ``d`` / ``n`` the determinant scales
        *linearly* with edge length (``|det| ≈ edge_len · |sin θ|``), so
        the tolerance is scaled by ``edge_len`` (``sqrt(edge_len²)``) to
        make the angular reject threshold invariant to mesh scale.  The
        comparison is ``<=`` so a zero-length edge (degenerate face with
        two coincident vertex positions) rejects cleanly instead of
        reaching the ``1.0 / det`` division below.
      - **s_tol** (``1e-4``): edge parametric bounds ``s ∈ [-s_tol, 1+s_tol]``
        accept intersections slightly outside the edge due to float
        rounding.  The hit point is clamped to ``[0, 1]``.
      - **t_min** (``-1e-8``): ``t >= t_min`` (with ``t_min = -1e-8``)
        instead of ``t > 0`` avoids rejecting intersections at the
        current position (common after an edge-to-edge advance of 1e-7).

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
        # Parallel-edge reject.  With unit *d* / *n*, |det| ≈ L·|sin θ|
        # (L = edge length), so a scale-invariant angular threshold needs
        # a tolerance linear in L — hence sqrt(fedge_len2), not
        # fedge_len2 itself (the old quadratic form made the effective
        # angular threshold scale with L, breaking the invariance the
        # docstring claims).  The scalar sqrt is a single hardware op and
        # runs at most 3×/crossing.  ``<=`` (not ``<``) makes a
        # zero-length edge — where fedge_len2 and det are both 0 — reject
        # via ``0.0 <= 0.0`` instead of dividing by zero at inv_det below
        # (undefined under fastmath).
        if abs(det) <= 1e-10 * _math_sqrt(fedge_len2[fid, i]):
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

    Returns ``(path_n, exit_code)``: *path_n* is the number of points
    written to *path_buf* (≤ 1 means the shoot failed — no valid path
    produced).  *exit_code* is 0 on a normal exit and 1 when the loop
    was terminated by the consecutive-fallback safeguard described
    below — a signal that the local mesh is not 2-manifold around the
    truncation point and the rendered geodesic ends short of the
    requested length.

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
    # Hoisted out of the per-step loop: candidate-direction scratch used
    # only by the Phase 2b vertex/edge fallback (~1% of iterations) but
    # previously re-allocated every iteration regardless.
    cand_d = np.empty(3)
    # Anti-bounce: count consecutive Phase 2b fallbacks.  On non-2-
    # manifold meshes (>2 faces sharing a vertex with mis-oriented
    # normals) the fallback can keep selecting candidates whose
    # parallel-transport leaves ``curr_p`` clustered around the same
    # vertex, so the next ``_ray_edge_jit`` fails again and Phase 2b
    # fires again — an infinite loop bounded only by ``max_steps`` and
    # the user's patience.  Breaking out once the streak EXCEEDS 2
    # (i.e. on the 3rd consecutive fallback) ends the geodesic at the
    # last good edge crossing rather than spending the whole step
    # budget thrashing.
    fallback_streak = 0
    exit_code = 0  # 0=normal, 1=non-manifold-fan truncation

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
                fallback_streak += 1
                # >2 consecutive fallbacks => degenerate vertex fan;
                # quit at the last good edge crossing rather than
                # re-entering the loop indefinitely.  Caller surfaces
                # this via the HUD so the user knows the mesh is
                # non-manifold here and the truncation is not a
                # software bug.
                if fallback_streak > 2:
                    exit_code = 1
                    break
        else:
            # Phase 2 succeeded — reset the streak.
            fallback_streak = 0

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

    return path_n, exit_code


@njit(cache=True, fastmath=True, parallel=True)
def _project_batch_kernel(pts, nearest_verts, vf_data, vf_off,
                          fverts, fnormals, out, out_faces):
    """Inner projection loop for ``project_smooth_batch``.

    Analytical face-plane projection + barycentric clamping for each
    point against all faces adjacent to its k nearest vertices.
    KDTree query is done in the Python wrapper before calling this kernel.

    Parallelism
    ~~~~~~~~~~~
    Each iteration of the outer loop writes only ``out[i]`` and
    ``out_faces[i]`` — independent across ``i`` — so the loop is
    parallelised via ``prange``.  Yields ~4-8× on multicore for large
    de Casteljau batches.  When Numba is unavailable, ``prange``
    degrades to ``range`` and the kernel runs serially with identical
    semantics.

    *out_faces* (``int32[N]``) receives the index of the face each point
    projected onto (``-1`` if no valid face was found).  Callers that
    don't need the face indices can pass a 1-element dummy buffer.
    """
    n = pts.shape[0]
    for i in prange(n):
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
    # Safety: splines are saved as 3D positions (origin + p_a/p_b handle
    # endpoints), never as vertex indices, so reordering does not break
    # JSON save/load.  The
    # flag is there purely for A/B benchmarking — leave it ON by default,
    # it is essentially free on small meshes and real on large ones.
    MORTON_REORDER = True

    # When True (default), ``_dijkstra_corridor`` uses a single-pair A*
    # with an admissible centroid-distance heuristic instead of scipy's
    # full single-source Dijkstra over the whole face dual graph.  A*
    # explores only the corridor between the two endpoints, so the
    # corridor step is ~8-130x faster (scales with mesh size; the win
    # grows on dense meshes where the full SSSP sweep is most wasteful).
    # The result is cost-optimal by construction (admissible heuristic,
    # same dual graph + centroid-distance weights as scipy); the *path*
    # at exact ties could differ from scipy's backtrace, but since the
    # corridor is only a seed (BFS-expanded 3 rings + unioned), that was
    # bit-for-bit exact (maxdiff = 0) across all 33 real sessions tested
    # (32k-245k faces).  Set False to fall back to scipy's full SSSP.
    USE_ASTAR_CORRIDOR = True

    def __init__(self, V: np.ndarray | object, F: np.ndarray | None = None,
                 *, build_locator: bool = True):
        """Build a GeodesicMesh from raw arrays or a ``pv.PolyData``.

        ``build_locator`` (default True) controls whether the
        ``vtkStaticCellLocator`` is constructed.  Background workers
        (orange / blue export) only call ``compute_endpoint_local``
        which uses the KDTree — they pass ``build_locator=False`` to
        skip the ~250 ms locator construction *per worker process*
        (~1 s saved across 4 workers on a 240 K-face mesh).
        """
        self._build_locator_enabled = build_locator
        if hasattr(V, 'points') and hasattr(V, 'faces'):
            self._pv_mesh = V
            self.V = np.asarray(V.points, dtype=float)
            f = np.asarray(V.faces, dtype=int).reshape(-1, 4); self.F = f[:, 1:]
        else:
            self._pv_mesh = None
            self.V = np.asarray(V, dtype=float); self.F = np.asarray(F, dtype=int)

        # Reject empty / degenerate meshes up front with a clear
        # message, instead of letting them crash deep inside the
        # downstream constructors with cryptic errors:
        #   - ``KDTree(V)`` raises ``ValueError`` on empty V
        #   - ``np.cross`` returns shape (0, 3) on empty F, which then
        #     propagates into ``_compute_face_normals`` and the JIT
        #     kernels with hard-to-debug array-shape mismatches
        #   - ``find_face`` falls through to the KDTree fallback and
        #     returns garbage indices on nearly-empty meshes
        if self.V.ndim != 2 or self.V.shape[1] != 3 or len(self.V) < 3:
            raise ValueError(
                f"GeodesicMesh requires V of shape (N>=3, 3); got {self.V.shape}")
        if self.F.ndim != 2 or self.F.shape[1] != 3 or len(self.F) < 1:
            raise ValueError(
                f"GeodesicMesh requires F of shape (M>=1, 3); got {self.F.shape}")
        if int(self.F.max()) >= len(self.V):
            raise ValueError(
                f"GeodesicMesh: F indexes vertex {int(self.F.max())} "
                f"but V has only {len(self.V)} vertices")

        # Topology sanitisation for ``pp3d.EdgeFlipGeodesicSolver``.
        # geometry-central's manifold check (``GC_SAFETY_ASSERT``) raises
        # on duplicate edges, which crop up in real-world meshes
        # (anatomical scans, CAD pieces merged from multiple sources)
        # via duplicate faces, non-2-manifold edge fans, or self-edge
        # triangles.  Detection passes are O(F log F) and short-circuit
        # when the mesh is already clean, so the cost on a valid input
        # is one ``np.unique`` over the face triples.  When defects
        # *are* found the offending faces are dropped and the cleanup
        # is logged so the user knows what was changed.  Sanitisation
        # runs **before** Morton reorder + every derived structure so
        # the rest of ``__init__`` only ever sees a clean topology.
        self.V, self.F, sanitize_report = self._sanitize_for_solver(
            self.V, self.F)
        _changed = (sanitize_report['total_faces_dropped'] > 0
                    or sanitize_report['vertex_splits'] > 0)
        if _changed:
            log.warning(
                "mesh sanitised for pp3d: dropped %d faces "
                "(%d duplicate, %d non-manifold, %d inconsistent-winding, "
                "%d self-edge), split %d non-manifold vertices, "
                "freed %d unreferenced vertices",
                sanitize_report['total_faces_dropped'],
                sanitize_report['duplicate_faces'],
                sanitize_report['non_manifold_faces'],
                sanitize_report['winding_faces'],
                sanitize_report['self_edge_faces'],
                sanitize_report['vertex_splits'],
                sanitize_report['unreferenced_verts'])
            # Re-run the basic shape guard: an extreme defect (every
            # face dropped) leaves an unusable mesh and we want a
            # clear error rather than a confusing crash deeper down.
            if len(self.F) < 1 or len(self.V) < 3:
                raise ValueError(
                    "GeodesicMesh: topology sanitisation left an empty "
                    "mesh; the input may be entirely non-manifold or "
                    "self-overlapping.")

        # One-shot Morton reorder BEFORE any downstream structure is built.
        # All later arrays (_face_*, _vf_*, KDTree, solver, VTK locator)
        # naturally inherit the improved spatial locality.  See the
        # ``MORTON_REORDER`` class-level docstring for the rationale.
        if self.MORTON_REORDER:
            self._morton_reorder_inplace()

        self._kdtree         = KDTree(self.V)
        self._face_normals   = self._compute_face_normals()
        # Pre-computed face geometry (avoids double-indexing V[F[i]] in hot loops)
        self._face_verts = self.V[self.F]                        # (N_faces, 3, 3)
        self._face_edges = np.roll(self._face_verts, -1, axis=1) - self._face_verts
        # Pre-computed squared edge lengths — avoids 3 muls + 2 adds per
        # edge in the ray-edge intersection inner loop.
        self._face_edge_len2 = np.sum(self._face_edges ** 2, axis=2)  # (N_faces, 3)
        self._face_centroids = self._face_verts.mean(axis=1)        # (N_faces, 3)

        # Static face adjacency matrix — built before smooth normals so
        # _smooth_face_normals_laplacian can use the vectorized adjacency.
        # ``_face_adj_edge[fi, e]`` mirrors ``_face_adj[fi, e]`` and
        # records which edge index of the neighbour matches edge ``e``
        # of face ``fi`` — needed by the cotangent-Laplacian smoother
        # to find the angle "opposite" to the shared edge inside the
        # neighbour triangle.
        self._face_adj, self._face_adj_edge = self._build_face_adjacency_matrix()
        self._face_components = self._compute_face_components()
        # Smoothing strategy selected by COTANGENT_WEIGHTS class variable.
        # See module docstring 'Normal field smoothing' for rationale.
        # Iteration counts differ on purpose: the discrete LBO is a
        # forward-Euler step, so 2 iterations of Pinkall-Polthier
        # already act like ~5 of the bounded uniform Laplacian.
        # Pushing PP to 5 iters overshoots on sharp creases and starts
        # flipping corner normals (verified empirically on fandisk).
        if self.COTANGENT_WEIGHTS:
            self._smooth_face_normals = self._smooth_face_normals_cotangent(iterations=2)
        else:
            self._smooth_face_normals = self._smooth_face_normals_laplacian(iterations=5)
        self._vertex_normals = self._compute_vertex_normals()
        self._vf_data, self._vf_offsets = self._build_vertex_faces()
        try:
            self._solver = pp3d.EdgeFlipGeodesicSolver(self.V, self.F)
        except RuntimeError as exc:
            # ``_sanitize_for_solver`` already ran above; if pp3d still
            # rejects the topology the defect is one we don't auto-fix
            # (non-orientable surface, edge with inconsistent winding,
            # disconnected fan past a saddle vertex, …).  Wrap the raw
            # ``GC_SAFETY_ASSERT`` text with context so the user knows
            # this is a mesh issue rather than a solver bug.
            raise RuntimeError(
                f"pp3d EdgeFlipGeodesicSolver rejected the mesh after "
                f"topology sanitisation ({sanitize_report}).  The "
                f"underlying defect is one this auto-cleaner cannot "
                f"repair (non-orientable surface, inconsistent winding, "
                f"or another geometry-central manifold violation).  "
                f"Try cleaning the mesh in MeshLab / Blender / trimesh "
                f"before loading.\n  pp3d error: {exc}"
            ) from exc

        # Monotonic counter incremented by ``compute_shoot`` whenever
        # the JIT inner loop bails out via the >2-consecutive-fallback
        # safeguard (non-2-manifold vertex fan).  Editor reads it on
        # the poll tick and surfaces a HUD warning so the user knows
        # the geodesic ended short due to a mesh defect, not a bug.
        # Single-threaded read is fine: ``compute_shoot`` is only
        # called from the main thread (the orange worker pool runs in
        # subprocesses with their own ``GeodesicMesh`` instances).
        self._shoot_truncation_count: int = 0

        # Central VTK locator — used for ALL surface queries (pick,
        # project, find_face).  Skipped when ``build_locator=False``
        # (background workers don't need it).
        self.locator = self._build_locator() if self._build_locator_enabled else None

        # Pre-allocated VTK refs (avoids per-call object creation).
        #
        # **Not thread-safe.**  These four buffers are reused across
        # ``find_face`` / ``project_to_surface`` / ``compute_shoot`` —
        # two concurrent calls on the same ``GeodesicMesh`` instance
        # will clobber each other's results.  The fallback-flag formerly
        # stored on ``self._last_was_fallback`` had the same hazard,
        # which is why ``compute_endpoint`` / ``compute_endpoint_local``
        # / ``compute_endpoint_from_origin`` now return ``(path,
        # was_fallback)`` tuples — see their docstrings.  When you do
        # need parallelism, give each worker its own ``GeodesicMesh``
        # (the orange-curve worker pool already does this).
        self._vtk_cp = [0.0, 0.0, 0.0]
        self._vtk_cell_id = vtk.reference(0)
        self._vtk_sub_id = vtk.reference(0)
        self._vtk_dist2 = vtk.reference(0.0)

        # Face dual graph (CSR sparse) — built lazily on first
        # ``_dijkstra_corridor`` call.  Most sessions never trigger
        # the Dijkstra fallback (the Euclidean tube succeeds for
        # convex-ish geometry), so we don't pay the construction
        # cost up-front.
        self._face_dual_graph = None

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
        # Scale into [0, 2^21 - 1] then cast to uint32 (21 bits fits
        # comfortably).  The max corner maps to exactly ``scale`` = 2^21 - 1
        # (since ``(pts - bbox_min) / extent`` is exactly 1.0 there), so it
        # can never overflow into the 22nd bit — no epsilon nudge needed.
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

    def _build_face_adjacency_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Static face adjacency for O(1) face-neighbor lookup.

        Returns
        -------
        adj : (M, 3) int32
            ``adj[fi, e]`` = index of the face sharing edge *e* of face
            *fi*, or -1 if boundary.  Edge *e* connects ``F[fi, e]`` →
            ``F[fi, (e+1)%3]``.
        adj_edge : (M, 3) int32
            ``adj_edge[fi, e]`` = the edge index *inside the neighbour*
            that matches edge *e* of face *fi*; -1 on boundary edges.
            Used by the cotangent-Laplacian smoother to look up the
            "opposite vertex" inside the neighbour triangle.

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
        adj_edge = np.full((nf, 3), -1, dtype=np.int32)
        same = keys_s[:-1] == keys_s[1:]
        idx = np.where(same)[0]
        fi_a, ei_a = fids_s[idx], elocal_s[idx]
        fi_b, ei_b = fids_s[idx + 1], elocal_s[idx + 1]
        adj[fi_a, ei_a] = fi_b
        adj[fi_b, ei_b] = fi_a
        adj_edge[fi_a, ei_a] = ei_b
        adj_edge[fi_b, ei_b] = ei_a
        return adj, adj_edge

    def _compute_face_components(self) -> I32Array:
        """Labels each face with its connected component index.

        Builds the dual graph (faces as nodes, edge-adjacency as
        edges) as a scipy CSR matrix and delegates to
        ``connected_components`` — a single C call vs the previous
        Python BFS that took ~2 s on 3M-face meshes.  Returns an
        int32 array of length ``len(F)`` where ``labels[fi]`` is the
        component id (0-based).
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        nf = len(self.F)
        adj = self._face_adj  # (nf, 3) int32, -1 for boundary edges
        # Build COO from valid (face, neighbour) pairs.  Each interior
        # edge contributes two symmetric entries (fi, nb) — the dual
        # graph is undirected.
        rows = np.repeat(np.arange(nf, dtype=np.int32), 3)
        cols = adj.ravel()
        mask = cols >= 0
        rows = rows[mask]
        cols = cols[mask]
        data = np.ones_like(rows, dtype=np.int8)
        graph = csr_matrix((data, (rows, cols)), shape=(nf, nf))
        _, labels = connected_components(graph, directed=False, return_labels=True)
        return labels.astype(np.int32, copy=False)

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

        Morton-reorder synchronisation
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ``_morton_reorder_inplace`` permutes ``self.V`` and ``self.F``
        for cache locality but cannot mutate the original
        ``pv.PolyData`` the caller may still hold.  Building the
        locator on the *original* PolyData would return cell ids in
        the pre-reorder layout, which then index into a different
        triangle when looked up via ``self.F[cid]`` — silently sending
        every ``find_face`` call into the slow KDTree fallback (~6×
        speed cost was measured on a 7K-face sphere).

        Fix: build a fresh ``pv.PolyData`` from the post-reorder
        ``self.V`` / ``self.F`` and use it as the locator's dataset.
        ``self._pv_mesh`` is replaced with the synced one so any
        future re-build also uses the consistent layout.
        """
        if self._pv_mesh is None and (self.V is None or self.F is None):
            return None
        # Lazy import — pyvista is a heavy dep; only paid when
        # GeodesicMesh is built from raw arrays *and* a locator is
        # requested (the editor path always wants one).
        import pyvista as pv
        synced = pv.PolyData()
        synced.points = self.V
        # PyVista expects an Nx4 face buffer with leading "3" per row.
        faces_with_3 = np.column_stack([
            np.full(len(self.F), 3, dtype=self.F.dtype),
            self.F,
        ])
        synced.faces = faces_with_3.ravel()
        # Replace _pv_mesh so subsequent code (e.g. a future
        # _build_locator rebuild) sees the post-Morton layout, not
        # the stale pre-Morton one.
        self._pv_mesh = synced
        loc = vtk.vtkStaticCellLocator()
        loc.SetDataSet(synced)
        loc.SetNumberOfCellsPerNode(8)
        loc.SetMaxNumberOfBuckets(max(len(self.F) // 4, 1000))
        loc.BuildLocator()
        return loc

    def _compute_face_normals(self) -> F64Array:
        A, B, C = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        cross = np.cross(B - A, C - A); norms = np.linalg.norm(cross, axis=1, keepdims=True)
        return cross / np.where(norms < 1e-15, 1.0, norms)

    def _smooth_face_normals_laplacian(self, iterations: int = 5) -> F64Array:
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

    def _smooth_face_normals_cotangent(self, iterations: int = 2) -> F64Array:
        """Cotangent-Laplacian smoothing of face normals (Pinkall-Polthier).

        Classical discrete Laplace-Beltrami discretization applied to the
        face dual graph.  For each edge shared by two triangles, the
        weight on the dual edge connecting the two faces is::

            w(fi, fj) = ½ · ( cot α  +  cot β )

        where α is the angle, in face *fi*, at the vertex **opposite to
        the shared edge** — and β is the analogous angle in face *fj*.
        These are the two "off-edge" angles that complete each triangle.

        Why the opposite-angle cotangents?
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        This is the canonical Pinkall-Polthier weighting (1993): it
        makes the discrete Laplace-Beltrami operator depend only on
        the **intrinsic** geometry of the triangle pair, so the
        smoother is genuinely **invariant to triangulation quality**.
        Long thin triangles get downweighted because their opposite
        angles are tiny (cot of a small angle is huge — but the *peer*
        cotangent on the other side of the edge is small, and the row
        normalisation rebalances the total contribution).

        A previous implementation used ``cot(dihedral_angle)`` between
        the two face normals — a different geometric quantity that
        produces a *feature-preserving anisotropic* smoother but does
        not address the long-thin-triangle bias the docstring
        previously claimed to solve.  This version implements the
        classical Pinkall-Polthier formulation as advertised.

        Numerical robustness
        ~~~~~~~~~~~~~~~~~~~~
        ``cot α = (e1 · e2) / (2 · area)`` where e1, e2 are the two
        edges meeting at the opposite vertex (closed-form identity,
        no trigonometry).  The denominator uses ``2 · area`` directly,
        avoiding the ``sin θ`` division that explodes at near-degenerate
        triangles.  Negative cotangents (obtuse opposite angles) are
        clipped to zero — the standard intrinsic-Delaunay-style
        mass-lumping that keeps the smoother stable on irregular
        meshes.  Faces with zero total weight (all neighbours
        clipped, or boundary triangles with one side missing) get an
        identity self-loop so their normal survives the smoothing pass.

        Selected by ``COTANGENT_WEIGHTS = True`` on the class (the
        default; set it to ``False`` for the uniform-weight variant).
        """
        from scipy.sparse import coo_matrix, diags

        F = self.F
        nf = len(F)
        adj = self._face_adj
        adj_edge = self._face_adj_edge

        # --- Per-face cotangents at each of the 3 corners --------------
        # cot(angle at corner k of face fi)
        #   = (e_out · e_in) / (2 · area_fi)
        # where:
        #   e_out = V[(k+1)%3] - V[k]                = self._face_edges[fi, k]
        #   e_in  = V[(k-1)%3] - V[k]
        #         = -(V[k] - V[(k-1)%3])
        #         = -self._face_edges[fi, (k-1)%3]
        #
        # The two edges meeting at corner k both emanate from k; the
        # second edge is the negation of the previous edge in the
        # closed loop around the triangle.  Pulling the sign through:
        #   e_out · e_in = -(self._face_edges[fi, k]
        #                    · self._face_edges[fi, (k-1)%3])
        edges = self._face_edges                          # (nf, 3, 3)
        # roll(+1, axis=1) shifts so prev[fi, k] = edges[fi, (k-1)%3]
        edges_prev = np.roll(edges, 1, axis=1)
        # 2·area is constant per face: |e0 × e1| = 2·area_fi.
        cross_e01 = np.cross(edges[:, 0], edges[:, 1])
        two_area = np.linalg.norm(cross_e01, axis=1)      # (nf,)
        two_area_safe = np.maximum(two_area, 1e-15)
        # Sign pulled out as discussed above.
        dots = np.einsum('ijk,ijk->ij', edges, edges_prev)  # (nf, 3)
        cot_at_corner = -dots / two_area_safe[:, None]      # (nf, 3)

        # --- cot of the corner OPPOSITE to each edge -------------------
        # Edge e of face fi connects F[fi, e] and F[fi, (e+1)%3]; the
        # opposite vertex is F[fi, (e+2)%3], i.e. corner (e+2)%3.
        # roll(-2, axis=1) gives row[e] = cot_at_corner[(e+2)%3].
        cot_opp = np.roll(cot_at_corner, -2, axis=1)        # (nf, 3)

        # --- Build sparse weighted adjacency ---------------------------
        fi_arr = np.repeat(np.arange(nf, dtype=np.int32), 3)
        e_arr  = np.tile(np.arange(3, dtype=np.int32), nf)
        fj_flat = adj.ravel()
        ej_flat = adj_edge.ravel()
        mask = fj_flat >= 0
        rows  = fi_arr[mask]
        cols  = fj_flat[mask]
        e_my  = e_arr[mask]
        e_nb  = ej_flat[mask]

        cot_alpha = cot_opp[rows, e_my]
        cot_beta  = cot_opp[cols, e_nb]
        weights = 0.5 * (cot_alpha + cot_beta)
        # Standard mass-lumped Pinkall-Polthier: clip negatives.
        weights = np.maximum(weights, 0.0)

        A = coo_matrix((weights, (rows, cols)), shape=(nf, nf)).tocsr()

        # Faces whose entire row is zero (all neighbours clipped to 0,
        # or isolated component) need a self-loop so their normal
        # passes through the smoother unchanged instead of collapsing
        # to (0, 0, 0).
        row_sums = np.asarray(A.sum(axis=1)).ravel()
        isolated = row_sums < 1e-15
        if isolated.any():
            iso_idx = np.where(isolated)[0]
            self_loops = coo_matrix(
                (np.ones(len(iso_idx)),
                 (iso_idx.astype(np.int32), iso_idx.astype(np.int32))),
                shape=(nf, nf)).tocsr()
            A = A + self_loops
            row_sums = np.asarray(A.sum(axis=1)).ravel()

        A = diags(1.0 / row_sums) @ A

        normals = self._face_normals.copy()
        for _ in range(iterations):
            normals = A @ normals
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.where(norms < 1e-15, 1.0, norms)
        return normals

    def _compute_vertex_normals(self) -> F64Array:
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

    def face_normal(self, face_id: int) -> F64Array:
        return self._face_normals[int(face_id)].copy()

    @staticmethod
    def _barycentric(p: np.ndarray, A: np.ndarray, B: np.ndarray,
                     C: np.ndarray) -> tuple[float, float, float]:
        """Barycentric coordinates of *p* w.r.t. triangle (A, B, C).

        Single canonical implementation shared by ``get_barycentric``
        (global mesh) and ``_bary_buf`` (work-buffer variant).

        The five dot products are spelled out as scalar arithmetic
        rather than ``np.dot`` on 3-vectors: ``np.dot`` carries a
        per-call Python↔C dispatch cost that dominates for length-3
        inputs, and this is a hot leaf (``find_face`` /
        ``_outside_score`` call it once per candidate face).  The
        summation order is left-to-right, matching NumPy's sequential
        reduction for a 3-element dot — verified bit-for-bit against the
        cascade parity oracle (``tests/benchmark_endpoint_local.py
        --check``, 0.000e+00 on both locator regimes).
        """
        v0x = B[0] - A[0]; v0y = B[1] - A[1]; v0z = B[2] - A[2]
        v1x = C[0] - A[0]; v1y = C[1] - A[1]; v1z = C[2] - A[2]
        v2x = p[0] - A[0]; v2y = p[1] - A[1]; v2z = p[2] - A[2]
        d00 = v0x * v0x + v0y * v0y + v0z * v0z
        d01 = v0x * v1x + v0y * v1y + v0z * v1z
        d11 = v1x * v1x + v1y * v1y + v1z * v1z
        d20 = v2x * v0x + v2y * v0y + v2z * v0z
        d21 = v2x * v1x + v2y * v1y + v2z * v1z
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-15:
            return 1/3, 1/3, 1/3
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        return 1.0 - v - w, v, w

    def get_barycentric(self, p: F64Array, face_id: int) -> tuple[float, float, float]:
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

    def _find_faces_batch(self, pts: np.ndarray) -> np.ndarray:
        """Vectorised equivalent of ``[self.find_face(p) for p in pts]``.

        Returns an ``(N,)`` int array that is **bit-for-bit identical** to
        calling ``find_face`` per point — it selects the same nearest
        vertex and the same incident face — but amortises the per-point
        Python↔C overhead of ``KDTree.query`` into one batched call.

        Regime split:

        * **Locator present** (interactive editor mesh): VTK
          ``FindClosestPoint`` has no vectorised API, so this falls back to
          the per-point loop.  That regime's ``find_face`` is already
          ~11 µs/call, and batching cannot help it.
        * **No locator** (orange worker / CLI export — both build the mesh
          with ``build_locator=False``): the KDTree path is fully
          batchable.  One batched ``KDTree.query`` over all points, then
          the same per-point arg-min of ``_outside_score`` over the nearest
          vertex's incident faces that ``find_face`` performs (same key,
          same candidate iteration order ⇒ same tie-break ⇒ identical
          face).  This is the hot path: the boundary check in
          ``_try_solve_on_region`` calls ``find_face`` once per geodesic
          path point, and the worker's no-locator ``find_face`` was
          profiled at ~46 % of ``compute_endpoint_local``.
        """
        pts = np.asarray(pts, dtype=float)
        n = len(pts)
        out = np.empty(n, dtype=np.int64)
        if self.locator is not None:
            for i in range(n):
                out[i] = self.find_face(pts[i])
            return out
        # No locator — batch the single expensive C call (KDTree.query),
        # then replicate find_face's candidate selection exactly.
        _, vis = self._kdtree.query(pts)
        vf_data = self._vf_data
        vf_off = self._vf_offsets
        for i in range(n):
            vi = int(vis[i])
            p = pts[i]
            cands = vf_data[vf_off[vi]:vf_off[vi + 1]]
            out[i] = int(min(cands, key=lambda fi: self._outside_score(p, int(fi))))
        return out

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
        path_n, exit_code = _shoot_loop(
            curr_p, curr_d, int(face_idx), float(length), max_steps,
            fast_mode, self._face_normals, self._face_adj,
            self._face_verts, self._face_edges, self._face_edge_len2,
            self.V, self.F, self._vf_data, self._vf_offsets, path_buf)

        # Non-manifold-fan truncation: bump the instance counter so the
        # editor's poll tick can surface a HUD message.  The path is
        # still returned (truncated at the last good edge crossing) so
        # the caller's draw / measurement code keeps working.
        if exit_code == 1:
            self._shoot_truncation_count += 1

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

    def diagnose_path(self, path: F64Array, label: str) -> None:
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
        # ``k`` clamped to vertex count: scipy's KDTree raises when
        # k > nv, which trips on tiny demo meshes (icosahedron has 12
        # verts, fine; degenerate test meshes can have <7).
        k = min(7, len(self.V))
        _, nearest_verts = self._kdtree.query(pts, k=k)
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
        k = min(7, len(self.V))
        _, nearest_verts = self._kdtree.query(pts, k=k)
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
                                max_depth: int = 6,
                                labels: F64Array | None = None
                                ) -> F64Array | tuple[F64Array, F64Array]:
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
        labels : optional (N,) per-point scalar labels propagated through
              subdivision.  When a chord is split, the new midpoint
              label is the linear midpoint of its two endpoints' labels.
              Used by the interp-curve cache to keep a per-rendered-
              point splprep ``u`` parameter, which lets node-insertion
              identify the right segment of a self-intersecting spline
              without resorting to ambiguous 3-D distance.

        Returns
        -------
        (M, 3) refined polyline with M >= N when *labels* is None, or
        ``(refined_pts, refined_labels)`` when *labels* is provided.
        Unchanged if no segment exceeds the tolerance.
        """
        if len(pts) < 2:
            return (pts, labels) if labels is not None else pts
        if tol is None:
            mean_edge = float(np.sqrt(self._face_edge_len2.mean()))
            tol = mean_edge * 0.01
        tol_sq = tol * tol

        pts = np.asarray(pts, dtype=float)
        labels_arr: F64Array | None = (
            np.asarray(labels, dtype=float) if labels is not None else None)
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
            base = np.arange(n_old) + cumsplit
            out[base] = pts
            # Midpoint of segment i (if split) goes right after original i
            seg_idx = np.nonzero(needs_split)[0]
            mid_dst = seg_idx + cumsplit[:-1][needs_split] + 1
            out[mid_dst] = projected[needs_split]
            if labels_arr is not None:
                lab_out = np.empty(n_old + n_new, dtype=float)
                lab_out[base] = labels_arr
                lab_out[mid_dst] = (labels_arr[seg_idx] + labels_arr[seg_idx + 1]) * 0.5
                labels_arr = lab_out
            pts = out
        return (pts, labels_arr) if labels_arr is not None else pts

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

        Both buffers are oversized (V +3 slots, F +10) so the origin
        insertion can write at ``V_buf[nv]`` / ``F_buf[nf]`` without
        copying the full arrays.  ``compute_endpoint_from_origin`` only
        READS the cached buffers (vertex-snap test); its non-snap path
        delegates to ``compute_endpoint_local``, which builds its own
        submesh — the cache is never mutated after construction.

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
        return OriginCache(V_buf=V_buf, F_buf=F_buf, nv=nv, nf=nf,
                           idx=idx_o, p=np.array(p_origin),
                           solver=solver, kdtree=self._kdtree)

    def compute_endpoint_from_origin(self, origin_cache: OriginCache,
                                     p_end: F64Array) -> tuple[F64Array, bool]:
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

        Returns ``(path, was_fallback)`` (see ``compute_endpoint`` for
        the rationale of the tuple-return contract).
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
                    return np.array([origin_cache['p'], p_end]), True
                path = origin_cache['solver'].find_geodesic_path(idx_s, snap_idx)
                if path is None or len(path) < 2:
                    return np.array([origin_cache['p'], p_end]), True
                self.diagnose_path(path, "endpoint-cached-snap")
                return path, False

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
        and *success* is True if the solver returned a usable path.

        ``success`` is False (and *path* is None) when the solver
        returns ``None`` or a degenerate single-point path — both of
        which historically slipped past the wrapper as "success" and
        propagated invalid geometry to the GUI.

        Builds a face-adjacency buffer once from the global ``self.F``
        and passes it to ``_add_point_buf`` so endpoints that fall on
        edges trigger the 2-to-4 split (smooth, no artificial nudge)
        instead of the legacy nudge-inward fallback.
        """
        V_buf, F_buf, nv, nf = self._make_work_buffers(extra_verts=2, extra_faces=6)
        # Seed adjacency from the precomputed global table; reserve
        # extra_faces (=6) extra slots so the split path can grow it in
        # lockstep with F_buf (which is also sized nf + extra_faces).
        adj_buf = np.full((nf + 6, 3), -1, dtype=np.int32)
        adj_buf[:nf] = self._face_adj
        idx_s, nv, nf = self._add_point_buf(p_start, V_buf, F_buf, nv, nf,
                                            adj_buf=adj_buf)
        idx_e, nv, nf = self._add_point_buf(p_end,   V_buf, F_buf, nv, nf,
                                            adj_buf=adj_buf)
        nf = self._remove_degenerate_faces(F_buf, nf)
        if idx_s == idx_e:
            return np.array([p_start, p_end]), True
        solver = pp3d.EdgeFlipGeodesicSolver(V_buf[:nv], F_buf[:nf])
        path = solver.find_geodesic_path(idx_s, idx_e)
        if path is None or len(path) < 2:
            return None, False
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

    @staticmethod
    def _subdivide_submesh_1to4(V_sub: np.ndarray,
                                F_sub: np.ndarray,
                                ) -> tuple[np.ndarray, np.ndarray]:
        """Loop-style 1-to-4 subdivision of a submesh, no smoothing.

        Each face ``(a, b, c)`` is split into 4 sub-faces by inserting
        a new vertex at the midpoint of every edge::

                        a
                       / \\
                      /   \\
                     mca   mab
                    /  \\  / \\
                   /    \\/   \\
                  c----mbc----b

        The 4 new sub-faces are ``(a, mab, mca)``, ``(b, mbc, mab)``,
        ``(c, mca, mbc)``, ``(mab, mbc, mca)``.  Winding is preserved
        on every sub-face.  Edges shared between two faces use the
        same midpoint vertex (deduplicated by ``(min(va, vb),
        max(va, vb))`` key) so the result is a manifold mesh.

        **Why this exists.**  The orange worker calls
        ``compute_endpoint_local`` to compute geodesics inside a
        submesh extracted around the cascade endpoints.  When the
        submesh is coarse, the discrete geodesic produced by
        ``EdgeFlipGeodesicSolver`` is forced to follow real mesh
        edges, which can be far from the geodesic of the underlying
        smooth surface.  Worse, between two cascade samples the
        solver can flip-flop between two different chains of edges
        whose lengths cross — producing a discrete jump in the path
        length.  Subdividing the submesh once (4× the face count)
        gives the solver finer edges to work with: the discrete
        geodesic converges to the smooth one and the flip-flop
        disappears.  Verified empirically on fandisk: a 4.5 cm jump
        in path length between two consecutive cascade samples drops
        to 0.3 mm after one round of subdivision.

        **Vertex layout.**  ``V_sub`` is preserved verbatim as the
        prefix of ``V_fine``; midpoints are appended after.  This
        lets the caller reuse the original ``vmap`` (submesh-local →
        global vertex index) for the first ``len(V_sub)`` entries —
        the new midpoint vertices have no global counterpart and are
        treated as submesh-only.

        Cost: O(nf) Python (one dict lookup per edge, ~3 per face).
        For a 200-face submesh ~0.5 ms; the dominant cost of the
        subsequent solver call (~25 ms → ~100 ms after subdivision)
        is what actually grows.
        """
        edge_to_mid: dict[tuple[int, int], int] = {}
        new_V: list = list(V_sub)

        def get_mid(va: int, vb: int) -> int:
            key = (va, vb) if va < vb else (vb, va)
            existing = edge_to_mid.get(key)
            if existing is not None:
                return existing
            idx = len(new_V)
            new_V.append((V_sub[va] + V_sub[vb]) * 0.5)
            edge_to_mid[key] = idx
            return idx

        nf = len(F_sub)
        new_F = np.empty((4 * nf, 3), dtype=np.int32)
        for fi in range(nf):
            a = int(F_sub[fi, 0]); b = int(F_sub[fi, 1]); c = int(F_sub[fi, 2])
            mab = get_mid(a, b)
            mbc = get_mid(b, c)
            mca = get_mid(c, a)
            base = 4 * fi
            new_F[base]     = (a, mab, mca)
            new_F[base + 1] = (b, mbc, mab)
            new_F[base + 2] = (c, mca, mbc)
            new_F[base + 3] = (mab, mbc, mca)
        return np.asarray(new_V, dtype=float), new_F

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

    def _bfs_init(self, seed_faces) -> tuple[set[int], set[int]]:
        """Initial BFS state from the seed: ``(visited, frontier)``.

        Both sets contain every seed face.  Use ``_bfs_advance`` to
        expand by additional rings without redoing prior work.
        """
        visited = {int(f) for f in seed_faces}
        frontier = set(visited)
        return visited, frontier

    def _bfs_advance(self, visited: set[int], frontier: set[int],
                     extra_rings: int) -> None:
        """Expand BFS state by *extra_rings* additional rings, in place.

        Avoids the O(N²) cost of restarting from scratch on every retry
        of ``compute_endpoint_local``: at depth 60 after passes of
        3 / 15 / 30 / 60 (phase A's 3 rings, then phase C's
        escalations), we do ``3 + 12 + 15 + 30 = 60`` rings of work
        (the increments) instead of ``3 + 15 + 30 + 60 = 108`` (the
        absolute depths).
        """
        adj = self._face_adj
        for _ in range(extra_rings):
            if not frontier:
                return
            # Gather every neighbour of the current frontier in one
            # vectorised index op (``adj[frontier]`` ⇒ (k, 3)) instead of a
            # Python double loop over (frontier × 3 edges) — this loop was
            # the bulk of the ``bfs`` profiling bucket.  Drop boundary
            # slots (-1) and dedupe; the membership filter against
            # ``visited`` then keeps the result identical to the scalar
            # loop.  ``visited`` / ``frontier`` stay plain sets, so the
            # arbitrary gather order is irrelevant: the final sets are the
            # same and callers consume ``sorted(visited)``.
            frontier_arr = np.fromiter(frontier, dtype=np.intp, count=len(frontier))
            nbrs = adj[frontier_arr].ravel()
            nbrs = nbrs[nbrs >= 0]
            if nbrs.size == 0:
                return
            next_f = set()
            for nb in np.unique(nbrs).tolist():
                if nb not in visited:
                    visited.add(nb)
                    next_f.add(nb)
            if not next_f:
                return
            frontier.clear()
            frontier.update(next_f)

    def _try_solve_on_region(self, p_start: np.ndarray,
                             p_end: np.ndarray,
                             face_region: np.ndarray,
                             submesh_subdiv: int = 0):
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

        ``submesh_subdiv`` (default 0) requests in-place 1-to-4 Loop
        subdivision of the submesh (no smoothing) before the solver
        runs.  See :meth:`_subdivide_submesh_1to4` for the rationale —
        in short, a finer submesh lets the discrete geodesic
        converge to the smooth-surface geodesic and removes the
        ~cm-scale jumps that arise when the discrete-geodesic
        topology flips between two cascade samples.  Cost grows
        ~4× per subdivision level (face count).  Caller decides
        based on the use case (orange worker uses 1; blue path_12
        and other latency-sensitive paths leave it at 0).
        """
        V_sub, F_sub, vmap = self._extract_submesh(
            self.V, self.F, face_region)

        # Optional in-place 1-to-4 Loop subdivision of the submesh.
        # See ``_subdivide_submesh_1to4`` for the rationale (forces
        # the discrete geodesic to converge to the smooth-surface
        # geodesic by giving the solver finer edges to work with).
        # ``vmap`` is intentionally NOT extended: the new midpoint
        # vertices have no global counterpart.  ``_to_local``'s primary
        # path maps the global-nearest vertex through ``vmap`` and can
        # therefore only return ORIGINAL corners — a point inside a
        # subdivided triangle's central subface seeds the insertion
        # from a corner its subface doesn't touch, and
        # ``_add_point_local``'s bary backstop rescans all faces in
        # that case (see the comment there).  The local-KDTree fallback
        # (global-nearest outside the submesh) is built on the
        # post-subdivision ``V_sub`` and CAN return a midpoint index;
        # that is harmless — the index only seeds the candidate pool,
        # and a midpoint seed is at least as close as a corner seed.
        for _ in range(max(0, int(submesh_subdiv))):
            V_sub, F_sub = self._subdivide_submesh_1to4(V_sub, F_sub)

        nv_sub, nf_sub = len(V_sub), len(F_sub)

        # Boundary faces of the submesh: any face with at least one
        # neighbour that is NOT in the submesh.  A geodesic whose endpoint
        # lies on such a face may have been truncated.
        # Note: this set indexes into the GLOBAL ``self.F`` (pre-
        # subdivision) — the ``find_face(pt)`` lookup in the boundary
        # check below also uses the global mesh, so the check remains
        # valid regardless of how many subdivision rounds we did.
        adj = self._face_adj
        region_set = set(face_region.tolist())
        boundary_faces_global: set[int] = set()
        for fi in face_region:
            for nb in adj[fi]:
                if nb < 0 or int(nb) not in region_set:
                    boundary_faces_global.add(int(fi))
                    break

        # Topology-insertion buffers (oversize so _add_point_local can
        # subdivide without reallocation).  The 2-to-4 edge-split path
        # also allocates 2 new faces per insertion (same growth as
        # the 1-to-3 path), so ``2 * extra`` headroom on F covers
        # both inserts even in the pathological case where every
        # endpoint lands on an edge.
        extra = 4
        V_buf = np.empty((nv_sub + extra, 3), dtype=float)
        V_buf[:nv_sub] = V_sub
        F_buf = np.empty((nf_sub + 2 * extra, 3), dtype=np.int32)
        F_buf[:nf_sub] = F_sub

        # Local face-adjacency buffer — built once from F_sub, then
        # mutated in lockstep with F_buf by the topology-insertion
        # path (1-to-3 leaves it stale but the ``find_face`` linear
        # scan inside ``_add_point_local`` doesn't depend on it,
        # while 2-to-4 updates it as part of its operation).  Sized
        # to accommodate the same ``2 * extra`` growth as F_buf.
        adj_buf = self._build_face_adj_buf(F_buf, nf_sub, extra=2 * extra)

        try:
            # One batched KDTree query for both endpoints instead of two
            # single-point queries: scipy's per-call Python wrapper (input
            # validation, output shaping) costs ~tens of µs, dominating the
            # tiny 3-point search itself, so halving the call count is a
            # measurable ~4% on the worker path (profiled, fandisk).  The
            # nearest vertices are identical to the per-point queries, so
            # this is bit-for-bit output-preserving (parity oracle 0.000).
            _, vi_globals = self._kdtree.query(np.array([p_start, p_end]))
            vi_global_s = int(vi_globals[0])
            vi_global_e = int(vi_globals[1])

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
                p_start, V_buf, F_buf, nv, nf, vi_local_s, nf_sub,
                adj_buf=adj_buf)
            idx_e, nv, nf = self._add_point_local(
                p_end, V_buf, F_buf, nv, nf, vi_local_e, nf_sub,
                adj_buf=adj_buf)

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
        #
        # ``_find_faces_batch`` returns the same per-point face the
        # per-point ``find_face`` loop would (identical nearest-vertex +
        # candidate selection), but amortises the no-locator
        # ``KDTree.query`` into one batched call.  The orange worker / CLI
        # export build the mesh with ``build_locator=False``, where this
        # query was profiled at ~46 % of ``compute_endpoint_local``.  The
        # verdict is unchanged: 'boundary' iff ANY path point lands on a
        # submesh-boundary face — checking them in order returns at the
        # same point the old loop did.
        for fi_global in self._find_faces_batch(path):
            if int(fi_global) in boundary_faces_global:
                return ('boundary', None)

        return ('ok', path)

    def _get_face_dual_graph(self):
        """Lazy-build the face dual graph as a CSR sparse matrix.

        Nodes = face indices.  Edges = pairs of edge-adjacent faces.
        Edge weight = Euclidean distance between centroids — a good
        proxy for geodesic arc-length on roughly equilateral meshes
        (and acceptable on irregular ones, since Dijkstra is only
        used to *seed* a submesh region, not to produce the final
        geodesic).

        Construction is O(F) over edges and runs once per
        ``GeodesicMesh`` instance.  Only triggered when
        ``_dijkstra_corridor`` is invoked from the boundary-fail
        path of ``compute_endpoint_local``; most sessions never
        pay the cost.
        """
        if self._face_dual_graph is not None:
            return self._face_dual_graph
        from scipy.sparse import coo_matrix

        nf = len(self.F)
        adj = self._face_adj
        centroids = self._face_centroids
        fi = np.repeat(np.arange(nf, dtype=np.int32), 3)
        fj = adj.ravel()
        mask = fj >= 0
        fi = fi[mask]
        fj = fj[mask]
        dists = np.linalg.norm(
            centroids[fi.astype(np.int64)] -
            centroids[fj.astype(np.int64)], axis=1)
        self._face_dual_graph = coo_matrix(
            (dists, (fi, fj)), shape=(nf, nf)).tocsr()
        return self._face_dual_graph

    def _dijkstra_corridor(self, p_start: F64Array,
                           p_end: F64Array) -> list[int] | None:
        """Topological shortest path on the face dual graph from
        ``find_face(p_start)`` to ``find_face(p_end)``.

        Returns a list of face indices ordered from end back to
        start (orientation does not matter to the caller — the list
        is used as a seed set).  Returns ``None`` when the two
        faces are in disconnected components, when either endpoint
        face cannot be located, or when scipy reports no path.

        Why a topological route here?
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ``compute_endpoint_local``'s default seed is the projection
        of the *Euclidean* line A→B onto the mesh.  On U-shaped /
        horseshoe geometry this line cuts through air and projects
        onto the *wrong* wall — the BFS expansion then has to climb
        out and reach the real opposite end, often falling out of
        the submesh boundary and forcing the global solver
        (~300 ms).  Dijkstra on the face graph respects the
        surface, so even on extreme concavity the corridor follows
        the actual route the geodesic will take.

        Cost: ~10-50 ms via scipy's C implementation.  Only paid
        when phase A's Euclidean seed fails by boundary touch.
        """
        from scipy.sparse.csgraph import dijkstra

        try:
            start_face = self.find_face(p_start)
            end_face = self.find_face(p_end)
        except (ValueError, IndexError, RuntimeError) as exc:
            log.debug("dijkstra: find_face failed: %s", exc)
            return None

        if start_face == end_face:
            return [int(start_face)]
        if not self.same_component(start_face, end_face):
            return None

        if self.USE_ASTAR_CORRIDOR:
            return self._astar_corridor(int(start_face), int(end_face))

        graph = self._get_face_dual_graph()
        try:
            _, predecessors = dijkstra(
                graph, indices=start_face,
                return_predecessors=True, directed=False)
        except (ValueError, RuntimeError) as exc:
            log.debug("dijkstra: scipy call failed: %s", exc)
            return None

        # Backtrace end → start.  scipy uses -9999 (or any negative
        # value in newer releases) as the "no predecessor" sentinel
        # for both unreachable nodes and the source itself.
        path: list[int] = []
        cur = int(end_face)
        seen: set[int] = set()
        # Bound the loop to nf to defend against cycles from
        # unexpected sentinel values.
        nf = len(self.F)
        for _ in range(nf):
            path.append(cur)
            if cur == int(start_face):
                return path
            seen.add(cur)
            cur = int(predecessors[cur])
            if cur < 0 or cur in seen:
                return None
        return None

    def _astar_corridor(self, start_face: int, end_face: int) -> list[int] | None:
        """Single-pair shortest path on the face dual graph via A*.

        Same graph and edge weights (centroid distances) as
        :meth:`_dijkstra_corridor`, but explores only the nodes between
        *start_face* and *end_face* instead of scipy's full single-source
        sweep.  Heuristic = Euclidean distance between a face centroid and
        the end centroid — admissible (straight line ≤ centroid-graph
        path), so A* returns a cost-optimal path.  The path at *ties* may
        differ from scipy's predecessor backtrace; the corridor is only a
        seed, but that can still shift the submesh — hence the
        ``USE_ASTAR_CORRIDOR`` flag and the parity measurement.

        Returns the path as a list of face indices ordered end → start
        (matching ``_dijkstra_corridor``), or ``None`` if unreachable.
        """
        import heapq

        graph = self._get_face_dual_graph()
        indptr = graph.indptr
        indices = graph.indices
        data = graph.data
        centroids = self._face_centroids
        end_c = centroids[end_face]

        def h(u: int) -> float:
            d = centroids[u] - end_c
            return float(np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]))

        g_score: dict[int, float] = {start_face: 0.0}
        prev: dict[int, int] = {}
        pq: list[tuple[float, int]] = [(h(start_face), start_face)]
        closed: set[int] = set()
        found = False
        while pq:
            _, u = heapq.heappop(pq)
            if u in closed:
                continue
            if u == end_face:
                found = True
                break
            closed.add(u)
            gu = g_score[u]
            for k in range(int(indptr[u]), int(indptr[u + 1])):
                v = int(indices[k])
                if v in closed:
                    continue
                ng = gu + float(data[k])
                if v not in g_score or ng < g_score[v]:
                    g_score[v] = ng
                    prev[v] = u
                    heapq.heappush(pq, (ng + h(v), v))
        if not found:
            return None
        path = [int(end_face)]
        cur = end_face
        while cur != start_face:
            cur = prev[cur]
            path.append(int(cur))
        return path

    def compute_endpoint_local(self, p_start: F64Array,
                               p_end: F64Array,
                               n_line_samples: int = 100,
                               submesh_subdiv: int = 0,
                               ) -> tuple[F64Array, bool]:
        """Geodesic path using a **projected-line** submesh pre-filter.

        Returns
        -------
        ``(path, was_fallback)`` — *path* is the geodesic polyline and
        *was_fallback* is ``True`` when the result is a degraded
        2-point straight-line stub (cross-component, solver gave up,
        endpoints collapsed to the same inserted vertex).  Callers
        usually colour the rendered span red when ``was_fallback`` is
        true.

        Returning the flag in the tuple (rather than via instance state
        ``self._last_was_fallback``) is deliberate — it keeps the
        function thread-safe across the orange-worker pool, where two
        workers calling on the same ``GeodesicMesh`` would otherwise
        race on the shared attribute.

        Submesh subdivision
        ~~~~~~~~~~~~~~~~~~~
        ``submesh_subdiv`` (default 0) requests *N* rounds of in-place
        1-to-4 subdivision of the local submesh BEFORE the solver
        runs.  The discrete geodesic produced by
        ``EdgeFlipGeodesicSolver`` has to follow real mesh edges, so
        on coarse meshes it can be far from the smooth-surface
        geodesic — and worse, between two cascade samples the solver
        can flip-flop between two near-equal-length edge chains,
        producing visible kinks (~cm scale) in the orange curve.
        Subdividing the submesh (4× faces per level) gives the
        solver finer edges and the discrete geodesic converges to
        the smooth one; the flip-flop disappears.

        Cost: ~4× per level (a 25 ms call becomes ~100 ms at level 1).
        Use 1 for the orange worker; leave at 0 for latency-sensitive
        paths (blue ``path_12`` consolidation, handle drag).

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

        Three-phase strategy
        ~~~~~~~~~~~~~~~~~~~~
          A. **Euclidean tube + 3 rings** (~25 ms): handles >90 % of
             real queries.  The straight A→B line projected onto the
             surface is a near-perfect seed for convex / mildly-curved
             geometry.

          B. **Dijkstra corridor + 3 rings** (~30-60 ms, only if A
             returns ``'boundary'`` or ``'error'``): topological
             shortest path on the face dual graph, weighted by
             centroid distance.  Handles the U-shape / horseshoe
             case where the Euclidean tube projects onto the wrong
             wall and the BFS expansion can't recover.  The
             Dijkstra graph is built lazily on first miss; most
             sessions never pay this cost.

          C. **BFS escalation 15 → 30 → 60 rings** (~50-200 ms,
             only if both A and B fail): legacy safety net for
             pathological cases.  Expands from whichever set
             phases A and B left in ``visited`` / ``frontier``.

        After all local phases fail the method falls back to
        ``compute_endpoint`` on the full mesh.  A ``'trivial'``
        result (two endpoints collapsed to one vertex) is returned
        immediately — it is degenerate but correct.
        """
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

        # --- Phase A: Euclidean tube + 3 rings ---
        visited, frontier = self._bfs_init(seed_set)
        self._bfs_advance(visited, frontier, 3)
        face_region = np.array(sorted(visited), dtype=np.int32)
        status, path = self._try_solve_on_region(
            p_start, p_end, face_region, submesh_subdiv=submesh_subdiv)
        if status == 'ok':
            assert path is not None
            return path, False
        if status == 'trivial':
            assert path is not None
            return path, True

        # --- Phase B: Dijkstra corridor + 3 rings (concave fallback) ---
        # Activated when Phase A fails by 'boundary' (Euclidean line
        # projected onto the wrong wall) or 'error' (solver couldn't
        # construct on the tight tube).  The Dijkstra path on the
        # face dual graph respects surface topology, so it bridges
        # U-shapes the Euclidean projection cannot.
        if status in ('boundary', 'error'):
            corridor = self._dijkstra_corridor(p_start, p_end)
            if corridor is not None:
                # Fresh BFS state seeded from the corridor + endpoint
                # 1-rings.  Union with phase A's exploration so we
                # don't lose the legitimate faces the Euclidean tube
                # already found.
                corr_seed: set[int] = {int(f) for f in corridor}
                corr_seed.update(self._faces_for_point(p_start))
                corr_seed.update(self._faces_for_point(p_end))
                visited_d, frontier_d = self._bfs_init(corr_seed)
                self._bfs_advance(visited_d, frontier_d, 3)
                visited |= visited_d
                # Union frontiers (NOT overwrite) — replacing
                # ``frontier`` with ``frontier_d`` would discard phase
                # A's outer boundary, leaving phase C's BFS unable to
                # grow outward from the Euclidean tube.  ``_bfs_advance``
                # already filters faces that are interior to ``visited``,
                # so unioning is safe and preserves both fronts.
                frontier |= frontier_d
                face_region = np.array(sorted(visited), dtype=np.int32)
                status, path = self._try_solve_on_region(
                    p_start, p_end, face_region, submesh_subdiv=submesh_subdiv)
                if status == 'ok':
                    assert path is not None
                    return path, False
                if status == 'trivial':
                    assert path is not None
                    return path, True

        # --- Phase C: BFS escalation 15 → 30 → 60 rings (safety net) ---
        # We've already advanced the frontier 3 rings in phase A
        # (and possibly 3 more from the corridor seed in phase B);
        # ``prev_depth`` tracks total advance so we only walk the
        # delta on each step instead of restarting BFS.
        prev_depth = 3
        for k_rings in (15, 30, 60):
            self._bfs_advance(visited, frontier, k_rings - prev_depth)
            prev_depth = k_rings
            face_region = np.array(sorted(visited), dtype=np.int32)
            status, path = self._try_solve_on_region(
                p_start, p_end, face_region, submesh_subdiv=submesh_subdiv)
            if status == 'ok':
                assert path is not None
                return path, False
            if status == 'trivial':
                assert path is not None
                return path, True

        # All local attempts exhausted — global solver as last resort.
        return self.compute_endpoint(p_start, p_end)

    @staticmethod
    def _build_face_adj_buf(F_buf: np.ndarray, nf: int,
                            extra: int = 0) -> np.ndarray:
        """Build a face-adjacency buffer for the first *nf* faces of *F_buf*.

        Returns an ``(nf + extra, 3)`` int32 buffer where ``adj[fi, e]``
        is the face index that shares edge ``e`` of face ``fi``, or
        ``-1`` for boundary edges.  Edge ``e`` of face ``fi`` is the
        directed edge ``F_buf[fi, e] → F_buf[fi, (e + 1) % 3]``.

        The *extra* slots are filled with ``-1`` so callers can grow
        the adjacency in lockstep with ``F_buf`` (each topological
        insertion adds 2 faces).

        Built by hashing each directed edge as the unordered pair
        ``(min(v0, v1), max(v0, v1))``: matching pairs share the edge.
        Two faces with the same unordered pair sit on opposite sides
        of that edge (their winding traverses the edge in opposite
        directions, which is always true for a 2-manifold).
        """
        adj = np.full((nf + extra, 3), -1, dtype=np.int32)
        if nf == 0:
            return adj

        # Directed edges in the original ``(fi, e)`` iteration order
        # (C-order ravel of an (nf, 3) array ⇒ index = fi*3 + e), hashed
        # as the unordered pair (min, max).  Vectorised equivalent of the
        # former Python double loop, which hashed nf*3 edges through a
        # dict — ~15 % of compute_endpoint_local on coarse spans.
        fv = F_buf[:nf]
        v0 = fv
        v1 = fv[:, (np.arange(3) + 1) % 3]
        a = np.minimum(v0, v1).ravel()
        b = np.maximum(v0, v1).ravel()
        face_of = np.repeat(np.arange(nf, dtype=np.int32), 3)
        slot_of = np.tile(np.arange(3, dtype=np.int32), nf)

        # Stable sort by (a, b) keeping original (fi, e) order within ties
        # (the trailing arange key), so a run of equal edge keys is in the
        # same order the dict saw them — matching its pairing exactly.
        order = np.lexsort((np.arange(len(a)), b, a))
        a_s, b_s = a[order], b[order]
        face_s, slot_s = face_of[order], slot_of[order]

        same = (a_s[1:] == a_s[:-1]) & (b_s[1:] == b_s[:-1])
        # Non-manifold edge shared by ≥3 faces ⇒ two adjacent ``same``
        # flags.  Rare (sanitised submesh is 2-manifold-with-boundary);
        # fall back to the exact scalar pairing to preserve dict order.
        if np.any(same[1:] & same[:-1]):
            edge_map: dict[tuple[int, int], tuple[int, int]] = {}
            for fi in range(nf):
                for e in range(3):
                    w0 = int(F_buf[fi, e])
                    w1 = int(F_buf[fi, (e + 1) % 3])
                    key = (w0, w1) if w0 < w1 else (w1, w0)
                    if key in edge_map:
                        f_other, e_other = edge_map.pop(key)
                        adj[fi, e] = f_other
                        adj[f_other, e_other] = fi
                    else:
                        edge_map[key] = (fi, e)
            return adj

        # Every edge appears once (boundary ⇒ stays -1) or twice.  Each
        # True in ``same`` marks a matched pair at (i, i+1); link both
        # directions (the result is symmetric, so order within the pair
        # doesn't matter).
        i = np.nonzero(same)[0]
        adj[face_s[i], slot_s[i]] = face_s[i + 1]
        adj[face_s[i + 1], slot_s[i + 1]] = face_s[i]
        return adj

    def _split_edge_2to4(self, p: np.ndarray,
                         V_buf: np.ndarray, F_buf: np.ndarray,
                         adj_buf: np.ndarray,
                         nv: int, nf: int,
                         face_a: int, edge_local_a: int,
                        ) -> tuple[int, int, int] | None:
        """Insert *p* on a shared edge by splitting both incident triangles.

        Mathematically the right operation when *p* lies on an edge:
        the 1-to-3 subdivision used by ``_add_point_local`` would
        produce a degenerate sliver triangle (one of its sub-faces
        would have all three vertices on the edge).  Splitting instead
        the *pair* of triangles that share the edge into 4 sub-faces
        keeps every result strictly non-degenerate while still
        introducing *p* as a new vertex of the mesh.

        Geometry — given:

            face_a = (Va, Vb, Vc)   — Va is at slot ``edge_local_a``
            face_b = (Vb, Va, Vd)   — its neighbour across (Va, Vb)

        the operation rewrites both faces in place and adds two new
        sub-faces at slots ``nf`` and ``nf+1``::

            face_a   ←  (Va, p, Vc)        (in place)
            face_b   ←  (Vb, p, Vd)        (in place)
            new_f1   =  (p,  Vb, Vc)        (slot nf)
            new_f2   =  (p,  Va, Vd)        (slot nf+1)

        Adjacency is updated in *adj_buf* for the four modified / new
        faces AND for the four outer neighbours whose entry pointing
        to ``face_a`` / ``face_b`` is re-routed to ``new_f1`` / ``new_f2``
        (since the half of the original face containing that edge now
        belongs to a different face id).

        Parameters
        ----------
        p : (3,) ndarray
            Surface point to insert.  Caller should have verified that
            ``p`` is on (or extremely close to) the shared edge of
            ``face_a`` and its neighbour at slot ``edge_local_a``.
        V_buf, F_buf : pre-allocated buffers — modified in place.
        adj_buf : pre-allocated face-adjacency buffer — modified in place.
        nv, nf : current vertex / face counts.
        face_a, edge_local_a : the face containing *p* and the local
            edge slot ``∈ {0, 1, 2}`` that ``p`` lies on.

        Returns
        -------
        ``(p_idx, new_nv, new_nf)`` on success.  ``None`` when the
        edge is on the mesh boundary (no neighbour to split with) —
        caller falls back to the 1-to-3 path.

        Edge cases:
          * **Boundary edge** (``adj_buf[face_a, edge_local_a] < 0``):
            no neighbour to split.  Return None — caller can either
            do a tiny nudge inward or accept the boundary truncation.
          * **Inconsistent adjacency** (the table claims face_b is the
            neighbour but face_b does not actually contain the edge
            (Vb, Va)): treated as failure.  Return None.  This should
            never happen on a correctly-built mesh but we are defensive.
        """
        face_b = int(adj_buf[face_a, edge_local_a])
        if face_b < 0:
            return None  # boundary edge — fallback path

        e_a = int(edge_local_a)
        Va = int(F_buf[face_a, e_a])
        Vb = int(F_buf[face_a, (e_a + 1) % 3])
        Vc = int(F_buf[face_a, (e_a + 2) % 3])

        # Find which slot of face_b holds the directed edge (Vb, Va) —
        # i.e., the same edge traversed in the opposite direction.
        e_b = -1
        for k in range(3):
            if (int(F_buf[face_b, k]) == Vb and
                    int(F_buf[face_b, (k + 1) % 3]) == Va):
                e_b = k
                break
        if e_b < 0:
            # Adjacency table claims face_b is the neighbour but
            # face_b does not contain the directed edge — table
            # inconsistency.  Bail to the safe path.
            return None
        Vd = int(F_buf[face_b, (e_b + 2) % 3])

        # Outer neighbours that survive the split (their entry pointing
        # to face_a / face_b may need to be re-routed to a new face id).
        nb_a_BC = int(adj_buf[face_a, (e_a + 1) % 3])  # face_a edge (Vb, Vc)
        nb_a_CA = int(adj_buf[face_a, (e_a + 2) % 3])  # face_a edge (Vc, Va)
        nb_b_AD = int(adj_buf[face_b, (e_b + 1) % 3])  # face_b edge (Va, Vd)
        nb_b_DB = int(adj_buf[face_b, (e_b + 2) % 3])  # face_b edge (Vd, Vb)

        # Insert the new vertex.
        p_idx = nv
        V_buf[p_idx] = p
        nv += 1

        new_f1 = nf       # (p, Vb, Vc)
        new_f2 = nf + 1   # (p, Va, Vd)

        # Rewrite F.  The two original faces are replaced in place;
        # two slots at the tail receive the new sub-faces.  Winding
        # is preserved on every sub-face, so the manifold orientation
        # is unchanged.
        F_buf[face_a, 0] = Va
        F_buf[face_a, 1] = p_idx
        F_buf[face_a, 2] = Vc
        F_buf[face_b, 0] = Vb
        F_buf[face_b, 1] = p_idx
        F_buf[face_b, 2] = Vd
        F_buf[new_f1, 0] = p_idx
        F_buf[new_f1, 1] = Vb
        F_buf[new_f1, 2] = Vc
        F_buf[new_f2, 0] = p_idx
        F_buf[new_f2, 1] = Va
        F_buf[new_f2, 2] = Vd

        # Update adjacency for the four faces that changed.  Edge slot
        # convention: ``adj[fi, e]`` is across the directed edge
        # ``F[fi, e] → F[fi, (e+1) % 3]``.
        # face_a now (Va, p, Vc):
        #   slot 0 (Va→p)  ↔ new_f2 (which has p→Va)
        #   slot 1 (p→Vc)  ↔ new_f1 (which has Vc→p)
        #   slot 2 (Vc→Va) ↔ nb_a_CA (unchanged outer)
        adj_buf[face_a, 0] = new_f2
        adj_buf[face_a, 1] = new_f1
        adj_buf[face_a, 2] = nb_a_CA

        # face_b now (Vb, p, Vd):
        #   slot 0 (Vb→p)  ↔ new_f1 (which has p→Vb)
        #   slot 1 (p→Vd)  ↔ new_f2 (which has Vd→p)
        #   slot 2 (Vd→Vb) ↔ nb_b_DB (unchanged outer)
        adj_buf[face_b, 0] = new_f1
        adj_buf[face_b, 1] = new_f2
        adj_buf[face_b, 2] = nb_b_DB

        # new_f1 (p, Vb, Vc):
        #   slot 0 (p→Vb)  ↔ face_b (which has Vb→p)
        #   slot 1 (Vb→Vc) ↔ nb_a_BC (was face_a's neighbour across (Vb, Vc))
        #   slot 2 (Vc→p)  ↔ face_a (which has p→Vc)
        adj_buf[new_f1, 0] = face_b
        adj_buf[new_f1, 1] = nb_a_BC
        adj_buf[new_f1, 2] = face_a

        # new_f2 (p, Va, Vd):
        #   slot 0 (p→Va)  ↔ face_a (which has Va→p)
        #   slot 1 (Va→Vd) ↔ nb_b_AD (was face_b's neighbour across (Va, Vd))
        #   slot 2 (Vd→p)  ↔ face_b (which has p→Vd)
        adj_buf[new_f2, 0] = face_a
        adj_buf[new_f2, 1] = nb_b_AD
        adj_buf[new_f2, 2] = face_b

        # Re-route outer neighbours whose entry pointed to one of the
        # original faces but whose edge now belongs to a new sub-face.
        # nb_a_BC: was across face_a's (Vb, Vc); that edge now lives
        # on new_f1.  Update the slot that pointed to face_a.
        if nb_a_BC >= 0:
            for k in range(3):
                if int(adj_buf[nb_a_BC, k]) == face_a:
                    adj_buf[nb_a_BC, k] = new_f1
                    break
        # nb_a_CA's edge is still owned by face_a — no change.
        # nb_b_AD: was across face_b's (Va, Vd); now on new_f2.
        if nb_b_AD >= 0:
            for k in range(3):
                if int(adj_buf[nb_b_AD, k]) == face_b:
                    adj_buf[nb_b_AD, k] = new_f2
                    break
        # nb_b_DB's edge is still owned by face_b — no change.

        return p_idx, nv, nf + 2

    def _add_point_local(self, p, V_buf, F_buf, nv, nf,
                         vi_local, nf_original, adj_buf=None):
        """Insert a point into submesh topology with 1-to-3 subdivision.

        *vi_local* is the submesh-local index of the nearest vertex to *p*
        (precomputed by the caller via the global KDTree + searchsorted).
        Avoids building a local KDTree per call.

        *nf_original* is the face count of the submesh **before** any
        prior insertions in this call sequence.  Faces with index ≥
        ``nf_original`` were created by previous ``_add_point_local``
        calls and are conservatively added to the candidate pool
        (they may contain the inserted vertex even though the linear
        scan over ``F_buf[fi]`` for ``vi`` would also catch them —
        this is belt-and-suspenders for the rare case where the
        bary scoring of the new sub-face wins over its parent).

        Earlier versions accepted ``nv_original`` (vertex count) here
        and used it as a face index, inflating ``candidates`` by
        ~hundreds per call on typical meshes (verified via cProfile,
        ~3500 candidates per call on a 12k-face submesh).  That was
        a real bug — fixed now.

        When *adj_buf* is provided (the caller pre-built local face
        adjacency in ``_try_solve_on_region``), points whose
        barycentric coordinate falls below ``edge_eps`` trigger the
        mathematically correct **2-to-4 edge split** (see
        :meth:`_split_edge_2to4`) instead of the conservative nudge.
        That eliminates the discrete jump in endpoint position when
        the cascade's *t* parameter sweeps the point across an edge —
        the dominant source of jitter in the rendered orange curve
        on dense meshes.  When *adj_buf* is None, falls back to the
        legacy nudge with very small ``edge_eps``.

        Returns ``(vertex_idx, nv, nf)``.
        """
        vi = int(vi_local)

        # Candidate faces: all containing vi + any from prior insertions.
        # Vectorised equivalent of the former two Python loops over
        # ``range(nf)`` — the O(nf) scan was the bulk of this method's
        # cost (~15 % of compute_endpoint_local, see
        # profile_endpoint_local.py).  Order is preserved exactly so the
        # ``min(..., key=...)`` tie-break below is unchanged: ascending
        # vi-containing faces first, then ascending prior-insertion faces
        # (indices ≥ nf_original) not already present.
        vi_faces = np.nonzero((F_buf[:nf] == vi).any(axis=1))[0]
        candidates = vi_faces.tolist()
        if nf > nf_original:
            seen = set(candidates)
            candidates.extend(fi for fi in range(nf_original, nf)
                              if fi not in seen)
        if not candidates:
            candidates = list(range(nf))

        face_idx = min(candidates,
                       key=lambda i: self._outside_score_buf(p, i, V_buf, F_buf))

        u, v, w = self._bary_buf(p, face_idx, V_buf, F_buf)
        # Defensive backstop against a mis-seeded candidate pool.  The
        # pool is faces incident to the nearest vertex *vi*; that misses
        # p's true containing face whenever vi is not one of that face's
        # corners.  The load-bearing case is ``submesh_subdiv >= 1``: the
        # 1-to-4 subdivision adds midpoint vertices that ``vmap`` does not
        # cover, so ``_to_local`` can only ever return an *original*
        # corner — and a point inside a triangle's central subface (whose
        # three corners are all midpoints) then seeds from a corner its
        # subface doesn't touch.  The ``min(outside_score)`` pick lands on
        # an adjacent subface, ``min_bary`` goes strongly negative, and
        # the 2-to-4 split below would weld p onto the wrong edge while
        # its 3-D position sits inside the neighbour — a local fold.  A
        # grossly-negative bary means "chosen face does not contain p";
        # rescan all faces so the insert targets the real one.  Never
        # fires on a well-seeded pool (min_bary >= ~0), so the
        # parity-oracle'd submesh_subdiv=0 path is unchanged.
        if min(u, v, w) < -1e-2 and len(candidates) < nf:
            face_idx = min(range(nf),
                           key=lambda i: self._outside_score_buf(p, i, V_buf, F_buf))
            u, v, w = self._bary_buf(p, face_idx, V_buf, F_buf)
        fa = int(F_buf[face_idx, 0])
        fb = int(F_buf[face_idx, 1])
        fc = int(F_buf[face_idx, 2])
        # Three-tier strategy:
        #   snap_eps  : bary very close to 1 → snap to that vertex.
        #   split_eps : bary close to 0 → 2-to-4 edge split (preserves
        #               the exact position of p as a new vertex; no
        #               jitter when the caller's parameter sweeps p
        #               across an edge).  Permissive (1e-3) so almost
        #               every near-edge insertion goes through it.
        #   nudge_eps : 2-to-4 unavailable (boundary edge / no
        #               adj_buf) AND bary extremely close to 0 →
        #               nudge inward as last-resort to keep the
        #               1-to-3 path manifold.  Tight (1e-7) so the
        #               nudge fires only when geometrically necessary.
        snap_eps = 1e-7
        split_eps = 1e-3
        nudge_eps = 1e-7

        if u > 1 - snap_eps: return fa, nv, nf
        if v > 1 - snap_eps: return fb, nv, nf
        if w > 1 - snap_eps: return fc, nv, nf

        # Identify which barycentric coord is the smallest — that
        # tells us which edge the point is on, so 2-to-4 can target
        # the right pair of triangles.
        bary = (u, v, w)
        min_bary = min(bary)
        if min_bary < split_eps and adj_buf is not None:
            # Map smallest bary → the local edge slot opposite to that
            # vertex.  In a triangle (V0, V1, V2) with bary (u, v, w):
            #   u → 0 (vertex V0), opposite edge slot 1 (V1→V2)
            #   v → 0 (vertex V1), opposite edge slot 2 (V2→V0)
            #   w → 0 (vertex V2), opposite edge slot 0 (V0→V1)
            min_idx = bary.index(min_bary)
            opposite_edge_slot = (min_idx + 1) % 3
            result = self._split_edge_2to4(
                p, V_buf, F_buf, adj_buf, nv, nf,
                face_idx, opposite_edge_slot)
            if result is not None:
                return result
            # Boundary edge or inconsistent adjacency → fall through
            # to the legacy nudge path below.

        if min_bary < nudge_eps:
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
            e1 = tri[1] - tri[0]
            e2 = tri[2] - tri[0]
            # Explicit cross-product magnitude — avoids np.cross's
            # axis-handling (moveaxis / normalize_axis_tuple) and norm's
            # nrm2 wrapper, both of which dominate a single 3-vector op.
            # The area only gates the <1e-15 degenerate check (real areas
            # are ~1e-6, nine orders away), so any last-ULP drift cannot
            # flip the branch — verified bit-for-bit by the parity oracle.
            cx = e1[1] * e2[2] - e1[2] * e2[1]
            cy = e1[2] * e2[0] - e1[0] * e2[2]
            cz = e1[0] * e2[1] - e1[1] * e2[0]
            area = 0.5 * (cx * cx + cy * cy + cz * cz) ** 0.5
            if area < 1e-15:
                F_buf[face_idx] = saved_face
                F_buf[nf] = saved_nf0
                F_buf[nf + 1] = saved_nf1
                dists = [np.linalg.norm(p - V_buf[fa]),
                         np.linalg.norm(p - V_buf[fb]),
                         np.linalg.norm(p - V_buf[fc])]
                return [fa, fb, fc][np.argmin(dists)], nv - 1, nf

        return p_idx, nv, nf + 2

    def compute_endpoint(self, p_start: F64Array,
                         p_end: F64Array) -> tuple[F64Array, bool]:
        """Geodesic path between two exact 3D points via buffer-based mesh insertion.

        If the first attempt fails (solver rejects the modified mesh),
        retries with points nudged toward their face centroids.  Only
        falls back to vertex-snap as a last resort.

        Returns
        -------
        ``(path, was_fallback)`` — the second element is True whenever
        the result is a 2-point straight-line stub (cross-component or
        solver failure).  Returning by tuple rather than via instance
        state is what makes this function safe to invoke concurrently
        from background workers sharing the same ``GeodesicMesh``.
        """
        # Reject cross-component queries early — no geodesic can exist
        fi_s = self.find_face(p_start)
        fi_e = self.find_face(p_end)
        if not self.same_component(fi_s, fi_e):
            return np.array([p_start, p_end]), True

        # Attempt 1: exact positions
        try:
            path, ok = self._try_endpoint_insertion(p_start, p_end)
            if ok:
                assert path is not None
                self.diagnose_path(path, "endpoint")
                return path, False
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
            min_edge_s = min(float(np.linalg.norm(verts_s[i] - verts_s[(i + 1) % 3]))
                             for i in range(3))
            min_edge_e = min(float(np.linalg.norm(verts_e[i] - verts_e[(i + 1) % 3]))
                             for i in range(3))
            nudge_s = max(1e-6, min(1e-2, min_edge_s * 0.01))
            nudge_e = max(1e-6, min(1e-2, min_edge_e * 0.01))
            p_s2 = p_start * (1.0 - nudge_s) + A_s * nudge_s
            p_e2 = p_end * (1.0 - nudge_e) + A_e * nudge_e
            path, ok = self._try_endpoint_insertion(p_s2, p_e2)
            if ok:
                assert path is not None
                self.diagnose_path(path, "endpoint-nudged")
                return path, False
        except (RuntimeError, ValueError, TypeError, IndexError) as exc:
            log.debug("compute_endpoint attempt-2 (nudged) failed: %s", exc)

        # Last resort: snap to nearest vertices and use pre-built solver
        if self.locator is not None:
            log.warning("endpoint insertion failed after retry; falling back to vertex snap")
        _, idx_s = self._kdtree.query(p_start)
        _, idx_e = self._kdtree.query(p_end)
        idx_s, idx_e = int(idx_s), int(idx_e)
        if idx_s == idx_e:
            return np.array([p_start, p_end]), True
        try:
            path = self._solver.find_geodesic_path(idx_s, idx_e)
            if path is None or len(path) < 2:
                return np.array([p_start, p_end]), True
            self.diagnose_path(path, "endpoint-snapped")
            return path, False
        except (RuntimeError, ValueError, TypeError) as exc:
            log.debug("vertex-snap solver failed: %s", exc)
            return np.array([p_start, p_end]), True

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

    @staticmethod
    def _sanitize_for_solver(V: np.ndarray, F: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Best-effort topology cleanup so ``pp3d.EdgeFlipGeodesicSolver``
        doesn't fire ``GC_SAFETY_ASSERT`` on the user's mesh.

        Four repair passes, run in order:

          1. **Self-edge faces** — drop any face whose three vertex
             indices are not distinct.  Same defect class as
             :meth:`_remove_degenerate_faces` but applied globally
             instead of to a working buffer.
          2. **Duplicate faces** — drop rows whose unordered vertex
             triple appears more than once.  Real-world cause: a CAD
             merge that brought in the same triangle from two source
             meshes.  Keeps the first occurrence by row order.
          3. **Non-manifold edge fans** — greedy peel: while any
             *undirected* edge has > 2 incident faces, remove the
             face that contributes to the most over-count edges;
             recount; repeat.  Converges because each removal
             strictly decreases total over-count.  Real-world cause:
             anatomical / scan meshes with T-junctions or "fins"
             along a seam.
          4. **Inconsistent winding** — same greedy peel as pass 3
             but on *directed* edges (count > 1).  In an orientable
             2-manifold each ``(a → b)`` directed edge belongs to
             exactly one face — its reverse ``(b → a)`` belongs to
             the neighbour.  Two faces traversing the same edge in
             the same direction break orientation; gc reports it
             as ``duplicate edge in list a -- b`` even when the
             undirected count is the manifold-valid value 2.
          5. **Non-manifold vertices** (vertex split) — a vertex
             whose incident-face fan splits into multiple edge-
             connected components is a "pinch point" (two surfaces
             meeting only at one vertex).  gc reports it as
             ``vertex N appears in more than one boundary loop``.
             Repair preserves geometry: keep the first component on
             the original vertex; for each extra component append a
             fresh vertex at the same 3-D position and reassign that
             component's face slots to it.  After the split the two
             surfaces are topologically disjoint (they no longer
             share a vertex) but visually identical.

        After the topology fixes, vertices that no surviving face
        references are dropped and ``F`` is remapped accordingly.

        Cheap on a clean mesh: each detection pass is ``O(F log F)``
        and the function early-exits when nothing needs changing
        (``F`` is returned unmodified, no copy).

        Returns
        -------
        ``(V_out, F_out, report)`` where *report* counts how many
        faces / vertices each pass removed.  Caller decides whether
        to log + warn or stay silent based on
        ``report['total_faces_dropped']``.

        What this **cannot** repair: non-orientable surfaces, edges
        with consistent winding but incompatible normals across the
        seam, and other defects that pp3d's manifold check rejects
        even after edge-fan cleanup.  Those still raise from the
        solver constructor; the caller's ``except`` clause prints a
        message pointing at MeshLab / Blender / trimesh.
        """
        F = np.asarray(F, dtype=np.int32)
        V = np.asarray(V, dtype=np.float64)
        n_orig = len(F)

        # Pass 1: self-edge faces ([A, A, B] etc.)
        valid = ((F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2])
                 & (F[:, 0] != F[:, 2]))
        n_self = int((~valid).sum())
        if n_self:
            F = F[valid]

        # Pass 2: duplicate faces (same unordered triple).  ``np.unique``
        # on ``axis=0`` operates row-wise; sorting per row first
        # collapses (a, b, c) ≡ (b, a, c) ≡ (c, b, a) — the manifold
        # check is winding-agnostic for duplicate detection.
        if len(F):
            F_sorted = np.sort(F, axis=1)
            _u, unique_idx = np.unique(F_sorted, axis=0, return_index=True)
            n_dup = len(F) - len(unique_idx)
            if n_dup:
                # Preserve original row order among the survivors.
                F = F[np.sort(unique_idx)]
        else:
            n_dup = 0

        # Pass 3 + 4 strategy: batch peel.
        #
        # Earlier implementations did greedy *single-face* peel inside a
        # ``while`` loop: each iteration recomputed ``np.unique`` over
        # all 3 × len(F) edges and removed the one worst-scoring face.
        # That is O(F × F_dropped) — quadratic when the mesh has many
        # bad faces.  On RVP.vtk (241 K faces, 6552 with inconsistent
        # winding) it ran for ~4.5 minutes before the editor became
        # responsive.
        #
        # Batch peel: in one ``np.unique`` per outer iteration, find
        # *every* over-incident edge and drop **all-but-``limit``**
        # faces from each, picking the first-occurrence face(s) as
        # keepers (stable argsort).  Typically converges in a single
        # iteration (the second is just a defensive re-check); each
        # iteration is O(F log F).  On RVP.vtk this drops the
        # sanitiser from ~260 s to <100 ms.
        #
        # Correctness vs. the old greedy: both produce a manifold-clean
        # submesh.  The two heuristics can differ on which specific
        # faces are kept when the mesh has multiple competing defect
        # patterns, but for the caller's contract (a topology pp3d will
        # accept) any valid sub-cover suffices.

        def _peel_overcount_batch(F_in: np.ndarray,
                                  directed: bool,
                                  limit: int) -> tuple[np.ndarray, int]:
            """One batch peel of over-incident edges.

            *directed* selects between undirected (sorted ``lo,hi`` —
            pass 3) and directed (``a → b`` as authored — pass 4)
            edge keys.  *limit* is the maximum count permitted per
            unique edge (2 for undirected, 1 for directed).  Returns
            ``(F_out, n_dropped)`` — *F_out* is *F_in* unchanged when
            no edge exceeds the limit.
            """
            total_dropped = 0
            while len(F_in) > 0:
                e0 = np.concatenate([F_in[:, 0], F_in[:, 1], F_in[:, 2]]).astype(np.int64)
                e1 = np.concatenate([F_in[:, 1], F_in[:, 2], F_in[:, 0]]).astype(np.int64)
                if directed:
                    keys = (e0 << 32) | e1
                else:
                    keys = (np.minimum(e0, e1) << 32) | np.maximum(e0, e1)
                _u, inverse, counts = np.unique(
                    keys, return_inverse=True, return_counts=True)
                if int(counts.max()) <= limit:
                    break
                # Stable argsort groups equal-key slots into contiguous
                # runs; the first ``limit`` slots of each run are
                # keepers, the rest are flagged for face removal.
                # ``slot_to_face[s]`` recovers which face owns slot s
                # (slots are concatenated edge-major: face_0_e0,
                # face_1_e0, …, face_0_e1, …).
                n_F = len(F_in)
                slot_to_face = np.tile(np.arange(n_F, dtype=np.int64), 3)
                order = np.argsort(keys, kind='stable')
                sorted_keys = keys[order]
                # rank_in_group: 0 for the first slot of each unique-key
                # run, 1 for the second, ...  Compute via cumulative
                # group ids and per-group start offsets.
                group_breaks = np.concatenate(
                    [[True], sorted_keys[1:] != sorted_keys[:-1]])
                group_starts = np.where(group_breaks)[0]
                group_id = np.cumsum(group_breaks) - 1
                rank_in_group = (np.arange(len(sorted_keys), dtype=np.int64)
                                 - group_starts[group_id])
                drop_sorted_mask = rank_in_group >= limit
                drop_slot_idx = order[drop_sorted_mask]
                faces_to_drop = np.unique(slot_to_face[drop_slot_idx])
                if faces_to_drop.size == 0:
                    break  # paranoid — should be unreachable when max > limit
                F_in = np.delete(F_in, faces_to_drop, axis=0)
                total_dropped += faces_to_drop.size
            return F_in, total_dropped

        # Pass 3: non-manifold edges (undirected count > 2).
        F, n_nonman = _peel_overcount_batch(F, directed=False, limit=2)

        # Pass 4: inconsistent winding (directed count > 1).
        # Pass 3 ran first so undirected counts are already ≤ 2 when we
        # get here; dropping a face only reduces counts, so we never
        # re-introduce pass-3 violations.
        F = np.ascontiguousarray(F, dtype=np.int32)
        F, n_winding = _peel_overcount_batch(F, directed=True, limit=1)

        # Pass 5: non-manifold vertices.  After passes 1-4 every
        # undirected edge has incidence ≤ 2 with consistent winding,
        # but the face fan around a vertex can still split into
        # multiple edge-connected components — gc reports this as
        # ``vertex N appears in more than one boundary loop``.  We
        # detect by per-vertex union-find on incident faces (linked
        # iff they share an edge through the vertex) and repair by
        # vertex split: keep the first component on the original
        # vertex, append a duplicate of ``V[v]`` for each extra
        # component, and rewrite that component's face slots to the
        # duplicate.  Geometry preserved (the duplicate sits at the
        # exact same 3-D position); topology cleaned.
        n_split = 0
        if len(F):
            # Per-vertex face index via argsort.  ``F.ravel()`` runs
            # face-major: face0_v0, face0_v1, face0_v2, face1_v0, …
            # ``face_id_flat`` mirrors this so we know which face each
            # entry came from.
            verts_flat = F.ravel()
            face_id_flat = np.repeat(np.arange(len(F), dtype=np.int64), 3)
            order = np.argsort(verts_flat, kind='stable')
            sorted_verts = verts_flat[order]
            sorted_face_ids = face_id_flat[order]
            unique_v, v_start = np.unique(sorted_verts, return_index=True)
            v_end = np.concatenate(
                [v_start[1:], np.array([len(sorted_verts)], dtype=v_start.dtype)])

            # Mutable working list for V (rows appended on split).
            V_list = [V]
            n_V_curr = len(V)
            # F is mutated in place via fancy indexing on a copy.
            F = F.copy()

            for vi_idx in range(len(unique_v)):
                v = int(unique_v[vi_idx])
                faces_v = sorted_face_ids[v_start[vi_idx]:v_end[vi_idx]]
                if len(faces_v) <= 1:
                    continue

                # For each face that contains v, find the two "other"
                # vertices on edges through v.  Build edge_other ->
                # list of (face_id, slot_of_v) pairs.
                edge_to_faces: dict[int, list[int]] = {}
                v_slot_per_face: dict[int, int] = {}
                for fi in faces_v:
                    fi_int = int(fi)
                    row = F[fi_int]
                    if int(row[0]) == v:
                        slot = 0
                    elif int(row[1]) == v:
                        slot = 1
                    else:
                        slot = 2
                    v_slot_per_face[fi_int] = slot
                    o1 = int(row[(slot + 1) % 3])
                    o2 = int(row[(slot + 2) % 3])
                    edge_to_faces.setdefault(o1, []).append(fi_int)
                    edge_to_faces.setdefault(o2, []).append(fi_int)

                # Union-find over faces_v keyed on edge_through_v
                # equivalence: two faces sharing an edge through v fall
                # into the same component.
                parent: dict[int, int] = {int(fi): int(fi) for fi in faces_v}

                # ``parent`` is bound as a default arg: the closure is
                # rebuilt (and used) within this loop iteration only, but
                # the explicit bind silences B023 and makes that obvious.
                def _find(x: int, parent: dict[int, int] = parent) -> int:
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                for face_list in edge_to_faces.values():
                    if len(face_list) < 2:
                        continue
                    root0 = _find(face_list[0])
                    for fj in face_list[1:]:
                        rj = _find(fj)
                        if rj != root0:
                            parent[rj] = root0

                # Group faces by component root.
                comps: dict[int, list[int]] = {}
                for fi in faces_v:
                    comps.setdefault(_find(int(fi)), []).append(int(fi))
                if len(comps) <= 1:
                    continue

                # Multi-component vertex: split.  Keep first component
                # on ``v``; reassign each extra component to a fresh
                # vertex at the same 3-D position.
                comp_lists = list(comps.values())
                for extra in comp_lists[1:]:
                    n_split += 1
                    V_list.append(V[v:v + 1].copy())
                    new_vid = n_V_curr
                    n_V_curr += 1
                    for fi in extra:
                        F[fi, v_slot_per_face[fi]] = np.int32(new_vid)

            if n_split:
                V = np.concatenate(V_list, axis=0)

        # Final pass: drop unreferenced vertices and remap F.
        if len(F):
            used = np.unique(F.ravel())
        else:
            used = np.empty(0, dtype=np.int64)
        n_unused = len(V) - len(used)
        if n_unused:
            remap = np.full(len(V), -1, dtype=np.int64)
            remap[used] = np.arange(len(used))
            V_out = V[used]
            F_out = remap[F].astype(np.int32) if len(F) else F.astype(np.int32)
        else:
            V_out = V
            F_out = F

        report = {
            'self_edge_faces': n_self,
            'duplicate_faces': n_dup,
            'non_manifold_faces': n_nonman,
            'winding_faces': n_winding,
            'vertex_splits': n_split,
            'unreferenced_verts': int(n_unused),
            'total_faces_dropped': n_orig - len(F_out),
        }
        return V_out, F_out, report

    def _add_point_buf(self, p, V_buf, F_buf, nv, nf, adj_buf=None):
        """Insert a point into mesh topology using pre-allocated buffers.

        Operates on numpy arrays in-place — no list/tuple conversion.
        Returns (vertex_idx, new_nv, new_nf).

        Three insertion strategies, in order of preference:

          1. **Snap to nearest vertex** (``snap_eps = 1e-7``): the
             point is essentially on a mesh vertex; reuse it as-is.
          2. **2-to-4 edge split** (``edge_eps = 1e-7``, requires
             *adj_buf*): the point is essentially on an edge; split
             the two triangles sharing that edge into 4 sub-faces via
             :meth:`_split_edge_2to4`.  Mathematically the right
             operation, eliminates the discrete jump in inserted-point
             position when the caller's parameter sweeps the point
             across an edge.
          3. **1-to-3 interior subdivision** (otherwise): standard
             three-way split of the containing triangle.  Falls back
             to a tiny nudge inward if all of the above fail (boundary
             edge with no neighbour to split, ``adj_buf`` missing,
             or near-zero-area sub-face after subdivision).

        The previous default thresholds (``snap_eps = 1e-4``,
        ``edge_eps = 1e-3``) were ~1000× more conservative than the
        post-subdivision area check needs.  When the caller's input
        sweeps continuously, those thresholds caused a step
        discontinuity in the inserted-point position whenever the
        bary coord crossed the boundary — visible as ~1e-4 jitter
        in the rendered cascade.  The 2-to-4 path removes that
        discontinuity entirely, since it preserves the exact
        position of *p* as a new mesh vertex.
        """
        face_idx = self._find_face_buf(p, V_buf, F_buf, nv, nf)
        u, v, w = self._bary_buf(p, face_idx, V_buf, F_buf)
        # Same grossly-negative-bary backstop as ``_add_point_local``:
        # ``_find_face_buf`` seeds candidates from the single nearest
        # vertex, and on irregular tessellation (sliver fans, strong
        # size gradients) the true containing face may not touch that
        # vertex at all.  A min-bary below -1e-2 means "the chosen face
        # does not contain p" — welding p onto one of its edges would
        # fold the local topology.  The full rescan is O(nf) with a
        # Python key function, but it only fires on the mis-seeded
        # case; well-seeded insertions (min_bary >= ~0) are unchanged.
        if min(u, v, w) < -1e-2:
            face_idx = min(range(nf),
                           key=lambda i: self._outside_score_buf(
                               p, i, V_buf, F_buf))
            u, v, w = self._bary_buf(p, face_idx, V_buf, F_buf)
        fa, fb, fc = int(F_buf[face_idx, 0]), int(F_buf[face_idx, 1]), int(F_buf[face_idx, 2])
        snap_eps = 1e-7
        edge_eps = 1e-7

        # Case 1: snap to nearest vertex
        if u > 1 - snap_eps: return fa, nv, nf
        if v > 1 - snap_eps: return fb, nv, nf
        if w > 1 - snap_eps: return fc, nv, nf

        # Case 2: 2-to-4 edge split (when adjacency available).
        bary = (u, v, w)
        min_bary = min(bary)
        if min_bary < edge_eps and adj_buf is not None:
            min_idx = bary.index(min_bary)
            opposite_edge_slot = (min_idx + 1) % 3
            result = self._split_edge_2to4(
                p, V_buf, F_buf, adj_buf, nv, nf,
                face_idx, opposite_edge_slot)
            if result is not None:
                return result
            # 2-to-4 failed (boundary edge or inconsistent adjacency)
            # — fall through to the 1-to-3 path with nudge below.

        # Case 3: 1-to-3 subdivision.  Tiny nudge toward centroid
        # when bary < edge_eps but 2-to-4 was unavailable, so the
        # subdivision does not create a degenerate triangle.
        if min_bary < edge_eps:
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
            e1 = tri[1] - tri[0]
            e2 = tri[2] - tri[0]
            # Explicit cross-product magnitude — avoids np.cross's
            # axis-handling (moveaxis / normalize_axis_tuple) and norm's
            # nrm2 wrapper, both of which dominate a single 3-vector op.
            # The area only gates the <1e-15 degenerate check (real areas
            # are ~1e-6, nine orders away), so any last-ULP drift cannot
            # flip the branch — verified bit-for-bit by the parity oracle.
            cx = e1[1] * e2[2] - e1[2] * e2[1]
            cy = e1[2] * e2[0] - e1[0] * e2[2]
            cz = e1[0] * e2[1] - e1[1] * e2[0]
            area = 0.5 * (cx * cx + cy * cy + cz * cz) ** 0.5
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

    def short_geodesic(
        self,
        p0: F64Array,
        p1: F64Array,
        face_a: int | None = None,
        face_b: int | None = None,
    ) -> F64Array | None:
        """Fast exact geodesic between two points in adjacent triangles.

        When *p0* and *p1* lie in the same triangle (or in two
        edge-adjacent triangles), the geodesic between them is either
        a straight 3-D segment (same triangle) or a two-segment polyline
        through the unique optimal crossing point on the shared edge —
        no edge-flip iteration, no submesh extraction, no
        ``EdgeFlipGeodesicSolver`` invocation.  This is the fast path
        used by the orange worker's phase-3 chord-bridging when the
        cascade samples are dense enough that consecutive samples land
        in adjacent (or identical) faces.

        The math is the classic *unfold-and-mirror* construction.  Both
        triangle planes are rotated around the shared edge until they
        are coplanar.  In that 2-D plane the geodesic is a straight
        line, and its crossing of the shared edge is the (unique)
        point ``q`` that minimises ``|p0 - q| + |q - p1|`` — found by
        reflecting ``p1`` across the edge and intersecting the segment
        ``(p0, mirror(p1))`` with the edge.

        Validation
        ----------
        Before returning a result, the crossing parameter ``s`` along
        the shared edge is checked against a margin:

            ``s ∈ [margin, edge_len - margin]``,
            ``margin = max(1e-7, 0.001 * edge_len)``.

        If ``s`` falls outside that interval, the optimal geodesic is
        passing *around* one of the shared-edge vertices — the
        unfolding is no longer a flat construction (cone-curvature at
        the vertex), so the result would be wrong.  The method then
        returns ``None`` and the caller is expected to fall back to
        ``compute_endpoint_local``.

        Parameters
        ----------
        p0, p1 : (3,) ndarrays
            Surface points (each must lie ON the mesh).  Caller is
            responsible for that — typically they are already cascade
            samples returned by ``compute_endpoint_local``.
        face_a, face_b : int, optional
            If known, the face indices containing *p0* / *p1*.  When
            omitted, ``find_face`` is called for each — adds ~5-15 µs
            per call.  Hot-path callers (orange worker phase 3) keep
            the face indices alive across cascade evaluations and
            should pass them.

        Returns
        -------
        np.ndarray, shape (2, 3) or (3, 3), or None.
            ``[p0, p1]`` if both points are in the same face.
            ``[p0, q, p1]`` when the optimal crossing falls strictly
            inside the shared edge.  ``None`` on any other condition
            (non-adjacent faces, degenerate edge, crossing on/near a
            shared-edge vertex) — caller must fall back to a full
            geodesic solver call.
        """
        if face_a is None:
            face_a = self.find_face(p0)
        if face_b is None:
            face_b = self.find_face(p1)
        if face_a < 0 or face_b < 0:
            return None
        if face_a == face_b:
            # Same triangle: straight 3-D segment is the geodesic
            # (triangle is flat).
            return np.stack([p0, p1])

        # Locate the shared edge inside face_a's adjacency row.  The
        # adjacency matrix stores the neighbour face index per local
        # edge slot; we want the slot that points to face_b.
        adj_row = self._face_adj[face_a]
        elocal = -1
        for k in range(3):
            if int(adj_row[k]) == face_b:
                elocal = k
                break
        if elocal < 0:
            return None  # not edge-adjacent

        # Shared edge = vertices (F[face_a, elocal], F[face_a, (elocal+1) % 3]).
        v_i_idx = int(self.F[face_a, elocal])
        v_j_idx = int(self.F[face_a, (elocal + 1) % 3])
        v_i = self.V[v_i_idx]
        v_j = self.V[v_j_idx]
        edge = v_j - v_i
        edge_len_sq = float(np.dot(edge, edge))
        if edge_len_sq < 1e-24:
            return None  # degenerate edge — caller falls back
        edge_len = float(np.sqrt(edge_len_sq))
        edge_unit = edge / edge_len

        # Project p0 and p1 onto the edge axis, get tangential and
        # perpendicular components.  In the unfolded 2-D plane the
        # x-coord is the tangential projection and the y-coord is the
        # perpendicular distance — positive for face_a's half-plane and
        # NEGATIVE for face_b's (face_b is rotated around the edge to
        # the opposite side of the unfolded plane).
        d0 = p0 - v_i
        d1 = p1 - v_i
        p0_x = float(np.dot(d0, edge_unit))
        p1_x = float(np.dot(d1, edge_unit))
        p0_perp = d0 - p0_x * edge_unit
        p1_perp = d1 - p1_x * edge_unit
        p0_y = float(np.linalg.norm(p0_perp))   # >= 0, p0 in face_a
        p1_y = -float(np.linalg.norm(p1_perp))  # <= 0, p1 in unfolded face_b

        # Straight line from (p0_x, p0_y) to (p1_x, p1_y) crosses y=0
        # at a unique point unless both have y=0 (both on edge axis,
        # degenerate).  Solve y(t) = 0 for t on the segment.
        denom = p0_y - p1_y
        if abs(denom) < 1e-15:
            return None
        t_cross = p0_y / denom            # in (0, 1) when signs differ
        s = p0_x + t_cross * (p1_x - p0_x)  # x-coord of crossing on the edge

        # Reject crossings that hit (or come within margin of) either
        # shared-edge vertex.  A vertex hit means the optimal geodesic
        # actually wraps around the vertex's curvature cone — the flat
        # unfolding is invalid there.
        margin = max(1e-7, 0.001 * edge_len)
        if s < margin or s > edge_len - margin:
            return None

        # Crossing point in 3-D, on the shared edge.
        q = v_i + (s / edge_len) * edge
        return np.stack([p0, q, p1])


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

        P0, H_out, H_in, P1 = (np.asarray(p, dtype=float) for p in ctrl)

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
        P0, H_out, H_in, P1 = (np.asarray(p, dtype=float) for p in ctrl)
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
        out = np.asarray(c0 * one_minus_t + c1 * t_col, dtype=float)

        if do_proj:
            out = self.project_smooth_batch(out)

        return out


def eval_cascade_at_t(
    geo, t: float,
    path_b: np.ndarray, cum_b: np.ndarray, total_b: float,
    path_a_rev: np.ndarray, cum_a: np.ndarray, total_a: float,
    path_12: np.ndarray, cum_12: np.ndarray, total_12: float,
    submesh_subdiv: int = 0,
    use_full_mesh: bool = False,
) -> tuple[np.ndarray, bool]:
    """Evaluate one full de Casteljau cascade level at parameter *t*.

    Shared by the editor's orange worker (geo_splines) and the headless
    exporter (spline_export) so both produce the identical fully-geodesic
    curve.  Returns ``(point, degraded)`` where ``degraded`` is True if any
    of the three inner solver calls fell back to a straight-line polyline
    (component break, solver failure).

    The cascade structure:
      level 1  →  b01 = lerp(path_b, t)
                  b12 = lerp(path_12, t)   (path_12 is shared / cached)
                  b23 = lerp(path_a_rev, t)
      level 2  →  path_c0 = geodesic(b01, b12)  → c0 = lerp(path_c0, t)
                  path_c1 = geodesic(b12, b23)  → c1 = lerp(path_c1, t)
      level 3  →  path_final = geodesic(c0, c1)  → result = lerp(path_final, t)

    Solver dispatch:

    * ``use_full_mesh=False`` (default) — uses ``compute_endpoint_local``
      with the supplied *submesh_subdiv*.  Fast (~25-100 ms per call)
      but the submesh extraction can return topologically different
      paths for slightly-perturbed inputs at certain ``t``, producing
      visible jumps in the rendered curve.
    * ``use_full_mesh=True`` — uses ``compute_endpoint`` instead.
      ~3-5× slower because the solver is built on the augmented full
      mesh, but the answer is stable: equal inputs → equal outputs,
      modulo only floating-point noise.  Eliminates submesh-extraction
      artifacts.  Genuine cascade-topology jumps (where ``c0`` and
      ``c1`` themselves cross a saddle vertex as ``t`` advances)
      persist — no solver swap can fix those.
    """
    log_w = logging.getLogger("geo_splines.worker")
    degraded = False

    if use_full_mesh:
        def _solve(p_a, p_b):
            return geo.compute_endpoint(p_a, p_b)
    else:
        def _solve(p_a, p_b):
            return geo.compute_endpoint_local(
                p_a, p_b, submesh_subdiv=submesh_subdiv)

    b01 = GeodesicMesh.geodesic_lerp(path_b, t, cum_b, total_b)
    b12 = GeodesicMesh.geodesic_lerp(path_12, t, cum_12, total_12)
    b23 = GeodesicMesh.geodesic_lerp(path_a_rev, t, cum_a, total_a)

    try:
        path_c0, fb_c0 = _solve(b01, b12)
        if fb_c0:
            degraded = True
    except (RuntimeError, ValueError, TypeError, IndexError) as exc:
        log_w.debug("solver(b01, b12) failed: %s", exc)
        path_c0, degraded = np.array([b01, b12]), True
    if path_c0 is None or len(path_c0) < 2:
        path_c0, degraded = np.array([b01, b12]), True

    try:
        path_c1, fb_c1 = _solve(b12, b23)
        if fb_c1:
            degraded = True
    except (RuntimeError, ValueError, TypeError, IndexError) as exc:
        log_w.debug("solver(b12, b23) failed: %s", exc)
        path_c1, degraded = np.array([b12, b23]), True
    if path_c1 is None or len(path_c1) < 2:
        path_c1, degraded = np.array([b12, b23]), True

    cum_c0, total_c0 = GeodesicMesh.compute_path_lengths(path_c0)
    cum_c1, total_c1 = GeodesicMesh.compute_path_lengths(path_c1)
    c0 = GeodesicMesh.geodesic_lerp(path_c0, t, cum_c0, total_c0)
    c1 = GeodesicMesh.geodesic_lerp(path_c1, t, cum_c1, total_c1)

    try:
        path_final, fb_f = _solve(c0, c1)
        if fb_f:
            degraded = True
    except (RuntimeError, ValueError, TypeError, IndexError) as exc:
        log_w.debug("solver(c0, c1) failed: %s", exc)
        path_final, degraded = np.array([c0, c1]), True
    if path_final is None or len(path_final) < 2:
        path_final, degraded = np.array([c0, c1]), True

    cum_f, total_f = GeodesicMesh.compute_path_lengths(path_final)
    return GeodesicMesh.geodesic_lerp(path_final, t, cum_f, total_f), degraded

