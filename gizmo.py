# SPDX-License-Identifier: Apache-2.0
"""
gizmo.py — Interactive geodesic segment widget and visualization.

Two classes implement a symmetric pair of geodesic rays emanating from a
shared midpoint (origin) on a triangulated surface:

  * ``SegmentData`` — pure geometric state and computation.  No VTK
    dependency.  Usable in tests, serialization, or offline processing.
  * ``GeodesicSegment(SegmentData)`` — adds VTK actor management and
    plotter integration for interactive editing.

Interaction model
-----------------
Three control handles define the segment:

  (A) ←—— path_a ——— (P) ——— path_b ——→ (B)
       blue handle     red      green handle

  * **P (origin)**: dragging translates the whole segment via parallel
    transport — the shooting direction is preserved in the local tangent
    frame across faces.
  * **A or B (endpoints)**: dragging one endpoint recomputes that ray as an
    exact geodesic (Edge-Flip solver with topology insertion), then shoots
    the opposite ray with:
      - the **same arc-length** (polyline sum of segments),
      - a **perfectly opposite tangent direction** (projected onto the
        tangent plane at the origin).

Symmetry guarantees
-------------------
  * Both rays always have identical arc-lengths (polyline sum, exact).
  * Departure directions are exactly antipodal in the tangent plane.

Exactness model
---------------
The final state is always mathematically exact on the discrete surface.
During drag, a fast preview (vertex-snapped geodesic via ``AGILE_DRAG``)
keeps the UI responsive (~17 ms); on mouse-pause or release a debounce
triggers the exact recalculation via topology insertion (~340 ms).
Approximate states are strictly transient — the user always converges to
the exact solution.

Handle visualization
--------------------
By default, all three handles (P, A, B) are rendered as point-spheres.
When the module constant ``ARROW_HANDLES`` is ``True``, handles A and B
are instead rendered as directional cones aligned with the geodesic
tangent at the endpoint.  This makes the "ray shooter" metaphor visually
explicit — the arrow tip points in the direction the ray arrived at (and
would continue past) the endpoint.  Handle P always remains a sphere
since it represents a position, not a direction.

The cone geometry is a ``pv.Cone`` template created lazily on first use
(shared across all instances).  Points are transformed in a pre-allocated
buffer via Rodrigues rotation — no per-frame mesh creation.

Shared utilities
----------------
  * ``update_line_inplace(pd, pts)`` — module-level function for updating
    a PolyData polyline in-place without creating intermediate objects.
    Used by both ``GeodesicSegment.update_visuals`` and
    ``GeodesicSplineApp`` span/stitch rendering — single canonical
    implementation, no duplication.

Performance conventions
-----------------------
  * ``_rotate_basis`` uses fully inlined scalar math (same pattern as
    ``_parallel_transport`` in geodesics.py) — avoids numpy call overhead
    for 3-element vector operations that runs once per drag-P frame.
  * ``update_visuals`` writes into a pre-allocated ``_line_buf`` to
    concatenate path_a and path_b without per-frame ``np.vstack``
    allocation.
"""

from __future__ import annotations

from math import sqrt as _math_sqrt, acos as _math_acos, sin as _math_sin, cos as _math_cos
from typing import TYPE_CHECKING
import numpy as np
import pyvista as pv

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

if TYPE_CHECKING:
    from geodesics import GeodesicMesh

# When True, drag uses vertex-snapped geodesic (~17 ms) as preview;
# exact topology-inserted geodesic (~340 ms) is computed on debounce.
# When False, every drag move computes the exact solution (slow but precise).
AGILE_DRAG = True

# Print warnings when geodesic shoots fail or are truncated.
WARN_SHOOT = False

# When True, handles A and B are rendered as directional cones (arrows)
# aligned with the geodesic tangent at the endpoint, instead of plain
# spheres.  Makes the "ray shooter" metaphor visually explicit.
# Handle P (origin) always remains a sphere (position marker, not a direction).
ARROW_HANDLES = True

# When True, arrow size scales with camera distance (constant screen size,
# like the sphere handles).  When False, arrow size is proportional to
# h_length (grows with the geodesic).
ARROW_FIXED_SCREEN_SIZE = True
# Scale factor for fixed-screen-size mode (relative to camera distance).
# Analogous to CURSOR_SCALE in geo_shoot.py but smaller — arrows are
# handle-sized, not cursor-sized.
ARROW_SCREEN_SCALE = 0.01

# Global opacity for all auxiliary visuals (geodesic lines, handles,
# arrows).  Cycled by the 't' key in geo_splines.py.
GIZMO_OPACITY = 0.2

# Cached pv.Color().float_rgb lookups — avoids per-frame Color object allocation.
_COLOR_CACHE: dict[str, tuple[float, float, float]] = {}


def _color_rgb(name: str) -> tuple[float, float, float]:
    """Cached color lookup.  Avoids creating a ``pv.Color`` on every frame."""
    try:
        return _COLOR_CACHE[name]
    except KeyError:
        rgb = pv.Color(name).float_rgb
        _COLOR_CACHE[name] = rgb
        return rgb


@njit(cache=True, fastmath=True)
def _rotation_x_to_jit(dx, dy, dz, out):
    """Rodrigues rotation matrix mapping ``(1, 0, 0)`` to unit vector *(dx, dy, dz)*.

    JIT-compiled, writes the 3×3 result into pre-allocated *out*.
    Fully inlined scalar math — no numpy array creation.

    Derivation: axis ``k = (0, ky, kz)`` (cross of x-axis with d,
    normalized), ``|k| = 1``.  ``K²`` for ``kx=0``::

        K² = [[-1,     0,       0     ],
              [ 0,    -kz²,    kz*ky  ],
              [ 0,    ky*kz,   -ky²   ]]

    ``R = I + K·sin(θ) + K²·(1-cos(θ))``.
    """
    sqrt = _math_sqrt
    s = sqrt(dy * dy + dz * dz)  # sin(θ)
    c = dx                        # cos(θ)
    if s < 1e-10:
        if c > 0:
            out[0, 0] = 1.0; out[0, 1] = 0.0; out[0, 2] = 0.0
            out[1, 0] = 0.0; out[1, 1] = 1.0; out[1, 2] = 0.0
            out[2, 0] = 0.0; out[2, 1] = 0.0; out[2, 2] = 1.0
        else:
            out[0, 0] = -1.0; out[0, 1] = 0.0; out[0, 2] = 0.0
            out[1, 0] = 0.0;  out[1, 1] = 1.0; out[1, 2] = 0.0
            out[2, 0] = 0.0;  out[2, 1] = 0.0; out[2, 2] = -1.0
        return
    inv_s = 1.0 / s
    ky = -dz * inv_s
    kz = dy * inv_s
    one_m_c = 1.0 - c
    ky2 = ky * ky; kz2 = kz * kz; kykz = ky * kz
    # R = I + K*s + K²*(1-c)
    out[0, 0] = 1.0 - one_m_c           # I[0,0] + K²[0,0]*(1-c) = 1 - 1*(1-c) = c
    out[0, 1] = -kz * s                  # K[0,1]*s + K²[0,1]=0
    out[0, 2] = ky * s                   # K[0,2]*s + K²[0,2]=0
    out[1, 0] = kz * s                   # K[1,0]*s + K²[1,0]=0
    out[1, 1] = 1.0 - kz2 * one_m_c     # I + K²[1,1]*(1-c)
    out[1, 2] = kykz * one_m_c           # K[1,2]=0, K²[1,2]=kz*ky
    out[2, 0] = -ky * s                  # K[2,0]*s + K²[2,0]=0
    out[2, 1] = kykz * one_m_c           # K[2,1]=0, K²[2,1]=ky*kz
    out[2, 2] = 1.0 - ky2 * one_m_c     # I + K²[2,2]*(1-c)


class _LineConnBuffer:
    """Pre-allocated connectivity buffer for ``update_line_inplace``.

    Encapsulates the growing buffer as an instance rather than module-level
    globals, enabling fresh instances for testing and multi-instance
    scenarios.  Indices ``1..N`` are pre-filled with sequential values;
    only index 0 (point count) is written per call.  The buffer is doubled
    on overflow (amortized O(1) growth).
    """
    __slots__ = ('cap', 'buf')

    def __init__(self, initial_cap: int = 1024):
        self.cap = initial_cap
        self.buf = np.empty(initial_cap, dtype=int)
        self.buf[1:initial_cap] = np.arange(initial_cap - 1)

    def get(self, n: int) -> np.ndarray:
        """Return a connectivity slice for *n* points, growing if needed."""
        needed = n + 1
        if needed > self.cap:
            self.cap = needed * 2
            self.buf = np.empty(self.cap, dtype=int)
            self.buf[1:self.cap] = np.arange(self.cap - 1)
        self.buf[0] = n
        return self.buf[:needed]


# Default module-level instance — same runtime behaviour as the old globals.
_line_conn = _LineConnBuffer()


def update_line_inplace(pd: pv.PolyData, pts: np.ndarray) -> None:
    """Update a PolyData polyline in-place without allocating a temporary object.

    Shared utility used by ``GeodesicSegment.update_visuals`` and
    ``GeodesicSplineApp`` span/stitch rendering.  Writes connectivity
    and point data directly into the existing PolyData — no intermediate
    ``pv.lines_from_points()`` call.

    Uses ``np.ascontiguousarray`` defensively so that if a non-contiguous
    slice or view is passed, VTK's zero-copy path is preserved instead
    of triggering a deep copy inside the C++ layer.
    """
    if pts is None or len(pts) < 2:
        return
    pts = np.ascontiguousarray(pts, dtype=float)
    pd.points = pts
    pd.lines = _line_conn.get(len(pts))
    pd.Modified()


def update_dashed_line_inplace(pd: pv.PolyData, pts: np.ndarray) -> None:
    """Renders *pts* as **alternating** 2-point segments instead of a
    single polyline.

    Used by the orange layer while the worker is still computing: the
    polyline appears as a dashed / intermittent curve that fills in
    more densely as more sample points arrive, giving a clear "still
    working" visual signal.  When the span consolidates, the renderer
    switches back to ``update_line_inplace`` for a solid polyline.

    VTK ``lines`` connectivity layout for independent 2-point segments:
    ``[2, p0, p1,  2, p2, p3,  2, p4, p5, ...]`` — one 3-int record
    per segment.  We emit the **odd** 1-indexed segments (which are
    indices ``0, 2, 4, ...`` in 0-indexed counting), so the dashing
    always starts at the first endpoint and leaves the second half of
    each pair invisible.

    Falls back to a simple polyline if fewer than 2 points are supplied.
    """
    if pts is None or len(pts) < 2:
        return
    pts = np.ascontiguousarray(pts, dtype=float)
    n = len(pts)
    # Start indices of "odd 1-indexed" segments (equivalently: 0-indexed
    # even positions) — each segment covers (i, i+1) for i in 0, 2, 4, ...
    seg_starts = np.arange(0, n - 1, 2, dtype=np.int64)
    n_segs = len(seg_starts)
    lines = np.empty(3 * n_segs, dtype=np.int64)
    lines[0::3] = 2
    lines[1::3] = seg_starts
    lines[2::3] = seg_starts + 1
    pd.points = pts
    pd.lines = lines
    pd.Modified()


def safe_remove_actor(plotter: pv.Plotter, actor) -> None:
    """Detaches the mapper before removing *actor* from *plotter*.

    VTK's Python wrapper holds cross-references between Python and C++ —
    dropping the last Python ref to an actor does not always release the
    underlying RAM because the mapper keeps a back-pointer.  Explicitly
    calling ``actor.SetMapper(None)`` severs that link so the C++ graph
    can be collected at the next GC pass.  Matters on long sessions that
    repeatedly create and destroy curve actors (drag + undo cycles).

    Silently ignores errors — the plotter may already be torn down
    (window-X cleanup path) when this runs.
    """
    if actor is None:
        return
    try:
        if hasattr(actor, 'SetMapper'):
            actor.SetMapper(None)
    except Exception:
        pass
    try:
        plotter.remove_actor(actor)
    except Exception:
        pass


class SegmentData:
    """Pure geometric state and computation for a geodesic segment.

    Contains all surface-geometry attributes and the methods that operate
    on them (shooting, symmetric rays, basis rotation).  **No VTK or
    PyVista dependency** — instantiate freely in tests or for
    serialization without a plotter.

    Parameters
    ----------
    origin : ndarray (3,)   — midpoint position on the mesh surface.
    face_idx : int           — face index containing the origin.
    normal : ndarray (3,)    — surface normal at origin.
    u, v : ndarray (3,)      — orthonormal tangent basis at origin.

    Key attributes
    --------------
    p_a, p_b      — current endpoint positions (or None).
    path_a, path_b — Nx3 polyline arrays from origin to each endpoint.
    h_length       — arc-length of each ray (always equal for both).
    local_v        — 2D direction in the (u, v) basis, preserved across
                     translations via parallel transport.
    """

    def __init__(self, origin: np.ndarray, face_idx: int, normal: np.ndarray, u: np.ndarray, v: np.ndarray):
        self.origin = np.array(origin, dtype=float)
        self.face_idx = int(face_idx)
        self.normal = np.array(normal, dtype=float)
        self.u, self.v = np.array(u, dtype=float), np.array(v, dtype=float)

        # Geometry
        self.p_a, self.p_b = None, None
        self.path_a, self.path_b = None, None

        # Translation State
        self.local_v: np.ndarray = np.array([1.0, 0.0])
        self.h_length: float = 0.1
        # Cache from GeodesicMesh.prepare_origin (typed as OriginCache
        # TypedDict; kept as plain dict here to avoid a heavy import).
        self._origin_cache: dict | None = None

        # Interaction flags — used by both logic and rendering decisions,
        # so they live in the data layer.
        self.hover_marker: str | None = None
        self.is_active: bool = False   # Part of active spline
        self.is_preview: bool = False  # During fast movement
        self.is_dragging: bool = False # Currently being dragged
        self.is_dimmed: bool = False   # Inactive spline state

    def update_local_v(self, geo: GeodesicMesh = None) -> None:
        """Calculates stable local parameters for the translation mode."""
        if self.path_b is None or len(self.path_b) < 2: return
        v_b = self.path_b[1] - self.path_b[0]
        vn = np.linalg.norm(v_b)
        if vn < 1e-12: return
        
        v_b /= vn
        self.local_v[0] = np.dot(v_b, self.u)
        self.local_v[1] = np.dot(v_b, self.v)
        lvn = np.linalg.norm(self.local_v)
        if lvn > 1e-12: self.local_v /= lvn
        self.h_length = float(np.sum(np.linalg.norm(np.diff(self.path_b, axis=0), axis=1)))

    @staticmethod
    def _tangent_direction(path: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Extract departure direction from path, projected onto the tangent plane."""
        v = path[1] - path[0]
        v = v - np.dot(v, normal) * normal  # project onto tangent plane
        vn = np.linalg.norm(v)
        return v / vn if vn > 1e-12 else v

    def _update_symmetric_ray(self, primary_path: np.ndarray, geo: GeodesicMesh,
                              shoot_sign: float) -> tuple:
        """Shoot the opposite ray with exact same length and perfectly opposite tangent direction.
        Returns (path, endpoint) for the opposite ray."""
        v_tan = self._tangent_direction(primary_path, self.normal)
        total_l = float(np.sum(np.linalg.norm(np.diff(primary_path, axis=0), axis=1)))
        opp_path = geo.compute_shoot(self.origin, shoot_sign * v_tan, total_l, self.face_idx)
        opp_end = opp_path[-1] if opp_path is not None else None
        return opp_path, opp_end

    def _fast_geodesic_from_origin(self, target: np.ndarray, geo: GeodesicMesh) -> np.ndarray | None:
        """Fast geodesic from cached origin to target via vertex-snap (~17ms).

        Uses the solver pre-built with the origin already inserted into the
        topology.  Only the target is snapped to its nearest vertex — the
        origin is exact.  This is ~20x faster than full topology insertion
        but the endpoint may be off by up to one edge length.
        """
        cache = self._origin_cache
        idx_s = cache['idx']
        _, idx_e = cache['kdtree'].query(target)
        idx_e = int(idx_e)
        if idx_s == idx_e:
            return np.array([self.origin, target])
        try:
            return cache['solver'].find_geodesic_path(idx_s, idx_e)
        except Exception:
            return np.array([self.origin, target])

    def update_from_a(self, new_a: np.ndarray, geo: GeodesicMesh, exact: bool = False) -> None:
        """Expansion/Rotation: recalculate symmetry from handle A.

        Two-phase strategy:
          - Preview (exact=False): vertex-snap geodesic via cached solver (~17ms).
          - Exact   (exact=True):  full topology insertion for endpoint (~340ms).
        In both cases the opposite ray (B) is shot with identical arc-length
        and perfectly opposite tangent direction.
        """
        self.is_preview = AGILE_DRAG and not exact
        if self._origin_cache is None:
            self._origin_cache = geo.prepare_origin(self.origin)

        if exact:
            self.path_a = geo.compute_endpoint_from_origin(self._origin_cache, new_a)
        else:
            self.path_a = self._fast_geodesic_from_origin(new_a, geo)

        if self.path_a is not None and len(self.path_a) > 1:
            self.p_a = self.path_a[-1]
            self.path_b, self.p_b = self._update_symmetric_ray(self.path_a, geo, -1.0)
            self.update_local_v(geo)

    def update_from_b(self, new_b: np.ndarray, geo: GeodesicMesh, exact: bool = False) -> None:
        """Expansion/Rotation: recalculate symmetry from handle B.

        Two-phase strategy (see update_from_a docstring).
        """
        self.is_preview = AGILE_DRAG and not exact
        if self._origin_cache is None:
            self._origin_cache = geo.prepare_origin(self.origin)

        if exact:
            self.path_b = geo.compute_endpoint_from_origin(self._origin_cache, new_b)
        else:
            self.path_b = self._fast_geodesic_from_origin(new_b, geo)

        if self.path_b is not None and len(self.path_b) > 1:
            self.p_b = self.path_b[-1]
            self.path_a, self.p_a = self._update_symmetric_ray(self.path_b, geo, -1.0)
            self.update_local_v(geo)

    def update_from_p(self, new_p: np.ndarray, new_fi: int, geo: GeodesicMesh, exact: bool = False) -> None:
        """Translation: moves midpoint while preserving shooting direction.

        The tangent basis (u, v) is rotated via Rodrigues to the new surface
        normal, keeping ``local_v`` constant — so the 3D shooting direction
        is parallel-transported across faces.

        When ``exact=False`` (drag preview), shoots with ``fast_mode=True``
        (no parallel transport across edges, ~17 ms).  When ``exact=True``
        (debounce/release), shoots with full parallel transport (~340 ms).
        """
        self.is_preview = AGILE_DRAG and not exact
        self._origin_cache = None
        self.origin, self.face_idx = new_p, new_fi
        self._rotate_basis(geo.get_interpolated_normal(new_p, new_fi))

        v_3d = self.local_v[0] * self.u + self.local_v[1] * self.v
        fast = self.is_preview
        self.path_b = geo.compute_shoot(self.origin, v_3d, self.h_length, self.face_idx, fast_mode=fast)
        self.path_a = geo.compute_shoot(self.origin, -v_3d, self.h_length, self.face_idx, fast_mode=fast)
        self.p_b = self.path_b[-1] if self.path_b is not None else None
        self.p_a = self.path_a[-1] if self.path_a is not None else None

    def _rotate_basis(self, new_normal: np.ndarray) -> None:
        """Rodrigues rotation of (u, v, normal) basis to align with new_normal.

        Preserves the orientation of ``local_v`` in the tangent plane —
        this is how the shooting direction is parallel-transported when
        the user drags P across faces with different normals.

        Fully inlined scalar math (same pattern as ``_parallel_transport``
        in geodesics.py) — avoids numpy call overhead for 3-element vectors
        that dominates at ~30 µs per ``np.cross`` call.  Runs once per
        drag-P frame.
        """
        sqrt, acos, sin, cos = _math_sqrt, _math_acos, _math_sin, _math_cos

        n1x, n1y, n1z = float(self.normal[0]), float(self.normal[1]), float(self.normal[2])
        n2x, n2y, n2z = float(new_normal[0]), float(new_normal[1]), float(new_normal[2])

        dot = n1x * n2x + n1y * n2y + n1z * n2z
        if dot > 1.0:
            dot = 1.0
        elif dot < -1.0:
            dot = -1.0
        if dot > 0.999999:
            return

        # axis = cross(n1, n2)
        ax = n1y * n2z - n1z * n2y
        ay = n1z * n2x - n1x * n2z
        az = n1x * n2y - n1y * n2x
        al = sqrt(ax * ax + ay * ay + az * az)

        if al < 1e-12:
            if abs(n1x) < 0.9:
                ax, ay, az = 0.0, n1z, -n1y
            else:
                ax, ay, az = -n1z, 0.0, n1x
            al = sqrt(ax * ax + ay * ay + az * az)
            inv_al = 1.0 / al
            ax *= inv_al; ay *= inv_al; az *= inv_al
            angle = 3.141592653589793
        else:
            inv_al = 1.0 / al
            ax *= inv_al; ay *= inv_al; az *= inv_al
            angle = acos(dot)

        sin_t = sin(angle)
        cos_t = cos(angle)
        one_m_cos = 1.0 - cos_t

        # Rodrigues: v' = v*cos(θ) + (a × v)*sin(θ) + a*(a·v)*(1-cos(θ))
        # Applied in-place to self.u and self.v
        for vec in (self.u, self.v):
            vx, vy, vz = float(vec[0]), float(vec[1]), float(vec[2])
            cx = ay * vz - az * vy
            cy = az * vx - ax * vz
            cz = ax * vy - ay * vx
            d = ax * vx + ay * vy + az * vz
            rx = vx * cos_t + cx * sin_t + ax * d * one_m_cos
            ry = vy * cos_t + cy * sin_t + ay * d * one_m_cos
            rz = vz * cos_t + cz * sin_t + az * d * one_m_cos
            rn = sqrt(rx * rx + ry * ry + rz * rz)
            if rn > 1e-12:
                inv_rn = 1.0 / rn
                vec[0] = rx * inv_rn; vec[1] = ry * inv_rn; vec[2] = rz * inv_rn
            else:
                vec[0] = rx; vec[1] = ry; vec[2] = rz

        self.normal = new_normal


class GeodesicSegment(SegmentData):
    """Interactive geodesic segment widget with VTK rendering.

    Inherits all geometric state and computation from ``SegmentData``.
    Adds VTK actor management, visual styling, and plotter integration.

    When ``ARROW_HANDLES`` is True, handles A and B are rendered as
    directional cones aligned with the geodesic tangent at the endpoint.
    Handle P always remains a sphere.

    Instantiate ``SegmentData`` directly when you need the geometry
    without a plotter (unit tests, serialization, offline processing).
    """

    # Cone template (lazy, shared across all instances).  Created on first
    # arrow-handle render; stores (points, faces) arrays from a pv.Cone.
    _cone_tpl_pts: np.ndarray | None = None
    _cone_tpl_faces: np.ndarray | None = None

    def __init__(self, origin: np.ndarray, face_idx: int, normal: np.ndarray, u: np.ndarray, v: np.ndarray):
        super().__init__(origin, face_idx, normal, u, v)

        # VTK Management
        self._pd_line, self._act_line = None, None
        self._handle_pd: dict[str, pv.PolyData | None] = {'p': None, 'a': None, 'b': None}
        self._handle_act: dict[str, object | None] = {'p': None, 'a': None, 'b': None}

        # Pre-allocated line buffer (max_steps*2 + 2 covers any path pair)
        self._line_buf = np.empty((802, 3), dtype=float)

        # Sphere handle: reusable 1×3 buffer (avoids np.array per frame)
        self._handle_pt_buf = np.empty((1, 3), dtype=float)

        # Arrow handle buffers (allocated lazily on first arrow render)
        self._arrow_buf_a: np.ndarray | None = None
        self._arrow_buf_b: np.ndarray | None = None
        # Arrow rotation matrix buffer (reused by JIT kernel)
        self._arrow_R_buf = np.empty((3, 3), dtype=float)
        # Arrow transform cache: {tag: (d_copy, scale)} — skip transform
        # when direction and scale haven't changed
        self._arrow_cache: dict[str, tuple[np.ndarray, float, bool]] = {}

    @staticmethod
    def _apply_depth_priority(actor, offset: float = -6.0) -> None:
        """Shifts an actor closer to camera in z-buffer so it draws on top."""
        mapper = actor.GetMapper()
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0, offset)
        mapper.SetRelativeCoincidentTopologyPointOffsetParameter(offset)

    def clear_actors(self, plotter: pv.Plotter) -> None:
        """Removes all internal actors from the VTK scene."""
        if self._act_line is not None:
            safe_remove_actor(plotter, self._act_line)
        self._act_line = None
        for tag in self._handle_act:
            if self._handle_act[tag] is not None:
                safe_remove_actor(plotter, self._handle_act[tag])
            self._handle_pd[tag] = None
            self._handle_act[tag] = None

    @classmethod
    def _ensure_cone_template(cls) -> tuple[np.ndarray, np.ndarray]:
        """Lazily creates the shared cone template (base at origin, tip at +X)."""
        if cls._cone_tpl_pts is None:
            cone = pv.Cone(center=(0.5, 0, 0), direction=(1, 0, 0),
                           height=1.0, radius=0.3, resolution=8, capping=True)
            cls._cone_tpl_pts = np.array(cone.points, dtype=float)
            cls._cone_tpl_faces = np.array(cone.faces)
        return cls._cone_tpl_pts, cls._cone_tpl_faces

    @staticmethod
    def _rotation_x_to(d: np.ndarray) -> np.ndarray:
        """Rodrigues rotation matrix mapping ``(1, 0, 0)`` to unit vector *d*.

        Used to orient the cone template along the geodesic tangent direction.
        Handles the degenerate case where *d* ≈ ``(-1, 0, 0)`` (180° rotation).
        """
        c = float(d[0])  # cos(θ) = dot([1,0,0], d)
        s = np.sqrt(d[1] * d[1] + d[2] * d[2])  # |cross([1,0,0], d)|
        if s < 1e-10:
            if c > 0:
                return np.eye(3)
            return np.diag(np.array([-1.0, 1.0, -1.0]))  # 180° around Y
        inv_s = 1.0 / s
        ky, kz = -d[2] * inv_s, d[1] * inv_s
        K = np.array([[0.0, -kz,  ky],
                      [kz,  0.0,  0.0],
                      [-ky, 0.0,  0.0]])
        return np.eye(3) + K * s + K @ K * (1.0 - c)

    def _update_handle_arrow(self, plotter, tag: str, pt, color) -> None:
        """Renders handle A or B as a directional cone aligned with the
        geodesic tangent at the endpoint.

        Uses ``_rotation_x_to_jit`` (Numba-compiled) for the rotation
        matrix and a per-tag transform cache: if the tangent direction,
        scale, and hover state haven't changed, the expensive
        ``np.dot(tpl_pts * scale, R.T)`` is skipped entirely.

        Size mode controlled by ``ARROW_FIXED_SCREEN_SIZE``:
          - **True** (default): scale from camera distance — constant
            apparent size regardless of zoom.
          - **False**: scale proportional to ``h_length``.
        """
        pd = self._handle_pd[tag]
        act = self._handle_act[tag]

        if self.is_dimmed or pt is None:
            if act: act.SetVisibility(False)
            return

        # Tangent direction at the endpoint (arrival direction)
        path = self.path_a if tag == 'a' else self.path_b
        if path is None or len(path) < 2:
            if act: act.SetVisibility(False)
            return
        d = path[-1] - path[-2]
        dn = np.linalg.norm(d)
        if dn < 1e-12:
            if act: act.SetVisibility(False)
            return
        d /= dn

        if ARROW_FIXED_SCREEN_SIZE:
            # Use camera distance to the node ORIGIN, not the handle tip.
            # Handles far from origin would otherwise appear bigger (farther
            # from camera = larger scale for same screen size).
            cam = plotter.camera.position
            ox = cam[0] - self.origin[0]
            oy = cam[1] - self.origin[1]
            oz = cam[2] - self.origin[2]
            scale = np.sqrt(ox*ox + oy*oy + oz*oz) * ARROW_SCREEN_SCALE
        else:
            scale = self.h_length * 0.2

        is_hovered = (self.hover_marker == tag)
        actual_col = 'black' if is_hovered else color
        final_scale = scale * 1.4 if is_hovered else scale

        # --- Transform cache: skip np.dot if direction+scale+hover unchanged ---
        cached = self._arrow_cache.get(tag)
        cache_hit = (cached is not None
                     and abs(cached[1] - final_scale) < 1e-6
                     and cached[2] == is_hovered
                     and abs(cached[0][0] - d[0]) < 1e-6
                     and abs(cached[0][1] - d[1]) < 1e-6
                     and abs(cached[0][2] - d[2]) < 1e-6)

        tpl_pts, tpl_faces = self._ensure_cone_template()
        buf = self._arrow_buf_a if tag == 'a' else self._arrow_buf_b
        if buf is None or buf.shape[0] != tpl_pts.shape[0]:
            buf = np.empty_like(tpl_pts)
            if tag == 'a':
                self._arrow_buf_a = buf
            else:
                self._arrow_buf_b = buf
            cache_hit = False

        if not cache_hit:
            _rotation_x_to_jit(float(d[0]), float(d[1]), float(d[2]),
                               self._arrow_R_buf)
            np.dot(tpl_pts * final_scale, self._arrow_R_buf.T, out=buf)
            self._arrow_cache[tag] = (d.copy(), final_scale, is_hovered)

        # Always update position (endpoint moves during drag)
        buf_translated = buf + pt

        # Hovered arrows go fully opaque
        opacity = 1.0 if is_hovered else GIZMO_OPACITY

        if pd is None:
            pd = pv.PolyData(buf_translated.copy(), tpl_faces.copy())
            act = plotter.add_mesh(pd, color=actual_col, lighting=True,
                                   opacity=opacity)
            self._apply_depth_priority(act, -8.0)
            self._handle_pd[tag] = pd
            self._handle_act[tag] = act
        else:
            pd.points = np.ascontiguousarray(buf_translated)
            pd.Modified()
            prop = act.GetProperty()
            prop.SetColor(_color_rgb(actual_col))
            prop.SetOpacity(opacity)
            act.SetVisibility(True)

    def _update_handle(self, plotter, tag: str, pt, color):
        """Unified sync for control markers, keyed by handle tag ('p', 'a', 'b').

        When ``ARROW_HANDLES`` is True, delegates A/B handles to
        ``_update_handle_arrow`` for directional cone rendering.
        Handle P always uses the sphere path.
        """
        if ARROW_HANDLES and tag in ('a', 'b'):
            self._update_handle_arrow(plotter, tag, pt, color)
            return

        pd = self._handle_pd[tag]
        act = self._handle_act[tag]

        # Hide handles entirely for dimmed splines
        if self.is_dimmed or pt is None:
            if act: act.SetVisibility(False)
            return

        # Style logic: Contrast marker on hover
        is_hovered = (self.hover_marker == tag)
        if is_hovered:
            actual_col = 'darkred' if tag == 'p' else 'black'
            sz = 11 if tag == 'p' else 10
        else:
            actual_col = color
            sz = 7

        buf = self._handle_pt_buf
        buf[0, 0] = pt[0]; buf[0, 1] = pt[1]; buf[0, 2] = pt[2]

        # Hovered handles go fully opaque for visual prominence
        opacity = 1.0 if is_hovered else GIZMO_OPACITY

        if pd is None:
            pd = pv.PolyData(buf.copy())
            act = plotter.add_mesh(pd, color=actual_col, point_size=sz,
                                   render_points_as_spheres=True, lighting=False,
                                   opacity=opacity)
            self._apply_depth_priority(act, -8.0)
            self._handle_pd[tag] = pd
            self._handle_act[tag] = act
        else:
            pd.points = np.ascontiguousarray(buf); pd.Modified()
            prop = act.GetProperty()
            prop.SetColor(_color_rgb(actual_col))
            prop.SetPointSize(sz)
            prop.SetOpacity(opacity)
            act.SetVisibility(True)

    def update_visuals(self, plotter: pv.Plotter, line_width: int = 2) -> None:
        """Refreshes the visual representation with state-dependent styling.

        Visual states (priority order):
          - **dimmed**: inactive spline — gray, thin, translucent.
          - **preview**: drag in motion — red, thin (lw-1), 60% opacity.
          - **normal / consolidated**: full width, full opacity.

        Uses ``_line_buf`` (pre-allocated in ``__init__``) to concatenate
        path_a and path_b without per-frame ``np.vstack`` allocation.
        """
        if self.is_dimmed:
            line_color = '#dddddd'
            line_opacity = 0.3
            lw = 1.0
        elif self.is_preview:
            line_color = 'red'
            line_opacity = min(0.6, GIZMO_OPACITY)
            lw = max(1, line_width - 1)
        else:
            line_color = 'red'
            line_opacity = GIZMO_OPACITY
            lw = line_width

        # 1. Main Geodesic Path — write into pre-allocated buffer
        n = 0
        if self.path_a is not None and len(self.path_a) > 1:
            na = len(self.path_a)
            self._line_buf[:na] = self.path_a[::-1]
            n = na
        if self.path_b is not None and len(self.path_b) > 1:
            nb = len(self.path_b) - 1
            self._line_buf[n:n + nb] = self.path_b[1:]
            n += nb

        if n > 0:
            pts_arr = self._line_buf[:n]
            if self._pd_line is None:
                self._pd_line = pv.PolyData()
                update_line_inplace(self._pd_line, pts_arr)
                self._act_line = plotter.add_mesh(
                    self._pd_line, color=line_color, line_width=lw,
                    opacity=line_opacity, lighting=False)
                self._apply_depth_priority(self._act_line, -8.0)
            else:
                update_line_inplace(self._pd_line, pts_arr)
                prop = self._act_line.GetProperty()
                prop.SetColor(_color_rgb(line_color))
                prop.SetOpacity(line_opacity)
                prop.SetLineWidth(lw)
            self._act_line.SetVisibility(True)
        elif self._act_line:
            self._act_line.SetVisibility(False)

        # 2. Control Markers — bright highlight during drag preview,
        #    reverts to base color on consolidation (debounce sets
        #    is_preview=False while is_dragging may still be True).
        if self.is_dragging and self.is_preview:
            p_col, a_col, b_col = 'orangered', 'deepskyblue', 'yellow'
        else:
            p_col, a_col, b_col = 'red', 'blue', 'lime'

        self._update_handle(plotter, 'p', self.origin, p_col)
        self._update_handle(plotter, 'a', self.p_a, a_col)
        self._update_handle(plotter, 'b', self.p_b, b_col)

    def refresh_arrows(self, plotter: pv.Plotter) -> None:
        """Lightweight arrow-only refresh for camera-distance scaling.

        Invalidates the arrow transform cache (forces scale recomputation
        from the new camera position) and re-renders both A/B arrows.
        Skips the line and sphere handle updates that ``update_visuals``
        would do — ~3× cheaper per node.
        """
        if not ARROW_HANDLES or self.is_dimmed:
            return
        # Invalidate cache so _update_handle_arrow recomputes scale
        self._arrow_cache.clear()
        if self.is_dragging and self.is_preview:
            a_col, b_col = 'deepskyblue', 'yellow'
        else:
            a_col, b_col = 'blue', 'lime'
        self._update_handle(plotter, 'a', self.p_a, a_col)
        self._update_handle(plotter, 'b', self.p_b, b_col)
