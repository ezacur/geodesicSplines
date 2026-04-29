# SPDX-License-Identifier: Apache-2.0
"""
geo_shoot.py — Interactive Geodesic Midpoint Shooter.

Base application for interactive geodesic editing on triangulated meshes.
Handles plotter setup, surface picking, and the drag/debounce lifecycle
for GeodesicSegment widgets.

``SurfaceCursor`` is a self-contained component that owns the surface-
aligned circle + geodesic crosshair visual.  It manages its own VTK
actors, geometry buffers, and caching state — extracted from the main
class to keep ``MidpointShooterApp`` focused on interaction logic.

Subclassed by ``GeodesicSplineApp`` (geo_splines.py) to add multi-node
spline editing on top of the same infrastructure.

Design philosophy
-----------------
This application is built around a strict **exactness-first** principle:
every geodesic segment, its arc-length, its endpoints, and the tangent
directions at the origin must be mathematically exact on the discrete
surface — not approximations, not vertex-snapped, not "close enough".

Concretely:
  - Endpoints are inserted into the mesh topology (face subdivision or
    edge split) so the Edge-Flip solver operates on the user's exact
    positions — never snapped to pre-existing vertices.
  - Arc-lengths are the true polyline sum of segments, exploiting the
    fact that a geodesic on a triangle mesh is a piecewise-linear path.
    No Euclidean shortcuts, no approximations.
  - Symmetric rays depart in perfectly opposite directions in the tangent
    plane at the origin, with identical arc-lengths.
  - Tangent directions are extracted from the geodesic path and projected
    onto the tangent plane — not taken as raw first-segment vectors.

The final state — what the user sees when they stop interacting — is
always the mathematically exact discrete geodesic.

Performance principle: **if it can be done fast and exact, do it fast.**
Every operation is pre-optimized — pre-allocated buffers, pre-computed
face geometry, cached solvers, inlined math in hot loops — even where
the current workload would tolerate slower code.  This is deliberate:
the tasks ahead (spline networks, dense sampling, real-time deformation)
are computationally heavy, and a slow foundation compounds into an
unusable application.

Hot-path allocation discipline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Mouse-move events fire at display refresh rate (60–144 Hz).  Any Python
object created inside ``_on_move`` or its callees becomes GC pressure
that compounds into frame drops.  The codebase follows a strict rule:

  - **Buffers** (NumPy arrays for screen projections, hover caches)
    are pre-allocated once in ``__init__`` and reused via slice writes —
    never re-created per frame.  Cursor-specific buffers (crosshair
    geometry, circle transform, Rodrigues rotation) live inside
    ``SurfaceCursor`` following the same discipline.
  - **Screen projection** (``_to_screen_batch``) dispatches to the
    ``@njit`` kernel ``_to_screen_kernel`` which fuses the VP matrix
    multiply, perspective divide, and viewport mapping into a single
    scalar loop — eliminates ``np.dot`` / slice overhead that dominates
    for the small batches (N < 50) typical of hover caches.
  - **Hover markers** (3D positions of all segment handles) are cached in
    a flat array and rebuilt only when segments are added or removed
    (``_rebuild_hover_cache``), not on every mouse move.
  - **Crosshair PolyData** is updated in-place on cache miss only — on
    cache hit the existing VTK geometry is reused without any Python-side
    copy.
  - **Circle projection** is cached by screen-pixel distance and face id
    (``CIRCLE_REUSE_PX``).  On cache hit the expensive
    ``project_smooth_batch`` (32 points) is skipped entirely.
  - **Screen-distance checks** use manual ``dx*dx + dy*dy`` against
    squared thresholds — ``np.linalg.norm`` is avoided on every path
    that runs per mouse event.

Approximate-now, correct-immediately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The one concession to interactivity is **temporal**: during a drag, the
application shows a fast *preview* (vertex-snapped geodesic, ~17 ms)
that is visually close but not exact.  As soon as the mouse pauses
(150 ms) or is released, the preview is silently replaced by the exact
solution (~340 ms).  The user always sees the exact result; the preview
exists only to keep the interface responsive during continuous motion.

This pattern is pervasive:
  - Drag of A/B handles: snap preview → debounce → exact topology insertion.
  - Drag of P (translate): fast shoot → debounce → exact shoot with
    parallel-transported basis.
  - Cursor crosshair: cached geodesic shoots, refreshed when the cursor
    moves beyond a pixel threshold.

Visual feedback reflects the state: preview lines are thin (lw 1, opacity
0.6); consolidated lines are full width (lw 2, opacity 1.0).

Robustness: surface picking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
VTK's ``vtkStaticCellLocator`` occasionally returns inconsistent
(point, face_id) pairs on irregular meshes — the intersection point does
not lie on the reported face (barycentric coords wildly outside [0, 1]).

``_pick()`` defends against this with a three-level strategy:

  1. **IntersectWithLine** — fast ray pick, O(log N).
  2. **Barycentric validation** — if min(u,v,w) < -0.1 or max > 1.1,
     the face is wrong.  Fall through to:
  3. **FindClosestPoint + KDTree fallback** — ``find_face()`` tries the
     locator's closest-point query; if that also fails validation, it
     uses a KDTree nearest-vertex search + barycentric scoring across
     all adjacent faces.

The same validation is applied in ``compute_shoot`` before the first
ray step, as a second line of defense.

Master Clock pattern (debounce)
--------------------------------
VTK only wakes from two sources: hardware events (mouse, keyboard) and
its own timers.  When the mouse is held still during a drag, there is no
hardware event — so we need a timer.  Alternatives evaluated and rejected:

  - **Win32 ``SetTimer`` without callback**: VTK ignores ``WM_TIMER``
    for timer IDs it did not create.
  - **Win32 ``SetTimer`` with ``TIMERPROC``**: VTK's event loop does not
    call ``DispatchMessage`` for external callbacks.
  - **``threading.Timer`` + ``PostMessageW(WM_MOUSEMOVE)``**: synthetic
    mouse messages with coordinates (0,0) are discarded.
  - **``threading.Timer`` + ``InvokeEvent``**: executes synchronously on
    the calling (background) thread — ``wglMakeCurrent`` crash.

The only mechanism that works: **a single ``CreateRepeatingTimer(50)``**
created once at startup, never destroyed, never recreated.  All debounce
logic is managed in Python — the timer just provides a 50 ms heartbeat.

``_on_poll_timer`` iterates ``SessionState.pending_debounces`` every tick
and fires all callbacks whose ``perf_counter`` deadline has expired.
A single ``render()`` is issued at the end of each tick that had work,
batching multiple consolidations into one frame.

To register a new debounce task from anywhere in the code::

    self.state.pending_debounces['my_task'] = (
        time.perf_counter() + delay_seconds,
        self._my_callback,       # called on main thread, no render()
    )

To cancel: ``self.state.pending_debounces.pop('my_task', None)``.

The drag consolidation uses task id ``'drag_exact'`` — see
``_schedule_debounce`` and ``_fire_debounce``.

Implementation details:
  - The timer is created from a one-shot ``RenderEvent`` callback in
    ``run()`` — VTK silently ignores timers created before ``Start()``.
  - ``_lock_camera()`` uses a bare ``vtkInteractorStyle`` (not ``None``)
    to keep the event loop and timers alive during drag.
  - ``_on_poll_timer()`` ignores the timer ID (VTK returns 0 on some
    backends).

Keyboard handling
-----------------
ALL default VTK and PyVista key bindings are permanently disabled:
  - PyVista callbacks are cleared and ``reset_key_events`` is monkey-patched
    to a no-op so ``show()`` cannot re-register them.
  - A no-op observer is installed on ``CharEvent`` with priority 100 on the
    interactor style, preventing VTK's C++ ``OnChar()`` from ever running
    (this is what maps ``q`` → TerminateApp, ``e`` → ExitApp, etc.).
  - After every camera style swap (``_unlock_camera``), the CharEvent block
    is re-applied to the restored style.

To add custom keyboard shortcuts, use ``self.plotter.add_key_event(key, fn)``
which hooks into PyVista's ``KeyPressEvent`` layer — unaffected by the
CharEvent block.

Spatial locator
---------------
All surface queries (ray picking, closest-point projection, face lookup)
go through a single ``vtkStaticCellLocator`` built once by ``GeodesicMesh``
and exposed as ``self.geo.locator``.  The ``_pick()`` method uses it for
O(log N) ray–mesh intersection via ``IntersectWithLine``.  Extensions that
need surface projection or containment tests should reuse this locator
rather than creating their own.

Subclass extension points
-------------------------
``GeodesicSplineApp`` (geo_splines.py) extends this class.  To avoid
duplicating infrastructure, the following hooks are exposed:

  - ``_try_hit_marker(x, y)`` — vectorized hit-test using the pre-built
    ``_hover_pts_3d`` cache.  Returns True and initiates drag if a handle
    is hit.  Subclasses override to add spline-index switching.
  - ``pick_override`` — keyword argument on ``_on_move``.  Subclasses
    that pick once per frame pass the result via
    ``super()._on_move(obj, event, pick_override=result)`` to avoid
    a redundant O(log N) ray-cast inside the parent.
  - ``_frame_pick`` — set by ``_on_move`` to the (q, cid) result used
    in the current frame.  Available after ``super()._on_move()`` returns
    for subclass post-processing.  ``(None, None)`` when the pick was
    skipped (hovering a marker, no active drag).

Next steps
----------
  - Spline tangent continuity enforcement (G1/C1 constraints at nodes).
  - Dense geodesic sampling for texture/UV mapping along splines.
  - Real-time mesh deformation with spline constraints.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv
import vtk

from geodesics import GeodesicMesh
from gizmo import GeodesicSegment, WARN_SHOOT, _color_rgb


# Module-level logger.  No handler is attached here; geo_splines
# configures the root for the editor session.  Keep WARNING by default
# so VTK observer-callback failures (which we cannot let propagate, but
# still must surface) reach the user via the same channel as the rest
# of the diagnostics.
log = logging.getLogger("geo_shoot")

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


@njit(cache=True, fastmath=True)
def _to_screen_kernel(pts, M, vp_w, vp_h, vp_ox, vp_oy, out_screen):
    """World-to-screen projection kernel — replaces np.dot + slice ops.

    For small batches (N < ~50, typical of hover caches and circle points)
    the per-call Python→C overhead of ``np.dot`` dominates over the actual
    arithmetic.  This kernel does the 4-component matrix multiply, perspective
    divide, and viewport mapping in a single tight loop with zero intermediate
    arrays.
    """
    n = pts.shape[0]
    for i in range(n):
        x = pts[i, 0]; y = pts[i, 1]; z = pts[i, 2]
        cx = M[0, 0]*x + M[0, 1]*y + M[0, 2]*z + M[0, 3]
        cy = M[1, 0]*x + M[1, 1]*y + M[1, 2]*z + M[1, 3]
        cw = M[3, 0]*x + M[3, 1]*y + M[3, 2]*z + M[3, 3]
        if abs(cw) < 1e-12:
            cw = 1e-12
        inv_w = 1.0 / cw
        out_screen[i, 0] = (cx * inv_w * 0.5 + 0.5) * vp_w + vp_ox
        out_screen[i, 1] = (cy * inv_w * 0.5 + 0.5) * vp_h + vp_oy


@njit(cache=True, fastmath=True)
def _hover_argmin_sq(screen_pts, n, mx, my):
    """Finds the index of the closest screen point to (mx, my).

    Returns ``(best_idx, best_sq_dist)``.  Replaces ``np.argmin`` on a
    temporary squared-distance array — avoids creating the intermediate
    array for the small N typical of hover caches (< 50 markers).
    """
    best_idx = 0
    best_sq = 1e30
    for i in range(n):
        dx = screen_pts[i, 0] - mx
        dy = screen_pts[i, 1] - my
        sq = dx * dx + dy * dy
        if sq < best_sq:
            best_sq = sq
            best_idx = i
    return best_idx, best_sq


@njit(cache=True, fastmath=True)
def _closest_seg_on_polyline_2d(screen_pts, n_pts, mx, my):
    """Finds the closest point on a 2D polyline to screen position (mx, my).

    Tests every segment (not just vertices) using perpendicular projection
    clamped to [0, 1].  Returns ``(min_sq_dist, seg_idx, frac)`` where
    the closest 3-D point is::

        pts_3d[seg_idx] * (1 - frac) + pts_3d[seg_idx + 1] * frac
    """
    best_sq = 1e30
    best_seg = 0
    best_frac = 0.0
    for i in range(n_pts - 1):
        # Segment P0→P1
        vx = screen_pts[i + 1, 0] - screen_pts[i, 0]
        vy = screen_pts[i + 1, 1] - screen_pts[i, 1]
        wx = mx - screen_pts[i, 0]
        wy = my - screen_pts[i, 1]
        vv = vx * vx + vy * vy
        if vv < 1e-12:
            t = 0.0
        else:
            t = (wx * vx + wy * vy) / vv
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
        cx = screen_pts[i, 0] + t * vx - mx
        cy = screen_pts[i, 1] + t * vy - my
        sq = cx * cx + cy * cy
        if sq < best_sq:
            best_sq = sq
            best_seg = i
            best_frac = t
    return best_sq, best_seg, best_frac


@dataclass
class UIConfig:
    """Centralized visual tokens and thresholds."""
    # Colors
    COLOR_MESH: str = 'white'
    COLOR_CURSOR: str = 'darkgoldenrod'
    COLOR_CROSSHAIR: str = 'darkslategrey'
    COLOR_HOVER: str = 'black'
    
    # Appearance
    CURSOR_OPACITY: float = 0.8
    CROSSHAIR_OPACITY: float = 0.5
    CURSOR_LINE_WIDTH: float = 1.5
    CROSSHAIR_LINE_WIDTH: float = 1.0
    
    # Interaction
    PICK_TOLERANCE: int = 15
    CURSOR_SCALE: float = 0.012            # Relative to camera dist
    TELESCOPE_FACTOR: float = 1.3
    THROTTLE_MIN: float = 0.012
    THROTTLE_MAX: float = 0.1
    CROSSHAIR_REUSE_PX: float = 1.5       # Screen-pixel threshold to reuse cached crosshair
    CIRCLE_REUSE_PX: float = 2.0          # Screen-pixel threshold to reuse cached circle projection
    RODRIGUES_DOT_THRESHOLD: float = 0.9999  # Normal-change threshold for cursor rotation recalc

    # Derived — computed from their linear counterparts in __post_init__
    # so they never go out of sync with the source values.
    PICK_TOLERANCE_SQ: int = field(init=False)
    CROSSHAIR_REUSE_PX_SQ: float = field(init=False)
    CIRCLE_REUSE_PX_SQ: float = field(init=False)

    def __post_init__(self):
        self.PICK_TOLERANCE_SQ = self.PICK_TOLERANCE ** 2
        self.CROSSHAIR_REUSE_PX_SQ = self.CROSSHAIR_REUSE_PX ** 2
        self.CIRCLE_REUSE_PX_SQ = self.CIRCLE_REUSE_PX ** 2


@dataclass
class SessionState:
    """Manages the current interaction lifecycle.

    ``pending_debounces`` is the extensible registry for the Master Clock
    pattern.  Each entry is ``{task_id: (deadline, callback)}``.  The
    heartbeat timer (``_on_poll_timer``) iterates this dict every 50 ms
    and fires expired callbacks.

    To add a new debounce type::

        self.state.pending_debounces['my_task'] = (
            time.perf_counter() + delay_sec,
            self._my_callback,
        )

    The callback will run on the main thread when the deadline expires.
    To cancel: ``del self.state.pending_debounces['my_task']``.
    """
    hover_seg: GeodesicSegment | None = None
    hover_marker: str | None = None

    active_seg: GeodesicSegment | None = None
    drag_marker: str | None = None

    # Performance
    last_move_t: float = 0.0
    interaction_dt: float = 0.02

    # Last drag position (used by debounce callbacks)
    last_drag_q: np.ndarray | None = None
    last_drag_cid: int | None = None

    # Master Clock debounce registry: {task_id: (deadline, callback)}
    pending_debounces: dict = field(default_factory=dict)

    # Pixel culling (tuple, not ndarray — avoids numpy in hot path)
    _last_mouse_px: tuple | None = None

    # VP matrix cache
    _vp_matrix: np.ndarray | None = None
    _vp_cam_mtime: int = 0
    _vp_ox: float = 0.0
    _vp_oy: float = 0.0
    _vp_w: float = 0.0
    _vp_h: float = 0.0


class SurfaceCursor:
    """Surface-aligned cursor with geodesic crosshair and projected circle.

    Self-contained visual component that manages its own VTK actors,
    geometry buffers, and caching state.  Extracted from
    ``MidpointShooterApp`` to keep the app class focused on interaction
    logic.

    Parameters
    ----------
    plotter : pv.Plotter
        The live plotter where actors are added.
    geo : GeodesicMesh
        Mesh solver — used for ``compute_shoot`` (crosshair),
        ``get_interpolated_normal``, and ``project_smooth_batch`` (circle).
    cfg : UIConfig
        Visual tokens and thresholds.
    to_screen_fn : callable (ndarray → ndarray)
        Maps a 3-D world point to a 2-D screen pixel.  Provided by the
        host app (backed by its VP matrix cache).

    Usage::

        cursor = SurfaceCursor(plotter, geo, cfg, app._to_screen_single)
        cursor.setup_actors()
        # in _on_move:
        needs_render = cursor.update(q, cid, hide=False)
    """

    def __init__(self, plotter: pv.Plotter, geo: GeodesicMesh,
                 cfg, to_screen_fn):
        self._plotter = plotter
        self._geo = geo
        self._cfg = cfg
        self._to_screen = to_screen_fn

        # Crosshair cache
        self._crosshair_screen: np.ndarray | None = None
        self._crosshair_cid: int | None = None
        self._crosshair_valid: bool = False

        # Circle projection cache
        self._circle_screen: np.ndarray | None = None
        self._circle_cid: int | None = None

        # Crosshair geometry buffers (4 rays × max_steps=10 → up to 44 pts)
        self._cross_pts_buf = np.empty((44, 3), dtype=float)
        self._cross_lines_buf = np.empty(60, dtype=int)

        # Rodrigues rotation pre-allocated buffers
        self._z_axis = np.array([0.0, 0.0, 1.0])
        self._K_buf = np.empty((3, 3), dtype=float)
        self._rodrigues_buf = np.empty((3, 3), dtype=float)
        self._cursor_R: np.ndarray | None = None
        self._cursor_n_cache: np.ndarray | None = None

        # VTK actors — created in setup_actors()
        self._circle_template = None
        self._circle_mesh = None
        self._circle_actor = None
        self._lines_mesh = None
        self._lines_actor = None
        self._circle_pts_buf: np.ndarray | None = None
        self._circle_pts_buf2: np.ndarray | None = None
        # Python-side visibility tracking (avoids VTK GetVisibility calls)
        self._circle_visible: bool = False
        self._lines_visible: bool = False

    def setup_actors(self) -> None:
        """Creates VTK actors for circle and crosshair.

        Must be called after the plotter is initialized and before the
        first ``update()`` call.
        """
        cfg = self._cfg
        self._circle_template = pv.Circle(radius=1.0, resolution=32)
        self._circle_mesh = self._circle_template.copy()
        self._circle_actor = self._plotter.add_mesh(
            self._circle_mesh, color=cfg.COLOR_CURSOR,
            style='wireframe', line_width=cfg.CURSOR_LINE_WIDTH,
            opacity=cfg.CURSOR_OPACITY, lighting=False, pickable=False)

        self._lines_mesh = pv.PolyData()
        self._lines_actor = self._plotter.add_mesh(
            self._lines_mesh, color=cfg.COLOR_CROSSHAIR,
            line_width=cfg.CROSSHAIR_LINE_WIDTH,
            opacity=cfg.CROSSHAIR_OPACITY, lighting=False, pickable=False)

        for actor in [self._circle_actor, self._lines_actor]:
            actor.SetVisibility(False)
            mapper = actor.GetMapper()
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0, -6.0)
            mapper.SetRelativeCoincidentTopologyPointOffsetParameter(-6.0)

        n_circle = len(self._circle_template.points)
        self._circle_pts_buf = np.empty((n_circle, 3), dtype=float)
        self._circle_pts_buf2 = np.empty((n_circle, 3), dtype=float)

    def update(self, q: np.ndarray | None, cid: int | None,
               hide: bool) -> bool:
        """Updates cursor position.  Returns True if a render is needed.

        Computes cursor size from camera distance, then delegates to
        ``_update_crosshair`` and ``_update_cursor_circle``.
        """
        if q is None or hide:
            changed = False
            if self._circle_visible:
                self._circle_actor.SetVisibility(False)
                self._circle_visible = False
                changed = True
            if self._lines_visible:
                self._lines_actor.SetVisibility(False)
                self._lines_visible = False
                changed = True
            return changed

        n = self._geo.get_interpolated_normal(q, cid)
        cam = self._plotter.camera.position
        dx = cam[0] - q[0]; dy = cam[1] - q[1]; dz = cam[2] - q[2]
        cursor_sz = np.sqrt(dx*dx + dy*dy + dz*dz) * self._cfg.CURSOR_SCALE

        self._update_crosshair(q, cid, n, cursor_sz)
        self._update_cursor_circle(q, cid, n, cursor_sz)
        return True

    def _update_crosshair(self, q: np.ndarray, cid: int,
                           n: np.ndarray, cursor_sz: float) -> None:
        """Updates the geodesic crosshair, cached by screen-pixel distance and face id.

        On cache hit, the existing VTK geometry is reused without any
        Python-side copy.  On cache miss, four short geodesic rays are
        computed and the mesh is updated in-place.
        """
        screen_q = self._to_screen(q)
        cache_hit = False
        if self._crosshair_screen is not None and self._crosshair_cid == cid:
            _cdx = screen_q[0] - self._crosshair_screen[0]
            _cdy = screen_q[1] - self._crosshair_screen[1]
            cache_hit = (_cdx * _cdx + _cdy * _cdy) < self._cfg.CROSSHAIR_REUSE_PX_SQ

        if not cache_hit:
            ref = np.array([0, 0, 1.0])
            if abs(np.dot(ref, n)) > 0.9: ref = np.array([1, 0, 0])
            u = np.cross(n, ref); u /= np.linalg.norm(u); v = np.cross(n, u)

            n_pts, n_lines = 0, 0
            shoot_dist = cursor_sz * self._cfg.TELESCOPE_FACTOR
            for dvec in [u, -u, v, -v]:
                path = self._geo.compute_shoot(q, dvec, shoot_dist, cid, max_steps=10, fast_mode=True)
                if path is not None:
                    k = len(path)
                    self._cross_pts_buf[n_pts:n_pts + k] = path
                    self._cross_lines_buf[n_lines] = k
                    self._cross_lines_buf[n_lines + 1:n_lines + 1 + k] = range(n_pts, n_pts + k)
                    n_lines += 1 + k
                    n_pts += k

            self._crosshair_valid = n_pts > 0
            if n_pts > 0:
                self._lines_mesh.points = self._cross_pts_buf[:n_pts]
                self._lines_mesh.lines = self._cross_lines_buf[:n_lines]
                self._lines_mesh.Modified()

            self._crosshair_screen = screen_q.copy()
            self._crosshair_cid = cid

        if self._crosshair_valid != self._lines_visible:
            self._lines_actor.SetVisibility(self._crosshair_valid)
            self._lines_visible = self._crosshair_valid

    def _update_cursor_circle(self, q: np.ndarray, cid: int,
                               n: np.ndarray, cursor_sz: float) -> None:
        """Updates the cursor circle via Rodrigues rotation + surface projection.

        Both the rotation matrix (cached by normal) and the projected circle
        (cached by screen-pixel distance + face id) are reused on cache hit.
        """
        screen_q = self._to_screen(q)
        cache_hit = False
        if self._circle_screen is not None and self._circle_cid == cid:
            _cdx = screen_q[0] - self._circle_screen[0]
            _cdy = screen_q[1] - self._circle_screen[1]
            cache_hit = (_cdx * _cdx + _cdy * _cdy) < self._cfg.CIRCLE_REUSE_PX_SQ

        if cache_hit:
            if not self._circle_visible:
                self._circle_actor.SetVisibility(True)
                self._circle_visible = True
            return

        if self._cursor_n_cache is None \
                or np.dot(self._cursor_n_cache, n) < self._cfg.RODRIGUES_DOT_THRESHOLD:
            z = self._z_axis
            dot = np.clip(np.dot(z, n), -1.0, 1.0)
            axis = np.cross(z, n)
            axis_n = np.linalg.norm(axis)
            if axis_n > 1e-9:
                axis /= axis_n
                theta = np.arccos(dot)
                K = self._K_buf
                K[0, 0] = 0;        K[0, 1] = -axis[2]; K[0, 2] = axis[1]
                K[1, 0] = axis[2];  K[1, 1] = 0;        K[1, 2] = -axis[0]
                K[2, 0] = -axis[1]; K[2, 1] = axis[0];  K[2, 2] = 0
                sin_t, cos_t = np.sin(theta), np.cos(theta)
                np.dot(K, K, out=self._rodrigues_buf)
                self._rodrigues_buf *= (1 - cos_t)
                self._rodrigues_buf += K * sin_t
                self._rodrigues_buf[0, 0] += 1.0
                self._rodrigues_buf[1, 1] += 1.0
                self._rodrigues_buf[2, 2] += 1.0
                self._cursor_R = self._rodrigues_buf
            elif dot < 0:
                self._cursor_R = np.diag([1.0, 1.0, -1.0])
            else:
                self._cursor_R = None
            self._cursor_n_cache = n.copy()

        np.multiply(self._circle_template.points, cursor_sz, out=self._circle_pts_buf)
        if self._cursor_R is not None:
            np.dot(self._circle_pts_buf, self._cursor_R.T, out=self._circle_pts_buf2)
            self._circle_pts_buf2 += q
            buf = self._circle_pts_buf2
        else:
            self._circle_pts_buf += q
            buf = self._circle_pts_buf

        eps = cursor_sz * 0.1
        pts_proj = self._geo.project_smooth_batch(buf)
        pts_proj += n * eps

        self._circle_mesh.points = pts_proj
        self._circle_mesh.Modified()
        if not self._circle_visible:
            self._circle_actor.SetVisibility(True)
            self._circle_visible = True

        self._circle_screen = screen_q.copy()
        self._circle_cid = cid


class MidpointShooterApp:
    """
    Core application for interactive geodesic shooting on 3D meshes.

    Args:
        mesh_path: Filename of the .ply or .obj surface to load.
    """
    # Maximum capacity for hover marker buffers.  Doubled on overflow.
    _HOVER_CAPACITY: int = 512

    def __init__(self, mesh_path: str):
        self._load_mesh(mesh_path)
        self.cfg = UIConfig()
        self.state = SessionState()
        self.segments: list[GeodesicSegment] = []

        # --- Pre-allocated buffers (see 'Hot-path allocation discipline') ---
        self._hover_pts_3d = np.empty((self._HOVER_CAPACITY, 3), dtype=float)
        self._hover_tags: list[tuple[GeodesicSegment, str]] = []
        self._hover_n: int = 0
        self._hover_dirty: bool = False
        self._screen_buf = np.empty((self._HOVER_CAPACITY, 2), dtype=float)
        self._single_screen_buf = np.empty(2, dtype=float)
        self._vp_matrix_buf = np.empty((4, 4), dtype=float)

        # Pre-allocated pick result buffer (avoids np.array() per pick)
        self._pick_result_buf = np.empty(3, dtype=float)

        # Lazy VTK helpers — explicit None avoids hasattr checks in hot paths.
        self._vtk_coord: vtk.vtkCoordinate | None = None
        self._saved_style: vtk.vtkInteractorStyle | None = None

        self._setup_plotter()
        # Surface cursor — self-contained component (circle + crosshair)
        self._cursor = SurfaceCursor(self.plotter, self.geo, self.cfg,
                                     self._to_screen_single)
        self._cursor.setup_actors()
        self._setup_interaction()
        self._setup_state()
        self._print_help()

    def _load_mesh(self, mesh_or_path) -> None:
        """Loads and pre-processes the mesh for geodesic computation.

        Accepts a file path (str) or a pre-built ``pv.PolyData`` mesh.
        """
        if isinstance(mesh_or_path, str):
            print(f"[*] Loading mesh: {mesh_or_path}...")
            if not os.path.exists(mesh_or_path):
                raise FileNotFoundError(f"Mesh file '{mesh_or_path}' not found.")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=pv.PyVistaFutureWarning)
                self.mesh = pv.read(mesh_or_path).extract_surface().triangulate().clean()
        else:
            print("[*] Using provided mesh...")
            self.mesh = mesh_or_path

        self.geo = GeodesicMesh(self.mesh)
        
        # Diagonal for scale-dependent thresholds
        b = self.mesh.bounds
        self.diag = np.sqrt((b[1]-b[0])**2 + (b[3]-b[2])**2 + (b[5]-b[4])**2)

    def _setup_plotter(self) -> None:
        """Initializes the PyVista plotter and lighting."""
        pv.global_theme.allow_empty_mesh = True
        self.plotter = pv.Plotter(title="Midpoint Geodesic Shooter")
        self.plotter.remove_all_lights()
        self.plotter.add_light(pv.Light(light_type='headlight', intensity=1.0))
        
        # Main Surface
        self.mesh_actor = self.plotter.add_mesh(
            self.mesh, color='white', opacity=1.0, 
            interpolation='gouraud', pickable=True
        )
        
        # Boundary edges
        self._edge_mesh = self.mesh.extract_feature_edges(
            boundary_edges=True, non_manifold_edges=True,
            feature_edges=False, manifold_edges=False
        )
        self._edge_actor = self.plotter.add_mesh(
            self._edge_mesh, color='magenta', opacity=0.5,
            line_width=3, lighting=False, pickable=False
        )
        self._set_depth_priority(self._edge_actor)

        # Wireframe overlay (toggle: 'w')
        self._wireframe_actor = self.plotter.add_mesh(
            self.mesh, color='#aaaaaa', opacity=0.15,
            style='wireframe', line_width=0.5, pickable=False
        )
        self._set_depth_priority(self._wireframe_actor, -4.0)
        self._wireframe_actor.SetVisibility(False)
        self._wireframe_visible = False

        # Opacity slider
        self._surface_opacity = 1.0
        self._opacity_slider = self.plotter.add_slider_widget(
            self._on_opacity_slider, [0.0, 1.0], value=1.0,
            title="", pointa=(0.0, 0.04 ), pointb=( 0.15 , 0.04 ),
            style='modern', fmt="", interaction_event='always'
        )

        # HUD
        self._hud_actor = self.plotter.add_text(
            "READY", position='upper_left', font_size=12,
            color='white', shadow=True, name="status_hud"
        )
        self._hud_text = "READY"
        self._hud_color = 'white'

    def _setup_interaction(self) -> None:
        """Configures picking, keyboard blocking, and VTK mouse observers."""
        # Disable all default VTK/PyVista key bindings
        self.plotter.iren.clear_key_event_callbacks()
        self.plotter.reset_key_events = lambda: None
        self.plotter.iren.interactor.RemoveObservers("CharEvent")
        self.plotter.iren.interactor.GetInteractorStyle().AddObserver(
            vtk.vtkCommand.CharEvent, lambda obj, ev: None, 100.0)

        # Fallback picker (if locator unavailable)
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.001)
        self.picker.AddPickList(self.mesh_actor)
        self.picker.PickFromListOn()

        # Pre-allocated refs for locator ray pick
        self._pick_t = vtk.reference(0.0)
        self._pick_pt = [0.0, 0.0, 0.0]
        self._pick_pcoords = [0.0, 0.0, 0.0]
        self._pick_sub_id = vtk.reference(0)
        self._pick_cell_id = vtk.reference(0)
        self._pick_cell = vtk.vtkGenericCell()
        self._ray_p0 = np.zeros(3, dtype=float)
        self._ray_p1 = np.zeros(3, dtype=float)

        # Mouse events
        vtki = self.plotter.iren.interactor
        vtki.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self._on_press, 1.0)
        vtki.AddObserver(vtk.vtkCommand.LeftButtonReleaseEvent, self._on_release, 1.0)
        vtki.AddObserver(vtk.vtkCommand.MouseMoveEvent, self._on_move, 1.0)

        self.plotter.add_key_event('e', self._on_export)
        # Delete key removed — node deletion requires spline-aware reconnection
        self.plotter.add_key_event('w', self._toggle_wireframe)
        self.plotter.add_key_event('a', self._cycle_opacity)

    def _setup_state(self) -> None:
        """Registers the Master Clock timer observer.  The actual repeating
        timer is created in ``run()`` via a one-shot ``RenderEvent`` — VTK
        ignores timers created before ``Start()``.
        """
        self._debounce_sec = 0.15
        self.plotter.iren.interactor.AddObserver(
            vtk.vtkCommand.TimerEvent, self._on_poll_timer, 1.0)

    @staticmethod
    def _print_help() -> None:
        print("""
========================================
--- MIDPOINT GEODESIC SHOOTER ---
  Double-click      : New midpoint segment
  Drag Red          : Translate segment
  Drag Blue/Lime    : Symmetric expansion/rotation
  Key 'e'           : Export paths to TXT
  Key 'w'           : Toggle wireframe overlay
  Key 'a'           : Cycle surface transparency
========================================""")

    @staticmethod
    def _set_depth_priority(actor, offset: float = -6.0) -> None:
        """Shifts an actor closer to the camera in z-buffer so it draws on top of surfaces."""
        mapper = actor.GetMapper()
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0, offset)
        mapper.SetRelativeCoincidentTopologyPointOffsetParameter(offset)

    def _is_marker_occluded(self, marker_pos: np.ndarray) -> bool:
        """Returns True if *marker_pos* is occluded by the mesh from the camera view.

        Performs a ray-cast from the camera to the marker. If the first hit on
        the mesh is significantly closer than the marker itself, it is
        considered occluded.
        """
        cam_pos = np.array(self.plotter.camera.position)
        # Ray cast from camera to marker
        hit = self.geo.locator.IntersectWithLine(
            cam_pos, marker_pos, 0.0001,
            self._pick_t, self._pick_pt, self._pick_pcoords,
            self._pick_sub_id, self._pick_cell_id, self._pick_cell
        )
        if hit:
            # First hit position
            hit_pt = np.array([self._pick_pt[0], self._pick_pt[1], self._pick_pt[2]])
            # dist_hit and dist_marker from camera
            dist_hit = np.linalg.norm(hit_pt - cam_pos)
            dist_marker = np.linalg.norm(marker_pos - cam_pos)
            # Threshold: 1e-4 * mesh diagonal to avoid self-occlusion artifacts.
            if dist_hit < dist_marker - self.diag * 1e-4:
                return True
        return False

    # --- UI Helpers ---

    def _pick(self) -> tuple[np.ndarray | None, int | None]:
        """Screen-to-surface ray pick.  Returns (point, cell_id) or (None, None).

        Uses ``IntersectWithLine`` for O(log N) ray–mesh intersection.
        The result is validated via barycentric coords — if the locator
        returns an inconsistent (point, face) pair (known issue on irregular
        meshes), the point is re-projected via ``find_face()`` which applies
        a KDTree fallback.  See module docstring 'Robustness: surface picking'.

        Note: this method implements levels 1–2 of the three-level strategy
        described in the module docstring.  Level 3 (KDTree nearest-vertex
        search + barycentric scoring across adjacent faces) is fully
        encapsulated inside ``find_face()`` and fires automatically when
        the locator's ``FindClosestPoint`` also fails validation.
        """
        if self.geo.locator is None:
            # Fallback to vtkCellPicker if no locator available
            x, y = self.plotter.iren.get_event_position()
            self.picker.Pick(x, y, 0, self.plotter.renderer)
            cid = self.picker.GetCellId()
            if cid == -1: return None, None
            return np.array(self.picker.GetPickPosition()), cid

        x, y = self.plotter.iren.get_event_position()
        renderer = self.plotter.renderer

        # Camera ray: screen (x,y) → world near/far
        renderer.SetDisplayPoint(x, y, 0)
        renderer.DisplayToWorld()
        wp0 = renderer.GetWorldPoint()
        w0 = wp0[3]
        self._ray_p0[0] = wp0[0]/w0; self._ray_p0[1] = wp0[1]/w0; self._ray_p0[2] = wp0[2]/w0

        renderer.SetDisplayPoint(x, y, 1)
        renderer.DisplayToWorld()
        wp1 = renderer.GetWorldPoint()
        w1 = wp1[3]
        self._ray_p1[0] = wp1[0]/w1; self._ray_p1[1] = wp1[1]/w1; self._ray_p1[2] = wp1[2]/w1

        # Intersect
        hit = self.geo.locator.IntersectWithLine(
            self._ray_p0, self._ray_p1, 0.001,
            self._pick_t, self._pick_pt, self._pick_pcoords,
            self._pick_sub_id, self._pick_cell_id, self._pick_cell
        )
        if not hit:
            return None, None

        buf = self._pick_result_buf
        buf[0] = self._pick_pt[0]; buf[1] = self._pick_pt[1]; buf[2] = self._pick_pt[2]
        cid = self._pick_cell_id.get()

        # Validate: ray intersection can return inconsistent (point, cell_id)
        # on irregular meshes.  If bary coords say pt is not on cid, re-project
        # via FindClosestPoint which returns a consistent pair.
        u, v, w = self.geo.get_barycentric(buf, cid)
        if min(u, v, w) < -0.1 or max(u, v, w) > 1.1:
            cid = self.geo.find_face(buf)
            buf[0] = self.geo._vtk_cp[0]; buf[1] = self.geo._vtk_cp[1]; buf[2] = self.geo._vtk_cp[2]

        return buf, cid

    def _to_screen(self, pt3d: np.ndarray) -> np.ndarray:
        """World 3D → display 2D pixels (single point).

        Prefer ``_to_screen_batch`` when projecting multiple points — it
        avoids per-point VTK overhead via a cached VP matrix multiply.
        """
        if self._vtk_coord is None:
            self._vtk_coord = vtk.vtkCoordinate()
            self._vtk_coord.SetCoordinateSystemToWorld()
        self._vtk_coord.SetValue(float(pt3d[0]), float(pt3d[1]), float(pt3d[2]))
        return np.array(self._vtk_coord.GetComputedDisplayValue(self.plotter.renderer), dtype=float)

    def _refresh_vp_cache(self) -> None:
        """Rebuilds cached VP matrix when camera MTime changes."""
        renderer = self.plotter.renderer
        cam = renderer.GetActiveCamera()
        mtime = cam.GetMTime()
        if mtime != self.state._vp_cam_mtime:
            vm4 = cam.GetCompositeProjectionTransformMatrix(
                renderer.GetTiledAspectRatio(), 0, 1)
            buf = self._vp_matrix_buf
            e = vm4.GetElement
            for i in range(4):
                buf[i, 0] = e(i, 0); buf[i, 1] = e(i, 1)
                buf[i, 2] = e(i, 2); buf[i, 3] = e(i, 3)
            self.state._vp_matrix = buf

            vp = renderer.GetViewport()
            sz = renderer.GetRenderWindow().GetSize()
            self.state._vp_ox = sz[0] * vp[0]
            self.state._vp_oy = sz[1] * vp[1]
            self.state._vp_w = sz[0] * (vp[2] - vp[0])
            self.state._vp_h = sz[1] * (vp[3] - vp[1])
            self.state._vp_cam_mtime = mtime

    def _to_screen_batch(self, pts_3d: np.ndarray) -> np.ndarray:
        """Vectorized world-to-screen using cached VP matrix.  Returns Nx2 screen pixels.

        Dispatches to the ``@njit`` kernel ``_to_screen_kernel`` which fuses
        the matrix multiply, perspective divide, and viewport mapping into a
        single loop — eliminates ``np.dot`` / slice overhead that dominates
        for the small batches typical of hover caches (N < 50).

        Returns a view into ``_screen_buf`` — valid until the next call.
        Callers that need to persist the result must ``.copy()`` it.
        """
        self._refresh_vp_cache()
        n = len(pts_3d)

        if n > self._screen_buf.shape[0]:
            self._screen_buf = np.empty((n * 2, 2), dtype=float)

        screen = self._screen_buf[:n]
        _to_screen_kernel(pts_3d, self.state._vp_matrix,
                          self.state._vp_w, self.state._vp_h,
                          self.state._vp_ox, self.state._vp_oy, screen)
        return screen

    def _to_screen_single(self, pt3d: np.ndarray) -> np.ndarray:
        """World 3D → display 2D for a single point via cached VP matrix.

        Avoids the buffer-slicing and vectorized-ops overhead of
        ``_to_screen_batch`` when projecting exactly one point.  Performs
        a manual 4-component dot product and writes into the pre-allocated
        ``_single_screen_buf``.

        Returns a view into ``_single_screen_buf`` — valid until the next
        call.  Callers that need to persist the result must ``.copy()`` it.
        """
        self._refresh_vp_cache()
        M = self.state._vp_matrix
        x, y, z = float(pt3d[0]), float(pt3d[1]), float(pt3d[2])
        cx = M[0, 0]*x + M[0, 1]*y + M[0, 2]*z + M[0, 3]
        cy = M[1, 0]*x + M[1, 1]*y + M[1, 2]*z + M[1, 3]
        cw = M[3, 0]*x + M[3, 1]*y + M[3, 2]*z + M[3, 3]
        if abs(cw) < 1e-12:
            cw = 1e-12
        buf = self._single_screen_buf
        buf[0] = (cx / cw * 0.5 + 0.5) * self.state._vp_w + self.state._vp_ox
        buf[1] = (cy / cw * 0.5 + 0.5) * self.state._vp_h + self.state._vp_oy
        return buf

    def _set_hud(self, message: str, color: str = 'white',
                 sticky_seconds: float = 0.0) -> None:
        """Updates the on-screen HUD text.

        Parameters
        ----------
        message : str
            Text to display (uppercased automatically).
        color : str
            Named PyVista color for the text.
        sticky_seconds : float
            When > 0, the message remains protected from being overwritten
            by other ``_set_hud`` calls for that many seconds — used by
            transient but important events (geodesic fallback, save /
            load failure, "nothing to undo") so the user actually sees
            the message before a routine progress / drag update covers
            it.  Subsequent calls with a non-zero ``sticky_seconds`` of
            their own bypass the protection (a newer urgent message wins).

        Sticky window is honoured per-process via ``time.monotonic``.  The
        progress HUD (``_t("computing_orange", ...)``) is *not* sticky and
        therefore won't fire while a stickier message is still on screen,
        which is desirable: the user sees the error first, then the
        progress text resumes once the window expires.
        """
        import time
        now = time.monotonic()
        sticky_until = getattr(self, '_hud_sticky_until', 0.0)
        # Non-sticky callers must wait out an in-progress sticky window.
        # Sticky callers always win (assumed to be at least as important).
        if sticky_seconds <= 0.0 and now < sticky_until:
            return
        msg = message.upper()
        if msg == self._hud_text and color == self._hud_color:
            # Still update the sticky deadline — repeated identical
            # warnings from different code paths should extend protection.
            if sticky_seconds > 0.0:
                self._hud_sticky_until = now + sticky_seconds
            return
        self._hud_text = msg
        self._hud_color = color
        self._hud_actor.SetText(2, msg)  # 2 = upper_left corner
        self._hud_actor.GetTextProperty().SetColor(_color_rgb(color))
        self._hud_sticky_until = now + sticky_seconds if sticky_seconds > 0.0 else 0.0

    # --- Interaction Logic ---

    def _rebuild_hover_cache(self) -> None:
        """Rebuilds the flat marker-position array used by hover detection.

        Called lazily (on dirty flag) when segments are added, removed, or
        repositioned — not on every mouse move.  Writes into the pre-allocated
        ``_hover_pts_3d`` buffer and updates ``_hover_n``.  The buffer is
        doubled if segment count exceeds current capacity.
        """
        n = 0
        self._hover_tags.clear()
        for s in self.segments:
            for tag, pos in [('p', s.origin), ('a', s.p_a), ('b', s.p_b)]:
                if pos is not None:
                    if n >= self._hover_pts_3d.shape[0]:
                        new_cap = self._hover_pts_3d.shape[0] * 2
                        new_buf = np.empty((new_cap, 3), dtype=float)
                        new_buf[:n] = self._hover_pts_3d[:n]
                        self._hover_pts_3d = new_buf
                        self._screen_buf = np.empty((new_cap, 2), dtype=float)
                    self._hover_pts_3d[n] = pos
                    self._hover_tags.append((s, tag))
                    n += 1
        self._hover_n = n
        self._hover_dirty = False

    def _create_segment(self, pt: np.ndarray, cid: int) -> None:
        """Creates a new geodesic segment at the given surface point.

        Computes the tangent frame from the interpolated normal, shoots
        symmetric geodesic rays (B then A), and appends the segment to
        ``self.segments``.  Marks the hover cache dirty so the next mouse
        move picks up the new handles.

        Args:
            pt:  Exact surface point (3D).
            cid: Face id containing *pt*.
        """
        n = self.geo.get_interpolated_normal(pt, cid)
        ref = np.array([0, 0, 1.0])
        if abs(np.dot(ref, n)) > 0.9: ref = np.array([1, 0, 0])
        u = np.cross(n, ref); u /= np.linalg.norm(u); v = np.cross(n, u)

        new_s = GeodesicSegment(pt, cid, n, u, v)
        new_s.h_length = self.diag * 0.05

        path_b = self.geo.compute_shoot(pt, u, new_s.h_length, cid)
        if path_b is not None:
            new_s.p_b, new_s.path_b = path_b[-1], path_b
            self.geo.diagnose_path(path_b, "shoot:B")

            path_a = self.geo.compute_shoot(pt, -u, new_s.h_length, cid)
            if path_a is not None:
                new_s.p_a, new_s.path_a = path_a[-1], path_a
                self.geo.diagnose_path(path_a, "shoot:A")
            elif WARN_SHOOT:
                print(f"[!] shoot:A FAILED  face={cid}  P=({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})")
        elif WARN_SHOOT:
            print(f"[!] shoot:B FAILED  face={cid}  P=({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})")

        self.segments.append(new_s)
        self._hover_dirty = True
        new_s._origin_cache = self.geo.prepare_origin(new_s.origin)
        new_s.update_local_v(self.geo)
        new_s.update_visuals(self.plotter)

    def _try_hit_marker(self, x: int, y: int) -> bool:
        """Hit-test all segment handles against screen position (x, y).

        Uses the pre-built ``_hover_pts_3d`` cache for vectorized
        screen-projection and squared-distance comparison — no per-marker
        VTK coordinate calls.

        Returns True and initiates drag if a handle is within
        ``PICK_TOLERANCE`` pixels; False otherwise.  Subclasses may
        override to add spline-index switching or other pre-drag logic.
        """
        if self._hover_dirty:
            self._rebuild_hover_cache()
        if self._hover_n == 0:
            return False
        pts_2d = self._to_screen_batch(self._hover_pts_3d[:self._hover_n])
        best, best_sq = _hover_argmin_sq(pts_2d, self._hover_n,
                                         float(x), float(y))
        if best_sq >= self.cfg.PICK_TOLERANCE_SQ:
            return False
        
        # Occlusion check: skip if hidden by mesh
        if self._is_marker_occluded(self._hover_pts_3d[best]):
            return False

        seg, tag = self._hover_tags[best]
        self.state.active_seg = seg
        self.state.drag_marker = tag
        seg.is_active = True
        seg.is_dragging = True
        self._lock_camera()
        self._set_hud(f"DRAGGING {tag.upper()}", 'gold')
        seg.update_visuals(self.plotter)
        self.plotter.render()
        return True

    def _on_press(self, obj, event) -> None:
        """Handles left-button press: initiates marker drag or creates a segment.

        Single click near a marker (< PICK_TOLERANCE px) starts a drag on
        that handle via ``_try_hit_marker``.  Double click on the surface
        creates a new geodesic segment at the pick point.
        """
        try:
            x, y = self.plotter.iren.get_event_position()

            is_dbl = self.plotter.iren.interactor.GetRepeatCount() >= 1
            if not is_dbl:
                if self._try_hit_marker(x, y):
                    return

            # Double-click: create new segment
            if is_dbl:
                pt, cid = self._pick()
                if pt is not None:
                    self._create_segment(pt, cid)
                    self.plotter.render()
        except Exception:  # noqa: BLE001 — VTK observer must not propagate
            # VTK's C++ event loop cannot tolerate a raised exception from
            # a Python observer callback (segfault risk).  We deliberately
            # catch any failure, log with full traceback, and return
            # cleanly.  log.exception emits at ERROR with traceback —
            # equivalent to the previous bare prints but routed through
            # the standard logger.
            log.exception("press handler failed")

    def _detect_hover(self, x: int, y: int) -> tuple[GeodesicSegment | None, str | None, bool]:
        """Tests all segment handles against screen position for hover highlight.

        Returns (hovered_segment, marker_tag, needs_render).  Updates the
        segment's visual state when hover changes.  Reads from the pre-built
        ``_hover_pts_3d`` cache (dirty-flag pattern).
        """
        new_h_s, new_h_m = None, None
        if self.state.active_seg is None and self.segments:
            if self._hover_dirty:
                self._rebuild_hover_cache()
            if self._hover_n > 0:
                pts_2d = self._to_screen_batch(self._hover_pts_3d[:self._hover_n])
                best, best_sq = _hover_argmin_sq(pts_2d, self._hover_n,
                                                 float(x), float(y))
                if best_sq < self.cfg.PICK_TOLERANCE_SQ:
                    # Occlusion check: skip if hidden by mesh
                    if not self._is_marker_occluded(self._hover_pts_3d[best]):
                        new_h_s, new_h_m = self._hover_tags[best]

        needs_render = False
        if (new_h_s, new_h_m) != (self.state.hover_seg, self.state.hover_marker):
            if self.state.hover_seg:
                self.state.hover_seg.hover_marker = None
                self.state.hover_seg.update_visuals(self.plotter)
            self.state.hover_seg, self.state.hover_marker = new_h_s, new_h_m
            if self.state.hover_seg:
                self.state.hover_seg.hover_marker = self.state.hover_marker
                self.state.hover_seg.update_visuals(self.plotter)
            needs_render = True

        return new_h_s, new_h_m, needs_render

    def _on_move(self, obj, event, *, pick_override=None) -> None:
        """Main interaction loop: hover detection, surface cursor, drag handling.

        Runs at display refresh rate (60–144 Hz).  Performance-critical:
        all screen-distance checks use squared thresholds, hover markers
        are read from the pre-built ``_hover_pts_3d`` cache, and
        ``_to_screen_batch`` reuses a pre-allocated homogeneous buffer.

        The surface ray-pick (``_pick()``) is skipped when a marker is
        hovered and no drag is active — the cursor would be hidden anyway,
        so the O(log N) ray-cast would be wasted.

        Parameters
        ----------
        pick_override : tuple[ndarray | None, int | None] | None
            When a subclass has already performed the ray-pick for this
            frame, it passes the result here to avoid a redundant O(log N)
            ray-cast.  ``None`` (the default) means "no pre-computed pick,
            ray-cast if needed".  ``(None, None)`` means "the subclass
            picked and got no hit".
        """
        try:
            now = time.perf_counter()
            x, y = self.plotter.iren.get_event_position()

            # Pixel culling
            if self.state._last_mouse_px is not None:
                dx = x - self.state._last_mouse_px[0]
                dy = y - self.state._last_mouse_px[1]
                if dx*dx + dy*dy < 4.0:  # 2px threshold, squared
                    return
            self.state._last_mouse_px = (x, y)

            # Drag throttle
            if self.state.active_seg is not None:
                if (now - self.state.last_move_t) < self.state.interaction_dt:
                    return

            needs_render = False

            # Hover detection
            new_h_s, new_h_m, hover_changed = self._detect_hover(x, y)
            if hover_changed:
                needs_render = True

            # Skip the expensive ray-pick when cursor will be hidden anyway
            # (hovering a marker with no active drag, or subclass requests
            # cursor suppression via _hide_cursor flag).
            hiding_cursor = ((new_h_s is not None)
                             or (self.state.active_seg is not None)
                             or getattr(self, '_hide_cursor', False))
            if hiding_cursor and self.state.active_seg is None:
                self._frame_pick = (None, None)
                cursor_changed = self._cursor.update(None, None, True)
                if cursor_changed:
                    needs_render = True
            else:
                # Use pre-computed pick from subclass, or ray-cast now
                if pick_override is not None:
                    q, cid = pick_override
                else:
                    q, cid = self._pick()
                self._frame_pick = (q, cid)
                cursor_changed = self._cursor.update(q, cid, hiding_cursor)
                if cursor_changed:
                    needs_render = True

                # Active segment drag
                if self.state.active_seg and q is not None:
                    start_proc = time.perf_counter()

                    self.state.active_seg.is_preview = True
                    if self.state.drag_marker == 'p':
                        self.state.active_seg.update_from_p(q, cid, self.geo)
                    elif self.state.drag_marker == 'a':
                        self.state.active_seg.update_from_a(q, self.geo)
                    elif self.state.drag_marker == 'b':
                        self.state.active_seg.update_from_b(q, self.geo)

                    self.state.active_seg.update_visuals(self.plotter)
                    needs_render = True

                    self.state.last_drag_q = q.copy()
                    self.state.last_drag_cid = cid
                    self._schedule_debounce()

                    # Adaptive throttle
                    proc_time = time.perf_counter() - start_proc
                    self.state.interaction_dt = max(self.cfg.THROTTLE_MIN, min(self.cfg.THROTTLE_MAX, proc_time * 1.5))
                    self.state.last_move_t = now

            if needs_render:
                if self.state.active_seg is None:
                    if self.state.hover_seg:
                        self._set_hud(f"READY: {self.state.hover_marker.upper()}", 'white')
                    else:
                        self._set_hud("READY", 'white')
                self.plotter.render()

        except Exception:  # noqa: BLE001 — VTK observer must not propagate
            # Same rationale as press handler: any uncaught exception
            # here would crash VTK's event loop.
            log.exception("interaction loop failed")

    def _schedule_debounce(self) -> None:
        """Registers the drag consolidation in the Master Clock.
        Overwrites any pending 'drag_exact' task with a fresh deadline."""
        self.state.pending_debounces['drag_exact'] = (
            time.perf_counter() + self._debounce_sec,
            self._fire_debounce,
        )

    def _on_poll_timer(self, obj, event) -> None:
        """Master Clock heartbeat (main thread, every ~50 ms).

        Iterates ``pending_debounces`` and fires all expired callbacks.
        A single ``render()`` is issued at the end if anything fired.
        See ``SessionState`` docstring for how to register new tasks.
        """
        if not self.state.pending_debounces:
            return
        now = time.perf_counter()
        fired = False
        for key in list(self.state.pending_debounces):
            dl, cb = self.state.pending_debounces[key]
            if now >= dl:
                del self.state.pending_debounces[key]
                cb()
                fired = True
        if fired:
            self.plotter.render()

    def _fire_debounce(self) -> None:
        """Exact geodesic recalculation (main thread).

        Updates geometry and actors but does NOT call ``render()`` — that
        is batched by ``_on_poll_timer`` after all expired tasks.
        """
        if self.state.active_seg and self.state.last_drag_q is not None:
            q, cid = self.state.last_drag_q, self.state.last_drag_cid
            self.state.active_seg.is_preview = False

            if self.state.drag_marker == 'p':
                self.state.active_seg.update_from_p(q, cid, self.geo, exact=True)
            elif self.state.drag_marker == 'a':
                self.state.active_seg.update_from_a(q, self.geo, exact=True)
            elif self.state.drag_marker == 'b':
                self.state.active_seg.update_from_b(q, self.geo, exact=True)

            self._set_hud("REFINED (EXACT)", 'cyan')
            self.state.active_seg.update_visuals(self.plotter)

    def _on_release(self, obj, event) -> None:
        """Finalizes dragging and releases the camera.

        Uses ``finally`` to guarantee drag state is cleaned up even if the
        exact-recalculation or visual update fails — a stale ``active_seg``
        would leave the camera locked and the UI frozen.

        Subclasses override ``_finalize_release`` for post-drag behavior
        (e.g. span recomputation) — no need to override ``_on_release``.
        """
        if not self.state.active_seg:
            return
        dragged = self.state.active_seg
        try:
            # Fire pending drag debounce synchronously
            entry = self.state.pending_debounces.pop('drag_exact', None)
            if entry:
                entry[1]()
            dragged.is_preview = False
            dragged.is_dragging = False
            self._finalize_release(dragged)
        except Exception:  # noqa: BLE001 — VTK observer must not propagate
            log.exception("release handler failed")
        finally:
            # Guarantee cleanup regardless of success/failure.
            dragged.is_preview = False
            dragged.is_dragging = False
            self.state.active_seg = None
            self.state.drag_marker = None
            self._hover_dirty = True
            self._unlock_camera()
            self.plotter.render()

    def _finalize_release(self, seg: GeodesicSegment) -> None:
        """Post-drag finalization hook.  Override for subclass-specific behavior
        (e.g. span recomputation in ``GeodesicSplineApp``).

        Called inside the ``try`` block of ``_on_release`` — exceptions are
        caught and drag state is always cleaned up by ``finally``.
        """
        seg.is_active = False
        seg.update_visuals(self.plotter)
        self._set_hud("FINISHED", 'lime')

    def _lock_camera(self) -> None:
        """Swaps to a bare vtkInteractorStyle that blocks camera interaction."""
        vtki = self.plotter.iren.interactor
        self._saved_style = vtki.GetInteractorStyle()
        noop_style = vtk.vtkInteractorStyle()
        noop_style.AddObserver(vtk.vtkCommand.CharEvent, lambda o, e: None, 100.0)
        vtki.SetInteractorStyle(noop_style)

    def _unlock_camera(self) -> None:
        """Restores saved interactor style, re-blocking CharEvent."""
        vtki = self.plotter.iren.interactor
        if self._saved_style is not None:
            vtki.SetInteractorStyle(self._saved_style)
            self._saved_style.RemoveObservers("CharEvent")
            self._saved_style.AddObserver(
                vtk.vtkCommand.CharEvent, lambda obj, ev: None, 100.0)

    def _toggle_wireframe(self) -> None:
        """Toggles wireframe overlay with 'w'."""
        self._wireframe_visible = not self._wireframe_visible
        self._wireframe_actor.SetVisibility(self._wireframe_visible)
        self._set_hud("WIREFRAME ON" if self._wireframe_visible else "WIREFRAME OFF", 'gray')
        self.plotter.render()

    def _on_opacity_slider(self, value: float) -> None:
        """Callback for the surface opacity slider."""
        self._surface_opacity = float(value)
        self.mesh_actor.GetProperty().SetOpacity(self._surface_opacity)
        self._edge_actor.GetProperty().SetOpacity(self._surface_opacity)

    def _cycle_opacity(self) -> None:
        """Cycles surface opacity in 0.25 steps."""
        ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        nxt = next((t for t in ticks if t > self._surface_opacity + 1e-3), ticks[0])
        self._surface_opacity = nxt
        self.mesh_actor.GetProperty().SetOpacity(nxt)
        self._edge_actor.GetProperty().SetOpacity(nxt)
        self._opacity_slider.GetRepresentation().SetValue(nxt)
        self._set_hud(f"ALPHA {nxt:.1f}", 'white')
        self.plotter.render()

    def _on_export(self) -> None:
        """Exports all geodesic paths to a text file with HUD progress.

        Updates the HUD text after each segment so the user sees progress
        instead of a frozen UI during large exports.
        """
        if not self.segments:
            print("[!] No segments to export.")
            return

        total = len(self.segments)
        timestamp = int(time.time())
        fname = f"geo_paths_{timestamp}.txt"
        print(f"[*] Exporting {total} paths to {fname}...")
        with open(fname, 'w') as f:
            for i, s in enumerate(self.segments):
                f.write(f"Segment {i} Origin: {s.origin}\n")
                if s.path_a is not None:
                    f.write(f"  Path A ({len(s.path_a)} pts)\n")
                    np.savetxt(f, s.path_a, fmt='    %.6f %.6f %.6f')
                if s.path_b is not None:
                    f.write(f"  Path B ({len(s.path_b)} pts)\n")
                    np.savetxt(f, s.path_b, fmt='    %.6f %.6f %.6f')
                # Progress feedback every 10 segments
                if (i + 1) % 10 == 0 or i + 1 == total:
                    self._set_hud(f"EXPORTING {i+1}/{total}...", 'gold')
                    self.plotter.render()
        self._set_hud(f"EXPORTED {total} PATHS → {fname}", 'gold')
        self.plotter.render()
        print("[*] Export complete.")

    def cleanup(self) -> None:
        """Releases VTK resources that are not automatically freed.

        Call this when the application exits or when resetting the session.
        Without explicit cleanup, long-running sessions accumulate:

          - **Interactor observers** (mouse, timer) that hold Python
            references and fire on every event — causes slowdown over time.
          - **Segment actors** (PolyData + VTK actors) that consume GPU
            memory even when visually removed.
          - **Pending debounce callbacks** that reference stale segments.

        Static resources (mesh actor, cursor, HUD, slider) are owned by
        the Plotter and cleaned up by ``plotter.close()``.
        """
        # Clear all geodesic segment actors
        for seg in self.segments:
            try:
                seg.clear_actors(self.plotter)
            except (AttributeError, RuntimeError) as exc:
                # VTK actor may already be detached or the plotter may
                # have torn down its renderer.  Log at debug — this is
                # cleanup, not a fatal path.
                log.debug("seg.clear_actors during cleanup: %s", exc)
        self.segments.clear()

        # Clear pending debounces
        self.state.pending_debounces.clear()

        # Remove interactor observers (mouse, timer).  The plotter/iren
        # may already be closed when cleanup runs (window X button),
        # in which case the attributes are None — skip silently.
        try:
            vtki = self.plotter.iren.interactor
            if vtki is not None:
                vtki.RemoveObservers(vtk.vtkCommand.LeftButtonPressEvent)
                vtki.RemoveObservers(vtk.vtkCommand.LeftButtonReleaseEvent)
                vtki.RemoveObservers(vtk.vtkCommand.MouseMoveEvent)
                vtki.RemoveObservers(vtk.vtkCommand.TimerEvent)
        except (AttributeError, RuntimeError):
            pass

    def run(self) -> None:
        """Starts the application.  Creates the master clock timer from inside
        the live event loop (one-shot RenderEvent), then enters ``show()``.

        ``cleanup()`` is guaranteed to run when the window closes — whether
        the user closes via the window's X button, ``q`` key (disabled by
        default), or a ``KeyboardInterrupt``.  This prevents leaked VTK
        observers and GPU-resident actors from accumulating.

        See module docstring 'Master Clock pattern' for rationale.
        """
        def _start_master_clock(obj, event):
            obj.RemoveObservers("RenderEvent")  # one-shot
            self.plotter.iren.interactor.CreateRepeatingTimer(50)
        self.plotter.iren.interactor.GetRenderWindow().AddObserver(
            "RenderEvent", _start_master_clock)
        try:
            self.plotter.show()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.cleanup()
            except Exception as exc:  # noqa: BLE001 — teardown best-effort
                # The window may already be closed by VTK; cleanup is a
                # belt-and-braces pass so any exception here is non-fatal.
                log.debug("cleanup in run() finally: %s", exc)


if __name__ == "__main__":
    MESH_FILE = "malla_compleja.ply"
    if not os.path.exists(MESH_FILE):
        print("[!] Default mesh missing, creating primitive sphere...")
        sphere = pv.Sphere(radius=10, theta_resolution=40, phi_resolution=40).triangulate()
        sphere.save(MESH_FILE)

    app = MidpointShooterApp(MESH_FILE)
    app.run()
