"""
geo_splines.py — Geodesic Spline Editor.

Interactive multi-spline editor on 3D meshes.  Extends ``MidpointShooterApp``
(geo_shoot.py) with multi-node spline chains, closed loops, and hybrid
geodesic Bézier curve rendering.

Architecture
------------
``GeodesicSplineApp`` inherits all infrastructure from the parent — plotter,
surface cursor, picking, hover detection, debounce — and overrides:

  - ``_on_press`` / ``_on_move`` — multi-node click/drag.
  - ``_finalize_release`` — span recomputation after drag ends.
  - ``_try_hit_marker`` — extends the parent's hover-cache hit-testing with
    spline-index switching (selects the spline that owns the hit marker).
  - ``_fire_debounce`` — adds span recomputation on top of the parent's
    exact geodesic recalculation.
  - ``_print_help`` / ``_setup_interaction`` — additional key bindings.

Each spline node reuses the ``GeodesicSegment`` widget from gizmo.py.
Between consecutive nodes, a **span** is drawn — a cubic Bézier curve
evaluated with the hybrid de Casteljau algorithm.

Spline model
------------
Each span is a cubic Bézier through four control points:

    [node_i.origin,  node_i.p_b,  node_{i+1}.p_a,  node_{i+1}.origin]

The handles ``p_a`` / ``p_b`` are geodesic endpoints computed by shooting
from the node origin — so they lie exactly on the surface with known
geodesic paths connecting them to their node.

The de Casteljau evaluation (``GeodesicMesh.hybrid_de_casteljau_curve``)
is **hybrid**: level-1 lerps use geodesic interpolation along precomputed
paths; levels 2–3 use Euclidean lerp with surface re-projection.  This is
not a fully geodesic de Casteljau (that would require recursive geodesic
interpolation at every level), but it produces curves that hug the surface
well and is fast enough for interactive editing.

All span points are batch-projected onto the surface via
``project_smooth_batch`` to guarantee visibility (no z-fighting with the
mesh) and correctness even during fast drag (when ``fast=True`` skips
per-point projection inside de Casteljau).

Secant chord subdivision
~~~~~~~~~~~~~~~~~~~~~~~~
Even after projection, consecutive polyline points that sit on opposite
sides of a mesh ridge (small dihedral angle) produce a straight chord
that passes *through* the mesh interior — visually, the line disappears
behind the surface.  ``GeodesicMesh.subdivide_secant_chords`` detects
these segments by projecting the chord midpoint onto the surface: if the
projected midpoint deviates from the Euclidean midpoint by more than a
tolerance (~10% of mean edge length), the segment is split at the
projected point and both halves are checked recursively (up to 6 levels,
i.e. 64× local refinement).  This runs as a post-processing step on
every displayed polyline (blue consolidated, orange) but is
skipped during drag for performance.

Shared infrastructure from parent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module deliberately avoids duplicating utilities already provided by
``MidpointShooterApp`` and ``gizmo.py``:

  - **Hover detection** reuses the parent's ``_hover_pts_3d`` cache and
    ``_try_hit_marker`` (squared-distance, vectorized screen projection).
  - **Line updates** use the module-level ``update_line_inplace`` from
    gizmo.py — the single canonical implementation shared with
    ``GeodesicSegment.update_visuals``.
  - **Surface pick** passes the pick result to the parent via the
    ``pick_override`` keyword argument on ``_on_move``, avoiding double
    ray-casts: geo_splines picks once per frame and the parent reuses it.

Stitch preview
--------------
When hovering over the surface with an active (non-closed) spline, a
**stitch preview** shows the prospective span from the last node to the
cursor position.  It computes an exact geodesic path from the last node to the cursor
using a topologically-inserted origin (``prepare_origin`` cache) for
~0.01 ms per-frame responsiveness.

LOD and visual feedback during drag
------------------------------------
During drag of a node, affected spans are recomputed at lower resolution
(``DRAG_*`` constants) and with ``fast=True`` (skip per-point projection
in de Casteljau — batch projection still applies).  They are also drawn
with the drag style: lighter color (``SPAN_DRAG_COLOR``), thinner line
width, and reduced opacity — signalling "approximate".

On debounce (mouse pauses 150 ms) the exact solution is computed and the
affected spans revert to full color/width/opacity, even if the mouse is
still held down.  On release, all spans are recomputed at full quality.

Fully geodesic de Casteljau (background)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A second curve layer (orange, thick — ``GEO_COLOR``, ``GEO_LINE_WIDTH``)
evaluates the de Casteljau algorithm with geodesic interpolation at
**every** level, not just level 1.  This is computationally expensive
(~4 ``compute_endpoint`` calls per sample point, ~1.4 s/point) so it
runs in background ``ProcessPoolExecutor`` workers (max 4).

Three curve layers
~~~~~~~~~~~~~~~~~~~
Three parallel curve layers with increasing accuracy and cost:

  - ``_interp_cache`` — **black / interpolation** B-spline (z-depth -6).
    Scipy ``splprep``/``splev`` through node origins, projected onto the
    surface.  No handles, no de Casteljau — purely node-defined.  The
    philosophy is **fast and rough**: a quick-and-dirty curve that gives
    immediate visual feedback of the overall spline shape.  Recomputed
    synchronously on every drag frame (~1-5 ms), no background worker.
    Uses dedicated secant subdivision params (``INTERP_SECANT_*``) that
    are 5× tighter than the Bézier layers, with 200+ initial samples,
    because the 3D B-spline has no geodesic awareness and can deviate
    further from the surface.  Keyed by spline index (one curve per
    spline, not per span).
  - ``_span_cache`` — **blue / Bézier** (z-depth -8).  Dual-mode:
      * During drag: fast hybrid — geodesic lerp on outer paths,
        Euclidean lerp + projection on H_out→H_in (~3 ms per span).
      * On consolidation (debounce fires): semi-geodesic —
        ``compute_endpoint_local`` provides an exact geodesic ``path_12``
        between H_out and H_in, so level-1 is fully geodesic (~25 ms
        per span).  Level-2/3 remain Euclidean+projection.
    This dual-mode strategy keeps interactive drag fluid while giving
    accurate geometry the moment the user stops moving.
  - ``_geo_span_cache`` — **orange / fully geodesic** de Casteljau
    (z-depth -20).  Computed in background (~4-7s per span, ~6× faster
    thanks to ``compute_endpoint_local`` used for all 4 endpoint calls
    per sample point).  Geodesic interpolation at every de Casteljau
    level.  Grows progressively.

The Bézier caches are keyed by ``(spline_id, span_index)``.
The interpolation cache is keyed by ``spline_id`` alone.

Worker architecture
~~~~~~~~~~~~~~~~~~~~

  - ``_geodesic_decasteljau_worker`` — computes fully geodesic
    de Casteljau (orange layer).  Sends points progressively.
  - Runs in ``ProcessPoolExecutor`` child processes (no VTK, no GIL
    contention).  Communication via ``mp.Pipe`` per span.
  - Cancellation: closing the read end → ``BrokenPipeError`` → exit.
  - ``_SpanWorkManager`` — coordinates the process pool, per-span
    pipes, and accumulated points.
  - ``_on_poll_timer`` — drains orange pipes (progressive update).
    Blue is computed synchronously in the main thread inside
    ``_recompute_spans`` — fast enough thanks to
    ``compute_endpoint_local``.

One-frame span lag during drag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
During drag, the parent's ``_on_move`` renders the frame with updated
handle positions.  The child then recomputes hybrid spans — these appear
one frame later (~16 ms at 60 Hz, imperceptible).  This avoids a
double-render per frame that would halve the frame rate.

Visual refresh optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~
``_refresh_visuals`` tracks ``_prev_active_spline_idx`` to avoid iterating
all nodes across all splines on every visual refresh.  Only nodes in the
previously active and currently active splines are visited; span visibility
is toggled only for relevant cache entries.  Falls back to a full sweep
when the previous index is out of range (after spline deletion).

Arrow scale on camera change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Handle arrows use camera distance for fixed-screen-size scaling
(``ARROW_FIXED_SCREEN_SIZE``).  Since ``update_visuals`` only runs on
hover/drag, arrows would go stale during camera rotation or zoom.  The
poll timer (``_on_poll_timer``, every 50 ms) compares ``camera.position``
with a cached value and calls ``refresh_arrows`` on all nodes when it
changes — a lightweight method that only recomputes cone scale and
transform, skipping line and sphere handle updates.

Undo/Redo (Ctrl+Z / Ctrl+Y)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Snapshot-based with **differential restoration** for speed.  Every
mutation (add/delete/insert/drag/close/load) snapshots the minimal state
(origin + tangent per node, plus closed flags — ~48 bytes/node) to the
undo stack.

On Ctrl+Z / Ctrl+Y, ``_restore_snapshot`` compares the target snapshot
with the current state:

  - **Structure match** (same spline count, same node count per spline,
    same closed flags): only nodes whose origin or tangent *actually
    differ* are rebuilt in place via ``_rebuild_node_inplace`` (no actor
    destruction, no full `_load_from_data`).  On a 50-node spline where
    a single node moved, this is ~50× faster than full rebuild.
  - **Structure changed**: falls back to ``_load_from_data`` (clears
    everything, rebuilds from scratch).

This keeps undo/redo responsive on large splines where the full rebuild
(~10 ms × 2 shoots × N nodes) would cause a visible freeze.

Orange progress HUD
~~~~~~~~~~~~~~~~~~~~
When orange workers are active, ``_on_poll_timer`` shows
``COMPUTING ORANGE k/N`` in the HUD where *N* is the batch total and
*k* is the number completed.  Clears automatically when all spans
finish.  The counter is reset between batches (each consolidation or
load is a new batch).

Curve hover and node insertion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When the cursor moves over a visible spline curve, a colored marker
appears at the closest point on the polyline.  The marker color matches
the curve layer (blue, orange, or interp), respecting visual z-priority
(orange > blue > interp when curves overlap).  Double-clicking while the
marker is visible inserts a new C1 node at that point using the de
Casteljau splitting property — the tangent direction comes from the
Bézier derivative at the insertion parameter.

Interaction
-----------
Spline-specific bindings:

  Double-click Left  : Add node / insert at curve hover
  Double-click Right : Start new spline (only if current has nodes)
  Drag Red marker    : Translate node on surface
  Drag Handle        : Adjust tangent direction / length
  Shift + Drag       : Snap drag target to nearest mesh vertex
  C                  : Toggle close/open spline loop (3+ nodes)
  Backspace          : Undo last node / break
  Ctrl+Z             : Undo (full state snapshot)
  Ctrl+Y             : Redo

Layer visibility (keyboard + checkboxes):

  Key 'b'            : Toggle blue (hybrid) curve
  Key 'o'            : Toggle orange (fully geodesic) curve
  Key 'k'            : Toggle black (interpolation) curve

Session:

  Key 's'            : Save splines to timestamped JSON
  Key 'l'            : Load splines from JSON (file dialog)
  Key 't'            : Cycle gizmo opacity (0.2 / 0.4 / 0.7 / 1.0)

Inherited from parent (``MidpointShooterApp``):

  Key 'e'            : Export paths to TXT
  Key 'w'            : Toggle wireframe overlay
  Key 'a'            : Cycle surface transparency

CLI export
----------
``spline_export.py`` reads a saved JSON and outputs curve points to stdout::

    python spline_export.py <file.json> <b|g|o|k> [--samples N]

See ``spline_export.py`` module docstring for format details.

Curvature-aware adaptive sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When ``SplineConfig.ADAPTIVE_SAMPLING`` is True (the default), sample
points are distributed non-uniformly in parameter space, concentrating
near high-curvature regions of the Bézier control polygon.

  - **Phase 1** (a priori, ~O(1) overhead): the two interior angles of the
    control polygon predict curvature peaks at ~t=1/3 and ~t=2/3.  A
    density ``ρ(t) = 1 + k₁·G(t,1/3,σ) + k₂·G(t,2/3,σ)`` is built and
    its CDF inverted to produce non-uniform ``t_vals``.
    See ``GeodesicMesh.curvature_adaptive_t_vals``.
  - **Phase 2** (post-evaluation refinement): after the initial curve
    evaluation, turning angles between consecutive chords are measured;
    parametric midpoints are inserted where the angle exceeds a threshold.
    See ``GeodesicMesh.refine_t_vals_by_curvature``.  Phase 2 runs only
    on consolidated (non-drag) evaluations and background workers.

Set ``ADAPTIVE_SAMPLING = False`` to revert to uniform ``linspace``.
"""

from __future__ import annotations

import atexit
import glob
import json
import multiprocessing as mp
import multiprocessing.shared_memory as _shm
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing.connection import Connection

import numpy as np
import pyvista as pv
import vtk

from geo_shoot import MidpointShooterApp, _hover_argmin_sq, _closest_seg_on_polyline_2d
from geodesics import GeodesicMesh
from scipy.interpolate import splprep, splev
from gizmo import (
    GeodesicSegment,
    update_line_inplace,
    update_dashed_line_inplace,
    safe_remove_actor,
)


# ---------------------------------------------------------------
# Global rendering flags (experimental)
# ---------------------------------------------------------------
# SSAO (Screen Space Ambient Occlusion) darkens crevices under the
# spline, making curves "pop" off the mesh surface.  Trial feature:
# set to True to enable, False to keep the legacy Gouraud look.
# May interact with the depth priority scheme for line actors — try
# both and keep whichever looks better on your mesh.
SSAO_ENABLED: bool = False


# --- Process-local GeodesicMesh for background workers ---
# Each worker process builds its own GeodesicMesh from (V, F) arrays
# passed via the pool initializer.  No VTK objects — workers only need
# compute_endpoint and geodesic_lerp which use KDTree + numpy buffers.
_process_geo: GeodesicMesh | None = None


def _process_initializer(v_shm_name: str, v_shape: tuple, v_dtype: str,
                         f_shm_name: str, f_shape: tuple, f_dtype: str) -> None:
    """Creates a process-local ``GeodesicMesh`` from shared-memory arrays.

    Called once per worker process by ``ProcessPoolExecutor``.  Maps V and
    F from ``multiprocessing.shared_memory.SharedMemory`` blocks created
    by ``_SpanWorkManager`` — zero-copy, no per-process duplication of
    the mesh arrays.  The mesh is built without PyVista, so no VTK
    locator is created.

    Also redirects the child process's stderr to ``os.devnull`` so that
    ``BrokenPipeError`` tracebacks during executor teardown never reach
    the parent's console.
    """
    import sys, os
    # Redirect at the OS file-descriptor level so that even C-level
    # writes (e.g. from multiprocessing internals) go to devnull.
    # This also avoids leaking an open Python file object.
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)  # 2 = stderr fd
    os.close(devnull_fd)
    sys.stderr = open(os.devnull, 'w')  # sync Python wrapper

    global _process_geo
    shm_v = _shm.SharedMemory(name=v_shm_name)
    V = np.ndarray(v_shape, dtype=np.dtype(v_dtype), buffer=shm_v.buf)
    shm_f = _shm.SharedMemory(name=f_shm_name)
    F = np.ndarray(f_shape, dtype=np.dtype(f_dtype), buffer=shm_f.buf)
    # GeodesicMesh copies V and F internally (np.asarray), so the shm
    # mapping can be closed after init without invalidating the mesh.
    _process_geo = GeodesicMesh(V.copy(), F.copy())
    shm_v.close()
    shm_f.close()


@dataclass
class SplineConfig:
    """Centralized spline editing tokens and thresholds."""
    # Bézier curve sampling
    ADAPTIVE_SAMPLING: bool = True   # curvature-aware non-uniform t distribution
    RESOLUTION: float = 0.5
    MIN_SAMPLES: int = 10
    MAX_SAMPLES: int = 60

    # Secant chord subdivision — eliminates chords that cut through mesh ridges
    SECANT_TOL_FACTOR: float = 0.01   # fraction of mean edge length
    SECANT_MAX_DEPTH: int = 6         # max recursive splits (6 → 64× local)

    # LOD during drag: aggressive coarse sampling for real-time feedback
    # (blue gets accuracy back on consolidation via geodesic path_12).
    DRAG_RESOLUTION_FACTOR: float = 3.5
    DRAG_MIN_DIVISOR: int = 3
    DRAG_MAX_DIVISOR: int = 3

    # Geometry
    HANDLE_FRACTION: float = 1 / 3
    INITIAL_H_FRACTION: float = 0.05    # h_length = diag * this
    NORMAL_ALIGN_THRESHOLD: float = 0.9  # dot(ref, n) above which ref is swapped

    # Visuals
    SPAN_COLOR_HEX: str = '#a0a0b8'
    SPAN_LINE_WIDTH: int = 2
    STITCH_SKIP_PX: float = 3.0

    # Drag preview: lighter / thinner spans while dragging (before debounce)
    SPAN_DRAG_COLOR_HEX: str = '#88bbff'
    SPAN_DRAG_LINE_WIDTH: int = 1
    SPAN_DRAG_OPACITY: float = 0.6

    # Fallback indicator: spans whose geodesic degraded to a straight line
    # (cross-component, solver failure) are repainted in saturated red so
    # the user notices a degraded result instead of trusting a phantom curve.
    SPAN_FALLBACK_COLOR_HEX: str = '#ff2020'

    # Fully geodesic de Casteljau (background, ~4-7s per span)
    # 33 = 2^5 + 1 — gives 5 clean binary-subdivision levels (2, 3, 5, 9,
    # 17, 33 points).  Endpoints are already known (node origins) so the
    # worker actually computes 31 points per span.
    GEO_SAMPLES: int = 33
    GEO_COLOR_HEX: str = '#ff8800'           # orange (final, consolidated)
    GEO_COLOR_COMPUTING_HEX: str = '#b85a00' # dimmer orange while computing
    GEO_LINE_WIDTH: int = 3
    GEO_OPACITY: float = 1.0
    MAX_GEO_WORKERS: int = 4        # max concurrent background processes
    # Dashed rendering while the span is still computing: only the
    # odd-indexed segments are drawn, creating a visual "dashes" pattern
    # that tells the user the curve is still being refined.  Disable for
    # a solid-curve-with-dimmer-color look.
    GEO_DASHED_WHILE_COMPUTING: bool = True

    # Interpolation curve (scipy B-spline through nodes, projected to surface).
    # Uses tighter secant subdivision than Bézier layers because the 3D
    # B-spline has no geodesic awareness and can deviate further from the
    # surface between nodes.
    INTERP_COLOR_HEX: str = '#000000'  # black
    INTERP_LINE_WIDTH: int = 2
    INTERP_OPACITY: float = 1.0
    INTERP_MIN_SAMPLES: int = 200      # high base count (short chords)
    INTERP_SECANT_TOL_FACTOR: float = 0.002  # 5× tighter than Bézier layers
    INTERP_SECANT_MAX_DEPTH: int = 8         # 256× local refinement

    # Z-depth priority (polygon offset) per visual layer.
    # Lower = closer to camera = drawn on top.  Layering from back to front:
    # mesh wireframe → interp → blue Bézier → orange Bézier → curve hover marker
    DEPTH_INTERP: float = -6.0
    DEPTH_BLUE: float = -8.0
    DEPTH_STITCH: float = -8.0
    DEPTH_ORANGE: float = -20.0
    DEPTH_CURVE_HOVER: float = -24.0

    # Derived — computed in __post_init__ so they stay in sync
    DRAG_RESOLUTION: float = field(init=False)
    DRAG_MIN_SAMPLES: int = field(init=False)
    DRAG_MAX_SAMPLES: int = field(init=False)
    SPAN_COLOR: tuple = field(init=False)
    SPAN_DRAG_COLOR: tuple = field(init=False)
    SPAN_FALLBACK_COLOR: tuple = field(init=False)
    GEO_COLOR: tuple = field(init=False)
    GEO_COLOR_COMPUTING: tuple = field(init=False)
    INTERP_COLOR: tuple = field(init=False)
    STITCH_SKIP_PX_SQ: float = field(init=False)

    def __post_init__(self):
        self.DRAG_RESOLUTION = self.RESOLUTION * self.DRAG_RESOLUTION_FACTOR
        self.DRAG_MIN_SAMPLES = max(3, self.MIN_SAMPLES // self.DRAG_MIN_DIVISOR)
        self.DRAG_MAX_SAMPLES = self.MAX_SAMPLES // self.DRAG_MAX_DIVISOR
        self.SPAN_COLOR = pv.Color(self.SPAN_COLOR_HEX).float_rgb
        self.SPAN_DRAG_COLOR = pv.Color(self.SPAN_DRAG_COLOR_HEX).float_rgb
        self.SPAN_FALLBACK_COLOR = pv.Color(self.SPAN_FALLBACK_COLOR_HEX).float_rgb
        self.GEO_COLOR = pv.Color(self.GEO_COLOR_HEX).float_rgb
        self.GEO_COLOR_COMPUTING = pv.Color(self.GEO_COLOR_COMPUTING_HEX).float_rgb
        self.INTERP_COLOR = pv.Color(self.INTERP_COLOR_HEX).float_rgb
        self.STITCH_SKIP_PX_SQ = self.STITCH_SKIP_PX ** 2


def _hierarchical_inner_order(total: int) -> list[int]:
    """Inner-index sequence for binary-subdivision progressive refinement.

    Given *total* sample points on ``[0, 1]`` (including the two endpoints
    at positions ``0`` and ``total-1``), returns the **inner** indices
    (the ``total - 2`` points strictly between the endpoints) in the
    order that progressively refines the curve:

      - Level 1: midpoint (``total // 2``)  → 1 point
      - Level 2: quarter points              → 2 points
      - Level 3: eighth points               → 4 points
      - ...

    The endpoints are intentionally excluded: they coincide with the node
    origins and the worker should not recompute them.

    Works for any ``total ≥ 3``; the sequence is optimal when
    ``total == 2**k + 1`` (each level doubles cleanly), which is why
    ``GEO_SAMPLES`` defaults to 33 = 2^5 + 1.
    """
    visited = {0, total - 1}
    order: list[int] = []
    step = (total - 1) // 2
    while step >= 1:
        for idx in range(step, total - 1, 2 * step):
            if idx not in visited:
                order.append(idx)
                visited.add(idx)
        step //= 2
    # Fill any indices the binary loop missed (only possible when
    # ``total - 1`` is not a power of 2; the shape is still progressive).
    for idx in range(1, total - 1):
        if idx not in visited:
            order.append(idx)
    return order


def _geodesic_decasteljau_worker(
    span_key: tuple,
    ctrl: list[np.ndarray],
    path_b: np.ndarray,
    path_a_rev: np.ndarray,
    t_grid: np.ndarray,
    inner_order: list[int],
    writer,
    path_12_cached: np.ndarray | None = None,
) -> None:
    """Background worker: computes fully geodesic de Casteljau points.

    Runs in a ``ProcessPoolExecutor`` child process.  Uses the process-
    local ``_process_geo`` (created by ``_process_initializer``) — no VTK
    objects, no GIL contention with the main thread.

    *t_grid* is the dense linspace in ``[0, 1]`` of length ``n_samples``.
    *inner_order* is the sequence of indices in ``t_grid`` to compute, in
    progressive-refinement order (midpoint first, then quarters, then
    eighths, etc.).  Endpoints (``t_grid[0]`` and ``t_grid[-1]``) are
    NOT computed by the worker — they coincide with node origins and
    are pre-seeded by the main thread.

    Each ``('point', span_key, idx, result)`` message carries the
    **final position** ``idx`` inside the sorted ``t_grid`` array, so the
    main thread stores results in a sparse buffer and renders in
    t-order regardless of arrival order.

    *path_12_cached* is kept for backwards compatibility but is typically
    ``None`` — the worker computes ``path_12`` itself via
    ``compute_endpoint_local`` (~25 ms).

    Results are sent via *writer* (a ``Connection`` from ``mp.Pipe``).
    If the pipe is closed by the main thread (span cancelled), the next
    ``send()`` raises ``BrokenPipeError`` and the worker exits silently.
    """
    geo = _process_geo
    P0, H_out, H_in, P1 = ctrl

    cum_b, total_b = GeodesicMesh.compute_path_lengths(path_b)
    cum_a, total_a = GeodesicMesh.compute_path_lengths(path_a_rev)

    # Track if ANY of the per-point solver calls fell back to a straight
    # line.  The flag is transmitted to the main thread once via the
    # 'done' message so the span can be flagged degraded without one
    # message per sample.
    degraded_any = False

    if path_12_cached is not None and len(path_12_cached) >= 2:
        path_12 = path_12_cached
    else:
        path_12 = geo.compute_endpoint_local(H_out, H_in)
        if path_12 is None or len(path_12) < 2:
            path_12 = np.array([H_out, H_in])
            degraded_any = True
        elif geo._last_was_fallback:
            degraded_any = True
    cum_12, total_12 = GeodesicMesh.compute_path_lengths(path_12)

    try:
        for idx in inner_order:
            t = float(t_grid[idx])
            b01 = GeodesicMesh.geodesic_lerp(path_b, t, cum_b, total_b)
            b12 = GeodesicMesh.geodesic_lerp(path_12, t, cum_12, total_12)
            b23 = GeodesicMesh.geodesic_lerp(path_a_rev, t, cum_a, total_a)

            try:
                path_c0 = geo.compute_endpoint_local(b01, b12)
                if geo._last_was_fallback:
                    degraded_any = True
            except Exception:
                path_c0 = np.array([b01, b12])
                degraded_any = True
            if path_c0 is None or len(path_c0) < 2:
                path_c0 = np.array([b01, b12])
                degraded_any = True

            try:
                path_c1 = geo.compute_endpoint_local(b12, b23)
                if geo._last_was_fallback:
                    degraded_any = True
            except Exception:
                path_c1 = np.array([b12, b23])
                degraded_any = True
            if path_c1 is None or len(path_c1) < 2:
                path_c1 = np.array([b12, b23])
                degraded_any = True

            cum_c0, total_c0 = GeodesicMesh.compute_path_lengths(path_c0)
            cum_c1, total_c1 = GeodesicMesh.compute_path_lengths(path_c1)
            c0 = GeodesicMesh.geodesic_lerp(path_c0, t, cum_c0, total_c0)
            c1 = GeodesicMesh.geodesic_lerp(path_c1, t, cum_c1, total_c1)

            try:
                path_final = geo.compute_endpoint_local(c0, c1)
                if geo._last_was_fallback:
                    degraded_any = True
            except Exception:
                path_final = np.array([c0, c1])
                degraded_any = True
            if path_final is None or len(path_final) < 2:
                path_final = np.array([c0, c1])
                degraded_any = True

            cum_f, total_f = GeodesicMesh.compute_path_lengths(path_final)
            result = GeodesicMesh.geodesic_lerp(path_final, t, cum_f, total_f)

            writer.send(('point', span_key, idx, result))

        writer.send(('done', span_key, degraded_any))
    except (BrokenPipeError, OSError):
        pass  # pipe closed — span was cancelled, exit silently
    finally:
        writer.close()


class _SpanWorkManager:
    """Coordinates background geodesic de Casteljau computation.

    Uses ``ProcessPoolExecutor`` to avoid GIL contention — each worker
    process has its own Python interpreter and ``GeodesicMesh`` instance.

    Communication uses ``mp.Pipe`` per span — no ``mp.Manager`` overhead.
    ``Connection.poll()`` is a non-blocking kernel call (~microseconds)
    so ``drain_queue()`` has zero cost when no results are pending.

    Cancellation: closing the read-end of the pipe causes the worker's
    next ``send()`` to raise ``BrokenPipeError`` and exit.
    """

    def __init__(self, V: np.ndarray, F: np.ndarray, max_workers: int = 4):
        # Share V and F via SharedMemory — avoids pickling ~20MB per
        # worker process on Windows (spawn).  Workers map the shared
        # block and copy into their own heap during init.
        V_c = np.ascontiguousarray(V, dtype=float)
        F_c = np.ascontiguousarray(F, dtype=int)
        self._shm_V = _shm.SharedMemory(create=True, size=V_c.nbytes)
        self._shm_F = _shm.SharedMemory(create=True, size=F_c.nbytes)
        np.ndarray(V_c.shape, dtype=V_c.dtype, buffer=self._shm_V.buf)[:] = V_c
        np.ndarray(F_c.shape, dtype=F_c.dtype, buffer=self._shm_F.buf)[:] = F_c

        self._executor = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_process_initializer,
            initargs=(self._shm_V.name, V_c.shape, str(V_c.dtype),
                      self._shm_F.name, F_c.shape, str(F_c.dtype)))

        # Safety net: if the parent crashes before ``shutdown()`` runs,
        # atexit still fires during interpreter teardown and releases the
        # /dev/shm blocks (on POSIX) so they don't leak across sessions.
        # ``shutdown()`` is idempotent so calling it twice is harmless.
        atexit.register(self.shutdown)

        # --- Orange (fully geodesic) tracking ---
        self._readers: dict[tuple, Connection] = {}
        self._futures: dict[tuple, Future] = {}
        self._points: dict[tuple, list[np.ndarray]] = {}
        self.dirty_spans: set[tuple] = set()
        self.done_spans: set[tuple] = set()  # spans whose worker sent 'done'
        # Spans whose worker reported a geodesic fallback.  The main
        # thread consumes this set after ``drain_queue`` and repaints
        # the affected orange/blue actors in red.
        self.degraded_spans: set[tuple] = set()

        # Spans whose worker died unexpectedly (pipe broken) — main
        # thread should clear the actor geometry on next poll tick.
        self.dead_spans: set[tuple] = set()

        # Spans that are actively being computed (submitted but not yet
        # done/cancelled/dead).  Used by the UI to show a progress HUD.
        self.active_spans: set[tuple] = set()

        # Warm up: force all worker processes to start now
        self._warmup_futures = [
            self._executor.submit(int, 0) for _ in range(max_workers)]

    # --- Fully geodesic (orange) ---

    def submit_span(self, span_key: tuple,
                    ctrl: list[np.ndarray], path_b: np.ndarray,
                    path_a_rev: np.ndarray, n_samples: int,
                    adaptive: bool = False) -> None:
        """Submits a fully geodesic worker.

        Sparse-array protocol: the main thread allocates a per-span list
        of *n_samples* slots pre-seeded with the two endpoints (node
        origins).  The worker fills the remaining slots in hierarchical
        refinement order (midpoint first, then quarters, etc.) — each
        message carries the final index so the renderer can place the
        point in its sorted t-position regardless of arrival order.
        The visible curve therefore refines from coarse to fine instead
        of growing from one end.
        """
        self.cancel_span(span_key)
        reader, writer = mp.Pipe(duplex=False)
        self._readers[span_key] = reader

        # Sparse buffer: indices 0..n_samples-1.  Endpoints are known
        # from ``ctrl[0]`` (P0) and ``ctrl[3]`` (P1); worker fills the
        # interior.  The list is the rendering source of truth — sort +
        # compaction happens in ``get_points``.
        pts = [None] * n_samples
        pts[0]  = np.asarray(ctrl[0], dtype=float)
        pts[-1] = np.asarray(ctrl[3], dtype=float)
        self._points[span_key] = pts

        if adaptive:
            t_grid = GeodesicMesh.curvature_adaptive_t_vals(ctrl, n_samples)
        else:
            t_grid = np.linspace(0.0, 1.0, n_samples)
        inner_order = _hierarchical_inner_order(n_samples)

        future = self._executor.submit(
            _geodesic_decasteljau_worker,
            span_key, ctrl, path_b.copy(), path_a_rev.copy(),
            t_grid, inner_order, writer, None)
        self._futures[span_key] = future
        self.active_spans.add(span_key)

    def cancel_span(self, span_key: tuple) -> None:
        """Closes the pipe for the fully geodesic worker on *span_key*."""
        reader = self._readers.pop(span_key, None)
        if reader is not None:
            reader.close()
        self._futures.pop(span_key, None)
        self._points.pop(span_key, None)
        self.done_spans.discard(span_key)
        self.active_spans.discard(span_key)

    # --- Shared ---

    def cancel_all_for_span(self, span_key: tuple) -> None:
        """Cancels the orange worker for *span_key*."""
        self.cancel_span(span_key)

    def cancel_all(self) -> None:
        """Cancels all active orange workers."""
        for r in self._readers.values():
            r.close()
        self._readers.clear()
        self._futures.clear()
        self._points.clear()
        self.active_spans.clear()

    def drain_queue(self) -> bool:
        """Polls all active orange pipes.  Returns True if any results."""
        had_results = False

        # --- Drain orange (fully geodesic) pipes ---
        for span_key in list(self._readers):
            reader = self._readers.get(span_key)
            if reader is None:
                continue
            try:
                while reader.poll():
                    msg = reader.recv()
                    kind = msg[0]
                    if kind == 'point':
                        _, _, idx, point = msg
                        pts = self._points.get(span_key)
                        # Sparse write: ``idx`` is the final t-position,
                        # not the order of arrival.  Slots already seeded
                        # with endpoints are never overwritten by the
                        # worker (inner_order excludes 0 and n_samples-1).
                        if pts is not None and 0 <= idx < len(pts):
                            pts[idx] = point
                            self.dirty_spans.add(span_key)
                            had_results = True
                    elif kind == 'done':
                        # Payload: ('done', span_key, degraded_any).  Older
                        # 2-field messages are still accepted for safety.
                        degraded = bool(msg[2]) if len(msg) > 2 else False
                        if degraded:
                            self.degraded_spans.add(span_key)
                        else:
                            self.degraded_spans.discard(span_key)
                        self.dirty_spans.add(span_key)
                        self.done_spans.add(span_key)
                        self.active_spans.discard(span_key)
                        had_results = True
                        # Worker exits after done — close reader now
                        # to prevent the normal EOF from being mistaken
                        # for a worker death on the next poll() cycle.
                        reader.close()
                        self._readers.pop(span_key, None)
                        break
            except (EOFError, OSError):
                # Worker died or pipe broken — mark for actor cleanup
                self._readers.pop(span_key, None)
                self.dead_spans.add(span_key)
                self.active_spans.discard(span_key)
                had_results = True
        return had_results

    def get_points(self, span_key: tuple) -> np.ndarray | None:
        """Compacts the sparse per-span buffer into a t-sorted polyline.

        The buffer is a list of length ``n_samples`` pre-seeded with the
        two node-origin endpoints and filled by the worker in
        hierarchical order.  This method drops the ``None`` slots (not
        yet computed) and returns only the populated ones, preserving
        their t-grid order.  Since endpoints are pre-seeded, the result
        always has at least 2 points from the moment the span is
        submitted — the initial render is a straight line between the
        node origins, refined as worker results arrive.
        """
        pts = self._points.get(span_key)
        if not pts:
            return None
        compact = [p for p in pts if p is not None]
        if len(compact) < 2:
            return None
        return np.asarray(compact, dtype=float)

    def shutdown(self) -> None:
        """Cancels all workers, shuts down the process pool, and releases
        shared memory blocks for V and F.  Safe to call multiple times.

        Worker processes have their stderr redirected to ``os.devnull``
        (see ``_process_initializer``), so they cannot print tracebacks.
        The parent-side stderr is also temporarily suppressed during
        ``executor.shutdown()`` to silence internal management threads.
        """
        if getattr(self, '_shutdown_done', False):
            return
        self._shutdown_done = True
        self.cancel_all()
        import sys, os
        devnull = open(os.devnull, 'w')
        old_stderr = sys.stderr
        try:
            sys.stderr = devnull
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Python < 3.9: cancel_futures not supported
            self._executor.shutdown(wait=False)
        finally:
            sys.stderr = old_stderr
            devnull.close()
        for shm_block in (self._shm_V, self._shm_F):
            try:
                shm_block.close()
                shm_block.unlink()
            except Exception:
                pass  # already cleaned up or platform quirk


class GeodesicSplineApp(MidpointShooterApp):

    def __init__(self, mesh_or_path, mesh_label: str | None = None):
        # mesh_label is what goes into the JSON "mesh_file" field.
        # For file paths it's the path itself; for in-memory meshes
        # (e.g. ICOSAHEDRON) it's the sentinel string.
        self.mesh_label = mesh_label if mesh_label is not None else str(mesh_or_path)
        self.scfg = SplineConfig()
        self.splines: list[list[GeodesicSegment]] = [[]]
        self.splines_closed: list[bool] = [False]
        self.active_spline_idx = 0
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []
        self._MAX_UNDO = 50
        self._prev_active_spline_idx = 0
        self._span_cache: dict[tuple, tuple[pv.PolyData, vtk.vtkActor]] = {}
        # Per-span style key (dragging, degraded) — repaints only fire on change.
        self._span_drag_state: dict[tuple, tuple[bool, bool]] = {}
        self._geo_span_cache: dict[tuple, tuple[pv.PolyData, vtk.vtkActor]] = {}
        # Spans whose geodesic solver fell back to a straight line.  Set
        # by ``_recompute_spans`` and orange-worker drain; consumed by
        # ``_set_span`` / ``_set_geo_span`` to repaint in red.
        self._degraded_spans: set[tuple] = set()
        # Interpolation curve: one actor per spline (keyed by spline index)
        self._interp_cache: dict[int, tuple[pv.PolyData, vtk.vtkActor]] = {}
        self._last_stitch_screen: tuple = (0.0, 0.0)
        self._stitch_origin_cache: dict | None = None  # prepare_origin cache for last node
        self._stitch_origin_node_id: int = -1           # id() of node that owns the cache
        self._node_to_spline: dict[int, int] = {}  # id(segment) → spline index
        self._pre_drag_spline_idx: int | None = None
        self._last_cam_pos: tuple = (0.0, 0.0, 0.0)  # for arrow scale refresh
        # _work_mgr created after super().__init__ when self.geo is available

        super().__init__(mesh_or_path)
        self._work_mgr = _SpanWorkManager(
            self.geo.V, self.geo.F, self.scfg.MAX_GEO_WORKERS)
        # Pre-compute secant subdivision tolerance from mesh density
        mean_edge = float(np.sqrt(self.geo._face_edge_len2.mean()))
        self._secant_tol = mean_edge * self.scfg.SECANT_TOL_FACTOR
        self.plotter.set_background('white')

        # Experimental SSAO — see SSAO_ENABLED module flag at top of file.
        if SSAO_ENABLED:
            try:
                self.plotter.enable_ssao()
            except Exception as exc:
                print(f"[!] SSAO unavailable: {exc}")

        # Resolve z-fighting: lines/points always render on top of solid surfaces
        vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
        vtk.vtkMapper.SetResolveCoincidentTopologyPolygonOffsetParameters(1.0, 1.0)

        self._stitch_pd = pv.PolyData()
        self._stitch_actor = self.plotter.add_mesh(
            self._stitch_pd, color='#666666', line_width=1.5,
            opacity=0.6, lighting=False, pickable=False, name="stitch_preview",
        )
        self._set_depth_priority(self._stitch_actor, self.scfg.DEPTH_STITCH)

        self._stitch_actor.SetVisibility(False)

        # Orange computation HUD tracking: total submitted in current batch.
        # Reset to 0 when all orange spans complete.
        self._orange_batch_total = 0
        self._orange_hud_active = False

        # Curve-layer visibility toggles (horizontal row above opacity slider)
        # Orange starts hidden by default (expensive computation, opt-in).
        self._layer_visible = {'blue': True, 'orange': False, 'interp': False}
        self._layer_widgets: dict[str, object] = {}
        self._cb_size = 20
        # Widget layout uses normalized Y (fraction of window height) so
        # checkboxes stay above the slider after resize.  X positions are
        # absolute pixels from the left edge (small, stable).
        self._cb_y_norm = 0.08   # 8% from bottom
        self._cb_x_positions = []  # pixel x per checkbox (fixed)
        defaults = {'blue': True, 'orange': False, 'interp': False}
        for i, (layer, color) in enumerate([
                ('blue', 'blue'), ('orange', 'orange'),
                ('interp', 'black')]):
            x_pos = 11 + i * 22
            self._cb_x_positions.append(x_pos)
            widget = self.plotter.add_checkbox_button_widget(
                lambda v, l=layer: self._toggle_layer(l, v),
                value=defaults[layer],
                position=(x_pos, 50), size=self._cb_size, border_size=2,
                color_on=color, color_off='grey')
            self._layer_widgets[layer] = widget

        # Help button "?" — toggles a shortcut reference overlay
        self._help_visible = False
        self._help_panel = None
        self._help_x = 80
        self._help_widget = self.plotter.add_checkbox_button_widget(
            self._toggle_help_panel,
            value=False,
            position=(self._help_x, 50), size=self._cb_size, border_size=2,
            color_on='white', color_off='grey')
        self._help_label = self.plotter.add_text(
            "?", position=(self._help_x + 5, 52),
            font_size=7, color='white', shadow=True, name="label_help")

        # Reposition widgets on window resize so they stay above the slider.
        self.plotter.iren.interactor.AddObserver(
            'ConfigureEvent', self._on_window_resize)

        # Curve hover marker — single colored sphere, layer-colored.
        self._curve_hover_pd = pv.PolyData(np.zeros((1, 3)))
        self._curve_hover_actor = self.plotter.add_mesh(
            self._curve_hover_pd, color='white', point_size=9,
            render_points_as_spheres=True, lighting=False, pickable=False)
        self._set_depth_priority(self._curve_hover_actor, self.scfg.DEPTH_CURVE_HOVER)
        self._curve_hover_actor.SetVisibility(False)
        self._curve_hover_pt_buf = np.empty((1, 3), dtype=float)
        # Pre-allocated buffer for batched curve hover projection
        self._curve_hover_3d_buf = np.empty((2048, 3), dtype=float)

        # Curve hover state — stored for future node insertion
        self.curve_hover_info: dict | None = None

    # Visual z-priority penalty for curve hover.  When multiple curves
    # overlap on screen, the one rendered on top should win the hover.
    # A small penalty (in squared pixels) is added to lower-priority
    # layers so that the visually topmost curve wins ties.
    _LAYER_HOVER_PENALTY = {'orange': 0.0, 'blue': 3.0, 'interp': 6.0}

    def _detect_curve_hover(self, x: int, y: int) -> bool:
        """Tests proximity of cursor to all visible spline curves.

        Collects all visible curve points into a single pre-allocated
        buffer, projects them to screen in one ``_to_screen_batch`` call,
        then finds the closest segment per polyline via
        ``_closest_seg_on_polyline_2d``.

        When curves overlap on screen (nearly equal distance), the
        layer with higher visual z-priority wins — matching what the
        user sees.  This is achieved by adding a small penalty to
        lower-priority layers' squared distances.

        Returns True if the marker visibility changed (needs render).
        """
        best_sq = self.cfg.PICK_TOLERANCE_SQ
        best_info = None
        best_pt_3d = None
        mx, my = float(x), float(y)

        layer_caches = []
        if self._layer_visible['blue']:
            layer_caches.append(('blue', self._span_cache))
        if self._layer_visible['orange']:
            layer_caches.append(('orange', self._geo_span_cache))

        # --- Phase 1: collect all visible curve points into one buffer ---
        meta = []  # (layer, sid, i, start, n_pts, pts_3d_ref)
        total_n = 0
        for layer, cache in layer_caches:
            for (sid, i), (pd, actor) in cache.items():
                if not actor.GetVisibility():
                    continue
                pts_3d = pd.points
                if pts_3d is None or len(pts_3d) < 2:
                    continue
                n = len(pts_3d)
                if total_n + n > self._curve_hover_3d_buf.shape[0]:
                    new_cap = (total_n + n) * 2
                    self._curve_hover_3d_buf = np.empty((new_cap, 3), dtype=float)
                self._curve_hover_3d_buf[total_n:total_n + n] = pts_3d
                meta.append((layer, sid, i, total_n, n, pts_3d))
                total_n += n
        # Interp cache: keyed by sid (not (sid, span_idx)), uses i=-1 sentinel
        if self._layer_visible['interp']:
            for sid, (pd, actor) in self._interp_cache.items():
                if not actor.GetVisibility():
                    continue
                pts_3d = pd.points
                if pts_3d is None or len(pts_3d) < 2:
                    continue
                n = len(pts_3d)
                if total_n + n > self._curve_hover_3d_buf.shape[0]:
                    new_cap = (total_n + n) * 2
                    self._curve_hover_3d_buf = np.empty((new_cap, 3), dtype=float)
                self._curve_hover_3d_buf[total_n:total_n + n] = pts_3d
                meta.append(('interp', sid, -1, total_n, n, pts_3d))
                total_n += n

        # --- Phase 2: single batched projection ---
        if total_n > 0:
            all_2d = self._to_screen_batch(self._curve_hover_3d_buf[:total_n])

            # --- Phase 3: find closest segment per polyline ---
            for layer, sid, i, start, n, pts_3d in meta:
                penalty = self._LAYER_HOVER_PENALTY[layer]
                sq, seg, frac = _closest_seg_on_polyline_2d(
                    all_2d[start:start + n], n, mx, my)
                effective_sq = sq + penalty
                if effective_sq < best_sq and seg + 1 < n:
                    # Potential hit, but check occlusion
                    p0 = pts_3d[seg]
                    p1 = pts_3d[seg + 1]
                    pt_3d = p0 * (1.0 - frac) + p1 * frac
                    if not self._is_marker_occluded(pt_3d):
                        best_sq = effective_sq
                        best_pt_3d = pt_3d
                        best_info = {
                            'spline_idx': sid, 'span_idx': i,
                            'layer': layer, 'seg': seg, 'frac': frac,
                            'point': best_pt_3d,
                        }

        changed = False
        if best_info is not None:
            self.curve_hover_info = best_info
            # Position and color the marker
            buf = self._curve_hover_pt_buf
            buf[0] = best_pt_3d
            self._curve_hover_pd.points = buf
            self._curve_hover_pd.Modified()
            color_map = {
                'blue': self.scfg.SPAN_COLOR,
                'orange': self.scfg.GEO_COLOR,
                'interp': self.scfg.INTERP_COLOR,
            }
            self._curve_hover_actor.GetProperty().SetColor(
                color_map[best_info['layer']])
            self._curve_hover_actor.SetVisibility(True)
            changed = True  # always render — position moved
        else:
            self.curve_hover_info = None
            if self._curve_hover_actor.GetVisibility():
                self._curve_hover_actor.SetVisibility(False)
                changed = True
        return changed

    def _cycle_gizmo_opacity(self) -> None:
        """Cycles the opacity of all auxiliary visuals (nodes, tangent lines,
        handle arrows, stitch preview) through 0.2 → 0.4 → 0.7 → 1.0 → 0.2.

        Modifies the module-level ``gizmo.GIZMO_OPACITY`` and refreshes
        all segment visuals to apply the new value.
        """
        import gizmo
        ticks = [0.2, 0.4, 0.7, 1.0]
        cur = gizmo.GIZMO_OPACITY
        nxt = next((t for t in ticks if t > cur + 1e-3), ticks[0])
        gizmo.GIZMO_OPACITY = nxt
        # Refresh all segment visuals to pick up the new opacity
        for _, _, node in self._iter_all_nodes():
            node.update_visuals(self.plotter)
        # Update stitch preview if visible
        if self._stitch_actor.GetVisibility():
            self._stitch_actor.GetProperty().SetOpacity(nxt)
        self._set_hud(f"GIZMO OPACITY {nxt:.0%}", 'white')
        self.plotter.render()

    def _toggle_layer_key(self, layer: str) -> None:
        """Keyboard shortcut: inverts the visibility of a curve layer
        and synchronizes the checkbox widget to match.

        Keys: ``b`` (blue), ``o`` (orange), ``k`` (interp).

        Keys: ``b`` (blue), ``o`` (orange), ``k`` (interp).
        """
        new_val = not self._layer_visible[layer]
        self._toggle_layer(layer, new_val)
        widget = self._layer_widgets.get(layer)
        if widget is not None:
            widget.GetRepresentation().SetState(int(new_val))

    def _toggle_layer(self, layer: str, visible: bool) -> None:
        """Checkbox callback: shows or hides all actors in a curve layer.

        *layer* is ``'blue'``, ``'orange'``, or ``'interp'``.
        """
        self._layer_visible[layer] = visible
        cache_map = {
            'blue': self._span_cache,
            'orange': self._geo_span_cache,
        }
        cache = cache_map.get(layer)
        if cache is not None:
            for _, actor in cache.values():
                actor.SetVisibility(visible)
        if layer == 'interp':
            for _, actor in self._interp_cache.values():
                actor.SetVisibility(visible)
        self.plotter.render()

    def _fire_debounce(self) -> None:
        """Exact recalculation + span recomputation + geodesic submit.

        Sequence:
          1. Recomputes the exact geodesic for the dragged segment
             (``update_from_a/b/p`` with ``exact=True``).
          2. Recomputes hybrid Bézier spans (blue) for affected indices.
          3. Calls ``_submit_geodesic_spans`` to start background workers
             for the fully-geodesic orange curve on the same indices.

        The ``is_preview`` flag is set to False before recomputation, so
        hybrid spans revert to full color/width (consolidated appearance)
        and LOD switches to full quality.

        No render() — batched by ``_on_poll_timer``.
        """
        seg = self.state.active_seg
        if seg is not None and self.state.last_drag_q is not None:
            q, cid = self.state.last_drag_q, self.state.last_drag_cid
            seg.is_preview = False

            if self.state.drag_marker == 'p':
                seg.update_from_p(q, cid, self.geo, exact=True)
            elif self.state.drag_marker == 'a':
                seg.update_from_a(q, self.geo, exact=True)
            elif self.state.drag_marker == 'b':
                seg.update_from_b(q, self.geo, exact=True)

            self._recompute_spans(node=seg)
            self._submit_geodesic_spans(node=seg)
            seg.update_visuals(self.plotter)
            # Origin may have moved — stitch cache uses same id() but stale solver
            if id(seg) == self._stitch_origin_node_id:
                self._invalidate_stitch_cache()
            self._set_hud("REFINED (EXACT)", 'cyan')

    def _finalize_release(self, seg: GeodesicSegment) -> None:
        """Post-drag: keep node active, recompute spans, restore active spline.

        If the drag required switching to a different spline (via
        ``_try_hit_marker``), restores the pre-drag active spline index.
        This prevents losing access to empty (break) splines that have
        no clickable nodes.

        Does NOT submit geodesic workers — that is already handled by
        ``_fire_debounce`` which runs synchronously inside ``_on_release``
        before this method.
        """
        seg.is_active = True
        self._recompute_spans(node=seg)
        seg.update_visuals(self.plotter)

        # Restore the active spline from before the drag
        pre = getattr(self, '_pre_drag_spline_idx', None)
        if pre is not None and pre != self.active_spline_idx:
            if 0 <= pre < len(self.splines):
                self.active_spline_idx = pre
                self._refresh_visuals()
        self._pre_drag_spline_idx = None

    def _setup_interaction(self) -> None:
        super()._setup_interaction()
        self.plotter.add_key_event('BackSpace', self._on_backspace)
        self.plotter.add_key_event('c', self._on_close_spline)
        self.plotter.add_key_event('b', lambda: self._toggle_layer_key('blue'))
        self.plotter.add_key_event('o', lambda: self._toggle_layer_key('orange'))
        self.plotter.add_key_event('k', lambda: self._toggle_layer_key('interp'))
        self.plotter.add_key_event('s', self._on_save)
        self.plotter.add_key_event('l', self._on_load)
        self.plotter.add_key_event('t', self._cycle_gizmo_opacity)
        self.plotter.iren.interactor.AddObserver(
            vtk.vtkCommand.RightButtonPressEvent, self._on_right_press, 1.0)
        # Ctrl+Z / Ctrl+Y — raw VTK observer (PyVista add_key_event
        # does not support modifier keys).
        self.plotter.iren.interactor.AddObserver(
            'KeyPressEvent', self._on_key_press_ctrl, 1.0)

    def _print_help(self) -> None:
        print("\n" + "=" * 48)
        print("  GEODESIC SPLINE EDITOR")
        print("  Dbl-click L : Add node    Dbl-click R : New spline")
        print("  Drag Red    : Translate   Drag Handle : Tangents")
        print("  Shift+Drag  : Snap to mesh vertex")
        print("  C           : Close/open loop  Backspace : Undo")
        # Delete key removed — node deletion requires spline-aware reconnection
        print("  b/o/k       : Toggle blue/orange/interp curves")
        print("  t           : Cycle gizmo opacity (20/40/70/100%)")
        print("  s           : Save splines to JSON")
        print("  l           : Load splines from JSON")
        print("  Ctrl+Z      : Undo     Ctrl+Y      : Redo")
        print("=" * 48 + "\n")

    _HELP_TEXT = (
        "  Dbl-click L : Add node\n"
        "  Dbl-click R : New spline\n"
        "  Drag Red    : Translate node\n"
        "  Drag Handle : Tangents\n"
        "  Shift+Drag  : Snap to vertex\n"
        "  C           : Close/open loop\n"
        "  Backspace   : Undo node\n"
        "  Ctrl+Z / Y  : Undo / Redo\n"
        "  b/o/k       : Toggle curves\n"
        "  t           : Gizmo opacity\n"
        "  s           : Save JSON\n"
        "  l           : Load JSON\n"
        "  e           : Export paths\n"
        "  w           : Wireframe\n"
        "  a           : Surface opacity"
    )

    def _on_window_resize(self, obj, event) -> None:
        """Repositions checkbox widgets, help button, and slider after
        a window resize.

        Layout (bottom-up, in pixels from bottom edge):
          - Slider: 20 px from bottom
          - Checkboxes + help "?": 20 px above slider top

        All positions are recomputed from the window height so the
        widget cluster stays compact and proportional.
        ``PlaceWidget`` takes 6-element bounds ``[xmin, xmax, ymin, ymax, 0, 0]``.
        """
        h = self.plotter.window_size[1]
        if h < 1:
            return
        sz = self._cb_size

        # Slider: fixed 20 px from bottom, normalized coords
        slider_y_px = 20
        slider_y_norm = slider_y_px / h
        rep_sl = self._opacity_slider.GetRepresentation()
        rep_sl.GetPoint1Coordinate().SetValue(0.0, slider_y_norm, 0.0)
        rep_sl.GetPoint2Coordinate().SetValue(0.15, slider_y_norm, 0.0)

        # Checkboxes: 20 px above slider
        cb_y = slider_y_px + 20

        for i, (layer, widget) in enumerate(self._layer_widgets.items()):
            x = self._cb_x_positions[i]
            widget.GetRepresentation().PlaceWidget(
                [float(x), float(x + sz), float(cb_y), float(cb_y + sz), 0.0, 0.0])

        hx = self._help_x
        self._help_widget.GetRepresentation().PlaceWidget(
            [float(hx), float(hx + sz), float(cb_y), float(cb_y + sz), 0.0, 0.0])

        # Reposition "?" label
        self._help_label.SetPosition(hx + 5, cb_y + 2)

    def _toggle_help_panel(self, visible: bool) -> None:
        """Toggles the on-screen shortcut reference panel."""
        if visible and self._help_panel is None:
            self._help_panel = self.plotter.add_text(
                self._HELP_TEXT, position=( 2 , 85 ),
                font_size=8, color='red', shadow=False,
                name="help_panel")
            self._help_visible = True
        elif not visible and self._help_panel is not None:
            self.plotter.remove_actor(self._help_panel)
            self._help_panel = None
            self._help_visible = False
        self.plotter.render()

    def _on_key_press_ctrl(self, obj, event) -> None:
        """Raw VTK KeyPress handler for Ctrl+Z (undo) and Ctrl+Y (redo).

        Used instead of PyVista's ``add_key_event`` because the latter
        does not support modifier keys.
        """
        iren = self.plotter.iren.interactor
        if not iren.GetControlKey():
            return
        key = iren.GetKeySym()
        if key in ('z', 'Z'):
            self._on_undo_ctrl_z()
        elif key in ('y', 'Y'):
            self._on_redo()

    # --- Helpers ---

    @property
    def _active_nodes(self) -> list[GeodesicSegment]:
        return self.splines[self.active_spline_idx]

    def _iter_all_nodes(self) -> tuple[int, int, GeodesicSegment]:
        """Yields ``(spline_idx, node_idx, node)`` for every node across all splines."""
        for s_idx, nodes in enumerate(self.splines):
            for n_idx, node in enumerate(nodes):
                yield s_idx, n_idx, node

    def _spline_for_node(self, seg: GeodesicSegment) -> int:
        """Returns the spline index that owns *seg*.  O(1) via ``_node_to_spline`` cache."""
        return self._node_to_spline.get(id(seg), self.active_spline_idx)

    def _rebuild_node_index(self) -> None:
        """Rebuilds the ``id(segment) → spline_index`` lookup dict.

        Called after any structural change (node add/remove, spline
        add/remove) to keep the O(1) lookup in ``_spline_for_node`` correct.
        """
        self._node_to_spline = {
            id(node): s_idx
            for s_idx, nodes in enumerate(self.splines)
            for node in nodes
        }

    def _build_local_frame(self, pt: np.ndarray, cid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Builds an orthonormal tangent frame ``(normal, u, v)`` at *pt* on face *cid*."""
        n = self.geo.get_interpolated_normal(pt, cid)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(ref, n)) > self.scfg.NORMAL_ALIGN_THRESHOLD:
            ref = np.array([1.0, 0.0, 0.0])
        u = np.cross(n, ref)
        u /= np.linalg.norm(u)
        return n, u, np.cross(n, u)

    # --- Undo / Redo ---

    def _snapshot(self) -> dict:
        """Captures the current spline state as a lightweight dict.

        Uses the same 2-field-per-node representation as the JSON save
        format (origin + tangent-with-magnitude), so ``_load_from_data``
        can restore it directly.  Typical size: ~48 bytes per node.
        """
        splines = []
        for sid, nodes in enumerate(self.splines):
            node_data = []
            for node in nodes:
                tangent_3d = (node.local_v[0] * node.u
                              + node.local_v[1] * node.v) * node.h_length
                node_data.append({
                    'origin': node.origin.tolist(),
                    'tangent': tangent_3d.tolist(),
                })
            splines.append({
                'closed': self.splines_closed[sid],
                'nodes': node_data,
            })
        return {
            'version': 1,
            'splines': splines,
            'active_spline_idx': self.active_spline_idx,
        }

    def _push_undo(self) -> None:
        """Saves a snapshot to the undo stack.  Clears the redo stack.

        Called immediately before any mutation (node add/insert/delete,
        close, break, drag start, load).  The snapshot captures the state
        *before* the mutation so that Ctrl+Z restores it.
        """
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self._MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _on_undo_ctrl_z(self) -> None:
        """Ctrl+Z: restores the previous spline state from the undo stack."""
        if not self._undo_stack:
            self._set_hud("NOTHING TO UNDO", 'grey')
            self.plotter.render()
            return
        self._redo_stack.append(self._snapshot())
        data = self._undo_stack.pop()
        self._restore_snapshot(data)
        self._set_hud("UNDO", 'yellow')
        self.plotter.render()

    def _on_redo(self) -> None:
        """Ctrl+Y: re-applies the last undone operation from the redo stack."""
        if not self._redo_stack:
            self._set_hud("NOTHING TO REDO", 'grey')
            self.plotter.render()
            return
        self._undo_stack.append(self._snapshot())
        data = self._redo_stack.pop()
        self._restore_snapshot(data)
        self._set_hud("REDO", 'cyan')
        self.plotter.render()

    def _restore_snapshot(self, data: dict) -> None:
        """Restores a snapshot, using differential reconstruction when possible.

        **Differential path** (fast, when the spline structure is identical):
        compare the current state node-by-node with *data*.  Only nodes
        whose origin or tangent changed are reconstructed (via ``compute_shoot``,
        ~10 ms per node).  On a 50-node spline where only 1 node moved,
        this is ~50× faster than the full rebuild.

        **Full rebuild** (fallback): when the structure differs (different
        number of splines, different node counts per spline, closed flag
        changed), delegates to ``_load_from_data`` which clears all actors
        and reconstructs everything from scratch.
        """
        active = data.pop('active_spline_idx', 0)

        # Structure comparison: if any of these differ, we need a full rebuild.
        structure_match = (len(data['splines']) == len(self.splines))
        if structure_match:
            for i, sd in enumerate(data['splines']):
                if (len(sd['nodes']) != len(self.splines[i])
                        or bool(sd.get('closed', False)) != self.splines_closed[i]):
                    structure_match = False
                    break

        if not structure_match:
            # Full rebuild path
            self._load_from_data(data)
            self.active_spline_idx = min(active, len(self.splines) - 1)
            self._prev_active_spline_idx = self.active_spline_idx
            self._refresh_visuals()
            return

        # Differential path: same structure, reconstruct only changed nodes.
        changed_splines: set[int] = set()
        for sid, sd in enumerate(data['splines']):
            current_nodes = self.splines[sid]
            for nid, nd in enumerate(sd['nodes']):
                target_origin = np.array(nd['origin'], dtype=float)
                target_tangent = np.array(nd['tangent'], dtype=float)
                seg = current_nodes[nid]
                cur_tangent = (seg.local_v[0] * seg.u
                               + seg.local_v[1] * seg.v) * seg.h_length
                if (np.allclose(seg.origin, target_origin, atol=1e-12)
                        and np.allclose(cur_tangent, target_tangent, atol=1e-12)):
                    continue  # node unchanged
                # Reconstruct this node's geometry in place
                self._rebuild_node_inplace(seg, target_origin, target_tangent)
                changed_splines.add(sid)

        # Recompute spans only for splines with changed nodes
        if changed_splines:
            # Cancel any running orange workers for affected splines
            saved_sid = self.active_spline_idx
            for sid in changed_splines:
                self.active_spline_idx = sid
                self._recompute_spans()
                self._submit_geodesic_spans()
            self.active_spline_idx = saved_sid

        self.active_spline_idx = min(active, len(self.splines) - 1)
        self._prev_active_spline_idx = self.active_spline_idx
        self._refresh_visuals()

    def _rebuild_node_inplace(self, seg: GeodesicSegment,
                              origin: np.ndarray,
                              tangent_full: np.ndarray) -> None:
        """Reconstructs a single ``GeodesicSegment`` in place from the
        canonical 2-field representation (origin + tangent-with-magnitude).

        Used by the differential undo/redo path to avoid destroying/recreating
        VTK actors for unchanged nodes.
        """
        h_length = float(np.linalg.norm(tangent_full))
        if h_length > 1e-15:
            tangent_dir = tangent_full / h_length
        else:
            tangent_dir = np.array([1.0, 0.0, 0.0])
            h_length = 0.01

        face_idx = self.geo.find_face(origin)
        normal, u, v = self._build_local_frame(origin, face_idx)
        seg.origin = origin
        seg.face_idx = face_idx
        seg.normal = normal
        seg.u = u
        seg.v = v
        seg.h_length = h_length

        seg.path_b = self.geo.compute_shoot(origin, tangent_dir, h_length, face_idx)
        seg.path_a = self.geo.compute_shoot(origin, -tangent_dir, h_length, face_idx)
        seg.p_b = seg.path_b[-1] if seg.path_b is not None else None
        seg.p_a = seg.path_a[-1] if seg.path_a is not None else None
        seg.update_local_v(self.geo)
        seg.update_visuals(self.plotter)

    def _init_tangents(self, node: GeodesicSegment, direction: np.ndarray, length: float):
        """Shoots symmetric geodesic rays from *node* in ``+-direction`` for *length*.

        Sets ``p_a``, ``p_b``, ``path_a``, ``path_b``, and ``h_length`` on the node,
        then updates ``local_v`` for future parallel-transport translations.
        """
        cid = node.face_idx
        for sign, attr_p, attr_path in [(-1, 'p_a', 'path_a'), (1, 'p_b', 'path_b')]:
            path = self.geo.compute_shoot(node.origin, sign * direction, length, cid)
            if path is not None:
                setattr(node, attr_p, path[-1])
                setattr(node, attr_path, path)
        node.h_length = length
        node.update_local_v(self.geo)

    # --- Mouse interactions ---

    def _try_hit_marker(self, x: int, y: int) -> bool:
        """Hit-test using the parent's pre-built hover cache.

        Extends the parent's ``_try_hit_marker`` with spline-index
        switching: if the closest marker belongs to a different spline,
        the active spline is switched before initiating the drag.  Uses
        the same squared-distance, vectorized screen-projection path —
        no per-marker VTK coordinate calls, no ``np.linalg.norm``.
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
        s_idx = self._spline_for_node(seg)
        self._push_undo()
        # Save the pre-drag active spline so _finalize_release can restore
        # it.  Without this, dragging a node in a non-active spline
        # permanently switches away from the active one — the user loses
        # access to an empty (break) spline that has no clickable nodes.
        self._pre_drag_spline_idx = self.active_spline_idx
        if s_idx != self.active_spline_idx:
            self.active_spline_idx = s_idx
            self._refresh_visuals()
        self.state.active_seg = seg
        self.state.drag_marker = tag
        seg.is_active = True
        seg.is_dragging = True
        self._lock_camera()
        self._set_hud(f"DRAGGING {tag.upper()}", 'gold')
        # Cancel background workers and hide orange immediately
        # — in the same render frame as the drag initiation, not on the
        # first _on_move (which would leave them visible for 1+ frame).
        self._cancel_geodesic_spans(seg)
        seg.update_visuals(self.plotter)
        self.plotter.render()
        return True

    def _clear_spline_spans(self, sid: int) -> None:
        """Removes all span actors and cancels all workers for spline *sid*.

        Called before structural changes (node insertion/deletion) that
        invalidate span indices.  After the change, ``_recompute_spans``
        rebuilds everything from scratch.
        """
        for cache in (self._span_cache, self._geo_span_cache):
            to_remove = [k for k in cache if k[0] == sid]
            for key in to_remove:
                _, actor = cache.pop(key)
                safe_remove_actor(self.plotter, actor)
                self._work_mgr.cancel_all_for_span(key)
                self._degraded_spans.discard(key)
        self._span_drag_state = {
            k: v for k, v in self._span_drag_state.items() if k[0] != sid}

    def _insert_node_from_interp(self, info: dict, sid: int,
                                nodes: list, closed: bool) -> None:
        """Inserts a node from a hover on the interpolation (black) curve.

        Unlike the Bézier layers, the interp curve has no span structure
        — it is a single polyline per spline.  The insertion index is
        determined by **arc-length fraction** along the interp polyline:

          1. Compute the cumulative arc-length along the displayed
             polyline up to the hover point.
          2. For each node origin, find its closest vertex on the
             polyline and its arc-length fraction.
          3. The hover fraction falls between two consecutive node
             fractions → insert between those two nodes.

        This is robust against self-intersecting splines (where
        Euclidean nearest-origin would pick the wrong segment).

        The tangent direction comes from the polyline segment at the
        hover point, projected onto the surface tangent plane at the
        insertion position.
        """
        pt = self.geo.project_to_surface(
            np.array(info['point'], dtype=float))
        cid = self.geo.find_face(pt)

        n_nodes = len(nodes)
        entry = self._interp_cache.get(sid)
        origins = np.array([n.origin for n in nodes], dtype=float)

        # --- Compute insertion position via arc-length fraction ---
        insert_pos = None
        if (entry is not None and entry[0].points is not None
                and len(entry[0].points) >= 2 and n_nodes >= 2):
            pts_3d = np.asarray(entry[0].points)
            # Cumulative arc-length along the polyline
            diffs = np.diff(pts_3d, axis=0)
            seg_lens = np.linalg.norm(diffs, axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
            total = cum[-1]

            if total > 1e-12:
                # Hover fraction: arc-length up to the hover segment + frac
                seg = info['seg']
                frac = info['frac']
                hover_s = cum[seg] + seg_lens[seg] * frac if seg < len(seg_lens) else cum[-1]
                hover_frac = hover_s / total

                # Fraction of each node origin along the polyline
                # (find closest polyline vertex to each origin)
                node_fracs = np.empty(n_nodes, dtype=float)
                for k in range(n_nodes):
                    d = np.linalg.norm(pts_3d - origins[k], axis=1)
                    node_fracs[k] = cum[int(np.argmin(d))] / total

                # Find which node-gap the hover_frac falls in.
                # Non-closed: gaps are [frac[0], frac[1]], ..., [frac[n-2], frac[n-1]].
                # Closed: an extra wrap gap [frac[n-1], frac[0]+1].
                if closed:
                    # Extend node_fracs with wrap entry
                    sorted_idx = np.argsort(node_fracs)
                    sorted_fracs = node_fracs[sorted_idx]
                    # Find position in sorted cycle
                    pos_in_sorted = int(np.searchsorted(sorted_fracs, hover_frac))
                    # pos_in_sorted indicates which sorted gap the hover is in.
                    # Insert after sorted_idx[pos_in_sorted - 1] in node order.
                    if pos_in_sorted == 0:
                        # Wrap-around: insert after the last node
                        insert_pos = (sorted_idx[-1] + 1) % n_nodes
                    elif pos_in_sorted >= n_nodes:
                        insert_pos = (sorted_idx[-1] + 1) % n_nodes
                    else:
                        insert_pos = sorted_idx[pos_in_sorted - 1] + 1
                else:
                    # Open: find the gap [node_fracs[i], node_fracs[i+1]]
                    for j in range(n_nodes - 1):
                        if node_fracs[j] <= hover_frac <= node_fracs[j + 1]:
                            insert_pos = j + 1
                            break
                    if insert_pos is None:
                        # Hover before first or after last → append at closer end
                        insert_pos = 0 if hover_frac < node_fracs[0] else n_nodes

        if insert_pos is None:
            # Fallback: nearest-origin Euclidean (degenerate cases)
            if n_nodes < 2:
                insert_pos = n_nodes
            else:
                dists = np.linalg.norm(origins - pt, axis=1)
                insert_pos = int(np.argmin(dists)) + 1

        # --- Tangent from the polyline direction at the hover segment ---
        if (entry is not None and entry[0].points is not None
                and len(entry[0].points) >= 2):
            seg_idx = info['seg']
            pts_3d = entry[0].points
            if seg_idx + 1 < len(pts_3d):
                tangent = pts_3d[seg_idx + 1] - pts_3d[seg_idx]
            else:
                tangent = pts_3d[-1] - pts_3d[-2]
        else:
            # Fallback: direction between neighbor nodes
            if n_nodes >= 2:
                i_prev = (insert_pos - 1) % n_nodes
                i_next = insert_pos % n_nodes
                tangent = origins[i_next] - origins[i_prev]
            else:
                tangent = np.array([1.0, 0.0, 0.0])

        # Project tangent onto surface tangent plane and normalize
        normal = self.geo.get_interpolated_normal(pt, cid)
        tangent = tangent - np.dot(tangent, normal) * normal
        tn = np.linalg.norm(tangent)
        if tn > 1e-12:
            tangent /= tn
        else:
            tangent = np.array([1.0, 0.0, 0.0])

        # Create node
        new_node = GeodesicSegment(pt, cid, *self._build_local_frame(pt, cid))
        new_node.is_active = True

        # Handle length from distances to neighbors
        if n_nodes >= 2:
            i_prev = max(0, insert_pos - 1)
            i_next = min(n_nodes - 1, insert_pos) if insert_pos < n_nodes else 0
            d0 = np.linalg.norm(pt - origins[i_prev])
            d1 = np.linalg.norm(pt - origins[i_next])
            h_len = min(d0, d1) * self.scfg.HANDLE_FRACTION
        else:
            h_len = self.diag * self.scfg.INITIAL_H_FRACTION
        self._init_tangents(new_node, tangent, h_len)

        # Clear span caches (indices shift)
        self._clear_spline_spans(sid)

        # Insert
        nodes.insert(insert_pos, new_node)
        self.segments.append(new_node)
        new_node.update_visuals(self.plotter)
        self._hover_dirty = True
        self._rebuild_node_index()
        self._refresh_visuals()
        self._recompute_spans()
        self._submit_geodesic_spans()
        self._set_hud("NODE INSERTED (INTERP)", 'lime')
        self.plotter.render()

    # --- Bézier split at curve hover — helper methods ---

    def _recover_t_from_hover(self, info: dict, layer: str,
                              sid: int, span_idx: int) -> float:
        """Recovers the Bézier parameter *t* from the polyline hover position.

        The displayed polyline may have non-uniform parameter spacing
        (adaptive sampling, secant subdivision), so ``info['seg']/N`` is
        wrong.  Uses **arc-length fraction** along the polyline — robust
        against any sampling distribution.  Returns 0.5 when the polyline
        is unavailable or degenerate, clamped to ``[0.01, 0.99]``.
        """
        layer_cache = {
            'blue': self._span_cache,
            'orange': self._geo_span_cache,
        }[layer]
        curve_entry = layer_cache.get((sid, span_idx))
        t = 0.5
        if curve_entry is not None and curve_entry[0].points is not None:
            pts_3d = curve_entry[0].points
            if len(pts_3d) >= 2:
                diffs = np.diff(pts_3d, axis=0)
                seg_lens = np.linalg.norm(diffs, axis=1)
                total_len = seg_lens.sum()
                if total_len > 1e-15:
                    seg_idx = info['seg']
                    frac = info['frac']
                    len_before = seg_lens[:seg_idx].sum()
                    len_partial = (seg_lens[seg_idx] * frac
                                   if seg_idx < len(seg_lens) else 0.0)
                    t = (len_before + len_partial) / total_len
        return max(0.01, min(0.99, t))

    def _de_casteljau_split(self, ctrl: tuple, paths: tuple, t: float,
                            use_geodesic: bool) -> dict:
        """Computes de Casteljau intermediate points at parameter *t*.

        de Casteljau triangle (cubic Bézier with 4 control points)::

            Level 0:   P0  ----  H_out  ----  H_in  ----  P1
                         \\        /  \\        /  \\        /
                          \\  lerp/    \\  lerp/    \\  lerp/
                           \\    /      \\    /      \\    /
            Level 1:       b01  -------  b12  -------  b23
                              \\          /  \\          /
                               \\   lerp /    \\   lerp /
                                \\      /      \\      /
            Level 2:             c0  -----------  c1
                                    \\           /
                                     \\   lerp  /
                                      \\       /
            Level 3:                    Q  (the split point)

        Each arrow is a ``lerp(A, B, t)``.  On our curved surface, level-1
        lerps are GEODESIC along the pre-existing paths (``path_b`` for
        P0→H_out, ``path_a`` reversed for H_in→P1, always a freshly
        computed geodesic for H_out→H_in).  Levels 2-3 use Euclidean
        lerp + ``project_to_surface``.

        Returns a dict with keys ``b01, b12, b23, c0, c1`` — the level-1
        and level-2 points.  Level-3 (Q) is NOT computed here; the caller
        uses the hover point directly instead (exactly where the user
        clicked, avoiding projection drift from levels 2-3).

        Parameters
        ----------
        ctrl : ``(P0, H_out, H_in, P1)`` tuple of control points.
        paths : ``(path_b, path_a)`` geodesic polylines for the outer
            segments (may be ``None`` when not available).
        t : parameter in (0, 1).
        use_geodesic : True for orange layer, False for blue — selects
            geodesic lerp on outer segments vs Euclidean + projection.
        """
        P0, H_out, H_in, P1 = ctrl
        path_b, path_a = paths

        if use_geodesic and path_b is not None and len(path_b) >= 2:
            cum, total = GeodesicMesh.compute_path_lengths(path_b)
            b01 = GeodesicMesh.geodesic_lerp(path_b, t, cum, total)
        else:
            b01 = self.geo.project_to_surface(P0 * (1 - t) + H_out * t)

        b12 = self.geo.project_to_surface(H_out * (1 - t) + H_in * t)

        if use_geodesic and path_a is not None and len(path_a) >= 2:
            path_a_rev = path_a[::-1]
            cum, total = GeodesicMesh.compute_path_lengths(path_a_rev)
            b23 = GeodesicMesh.geodesic_lerp(path_a_rev, t, cum, total)
        else:
            b23 = self.geo.project_to_surface(H_in * (1 - t) + P1 * t)

        c0 = self.geo.project_to_surface(b01 * (1 - t) + b12 * t)
        c1 = self.geo.project_to_surface(b12 * (1 - t) + b23 * t)

        return {'b01': b01, 'b12': b12, 'b23': b23, 'c0': c0, 'c1': c1}

    @staticmethod
    def _bezier_derivative_tangent(ctrl: tuple, t: float,
                                   normal: np.ndarray,
                                   fallback: np.ndarray) -> np.ndarray:
        """Computes the tangent direction at parameter *t* of a cubic Bézier.

        Uses ``B'(t) = 3(1-t)²(H-P0) + 6(1-t)t(H_in-H_out) + 3t²(P1-H_in)``,
        projects onto the surface tangent plane at the insertion point
        (removes component along *normal*), and normalizes.

        Falls back to the given *fallback* vector (typically ``c1 - c0``
        from the de Casteljau split) when the derivative is degenerate.
        """
        P0, H_out, H_in, P1 = ctrl
        omt = 1.0 - t
        deriv = (3.0 * omt * omt * (H_out - P0) +
                 6.0 * omt * t * (H_in - H_out) +
                 3.0 * t * t * (P1 - H_in))
        deriv -= np.dot(deriv, normal) * normal
        dn = np.linalg.norm(deriv)
        if dn > 1e-12:
            return deriv / dn
        tn2 = np.linalg.norm(fallback)
        if tn2 > 1e-12:
            return fallback / tn2
        return np.array([1.0, 0.0, 0.0])

    def _shorten_endpoint_handle(self, node: GeodesicSegment,
                                 origin: np.ndarray, new_tip: np.ndarray,
                                 which: str) -> None:
        """Shortens a node's outgoing handle to reach *new_tip* exactly.

        *which* is ``'b'`` (outgoing ``p_b``/``path_b``) or ``'a'``
        (incoming ``p_a``/``path_a``).  Recomputes the geodesic path,
        updates ``h_length`` from the actual arc-length, and syncs
        ``local_v``.

        Used in the endpoint rule of node insertion to give an exact
        de Casteljau split on endpoint nodes of open splines (where
        neighbor handle modification doesn't break C1 with another span).
        """
        direction_raw = new_tip - origin
        dist = np.linalg.norm(direction_raw)
        if dist < 1e-12:
            return
        direction = direction_raw / dist
        path = self.geo.compute_shoot(origin, direction, dist, node.face_idx)
        if path is None:
            path = np.array([origin, new_tip])
        if which == 'b':
            node.p_b = new_tip
            node.path_b = path
        else:
            node.p_a = new_tip
            node.path_a = path
        node.h_length = float(np.sum(np.linalg.norm(
            np.diff(path, axis=0), axis=1)))
        node.update_local_v(self.geo)

    def _insert_node_at_curve(self, info: dict) -> None:
        """Inserts a new C1 node at the curve hover point.

        The new node is placed at ``info['point']`` (projected onto the
        surface) — exactly where the user clicked, independent of the
        de Casteljau approximation.

        Orchestrates four helpers:
          1. ``_recover_t_from_hover``: arc-length fraction along the
             displayed polyline gives the Bézier parameter *t*.
          2. ``_de_casteljau_split``: level-1 and level-2 intermediate
             points used for handle shortening and derivative fallback.
          3. ``_bezier_derivative_tangent``: tangent direction for the
             new node's symmetric C1 handles.
          4. ``_shorten_endpoint_handle``: exact endpoint-rule handle
             shortening on open-spline endpoints.

        Interp-layer hovers are delegated to ``_insert_node_from_interp``
        (different logic: no de Casteljau, tangent from polyline
        direction).

        Endpoint rule for neighbor handle modification:
          - Open spline, first span: ``n0.p_b`` shortened to ``b01``.
          - Open spline, last span: ``n1.p_a`` shortened to ``b23``.
          - Closed spline or interior span: neighbors untouched
            (preserves C1 with adjacent spans).

        Span indices shift after insertion, so span caches for the
        affected spline are cleared and rebuilt.
        """
        sid = info['spline_idx']
        span_idx = info['span_idx']
        layer = info['layer']

        # Switch to the correct spline if needed
        if sid != self.active_spline_idx:
            self.active_spline_idx = sid
            self._refresh_visuals()

        nodes = self.splines[sid]
        closed = self.splines_closed[sid]

        # Interp layer: different logic (no de Casteljau, polyline tangent)
        if layer == 'interp':
            self._insert_node_from_interp(info, sid, nodes, closed)
            return

        n0 = nodes[span_idx]
        n1 = nodes[(span_idx + 1) % len(nodes)]
        P0, H_out, H_in, P1 = n0.origin, n0.p_b, n1.p_a, n1.origin

        # --- Step 1: parameter t from arc-length fraction ---
        t = self._recover_t_from_hover(info, layer, sid, span_idx)

        # --- Node position: exactly where the user clicked ---
        Q = self.geo.project_to_surface(np.array(info['point'], dtype=float))
        cid = self.geo.find_face(Q)
        new_node = GeodesicSegment(Q, cid, *self._build_local_frame(Q, cid))
        new_node.is_active = True

        # Fallback path: if the span has no handles yet, use simple midpoint
        if H_out is None or H_in is None:
            tangent = P1 - P0
            tn = np.linalg.norm(tangent)
            if tn > 1e-12:
                tangent /= tn
            d0 = np.linalg.norm(Q - P0)
            d1 = np.linalg.norm(Q - P1)
            self._init_tangents(new_node, tangent,
                                min(d0, d1) * self.scfg.HANDLE_FRACTION)
            can_modify_n0 = False
            can_modify_n1 = False
        else:
            # --- Step 2: de Casteljau intermediate points ---
            ctrl = (P0, H_out, H_in, P1)
            paths = (n0.path_b, n1.path_a)
            use_geodesic = (layer != 'blue')
            dc = self._de_casteljau_split(ctrl, paths, t, use_geodesic)

            # --- Step 3: tangent direction from Bézier derivative ---
            tangent = self._bezier_derivative_tangent(
                ctrl, t, new_node.normal, dc['c1'] - dc['c0'])

            # Handle length proportional to min distance to neighbors
            h_len = (min(np.linalg.norm(Q - P0), np.linalg.norm(P1 - Q))
                     * self.scfg.HANDLE_FRACTION)
            self._init_tangents(new_node, tangent, h_len)

            # --- Step 4: endpoint-rule neighbor handle shortening ---
            n_spans = self._span_count(sid)
            can_modify_n0 = (span_idx == 0) and not closed
            can_modify_n1 = (span_idx == n_spans - 1) and not closed
            if can_modify_n0:
                self._shorten_endpoint_handle(n0, P0, dc['b01'], which='b')
            if can_modify_n1:
                self._shorten_endpoint_handle(n1, P1, dc['b23'], which='a')

        # --- Insert into data structures ---
        self._clear_spline_spans(sid)
        insert_pos = span_idx + 1
        if closed and insert_pos >= len(nodes):
            insert_pos = len(nodes)
        nodes.insert(insert_pos, new_node)
        self.segments.append(new_node)

        # Rebuild visuals — new node always, neighbors only if modified
        new_node.update_visuals(self.plotter)
        if can_modify_n0:
            n0.update_visuals(self.plotter)
        if can_modify_n1:
            n1.update_visuals(self.plotter)
        self._hover_dirty = True
        self._rebuild_node_index()
        self._refresh_visuals()
        self._recompute_spans()
        self._submit_geodesic_spans()
        self._set_hud("NODE INSERTED", 'lime')
        self.plotter.render()

    def _on_press(self, obj, event) -> None:
        """Left-button press: single-click drags a marker, double-click adds/inserts a node.

        On double-click:
          - If the cursor is hovering a curve (``curve_hover_info`` set),
            a new node is **inserted** at that point, splitting the span.
          - Otherwise, a new node is **appended** to the end of the
            active spline.

        ``new_node.update_visuals()`` is called explicitly after insertion
        because ``_refresh_visuals`` only updates nodes whose state
        *changed* — a freshly created node has ``is_active=True`` by
        default, so it would be skipped and remain invisible.
        """
        x, y = self.plotter.iren.get_event_position()
        is_double = self.plotter.iren.interactor.GetRepeatCount() >= 1

        if not is_double:
            self._try_hit_marker(x, y)
            return

        # Node insertion on curve hover takes priority
        if self.curve_hover_info is not None:
            self._push_undo()
            self._insert_node_at_curve(self.curve_hover_info)
            self.curve_hover_info = None
            self._curve_hover_actor.SetVisibility(False)
            return

        pt, cid = self._pick()
        if pt is None:
            return
        if self.splines_closed[self.active_spline_idx]:
            return

        self._push_undo()
        nodes = self._active_nodes
        new_node = GeodesicSegment(pt, cid, *self._build_local_frame(pt, cid))
        new_node.is_active = True
        new_node.h_length = self.diag * self.scfg.INITIAL_H_FRACTION

        if nodes:
            vec = pt - nodes[-1].origin
            vn = np.linalg.norm(vec)
            if vn > 1e-9:
                v_dir, h_len = vec / vn, vn * self.scfg.HANDLE_FRACTION
                self._init_tangents(new_node, v_dir, h_len)
                if len(nodes) == 1 and nodes[0].p_b is None:
                    self._init_tangents(nodes[0], v_dir, h_len)
                    nodes[0].update_visuals(self.plotter)

        nodes.append(new_node)
        self.segments.append(new_node)
        new_node.update_visuals(self.plotter)
        self._hover_dirty = True
        self._rebuild_node_index()
        self._refresh_visuals()
        self._recompute_spans()
        # Submit only the new span (last node), not all — previous spans
        # already have their geodesic curves computed or in progress.
        self._submit_geodesic_spans(node=new_node)
        self.plotter.render()

    def _on_right_press(self, obj, event) -> None:
        """Double-right-click: starts a new spline (break).

        Only fires when the current spline has at least one node —
        prevents consecutive empty splines.
        """
        if self.plotter.iren.interactor.GetRepeatCount() >= 1 and self._active_nodes:
            self._push_undo()
            self.splines.append([])
            self.splines_closed.append(False)
            self.active_spline_idx = len(self.splines) - 1
            self._refresh_visuals()
            self._set_hud("NEW SPLINE STARTED", 'lime')
            self.plotter.render()

    def _snap_point_to_edge(self, p: np.ndarray, face_idx: int | None
                            ) -> tuple[np.ndarray | None,
                                       tuple[int, int, float] | None]:
        """Projects *p* onto the closest edge of *face_idx* (clamped).

        Tests the 3 edges of the containing face; for each edge ``(a, b)``
        computes the perpendicular foot of *p*, clamped to ``t ∈ [0, 1]``
        so the snap point always lies within the edge segment (never
        outside).  Returns ``(snap_xyz, (va, vb, t))`` or ``(None, None)``
        when the face is degenerate.

        Edge snap is cheap (3 dot products + 3 comparisons) and stays on
        the surface by construction: every edge is a real mesh edge, so
        no re-projection is needed.  Useful for landing splines on CAD
        seam edges and feature creases.
        """
        if face_idx is None:
            return None, None
        F = self.geo.F[face_idx]
        V = self.geo.V
        best_d = np.inf
        best_pt = None
        best_info = None
        for i in range(3):
            ia, ib = int(F[i]), int(F[(i + 1) % 3])
            a = V[ia]
            edge = V[ib] - a
            L2 = float(np.dot(edge, edge))
            if L2 < 1e-18:
                continue
            t = float(np.dot(p - a, edge)) / L2
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            closest = a + t * edge
            d = float(np.linalg.norm(p - closest))
            if d < best_d:
                best_d = d
                best_pt = closest.copy()
                best_info = (ia, ib, t)
        return best_pt, best_info

    def _on_move(self, obj, event) -> None:
        """Spline-aware move handler.

        Picks once per frame and passes the result to the parent via
        ``pick_override`` so its ``_on_move`` skips the redundant
        O(log N) ray-cast.  After the parent processes hover detection
        and drag geometry, this method handles span recomputation,
        stitch preview, and **curve hover detection**.

        Processing order (when not dragging or hovering a handle):
          1. Parent: hover detection, cursor, render.
          2. Stitch preview update (vertex-snapped geodesic).
          3. Curve hover: ``_detect_curve_hover`` finds the closest
             visible curve segment (blue/orange/interp), positions a
             colored marker, and stores metadata in ``curve_hover_info``
             for future node insertion.

        During drag, hybrid spans are recomputed AFTER the parent's
        render — they lag by one frame (~16 ms at 60 Hz, imperceptible).
        Geodesic (orange) spans are cancelled on drag start and
        resubmitted on debounce consolidation.
        """
        # Skip expensive ray-pick when hovering a marker (no drag active).
        hovering_marker = self.state.hover_seg is not None and self.state.active_seg is None
        dragged = self.state.active_seg
        # Tell the parent to suppress the cursor when curve hover is active
        # (set BEFORE super()._on_move so the parent reads it).
        self._hide_cursor = self.curve_hover_info is not None
        if hovering_marker:
            pick_result = (None, None)
            super()._on_move(obj, event)
        else:
            pick_result = self._pick()
            if dragged:
                iren = self.plotter.iren.interactor
                if pick_result[0] is not None:
                    # Shift wins over Ctrl when both are held — vertex snap
                    # is a strict subset of edge snap (edge endpoints are
                    # vertices), so no disambiguation is needed.
                    if iren.GetShiftKey():
                        _, vi = self.geo._kdtree.query(pick_result[0])
                        snapped = self.geo.V[int(vi)].copy()
                        pick_result = (snapped, self.geo.find_face(snapped))
                        self._set_hud(f"SNAP → vertex {int(vi)}", 'gold')
                    elif iren.GetControlKey():
                        snapped, info = self._snap_point_to_edge(
                            pick_result[0], pick_result[1])
                        if snapped is not None:
                            pick_result = (snapped,
                                           self.geo.find_face(snapped))
                            va, vb, t = info
                            self._set_hud(
                                f"SNAP → edge {va}-{vb} t={t:.2f}",
                                'cyan')
                self._stitch_actor.SetVisibility(False)
                self._cancel_geodesic_spans(dragged)
            super()._on_move(obj, event, pick_override=pick_result)

        if dragged:
            # Recompute spans after the parent processed the drag geometry.
            self._recompute_spans(node=dragged)
            self._curve_hover_actor.SetVisibility(False)
            self.curve_hover_info = None
            return

        # Skip stitch preview and curve hover when hovering a handle marker
        if hovering_marker:
            self._curve_hover_actor.SetVisibility(False)
            self.curve_hover_info = None
            return

        # Stitch preview — screen-pixel threshold (squared, no sqrt)
        x, y = self.plotter.iren.get_event_position()
        sdx = x - self._last_stitch_screen[0]
        sdy = y - self._last_stitch_screen[1]
        if sdx * sdx + sdy * sdy < self.scfg.STITCH_SKIP_PX_SQ:
            return
        self._last_stitch_screen = (x, y)

        self._update_stitch(q=pick_result[0])

        # Curve hover detection — show marker on nearest visible curve
        curve_changed = self._detect_curve_hover(x, y)
        if curve_changed:
            self.plotter.render()

    def _on_close_spline(self) -> None:
        """'C' key: toggles open/closed state of the active spline.

        If the spline is open (3+ nodes): closes it by computing a
        closing tangent (first node's ``p_a`` toward the last node),
        then auto-breaks into a new empty spline.

        If the spline is already closed: reopens it by clearing the
        first node's ``p_a`` tangent, removing the wrap-around span,
        and marking the spline open.  Stays on the same spline (no
        auto-break on reopen).
        """
        sid = self.active_spline_idx
        nodes = self.splines[sid]

        if not nodes:
            return

        # Already closed → reopen
        if self.splines_closed[sid]:
            self._push_undo()
            self.splines_closed[sid] = False
            # Clear the closing tangent on the first node (phantom handle
            # that belonged only to the wrap-around span)
            first = nodes[0]
            first.p_a = None
            first.path_a = None
            first.update_visuals(self.plotter)
            # Remove the wrap-around span cache entries + cancel workers
            closing_idx = len(nodes) - 1
            key = (sid, closing_idx)
            entry = self._span_cache.pop(key, None)
            if entry:
                safe_remove_actor(self.plotter, entry[1])
            self._work_mgr.cancel_all_for_span(key)
            entry_g = self._geo_span_cache.pop(key, None)
            if entry_g:
                safe_remove_actor(self.plotter, entry_g[1])
            self._hover_dirty = True
            self._recompute_spans()
            self._submit_geodesic_spans()
            self._refresh_visuals()
            self._update_stitch()
            self._set_hud("LOOP OPENED", 'yellow')
            self.plotter.render()
            return

        # Open → close (requires 3+ nodes)
        if len(nodes) < 3:
            return

        self._push_undo()
        did_close = False
        first, last = nodes[0], nodes[-1]
        vec = first.origin - last.origin
        # Project onto first node's tangent plane for a rigorous
        # surface-tangent direction.  Raw Euclidean vec can point
        # through the mesh interior on curved surfaces.
        vec -= np.dot(vec, first.normal) * first.normal
        vn = np.linalg.norm(vec)
        if vn > 1e-9:
            v_dir = vec / vn
            h_len = np.linalg.norm(first.origin - last.origin) * self.scfg.HANDLE_FRACTION
            # Compute closing tangent for first node
            if first.p_a is None:
                path_a = self.geo.compute_shoot(
                    first.origin, v_dir, h_len, first.face_idx)
                if path_a is not None:
                    first.p_a, first.path_a = path_a[-1], path_a
                    first.update_visuals(self.plotter)
            # Only close if first.p_a is valid (shoot succeeded)
            if first.p_a is not None:
                self.splines_closed[sid] = True
                did_close = True
                self._recompute_spans()
                self._submit_geodesic_spans(node=first)

        if not did_close:
            return

        # Auto-break: start a new spline (only after successful close)
        self.splines.append([])
        self.splines_closed.append(False)
        self.active_spline_idx = len(self.splines) - 1
        self._stitch_actor.SetVisibility(False)
        self._refresh_visuals()
        self._set_hud("LOOP CLOSED + BREAK", 'cyan')
        self.plotter.render()

    def _on_backspace(self) -> None:
        """Backspace: removes the last node from the active spline, or undoes
        the last break if the spline is empty.

        Undo of a break restores the previous spline as active and reopens
        it if it was closed.  Affected span cache entries are cleaned up
        so no stale actors remain in the scene.
        """
        sid = self.active_spline_idx
        nodes = self.splines[sid]

        # Nothing to undo: single empty spline
        if not nodes and len(self.splines) <= 1:
            return

        self._push_undo()
        # Empty spline = undo the break
        if not nodes and len(self.splines) > 1:
            self.splines.pop(sid)
            self.splines_closed.pop(sid)
            self.active_spline_idx = len(self.splines) - 1
            self._rebuild_node_index()
            if self.splines_closed[self.active_spline_idx]:
                self.splines_closed[self.active_spline_idx] = False
                prev_nodes = self.splines[self.active_spline_idx]

                # Clear the closing tangent on the first node — it was
                # set by _on_close_spline and doesn't belong to any open
                # span.  Without this, a phantom handle A remains visible,
                # hoverable, and draggable after reopening.
                first = prev_nodes[0]
                first.p_a = None
                first.path_a = None
                first.update_visuals(self.plotter)

                # Remove the closing span from cache
                closing_idx = len(prev_nodes) - 1
                key = (self.active_spline_idx, closing_idx)
                entry = self._span_cache.pop(key, None)
                if entry:
                    safe_remove_actor(self.plotter, entry[1])
                # Cancel all workers + hide actors for closing span
                self._work_mgr.cancel_all_for_span(key)
                for cache in (self._geo_span_cache,):
                    removed = cache.pop(key, None)
                    if removed:
                        safe_remove_actor(self.plotter, removed[1])
                self._hover_dirty = True
                self._set_hud("LOOP OPENED", 'yellow')
            else:
                self._set_hud("BREAK REMOVED", 'yellow')
            self._refresh_visuals()
            self._update_stitch()
            self.plotter.render()
            return

        if nodes:
            node = nodes.pop()
            if node in self.segments:
                self.segments.remove(node)
            node.clear_actors(self.plotter)
            if self.state.hover_seg is node:
                self.state.hover_seg = None
                self.state.hover_marker = None
            self._hover_dirty = True
            self._rebuild_node_index()
            removed_idx = len(nodes)
            if removed_idx > 0:
                key = (sid, removed_idx - 1)
                entry = self._span_cache.pop(key, None)
                if entry:
                    safe_remove_actor(self.plotter, entry[1])
                # Cancel all workers + hide actors for removed span
                self._work_mgr.cancel_all_for_span(key)
                for cache in (self._geo_span_cache,):
                    removed_entry = cache.pop(key, None)
                    if removed_entry:
                        safe_remove_actor(self.plotter, removed_entry[1])
            self._recompute_spans()
        self._update_stitch()
        self.plotter.render()

    # --- Stitch preview ---

    def _refresh_stitch_cache(self) -> None:
        """Rebuilds the ``prepare_origin`` cache for the stitch preview.

        Called when the last node of the active spline changes (node
        placement, deletion, drag consolidation).  The cache inserts the
        node's origin into the mesh topology and builds a local solver,
        so that ``_update_stitch`` can compute geodesics starting from the
        exact node position (~0.01 ms per query) instead of the nearest
        mesh vertex.

        Cost: ~2-5 ms (one-off).  Subsequent ``_update_stitch`` calls
        reuse the cached solver with vertex-snapped endpoint only.
        """
        nodes = self._active_nodes
        if not nodes or self.splines_closed[self.active_spline_idx]:
            self._stitch_origin_cache = None
            self._stitch_origin_node_id = -1
            return
        last = nodes[-1]
        nid = id(last)
        if nid == self._stitch_origin_node_id:
            return  # cache already valid for this node
        self._stitch_origin_cache = self.geo.prepare_origin(last.origin)
        self._stitch_origin_node_id = nid

    def _invalidate_stitch_cache(self) -> None:
        """Forces stitch cache rebuild on next ``_refresh_stitch_cache``."""
        self._stitch_origin_cache = None
        self._stitch_origin_node_id = -1

    def _update_stitch(self, q=None) -> None:
        """Updates the prospective-span preview from last node to cursor.

        Uses the topologically-inserted origin (via ``prepare_origin``
        cache) for the start point, and vertex-snap for the cursor
        endpoint.  This gives an exact geodesic departure from the node
        while keeping per-frame cost at ~0.01 ms.

        Parameters
        ----------
        q : optional surface point.  When provided (from ``_on_move``),
            avoids a redundant ray-cast.  When None (from ``_on_backspace`` /
            ``_on_close_spline``), falls back to a fresh ``_pick()``.
        """
        nodes = self._active_nodes
        if self.splines_closed[self.active_spline_idx]:
            self._stitch_actor.SetVisibility(False)
            return
        if q is None and nodes:
            q, _ = self._pick()
        if not nodes or q is None:
            self._stitch_actor.SetVisibility(False)
            return

        last = nodes[-1]
        vec = q - last.origin
        vn = np.linalg.norm(vec)
        if vn < 1e-9:
            self._stitch_actor.SetVisibility(False)
            return

        # Ensure origin cache is valid for current last node
        self._refresh_stitch_cache()
        cache = self._stitch_origin_cache

        if cache is not None:
            # Exact origin + vertex-snapped endpoint (~0.01 ms)
            idx_s = cache['idx']
            _, idx_e = cache['kdtree'].query(q)
            idx_e = int(idx_e)
            if idx_s == idx_e:
                self._stitch_actor.SetVisibility(False)
                return
            try:
                pts = cache['solver'].find_geodesic_path(idx_s, idx_e)
            except Exception:
                self._stitch_actor.SetVisibility(False)
                return
        else:
            # Fallback: both endpoints vertex-snapped (degenerate mesh)
            _, idx_s = self.geo._kdtree.query(last.origin)
            _, idx_e = self.geo._kdtree.query(q)
            idx_s, idx_e = int(idx_s), int(idx_e)
            if idx_s == idx_e:
                self._stitch_actor.SetVisibility(False)
                return
            try:
                pts = self.geo._solver.find_geodesic_path(idx_s, idx_e)
            except Exception:
                self._stitch_actor.SetVisibility(False)
                return

        if pts is None or len(pts) < 2:
            self._stitch_actor.SetVisibility(False)
            return

        update_line_inplace(self._stitch_pd, pts)
        import gizmo
        self._stitch_actor.GetProperty().SetOpacity(gizmo.GIZMO_OPACITY)
        self._stitch_actor.SetVisibility(True)

    # --- Span rendering ---

    def _mark_span_degraded(self, key: tuple, degraded: bool) -> None:
        """Updates the ``_degraded_spans`` set and flashes a HUD warning.

        Called by ``_recompute_spans`` after every blue-layer
        ``compute_endpoint_local`` and by the orange worker drain loop.
        A transient HUD message fires only on the ``False → True``
        transition so the user isn't spammed while the flag remains set
        across redundant recomputations.
        """
        was_degraded = key in self._degraded_spans
        if degraded and not was_degraded:
            self._degraded_spans.add(key)
            self._set_hud(f"GEODESIC FALLBACK on span {key[0]}:{key[1]}",
                          'red')
            # Clear any cached drag-state so _set_span will repaint.
            self._span_drag_state.pop(key, None)
        elif not degraded and was_degraded:
            self._degraded_spans.discard(key)
            self._span_drag_state.pop(key, None)

    def _set_span(self, sid: int, i: int, pts, dragging: bool = False) -> None:
        """Updates the geometry and style of span *(sid, i)*.

        When *dragging* is True (node is being dragged, preview state),
        the span is drawn thinner and lighter to signal "approximate".
        On consolidation (debounce fires, *dragging* becomes False) the
        span reverts to full color/width.

        Degraded spans (``key in self._degraded_spans`` — geodesic fell
        back to a straight line) are painted red regardless of drag state.
        """
        key = (sid, i)
        if key not in self._span_cache:
            pd = pv.PolyData()
            actor = self.plotter.add_mesh(pd, lighting=False, pickable=False)

            self._set_depth_priority(actor, self.scfg.DEPTH_BLUE)
            self._span_cache[key] = (pd, actor)

        pd, actor = self._span_cache[key]
        if pts is None or len(pts) < 2:
            actor.SetVisibility(False)
            return
        update_line_inplace(pd, pts)

        degraded = key in self._degraded_spans
        # Tri-state style key: (dragging, degraded).  Using ``None`` as the
        # unseen sentinel lets the first call always repaint.
        style_key = (dragging, degraded)
        if self._span_drag_state.get(key) != style_key:
            self._span_drag_state[key] = style_key
            sc = self.scfg
            prop = actor.GetProperty()
            if degraded:
                prop.SetColor(sc.SPAN_FALLBACK_COLOR)
                prop.SetLineWidth(sc.SPAN_LINE_WIDTH)
                prop.SetOpacity(1.0)
            elif dragging:
                prop.SetColor(sc.SPAN_DRAG_COLOR)
                prop.SetLineWidth(sc.SPAN_DRAG_LINE_WIDTH)
                prop.SetOpacity(sc.SPAN_DRAG_OPACITY)
            else:
                prop.SetColor(sc.SPAN_COLOR)
                prop.SetLineWidth(sc.SPAN_LINE_WIDTH)
                prop.SetOpacity(1.0)
        actor.SetVisibility(self._layer_visible['blue'])

    def _span_count(self, sid: int) -> int:
        """Number of Bézier spans in spline *sid* (closed loops have one extra wrap-around span)."""
        n = len(self.splines[sid])
        if n < 2:
            return 0
        return n if self.splines_closed[sid] else n - 1

    def _span_pair(self, sid: int, i: int) -> tuple[GeodesicSegment, GeodesicSegment]:
        """Returns the ``(node_start, node_end)`` pair for span *i* of spline *sid*.

        For closed splines, index wraps around so that the last span
        connects the last node back to the first.
        """
        nodes = self.splines[sid]
        return nodes[i], nodes[(i + 1) % len(nodes)]

    @staticmethod
    def _adjacent_span_indices(idx: int, total: int, closed: bool) -> list[int]:
        """Returns the 1–2 span indices adjacent to node *idx*.

        A node participates in span ``idx-1`` (as end-node) and span
        ``idx`` (as start-node).  For closed splines, ``idx-1`` wraps
        via modulo so that node 0 correctly includes the closing span.
        For open splines, ``idx-1 < 0`` is discarded (no wrap).
        """
        if closed:
            candidates = [(idx - 1) % total, idx % total]
        else:
            candidates = [idx - 1, idx]
        # Deduplicate while preserving order, discard out-of-range
        seen = set()
        result = []
        for j in candidates:
            if 0 <= j < total and j not in seen:
                seen.add(j)
                result.append(j)
        return result

    def _recompute_spans(self, node=None) -> None:
        """Recomputes Bézier spans for the active spline.

        When *node* is provided, only spans adjacent to that node are
        recomputed (exactly 2 for interior nodes, 1 for endpoints of
        open splines).  Otherwise all spans are recomputed.

        During drag preview (``node.is_dragging and node.is_preview``),
        affected spans use LOD sampling and are drawn with the lighter
        drag style (``SPAN_DRAG_COLOR``, thinner).  On consolidation
        (debounce sets ``is_preview=False``), the same spans are
        recomputed at full quality with normal appearance.
        """
        sid = self.active_spline_idx
        nodes = self.splines[sid]
        total = self._span_count(sid)

        if node is not None:
            try:
                idx = nodes.index(node)
            except ValueError:
                return
            indices = self._adjacent_span_indices(
                idx, total, self.splines_closed[sid])
        else:
            indices = range(total)

        is_dragging = node is not None and node.is_dragging
        # Visual drag style: lighter/thinner while preview, normal on consolidation
        is_preview_drag = is_dragging and node.is_preview
        sc = self.scfg
        res = sc.DRAG_RESOLUTION if is_dragging else sc.RESOLUTION
        min_s = sc.DRAG_MIN_SAMPLES if is_dragging else sc.MIN_SAMPLES
        max_s = sc.DRAG_MAX_SAMPLES if is_dragging else sc.MAX_SAMPLES

        adaptive = sc.ADAPTIVE_SAMPLING
        for i in indices:
            n0, n1 = self._span_pair(sid, i)
            if n0.p_b is None or n1.p_a is None:
                self._set_span(sid, i, None)
                continue
            ctrl = [n0.origin, n0.p_b, n1.p_a, n1.origin]
            n = self.geo.adaptive_samples(ctrl, res, min_s, max_s)
            t_vals = GeodesicMesh.curvature_adaptive_t_vals(ctrl, n) if adaptive else None
            # Two-mode Bézier: during drag use fast Euclidean+projection
            # for H_out→H_in (path_12=None); on consolidation compute the
            # exact geodesic path_12 via compute_endpoint_local for a
            # semi-geodesic curve (~25ms extra per span).
            path_12 = None if is_dragging else self.geo.compute_endpoint_local(
                n0.p_b, n1.p_a)
            # Track fallbacks only on consolidation (when path_12 is
            # actually computed).  During drag the hybrid skips the
            # solver entirely so there is nothing to flag.
            if not is_dragging:
                self._mark_span_degraded(
                    (sid, i), self.geo._last_was_fallback)
            pts = self.geo.hybrid_de_casteljau_curve(
                ctrl, n0.path_b, n1.path_a, n, fast=is_dragging,
                t_vals=t_vals, path_12=path_12)
            # Phase 2 refinement: only when not dragging (no time pressure)
            if adaptive and not is_dragging and len(pts) >= 3:
                t2 = GeodesicMesh.refine_t_vals_by_curvature(pts, t_vals)
                if len(t2) > len(t_vals):
                    pts = self.geo.hybrid_de_casteljau_curve(
                        ctrl, n0.path_b, n1.path_a, len(t2),
                        t_vals=t2, path_12=path_12)
            projected = self.geo.project_smooth_batch(pts)
            if not is_dragging:
                projected = self.geo.subdivide_secant_chords(
                    projected, tol=self._secant_tol,
                    max_depth=self.scfg.SECANT_MAX_DEPTH)
            self._set_span(sid, i, projected, dragging=is_preview_drag)

        # Interpolation curve tracks node origins — recompute on every call.
        self._recompute_interp_curve(sid, is_dragging=is_dragging)

    # --- Interpolation curve (scipy B-spline through nodes, black) ---

    def _set_interp_curve(self, sid: int, pts: np.ndarray | None) -> None:
        """Updates the interpolation curve actor for spline *sid*."""
        if pts is None or len(pts) < 2:
            entry = self._interp_cache.get(sid)
            if entry is not None:
                entry[1].SetVisibility(False)
            return

        if sid not in self._interp_cache:
            pd = pv.PolyData()
            actor = self.plotter.add_mesh(pd, lighting=False, pickable=False)
            sc = self.scfg
            prop = actor.GetProperty()
            prop.SetColor(sc.INTERP_COLOR)
            prop.SetLineWidth(sc.INTERP_LINE_WIDTH)
            prop.SetOpacity(sc.INTERP_OPACITY)
            self._set_depth_priority(actor, sc.DEPTH_INTERP)
            self._interp_cache[sid] = (pd, actor)

        pd, actor = self._interp_cache[sid]
        update_line_inplace(pd, pts)
        actor.SetVisibility(self._layer_visible['interp'])

    def _recompute_interp_curve(self, sid: int | None = None,
                                is_dragging: bool = False) -> None:
        """Recomputes the scipy B-spline interpolation curve for spline *sid*.

        Fits a B-spline (``splprep`` with ``s=0``) through the node
        origins, evaluates it at adaptively-sampled parameter values,
        projects all points onto the surface, and (on consolidation)
        subdivides secant chords.  For closed splines, ``per=True``
        produces a periodic (wrap-around) curve.

        During drag (*is_dragging=True*) the expensive secant chord
        subdivision is skipped — the curve stays fast (~1-3 ms).  On
        consolidation, the full refinement runs (~5-10 ms).

        When *sid* is None, recomputes all splines.
        """
        if sid is None:
            for s in range(len(self.splines)):
                self._recompute_interp_curve(s, is_dragging=is_dragging)
            return

        nodes = self.splines[sid]
        if len(nodes) < 2:
            self._set_interp_curve(sid, None)
            return

        origins = np.array([n.origin for n in nodes], dtype=float)
        closed = self.splines_closed[sid]

        # Need at least k+1 points for degree k
        k = min(3, len(origins) - 1)
        if closed and len(origins) < k + 1:
            self._set_interp_curve(sid, None)
            return

        try:
            tck, u = splprep(
                [origins[:, 0], origins[:, 1], origins[:, 2]],
                s=0, k=k, per=closed)
        except Exception:
            self._set_interp_curve(sid, None)
            return

        # High base sample count — the 3D B-spline has no geodesic
        # awareness, so shorter initial chords reduce surface deviation.
        sc = self.scfg
        n = max(sc.INTERP_MIN_SAMPLES,
                self.geo.adaptive_samples(origins, sc.RESOLUTION,
                                          sc.INTERP_MIN_SAMPLES, 500))

        u_fine = np.linspace(0.0, 1.0, n)
        x, y, z = splev(u_fine, tck)
        raw_pts = np.column_stack((x, y, z))

        projected = self.geo.project_smooth_batch(raw_pts)
        # Secant subdivision only on consolidation — too slow for drag
        if not is_dragging:
            mean_edge = float(np.sqrt(self.geo._face_edge_len2.mean()))
            interp_tol = mean_edge * sc.INTERP_SECANT_TOL_FACTOR
            projected = self.geo.subdivide_secant_chords(
                projected, tol=interp_tol,
                max_depth=sc.INTERP_SECANT_MAX_DEPTH)

        self._set_interp_curve(sid, projected)

    # --- Background curve layer (fully geodesic orange) ---

    def _set_geo_span(self, sid: int, i: int, pts: np.ndarray | None,
                      computing: bool = False) -> None:
        """Updates the orange geodesic-curve actor for span *(sid, i)*.

        Mirrors ``_set_span`` but operates on ``_geo_span_cache`` with the
        geodesic visual style (orange, thick).

        When *pts* is None, clears the PolyData geometry AND hides the
        actor.  This prevents stale data from reappearing if
        ``_refresh_visuals`` later re-shows the actor.

        *computing* controls the visual style used while the worker
        is still producing points:

          - **Color**: ``GEO_COLOR_COMPUTING`` (dimmer orange) while
            computing, ``GEO_COLOR`` (full orange) on consolidation.
            Clear binary signal for "still working / done".
          - **Dashed** (optional, ``GEO_DASHED_WHILE_COMPUTING``): when
            True, the partial curve is rendered as alternating odd
            1-indexed segments so the polyline looks like a dashed
            line while refining.  Switches to a solid polyline at
            consolidation.  Disable the flag for a solid-dimmer look
            without dashes.

        Degraded spans (geodesic fell back to a straight line) override
        both and use ``SPAN_FALLBACK_COLOR`` — a failure signal that
        should dominate any progress indicator.
        """
        key = (sid, i)
        if pts is None or len(pts) < 2:
            entry = self._geo_span_cache.get(key)
            if entry is not None:
                # Clear geometry so stale data can't reappear
                entry[0].points = np.zeros((0, 3), dtype=float)
                entry[0].Modified()
                entry[1].SetVisibility(False)
            return

        if key not in self._geo_span_cache:
            pd = pv.PolyData()
            actor = self.plotter.add_mesh(pd, lighting=False, pickable=False)
            sc = self.scfg
            prop = actor.GetProperty()
            prop.SetColor(sc.GEO_COLOR)
            prop.SetLineWidth(sc.GEO_LINE_WIDTH)
            prop.SetOpacity(sc.GEO_OPACITY)

            self._set_depth_priority(actor, sc.DEPTH_ORANGE)
            self._geo_span_cache[key] = (pd, actor)

        pd, actor = self._geo_span_cache[key]
        sc = self.scfg
        use_dashed = computing and sc.GEO_DASHED_WHILE_COMPUTING
        if use_dashed:
            update_dashed_line_inplace(pd, pts)
        else:
            update_line_inplace(pd, pts)

        # Color priority: fallback > computing > final.
        prop = actor.GetProperty()
        if key in self._degraded_spans:
            prop.SetColor(sc.SPAN_FALLBACK_COLOR)
        elif computing:
            prop.SetColor(sc.GEO_COLOR_COMPUTING)
        else:
            prop.SetColor(sc.GEO_COLOR)
        actor.SetVisibility(self._layer_visible['orange'])

    def _submit_geodesic_spans(self, node: GeodesicSegment | None = None) -> None:
        """Submits affected spans for background orange de Casteljau computation.

        Identifies the same span indices as ``_recompute_spans`` (adjacent
        to *node*, or all spans if *node* is None).  For each span with
        complete control points, submits a worker to ``_work_mgr``.  Hides
        the orange actor while computation is in progress — the Master
        Clock timer will progressively reveal it as points arrive.
        """
        sid = self.active_spline_idx
        nodes = self.splines[sid]
        total = self._span_count(sid)
        if total == 0:
            return

        if node is not None:
            try:
                idx = nodes.index(node)
            except ValueError:
                return
            indices = self._adjacent_span_indices(
                idx, total, self.splines_closed[sid])
        else:
            indices = range(total)

        sc = self.scfg
        for i in indices:
            span_key = (sid, i)
            n0, n1 = self._span_pair(sid, i)
            if n0.p_b is None or n1.p_a is None:
                continue
            ctrl = [n0.origin, n0.p_b, n1.p_a, n1.origin]
            if n0.path_b is None or len(n0.path_b) < 2:
                continue
            if n1.path_a is None or len(n1.path_a) < 2:
                continue
            # Hide orange actor while recomputing
            self._set_geo_span(sid, i, None)
            # Submit the orange worker directly (no more green pipeline).
            self._work_mgr.submit_span(
                span_key, ctrl,
                n0.path_b, n1.path_a[::-1],
                sc.GEO_SAMPLES, adaptive=sc.ADAPTIVE_SAMPLING)
            self._orange_batch_total += 1

    def _cancel_geodesic_spans(self, node: GeodesicSegment) -> None:
        """Cancels running workers and hides geodesic actors for spans
        adjacent to *node*.  Called at the start of a drag.

        Cancellation works by closing the read end of the ``mp.Pipe``
        for each affected span (via ``_work_mgr.cancel_span``).  The
        worker's next ``send()`` raises ``BrokenPipeError`` and exits.
        """
        sid = self.active_spline_idx
        nodes = self.splines[sid]
        total = self._span_count(sid)
        if total == 0:
            return
        try:
            idx = nodes.index(node)
        except ValueError:
            return
        for j in self._adjacent_span_indices(
                idx, total, self.splines_closed[sid]):
            span_key = (sid, j)
            self._work_mgr.cancel_all_for_span(span_key)
            # Orange: hide immediately (too inaccurate to show stale)
            self._set_geo_span(sid, j, None)

    def _on_poll_timer(self, obj, event) -> None:
        """Master Clock heartbeat.

        Polls the orange worker pipes for progressive curve updates.
        Also detects camera movement and refreshes handle arrow scales.
        """
        super()._on_poll_timer(obj, event)
        has_worker_results = self._work_mgr.drain_queue()

        # Orange worker fallback reports — merge into the app-level set
        # so ``_set_geo_span`` (and blue ``_set_span``) can paint red.
        # Workers only mutate ``degraded_spans`` during 'done', which
        # coincides with ``dirty_spans`` so the state is consumed in the
        # same drain pass without extra synchronization.
        if self._work_mgr.degraded_spans:
            for span_key in list(self._work_mgr.degraded_spans):
                self._mark_span_degraded(span_key, True)
            self._work_mgr.degraded_spans.clear()

        needs_render = False

        # --- Orange progress HUD ---
        # Show "COMPUTING ORANGE k/N" while any orange span is active.
        # Clears when all spans finish.
        n_active = len(self._work_mgr.active_spans)
        if n_active > 0:
            done = self._orange_batch_total - n_active
            self._set_hud(
                f"COMPUTING ORANGE {done}/{self._orange_batch_total}",
                'orange')
            self._orange_hud_active = True
        elif self._orange_hud_active:
            # Just finished — reset batch counter and clear HUD
            self._orange_batch_total = 0
            self._orange_hud_active = False
            self._set_hud("ORANGE DONE", 'lime')

        # --- Camera-change detection: refresh arrow scales ---
        # Must run before the worker-results guard — arrows must update
        # even when no background workers are active (e.g. user is just
        # navigating the camera after finishing edits).
        cam = self.plotter.camera.position
        if cam != self._last_cam_pos:
            self._last_cam_pos = cam
            for _, _, node in self._iter_all_nodes():
                node.refresh_arrows(self.plotter)
            needs_render = True

        if not has_worker_results:
            if needs_render:
                self.plotter.render()
            return

        # --- Orange results: progressive update ---
        # Exclude dead spans — their worker died, don't show partial data
        dirty_orange = self._work_mgr.dirty_spans - self._work_mgr.dead_spans
        self._work_mgr.dirty_spans = set()
        for span_key in dirty_orange:
            sid, i = span_key
            if sid >= len(self.splines) or i >= self._span_count(sid):
                continue
            pts = self._work_mgr.get_points(span_key)
            if pts is not None:
                # Secant subdivision only on completion — avoids O(N²)
                # cost of re-subdividing the growing polyline on every
                # progressive point arrival.
                is_done = span_key in self._work_mgr.done_spans
                if is_done:
                    pts = self.geo.subdivide_secant_chords(
                        pts, tol=self._secant_tol,
                        max_depth=self.scfg.SECANT_MAX_DEPTH)
                    self._work_mgr.done_spans.discard(span_key)
                # ``computing`` is True while the worker is still active
                # (dimmer color, optional dashed pattern).  Flips to
                # False on the 'done' message and the curve consolidates
                # to full-orange solid.
                self._set_geo_span(*span_key, pts,
                                   computing=not is_done)
                needs_render = True

        # --- Dead workers: clear orphaned actor geometry ---
        dead = self._work_mgr.dead_spans
        if dead:
            self._work_mgr.dead_spans = set()
            for span_key in dead:
                self._set_geo_span(*span_key, None)
            needs_render = True

        if needs_render:
            self.plotter.render()

    # --- Save / Load ---

    def _on_save(self) -> None:
        """Saves all splines to a timestamped JSON file.

        Format: ``yyyymmdd_HHMMSS.json`` in the current directory.
        Each node stores only **2 fields** at full float64 precision:

          - **origin** (x, y, z) — 3-D position on the surface.
          - **tangent** (dx, dy, dz) — 3-D vector whose direction is the
            shoot direction for path_b and whose magnitude is h_length.

        On load, ``h_length = |tangent|``, direction = ``tangent / |tangent|``.
        ``face_idx``, normal, frame (u, v), ``local_v``, and all paths/handles
        are reconstructed via ``find_face`` + ``_build_local_frame`` +
        ``compute_shoot``.

        The 3-D tangent-with-magnitude is the most compact representation —
        two fields fully define the node's geometry, and it survives mesh
        remeshing as long as the surface shape is preserved.
        """
        data = {
            'version': 1,
            'mesh_file': self.mesh_label,
            'splines': [],
        }
        for sid, nodes in enumerate(self.splines):
            spline_data = {
                'closed': self.splines_closed[sid],
                'nodes': [],
            }
            for node in nodes:
                # 3D tangent with magnitude = shoot direction × h_length
                tangent_3d = (node.local_v[0] * node.u
                              + node.local_v[1] * node.v) * node.h_length
                spline_data['nodes'].append({
                    'origin': node.origin.tolist(),
                    'tangent': tangent_3d.tolist(),
                })
            data['splines'].append(spline_data)

        # numpy .tolist() produces Python floats which json.dump writes
        # with full repr precision (~17 significant digits) by default.
        fname = datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'
        with open(fname, 'w') as f:
            json.dump(data, f, indent=2)
        n_nodes = sum(len(s) for s in self.splines)
        self._set_hud(f"SAVED {n_nodes} nodes → {fname}", 'gold')
        print(f"[*] Saved {n_nodes} nodes across {len(self.splines)} "
              f"splines to {fname}")
        self.plotter.render()

    def _on_load(self) -> None:
        """Loads splines from a JSON file, replacing all current splines.

        Opens a file dialog defaulting to the most recent ``*.json`` in
        the current directory.
        Reconstructs each node via:

          1. ``|tangent|`` → h_length; ``tangent / |tangent|`` → direction
          2. ``find_face(origin)`` → face_idx
          3. ``_build_local_frame(origin, face_idx)`` → normal, u, v
          4. ``compute_shoot(origin, ±direction, h_length)`` → paths, handles
          5. ``update_local_v()`` → local_v from reconstructed paths

        All derived state is recomputed from the 2 saved fields per node.
        """
        import tkinter as tk
        from tkinter import filedialog

        # Find the most recent JSON for the default
        jsons = sorted(glob.glob('*.json'), reverse=True)
        initial_file = jsons[0] if jsons else ''

        root = tk.Tk()
        root.withdraw()
        fpath = filedialog.askopenfilename(
            title="Load splines",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=initial_file)
        root.destroy()

        if not fpath:
            return

        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[!] Failed to read {fpath}: {exc}")
            self._set_hud("LOAD FAILED", 'red')
            self.plotter.render()
            return

        version = data.get('version')
        if version != 1:
            print(f"[!] Unknown format version: {version}")
            self._set_hud("LOAD FAILED: unknown version", 'red')
            self.plotter.render()
            return

        self._push_undo()
        n_nodes = self._load_from_data(data)
        self._set_hud(f"LOADED {n_nodes} nodes from {fpath}", 'lime')
        print(f"[*] Loaded {n_nodes} nodes across {len(self.splines)} "
              f"splines from {fpath}")
        self.plotter.render()

    def _load_from_data(self, data: dict) -> int:
        """Replaces all splines with those described in *data*.

        Clears existing state (workers, actors, caches), reconstructs
        each node from the 2 saved fields (origin + tangent), and
        recomputes all derived geometry.

        Returns the total number of nodes loaded.
        """
        # --- Clear existing splines ---
        self._work_mgr.cancel_all()
        for seg in list(self.segments):
            seg.clear_actors(self.plotter)
        self.segments.clear()
        for cache in (self._span_cache, self._geo_span_cache):
            for pd, actor in cache.values():
                safe_remove_actor(self.plotter, actor)
            cache.clear()
        for pd, actor in self._interp_cache.values():
            safe_remove_actor(self.plotter, actor)
        self._interp_cache.clear()
        self._span_drag_state.clear()
        self._degraded_spans.clear()
        self.state.hover_seg = None
        self.state.hover_marker = None
        self.state.active_seg = None
        self._hover_dirty = True

        # --- Rebuild from JSON ---
        self.splines = []
        self.splines_closed = []

        for spline_data in data['splines']:
            nodes = []
            for nd in spline_data['nodes']:
                origin = np.array(nd['origin'], dtype=float)
                tangent_full = np.array(nd['tangent'], dtype=float)

                # Decompose tangent vector into direction + magnitude
                h_length = float(np.linalg.norm(tangent_full))
                if h_length > 1e-15:
                    tangent_dir = tangent_full / h_length
                else:
                    tangent_dir = np.array([1.0, 0.0, 0.0])
                    h_length = 0.01

                # Reconstruct face, frame, paths from origin + tangent
                face_idx = self.geo.find_face(origin)
                normal, u, v = self._build_local_frame(origin, face_idx)
                seg = GeodesicSegment(origin, face_idx, normal, u, v)
                seg.h_length = h_length
                seg.is_active = True

                seg.path_b = self.geo.compute_shoot(
                    origin, tangent_dir, h_length, face_idx)
                seg.path_a = self.geo.compute_shoot(
                    origin, -tangent_dir, h_length, face_idx)
                if seg.path_b is not None:
                    seg.p_b = seg.path_b[-1]
                if seg.path_a is not None:
                    seg.p_a = seg.path_a[-1]
                seg.update_local_v(self.geo)

                nodes.append(seg)
                self.segments.append(seg)
                seg.update_visuals(self.plotter)

            self.splines.append(nodes)
            self.splines_closed.append(bool(spline_data.get('closed', False)))

        # Ensure at least one spline exists
        if not self.splines:
            self.splines.append([])
            self.splines_closed.append(False)

        self.active_spline_idx = 0
        self._prev_active_spline_idx = 0
        self._rebuild_node_index()
        self._refresh_visuals()
        self._recompute_spans()
        self._submit_geodesic_spans()
        self._update_stitch()

        return sum(len(s) for s in self.splines)

    def cleanup(self) -> None:
        """Shuts down background workers and clears all curve-layer actors.

        Wraps actor removal in try/except because the plotter may already
        be closed (window X button) when cleanup runs.
        """
        self._work_mgr.shutdown()
        for cache in (self._span_cache, self._geo_span_cache):
            for pd, actor in cache.values():
                safe_remove_actor(self.plotter, actor)
            cache.clear()
        for pd, actor in self._interp_cache.values():
            safe_remove_actor(self.plotter, actor)
        self._interp_cache.clear()
        super().cleanup()

    # --- Visuals ---

    def _refresh_visuals(self) -> None:
        """Resets visual state for affected splines only.

        Tracks the previously active spline index to avoid iterating all
        nodes across all splines.  Only nodes in the old and new active
        splines are visited; span visibility is toggled only for relevant
        cache entries.  Falls back to a full sweep when the previous index
        is out of range (after spline deletion).
        """
        prev = self._prev_active_spline_idx
        curr = self.active_spline_idx
        self._prev_active_spline_idx = curr

        # Determine which spline indices need updating
        if prev == curr:
            affected = {curr}
        elif 0 <= prev < len(self.splines):
            affected = {prev, curr}
        else:
            # prev is stale (spline was deleted) — full sweep
            affected = None

        if affected is not None:
            for s_idx in affected:
                for node in self.splines[s_idx]:
                    changed = node.is_dimmed or not node.is_active
                    node.is_dimmed = False
                    node.is_active = True
                    if changed:
                        node.update_visuals(self.plotter)
            layer_vis = self._layer_visible
            for cache, layer in ((self._span_cache, 'blue'),
                                 (self._geo_span_cache, 'orange')):
                vis = layer_vis[layer]
                for (sid, _), (_, actor) in cache.items():
                    if sid in affected:
                        actor.SetVisibility(vis)
            interp_vis = layer_vis['interp']
            for sid, (_, actor) in self._interp_cache.items():
                if sid in affected:
                    actor.SetVisibility(interp_vis)
        else:
            for _, _, node in self._iter_all_nodes():
                changed = node.is_dimmed or not node.is_active
                node.is_dimmed = False
                node.is_active = True
                if changed:
                    node.update_visuals(self.plotter)
            layer_vis = self._layer_visible
            for cache, layer in ((self._span_cache, 'blue'),
                                 (self._geo_span_cache, 'orange')):
                vis = layer_vis[layer]
                for _, actor in cache.values():
                    actor.SetVisibility(vis)
            interp_vis = layer_vis['interp']
            for _, actor in self._interp_cache.values():
                actor.SetVisibility(interp_vis)


def _make_icosahedron(radius: float = 10.0, subdivisions: int = 2) -> pv.PolyData:
    """Creates a subdivided icosahedron with flat faces.

    Starts from a regular icosahedron (12 vertices, 20 faces) and applies
    *subdivisions* rounds of linear midpoint subdivision (each tri → 4).
    Midpoints are placed at the EDGE midpoint (linear interpolation),
    NOT re-projected to a sphere — so the surface stays polyhedral with
    flat triangular faces.  2 subdivisions give 80 faces.
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    verts *= radius
    F = np.array([
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
    ], dtype=int)
    V = verts
    for _ in range(subdivisions):
        edge_mids = {}
        new_V = list(V)
        new_F = []
        for f in F:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            mids = []
            for i, j in [(a, b), (b, c), (c, a)]:
                key = (min(i, j), max(i, j))
                if key not in edge_mids:
                    # Linear midpoint — NO sphere projection → flat faces
                    mid = (V[i] + V[j]) * 0.5
                    edge_mids[key] = len(new_V)
                    new_V.append(mid)
                mids.append(edge_mids[key])
            ab, bc, ca = mids
            new_F.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        V = np.array(new_V)
        F = np.array(new_F, dtype=int)
    n = len(F)
    pv_faces = np.column_stack([np.full(n, 3, dtype=int), F]).ravel()
    return pv.PolyData(V, faces=pv_faces).triangulate().clean()


# Sentinel mesh_path for the built-in icosahedron demo mesh.
ICOSAHEDRON = "ICOSAHEDRON"


if __name__ == "__main__":
    import json as _json
    import os
    import sys

    usage = (
        "Usage:\n"
        "  python geo_splines.py                  "
        "— demo icosahedron, no splines\n"
        "  python geo_splines.py mesh.ply          "
        "— open mesh, no splines\n"
        "  python geo_splines.py session.json       "
        "— open mesh + splines from JSON\n"
    )

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    json_path = None

    if arg is None:
        # No arguments → demo icosahedron (in-memory, no file)
        mesh_or_path = _make_icosahedron(radius=10.0)
        mesh_label = ICOSAHEDRON
    elif arg.lower().endswith('.json'):
        # JSON file → read mesh_path from it, load splines after init
        json_path = arg
        if not os.path.exists(json_path):
            print(f"[!] JSON file not found: {json_path}")
            sys.exit(1)
        with open(json_path, 'r') as f:
            data = _json.load(f)
        mesh_label = data.get('mesh_file', '')
        if mesh_label == ICOSAHEDRON:
            mesh_or_path = _make_icosahedron(radius=10.0)
        else:
            if not os.path.exists(mesh_label):
                print(f"[!] Mesh file not found: {mesh_label}")
                sys.exit(1)
            mesh_or_path = mesh_label
    else:
        # Mesh file directly
        if not os.path.exists(arg):
            print(f"[!] Mesh file not found: {arg}")
            sys.exit(1)
        mesh_or_path = arg
        mesh_label = arg

    try:
        app = GeodesicSplineApp(mesh_or_path, mesh_label=mesh_label)

        # Load splines from JSON if provided
        if json_path is not None:
            with open(json_path, 'r') as f:
                data = _json.load(f)
            if data.get('version') != 1:
                print(f"[!] Unknown JSON version: {data.get('version')}")
            else:
                n_nodes = app._load_from_data(data)
                print(f"[*] Loaded {n_nodes} nodes from {json_path}")

        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure workers are cleaned up even if init/load was interrupted
        if 'app' in locals() and hasattr(app, '_work_mgr'):
            try:
                app._work_mgr.shutdown()
            except Exception:
                pass
