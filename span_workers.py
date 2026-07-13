# SPDX-License-Identifier: Apache-2.0
"""
span_workers.py — orange-layer background-worker pipeline.

This light module holds the machinery that computes the *orange* (fully
geodesic) curve layer off the main thread, extracted out of the ~7k-line
``geo_splines.py`` GUI editor so the two concerns no longer share a module:

  - ``_geodesic_decasteljau_worker`` and its phase helpers
    (``_phase1_canonical``, ``_phase2_densify``, ``_phase3_chord_bridge``,
    plus ``_build_chord_geodesic`` and ``_hierarchical_inner_order``) — the
    three-phase de Casteljau cascade run inside each worker process.
  - ``_SpanWorkManager`` — the ``ProcessPoolExecutor`` coordinator: it
    publishes the mesh V/F via ``multiprocessing.shared_memory``, owns a
    per-span ``mp.Pipe`` for streaming results, and recovers a broken pool.
  - ``_process_initializer`` / the ``_process_geo`` module global — the
    per-worker setup that maps the shared V/F into a process-local
    ``GeodesicMesh``.

It imports only ``geodesics`` + ``numpy`` + the standard library — **not**
pyvista / vtk-GUI code.  ``ProcessPoolExecutor`` spawn children re-import
the module that defines the worker function by qualified name; because that
module is now ``span_workers`` (light) rather than ``geo_splines`` (the full
GUI editor), the children no longer drag the entire pyvista/VTK editor stack
into every worker process on startup.

(``geodesics`` transitively imports ``vtk``, so ``vtk`` *is* pulled in here —
that is expected and unavoidable; the win is avoiding the 7k-line editor
module and its pyvista imports.)

See ``docs/ARCHITECTURE.md`` for the worker pipeline internals and the
per-span-pipe ticket/generation invariant.
"""

from __future__ import annotations

import bisect
import logging
import multiprocessing as mp
import multiprocessing.shared_memory as _shm
import os
import signal
import sys
import weakref
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing.connection import Connection
from typing import Any

import numpy as np

from geodesics import GeodesicMesh, eval_cascade_at_t

# ---------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------
# A span is identified by ``(spline_index, span_index_within_spline)``.
# This tuple is used as a dict key in 4 caches (``_span_cache``,
# ``_geo_span_cache``, ``_span_drag_state``, ``_SpanWorkManager._points``)
# and as a set element in 5 sets (degraded / dead / dirty / done /
# active spans).  Naming the type makes those signatures legible and
# documents the intent — "tuple" alone could mean RGB, screen coords,
# or anything else.
SpanKey = tuple[int, int]

# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
# Reuse the ``geo_splines`` logger name so this module's diagnostics land
# on the same handler / stream as the editor's — log output is unchanged
# by the extraction.  ``_process_initializer`` still installs its own
# ``geo_splines.worker`` child-side logger (unchanged).
log = logging.getLogger("geo_splines")


# --- Process-local GeodesicMesh for background workers ---
# A bound on this module-level state: child processes spawn with a
# fresh interpreter so this attribute is set exactly once per worker
# (in ``_process_initializer``).  The parent process never reads it.
# It lives at module scope because ``ProcessPoolExecutor`` only
# accepts top-level callables for the initializer / worker functions
# (they must be importable by name on the child side).
_process_geo: GeodesicMesh | None = None


def _process_initializer(v_shm_name: str, v_shape: tuple, v_dtype: str,
                         f_shm_name: str, f_shape: tuple, f_dtype: str) -> None:
    """Creates a process-local ``GeodesicMesh`` from shared-memory arrays.

    Called once per worker process by ``ProcessPoolExecutor``.  Maps V and
    F from ``multiprocessing.shared_memory.SharedMemory`` blocks created
    by ``_SpanWorkManager``.  The mesh is built without PyVista, so no
    VTK locator is created.

    The previous implementation hard-redirected the worker's stderr to
    /dev/null to silence ``BrokenPipeError`` tracebacks during shutdown.
    That also swallowed legitimate import / runtime errors and made
    debugging painful.  We now install a logger-only handler at WARNING
    level: workers stay quiet on normal operation, but real failures
    surface to stderr with module + level prefixes that the parent can
    distinguish from the line noise.

    SIGINT handling
    ---------------
    Workers ignore ``SIGINT``.  On Ctrl+C the OS sends SIGINT to the
    parent **and** every child in the process group.  Without this
    guard each worker would interrupt its in-flight scipy / Intel-MKL
    Fortran call and the runtime would dump
    ``forrtl: error (200): program interrupted`` to stderr — four
    workers writing concurrently produced the unreadable stack-trace
    soup the user reported.  Ignoring SIGINT in the children leaves
    the parent's ``KeyboardInterrupt`` handler the sole graceful-exit
    path: it calls ``_work_mgr.shutdown()`` which uses
    ``executor.shutdown(cancel_futures=True)`` →
    ``TerminateProcess`` on Windows, killing the children at the OS
    level before any Fortran cleanup can run.
    """
    # Belt-and-braces against MKL's Console Control Handler: the
    # module-level ``os.environ.setdefault`` already runs in this
    # spawned child (Windows ``spawn`` inherits the parent's env, and
    # the child re-imports geo_splines to resolve this initializer),
    # but we re-assert it here in case a custom launcher cleared the
    # variable between fork and ``_process_initializer``.  Must precede
    # the scipy / pp3d import chain (triggered below by GeodesicMesh
    # construction) for the same reason as in the parent.
    os.environ.setdefault('FOR_DISABLE_CONSOLE_CTRL_HANDLER', '1')

    # Block SIGINT before anything else — the import of scipy / pp3d
    # below already pulls in MKL, and we want the SIGINT mask in place
    # before any Fortran call starts.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Install a stderr logger at WARNING level on the worker side.
    # Child sees its own copy of `log` after spawn; reset handlers so
    # the parent's stderr handler does not leak in via fork on POSIX.
    worker_log = logging.getLogger("geo_splines.worker")
    worker_log.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter(
        "[%(levelname)s] geo_splines.worker[%(process)d]: %(message)s"))
    worker_log.addHandler(h)
    worker_log.propagate = False
    worker_log.setLevel(
        logging.DEBUG if os.environ.get("GEO_SPLINES_DEBUG") else logging.WARNING)

    global _process_geo
    shm_v = _shm.SharedMemory(name=v_shm_name)
    V: np.ndarray = np.ndarray(v_shape, dtype=np.dtype(v_dtype), buffer=shm_v.buf)
    shm_f = _shm.SharedMemory(name=f_shm_name)
    F: np.ndarray = np.ndarray(f_shape, dtype=np.dtype(f_dtype), buffer=shm_f.buf)
    # GeodesicMesh copies V and F internally (np.asarray), so the shm
    # mapping can be closed after init without invalidating the mesh.
    # ``copy()`` defends against premature shm.close() on platforms
    # where the slice would otherwise stay attached to the buffer.
    #
    # ``build_locator=False``: the orange worker only calls
    # ``compute_endpoint_local`` which uses the KDTree, never the
    # VTK locator.  Skipping ``_build_locator`` saves ~250 ms of
    # PyVista + VTK init per worker and — critically on Windows
    # spawn-mode children — avoids the import-time chain that has
    # been observed to abort the worker (manifesting as a continuous
    # stream of "orange worker pipe broken on span (0, 0): WinError
    # 109" warnings as each respawned worker dies before sending
    # any results).
    _process_geo = GeodesicMesh(V.copy(), F.copy(), build_locator=False)
    shm_v.close()
    shm_f.close()


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


def _build_chord_geodesic(
    geo, p_left: np.ndarray, p_right: np.ndarray,
    submesh_subdiv: int = 0,
) -> tuple[np.ndarray, bool]:
    """Return ``(polyline, degraded)`` — a polyline that follows the
    mesh geodesic from *p_left* to *p_right*, plus a degraded flag.

    Used by the orange worker's phase-3 chord-bridging — once all
    cascade samples are computed, every pair of consecutive samples
    is connected by an exact mesh geodesic so the rendered polyline
    hugs the surface instead of cutting across in straight 3-D
    chords.

    Tries the cheap fast path first: if both points fall in adjacent
    triangles, ``short_geodesic`` returns a 2- or 3-point polyline in
    a few µs (no solver call) — the submesh subdivision flag is
    irrelevant for this case (the answer is exact regardless).
    Otherwise falls back to the full ``compute_endpoint_local`` with
    the requested *submesh_subdiv* level (~25-100 ms).  Last resort,
    when even the full solver fails (disconnected components,
    malformed input), is a degenerate two-point Euclidean polyline
    — at least the geometry can still be rendered.

    *degraded* is ``True`` whenever the returned polyline is that
    Euclidean stand-in rather than a real geodesic: the solver raised,
    returned nothing usable, or reported its own ``was_fallback``
    straight-line stub.  The caller must OR it into the span's
    ``degraded_any`` flag — a straight chord that renders without the
    red fallback repaint is exactly the "phantom curve" the editor
    promises never to show.
    """
    seg = geo.short_geodesic(p_left, p_right)
    if seg is not None:
        return seg, False
    try:
        seg, was_fallback = geo.compute_endpoint_local(
            p_left, p_right, submesh_subdiv=submesh_subdiv)
    except (RuntimeError, ValueError, TypeError, IndexError):
        seg, was_fallback = None, True
    if seg is None or len(seg) < 2:
        return np.stack([p_left, p_right]), True
    return seg, bool(was_fallback)


def _phase1_canonical(
    geo, span_key: SpanKey,
    P0: np.ndarray, P1: np.ndarray,
    t_grid: np.ndarray, inner_order: list[int],
    eval_args: tuple, writer, *,
    submesh_subdiv: int, use_full_mesh: bool,
) -> tuple[list[float], list[np.ndarray], bool]:
    """Phase 1 of the orange worker: canonical *t_grid* samples.

    Evaluates the de Casteljau cascade at every inner index of *t_grid*
    in *inner_order* (hierarchical: midpoint → quarters → eighths …),
    sending each as ``('point', span_key, t, point)``.  Returns the
    worker-side t-sorted polyline state ``(t_list, p_list)`` plus the
    degraded flag.
    """
    # Worker-side sorted polyline state, mirrored on the parent.
    # Endpoints are seeded so phase-2 chord pairs starting at idx 0
    # and ending at idx N-1 are well-defined from the first iter.
    t_list: list[float] = [float(t_grid[0]), float(t_grid[-1])]
    p_list: list[np.ndarray] = [np.asarray(P0, dtype=float),
                                np.asarray(P1, dtype=float)]
    degraded = False

    for idx in inner_order:
        t = float(t_grid[idx])
        point, deg = eval_cascade_at_t(
            geo, t, *eval_args,
            submesh_subdiv=submesh_subdiv,
            use_full_mesh=use_full_mesh)
        if deg:
            degraded = True
        # Insert in t-sorted order on the worker side too — phase 2
        # walks consecutive pairs and needs them ordered.
        pos = bisect.bisect_left(t_list, t)
        t_list.insert(pos, t)
        p_list.insert(pos, point)
        writer.send(('point', span_key, t, point))

    return t_list, p_list, degraded


def _phase2_densify(
    geo, span_key: SpanKey,
    t_list: list[float], p_list: list[np.ndarray],
    eval_args: tuple, writer, *,
    deviation_mode: str, subdiv_tol_factor: float, subdiv_max_depth: int,
    submesh_subdiv: int, use_full_mesh: bool,
) -> bool:
    """Phase 2 of the orange worker: cascade densification.

    Walks consecutive sample pairs of the t-sorted state, decides which
    to subdivide per *deviation_mode* (see the worker docstring), and
    inserts fresh cascade evaluations at midpoint *t*-values — each sent
    immediately as ``('point', span_key, t_mid, pt)`` and merged into
    *t_list* / *p_list* in place.  Returns the degraded flag.
    """
    degraded = False
    mean_edge = float(np.sqrt(geo._face_edge_len2.mean()))
    tol_sq = (mean_edge * subdiv_tol_factor) ** 2

    for _level in range(subdiv_max_depth):
        n_pairs = len(t_list) - 1
        if n_pairs <= 0:
            break

        # Build per-pair midpoints in 3D (chord midpoints) and
        # decide which need a cascade insertion.
        pts_arr = np.asarray(p_list)
        mids = (pts_arr[:-1] + pts_arr[1:]) * 0.5

        # Where to evaluate the cascade.  Done per-pair below; in
        # 'cascade' mode every pair is evaluated; in 'surface' mode
        # only the pairs flagged by the projection test.
        inserts: list[tuple[float, np.ndarray]] = []

        if deviation_mode == 'surface':
            projected = geo.project_smooth_batch(mids)
            diff = projected - mids
            dev_sq = np.sum(diff * diff, axis=1)
            needs_split = dev_sq > tol_sq
            if not needs_split.any():
                break
            for i in range(n_pairs):
                if not needs_split[i]:
                    continue
                t_mid = (t_list[i] + t_list[i + 1]) * 0.5
                pt, deg = eval_cascade_at_t(
                    geo, t_mid, *eval_args,
                    submesh_subdiv=submesh_subdiv,
                    use_full_mesh=use_full_mesh)
                if deg:
                    degraded = True
                inserts.append((t_mid, pt))
                writer.send(('point', span_key, t_mid, pt))
        else:  # 'cascade' (default)
            any_split = False
            for i in range(n_pairs):
                t_mid = (t_list[i] + t_list[i + 1]) * 0.5
                pt, deg = eval_cascade_at_t(
                    geo, t_mid, *eval_args,
                    submesh_subdiv=submesh_subdiv,
                    use_full_mesh=use_full_mesh)
                if deg:
                    degraded = True
                diff = pt - mids[i]
                if float(np.dot(diff, diff)) > tol_sq:
                    inserts.append((t_mid, pt))
                    writer.send(('point', span_key, t_mid, pt))
                    any_split = True
            if not any_split:
                break

        # Merge inserts into the sorted state — bisect each so the
        # invariant is preserved without a full re-sort.
        for t_mid, pt in inserts:
            pos = bisect.bisect_left(t_list, t_mid)
            t_list.insert(pos, t_mid)
            p_list.insert(pos, pt)

    return degraded


def _phase3_chord_bridge(
    geo, span_key: SpanKey,
    p_list: list[np.ndarray], writer, *,
    submesh_subdiv: int,
) -> bool:
    """Phase 3 of the orange worker: geodesic chord bridging.

    Connects every consecutive sample pair with an exact mesh geodesic
    (:func:`_build_chord_geodesic`) and sends the concatenated polyline
    once as ``('chord_geo', span_key, polyline)``.  Returns ``True``
    when any chord degraded to a straight Euclidean segment, so the
    caller can fold it into the ``('done', ...)`` degraded flag and
    the parent repaints the span red.
    """
    degraded = False
    polyline_segs: list[np.ndarray] = []
    for i in range(len(p_list) - 1):
        seg, deg = _build_chord_geodesic(geo, p_list[i], p_list[i + 1],
                                         submesh_subdiv=submesh_subdiv)
        degraded |= deg
        polyline_segs.append(seg)
    # Concatenate, dropping the duplicated joint between
    # consecutive segments so the polyline has no zero-length
    # segments.
    full = [polyline_segs[0]]
    for seg in polyline_segs[1:]:
        full.append(seg[1:])
    polyline = np.concatenate(full, axis=0)
    writer.send(('chord_geo', span_key, polyline))
    return degraded


def _geodesic_decasteljau_worker(
    span_key: SpanKey,
    ctrl: list[np.ndarray],
    path_b: np.ndarray,
    path_a_rev: np.ndarray,
    t_grid: np.ndarray,
    inner_order: list[int],
    writer,
    *,
    deviation_mode: str = 'cascade',
    subdiv_tol_factor: float = 0.01,
    subdiv_max_depth: int = 6,
    chord_bridging: bool = True,
    submesh_subdiv: int = 0,
    use_full_mesh: bool = False,
) -> None:
    """Background worker: computes the orange (fully geodesic) curve.

    Runs in a ``ProcessPoolExecutor`` child process.  Uses the process-
    local ``_process_geo`` (created by ``_process_initializer``) — no VTK
    objects, no GIL contention with the main thread.

    Three-phase pipeline
    ====================

    **Phase 1 — canonical samples** (:func:`_phase1_canonical`).
    Evaluates the de Casteljau cascade
    at every position of *t_grid* (length ``n_samples``, default 33),
    visiting indices in *inner_order* (hierarchical refinement: midpoint
    → quarters → eighths …) so the parent can render the curve coarse-
    to-fine.  Each sample is sent as ``('point', span_key, t, result)``.
    The two endpoints (idx 0, idx N-1) are NOT computed by the worker —
    they coincide with node origins and are pre-seeded by the parent.

    **Phase 2 — cascade densification** (:func:`_phase2_densify`).
    Walks all consecutive sample
    pairs, decides which to subdivide using *deviation_mode*, and
    inserts new cascade samples at the midpoint *t*-value.  Each
    insertion is sent immediately as ``('point', span_key, t_mid,
    point)`` — the parent re-sorts on arrival, so the rendered
    polyline refines progressively in problem regions.  Recursive up
    to *subdiv_max_depth* levels.  Two criteria are supported:

      * ``deviation_mode='cascade'`` (default): for each pair, evaluate
        the cascade at ``t_mid`` and split if
        ``|chord_midpoint - cascade_eval| > tol``.  Always pays the
        cascade cost (3 × ``compute_endpoint_local`` per pair) but
        measures deviation from the *true* curve.
      * ``deviation_mode='surface'``: project chord midpoint onto the
        mesh and split if ``|chord_midpoint - projection| > tol``.
        Cheaper (one ``project_smooth_batch`` per pair), only triggers
        when the chord pierces the surface — useful as a fallback when
        ``cascade`` is too slow.  Inserted point is still the cascade
        evaluation, so geometric quality is identical between modes;
        only the *decision* of whether to split differs.

    Tolerance: ``tol = mean_edge_length * subdiv_tol_factor``.

    **Phase 3 — chord bridging** (:func:`_phase3_chord_bridge`).
    Once all densification is complete,
    every consecutive sample pair is connected by an exact mesh
    geodesic via :func:`_build_chord_geodesic` (``short_geodesic`` fast
    path, ``compute_endpoint_local`` fallback).  The polyline is sent
    once as ``('chord_geo', span_key, polyline)``; the parent replaces
    the actor geometry wholesale.  Chords whose solvers all fail
    degrade to straight Euclidean segments and feed the degraded
    flag, same as the cascade phases.  Skipped when *chord_bridging*
    is False — in that case the parent renders the t-sorted polyline
    as Euclidean chords between samples.

    Finally a ``('done', span_key, degraded_any)`` message terminates
    the worker.  The *degraded_any* flag triggers the red-fallback
    repaint on the parent if any solver call — in any of the three
    phases — hit a straight-line path.

    Failure modes
    -------------
    Pipe closed (span cancelled): next ``send()`` raises
    ``BrokenPipeError``; worker exits silently.

    Any other exception: the outer ``except Exception`` (BLE001) is
    deliberate — we capture the traceback and forward it as
    ``('error', span_key, repr, traceback)`` so a real diagnostic
    surfaces in the editor's HUD / stderr instead of the parent's
    drain loop seeing a mysterious "pipe broken" warning.
    """
    try:
        geo = _process_geo
        assert geo is not None, "_process_initializer must run before _geodesic_decasteljau_worker"
        P0, H_out, H_in, P1 = ctrl

        cum_b, total_b = GeodesicMesh.compute_path_lengths(path_b)
        cum_a, total_a = GeodesicMesh.compute_path_lengths(path_a_rev)

        degraded_any = False

        # Cache the level-1 middle path (constant across all t).
        # Honours ``use_full_mesh`` so the cached ``path_12`` matches
        # the per-t inner solver behaviour — mixing full-mesh inner
        # calls with a submesh-derived ``path_12`` would let the
        # submesh artifact leak in via ``b12 = lerp(path_12, t)``.
        try:
            if use_full_mesh:
                path_12, fb12 = geo.compute_endpoint(H_out, H_in)
            else:
                path_12, fb12 = geo.compute_endpoint_local(
                    H_out, H_in, submesh_subdiv=submesh_subdiv)
        except (RuntimeError, ValueError, TypeError, IndexError) as exc:
            logging.getLogger("geo_splines.worker").debug(
                "level-1 path_12 solver failed: %s", exc)
            path_12, fb12 = None, True
        if path_12 is None or len(path_12) < 2:
            path_12, degraded_any = np.array([H_out, H_in]), True
        elif fb12:
            degraded_any = True
        cum_12, total_12 = GeodesicMesh.compute_path_lengths(path_12)

        eval_args = (path_b, cum_b, total_b,
                     path_a_rev, cum_a, total_a,
                     path_12, cum_12, total_12)

        # Phase 1 — canonical N=GEO_SAMPLES grid.
        t_list, p_list, deg = _phase1_canonical(
            geo, span_key, P0, P1, t_grid, inner_order, eval_args, writer,
            submesh_subdiv=submesh_subdiv, use_full_mesh=use_full_mesh)
        degraded_any |= deg

        # Phase 2 — cascade densification (mutates t_list / p_list).
        degraded_any |= _phase2_densify(
            geo, span_key, t_list, p_list, eval_args, writer,
            deviation_mode=deviation_mode,
            subdiv_tol_factor=subdiv_tol_factor,
            subdiv_max_depth=subdiv_max_depth,
            submesh_subdiv=submesh_subdiv, use_full_mesh=use_full_mesh)

        # Phase 3 — chord bridging via short / full geodesics.
        if chord_bridging and len(p_list) >= 2:
            degraded_any |= _phase3_chord_bridge(
                geo, span_key, p_list, writer,
                submesh_subdiv=submesh_subdiv)

        writer.send(('done', span_key, degraded_any))
    except (BrokenPipeError, OSError):
        pass  # pipe closed — span was cancelled, exit silently
    except Exception as exc:  # noqa: BLE001 — must surface unknown failures
        import traceback as _tb
        tb_str = _tb.format_exc()
        try:
            writer.send(('error', span_key, repr(exc), tb_str))
        except (BrokenPipeError, OSError):
            pass
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

    Pipe-per-span as an implicit ticket / generation system
    -------------------------------------------------------
    A common review question on multiprocessing curve renderers is:
    *"What stops a stale background result from overwriting the curve
    after the user has already deleted or undone the segment?"*
    The textbook answer is to attach a generation counter (a ticket) to
    every job and discard incoming results whose ticket no longer
    matches the current generation for that span_key.

    This module **does not** carry an explicit generation counter and
    does not need one — the per-span pipe topology gives the same
    isolation guarantee for free:

      1. ``submit_span(span_key, ...)`` always calls
         ``cancel_span(span_key)`` first, which closes the *old*
         reader end of the pipe.
      2. A **brand-new** ``mp.Pipe(duplex=False)`` is then created.
         The fresh ``writer`` is shipped to the new worker; the fresh
         ``reader`` is mapped to ``span_key`` in ``self._readers``.
      3. The previous worker, on its next ``send()``, hits
         ``BrokenPipeError`` (its writer is no longer connected to a
         live reader) and exits silently.  Any partial messages it had
         already pushed into the OS pipe buffer are discarded the
         moment the parent's ``reader.close()`` returns — there is no
         shared queue from which they could be re-read.
      4. The new worker writes only to ``writer_new``; the parent only
         reads from ``reader_new``.  Cross-batch contamination is
         topologically impossible.

    In other words, **the pipe object itself acts as the ticket**:
    creating a new pipe is equivalent to incrementing a generation
    counter, and the old pipe's death is equivalent to discarding any
    result that carries the previous generation.  This is enforced by
    the OS / Python runtime, not by application code, which makes the
    invariant easier to reason about and impossible to forget.

    The same logic applies to span-key reuse: when a node is deleted
    and a new one is added at the same ``(sid, i)``, the ``submit_span``
    call clears all state for that key (``_points``, ``_futures``,
    ``done_spans``, ``active_spans``) before installing the new pipe,
    so even a same-key resubmit cannot inherit stale data from the
    previous lifetime.

    The single remaining race-window is between ``cancel_span`` and the
    next ``drain_queue``: an in-flight ``'point'`` message that arrived
    just before ``reader.close()`` is silently dropped by the OS.  That
    is the desired behaviour — we *want* cancelled work to be invisible.
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

        self._max_workers = max_workers
        # Init args captured for ``_rebuild_executor``: when a worker
        # dies abnormally (segfault in pp3d / VTK, OOM kill, etc.)
        # the entire ``ProcessPoolExecutor`` becomes unusable —
        # ``submit()`` raises ``BrokenProcessPool`` permanently.
        # Re-creating the pool is the only recovery path; the V/F
        # shared memory blocks survive intact, so we just spin up a
        # fresh executor with the same initializer.
        self._init_args = (
            self._shm_V.name, V_c.shape, str(V_c.dtype),
            self._shm_F.name, F_c.shape, str(F_c.dtype))
        self._executor = self._build_executor()

        # Safety net: if the parent crashes before ``shutdown()`` runs,
        # atexit still fires during interpreter teardown and releases the
        # /dev/shm blocks (on POSIX) so they don't leak across sessions.
        # ``shutdown()`` is idempotent so calling it twice is harmless.
        # We register via ``weakref.finalize`` so a per-instance handle
        # is recorded — multiple managers in one interpreter (rare, but
        # possible in tests) do not share a single ``atexit`` slot, and
        # a manager that is garbage-collected before interpreter exit
        # releases its handler eagerly.
        self._finalizer = weakref.finalize(self, _SpanWorkManager._cleanup_at_exit,
                                           weakref.ref(self))

        # --- Orange (fully geodesic) tracking ---
        # ``_points[key]`` is a per-span state dict with three keys:
        #   - 't_list':   list[float]            (t values, sorted)
        #   - 'p_list':   list[np.ndarray]       (cascade samples, t-aligned)
        #   - 'polyline': np.ndarray | None      (phase-3 chord-bridged override)
        # ``Any``-valued to keep static-typed setitem (e.g. ``state['polyline']
        # = ndarray``) checkers happy; a TypedDict would be more precise but
        # the dict is read in only a handful of places.
        self._readers: dict[SpanKey, Connection] = {}
        self._futures: dict[SpanKey, Future] = {}
        self._points: dict[SpanKey, dict[str, Any]] = {}
        self.dirty_spans: set[SpanKey] = set()
        self.done_spans: set[SpanKey] = set()  # spans whose worker sent 'done'
        # Spans whose worker reported a geodesic fallback.  The main
        # thread consumes this set after ``drain_queue`` and repaints
        # the affected orange/blue actors in red.
        self.degraded_spans: set[SpanKey] = set()

        # Spans whose worker died unexpectedly (pipe broken) — main
        # thread should clear the actor geometry on next poll tick.
        self.dead_spans: set[SpanKey] = set()

        # Spans that are actively being computed (submitted but not yet
        # done/cancelled/dead).  Used by the UI to show a progress HUD.
        self.active_spans: set[SpanKey] = set()

        # Batch progress counters.  ``_batch_submitted`` reflects the
        # current outstanding work plus completed-since-idle; cancelling
        # a span decrements it (so the HUD does not lie when the user
        # rapid-fires submit/cancel cycles).  ``_batch_done`` only grows
        # via real ``'done'`` messages.  Both reset to 0 the moment
        # ``active_spans`` becomes empty (see ``maybe_reset_progress``).
        self._batch_submitted: int = 0
        self._batch_done: int = 0

        # Warm up: force all worker processes to start now.  The futures
        # are intentionally discarded — submitting is enough to spin up
        # the child processes via ``ProcessPoolExecutor``'s lazy spawn.
        # Holding them in ``self.`` would just be a zombie attribute.
        for _ in range(max_workers):
            self._executor.submit(int, 0)

    def _build_executor(self) -> ProcessPoolExecutor:
        """Spins up a fresh ``ProcessPoolExecutor`` with the saved
        initializer + initargs.

        Used both at construction and by ``_rebuild_executor`` after a
        ``BrokenProcessPool``.  The shared-memory blocks for V / F are
        unchanged across rebuilds — only the worker processes are
        replaced — so the new pool sees the same mesh.
        """
        return ProcessPoolExecutor(
            max_workers=self._max_workers,
            initializer=_process_initializer,
            initargs=self._init_args)

    def _rebuild_executor(self) -> None:
        """Replace a broken executor with a fresh one and clear pending state.

        ``ProcessPoolExecutor`` becomes permanently unusable after any
        worker dies abnormally (segfault in pp3d / VTK, signal, OOM
        kill).  All subsequent ``submit()`` calls raise
        ``BrokenProcessPool``.  This method is the recovery path:

          1. Force-shutdown the broken pool (``cancel_futures=True`` so
             any pending futures fail fast — they cannot complete on a
             broken pool anyway).
          2. Drop all bookkeeping for the orange batch: ``_readers`` /
             ``_futures`` / ``_points`` / ``active_spans`` /
             ``done_spans`` / ``dirty_spans`` / counters.  Spans that
             were mid-flight will simply have to be resubmitted by the
             caller (the editor's next ``_recompute_spans`` does this
             automatically for the active spline).
          3. Build a fresh executor with the same initializer + V / F
             shared-memory args.

        Called from ``submit_span`` when ``executor.submit`` raises.
        Idempotent only insofar as a freshly-built executor will not
        immediately be broken — if pp3d crashes again on the next
        submit it will re-trigger this path.
        """
        log.warning("orange worker pool broken; rebuilding executor")
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:  # noqa: BLE001 — broken pool teardown
            log.debug("broken-pool shutdown raised: %s", exc)

        # Drop all per-span state — those readers / futures point at
        # workers from the dead pool and can never produce results.
        for r in self._readers.values():
            try:
                r.close()
            except OSError:
                pass
        self._readers.clear()
        self._futures.clear()
        self._points.clear()
        self.active_spans.clear()
        self.dirty_spans.clear()
        self.done_spans.clear()
        self.dead_spans.clear()
        self.degraded_spans.clear()
        self._batch_submitted = 0
        self._batch_done = 0

        self._executor = self._build_executor()

    @staticmethod
    def _cleanup_at_exit(weak_self) -> None:
        """``weakref.finalize`` callback — calls ``shutdown`` if alive."""
        target = weak_self()
        if target is not None:
            try:
                target.shutdown()
            except Exception as exc:  # noqa: BLE001 — interpreter teardown is best-effort
                log.debug("worker manager finalize: %s", exc)

    # --- Fully geodesic (orange) ---

    def submit_span(self, span_key: SpanKey,
                    ctrl: list[np.ndarray], path_b: np.ndarray,
                    path_a_rev: np.ndarray, n_samples: int,
                    adaptive: bool = False,
                    *,
                    deviation_mode: str = 'cascade',
                    subdiv_tol_factor: float = 0.01,
                    subdiv_max_depth: int = 6,
                    chord_bridging: bool = True,
                    submesh_subdiv: int = 0,
                    use_full_mesh: bool = False) -> None:
        """Submits a fully geodesic (orange) worker.

        Per-span state on the parent is a t-sorted ``(t_list, p_list)``
        pair plus an optional ``polyline`` override:

          * ``t_list``, ``p_list``: the cascade samples, t-sorted.  Phase
            1 fills it with the canonical *n_samples* grid; phase 2
            inserts midpoint samples wherever the chord deviates from
            the true curve beyond *subdiv_tol_factor × mean_edge*.  The
            rendered polyline at any moment is just ``p_list`` connected
            by straight chords — it refines coarse-to-fine as worker
            messages arrive.
          * ``polyline``: phase-3 chord-bridging output, when
            *chord_bridging* is True.  Replaces the chord-connected
            view wholesale once the worker is done — every consecutive
            sample pair is now joined by an exact mesh geodesic
            (``short_geodesic`` fast path or ``compute_endpoint_local``
            fallback) so the rendered curve hugs the surface in
            problematic regions instead of cutting through.

        Cascade-densification config
        ----------------------------
        *deviation_mode*, *subdiv_tol_factor*, *subdiv_max_depth*,
        *chord_bridging* are forwarded verbatim to the worker — see
        :func:`_geodesic_decasteljau_worker` for the contract.

        Cancellation-by-pipe
        --------------------
        Re-submitting an already-active span replaces it.  We let
        ``cancel_span`` decrement ``_batch_submitted`` so that the
        increment below is balanced (otherwise rapid resubmits inflate
        the HUD numerator forever).

        Cancel-then-new-pipe is also our **ticket / generation system**:
        the freshly created pipe below is unreachable to the previous
        worker (its writer end is now dangling), so any stale result
        it might still produce can never reach this reader.  See the
        class docstring for the full rationale.
        """
        self.cancel_span(span_key)
        reader, writer = mp.Pipe(duplex=False)
        self._readers[span_key] = reader

        # Per-span state — t-sorted polyline buffers + phase-3 override.
        # Endpoints (P0, P1) are seeded so the very first render is a
        # straight line between the two node origins, refined as the
        # worker streams cascade samples in.
        state = {
            't_list': [0.0, 1.0],
            'p_list': [np.asarray(ctrl[0], dtype=float),
                       np.asarray(ctrl[3], dtype=float)],
            'polyline': None,
        }
        self._points[span_key] = state

        if adaptive:
            t_grid = GeodesicMesh.curvature_adaptive_t_vals(ctrl, n_samples)
        else:
            t_grid = np.linspace(0.0, 1.0, n_samples)
        inner_order = _hierarchical_inner_order(n_samples)

        # ``executor.submit`` raises ``BrokenProcessPool`` if a worker
        # has died abnormally since the last call (segfault in pp3d /
        # VTK, OOM kill).  The pool is permanently unusable in that
        # state — we rebuild it once and retry.  If the second submit
        # also fails the workers are likely dying on a malformed input
        # we'd just keep retrying; mark the span dead so the editor's
        # poll-tick clears its (stale) actor and moves on.
        from concurrent.futures.process import BrokenProcessPool
        worker_kwargs = dict(
            deviation_mode=deviation_mode,
            subdiv_tol_factor=subdiv_tol_factor,
            subdiv_max_depth=subdiv_max_depth,
            chord_bridging=chord_bridging,
            submesh_subdiv=submesh_subdiv,
            use_full_mesh=use_full_mesh,
        )
        try:
            future = self._executor.submit(
                _geodesic_decasteljau_worker,
                span_key, ctrl, path_b.copy(), path_a_rev.copy(),
                t_grid, inner_order, writer,
                **worker_kwargs)
        except BrokenProcessPool:
            self._rebuild_executor()
            # ``_rebuild_executor`` closed EVERY reader (including the one
            # created above) and cleared self._readers / self._points.
            # The original (reader, writer) pair is therefore dead: its
            # read-end is closed, so a retried worker writing to the old
            # ``writer`` would surface only as an ``OSError`` on the next
            # ``drain_queue`` poll — killing the very span we are trying
            # to recover.  Close the orphaned write-end and mint a FRESH
            # pipe for the retry.
            try:
                writer.close()
            except OSError:
                pass
            reader, writer = mp.Pipe(duplex=False)
            self._readers[span_key] = reader
            self._points[span_key] = state
            try:
                future = self._executor.submit(
                    _geodesic_decasteljau_worker,
                    span_key, ctrl, path_b.copy(), path_a_rev.copy(),
                    t_grid, inner_order, writer,
                    **worker_kwargs)
            except BrokenProcessPool as exc:
                log.error("orange worker pool broken twice in a row "
                          "for span %s: %s — giving up", span_key, exc)
                try:
                    reader.close()
                except OSError:
                    pass
                self._readers.pop(span_key, None)
                self._points.pop(span_key, None)
                self.dead_spans.add(span_key)
                return
        self._futures[span_key] = future
        self.active_spans.add(span_key)
        self._batch_submitted += 1

        # Surface "task itself failed before running" exceptions
        # (e.g. signature mismatch, pickling error) — without this,
        # ProcessPoolExecutor swallows them silently into the Future
        # and the parent only sees a generic "pipe broken" warning
        # when the writer goes out of scope on the worker side.
        # Done via add_done_callback so the wait happens on the
        # executor's bookkeeping thread, never blocking the main
        # thread; we only act on a non-None exception().
        def _surface_future_error(fut, _key=span_key):
            exc = fut.exception()
            if exc is not None and not isinstance(
                    exc, (BrokenPipeError, OSError)):
                log.error(
                    "orange worker future for span %s raised %s: %s",
                    _key, type(exc).__name__, exc)
        future.add_done_callback(_surface_future_error)

    def cancel_span(self, span_key: SpanKey) -> None:
        """Closes the pipe for the fully geodesic worker on *span_key*.

        If the span was actively counted in the current batch, the
        ``_batch_submitted`` counter is decremented so the progress HUD
        stays accurate across submit/cancel/submit cycles.
        """
        was_active = span_key in self.active_spans
        reader = self._readers.pop(span_key, None)
        if reader is not None:
            try:
                reader.close()
            except OSError as exc:
                log.debug("cancel_span: reader close failed (%s)", exc)
        self._futures.pop(span_key, None)
        self._points.pop(span_key, None)
        self.done_spans.discard(span_key)
        self.active_spans.discard(span_key)
        if was_active and self._batch_submitted > 0:
            self._batch_submitted -= 1

    # --- Shared ---

    def cancel_all_for_span(self, span_key: SpanKey) -> None:
        """Cancels the orange worker for *span_key*."""
        self.cancel_span(span_key)

    def cancel_all(self) -> None:
        """Cancels all active orange workers and resets batch counters."""
        for r in self._readers.values():
            try:
                r.close()
            except OSError as exc:
                log.debug("cancel_all: reader close failed (%s)", exc)
        self._readers.clear()
        self._futures.clear()
        self._points.clear()
        self.active_spans.clear()
        # Clear the per-span result flags too (matches _rebuild_executor).
        # These are keyed by (spline_idx, span_idx); leaving them behind
        # lets a stale generation bleed into the next one — a freshly
        # submitted span whose key happens to sit in done_spans renders
        # "final" on its first partial polyline, and a leftover
        # degraded_spans key repaints a healthy new span red.
        self.dirty_spans.clear()
        self.done_spans.clear()
        self.dead_spans.clear()
        self.degraded_spans.clear()
        self._batch_submitted = 0
        self._batch_done = 0

    def progress(self) -> tuple[int, int]:
        """Returns ``(done, total)`` for the orange progress HUD.

        ``total`` is the number of spans submitted in the current batch
        (decremented on cancellation), ``done`` is the number that
        actually emitted a ``'done'`` message.  Both reset to zero the
        moment no spans are active anymore.
        """
        return self._batch_done, self._batch_submitted

    def maybe_reset_progress(self) -> None:
        """Resets batch counters when no work is outstanding."""
        if not self.active_spans:
            self._batch_submitted = 0
            self._batch_done = 0

    @staticmethod
    def _insert_sample_sorted(state: dict, t: float, point: np.ndarray) -> None:
        """Bisect-insert a (t, point) pair into the per-span sorted state.

        State invariant: ``t_list`` strictly ascending, ``p_list``
        parallel.  Worker phase 1 + phase 2 both call this (they only
        differ in the *t* values they emit; no special-casing needed).
        Duplicates within ``1e-12`` are treated as overwrites — defensive
        against numerical noise on midpoint subdivision.
        """
        t_list = state['t_list']
        p_list = state['p_list']
        pos = bisect.bisect_left(t_list, t)
        if pos < len(t_list) and abs(t_list[pos] - t) < 1e-12:
            p_list[pos] = point
            return
        t_list.insert(pos, t)
        p_list.insert(pos, point)

    def drain_queue(self) -> bool:
        """Polls all active orange pipes.  Returns True if any results.

        Message types
        -------------
        ``('point', span_key, t, point)``
            Cascade sample at parameter *t*.  Inserted into the
            per-span sorted state — phase-1 canonical samples and
            phase-2 densification midpoints share this message type
            because both refine the same t-sorted polyline.
        ``('chord_geo', span_key, polyline)``
            Phase-3 output: a polyline that connects the cascade
            samples by exact mesh geodesics.  Replaces the t-sorted
            chord polyline as the rendering source.
        ``('done', span_key, degraded_any)``
            Worker completed.  ``degraded_any`` toggles the red
            fallback repaint if any solver call hit a straight line.
        ``('error', span_key, repr, traceback)``
            Worker raised — span goes dead, traceback is logged.
        """
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
                        # Worker phase 1 + 2 both emit this — t-keyed
                        # cascade sample, sorted-insert into state.
                        _, _, t_val, point = msg
                        state = self._points.get(span_key)
                        if state is not None:
                            # Phase-3 output (when present) supersedes
                            # the t-sorted polyline; we keep updating
                            # the sample buffers so a hypothetical
                            # later re-render path can fall back to
                            # them, but the visible curve already comes
                            # from ``polyline``.
                            self._insert_sample_sorted(state, float(t_val), point)
                            self.dirty_spans.add(span_key)
                            had_results = True
                    elif kind == 'chord_geo':
                        # Phase-3 polyline override — surface-hugging
                        # geodesic between every consecutive sample.
                        _, _, polyline = msg
                        state = self._points.get(span_key)
                        if state is not None:
                            state['polyline'] = np.asarray(polyline, dtype=float)
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
                        if span_key in self.active_spans:
                            self.active_spans.discard(span_key)
                            self._batch_done += 1
                        had_results = True
                        # Worker exits after done — close reader now
                        # to prevent the normal EOF from being mistaken
                        # for a worker death on the next poll() cycle.
                        try:
                            reader.close()
                        except OSError as exc:
                            log.debug("drain: reader close failed (%s)", exc)
                        self._readers.pop(span_key, None)
                        break
                    elif kind == 'error':
                        # Worker caught an unhandled exception and forwarded
                        # the traceback before exiting.  Surface it loudly
                        # — without this the parent only sees a generic
                        # "pipe broken" warning when drain hits EOF.
                        # Payload: ('error', span_key, repr, traceback_str).
                        repr_exc = msg[2] if len(msg) > 2 else '?'
                        tb_str = msg[3] if len(msg) > 3 else ''
                        log.error(
                            "orange worker on span %s raised %s\n%s",
                            span_key, repr_exc, tb_str)
                        try:
                            reader.close()
                        except OSError:
                            pass
                        self._readers.pop(span_key, None)
                        self.dead_spans.add(span_key)
                        if span_key in self.active_spans:
                            self.active_spans.discard(span_key)
                            if self._batch_submitted > 0:
                                self._batch_submitted -= 1
                        had_results = True
                        break
            except (EOFError, OSError) as exc:
                # Worker died or pipe broken — mark for actor cleanup
                log.warning("orange worker pipe broken on span %s: %s",
                            span_key, exc)
                self._readers.pop(span_key, None)
                self.dead_spans.add(span_key)
                if span_key in self.active_spans:
                    self.active_spans.discard(span_key)
                    if self._batch_submitted > 0:
                        self._batch_submitted -= 1
                had_results = True
        return had_results

    def get_points(self, span_key: SpanKey) -> np.ndarray | None:
        """Returns the renderable polyline for *span_key*, or None.

        Two render sources, picked by precedence:

          1. ``state['polyline']`` — the phase-3 chord-bridged polyline
             (surface-hugging geodesics between every consecutive
             cascade sample).  Set once the worker is fully done.
          2. ``state['p_list']`` — the t-sorted cascade samples
             connected by straight 3-D chords.  Used during phase 1
             and phase 2 (progressive refine) and as the final result
             when ``chord_bridging`` is disabled.

        Since the endpoints (node origins) are seeded by ``submit_span``,
        the result always has at least 2 points from the moment the
        span is submitted — the initial render is a straight chord
        between the node origins, refined as worker results arrive.
        """
        state = self._points.get(span_key)
        if state is None:
            return None
        polyline = state.get('polyline')
        if polyline is not None and len(polyline) >= 2:
            return polyline
        p_list = state.get('p_list')
        if p_list is None or len(p_list) < 2:
            return None
        return np.asarray(p_list, dtype=float)

    def shutdown(self) -> None:
        """Cancels all workers, shuts down the process pool, and releases
        shared memory blocks for V and F.  Safe to call multiple times.

        Runs each phase under its own try / except so a failure in one
        (e.g. ``shm.close`` raising on a half-mapped buffer) cannot stop
        the others (executor shutdown, ``shm.unlink`` of the second
        block, finalizer detach).  Pre-refactor a single ``except`` would
        skip the rest of the cleanup and leak the un-unlinked block on
        POSIX ``/dev/shm``.

        Worker termination is **two-phase**:

          1. ``executor.shutdown(wait=False, cancel_futures=True)`` —
             cancels every *pending* future and signals the executor to
             stop dispatching new work.  Running futures, however, cannot
             be cancelled by this call — and our workers have
             ``SIG_IGN`` on SIGINT (installed in ``_process_initializer``
             to silence MKL's forrtl traceback), so Ctrl+C does NOT kill
             them either.  If we stopped here, ``concurrent.futures``'s
             atexit hook (``_python_exit``) would re-call ``shutdown(wait=True)``
             and block the interpreter until every worker finished its
             current orange-geodesic compute (often 2-5 s).  The user
             sees: "Caught a Ctrl-C within python, exiting program" plus
             a window that stays open until the workers drain.
          2. Force-kill via ``Process.kill()`` (Windows: ``TerminateProcess``,
             POSIX: ``SIGKILL``) on every process in ``executor._processes``.
             This is the only way to break out promptly when SIGINT was
             intercepted at Python level *by design*.  Uses the private
             ``_processes`` attribute because ``concurrent.futures`` does
             not expose a public force-kill API; the attribute has been
             stable since 3.5.  Wrapped in try/except so a worker that
             has already exited cannot break the rest of the loop.
        """
        if getattr(self, '_shutdown_done', False):
            return
        self._shutdown_done = True
        try:
            self.cancel_all()
        except Exception as exc:  # noqa: BLE001 — best-effort
            log.debug("cancel_all during shutdown: %s", exc)
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Python < 3.9: cancel_futures not supported (defensive — pyproject pins >=3.10)
            try:
                self._executor.shutdown(wait=False)
            except Exception as exc:  # noqa: BLE001
                log.debug("executor.shutdown fallback: %s", exc)
        except Exception as exc:  # noqa: BLE001
            log.debug("executor.shutdown: %s", exc)
        # Phase 2: force-kill any worker still alive (see docstring).
        # ``_processes`` is a dict[pid, multiprocessing.Process] on every
        # supported Python version; defensive ``getattr`` for the
        # vanishingly small chance the attribute is renamed upstream.
        for proc in getattr(self._executor, '_processes', {}).values():
            try:
                if proc.is_alive():
                    proc.kill()
            except (AttributeError, OSError, ValueError) as exc:
                # OSError: already-reaped pid; ValueError: closed handle
                log.debug("worker force-kill skipped: %s", exc)
        # Each shm block is cleaned independently: a failure to close()
        # one must not skip unlink() of either.  Both close & unlink are
        # idempotent and safe to call after the executor is gone.
        for shm_block in (self._shm_V, self._shm_F):
            try:
                shm_block.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("shm.close (%s): %s", shm_block.name, exc)
            try:
                shm_block.unlink()
            except FileNotFoundError:
                pass  # already unlinked by another process
            except Exception as exc:  # noqa: BLE001
                log.debug("shm.unlink (%s): %s", shm_block.name, exc)
        # Detach the finalizer so atexit will not retry this work.
        finalizer = getattr(self, '_finalizer', None)
        if finalizer is not None:
            try:
                finalizer.detach()
            except Exception as exc:  # noqa: BLE001
                log.debug("finalizer.detach: %s", exc)

    # Context-manager protocol so callers (tests, scripts) can wrap the
    # manager in ``with _SpanWorkManager(...) as wm:`` and be sure the
    # process pool + shared memory are released on exit, including on
    # KeyboardInterrupt / unhandled exception.
    def __enter__(self) -> _SpanWorkManager:
        return self

    def __exit__(self, exc_type, exc_value, tb) -> None:
        self.shutdown()
