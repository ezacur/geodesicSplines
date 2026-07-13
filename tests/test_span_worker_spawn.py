"""End-to-end spawn test for the background worker pool.

Unlike ``test_span_work_manager`` (which drives ``submit_span`` against a
fake executor) and ``test_chord_degraded_flag`` (which runs the worker
in-process), this exercises the REAL path: a genuine
``ProcessPoolExecutor`` spawns a child, the child imports ``span_workers``
and runs ``_process_initializer`` to rebuild the mesh from shared memory,
and ``_geodesic_decasteljau_worker`` — pickled by qualified name — streams
results back over the pipe.  This is the machinery the ``span_workers``
extraction is meant to keep working (and keep light: the child imports
neither pyvista nor the GUI module).

Needs the full runtime (vtk / pyvista / potpourri3d) and is slow (real
process spawn + Numba warm-up), so it is skipped in the slim CI matrix
and runs only in the full-dependency job.
"""
import time

import numpy as np
import pytest

pytest.importorskip("vtk")
pytest.importorskip("pyvista")
pytest.importorskip("potpourri3d")

from span_workers import _SpanWorkManager  # noqa: E402


def _flat_mesh():
    V = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.5, -1.0, 0.0],
    ], dtype=float)
    F = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int64)
    return V, F


def _flat_span():
    P0 = np.array([0.20, 0.20, 0.0])
    H_out = np.array([0.35, 0.25, 0.0])
    H_in = np.array([0.50, 0.30, 0.0])
    P1 = np.array([0.65, 0.20, 0.0])
    return [P0, H_out, H_in, P1], np.stack([P0, H_out]), np.stack([H_in, P1])


def test_real_spawn_span_completes_and_shuts_down_clean():
    V, F = _flat_mesh()
    mgr = _SpanWorkManager(V, F, 2)
    try:
        ctrl, path_b, path_a_rev = _flat_span()
        span_key = (0, 0)
        mgr.submit_span(span_key, ctrl, path_b, path_a_rev, 5)

        deadline = time.perf_counter() + 90.0
        while time.perf_counter() < deadline:
            mgr.drain_queue()
            if span_key in mgr.done_spans:
                break
            time.sleep(0.05)

        assert span_key in mgr.done_spans, "worker never sent 'done'"
        assert span_key not in mgr.dead_spans, "span died"
        pts = mgr.get_points(span_key)
        assert pts is not None and len(pts) >= 2
        assert np.isfinite(pts).all()

        # Deterministically exercise the shutdown guard: after
        # executor.shutdown(wait=False) the pool's management thread can
        # null ``_processes`` asynchronously; the force-kill loop must not
        # crash on ``None.values()`` before the shared-memory cleanup runs.
        mgr._executor._processes = None
        mgr.shutdown()   # must not raise
        assert mgr._shutdown_done is True
    finally:
        mgr.shutdown()   # idempotent second call (no-op)
