"""Regression test for ``_SpanWorkManager.submit_span`` pool recovery.

When ``executor.submit`` raises ``BrokenProcessPool`` (a worker died
abnormally), ``submit_span`` calls ``_rebuild_executor`` — which closes
*every* reader, including the one just registered for this span — and
then retries.  The retry must mint a **fresh** pipe: re-registering the
original (now-closed) reader and re-shipping its writer means the
recovered worker writes into a closed pipe, so the next ``drain_queue``
poll raises ``OSError`` and the span is marked dead.  I.e. the recovery
path would reliably kill the one span that triggered it.

This exercises ``submit_span`` in isolation (no real process pool /
shared memory) via a flaky fake executor.
"""
import multiprocessing as mp
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pytest

pytest.importorskip("vtk")  # geo_splines imports vtk transitively

from geo_splines import _SpanWorkManager  # noqa: E402


class _DummyFuture:
    def add_done_callback(self, fn):
        pass

    def exception(self):
        return None


class _FlakyExecutor:
    """``submit`` raises BrokenProcessPool once, then captures the writer
    (7th positional arg) and returns a dummy future."""

    def __init__(self):
        self.calls = 0
        self.captured_writer = None

    def submit(self, fn, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise BrokenProcessPool("simulated broken pool")
        self.captured_writer = args[6]
        return _DummyFuture()


def _bare_manager(executor):
    """A ``_SpanWorkManager`` with only the state ``submit_span`` touches
    — bypasses ``__init__`` (which spawns a real pool + shared memory)."""
    m = _SpanWorkManager.__new__(_SpanWorkManager)
    m._readers = {}
    m._futures = {}
    m._points = {}
    m.active_spans = set()
    m.dirty_spans = set()
    m.done_spans = set()
    m.dead_spans = set()
    m.degraded_spans = set()
    m._batch_submitted = 0
    m._batch_done = 0
    m._executor = executor
    return m


def test_broken_pool_retry_uses_fresh_open_pipe():
    ex = _FlakyExecutor()
    m = _bare_manager(ex)

    # Replicate _rebuild_executor's contract (close every reader, clear
    # state) without spawning a real pool; keep the recovered executor.
    def _fake_rebuild():
        for r in m._readers.values():
            try:
                r.close()
            except OSError:
                pass
        m._readers.clear()
        m._futures.clear()
        m._points.clear()
        m.active_spans.clear()
        m.dirty_spans.clear()
        m.done_spans.clear()
        m.dead_spans.clear()
        m.degraded_spans.clear()
        m._batch_submitted = 0
        m._batch_done = 0

    m._rebuild_executor = _fake_rebuild

    span_key = (0, 0)
    ctrl = [np.zeros(3), np.ones(3), np.ones(3) * 2.0, np.ones(3) * 3.0]
    path_b = np.stack([ctrl[0], ctrl[1]])
    path_a_rev = np.stack([ctrl[2], ctrl[3]])

    m.submit_span(span_key, ctrl, path_b, path_a_rev, n_samples=5)

    assert ex.calls == 2                 # failed once, retried once
    assert span_key not in m.dead_spans  # span survived the recovery
    assert span_key in m._readers

    reader = m._readers[span_key]
    assert not reader.closed             # a fresh, open read-end

    # End-to-end: the retry's writer must be paired with the registered
    # reader (proving a brand-new pipe, not the closed original).
    ex.captured_writer.send(('done', span_key, False))
    assert reader.poll(1.0)
    assert reader.recv() == ('done', span_key, False)


def test_cancel_all_clears_result_flags():
    """cancel_all must clear the per-span result flags too, or a stale
    generation bleeds into the next batch: a resubmitted span whose key
    sits in done_spans renders 'final' on its first partial polyline, and
    a leftover degraded_spans key repaints a healthy new span red."""
    m = _bare_manager(executor=None)
    key = (0, 0)
    m.dirty_spans.add(key)
    m.done_spans.add(key)
    m.dead_spans.add(key)
    m.degraded_spans.add(key)
    m.active_spans.add(key)
    m._batch_submitted = 3
    m._batch_done = 1

    m.cancel_all()

    assert not m.dirty_spans
    assert not m.done_spans
    assert not m.dead_spans
    assert not m.degraded_spans
    assert not m.active_spans
    assert m._batch_submitted == 0
    assert m._batch_done == 0


def test_cancel_span_discards_dead_flag():
    """A cancelled span's 'worker died — clear the actor' signal is
    obsolete; leaving it hid the actor of whatever replaced the span."""
    m = _bare_manager(executor=None)
    key = (0, 0)
    reader, writer = mp.Pipe(duplex=False)
    m._readers[key] = reader
    m.dead_spans.add(key)

    m.cancel_span(key)

    assert key not in m.dead_spans
    writer.close()


def test_shift_spline_keys_renumbers_and_cancels():
    """After the app pops spline 1, the manager must cancel in-flight
    spans of splines >= 1 (their pipe messages embed the OLD key) and
    renumber the parent-side flag sets."""
    m = _bare_manager(executor=None)
    live_reader, live_writer = mp.Pipe(duplex=False)
    m._readers[(2, 0)] = live_reader          # in-flight span of spline 2
    m.active_spans.add((2, 0))
    m._batch_submitted = 1
    # Flags across all three splines.
    m.done_spans.update({(0, 1), (2, 1)})
    m.degraded_spans.update({(1, 0), (2, 1)})
    m.dirty_spans.add((0, 0))
    m.dead_spans.add((2, 2))

    affected = m.shift_spline_keys(1)

    assert affected == {2}                     # caller resubmits new sid 1
    assert (2, 0) not in m._readers and live_reader.closed
    assert not m.active_spans
    # Spline 0 untouched; spline 1's flags dropped; spline 2 shifted → 1.
    assert m.done_spans == {(0, 1), (1, 1)}
    assert m.degraded_spans == {(1, 1)}
    assert m.dirty_spans == {(0, 0)}
    assert m.dead_spans == {(1, 2)}
    live_writer.close()


def test_release_resources_shuts_down_executor_and_shm():
    """The GC-time finalizer receives the resources directly — a
    weakref-to-self callback always resolved to None (finalize fires
    after the referent dies) and cleaned nothing."""
    class _Exec:
        def __init__(self):
            self.shutdowns = []
            self._processes = {}

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdowns.append((wait, cancel_futures))

    class _Shm:
        def __init__(self):
            self.closed = False
            self.unlinked = False
            self.name = "test"

        def close(self):
            self.closed = True

        def unlink(self):
            self.unlinked = True

    ex, shm_v, shm_f = _Exec(), _Shm(), _Shm()
    _SpanWorkManager._release_resources(ex, shm_v, shm_f)

    assert ex.shutdowns == [(False, True)]
    assert shm_v.closed and shm_v.unlinked
    assert shm_f.closed and shm_f.unlinked


def test_pipe_pairing_helper_sanity():
    """Guards the assumption the test above relies on: mp.Pipe(duplex=
    False) returns (reader, writer) and a closed reader reports closed."""
    reader, writer = mp.Pipe(duplex=False)
    writer.send(42)
    assert reader.recv() == 42
    reader.close()
    assert reader.closed
