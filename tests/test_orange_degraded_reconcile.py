"""Regression test for ``_apply_worker_fallbacks`` degraded reconciliation.

The old code merged only ``True`` flags from the worker manager, so a
span painted red once stayed red forever: rebuilding it (``R``) with a
clean geodesic never cleared the app-level ``_degraded_spans`` entry,
because the manager's clean-``done`` discard only emptied its own set —
which the merge then skipped via an early return.

``_apply_worker_fallbacks`` now reconciles every span that emitted a
``'done'`` this tick against the manager's verdict: mark on fallback,
clear on a clean finish.
"""
import types

import pytest

pytest.importorskip("vtk")
pytest.importorskip("pyvista")

from geo_splines import GeodesicSplineApp  # noqa: E402


def _fake_app(done, degraded, already_red):
    app = GeodesicSplineApp.__new__(GeodesicSplineApp)
    app._degraded_spans = set(already_red)
    app._span_drag_state = {}
    app._work_mgr = types.SimpleNamespace(
        done_spans=set(done), degraded_spans=set(degraded))
    app._set_hud = types.MethodType(lambda self, *a, **k: None, app)
    return app


def test_clean_done_clears_sticky_red():
    key = (0, 0)
    # Span was painted red earlier; it just finished a clean rebuild
    # (in done_spans, NOT in degraded_spans).
    app = _fake_app(done={key}, degraded=set(), already_red={key})

    app._apply_worker_fallbacks()

    assert key not in app._degraded_spans          # red cleared
    assert not app._work_mgr.degraded_spans         # manager set drained


def test_degraded_done_marks_red():
    key = (1, 2)
    app = _fake_app(done={key}, degraded={key}, already_red=set())

    app._apply_worker_fallbacks()

    assert key in app._degraded_spans               # newly red


def test_untouched_span_not_reconciled():
    """A span that did NOT finish this tick keeps its current flag."""
    finished, other = (0, 0), (0, 1)
    app = _fake_app(done={finished}, degraded=set(), already_red={other})

    app._apply_worker_fallbacks()

    # 'other' didn't emit 'done' this tick, so its red is left alone.
    assert other in app._degraded_spans
    # 'finished' was clean → not red.
    assert finished not in app._degraded_spans


def test_no_done_is_noop():
    key = (0, 0)
    app = _fake_app(done=set(), degraded=set(), already_red={key})
    app._apply_worker_fallbacks()
    assert key in app._degraded_spans               # untouched
