"""Regression test for Backspace on a *closed* spline.

Old behaviour popped the last node regardless of closed state, which:

  1. stranded the wrap-around span's actor (only span N-2 was cleaned,
     never span N-1), and
  2. could turn a closed 3-node spline into a closed 2-node one — a
     state ``_validate_session_dict`` rejects, silently dead-ending the
     undo/redo chain when that snapshot is later restored.

Backspace on a closed spline now reopens the loop (undo the close)
instead.  This drives the real ``_on_backspace`` / ``_reopen_spline_loop``
control flow on a lightweight fake app (no window / mesh / solver).
"""
import types

import numpy as np
import pytest

pytest.importorskip("vtk")
pytest.importorskip("pyvista")

import geo_splines  # noqa: E402
from geo_splines import GeodesicSplineApp  # noqa: E402


class _FakeNode:
    def __init__(self):
        self.origin = np.zeros(3)
        self.p_a = np.ones(3)
        self.path_a = np.stack([np.zeros(3), np.ones(3)])
        self.updated = False

    def update_visuals(self, plotter):
        self.updated = True


class _FakePlotter:
    def __init__(self):
        self.removed = []
        self.rendered = 0

    def remove_actor(self, actor, **kw):
        self.removed.append(actor)

    def render(self):
        self.rendered += 1


class _FakeWorkMgr:
    def __init__(self):
        self.cancelled = []

    def cancel_all_for_span(self, key):
        self.cancelled.append(key)


def _make_closed_3node_app():
    app = GeodesicSplineApp.__new__(GeodesicSplineApp)
    nodes = [_FakeNode(), _FakeNode(), _FakeNode()]
    app.splines = [nodes]
    app.splines_closed = [True]
    app.active_spline_idx = 0
    app.plotter = _FakePlotter()
    app._work_mgr = _FakeWorkMgr()
    # Wrap-around span (N-1 == 2) present in both caches — must be cleared.
    wrap_actor = object()
    wrap_actor_g = object()
    app._span_cache = {(0, 0): (None, object()), (0, 2): (None, wrap_actor)}
    app._geo_span_cache = {(0, 2): (None, wrap_actor_g)}
    app._hover_dirty = False
    app._wrap_actors = (wrap_actor, wrap_actor_g)

    # Stub the heavy collaborators the closed-Backspace path calls.
    for name in ("_push_undo", "_recompute_spans", "_submit_geodesic_spans",
                 "_refresh_visuals", "_update_stitch", "_rebuild_node_index"):
        setattr(app, name, types.MethodType(lambda self, *a, **k: None, app))
    app._set_hud = types.MethodType(lambda self, *a, **k: None, app)
    return app


def test_backspace_on_closed_spline_reopens_not_pops():
    app = _make_closed_3node_app()
    wrap_actor, wrap_actor_g = app._wrap_actors

    app._on_backspace()

    # Reopened, not popped: still 3 nodes, now open.
    assert len(app.splines[0]) == 3
    assert app.splines_closed[0] is False
    # First node's closing tangent cleared.
    assert app.splines[0][0].p_a is None
    assert app.splines[0][0].path_a is None
    # Wrap-around span removed from BOTH caches, workers cancelled,
    # actors removed — no stale wrap actor left behind.
    assert (0, 2) not in app._span_cache
    assert (0, 2) not in app._geo_span_cache
    assert (0, 2) in app._work_mgr.cancelled
    assert wrap_actor in app.plotter.removed
    assert wrap_actor_g in app.plotter.removed
    # The non-wrap span is untouched.
    assert (0, 0) in app._span_cache


def test_backspace_open_spline_still_pops():
    """The open-spline path must be unchanged: a node is popped."""
    app = _make_closed_3node_app()
    app.splines_closed = [False]

    # Extra stubs the open pop-path touches.
    popped = app.splines[0][-1]
    app.segments = set()
    app.state = types.SimpleNamespace(
        hover_seg=None, hover_marker=None,
        pending_hover_revert_seg=None, pending_debounces={})
    popped.clear_actors = types.MethodType(lambda self, p: None, popped)

    app._on_backspace()

    assert len(app.splines[0]) == 2      # one node popped
    assert app.splines_closed[0] is False
