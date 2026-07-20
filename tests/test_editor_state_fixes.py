"""Regression tests for editor state-management fixes.

Four independent bugs, all driven on the lightweight fake-app pattern
(no window / mesh / solver — same approach as
``test_backspace_closed_spline.py``):

1. ``_load_from_data`` recomputed/submitted spans for the active spline
   (index 0) only — splines 1..N-1 of a loaded session had nodes but no
   blue/orange curves.
2. ``_finalize_release`` after a click-without-drag consolidated blue
   only; the orange spans that ``_try_hit_marker`` cancelled + cleared
   at press time were never resubmitted.
3. ``_restore_snapshot``'s differential path moved node origins/handles
   without setting ``_hover_dirty``, so marker hover kept operating at
   the pre-undo screen positions.
4. ``_reopen_spline_loop`` / ``_on_backspace`` popped a span's actor
   from the caches but left its ``_span_drag_state`` entry behind, so
   the style gate in ``_set_span`` skipped painting a recreated actor
   (default theme look instead of SPAN_COLOR/width).
"""
import types

import numpy as np
import pytest

pytest.importorskip("vtk")
pytest.importorskip("pyvista")

from geo_splines import GeodesicSplineApp  # noqa: E402


class _FakeNode:
    def __init__(self, origin=(0.0, 0.0, 0.0), p_a=(1.0, 0.0, 0.0),
                 p_b=(-1.0, 0.0, 0.0)):
        self.origin = np.asarray(origin, dtype=float)
        self.p_a = np.asarray(p_a, dtype=float) if p_a is not None else None
        self.p_b = np.asarray(p_b, dtype=float) if p_b is not None else None
        self.path_a = (np.stack([self.origin, self.p_a])
                       if self.p_a is not None else None)
        self.path_b = (np.stack([self.origin, self.p_b])
                       if self.p_b is not None else None)
        self.is_active = False
        self.is_dragging = False
        self.is_preview = False

    def update_visuals(self, plotter):
        pass

    def clear_actors(self, plotter):
        pass


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


def _noop(self, *a, **k):
    return None


def _stub(app, *names):
    for name in names:
        setattr(app, name, types.MethodType(_noop, app))


# ---------------------------------------------------------------------------
# 1. _load_from_data walks every spline
# ---------------------------------------------------------------------------

def test_load_from_data_recomputes_every_spline():
    app = GeodesicSplineApp.__new__(GeodesicSplineApp)
    app.plotter = _FakePlotter()
    app.segments = []
    app.state = types.SimpleNamespace(
        hover_seg=None, hover_marker=None,
        pending_hover_revert_seg=None, pending_debounces={},
        active_seg=None)
    app._hover_dirty = False
    _stub(app, "_clear_all_curve_caches", "_rebuild_node_index",
          "_refresh_visuals", "_update_stitch")
    app._node_from_record = types.MethodType(
        lambda self, nd: _FakeNode(), app)

    recompute_sids, submit_sids = [], []
    app._recompute_spans = types.MethodType(
        lambda self, node=None, sid=None: recompute_sids.append(sid), app)
    app._submit_geodesic_spans = types.MethodType(
        lambda self, node=None, sid=None: submit_sids.append(sid), app)

    data = {'splines': [
        {'nodes': [{}, {}], 'closed': False},
        {'nodes': [{}, {}, {}], 'closed': False},
        {'nodes': [{}, {}], 'closed': False},
    ]}
    n_loaded = app._load_from_data(data)

    assert n_loaded == 7
    # Every spline gets an explicit per-sid recompute + submit — not
    # just the active one (the old sid-less calls meant [None]).
    assert recompute_sids == [0, 1, 2]
    assert submit_sids == [0, 1, 2]


# ---------------------------------------------------------------------------
# 2. click-without-drag resubmits orange
# ---------------------------------------------------------------------------

def _release_app():
    app = GeodesicSplineApp.__new__(GeodesicSplineApp)
    app.plotter = _FakePlotter()
    app.splines = [[]]
    app._pre_drag_spline_idx = None
    app.calls = []
    app._recompute_spans = types.MethodType(
        lambda self, node=None, sid=None:
        self.calls.append(('blue', node)), app)
    app._submit_geodesic_spans = types.MethodType(
        lambda self, node=None, sid=None:
        self.calls.append(('orange', node)), app)
    return app


def test_finalize_release_without_debounce_resubmits_orange():
    app = _release_app()
    seg = _FakeNode()
    app._consolidated_seg = None   # no debounce fired for this drag

    app._finalize_release(seg)

    assert ('blue', seg) in app.calls
    assert ('orange', seg) in app.calls   # the missing resubmit
    assert app._consolidated_seg is None


def test_finalize_release_after_debounce_skips_recompute():
    app = _release_app()
    seg = _FakeNode()
    app._consolidated_seg = seg    # _fire_debounce already consolidated

    app._finalize_release(seg)

    assert app.calls == []         # no redundant second solve
    assert app._consolidated_seg is None


# ---------------------------------------------------------------------------
# 3. differential undo/redo restore marks the marker hover cache dirty
# ---------------------------------------------------------------------------

def test_diff_restore_marks_hover_dirty():
    app = GeodesicSplineApp.__new__(GeodesicSplineApp)
    app.plotter = _FakePlotter()
    node_a = _FakeNode(origin=(0.0, 0.0, 0.0))
    node_b = _FakeNode(origin=(1.0, 0.0, 0.0))
    app.splines = [[node_a, node_b]]
    app.splines_closed = [False]
    app.active_spline_idx = 0
    app._prev_active_spline_idx = 0
    app._didactic_geo_cache = None
    app._hover_dirty = False
    _stub(app, "_refresh_visuals", "_invalidate_stitch_cache",
          "_recompute_spans", "_submit_geodesic_spans")

    rebuilt = []
    app._rebuild_node_inplace = types.MethodType(
        lambda self, seg, nd: rebuilt.append(seg), app)

    # Same structure (diff path), node 0 origin moved.
    snapshot = {
        'version': 2,
        'active_spline_idx': 0,
        'splines': [{
            'closed': False,
            'nodes': [
                {'origin': [0.5, 0.5, 0.0],
                 'p_a': [1.0, 0.0, 0.0], 'p_b': [-1.0, 0.0, 0.0]},
                {'origin': [1.0, 0.0, 0.0],
                 'p_a': [1.0, 0.0, 0.0], 'p_b': [-1.0, 0.0, 0.0]},
            ],
        }],
    }
    app._restore_snapshot(snapshot)

    assert rebuilt == [node_a]        # differential path was taken
    assert app._hover_dirty is True   # markers moved -> cache must rebuild


# ---------------------------------------------------------------------------
# 4. span style state is dropped together with the span actor
# ---------------------------------------------------------------------------

def _style_app(closed):
    app = GeodesicSplineApp.__new__(GeodesicSplineApp)
    nodes = [_FakeNode(), _FakeNode(), _FakeNode()]
    app.splines = [nodes]
    app.splines_closed = [closed]
    app.active_spline_idx = 0
    app.plotter = _FakePlotter()
    app._work_mgr = _FakeWorkMgr()
    app._hover_dirty = False
    _stub(app, "_push_undo", "_recompute_spans", "_submit_geodesic_spans",
          "_refresh_visuals", "_update_stitch", "_rebuild_node_index",
          "_set_hud")
    return app


def test_reopen_loop_drops_span_style_state():
    app = _style_app(closed=True)
    wrap_key = (0, 2)
    app._span_cache = {wrap_key: (None, object())}
    app._geo_span_cache = {wrap_key: (None, object())}
    app._span_drag_state = {wrap_key: (False, False)}
    app._degraded_spans = {wrap_key}

    app._reopen_spline_loop(0)

    assert wrap_key not in app._span_cache
    assert wrap_key not in app._span_drag_state
    assert wrap_key not in app._degraded_spans


def test_backspace_node_pop_drops_span_style_state():
    app = _style_app(closed=False)
    removed_key = (0, 1)   # span adjacent to the popped 3rd node
    app._span_cache = {removed_key: (None, object())}
    app._geo_span_cache = {removed_key: (None, object())}
    app._span_drag_state = {removed_key: (False, False)}
    app._degraded_spans = {removed_key}
    app.segments = []
    app._hover_curve_dirty = False
    app.state = types.SimpleNamespace(
        active_seg=None, drag_marker=None,
        hover_seg=None, hover_marker=None,
        pending_hover_revert_seg=None, pending_debounces={})
    app.stitch_invalidations = []
    app._invalidate_stitch_cache = types.MethodType(
        lambda self: self.stitch_invalidations.append(True), app)

    app._on_backspace()

    assert len(app.splines[0]) == 2
    assert removed_key not in app._span_cache
    assert removed_key not in app._span_drag_state
    assert removed_key not in app._degraded_spans
    # The popped node's curve is gone — hover must rebuild, and the
    # id()-keyed stitch cache must drop (the popped node may be the
    # very node it is keyed to).
    assert app._hover_curve_dirty is True
    assert app.stitch_invalidations == [True]
