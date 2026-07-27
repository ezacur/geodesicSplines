"""Regression tests for three editor state bugs.

Same lightweight fake-app pattern as ``test_editor_state_fixes.py``
(no window / mesh / solver).

1. ``curve_hover_info`` survived every structural mutation.  The payload
   is captured on a mouse-move but consumed by a later double-click, and
   every mutation that can happen in between (Backspace, ``C``, ``l``,
   Ctrl+Z) is keyboard-driven — so the cursor never moves and the
   payload is never refreshed.  Acting on it raised ``IndexError`` deep
   in ``_insert_node_at_curve`` *after* ``_push_undo`` had already spent
   an undo slot and wiped the redo stack; when the stale index happened
   to stay in range it silently inserted into a different span.

2. ``_set_span(sid, i, None)`` hid the actor but left ``pd.points``
   populated, so any blanket re-show (``_refresh_visuals``,
   ``_toggle_layer`` — i.e. pressing ``b`` twice) resurrected the stale
   polyline.  Its two sibling methods clear the geometry and say why.

3. ``_on_close_spline`` pushed an undo snapshot *before* a close that
   can bail out (coincident first/last node, or a failed closing
   shoot), spending an undo slot and clearing the redo stack for a
   gesture that mutated nothing.
"""
import types

import numpy as np
import pytest

pytest.importorskip("vtk")
pytest.importorskip("pyvista")

import pyvista as pv  # noqa: E402

from geo_splines import GeodesicSplineApp  # noqa: E402


class _FakeNode:
    def __init__(self, origin=(0.0, 0.0, 0.0)):
        self.origin = np.asarray(origin, dtype=float)
        self.p_a = np.array([1.0, 0.0, 0.0])
        self.p_b = np.array([-1.0, 0.0, 0.0])
        self.normal = np.array([0.0, 0.0, 1.0])
        self.face_idx = 0

    def update_visuals(self, plotter):
        pass


class _FakePlotter:
    def __init__(self):
        self.rendered = 0

    def render(self):
        self.rendered += 1


# ---------------------------------------------------------------------------
# 1. stale curve_hover_info
# ---------------------------------------------------------------------------

def _hover_app(n_nodes=5, closed=False):
    app = GeodesicSplineApp.__new__(GeodesicSplineApp)
    app.splines = [[_FakeNode((i, 0, 0)) for i in range(n_nodes)]]
    app.splines_closed = [closed]
    return app


def _payload(app, sid=0, span_idx=3):
    """The stamp ``_update_hover_marker`` attaches to a live payload."""
    return {
        'spline_idx': sid,
        'span_idx': span_idx,
        'layer': 'blue',
        'point': np.zeros(3),
        'nodes_snapshot': tuple(app.splines[sid]),
        'closed_snapshot': bool(app.splines_closed[sid]),
    }


def test_fresh_payload_is_live():
    app = _hover_app()
    assert app._hover_info_live(_payload(app)) is True


def test_payload_dies_when_a_node_is_removed():
    """Backspace twice with the sight parked on span 3 of a 5-node
    spline: span 3 no longer exists."""
    app = _hover_app(n_nodes=5)
    info = _payload(app, span_idx=3)
    app.splines[0].pop()
    app.splines[0].pop()
    assert app._hover_info_live(info) is False


def test_payload_dies_when_a_node_object_is_replaced():
    """Undo rebuilds nodes in place — same count, different objects.
    Identity, not length, is the contract."""
    app = _hover_app(n_nodes=5)
    info = _payload(app, span_idx=2)
    app.splines[0][2] = _FakeNode((99, 0, 0))
    assert app._hover_info_live(info) is False


def test_payload_dies_when_the_loop_closes():
    """Closing adds the wrap-around span, so span indices change
    meaning even though the node list is untouched."""
    app = _hover_app(n_nodes=5)
    info = _payload(app, span_idx=3)
    app.splines_closed[0] = True
    assert app._hover_info_live(info) is False


def test_payload_dies_when_its_spline_is_gone():
    app = _hover_app(n_nodes=5)
    info = _payload(app, span_idx=1)
    app.splines.clear()
    app.splines_closed.clear()
    assert app._hover_info_live(info) is False


def test_in_range_but_stale_span_is_still_rejected():
    """The silent-corruption case: after removing one node span 2 is
    still a valid index, but it addresses different geometry."""
    app = _hover_app(n_nodes=6)
    info = _payload(app, span_idx=2)
    app.splines[0].pop(0)          # every span shifts down one
    assert app._hover_info_live(info) is False


def test_interp_payload_uses_the_node_contract_only():
    """``span_idx == -1`` is the interp sentinel — it addresses the
    whole spline, so the span-range check must not reject it."""
    app = _hover_app(n_nodes=4)
    info = _payload(app, span_idx=-1)
    assert app._hover_info_live(info) is True
    app.splines[0].pop()
    assert app._hover_info_live(info) is False


def test_stale_double_click_does_not_spend_undo():
    """A dropped gesture must not clear the redo stack."""
    app = _hover_app(n_nodes=5)
    app.plotter = _FakePlotter()
    info = _payload(app, span_idx=3)
    app.splines[0].pop()
    app.splines[0].pop()
    app.curve_hover_info = info

    calls = []
    app._push_undo = types.MethodType(lambda self: calls.append('undo'), app)
    app._insert_node_at_curve = types.MethodType(
        lambda self, i: calls.append('insert'), app)
    app._hide_curve_hover_marker = types.MethodType(lambda self: None, app)
    app._set_hud = types.MethodType(lambda self, *a, **k: None, app)
    app._try_hit_marker = types.MethodType(lambda self, x, y: None, app)
    app.plotter.iren = types.SimpleNamespace(
        get_event_position=lambda: (0, 0),
        interactor=types.SimpleNamespace(GetRepeatCount=lambda: 1))

    app._on_press_impl(None, None)

    assert calls == []                      # neither undo nor insert
    assert app.curve_hover_info is None     # payload dropped


# ---------------------------------------------------------------------------
# 2. _set_span blanking clears geometry
# ---------------------------------------------------------------------------

def test_blanking_a_span_clears_its_geometry():
    """Hiding is not enough — ``_refresh_visuals`` / ``_toggle_layer``
    re-show every span actor unconditionally."""
    app = GeodesicSplineApp.__new__(GeodesicSplineApp)
    pd = pv.PolyData()
    actor = types.SimpleNamespace(
        _vis=True,
        GetVisibility=lambda s=None: True,
        SetVisibility=lambda v: None,
    )
    pd.points = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    app._span_cache = {(0, 0): (pd, actor)}
    app._hover_curve_dirty = False

    app._set_span(0, 0, None)

    assert pd.n_points == 0, "stale polyline can reappear on a blanket re-show"
    assert app._hover_curve_dirty is True


# ---------------------------------------------------------------------------
# 3. _on_close_spline does not spend undo on a no-op
# ---------------------------------------------------------------------------

def _close_app(vec_degenerate):
    app = GeodesicSplineApp.__new__(GeodesicSplineApp)
    app.plotter = _FakePlotter()
    first = _FakeNode((0.0, 0.0, 0.0))
    first.p_a = None                       # forces the closing shoot
    # Coincident first/last -> the ``vn > 1e-9`` guard bails.
    last_pos = (0.0, 0.0, 0.0) if vec_degenerate else (5.0, 0.0, 0.0)
    app.splines = [[first, _FakeNode((2.0, 0, 0)), _FakeNode(last_pos)]]
    app.splines_closed = [False]
    app.active_spline_idx = 0
    app.undo_calls = []
    app._push_undo = types.MethodType(
        lambda self: self.undo_calls.append(1), app)
    for name in ("_recompute_spans", "_submit_geodesic_spans",
                 "_refresh_visuals", "_update_stitch"):
        setattr(app, name, types.MethodType(lambda self, *a, **k: None, app))
    app._set_hud = types.MethodType(lambda self, *a, **k: None, app)
    app._stitch_actor = types.SimpleNamespace(SetVisibility=lambda v: None)
    app.scfg = types.SimpleNamespace(HANDLE_FRACTION=0.3)
    return app


def test_degenerate_close_does_not_push_undo():
    """First and last node coincident: nothing is mutated, so no undo
    slot may be spent and the redo stack must survive."""
    app = _close_app(vec_degenerate=True)
    app.geo = types.SimpleNamespace(compute_shoot=lambda *a: None)

    app._on_close_spline()

    assert app.undo_calls == []
    assert app.splines_closed[0] is False


def test_failed_closing_shoot_does_not_push_undo():
    app = _close_app(vec_degenerate=False)
    app.geo = types.SimpleNamespace(compute_shoot=lambda *a: None)

    app._on_close_spline()

    assert app.undo_calls == []
    assert app.splines_closed[0] is False


def test_successful_close_does_push_undo():
    app = _close_app(vec_degenerate=False)
    app.geo = types.SimpleNamespace(
        compute_shoot=lambda *a: np.array([[0.0, 0, 0], [0.5, 0, 0]]))

    app._on_close_spline()

    assert app.undo_calls == [1]
    assert app.splines_closed[0] is True
