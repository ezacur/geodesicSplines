"""Regression tests for the second batch of editor state fixes.

Covers (fake-app pattern, no window / mesh / solver):

- **M1**: popping an empty spline via Backspace re-keys every
  sid-keyed cache so later splines keep their actors (previously the
  old keys became unreachable ghosts and new edits drew duplicates).
- **M2**: Backspace mid-drag aborts the gesture cleanly (no ghost
  gizmo on the popped node, camera unlocked, no stale
  ``_consolidated_seg``).
- **M3**: a session load mid-drag unwinds the drag instead of only
  nulling ``active_seg`` (which left the camera locked forever).
- **B1**: a marker click captures the undo snapshot WITHOUT pushing
  it; the first drag movement commits it; a click that never moves
  discards it (no undo-history flooding, redo stack preserved).
- **B10**: ``allowed_tags`` filters marker candidates BEFORE the
  argmin, so a P marker within tolerance is found even when an A/B
  tip is sub-pixel closer.
"""
import types

import numpy as np
import pytest

pytest.importorskip("vtk")
pytest.importorskip("pyvista")

from geo_shoot import MidpointShooterApp  # noqa: E402
from geo_splines import GeodesicSplineApp  # noqa: E402


class _FakeNode:
    def __init__(self, origin=(0.0, 0.0, 0.0)):
        self.origin = np.asarray(origin, dtype=float)
        self.p_a = np.array([1.0, 0.0, 0.0])
        self.p_b = np.array([-1.0, 0.0, 0.0])
        self.path_a = np.stack([self.origin, self.p_a])
        self.path_b = np.stack([self.origin, self.p_b])
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


def _noop(self, *a, **k):
    return None


def _stub(app, *names):
    for name in names:
        setattr(app, name, types.MethodType(_noop, app))


def _idle_state():
    return types.SimpleNamespace(
        active_seg=None, drag_marker=None,
        hover_seg=None, hover_marker=None,
        pending_hover_revert_seg=None, pending_debounces={},
        last_drag_q=None, last_drag_cid=None)


# ---------------------------------------------------------------------------
# M1 — empty-spline pop re-keys sid-keyed caches
# ---------------------------------------------------------------------------

class _RecordingWorkMgr:
    def __init__(self):
        self.cancelled = []
        self.shifted = []

    def cancel_all_for_span(self, key):
        self.cancelled.append(key)

    def shift_spline_keys(self, removed_sid):
        self.shifted.append(removed_sid)
        return set()


def test_empty_spline_pop_rekeys_caches():
    app = GeodesicSplineApp.__new__(GeodesicSplineApp)
    survivors = [_FakeNode(), _FakeNode(), _FakeNode()]
    app.splines = [[], survivors]          # active spline 0 is empty
    app.splines_closed = [False, False]
    app.active_spline_idx = 0
    app.plotter = _FakePlotter()
    app.state = _idle_state()
    app._work_mgr = _RecordingWorkMgr()
    app._hover_dirty = False
    app._hover_curve_dirty = False
    app._didactic_geo_cache = None
    _stub(app, "_push_undo", "_rebuild_node_index", "_refresh_visuals",
          "_update_stitch", "_set_hud", "_submit_geodesic_spans")

    pd_actor = (object(), object())
    app._span_cache = {(1, 0): pd_actor, (1, 1): pd_actor}
    app._geo_span_cache = {(1, 0): pd_actor}
    app._span_drag_state = {(1, 0): (False, False)}
    app._degraded_spans = {(1, 1)}
    app._interp_cache = {1: pd_actor}
    app._interp_origins_buf = {1: np.zeros((3, 3))}
    app._interp_result_cache = {1: ('fp', None, None, None)}

    app._on_backspace()

    # Spline popped; every cache now keyed by the survivor's NEW sid 0.
    assert len(app.splines) == 1 and app.splines[0] is survivors
    assert set(app._span_cache) == {(0, 0), (0, 1)}
    assert set(app._geo_span_cache) == {(0, 0)}
    assert set(app._span_drag_state) == {(0, 0)}
    assert app._degraded_spans == {(0, 1)}
    assert set(app._interp_cache) == {0}
    assert set(app._interp_origins_buf) == {0}
    assert set(app._interp_result_cache) == {0}
    # The worker bookkeeping was renumbered through the manager.
    assert app._work_mgr.shifted == [0]
    assert app._hover_curve_dirty is True


# ---------------------------------------------------------------------------
# M2 / M3 — structural mutations mid-drag abort the gesture
# ---------------------------------------------------------------------------

def _dragging_app_for_backspace():
    app = GeodesicSplineApp.__new__(GeodesicSplineApp)
    nodes = [_FakeNode(), _FakeNode(), _FakeNode()]
    app.splines = [nodes]
    app.splines_closed = [False]
    app.active_spline_idx = 0
    app.plotter = _FakePlotter()
    app._work_mgr = _RecordingWorkMgr()
    app._hover_dirty = False
    app._hover_curve_dirty = False
    app.segments = []
    app._span_cache = {}
    app._geo_span_cache = {}
    app._span_drag_state = {}
    app._degraded_spans = set()
    _stub(app, "_push_undo", "_rebuild_node_index", "_refresh_visuals",
          "_update_stitch", "_set_hud", "_recompute_spans",
          "_submit_geodesic_spans", "_invalidate_stitch_cache")

    dragged = nodes[-1]
    dragged.is_dragging = True
    dragged.is_preview = True
    app.state = _idle_state()
    app.state.active_seg = dragged
    app.state.drag_marker = 'p'
    app.state.pending_debounces['drag_exact'] = (0.0, lambda: None)
    app._consolidated_seg = dragged
    app._pending_drag_snapshot = {'version': 2}

    app.unlocks = []
    app._unlock_camera = types.MethodType(
        lambda self: self.unlocks.append(True), app)
    return app, dragged


def test_backspace_mid_drag_aborts_gesture():
    app, dragged = _dragging_app_for_backspace()

    app._on_backspace()

    # Node popped AND the gesture fully unwound.
    assert dragged not in app.splines[0]
    assert app.state.active_seg is None
    assert app.state.drag_marker is None
    assert 'drag_exact' not in app.state.pending_debounces
    assert app.unlocks == [True]
    assert dragged.is_dragging is False and dragged.is_preview is False
    assert app._consolidated_seg is None
    assert app._pending_drag_snapshot is None


def test_load_mid_drag_unlocks_camera():
    app = GeodesicSplineApp.__new__(GeodesicSplineApp)
    app.plotter = _FakePlotter()
    dragged = _FakeNode()
    dragged.is_dragging = True
    app.segments = [dragged]
    app.state = _idle_state()
    app.state.active_seg = dragged
    app.state.drag_marker = 'a'
    app._consolidated_seg = dragged
    app._pending_drag_snapshot = {'version': 2}
    app._hover_dirty = False
    _stub(app, "_clear_all_curve_caches", "_rebuild_node_index",
          "_refresh_visuals", "_update_stitch",
          "_recompute_spans", "_submit_geodesic_spans")
    app._node_from_record = types.MethodType(
        lambda self, nd: _FakeNode(), app)
    app.unlocks = []
    app._unlock_camera = types.MethodType(
        lambda self: self.unlocks.append(True), app)

    app._load_from_data({'splines': [{'nodes': [{}], 'closed': False}]})

    assert app.state.active_seg is None
    assert app.state.drag_marker is None
    assert app.unlocks == [True]
    assert app._consolidated_seg is None


# ---------------------------------------------------------------------------
# B1 — deferred drag undo
# ---------------------------------------------------------------------------

def _click_app():
    app = GeodesicSplineApp.__new__(GeodesicSplineApp)
    app.plotter = _FakePlotter()
    app.splines = [[]]
    app.active_spline_idx = 0
    app.state = _idle_state()
    app._undo_stack = []
    app._redo_stack = ['old-redo-entry']
    app._pending_drag_snapshot = None
    seg = _FakeNode()
    app._closest_marker_under_cursor = types.MethodType(
        lambda self, x, y, allowed_tags=None: (seg, 'p'), app)
    app._spline_for_node = types.MethodType(lambda self, s: 0, app)
    app._snapshot = types.MethodType(
        lambda self: {'sentinel': True}, app)
    _stub(app, "_lock_camera", "_set_hud", "_cancel_geodesic_spans",
          "_refresh_visuals", "_recompute_spans", "_submit_geodesic_spans")
    return app, seg


def test_marker_click_defers_undo_push():
    app, seg = _click_app()

    assert app._try_hit_marker(5, 5) is True

    # Snapshot captured but NOT pushed; redo stack untouched.
    assert app._pending_drag_snapshot == {'sentinel': True}
    assert app._undo_stack == []
    assert app._redo_stack == ['old-redo-entry']

    # First drag movement commits it (and clears redo, standard
    # semantics for a real mutation).
    app._commit_pending_drag_undo()
    assert app._undo_stack == [{'sentinel': True}]
    assert app._redo_stack == []
    assert app._pending_drag_snapshot is None

    # Idempotent: a second commit with nothing pending is a no-op.
    app._commit_pending_drag_undo()
    assert app._undo_stack == [{'sentinel': True}]


def test_release_without_movement_discards_snapshot():
    app, seg = _click_app()
    app._try_hit_marker(5, 5)
    app._consolidated_seg = None
    app._pre_drag_spline_idx = None

    app._finalize_release(seg)

    assert app._pending_drag_snapshot is None
    assert app._undo_stack == []             # nothing mutated → no entry
    assert app._redo_stack == ['old-redo-entry']   # redo preserved


# ---------------------------------------------------------------------------
# B10 — allowed_tags filters before the argmin
# ---------------------------------------------------------------------------

def _hover_app():
    app = MidpointShooterApp.__new__(MidpointShooterApp)
    seg_a, seg_p = _FakeNode(), _FakeNode()
    app._hover_dirty = False
    app._hover_n = 2
    app._hover_pts_3d = np.zeros((2, 3))
    app._hover_tags = [(seg_a, 'a'), (seg_p, 'p')]
    # Screen positions: the 'a' marker is CLOSER to the cursor (10, 10).
    app._to_screen_batch = types.MethodType(
        lambda self, pts: np.array([[10.0, 10.0], [13.0, 10.0]]), app)
    app._is_marker_occluded = types.MethodType(
        lambda self, p: False, app)
    app.cfg = types.SimpleNamespace(PICK_TOLERANCE_SQ=100.0)
    return app, seg_a, seg_p


def test_allowed_tags_filter_applies_before_argmin():
    app, seg_a, seg_p = _hover_app()

    # Unfiltered: the nearer 'a' marker wins.
    assert app._closest_marker_under_cursor(10, 10) == (seg_a, 'a')
    # Filtered to 'p': the P marker must be found even though 'a' is
    # nearer (the old code argmin'd first and returned None).
    assert app._closest_marker_under_cursor(
        10, 10, allowed_tags=('p',)) == (seg_p, 'p')
    # Nothing allowed → clean None.
    assert app._closest_marker_under_cursor(
        10, 10, allowed_tags=('x',)) is None
