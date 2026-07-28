"""Tests for two load-bearing pieces of ``geo_shoot.py``.

``geo_shoot`` is the interaction layer and was the least-covered module
in the repo.  These two pieces are the ones worth locking first because
everything else depends on them and neither needs a window:

1. **The Master Clock** (``_on_poll_timer``) — the single 50 ms
   heartbeat that fires every debounced task in the editor.  If it stops
   iterating, drag consolidation, the exact stitch line, hover revert and
   the guide fade all silently stop happening.
2. **``_closest_seg_on_polyline_2d``** — the per-segment distance kernel
   behind curve-hover detection, i.e. behind node insertion.  Pure
   numerics, so it can be pinned exactly.
"""
import time
import types

import numpy as np
import pytest

pytest.importorskip("vtk")
pytest.importorskip("pyvista")

from geo_shoot import MidpointShooterApp, _closest_seg_on_polyline_2d  # noqa: E402

# ---------------------------------------------------------------------------
# _closest_seg_on_polyline_2d
# ---------------------------------------------------------------------------

def _poly(*pts):
    a = np.asarray(pts, dtype=float)
    return a, len(a)


def test_closest_point_inside_a_segment():
    pts, n = _poly((0, 0), (10, 0))
    sq, seg, frac = _closest_seg_on_polyline_2d(pts, n, 3.0, 4.0)
    assert seg == 0
    assert frac == pytest.approx(0.3)
    assert sq == pytest.approx(16.0)          # perpendicular distance 4


def test_projection_is_clamped_before_the_first_vertex():
    pts, n = _poly((0, 0), (10, 0))
    sq, seg, frac = _closest_seg_on_polyline_2d(pts, n, -5.0, 0.0)
    assert (seg, frac) == (0, 0.0)
    assert sq == pytest.approx(25.0)


def test_projection_is_clamped_past_the_last_vertex():
    pts, n = _poly((0, 0), (10, 0))
    sq, seg, frac = _closest_seg_on_polyline_2d(pts, n, 15.0, 0.0)
    assert (seg, frac) == (0, 1.0)
    assert sq == pytest.approx(25.0)


def test_picks_the_nearer_of_two_segments():
    """An L: the cursor sits beside the second leg."""
    pts, n = _poly((0, 0), (10, 0), (10, 10))
    sq, seg, frac = _closest_seg_on_polyline_2d(pts, n, 12.0, 7.0)
    assert seg == 1
    assert frac == pytest.approx(0.7)
    assert sq == pytest.approx(4.0)


def test_exactly_on_a_vertex_gives_zero_distance():
    pts, n = _poly((0, 0), (10, 0), (10, 10))
    sq, seg, frac = _closest_seg_on_polyline_2d(pts, n, 10.0, 0.0)
    assert sq == pytest.approx(0.0)
    # Either the end of seg 0 or the start of seg 1 is correct.
    assert (seg, frac) in {(0, 1.0), (1, 0.0)}


def test_zero_length_segment_does_not_divide_by_zero():
    """Consecutive coincident points occur whenever a projected polyline
    collapses in screen space."""
    pts, n = _poly((5, 5), (5, 5), (9, 5))
    sq, seg, frac = _closest_seg_on_polyline_2d(pts, n, 5.0, 8.0)
    assert np.isfinite(sq)
    assert sq == pytest.approx(9.0)
    assert frac == 0.0


def test_single_point_polyline_returns_the_sentinel():
    """No segments to test — the caller's distance threshold must reject
    this rather than the kernel inventing a hit."""
    pts, n = _poly((1, 2))
    sq, seg, frac = _closest_seg_on_polyline_2d(pts, n, 0.0, 0.0)
    assert sq == 1e30
    assert (seg, frac) == (0, 0.0)


def test_frac_reconstructs_the_3d_point_as_documented():
    """The docstring's contract: the 3-D hit is
    ``pts_3d[seg] * (1 - frac) + pts_3d[seg + 1] * frac``."""
    screen, n = _poly((0, 0), (10, 0), (10, 10))
    pts_3d = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
    _, seg, frac = _closest_seg_on_polyline_2d(screen, n, 12.0, 7.0)
    hit = pts_3d[seg] * (1 - frac) + pts_3d[seg + 1] * frac
    np.testing.assert_allclose(hit, [1.0, 0.7, 0.0])


# ---------------------------------------------------------------------------
# Master Clock  (_on_poll_timer)
# ---------------------------------------------------------------------------

class _FakePlotter:
    def __init__(self):
        self.renders = 0

    def render(self):
        self.renders += 1


def _clock_app():
    app = MidpointShooterApp.__new__(MidpointShooterApp)
    app.plotter = _FakePlotter()
    app.state = types.SimpleNamespace(pending_debounces={})
    return app


def _expired(cb):
    return (time.perf_counter() - 1.0, cb)


def _pending(cb):
    return (time.perf_counter() + 3600.0, cb)


def test_empty_registry_does_not_render():
    app = _clock_app()
    app._on_poll_timer(None, None)
    assert app.plotter.renders == 0


def test_expired_task_fires_and_is_removed_exactly_once():
    app = _clock_app()
    calls = []
    app.state.pending_debounces['t'] = _expired(lambda: calls.append(1))

    app._on_poll_timer(None, None)
    app._on_poll_timer(None, None)          # must not fire again

    assert calls == [1]
    assert 't' not in app.state.pending_debounces
    assert app.plotter.renders == 1


def test_unexpired_task_is_left_alone_and_nothing_renders():
    app = _clock_app()
    calls = []
    app.state.pending_debounces['t'] = _pending(lambda: calls.append(1))

    app._on_poll_timer(None, None)

    assert calls == []
    assert 't' in app.state.pending_debounces
    assert app.plotter.renders == 0


def test_a_raising_callback_does_not_starve_the_others():
    """One failing debounce must not skip the remaining expired tasks or
    the batched render — otherwise a single bad task silently freezes
    drag consolidation and the stitch line."""
    app = _clock_app()
    calls = []

    def boom():
        calls.append('boom')
        raise RuntimeError("callback exploded")

    app.state.pending_debounces['a'] = _expired(boom)
    app.state.pending_debounces['b'] = _expired(lambda: calls.append('b'))

    app._on_poll_timer(None, None)          # must not propagate

    assert set(calls) == {'boom', 'b'}
    assert app.state.pending_debounces == {}
    assert app.plotter.renders == 1


def test_a_callback_cancelling_another_pending_task_does_not_raise():
    """Latent ``KeyError``: the ``list()`` snapshot survives a resize but
    not a key vanishing mid-tick.  Cancel-from-callback is one ``pop``
    away, so the loop reads defensively."""
    app = _clock_app()
    calls = []

    def cancels_b():
        calls.append('a')
        app.state.pending_debounces.pop('b', None)

    app.state.pending_debounces['a'] = _expired(cancels_b)
    app.state.pending_debounces['b'] = _expired(lambda: calls.append('b'))

    app._on_poll_timer(None, None)

    assert calls == ['a']                   # 'b' was cancelled, not fired
    assert app.state.pending_debounces == {}
    assert app.plotter.renders == 1


def test_a_callback_may_reschedule_itself_without_refiring_this_tick():
    """The pattern ``_tick_guides_fade`` uses to animate."""
    app = _clock_app()
    calls = []

    def tick():
        calls.append(1)
        if len(calls) < 3:
            app.state.pending_debounces['fade'] = _expired(tick)

    app.state.pending_debounces['fade'] = _expired(tick)

    for _ in range(5):
        app._on_poll_timer(None, None)

    assert calls == [1, 1, 1]
    assert app.state.pending_debounces == {}


def test_overwriting_a_key_slides_the_deadline():
    """How every mouse-move debounce works: re-registering the same key
    replaces the deadline so a moving cursor never lets it fire."""
    app = _clock_app()
    calls = []
    app.state.pending_debounces['drag_exact'] = _expired(lambda: calls.append(1))
    # A new mouse-move arrives before the tick.
    app.state.pending_debounces['drag_exact'] = _pending(lambda: calls.append(2))

    app._on_poll_timer(None, None)

    assert calls == []
    assert 'drag_exact' in app.state.pending_debounces


def test_cancelling_before_the_tick_prevents_the_fire():
    app = _clock_app()
    calls = []
    app.state.pending_debounces['t'] = _expired(lambda: calls.append(1))
    app.state.pending_debounces.pop('t', None)

    app._on_poll_timer(None, None)

    assert calls == []
    assert app.plotter.renders == 0
