"""Regression test for Shift+drag consolidation in the BASE app.

``MidpointShooterApp._on_move`` dispatches ``update_magnitude`` for
Shift+drag of A/B (magnitude-only mode, direction frozen), but the base
``_fire_debounce`` had no Shift branch: when the cursor paused, the
consolidation re-aimed the tangent at the cursor via
``update_from_a/b`` — snapping the direction the user explicitly froze.
The spline subclass carried the correct branch all along
(``GeodesicSplineApp._fire_debounce``); the base must mirror it.
"""
import types

import numpy as np
import pytest

pytest.importorskip("vtk")
pytest.importorskip("pyvista")

from geo_shoot import MidpointShooterApp  # noqa: E402


class _RecordingSeg:
    def __init__(self):
        self.calls = []
        self.is_preview = True

    def update_magnitude(self, q, marker, geo, exact=False):
        self.calls.append(('update_magnitude', marker, exact))

    def update_from_a(self, q, geo, exact=False):
        self.calls.append(('update_from_a', exact))

    def update_from_b(self, q, geo, exact=False):
        self.calls.append(('update_from_b', exact))

    def update_from_p(self, q, cid, geo, exact=False):
        self.calls.append(('update_from_p', exact))

    def update_visuals(self, plotter):
        self.calls.append(('update_visuals',))


def _app(marker, shift):
    app = MidpointShooterApp.__new__(MidpointShooterApp)
    seg = _RecordingSeg()
    app.state = types.SimpleNamespace(
        active_seg=seg,
        drag_marker=marker,
        last_drag_q=np.array([1.0, 2.0, 3.0]),
        last_drag_cid=7,
    )
    app.plotter = types.SimpleNamespace(
        iren=types.SimpleNamespace(
            interactor=types.SimpleNamespace(
                GetShiftKey=lambda: 1 if shift else 0)))
    app.geo = object()
    app._set_hud = types.MethodType(lambda self, *a, **k: None, app)
    return app, seg


@pytest.mark.parametrize("marker", ["a", "b"])
def test_shift_drag_consolidates_magnitude_only(marker):
    app, seg = _app(marker, shift=True)

    app._fire_debounce()

    assert ('update_magnitude', marker, True) in seg.calls
    assert not any(c[0] in ('update_from_a', 'update_from_b')
                   for c in seg.calls)
    assert seg.is_preview is False


@pytest.mark.parametrize("marker,expected", [
    ("a", "update_from_a"), ("b", "update_from_b")])
def test_plain_drag_consolidates_exact_endpoint(marker, expected):
    app, seg = _app(marker, shift=False)

    app._fire_debounce()

    assert (expected, True) in seg.calls
    assert not any(c[0] == 'update_magnitude' for c in seg.calls)


def test_p_drag_unaffected_by_shift():
    app, seg = _app('p', shift=True)

    app._fire_debounce()

    assert ('update_from_p', True) in seg.calls
    assert not any(c[0] == 'update_magnitude' for c in seg.calls)
