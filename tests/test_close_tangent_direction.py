"""Regression test for the closing-tangent direction in ``_on_close_spline``.

When the first node has no ``p_a`` (a reopened loop cleared it), closing
computes a fresh closing tangent.  ``p_a`` must point BACKWARD — toward
the last node — so the wrap-around span ``[last, last.p_b, first.p_a,
first]`` arrives at the first node moving forward (G1 with span 0).
The old code shot ``+v_dir`` (away from the last node), hooking the
closing span around node 0.

Convention cross-checks: ``_init_tangents`` builds ``p_a`` with the
``sign=-1`` ray, and ARCHITECTURE.md states "closing tangent on the
first node (``p_a`` toward the last node)".
"""
import types

import numpy as np
import pytest

pytest.importorskip("vtk")
pytest.importorskip("pyvista")

from geo_splines import GeodesicSplineApp, SplineConfig  # noqa: E402


class _FakeNode:
    def __init__(self, origin):
        self.origin = np.asarray(origin, dtype=float)
        self.normal = np.array([0.0, 0.0, 1.0])
        self.face_idx = 0
        self.p_a = None
        self.path_a = None
        self.p_b = np.asarray(origin, dtype=float) + [0.1, 0.0, 0.0]

    def update_visuals(self, plotter):
        pass


def _noop(self, *a, **k):
    return None


def test_reclose_shoots_closing_tangent_toward_last_node():
    app = GeodesicSplineApp.__new__(GeodesicSplineApp)
    first = _FakeNode([0.0, 0.0, 0.0])
    last = _FakeNode([1.0, 1.0, 0.0])
    app.splines = [[first, _FakeNode([2.0, 0.0, 0.0]), last]]
    app.splines_closed = [False]
    app.active_spline_idx = 0
    app.scfg = SplineConfig()
    app.plotter = types.SimpleNamespace(render=lambda: None)
    app._stitch_actor = types.SimpleNamespace(SetVisibility=lambda v: None)
    for name in ("_push_undo", "_recompute_spans", "_submit_geodesic_spans",
                 "_refresh_visuals", "_set_hud"):
        setattr(app, name, types.MethodType(_noop, app))

    shots = []

    def _compute_shoot(origin, direction, h_len, face_idx):
        shots.append((np.asarray(origin, float),
                      np.asarray(direction, float), h_len))
        return np.stack([origin, origin + direction * h_len])

    app.geo = types.SimpleNamespace(compute_shoot=_compute_shoot)

    app._on_close_spline()

    assert app.splines_closed[0] is True
    assert len(shots) == 1
    _origin, direction, _h = shots[0]
    # The tangent-plane direction from last toward first is
    # normalize(first - last) = (-0.7071, -0.7071, 0); the closing
    # tangent must be its NEGATION (toward the last node).
    expected = (last.origin - first.origin)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(direction, expected, atol=1e-12)
    # And the resulting handle sits between first and last, not on the
    # far side of the first node.
    assert np.dot(first.p_a - first.origin,
                  last.origin - first.origin) > 0
