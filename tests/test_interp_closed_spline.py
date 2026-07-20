"""Regression tests for the interpolation (black) curve layer.

1. Closed splines: scipy's ``splprep(per=True)`` requires the last
   input point to duplicate the first — otherwise it silently
   overwrites the last point with the first, so the fitted curve
   missed the true last node entirely (verified: ~0.4 units off on a
   unit square).  ``_recompute_interp_curve`` now fits on an
   explicitly wrapped copy and drops the duplicate's parameter from
   ``u_at_nodes``.

2. ``_set_interp_curve(sid, None)`` must clear the PolyData geometry
   (not just hide the actor): ``_toggle_layer`` / ``_refresh_visuals``
   blanket re-show interp actors, which used to resurrect stale
   geometry cached while the layer was hidden.
"""
import types

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("vtk")
pytest.importorskip("pyvista")

from geo_splines import GeodesicSplineApp, LayerKind, SplineConfig  # noqa: E402


class _FakeNode:
    def __init__(self, origin):
        self.origin = np.asarray(origin, dtype=float)


def _interp_app(origins, closed):
    """Fake app exposing the real ``_recompute_interp_curve`` code path
    over an identity-projection fake mesh."""
    app = GeodesicSplineApp.__new__(GeodesicSplineApp)
    app.splines = [[_FakeNode(o) for o in origins]]
    app.splines_closed = [closed]
    app.scfg = SplineConfig()
    app._layer_visible = {LayerKind.BLUE: True, LayerKind.ORANGE: False,
                          LayerKind.INTERP: True}
    app._interp_origins_buf = {}
    app._interp_result_cache = {}

    app.captured = {}
    app._set_interp_curve = types.MethodType(
        lambda self, sid, pts: self.captured.__setitem__(sid, pts), app)

    app.geo = types.SimpleNamespace(
        adaptive_samples=lambda pts, res, mn, mx: mn,
        project_smooth_batch=lambda pts: np.asarray(pts, dtype=float),
        subdivide_secant_chords=lambda pts, tol, max_depth, labels:
            (pts, labels),
        _face_edge_len2=np.array([0.01]),
    )
    return app


def _min_dist_to_polyline(polyline, point):
    return float(np.min(np.linalg.norm(polyline - point, axis=1)))


def test_closed_interp_curve_passes_through_last_node():
    square = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
              (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    app = _interp_app(square, closed=True)

    app._recompute_interp_curve(0)

    curve = app.captured[0]
    assert curve is not None
    # The periodic fit must interpolate EVERY node — with the silent
    # last-point overwrite, the curve missed (0, 1, 0) by ~0.4.
    for origin in square:
        assert _min_dist_to_polyline(curve, np.asarray(origin)) < 0.05

    _fp, _projected, u_at_nodes, _u_per_pt = app._interp_result_cache[0]
    # One parameter per node, and the last node keeps its own distinct
    # parameter strictly inside (0, 1) — not the wrap duplicate's 1.0.
    assert len(u_at_nodes) == 4
    assert u_at_nodes[0] == 0.0
    assert np.all(np.diff(u_at_nodes) > 0)
    assert u_at_nodes[-1] < 1.0


def test_open_interp_curve_unchanged():
    line = [(0.0, 0.0, 0.0), (1.0, 0.2, 0.0),
            (2.0, -0.2, 0.0), (3.0, 0.0, 0.0)]
    app = _interp_app(line, closed=False)

    app._recompute_interp_curve(0)

    curve = app.captured[0]
    for origin in line:
        assert _min_dist_to_polyline(curve, np.asarray(origin)) < 0.05
    _fp, _projected, u_at_nodes, _u_per_pt = app._interp_result_cache[0]
    assert len(u_at_nodes) == 4
    assert u_at_nodes[-1] == 1.0   # open: last node IS the u=1 endpoint


def test_set_interp_curve_none_clears_stale_geometry():
    pv = pytest.importorskip("pyvista")
    app = GeodesicSplineApp.__new__(GeodesicSplineApp)
    app._layer_visible = {LayerKind.INTERP: True}
    app._hover_curve_dirty = False

    stale = pv.PolyData(np.array([[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0],
                                  [2.0, 0.0, 0.0]]))
    visibility = []
    actor = types.SimpleNamespace(GetVisibility=lambda: True,
                                  SetVisibility=visibility.append)
    app._interp_cache = {0: (stale, actor)}

    # Real method under test (bound to the fake app).
    GeodesicSplineApp._set_interp_curve(app, 0, None)

    # Geometry cleared — a later blanket SetVisibility(True) renders
    # nothing instead of resurrecting the 3-point stale polyline.
    assert stale.n_points == 0
    assert visibility == [False]
    assert app._hover_curve_dirty is True
