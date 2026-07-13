"""Unit tests for the phase-3 degraded-flag propagation.

``_build_chord_geodesic`` is the orange worker's phase-3 chord bridge.
Historically it silently returned a straight Euclidean segment when
both solvers failed, so the span rendered without the red fallback
repaint — exactly the "phantom curve" the editor promises never to
show.  The contract now is:

  - solvable chord   → ``(geodesic polyline, False)``
  - unsolvable chord → ``([p_left, p_right], True)``

and ``_phase3_chord_bridge`` ORs the per-chord flags into a single
bool that the worker folds into the ``('done', ...)`` degraded flag.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("vtk")
pytest.importorskip("pyvista")

from geo_splines import _build_chord_geodesic, _phase3_chord_bridge  # noqa: E402
from geodesics import GeodesicMesh  # noqa: E402


@pytest.fixture
def two_triangle_mesh():
    """Flat two-triangle mesh in the XY plane sharing edge (v0, v1)."""
    V = np.array([
        [0.0, 0.0, 0.0],   # v0
        [1.0, 0.0, 0.0],   # v1
        [0.0, 1.0, 0.0],   # v2
        [0.5, -1.0, 0.0],  # v3
    ], dtype=float)
    F = np.array([
        [0, 1, 2],
        [1, 0, 3],
    ], dtype=int)
    return GeodesicMesh(V, F, build_locator=False)


@pytest.fixture
def disconnected_mesh():
    """Two triangles in separate connected components, 100 units apart."""
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [100.0, 0.0, 0.0],
        [101.0, 0.0, 0.0],
        [100.0, 1.0, 0.0],
    ], dtype=float)
    F = np.array([
        [0, 1, 2],
        [3, 4, 5],
    ], dtype=int)
    return GeodesicMesh(V, F, build_locator=False)


class _CollectingWriter:
    """Stub for the worker's pipe write-end: records every send()."""

    def __init__(self):
        self.msgs = []

    def send(self, msg):
        self.msgs.append(msg)


def test_solvable_chord_is_not_degraded(two_triangle_mesh):
    gm = two_triangle_mesh
    p0 = np.array([0.2, 0.3, 0.0])
    p1 = np.array([0.4, 0.4, 0.0])
    seg, degraded = _build_chord_geodesic(gm, p0, p1)
    assert degraded is False
    assert len(seg) >= 2
    np.testing.assert_allclose(seg[0], p0, atol=1e-9)
    np.testing.assert_allclose(seg[-1], p1, atol=1e-9)


def test_cross_component_chord_is_degraded(disconnected_mesh):
    gm = disconnected_mesh
    p_left = gm.V[[0, 1, 2]].mean(axis=0)   # centroid, component 0
    p_right = gm.V[[3, 4, 5]].mean(axis=0)  # centroid, component 1
    seg, degraded = _build_chord_geodesic(gm, p_left, p_right)
    assert degraded is True
    # Last-resort geometry: the plain 2-point Euclidean stand-in.
    assert len(seg) == 2
    np.testing.assert_allclose(seg[0], p_left, atol=1e-9)
    np.testing.assert_allclose(seg[1], p_right, atol=1e-9)


def test_phase3_returns_false_when_all_chords_solve(two_triangle_mesh):
    gm = two_triangle_mesh
    p_list = [
        np.array([0.2, 0.3, 0.0]),
        np.array([0.4, 0.4, 0.0]),
        np.array([0.3, 0.5, 0.0]),
    ]
    writer = _CollectingWriter()
    degraded = _phase3_chord_bridge(gm, (0, 0), p_list, writer,
                                    submesh_subdiv=0)
    assert degraded is False
    assert len(writer.msgs) == 1
    kind, span_key, polyline = writer.msgs[0]
    assert kind == 'chord_geo'
    assert span_key == (0, 0)
    assert len(polyline) >= len(p_list)


def test_phase3_propagates_degraded_chord(disconnected_mesh):
    gm = disconnected_mesh
    p_list = [
        gm.V[[0, 1, 2]].mean(axis=0),
        gm.V[[3, 4, 5]].mean(axis=0),
    ]
    writer = _CollectingWriter()
    degraded = _phase3_chord_bridge(gm, (0, 0), p_list, writer,
                                    submesh_subdiv=0)
    assert degraded is True
    # The polyline is still sent — degraded geometry renders, but the
    # flag lets the parent repaint the span red.
    assert len(writer.msgs) == 1
    assert writer.msgs[0][0] == 'chord_geo'
