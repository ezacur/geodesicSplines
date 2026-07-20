"""Regression test for the CLI blue export's ``path_a`` orientation.

``hybrid_de_casteljau_curve`` expects ``path_in`` oriented P1 -> H_in
exactly as stored on the node (it reverses internally); the editor
passes ``n1.path_a`` unreversed.  ``compute_blue`` used to pass a
pre-reversed copy, double-reversing the third level-1 control segment:
every exported span ended at the H_in handle instead of the destination
node, leaving inter-span gaps.

Uses a flat grid mesh — geodesics are straight lines there, so the
span-endpoint contract can be asserted exactly.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("vtk")
pytest.importorskip("potpourri3d")

from geodesics import GeodesicMesh  # noqa: E402
from spline_export import compute_blue  # noqa: E402


@pytest.fixture(scope="module")
def flat_grid():
    """A flat 5x5 XY grid (32 triangles), no VTK locator needed."""
    n = 5
    xs, ys = np.meshgrid(np.arange(n, dtype=float),
                         np.arange(n, dtype=float))
    V = np.column_stack(
        [xs.ravel(), ys.ravel(), np.zeros(n * n)])
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            v0 = j * n + i
            v1 = v0 + 1
            v2 = v0 + n
            v3 = v2 + 1
            faces.append([v0, v1, v3])
            faces.append([v0, v3, v2])
    F = np.array(faces, dtype=int)
    return GeodesicMesh(V, F, build_locator=False)


def _node(origin, p_a, p_b):
    """Node dict in the shape ``rebuild_mesh_and_nodes`` produces.

    ``path_a`` / ``path_b`` follow the storage convention:
    origin -> handle (path[0] == origin, path[-1] == handle).
    """
    origin = np.asarray(origin, dtype=float)
    p_a = np.asarray(p_a, dtype=float) if p_a is not None else None
    p_b = np.asarray(p_b, dtype=float) if p_b is not None else None
    return {
        'origin': origin,
        'p_a': p_a, 'p_b': p_b,
        'path_a': np.array([origin, p_a]) if p_a is not None else None,
        'path_b': np.array([origin, p_b]) if p_b is not None else None,
    }


def test_blue_span_ends_at_destination_node(flat_grid):
    n0 = _node([1.2, 2.1, 0.0], [0.7, 2.1, 0.0], [1.7, 2.1, 0.0])
    n1 = _node([3.1, 2.3, 0.0], [2.6, 2.3, 0.0], [3.6, 2.3, 0.0])

    spans = compute_blue(flat_grid, [n0, n1], closed=False, n_samples=33)

    assert len(spans) == 1
    pts = spans[0]
    # The Bezier span must start at n0.origin and end at n1.origin.
    # With the double-reversal bug it ended at n1.p_a (0.5 away).
    np.testing.assert_allclose(pts[0], n0['origin'], atol=1e-6)
    np.testing.assert_allclose(pts[-1], n1['origin'], atol=1e-6)


def test_blue_spans_are_gap_free_on_closed_spline(flat_grid):
    nodes = [
        _node([1.1, 1.1, 0.0], [0.8, 1.4, 0.0], [1.4, 0.8, 0.0]),
        _node([3.1, 1.2, 0.0], [2.8, 0.9, 0.0], [3.4, 1.5, 0.0]),
        _node([2.1, 3.2, 0.0], [2.5, 3.1, 0.0], [1.7, 3.3, 0.0]),
    ]

    spans = compute_blue(flat_grid, nodes, closed=True, n_samples=17)

    assert len(spans) == 3   # N spans on a closed N-node spline
    for i, pts in enumerate(spans):
        start = nodes[i]['origin']
        end = nodes[(i + 1) % 3]['origin']
        np.testing.assert_allclose(pts[0], start, atol=1e-6)
        np.testing.assert_allclose(pts[-1], end, atol=1e-6)
