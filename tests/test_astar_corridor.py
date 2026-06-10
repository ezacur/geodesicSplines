"""Regression test for the A* single-pair corridor.

``GeodesicMesh._astar_corridor`` replaces scipy's full single-source
Dijkstra in ``_dijkstra_corridor`` (gated by ``USE_ASTAR_CORRIDOR``,
default on) with a single-pair A* using an admissible centroid-distance
heuristic.  It explores only the corridor between the two faces, so it
is much faster on large meshes — but it must return a **cost-optimal**
path (same total weight as scipy), otherwise the seed corridor would be
wrong.

This locks the invariant that actually matters: A* agrees with scipy's
shortest-path COST.  (The exact sequence of faces at ties may differ;
that is fine — the corridor is only a BFS seed, and parity of the final
curve was verified bit-for-bit across 33 real sessions.)
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("vtk")

from geodesics import GeodesicMesh  # noqa: E402


def _grid_mesh(n):
    """An (n x n) triangulated grid in the z=0 plane → 2*(n-1)^2 faces."""
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    V = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n)])
    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            a = r * n + c
            b = a + 1
            d = a + n
            e = d + 1
            faces.append([a, b, d])
            faces.append([b, e, d])
    return V, np.array(faces, dtype=np.int32)


@pytest.fixture
def grid_geo():
    V, F = _grid_mesh(8)  # 98 faces
    return GeodesicMesh(V, F, build_locator=False)


def _scipy_cost(geo, start, end):
    from scipy.sparse.csgraph import dijkstra
    graph = geo._get_face_dual_graph()
    dist = dijkstra(graph, indices=start, directed=False)
    return float(dist[end])


def _path_cost(geo, path):
    """Sum of centroid-distance edge weights along a face path."""
    c = geo._face_centroids
    return float(sum(np.linalg.norm(c[path[i]] - c[path[i + 1]])
                     for i in range(len(path) - 1)))


@pytest.mark.parametrize("start,end", [(0, 97), (5, 90), (0, 49), (12, 60), (3, 80)])
def test_astar_cost_optimal(grid_geo, start, end):
    geo = grid_geo
    path = geo._astar_corridor(start, end)
    assert path is not None, f"A* found no path {start}->{end}"
    # Endpoints (A* returns end -> start, matching _dijkstra_corridor).
    assert path[0] == end and path[-1] == start
    # Cost-optimal: same total weight as scipy's shortest path.
    np.testing.assert_allclose(_path_cost(geo, path), _scipy_cost(geo, start, end),
                               rtol=1e-9, atol=1e-9)


def test_astar_path_is_adjacent_chain(grid_geo):
    """Consecutive faces in the returned path must be edge-adjacent."""
    geo = grid_geo
    adj = geo._face_adj
    path = geo._astar_corridor(0, 97)
    for u, v in zip(path[:-1], path[1:], strict=False):
        assert v in set(int(x) for x in adj[u]), f"{u}->{v} not adjacent"


def test_astar_matches_dijkstra_corridor_cost_via_flag(grid_geo):
    """The corridor returned with USE_ASTAR_CORRIDOR on/off has equal cost."""
    geo = grid_geo
    # Drive through the public method by toggling the flag.  Pick faces
    # far apart; feed their centroids as the query points.
    c = geo._face_centroids
    p_start, p_end = c[0], c[97]

    geo.USE_ASTAR_CORRIDOR = False
    scipy_path = geo._dijkstra_corridor(p_start, p_end)
    geo.USE_ASTAR_CORRIDOR = True
    astar_path = geo._dijkstra_corridor(p_start, p_end)

    assert scipy_path is not None and astar_path is not None
    np.testing.assert_allclose(_path_cost(geo, astar_path),
                               _path_cost(geo, scipy_path),
                               rtol=1e-9, atol=1e-9)
