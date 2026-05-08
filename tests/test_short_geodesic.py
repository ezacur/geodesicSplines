"""Unit tests for ``GeodesicMesh.short_geodesic``.

The method is the fast path used by the orange worker's phase-3 chord-
bridging.  The contract is:

  - same face       → ``[p0, p1]`` (straight 3-D segment)
  - adjacent face,
    crossing inside shared edge with margin
                    → ``[p0, q, p1]`` (two-segment polyline through *q*)
  - any other case  → ``None`` (caller falls back to compute_endpoint_local)

Tests cover all three branches plus the on-vertex rejection.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("vtk")

from geodesics import GeodesicMesh  # noqa: E402


# ---------------------------------------------------------------
# A flat two-triangle mesh in the XY plane:
#
#         v2 (0, 1, 0)
#         /\
#        /  \
#       /    \
#  v0 *------* v1   shared edge = (v0, v1)
#   (0,0,0) (1,0,0)
#       \    /
#        \  /
#         \/
#         v3 (0.5, -1, 0)
#
# Triangles: T0 = (v0, v1, v2), T1 = (v1, v0, v3) — opposite winding so
# they share the edge (v0, v1) with the same orientation.
# ---------------------------------------------------------------
@pytest.fixture
def two_triangle_mesh():
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
    # build_locator=False — the test does not exercise VTK pickers
    return GeodesicMesh(V, F, build_locator=False)


def test_same_face_returns_straight_segment(two_triangle_mesh):
    gm = two_triangle_mesh
    # Two interior points in T0
    p0 = np.array([0.2, 0.3, 0.0])
    p1 = np.array([0.4, 0.4, 0.0])
    out = gm.short_geodesic(p0, p1, face_a=0, face_b=0)
    assert out is not None
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[0], p0)
    np.testing.assert_allclose(out[1], p1)


def test_adjacent_face_inserts_edge_crossing(two_triangle_mesh):
    gm = two_triangle_mesh
    # p0 in T0, p1 in T1 — symmetric across the shared edge so the
    # crossing must land at x=0.5, y=0 (mid-edge).
    p0 = np.array([0.5, 0.4, 0.0])   # T0
    p1 = np.array([0.5, -0.4, 0.0])  # T1
    out = gm.short_geodesic(p0, p1, face_a=0, face_b=1)
    assert out is not None
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out[0], p0)
    np.testing.assert_allclose(out[2], p1)
    # Crossing must be on the shared edge (z=0, y=0), midway by symmetry.
    np.testing.assert_allclose(out[1], [0.5, 0.0, 0.0], atol=1e-12)


def test_off_axis_adjacent_inserts_correct_crossing(two_triangle_mesh):
    gm = two_triangle_mesh
    # Asymmetric: p0 closer to v0, p1 closer to v1 → crossing skewed.
    p0 = np.array([0.1, 0.1, 0.0])   # T0
    p1 = np.array([0.9, -0.1, 0.0])  # T1
    out = gm.short_geodesic(p0, p1, face_a=0, face_b=1)
    assert out is not None
    assert out.shape == (3, 3)
    # Mesh is flat, so the geodesic equals the Euclidean line.
    # Solve y = 0: parametric line p0 + t*(p1 - p0), y = 0.1 + t*(-0.2) = 0 → t = 0.5
    expected_q = p0 + 0.5 * (p1 - p0)
    np.testing.assert_allclose(out[1], expected_q, atol=1e-12)


def test_crossing_near_vertex_returns_none(two_triangle_mesh):
    gm = two_triangle_mesh
    # Construct p0/p1 so the straight line (in unfolded plane) lands
    # almost exactly on the shared-edge vertex v0 = (0, 0, 0).
    # margin = max(1e-7, 0.001 * edge_len=1.0) = 0.001 → reject if
    # crossing s < 0.001 from v0.
    p0 = np.array([1e-5, 0.5, 0.0])    # straight down crosses (1e-5, 0)
    p1 = np.array([1e-5, -0.5, 0.0])
    out = gm.short_geodesic(p0, p1, face_a=0, face_b=1)
    assert out is None, "vertex-adjacent crossing must fall back"


def test_non_adjacent_returns_none():
    # Build a 3-triangle strip where the two outer triangles share no
    # edge.  ``find_face`` is used to resolve indices because Morton
    # reorder permutes F at construction.
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.5, 1.0, 0.0],
    ], dtype=float)
    F = np.array([
        [0, 1, 3],   # left
        [1, 4, 3],   # middle (adjacent to both)
        [1, 2, 4],   # right (NOT adjacent to left)
    ], dtype=int)
    gm = GeodesicMesh(V, F, build_locator=False)
    p0 = np.array([0.3, 0.3, 0.0])
    p1 = np.array([1.7, 0.3, 0.0])
    fa = gm.find_face(p0)
    fb = gm.find_face(p1)
    assert fa != fb
    # Confirm the test premise: these two are NOT edge-adjacent
    # (they share at most vertex 1).
    adj = gm._face_adj[fa]
    assert fb not in [int(a) for a in adj], "test premise violated"
    out = gm.short_geodesic(p0, p1, face_a=fa, face_b=fb)
    assert out is None, "non-adjacent triangles must fall back"


def test_face_indices_inferred_when_omitted(two_triangle_mesh):
    """When *face_a* / *face_b* are not passed, the method must call
    ``find_face`` internally and still produce the correct result."""
    gm = two_triangle_mesh
    p0 = np.array([0.5, 0.4, 0.0])
    p1 = np.array([0.5, -0.4, 0.0])
    out = gm.short_geodesic(p0, p1)  # no face hints
    assert out is not None
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out[1], [0.5, 0.0, 0.0], atol=1e-12)
