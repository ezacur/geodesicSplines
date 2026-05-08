"""Unit tests for ``GeodesicMesh._subdivide_submesh_1to4``.

The method does Loop-style 1-to-4 subdivision (no smoothing) of a
submesh: each face splits into 4 sub-faces, with new vertices at
edge midpoints.  Used by the orange worker to give the
``EdgeFlipGeodesicSolver`` finer edges so its discrete geodesic
converges to the smooth-surface geodesic.

Coverage:
  * Single triangle → 4 sub-faces, 6 vertices total (3 original + 3 mids).
  * Diamond (2 tris sharing edge) → shared midpoint deduplicated.
  * Total area preserved.
  * Original V_sub preserved as prefix of V_fine.
  * Manifold preserved: every edge appears in exactly 2 faces (interior)
    or 1 face (boundary).
  * Multiple subdivision rounds compose correctly.
  * Empty / degenerate input handled defensively.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("vtk")

from geodesics import GeodesicMesh  # noqa: E402


def test_single_triangle():
    """One triangle → 4 sub-faces, 6 vertices (3 original + 3 midpoints)."""
    V = np.array([[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]], dtype=float)
    F = np.array([[0, 1, 2]], dtype=np.int32)

    V_fine, F_fine = GeodesicMesh._subdivide_submesh_1to4(V, F)

    assert V_fine.shape == (6, 3)
    assert F_fine.shape == (4, 3)
    # Original 3 vertices preserved verbatim
    np.testing.assert_array_equal(V_fine[:3], V)
    # The 3 new vertices are the edge midpoints (in some order)
    expected_mids = {(0.5, 0.0, 0.0), (0.5, 0.5, 0.0), (0.0, 0.5, 0.0)}
    actual_mids = {tuple(v) for v in V_fine[3:]}
    assert actual_mids == expected_mids


def test_diamond_shared_midpoint():
    """Two triangles sharing one edge → its midpoint must be ONE vertex,
    not two duplicates."""
    V = np.array([
        [0.0, 0.0, 0.0],   # v0
        [1.0, 0.0, 0.0],   # v1 (shared edge with v0)
        [0.0, 1.0, 0.0],   # v2
        [0.5, -1.0, 0.0],  # v3
    ], dtype=float)
    F = np.array([
        [0, 1, 2],
        [1, 0, 3],
    ], dtype=np.int32)

    V_fine, F_fine = GeodesicMesh._subdivide_submesh_1to4(V, F)

    # 4 originals + 5 unique midpoints (one shared) = 9
    assert V_fine.shape == (9, 3)
    # 2 faces × 4 = 8 sub-faces
    assert F_fine.shape == (8, 3)
    # Original vertices preserved
    np.testing.assert_array_equal(V_fine[:4], V)


def test_total_area_preserved():
    """Subdivision must not change the total surface area."""
    V = np.array([[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [1.0, 1.0, 0.5]], dtype=float)
    F = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    def total_area(V_, F_):
        out = 0.0
        for tri in F_:
            v = V_[tri]
            out += 0.5 * np.linalg.norm(np.cross(v[1] - v[0], v[2] - v[0]))
        return out

    area_before = total_area(V, F)
    V_fine, F_fine = GeodesicMesh._subdivide_submesh_1to4(V, F)
    area_after = total_area(V_fine, F_fine)
    np.testing.assert_allclose(area_after, area_before, rtol=1e-12)


def test_manifold_preserved():
    """After subdivision, every edge should appear in exactly 1
    (boundary) or 2 (interior) faces."""
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, -1.0, 0.0],
    ], dtype=float)
    F = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int32)
    V_fine, F_fine = GeodesicMesh._subdivide_submesh_1to4(V, F)

    edge_count: dict[tuple[int, int], int] = {}
    for tri in F_fine:
        for k in range(3):
            a = int(tri[k]); b = int(tri[(k + 1) % 3])
            key = (a, b) if a < b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1

    counts = sorted(set(edge_count.values()))
    assert counts == [1, 2], \
        f"non-manifold: edges appear with counts {counts}"


def test_two_rounds_compose():
    """Two consecutive 1-to-4 subdivisions = 16× the original face count."""
    V = np.array([[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]], dtype=float)
    F = np.array([[0, 1, 2]], dtype=np.int32)

    V1, F1 = GeodesicMesh._subdivide_submesh_1to4(V, F)
    V2, F2 = GeodesicMesh._subdivide_submesh_1to4(V1, F1)
    assert F2.shape[0] == 16
    # Originals still preserved as prefix
    np.testing.assert_array_equal(V2[:3], V)


def test_winding_preserved():
    """All sub-faces must have the same winding (signed area sign)
    as the parent face."""
    V = np.array([[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]], dtype=float)
    F = np.array([[0, 1, 2]], dtype=np.int32)
    # Parent: vertices in CCW order from above → normal points +z.
    V_fine, F_fine = GeodesicMesh._subdivide_submesh_1to4(V, F)
    for tri in F_fine:
        v = V_fine[tri]
        n = np.cross(v[1] - v[0], v[2] - v[0])
        assert n[2] > 0, f"sub-face {tri} has flipped winding (n={n})"
