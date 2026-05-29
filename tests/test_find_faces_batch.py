"""Exactness regression for the batched ``_find_faces_batch``.

``GeodesicMesh._find_faces_batch`` is a vectorised equivalent of the
per-point ``find_face`` loop used by the boundary check in
``_try_solve_on_region``.  It amortises the per-point Python↔C overhead
of ``KDTree.query`` into one batched call — profiled at ~46 % of
``compute_endpoint_local`` on the no-locator orange-worker / CLI-export
path (see ``tests/benchmark_endpoint_local.py``).

The contract it MUST keep: for every point it returns **exactly** the
face ``find_face`` returns — same nearest vertex, same incident-face
arg-min, same tie-break.  Any divergence would change which path points
the boundary check flags, hence the escalation path and the final
geodesic, so this is locked here in both regimes:

  * **no locator** (``build_locator=False``): the batched KDTree path —
    the one that actually reimplements the selection;
  * **locator on** (``build_locator=True``): the per-point fall-back —
    confirms the regime split is wired correctly.

This test owns the correctness contract; the benchmark only measures
speed.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("vtk")
pytest.importorskip("potpourri3d")

from geodesics import GeodesicMesh  # noqa: E402


def _octahedron():
    """Unit octahedron — a small closed 2-manifold, non-degenerate."""
    V = np.array([
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
    ], dtype=float)
    F = np.array([
        [4, 0, 2], [4, 2, 1], [4, 1, 3], [4, 3, 0],
        [5, 2, 0], [5, 1, 2], [5, 3, 1], [5, 0, 3],
    ], dtype=np.int32)
    return V, F


def _query_points(geo, seed=0):
    """A spread of query points that exercise every selection branch:
    face centroids, vertices (corner ⇒ several candidate faces, ties),
    edge midpoints (two candidates), and points perturbed off the
    surface along random directions."""
    rng = np.random.default_rng(seed)
    pts = [geo._face_centroids.copy(), geo.V.copy()]

    # Edge midpoints (each interior edge shared by two faces).
    mids = []
    for f in geo.F:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            mids.append(0.5 * (geo.V[f[a]] + geo.V[f[b]]))
    pts.append(np.array(mids, dtype=float))

    # Centroids nudged off the surface along a random direction.
    base = geo._face_centroids
    pts.append(base + 0.05 * rng.standard_normal(base.shape))

    # Random points on a slightly larger sphere (clearly off-surface).
    dirs = rng.standard_normal((150, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts.append(1.3 * dirs)

    return np.concatenate(pts, axis=0)


@pytest.mark.parametrize("build_locator", [False, True])
def test_batch_matches_per_point(build_locator):
    V, F = _octahedron()
    geo = GeodesicMesh(V, F, build_locator=build_locator)
    pts = _query_points(geo)

    want = np.array([geo.find_face(p) for p in pts], dtype=np.int64)
    got = geo._find_faces_batch(pts)

    assert got.shape == (len(pts),)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("build_locator", [False, True])
def test_single_point(build_locator):
    """A length-1 input (the boundary check never sees this, but the
    batched ``KDTree.query`` returns scalars for a 1-D input — confirm
    the (n, 3) contract holds for n == 1)."""
    V, F = _octahedron()
    geo = GeodesicMesh(V, F, build_locator=build_locator)
    p = geo._face_centroids[0]
    got = geo._find_faces_batch(p[None, :])
    assert got.shape == (1,)
    assert int(got[0]) == geo.find_face(p)
