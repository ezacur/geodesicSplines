"""Exactness contracts for two output-preserving micro-optimisations on
the geodesic core hot path:

* ``GeodesicMesh._barycentric`` — the five dot products were spelled out
  as scalar arithmetic instead of ``np.dot`` on 3-vectors (``np.dot``
  carries a per-call Python↔C dispatch cost that dominates for length-3
  inputs, and this is a leaf called once per candidate face inside
  ``find_face`` / ``_outside_score``).

* ``GeodesicMesh._bfs_advance`` — the per-ring neighbour expansion was
  vectorised (``adj[frontier]`` gather + dedupe) instead of a Python
  double loop over (frontier × 3 edges), while keeping the plain-set
  interface.

Both are byte-for-byte output-preserving end to end (verified against the
cascade parity oracle at ``tests/benchmark_endpoint_local.py --check``,
0.000e+00).  These fast unit tests own the local correctness contract so
a future refactor can't silently break it.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("vtk")
pytest.importorskip("potpourri3d")

from geodesics import GeodesicMesh  # noqa: E402


# ---------------------------------------------------------------------------
# _barycentric
# ---------------------------------------------------------------------------
def test_barycentric_recovers_known_coords():
    """For p built from known (u, v, w), _barycentric must recover them."""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        A, B, C = rng.standard_normal((3, 3))
        # Skip near-degenerate triangles (the method returns 1/3,1/3,1/3
        # there by design; correctness on real triangles is the contract).
        n = np.cross(B - A, C - A)
        if np.linalg.norm(n) < 1e-6:
            continue
        bary = rng.random(3)
        bary /= bary.sum()
        p = bary[0] * A + bary[1] * B + bary[2] * C
        u, v, w = GeodesicMesh._barycentric(p, A, B, C)
        np.testing.assert_allclose((u, v, w), bary, atol=1e-12, rtol=0)
        # Barycentric coords always sum to 1.
        assert abs((u + v + w) - 1.0) < 1e-12


def test_barycentric_degenerate_triangle():
    """A zero-area triangle returns the centroid sentinel (1/3, 1/3, 1/3)."""
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([1.0, 0.0, 0.0])
    C = np.array([2.0, 0.0, 0.0])  # collinear ⇒ denom ~0
    u, v, w = GeodesicMesh._barycentric(np.array([0.5, 0.0, 0.0]), A, B, C)
    assert (u, v, w) == (1 / 3, 1 / 3, 1 / 3)


def test_barycentric_deterministic():
    rng = np.random.default_rng(1)
    A, B, C = rng.standard_normal((3, 3))
    p = rng.standard_normal(3)
    assert GeodesicMesh._barycentric(p, A, B, C) == \
        GeodesicMesh._barycentric(p, A, B, C)


# ---------------------------------------------------------------------------
# _bfs_advance
# ---------------------------------------------------------------------------
def _octahedron():
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


def _grid(n):
    """An (n x n) triangulated grid in the z=0 plane — many shared edges,
    real boundary faces (so the -1 neighbour branch is exercised)."""
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    V = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n)])
    F = []
    for r in range(n - 1):
        for c in range(n - 1):
            a = r * n + c
            F.append([a, a + 1, a + n])
            F.append([a + 1, a + n + 1, a + n])
    return V, np.array(F, dtype=np.int32)


def _scalar_bfs_advance(adj, visited, frontier, extra_rings):
    """Reference: the original scalar double-loop, verbatim."""
    for _ in range(extra_rings):
        if not frontier:
            return
        next_f = set()
        for fi in frontier:
            for nb in adj[fi]:
                nb_i = int(nb)
                if nb_i >= 0 and nb_i not in visited:
                    visited.add(nb_i)
                    next_f.add(nb_i)
        if not next_f:
            return
        frontier.clear()
        frontier.update(next_f)


@pytest.mark.parametrize("mesh", ["octahedron", "grid"])
@pytest.mark.parametrize("rings", [1, 2, 3, 10])
def test_bfs_advance_matches_scalar(mesh, rings):
    V, F = _octahedron() if mesh == "octahedron" else _grid(7)
    geo = GeodesicMesh(V, F, build_locator=False)

    for seed in range(len(geo.F)):
        # Vectorised (production) state.
        vis_v, fr_v = geo._bfs_init([seed])
        geo._bfs_advance(vis_v, fr_v, rings)
        # Scalar reference state from the same seed.
        vis_s, fr_s = {seed}, {seed}
        _scalar_bfs_advance(geo._face_adj, vis_s, fr_s, rings)

        assert vis_v == vis_s, f"visited diverged (mesh={mesh}, seed={seed})"
        assert fr_v == fr_s, f"frontier diverged (mesh={mesh}, seed={seed})"
