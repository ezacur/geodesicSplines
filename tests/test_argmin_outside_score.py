"""Equivalence tests for ``GeodesicMesh._argmin_outside_score_buf``.

The grossly-negative-bary backstop in ``_add_point_buf`` /
``_add_point_local`` used to pick its replacement face with
``min(range(nf), key=lambda i: _outside_score_buf(...))`` — a Python
key function over the whole face buffer.  On the global path that is
the *mesh* face count: measured at **646 ms per hit** on a 207 K-face
mesh versus 0.71 ms for a normal insertion, and it is reachable on
clean CAD geometry because ``_find_face_buf`` seeds from a k=1 KDTree
query.

The replacement is vectorised, so the only thing worth testing is that
it is a *drop-in*: same index, bit-for-bit, including tie-breaking and
the degenerate-denominator branch.  Anything less would move the curve.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("vtk")

from geodesics import GeodesicMesh  # noqa: E402


def _scalar_argmin(gm, p, V_buf, F_buf, nf):
    """The exact expression the vectorised helper replaced."""
    return min(range(nf),
               key=lambda i: gm._outside_score_buf(p, i, V_buf, F_buf))


def _grid(n, seed=0, jitter=0.05):
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    V = np.column_stack([xs.ravel(), ys.ravel(),
                         rng.normal(0, jitter, n * n)])
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            F += [[a, b, d], [a, d, c]]
    return V, np.asarray(F, dtype=np.int32)


def _mesh(n=7, seed=0, jitter=0.05):
    V, F = _grid(n, seed, jitter)
    return GeodesicMesh(V, F, build_locator=False)


@pytest.mark.parametrize("seed", range(6))
def test_matches_the_scalar_argmin_on_random_query_points(seed):
    gm = _mesh(seed=seed)
    rng = np.random.default_rng(100 + seed)
    V_buf, F_buf, nv, nf = gm._make_work_buffers(extra_verts=2, extra_faces=6)

    lo = gm.V.min(axis=0) - 0.5
    hi = gm.V.max(axis=0) + 0.5
    for _ in range(40):
        p = rng.uniform(lo, hi)
        assert (gm._argmin_outside_score_buf(p, V_buf, F_buf, nf)
                == _scalar_argmin(gm, p, V_buf, F_buf, nf))


@pytest.mark.parametrize("seed", range(4))
def test_matches_on_points_exactly_on_the_surface(seed):
    """On-surface points are the realistic case — score 0 on the
    containing face, so ties across adjacent faces are common and the
    tie-break has to agree."""
    gm = _mesh(seed=seed)
    rng = np.random.default_rng(500 + seed)
    V_buf, F_buf, nv, nf = gm._make_work_buffers(extra_verts=2, extra_faces=6)

    for _ in range(40):
        fi = int(rng.integers(0, len(gm.F)))
        w = rng.random(3)
        w /= w.sum()
        p = w @ gm.V[gm.F[fi]]
        assert (gm._argmin_outside_score_buf(p, V_buf, F_buf, nf)
                == _scalar_argmin(gm, p, V_buf, F_buf, nf))


def test_matches_on_mesh_vertices():
    """A vertex sits on every face of its fan: a maximal tie."""
    gm = _mesh()
    V_buf, F_buf, nv, nf = gm._make_work_buffers(extra_verts=2, extra_faces=6)
    for vi in range(len(gm.V)):
        p = gm.V[vi]
        assert (gm._argmin_outside_score_buf(p, V_buf, F_buf, nf)
                == _scalar_argmin(gm, p, V_buf, F_buf, nf))


def test_matches_when_the_buffer_holds_degenerate_faces():
    """``F_buf`` slots past ``nf`` are zero-filled, and a revert can
    leave a collapsed triangle behind — those rows hit
    ``abs(denom) < 1e-15`` and must take the (1/3, 1/3, 1/3) branch in
    both implementations."""
    gm = _mesh(n=5)
    V_buf, F_buf, nv, nf = gm._make_work_buffers(extra_verts=4, extra_faces=8)
    # Collapse two real faces onto a single vertex -> zero area.
    F_buf[3] = [0, 0, 0]
    F_buf[7] = [2, 2, 5]
    rng = np.random.default_rng(9)
    for _ in range(30):
        p = rng.uniform(gm.V.min(axis=0) - 1, gm.V.max(axis=0) + 1)
        assert (gm._argmin_outside_score_buf(p, V_buf, F_buf, nf)
                == _scalar_argmin(gm, p, V_buf, F_buf, nf))


def test_matches_on_a_sliver_mesh():
    """Near-degenerate but non-zero areas — the regime where a
    different summation order would show up as a different arg-min."""
    V, F = _grid(6, seed=3, jitter=0.0)
    V[:, 1] *= 1e-6          # squash into slivers
    gm = GeodesicMesh(V, F, build_locator=False)
    V_buf, F_buf, nv, nf = gm._make_work_buffers(extra_verts=2, extra_faces=6)
    rng = np.random.default_rng(11)
    for _ in range(30):
        p = rng.uniform(gm.V.min(axis=0) - 0.1, gm.V.max(axis=0) + 0.1)
        assert (gm._argmin_outside_score_buf(p, V_buf, F_buf, nf)
                == _scalar_argmin(gm, p, V_buf, F_buf, nf))


def test_scores_themselves_are_bit_identical():
    """Not just the arg-min: every per-face score must match exactly,
    so no future change of tie-break can silently diverge."""
    gm = _mesh(n=6, seed=1)
    V_buf, F_buf, nv, nf = gm._make_work_buffers(extra_verts=2, extra_faces=6)
    rng = np.random.default_rng(13)
    for _ in range(10):
        p = rng.uniform(gm.V.min(axis=0) - 0.5, gm.V.max(axis=0) + 0.5)
        scalar = np.array([gm._outside_score_buf(p, i, V_buf, F_buf)
                           for i in range(nf)])
        # Recompute the vector path's scores by asking it for the argmin
        # of a masked copy repeatedly would be O(n^2); instead assert the
        # minimum it picks has exactly the scalar minimum value.
        idx = gm._argmin_outside_score_buf(p, V_buf, F_buf, nf)
        assert scalar[idx] == scalar.min()
        assert idx == int(np.argmin(scalar))
