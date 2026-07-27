"""Unit tests for ``GeodesicMesh._relink_adj_1to3``.

The 1-to-3 interior split used to leave ``adj_buf`` untouched.  That is
harmless for ``find_face`` (which scans), but both endpoints of a span
share one ``adj_buf``, so a stale entry silently disabled the **2-to-4
edge split for the second insertion** in that neighbourhood — the
mechanism ``docs/ARCHITECTURE.md`` calls "the load-bearing fix for
smooth orange / didactic agreement on dense meshes".  The degradation
was invisible: the insertion just fell through to a 1-to-3 on a
near-edge point.

The oracle here is ``_build_face_adj_buf`` — a fresh rebuild from the
post-split ``F_buf`` must agree with the incrementally maintained
buffer, entry for entry.  That is the same standard the existing
``_split_edge_2to4`` bookkeeping is held to.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("vtk")

from geodesics import GeodesicMesh  # noqa: E402


def _grid(n):
    """(n x n) vertex grid triangulated into 2(n-1)^2 faces."""
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    V = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n)])
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            F += [[a, b, d], [a, d, c]]
    return V, np.asarray(F, dtype=np.int32)


def _split_1to3(V_buf, F_buf, adj_buf, nv, nf, face_idx, p):
    """Apply the same 1-to-3 rewrite ``_add_point_local`` performs."""
    fa, fb, fc = (int(F_buf[face_idx, 0]), int(F_buf[face_idx, 1]),
                  int(F_buf[face_idx, 2]))
    p_idx = nv
    V_buf[p_idx] = p
    F_buf[face_idx] = [p_idx, fa, fb]
    F_buf[nf] = [p_idx, fb, fc]
    F_buf[nf + 1] = [p_idx, fc, fa]
    GeodesicMesh._relink_adj_1to3(adj_buf, face_idx, nf)
    return nv + 1, nf + 2


def _buffers(V, F, extra_faces=12, extra_verts=6):
    nv, nf = len(V), len(F)
    V_buf = np.zeros((nv + extra_verts, 3), dtype=float)
    V_buf[:nv] = V
    F_buf = np.zeros((nf + extra_faces, 3), dtype=np.int32)
    F_buf[:nf] = F
    adj_buf = GeodesicMesh._build_face_adj_buf(F_buf, nf, extra=extra_faces)
    return V_buf, F_buf, adj_buf, nv, nf


def _assert_matches_rebuild(F_buf, adj_buf, nf):
    fresh = GeodesicMesh._build_face_adj_buf(F_buf, nf, extra=0)
    np.testing.assert_array_equal(adj_buf[:nf], fresh[:nf])


@pytest.mark.parametrize("face_idx", [0, 1, 5, 17])
def test_single_split_matches_a_fresh_rebuild(face_idx):
    V, F = _grid(5)
    V_buf, F_buf, adj_buf, nv, nf = _buffers(V, F)
    tri = V_buf[F_buf[face_idx]]
    nv, nf = _split_1to3(V_buf, F_buf, adj_buf, nv, nf, face_idx,
                         tri.mean(axis=0))
    _assert_matches_rebuild(F_buf, adj_buf, nf)


def test_chained_splits_stay_consistent():
    """Both endpoints of a span share one ``adj_buf``, so the second
    insertion must see the first one's topology."""
    V, F = _grid(6)
    V_buf, F_buf, adj_buf, nv, nf = _buffers(V, F, extra_faces=16,
                                             extra_verts=8)
    for face_idx in (0, 3, 9, 12):
        tri = V_buf[F_buf[face_idx]]
        nv, nf = _split_1to3(V_buf, F_buf, adj_buf, nv, nf, face_idx,
                             tri.mean(axis=0))
        _assert_matches_rebuild(F_buf, adj_buf, nf)


def test_split_on_a_boundary_face_keeps_minus_one():
    """Face 0 of the grid has a real mesh boundary edge; the sub-face
    that inherits it must keep ``-1``, not point at a neighbour."""
    V, F = _grid(4)
    V_buf, F_buf, adj_buf, nv, nf = _buffers(V, F)
    assert (adj_buf[0] < 0).any(), "fixture expects a boundary face"
    tri = V_buf[F_buf[0]]
    nv, nf = _split_1to3(V_buf, F_buf, adj_buf, nv, nf, 0, tri.mean(axis=0))
    _assert_matches_rebuild(F_buf, adj_buf, nf)
    assert (adj_buf[:nf] < 0).any()


def test_relink_is_symmetric():
    """Adjacency must stay a two-way relation: if A says B, B says A."""
    V, F = _grid(5)
    V_buf, F_buf, adj_buf, nv, nf = _buffers(V, F)
    tri = V_buf[F_buf[7]]
    nv, nf = _split_1to3(V_buf, F_buf, adj_buf, nv, nf, 7, tri.mean(axis=0))

    for fi in range(nf):
        for s in range(3):
            nb = int(adj_buf[fi, s])
            if nb < 0:
                continue
            assert fi in [int(x) for x in adj_buf[nb]], (
                f"face {fi} claims neighbour {nb}, which does not claim it back")


# ---------------------------------------------------------------------------
# The relink makes the 2-to-4 split *reachable* where it used to bail.
# pp3d segfaults on non-manifold input, so that new reachability has to
# be shown not to break the manifold invariant.
# ---------------------------------------------------------------------------

def _manifold_maxima(F):
    """(max undirected edge incidence, max directed edge incidence)."""
    e0 = np.concatenate([F[:, 0], F[:, 1], F[:, 2]]).astype(np.int64)
    e1 = np.concatenate([F[:, 1], F[:, 2], F[:, 0]]).astype(np.int64)
    und = (np.minimum(e0, e1) << 32) | np.maximum(e0, e1)
    dir_ = (e0 << 32) | e1
    return (int(np.unique(und, return_counts=True)[1].max()),
            int(np.unique(dir_, return_counts=True)[1].max()))


@pytest.mark.parametrize("seed", range(6))
def test_two_insertions_sharing_one_adj_buf_stay_manifold(seed):
    """``_try_endpoint_insertion`` inserts both span endpoints against a
    single ``adj_buf``.  Points are biased onto edges so the 2-to-4 path
    fires; the result must stay 2-manifold and the maintained adjacency
    must still match a rebuild."""
    rng = np.random.default_rng(seed)
    V, F = _grid(7)
    gm = GeodesicMesh(V, F, build_locator=False)

    V_buf, F_buf, nv, nf = gm._make_work_buffers(extra_verts=2, extra_faces=6)
    adj_buf = np.full((nf + 6, 3), -1, dtype=np.int32)
    adj_buf[:nf] = gm._face_adj

    for _ in range(2):
        fi = int(rng.integers(0, len(gm.F)))
        tri = gm.V[gm.F[fi]]
        w = rng.random(3)
        w[int(rng.integers(0, 3))] = 10.0 ** rng.uniform(-9, -2)  # onto an edge
        w /= w.sum()
        _, nv, nf = gm._add_point_buf(w @ tri, V_buf, F_buf, nv, nf,
                                      adj_buf=adj_buf)

    undirected, directed = _manifold_maxima(F_buf[:nf])
    assert undirected <= 2, "non-manifold edge — pp3d would reject this"
    assert directed <= 1, "inconsistent winding — pp3d would reject this"
    _assert_matches_rebuild(F_buf, adj_buf, nf)
