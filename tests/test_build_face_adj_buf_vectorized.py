"""Exactness regression for the vectorised ``_build_face_adj_buf``.

``GeodesicMesh._build_face_adj_buf`` was rewritten from a Python
double loop hashing each directed edge through a dict, to a NumPy
lexsort-based pairing (the dict build was ~15 % of
``compute_endpoint_local`` on coarse spans — see
``tests/benchmark_endpoint_local.py``).

The vectorised version must produce **byte-for-byte identical**
adjacency to the old scalar one for every input, including:

  * boundary edges (appear once → stay -1),
  * interior manifold edges (appear twice → symmetric cross-link),
  * the non-manifold case (an edge shared by ≥3 faces), which takes
    the scalar fall-back inside the method — here we assert the
    fall-back is in fact bit-identical to the reference, locking the
    contract even on degenerate input.

This test owns the correctness contract; the benchmark only measures
speed.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("vtk")

from geodesics import GeodesicMesh  # noqa: E402


def _scalar_adj(F_buf, nf, extra):
    """Reference: the original dict-based implementation, verbatim."""
    adj = np.full((nf + extra, 3), -1, dtype=np.int32)
    edge_map = {}
    for fi in range(nf):
        for e in range(3):
            v0 = int(F_buf[fi, e])
            v1 = int(F_buf[fi, (e + 1) % 3])
            key = (v0, v1) if v0 < v1 else (v1, v0)
            if key in edge_map:
                f_other, e_other = edge_map.pop(key)
                adj[fi, e] = f_other
                adj[f_other, e_other] = fi
            else:
                edge_map[key] = (fi, e)
    return adj


def _grid_faces(n):
    """Triangulated (n x n) quad grid → 2*(n-1)^2 faces, many shared edges."""
    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            a = r * n + c
            b = a + 1
            d = a + n
            e = d + 1
            faces.append([a, b, d])
            faces.append([b, e, d])
    return np.array(faces, dtype=np.int32)


# (label, F, extra) cases.
_CASES = {
    "diamond": (np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int32), 2),
    "single": (np.array([[0, 1, 2]], dtype=np.int32), 0),
    "grid3": (_grid_faces(3), 4),
    "grid6": (_grid_faces(6), 8),
    # Non-manifold: edge (0,1) shared by THREE faces → exercises the
    # scalar fall-back branch (run length ≥ 3).
    "nonmanifold": (np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int32), 2),
}


@pytest.mark.parametrize("label", list(_CASES))
def test_vectorized_matches_scalar(label):
    F, extra = _CASES[label]
    nf = len(F)
    # Oversize buffer with stale rows beyond nf, to confirm they're ignored.
    F_buf = np.full((nf + extra, 3), 999, dtype=np.int32)
    F_buf[:nf] = F

    got = GeodesicMesh._build_face_adj_buf(F_buf, nf=nf, extra=extra)
    want = _scalar_adj(F_buf, nf=nf, extra=extra)

    np.testing.assert_array_equal(got, want)
    assert got.shape == (nf + extra, 3)
    assert (got[nf:] == -1).all(), "extra slots must stay -1"


def test_partial_nf_ignores_trailing_faces():
    """When nf < len(F_buf), trailing faces must not affect adjacency."""
    F_buf = np.array([[0, 1, 2], [1, 0, 3], [5, 6, 7]], dtype=np.int32)
    # Only the first two faces are "real"; the third is stale.
    got = GeodesicMesh._build_face_adj_buf(F_buf, nf=2, extra=1)
    want = _scalar_adj(F_buf, nf=2, extra=1)
    np.testing.assert_array_equal(got, want)


def test_empty_mesh():
    F_buf = np.empty((4, 3), dtype=np.int32)
    got = GeodesicMesh._build_face_adj_buf(F_buf, nf=0, extra=4)
    assert got.shape == (4, 3)
    assert (got == -1).all()
