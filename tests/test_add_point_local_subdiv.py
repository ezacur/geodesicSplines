"""Regression test for the ``_add_point_local`` candidate-seed backstop.

After a 1-to-4 submesh subdivision, ``vmap`` only covers the original
(pre-subdivision) vertices, so ``_try_solve_on_region`` can only ever
hand ``_add_point_local`` an *original corner* as the nearest-vertex
seed ``vi_local``.  A point that lands in a subdivided triangle's
*central* subface — whose three corners are all edge-midpoints — is then
seeded from a corner its containing subface does not touch.  Before the
fix the ``min(outside_score)`` pick selected an adjacent corner subface,
the barycentric coord went strongly negative, and the 2-to-4 edge split
welded the point onto the wrong edge while its 3-D position sat inside
the neighbour — a local mesh fold that silently degrades exactly the
``submesh_subdiv=1`` orange-worker path the subdivision is meant to
improve.

The backstop rescans all faces when the seeded face clearly does not
contain the point, so the insertion targets the true central subface.
"""
import numpy as np
import pytest

pytest.importorskip("vtk")

from geodesics import GeodesicMesh  # noqa: E402


@pytest.fixture
def flat_mesh():
    """Two coplanar triangles sharing edge (v0, v1)."""
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, -1.0, 0.0],
    ], dtype=float)
    F = np.array([[0, 1, 2], [1, 0, 3]], dtype=int)
    return GeodesicMesh(V, F, build_locator=False)


def test_central_subface_point_is_not_folded(flat_mesh):
    gm = flat_mesh

    # Subdivide triangle 0 (and its neighbour) 1-to-4.  Per
    # _subdivide_submesh_1to4, midpoints are appended after the original
    # vertices; triangle 0 = (0,1,2) yields midpoints mab, mbc, mca and a
    # central subface (mab, mbc, mca).
    V_sub, F_sub = np.array(gm.V), np.array(gm.F, dtype=np.int32)
    V_fine, F_fine = gm._subdivide_submesh_1to4(V_sub, F_sub)
    nf = len(F_fine)

    # Locate triangle 0's central subface: the one whose three corners
    # are all >= len(V_sub) (i.e. all edge-midpoints).
    orig_n = len(V_sub)
    central = [fi for fi in range(nf)
               if all(int(v) >= orig_n for v in F_fine[fi])]
    assert len(central) >= 1
    central_face = central[0]
    central_verts = set(int(v) for v in F_fine[central_face])

    # Point at the centroid of that central subface — strictly inside it,
    # bounded only by midpoint vertices.
    p = V_fine[F_fine[central_face]].mean(axis=0)

    extra = 4
    V_buf = np.empty((len(V_fine) + extra, 3), dtype=float)
    V_buf[:len(V_fine)] = V_fine
    F_buf = np.empty((nf + 2 * extra, 3), dtype=np.int32)
    F_buf[:nf] = F_fine
    adj_buf = gm._build_face_adj_buf(F_buf, nf, extra=2 * extra)

    nv0 = len(V_fine)
    # Deliberately mis-seed with an ORIGINAL corner (vertex 0) — exactly
    # what _to_local returns for this point after subdivision.
    idx, nv, nf_after = gm._add_point_local(
        p, V_buf, F_buf, nv0, nf, vi_local=0, nf_original=nf,
        adj_buf=adj_buf)

    # A genuinely-interior point must be inserted as a NEW vertex at its
    # exact position (not snapped to an existing one, not nudged).
    assert idx == nv0
    assert nv == nv0 + 1
    np.testing.assert_allclose(V_buf[idx], p, atol=1e-12)

    # The insertion must be the 1-to-3 interior split of the CENTRAL
    # subface: every face now incident to the new vertex must have its
    # other two corners drawn from the central subface's midpoints.  A
    # fold (2-to-4 on the wrong edge) would connect the new vertex to an
    # original corner instead.
    incident = [fi for fi in range(nf_after)
                if idx in (int(F_buf[fi, 0]), int(F_buf[fi, 1]),
                           int(F_buf[fi, 2]))]
    assert len(incident) == 3
    for fi in incident:
        others = {int(v) for v in F_buf[fi]} - {idx}
        assert others <= central_verts, (
            f"new vertex welded outside the central subface: {others} "
            f"not subset of midpoints {central_verts}")
