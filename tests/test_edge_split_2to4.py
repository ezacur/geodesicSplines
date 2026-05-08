"""Unit tests for ``GeodesicMesh._split_edge_2to4`` and the
adjacency buffer helper.

The 2-to-4 edge split is the right operation when a point falls on
an edge of the mesh: it preserves the exact position of the point as
a new vertex (no nudge) and rewrites the two incident triangles plus
adds two new ones — manifold by construction.

Coverage:
  * Build face adjacency from a tiny F buffer.
  * Split a shared edge in a flat two-triangle mesh — verify the
    new face count, vertex count, F entries, and adjacency.
  * Boundary edge → split should return None (caller falls back).
  * Inserted vertex equals *p* exactly (no nudge, no projection).
  * Adjacency stays consistent: every entry has a symmetric
    counterpart, and outer neighbours that previously pointed to the
    split faces are re-routed to the new sub-faces.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("vtk")

from geodesics import GeodesicMesh  # noqa: E402


# Diamond mesh used by several tests:
#
#         v2 (0, 1, 0)
#         /\
#        /  \
#  v0 *------* v1   shared edge (v0, v1)
#   (0,0,0) (1,0,0)
#        \  /
#         \/
#         v3 (0.5, -1, 0)
#
# T0 = (v0, v1, v2)  CCW from above
# T1 = (v1, v0, v3)  CCW from above (opposite-direction edge to share with T0)
@pytest.fixture
def diamond_mesh(monkeypatch):
    # Disable Morton reorder for this test so vertex / face indices in
    # the assertions match the V / F arrays we constructed by hand.
    # ``MORTON_REORDER`` is a class-level attribute consulted in
    # ``GeodesicMesh.__init__``.
    monkeypatch.setattr(GeodesicMesh, "MORTON_REORDER", False)
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, -1.0, 0.0],
    ], dtype=float)
    F = np.array([
        [0, 1, 2],
        [1, 0, 3],
    ], dtype=int)
    return GeodesicMesh(V, F, build_locator=False)


def _adj_is_symmetric(adj_buf, nf):
    """Check that every directed-edge adjacency entry has a symmetric
    counterpart in the partner face."""
    for fi in range(nf):
        for e in range(3):
            nb = int(adj_buf[fi, e])
            if nb < 0:
                continue
            # Find the slot in nb that points back to fi.
            found = any(int(adj_buf[nb, k]) == fi for k in range(3))
            if not found:
                return False
    return True


def test_build_face_adj_buf(diamond_mesh):
    gm = diamond_mesh
    F_buf = np.empty((4, 3), dtype=np.int32)
    F_buf[:2] = gm.F
    adj = GeodesicMesh._build_face_adj_buf(F_buf, nf=2, extra=2)
    assert adj.shape == (4, 3)
    # The two triangles share one edge → exactly one cross-link
    cross_links = [(fi, e) for fi in range(2) for e in range(3) if adj[fi, e] >= 0]
    assert len(cross_links) == 2, f"expected exactly 2 adjacency entries, got {cross_links}"
    # Symmetric
    assert _adj_is_symmetric(adj, 2)
    # Extra slots stay -1
    assert (adj[2:] == -1).all()


def test_split_edge_returns_correct_indices(diamond_mesh):
    gm = diamond_mesh
    # Buffers oversized for one split (adds 2 faces, 1 vertex).
    nf0 = len(gm.F)
    F_buf = np.empty((nf0 + 4, 3), dtype=np.int32)
    F_buf[:nf0] = gm.F
    V_buf = np.empty((len(gm.V) + 2, 3), dtype=float)
    V_buf[:len(gm.V)] = gm.V
    adj_buf = GeodesicMesh._build_face_adj_buf(F_buf, nf=nf0, extra=4)

    # Mid-edge point on the shared edge.  v0=(0,0,0), v1=(1,0,0) →
    # midpoint (0.5, 0, 0).
    p = np.array([0.5, 0.0, 0.0])

    # Find face_a = a face containing both v0 and v1.
    fa = next(fi for fi in range(nf0)
              if 0 in F_buf[fi].tolist() and 1 in F_buf[fi].tolist())
    # Find which edge slot of fa is (v0, v1) or (v1, v0).
    for e in range(3):
        v0_e = int(F_buf[fa, e])
        v1_e = int(F_buf[fa, (e + 1) % 3])
        if {v0_e, v1_e} == {0, 1}:
            edge_slot = e
            break

    result = gm._split_edge_2to4(
        p, V_buf, F_buf, adj_buf, nv=len(gm.V), nf=nf0,
        face_a=fa, edge_local_a=edge_slot)
    assert result is not None
    p_idx, new_nv, new_nf = result
    assert new_nv == len(gm.V) + 1
    assert new_nf == nf0 + 2
    assert p_idx == len(gm.V)
    # The inserted vertex equals p exactly — no nudge, no projection.
    np.testing.assert_array_equal(V_buf[p_idx], p)


def test_split_edge_preserves_manifold(diamond_mesh):
    gm = diamond_mesh
    nf0 = len(gm.F)
    F_buf = np.empty((nf0 + 4, 3), dtype=np.int32)
    F_buf[:nf0] = gm.F
    V_buf = np.empty((len(gm.V) + 2, 3), dtype=float)
    V_buf[:len(gm.V)] = gm.V
    adj_buf = GeodesicMesh._build_face_adj_buf(F_buf, nf=nf0, extra=4)

    p = np.array([0.3, 0.0, 0.0])  # off-centre on shared edge
    fa = next(fi for fi in range(nf0)
              if 0 in F_buf[fi].tolist() and 1 in F_buf[fi].tolist())
    for e in range(3):
        v0_e = int(F_buf[fa, e]); v1_e = int(F_buf[fa, (e + 1) % 3])
        if {v0_e, v1_e} == {0, 1}:
            edge_slot = e
            break

    p_idx, _, new_nf = gm._split_edge_2to4(
        p, V_buf, F_buf, adj_buf, nv=len(gm.V), nf=nf0,
        face_a=fa, edge_local_a=edge_slot)

    # Adjacency must remain symmetric across all 4 faces.
    assert _adj_is_symmetric(adj_buf, new_nf), \
        "adjacency lost symmetry after 2-to-4 split"

    # The inserted vertex must appear in exactly 4 of the new faces
    # (the 2 sub-faces of T0 + the 2 sub-faces of T1).
    faces_with_p = sum(1 for fi in range(new_nf) if p_idx in F_buf[fi].tolist())
    assert faces_with_p == 4

    # Every face now has 3 distinct vertex indices (no degenerate).
    for fi in range(new_nf):
        f = F_buf[fi]
        assert len({int(f[0]), int(f[1]), int(f[2])}) == 3, \
            f"face {fi} = {f} is degenerate"

    # Total area is preserved (no overlap, no gap).  The original
    # diamond has area 1/2 + 1/2 = 1.0 (two triangles each of base
    # 1.0 and height 1.0).
    total_area = 0.0
    for fi in range(new_nf):
        tri = V_buf[F_buf[fi]]
        total_area += 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    np.testing.assert_allclose(total_area, 1.0, rtol=1e-12)


def test_split_edge_boundary_returns_none(diamond_mesh):
    """Trying to split an edge that has no neighbour (boundary edge)
    must return None so the caller can fall back."""
    gm = diamond_mesh
    nf0 = len(gm.F)
    F_buf = np.empty((nf0 + 4, 3), dtype=np.int32)
    F_buf[:nf0] = gm.F
    V_buf = np.empty((len(gm.V) + 2, 3), dtype=float)
    V_buf[:len(gm.V)] = gm.V
    adj_buf = GeodesicMesh._build_face_adj_buf(F_buf, nf=nf0, extra=4)

    # Pick face 0 and an edge that does NOT face the other triangle —
    # i.e., a boundary edge of T0.
    fa = 0
    boundary_slot = next(e for e in range(3) if int(adj_buf[fa, e]) < 0)
    Va = int(F_buf[fa, boundary_slot])
    Vb = int(F_buf[fa, (boundary_slot + 1) % 3])
    p = (gm.V[Va] + gm.V[Vb]) * 0.5  # midpoint of the boundary edge

    result = gm._split_edge_2to4(
        p, V_buf, F_buf, adj_buf, nv=len(gm.V), nf=nf0,
        face_a=fa, edge_local_a=boundary_slot)
    assert result is None
