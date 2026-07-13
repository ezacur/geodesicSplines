"""Characterisation tests for ``GeodesicMesh.compute_shoot``.

The geodesic-shooting path (``compute_shoot`` → ``_shoot_loop`` →
``_ray_edge_jit`` + ``_parallel_transport``) had no direct unit test,
yet it is the kernel that builds every handle ray.  These tests pin two
guarantees:

* On a **flat** mesh a geodesic is a straight segment, so the shot
  endpoint is ``p_start + length·d̂`` and the polyline arc-length equals
  the requested length — even when the ray crosses an internal edge.
* The result is **invariant to mesh scale** when scaling up: shooting
  on a mesh scaled by *k ≥ 1* (with the length scaled by *k*) yields the
  same path scaled by *k*.  This is the property the ``_ray_edge_jit``
  parallel-reject tolerance must preserve; it regressed silently when
  that tolerance scaled with ``edge_len²`` instead of ``edge_len``.

  Note the *sub-unit* regime (*k ≪ 1*) is only invariant down to a
  floor set by the shooting loop's **absolute** constants — the 1e-7
  edge-crossing nudge, ``s_tol = 1e-4``, ``t_min = 1e-8`` — which do not
  scale with the mesh.  ``test_shoot_sub_unit_scale_floor`` characterises
  that floor rather than pretending it is zero.
"""
import numpy as np
import pytest

pytest.importorskip("vtk")

from geodesics import GeodesicMesh  # noqa: E402


def _two_triangle_mesh(scale=1.0):
    """Two coplanar triangles in the XY plane sharing edge (v0, v1) on
    the x-axis; triangle 0 is above it, triangle 1 below."""
    V = np.array([
        [0.0, 0.0, 0.0],   # v0
        [1.0, 0.0, 0.0],   # v1
        [0.0, 1.0, 0.0],   # v2  (upper triangle [0,1,2])
        [0.5, -1.0, 0.0],  # v3  (lower triangle [1,0,3])
    ], dtype=float) * scale
    F = np.array([[0, 1, 2], [1, 0, 3]], dtype=int)
    return GeodesicMesh(V, F, build_locator=False)


def _arc_length(path):
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def test_shoot_straight_across_internal_edge():
    gm = _two_triangle_mesh()
    p0 = np.array([0.3, 0.3, 0.0])
    d = np.array([0.0, -1.0, 0.0])   # crosses the shared edge at y=0
    length = 0.8

    path = gm.compute_shoot(p0, d, length, face_idx=0)
    assert path is not None
    assert len(path) >= 3               # start, edge crossing, end

    np.testing.assert_allclose(path[0], p0, atol=1e-9)
    np.testing.assert_allclose(path[-1], [0.3, -0.5, 0.0], atol=1e-7)
    # A crossing vertex must sit exactly on the shared edge (y == 0).
    assert np.any(np.abs(path[1:-1, 1]) < 1e-7)
    assert _arc_length(path) == pytest.approx(length, abs=1e-7)


@pytest.mark.parametrize("k", [1.0, 1e2, 1e4, 1e6])
def test_shoot_is_scale_invariant_scaling_up(k):
    p0 = np.array([0.3, 0.3, 0.0])
    d = np.array([0.0, -1.0, 0.0])
    length = 0.8

    base = _two_triangle_mesh(scale=1.0).compute_shoot(
        p0, d, length, face_idx=0)
    scaled = _two_triangle_mesh(scale=k).compute_shoot(
        p0 * k, d, length * k, face_idx=0)

    assert base is not None and scaled is not None
    assert len(scaled) == len(base)
    # Same path geometry, just uniformly scaled by k.  Endpoints agree
    # to ~1e-7 mesh units — the base mesh's own edge-crossing nudge,
    # applied in its own units and hence not removed by scaled/k.  Still
    # far tighter than any structural change (a wrong / missed edge
    # crossing would shift the path by O(0.1)), so this pins the det_tol
    # behaviour without over-claiming float-eps invariance.
    np.testing.assert_allclose(scaled / k, base, rtol=1e-6, atol=1e-6)


def test_shoot_sub_unit_scale_floor():
    """Sub-unit meshes stay invariant only down to the ~1e-7 mesh-unit
    floor set by the un-scaled edge-crossing nudge — not to float eps.

    This is a *characterisation* of the known limitation (see finding
    3.2 in the review): the ``det_tol`` fix makes the parallel-reject
    scale-invariant, but the nudge / ``s_tol`` / ``t_min`` constants are
    still absolute.  Tighten this test only alongside a fix that scales
    those constants to the mesh.
    """
    k = 1e-3
    p0 = np.array([0.3, 0.3, 0.0])
    d = np.array([0.0, -1.0, 0.0])
    length = 0.8

    base = _two_triangle_mesh(scale=1.0).compute_shoot(
        p0, d, length, face_idx=0)
    scaled = _two_triangle_mesh(scale=k).compute_shoot(
        p0 * k, d, length * k, face_idx=0)

    assert base is not None and scaled is not None
    # Endpoint agrees only to the nudge floor: ~1e-7 (mesh units) / k.
    err = float(np.abs(scaled[-1] / k - base[-1]).max())
    assert err < 1e-3            # not float eps — the nudge floor
    assert err > 1e-6            # and it really is non-trivial here
