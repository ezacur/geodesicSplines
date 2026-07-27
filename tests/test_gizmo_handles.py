"""Regression tests for gizmo handle-marker state.

1. **Sphere-handle buffer aliasing**: ``_update_handle``'s update
   branch used ``pd.points = np.ascontiguousarray(buf)``, which is
   zero-copy for an already-contiguous buffer — the p/a/b polydata all
   ended up wrapping the ONE shared ``_handle_pt_buf`` and every
   marker rendered at the last-written handle's position (visible with
   ``ARROW_HANDLES = False``, where A/B also take the sphere path).

2. **``update_magnitude`` stale endpoints**: a failed ``compute_shoot``
   left the previous ``p_a`` / ``p_b`` in place while nulling the
   paths — an invisible but hoverable/draggable marker that also
   leaked into save/undo snapshots.  It must null the endpoints like
   ``update_from_p`` does.
"""
import types

import numpy as np
import pytest

pytest.importorskip("vtk")
pytest.importorskip("pyvista")

from gizmo import GeodesicSegment, SegmentData  # noqa: E402


def _frame():
    origin = np.zeros(3)
    normal = np.array([0.0, 0.0, 1.0])
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    return origin, 0, normal, u, v


class _FakeMapper:
    """Records the offsets ``set_depth_priority`` applies.

    The polygon parameter matters: the handle arrows are cone glyphs
    made entirely of polygons, so without it they get no z-bias at all.
    """

    def __init__(self):
        self.offsets = {}

    def SetResolveCoincidentTopologyToPolygonOffset(self):
        pass

    def SetRelativeCoincidentTopologyPolygonOffsetParameters(self, a, b):
        self.offsets['polygon'] = (a, b)

    def SetRelativeCoincidentTopologyLineOffsetParameters(self, a, b):
        self.offsets['line'] = (a, b)

    def SetRelativeCoincidentTopologyPointOffsetParameter(self, a):
        self.offsets['point'] = a


class _FakeProp:
    def SetColor(self, *a):
        pass

    def SetPointSize(self, s):
        pass

    def SetOpacity(self, o):
        pass


class _FakeActor:
    def __init__(self):
        self._prop = _FakeProp()
        self._mapper = _FakeMapper()

    def GetProperty(self):
        return self._prop

    def GetMapper(self):
        return self._mapper

    def SetVisibility(self, v):
        pass


class _FakePlotter:
    def add_mesh(self, pd, **kw):
        return _FakeActor()


def test_handle_polydata_does_not_alias_shared_buffer():
    seg = GeodesicSegment(*_frame())
    plotter = _FakePlotter()

    pt1 = np.array([1.0, 2.0, 3.0])
    pt2 = np.array([4.0, 5.0, 6.0])
    seg._update_handle(plotter, 'p', pt1, 'red')   # creation branch
    seg._update_handle(plotter, 'p', pt2, 'red')   # update branch

    pd = seg._handle_pd['p']
    np.testing.assert_allclose(np.asarray(pd.points)[0], pt2)

    # Simulate the next handle's write into the SHARED scratch buffer —
    # with the aliasing bug this dragged the polydata along with it.
    seg._handle_pt_buf[0, :] = 99.0
    np.testing.assert_allclose(np.asarray(pd.points)[0], pt2)


def test_update_magnitude_nulls_endpoints_on_failed_shoot():
    seg = SegmentData(*_frame())
    seg.local_v = np.array([1.0, 0.0])
    seg.h_length = 1.0
    # Pre-existing valid handles from an earlier successful shoot.
    seg.p_b = np.array([1.0, 0.0, 0.0])
    seg.p_a = np.array([-1.0, 0.0, 0.0])
    seg.path_b = np.stack([np.zeros(3), seg.p_b])
    seg.path_a = np.stack([np.zeros(3), seg.p_a])

    geo = types.SimpleNamespace(
        compute_shoot=lambda *a, **k: None)   # directional shoot fails

    seg.update_magnitude(np.array([0.5, 0.0, 0.0]), 'b', geo)

    assert seg.path_b is None and seg.path_a is None
    assert seg.p_b is None, "stale p_b must not survive a failed shoot"
    assert seg.p_a is None, "stale p_a must not survive a failed shoot"


def test_update_magnitude_keeps_endpoints_on_success():
    seg = SegmentData(*_frame())
    seg.local_v = np.array([1.0, 0.0])
    seg.h_length = 1.0

    def _shoot(origin, direction, length, face_idx, fast_mode=False):
        return np.stack([origin, origin + direction * length])

    geo = types.SimpleNamespace(compute_shoot=_shoot)

    seg.update_magnitude(np.array([0.5, 0.0, 0.0]), 'b', geo)

    np.testing.assert_allclose(seg.p_b, [0.5, 0.0, 0.0])
    np.testing.assert_allclose(seg.p_a, [-0.5, 0.0, 0.0])
    assert seg.h_length == pytest.approx(0.5)


def test_set_depth_priority_biases_polygons_too():
    """The A / B handle arrows are cone glyphs — pure polygons
    (``polys=9, lines=0, verts=0`` at ``resolution=8``).  Setting only
    the line and point offsets left them with no z-bias at all, so the
    blue / orange / interp curves (line primitives, which *did* get
    their offset) drew over the hovered gizmo — the inverse of the
    documented behaviour."""
    from gizmo import set_depth_priority

    actor = _FakeActor()
    set_depth_priority(actor, -26.0)

    offsets = actor.GetMapper().offsets
    assert offsets['polygon'] == (0, -26.0)
    assert offsets['line'] == (0, -26.0)
    assert offsets['point'] == -26.0
