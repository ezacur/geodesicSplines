"""Unit tests for the phase-3 degraded-flag propagation.

``_build_chord_geodesic`` is the orange worker's phase-3 chord bridge.
Historically it silently returned a straight Euclidean segment when
both solvers failed, so the span rendered without the red fallback
repaint — exactly the "phantom curve" the editor promises never to
show.  The contract now is:

  - solvable chord   → ``(geodesic polyline, False)``
  - unsolvable chord → ``([p_left, p_right], True)``

and ``_phase3_chord_bridge`` ORs the per-chord flags into a single
bool that the worker folds into the ``('done', ...)`` degraded flag.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("vtk")
pytest.importorskip("pyvista")

import span_workers  # noqa: E402
from geodesics import GeodesicMesh  # noqa: E402
from span_workers import (  # noqa: E402
    _build_chord_geodesic,
    _geodesic_decasteljau_worker,
    _hierarchical_inner_order,
    _phase3_chord_bridge,
)


@pytest.fixture
def two_triangle_mesh():
    """Flat two-triangle mesh in the XY plane sharing edge (v0, v1)."""
    V = np.array([
        [0.0, 0.0, 0.0],   # v0
        [1.0, 0.0, 0.0],   # v1
        [0.0, 1.0, 0.0],   # v2
        [0.5, -1.0, 0.0],  # v3
    ], dtype=float)
    F = np.array([
        [0, 1, 2],
        [1, 0, 3],
    ], dtype=int)
    return GeodesicMesh(V, F, build_locator=False)


@pytest.fixture
def disconnected_mesh():
    """Two triangles in separate connected components, 100 units apart."""
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [100.0, 0.0, 0.0],
        [101.0, 0.0, 0.0],
        [100.0, 1.0, 0.0],
    ], dtype=float)
    F = np.array([
        [0, 1, 2],
        [3, 4, 5],
    ], dtype=int)
    return GeodesicMesh(V, F, build_locator=False)


class _CollectingWriter:
    """Stub for the worker's pipe write-end: records every send()."""

    def __init__(self):
        self.msgs = []
        self.closed = False

    def send(self, msg):
        self.msgs.append(msg)

    def close(self):
        self.closed = True


def test_solvable_chord_is_not_degraded(two_triangle_mesh):
    gm = two_triangle_mesh
    p0 = np.array([0.2, 0.3, 0.0])
    p1 = np.array([0.4, 0.4, 0.0])
    seg, degraded = _build_chord_geodesic(gm, p0, p1)
    assert degraded is False
    assert len(seg) >= 2
    np.testing.assert_allclose(seg[0], p0, atol=1e-9)
    np.testing.assert_allclose(seg[-1], p1, atol=1e-9)


def test_cross_component_chord_is_degraded(disconnected_mesh):
    gm = disconnected_mesh
    p_left = gm.V[[0, 1, 2]].mean(axis=0)   # centroid, component 0
    p_right = gm.V[[3, 4, 5]].mean(axis=0)  # centroid, component 1
    seg, degraded = _build_chord_geodesic(gm, p_left, p_right)
    assert degraded is True
    # Last-resort geometry: the plain 2-point Euclidean stand-in.
    assert len(seg) == 2
    np.testing.assert_allclose(seg[0], p_left, atol=1e-9)
    np.testing.assert_allclose(seg[1], p_right, atol=1e-9)


def test_phase3_returns_false_when_all_chords_solve(two_triangle_mesh):
    gm = two_triangle_mesh
    p_list = [
        np.array([0.2, 0.3, 0.0]),
        np.array([0.4, 0.4, 0.0]),
        np.array([0.3, 0.5, 0.0]),
    ]
    writer = _CollectingWriter()
    degraded = _phase3_chord_bridge(gm, (0, 0), p_list, writer,
                                    submesh_subdiv=0)
    assert degraded is False
    assert len(writer.msgs) == 1
    kind, span_key, polyline = writer.msgs[0]
    assert kind == 'chord_geo'
    assert span_key == (0, 0)
    assert len(polyline) >= len(p_list)


def test_phase3_propagates_degraded_chord(disconnected_mesh):
    gm = disconnected_mesh
    p_list = [
        gm.V[[0, 1, 2]].mean(axis=0),
        gm.V[[3, 4, 5]].mean(axis=0),
    ]
    writer = _CollectingWriter()
    degraded = _phase3_chord_bridge(gm, (0, 0), p_list, writer,
                                    submesh_subdiv=0)
    assert degraded is True
    # The polyline is still sent — degraded geometry renders, but the
    # flag lets the parent repaint the span red.
    assert len(writer.msgs) == 1
    assert writer.msgs[0][0] == 'chord_geo'


def test_build_chord_flags_solver_fallback(two_triangle_mesh, monkeypatch):
    """Degraded is True when the solver *succeeds* but reports its own
    straight-line stub (``was_fallback=True``).

    The disconnected-mesh test above only covers the solver-*raises* /
    returns-nothing path.  Here the fast path misses and
    ``compute_endpoint_local`` returns a usable 2-point polyline while
    flagging it as a fallback — ``_build_chord_geodesic`` must still
    mark the chord degraded and return that polyline verbatim.
    """
    gm = two_triangle_mesh
    p0 = np.array([0.2, 0.3, 0.0])
    p1 = np.array([0.4, 0.4, 0.0])
    stub = np.stack([p0, p1])

    monkeypatch.setattr(gm, "short_geodesic", lambda *a, **k: None)
    monkeypatch.setattr(
        gm, "compute_endpoint_local", lambda *a, **k: (stub, True))

    seg, degraded = _build_chord_geodesic(gm, p0, p1)
    assert degraded is True
    np.testing.assert_array_equal(seg, stub)


def _flat_span_inputs():
    """Four coplanar control points inside triangle [v0,v1,v2] plus the
    two outer handle paths, as fed to ``_geodesic_decasteljau_worker``."""
    P0 = np.array([0.20, 0.20, 0.0])
    H_out = np.array([0.35, 0.25, 0.0])
    H_in = np.array([0.50, 0.30, 0.0])
    P1 = np.array([0.65, 0.20, 0.0])
    ctrl = [P0, H_out, H_in, P1]
    path_b = np.stack([P0, H_out])          # node-0 B handle path
    path_a_rev = np.stack([H_in, P1])       # node-1 A handle path, reversed
    return ctrl, path_b, path_a_rev


def test_worker_reports_clean_done_on_flat_mesh(two_triangle_mesh, monkeypatch):
    """End-to-end: a fully solvable span on a flat mesh finishes with
    ``('done', span_key, False)`` — phase 3 does not spuriously flag."""
    monkeypatch.setattr(span_workers, "_process_geo", two_triangle_mesh)
    ctrl, path_b, path_a_rev = _flat_span_inputs()
    t_grid = np.linspace(0.0, 1.0, 5)
    writer = _CollectingWriter()

    _geodesic_decasteljau_worker(
        (0, 0), ctrl, path_b, path_a_rev, t_grid,
        _hierarchical_inner_order(5), writer)

    assert writer.closed is True
    done = [m for m in writer.msgs if m[0] == 'done']
    assert len(done) == 1
    assert done[0] == ('done', (0, 0), False)
    assert not any(m[0] == 'error' for m in writer.msgs)


def test_worker_folds_phase3_degraded_into_done(two_triangle_mesh, monkeypatch):
    """The worker ORs ``_phase3_chord_bridge``'s return into the final
    ``degraded_any``.  Force phase 3 to report a degraded chord while
    phases 1-2 run clean on the flat mesh; ``'done'`` must be True."""
    monkeypatch.setattr(span_workers, "_process_geo", two_triangle_mesh)

    def _fake_phase3(geo, span_key, p_list, writer, **kwargs):
        writer.send(('chord_geo', span_key, np.stack(p_list)))
        return True

    monkeypatch.setattr(span_workers, "_phase3_chord_bridge", _fake_phase3)

    ctrl, path_b, path_a_rev = _flat_span_inputs()
    t_grid = np.linspace(0.0, 1.0, 5)
    writer = _CollectingWriter()

    _geodesic_decasteljau_worker(
        (0, 0), ctrl, path_b, path_a_rev, t_grid,
        _hierarchical_inner_order(5), writer)

    done = [m for m in writer.msgs if m[0] == 'done']
    assert len(done) == 1
    assert done[0] == ('done', (0, 0), True)
