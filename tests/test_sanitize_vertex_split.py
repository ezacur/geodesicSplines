"""Unit tests for ``_sanitize_for_solver`` pass 5 (non-manifold vertices)
and its vectorised detector ``GeodesicMesh._corner_fan_labels``.

Pass 5 splits "pinch point" vertices — a vertex whose incident-face fan
falls apart into several edge-connected components, which
geometry-central rejects with ``vertex N appears in more than one
boundary loop``.  Detection is one ``connected_components`` call on the
corner graph; the Python repair loop then runs only over the vertices
that actually split.

Coverage:
  * Clean meshes (open grid, closed sphere) → no split, no faces dropped.
  * Bowtie (two *open* fans at one vertex) → 1 split.
  * Double cone (two *closed* fans at one apex) → 1 split.  This is the
    case a purely arithmetic ``#faces vs #edges`` test cannot see, so it
    is what forces the connected-components formulation.
  * Closed fan + open fan at the same vertex → 1 split.
  * Three fans at one vertex → 2 splits.
  * Geometry is preserved: every appended vertex duplicates the position
    of the vertex it was split off.
  * After the split no vertex has a multi-component fan any more.
  * Idempotence — the property ``GeodesicMesh(..., sanitize=False)``
    relies on, since the orange workers rebuild from the parent's
    already-sanitised ``geo.V`` / ``geo.F``.
  * ``sanitize=False`` yields a bit-identical mesh to a full build when
    the input is already sanitised.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("vtk")

from geodesics import GeodesicMesh  # noqa: E402

# --------------------------------------------------------------- helpers

def _grid(n, seed=0):
    """(n x n) vertex grid triangulated into 2(n-1)^2 faces."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    V = np.column_stack([xs.ravel(), ys.ravel(),
                         rng.normal(0, 0.05, n * n)]).astype(float)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            F += [[a, b, d], [a, d, c]]
    return V, np.asarray(F, dtype=np.int32)


def _ring(z, k=6):
    return [[np.cos(2 * np.pi * i / k), np.sin(2 * np.pi * i / k), z]
            for i in range(k)]


def _fan_component_counts(F):
    """Number of edge-connected fan components per vertex."""
    labels = GeodesicMesh._corner_fan_labels(F)
    counts = {}
    for corner, v in enumerate(F.ravel()):
        counts.setdefault(int(v), set()).add(int(labels[corner]))
    return {v: len(s) for v, s in counts.items()}


# ----------------------------------------------------------- clean input

@pytest.mark.parametrize("n", [4, 9])
def test_clean_grid_needs_no_repair(n):
    V, F = _grid(n)
    V_out, F_out, report = GeodesicMesh._sanitize_for_solver(V, F)

    assert report['vertex_splits'] == 0
    assert report['total_faces_dropped'] == 0
    assert report['unreferenced_verts'] == 0
    np.testing.assert_array_equal(F_out, F)
    np.testing.assert_array_equal(V_out, V)


def test_clean_closed_sphere_needs_no_repair():
    pv = pytest.importorskip("pyvista")
    sph = pv.Sphere(theta_resolution=16, phi_resolution=16).triangulate()
    V = np.asarray(sph.points, dtype=float)
    F = np.asarray(sph.faces).reshape(-1, 4)[:, 1:].astype(np.int32)

    _, F_out, report = GeodesicMesh._sanitize_for_solver(V, F)

    assert report['vertex_splits'] == 0
    # A closed sphere is one cycle fan per vertex — the case that would
    # be misread as "needs split" if cycles were counted as extra
    # components.
    assert set(_fan_component_counts(F_out).values()) == {1}


# ------------------------------------------------------ pinch geometries

def test_bowtie_two_open_fans_splits_once():
    """Two triangle pairs meeting at exactly one vertex."""
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [-1, 0, 0], [-1, -1, 0], [0, -1, 0]], dtype=float)
    F = np.array([[0, 1, 2], [0, 2, 3], [0, 4, 5], [0, 5, 6]], dtype=np.int32)

    assert _fan_component_counts(F)[0] == 2

    V_out, F_out, report = GeodesicMesh._sanitize_for_solver(V, F)

    assert report['vertex_splits'] == 1
    assert report['total_faces_dropped'] == 0
    assert len(V_out) == len(V) + 1
    # The duplicate sits exactly on top of the original apex.
    np.testing.assert_array_equal(V_out[-1], V[0])
    assert max(_fan_component_counts(F_out).values()) == 1


def test_double_cone_two_closed_fans_splits_once():
    """Two *closed* fans sharing an apex.

    Both components are cycles, so the vertex has the same face count as
    edge count and no arithmetic shortcut can distinguish it from a
    single closed fan — only real connectivity does.
    """
    k = 6
    V = np.asarray([[0, 0, 0]] + _ring(1.0, k) + _ring(-1.0, k), dtype=float)
    F = [[0, 1 + i, 1 + (i + 1) % k] for i in range(k)]
    F += [[0, 1 + k + i, 1 + k + (i + 1) % k] for i in range(k)]
    F = np.asarray(F, dtype=np.int32)

    assert _fan_component_counts(F)[0] == 2

    V_out, F_out, report = GeodesicMesh._sanitize_for_solver(V, F)

    assert report['vertex_splits'] == 1
    assert len(V_out) == len(V) + 1
    np.testing.assert_array_equal(V_out[-1], V[0])
    assert max(_fan_component_counts(F_out).values()) == 1


def test_closed_fan_plus_open_fan_splits_once():
    k = 6
    V = np.asarray([[0, 0, 0]] + _ring(1.0, k) + [[2, 0, -1], [2, 1, -1]],
                   dtype=float)
    F = [[0, 1 + i, 1 + (i + 1) % k] for i in range(k)]
    F.append([0, 1 + k, 2 + k])
    F = np.asarray(F, dtype=np.int32)

    assert _fan_component_counts(F)[0] == 2

    _, F_out, report = GeodesicMesh._sanitize_for_solver(V, F)

    assert report['vertex_splits'] == 1
    assert max(_fan_component_counts(F_out).values()) == 1


def test_three_fans_at_one_vertex_splits_twice():
    V = [[0, 0, 0]]
    F = []
    for blk in range(3):
        base = len(V)
        V += [[blk * 3 + 1, 0, 0], [blk * 3 + 1, 1, 0], [blk * 3 + 2, 1, 0]]
        F += [[0, base, base + 1], [0, base + 1, base + 2]]
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=np.int32)

    assert _fan_component_counts(F)[0] == 3

    V_out, F_out, report = GeodesicMesh._sanitize_for_solver(V, F)

    assert report['vertex_splits'] == 2
    assert len(V_out) == len(V) + 2
    for extra in V_out[len(V):]:
        np.testing.assert_array_equal(extra, V[0])
    assert max(_fan_component_counts(F_out).values()) == 1


# ------------------------------------------------------------ properties

def test_split_preserves_face_count_and_triangle_positions():
    """A vertex split rewrites indices only — no face is added or lost,
    and every triangle keeps its three 3-D corner positions."""
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [-1, 0, 0], [-1, -1, 0], [0, -1, 0]], dtype=float)
    F = np.array([[0, 1, 2], [0, 2, 3], [0, 4, 5], [0, 5, 6]], dtype=np.int32)

    V_out, F_out, report = GeodesicMesh._sanitize_for_solver(V, F)

    assert report['vertex_splits'] == 1
    assert len(F_out) == len(F)
    np.testing.assert_allclose(V_out[F_out], V[F])


@pytest.mark.parametrize("seed", range(8))
def test_sanitise_is_idempotent_on_dirty_meshes(seed):
    """``GeodesicMesh(..., sanitize=False)`` for the orange workers is
    only sound because a second pass provably finds nothing."""
    rng = np.random.default_rng(1000 + seed)
    V, F = _grid(6, seed=seed)
    F = np.vstack([F, rng.integers(0, len(V), size=(rng.integers(1, 8), 3))])
    F = F[rng.random(len(F)) > 0.12].astype(np.int32)
    if len(F) < 2:
        pytest.skip("degenerate draw")

    V1, F1, _ = GeodesicMesh._sanitize_for_solver(V, F)
    V2, F2, report2 = GeodesicMesh._sanitize_for_solver(V1, F1)

    np.testing.assert_array_equal(V1, V2)
    np.testing.assert_array_equal(F1, F2)
    assert report2['total_faces_dropped'] == 0
    assert report2['vertex_splits'] == 0
    assert report2['unreferenced_verts'] == 0


def test_sanitize_false_matches_full_build_on_clean_input():
    """The worker path (``sanitize=False``) must reproduce the parent
    mesh bit-for-bit when handed already-sanitised V / F."""
    V, F = _grid(9)
    full = GeodesicMesh(V, F, build_locator=False)

    worker = GeodesicMesh(full.V.copy(), full.F.copy(),
                          build_locator=False, sanitize=False)
    again = GeodesicMesh(full.V.copy(), full.F.copy(), build_locator=False)

    np.testing.assert_array_equal(worker.V, again.V)
    np.testing.assert_array_equal(worker.F, again.F)


# ----------------------------------------------------- detector contract

def test_corner_fan_labels_shape_and_vertex_locality():
    """Labels are per corner and never span two vertices."""
    V, F = _grid(5)
    labels = GeodesicMesh._corner_fan_labels(F)

    assert labels.shape == (3 * len(F),)
    by_label = {}
    for corner, v in enumerate(F.ravel()):
        by_label.setdefault(int(labels[corner]), set()).add(int(v))
    assert all(len(vs) == 1 for vs in by_label.values())


def test_corner_fan_labels_empty_mesh():
    labels = GeodesicMesh._corner_fan_labels(np.empty((0, 3), dtype=np.int32))
    assert labels.shape == (0,)
