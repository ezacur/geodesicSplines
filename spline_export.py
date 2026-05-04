# SPDX-License-Identifier: Apache-2.0
"""
spline_export.py — Command-line exporter for geodesic spline curves.

Reads a spline JSON file (saved by geo_splines.py) and outputs the
curve points for a selected layer (blue/orange/interp) to stdout.

Usage::

    python spline_export.py <splines.json> <b|o|k> [--samples N]

Output format (CSV, one point per line):

    x , y , z          — curve point
    NaN , NaN , NaN    — break between splines
    NaN , NaN , NaN
    x , y , z          — landmark (node origin)
    NaN , NaN , NaN

Landmarks are node origins wrapped in NaN sentinels so downstream
tools can distinguish them from curve points.

Layers
------
  - **b** (blue): semi-geodesic Bézier — level-1 uses the exact geodesic
    between H_out and H_in (via ``compute_endpoint_local``); levels 2-3
    Euclidean + projection.  ~seconds per spline on a 240K-face mesh.
  - **o** (orange): fully geodesic de Casteljau — geodesic interpolation
    at every level.  ~minutes to hours on large meshes.
  - **k** (black): interpolation B-spline through node origins (scipy
    splprep/splev), projected onto the surface.  Fastest; no handles.

Examples::

    python spline_export.py 20260414_153022.json b > blue_curve.csv
    python spline_export.py 20260414_153022.json o --samples 50
    python spline_export.py 20260414_153022.json k > interp_curve.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from geodesics import GeodesicMesh


NAN_LINE = "NaN , NaN , NaN"

# Diagnostics on stderr.  Aligned with geo_splines.log so users see the
# same "[LEVEL] module: msg" prefix across both tools and a single
# environment variable controls verbosity.  CSV output stays on stdout.
log = logging.getLogger("spline_export")
if not log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(_h)
    log.propagate = False
log.setLevel(logging.DEBUG if os.environ.get("GEO_SPLINES_DEBUG") else logging.INFO)


def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _read_mesh_VF(mesh_file: str) -> tuple[np.ndarray, np.ndarray]:
    """Reads ``mesh_file`` and returns ``(V, F)`` as plain numpy arrays.

    Uses the same pipeline as the interactive editor
    (``geo_shoot.py:_load_mesh`` / [geo_shoot.py:698]):

        pv.read(path).extract_surface().triangulate().clean()

    This guarantees byte-for-byte parity with what ``geo_splines`` sees
    when it loads the same file: the same V / F arrays, same
    deduplication of coincident vertices, same removal of degenerate
    triangles.  Without this parity, the orange worker downstream
    builds its face-adjacency matrix on slightly different topology
    (duplicate vertices break edge-key matching) and ``compute_shoot``
    truncates short of where it should — producing a visibly **shorter
    curve** than the editor displays.

    A previous version of this function had a "meshio fast path" for
    ``.obj`` / ``.ply`` / ``.stl`` to avoid the PyVista import in
    headless CI, but meshio does not deduplicate vertices or remove
    degenerate triangles, and the geometry mismatch silently shifted
    the orange curve.  Parity with the editor is more important than
    a one-time PyVista import cost (~1 s); ``pv.read`` itself does
    not require a display, so this is still safe in offscreen
    contexts (``Plotter()`` is the only PyVista API that needs X).

    Built-in icosahedron sentinel is handled by the caller — this
    helper deals only with on-disk meshes.
    """
    import warnings
    import pyvista as pv
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        mesh_pv = pv.read(mesh_file).extract_surface().triangulate().clean()
    V = np.asarray(mesh_pv.points, dtype=float)
    # PyVista stores faces as flat [n, i0, i1, ..., n, i0, ...] — for an
    # all-triangle mesh after .triangulate() this is [3, a, b, c, 3, ...].
    F = np.asarray(mesh_pv.faces, dtype=int).reshape(-1, 4)[:, 1:]
    return V, F


def rebuild_mesh_and_nodes(data: dict):
    """Rebuilds GeodesicMesh and node data from JSON.

    Two schemas are accepted, dispatched per-node by which keys are
    present (matches ``geo_splines._apply_record_to_node``):

      • **v2** (preferred): ``{origin, p_a, p_b}`` — handle endpoints
        as literal 3-D positions.  Reconstructed via the same
        ``compute_endpoint_from_origin`` (EdgeFlipGeodesicSolver) call
        the editor uses during drag, so the geodesic on reload is
        identical to what the user saw on screen at save time.
      • **v1** (legacy): ``{origin, tangent}`` — direction × h_length.
        Reconstructed via ``compute_shoot`` ± tangent_dir; loses
        solver-curving information on curved surfaces (this is the
        bug v2 was introduced to fix).

    Returns ``(geo, splines, splines_closed)`` where *splines* is a list
    of lists of dicts with 'origin', 'face_idx', 'p_a', 'p_b', 'path_a',
    'path_b'.
    """
    mesh_file = data['mesh_file']
    log.info("loading mesh: %s", mesh_file)
    # Both the prefixed sentinel ("__builtin__:icosahedron") and the
    # legacy plain string ("ICOSAHEDRON") map to the in-memory demo
    # mesh.  This branch is rare (demo only); the lazy import of
    # ``_make_icosahedron`` keeps PyVista out of the default
    # mesh-file path that runs in CI and headless containers.
    if mesh_file in ("__builtin__:icosahedron", "ICOSAHEDRON"):
        from geo_splines import _make_icosahedron
        mesh = _make_icosahedron(radius=10.0)
        # Extract V, F from the pv.PolyData so GeodesicMesh receives
        # plain arrays (no locator built — fine, we don't pick).
        V = np.asarray(mesh.points, dtype=float)
        F = np.asarray(mesh.faces, dtype=int).reshape(-1, 4)[:, 1:]
    else:
        V, F = _read_mesh_VF(mesh_file)
    geo = GeodesicMesh(V, F)

    def _build_node_v2(nd, origin):
        """v2 schema: handles persisted as literal 3-D positions.

        face_idx isn't needed here — ``compute_endpoint_from_origin``
        works off the origin_cache (the solver's own per-origin
        topology insertion), not a starting face index.
        """
        p_a_rec = nd.get('p_a')
        p_b_rec = nd.get('p_b')
        try:
            cache = geo.prepare_origin(origin)
        except (RuntimeError, ValueError, TypeError) as exc:
            log.warning("v2 load: prepare_origin failed at %s (%s); "
                        "node will have null paths", origin.tolist(), exc)
            cache = None

        def _resolve(p_rec):
            if p_rec is None or cache is None:
                return None, None
            p_target = np.asarray(p_rec, dtype=float)
            try:
                path = geo.compute_endpoint_from_origin(cache, p_target)
            except (RuntimeError, ValueError, TypeError, IndexError) as exc:
                log.debug("v2 load: solver failed for handle %s (%s)",
                          p_target.tolist(), exc)
                path = np.array([origin, p_target])
            if path is None or len(path) < 2:
                path = np.array([origin, p_target])
            return path, path[-1]

        path_a, p_a = _resolve(p_a_rec)
        path_b, p_b = _resolve(p_b_rec)
        return path_a, path_b, p_a, p_b

    def _build_node_v1(nd, origin, face_idx):
        """v1 schema: tangent vector → compute_shoot ± direction."""
        tangent_full = np.array(nd['tangent'], dtype=float)
        h_length = float(np.linalg.norm(tangent_full))
        if h_length > 1e-15:
            tangent_dir = tangent_full / h_length
        else:
            tangent_dir = np.array([1.0, 0.0, 0.0])
            h_length = 0.01
        path_b = geo.compute_shoot(origin, tangent_dir, h_length, face_idx)
        path_a = geo.compute_shoot(origin, -tangent_dir, h_length, face_idx)
        p_b = path_b[-1] if path_b is not None else None
        p_a = path_a[-1] if path_a is not None else None
        return path_a, path_b, p_a, p_b

    splines = []
    splines_closed = []
    for sd in data['splines']:
        nodes = []
        for nd in sd['nodes']:
            origin = np.array(nd['origin'], dtype=float)
            face_idx = geo.find_face(origin)
            # Per-node schema dispatch — same logic as the editor's
            # _apply_record_to_node.
            if 'p_a' in nd and 'p_b' in nd:
                path_a, path_b, p_a, p_b = _build_node_v2(nd, origin)
            else:
                path_a, path_b, p_a, p_b = _build_node_v1(nd, origin, face_idx)

            nodes.append({
                'origin': origin, 'face_idx': face_idx,
                'p_a': p_a, 'p_b': p_b,
                'path_a': path_a, 'path_b': path_b,
            })
        splines.append(nodes)
        splines_closed.append(bool(sd.get('closed', False)))

    return geo, splines, splines_closed


def compute_blue(geo, nodes, closed, n_samples):
    """Computes semi-geodesic Bézier (blue) curve points for one spline.

    Matches the interactive app's consolidated blue: level-1 geodesic lerp
    on all three control segments (including H_out→H_in via
    ``compute_endpoint_local``); levels 2-3 Euclidean + projection.
    """
    all_pts = []
    n_nodes = len(nodes)
    if n_nodes < 2:
        return all_pts

    n_spans = n_nodes if closed else n_nodes - 1
    for i in range(n_spans):
        n0 = nodes[i]
        n1 = nodes[(i + 1) % n_nodes]
        ctrl = [n0['origin'], n0['p_b'], n1['p_a'], n1['origin']]
        if any(p is None for p in ctrl):
            continue
        path_b = n0['path_b']
        path_a_rev = n1['path_a'][::-1] if n1['path_a'] is not None else None

        # Geodesic H_out → H_in via local submesh solver
        log.debug("span %d: computing path_12 (H_out -> H_in)", i)
        path_12 = geo.compute_endpoint_local(n0['p_b'], n1['p_a'])
        if path_12 is None or len(path_12) < 2:
            path_12 = None

        n = geo.adaptive_samples(ctrl, 0.3, 15, 100)
        n = max(n, n_samples)
        pts = geo.hybrid_de_casteljau_curve(
            ctrl, path_b, path_a_rev, n, fast=False, path_12=path_12)
        pts = geo.project_smooth_batch(pts)
        all_pts.append(pts)

    return all_pts


def _orange_worker_init() -> None:
    """ProcessPoolExecutor initializer: blocks SIGINT in worker children.

    On Ctrl+C the OS sends SIGINT to the parent and every child in the
    process group.  Without this guard, each worker would interrupt
    its in-flight scipy / Intel-MKL Fortran call and the runtime would
    dump ``forrtl: error (200): program interrupted`` to stderr — with
    several workers writing concurrently the output became unreadable.

    Ignoring SIGINT in the children leaves the parent's
    ``KeyboardInterrupt`` handler the sole graceful-exit path:
    ``with ProcessPoolExecutor() as executor`` triggers
    ``executor.shutdown(wait=True)`` on context exit, which kills the
    children at the OS level (``TerminateProcess`` on Windows) without
    giving Fortran cleanup a chance to run.
    """
    import signal as _signal
    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)


def _orange_span_worker(task_data):
    """Worker function to compute a single orange span in a separate process.

    Mirrors ``_geodesic_decasteljau_worker`` in ``geo_splines.py`` for
    bit-for-bit parity with the editor's orange layer:

      - **Endpoints are pre-seeded with the literal P0 / P1**
        (``ctrl[0]`` / ``ctrl[3]`` = node origins).  Computing the
        endpoints via ``de_casteljau(t=0)`` and ``de_casteljau(t=1)``
        chains five ``compute_endpoint_local`` calls, each of which
        inserts points into the mesh with a topology-tolerance / nudge
        step that drifts up to ~0.2 units away from the true node
        position.  The editor avoids this by seeding the endpoints
        explicitly and only sampling the *interior* t values; we do
        the same here.

      - The t grid is ``curvature_adaptive_t_vals`` when adaptive=True
        (matches the editor's default ``ADAPTIVE_SAMPLING=True``),
        falling back to ``np.linspace`` otherwise.

    The caller is responsible for the secant-chord subdivision pass
    (the editor runs it post-worker in ``_apply_orange_progress``).
    """
    (v, f, ctrl, path_b, path_a_rev, t_grid) = task_data

    # Local imports — needed inside spawn-mode worker children.
    import numpy as np
    from geodesics import GeodesicMesh

    # GeodesicMesh accepts (V, F) arrays directly.
    geo = GeodesicMesh(v, f)

    P0, H_out, H_in, P1 = ctrl

    path_12 = geo.compute_endpoint_local(H_out, H_in)
    if path_12 is None or len(path_12) < 2:
        path_12 = np.array([H_out, H_in])

    cum_b, total_b = GeodesicMesh.compute_path_lengths(path_b)
    cum_a, total_a = GeodesicMesh.compute_path_lengths(path_a_rev)
    cum_12, total_12 = GeodesicMesh.compute_path_lengths(path_12)

    n = len(t_grid)
    span_pts = [None] * n
    # Pre-seed endpoints with the literal node origins (matches editor).
    span_pts[0]  = np.asarray(P0, dtype=float)
    span_pts[-1] = np.asarray(P1, dtype=float)

    # Inner indices only — endpoints are already seeded.
    for idx in range(1, n - 1):
        t = float(t_grid[idx])

        b01 = GeodesicMesh.geodesic_lerp(path_b, t, cum_b, total_b)
        b12 = GeodesicMesh.geodesic_lerp(path_12, t, cum_12, total_12)
        b23 = GeodesicMesh.geodesic_lerp(path_a_rev, t, cum_a, total_a)

        try:
            path_c0 = geo.compute_endpoint_local(b01, b12)
        except (RuntimeError, ValueError) as exc:
            log.debug("compute_endpoint_local(b01, b12) failed: %s", exc)
            path_c0 = np.array([b01, b12])
        if path_c0 is None or len(path_c0) < 2:
            path_c0 = np.array([b01, b12])

        try:
            path_c1 = geo.compute_endpoint_local(b12, b23)
        except (RuntimeError, ValueError) as exc:
            log.debug("compute_endpoint_local(b12, b23) failed: %s", exc)
            path_c1 = np.array([b12, b23])
        if path_c1 is None or len(path_c1) < 2:
            path_c1 = np.array([b12, b23])

        cum_c0, total_c0 = GeodesicMesh.compute_path_lengths(path_c0)
        cum_c1, total_c1 = GeodesicMesh.compute_path_lengths(path_c1)
        c0 = GeodesicMesh.geodesic_lerp(path_c0, t, cum_c0, total_c0)
        c1 = GeodesicMesh.geodesic_lerp(path_c1, t, cum_c1, total_c1)

        try:
            path_f = geo.compute_endpoint_local(c0, c1)
        except (RuntimeError, ValueError) as exc:
            log.debug("compute_endpoint_local(c0, c1) failed: %s", exc)
            path_f = np.array([c0, c1])
        if path_f is None or len(path_f) < 2:
            path_f = np.array([c0, c1])

        cum_f, total_f = GeodesicMesh.compute_path_lengths(path_f)
        span_pts[idx] = GeodesicMesh.geodesic_lerp(path_f, t, cum_f, total_f)

    return np.array(span_pts)


def compute_orange(geo, nodes, closed, n_samples, adaptive: bool = True):
    """Computes fully geodesic (orange) de Casteljau points for one spline.

    Mirrors the editor's orange-layer pipeline end-to-end so the export
    produces the exact curve the user sees on screen:

      1. Per-span control points ``[P0, H_out, H_in, P1]`` built from
         the node origins + handle endpoints.
      2. ``t_grid`` from ``curvature_adaptive_t_vals`` when ``adaptive``
         (matches editor's default ``ADAPTIVE_SAMPLING=True``), else
         ``np.linspace``.
      3. Worker computes only the *inner* points; the parent (here)
         pre-seeds endpoints with the literal node origins via the
         worker's seed logic.
      4. ``subdivide_secant_chords`` post-processing identical to
         ``_apply_orange_progress`` in the editor.
    """
    n_nodes = len(nodes)
    if n_nodes < 2:
        return []

    n_spans = n_nodes if closed else n_nodes - 1
    tasks = []

    # Mesh arrays passed by reference once per task — avoids re-pickling
    # the large V / F per worker call.
    v = geo.V
    f = geo.F

    for i in range(n_spans):
        n0 = nodes[i]
        n1 = nodes[(i + 1) % n_nodes]
        if n0['p_b'] is None or n1['p_a'] is None:
            tasks.append(None)
            continue
        if n0['path_b'] is None or n1['path_a'] is None:
            tasks.append(None)
            continue

        ctrl = [
            np.asarray(n0['origin'], dtype=float),  # P0
            np.asarray(n0['p_b'], dtype=float),     # H_out
            np.asarray(n1['p_a'], dtype=float),     # H_in
            np.asarray(n1['origin'], dtype=float),  # P1
        ]
        path_b = np.asarray(n0['path_b'], dtype=float)
        path_a_rev = np.asarray(n1['path_a'], dtype=float)[::-1].copy()

        if adaptive:
            t_grid = GeodesicMesh.curvature_adaptive_t_vals(ctrl, n_samples)
        else:
            t_grid = np.linspace(0.0, 1.0, n_samples)

        tasks.append((v, f, ctrl, path_b, path_a_rev, t_grid))

    log.info("computing %d spans in parallel...", n_spans)

    all_pts = [None] * n_spans
    valid_task_indices = [i for i, t in enumerate(tasks) if t is not None]
    valid_tasks = [tasks[i] for i in valid_task_indices]

    with ProcessPoolExecutor(initializer=_orange_worker_init) as executor:
        results = list(executor.map(_orange_span_worker, valid_tasks))

    # Post-process: same secant chord subdivision the editor applies in
    # _apply_orange_progress when the worker emits 'done'.  Keeps the
    # polyline visibly close to the surface even when the de Casteljau
    # samples land on opposite sides of a ridge.
    mean_edge = float(np.sqrt(geo._face_edge_len2.mean()))
    secant_tol = mean_edge * 0.01
    for i, res in zip(valid_task_indices, results):
        if res is None or len(res) < 2:
            all_pts[i] = res
            continue
        all_pts[i] = geo.subdivide_secant_chords(res, tol=secant_tol, max_depth=6)

    return [p for p in all_pts if p is not None]


def compute_interp(geo, nodes, closed, n_samples):
    """Computes interpolation B-spline (black) curve points for one spline.

    Uses scipy ``splprep``/``splev`` through node origins, projected onto
    the surface.  Fast (~ms), no geodesic awareness — purely node-defined.
    """
    from scipy.interpolate import splprep, splev

    n_nodes = len(nodes)
    if n_nodes < 2:
        return []

    origins = np.array([nd['origin'] for nd in nodes], dtype=float)
    k = min(3, n_nodes - 1)

    if closed and n_nodes < k + 1:
        return []

    try:
        tck, _ = splprep(
            [origins[:, 0], origins[:, 1], origins[:, 2]],
            s=0, k=k, per=closed)
    except (TypeError, ValueError) as exc:
        log.debug("splprep failed (degenerate node layout): %s", exc)
        return []

    n = max(n_samples, 200)
    u_fine = np.linspace(0.0, 1.0, n)
    x, y, z = splev(u_fine, tck)
    raw_pts = np.column_stack((x, y, z))
    projected = geo.project_smooth_batch(raw_pts)

    # Return as a single list (one curve per spline, not per span)
    return [projected]


def write_obj(path, spline_points_list):
    """Writes curve points as an OBJ file with vertices and lines."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Geodesic Spline Export\n")
        v_offset = 1
        for spline_idx, spans in enumerate(spline_points_list):
            f.write(f"g spline_{spline_idx}\n")
            
            # Collect all points for this spline to create a continuous line
            # Spans are lists of numpy arrays
            all_points = []
            for span in spans:
                for pt in span:
                    all_points.append(pt)
            
            if not all_points:
                continue
                
            for pt in all_points:
                f.write(f"v {pt[0]:.8f} {pt[1]:.8f} {pt[2]:.8f}\n")
            
            # Create individual segments connecting the vertices using 'f'
            for i in range(v_offset, v_offset + len(all_points) - 1):
                f.write(f"f {i} {i+1}\n")
            v_offset += len(all_points)


def write_vtk(path, spline_points_list, landmarks=None):
    """Writes curve points + optional landmarks as a legacy BINARY VTK
    UnstructuredGrid file.

    *spline_points_list*: list of splines, each a list of (M, 3) span
    polylines.  Each span is written as M-1 ``VTK_LINE`` (cell type 3)
    segments.

    *landmarks*: optional list of (3,) points written as ``VTK_VERTEX``
    (cell type 1) cells — one per landmark.  Used by the editor's 'v'
    export for splines that have only a single node (interpreted as a
    user-marked point rather than a curve).  Pre-existing CLI callers
    that pass only ``spline_points_list`` keep their previous behaviour
    (no vertex cells written).

    Mixed cell types are valid in legacy VTK UnstructuredGrid; ParaView
    and other VTK consumers handle the combination natively.
    """
    import numpy as np
    all_points = []
    line_segments: list[tuple[int, int]] = []
    vertex_cells: list[int] = []

    # Flatten spans into individual line segments
    for spans in spline_points_list:
        for span in spans:
            if span is not None and len(span) >= 2:
                v_offset = len(all_points)
                all_points.extend(span)
                for i in range(len(span) - 1):
                    line_segments.append((v_offset + i, v_offset + i + 1))

    # Append landmark points as VTK_VERTEX cells
    if landmarks:
        for lm in landmarks:
            v_offset = len(all_points)
            all_points.append(np.asarray(lm, dtype=float))
            vertex_cells.append(v_offset)

    if not all_points:
        return

    n_lines = len(line_segments)
    n_verts = len(vertex_cells)
    n_cells = n_lines + n_verts

    with open(path, 'wb') as f:
        # Header (ASCII part)
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(b"Geodesic Splines Export\n")
        f.write(b"BINARY\n")
        f.write(b"DATASET UNSTRUCTURED_GRID\n\n")

        # Points (Binary Big-Endian)
        f.write(f"POINTS {len(all_points)} double\n".encode('ascii'))
        pts_bin = np.array(all_points, dtype='>f8').tobytes()
        f.write(pts_bin)
        f.write(b"\n")

        # CELLS section: each cell is laid out as [n_pts, p0, p1, ...].
        # Lines contribute [2, a, b], vertices contribute [1, p].
        cells_data: list[int] = []
        for a, b in line_segments:
            cells_data.extend([2, a, b])
        for p in vertex_cells:
            cells_data.extend([1, p])

        f.write(f"CELLS {n_cells} {len(cells_data)}\n".encode('ascii'))
        cells_bin = np.array(cells_data, dtype='>i4').tobytes()
        f.write(cells_bin)
        f.write(b"\n")

        # CELL_TYPES: VTK_LINE = 3, VTK_VERTEX = 1.
        f.write(f"CELL_TYPES {n_cells}\n".encode('ascii'))
        types_bin = np.array(
            [3] * n_lines + [1] * n_verts, dtype='>i4').tobytes()
        f.write(types_bin)
        f.write(b"\n")


def format_point(pt):
    return f"{pt[0]:.16e} , {pt[1]:.16e} , {pt[2]:.16e}"


def main():
    if len(sys.argv) == 1:
        print("Usage: python spline_export.py <splines.json> [layer] [--samples N] [--obj]")
        print("\nExport geodesic spline curves from a JSON session file.")
        print("\nArguments:")
        print("  json_file     Path to the splines JSON file")
        print("  layer         Curve layer: b=blue(semi-geodesic), o=orange(exact), k=interp(black)")
        print("                (default: o)")
        print("\nOptions:")
        print("  --samples N   Minimum samples per span (default: 60)")
        print("  --obj         Export as .obj file instead of CSV to stdout")
        print("  --vtk         Export as binary legacy .vtk file instead of CSV to stdout")
        print("  -h, --help    Show this help message and exit")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Export geodesic spline curves from a JSON session file.")
    parser.add_argument('json_file', help="Path to the splines JSON file")
    parser.add_argument('layer', nargs='?', choices=['b', 'o', 'k'], default='o',
                        help="Curve layer: b=blue(semi-geodesic), o=orange(exact), k=interp(black) (default: o)")
    parser.add_argument('--samples', type=int, default=60,
                        help="Minimum samples per span (default: 60)")
    parser.add_argument('--obj', action='store_true',
                        help="Export to .obj file (basename.obj)")
    parser.add_argument('--vtk', action='store_true',
                        help="Export to binary legacy .vtk file (basename.vtk)")
    args = parser.parse_args()

    data = load_json(args.json_file)
    geo, splines, splines_closed = rebuild_mesh_and_nodes(data)

    compute_fn = {'b': compute_blue, 'o': compute_orange,
                  'k': compute_interp}
    layer_name = {'b': 'blue (semi-geodesic)',
                  'o': 'orange (fully geodesic)',
                  'k': 'black (interpolation)'}

    log.info("layer: %s", layer_name[args.layer])
    log.info("splines: %d", len(splines))
    log.info("samples/span: %d", args.samples)

    all_spline_points = []
    for sid, (nodes, closed) in enumerate(zip(splines, splines_closed)):
        n_nodes = len(nodes)
        log.info("spline %d: %d nodes, %s",
                 sid, n_nodes, 'closed' if closed else 'open')

        if n_nodes < 2:
            all_spline_points.append([])
            continue

        # Compute curve
        span_pts_list = compute_fn[args.layer](
            geo, nodes, closed, args.samples)
        all_spline_points.append(span_pts_list)

    if args.obj:
        obj_path = os.path.splitext(args.json_file)[0] + ".obj"
        log.info("exporting to OBJ: %s", obj_path)
        write_obj(obj_path, all_spline_points)
    elif args.vtk:
        vtk_path = os.path.splitext(args.json_file)[0] + ".vtk"
        log.info("exporting to binary legacy VTK: %s", vtk_path)
        write_vtk(vtk_path, all_spline_points)
    else:
        # CSV output to stdout
        first_spline = True
        for span_pts_list in all_spline_points:
            if not first_spline:
                # Break between splines
                print(NAN_LINE)
            first_spline = False

            # Print all points for all spans of this spline
            for span in span_pts_list:
                for pt in span:
                    print(format_point(pt))

    log.info("done.")


if __name__ == '__main__':
    main()
