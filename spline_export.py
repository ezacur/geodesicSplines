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
import sys

import numpy as np

from geodesics import GeodesicMesh


NAN_LINE = "NaN , NaN , NaN"


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def rebuild_mesh_and_nodes(data: dict):
    """Rebuilds GeodesicMesh and node data from JSON.

    Each node is reconstructed from 2 saved fields: ``origin`` (3-D
    position) and ``tangent`` (3-D vector whose magnitude is h_length).

    Returns ``(geo, splines, splines_closed)`` where *splines* is a list
    of lists of dicts with 'origin', 'face_idx', 'p_a', 'p_b', 'path_a',
    'path_b'.
    """
    import pyvista as pv
    import warnings

    mesh_file = data['mesh_file']
    print(f"# Loading mesh: {mesh_file}", file=sys.stderr)
    if mesh_file == "ICOSAHEDRON":
        # Import the generator from geo_splines
        from geo_splines import _make_icosahedron
        mesh = _make_icosahedron(radius=10.0)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            mesh = pv.read(mesh_file).extract_surface().triangulate().clean()
    geo = GeodesicMesh(mesh)

    splines = []
    splines_closed = []
    for sd in data['splines']:
        nodes = []
        for nd in sd['nodes']:
            origin = np.array(nd['origin'], dtype=float)
            tangent_full = np.array(nd['tangent'], dtype=float)

            h_length = float(np.linalg.norm(tangent_full))
            if h_length > 1e-15:
                tangent_dir = tangent_full / h_length
            else:
                tangent_dir = np.array([1.0, 0.0, 0.0])
                h_length = 0.01

            face_idx = geo.find_face(origin)
            path_b = geo.compute_shoot(origin, tangent_dir, h_length, face_idx)
            path_a = geo.compute_shoot(origin, -tangent_dir, h_length, face_idx)
            p_b = path_b[-1] if path_b is not None else None
            p_a = path_a[-1] if path_a is not None else None

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
        print(f"#   span {i}: computing path_12...", file=sys.stderr)
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


def compute_orange(geo, nodes, closed, n_samples):
    """Computes fully geodesic (orange) de Casteljau points for one spline."""
    all_pts = []
    n_nodes = len(nodes)
    if n_nodes < 2:
        return all_pts

    n_spans = n_nodes if closed else n_nodes - 1
    for i in range(n_spans):
        n0 = nodes[i]
        n1 = nodes[(i + 1) % n_nodes]
        if n0['p_b'] is None or n1['p_a'] is None:
            continue
        if n0['path_b'] is None or n1['path_a'] is None:
            continue

        H_out, H_in = n0['p_b'], n1['p_a']
        path_b = n0['path_b']
        path_a_rev = n1['path_a'][::-1]

        print(f"#   span {i}: computing path_12...", file=sys.stderr)
        path_12 = geo.compute_endpoint_local(H_out, H_in)
        if path_12 is None or len(path_12) < 2:
            path_12 = np.array([H_out, H_in])
        cum_b, total_b = GeodesicMesh.compute_path_lengths(path_b)
        cum_a, total_a = GeodesicMesh.compute_path_lengths(path_a_rev)
        cum_12, total_12 = GeodesicMesh.compute_path_lengths(path_12)

        t_vals = np.linspace(0.0, 1.0, n_samples)
        span_pts = []

        for idx, t in enumerate(t_vals):
            b01 = GeodesicMesh.geodesic_lerp(path_b, t, cum_b, total_b)
            b12 = GeodesicMesh.geodesic_lerp(path_12, t, cum_12, total_12)
            b23 = GeodesicMesh.geodesic_lerp(path_a_rev, t, cum_a, total_a)

            try:
                path_c0 = geo.compute_endpoint_local(b01, b12)
            except Exception:
                path_c0 = np.array([b01, b12])
            if path_c0 is None or len(path_c0) < 2:
                path_c0 = np.array([b01, b12])

            try:
                path_c1 = geo.compute_endpoint_local(b12, b23)
            except Exception:
                path_c1 = np.array([b12, b23])
            if path_c1 is None or len(path_c1) < 2:
                path_c1 = np.array([b12, b23])

            cum_c0, total_c0 = GeodesicMesh.compute_path_lengths(path_c0)
            cum_c1, total_c1 = GeodesicMesh.compute_path_lengths(path_c1)
            c0 = GeodesicMesh.geodesic_lerp(path_c0, t, cum_c0, total_c0)
            c1 = GeodesicMesh.geodesic_lerp(path_c1, t, cum_c1, total_c1)

            try:
                path_f = geo.compute_endpoint_local(c0, c1)
            except Exception:
                path_f = np.array([c0, c1])
            if path_f is None or len(path_f) < 2:
                path_f = np.array([c0, c1])

            cum_f, total_f = GeodesicMesh.compute_path_lengths(path_f)
            result = GeodesicMesh.geodesic_lerp(path_f, t, cum_f, total_f)
            span_pts.append(result)

            if (idx + 1) % 5 == 0 or idx == len(t_vals) - 1:
                print(f"#   span {i}: {idx+1}/{n_samples} points",
                      file=sys.stderr)

        all_pts.append(np.array(span_pts))

    return all_pts


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
    except Exception:
        return []

    n = max(n_samples, 200)
    u_fine = np.linspace(0.0, 1.0, n)
    x, y, z = splev(u_fine, tck)
    raw_pts = np.column_stack((x, y, z))
    projected = geo.project_smooth_batch(raw_pts)

    # Return as a single list (one curve per spline, not per span)
    return [projected]


def format_point(pt):
    return f"{pt[0]:.16e} , {pt[1]:.16e} , {pt[2]:.16e}"


def main():
    parser = argparse.ArgumentParser(
        description="Export geodesic spline curves from a JSON session file.")
    parser.add_argument('json_file', help="Path to the splines JSON file")
    parser.add_argument('layer', choices=['b', 'o', 'k'],
                        help="Curve layer: b=blue(semi-geodesic), o=orange(exact), k=interp(black)")
    parser.add_argument('--samples', type=int, default=60,
                        help="Minimum samples per span (default: 60)")
    args = parser.parse_args()

    data = load_json(args.json_file)
    geo, splines, splines_closed = rebuild_mesh_and_nodes(data)

    compute_fn = {'b': compute_blue, 'o': compute_orange,
                  'k': compute_interp}
    layer_name = {'b': 'blue (semi-geodesic)',
                  'o': 'orange (fully geodesic)',
                  'k': 'black (interpolation)'}

    print(f"# Layer: {layer_name[args.layer]}", file=sys.stderr)
    print(f"# Splines: {len(splines)}", file=sys.stderr)
    print(f"# Samples/span: {args.samples}", file=sys.stderr)

    first_spline = True
    for sid, (nodes, closed) in enumerate(zip(splines, splines_closed)):
        n_nodes = len(nodes)
        print(f"# Spline {sid}: {n_nodes} nodes, "
              f"{'closed' if closed else 'open'}", file=sys.stderr)

        if not first_spline:
            # Break between splines
            print(NAN_LINE)
        first_spline = False

        if n_nodes == 0:
            continue

        # Compute curve
        span_pts_list = compute_fn[args.layer](
            geo, nodes, closed, args.samples)

        # Output: interleave landmarks and curve segments
        for i, node in enumerate(nodes):
            # Landmark
            print(NAN_LINE)
            print(format_point(node['origin']))
            print(NAN_LINE)

            # Curve segment from this node to the next
            if i < len(span_pts_list):
                for pt in span_pts_list[i]:
                    print(format_point(pt))

        # For closed splines, the last span wraps to node 0 — its curve
        # was already output as span_pts_list[-1].  Add the first node
        # as closing landmark.
        if closed and n_nodes >= 2:
            print(NAN_LINE)
            print(format_point(nodes[0]['origin']))
            print(NAN_LINE)

    print(f"# Done.", file=sys.stderr)


if __name__ == '__main__':
    main()
