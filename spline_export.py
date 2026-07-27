# SPDX-License-Identifier: Apache-2.0
"""
spline_export.py — Command-line exporter for geodesic spline curves.

Reads a spline JSON file (saved by geo_splines.py) and outputs the
curve points for a selected layer (blue/orange/interp) to stdout (CSV)
or to a file on disk (OBJ / VTK).

Usage
-----

::

    python spline_export.py <session.json> [layer] [mesh_override] \\
                            [--samples N] [--obj | --vtk]

Positional arguments:

  * ``session.json``     Required.  Path to a session file produced by
                         ``geo_splines.py`` (v1 or v2 schema).
  * ``layer``            Optional.  Curve layer letter — one of
                         ``b`` (blue / semi-geodesic), ``o`` (orange /
                         fully geodesic, the default), or ``k`` (black /
                         scipy interp through node origins).
  * ``mesh_override``    Optional.  Path to a mesh file
                         (``.vtk``/``.obj``/``.ply``/``.stl``) used in
                         place of the session's ``mesh_file`` field.
                         Lets you export the same splines onto a
                         different geometry — registered counterpart,
                         higher-res resampling, alternative mesh —
                         since the JSON stores 3-D positions, not
                         vertex indices.

Layer and mesh_override are **order-agnostic**: ``b L.vtk`` and
``L.vtk b`` are equivalent.  ``--mesh PATH`` is an explicit
alternative to the positional form.

Options:

  * ``--samples N``      Minimum samples per span (>= 2, default 60).
  * ``--obj``            Write ``<basename>.obj`` instead of CSV.
                         Mutually exclusive with ``--vtk``.
  * ``--vtk``            Write ``<basename>.vtk`` (binary legacy
                         UnstructuredGrid) instead of CSV.  Mutually
                         exclusive with ``--obj``.
  * ``-h``, ``--help``   Show argparse-generated help and exit.

Without ``--obj`` / ``--vtk`` the export is CSV to stdout; pipe or
redirect to capture it.

Output format (CSV, one point per line)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    x , y , z          — curve point
    NaN , NaN , NaN    — break between splines (also emitted at any
                         gap left by a skipped span)

Consecutive spans of one spline share their endpoint, so each intact
spline prints as one contiguous polyline; downstream tools
(Matplotlib, R, MATLAB) cut polylines at the NaN rows without joining
disjoint splines.  Node-origin landmark rows are no longer emitted
(an earlier revision wrapped each origin in NaN sentinels); splines
with fewer than 2 nodes contribute no rows at all.

Layers
------
  - **b** (blue): semi-geodesic Bézier — level-1 uses the exact geodesic
    between H_out and H_in (via ``compute_endpoint_local``); levels 2-3
    Euclidean + projection.  ~seconds per spline on a 240K-face mesh.
  - **o** (orange): fully geodesic de Casteljau — geodesic interpolation
    at every level.  ~minutes to hours on large meshes.
  - **k** (black): interpolation B-spline through node origins (scipy
    splprep/splev), projected onto the surface.  Fastest; no handles.

Diagnostics
-----------

All ``log.info`` / ``log.error`` go to **stderr** with the
``[LEVEL] spline_export: ...`` prefix.  Stdout stays clean for the
CSV stream.  ``GEO_SPLINES_DEBUG=1`` in the environment raises the
log level to DEBUG.

Exit codes:

  * ``0`` success.
  * ``2`` JSON missing / unreadable / malformed / failing schema
    validation; override mesh missing, or the session's own
    ``mesh_file`` not found (looked up next to the session as well as
    in the CWD); the ``--obj`` / ``--vtk`` output path resolving onto
    one of this run's own inputs (the session or the mesh) — see
    ``_guard_output_path``; or nothing to export (no spline with >= 2
    nodes and no landmark).

Examples
--------

::

    # Default orange layer, CSV to stdout, redirected to a file:
    python spline_export.py 20260414_153022.json > orange.csv

    # Blue layer:
    python spline_export.py 20260414_153022.json b > blue.csv

    # Orange with 50 samples / span:
    python spline_export.py 20260414_153022.json o --samples 50

    # Same session but exported against a different mesh:
    python spline_export.py 20260414_153022.json L_hires.vtk --vtk

    # Order-agnostic positionals + interp layer to OBJ:
    python spline_export.py 20260414_153022.json k other_mesh.obj --obj

    # Explicit --mesh option (equivalent to a positional .vtk):
    python spline_export.py 20260414_153022.json --mesh L_hires.vtk --vtk
"""

from __future__ import annotations

import argparse
import json
import logging
import os

# Disable Intel Fortran's Win32 Console Control Handler before MKL
# loads (via numpy/scipy/pp3d below).  Without this, Ctrl+C on the
# parent or any worker triggers ``libifcoremd.dll``'s ``forrtl: error
# (200)`` traceback + ``abort()`` before Python can run KeyboardInterrupt
# cleanup, hanging the terminal with four interleaved tracebacks.
# Mirror of the same assignment in ``geo_splines.py``; see that file
# for the full rationale.
os.environ.setdefault('FOR_DISABLE_CONSOLE_CTRL_HANDLER', '1')

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from geodesics import GeodesicMesh, eval_cascade_at_t

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
    """Load *path* as JSON, returning the top-level dict.

    Surfaces friendly diagnostics (and exits with code 2) for the
    three classes of failure end users actually hit:

      * file missing / unreadable
      * not valid JSON
      * top level is not an object

    Also runs the same ``_validate_session_dict`` schema check the
    interactive editor does, so a JSON the editor rejects can never
    silently mis-export here.  Cross-tool consistency guarantee.
    """
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        log.error("session file not found: %s", path)
        sys.exit(2)
    except PermissionError as exc:
        log.error("cannot read %s: %s", path, exc)
        sys.exit(2)
    except json.JSONDecodeError as exc:
        log.error("malformed JSON in %s: %s", path, exc)
        sys.exit(2)
    except OSError as exc:
        log.error("I/O error reading %s: %s", path, exc)
        sys.exit(2)

    if not isinstance(data, dict):
        log.error("session JSON must be an object; got %s",
                  type(data).__name__)
        sys.exit(2)

    # Validate against the same schema the editor enforces — keeps
    # CLI / GUI behaviour aligned and rejects hostile / hand-edited
    # inputs (NaN/Inf coordinates, missing splines, malformed nodes)
    # with a clear error rather than crashing deep inside the solver.
    # ``session_io`` is stdlib-only (no pyvista / vtk), so validation
    # runs before we touch the heavy geometry stack — a malformed
    # session is rejected cheaply, and this no longer imports the whole
    # GUI module just to reach the validator.
    from session_io import _validate_session_dict as _validate
    try:
        _validate(data)
    except ValueError as exc:
        log.error("invalid session %s: %s", path, exc)
        sys.exit(2)

    # Mirror the editor's version gate (``_on_load``): a session the
    # editor refuses to open must not silently export here under a
    # v1/v2 interpretation its (future) version may not have.
    version = data.get('version')
    if version not in (1, 2):
        log.error("unknown session version %r in %s (supported: 1, 2)",
                  version, path)
        sys.exit(2)

    return data


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


def _resolve_mesh_path(mesh_file: str, session_path: str | None) -> str:
    """Resolve a session's ``mesh_file`` to an existing path.

    ``mesh_file`` is whatever relative name the editor was launched with
    (``geo_splines.py:mesh_label``), so resolving it against the process
    CWD only works when the CLI happens to run from the directory the
    editor ran from.  Sessions are normally kept next to their mesh, so
    try the session's own directory first — the same fix
    ``tests/benchmark_endpoint_local.py`` already applies internally.

    Returns the first candidate that exists; falls back to the literal
    value so the caller's ``os.path.exists`` check produces the message.
    """
    if os.path.isabs(mesh_file) or os.path.exists(mesh_file):
        return mesh_file
    if session_path:
        cand = os.path.join(os.path.dirname(os.path.abspath(session_path)),
                            mesh_file)
        if os.path.exists(cand):
            return cand
    return mesh_file


def rebuild_mesh_and_nodes(data: dict, session_path: str | None = None):
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
    # ``_validate_session_dict`` does not require ``mesh_file`` (undo
    # snapshots legitimately omit it) — reject it here with the CLI's
    # clean exit-2 diagnostic instead of a raw KeyError traceback.
    mesh_file = data.get('mesh_file')
    if not mesh_file:
        log.error("session has no 'mesh_file' key — pass --mesh to "
                  "supply the mesh explicitly")
        sys.exit(2)
    if mesh_file not in ("__builtin__:icosahedron", "ICOSAHEDRON"):
        resolved = _resolve_mesh_path(mesh_file, session_path)
        if not os.path.exists(resolved):
            # Previously this fell straight into ``pv.read`` and surfaced
            # as a raw FileNotFoundError traceback with exit 1, while the
            # module's own exit-code table promises a clean 2 for a
            # missing mesh.
            log.error("mesh referenced by the session not found: %s "
                      "(looked next to the session too) — pass --mesh to "
                      "supply it explicitly", mesh_file)
            sys.exit(2)
        if resolved != mesh_file:
            log.info("resolved session mesh relative to the session "
                     "directory: %s", resolved)
        mesh_file = resolved
        # Write the resolved path back so downstream consumers see the
        # mesh actually loaded — ``_guard_output_path`` compares the
        # export target against it, and an unresolved relative name
        # would fail its ``os.path.exists`` test and silently disarm
        # the overwrite guard.  ``main`` already mutates this key for
        # ``--mesh``, so the ownership convention is established.
        data['mesh_file'] = mesh_file
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
    # CLI exporter never picks: ``compute_blue`` / ``compute_orange``
    # / ``compute_interp`` only consume KDTree + solver + projection.
    # Skip locator construction (~250 ms on dense meshes).
    geo = GeodesicMesh(V, F, build_locator=False)

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
                path, _ = geo.compute_endpoint_from_origin(cache, p_target)
            except (RuntimeError, ValueError, TypeError, IndexError) as exc:
                log.warning("v2 load: solver failed for handle %s (%s); "
                            "handle path degrades to a straight segment",
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


def compute_blue(geo, nodes, closed, n_samples) -> list[np.ndarray]:
    """Computes semi-geodesic Bézier (blue) curve points for one spline.

    Matches the interactive app's consolidated blue: level-1 geodesic lerp
    on all three control segments (including H_out→H_in via
    ``compute_endpoint_local``); levels 2-3 Euclidean + projection.

    Return contract
    ~~~~~~~~~~~~~~~
    ``list[np.ndarray]`` — one ``(M_i, 3)`` polyline **per Bézier span**.
    A spline of N nodes yields ``N - 1`` open or ``N`` closed entries.
    Returns an empty list when the spline has fewer than 2 nodes.

    Same shape as ``compute_orange``.  ``compute_interp`` is the
    intentional outlier (one polyline for the whole spline) — see its
    docstring for why.
    """
    all_pts: list[np.ndarray] = []
    n_nodes = len(nodes)
    if n_nodes < 2:
        return all_pts

    n_spans = n_nodes if closed else n_nodes - 1
    n_skipped = 0
    degraded_spans: list[int] = []
    for i in range(n_spans):
        n0 = nodes[i]
        n1 = nodes[(i + 1) % n_nodes]
        ctrl = [n0['origin'], n0['p_b'], n1['p_a'], n1['origin']]
        if any(p is None for p in ctrl):
            n_skipped += 1
            continue
        path_b = n0['path_b']
        # ``hybrid_de_casteljau_curve`` expects ``path_in`` oriented
        # P1 -> H_in exactly as stored (it reverses internally, same as
        # the editor's call in ``_recompute_spans``).  Passing a
        # pre-reversed copy double-reverses: every span then ends at the
        # H_in handle instead of the destination node.  Only the
        # ``eval_cascade_at_t`` orange path takes a pre-reversed copy.
        path_a = n1['path_a']

        # Geodesic H_out → H_in via local submesh solver.  Guarded for
        # the same reason ``_orange_span_worker`` guards its level-1
        # solve: one bad span must degrade to the hybrid fallback, not
        # abort the whole export with a traceback and exit 1.
        log.debug("span %d: computing path_12 (H_out -> H_in)", i)
        try:
            path_12, was_fallback = geo.compute_endpoint_local(
                n0['p_b'], n1['p_a'])
        except (RuntimeError, ValueError, TypeError, IndexError) as exc:
            log.debug("span %d: path_12 solver failed: %s", i, exc)
            path_12, was_fallback = None, True
        if path_12 is None or len(path_12) < 2:
            path_12 = None
            degraded_spans.append(i)
        elif was_fallback:
            degraded_spans.append(i)

        n = geo.adaptive_samples(ctrl, 0.3, 15, 100)
        n = max(n, n_samples)
        pts = geo.hybrid_de_casteljau_curve(
            ctrl, path_b, path_a, n, fast=False, path_12=path_12)
        pts = geo.project_smooth_batch(pts)
        all_pts.append(pts)

    if n_skipped:
        log.warning("blue: %d of %d spans SKIPPED (handles missing or "
                    "unsolvable) — output is incomplete", n_skipped, n_spans)
    if degraded_spans:
        log.warning("blue: %d of %d spans DEGRADED (level-1 geodesic "
                    "H_out->H_in fell back to a straight line; spans %s)",
                    len(degraded_spans), n_spans, degraded_spans)

    return all_pts


# Per-worker GeodesicMesh, built ONCE in the initializer and reused
# by every task running on that worker.  The previous implementation
# pickled (V, F) into every task tuple — at 240 K faces that's
# 10-20 MB serialised per span, hammered through the IPC pipe
# n_spans times.  The initializer pattern picks (V, F) up exactly
# once per worker process.
_worker_geo: GeodesicMesh | None = None


def _orange_worker_init(v: np.ndarray, f: np.ndarray) -> None:
    """ProcessPoolExecutor initializer: blocks SIGINT and builds the
    per-worker ``GeodesicMesh`` once.

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

    *v* / *f* are required and passed as ``initargs`` from the parent
    — there is no legitimate caller without them.  Defaults were
    removed so accidental misuse fails immediately at executor
    startup with a clear ``TypeError`` rather than producing a worker
    with ``_worker_geo = None`` and crashing later inside
    ``_orange_span_worker`` with an opaque ``AttributeError``.

    Workers don't need the VTK locator (only the KDTree + solver),
    so we pass ``build_locator=False`` to skip the ~250 ms locator
    construction per worker.
    """
    # Belt-and-braces against MKL's Console Control Handler — same
    # rationale as in ``geo_splines._process_initializer``.  Re-assert
    # the env var in case a custom launcher cleared it between fork
    # and the worker initializer.
    os.environ.setdefault('FOR_DISABLE_CONSOLE_CTRL_HANDLER', '1')
    import signal as _signal
    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)

    global _worker_geo
    from geodesics import GeodesicMesh as _GM
    _worker_geo = _GM(v, f, build_locator=False)


def _orange_span_worker(task_data):
    """Worker function to compute a single orange span in a separate process.

    Mirrors **phase 1** (canonical cascade samples) of the editor's
    ``span_workers._geodesic_decasteljau_worker``:

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

    It does NOT run the editor worker's phase 2 (cascade
    densification) or phase 3 (geodesic chord-bridging); the caller
    applies a legacy ``subdivide_secant_chords`` pass instead — see
    ``compute_orange``'s docstring for how the exported curve can
    therefore differ from the rendered one.

    Returns ``(span_pts, degraded)`` — *degraded* mirrors the editor
    worker's ``degraded_any``: ``True`` when the level-1 middle path
    or any cascade evaluation fell back to a straight-line stub, so
    the parent can warn instead of exporting a phantom curve silently.
    """
    (ctrl, path_b, path_a_rev, t_grid) = task_data

    # Local imports — needed inside spawn-mode worker children.
    import numpy as np

    # Per-worker mesh built once by ``_orange_worker_init`` — we trust
    # it is present (the initializer is always passed via initargs in
    # the parent); a missing worker-local mesh is a programmer error
    # and an AttributeError here is the right diagnostic.
    geo = _worker_geo

    P0, H_out, H_in, P1 = ctrl

    # Guard the level-1 solver exactly as the editor worker does
    # (span_workers._geodesic_decasteljau_worker): a solver exception on
    # one span must degrade to a straight stub, not abort the whole
    # export by propagating a pickled traceback through executor.map.
    degraded = False
    try:
        path_12, was_fallback = geo.compute_endpoint_local(H_out, H_in)
    except (RuntimeError, ValueError, TypeError, IndexError) as exc:
        log.debug("orange span: level-1 path_12 solver failed: %s", exc)
        path_12, was_fallback = None, True
    if path_12 is None or len(path_12) < 2:
        path_12 = np.array([H_out, H_in])
        degraded = True
    elif was_fallback:
        degraded = True

    cum_b, total_b = GeodesicMesh.compute_path_lengths(path_b)
    cum_a, total_a = GeodesicMesh.compute_path_lengths(path_a_rev)
    cum_12, total_12 = GeodesicMesh.compute_path_lengths(path_12)

    n = len(t_grid)
    span_pts: list[np.ndarray] = [np.empty(0)] * n
    # Pre-seed endpoints with the literal node origins (matches editor).
    span_pts[0]  = np.asarray(P0, dtype=float)
    span_pts[-1] = np.asarray(P1, dtype=float)

    # Inner indices only — endpoints are already seeded.  Delegates to the
    # shared cascade (``geodesics.eval_cascade_at_t``) — the exact routine
    # the editor's orange worker runs — for bit-for-bit parity.
    for idx in range(1, n - 1):
        t = float(t_grid[idx])
        span_pts[idx], deg = eval_cascade_at_t(
            geo, t, path_b, cum_b, total_b,
            path_a_rev, cum_a, total_a,
            path_12, cum_12, total_12)
        degraded |= deg

    return np.array(span_pts), degraded


def compute_orange(geo, nodes, closed, n_samples,
                   adaptive: bool = True) -> list[np.ndarray]:
    """Computes fully geodesic (orange) de Casteljau points for one spline.

    Mirrors **phase 1** of the editor's orange-layer pipeline (the
    canonical cascade samples):

      1. Per-span control points ``[P0, H_out, H_in, P1]`` built from
         the node origins + handle endpoints.
      2. ``t_grid`` from ``curvature_adaptive_t_vals`` when ``adaptive``
         (matches editor's default ``ADAPTIVE_SAMPLING=True``), else
         ``np.linspace``.
      3. Worker computes only the *inner* points; the parent (here)
         pre-seeds endpoints with the literal node origins via the
         worker's seed logic.

    Known divergence from the rendered curve
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The editor's worker additionally runs phase 2 (cascade
    densification) and phase 3 (geodesic chord-bridging); this export
    path instead applies the legacy ``subdivide_secant_chords``
    post-pass, which inserts surface-projected chord midpoints that
    are NOT on the de Casteljau cascade.  The exported polyline can
    therefore deviate from the on-screen orange curve wherever a chord
    crosses a ridge.  (The editor's ``v``-key export avoids this only
    when ``EXPORT_VTK_SAMPLES >= GEO_SAMPLES`` lets it reuse the live
    rendered polylines — see ``_on_export_vtk``.)

    Return contract
    ~~~~~~~~~~~~~~~
    ``list[np.ndarray]`` — one ``(M_i, 3)`` polyline **per Bézier span**.
    Spans whose endpoints could not be solved are filtered out, so the
    list length may be less than ``N - 1`` (open) / ``N`` (closed).
    Same shape as ``compute_blue``.

    Skipped spans (unsolvable handles) and degraded spans (worker fell
    back to a straight-line path somewhere) are reported via
    ``log.warning`` — the export succeeds but the user must know the
    output is not the full exact curve.
    """
    n_nodes = len(nodes)
    if n_nodes < 2:
        return []

    n_spans = n_nodes if closed else n_nodes - 1
    tasks: list[tuple | None] = []

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

        # V / F NOT in the task tuple — sent once via ``initargs``
        # below.  Saves ~10-20 MB of pickling per span on dense meshes.
        tasks.append((ctrl, path_b, path_a_rev, t_grid))

    log.info("computing %d spans in parallel...", n_spans)

    all_pts: list[np.ndarray | None] = [None] * n_spans
    valid_task_indices = [i for i, t in enumerate(tasks) if t is not None]
    valid_tasks = [tasks[i] for i in valid_task_indices]

    with ProcessPoolExecutor(
            initializer=_orange_worker_init,
            initargs=(geo.V, geo.F)) as executor:
        results = list(executor.map(_orange_span_worker, valid_tasks))

    # Post-process: legacy secant-chord subdivision.  Keeps the
    # polyline visibly close to the surface even when the de Casteljau
    # samples land on opposite sides of a ridge, but the inserted
    # midpoints are NOT on the cascade — the editor retired this pass
    # in favour of the worker's phase-2/3 pipeline (see the
    # "Known divergence" note in the docstring above).
    mean_edge = float(np.sqrt(geo._face_edge_len2.mean()))
    secant_tol = mean_edge * 0.01
    degraded_spans: list[int] = []
    for i, (res, degraded) in zip(valid_task_indices, results, strict=False):
        if degraded:
            degraded_spans.append(i)
        if res is None or len(res) < 2:
            all_pts[i] = res
            continue
        all_pts[i] = geo.subdivide_secant_chords(res, tol=secant_tol, max_depth=6)

    n_skipped = n_spans - len(valid_task_indices)
    if n_skipped:
        log.warning("orange: %d of %d spans SKIPPED (handles missing or "
                    "unsolvable) — output is incomplete", n_skipped, n_spans)
    if degraded_spans:
        log.warning("orange: %d of %d spans DEGRADED to straight-line "
                    "fallback (spans %s) — the editor paints these red",
                    len(degraded_spans), n_spans, degraded_spans)

    return [p for p in all_pts if p is not None]


def compute_interp(geo, nodes, closed, n_samples) -> list[np.ndarray]:
    """Computes interpolation B-spline (black) curve points for one spline.

    Uses scipy ``splprep``/``splev`` through node origins, projected onto
    the surface.  Fast (~ms), no geodesic awareness — purely node-defined.

    Return contract
    ~~~~~~~~~~~~~~~
    ``list[np.ndarray]`` — at most one ``(N, 3)`` polyline covering the
    **entire spline**.  Unlike ``compute_blue`` / ``compute_orange``,
    interp is a single fitted curve (not per-span Bezier sampling), so
    the list always has 0 or 1 element.  Downstream writers iterate
    "for spans in list, for span in spans" so the shape is compatible
    even though the semantics differ.
    """
    from scipy.interpolate import splev, splprep

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
    """Writes curve points as an OBJ file: ``v`` records plus one ``l``
    (polyline) record per span.

    ``l`` — not 2-vertex ``f`` records — is the OBJ element for
    polylines: the spec requires faces to have >= 3 vertices, and
    vtkOBJReader / ParaView / MeshLab reject 2-vertex ``f`` files
    outright (empty dataset).  One ``l`` chain per SPAN (not per
    spline): consecutive spans share their endpoint so an intact
    spline still reads as a continuous curve, while a skipped span
    (missing handles) leaves a real gap instead of a fabricated
    straight bridge — matching the VTK writer's spans-as-separate-
    cells behaviour.
    """
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Geodesic Spline Export\n")
        v_offset = 1
        for spline_idx, spans in enumerate(spline_points_list):
            f.write(f"g spline_{spline_idx}\n")
            for span in spans:
                if len(span) < 2:
                    continue
                for pt in span:
                    f.write(f"v {pt[0]:.8f} {pt[1]:.8f} {pt[2]:.8f}\n")
                chain = " ".join(
                    str(i) for i in range(v_offset, v_offset + len(span)))
                f.write(f"l {chain}\n")
                v_offset += len(span)


def write_csv(spline_points_list, stream):
    """Writes curve points as CSV rows to *stream*.

    One ``x, y, z`` row per point; a NaN row between splines.  Adjacent
    spans share their endpoint bit-for-bit; when they do NOT (a span
    was skipped for missing handles), an extra NaN row marks the gap so
    downstream parsers don't fabricate a straight bridge across it.
    See the module docstring for the full format contract.
    """
    first_spline = True
    for span_pts_list in spline_points_list:
        if not first_spline:
            # Break between splines
            print(NAN_LINE, file=stream)
        first_spline = False

        prev_end = None
        for span in span_pts_list:
            if len(span) == 0:
                continue
            if prev_end is not None and not np.allclose(
                    prev_end, span[0], atol=1e-9):
                print(NAN_LINE, file=stream)
            for pt in span:
                print(format_point(pt), file=stream)
            prev_end = span[-1]


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


_MESH_EXTS = ('.vtk', '.obj', '.ply', '.stl')
_LAYER_CHOICES = ('b', 'o', 'k')


def _same_path(a: str, b: str) -> bool:
    """True when *a* and *b* name the same file on disk.

    ``os.path.samefile`` is authoritative (it resolves hardlinks,
    junctions and 8.3 short names) but needs both paths to exist —
    the export target usually does not.  Fall back to a normalised
    realpath comparison, which handles ``..`` segments, symlinks and
    Windows case-insensitivity.
    """
    try:
        return os.path.samefile(a, b)
    except OSError:
        return (os.path.normcase(os.path.realpath(a))
                == os.path.normcase(os.path.realpath(b)))


def _guard_output_path(out_path: str, inputs: dict[str, str | None]) -> None:
    """Abort before an export write that would clobber its own input.

    ``--obj`` / ``--vtk`` derive the output name from the *session*
    basename alone, so the natural pairing of a session with a
    same-named mesh (``heart.json`` next to ``heart.obj``) resolves the
    output straight onto the mesh.  ``rebuild_mesh_and_nodes`` has
    already read the mesh into memory by then, so the write succeeded
    silently and destroyed the source geometry while still exiting 0.

    Exits 2 (the CLI's input-error code) rather than raising, so the
    failure reads like the other pre-flight checks.
    """
    for label, src in inputs.items():
        if src and os.path.exists(src) and _same_path(src, out_path):
            log.error("refusing to write the export over the %s: %s",
                      label, os.path.abspath(out_path))
            log.error("the output name is derived from the session basename "
                      "— rename the session (or the %s) so they differ", label)
            sys.exit(2)


def main():
    if len(sys.argv) == 1:
        # ``argparse``'s ``--help`` is the canonical reference; this
        # condensed banner is the quick reminder when the user types
        # ``spline_export.py`` with no args at all (a common
        # "wait, what did this take again" moment).
        print("Usage: python spline_export.py <session.json> [layer] "
              "[mesh_override] [--samples N] [--obj | --vtk]")
        print()
        print("Export geodesic spline curves from a JSON session file.")
        print()
        print("Positional:")
        print("  session.json   Path to the JSON session (v1 or v2 schema).")
        print("  layer          Curve layer letter (order-agnostic with mesh_override):")
        print("                   b  blue   — semi-geodesic Bezier")
        print("                   o  orange — fully geodesic de Casteljau (default)")
        print("                   k  black  — scipy interp through node origins")
        print("  mesh_override  Optional .vtk/.obj/.ply/.stl mesh used in place of the")
        print("                 session's mesh_file field.  Lets you export the same")
        print("                 splines onto a different geometry.")
        print()
        print("Options:")
        print("  --samples N    Minimum samples per span (>= 2, default: 60).")
        print("  --mesh PATH    Explicit mesh override (alternative to positional).")
        print("  --obj          Write <basename>.obj instead of CSV to stdout.")
        print("  --vtk          Write <basename>.vtk instead of CSV to stdout.")
        print("  -h, --help     Show the full argparse help and exit.")
        print()
        print("Default output is CSV to stdout — pipe or redirect to capture it.")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description=(
            "Export geodesic spline curves from a JSON session file.  "
            "Positional 'layer' (b/o/k) and 'mesh_override' (path to "
            ".vtk/.obj/.ply/.stl) are order-agnostic and both optional; "
            "the default layer is 'o' (orange / fully geodesic)."),
        epilog=(
            "Examples: "
            "spline_export.py s.json b > b.csv  |  "
            "spline_export.py s.json o --samples 80 --vtk  |  "
            "spline_export.py s.json L_hires.vtk k --obj"))
    parser.add_argument('json_file',
                        help="Path to the JSON session (v1 or v2 schema).")
    parser.add_argument(
        'extras', nargs='*',
        metavar='LAYER_OR_MESH',
        help=("Up to two optional positionals, order-agnostic: a layer "
              "letter ('b'/'o'/'k') and/or a mesh path "
              "(.vtk/.obj/.ply/.stl) to override session's mesh_file."))

    def _samples_type(value: str) -> int:
        # ``int(value)`` itself raises ArgumentTypeError-equivalent if
        # non-numeric.  We add a lower-bound check: ``< 2`` would feed
        # ``np.linspace(0, 1, 0)`` or worse to the curve evaluator and
        # crash with a cryptic shape error far from the input boundary.
        n = int(value)
        if n < 2:
            raise argparse.ArgumentTypeError(
                f"--samples must be >= 2 (got {n})")
        return n

    parser.add_argument('--samples', type=_samples_type, default=60,
                        help="Minimum samples per span (>= 2, default: 60).")
    parser.add_argument('--mesh', dest='mesh_option', metavar='PATH',
                        default=None,
                        help=("Mesh file (.vtk/.obj/.ply/.stl) to use in "
                              "place of the session's mesh_file.  Equivalent "
                              "to passing the same path as a positional "
                              "argument."))
    # Mutually-exclusive output formats.  Without the group, passing
    # both ``--obj --vtk`` silently dispatched to ``--obj`` because of
    # the ``elif`` chain in main() — surprising and undocumented.
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument('--obj', action='store_true',
                           help="Export to .obj file (basename.obj).")
    out_group.add_argument('--vtk', action='store_true',
                           help=("Export to binary legacy .vtk "
                                 "(basename.vtk).  Note this is the OUTPUT "
                                 "format flag — to override the INPUT mesh, "
                                 "pass the .vtk path as a positional or "
                                 "use --mesh."))
    args = parser.parse_args()

    # Disambiguate ``extras`` by extension / value.  A layer letter and
    # a mesh path are both optional and order-agnostic, so the loop
    # routes each into its own slot and errors clearly on duplicates
    # or unrecognised tokens.
    layer = 'o'
    mesh_override = args.mesh_option
    seen_layer = False
    for x in args.extras:
        xl = x.lower()
        if xl.endswith(_MESH_EXTS):
            if mesh_override is not None and mesh_override != x:
                parser.error(
                    f"two mesh overrides given: {mesh_override!r} and {x!r}")
            mesh_override = x
        elif x in _LAYER_CHOICES:
            if seen_layer:
                parser.error(f"layer specified twice: {x!r}")
            layer = x
            seen_layer = True
        else:
            parser.error(
                f"unrecognised argument {x!r}: expected a layer letter "
                f"({'/'.join(_LAYER_CHOICES)}) or a mesh path with one of "
                f"{', '.join(_MESH_EXTS)}")
    args.layer = layer
    args.mesh_override = mesh_override

    data = load_json(args.json_file)
    if args.mesh_override is not None:
        if not os.path.exists(args.mesh_override):
            log.error("override mesh not found: %s", args.mesh_override)
            sys.exit(2)
        log.info("mesh override: %s (replacing session's '%s')",
                 args.mesh_override, data.get('mesh_file', '<unset>'))
        data['mesh_file'] = args.mesh_override
    geo, splines, splines_closed = rebuild_mesh_and_nodes(
        data, session_path=args.json_file)

    compute_fn = {'b': compute_blue, 'o': compute_orange,
                  'k': compute_interp}
    layer_name = {'b': 'blue (semi-geodesic)',
                  'o': 'orange (fully geodesic)',
                  'k': 'black (interpolation)'}

    log.info("layer: %s", layer_name[args.layer])
    log.info("splines: %d", len(splines))
    log.info("samples/span: %d", args.samples)

    all_spline_points = []
    # Single-node splines are *landmarks*, not curves.  The GUI's ``v``
    # export emits them as ``VTK_VERTEX`` cells (``write_vtk`` has taken
    # a ``landmarks`` argument all along); the CLI silently dropped
    # them, so the same session exported through the two paths produced
    # different files.
    landmarks: list[np.ndarray] = []
    for sid, (nodes, closed) in enumerate(zip(splines, splines_closed, strict=False)):
        n_nodes = len(nodes)
        log.info("spline %d: %d nodes, %s",
                 sid, n_nodes, 'closed' if closed else 'open')

        if n_nodes == 1:
            landmarks.append(np.asarray(nodes[0]['origin'], dtype=float))
            all_spline_points.append([])
            continue
        if n_nodes < 2:
            all_spline_points.append([])
            continue

        # Compute curve
        span_pts_list = compute_fn[args.layer](
            geo, nodes, closed, args.samples)
        all_spline_points.append(span_pts_list)

    n_spans_total = sum(len(s) for s in all_spline_points)
    if landmarks and not args.vtk:
        # Only the VTK writer has a cell type for a bare point.
        log.warning("%d landmark (1-node) spline(s) are only representable "
                    "in the VTK output — not written to %s",
                    len(landmarks), "OBJ" if args.obj else "CSV")
    if n_spans_total == 0 and not landmarks:
        # Refuse instead of "succeeding" with nothing: ``write_vtk``
        # returns before opening the file, so a stale export from a
        # previous run stayed on disk and read as the current one, and
        # ``write_obj`` / ``write_csv`` emitted headers-only output —
        # all with "done." and exit 0.
        log.error("nothing to export: no spline in %s has >= 2 nodes "
                  "(layer %r)", args.json_file, args.layer)
        sys.exit(2)

    # The output basename comes from the session, so it can land on the
    # mesh this very run just read (``heart.json`` + ``heart.obj``).
    # ``data['mesh_file']`` is post-override, i.e. the mesh actually
    # loaded; the built-in icosahedron sentinel is not a path and is
    # skipped by the ``os.path.exists`` test inside the guard.
    _inputs = {'session': args.json_file, 'input mesh': data.get('mesh_file')}

    if args.obj:
        obj_path = os.path.splitext(args.json_file)[0] + ".obj"
        _guard_output_path(obj_path, _inputs)
        log.info("exporting to OBJ: %s", obj_path)
        write_obj(obj_path, all_spline_points)
    elif args.vtk:
        vtk_path = os.path.splitext(args.json_file)[0] + ".vtk"
        _guard_output_path(vtk_path, _inputs)
        log.info("exporting to binary legacy VTK: %s (%d landmark(s))",
                 vtk_path, len(landmarks))
        write_vtk(vtk_path, all_spline_points, landmarks=landmarks)
    else:
        write_csv(all_spline_points, sys.stdout)

    log.info("done.")


if __name__ == '__main__':
    main()
