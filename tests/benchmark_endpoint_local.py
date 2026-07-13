#!/usr/bin/env python
"""Profiling benchmark for ``GeodesicMesh.compute_endpoint_local``.

Two faces:

* **As a script** (``python tests/benchmark_endpoint_local.py [mesh]
  [--nodes K] [--samples N] [--no-locator] [--baseline f] [--check f]``)
  it builds a realistic closed spline over a real mesh, runs the exact
  export cascade in-process, and prints a per-phase time breakdown of
  ``compute_endpoint_local`` (projection / bfs / extract / solver /
  find_face / dijkstra / topology insertion / adjacency / residual).
  Use ``--baseline``/``--check`` to capture and diff the cascade output
  as a parity oracle when changing the geodesic core.

* **As a pytest test** (``test_orange_cascade_benchmark``) marked
  ``benchmark`` and skipped when the heavy deps (vtk / pyvista /
  potpourri3d) or ``fandisk.obj`` are absent — so it never runs in the
  slim CI matrix.  It asserts the cascade is deterministic and finite
  (a cheap correctness net) and prints the breakdown for the record.

Method
======
The cascade is run **in-process** (no ``ProcessPoolExecutor`` — the
monkeypatched timers must see the calls), which fires the same
``compute_endpoint_local`` calls the editor's orange worker fires.
Leaf operations are timed by monkeypatching bound methods + the pp3d
solver class.  Buckets are NON-overlapping by construction (no timed
method contains another), so the bucket times plus a ``residual``
(topology-insertion math, numpy/set glue, boundary-loop overhead)
sum to the total ``compute_endpoint_local`` time.

History
=======
This harness measured the phase breakdown that motivated two exact
(bit-for-bit output-preserving) optimisations to ``geodesics.py``:
vectorising the candidate-face scan in ``_add_point_local`` and the
edge-hash loop in ``_build_face_adj_buf``.  It also caught that the
"batch the boundary-check find_face" idea (swapping ``find_face`` for
``project_smooth_batch_with_faces``) *changes the curve* by ~2e-2 —
the same order as the documented cascade flip-flop — so it was rejected
as a non-transparent optimisation.  Keep this tool to vet future ones.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_MESH = os.path.join(_REPO_ROOT, "fandisk.obj")

# Allow ``python tests/benchmark_endpoint_local.py`` to find the top-level
# modules: when run as a script, sys.path[0] is the tests/ dir, not the
# repo root.  Under pytest the root is already importable, so this is a
# no-op there.
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Heavy deps (vtk via geodesics, pyvista for mesh IO, potpourri3d for the
# solver) are imported lazily so this module imports cleanly in the slim
# CI environment; the pytest test skips when they are unavailable.
try:
    import geodesics
    import spline_export
    from geodesics import GeodesicMesh
    _HAVE_DEPS = True
except Exception:  # pragma: no cover - exercised only in slim CI
    _HAVE_DEPS = False

# ---------------------------------------------------------------------------
# Timing infrastructure
# ---------------------------------------------------------------------------
_TIME: dict[str, float] = defaultdict(float)
_COUNT: dict[str, int] = defaultdict(int)


def _reset_counters():
    _TIME.clear()
    _COUNT.clear()


class _timer:
    """Accumulate elapsed wall time + call count into a named bucket."""

    __slots__ = ("name", "_t0")

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _TIME[self.name] += time.perf_counter() - self._t0
        _COUNT[self.name] += 1
        return False


def _wrap_bound(geo, attr: str, bucket: str):
    """Replace ``geo.attr`` (bound method) with a timed wrapper.

    Setting an *instance* attribute to a plain function shadows the
    class method; ``self.attr(...)`` then calls our wrapper with no
    implicit ``self`` (instance-dict callables are not re-bound), and
    the wrapper forwards to the captured *bound* original.
    """
    orig = getattr(geo, attr)

    def wrapper(*a, **k):
        with _timer(bucket):
            return orig(*a, **k)

    setattr(geo, attr, wrapper)


def _wrap_count_only(geo, attr: str, bucket: str):
    """Count calls to ``geo.attr`` without attributing its time."""
    orig = getattr(geo, attr)

    def wrapper(*a, **k):
        _COUNT[bucket] += 1
        return orig(*a, **k)

    setattr(geo, attr, wrapper)


_ORIG_SOLVER = None


def _install_solver_timer():
    """Wrap pp3d's solver: time construction (O(F)) and path search.

    Captures the *original* class once so re-installing across multiple
    sessions in one process never nests the wrapper (which would
    double-count solver time and grow without bound).
    """
    global _ORIG_SOLVER
    if _ORIG_SOLVER is None:
        _ORIG_SOLVER = geodesics.pp3d.EdgeFlipGeodesicSolver
    orig_cls = _ORIG_SOLVER

    class TimedSolver:
        def __init__(self, V, F):
            with _timer("solver_build"):
                self._s = orig_cls(V, F)

        def find_geodesic_path(self, a, b):
            with _timer("solver_path"):
                return self._s.find_geodesic_path(a, b)

    geodesics.pp3d.EdgeFlipGeodesicSolver = TimedSolver


def install_instrumentation(geo):
    # Leaf operations (non-overlapping — none calls another timed leaf).
    _wrap_bound(geo, "project_smooth_batch_with_faces", "projection")
    _wrap_bound(geo, "_bfs_init", "bfs")
    _wrap_bound(geo, "_bfs_advance", "bfs")
    _wrap_bound(geo, "find_face", "find_face")
    _wrap_bound(geo, "_dijkstra_corridor", "dijkstra")
    # _extract_submesh is a staticmethod; instance-dict shadow still works.
    _wrap_bound(geo, "_extract_submesh", "extract_submesh")
    # Residual decomposition: topology insertion + adjacency + cleanup.
    # None of these calls a timed method (verified: _add_point_local does
    # its own F_buf linear scan, not self.find_face), so buckets stay
    # non-overlapping.
    _wrap_bound(geo, "_add_point_local", "topology_add")
    _wrap_bound(geo, "_build_face_adj_buf", "adj_buf")
    _wrap_bound(geo, "_remove_degenerate_faces", "remove_degen")
    _install_solver_timer()
    # Totals / counts (count-only so their time is not double-attributed).
    _wrap_bound(geo, "compute_endpoint_local", "TOTAL_cel")
    _wrap_count_only(geo, "_try_solve_on_region", "try_solve_attempts")


# ---------------------------------------------------------------------------
# Realistic closed spline construction
# ---------------------------------------------------------------------------
def _farthest_point_sample(V: np.ndarray, k: int, seed: int) -> list[int]:
    """K vertex indices spread across the mesh (euclidean FPS seed)."""
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(V)))
    chosen = [first]
    d = np.linalg.norm(V - V[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(V - V[nxt], axis=1))
    return chosen


def _truncate_geodesic(path: np.ndarray, frac: float):
    """Return ``(point_at_frac, polyline_origin..point)`` by arclength."""
    cum, total = GeodesicMesh.compute_path_lengths(path)
    if total <= 0.0:
        return path[0].copy(), path[:1].copy()
    pt = GeodesicMesh.geodesic_lerp(path, frac, cum, total)
    target = frac * total
    # Per-point cumulative arclength (length N, starts at 0) for masking.
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum_pts = np.concatenate([[0.0], np.cumsum(seg)])
    keep = path[cum_pts <= target]
    if len(keep) == 0:
        keep = path[:1]
    poly = np.vstack([keep, pt[None, :]])
    return pt, poly


def build_closed_spline(geo, k: int, seed: int):
    """Build K nodes with real geodesic handles toward both neighbours."""
    V = geo.V
    idx = _farthest_point_sample(V, k, seed)
    origins = [V[i].astype(float) for i in idx]

    nodes = []
    for i in range(k):
        o = origins[i]
        nxt = origins[(i + 1) % k]
        prv = origins[(i - 1) % k]
        path_next, _ = geo.compute_endpoint_local(o, nxt)
        path_prev, _ = geo.compute_endpoint_local(o, prv)
        if path_next is None or len(path_next) < 2:
            path_next = np.array([o, nxt])
        if path_prev is None or len(path_prev) < 2:
            path_prev = np.array([o, prv])
        p_b, path_b = _truncate_geodesic(path_next, 1.0 / 3.0)
        p_a, path_a = _truncate_geodesic(path_prev, 1.0 / 3.0)
        nodes.append({
            "origin": o,
            "p_a": p_a, "p_b": p_b,
            "path_a": path_a, "path_b": path_b,
        })
    return nodes


def build_tasks(nodes, n_samples: int, closed: bool = True):
    """Replicates compute_orange's task-building loop in spline_export."""
    n = len(nodes)
    n_spans = n if closed else n - 1
    tasks = []
    for i in range(n_spans):
        n0 = nodes[i]
        n1 = nodes[(i + 1) % n]
        ctrl = [
            np.asarray(n0["origin"], dtype=float),
            np.asarray(n0["p_b"], dtype=float),
            np.asarray(n1["p_a"], dtype=float),
            np.asarray(n1["origin"], dtype=float),
        ]
        path_b = np.asarray(n0["path_b"], dtype=float)
        path_a_rev = np.asarray(n1["path_a"], dtype=float)[::-1].copy()
        t_grid = GeodesicMesh.curvature_adaptive_t_vals(ctrl, n_samples)
        tasks.append((ctrl, path_b, path_a_rev, t_grid))
    return tasks


def build_scenario(mesh_path: str, nodes: int, samples: int,
                   seed: int, locator: bool):
    """Load mesh, build a closed spline, return ``(geo, tasks, n_spans)``.

    NB: not named ``setup`` — pytest treats a module-level ``setup`` as a
    legacy xunit setup hook and would call it with no arguments.
    """
    V, F = spline_export._read_mesh_VF(mesh_path)
    geo = GeodesicMesh(V, F, build_locator=locator)
    spline_nodes = build_closed_spline(geo, nodes, seed)
    tasks = build_tasks(spline_nodes, samples, closed=True)
    return geo, tasks, len(tasks)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(n_spans: int, wall_cascade: float):
    total = _TIME["TOTAL_cel"]
    n_cel = _COUNT["TOTAL_cel"]
    leaves = ["projection", "bfs", "extract_submesh",
              "solver_build", "solver_path", "find_face", "dijkstra",
              "topology_add", "adj_buf", "remove_degen"]
    leaf_sum = sum(_TIME[b] for b in leaves)
    residual = max(0.0, total - leaf_sum)

    print("\n" + "=" * 70)
    print("  compute_endpoint_local  —  phase breakdown")
    print("=" * 70)
    print(f"  spans                       : {n_spans}")
    print(f"  compute_endpoint_local calls: {n_cel}")
    if n_spans:
        print(f"  calls / span                : {n_cel / n_spans:.1f}")
    print(f"  _try_solve_on_region calls  : {_COUNT['try_solve_attempts']} "
          f"(= solver attempts incl. phase B/C escalation)")
    if n_cel:
        print(f"  mean time / call            : {1e3 * total / n_cel:.3f} ms")
    print(f"  total time in CEL           : {total:.3f} s")
    print(f"  wall time (in-proc cascade) : {wall_cascade:.3f} s")
    print("-" * 70)
    print(f"  {'bucket':<18}{'time (s)':>10}{'% CEL':>8}{'calls':>9}"
          f"{'us/call':>10}")
    print("-" * 70)

    def line(name, key):
        t = _TIME[key]
        c = _COUNT[key]
        pct = 100.0 * t / total if total else 0.0
        per = 1e6 * t / c if c else 0.0
        print(f"  {name:<18}{t:>10.3f}{pct:>7.1f}%{c:>9}{per:>10.1f}")

    for b in ["projection", "bfs", "extract_submesh", "solver_build",
              "solver_path", "find_face", "dijkstra", "topology_add",
              "adj_buf", "remove_degen"]:
        line(b, b)
    pct_r = 100.0 * residual / total if total else 0.0
    print(f"  {'residual(py glue)':<18}{residual:>10.3f}{pct_r:>7.1f}%"
          f"{'-':>9}{'-':>10}")
    print("-" * 70)
    solver_pct = (100.0 * (_TIME['solver_build'] + _TIME['solver_path'])
                  / total if total else 0.0)
    print(f"  SOLVER (build+path) total   : {solver_pct:.1f}%")
    print("=" * 70)
    print()


def profile(mesh_path, nodes, samples, seed, locator):
    """Instrumented in-process cascade run; prints the breakdown."""
    _reset_counters()
    geo, tasks, n_spans = build_scenario(mesh_path, nodes, samples, seed, locator)
    install_instrumentation(geo)
    spline_export._worker_geo = geo
    t0 = time.perf_counter()
    # _orange_span_worker returns (span_pts, degraded); the profiling
    # harness only cares about the geometry.
    results = [spline_export._orange_span_worker(task)[0] for task in tasks]
    wall = time.perf_counter() - t0
    report(n_spans, wall)
    return results


_BUCKETS = ["projection", "bfs", "extract_submesh", "solver_build",
            "solver_path", "find_face", "dijkstra", "topology_add",
            "adj_buf", "remove_degen"]


def _metrics_dict():
    """Snapshot the current counters into a JSON-friendly metrics dict."""
    total = _TIME["TOTAL_cel"]
    n = _COUNT["TOTAL_cel"]
    buckets = {}
    for b in _BUCKETS:
        buckets[b] = {
            "pct": round(100.0 * _TIME[b] / total, 1) if total else 0.0,
            "us_per_call": round(1e6 * _TIME[b] / _COUNT[b], 1) if _COUNT[b] else 0.0,
            "calls": _COUNT[b],
        }
    leaf = sum(_TIME[b] for b in _BUCKETS)
    resid = max(0.0, total - leaf)
    buckets["residual"] = {"pct": round(100.0 * resid / total, 1) if total else 0.0}
    solver = _TIME["solver_build"] + _TIME["solver_path"]
    return {
        "n_cel": n,
        "mean_ms": round(1e3 * total / n, 3) if n else 0.0,
        "total_s": round(total, 3),
        "solver_pct": round(100.0 * solver / total, 1) if total else 0.0,
        "try_solve_attempts": _COUNT["try_solve_attempts"],
        "buckets": buckets,
    }


def _enable_mesh_cache():
    """Memoise ``spline_export._read_mesh_VF`` so profiling many sessions
    that share a mesh (common — a whole study reuses one anatomy file)
    loads + cleans it once per process instead of per session."""
    if getattr(spline_export._read_mesh_VF, "_memoized", False):
        return
    orig = spline_export._read_mesh_VF
    cache: dict[str, tuple] = {}

    def memo(path, _orig=orig, _cache=cache):
        key = os.path.abspath(path)
        if key not in _cache:
            _cache[key] = _orig(path)
        return _cache[key]

    memo._memoized = True
    spline_export._read_mesh_VF = memo


def run_session(session_path, samples, locator):
    """Profile the orange cascade for a real saved session (.json).

    Loads via ``spline_export.rebuild_mesh_and_nodes`` (the editor's
    exact reconstruction), builds per-span tasks for **every** spline,
    and runs the instrumented in-process cascade.  Returns a metrics
    dict (also captures non-finite output + which fallbacks fired).

    ``locator`` selects the find_face regime: ``False`` is the literal
    CLI-export path (``rebuild_mesh_and_nodes`` builds no locator);
    ``True`` profiles a locator twin over the same V/F (tasks are plain
    arrays, so they are geo-independent) to mirror the interactive editor.
    """
    data = spline_export.load_json(session_path)
    # mesh_file is stored relative to the session dir — resolve it.
    mf = data.get("mesh_file")
    if (mf and mf not in ("__builtin__:icosahedron", "ICOSAHEDRON")
            and not os.path.isabs(mf) and not os.path.exists(mf)):
        cand = os.path.join(os.path.dirname(os.path.abspath(session_path)), mf)
        if os.path.exists(cand):
            data["mesh_file"] = cand

    geo, splines, closed = spline_export.rebuild_mesh_and_nodes(data)

    tasks = []
    for nodes, is_closed in zip(splines, closed, strict=False):
        if len(nodes) >= 2:
            tasks.extend(build_tasks(nodes, samples, closed=is_closed))

    prof_geo = (GeodesicMesh(geo.V, geo.F, build_locator=True)
                if locator else geo)

    _reset_counters()
    install_instrumentation(prof_geo)
    spline_export._worker_geo = prof_geo

    n_nonfinite = 0
    t0 = time.perf_counter()
    for task in tasks:
        res, _degraded = spline_export._orange_span_worker(task)
        if res is not None and res.size and not np.isfinite(res).all():
            n_nonfinite += 1
    wall = time.perf_counter() - t0

    return {
        "session": os.path.basename(session_path),
        "mesh_file": os.path.basename(str(data.get("mesh_file"))),
        "verts": int(len(geo.V)),
        "faces": int(len(geo.F)),
        "locator": bool(locator),
        "n_splines": len(splines),
        "total_nodes": int(sum(len(s) for s in splines)),
        "n_spans": len(tasks),
        "n_nonfinite_spans": n_nonfinite,
        "wall_s": round(wall, 3),
        **_metrics_dict(),
    }


# ---------------------------------------------------------------------------
# pytest entry point (skipped in slim CI)
# ---------------------------------------------------------------------------
# Apply the ``benchmark`` marker when running under pytest, but stay
# importable as a plain script even if pytest is not installed.
try:
    import pytest as _pytest
    _benchmark = _pytest.mark.benchmark
except Exception:  # pragma: no cover
    def _benchmark(fn):
        return fn


@_benchmark
def test_orange_cascade_benchmark():
    import pytest

    if not _HAVE_DEPS:
        pytest.skip("geodesics/vtk not importable (slim environment)")
    pytest.importorskip("pyvista")
    pytest.importorskip("potpourri3d")
    if not os.path.exists(_DEFAULT_MESH):
        pytest.skip(f"benchmark mesh not present: {_DEFAULT_MESH}")

    # Small but representative: closed spline, real geodesic handles.
    geo, tasks, n_spans = build_scenario(_DEFAULT_MESH, nodes=5, samples=21,
                                         seed=0, locator=True)
    assert n_spans == 5

    spline_export._worker_geo = geo
    run1 = [spline_export._orange_span_worker(t)[0] for t in tasks]
    run2 = [spline_export._orange_span_worker(t)[0] for t in tasks]

    assert len(run1) == n_spans
    for a, b in zip(run1, run2, strict=False):
        assert a.shape == b.shape
        assert np.isfinite(a).all(), "cascade produced non-finite samples"
        # The cascade must be deterministic run-to-run.
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="profile compute_endpoint_local")
    ap.add_argument("mesh", nargs="?", default=_DEFAULT_MESH)
    ap.add_argument("--nodes", type=int, default=6)
    ap.add_argument("--samples", type=int, default=33)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-locator", action="store_true",
                    help="build without VTK locator (matches CLI export; "
                         "find_face uses the slower KDTree fallback)")
    ap.add_argument("--baseline", metavar="FILE",
                    help="save per-span cascade output to FILE (.npz)")
    ap.add_argument("--check", metavar="FILE",
                    help="diff per-span cascade output against a --baseline")
    ap.add_argument("--sessions", nargs="+", metavar="JSON",
                    help="profile one or more real saved sessions (.json) "
                         "instead of a synthetic spline")
    ap.add_argument("--json", action="store_true",
                    help="with --sessions: emit one JSON metrics object per "
                         "session on stdout (machine-readable)")
    args = ap.parse_args()

    if not _HAVE_DEPS:
        print("ERROR: geodesics/vtk not importable in this environment.",
              file=sys.stderr)
        return 1

    # --- Real-session mode -------------------------------------------------
    if args.sessions:
        import json as _json
        _enable_mesh_cache()
        for path in args.sessions:
            try:
                m = run_session(path, args.samples, locator=not args.no_locator)
            except Exception as exc:  # noqa: BLE001 - report & continue
                print(_json.dumps({"session": os.path.basename(path),
                                   "error": repr(exc)}))
                continue
            if args.json:
                print(_json.dumps(m))
            else:
                b = m["buckets"]
                print(f"\n{m['session']}  [{m['mesh_file']}  "
                      f"{m['verts']}v/{m['faces']}f  "
                      f"loc={'on' if m['locator'] else 'off'}]")
                print(f"  splines={m['n_splines']} nodes={m['total_nodes']} "
                      f"spans={m['n_spans']} CEL={m['n_cel']} "
                      f"mean={m['mean_ms']}ms nonfinite={m['n_nonfinite_spans']}")
                top = sorted(((k, v['pct']) for k, v in b.items()),
                             key=lambda kv: -kv[1])[:6]
                print("  top buckets: " + ", ".join(f"{k} {p}%" for k, p in top))
        return 0

    print(f"Numba JIT active : {geodesics.HAS_NUMBA}")
    print(f"mesh             : {args.mesh}")
    print(f"locator          : "
          f"{'OFF (export path)' if args.no_locator else 'ON (editor path)'}")
    print(f"nodes / samples  : {args.nodes} / {args.samples}")

    results = profile(args.mesh, args.nodes, args.samples, args.seed,
                      locator=not args.no_locator)

    if args.baseline:
        np.savez(args.baseline, *results)
        print(f"baseline saved: {args.baseline} ({len(results)} spans)")

    if args.check:
        data = np.load(args.check)
        base = [data[k] for k in data.files]
        max_diff = 0.0
        worst = -1
        for i, (a, b) in enumerate(zip(base, results, strict=False)):
            if a.shape != b.shape:
                print(f"  span {i}: SHAPE MISMATCH {a.shape} vs {b.shape}")
                max_diff = float("inf")
                continue
            d = float(np.max(np.abs(a - b))) if a.size else 0.0
            if d > max_diff:
                max_diff, worst = d, i
        print(f"PARITY vs {args.check}: max abs diff = {max_diff:.3e} "
              f"(worst span {worst})")
        print("  -> " + ("IDENTICAL (< 1e-9)" if max_diff < 1e-9
                          else "DIVERGENT" if max_diff != max_diff
                          or max_diff > 1e-6
                          else "within 1e-6 (cascade-stable)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
