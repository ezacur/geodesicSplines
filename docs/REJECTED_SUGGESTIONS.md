# Rejected Suggestions

A running log of optimisation / refactor ideas that were proposed and
considered, but **not** applied — together with the concrete reason
they were rejected.  Kept here so the same ideas don't recur in
future review passes without new evidence.

A suggestion may move out of this file later if a measurement,
benchmark, or bug report invalidates the original rejection rationale.
When that happens, update the entry rather than removing it: history
is part of the value.

---

## Rendering / VTK

### Replace `IntersectWithLine` occlusion check with `vtkHardwareSelector`

**Proposed**: in [`_is_marker_occluded`](../geo_shoot.py), replace
the ray-cast (`self.geo.locator.IntersectWithLine`) with a Z-buffer
read via `vtkHardwareSelector`, claiming O(1) GPU lookup vs O(log N)
ray traversal and "fails on holes".

This proposal has resurfaced under different framings (dirty topology,
noisy 3D scans).  The analysis below covers all of them.

**Rejected because**:
- The "fails on holes" premise is mechanically wrong.
  `vtkStaticCellLocator.IntersectWithLine` tests against actual face
  geometry.  A hole = empty space with no face to intersect → method
  returns "no hit" → "not occluded".  That is the correct answer for
  a hole; nothing "slips through" because there is no face for the
  ray to slip around.
- O(log N) vs O(1) is a misleading comparison.  For N = 240k faces,
  log N ≈ 18 native operations.  Negligible compared with the Python
  overhead around the call.
- `vtkHardwareSelector` requires an extra render pass, behaves
  differently across VTK versions, and has known issues in offscreen
  contexts.  Trades a robust mechanism for one with more quirks and
  no measured speedup.
- Pathological cases for `IntersectWithLine` (silhouette edges,
  non-2-manifold triangles, mis-oriented normals) are **not** fixed by
  `vtkHardwareSelector` — Z-buffer reads have their own pathologies
  (Z-fighting near silhouettes, off-by-one at pixel edges).
- For noisy 3D scans the actual problem is normal-field instability,
  which the codebase already addresses via the smoothing pipeline at
  [`_smooth_face_normals_cotangent`](../geodesics.py).  The occlusion
  test is downstream of that and not the right place to compensate
  for upstream geometry quality.

**Re-open if**: a real reproduction surfaces (mesh + camera + marker
position + screenshot) where `IntersectWithLine` returns the wrong
occlusion answer *and* `vtkHardwareSelector` returns the right one
on the same input.

**Measured (2026-05-31)**: micro-benchmarked `IntersectWithLine`
camera→marker rays on fandisk (12.9k faces) — **2.1 µs/call**.  The
check fires at most once per frame (only the single hovered marker), so
its cost is **~0.002 ms/frame** (~0.01 % of a 60 fps budget).  Confirms
"negligible": a GPU Z-buffer read cannot be meaningfully faster than a
2 µs op and would add a whole render pass.

### Replace cone arrows with `vtkGlyph3D`

**Proposed**: in [`gizmo.py:_update_handle_arrow`](../gizmo.py),
let the GPU instance + rotate the cone via `vtkGlyph3D` (passing one
point + one direction vector per handle) instead of computing the
Rodrigues rotation in Python and uploading transformed points each
frame.

**Rejected because**:
- A transform cache already exists in [`_update_handle_arrow`](../gizmo.py).
  `np.dot(tpl_pts * scale, R.T)` is skipped entirely when direction +
  scale + hover state are unchanged.
- The cone is only ~30 vertices.  Even on cache miss, the matrix
  multiply is sub-microsecond — not the bottleneck.
- The cone orientation depends on the **last segment of the geodesic**
  (`path[-1] - path[-2]`), not a simple "handle direction".
  A glyph approach needs to keep that derivation, which adds
  complexity without removing it.
- No profiling shows arrow rendering as a hot path during drag.

**Measured (2026-05-31)**: micro-benchmarked the cone transform
(`_rotation_x_to_jit` + `np.dot`) — **1.7 µs per cache-miss**, 0.66 µs
for the cache-hit comparison; the cone template is 9 vertices
(`resolution=8`), not 30.  Worst case is a camera drag with
screen-fixed sizing, where the scale changes every frame so every arrow
misses: even then it is **0.02 ms/frame at 6 nodes, 0.10 ms at 30, and
0.34 ms at 100** (2 arrows/node) — all far under a 6.9 ms (144 fps)
frame.  A `vtkGlyph3D` rewrite would shave a fraction of a sub-ms cost
while still needing the geodesic-tangent derivation.  Confirms "not the
bottleneck".

### Float32 for `V` / `F` in shared memory

**Proposed**: store `V_c` (vertices) as `float32` in the shared
memory blocks at [_SpanWorkManager.__init__](../span_workers.py)
(the manager lived in ``geo_splines.py`` when this was proposed)
to halve memory bandwidth and double L1/L2 effective cache.

**Rejected because**:
- `potpourri3d`'s C++ bindings expect `float64` arrays.  Workers
  would have to upcast on every solver call — net zero or worse.
- `_shoot_loop` is precision-sensitive: `float32` accumulates drift
  along long geodesic paths and would visibly degrade the editor's
  exactness guarantee.
- `F` (face indices) is already `int` and could be downcast to
  `int32` (assumes < 2 billion vertices).  This would be safe but
  saves so little memory (~120 KB on a 30k-face mesh) it's not worth
  a change-on-every-mesh-load.

## State / Architecture

### Clear `_didactic_geo_cache` in `_hide_didactic_actors` to prevent a "memory leak"

**Proposed**: in [`_hide_didactic_actors`](../geo_splines.py),
add `self._didactic_geo_cache = None` so toggling didactic mode off
releases the strong references the cache holds (``'refs': (n0, n1,
n0.origin, n0.p_b, n0.path_b, ...)``).  Framed as a memory leak that
"grows indefinitely" if the user toggles didactic on/off repeatedly.

**Rejected because**:
- The premise that the cache grows is wrong.  The dict has exactly
  two slots (``'fast'`` and ``'exact'``) and every write at
  [`_compute_didactic`](../geo_splines.py)
  **overwrites** the slot's previous entry.  Bound: ~12 KB total
  (2 slots × ~5 paths × ~50 points × 24 bytes), regardless of how
  many times the user toggles.
- The "strong refs" tuple pins ``n0``, ``n1`` and their numpy arrays
  — but those are *already* alive as members of
  ``self.splines[sid]`` while they exist in the editor.  Pinning
  them adds zero net memory; the only effect is preventing GC after
  a structural change recycles the IDs.  That hazard is what the
  ``id()``-keyed cache is designed to detect, hence the explicit
  refs comment in [`_compute_didactic`](../geo_splines.py).
- The four code paths that *can* invalidate the cache content
  ([`_restore_snapshot` / `_clear_spline_spans` / `_shift_spline_caches` / `_clear_all_curve_caches`](../geo_splines.py))
  already set it to ``None`` on the structural events that matter
  (active-spline switch, spline clear, full reload).
- A toggle-off clear is a tiny cosmetic improvement (~12 KB freed
  for the duration of the off-state) sold as a leak fix.  Done
  routinely, this kind of "while we're here" addition accretes into
  invalidation-path complexity.  Re-open if profiling on a real
  workflow shows didactic-cache-related growth.

### Command pattern for undo / redo

**Proposed**: replace the snapshot-based undo at
[`_push_undo`](../geo_splines.py) with a Command pattern
(`MoveNodeCommand(node_id, old_pos, new_pos)` etc.) to reduce undo
memory and enable a non-linear history tree.

**Rejected because**:
- Snapshots are already tiny.  Each node persists `(origin: 3 floats,
  tangent: 3 floats)`.  At the configured 50-level cap × 100 nodes
  that is ~30 KB total.  Memory is not a problem.
  *(Update 2026-06-10: snapshots now use the v2 schema — `origin` +
  `p_a` + `p_b`, ~96 bytes/node, i.e. ~2× the figure above.  Still
  far below anything that matters; the rejection stands unchanged.)*
- The diff-restore in
  [`_can_use_diff_restore`](../geo_splines.py) +
  [`_restore_snapshot`](../geo_splines.py) already avoids
  rebuilding unchanged VTK actors — that was the actual perf win
  worth chasing.
- Command pattern would require one class per mutation kind (add,
  insert, delete, drag, close, reopen, load) plus apply / undo
  bookkeeping.  Substantial complexity; non-linear history isn't a
  feature anyone has asked for.

### `jaxtyping` / `Annotated[NDArray, "N, 3"]` shape hints

**Proposed**: annotate every numpy parameter with shape information
via `jaxtyping` or `numpy.typing.Annotated` for better static
documentation.

**Rejected because**:
- Shape annotations are not verified at runtime by mypy or by Python.
  They are documentation in another syntax.
- The same information is already in docstrings (`"(N, 3) surface
  polyline — should already be projected."` at
  [`subdivide_secant_chords`](../geodesics.py)).
- Adds a third-party runtime dependency for stylistic value only.

## Algorithm-level

### KDTree batched query in `compute_endpoint_local`

**Proposed**: at [`compute_endpoint_local`'s `_to_local`](../geodesics.py),
combine `_kdtree.query(p_start)` + `_kdtree.query(p_end)` into a
single `_kdtree.query([p_start, p_end])`.

**Rejected because**:
- scipy's `KDTree.query` is C-coded.  The Python ↔ C transition cost
  for two 1-point queries vs one 2-point query is ~1-2 µs total.
- The path is dominated by the C++ solver
  (`EdgeFlipGeodesicSolver.find_geodesic_path`), which costs
  milliseconds.
- Indistinguishable speedup; adds no clarity.

**Overturned (2026-05-30 — re-profiled and shipped, commit `37c15b8`)**:
the "~1-2 µs" estimate was wrong.  scipy's `KDTree.query` search is
C-coded, but its Python *wrapper* (input validation, `k` / `workers`
handling, output shaping) costs ~tens of µs per call — and that wrapper,
not the search, dominates a 3-point query.  After the `find_face` batch
the two `_to_local` queries became a visible slice; batching them into
one `query([p_start, p_end])` cut `KDTree.query` from 2186 → 1375 calls
(cProfile; the −811 are exactly the per-`_try_solve_on_region` pairs) and measured
a consistent **~4 %** on the worker path (interleaved A/B, both
orderings, fandisk no-locator).  Nearest vertices are identical to the
per-point queries, so it is bit-for-bit output-preserving (parity oracle
`0.000e+00`, both locator regimes).  Lesson: "C-coded, so negligible"
ignored the per-call Python wrapper cost — measure, don't estimate.

### Make the `area < 1e-15` check in `_add_point_local` scale-relative

**Proposed**: in [`_add_point_local`](../geodesics.py), the
post-subdivision area check uses an **absolute** threshold
``area < 1e-15`` to decide a sub-face is degenerate (revert + snap to
nearest existing vertex).  Replace with a **relative** threshold
``area < ε_rel × edge_len_1 × edge_len_2`` so the check stays
meaningful at any mesh scale.  The framing of the proposal: at
nano-scale absolute meshes (e.g. a microchip in metres) every
subdivision would hit the threshold and degrade to vertex-snap.

**Rejected because**:
- The threshold is for catching machine-precision degenerates, not
  small absolute scales.  ``1e-15`` is roughly ``5 × ε_machine`` for
  float64 — the scale at which an area is computationally
  indistinguishable from zero.  A typical-shape triangle at
  micrometre absolute scale has area ``~1e-13`` (*1000× above* the
  threshold), micro-chip CAD in millimetres has ``~1e-7``, and
  fandisk-style normalised meshes have area ``~1e-3``.  None
  approach ``1e-15``.
- The "100% of insertions fail at nano scale" claim is a
  hypothetical for sub-nanometre absolute meshes, which neither
  PyVista nor VTK is typically used with — workflows in those
  domains rescale to unit coordinates before importing.
- The fix is dimensionally cleaner but adds two extra distance
  computations per sub-face (six per insertion) for a scenario
  nobody has reproduced.  The other tolerances in the same function
  (``snap_eps``, ``split_eps``, ``nudge_eps`` — see
  [`_add_point_local`](../geodesics.py)) are *already*
  relative-by-design (barycentric coordinates) — the area check is the
  only absolute one and only fires after ``_add_point_local`` has
  chosen a 1-to-3 subdivision over the cheaper snap / 2-to-4 paths.
- **Confirmed empirically (2026-05-30)**: ran the orange cascade on
  fandisk uniformly scaled by ``s`` (insertion sub-triangle areas scale
  by ``s²``) and counted revert-branch hits.  At scale 1.0 (bbox diag
  7.6) the smallest insertion sub-triangle was ``5.3e-6`` with **0
  reverts**; at 1e-2 (bbox 0.076) min ``5.3e-10``, **0 reverts**; at
  1e-4 (bbox 7.6e-4) min ``6.7e-15``, still **0 reverts**.  The branch
  first fires at scale ~1e-5 (bbox ~7.6e-5, i.e. tens of µm) and only
  goes all-revert at ~1e-7 (sub-µm bbox).  So at any realistic scale
  (bbox ≳ 1e-3 — every normalised mesh and mm/cm CAD model) there is a
  ≥6-order-of-magnitude margin and the absolute threshold cannot
  misfire.
- Re-open with a reproduction: sub-nanometre mesh + spline +
  log line showing every insertion taking the area-degenerate
  branch.  The fix will then be straightforward.

### Ray-cast secant midpoint instead of nearest-point projection

**Proposed**: in [`subdivide_secant_chords`](../geodesics.py),
replace `project_smooth_batch(midpoints)` with a ray-cast along the
chord's average normal.  The current projection can land on the
opposite side of a thin feature ("ear" on a 3D scan).

**Rejected because**:
- The pathological case is real but rare in practice: spans are
  already short after the level-1 geodesic decomposition, and
  midpoints land on adjacent faces in the vast majority of cases.
- "Average normal of a chord" is not well-defined (chord has no
  intrinsic normal).  A correct implementation would need a precise
  normal source — face under midpoint? mean of endpoints' face
  normals? — and ambiguity-handling for ridge-crossing chords.
- The current tolerance + max-depth cap (`mean_edge * 0.01`,
  `max_depth=6`) self-limits the damage from a bad midpoint.
- Re-open if a real-world reproduction (mesh + spline + screenshot)
  shows the artefact in production.

### Replace `pp3d.EdgeFlipGeodesicSolver` with MMP via `pygeodesic`

**Proposed**: integrate `pygeodesic.PyGeodesicAlgorithmExact` (Mitchell-
Mount-Papadimitriou exact algorithm) as either a drop-in replacement or
a fallback for `pp3d.EdgeFlipGeodesicSolver`.  Hypothesis: visible
discontinuities of the orange curve (~1.5e-2 in 3-D, observed at
several `t` values on `20260506_110713.json` + `fandisk.obj`) were
attributable to Edge-Flip converging to non-global local minima for
slightly-perturbed input pairs in the de Casteljau cascade.

**Rejected because**:
- Side-by-side test on the suspect span: pp3d Edge-Flip and MMP
  produced **identical** paths for `path_b`, `path_a`, `path_12` —
  same point count and arclength agreement of `~1e-13`.  Cascade
  outputs at the 8 jump locations across `t∈[0.4, 0.9]` agreed to
  `~1e-11`.  The two solvers are functionally equivalent on this
  geometry; Edge-Flip is **not** finding non-global minima.
- The visible discontinuities are an instance of the discrete-
  geodesic flip-flop already documented in
  [`compute_endpoint_local`](../geodesics.py) — at
  `submesh_subdiv=1` (the configuration under test during this
  analysis; the shipped default is `ORANGE_SUBMESH_SUBDIV = 0`,
  and no committed default has ever been 1) the 1-to-4 submesh subdivision
  is fine enough to expose the flip-flop between two near-equal-
  length edge chains but not fine enough for the discrete geodesic
  to converge to the smooth one.  Solver choice does not matter:
  Edge-Flip and MMP both pick the same edge chain because both
  return a globally optimal polyline on the discrete mesh.  The
  ambiguity is in the mesh, not the algorithm.
- Two stabilisation attempts were tried and reverted:

  1. **Vertex-1-ring enrichment of the projection-hit seed faces**
     in `compute_endpoint_local`.  Did nothing — empirically the
     seeds were already identical across the jump (`set` equality
     at `t=0.41048→0.41049→0.41050`), so the boundary jitter was
     a red herring.
  2. **Bumping `ORANGE_SUBMESH_SUBDIV` from 1 to 2**.  Targeted tests
     at the jumps found at subdiv=1 looked clean (jumps drop from
     `~1.5e-2` to `~1e-5`), but a dense `t`-sweep over the same
     interval showed the flip-flop **redistributes** to new `t`
     values: subdiv=2 produced *more* anomalies (11 vs 2 at subdiv=1)
     totalling a larger fraction of the path length, at ~3.5× the
     per-call cost.  Higher subdivision shifts the flip-flop
     boundaries instead of removing them.

  The visible-cascade-jump remediation therefore has to come from
  somewhere else — chord-bridging in phase 3 already smooths the
  *rendered* polyline (consecutive samples are connected by mesh
  geodesics rather than 3-D chords), so the residual visual artefact
  is bounded by the cascade-sample positional jump itself.  Genuine
  cascade-topology jumps (e.g. `t≈0.759`, `t≈0.7838`) persist or
  grow at subdiv=2 — those are intrinsic and no discretisation
  level fixes them.
- MMP costs O(F) construction per call (no incremental insertion API)
  vs Edge-Flip's reusable solver, so even if it had been better it
  would have been slower for the orange worker's per-`t` calls.
- `pygeodesic` adds a third compiled native dependency (Kirsanov C++
  via Cython) on top of pp3d's geometry-central, doubling the build-
  toolchain surface for contributors.

**Re-open if**: a reproduction surfaces where pp3d Edge-Flip and MMP
disagree on the geodesic path between two arbitrary surface points
on a clean 2-manifold mesh, by more than `1e-6` in arclength or
position.  The `mmp_compare.py`, `worker_fine_sweep.py`, and
`mmp_cascade_jumps.py` analysis scripts (uncommitted, in repo root)
reproduce the original test if needed.

### Heartbeat / timeout for orange workers

**Proposed**: track the last activity per worker pipe and kill spans
that go silent for > N seconds (worker stuck in an infinite loop
inside the C++ solver), via `psutil` or a SIGTERM equivalent.

**Rejected because**:
- The `potpourri3d` solver does not hang in practice — it raises an
  exception or returns `None` on degenerate input.  No reported case
  of a worker stuck.
- [`drain_queue`](../span_workers.py) already detects worker
  death (`BrokenPipeError` / `EOFError`) and the
  [per-phase shutdown](../span_workers.py)
  hardens the cleanup path.  (Both lived in ``geo_splines.py`` when
  this was proposed; they moved to ``span_workers.py``.)
- A watchdog adds cross-platform `psutil` plumbing, false-positive
  risk (a slow span is not a hung span), and per-pipe last-seen
  state.  Re-open with concrete logs if a real hang is seen.

## Platform / packaging

### Make the `submit(int, 0)` worker warmup async to avoid "blocking the UI"

**Proposed**: in [`_SpanWorkManager.__init__`](../span_workers.py)
(then in ``geo_splines.py``),
the loop ``for _ in range(max_workers): self._executor.submit(int, 0)``
forces all worker subprocesses to spawn during construction.  Framed
as blocking the GUI for 1-2 seconds on Windows while ``spawn`` brings
up the children.  The proposal: do the warmup asynchronously on a
secondary thread.

**Rejected because**:
- The premise is wrong.  ``ProcessPoolExecutor.submit()`` is
  non-blocking by design: it appends a work item to the executor's
  internal queue and returns a ``Future`` immediately.  The actual
  ``CreateProcess`` calls happen on the executor's manager thread,
  not the caller's.  The four ``submit(int, 0)`` calls return in
  microseconds, so ``__init__`` does not block the GUI thread.
- What the warmup *does* do is ensure the spawn cost is paid during
  editor startup (where the user expects loading time) rather than
  at the moment of the first orange-curve computation (where they
  expect interactive latency).  That trade-off is intentional —
  see the existing comments at
  the warmup block of [`_SpanWorkManager.__init__`](../span_workers.py).
- "Async pool on a secondary thread" is what Python's executor
  *already does internally* — adding another layer would be
  redundant.
- Re-open with a profiling trace showing GUI freeze attributable
  to ``_SpanWorkManager.__init__`` on the main thread.  The four
  ``submit`` calls themselves cannot freeze it; if a freeze is
  observed, the cause is elsewhere (e.g. the *first* job's result
  being awaited synchronously somewhere downstream).

### Windows shared-memory leak from `atexit`-only cleanup

**Proposed**: replace `multiprocessing.shared_memory` with `mmap`
backed by a `tempfile.NamedTemporaryFile`, on the assumption that on
Windows a parent-process segfault leaks the shared memory blocks
until reboot.

**Rejected because**:
- The premise is wrong on Windows.  `multiprocessing.shared_memory`
  uses `CreateFileMapping` (kernel object).  When the last process
  vanishes — including via segfault — Windows releases the kernel
  object automatically.  No leak across reboots.
- On Linux (`/dev/shm`) the leak does happen; the
  [hardened shutdown](../span_workers.py) covers normal exits and
  KeyboardInterrupt, plus the `weakref.finalize` safety net covers
  garbage collection and interpreter teardown (since 2026-07 it
  receives the executor + shm blocks directly, so it fires usefully
  even when the manager object is already dead).  A hard segfault
  still leaks on Linux but that requires a Monitor process (overkill).
- `mmap` + `tempfile` has its own quirks (Windows file locking;
  POSIX file descriptor inheritance with spawn vs fork) that would
  trade one rare-edge-case leak for several common-case headaches.

## Algorithm-level (continued)

### Rewrite `compute_endpoint_local` in C++ / Cython / Rust via pybind11

**Proposed**: rewrite [`compute_endpoint_local`](../geodesics.py)
in C++ (or Cython, or Rust + PyO3) on the premise that the
"Python ↔ C++ bridge cost" is the bottleneck and that doing the
BFS / submesh extraction / geodesic call entirely in C++ would
take the per-call time from ~25 ms down to **<5 ms**.

**Rejected because**:

- **Premise is wrong about where time goes.** Profiling shows the
  ~25 ms breakdown is roughly:
  * ~15-20 ms inside `pp3d.EdgeFlipGeodesicSolver.find_geodesic_path`
    (already C++).  This is the dominant cost.
  * ~5-10 ms in the Python BFS / set ops / submesh extraction.
  * ~10-100 µs of pybind11 transition cost — negligible, not
    "hundreds of times per second" (the function fires at most
    ~6 calls/sec sustained, gated behind a 150 ms debounce).
- **The 5 ms target is physically implausible.** The C++ solver
  alone already costs 15-20 ms.  Reaching <5 ms would require
  replacing `EdgeFlipGeodesicSolver` itself, which is a research
  problem, not a refactor.
- **Cost / benefit is terrible.** ~600 lines of complex algorithm
  (BFS expansion, submesh extraction, Dijkstra fallback, topology
  insertion) reimplemented in C++; pybind11 + CMake setup;
  cross-platform binary wheels (cibuildwheel × 3 OSes × 4 Pythons);
  loss of the existing Numba JIT story; debugging across the
  Python/C++ boundary; experimentation friction for an editor
  whose value depends on tweakability.  Best-case savings: ~5-8 ms
  of Python overhead per call → ~17 ms total, not <5 ms.
- **Cheaper paths exist for the same speedup target**: Numba-JIT
  the BFS expansion (set → numpy bool mask, ~50 lines), cache the
  submesh extraction across consecutive calls with similar
  `face_region`, hoist `find_face` calls out of the boundary check
  in `_try_solve_on_region`.  These deliver the genuine
  ~5-8 ms savings without any of the build / maintenance pain.

**Re-open if**: someone shows a profile demonstrating that
non-solver Python time exceeds ~10 ms per call AND a working
proof-of-concept beats Numba-JIT'd BFS on a real mesh by ≥3×.

### Batch the boundary-check `find_face` via `project_smooth_batch_with_faces`

**Proposed**: the C++-rewrite entry above lists "batch the per-point
`find_face` in the boundary check" as a cheap, viable speedup.  The
boundary check at the end of
[`_try_solve_on_region`](../geodesics.py) calls
`self.find_face(pt)` once per path point (a Python↔VTK round-trip).
The obvious batch is one vectorised
`project_smooth_batch_with_faces(path)` call + an `np.isin` membership
test against `boundary_faces_global` — the *same* projection kernel
that already seeds the region.

**Rejected because**:
- **It changes the curve.**  Implemented and validated with the
  in-process cascade parity oracle
  ([`tests/benchmark_endpoint_local.py --baseline/--check`](../tests/benchmark_endpoint_local.py)):
  the output diverged by **~2-3e-2** (max abs, fandisk + 6-node closed
  spline) on *both* the locator and no-locator paths.  That is the
  same order as the documented discrete-geodesic flip-flop
  (`~1.5e-2`), and for the same reason: `find_face` (VTK
  `FindClosestPoint` + barycentric validation) and the projection
  kernel (KDTree + analytical projection) disagree on which face a
  point lands on *at submesh boundaries* — exactly the zone the check
  inspects.  Flipping a `'boundary'`/`'ok'` verdict changes the
  escalation path and the final edge chain.  This is the same class
  of "redistributes the flip-flop" non-transparency that got the
  `ORANGE_SUBMESH_SUBDIV` bump reverted.
- A semantics-preserving batch (keeping `find_face` exactly) cannot
  batch the dominant locator call: VTK `FindClosestPoint` has no
  vectorised API, so only the no-locator KDTree path is batchable —
  and that path is the export-only one.
- **Better targets existed and were taken instead.**  The same
  profiling run showed the cost is spread across several ~10-15 %
  buckets, not concentrated in the solver (solver was only ~12-15 %
  of a ~3.9 ms/call mean on the export cascade — note this is the
  `submesh_subdiv=0` regime, lighter than the ~25 ms editor-drag
  figure the C++ entry cites).  Two **bit-for-bit output-preserving**
  vectorisations were applied instead: the candidate-face scan in
  [`_add_point_local`](../geodesics.py) (206→~110 µs/call) and
  the edge-hash loop in
  [`_build_face_adj_buf`](../geodesics.py) (396→~100 µs/call),
  together ~−20 % on `compute_endpoint_local` with zero curve change
  (parity `0.000e+00`, locked by
  [`tests/test_build_face_adj_buf_vectorized.py`](../tests/test_build_face_adj_buf_vectorized.py)).

**Re-open if**: a batch face-classifier is found that returns the
*identical* face as `find_face` for every path point (so the parity
oracle stays at `0.000e+00`), and profiling still shows the
boundary-check `find_face` as a top bucket after the two
vectorisations above.

**Update (2026-05-29 — partially reopened and shipped)**: the
parenthetical above ("only the no-locator KDTree path is batchable —
and that path is the export-only one") was incomplete.  The no-locator
path is **also the orange worker's** path — the worker and the CLI
export both build with `build_locator=False` — where `find_face` was
profiled at **~46 %** of `compute_endpoint_local` (no-locator, fandisk,
via [`tests/benchmark_endpoint_local.py`](../tests/benchmark_endpoint_local.py)).
A **semantics-preserving** batch — [`_find_faces_batch`](../geodesics.py),
which runs the *same* batched `KDTree.query` + candidate arg-min and
returns the bit-identical face per point — satisfies this entry's own
re-open trigger and was shipped (commit `68301fb`): parity oracle
`0.000e+00` on both regimes, ~13 % faster on the worker path, locked by
[`tests/test_find_faces_batch.py`](../tests/test_find_faces_batch.py).
The **originally-proposed** swap to `project_smooth_batch_with_faces`
(a *different* classifier) stays rejected — it still changes the curve
~2e-2.

### Reduce the global-fallback rate by ignoring real mesh-boundary edges in the truncation check

**Proposed**: real-session profiling (34 heart-SSM sessions, meshes
32k–246k faces) showed the dominant cost of `compute_endpoint_local`
is the **full-mesh global solver** reached when all local phases fail
(~28–29 % of calls on the hard sessions; `solver_build` was 87–98 % of
the time, almost all of it full-mesh).  Root cause: these are **open**
surfaces (cut at valve planes, 0.5–5 % of faces on a real boundary),
and the truncation check in
[`_try_solve_on_region`](../geodesics.py) flags a path point on a
*real* mesh-boundary face (`nb < 0`) as possible truncation → escalate
→ but escalation can never resolve a real boundary → exhaust local
phases → fall to the full-mesh solver.  Proposal: treat only
**artificial** submesh boundaries (a neighbour that exists outside the
region) as truncating; ignore real mesh-boundary edges.  Optionally
gate it to dense meshes (`faces > 100k`) to stay exact.

**Rejected because**:
- **Huge speedups, but it changes the curve unpredictably.**  A/B
  measured with a parity oracle (cascade output, flag ON vs OFF) on all
  34 sessions: speedups up to **46×** (`RVN_tricuspide_septum`,
  24.5 s→0.53 s, exact) — but the curve **diverged** on specific
  geometries, worst `RVP_pulmonary_septum` **0.130** (235k faces),
  `RVN_pulmonary_septum` 5.85e-4 (175k), `LVN_240` 0.0167 (32k).  For
  those spans the global full-mesh solver returns a *genuinely
  different* geodesic — i.e. the real-boundary touch was a **legitimate**
  truncation signal there, not an artefact.
- **The density gate is refuted.**  Divergence does NOT vanish on fine
  meshes: the densest mesh tested (235k) showed the *largest* divergence
  (0.130), while a 175k mesh was exact on 5/6 spans.  Divergence tracks
  the span's start/end geometry (the `*_pulmonary_septum` seeds flip on
  multiple meshes), not face count — so no `faces > N` threshold
  separates exact from divergent.
- **"Gate on agreement" defeats the purpose.**  Verifying the fast
  result against the full-mesh oracle per call means paying for the
  full-mesh build anyway — no net speedup.
- 0.130 is far beyond the documented discrete flip-flop noise
  (~1.5e-2); shipping it would visibly move the rendered curve on
  exactly the clinically-interesting septum spans.

**Re-open if**: a *per-call*, *cheap* criterion is found that decides
when suppressing real-boundary escalation yields the same path as the
global solver (parity oracle `< 1e-9`), without computing the global
solve.  The exact-everywhere subset is large (most sessions were
bit-identical with multi-× speedups) — the blocker is detecting the
divergent spans without the oracle.  Probe lived in
`fallback_probe.py` (uncommitted); the per-session A/B numbers are in
the workflow run `validate-fallback-gate`.

### Numba / bool-mask rewrite of `_bfs_advance`

**Proposed**: replace the set-based BFS frontier expansion in
[`_bfs_advance`](../geodesics.py) with a length-`nf` NumPy bool
`visited` mask (optionally a Numba kernel), eliminating the per-ring
Python loop entirely.  Endorsed in the abstract by the "cheaper paths"
note of the C++-rewrite entry above ("Numba-JIT the BFS expansion
(set → numpy bool mask)").

**Rejected because**:
- The cheaper, lower-risk half was taken first: `_bfs_advance` now
  vectorises the neighbour gather (`adj[frontier]` + dedupe) while
  keeping the plain-set interface (commit `b8184bd`).  That alone cut
  its self-time from 0.207 s → 0.075 s (−64 %) and dropped the `bfs`
  profiling bucket to **~4.3 %** of `compute_endpoint_local`
  (no-locator worker, fandisk).
- The full bool-mask variant was then implemented and A/B-measured
  against that vectorised-gather version: **~1.8 %, within run-to-run
  noise** (3.29 vs 3.35 ms mean/call; the runs overlapped — in one
  cycle the bool-mask was *slower*).  Parity oracle `0.000e+00` — it
  was correct, just not faster.  4.3 % is the hard ceiling (the whole
  bucket), so no implementation can exceed it.
- Cost rejected for that noise-level gain: it changes the
  `visited` / `frontier` contract from sets to `(bool-mask, array)`
  across `_bfs_init`, `_bfs_advance`, `_try_solve_on_region` and the
  three phases of `compute_endpoint_local` (phase-B
  `frontier |= frontier_d` → `np.union1d`, `sorted(visited)` →
  `np.flatnonzero`), and forces a rewrite of the `_bfs_advance`
  contract test.

**Re-open if**: profiling shows the `bfs` bucket exceeding ~15 % on
some mesh / regime (a much denser mesh, or a `submesh_subdiv` level
where the BFS dominates), making the ceiling worth chasing.

---

## How to add an entry

When a future review proposes something already debunked here, point
the reviewer at this file and the entry that covers it.  When a new
suggestion is proposed and rejected:

1. State the proposal (what + where in the code).
2. State the rejection reason (measured number, code reference, or
   architectural argument — not opinion).
3. Add a "re-open if" trigger so the rejection is falsifiable.

### Link to symbols, never to line numbers

Write ``[`some_function`](../module.py)`` — **not**
``[...](../module.py#L1234)``.

Line anchors rot on the first refactor that touches the file above
them, and they rot *silently*: the link still resolves, it just points
at unrelated code.  This file had 26 of them and, when audited, **all
26 pointed somewhere wrong** — several off by more than a thousand
lines after the ``span_workers`` extraction.  A reviewer following one
of those lands on a random statement and concludes the entry is stale,
which is exactly the outcome this file exists to prevent.

Naming the symbol costs one grep to follow and never goes out of date;
if the symbol is renamed or deleted, the link becomes *obviously*
wrong rather than quietly wrong.
