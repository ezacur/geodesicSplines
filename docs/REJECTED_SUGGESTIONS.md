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

**Proposed**: in [`_is_marker_occluded`](../geo_shoot.py#L816), replace
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
  [geodesics.py:97-117](../geodesics.py#L97-L117).  The occlusion
  test is downstream of that and not the right place to compensate
  for upstream geometry quality.

**Re-open if**: a real reproduction surfaces (mesh + camera + marker
position + screenshot) where `IntersectWithLine` returns the wrong
occlusion answer *and* `vtkHardwareSelector` returns the right one
on the same input.

### Replace cone arrows with `vtkGlyph3D`

**Proposed**: in [`gizmo.py:_update_handle_arrow`](../gizmo.py#L620),
let the GPU instance + rotate the cone via `vtkGlyph3D` (passing one
point + one direction vector per handle) instead of computing the
Rodrigues rotation in Python and uploading transformed points each
frame.

**Rejected because**:
- A transform cache already exists at [gizmo.py:669-676](../gizmo.py#L669-L676).
  `np.dot(tpl_pts * scale, R.T)` is skipped entirely when direction +
  scale + hover state are unchanged.
- The cone is only ~30 vertices.  Even on cache miss, the matrix
  multiply is sub-microsecond — not the bottleneck.
- The cone orientation depends on the **last segment of the geodesic**
  (`path[-1] - path[-2]`), not a simple "handle direction".
  A glyph approach needs to keep that derivation, which adds
  complexity without removing it.
- No profiling shows arrow rendering as a hot path during drag.

### Float32 for `V` / `F` in shared memory

**Proposed**: store `V_c` (vertices) as `float32` in the shared
memory blocks at [_SpanWorkManager.__init__](../geo_splines.py#L674-L685)
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

**Proposed**: in [`_hide_didactic_actors`](../geo_splines.py#L4740),
add `self._didactic_geo_cache = None` so toggling didactic mode off
releases the strong references the cache holds (``'refs': (n0, n1,
n0.origin, n0.p_b, n0.path_b, ...)``).  Framed as a memory leak that
"grows indefinitely" if the user toggles didactic on/off repeatedly.

**Rejected because**:
- The premise that the cache grows is wrong.  The dict has exactly
  two slots (``'fast'`` and ``'exact'``) and every write at
  [geo_splines.py:_compute_didactic](../geo_splines.py#L4962)
  **overwrites** the slot's previous entry.  Bound: ~12 KB total
  (2 slots × ~5 paths × ~50 points × 24 bytes), regardless of how
  many times the user toggles.
- The "strong refs" tuple pins ``n0``, ``n1`` and their numpy arrays
  — but those are *already* alive as members of
  ``self.splines[sid]`` while they exist in the editor.  Pinning
  them adds zero net memory; the only effect is preventing GC after
  a structural change recycles the IDs.  That hazard is what the
  ``id()``-keyed cache is designed to detect, hence the explicit
  refs comment at [geo_splines.py:4964-4969](../geo_splines.py#L4964).
- The four code paths that *can* invalidate the cache content
  ([geo_splines.py:1844, 2592, 2856, 5735](../geo_splines.py#L2592))
  already set it to ``None`` on the structural events that matter
  (active-spline switch, spline clear, full reload).
- A toggle-off clear is a tiny cosmetic improvement (~12 KB freed
  for the duration of the off-state) sold as a leak fix.  Done
  routinely, this kind of "while we're here" addition accretes into
  invalidation-path complexity.  Re-open if profiling on a real
  workflow shows didactic-cache-related growth.

### Command pattern for undo / redo

**Proposed**: replace the snapshot-based undo at
[`_push_undo`](../geo_splines.py#L1535) with a Command pattern
(`MoveNodeCommand(node_id, old_pos, new_pos)` etc.) to reduce undo
memory and enable a non-linear history tree.

**Rejected because**:
- Snapshots are already tiny.  Each node persists `(origin: 3 floats,
  tangent: 3 floats)`.  At the configured 50-level cap × 100 nodes
  that is ~30 KB total.  Memory is not a problem.
- The diff-restore in
  [`_can_use_diff_restore`](../geo_splines.py#L1571) +
  [`_restore_snapshot`](../geo_splines.py#L1590) already avoids
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
  [geodesics.py:1493](../geodesics.py#L1493)).
- Adds a third-party runtime dependency for stylistic value only.

## Algorithm-level

### KDTree batched query in `compute_endpoint_local`

**Proposed**: at [geodesics.py:2032-2033](../geodesics.py#L2032-L2033),
combine `_kdtree.query(p_start)` + `_kdtree.query(p_end)` into a
single `_kdtree.query([p_start, p_end])`.

**Rejected because**:
- scipy's `KDTree.query` is C-coded.  The Python ↔ C transition cost
  for two 1-point queries vs one 2-point query is ~1-2 µs total.
- The path is dominated by the C++ solver
  (`EdgeFlipGeodesicSolver.find_geodesic_path`), which costs
  milliseconds.
- Indistinguishable speedup; adds no clarity.

### Make the `area < 1e-15` check in `_add_point_local` scale-relative

**Proposed**: in [`_add_point_local`](../geodesics.py#L2844), the
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
  (``snap_eps``, ``split_eps``, ``nudge_eps`` at lines 2789-2791)
  are *already* relative-by-design (barycentric coordinates) — the
  area check is the only absolute one and only fires after
  ``_add_point_buf`` has chosen a 1-to-3 subdivision over the
  cheaper snap / 2-to-4 paths.
- Re-open with a reproduction: sub-nanometre mesh + spline +
  log line showing every insertion taking the area-degenerate
  branch.  The fix will then be straightforward.

### Ray-cast secant midpoint instead of nearest-point projection

**Proposed**: in [`subdivide_secant_chords`](../geodesics.py#L1464),
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
  [`compute_endpoint_local`](../geodesics.py#L2328) — at
  `submesh_subdiv=1` (editor default) the 1-to-4 submesh subdivision
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
- [`drain_queue`](../geo_splines.py#L848) already detects worker
  death (`BrokenPipeError` / `EOFError`) and the
  [recently-added](../geo_splines.py#L955) per-phase shutdown
  hardens the cleanup path.
- A watchdog adds cross-platform `psutil` plumbing, false-positive
  risk (a slow span is not a hung span), and per-pipe last-seen
  state.  Re-open with concrete logs if a real hang is seen.

## Platform / packaging

### Make the `submit(int, 0)` worker warmup async to avoid "blocking the UI"

**Proposed**: in [`_SpanWorkManager.__init__`](../geo_splines.py#L1095),
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
  [geo_splines.py:1091-1094](../geo_splines.py#L1091-L1094).
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
  [hardened shutdown](../geo_splines.py#L935) covers normal exits and
  KeyboardInterrupt, plus the existing `weakref.finalize` covers
  interpreter teardown.  A hard segfault still leaks on Linux but
  that requires a Monitor process (overkill).
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

---

## How to add an entry

When a future review proposes something already debunked here, point
the reviewer at this file and the entry that covers it.  When a new
suggestion is proposed and rejected:

1. State the proposal (what + where in the code).
2. State the rejection reason (measured number, code reference, or
   architectural argument — not opinion).
3. Add a "re-open if" trigger so the rejection is falsifiable.
