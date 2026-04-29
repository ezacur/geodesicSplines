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

**Rejected because**:
- The "fails on holes" premise is wrong. `vtkStaticCellLocator.IntersectWithLine`
  tests against actual face geometry — holes are handled correctly.
- O(log N) vs O(1) is a misleading comparison.  For N = 240k faces,
  log N ≈ 18 native operations.  Negligible compared with the Python
  overhead around the call.
- `vtkHardwareSelector` requires an extra render pass, behaves
  differently across VTK versions, and has known issues in offscreen
  contexts.  Trades a robust mechanism for one with more quirks and
  no measured speedup.

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

---

## How to add an entry

When a future review proposes something already debunked here, point
the reviewer at this file and the entry that covers it.  When a new
suggestion is proposed and rejected:

1. State the proposal (what + where in the code).
2. State the rejection reason (measured number, code reference, or
   architectural argument — not opinion).
3. Add a "re-open if" trigger so the rejection is falsifiable.
