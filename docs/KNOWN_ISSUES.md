# Known Issues

Reproducible-but-unfixed defects, with the evidence gathered so far.
Distinct from [REJECTED_SUGGESTIONS.md](REJECTED_SUGGESTIONS.md), which
records *proposals* that were measured and turned down: entries here are
real problems nobody has root-caused yet.

The point of the file is that a symptom which looks like noise —
a red CI run that goes green on re-run — does not get dismissed twice.

---

## Intermittent native crash in the full test suite

**Symptom.** Running the whole suite occasionally dies with a fatal
native signal instead of a test failure.  Two signatures observed:

```
Windows fatal exception: code 0xc000001d      # ILLEGAL_INSTRUCTION
Fatal Python error: Illegal instruction
...
  geodesics.py in _astar_corridor
  geodesics.py in _dijkstra_corridor
  geodesics.py in compute_endpoint_local
  geodesics.py in eval_cascade_at_t
  spline_export.py in _orange_span_worker
  tests/benchmark_endpoint_local.py in test_orange_cascade_benchmark
```

pytest exits **3** (internal error) in that case.  The second signature
is a plain `Segmentation fault`, exit **139**, same reporting frame.

**Rate.** Roughly **1 run in 15** of the full suite (2 crashes across
~30 measured runs, 2026-07-27).  Both signatures appeared once each.

**What is ruled out.**

- **Not a regression.** Reproduced on `cf82004` — the commit before the
  2026-07-27 audit work started — via a clean `git worktree`:
  `Segmentation fault`, exit 139, 1 run in 8.  Whatever this is, it
  predates the current changes.
- **Not the reported Python frame.** `_astar_corridor` is plain Python
  (no `@njit`): a loop over `scipy.sparse` CSR arrays with `int()`
  conversions and set membership.  There is no instruction there that
  can raise SIGILL.  `faulthandler` reports where the *main thread*
  stood when the signal arrived, which for a fault raised inside a
  native extension (or by state corrupted earlier) is not the fault
  site.
- **Not reproducible in isolation.** `tests/benchmark_endpoint_local.py`
  alone: 0 crashes in 6 runs, both on current code and on `cf82004`.
  The process-pool tests plus the benchmark
  (`test_span_worker_spawn.py` + `test_span_work_manager.py` +
  `benchmark_endpoint_local.py`): 0 in 12.  Only the full suite has
  produced it, which points at state left behind by earlier tests
  rather than at any single test.

**Leading hypothesis (untested).** Native-library state after the
suite has spun up and torn down real `ProcessPoolExecutor` children:
`potpourri3d` (geometry-central, C++), VTK, Numba and Intel MKL all
load into the same process, and MKL/OpenMP thread pools surviving a
`spawn` cycle on Windows are a known source of instability.  No
evidence for this yet — it is where to look, not a conclusion.

**Why it is not "fixed" here.** At a ~7 % per-run rate, telling a
mitigation from luck needs on the order of 40+ runs per arm.  An A/B on
thread-limit environment variables was attempted and abandoned as
underpowered (0/10 on the control arm).  Shipping a retry wrapper or
`OMP_NUM_THREADS=1` without that evidence would convert a visible crash
into an invisible one.

**Next steps for whoever picks this up.**

1. Run the suite under a native debugger / `Application Verifier` (or
   `gdb --args` on Linux) to get the real faulting frame instead of the
   Python-side one.
2. Bisect the *test order*: `pytest -p no:randomly --deselect` halves of
   the suite ahead of the benchmark until the pairing that matters
   shows up.  ~15 runs per candidate set to have any power.
3. Check whether it survives `PYTHONMALLOC=malloc` — if the signature
   changes, it is heap corruption rather than a bad instruction.
4. Confirm or kill the MKL hypothesis by running with
   `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1` for 40+
   runs against a 40+-run control.

**Practical impact today.** CI can go red without anybody having broken
anything.  Re-run before investigating a failure whose log contains
`Fatal Python error` or `Segmentation fault` and no assertion — but
record the occurrence here so the rate stays measured rather than
folklore.
