# SPDX-License-Identifier: Apache-2.0
"""
geo_splines.py — Geodesic Spline Editor (interactive UI).

This module hosts ``GeodesicSplineApp``, the multi-spline editor that
sits on top of ``MidpointShooterApp`` (geo_shoot.py) and adds:

  - Multi-node spline chains and closed loops.
  - Three parallel curve layers (interp / blue Bezier / orange fully
    geodesic) computed at increasing accuracy and cost.
  - Background workers (``_SpanWorkManager``) for the orange layer.
  - Snapshot-based undo/redo with differential restoration.
  - JSON save / load and CLI entry point.
  - Imported guide polylines (Ctrl+X to load, X to toggle).
  - Hold-to-show node-index labels (key 'n').

For the user-facing description (interaction model, curve layers,
save/load format, dependencies) see README.md and userManual.md; for
the internals (geodesic algorithms, worker pipeline, performance notes)
see docs/ARCHITECTURE.md.  Those are the canonical references and this
module avoids duplicating them to prevent rot.

CLI usage
---------

::

    python geo_splines.py                                 # default mesh
    python geo_splines.py <mesh.{obj,ply,stl,vtk}>        # mesh only
    python geo_splines.py <session.json>                  # session + its own mesh
    python geo_splines.py <session.json> <mesh.{...}>     # session + override mesh

The fourth form replaces the session's ``mesh_file`` reference with
the explicit second argument — useful for inspecting the same splines
against a different geometry (registered counterpart, higher-res
resampling).  Splines store 3-D positions, not vertex indices, so
they re-project onto the alternate surface at load time.

Pass ``-h`` / ``--help`` for the same usage block from the command
line.  See ``_cli_main`` for the canonical reference.

Quick map of the main classes
-----------------------------
``SplineConfig``       Centralised numeric / visual constants.
``_SpanWorkManager``   ProcessPoolExecutor + per-span ``mp.Pipe``.
``GeodesicSplineApp``  Subclass of ``MidpointShooterApp`` — overrides
                       ``_on_press``, ``_on_move``, ``_finalize_release``,
                       ``_try_hit_marker``, ``_fire_debounce``,
                       ``_setup_interaction``, ``_on_poll_timer``,
                       ``_print_help``.
"""

from __future__ import annotations

import glob
import json
import logging
import multiprocessing as mp
import os

# Disable the Intel Fortran runtime's Win32 Console Control Handler
# **before** anything that loads MKL (numpy / scipy / potpourri3d).
# When Anaconda's scientific stack imports MKL, ``libifcoremd.dll``
# installs its own ``SetConsoleCtrlHandler`` ahead of the Python
# signal machinery — that handler runs on Ctrl+C, prints the
# unreadable ``forrtl: error (200)`` traceback to stderr and calls
# ``abort()``, killing the process before Python can raise
# ``KeyboardInterrupt`` or run any ``finally`` cleanup.  With four
# orange workers each holding their own MKL load, the user sees four
# interleaved tracebacks and the terminal hangs (parent never gets
# its KeyboardInterrupt either, because its own MKL handler aborted
# too).
#
# Setting this env var **before MKL is dlopen'd** instructs the
# Intel Fortran runtime to skip installing the Console Control
# Handler entirely; Python's ``signal.signal(SIGINT, SIG_IGN)`` in
# the worker initialiser then becomes the only active handler in
# children, and the parent's normal ``KeyboardInterrupt`` path takes
# over.  Workers inherit ``os.environ`` via ``spawn`` so this single
# assignment covers both ranks.  Harmless on non-Intel toolchains —
# the var is read only by ``libifcoremd.dll``.
os.environ.setdefault('FOR_DISABLE_CONSOLE_CTRL_HANDLER', '1')

import sys
import tempfile
import time
import weakref
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk
from scipy.interpolate import splev, splprep

from geo_shoot import MidpointShooterApp, _closest_seg_on_polyline_2d
from geodesics import HAS_NUMBA, GeodesicMesh
from gizmo import (
    GeodesicSegment,
    safe_remove_actor,
    update_dashed_line_inplace,
    update_line_inplace,
)

# Session-schema handling — validation, JSON-error hints and the
# compact save layout — lives in the light ``session_io`` module
# (stdlib-only) so the CLI exporter can reach it without importing this
# GUI module's pyvista / vtk stack.  Re-exported here for the editor's
# own call sites (``_on_load``, ``_restore_snapshot``, ``_on_save``,
# ``_cli_main``, ``_resolve_mesh``) and for backward compat.
from session_io import (
    _format_session_json,
    _json_decode_hint,
    _validate_session_dict,
)

# Orange-layer background-worker pipeline (worker functions + the
# ``ProcessPoolExecutor`` / shared-memory ``_SpanWorkManager``) lives in the
# light, GUI-free ``span_workers`` module so ``ProcessPoolExecutor`` spawn
# children re-import *it* — not this 7k-line pyvista/VTK editor module — to
# resolve the worker callable by qualified name.  Re-exported here so the
# editor's own call sites and the existing tests keep using
# ``geo_splines._SpanWorkManager`` / ``geo_splines.SpanKey`` etc.  ``SpanKey``
# in particular annotates dict keys / set members throughout this module.
from span_workers import (
    SpanKey,
    _build_chord_geodesic,  # noqa: F401 — re-exported for tests
    _geodesic_decasteljau_worker,  # noqa: F401 — re-exported for tests
    _hierarchical_inner_order,  # noqa: F401 — re-exported for tests
    _phase1_canonical,  # noqa: F401 — re-exported for tests
    _phase2_densify,  # noqa: F401 — re-exported for tests
    _phase3_chord_bridge,  # noqa: F401 — re-exported for tests
    _process_initializer,  # noqa: F401 — re-exported for tests
    _SpanWorkManager,
)

# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
# Every module-level / class-level diagnostic goes through this logger
# instead of bare ``print``.  Default level is WARNING so the console
# stays quiet for end users; set ``GEO_SPLINES_DEBUG=1`` to flip to
# DEBUG for development.
log = logging.getLogger("geo_splines")
if not log.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(_handler)
    log.propagate = False
log.setLevel(logging.DEBUG if os.environ.get("GEO_SPLINES_DEBUG") else logging.INFO)


# ---------------------------------------------------------------
# Numba availability — visible warning when the JIT is absent.
# ---------------------------------------------------------------
# When Numba is missing, the @njit decorator in geodesics.py is a no-op
# and the hot kernels fall back to pure-Python execution (50-2000x
# slower).  The ``@njit`` no-op silently masks this regression — the
# editor still works, but real-time drag becomes unresponsive on any
# non-trivial mesh.  We surface a one-time WARNING at import so users
# notice during the first session instead of mistaking the slowness for
# a different bug.  Skipped inside spawn-mode worker children: the warning
# would otherwise fire once per worker on every session start.
if not HAS_NUMBA and mp.current_process().name == "MainProcess":
    log.warning(
        "Numba not installed — geodesic shooting and projection kernels "
        "fall back to pure Python (~50-2000x slower).  "
        "Install with `pip install numba` for interactive performance.")


# ---------------------------------------------------------------
# Global rendering flags (experimental)
# ---------------------------------------------------------------
# SSAO (Screen Space Ambient Occlusion) darkens crevices under the
# spline, making curves "pop" off the mesh surface.  Trial feature:
# set to True to enable, False to keep the legacy Gouraud look.
# May interact with the depth priority scheme for line actors — try
# both and keep whichever looks better on your mesh.
SSAO_ENABLED: bool = False


# ---------------------------------------------------------------
# Built-in mesh sentinel
# ---------------------------------------------------------------
# A reserved string used as ``mesh_file`` in JSON sessions to indicate
# the in-memory icosahedron demo mesh.  The prefix ``__builtin__:``
# makes accidental collisions with real filenames impossible (the
# legacy plain ``"ICOSAHEDRON"`` value is still accepted on load for
# backwards compatibility; new saves always use the prefixed form).
BUILTIN_ICOSAHEDRON: str = "__builtin__:icosahedron"
_LEGACY_ICOSAHEDRON: str = "ICOSAHEDRON"  # accepted on load only

# Default mesh used by the CLI when no argument is given.  Falls back
# to the in-memory icosahedron if the file is not present.
DEFAULT_MESH_FILENAME: str = "fandisk.obj"


# ---------------------------------------------------------------
# HUD strings
# ---------------------------------------------------------------
# Centralised string table for HUD messages.  All call sites resolve
# through ``_t(key, **kw)`` so the wording lives in exactly one place.
# A previous version of this module shipped a parallel Spanish table
# selected by ``GEO_SPLINES_LANG``; the project is English-only now,
# so the i18n dispatch (and its env var) was removed.
_HUD_TEXTS: dict[str, str] = {
    "ready": "READY",
    "dragging": "DRAGGING {marker}",
    "snap_vertex": "SNAP -> vertex {idx}",
    "snap_edge": "SNAP -> edge {va}-{vb} t={t:.2f}",
    "refined_exact": "REFINED (EXACT)",
    "node_inserted": "NODE INSERTED",
    "node_inserted_interp": "NODE INSERTED (INTERP)",
    "hover_stale": "SPLINE CHANGED — move the cursor to re-aim, then double-click",
    "span_no_handles": "SPLINE {sid}: {n} SPAN(S) HAVE NO CURVE (handles unsolvable)",
    "handler_failed": "{name} FAILED — see the console log",
    "loop_closed_break": "LOOP CLOSED + BREAK",
    "loop_opened": "LOOP OPENED",
    "break_removed": "BREAK REMOVED",
    "new_spline_started": "NEW SPLINE STARTED",
    "nothing_to_undo": "NOTHING TO UNDO",
    "nothing_to_redo": "NOTHING TO REDO",
    "undo": "UNDO",
    "redo": "REDO",
    "saved": "SAVED {n} nodes -> {fname}",
    "save_failed": "SAVE FAILED: {err}",
    "loaded": "LOADED {n} nodes from {fname}",
    "load_failed": "LOAD FAILED",
    "load_failed_version": "LOAD FAILED: unknown version",
    "load_failed_format": "LOAD FAILED: invalid format",
    "load_failed_json": "LOAD FAILED: invalid JSON at line {line} col {col}: {msg}{hint}",
    "computing_orange": "COMPUTING ORANGE {done}/{total}",
    "orange_done": "ORANGE DONE",
    "orange_rebuilt": "ORANGE REBUILT",
    "geodesic_fallback": "GEODESIC FALLBACK on span {sid}:{i}",
    "shoot_truncated": "GEODESIC TRUNCATED at non-manifold vertex (mesh defect)",
    "gizmo_opacity": "GIZMO OPACITY {pct}",
    "orange_fullmesh_on":  "ORANGE: FULL-MESH ON (slower, no submesh artifacts) — press R to rebuild",
    "orange_fullmesh_off": "ORANGE: FULL-MESH OFF (fast submesh) — press R to rebuild",
    "guides_loaded":     "GUIDES LOADED ({n_files} file(s), {n_segs} segments)",
    "guides_load_failed": "GUIDE LOAD FAILED: {fname}: {err}",
    "guides_empty":      "GUIDE LOAD FAILED: {fname} has no line cells (cell type 3)",
    "guides_on":         "GUIDES ON",
    "guides_off":        "GUIDES OFF",
    "guides_none":       "NO GUIDES LOADED — use Ctrl+X to import",
}


# Surplus node-label actors kept alive before ``_ensure_node_labels``
# trims the pool.  Large enough that ordinary add / remove churn never
# destroys an actor, small enough that the pool cannot silently grow to
# the largest session ever opened in this process.
_NODE_LABEL_POOL_SLACK: int = 32


def _t(key: str, **kwargs) -> str:
    """Resolves a HUD string by key with optional ``str.format`` kwargs.

    Returns the template unchanged if ``kwargs`` is empty.  Falls back
    to the bare key (rather than crashing) when a format placeholder
    is missing from the supplied kwargs — useful so a typo in the
    caller doesn't break the HUD update.
    """
    template = _HUD_TEXTS.get(key, key)
    if not kwargs:
        return template
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        return template




# ---------------------------------------------------------------
# Curve layer identification
# ---------------------------------------------------------------
# A small enum-like namespace.  Kept as plain strings (not ``Enum``)
# because the values are also dict keys in ``self._layer_visible``
# and the JSON-style ``curve_hover_info`` payload — switching them to
# ``Enum`` would force string conversions at every read site.
class LayerKind:
    BLUE: str = 'blue'
    ORANGE: str = 'orange'
    INTERP: str = 'interp'


@dataclass
class _CurveHoverItem:
    """One visible polyline indexed by curve-hover detection.

    Replaces the historical 6-element tuple plus ``i = -1`` sentinel for
    the interpolation layer.  ``span_idx`` is ``None`` for the interp
    layer (one polyline per spline) and an integer for the blue/orange
    layers (one polyline per span).
    """
    layer: str
    sid: int
    span_idx: int | None
    start: int            # offset into the batched 2-D screen buffer
    n_pts: int
    pts_3d: np.ndarray    # shared reference, not a copy


@dataclass
class SplineConfig:
    """Centralized spline editing tokens and thresholds."""
    # Bézier curve sampling — base resolution before curvature & secant
    # refinement.  ``adaptive_samples`` in geodesics.py converts a span's
    # control-polygon length into a sample count via
    # ``n = poly_len / RESOLUTION + 1``, clamped to ``[MIN, MAX]``.
    # The previous defaults (RESOLUTION=0.5, MAX=60) capped a typical
    # span at ~30 samples, leaving the blue layer visibly polygonal in
    # smooth regions where neither the curvature refiner nor the secant
    # subdivision had a feature to refine on.  Tightened so the base
    # density is enough for visual smoothness; consolidation cost rises
    # by a few ms per span (still dominated by ``compute_endpoint_local``,
    # which is independent of sample count).
    ADAPTIVE_SAMPLING: bool = True   # curvature-aware non-uniform t distribution
    RESOLUTION: float = 0.2
    MIN_SAMPLES: int = 16
    MAX_SAMPLES: int = 200

    # Secant chord subdivision — eliminates chords that cut through mesh ridges
    SECANT_TOL_FACTOR: float = 0.01   # fraction of mean edge length
    SECANT_MAX_DEPTH: int = 6         # max recursive splits (6 → 64× local)

    # LOD during drag: historically these divided the resting sample
    # count by 3-4x to "save time".  In practice drag time is dominated
    # by ``project_smooth_batch`` (Numba-JIT, ~µs per sample), and the
    # only expensive operator — ``compute_endpoint_local`` (~25 ms) —
    # is already skipped during drag (path_12=None forces the Euclidean
    # middle segment).  Keeping the divisors only degraded the visual
    # quality of the drag preview (handles with short poly_length hit
    # the floor at 5 samples → visibly polygonal) without saving any
    # perceptible CPU.  All set to 1 so drag uses the same density as
    # the consolidated curve; the cheap LOD switch is *what* operator
    # runs (no path_12), not *how many* samples.
    DRAG_RESOLUTION_FACTOR: float = 1.0
    DRAG_MIN_DIVISOR: int = 1
    DRAG_MAX_DIVISOR: int = 1

    # Geometry
    HANDLE_FRACTION: float = 1 / 3
    INITIAL_H_FRACTION: float = 0.05    # h_length = diag * this
    NORMAL_ALIGN_THRESHOLD: float = 0.9  # dot(ref, n) above which ref is swapped

    # Visuals
    SPAN_COLOR_HEX: str = '#a0a0b8'
    SPAN_LINE_WIDTH: int = 2
    STITCH_SKIP_PX: float = 3.0
    # Delay after the cursor stops moving before the stitch line is
    # recomputed with the exact (topology-inserted) endpoint geodesic
    # instead of the vertex-snapped fast path.  ~25 ms compute via
    # ``compute_endpoint_from_origin`` — small enough to feel instant
    # once the cursor settles, large enough that continuous motion never
    # triggers it.
    STITCH_EXACT_DEBOUNCE_SEC: float = 0.15

    # Drag preview: lighter / thinner spans while dragging (before debounce)
    SPAN_DRAG_COLOR_HEX: str = '#88bbff'
    SPAN_DRAG_LINE_WIDTH: int = 1
    SPAN_DRAG_OPACITY: float = 0.6

    # Fallback indicator: spans whose geodesic degraded to a straight line
    # (cross-component, solver failure) are repainted in saturated red so
    # the user notices a degraded result instead of trusting a phantom curve.
    SPAN_FALLBACK_COLOR_HEX: str = '#ff2020'

    # Fully geodesic de Casteljau (background, ~4-7s per span)
    # 33 = 2^5 + 1 — gives 5 clean binary-subdivision levels (2, 3, 5, 9,
    # 17, 33 points).  Endpoints are already known (node origins) so the
    # worker actually computes 31 points per span.
    GEO_SAMPLES: int = 33
    GEO_COLOR_HEX: str = '#ff8800'           # orange (final, consolidated)
    GEO_COLOR_COMPUTING_HEX: str = '#b85a00' # dimmer orange while computing
    GEO_LINE_WIDTH: int = 3
    GEO_OPACITY: float = 1.0
    MAX_GEO_WORKERS: int = 4        # max concurrent background processes
    # Dashed rendering while the span is still computing: only the
    # odd-indexed segments are drawn, creating a visual "dashes" pattern
    # that tells the user the curve is still being refined.  Disable for
    # a solid-curve-with-dimmer-color look.
    GEO_DASHED_WHILE_COMPUTING: bool = True

    # Orange post-densification — eliminates the visible mismatch
    # between the orange polyline and the didactic point trajectory.
    # Two phases run inside the worker after the GEO_SAMPLES samples
    # are computed:
    #   Phase 2 (cascade densification): for every adjacent sample
    #       pair whose chord deviates beyond ``ORANGE_SUBDIV_TOL_FACTOR``
    #       of mean edge length, evaluate the cascade at the midpoint
    #       *t* and insert that point.  Recursive up to
    #       ``ORANGE_SUBDIV_MAX_DEPTH``.  The deviation criterion is
    #       selected by ``ORANGE_DEVIATION_MODE``:
    #         'cascade' → measure |chord_mid − cascade(t_mid)|.  Honest
    #             metric (deviation from the true curve).
    #         'surface' → measure |chord_mid − project(chord_mid)|.
    #             Default.  Cheaper (no extra cascade evaluation per
    #             chord) but only catches mesh-piercing chords, not
    #             curve drift on flat regions where the cascade still
    #             curls.
    #   Phase 3 (geodesic chord-bridging): each consecutive sample pair
    #       in the densified polyline is connected by an exact mesh
    #       geodesic (``short_geodesic`` if the endpoints are in
    #       adjacent triangles — fast — else ``compute_endpoint_local``).
    #       Disabled by ``ORANGE_CHORD_BRIDGING = False``.
    ORANGE_DEVIATION_MODE: str = 'surface'   # 'surface' (default) | 'cascade'
    ORANGE_SUBDIV_TOL_FACTOR: float = 0.01   # fraction of mean edge length
    ORANGE_SUBDIV_MAX_DEPTH: int = 6         # recursion cap for densification
    ORANGE_CHORD_BRIDGING: bool = True       # phase-3 short-geodesic polyline
    # When True, the worker's level-2 / level-3 cascade geodesics
    # (``solver(b01, b12)``, ``solver(b12, b23)``, ``solver(c0, c1)``)
    # use ``compute_endpoint`` (full-mesh insertion) instead of
    # ``compute_endpoint_local`` (submesh-bounded).  Trades ~3-5×
    # per-evaluation cost for elimination of submesh-extraction
    # artifacts — slight changes in the inputs no longer flip the
    # solver between regions whose discrete geodesics diverge.  Off
    # by default; flip on for meshes where the orange curve shows
    # noisy / wandering segments that don't look like genuine
    # cascade-topology changes (verify by comparing the same span
    # with the flag on vs off — if the noise drops dramatically,
    # those jumps were submesh artifacts).  Genuine cascade-topology
    # jumps (where ``c0`` and ``c1`` themselves cross a saddle as
    # ``t`` advances) persist regardless of this flag — no solver
    # swap can fix those.
    ORANGE_USE_FULL_MESH: bool = False

    # Submesh subdivision for the orange worker's geodesic solver.
    # ``compute_endpoint_local`` extracts a small submesh around the
    # cascade endpoints and runs ``EdgeFlipGeodesicSolver`` on it.  On
    # coarse meshes the discrete geodesic the solver returns can
    # diverge from the smooth-surface geodesic — and worse, can flip-
    # flop between two near-equal-length edge chains as the cascade
    # parameter sweeps, producing visible kinks (~cm scale) in the
    # rendered curve.  Subdividing the submesh once (4× faces) gives
    # the solver finer edges to work with; the discrete geodesic
    # converges to the smooth one and the flip-flop disappears.
    # Empirically: a 4.5 cm jump on fandisk drops to 0.3 mm at level 1;
    # level 2 gives the same answer (already converged).
    # Cost: ~4× per level (each ``compute_endpoint_local`` call goes
    # from ~25 ms to ~100 ms at level 1).  Done in background workers
    # so it does not block UI.  Set to 0 if you find the orange curve
    # already smooth enough on your meshes and want faster batches.
    ORANGE_SUBMESH_SUBDIV: int = 0           # 0 = off, 1 = 4× faces, 2 = 16×

    # Sample count for the per-key 'v' VTK export (``_on_export_vtk``).
    # Mirrors the ``--samples`` flag of ``spline_export.py`` — using the
    # same value here is the parity contract.  If this matches
    # ``GEO_SAMPLES`` and no orange workers are still active, the live
    # cache is reused to skip recomputation.
    EXPORT_VTK_SAMPLES: int = 20

    # Default parameter value for the didactic scaffold (key 'd').
    # Visible as the slider's initial position and the value used when
    # the scaffold is toggled on with the slider not yet created.  0.5
    # is the canonical "midpoint of the curve" cascade — useful for
    # most teaching contexts; the user can drag the slider in [0, 1].
    DIDACTIC_T_DEFAULT: float = 0.5

    # Interpolation curve (scipy B-spline through nodes, projected to surface).
    # Uses tighter secant subdivision than Bézier layers because the 3D
    # B-spline has no geodesic awareness and can deviate further from the
    # surface between nodes.
    INTERP_COLOR_HEX: str = '#000000'  # black
    INTERP_LINE_WIDTH: int = 2
    INTERP_OPACITY: float = 1.0
    INTERP_MIN_SAMPLES: int = 200      # high base count (short chords)
    INTERP_DRAG_SAMPLES: int = 50      # downsampled count during drag
    INTERP_SECANT_TOL_FACTOR: float = 0.002  # 5x tighter than Bezier layers
    INTERP_SECANT_MAX_DEPTH: int = 8         # 256x local refinement

    # Node-index labels (visible while 'n' key is held).  Black bold
    # text floating just above each node origin; size in screen-pixels
    # so it stays legible at any zoom.  Used for "which sphere is
    # node #4?" debugging during multi-node edits.
    NODE_LABEL_FONT_SIZE: int = 16
    NODE_LABEL_COLOR_HEX: str = '#000000'
    # Pixel offset (dx, dy) applied to every label relative to the
    # node's screen-projected position.  Pushes the text above the
    # red sphere so it does not occlude the node marker.
    NODE_LABEL_OFFSET_PX: tuple = (0, 14)

    # Curve-hover marker (telescopic-sight crosshair on a billboarded
    # circle): radius of the circle / half-extent of the crosshair,
    # expressed as a fraction of camera-to-point distance so the
    # indicator keeps a constant on-screen size at any zoom.  Both
    # diameters are aligned with the camera's view-plane axes (right
    # = horizontal, up = vertical) — the curve tangent is NOT used,
    # so the marker reads as a real telescopic sight regardless of
    # how the underlying curve is oriented in 3D.
    HOVER_MARKER_SCREEN_SCALE: float = 0.006
    # Sample count for the circumference polyline.  32 is enough to
    # look smooth at typical zoom levels; doubling it doubles the
    # geometry written on every camera refresh.
    HOVER_MARKER_CIRCLE_SAMPLES: int = 32
    # Independent line widths so the circle is visibly "the frame"
    # and the crosshair reads as a thinner aim guide.
    HOVER_MARKER_CIRCLE_LINE_WIDTH: int = 2
    HOVER_MARKER_CROSS_LINE_WIDTH: int = 1

    # Imported guide polylines (key Ctrl+X to load, key X to toggle).
    # Rendered behind every spline layer so the user's actual curves
    # remain visually dominant; ``GUIDE_OPACITY`` is the alpha applied
    # to the line segments (low transparency = high alpha).
    GUIDE_COLOR_HEX: str = '#00aa00'   # green
    GUIDE_LINE_WIDTH: int = 3
    GUIDE_OPACITY: float = 0.1
    # Hold X to preview guides at full opacity; on release, when the
    # previous state was 'hidden', the opacity fades from 1.0 back to
    # ``GUIDE_OPACITY`` over this duration (ease-out quadratic).  Tied
    # to the Master Clock's 50 ms cadence → ~10 frames of animation.
    GUIDE_FADE_DURATION_SEC: float = 0.5

    # Z-depth priority (polygon offset) per visual layer.
    # Lower = closer to camera = drawn on top.  Layering from back to front:
    # mesh wireframe → guides → interp → blue Bézier → orange Bézier → curve hover marker
    DEPTH_GUIDE: float = -3.0
    DEPTH_INTERP: float = -6.0
    DEPTH_BLUE: float = -8.0
    DEPTH_STITCH: float = -8.0
    DEPTH_ORANGE: float = -20.0
    DEPTH_CURVE_HOVER: float = -24.0

    # Derived — computed in __post_init__ so they stay in sync
    DRAG_RESOLUTION: float = field(init=False)
    DRAG_MIN_SAMPLES: int = field(init=False)
    DRAG_MAX_SAMPLES: int = field(init=False)
    SPAN_COLOR: tuple = field(init=False)
    SPAN_DRAG_COLOR: tuple = field(init=False)
    SPAN_FALLBACK_COLOR: tuple = field(init=False)
    GEO_COLOR: tuple = field(init=False)
    GEO_COLOR_COMPUTING: tuple = field(init=False)
    INTERP_COLOR: tuple = field(init=False)
    STITCH_SKIP_PX_SQ: float = field(init=False)

    def __post_init__(self):
        self.DRAG_RESOLUTION = self.RESOLUTION * self.DRAG_RESOLUTION_FACTOR
        self.DRAG_MIN_SAMPLES = max(3, self.MIN_SAMPLES // self.DRAG_MIN_DIVISOR)
        self.DRAG_MAX_SAMPLES = self.MAX_SAMPLES // self.DRAG_MAX_DIVISOR
        self.SPAN_COLOR = pv.Color(self.SPAN_COLOR_HEX).float_rgb
        self.SPAN_DRAG_COLOR = pv.Color(self.SPAN_DRAG_COLOR_HEX).float_rgb
        self.SPAN_FALLBACK_COLOR = pv.Color(self.SPAN_FALLBACK_COLOR_HEX).float_rgb
        self.GEO_COLOR = pv.Color(self.GEO_COLOR_HEX).float_rgb
        self.GEO_COLOR_COMPUTING = pv.Color(self.GEO_COLOR_COMPUTING_HEX).float_rgb
        self.INTERP_COLOR = pv.Color(self.INTERP_COLOR_HEX).float_rgb
        self.STITCH_SKIP_PX_SQ = self.STITCH_SKIP_PX ** 2


class GeodesicSplineApp(MidpointShooterApp):

    def __init__(self, mesh_or_path, mesh_label: str | None = None):
        # mesh_label is what goes into the JSON "mesh_file" field.
        # For file paths it's the path itself; for in-memory meshes
        # (e.g. ICOSAHEDRON) it's the sentinel string.
        self.mesh_label = mesh_label if mesh_label is not None else str(mesh_or_path)
        self.scfg = SplineConfig()
        self.splines: list[list[GeodesicSegment]] = [[]]
        self.splines_closed: list[bool] = [False]
        self.active_spline_idx = 0
        # Bounded undo / redo history.  ``deque(maxlen=...)`` evicts
        # the oldest entry in O(1) when capacity is reached, vs the
        # previous ``list.pop(0)`` which was O(n) on every push past
        # the cap.  Both stacks share the cap.
        self._MAX_UNDO = 50
        self._undo_stack: deque[dict] = deque(maxlen=self._MAX_UNDO)
        self._redo_stack: deque[dict] = deque(maxlen=self._MAX_UNDO)

        # Session-name state used by ``_on_save`` and ``_on_export_vtk``
        # to share a base filename across save / export.  Set when a
        # session JSON is loaded (CLI or ``L`` key) — the JSON's stem
        # becomes the name and persists until another load replaces
        # it.  ``None`` means "no session opened yet" — saves / exports
        # fall back to the legacy ``yyyymmdd_HHMMSS`` timestamp.
        # Both save and export append a numeric ``_NN`` suffix when
        # the file already exists so the original loaded JSON is never
        # silently overwritten.
        self._session_name: str | None = None
        self._prev_active_spline_idx = 0
        self._span_cache: dict[SpanKey, tuple[pv.PolyData, vtk.vtkActor]] = {}
        # Per-span style key (dragging, degraded) — repaints only fire on change.
        self._span_drag_state: dict[SpanKey, tuple[bool, bool]] = {}
        self._geo_span_cache: dict[SpanKey, tuple[pv.PolyData, vtk.vtkActor]] = {}
        # Spans whose geodesic solver fell back to a straight line.  Set
        # by ``_recompute_spans`` and orange-worker drain; consumed by
        # ``_set_span`` / ``_set_geo_span`` to repaint in red.
        self._degraded_spans: set[SpanKey] = set()
        # Node most recently consolidated to exact quality by
        # ``_fire_debounce``.  Lets ``_finalize_release`` skip a redundant
        # second exact ``_recompute_spans`` when the release-time debounce
        # already did it; reset on every fast-preview move so a later
        # release with no fired debounce still consolidates.
        self._consolidated_seg = None
        # Snapshot captured at marker-press time but NOT yet pushed to
        # the undo stack.  Committed by the first actual drag movement
        # (``_commit_pending_drag_undo``); discarded on a release with
        # no movement.  Pushing eagerly at press time flooded the
        # 50-deep undo history with no-op entries (and cleared the redo
        # stack) on every plain marker click.
        self._pending_drag_snapshot: dict | None = None
        # Interpolation curve: one actor per spline (keyed by spline index)
        self._interp_cache: dict[int, tuple[pv.PolyData, vtk.vtkActor]] = {}
        # Per-spline pre-allocated origin buffer + content-fingerprint
        # cache for ``_recompute_interp_curve``.  The buffer avoids
        # the per-frame ``np.array([n.origin for n in nodes])``
        # allocation; the fingerprint cache short-circuits the splprep
        # → splev → project_smooth_batch chain when the origins list
        # is bit-identical to the previous call (typical between
        # back-to-back consolidations or when the user hovers without
        # moving anything).
        self._interp_origins_buf: dict[int, np.ndarray] = {}
        self._interp_result_cache: dict[int, tuple] = {}
        self._last_stitch_screen: tuple = (0.0, 0.0)
        self._stitch_origin_cache: dict | None = None  # prepare_origin cache for last node
        self._stitch_origin_node_id: int = -1           # id() of node that owns the cache
        # Cursor position captured at every mouse-move while a stitch is
        # eligible; read by ``_fire_stitch_exact`` when the Master Clock
        # fires after STITCH_EXACT_DEBOUNCE_SEC of stillness.  None means
        # no refinement is pending (either nothing to refine, or the task
        # already fired and consumed it).
        self._stitch_pending_q: np.ndarray | None = None

        # Imported guide polylines (Ctrl+X loads; hold X to preview at
        # full opacity, release X to toggle hidden / low-opacity).
        # One actor per file so re-imports replace the previous set
        # cleanly via ``_clear_guides`` (``safe_remove_actor`` on each).
        # Parallel lists let the X-key handlers flip every actor in one
        # pass without re-querying the plotter.
        self._guide_pds: list[pv.PolyData] = []
        self._guide_actors: list[vtk.vtkActor] = []
        self._guide_visible: bool = True

        # X hold-to-preview state.  ``_x_hold_was_visible`` captures the
        # logical visibility at the moment the user *first* pressed X
        # (subsequent OS key-repeats are ignored).  ``None`` means the
        # key is not currently held.  Read by the KeyRelease handler to
        # decide whether to hide (was visible) or fade in (was hidden).
        self._x_hold_was_visible: bool | None = None

        # Guides fade-in animation state.  ``_guides_fade_start_t`` is
        # the ``perf_counter`` at which the fade began; ``None`` means
        # no fade is active.  Driven by the Master Clock via
        # ``_tick_guides_fade`` self-rescheduling on the poll timer.
        self._guides_fade_start_t: float | None = None
        # node → spline index map.  ``WeakKeyDictionary`` avoids the
        # ``id()``-recycling hazard a plain ``dict[int, int]`` keyed by
        # ``id(seg)`` would have: when CPython GC's a deleted segment
        # and a new one happens to land at the same address, the dict
        # would silently return the OLD spline index for the new
        # object.  Weak refs let the entry vanish automatically the
        # moment the segment is freed.
        self._node_to_spline: weakref.WeakKeyDictionary[GeodesicSegment, int] = (
            weakref.WeakKeyDictionary())
        self._pre_drag_spline_idx: int | None = None
        self._last_cam_pos: tuple = (0.0, 0.0, 0.0)  # for arrow scale refresh
        # Pre-allocated 4×3 scratch for the cubic-Bezier control points
        # ``[P0, H_out, H_in, P1]`` — reused across every span on every
        # frame of a drag instead of building a fresh Python list per
        # iteration.  ``adaptive_samples`` / ``curvature_adaptive_t_vals``
        # / ``hybrid_de_casteljau_curve`` all accept anything indexable
        # along axis 0 with shape (3,) per row, so the (4, 3) ndarray
        # is a drop-in replacement.
        self._ctrl_scratch = np.empty((4, 3), dtype=float)
        # _work_mgr created after super().__init__ when self.geo is available

        super().__init__(mesh_or_path)
        self._work_mgr = _SpanWorkManager(
            self.geo.V, self.geo.F, self.scfg.MAX_GEO_WORKERS)
        # Pre-compute secant subdivision tolerance from mesh density
        mean_edge = float(np.sqrt(self.geo._face_edge_len2.mean()))
        self._secant_tol = mean_edge * self.scfg.SECANT_TOL_FACTOR
        self.plotter.set_background('white')

        # Experimental SSAO — see SSAO_ENABLED module flag at top of file.
        if SSAO_ENABLED:
            try:
                self.plotter.enable_ssao()  # type: ignore[call-arg]
            except (AttributeError, RuntimeError) as exc:
                # Older PyVista lacks enable_ssao (AttributeError); some
                # OpenGL contexts reject the SSAO render pass at runtime
                # (RuntimeError from VTK).  Both are non-fatal: SSAO is
                # purely cosmetic and the editor works without it.
                log.warning("SSAO unavailable: %s", exc)

        # Resolve z-fighting: lines/points always render on top of solid surfaces
        vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
        vtk.vtkMapper.SetResolveCoincidentTopologyPolygonOffsetParameters(1.0, 1.0)

        self._stitch_pd, self._stitch_actor = self._create_aux_actor(
            kind='line', color='#666666', line_width=1.5, opacity=0.6,
            depth=self.scfg.DEPTH_STITCH, name="stitch_preview")

        # Orange computation HUD: tracks whether we are in the middle of
        # showing a "computing" message so we can flip to "ORANGE DONE"
        # exactly once per batch.  The numeric progress lives in
        # ``_work_mgr`` (``progress()`` returns ``(done, total)``).
        self._orange_hud_active = False
        # Snapshot of ``self.geo._shoot_truncation_count`` from the
        # previous poll tick.  Fresh increments mean ``compute_shoot``
        # bailed out via the non-2-manifold safeguard since we last
        # looked; surface that as a HUD warning so the user knows the
        # geodesic ended short due to a mesh defect.
        self._shoot_truncation_seen: int = int(
            getattr(self.geo, '_shoot_truncation_count', 0))

        # Curve-layer visibility toggles (horizontal row above opacity slider)
        # Orange starts hidden by default (expensive computation, opt-in).
        self._layer_visible = {'blue': True, 'orange': False, 'interp': False}
        self._layer_widgets: dict[str, object] = {}
        self._cb_size = 20
        # Widget layout uses normalized Y (fraction of window height) so
        # checkboxes stay above the slider after resize.  X positions are
        # absolute pixels from the left edge (small, stable).
        self._cb_y_norm = 0.08   # 8% from bottom
        self._cb_x_positions = []  # pixel x per checkbox (fixed)
        defaults = {'blue': True, 'orange': False, 'interp': False}
        for i, (layer, color) in enumerate([
                ('blue', 'blue'), ('orange', 'orange'),
                ('interp', 'black')]):
            x_pos = 11 + i * 22
            self._cb_x_positions.append(x_pos)
            widget = self.plotter.add_checkbox_button_widget(
                lambda v, lyr=layer: self._toggle_layer(lyr, v),
                value=defaults[layer],
                position=(x_pos, 50), size=self._cb_size, border_size=2,
                color_on=color, color_off='grey')
            self._layer_widgets[layer] = widget

        # Help button "?" — toggles a shortcut reference overlay
        self._help_visible = False
        self._help_panel = None
        self._help_x = 80
        self._help_widget = self.plotter.add_checkbox_button_widget(
            self._toggle_help_panel,
            value=False,
            position=(self._help_x, 50), size=self._cb_size, border_size=2,
            color_on='white', color_off='grey')
        self._help_label = self.plotter.add_text(
            "?", position=(self._help_x + 5, 52),
            font_size=7, color='white', shadow=True, name="label_help")

        # Full-mesh orange toggle: when on, ``eval_cascade_at_t`` swaps
        # ``compute_endpoint_local`` for ``compute_endpoint`` in level-2
        # / level-3 calls, eliminating submesh-extraction artifacts at
        # ~3-5× per-evaluation cost.  Toggling does NOT trigger a
        # rebuild — the user re-issues ``R`` when ready.  Visual:
        # orange when on (matches the affected layer), grey when off.
        self._fm_x = self._help_x + 22
        self._fm_widget = self.plotter.add_checkbox_button_widget(
            self._toggle_full_mesh_orange,
            value=self.scfg.ORANGE_USE_FULL_MESH,
            position=(self._fm_x, 50), size=self._cb_size, border_size=2,
            color_on='orange', color_off='grey')
        self._fm_label = self.plotter.add_text(
            "FM", position=(self._fm_x + 2, 52),
            font_size=7, color='white', shadow=True, name="label_fm")

        # Reposition widgets on window resize so they stay above the slider.
        # Capture the tag so ``cleanup`` (via the parent's ``_observer_tags``
        # mechanism) can detach this observer when the same plotter is
        # reused (notebook / repeated-instance flows) — otherwise dead
        # GeodesicSplineApp instances stay alive via the observer's strong
        # reference to ``self._on_window_resize``.
        _vtki = self.plotter.iren.interactor
        self._observer_tags.append(
            (_vtki, _vtki.AddObserver('ConfigureEvent', self._on_window_resize)))

        # Overlay renderer (layer 1) hosts both the curve-hover marker
        # and the node-index labels.  Lives in the same render window
        # as the main scene and shares its camera, so any orbit / zoom
        # is mirrored for free.  Layer-1 renderers draw *after* layer-0
        # with no shared depth buffer — exactly the "always on top, no
        # z-fighting with the mesh" behaviour we want.  Per-overlay
        # visibility (which marker / label is actually drawn) is gated
        # by the consumers' own occlusion logic so the overlay only
        # shows what the user could plausibly need to see.
        rwin = self.plotter.render_window
        assert rwin is not None, "render window not yet created"
        if rwin.GetNumberOfLayers() < 2:
            rwin.SetNumberOfLayers(2)
        self._overlay_renderer = vtk.vtkRenderer()
        self._overlay_renderer.SetLayer(1)
        self._overlay_renderer.SetActiveCamera(
            self.plotter.renderer.GetActiveCamera())
        self._overlay_renderer.SetInteractive(False)
        rwin.AddRenderer(self._overlay_renderer)

        # Curve hover marker — a telescopic-sight overlay built from
        # **two** PolyDatas (different line widths require different
        # actors).  Both live in the overlay renderer (same layer-1
        # treatment as the node labels) so they ignore the main z-
        # buffer; the per-frame orientation is camera right / up,
        # giving a true 2-D "sight" look regardless of curve direction.
        # See ``_orient_hover_marker`` for the geometry update.
        N_circle = self.scfg.HOVER_MARKER_CIRCLE_SAMPLES
        # Circumference: closed polyline of (N+1) points where the
        # last point coincides with the first.  Stable line cell so
        # the refresh path only writes ``.points``.
        circle_pd = pv.PolyData()
        circle_pd.points = np.zeros((N_circle + 1, 3), dtype=float)
        circle_pd.lines = np.concatenate(
            [[N_circle + 1], np.arange(N_circle + 1, dtype=np.int64)]
        ).astype(np.int64)
        circle_actor = vtk.vtkActor()
        circle_mapper = vtk.vtkPolyDataMapper()
        circle_mapper.SetInputData(circle_pd)
        circle_actor.SetMapper(circle_mapper)
        circle_prop = circle_actor.GetProperty()
        circle_prop.SetLineWidth(self.scfg.HOVER_MARKER_CIRCLE_LINE_WIDTH)
        circle_prop.LightingOff()
        circle_actor.SetVisibility(False)
        circle_actor.PickableOff()
        self._overlay_renderer.AddViewProp(circle_actor)

        # Crosshair: two lines (horizontal + vertical) sharing 4 points.
        # Points order: [-right, +right, -up, +up].
        cross_pd = pv.PolyData()
        cross_pd.points = np.zeros((4, 3), dtype=float)
        cross_pd.lines = np.array([2, 0, 1, 2, 2, 3], dtype=np.int64)
        cross_actor = vtk.vtkActor()
        cross_mapper = vtk.vtkPolyDataMapper()
        cross_mapper.SetInputData(cross_pd)
        cross_actor.SetMapper(cross_mapper)
        cross_prop = cross_actor.GetProperty()
        cross_prop.SetLineWidth(self.scfg.HOVER_MARKER_CROSS_LINE_WIDTH)
        cross_prop.LightingOff()
        cross_actor.SetVisibility(False)
        cross_actor.PickableOff()
        self._overlay_renderer.AddViewProp(cross_actor)

        self._curve_hover_circle_pd = circle_pd
        self._curve_hover_circle_actor = circle_actor
        self._curve_hover_cross_pd = cross_pd
        self._curve_hover_cross_actor = cross_actor
        # Pre-allocated point buffers reused on every orient call.
        self._curve_hover_circle_buf = np.empty((N_circle + 1, 3), dtype=float)
        self._curve_hover_cross_buf = np.empty((4, 3), dtype=float)
        # Pre-computed unit-circle samples (cos/sin per angle) — only
        # need to be multiplied by radius + (right, up) basis at
        # render time, never recomputed.
        theta = np.linspace(0.0, 2.0 * np.pi, N_circle + 1)
        self._curve_hover_cos = np.cos(theta)
        self._curve_hover_sin = np.sin(theta)
        # Pre-allocated buffer for batched curve hover projection
        self._curve_hover_3d_buf = np.empty((2048, 3), dtype=float)

        # Curve hover state — stored for future node insertion AND for
        # the camera-orbit re-orientation hook
        # (``_refresh_arrows_on_camera_change`` reads ``_curve_hover_state``).
        self.curve_hover_info: dict | None = None
        self._curve_hover_state: dict | None = None

        # Node-index labels — pool of ``vtkBillboardTextActor3D``s
        # grown as needed when 'n' is pressed.  Format:
        #   * single spline  → ``"3"``            (just the node index)
        #   * multi-spline   → ``"s1:3"``         (spline index + node)
        # All labels carry the same screen-pixel font size + color so
        # the rendering stays consistent regardless of zoom.  Held in
        # a plain list (not WeakRef) because we own the lifetime —
        # attached to the plotter renderer until ``cleanup()`` removes
        # them or the pool is resized down.
        self._node_labels: list[vtk.vtkBillboardTextActor3D] = []
        self._node_labels_visible: bool = False

        # Snap indicator — appears on drag while Shift (vertex) or Ctrl
        # (edge) is held, marking the exact target the drag will land on.
        # Smaller and brighter than the curve-hover marker so it doesn't
        # compete visually but is impossible to miss.
        self._snap_indicator_pd, self._snap_indicator_actor = self._create_aux_actor(
            kind='point', color='gold', point_size=14,
            depth=self.scfg.DEPTH_CURVE_HOVER - 1, name="snap_indicator")
        self._snap_indicator_buf = np.empty((1, 3), dtype=float)

        # Coordinate-edit preview — shown live while the right-double-
        # click coordinate dialog is open and the typed input parses
        # successfully.  Three actors form the preview group:
        #
        #   * ``_coord_preview_actor``        — sphere on the surface,
        #     darker grey, slightly larger.  Marks where the node will
        #     actually land (the projected point).
        #   * ``_coord_preview_input_actor``  — sphere at the literal
        #     typed coordinate, lighter grey, slightly smaller.  Often
        #     floats above the surface; visually communicates the
        #     "before projection" position.
        #   * ``_coord_preview_line_actor``   — thin grey line between
        #     the two spheres.  Its length is the projection distance
        #     and gives the user instant feedback on how far off-
        #     surface their typed point is.
        #
        # All three share the same visibility — toggled together by
        # ``_update_coord_preview`` / ``_hide_coord_preview``.  Depth
        # priority is in front of every other layer (CURVE_HOVER - 2)
        # so the preview is visible even on top of the orange curve.
        depth_coord = self.scfg.DEPTH_CURVE_HOVER - 2
        self._coord_preview_pd, self._coord_preview_actor = self._create_aux_actor(
            kind='point', color='#888888', point_size=11,
            depth=depth_coord, name="coord_preview")
        self._coord_preview_buf = np.empty((1, 3), dtype=float)

        (self._coord_preview_input_pd,
         self._coord_preview_input_actor) = self._create_aux_actor(
            kind='point', color='#bbbbbb', point_size=8,
            depth=depth_coord, name="coord_preview_input")
        self._coord_preview_input_buf = np.empty((1, 3), dtype=float)

        (self._coord_preview_line_pd,
         self._coord_preview_line_actor) = self._create_aux_actor(
            kind='line', color='#888888', line_width=1,
            depth=depth_coord, name="coord_preview_line")

        # --- Didactic visualization (key 'd') ---
        # Toggleable preview of the de Casteljau scaffold for the LAST
        # span of the active spline at t=0.5.  Four geodesic auxiliary
        # lines, all at the same gray + the global handle opacity (the
        # one cycled with 't'):
        #
        #   index 0: path_12      H_out  -> H_in    (level 1 middle)
        #   index 1: path_c0      b01    -> b12     (level 2 first)
        #   index 2: path_c1      b12    -> b23     (level 2 second)
        #   index 3: path_final   c0     -> c1      (level 3, collapses
        #                                           to the orange curve
        #                                           sample at t=0.5)
        #
        # On-demand semantics: while invisible the actors stay empty
        # and ``_compute_didactic`` is not called (``_didactic_dirty``
        # is set so the next toggle ON triggers a rebuild).  Toggle ON
        # triggers a fresh exact compute (~75-125 ms; four
        # ``compute_endpoint_local`` calls).  During node drag the
        # scaffold updates **live** in fast mode (~5-10 ms via
        # Euclidean line + ``project_smooth_batch``, the same trick
        # blue uses for ``path_12`` while dragging); on consolidation
        # it re-renders with exact geodesics and the lines visibly
        # snap from the approximation to the truth.  Drags that don't
        # touch one of the last span's two endpoint nodes are skipped
        # outright by ``_recompute_spans`` (see the
        # ``_is_node_in_last_span`` guard).
        self._didactic_visible: bool = False
        # NOTE: ``_didactic_dirty`` is write-only (dead) — it is set here and
        # in ``_compute_didactic``/``_recompute_spans`` but never read in any
        # condition.  Toggle-ON recomputes unconditionally; cache validity is
        # decided per-slot inside ``_compute_didactic`` via ``cache_key``.
        # Kept for now (removal is out of scope for the current doc pass).
        self._didactic_dirty: bool = True
        # Cache of the t-INVARIANT pieces of the cascade (path_12 plus
        # the cumulative lengths of path_b / path_a / path_12).  These
        # depend ONLY on the geometry of the last span's two endpoint
        # nodes — moving the t-slider must NOT trigger a recompute of
        # path_12 (which would call compute_endpoint_local at ~75 ms
        # per slider tick, plus a visible jump between the Euclidean
        # and geodesic approximations).
        #
        # Two slots — ``'fast'`` and ``'exact'`` — held simultaneously
        # so the slider tick path (``fast=True``) and the debounce
        # consolidation (``fast=False``) each hit their own cached
        # entry.  An earlier single-slot design alternated and
        # recomputed path_12 every tick→debounce→tick cycle.
        # The cache invalidates by object identity: any handle drag
        # rebuilds n0.path_b / n1.path_a / origins, so ``id(...)`` of
        # those buffers changes and the next call sees a miss.
        # None = "no valid cached entry, recompute everything".
        self._didactic_geo_cache: dict | None = None
        # Parameter value of the cascade.  The slider widget binds to
        # this attribute via ``_on_didactic_t_change``; while the
        # slider doesn't exist yet (first toggle pending), the default
        # from SplineConfig is used.  Keeping the value on the instance
        # rather than the slider lets ``_compute_didactic`` work
        # without needing the widget present.
        self._didactic_t: float = self.scfg.DIDACTIC_T_DEFAULT
        # Lazy-created the first time the user toggles 'd' on.  None
        # means "not yet built".  Lifecycle: build once, enable /
        # disable per toggle.  Tearing down on toggle-off would force
        # re-creation each cycle and PyVista's ``add_slider_widget``
        # is non-trivial.
        self._didactic_slider = None
        import gizmo as _gizmo_mod  # local alias for opacity read-back
        self._didactic_pds: list[pv.PolyData] = []
        self._didactic_actors: list[vtk.vtkActor] = []
        # In front of the orange curve so the scaffold reads cleanly
        # on top of the final spline.
        for _i in range(4):
            pd, actor = self._create_aux_actor(
                kind='line', color='#2d6b3a', line_width=1.5,
                opacity=_gizmo_mod.GIZMO_OPACITY,
                depth=self.scfg.DEPTH_ORANGE - 4,
                name=f"didactic_line_{_i}")
            self._didactic_pds.append(pd)
            self._didactic_actors.append(actor)

        # Level-3 evaluation point: small dark-green sphere placed at
        # ``geodesic_lerp(path_final, t)``.  It is the point on the
        # orange curve at the chosen ``t`` — visualising it on top of
        # the cascade makes the collapse explicit (the entire scaffold
        # converges to this single point).  Tracks the same opacity as
        # the lines so the whole scaffold fades together with the 't'
        # key.
        # Slightly more in-front than the lines (so the sphere reads
        # crisply on top of path_final at the collapse point).
        (self._didactic_point_pd,
         self._didactic_point_actor) = self._create_aux_actor(
            kind='point', color='#1f5232', point_size=10,
            opacity=_gizmo_mod.GIZMO_OPACITY,
            depth=self.scfg.DEPTH_ORANGE - 5,
            name="didactic_point")
        self._didactic_point_buf = np.empty((1, 3), dtype=float)

        # --- Hover-curve cache ---
        # ``_collect_visible_curves`` packs every visible polyline into
        # a single (N, 3) buffer for batched screen projection.  Hover
        # detection is gated to "not dragging, not hovering a marker",
        # so the buffer changes only when geometry of a visible span
        # changes (not on every move).  We invalidate the cache via
        # ``_hover_curve_dirty`` and rebuild lazily; reuse otherwise.
        # On a session with several splines this saves a few ms per
        # mouse-move event when the cursor wanders the surface between
        # edits.
        self._hover_curve_dirty: bool = True
        self._hover_curve_items_cached: list[_CurveHoverItem] = []
        self._hover_curve_buf_total: int = 0

    # Visual z-priority penalty for curve hover.  When multiple curves
    # overlap on screen, the one rendered on top should win the hover.
    # A small penalty (in squared pixels) is added to lower-priority
    # layers so that the visually topmost curve wins ties.
    _LAYER_HOVER_PENALTY = {'orange': 0.0, 'blue': 3.0, 'interp': 6.0}

    def _detect_curve_hover(self, x: int, y: int) -> bool:
        """Tests proximity of cursor to all visible spline curves.

        Orchestrator: delegates point collection, batched screen
        projection, and closest-segment search to dedicated helpers.
        Returns True when the hover marker's visibility or position
        changed (the caller renders accordingly).

        When curves overlap on screen (nearly equal distance), the
        layer with higher visual z-priority wins — see
        ``_LAYER_HOVER_PENALTY``.
        """
        items = self._collect_visible_curves()
        if not items:
            return self._update_hover_marker(None, None)

        all_2d = self._to_screen_batch(self._curve_hover_3d_buf[:items[-1].start + items[-1].n_pts])
        best_info, best_pt_3d = self._pick_closest_curve(items, all_2d, float(x), float(y))
        return self._update_hover_marker(best_info, best_pt_3d)

    def _collect_visible_curves(self) -> list[_CurveHoverItem]:
        """Concatenates every visible curve's 3-D points into one buffer.

        Grows ``self._curve_hover_3d_buf`` if the running total exceeds
        capacity (rare — initial 2048 fits ~10 medium splines).
        Returns one ``_CurveHoverItem`` per visible polyline; an empty
        list short-circuits the caller.

        Cached behaviour: the result is memoised in
        ``self._hover_curve_items_cached`` until ``_hover_curve_dirty``
        is set by any callsite that mutates curve geometry, layer
        visibility, or actor membership (``_set_span``, ``_set_geo_span``,
        ``_set_interp_curve``, ``_toggle_layer``, ``_load_from_data``,
        ``_clear_*`` family).  Hover detection is gated to mouse-moves
        without an active drag, so this cache is only hit while geometry
        is stable — exactly the regime where rebuilding it per move was
        wasteful.  Cost of marking dirty is one bool assignment.
        """
        if not self._hover_curve_dirty:
            return self._hover_curve_items_cached

        items: list[_CurveHoverItem] = []
        total_n = 0

        def _ensure_capacity(needed: int) -> None:
            """Grow ``_curve_hover_3d_buf`` to fit *needed* rows, preserving
            the contents already written into ``[:total_n]``.  A previous
            implementation used ``np.empty`` which left those slots
            uninitialised, producing ghost hover hits when a session
            crossed the initial 2048-row threshold.
            """
            cur_cap = self._curve_hover_3d_buf.shape[0]
            if needed <= cur_cap:
                return
            new_cap = max(needed, cur_cap * 2)
            new_buf = np.empty((new_cap, 3), dtype=float)
            if total_n > 0:
                new_buf[:total_n] = self._curve_hover_3d_buf[:total_n]
            self._curve_hover_3d_buf = new_buf

        layer_caches = []
        if self._layer_visible[LayerKind.BLUE]:
            layer_caches.append((LayerKind.BLUE, self._span_cache))
        if self._layer_visible[LayerKind.ORANGE]:
            layer_caches.append((LayerKind.ORANGE, self._geo_span_cache))

        for layer, cache in layer_caches:
            for (sid, i), (pd, actor) in cache.items():
                if not actor.GetVisibility():
                    continue
                pts_3d = pd.points
                if pts_3d is None or len(pts_3d) < 2:
                    continue
                n = len(pts_3d)
                _ensure_capacity(total_n + n)
                self._curve_hover_3d_buf[total_n:total_n + n] = pts_3d
                items.append(_CurveHoverItem(layer, sid, i, total_n, n, pts_3d))
                total_n += n

        # Interp layer: keyed by sid only (one polyline per spline, not
        # per span).  ``span_idx=None`` records the absence of a span
        # index — downstream code switches behaviour on that None.
        if self._layer_visible[LayerKind.INTERP]:
            for sid, (pd, actor) in self._interp_cache.items():
                if not actor.GetVisibility():
                    continue
                pts_3d = pd.points
                if pts_3d is None or len(pts_3d) < 2:
                    continue
                n = len(pts_3d)
                _ensure_capacity(total_n + n)
                self._curve_hover_3d_buf[total_n:total_n + n] = pts_3d
                items.append(_CurveHoverItem(
                    LayerKind.INTERP, sid, None, total_n, n, pts_3d))
                total_n += n

        self._hover_curve_items_cached = items
        self._hover_curve_buf_total = total_n
        self._hover_curve_dirty = False
        return items

    def _pick_closest_curve(self, items: list[_CurveHoverItem],
                            all_2d: np.ndarray, mx: float, my: float
                            ) -> tuple[dict | None, np.ndarray | None]:
        """Finds the closest curve segment under the cursor.

        Returns ``(info_dict, pt_3d)`` ready to feed
        ``_update_hover_marker``, or ``(None, None)`` if no curve is
        within the pick tolerance after applying z-priority penalties
        and the occlusion check.
        """
        best_sq = self.cfg.PICK_TOLERANCE_SQ
        best_info: dict | None = None
        best_pt_3d: np.ndarray | None = None
        for item in items:
            penalty = self._LAYER_HOVER_PENALTY[item.layer]
            sq, seg, frac = _closest_seg_on_polyline_2d(
                all_2d[item.start:item.start + item.n_pts], item.n_pts, mx, my)
            effective_sq = sq + penalty
            if effective_sq < best_sq and seg + 1 < item.n_pts:
                p0 = item.pts_3d[seg]
                p1 = item.pts_3d[seg + 1]
                pt_3d = p0 * (1.0 - frac) + p1 * frac
                if not self._is_marker_occluded(pt_3d):
                    best_sq = effective_sq
                    best_pt_3d = pt_3d
                    # Local curve tangent at the hover point.  The
                    # current telescopic-sight marker billboards to
                    # camera and does NOT consume this field, but the
                    # tangent is kept in the payload for downstream
                    # callers (potential future affordances, programmatic
                    # hover analysis, or a debug overlay) — computing it
                    # here is free since p0 / p1 are already in scope.
                    edge = p1 - p0
                    edge_len = np.linalg.norm(edge)
                    if edge_len > 1e-9:
                        tangent = edge / edge_len
                    else:
                        # Degenerate segment (two coincident polyline
                        # points) — pick any unit vector so consumers
                        # always see a well-formed entry.
                        tangent = np.array([1.0, 0.0, 0.0])
                    # ``span_idx`` is ``-1`` for interp to keep historical
                    # ``info['span_idx']`` semantics for callers that
                    # compare with integers.  The dataclass uses ``None``
                    # internally; the public dict translation is here.
                    span_idx = item.span_idx if item.span_idx is not None else -1
                    best_info = {
                        'spline_idx': item.sid,
                        'span_idx': span_idx,
                        'layer': item.layer,
                        'seg': seg,
                        'frac': frac,
                        'point': best_pt_3d,
                        'tangent': tangent,
                    }
        return best_info, best_pt_3d

    def _hover_info_live(self, info: dict | None) -> bool:
        """True when *info* still describes the spline it was built from.

        The telescopic-sight payload carries ``spline_idx`` / ``span_idx``
        captured on a mouse-move, but it is consumed later, by a
        double-click.  Every structural mutation that can happen in
        between is keyboard-driven (Backspace, ``C``, ``l``, Ctrl+Z,
        Dbl-click R), so the cursor never moves and the payload is never
        refreshed.  Acting on it then either raises ``IndexError`` deep
        in ``_insert_node_at_curve`` — after ``_push_undo`` has already
        spent an undo slot and cleared the redo stack — or, when the
        stale index happens to stay in range, silently inserts into a
        *different* span at a 3-D point taken from a curve that no
        longer exists there.

        Identity comparison, not ``==``: the node objects must be the
        very same instances, so a rebuild that produced equal-looking
        nodes still counts as stale.
        """
        if info is None:
            return False
        sid = info.get('spline_idx')
        snap = info.get('nodes_snapshot')
        if snap is None or not isinstance(sid, int):
            return False
        if not 0 <= sid < len(self.splines):
            return False
        if bool(self.splines_closed[sid]) != info.get('closed_snapshot'):
            return False
        cur = self.splines[sid]
        if len(cur) != len(snap) or any(
                a is not b for a, b in zip(cur, snap, strict=True)):
            return False
        # ``span_idx == -1`` is the interp layer's sentinel — it addresses
        # the whole spline, not one span, so the node check above is the
        # whole contract.
        span_idx = info.get('span_idx', -1)
        return span_idx == -1 or 0 <= span_idx < self._span_count(sid)

    def _hide_curve_hover_marker(self) -> None:
        """Hide both actors that make up the telescopic-sight hover
        marker (circumference + crosshair).  Idempotent.
        """
        self._curve_hover_circle_actor.SetVisibility(False)
        self._curve_hover_cross_actor.SetVisibility(False)

    def _orient_hover_marker(self, pt_3d: np.ndarray) -> None:
        """Writes the geometry of the telescopic-sight hover marker.

        The marker is a circle (frame) plus a centered crosshair
        (horizontal + vertical diameters) drawn on the camera's
        view-plane at world position *pt_3d*.  Orientation:

            view = focal − position             (camera forward, world)
            right = normalise(view × view_up)   (screen-horizontal in world)
            up    = normalise(right × view)     (screen-vertical, re-orthogonalised)

        The crosshair's two diameters are aligned with *right* and *up*
        — i.e. always horizontal / vertical on screen, like a real
        telescopic sight — regardless of the underlying curve's
        direction.  Radius scales with camera-to-point distance so the
        marker keeps a constant on-screen footprint at any zoom
        (``HOVER_MARKER_SCREEN_SCALE``).

        Degenerate case: when the camera's view direction is parallel
        to ``view_up`` (e.g. looking straight down a vertical-up axis
        and the camera's roll happens to put up = view), the
        ``view × view_up`` cross collapses.  Falls back to a world-axis
        right vector that is not parallel to view.
        """
        cam = self.plotter.camera
        cam_pos = np.asarray(cam.position, dtype=float)
        focal = np.asarray(cam.focal_point, dtype=float)
        view_up = np.asarray(cam.up, dtype=float)

        view = focal - cam_pos
        vn = float(np.linalg.norm(view))
        if vn < 1e-9:
            return  # paranoid — camera collapsed onto its focal point
        view_unit = view / vn

        right = np.cross(view_unit, view_up)
        rn = float(np.linalg.norm(right))
        if rn < 1e-6:
            # view ∥ view_up.  Swap in a non-parallel world axis.
            fallback = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(fallback, view_unit)) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0])
            right = np.cross(view_unit, fallback)
            rn = float(np.linalg.norm(right))
            if rn < 1e-9:
                return
        right_unit = right / rn
        # Re-orthogonalise: the user-supplied view_up may not be
        # perfectly ⟂ to view (VTK does not enforce it).  ``up_unit``
        # = right × view is always exactly ⟂ to both.
        up_unit = np.cross(right_unit, view_unit)
        upn = float(np.linalg.norm(up_unit))
        if upn < 1e-9:
            return
        up_unit = up_unit / upn

        # Radius / half-extent in world units, screen-scaled.
        cam_to_pt = cam_pos - pt_3d
        dist = float(np.linalg.norm(cam_to_pt))
        if dist < 1e-9:
            dist = 1e-9
        r = dist * self.scfg.HOVER_MARKER_SCREEN_SCALE

        # Circumference: pt + r·(cosθ·right + sinθ·up) for θ ∈ [0, 2π].
        # The pre-computed cos/sin tables avoid the trig call per refresh.
        circle_buf = self._curve_hover_circle_buf
        np.multiply.outer(self._curve_hover_cos, right_unit, out=circle_buf)
        circle_buf += np.multiply.outer(self._curve_hover_sin, up_unit)
        circle_buf *= r
        circle_buf += pt_3d
        self._curve_hover_circle_pd.points = circle_buf
        self._curve_hover_circle_pd.Modified()

        # Crosshair: 4 points at pt ± r·right, pt ± r·up.
        cross_buf = self._curve_hover_cross_buf
        r_right = r * right_unit
        r_up = r * up_unit
        cross_buf[0] = pt_3d - r_right
        cross_buf[1] = pt_3d + r_right
        cross_buf[2] = pt_3d - r_up
        cross_buf[3] = pt_3d + r_up
        self._curve_hover_cross_pd.points = cross_buf
        self._curve_hover_cross_pd.Modified()

    def _update_hover_marker(self, info: dict | None,
                             pt_3d: np.ndarray | None) -> bool:
        """Repositions / shows / hides the hover marker actor.

        Returns True when the visibility or position changed and the
        caller should issue a render.
        """
        if info is not None and pt_3d is not None:
            # Stamp the structure the payload describes.  ``info`` is
            # consumed on a later event (the double-click that inserts a
            # node), and every structural mutation in between —
            # backspace, close / reopen, load, undo, spline delete — is
            # keyboard-driven, so no intervening mouse-move refreshes
            # it.  Validating the stamp at the point of use is one
            # check that cannot rot; clearing the payload in each of the
            # eight mutators would be a new invalidation path to keep in
            # sync forever.  See ``_hover_info_live``.
            sid = info['spline_idx']
            if 0 <= sid < len(self.splines):
                info['nodes_snapshot'] = tuple(self.splines[sid])
                info['closed_snapshot'] = bool(self.splines_closed[sid])
            else:
                info['nodes_snapshot'] = None
                info['closed_snapshot'] = None
            self.curve_hover_info = info
            self._orient_hover_marker(pt_3d)
            # Stash the geometry inputs so the camera-orbit hook can
            # re-orient the marker without re-running curve hover
            # detection.  Tangent is no longer needed (the marker is
            # billboarded to camera, not aligned to the curve) but
            # ``info['tangent']`` stays in the payload — other consumers
            # may rely on it and recomputing in the orient hook is free.
            self._curve_hover_state = {
                'pt': np.array(pt_3d, dtype=float),
            }
            color_map = {
                LayerKind.BLUE: self.scfg.SPAN_COLOR,
                LayerKind.ORANGE: self.scfg.GEO_COLOR,
                LayerKind.INTERP: self.scfg.INTERP_COLOR,
            }
            color = color_map[info['layer']]
            self._curve_hover_circle_actor.GetProperty().SetColor(*color)
            self._curve_hover_cross_actor.GetProperty().SetColor(*color)
            self._curve_hover_circle_actor.SetVisibility(True)
            self._curve_hover_cross_actor.SetVisibility(True)
            return True  # always render — position moved
        self.curve_hover_info = None
        self._curve_hover_state = None
        changed = False
        if self._curve_hover_circle_actor.GetVisibility():
            self._curve_hover_circle_actor.SetVisibility(False)
            changed = True
        if self._curve_hover_cross_actor.GetVisibility():
            self._curve_hover_cross_actor.SetVisibility(False)
            changed = True
        return changed

    def _cycle_gizmo_opacity(self) -> None:
        """Cycles the opacity of all auxiliary visuals (nodes, tangent lines,
        handle arrows, stitch preview, didactic scaffold) through
        0.2 → 0.4 → 0.7 → 1.0 → 0.2.

        Modifies the module-level ``gizmo.GIZMO_OPACITY`` and refreshes
        every actor that reads from it.
        """
        import gizmo
        ticks = [0.2, 0.4, 0.7, 1.0]
        cur = gizmo.GIZMO_OPACITY
        nxt = next((t for t in ticks if t > cur + 1e-3), ticks[0])
        gizmo.GIZMO_OPACITY = nxt
        # Refresh all segment visuals to pick up the new opacity
        for _, _, node in self._iter_all_nodes():
            node.update_visuals(self.plotter)
        # Update stitch preview if visible
        if self._stitch_actor.GetVisibility():
            self._stitch_actor.GetProperty().SetOpacity(nxt)
        # Update didactic scaffold opacity (lines stay fixed-color; the
        # only thing that tracks the gizmo opacity is the alpha).  Also
        # the level-3 collapse-point sphere shares the same alpha.
        for actor in (*self._didactic_actors, self._didactic_point_actor):
            actor.GetProperty().SetOpacity(nxt)
        self._set_hud(_t("gizmo_opacity", pct=f"{nxt:.0%}"), 'white')
        self.plotter.render()

    def _toggle_layer_key(self, layer: str) -> None:
        """Keyboard shortcut: inverts the visibility of a curve layer
        and synchronizes the checkbox widget to match.

        Keys: ``b`` (blue), ``o`` (orange), ``k`` (interp).
        """
        new_val = not self._layer_visible[layer]
        self._toggle_layer(layer, new_val)
        widget = self._layer_widgets.get(layer)
        if widget is not None:
            widget.GetRepresentation().SetState(int(new_val))

    def _toggle_layer(self, layer: str, visible: bool) -> None:
        """Checkbox callback: shows or hides all actors in a curve layer.

        *layer* is ``'blue'``, ``'orange'``, or ``'interp'``.

        Special case for ``interp``: while hidden, ``_recompute_interp_curve``
        is short-circuited so the synchronous splprep / splev / projection
        chain does not steal main-thread frames from the visible layers.
        That means the cached actor geometry can be stale when the user
        toggles the layer ON.  We compensate by forcing a full recompute
        across all splines on the OFF→ON transition, so the curve appears
        immediately at full quality (no perceptible lag).  Blue and orange
        have separate behaviour: blue is recomputed live during drag, and
        orange is computed by background workers regardless of visibility,
        so neither needs this hand-off.
        """
        was_visible = self._layer_visible.get(layer, False)
        self._layer_visible[layer] = visible
        cache_map = {
            'blue': self._span_cache,
            'orange': self._geo_span_cache,
        }
        cache = cache_map.get(layer)
        if cache is not None:
            for _, actor in cache.values():
                actor.SetVisibility(visible)
        if layer == 'interp':
            # OFF → ON transition: regenerate from scratch since
            # _recompute_interp_curve was no-op'd while hidden.
            if visible and not was_visible:
                for s in range(len(self.splines)):
                    self._recompute_interp_curve(s, is_dragging=False)
            for _, actor in self._interp_cache.values():
                actor.SetVisibility(visible)
        # Hover detection scans visible curves only — visibility change
        # invalidates the cached buffer.
        self._hover_curve_dirty = True
        self.plotter.render()

    def _fire_debounce(self) -> None:
        """Exact recalculation + span recomputation + geodesic submit.

        Sequence:
          1. Recomputes the exact geodesic for the dragged segment
             (``update_from_a/b/p`` with ``exact=True``).
          2. Recomputes hybrid Bézier spans (blue) for affected indices.
          3. Calls ``_submit_geodesic_spans`` to start background workers
             for the fully-geodesic orange curve on the same indices.

        The ``is_preview`` flag is set to False before recomputation, so
        hybrid spans revert to full color/width (consolidated appearance)
        and LOD switches to full quality.

        No render() — batched by ``_on_poll_timer``.
        """
        seg = self.state.active_seg
        if seg is not None and self.state.last_drag_q is not None:
            q, cid = self.state.last_drag_q, self.state.last_drag_cid
            seg.is_preview = False

            # Shift+drag of A/B = magnitude-only mode (same dispatch as
            # the live-preview path in geo_shoot._on_move).  ``exact=True``
            # would normally route through ``compute_endpoint_from_origin``
            # to land precisely on the cursor, but for magnitude mode we
            # use ``compute_shoot`` (a directional ray for a target
            # arc-length) — the right primitive when the input is "scrub
            # along this fixed axis".
            shift_held = bool(
                self.plotter.iren.interactor.GetShiftKey())
            if self.state.drag_marker == 'p':
                seg.update_from_p(q, cid, self.geo, exact=True)
            elif self.state.drag_marker in ('a', 'b') and shift_held:
                seg.update_magnitude(q, self.state.drag_marker, self.geo,
                                     exact=True)
            elif self.state.drag_marker == 'a':
                seg.update_from_a(q, self.geo, exact=True)
            elif self.state.drag_marker == 'b':
                seg.update_from_b(q, self.geo, exact=True)

            self._recompute_spans(node=seg)
            self._submit_geodesic_spans(node=seg)
            seg.update_visuals(self.plotter)
            # Origin may have moved — stitch cache uses same id() but stale solver
            if id(seg) == self._stitch_origin_node_id:
                self._invalidate_stitch_cache()
            # Record the exact consolidation so a release following this
            # debounce doesn't recompute the same spans a second time.
            self._consolidated_seg = seg
            self._set_hud(_t("refined_exact"), 'cyan')

    def _finalize_release(self, seg: GeodesicSegment) -> None:
        """Post-drag: keep node active, recompute spans, restore active spline.

        If the drag required switching to a different spline (via
        ``_try_hit_marker``), restores the pre-drag active spline index.
        This prevents losing access to empty (break) splines that have
        no clickable nodes.

        When ``_fire_debounce`` ran synchronously inside ``_on_release``
        for this same node it already produced the exact spans AND
        resubmitted the orange workers (``is_preview`` was cleared, so
        ``_recompute_spans`` took the exact branch), so we skip a
        redundant second solve here.  Otherwise (release with no pending
        debounce, e.g. a click without drag) we consolidate now — blue
        AND orange: ``_try_hit_marker`` cancelled and cleared the
        adjacent orange spans at press time, so without the resubmit
        below a plain click on a marker left them blank until the next
        drag or an ``R`` rebuild.
        """
        # A release whose gesture never moved never committed the
        # press-time snapshot — discard it (nothing mutated, nothing
        # to undo).
        self._pending_drag_snapshot = None
        seg.is_active = True
        if self._consolidated_seg is not seg:
            self._recompute_spans(node=seg)
            self._submit_geodesic_spans(node=seg)
        self._consolidated_seg = None
        seg.update_visuals(self.plotter)

        # Restore the active spline from before the drag
        pre = getattr(self, '_pre_drag_spline_idx', None)
        if pre is not None and pre != self.active_spline_idx:
            if 0 <= pre < len(self.splines):
                self.active_spline_idx = pre
                self._refresh_visuals()
        self._pre_drag_spline_idx = None

    def _abort_active_drag(self) -> None:
        """Spline-aware drag abort — see the base class docstring.

        Additionally drops the spline-level per-gesture state: a stale
        ``_consolidated_seg`` from the aborted gesture would make the
        NEXT drag of the same node skip its release consolidation, and
        an uncommitted press-time undo snapshot no longer corresponds
        to any mutation.
        """
        had_drag = self.state.active_seg is not None
        super()._abort_active_drag()
        if had_drag:
            self._consolidated_seg = None
            self._pending_drag_snapshot = None

    def _setup_interaction(self) -> None:
        super()._setup_interaction()
        self.plotter.add_key_event('BackSpace', self._on_backspace)
        self.plotter.add_key_event('c', self._on_close_spline)
        self.plotter.add_key_event('b', lambda: self._toggle_layer_key('blue'))
        self.plotter.add_key_event('o', lambda: self._toggle_layer_key('orange'))
        self.plotter.add_key_event('k', lambda: self._toggle_layer_key('interp'))
        self.plotter.add_key_event('s', self._on_save)
        self.plotter.add_key_event('l', self._on_load)
        self.plotter.add_key_event('t', self._cycle_gizmo_opacity)
        self.plotter.add_key_event('r', self._rebuild_all_orange)
        self.plotter.add_key_event('v', self._on_export_vtk)
        self.plotter.add_key_event('d', self._toggle_didactic)
        # Note: 'x' is wired below as a raw VTK press/release pair (not
        # ``add_key_event``) so the handler can distinguish hold from
        # tap — see ``_on_key_press_guides`` / ``_on_key_release_guides``.
        # Capture the tags into the parent's ``_observer_tags`` list
        # so ``cleanup()`` detaches them.  Without this the lambdas /
        # bound methods keep ``self`` alive after the window is closed.
        _vtki = self.plotter.iren.interactor
        self._observer_tags.append((_vtki, _vtki.AddObserver(
            vtk.vtkCommand.RightButtonPressEvent, self._on_right_press, 1.0)))
        # Ctrl+Z / Ctrl+Y — raw VTK observer (PyVista add_key_event
        # does not support modifier keys).
        self._observer_tags.append((_vtki, _vtki.AddObserver(
            'KeyPressEvent', self._guard_observer(self._on_key_press_ctrl), 1.0)))
        # Node-index labels: 'n' must be a hold-to-show shortcut, not a
        # toggle, so PyVista's add_key_event (press-only) is not enough.
        # Raw VTK observers on both press AND release let us track the
        # held state precisely.  Press fires repeatedly under OS key-
        # repeat (~30 Hz) — ``_show_node_labels`` is idempotent and
        # cheap so that's fine and also keeps labels positioned when
        # the user drags a node while holding 'n'.
        self._observer_tags.append((_vtki, _vtki.AddObserver(
            'KeyPressEvent', self._guard_observer(self._on_key_press_labels), 1.0)))
        self._observer_tags.append((_vtki, _vtki.AddObserver(
            'KeyReleaseEvent', self._guard_observer(self._on_key_release_labels), 1.0)))
        # Guide polylines: 'x' is hold-to-preview-opaque + release-to-
        # toggle.  Same raw-observer pattern as 'n' for the same
        # reason — PyVista's ``add_key_event`` is press-only.
        self._observer_tags.append((_vtki, _vtki.AddObserver(
            'KeyPressEvent', self._guard_observer(self._on_key_press_guides), 1.0)))
        self._observer_tags.append((_vtki, _vtki.AddObserver(
            'KeyReleaseEvent', self._guard_observer(self._on_key_release_guides), 1.0)))

    def _guard_observer(self, fn):
        """Wrap a raw VTK observer so a failure is *visible*.

        ``_on_press`` / ``_on_move`` / ``_on_right_press`` /
        ``_on_poll_timer`` each have a hand-written ``_impl`` guard; the
        key observers wired above had none, and neither VTK nor
        PyVista's dispatcher guards for us.  VTK does not die on a
        raising Python observer — it calls ``PyErr_Print()`` and carries
        on — so the real consequence is a *silent* one: the traceback
        goes to stderr, which a GUI user never sees, the HUD is
        unchanged, and the shortcut simply appears to do nothing.

        Returns a plain function (not a bound method) — ``AddObserver``
        holds the reference, and the tag is captured into
        ``_observer_tags`` so ``cleanup()`` still detaches it.
        """
        name = getattr(fn, '__name__', repr(fn))

        def _wrapper(obj, event):
            try:
                fn(obj, event)
            except Exception:  # noqa: BLE001 — VTK observer must not propagate
                log.exception("%s failed", name)
                try:
                    self._set_hud(_t("handler_failed", name=name), 'red',
                                  sticky_seconds=4.0)
                    self.plotter.render()
                except Exception:  # noqa: BLE001 — HUD is best-effort
                    pass
        return _wrapper

    # Single source of truth for the editor's keybinding help.  Used
    # both by the on-screen panel (``_HELP_TEXT``, narrow column) and
    # by ``_print_help`` for the console banner.  Adding a new shortcut
    # only requires adding one row here.
    _HELP_ROWS: tuple[tuple[str, str], ...] = (
        ("Dbl-click L",     "Add node"),
        ("Dbl-click R",     "New spline / Edit P coords"),
        ("Drag Red",        "Translate node"),
        ("Drag Handle",     "Tangents"),
        ("Shift+Drag P",    "Snap to mesh vertex"),
        ("Shift+Drag A/B",  "Magnitude only (no snap, no rotation)"),
        ("C",               "Close/open loop"),
        ("Backspace",       "Undo node"),
        ("Ctrl+Z / Ctrl+Y", "Undo / Redo"),
        ("b/o/k",           "Toggle blue/orange/interp curves"),
        ("t",               "Cycle gizmo opacity (20/40/70/100%)"),
        ("r",               "Rebuild orange (all splines)"),
        ("s",               "Save splines to JSON"),
        ("l",               "Load splines from JSON"),
        ("v",               "Export orange curve to .vtk"),
        ("d",               "Toggle didactic scaffold (t=0.5)"),
        ("Ctrl+X / X (hold)", "Import guides / hold X for opaque preview, release toggles"),
        ("n (hold)",        "Show node-index labels while pressed"),
        ("e",               "Export paths"),
        ("w",               "Wireframe"),
        ("a",               "Surface opacity"),
    )

    def _print_help(self) -> None:
        # Console help -- ASCII only (Windows codepage 850 / cp1252 friendly).
        print("\n" + "=" * 60)
        print("  GEODESIC SPLINE EDITOR")
        for key, desc in self._HELP_ROWS:
            print(f"  {key:<16}: {desc}")
        print("=" * 60 + "\n")

    @classmethod
    def _build_help_text(cls) -> str:
        """Builds the on-screen narrow-column help string from
        ``_HELP_ROWS``.  Wraps long descriptions onto a continuation
        line so the panel stays inside the 28-char column."""
        col = 14  # key column width inside the panel
        lines: list[str] = []
        for key, desc in cls._HELP_ROWS:
            lines.append(f"  {key:<{col}}: {desc}")
        return "\n".join(lines)

    @property
    def _HELP_TEXT(self) -> str:  # noqa: N802 — preserved name for compat
        return self._build_help_text()

    def _on_window_resize(self, obj, event) -> None:
        """Repositions checkbox widgets, help button, and slider after
        a window resize.

        Layout (bottom-up, in pixels from bottom edge):
          - Slider: 20 px from bottom
          - Checkboxes + help "?": 20 px above slider top

        All positions are recomputed from the window height so the
        widget cluster stays compact and proportional.
        ``PlaceWidget`` takes 6-element bounds ``[xmin, xmax, ymin, ymax, 0, 0]``.
        """
        h = self.plotter.window_size[1]
        if h < 1:
            return
        sz = self._cb_size

        # Slider: fixed 20 px from bottom, normalized coords
        slider_y_px = 20
        slider_y_norm = slider_y_px / h
        rep_sl = self._opacity_slider.GetRepresentation()
        rep_sl.GetPoint1Coordinate().SetValue(0.0, slider_y_norm, 0.0)
        rep_sl.GetPoint2Coordinate().SetValue(0.15, slider_y_norm, 0.0)

        # Checkboxes: 20 px above slider
        cb_y = slider_y_px + 20

        for i, (_layer, widget) in enumerate(self._layer_widgets.items()):
            x = self._cb_x_positions[i]
            widget.GetRepresentation().PlaceWidget(
                [float(x), float(x + sz), float(cb_y), float(cb_y + sz), 0.0, 0.0])

        hx = self._help_x
        self._help_widget.GetRepresentation().PlaceWidget(
            [float(hx), float(hx + sz), float(cb_y), float(cb_y + sz), 0.0, 0.0])

        # Reposition "?" label
        self._help_label.SetPosition(hx + 5, cb_y + 2)

        fmx = self._fm_x
        self._fm_widget.GetRepresentation().PlaceWidget(
            [float(fmx), float(fmx + sz), float(cb_y), float(cb_y + sz), 0.0, 0.0])
        self._fm_label.SetPosition(fmx + 2, cb_y + 2)

    def _toggle_full_mesh_orange(self, value: bool) -> None:
        """Toggles ``SplineConfig.ORANGE_USE_FULL_MESH``.

        Does **not** trigger a rebuild — the user re-issues ``R`` when
        ready to recompute the orange polylines with the new setting.
        Rationale: rebuilding all spans is multi-second on big sessions
        and the user typically wants to compare before/after side-by-
        side, which requires staging the toggle without paying the
        compute cost yet.

        HUD message includes the rebuild reminder so the user is not
        confused by the on-screen orange not changing immediately.
        """
        self.scfg.ORANGE_USE_FULL_MESH = value
        key = "orange_fullmesh_on" if value else "orange_fullmesh_off"
        self._set_hud(_t(key), 'orange', sticky_seconds=4.0)
        self.plotter.render()

    def _toggle_help_panel(self, visible: bool) -> None:
        """Toggles the on-screen shortcut reference panel."""
        if visible and self._help_panel is None:
            self._help_panel = self.plotter.add_text(
                self._HELP_TEXT, position=( 2 , 85 ),
                font_size=8, color='red', shadow=False,
                name="help_panel")
            self._help_visible = True
        elif not visible and self._help_panel is not None:
            self.plotter.remove_actor(self._help_panel)
            self._help_panel = None
            self._help_visible = False
        self.plotter.render()

    def _on_key_press_ctrl(self, obj, event) -> None:
        """Raw VTK KeyPress handler for Ctrl-modified shortcuts.

        Recognised combinations:
          - Ctrl+Z — undo
          - Ctrl+Y — redo
          - Ctrl+X — import guide polylines (see ``_on_load_guides``)

        Used instead of PyVista's ``add_key_event`` because the latter
        does not support modifier keys.
        """
        iren = self.plotter.iren.interactor
        if not iren.GetControlKey():
            return
        key = iren.GetKeySym()
        if key in ('z', 'Z'):
            self._on_undo_ctrl_z()
        elif key in ('y', 'Y'):
            self._on_redo()
        elif key in ('x', 'X'):
            self._on_load_guides()

    def _on_key_press_labels(self, obj, event) -> None:
        """Raw VTK KeyPress handler for the 'n' hold-to-show shortcut.

        Fires on every key-press tick (incl. OS key-repeat) — that
        cadence is convenient because it refreshes label positions
        when the user drags a node while holding 'n'.  Skips when a
        modifier is held so 'n' is unambiguously the labels gesture
        (Ctrl+N / Shift+N stay free for future use without surprise
        side effects).
        """
        iren = self.plotter.iren.interactor
        if iren.GetControlKey() or iren.GetAltKey():
            return
        key = iren.GetKeySym()
        if key in ('n', 'N'):
            self._show_node_labels()

    def _on_key_release_labels(self, obj, event) -> None:
        """Raw VTK KeyRelease handler — hides node-index labels when
        the user releases 'n'.  See ``_on_key_press_labels`` for the
        held-down counterpart.

        No modifier re-check here: ``_hide_node_labels`` is a no-op
        unless a press actually showed the labels, so that flag is the
        correct gate.  Re-checking would swallow the release of a user
        who pressed Ctrl *after* 'n', leaving the labels stuck on."""
        iren = self.plotter.iren.interactor
        key = iren.GetKeySym()
        if key in ('n', 'N'):
            self._hide_node_labels()

    # --- Helpers ---

    @property
    def _active_nodes(self) -> list[GeodesicSegment]:
        return self.splines[self.active_spline_idx]

    def _iter_all_nodes(self) -> Iterator[tuple[int, int, GeodesicSegment]]:
        """Yields ``(spline_idx, node_idx, node)`` for every node across all splines."""
        for s_idx, nodes in enumerate(self.splines):
            for n_idx, node in enumerate(nodes):
                yield s_idx, n_idx, node

    def _spline_for_node(self, seg: GeodesicSegment) -> int:
        """Returns the spline index that owns *seg*.

        O(1) via the ``_node_to_spline`` ``WeakKeyDictionary``.  Falls
        back to the currently-active spline when the cache is stale
        (logs a debug message — visible only when
        ``GEO_SPLINES_DEBUG=1`` so the user is not spammed during
        normal use).  A stale entry is repaired on the next
        ``_rebuild_node_index`` call, which every mutation already
        triggers.
        """
        sid = self._node_to_spline.get(seg)
        if sid is not None:
            return sid
        log.debug("_spline_for_node: segment missing from cache, "
                  "falling back to active spline %d",
                  self.active_spline_idx)
        return self.active_spline_idx

    def _rebuild_node_index(self) -> None:
        """Rebuilds the ``segment → spline_index`` lookup dict.

        Called after any structural change (node add/remove, spline
        add/remove) to keep the O(1) lookup in ``_spline_for_node`` correct.
        """
        self._node_to_spline = weakref.WeakKeyDictionary({
            node: s_idx
            for s_idx, nodes in enumerate(self.splines)
            for node in nodes
        })

    def _build_local_frame(self, pt: np.ndarray, cid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Builds an orthonormal tangent frame ``(normal, u, v)`` at *pt* on face *cid*."""
        n = self.geo.get_interpolated_normal(pt, cid)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(ref, n)) > self.scfg.NORMAL_ALIGN_THRESHOLD:
            ref = np.array([1.0, 0.0, 0.0])
        u = np.cross(n, ref)
        u /= np.linalg.norm(u)
        return n, u, np.cross(n, u)

    # --- Undo / Redo ---

    def _snapshot(self) -> dict:
        """Captures the current spline state as a lightweight dict.

        Uses the **v2** schema: each node serialises ``origin``, ``p_a``,
        and ``p_b`` as literal 3-D positions.  This lossless layout is
        what the JSON save format also writes — see ``_on_save`` for
        the rationale.  Typical size: ~96 bytes per node (3× the v1
        layout's 32 bytes; trivial in the snapshot stack).
        """
        splines = []
        for sid, nodes in enumerate(self.splines):
            node_data = []
            for node in nodes:
                node_data.append({
                    'origin': node.origin.tolist(),
                    'p_a': node.p_a.tolist() if node.p_a is not None else None,
                    'p_b': node.p_b.tolist() if node.p_b is not None else None,
                })
            splines.append({
                'closed': self.splines_closed[sid],
                'nodes': node_data,
            })
        return {
            'version': 2,
            'splines': splines,
            'active_spline_idx': self.active_spline_idx,
        }

    def _push_undo(self) -> None:
        """Saves a snapshot to the undo stack.  Clears the redo stack.

        Called immediately before any mutation (node add/insert/delete,
        close, break, drag start, load).  The snapshot captures the state
        *before* the mutation so that Ctrl+Z restores it.
        """
        # ``deque.append`` with ``maxlen`` evicts the oldest entry
        # automatically — no manual cap check needed.
        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()

    def _commit_pending_drag_undo(self) -> None:
        """Pushes the press-time snapshot once a drag actually moved.

        ``_try_hit_marker`` captures the pre-drag state without pushing
        it; the first processed drag movement commits it here.  A click
        that never moves discards the snapshot in ``_finalize_release``
        instead, so plain marker clicks (spline switching, accidental
        clicks) no longer spend undo entries or clear the redo stack.
        """
        snap = self._pending_drag_snapshot
        if snap is not None:
            self._pending_drag_snapshot = None
            self._undo_stack.append(snap)
            self._redo_stack.clear()

    def _on_undo_ctrl_z(self) -> None:
        """Ctrl+Z: restores the previous spline state from the undo stack."""
        if not self._undo_stack:
            # Sticky: silent no-op is too easy to miss otherwise.
            self._set_hud(_t("nothing_to_undo"), 'grey', sticky_seconds=1.5)
            self.plotter.render()
            return
        self._redo_stack.append(self._snapshot())
        data = self._undo_stack.pop()
        self._restore_snapshot(data)
        self._set_hud(_t("undo"), 'yellow')
        self.plotter.render()

    def _on_redo(self) -> None:
        """Ctrl+Y: re-applies the last undone operation from the redo stack."""
        if not self._redo_stack:
            self._set_hud(_t("nothing_to_redo"), 'grey', sticky_seconds=1.5)
            self.plotter.render()
            return
        self._undo_stack.append(self._snapshot())
        data = self._redo_stack.pop()
        self._restore_snapshot(data)
        self._set_hud(_t("redo"), 'cyan')
        self.plotter.render()

    def _can_use_diff_restore(self, data: dict) -> bool:
        """True when *data* and the current splines share the same shape.

        The differential restore path only handles geometry changes
        (origin / tangent edits) — any structural change (different
        spline count, node count, closed flag) forces a full rebuild.
        Centralising the predicate keeps ``_restore_snapshot`` readable.
        """
        target = data.get('splines')
        if not isinstance(target, list) or len(target) != len(self.splines):
            return False
        for i, sd in enumerate(target):
            nodes = sd.get('nodes', [])
            if len(nodes) != len(self.splines[i]):
                return False
            if bool(sd.get('closed', False)) != self.splines_closed[i]:
                return False
        return True

    def _restore_snapshot(self, data: dict) -> None:
        """Restores a snapshot, using differential reconstruction when possible.

        **Differential path** (fast, when the spline structure is identical):
        compare the current state node-by-node with *data*.  Only nodes
        whose origin or tangent changed are reconstructed (via ``compute_shoot``,
        ~10 ms per node).  On a 50-node spline where only 1 node moved,
        this is ~50× faster than the full rebuild.

        **Full rebuild** (fallback): when the structure differs (different
        number of splines, different node counts per spline, closed flag
        changed), delegates to ``_load_from_data`` which clears all actors
        and reconstructs everything from scratch.

        Snapshots are produced internally by ``_snapshot()`` so they are
        well-formed by construction.  The validation pass below is
        defence-in-depth: a future bug in ``_snapshot`` (or a manual
        injection of a malformed dict) would otherwise crash inside the
        renderer.  ``_validate_session_dict`` rejects closed splines with
        < 3 nodes — the same invariant the interactive editor enforces.
        """
        try:
            _validate_session_dict(data)
        except ValueError as exc:
            log.error("invalid undo/redo snapshot — refusing to restore: %s", exc)
            return
        active = data.pop('active_spline_idx', 0)

        if not self._can_use_diff_restore(data):
            # Full rebuild path
            self._load_from_data(data)
            self.active_spline_idx = self._clamp_spline_idx(active)
            self._prev_active_spline_idx = self.active_spline_idx
            self._refresh_visuals()
            return

        # Differential path: same structure, reconstruct only changed nodes.
        # ``_snapshot`` always emits v2 records (with p_a / p_b); v1 records
        # only ever reach _load_from_data (full rebuild path).  So here we
        # compare on (origin, p_a, p_b) — the same invariant the live state
        # carries.
        changed_splines: set[int] = set()
        for sid, sd in enumerate(data['splines']):
            current_nodes = self.splines[sid]
            for nid, nd in enumerate(sd['nodes']):
                target_origin = np.asarray(nd['origin'], dtype=float)
                seg = current_nodes[nid]
                if not np.allclose(seg.origin, target_origin, atol=1e-12):
                    self._rebuild_node_inplace(seg, nd)
                    changed_splines.add(sid)
                    continue

                def _arr_or_none(v):
                    return np.asarray(v, dtype=float) if v is not None else None

                tgt_pa = _arr_or_none(nd.get('p_a'))
                tgt_pb = _arr_or_none(nd.get('p_b'))
                cur_pa = _arr_or_none(seg.p_a)
                cur_pb = _arr_or_none(seg.p_b)

                def _same(a, b):
                    if a is None and b is None:
                        return True
                    if a is None or b is None:
                        return False
                    return np.allclose(a, b, atol=1e-12)

                if _same(cur_pa, tgt_pa) and _same(cur_pb, tgt_pb):
                    continue  # node geometry unchanged
                self._rebuild_node_inplace(seg, nd)
                changed_splines.add(sid)

        # Recompute spans only for splines with changed nodes
        if changed_splines:
            for sid in changed_splines:
                self._recompute_spans(sid=sid)
                self._submit_geodesic_spans(sid=sid)
            # ``_rebuild_node_inplace`` keeps the node objects (and their
            # ids) but moves their origins, so the stitch cache — keyed by
            # id(last_node) — can now point a pre-built solver at the old
            # origin.  Drop it; the full-rebuild branch above does the same
            # via ``_clear_all_curve_caches``.
            self._invalidate_stitch_cache()
            # Marker positions moved too — without this the hover cache
            # keeps serving the pre-restore screen positions (markers
            # unhoverable at their restored location, ghost hover zone
            # at the old one).  The full-rebuild branch gets it via
            # ``_load_from_data``.
            self._hover_dirty = True

        self.active_spline_idx = self._clamp_spline_idx(active)
        self._prev_active_spline_idx = self.active_spline_idx
        # Differential restore swaps node arrays in place — the
        # didactic cache's keyed buffers are now stale.  Drop it so
        # the next ``_compute_didactic`` rebuilds.
        self._didactic_geo_cache = None
        self._refresh_visuals()

    def _clamp_spline_idx(self, idx: int) -> int:
        """Clamps *idx* into the valid range of ``self.splines``.

        Returns 0 when the splines list is empty (instead of -1, which the
        naive ``min(idx, len(self.splines) - 1)`` would yield).  The
        downstream logic always expects at least one (possibly empty)
        spline; ``_load_from_data`` guarantees that invariant.
        """
        n = len(self.splines)
        if n == 0:
            return 0
        return max(0, min(int(idx), n - 1))

    @staticmethod
    def _decompose_tangent(tangent_full: np.ndarray) -> tuple[np.ndarray, float]:
        """Splits a 3-D tangent vector into ``(unit_direction, h_length)``.

        Falls back to ``(+x, 0.01)`` for a near-zero tangent so the node
        still renders something sensible after a degenerate save.
        """
        h_length = float(np.linalg.norm(tangent_full))
        if h_length > 1e-15:
            return tangent_full / h_length, h_length
        return np.array([1.0, 0.0, 0.0]), 0.01

    def _apply_record_to_node(self, seg: GeodesicSegment,
                              record: dict) -> None:
        """Repopulates *seg*'s geometry from a serialized node record.

        Accepts either of two schemas:

        **v2** (preferred, written by the current ``_on_save``):
            ``{origin, p_a, p_b}`` — both handle endpoints as literal
            3-D positions.  Reconstructed via the same solver call
            (``compute_endpoint_from_origin``) the editor uses during
            drag, so the geodesic between origin and each handle is
            identical (down to float precision) to what the user saw
            on screen at save time.  This is the only path that
            preserves user edits exactly: with a single tangent vector
            the solver-curving information is lost on round-trip.

        **v1** (legacy, ``{origin, tangent}``):
            ``tangent`` = direction × h_length.  Reconstructed via
            ``compute_shoot`` (parallel-transport ray) ± tangent_dir.
            Path_a is the symmetric ray of path_b.  This is what
            broke for the user: a handle dragged via the solver to a
            curved surface point landed ~0.2 units away on reload
            because compute_shoot does not curve to a target point.

        Schema dispatch is done by presence of ``'p_a'`` / ``'p_b'``
        keys.  If both are present the v2 branch runs; otherwise we
        fall back to ``'tangent'`` (v1).  Mixed schemas are rejected
        upstream by ``_validate_session_dict``.
        """
        origin = np.asarray(record['origin'], dtype=float)
        face_idx = self.geo.find_face(origin)
        normal, u, v = self._build_local_frame(origin, face_idx)
        seg.origin = origin
        seg.face_idx = face_idx
        seg.normal = normal
        seg.u = u
        seg.v = v
        # Invalidate the per-node solver cache: the cache (built by
        # ``GeodesicSegment._update_handle`` on first drag) is keyed
        # implicitly by the segment's origin — using a stale cache
        # after origin moves (undo / redo / load) would feed the
        # solver topology built around the *previous* origin to the
        # next ``compute_endpoint_from_origin`` call, drifting the
        # first preview frame post-restore.
        seg._origin_cache = None

        if 'p_a' in record and 'p_b' in record:
            self._apply_v2_handles(seg, origin, record)
        else:
            self._apply_v1_tangent(seg, origin, face_idx,
                                   np.asarray(record['tangent'], dtype=float))

        seg.update_local_v(self.geo)

    def _apply_v2_handles(self, seg: GeodesicSegment, origin: np.ndarray,
                          record: dict) -> None:
        """v2 schema: rebuild path_a / path_b via the same solver
        ``update_from_a`` / ``update_from_b`` use during drag.

        ``compute_endpoint_from_origin`` requires an origin cache
        (``prepare_origin``, ~2-5 ms) — the same one the drag handler
        builds on first move and reuses for subsequent debounces.  We
        only need it once per node here.

        Each handle is independent: a None entry in the record yields
        path=None / p=None for that side (used by single-node placeholder
        splines).  If the solver fails for one side we log and degrade
        to ``compute_shoot`` along the straight-line direction — better
        than losing the node entirely.
        """
        p_a_rec = record.get('p_a')
        p_b_rec = record.get('p_b')

        # Origin cache for the solver — built once, reused for both
        # handles.  If ``prepare_origin`` fails (degenerate face under
        # the saved origin, near-zero-area triangle), we cannot run
        # the solver here.  We don't fall back to ``compute_shoot``
        # automatically because v2 records do not store a tangent
        # direction — only the two handle endpoints.  Instead the
        # node loads with ``path_a = path_b = None`` and the editor's
        # span recomputation will skip its spans (visible as a gap in
        # the curve at that node).  The user's first drag of a handle
        # rebuilds the cache from the new mouse position and the node
        # recovers.
        try:
            origin_cache = self.geo.prepare_origin(origin)
        except (RuntimeError, ValueError, TypeError) as exc:
            log.warning(
                "v2 load: prepare_origin failed at %s (%s); "
                "node will load with no handles — drag any handle to recover.",
                origin.tolist(), exc)
            origin_cache = None

        def _resolve_handle(p_rec):
            if p_rec is None:
                return None, None
            p_target = np.asarray(p_rec, dtype=float)
            if origin_cache is None:
                return None, None
            try:
                path, _ = self.geo.compute_endpoint_from_origin(
                    origin_cache, p_target)
            except (RuntimeError, ValueError, TypeError, IndexError) as exc:
                log.debug("v2 load: solver failed for handle %s (%s); using straight line",
                          p_target.tolist(), exc)
                path = np.array([origin, p_target])
            if path is None or len(path) < 2:
                path = np.array([origin, p_target])
            return path, path[-1]

        seg.path_a, seg.p_a = _resolve_handle(p_a_rec)
        seg.path_b, seg.p_b = _resolve_handle(p_b_rec)

        # h_length: the editor maintains symmetric arc-length on path_a
        # / path_b after every drag (_update_symmetric_ray ensures this).
        # On reload pick whichever is available; if both, average so a
        # tiny solver asymmetry doesn't bias one side.
        lengths: list[float] = []
        for path in (seg.path_b, seg.path_a):
            if path is not None and len(path) >= 2:
                lengths.append(float(np.sum(
                    np.linalg.norm(np.diff(path, axis=0), axis=1))))
        seg.h_length = sum(lengths) / len(lengths) if lengths else 0.01

    def _apply_v1_tangent(self, seg: GeodesicSegment, origin: np.ndarray,
                          face_idx: int, tangent_full: np.ndarray) -> None:
        """v1 schema: rebuild via compute_shoot ± tangent_dir.

        Loses solver-curving information that may have been baked into
        the editor state when the user dragged a handle on a curved
        surface — that's the historical reason v2 was introduced.
        Kept for backwards compatibility with sessions saved before
        the format bump.
        """
        tangent_dir, h_length = self._decompose_tangent(tangent_full)
        seg.h_length = h_length
        path_b = self.geo.compute_shoot(origin, tangent_dir, h_length, face_idx)
        path_a = self.geo.compute_shoot(origin, -tangent_dir, h_length, face_idx)
        seg.path_b = path_b
        seg.path_a = path_a
        seg.p_b = path_b[-1] if path_b is not None else None
        seg.p_a = path_a[-1] if path_a is not None else None

    def _node_from_record(self, record: dict) -> GeodesicSegment:
        """Creates a new ``GeodesicSegment`` from a serialized node record (v1 or v2)."""
        origin = np.asarray(record['origin'], dtype=float)
        face_idx = self.geo.find_face(origin)
        normal, u, v = self._build_local_frame(origin, face_idx)
        seg = GeodesicSegment(origin, face_idx, normal, u, v)
        seg.is_active = True
        self._apply_record_to_node(seg, record)
        return seg

    def _rebuild_node_inplace(self, seg: GeodesicSegment, record: dict) -> None:
        """Reconstructs a single ``GeodesicSegment`` in place from a
        v1 or v2 serialized record.  Used by the differential undo/redo
        path to avoid destroying/recreating VTK actors for unchanged nodes.
        """
        self._apply_record_to_node(seg, record)
        seg.update_visuals(self.plotter)

    def _init_tangents(self, node: GeodesicSegment, direction: np.ndarray, length: float):
        """Shoots symmetric geodesic rays from *node* in ``+-direction`` for *length*.

        Sets ``p_a``, ``p_b``, ``path_a``, ``path_b``, and ``h_length`` on the node,
        then updates ``local_v`` for future parallel-transport translations.
        """
        cid = node.face_idx
        for sign, attr_p, attr_path in [(-1, 'p_a', 'path_a'), (1, 'p_b', 'path_b')]:
            path = self.geo.compute_shoot(node.origin, sign * direction, length, cid)
            if path is not None:
                setattr(node, attr_p, path[-1])
                setattr(node, attr_path, path)
        node.h_length = length
        node.update_local_v(self.geo)

    # --- Mouse interactions ---

    def _try_hit_marker(self, x: int, y: int) -> bool:
        """Hit-test using the parent's pre-built hover cache.

        Extends the parent's ``_try_hit_marker`` with spline-index
        switching: if the closest marker belongs to a different spline,
        the active spline is switched before initiating the drag.  Uses
        the same squared-distance, vectorized screen-projection path —
        no per-marker VTK coordinate calls, no ``np.linalg.norm``.
        """
        hit = self._closest_marker_under_cursor(x, y)
        if hit is None:
            return False
        seg, tag = hit
        s_idx = self._spline_for_node(seg)
        # Capture the pre-drag snapshot WITHOUT pushing it — the first
        # actual drag movement commits it (``_commit_pending_drag_undo``
        # in ``_on_move``); a click that never moves discards it in
        # ``_finalize_release``.
        self._pending_drag_snapshot = self._snapshot()
        # Save the pre-drag active spline so _finalize_release can restore
        # it.  Without this, dragging a node in a non-active spline
        # permanently switches away from the active one — the user loses
        # access to an empty (break) spline that has no clickable nodes.
        self._pre_drag_spline_idx = self.active_spline_idx
        # Lock camera FIRST so the setup below can fail safely — same
        # rollback contract as the base class: any exception unwinds
        # drag state and unlocks before re-raising, so the user is
        # never stranded with a frozen viewport.
        self._lock_camera()
        try:
            if s_idx != self.active_spline_idx:
                self.active_spline_idx = s_idx
                self._refresh_visuals()
            self.state.active_seg = seg
            self.state.drag_marker = tag
            seg.is_active = True
            seg.is_dragging = True
            self._set_hud(_t("dragging", marker=tag.upper()), 'gold')
            # Cancel background workers and hide orange immediately
            # — in the same render frame as the drag initiation, not on
            # the first _on_move (which would leave them visible for
            # 1+ frame).
            self._cancel_geodesic_spans(seg)
            seg.update_visuals(self.plotter)
            self.plotter.render()
        except Exception:  # noqa: BLE001 — must always reach unlock
            self.state.active_seg = None
            self.state.drag_marker = None
            seg.is_active = False
            seg.is_dragging = False
            self.active_spline_idx = self._pre_drag_spline_idx
            self._pending_drag_snapshot = None
            self._unlock_camera()
            raise
        return True

    def _clear_spline_spans(self, sid: int) -> None:
        """Removes all span actors and cancels all workers for spline *sid*.

        Called before structural changes (node insertion/deletion) that
        invalidate span indices.  After the change, ``_recompute_spans``
        rebuilds everything from scratch.
        """
        for cache in (self._span_cache, self._geo_span_cache):
            to_remove = [k for k in cache if k[0] == sid]
            for key in to_remove:
                _, actor = cache.pop(key)
                safe_remove_actor(self.plotter, actor)
                self._work_mgr.cancel_all_for_span(key)
                self._degraded_spans.discard(key)
        self._span_drag_state = {
            k: v for k, v in self._span_drag_state.items() if k[0] != sid}
        # Span set for this spline changed — invalidate hover cache.
        self._hover_curve_dirty = True
        # If the cleared spline is the active one, the didactic
        # scaffold's last-span identity may have shifted; drop the
        # cache so the next ``_compute_didactic`` rebuilds against
        # the new node objects.
        if sid == self.active_spline_idx:
            self._didactic_geo_cache = None

    def _shift_spline_caches(self, removed_sid: int) -> None:
        """Re-keys every sid-keyed cache after ``splines.pop(removed_sid)``.

        ``self.splines`` / ``splines_closed`` are already popped when
        this runs.  Entries of the removed spline are dropped (removing
        their actors — normally none exist, the removed spline is
        empty); entries of later splines shift down one index so they
        keep pointing at the same spline data.  In-flight orange
        workers of shifted splines are cancelled by the work manager —
        their pipe messages embed the OLD key and would land on the
        wrong spline — and resubmitted here under the new numbering.
        """
        def _shift(key):
            return (key[0] - 1, key[1]) if key[0] > removed_sid else key

        for name in ('_span_cache', '_geo_span_cache'):
            old = getattr(self, name)
            new = {}
            for k, v in old.items():
                if k[0] == removed_sid:
                    safe_remove_actor(self.plotter, v[1])
                else:
                    new[_shift(k)] = v
            setattr(self, name, new)
        self._span_drag_state = {
            _shift(k): v for k, v in self._span_drag_state.items()
            if k[0] != removed_sid}
        self._degraded_spans = {
            _shift(k) for k in self._degraded_spans if k[0] != removed_sid}
        for name in ('_interp_cache', '_interp_origins_buf',
                     '_interp_result_cache'):
            old = getattr(self, name)
            new = {}
            for s, v in old.items():
                if s == removed_sid:
                    if name == '_interp_cache':
                        safe_remove_actor(self.plotter, v[1])
                    continue
                new[s - 1 if s > removed_sid else s] = v
            setattr(self, name, new)
        # Cancel in-flight workers keyed with the old numbering and
        # resubmit the affected splines under their new index.
        affected = self._work_mgr.shift_spline_keys(removed_sid)
        for old_sid in affected:
            if old_sid == removed_sid:
                continue    # the removed spline itself — nothing to resubmit
            new_sid = old_sid - 1
            if 0 <= new_sid < len(self.splines):
                self._submit_geodesic_spans(sid=new_sid)
        self._hover_curve_dirty = True
        # Spline indices shifted under the didactic scaffold's feet —
        # its id()-keyed cache may now describe a different spline.
        self._didactic_geo_cache = None

    def _insert_node_from_interp(self, info: dict, sid: int,
                                nodes: list, closed: bool) -> None:
        """Inserts a node from a hover on the interpolation (black) curve.

        Unlike the Bezier layers, the interp curve has no span structure
        -- it is a single polyline per spline.  The insertion index is
        determined by the **splprep parameter** ``u`` along the curve:

          1. Each input origin's ``u`` is known directly from
             ``splprep`` (cached as ``u_at_nodes`` in
             ``_interp_result_cache``); no 3-D search needed.
          2. The hovered segment of the rendered polyline carries a
             per-point ``u`` (cached as ``u_per_pt``) propagated through
             secant subdivision; the hover ``u`` is the linear
             interpolation of the segment's endpoint ``u`` values.
          3. Insertion goes into the unique node-gap whose ``u``
             interval contains the hover ``u``.

        ``u`` is parametric and strictly monotonic, so this is robust
        on self-intersecting splines where 3-D nearest-vertex search
        previously mis-attributed a node to the wrong arm of a loop
        (the limitation called out in the legacy implementation).

        The tangent direction comes from the polyline segment at the
        hover point, projected onto the surface tangent plane at the
        insertion position.
        """
        pt = self.geo.project_to_surface(
            np.array(info['point'], dtype=float))
        cid = self.geo.find_face(pt)

        n_nodes = len(nodes)
        entry = self._interp_cache.get(sid)
        origins = np.array([n.origin for n in nodes], dtype=float)
        result_cached = self._interp_result_cache.get(sid)

        # --- Compute insertion position via splprep ``u`` parameter ---
        insert_pos = None
        if (entry is not None and entry[0].points is not None
                and len(entry[0].points) >= 2 and n_nodes >= 2
                and result_cached is not None and len(result_cached) >= 4):
            _fp, _projected, u_at_nodes, u_per_pt = result_cached
            n_pts = len(u_per_pt)
            seg = int(info['seg'])
            frac = float(info['frac'])
            if 0 <= seg < n_pts - 1:
                hover_u = float(u_per_pt[seg]
                                + (u_per_pt[seg + 1] - u_per_pt[seg]) * frac)
            elif seg >= n_pts - 1 and n_pts > 0:
                hover_u = float(u_per_pt[-1])
            else:
                hover_u = float(u_per_pt[0]) if n_pts > 0 else 0.0

            node_us = np.asarray(u_at_nodes, dtype=float)
            if closed:
                # ``node_us`` from splprep with ``per=True`` is monotonic
                # in [0, 1]; find the cyclic gap whose [u_i, u_{i+1})
                # contains hover_u, treating the last→first wrap as a
                # gap that crosses 1.0.
                sorted_idx = np.argsort(node_us)
                sorted_us = node_us[sorted_idx]
                pos_in_sorted = int(np.searchsorted(sorted_us, hover_u))
                if pos_in_sorted == 0 or pos_in_sorted >= n_nodes:
                    insert_pos = (sorted_idx[-1] + 1) % n_nodes
                else:
                    insert_pos = sorted_idx[pos_in_sorted - 1] + 1
            else:
                # Open: ``node_us`` is sorted by construction (splprep
                # without per= preserves input order along an open
                # curve).  Find the gap containing hover_u.
                for j in range(n_nodes - 1):
                    if node_us[j] <= hover_u <= node_us[j + 1]:
                        insert_pos = j + 1
                        break
                if insert_pos is None:
                    # Hover before first or after last → append at closer end
                    insert_pos = 0 if hover_u < node_us[0] else n_nodes

        if insert_pos is None:
            # Fallback: nearest-origin Euclidean (cache miss / degenerate)
            if n_nodes < 2:
                insert_pos = n_nodes
            else:
                dists = np.linalg.norm(origins - pt, axis=1)
                insert_pos = int(np.argmin(dists)) + 1

        # --- Tangent from the polyline direction at the hover segment ---
        if (entry is not None and entry[0].points is not None
                and len(entry[0].points) >= 2):
            seg_idx = info['seg']
            pts_3d = entry[0].points
            if seg_idx + 1 < len(pts_3d):
                tangent = pts_3d[seg_idx + 1] - pts_3d[seg_idx]
            else:
                tangent = pts_3d[-1] - pts_3d[-2]
        else:
            # Fallback: direction between neighbor nodes
            if n_nodes >= 2:
                i_prev = (insert_pos - 1) % n_nodes
                i_next = insert_pos % n_nodes
                tangent = origins[i_next] - origins[i_prev]
            else:
                tangent = np.array([1.0, 0.0, 0.0])

        # Project tangent onto surface tangent plane and normalize
        normal = self.geo.get_interpolated_normal(pt, cid)
        tangent = tangent - np.dot(tangent, normal) * normal
        tn = np.linalg.norm(tangent)
        if tn > 1e-12:
            tangent /= tn
        else:
            tangent = np.array([1.0, 0.0, 0.0])

        # Create node
        new_node = GeodesicSegment(pt, cid, *self._build_local_frame(pt, cid))
        new_node.is_active = True

        # Handle length from distances to neighbors
        if n_nodes >= 2:
            i_prev = max(0, insert_pos - 1)
            i_next = min(n_nodes - 1, insert_pos) if insert_pos < n_nodes else 0
            d0 = np.linalg.norm(pt - origins[i_prev])
            d1 = np.linalg.norm(pt - origins[i_next])
            h_len = min(d0, d1) * self.scfg.HANDLE_FRACTION
        else:
            h_len = self.diag * self.scfg.INITIAL_H_FRACTION
        self._init_tangents(new_node, tangent, h_len)

        # Clear span caches (indices shift)
        self._clear_spline_spans(sid)

        # Insert
        nodes.insert(insert_pos, new_node)
        self.segments.append(new_node)
        new_node.update_visuals(self.plotter)
        self._hover_dirty = True
        self._rebuild_node_index()
        self._refresh_visuals()
        self._recompute_spans()
        self._submit_geodesic_spans()
        self._set_hud(_t("node_inserted_interp"), 'lime')
        self.plotter.render()

    # --- Bézier split at curve hover — helper methods ---

    def _recover_t_from_hover(self, info: dict, layer: str,
                              sid: int, span_idx: int) -> float:
        """Recovers the Bézier parameter *t* from the polyline hover position.

        The displayed polyline may have non-uniform parameter spacing
        (adaptive sampling, secant subdivision), so ``info['seg']/N`` is
        wrong.  Uses **arc-length fraction** along the polyline — robust
        against any sampling distribution.  Returns 0.5 when the polyline
        is unavailable or degenerate, clamped to ``[0.01, 0.99]``.
        """
        layer_cache = {
            'blue': self._span_cache,
            'orange': self._geo_span_cache,
        }[layer]
        curve_entry = layer_cache.get((sid, span_idx))
        t = 0.5
        if curve_entry is not None and curve_entry[0].points is not None:
            pts_3d = curve_entry[0].points
            if len(pts_3d) >= 2:
                diffs = np.diff(pts_3d, axis=0)
                seg_lens = np.linalg.norm(diffs, axis=1)
                total_len = seg_lens.sum()
                if total_len > 1e-15:
                    seg_idx = info['seg']
                    frac = info['frac']
                    len_before = seg_lens[:seg_idx].sum()
                    len_partial = (seg_lens[seg_idx] * frac
                                   if seg_idx < len(seg_lens) else 0.0)
                    t = (len_before + len_partial) / total_len
        return max(0.01, min(0.99, t))

    def _de_casteljau_split(self, ctrl: tuple, paths: tuple, t: float,
                            use_geodesic: bool) -> dict:
        """Computes de Casteljau intermediate points at parameter *t*.

        de Casteljau triangle (cubic Bézier with 4 control points)::

            Level 0:   P0  ----  H_out  ----  H_in  ----  P1
                         \\        /  \\        /  \\        /
                          \\  lerp/    \\  lerp/    \\  lerp/
                           \\    /      \\    /      \\    /
            Level 1:       b01  -------  b12  -------  b23
                              \\          /  \\          /
                               \\   lerp /    \\   lerp /
                                \\      /      \\      /
            Level 2:             c0  -----------  c1
                                    \\           /
                                     \\   lerp  /
                                      \\       /
            Level 3:                    Q  (the split point)

        Each arrow is a ``lerp(A, B, t)``.  On our curved surface, level-1
        lerps are GEODESIC along the pre-existing paths (``path_b`` for
        P0→H_out, ``path_a`` reversed for H_in→P1, always a freshly
        computed geodesic for H_out→H_in).  Levels 2-3 use Euclidean
        lerp + ``project_to_surface``.

        Returns a dict with keys ``b01, b12, b23, c0, c1`` — the level-1
        and level-2 points.  Level-3 (Q) is NOT computed here; the caller
        uses the hover point directly instead (exactly where the user
        clicked, avoiding projection drift from levels 2-3).

        Parameters
        ----------
        ctrl : ``(P0, H_out, H_in, P1)`` tuple of control points.
        paths : ``(path_b, path_a)`` geodesic polylines for the outer
            segments (may be ``None`` when not available).
        t : parameter in (0, 1).
        use_geodesic : True for orange layer, False for blue — selects
            geodesic lerp on outer segments vs Euclidean + projection.
        """
        P0, H_out, H_in, P1 = ctrl
        path_b, path_a = paths

        if use_geodesic and path_b is not None and len(path_b) >= 2:
            cum, total = GeodesicMesh.compute_path_lengths(path_b)
            b01 = GeodesicMesh.geodesic_lerp(path_b, t, cum, total)
        else:
            b01 = self.geo.project_to_surface(P0 * (1 - t) + H_out * t)

        b12 = self.geo.project_to_surface(H_out * (1 - t) + H_in * t)

        if use_geodesic and path_a is not None and len(path_a) >= 2:
            path_a_rev = path_a[::-1]
            cum, total = GeodesicMesh.compute_path_lengths(path_a_rev)
            b23 = GeodesicMesh.geodesic_lerp(path_a_rev, t, cum, total)
        else:
            b23 = self.geo.project_to_surface(H_in * (1 - t) + P1 * t)

        c0 = self.geo.project_to_surface(b01 * (1 - t) + b12 * t)
        c1 = self.geo.project_to_surface(b12 * (1 - t) + b23 * t)

        return {'b01': b01, 'b12': b12, 'b23': b23, 'c0': c0, 'c1': c1}

    @staticmethod
    def _bezier_derivative_tangent(ctrl: tuple, t: float,
                                   normal: np.ndarray,
                                   fallback: np.ndarray) -> np.ndarray:
        """Computes the tangent direction at parameter *t* of a cubic Bézier.

        Uses ``B'(t) = 3(1-t)²(H-P0) + 6(1-t)t(H_in-H_out) + 3t²(P1-H_in)``,
        projects onto the surface tangent plane at the insertion point
        (removes component along *normal*), and normalizes.

        Falls back to the given *fallback* vector (typically ``c1 - c0``
        from the de Casteljau split) when the derivative is degenerate.
        """
        P0, H_out, H_in, P1 = ctrl
        omt = 1.0 - t
        deriv = (3.0 * omt * omt * (H_out - P0) +
                 6.0 * omt * t * (H_in - H_out) +
                 3.0 * t * t * (P1 - H_in))
        deriv -= np.dot(deriv, normal) * normal
        dn = np.linalg.norm(deriv)
        if dn > 1e-12:
            return deriv / dn
        tn2 = np.linalg.norm(fallback)
        if tn2 > 1e-12:
            return fallback / tn2
        return np.array([1.0, 0.0, 0.0])

    def _shorten_endpoint_handle(self, node: GeodesicSegment,
                                 origin: np.ndarray, new_tip: np.ndarray,
                                 which: str) -> None:
        """Shortens a node's outgoing handle to reach *new_tip* exactly.

        *which* is ``'b'`` (outgoing ``p_b``/``path_b``) or ``'a'``
        (incoming ``p_a``/``path_a``).  Recomputes the geodesic path,
        updates ``h_length`` from the actual arc-length, and syncs
        ``local_v``.

        Used in the endpoint rule of node insertion to give an exact
        de Casteljau split on endpoint nodes of open splines (where
        neighbor handle modification doesn't break C1 with another span).
        """
        direction_raw = new_tip - origin
        dist = np.linalg.norm(direction_raw)
        if dist < 1e-12:
            return
        direction = direction_raw / dist
        path = self.geo.compute_shoot(origin, direction, dist, node.face_idx)
        if path is None:
            path = np.array([origin, new_tip])
        if which == 'b':
            node.p_b = new_tip
            node.path_b = path
        else:
            node.p_a = new_tip
            node.path_a = path
        node.h_length = float(np.sum(np.linalg.norm(
            np.diff(path, axis=0), axis=1)))
        node.update_local_v(self.geo)

    def _insert_node_at_curve(self, info: dict) -> None:
        """Inserts a new C1 node at the curve hover point.

        The new node is placed at ``info['point']`` (projected onto the
        surface) — exactly where the user clicked, independent of the
        de Casteljau approximation.

        Orchestrates four helpers:
          1. ``_recover_t_from_hover``: arc-length fraction along the
             displayed polyline gives the Bézier parameter *t*.
          2. ``_de_casteljau_split``: level-1 and level-2 intermediate
             points used for handle shortening and derivative fallback.
          3. ``_bezier_derivative_tangent``: tangent direction for the
             new node's symmetric C1 handles.
          4. ``_shorten_endpoint_handle``: exact endpoint-rule handle
             shortening on open-spline endpoints.

        Interp-layer hovers are delegated to ``_insert_node_from_interp``
        (different logic: no de Casteljau, tangent from polyline
        direction).

        Endpoint rule for neighbor handle modification:
          - Open spline, first span: ``n0.p_b`` shortened to ``b01``.
          - Open spline, last span: ``n1.p_a`` shortened to ``b23``.
          - Closed spline or interior span: neighbors untouched
            (preserves C1 with adjacent spans).

        Span indices shift after insertion, so span caches for the
        affected spline are cleared and rebuilt.
        """
        sid = info['spline_idx']
        span_idx = info['span_idx']
        layer = info['layer']

        # Switch to the correct spline if needed
        if sid != self.active_spline_idx:
            self.active_spline_idx = sid
            self._refresh_visuals()

        nodes = self.splines[sid]
        closed = self.splines_closed[sid]

        # Interp layer: different logic (no de Casteljau, polyline tangent)
        if layer == 'interp':
            self._insert_node_from_interp(info, sid, nodes, closed)
            return

        n0 = nodes[span_idx]
        n1 = nodes[(span_idx + 1) % len(nodes)]
        P0, H_out, H_in, P1 = n0.origin, n0.p_b, n1.p_a, n1.origin

        # --- Step 1: parameter t from arc-length fraction ---
        t = self._recover_t_from_hover(info, layer, sid, span_idx)

        # --- Node position: exactly where the user clicked ---
        Q = self.geo.project_to_surface(np.array(info['point'], dtype=float))
        cid = self.geo.find_face(Q)
        new_node = GeodesicSegment(Q, cid, *self._build_local_frame(Q, cid))
        new_node.is_active = True

        # Fallback path: if the span has no handles yet, use simple midpoint
        if H_out is None or H_in is None:
            tangent = P1 - P0
            tn = np.linalg.norm(tangent)
            if tn > 1e-12:
                tangent /= tn
            d0 = np.linalg.norm(Q - P0)
            d1 = np.linalg.norm(Q - P1)
            self._init_tangents(new_node, tangent,
                                min(d0, d1) * self.scfg.HANDLE_FRACTION)
            can_modify_n0 = False
            can_modify_n1 = False
        else:
            # --- Step 2: de Casteljau intermediate points ---
            ctrl = (P0, H_out, H_in, P1)
            paths = (n0.path_b, n1.path_a)
            use_geodesic = (layer != 'blue')
            dc = self._de_casteljau_split(ctrl, paths, t, use_geodesic)

            # --- Step 3: tangent direction from Bézier derivative ---
            tangent = self._bezier_derivative_tangent(
                ctrl, t, new_node.normal, dc['c1'] - dc['c0'])

            # Handle length proportional to min distance to neighbors
            h_len = (min(np.linalg.norm(Q - P0), np.linalg.norm(P1 - Q))
                     * self.scfg.HANDLE_FRACTION)
            self._init_tangents(new_node, tangent, h_len)

            # --- Step 4: endpoint-rule neighbor handle shortening ---
            n_spans = self._span_count(sid)
            can_modify_n0 = (span_idx == 0) and not closed
            can_modify_n1 = (span_idx == n_spans - 1) and not closed
            if can_modify_n0:
                self._shorten_endpoint_handle(n0, P0, dc['b01'], which='b')
            if can_modify_n1:
                self._shorten_endpoint_handle(n1, P1, dc['b23'], which='a')

        # --- Insert into data structures ---
        self._clear_spline_spans(sid)
        insert_pos = span_idx + 1
        if closed and insert_pos >= len(nodes):
            insert_pos = len(nodes)
        nodes.insert(insert_pos, new_node)
        self.segments.append(new_node)

        # Rebuild visuals — new node always, neighbors only if modified
        new_node.update_visuals(self.plotter)
        if can_modify_n0:
            n0.update_visuals(self.plotter)
        if can_modify_n1:
            n1.update_visuals(self.plotter)
        self._hover_dirty = True
        self._rebuild_node_index()
        self._refresh_visuals()
        self._recompute_spans()
        self._submit_geodesic_spans()
        self._set_hud(_t("node_inserted"), 'lime')
        self.plotter.render()

    def _on_press(self, obj, event) -> None:
        """VTK observer wrapper — must not propagate exceptions.

        The base class guards its own handler body; this override
        replaces it entirely, so it needs the same guard or a raising
        insertion/pick would crash VTK's event loop.
        """
        try:
            self._on_press_impl(obj, event)
        except Exception:  # noqa: BLE001 — VTK observer must not propagate
            log.exception("press handler failed")

    def _on_press_impl(self, obj, event) -> None:
        """Left-button press: single-click drags a marker, double-click adds/inserts a node.

        On double-click:
          - If the cursor is hovering a curve (``curve_hover_info`` set),
            a new node is **inserted** at that point, splitting the span.
          - Otherwise, a new node is **appended** to the end of the
            active spline.

        ``new_node.update_visuals()`` is called explicitly after insertion
        because ``_refresh_visuals`` only updates nodes whose state
        *changed* — a freshly created node has ``is_active=True`` by
        default, so it would be skipped and remain invisible.
        """
        x, y = self.plotter.iren.get_event_position()
        is_double = self.plotter.iren.interactor.GetRepeatCount() >= 1

        if not is_double:
            self._try_hit_marker(x, y)
            return

        # Node insertion on curve hover takes priority
        if self.curve_hover_info is not None:
            info = self.curve_hover_info
            self.curve_hover_info = None
            self._hide_curve_hover_marker()
            if not self._hover_info_live(info):
                # The spline changed under a marker the user never moved
                # off.  Drop the gesture instead of inserting into a span
                # that moved or vanished — and do it *before* the
                # ``_push_undo`` below, so a no-op click cannot spend an
                # undo slot and wipe the redo stack.
                self._set_hud(_t("hover_stale"), 'yellow')
                self.plotter.render()
                return
            self._push_undo()
            self._insert_node_at_curve(info)
            return

        pt, cid = self._pick()
        if pt is None:
            return
        if self.splines_closed[self.active_spline_idx]:
            return

        self._push_undo()
        nodes = self._active_nodes
        new_node = GeodesicSegment(pt, cid, *self._build_local_frame(pt, cid))
        new_node.is_active = True
        new_node.h_length = self.diag * self.scfg.INITIAL_H_FRACTION

        if nodes:
            vec = pt - nodes[-1].origin
            vn = np.linalg.norm(vec)
            if vn > 1e-9:
                v_dir, h_len = vec / vn, vn * self.scfg.HANDLE_FRACTION
                self._init_tangents(new_node, v_dir, h_len)
                if len(nodes) == 1 and nodes[0].p_b is None:
                    self._init_tangents(nodes[0], v_dir, h_len)
                    nodes[0].update_visuals(self.plotter)

        nodes.append(new_node)
        self.segments.append(new_node)
        new_node.update_visuals(self.plotter)
        self._hover_dirty = True
        self._rebuild_node_index()
        self._refresh_visuals()
        self._recompute_spans()
        # Submit only the new span (last node), not all — previous spans
        # already have their geodesic curves computed or in progress.
        self._submit_geodesic_spans(node=new_node)
        self.plotter.render()

    def _on_right_press(self, obj, event) -> None:
        """VTK observer wrapper — must not propagate exceptions."""
        try:
            self._on_right_press_impl(obj, event)
        except Exception:  # noqa: BLE001 — VTK observer must not propagate
            log.exception("right-press handler failed")

    def _on_right_press_impl(self, obj, event) -> None:
        """Right-button handler with two double-click behaviours.

        Single right-click: ignored (no behaviour bound).

        Double right-click:
          - **Over a red P marker** → open the coordinate-edit dialog
            (``_open_coordinates_dialog``).  The user types the desired
            world-space coordinates; the input is projected to the
            closest point on the surface and the node is moved there
            via ``update_from_p`` (parallel-transports the tangent).
            Right-button is used (not left) precisely so the gesture
            cannot be confused with the left-button drag-start.
          - **Over empty surface** → starts a new spline (break),
            preserving the historical behaviour.  Only fires when the
            current spline has at least one node so we don't create
            consecutive empty splines.
        """
        if self.plotter.iren.interactor.GetRepeatCount() < 1:
            return  # only double-click triggers a behaviour

        x, y = self.plotter.iren.get_event_position()
        hit = self._hit_test_marker(x, y, allowed_tags=('p',))
        if hit is not None:
            seg, _tag = hit
            # Re-entry guard: a fast triple-click could fire two right-press
            # events while the first dialog's mainloop is still active.
            # tkinter doesn't recover gracefully from nested mainloops in
            # the same root, so we just drop subsequent calls.
            if getattr(self, '_dialog_open', False):
                return
            # Set the flag *inside* the try so an exception thrown while
            # constructing the dialog (e.g. tkinter failing in headless
            # mode) cannot leave ``_dialog_open=True`` permanently
            # stuck — which would block every subsequent right-double-
            # click for the rest of the session.
            try:
                self._dialog_open = True
                parsed = self._open_coordinates_dialog(seg)
            finally:
                self._dialog_open = False
            if parsed is not None:
                self._move_node_to_coordinates(seg, parsed)
            return

        # Empty surface → new-spline (break) shortcut
        if self._active_nodes:
            self._push_undo()
            self.splines.append([])
            self.splines_closed.append(False)
            self.active_spline_idx = len(self.splines) - 1
            self._refresh_visuals()
            self._set_hud(_t("new_spline_started"), 'lime')
            self.plotter.render()

    # --- Coordinate-edit dialog (right-double-click on P marker) ---

    def _hit_test_marker(self, x: int, y: int,
                         allowed_tags: tuple[str, ...] | None = None
                         ) -> tuple[GeodesicSegment, str] | None:
        """Pure hit-test against the hover cache — no drag side-effects.

        Thin wrapper around the parent's
        ``_closest_marker_under_cursor``; kept as a separate method so
        the right-click ``coord-edit`` dialog (the only caller) reads
        cleanly and so future spline-aware filtering can be added here
        without touching the base class.
        """
        return self._closest_marker_under_cursor(x, y, allowed_tags)

    @staticmethod
    def _parse_coordinates(text: str) -> tuple[float, float, float] | None:
        """Parse an ``[x, y, z]`` / ``x, y, z`` / ``x y z`` string into a tuple.

        All three forms are accepted, with arbitrary whitespace and
        any combination of commas / spaces / square brackets.  The
        parser is deliberately strict on the *shape* (exactly three
        numeric tokens) and lenient on formatting: ``[ 1.2,3.4 5.6 ]``
        parses fine.

        Returns ``None`` on any of these failure modes (used by the
        dialog's live-validate feedback to colour the entry red and
        disable the OK button):

          * empty string
          * non-numeric token
          * not exactly three tokens
          * any token is NaN or +/- infinity
        """
        s = text.strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        # Treat both commas and arbitrary whitespace as separators.
        s = s.replace(',', ' ')
        parts = s.split()
        if len(parts) != 3:
            return None
        try:
            x, y, z = (float(p) for p in parts)
        except ValueError:
            return None
        # Reject NaN and +/- inf (find_face / projection don't handle them).
        for v in (x, y, z):
            if v != v or abs(v) == float('inf'):
                return None
        return (x, y, z)

    def _open_coordinates_dialog(self, seg: GeodesicSegment
                                 ) -> tuple[float, float, float] | None:
        """Modal Tk dialog for entering target coordinates with live preview.

        Visual style is intentionally minimal: no header label, a
        single monospace entry, and right-aligned OK / Cancel buttons
        below.  Uses ``ttk`` for the buttons (native theme on each
        platform) and a slightly larger UI font (``Segoe UI 11`` /
        platform fallback) for legibility.

        Behaviour
        ---------
        * Pre-fills the entry with the node's current ``origin`` so a
          small numerical adjustment is just a few keystrokes.
        * Live validation on every keystroke via a ``StringVar`` trace:
            - valid input → entry text black, OK enabled, **preview
              sphere shown** at the projected surface point.
            - invalid input → entry text red, OK disabled, **preview
              sphere hidden**.
        * Keyboard: ``<Return>`` accepts (only when OK is enabled);
          ``<Escape>`` and the window-X cancel.
        * The preview sphere is unconditionally hidden on dialog exit
          (OK or Cancel) — handled by the surrounding ``finally``.

        Returns the parsed ``(x, y, z)`` tuple on OK, or ``None`` on
        cancel.  Blocks the main thread for the duration of
        ``mainloop`` — same modal pattern as ``_on_load``'s file
        dialog.
        """
        import tkinter as tk
        from tkinter import ttk

        cur = seg.origin
        initial = f"{cur[0]:.6f}, {cur[1]:.6f}, {cur[2]:.6f}"

        # Pick a UI font that maps to a modern face on each platform.
        # Tk's default lookup falls back gracefully if the named family
        # is not installed (Windows ships Segoe UI; macOS and Linux
        # rebind to their native sans).
        ui_font = ('Segoe UI', 11)
        mono_font = ('Consolas', 11)

        root = tk.Tk()
        root.title("Set node coordinates")
        try:
            root.attributes('-topmost', True)
        except tk.TclError:
            pass  # platform-dependent, non-fatal

        # ttk styling: clam theme is consistent and modern across OSes.
        # A custom "Invalid.TEntry" style would be cleaner but tk.Entry
        # supports direct fg switching, which keeps the live-validate
        # path simple — so we use tk.Entry for the input and ttk for
        # the buttons / frames.
        try:
            ttk.Style(root).theme_use('clam')
        except tk.TclError:
            pass

        result: dict[str, tuple[float, float, float] | None] = {'value': None}

        container = ttk.Frame(root, padding=(18, 16, 18, 14))
        container.pack(fill='both', expand=True)

        entry_var = tk.StringVar(value=initial)
        entry = tk.Entry(
            container, textvariable=entry_var,
            font=mono_font, fg='black',
            relief='flat', borderwidth=0,
            highlightthickness=1,
            highlightbackground='#cccccc',
            highlightcolor='#666666',
        )
        entry.pack(fill='x', pady=(0, 14), ipady=4)
        entry.select_range(0, 'end')
        entry.icursor('end')
        entry.focus_set()

        btn_frame = ttk.Frame(container)
        btn_frame.pack(fill='x')

        def _on_ok() -> None:
            parsed = self._parse_coordinates(entry_var.get())
            if parsed is not None:
                result['value'] = parsed
                root.destroy()

        def _on_cancel() -> None:
            root.destroy()

        # Right-aligned OK / Cancel.  Pack from the right so the rightmost
        # button is Cancel (Windows convention).
        cancel_btn = ttk.Button(btn_frame, text='Cancel', command=_on_cancel)
        cancel_btn.pack(side='right')
        ok_btn = ttk.Button(btn_frame, text='OK', command=_on_ok)
        ok_btn.pack(side='right', padx=(0, 8))

        # Apply the chosen UI font globally to ttk widgets in this dialog.
        try:
            ttk.Style(root).configure('TButton', font=ui_font, padding=(12, 4))
            ttk.Style(root).configure('TFrame', background=root.cget('background'))
        except tk.TclError:
            pass

        def _validate(*_args) -> None:
            parsed = self._parse_coordinates(entry_var.get())
            if parsed is not None:
                entry.config(fg='black')
                ok_btn.state(['!disabled'])
                self._update_coord_preview(parsed)
            else:
                entry.config(fg='#c43030')
                ok_btn.state(['disabled'])
                self._hide_coord_preview()

        entry_var.trace_add('write', _validate)
        _validate()  # initial colour / button state / preview sphere

        # Enter accepts (only when OK is enabled); Escape cancels.
        def _on_enter(_event) -> None:
            if 'disabled' not in ok_btn.state():
                _on_ok()

        root.bind('<Return>', _on_enter)
        root.bind('<Escape>', lambda _e: _on_cancel())
        root.protocol('WM_DELETE_WINDOW', _on_cancel)

        try:
            root.mainloop()
        finally:
            # Always clean up the preview sphere — even if mainloop
            # exits abnormally (uncaught Tk exception, signal).
            self._hide_coord_preview()
            # ...and the interpreter.  ``_on_ok`` / ``_on_cancel``
            # normally destroy it, so this only fires on the abnormal
            # exit; destroying twice raises TclError, hence the guard.
            try:
                root.destroy()
            except tk.TclError:
                pass
        return result['value']

    def _update_coord_preview(self, target_xyz: tuple[float, float, float]) -> None:
        """Project *target_xyz* to the surface and show the 3-actor preview.

        The preview communicates two pieces of information:

          * **Where the node will land** — grey sphere on the surface.
          * **How far the typed point is off the surface** — second
            grey sphere at the typed coordinate (often floating above
            the mesh) plus a thin line between the two.  The line's
            length is exactly the projection distance.

        Uses the same projection path as ``_move_node_to_coordinates``:
        ``find_face`` populates ``self.geo._vtk_cp`` with the closest
        surface point; we snapshot that into the actor PolyData.  The
        ensuing ``plotter.render`` is necessary because tk's mainloop
        blocks the editor's normal Master Clock — without an explicit
        render none of the actors would update until the dialog closes.
        """
        input_pt = np.asarray(target_xyz, dtype=float)
        self.geo.find_face(input_pt)
        projected = np.array(self.geo._vtk_cp, dtype=float)

        # Projected sphere (on the surface)
        self._coord_preview_buf[0] = projected
        self._coord_preview_pd.points = self._coord_preview_buf
        self._coord_preview_pd.Modified()

        # Input sphere (typed coordinate, may be floating above surface)
        self._coord_preview_input_buf[0] = input_pt
        self._coord_preview_input_pd.points = self._coord_preview_input_buf
        self._coord_preview_input_pd.Modified()

        # Connector line.  ``update_line_inplace`` from gizmo writes
        # both points and the polyline connectivity into the existing
        # PolyData without allocating a temporary, matching the rest
        # of the editor's per-frame rendering style.
        update_line_inplace(
            self._coord_preview_line_pd,
            np.array([input_pt, projected], dtype=float))

        for actor in (self._coord_preview_actor,
                      self._coord_preview_input_actor,
                      self._coord_preview_line_actor):
            if not actor.GetVisibility():
                actor.SetVisibility(True)
        self.plotter.render()

    def _hide_coord_preview(self) -> None:
        """Hide all three preview actors and force a render.

        Idempotent — safe to call when actors are already invisible.
        Renders only once at the end so we don't pay three paints for
        what is logically a single state change.
        """
        any_visible = False
        for actor in (self._coord_preview_actor,
                      self._coord_preview_input_actor,
                      self._coord_preview_line_actor):
            if actor.GetVisibility():
                actor.SetVisibility(False)
                any_visible = True
        if any_visible:
            self.plotter.render()

    def _move_node_to_coordinates(self, seg: GeodesicSegment,
                                  target_xyz: tuple[float, float, float]) -> None:
        """Project *target_xyz* onto the surface and move *seg* there.

        The projection uses ``GeodesicMesh.find_face`` which goes through
        the VTK locator and writes the closest surface point into
        ``self.geo._vtk_cp`` as a side-effect.  We snapshot that
        immediately, then re-call ``find_face`` on the snapshot to get
        the face index that contains it (the original input may be far
        from the surface and would route through the KDTree fallback).

        The actual move is delegated to ``GeodesicSegment.update_from_p``
        with ``exact=True`` — same call the drag-of-P consolidation
        debounce uses, so the tangent is parallel-transported across
        the new face's normal exactly the same way as a manual drag.

        Records an undo snapshot, recomputes the affected spans, and
        submits orange workers — symmetric with ``_finalize_release``.
        """
        input_pt = np.asarray(target_xyz, dtype=float)

        # Project to the closest surface point.
        self.geo.find_face(input_pt)
        projected = np.array(self.geo._vtk_cp, dtype=float).copy()
        face_idx = self.geo.find_face(projected)

        self._push_undo()
        # If the node belongs to a different spline than the active one,
        # switch active first (mirrors _try_hit_marker's behaviour).
        s_idx = self._spline_for_node(seg)
        if s_idx != self.active_spline_idx:
            self.active_spline_idx = s_idx
            self._refresh_visuals()

        seg.update_from_p(projected, face_idx, self.geo, exact=True)
        seg.is_preview = False
        seg.is_dragging = False
        seg.update_visuals(self.plotter)

        self._hover_dirty = True
        self._invalidate_stitch_cache()
        self._recompute_spans(node=seg)
        self._submit_geodesic_spans(node=seg)
        self._refresh_visuals()

        self._set_hud(
            f"NODE MOVED TO [{projected[0]:.4f}, "
            f"{projected[1]:.4f}, {projected[2]:.4f}]",
            'lime', sticky_seconds=4.0)
        self.plotter.render()

    def _snap_point_to_edge(self, p: np.ndarray, face_idx: int | None
                            ) -> tuple[np.ndarray | None,
                                       tuple[int, int, float] | None]:
        """Projects *p* onto the closest edge of *face_idx* (clamped).

        Tests the 3 edges of the containing face; for each edge ``(a, b)``
        computes the perpendicular foot of *p*, clamped to ``t ∈ [0, 1]``
        so the snap point always lies within the edge segment (never
        outside).  Returns ``(snap_xyz, (va, vb, t))`` or ``(None, None)``
        when the face is degenerate.

        Edge snap is cheap (3 dot products + 3 comparisons) and stays on
        the surface by construction: every edge is a real mesh edge, so
        no re-projection is needed.  Useful for landing splines on CAD
        seam edges and feature creases.
        """
        if face_idx is None:
            return None, None
        F = self.geo.F[face_idx]
        V = self.geo.V
        best_d = np.inf
        best_pt = None
        best_info = None
        for i in range(3):
            ia, ib = int(F[i]), int(F[(i + 1) % 3])
            a = V[ia]
            edge = V[ib] - a
            L2 = float(np.dot(edge, edge))
            if L2 < 1e-18:
                continue
            t = float(np.dot(p - a, edge)) / L2
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            closest = a + t * edge
            d = float(np.linalg.norm(p - closest))
            if d < best_d:
                best_d = d
                best_pt = closest.copy()
                best_info = (ia, ib, t)
        return best_pt, best_info

    def _on_move(self, obj, event, *, pick_override=None) -> None:
        """VTK observer wrapper — must not propagate exceptions.

        The parent's ``_on_move`` guards its own body, but this
        override adds pre/post work (pick, snap, stitch preview via
        ``prepare_origin`` — which can raise on degenerate faces,
        curve hover) that runs outside that guard.
        """
        try:
            self._on_move_impl(obj, event, pick_override=pick_override)
        except Exception:  # noqa: BLE001 — VTK observer must not propagate
            log.exception("spline move handler failed")

    def _on_move_impl(self, obj, event, *, pick_override=None) -> None:
        """Spline-aware move handler.

        Picks once per frame and passes the result to the parent via
        ``pick_override`` so its ``_on_move`` skips the redundant
        O(log N) ray-cast.  After the parent processes hover detection
        and drag geometry, this method handles span recomputation,
        stitch preview, and **curve hover detection**.

        Processing order (when not dragging or hovering a handle):
          1. Parent: hover detection, cursor, render.
          2. Schedule ``'stitch_exact'`` debounce — every move resets
             the timer; fires the topology-inserted refinement
             (``_fire_stitch_exact``) once the cursor settles.
          3. Stitch preview fast update (vertex-snapped geodesic),
             gated by ``STITCH_SKIP_PX`` so sub-pixel twitches are
             cheap.  See ``_update_stitch`` for the two-tier pipeline.
          4. Curve hover: ``_detect_curve_hover`` finds the closest
             visible curve segment (blue/orange/interp), positions a
             colored marker, and stores metadata in ``curve_hover_info``
             for future node insertion.

        During drag, hybrid spans are recomputed AFTER the parent's
        render — they lag by one frame (~16 ms at 60 Hz, imperceptible).
        Geodesic (orange) spans are cancelled on drag start and
        resubmitted on debounce consolidation.
        """
        # Skip expensive ray-pick when hovering a marker (no drag active).
        hovering_marker = self.state.hover_seg is not None and self.state.active_seg is None
        dragged = self.state.active_seg
        # The parent's drag branch assigns a FRESH ``last_drag_q`` object
        # (``q.copy()``) on every processed drag movement — an identity
        # change after ``super()._on_move`` marks a real mutation this
        # frame, which is what commits the press-time undo snapshot.
        prev_drag_q = self.state.last_drag_q if dragged else None
        # Tell the parent to suppress the cursor when curve hover is active
        # (set BEFORE super()._on_move so the parent reads it).
        self._hide_cursor = self.curve_hover_info is not None
        if hovering_marker:
            pick_result = (None, None)
            super()._on_move(obj, event)
        else:
            pick_result = pick_override if pick_override is not None else self._pick()
            snap_indicator_pt: np.ndarray | None = None
            if dragged:
                iren = self.plotter.iren.interactor
                # Vertex / edge snap modifiers only make sense on P
                # (the node origin) — that's where the user wants to
                # land on a precise mesh feature.  On A / B handles the
                # same Shift modifier means "magnitude only" (direction
                # preserved); snapping the cursor there would
                # discretise the magnitude scalar and is undesirable.
                # See ``GeodesicSegment.update_magnitude`` and the
                # Shift dispatch in ``geo_shoot._on_move`` for the A/B
                # branch.
                snap_eligible = (self.state.drag_marker == 'p')
                if snap_eligible and pick_result[0] is not None:
                    # Shift wins over Ctrl when both are held -- vertex
                    # snap is a strict subset of edge snap (edge endpoints
                    # are vertices), so no disambiguation is needed.
                    if iren.GetShiftKey():
                        _, vi = self.geo._kdtree.query(pick_result[0])
                        snapped = self.geo.V[int(vi)].copy()
                        pick_result = (snapped, self.geo.find_face(snapped))
                        self._set_hud(_t("snap_vertex", idx=int(vi)), 'gold')
                        snap_indicator_pt = snapped
                    elif iren.GetControlKey():
                        snapped, info = self._snap_point_to_edge(
                            pick_result[0], pick_result[1])
                        if snapped is not None and info is not None:
                            pick_result = (snapped,
                                           self.geo.find_face(snapped))
                            va, vb, t = info
                            self._set_hud(
                                _t("snap_edge", va=va, vb=vb, t=t), 'cyan')
                            snap_indicator_pt = snapped
                self._stitch_actor.SetVisibility(False)
                self._cancel_geodesic_spans(dragged)
                self._update_snap_indicator(snap_indicator_pt)
            super()._on_move(obj, event, pick_override=pick_result)

        if dragged:
            # First processed movement of the gesture: commit the
            # press-time undo snapshot (see _try_hit_marker).
            if self.state.last_drag_q is not prev_drag_q:
                self._commit_pending_drag_undo()
            # A fresh preview move invalidates any prior exact
            # consolidation, so a subsequent release must recompute.
            self._consolidated_seg = None
            # Recompute spans after the parent processed the drag geometry.
            self._recompute_spans(node=dragged)
            self._hide_curve_hover_marker()
            self.curve_hover_info = None
            return

        # Not dragging anymore -- hide snap indicator if it was shown.
        if self._snap_indicator_actor.GetVisibility():
            self._snap_indicator_actor.SetVisibility(False)

        # Skip stitch preview and curve hover when hovering a handle marker
        if hovering_marker:
            self._hide_curve_hover_marker()
            self.curve_hover_info = None
            # Hide the grey stitch too — when the cursor lands on a
            # node / handle the stitch becomes a distracting third
            # visual that no longer indicates "next insertion".
            self._stitch_actor.SetVisibility(False)
            return

        # Schedule the exact-stitch refinement on every mouse-move event
        # (cheap dict update).  The Master Clock fires it once the cursor
        # has been still for STITCH_EXACT_DEBOUNCE_SEC.  Done before the
        # 3-px gate so a sub-3-px twitch still resets the timer.
        self._schedule_stitch_exact(pick_result[0])

        # Stitch preview — screen-pixel threshold (squared, no sqrt).
        # Skip the fast vertex-to-vertex redraw + curve-hover detection
        # on sub-3 px moves to avoid wasted work on tiny mouse twitches.
        x, y = self.plotter.iren.get_event_position()
        sdx = x - self._last_stitch_screen[0]
        sdy = y - self._last_stitch_screen[1]
        if sdx * sdx + sdy * sdy < self.scfg.STITCH_SKIP_PX_SQ:
            return
        self._last_stitch_screen = (x, y)

        self._update_stitch(q=pick_result[0])

        # Curve hover detection — show marker on nearest visible curve
        curve_changed = self._detect_curve_hover(x, y)
        # When the curve hover marker takes over, suppress the grey
        # stitch — the cursor is no longer on a "free" surface point
        # so the "from last node to here" preview becomes misleading
        # (the next click inserts at the curve's hover point, not at
        # the surface position the stitch would advertise).  Fast
        # path: only touch the actor when it is actually visible.
        if self.curve_hover_info is not None and self._stitch_actor.GetVisibility():
            self._stitch_actor.SetVisibility(False)
            curve_changed = True
        if curve_changed:
            self.plotter.render()

    def _reopen_spline_loop(self, sid: int) -> None:
        """Reopen a closed spline *in place*.

        Clears the first node's closing tangent (the phantom ``p_a`` that
        belonged only to the wrap-around span), drops that wrap-around
        span from both caches (removing its actors and cancelling its
        orange workers), marks the spline open, and flags hover dirty.

        Pure state mutation — the caller drives ``_recompute_spans`` /
        ``_refresh_visuals`` / ``render``.  Shared by the ``C``-key
        reopen, the Backspace-on-closed-spline path, and the
        break-undo path so all three stay in lockstep (a past drift
        between them stranded the wrap-around actor — see
        ``_on_backspace``).
        """
        nodes = self.splines[sid]
        self.splines_closed[sid] = False
        if nodes:
            first = nodes[0]
            first.p_a = None
            first.path_a = None
            first.update_visuals(self.plotter)
            # Wrap-around span of a closed N-node spline is span N-1.
            key = (sid, len(nodes) - 1)
            entry = self._span_cache.pop(key, None)
            if entry:
                safe_remove_actor(self.plotter, entry[1])
            self._work_mgr.cancel_all_for_span(key)
            entry_g = self._geo_span_cache.pop(key, None)
            if entry_g:
                safe_remove_actor(self.plotter, entry_g[1])
            # The actor is destroyed — drop its cached style state too,
            # or ``_set_span``'s style gate will skip painting the actor
            # recreated later under the same key (it would keep the
            # PyVista theme defaults instead of SPAN_COLOR / width).
            self._span_drag_state.pop(key, None)
            self._degraded_spans.discard(key)
        self._hover_dirty = True

    def _on_close_spline(self) -> None:
        """'C' key: toggles open/closed state of the active spline.

        If the spline is open (3+ nodes): closes it by computing a
        closing tangent (first node's ``p_a`` toward the last node),
        then auto-breaks into a new empty spline.

        If the spline is already closed: reopens it by clearing the
        first node's ``p_a`` tangent, removing the wrap-around span,
        and marking the spline open.  Stays on the same spline (no
        auto-break on reopen).
        """
        sid = self.active_spline_idx
        nodes = self.splines[sid]

        if not nodes:
            return

        # Already closed → reopen
        if self.splines_closed[sid]:
            self._push_undo()
            self._reopen_spline_loop(sid)
            self._recompute_spans()
            self._submit_geodesic_spans()
            self._refresh_visuals()
            self._update_stitch()
            self._set_hud(_t("loop_opened"), 'yellow')
            self.plotter.render()
            return

        # Open → close (requires 3+ nodes)
        if len(nodes) < 3:
            return

        # ``_push_undo`` is deliberately NOT called yet: both the
        # degenerate-vector check below and a failed closing shoot bail
        # out having mutated nothing, and an undo entry pushed for a
        # no-op click still clears the redo stack.  Same defect class
        # ``_commit_pending_drag_undo`` was introduced to fix for marker
        # clicks.  ``compute_shoot`` is a pure query, so everything the
        # decision needs can be resolved first.
        did_close = False
        first, last = nodes[0], nodes[-1]
        vec = first.origin - last.origin
        # Project onto first node's tangent plane for a rigorous
        # surface-tangent direction.  Raw Euclidean vec can point
        # through the mesh interior on curved surfaces.
        vec -= np.dot(vec, first.normal) * first.normal
        vn = np.linalg.norm(vec)
        if vn > 1e-9:
            v_dir = vec / vn
            h_len = np.linalg.norm(first.origin - last.origin) * self.scfg.HANDLE_FRACTION
            # Compute closing tangent for first node.  ``p_a`` points
            # BACKWARD, toward the last node (the same convention as
            # ``_init_tangents``' sign=-1 ray and the mirror handle the
            # first-close path reuses): the wrap-around span then
            # arrives at the first node moving along +v_dir, G1 with
            # span 0.  Shooting +v_dir placed the handle on the far
            # side and hooked the closing span around the node.
            path_a = None
            if first.p_a is None:
                path_a = self.geo.compute_shoot(
                    first.origin, -v_dir, h_len, first.face_idx)
            # Only close if the closing handle exists (already set, or
            # the shoot just succeeded).  Everything above this point is
            # read-only, so the undo slot is spent exactly when the
            # close actually happens.
            if first.p_a is not None or path_a is not None:
                self._push_undo()
                if path_a is not None:
                    first.p_a, first.path_a = path_a[-1], path_a
                    first.update_visuals(self.plotter)
                self.splines_closed[sid] = True
                did_close = True
                self._recompute_spans()
                self._submit_geodesic_spans(node=first)

        if not did_close:
            return

        # Auto-break: start a new spline (only after successful close)
        self.splines.append([])
        self.splines_closed.append(False)
        self.active_spline_idx = len(self.splines) - 1
        self._stitch_actor.SetVisibility(False)
        self._refresh_visuals()
        self._set_hud(_t("loop_closed_break"), 'cyan')
        self.plotter.render()

    def _on_backspace(self) -> None:
        """Backspace: removes the last node from the active spline, or undoes
        the last break if the spline is empty.

        Undo of a break restores the previous spline as active and reopens
        it if it was closed.  Affected span cache entries are cleaned up
        so no stale actors remain in the scene.
        """
        sid = self.active_spline_idx
        nodes = self.splines[sid]

        # Nothing to undo: single empty spline
        if not nodes and len(self.splines) <= 1:
            return

        # A structural mutation mid-gesture would strand the drag (ghost
        # gizmo on the popped node, locked camera on release) — abort it
        # cleanly first.
        self._abort_active_drag()

        self._push_undo()
        # Empty spline = undo the break
        if not nodes and len(self.splines) > 1:
            self.splines.pop(sid)
            self.splines_closed.pop(sid)
            # Every spline after ``sid`` just shifted down one index —
            # re-key the sid-keyed caches (and worker bookkeeping) or
            # their existing actors become unreachable ghosts and new
            # edits draw duplicates.
            self._shift_spline_caches(sid)
            self.active_spline_idx = len(self.splines) - 1
            self._rebuild_node_index()
            if self.splines_closed[self.active_spline_idx]:
                self._reopen_spline_loop(self.active_spline_idx)
                # ``_reopen_spline_loop`` is a pure state mutation — its
                # docstring makes the caller drive the recompute, and the
                # sibling reopen path below does exactly that.  Without
                # it the interp (black) curve keeps rendering the
                # periodic fit for a now-open spline (its fingerprint at
                # ``_recompute_interp_curve`` would catch the flag flip,
                # but nothing calls it) and the didactic scaffold keeps
                # drawing the wrap-around span that was just deleted.
                self._recompute_spans()
                self._submit_geodesic_spans()
                self._set_hud(_t("loop_opened"), 'yellow')
            else:
                self._set_hud(_t("break_removed"), 'yellow')
            self._refresh_visuals()
            self._update_stitch()
            self.plotter.render()
            return

        # Closed spline: the most recent structural op was the close, so
        # Backspace reopens the loop (undo the close) rather than popping
        # a node.  Popping a closed spline stranded the wrap-around span's
        # actor (only span N-2 was cleaned) and could drop a closed
        # 3-node spline to a closed 2-node one — a state
        # ``_validate_session_dict`` rejects, which silently dead-ends the
        # undo/redo chain the next time that snapshot is restored.
        if nodes and self.splines_closed[sid]:
            self._reopen_spline_loop(sid)
            self._recompute_spans()
            self._submit_geodesic_spans()
            self._refresh_visuals()
            self._set_hud(_t("loop_opened"), 'yellow')
            self._update_stitch()
            self.plotter.render()
            return

        if nodes:
            node = nodes.pop()
            if node in self.segments:
                self.segments.remove(node)
            node.clear_actors(self.plotter)
            if self.state.hover_seg is node:
                self.state.hover_seg = None
                self.state.hover_marker = None
            if self.state.pending_hover_revert_seg is node:
                self.state.pending_hover_revert_seg = None
                self.state.pending_debounces.pop('hover_revert', None)
            self._hover_dirty = True
            self._rebuild_node_index()
            removed_idx = len(nodes)
            if removed_idx > 0:
                key = (sid, removed_idx - 1)
                entry = self._span_cache.pop(key, None)
                if entry:
                    safe_remove_actor(self.plotter, entry[1])
                # Cancel all workers + hide actors for removed span
                self._work_mgr.cancel_all_for_span(key)
                for cache in (self._geo_span_cache,):
                    removed_entry = cache.pop(key, None)
                    if removed_entry:
                        safe_remove_actor(self.plotter, removed_entry[1])
                # Same style-state cleanup as ``_reopen_spline_loop`` —
                # a span recreated under this key must repaint.
                self._span_drag_state.pop(key, None)
                self._degraded_spans.discard(key)
            # The popped node's curve geometry is gone (or about to be
            # recomputed) — the curve-hover cache must rebuild, or the
            # telescopic-sight marker keeps appearing on the deleted
            # span (2→1-node pops trigger no _set_span call at all).
            self._hover_curve_dirty = True
            # The stitch cache is keyed by id(last node); the popped
            # node might BE that node, and a recycled address could
            # silently revive its solver for a different segment.
            self._invalidate_stitch_cache()
            self._recompute_spans()
        self._update_stitch()
        self.plotter.render()

    # --- Stitch preview ---

    def _refresh_stitch_cache(self) -> None:
        """Rebuilds the ``prepare_origin`` cache for the stitch preview.

        Called when the last node of the active spline changes (node
        placement, deletion, drag consolidation).  The cache inserts the
        node's origin into the mesh topology and builds a local solver,
        so that ``_update_stitch`` can compute geodesics starting from the
        exact node position (~0.01 ms per query) instead of the nearest
        mesh vertex.

        Cost: ~2-5 ms (one-off).  Subsequent ``_update_stitch`` calls
        reuse the cached solver with vertex-snapped endpoint only.
        """
        nodes = self._active_nodes
        if not nodes or self.splines_closed[self.active_spline_idx]:
            self._stitch_origin_cache = None
            self._stitch_origin_node_id = -1
            return
        last = nodes[-1]
        nid = id(last)
        if nid == self._stitch_origin_node_id:
            return  # cache already valid for this node
        self._stitch_origin_cache = self.geo.prepare_origin(last.origin)
        self._stitch_origin_node_id = nid

    def _schedule_stitch_exact(self, q: np.ndarray | None) -> None:
        """Registers the exact-stitch refinement in the Master Clock.

        The fast vertex-to-vertex stitch is drawn synchronously on every
        mouse-move (see ``_update_stitch``).  This method overwrites the
        ``'stitch_exact'`` debounce entry with a fresh deadline, so the
        callback only fires once the cursor has been still for
        ``STITCH_EXACT_DEBOUNCE_SEC``.  Cheap (single dict update); called
        on every mouse-move event regardless of the 3-px stitch gate.
        """
        if q is None or self.state.active_seg is not None:
            self.state.pending_debounces.pop('stitch_exact', None)
            self._stitch_pending_q = None
            return
        nodes = self._active_nodes
        if not nodes or self.splines_closed[self.active_spline_idx]:
            self.state.pending_debounces.pop('stitch_exact', None)
            self._stitch_pending_q = None
            return
        self._stitch_pending_q = np.asarray(q, dtype=float).copy()
        self.state.pending_debounces['stitch_exact'] = (
            time.perf_counter() + self.scfg.STITCH_EXACT_DEBOUNCE_SEC,
            self._fire_stitch_exact,
        )

    def _fire_stitch_exact(self) -> None:
        """Replaces the fast stitch with the exact origin→cursor geodesic.

        Fires from the Master Clock after the cursor has been still for
        ``STITCH_EXACT_DEBOUNCE_SEC``.  Uses
        ``compute_endpoint_from_origin`` (~25 ms): topology-inserted
        origin from the cache + topology-inserted endpoint at the exact
        cursor position, no vertex snap.  ``_on_poll_timer`` batches the
        render — do not call ``render()`` here.

        Defensive checks abort the refinement if spline state has
        changed since the task was scheduled (drag started, spline
        closed, last node removed, etc.) or if the solver returned a
        degraded path; in those cases the fast vertex-snapped line stays
        on screen unchanged.
        """
        q = self._stitch_pending_q
        self._stitch_pending_q = None
        if q is None or self.state.active_seg is not None:
            return
        nodes = self._active_nodes
        if not nodes or self.splines_closed[self.active_spline_idx]:
            return
        cache = self._stitch_origin_cache
        if cache is None or self._stitch_origin_node_id != id(nodes[-1]):
            return
        # Fast stitch must already be on-screen — refine, don't conjure.
        if not self._stitch_actor.GetVisibility():
            return
        try:
            pts, was_fallback = self.geo.compute_endpoint_from_origin(cache, q)
        except (RuntimeError, ValueError, TypeError, IndexError) as exc:
            log.debug("stitch exact refinement failed: %s", exc)
            return
        if was_fallback or pts is None or len(pts) < 2:
            return
        update_line_inplace(self._stitch_pd, pts)

    def _invalidate_stitch_cache(self) -> None:
        """Forces stitch cache rebuild on next ``_refresh_stitch_cache``."""
        self._stitch_origin_cache = None
        self._stitch_origin_node_id = -1

    def _update_snap_indicator(self, pt: np.ndarray | None) -> None:
        """Shows / hides / repositions the gold snap-target sphere.

        *pt* is the on-surface point the drag will land on after Shift
        (vertex) or Ctrl (edge) snapping.  Pass ``None`` to hide the
        indicator (no snap modifier held, or no valid snap target).
        """
        if pt is None:
            if self._snap_indicator_actor.GetVisibility():
                self._snap_indicator_actor.SetVisibility(False)
            return
        self._snap_indicator_buf[0] = pt
        self._snap_indicator_pd.points = self._snap_indicator_buf
        self._snap_indicator_pd.Modified()
        self._snap_indicator_actor.SetVisibility(True)

    def _update_stitch(self, q=None) -> None:
        """Updates the prospective-span preview from last node to cursor.

        Two-tier pipeline:

          1. **Fast vertex-snap** (this method, ~0.01 ms): exact
             topology-inserted origin via the ``prepare_origin`` cache,
             cursor endpoint snapped to its nearest mesh vertex.  Runs
             on every qualifying mouse-move (gated by ``STITCH_SKIP_PX``).
          2. **Exact refinement** (``_fire_stitch_exact``, ~25 ms): fires
             from the Master Clock after the cursor has been still for
             ``STITCH_EXACT_DEBOUNCE_SEC``.  Replaces the vertex-snapped
             endpoint with a topology-inserted endpoint at the exact
             cursor position via ``compute_endpoint_from_origin``.
             Scheduled in ``_on_move`` independently of the 3-px gate, so
             sub-pixel movement still resets the timer.

        The Master Clock task key is ``'stitch_exact'``; it is overwritten
        on every move and discarded by defensive checks if spline state
        has changed since scheduling.

        Parameters
        ----------
        q : optional surface point.  When provided (from ``_on_move``),
            avoids a redundant ray-cast.  When None (from ``_on_backspace`` /
            ``_on_close_spline``), falls back to a fresh ``_pick()``.
        """
        nodes = self._active_nodes
        if self.splines_closed[self.active_spline_idx]:
            self._stitch_actor.SetVisibility(False)
            return
        if q is None and nodes:
            q, _ = self._pick()
        if not nodes or q is None:
            self._stitch_actor.SetVisibility(False)
            return

        last = nodes[-1]
        vec = q - last.origin
        vn = np.linalg.norm(vec)
        if vn < 1e-9:
            self._stitch_actor.SetVisibility(False)
            return

        # Ensure origin cache is valid for current last node
        self._refresh_stitch_cache()
        cache = self._stitch_origin_cache

        if cache is not None:
            # Exact origin + vertex-snapped endpoint (~0.01 ms)
            idx_s = cache['idx']
            _, idx_e = cache['kdtree'].query(q)
            idx_e = int(idx_e)
            if idx_s == idx_e:
                self._stitch_actor.SetVisibility(False)
                return
            try:
                pts = cache['solver'].find_geodesic_path(idx_s, idx_e)
            except (RuntimeError, ValueError, TypeError, IndexError) as exc:
                # The wrapper raises a generic ``Exception`` from the
                # native solver; we cannot narrow the type and the
                # fallback (hide the stitch) is the same regardless.
                log.debug("stitch local solver failed: %s", exc)
                self._stitch_actor.SetVisibility(False)
                return
        else:
            # Fallback: both endpoints vertex-snapped (degenerate mesh)
            _, idx_s = self.geo._kdtree.query(last.origin)
            _, idx_e = self.geo._kdtree.query(q)
            idx_s, idx_e = int(idx_s), int(idx_e)
            if idx_s == idx_e:
                self._stitch_actor.SetVisibility(False)
                return
            try:
                pts = self.geo._solver.find_geodesic_path(idx_s, idx_e)
            except (RuntimeError, ValueError, TypeError, IndexError) as exc:
                log.debug("stitch global solver failed: %s", exc)
                self._stitch_actor.SetVisibility(False)
                return

        if pts is None or len(pts) < 2:
            self._stitch_actor.SetVisibility(False)
            return

        update_line_inplace(self._stitch_pd, pts)
        import gizmo
        self._stitch_actor.GetProperty().SetOpacity(gizmo.GIZMO_OPACITY)
        self._stitch_actor.SetVisibility(True)

    # --- Span rendering ---

    def _mark_span_degraded(self, key: SpanKey, degraded: bool) -> None:
        """Updates the ``_degraded_spans`` set and flashes a HUD warning.

        Called by ``_recompute_spans`` after every blue-layer
        ``compute_endpoint_local`` and by the orange worker drain loop.
        A transient HUD message fires only on the ``False → True``
        transition so the user isn't spammed while the flag remains set
        across redundant recomputations.
        """
        was_degraded = key in self._degraded_spans
        if degraded and not was_degraded:
            self._degraded_spans.add(key)
            # Sticky: a geodesic fallback is a real correctness signal
            # (the curve is no longer geodesic).  Without stickiness the
            # next routine HUD update — drag preview, hover, orange
            # progress — overwrites it within a frame and the user never
            # sees the warning.
            self._set_hud(_t("geodesic_fallback", sid=key[0], i=key[1]),
                          'red', sticky_seconds=3.0)
            # Clear any cached drag-state so _set_span will repaint.
            self._span_drag_state.pop(key, None)
        elif not degraded and was_degraded:
            self._degraded_spans.discard(key)
            self._span_drag_state.pop(key, None)

    def _set_span(self, sid: int, i: int, pts, dragging: bool = False) -> None:
        """Updates the geometry and style of span *(sid, i)*.

        When *dragging* is True (node is being dragged, preview state),
        the span is drawn thinner and lighter to signal "approximate".
        On consolidation (debounce fires, *dragging* becomes False) the
        span reverts to full color/width.

        Degraded spans (``key in self._degraded_spans`` — geodesic fell
        back to a straight line) are painted red regardless of drag state.
        """
        key = (sid, i)
        if key not in self._span_cache:
            pd = pv.PolyData()
            actor = self.plotter.add_mesh(pd, lighting=False, pickable=False)

            self._set_depth_priority(actor, self.scfg.DEPTH_BLUE)
            self._span_cache[key] = (pd, actor)

        pd, actor = self._span_cache[key]
        if pts is None or len(pts) < 2:
            if actor.GetVisibility():
                self._hover_curve_dirty = True
            # Clear geometry so stale data can't reappear — same
            # rationale as ``_set_geo_span`` / ``_set_interp_curve``.
            # Hiding alone is not enough: ``_refresh_visuals`` and
            # ``_toggle_layer`` both re-show every span actor of a
            # spline unconditionally, so a blanked span (e.g. a node
            # whose handles failed to solve, or an undo back to that
            # state) would pop back at its pre-blank shape.
            pd.points = np.zeros((0, 3), dtype=float)
            pd.Modified()
            actor.SetVisibility(False)
            return
        update_line_inplace(pd, pts)
        # Geometry changed — invalidate the hover cache so the next
        # mouse-move (without drag) rebuilds the buffer.
        self._hover_curve_dirty = True

        degraded = key in self._degraded_spans
        # Tri-state style key: (dragging, degraded).  Using ``None`` as the
        # unseen sentinel lets the first call always repaint.
        style_key = (dragging, degraded)
        if self._span_drag_state.get(key) != style_key:
            self._span_drag_state[key] = style_key
            sc = self.scfg
            prop = actor.GetProperty()
            if degraded:
                prop.SetColor(*sc.SPAN_FALLBACK_COLOR)
                prop.SetLineWidth(sc.SPAN_LINE_WIDTH)
                prop.SetOpacity(1.0)
            elif dragging:
                prop.SetColor(*sc.SPAN_DRAG_COLOR)
                prop.SetLineWidth(sc.SPAN_DRAG_LINE_WIDTH)
                prop.SetOpacity(sc.SPAN_DRAG_OPACITY)
            else:
                prop.SetColor(*sc.SPAN_COLOR)
                prop.SetLineWidth(sc.SPAN_LINE_WIDTH)
                prop.SetOpacity(1.0)
        actor.SetVisibility(self._layer_visible['blue'])

    def _span_count(self, sid: int) -> int:
        """Number of Bézier spans in spline *sid* (closed loops have one extra wrap-around span)."""
        n = len(self.splines[sid])
        if n < 2:
            return 0
        return n if self.splines_closed[sid] else n - 1

    def _span_pair(self, sid: int, i: int) -> tuple[GeodesicSegment, GeodesicSegment]:
        """Returns the ``(node_start, node_end)`` pair for span *i* of spline *sid*.

        For closed splines, index wraps around so that the last span
        connects the last node back to the first.
        """
        nodes = self.splines[sid]
        return nodes[i], nodes[(i + 1) % len(nodes)]

    @staticmethod
    def _adjacent_span_indices(idx: int, total: int, closed: bool) -> list[int]:
        """Returns the 1–2 span indices adjacent to node *idx*.

        A node participates in span ``idx-1`` (as end-node) and span
        ``idx`` (as start-node).  For closed splines, ``idx-1`` wraps
        via modulo so that node 0 correctly includes the closing span.
        For open splines, ``idx-1 < 0`` is discarded (no wrap).
        """
        if closed:
            candidates = [(idx - 1) % total, idx % total]
        else:
            candidates = [idx - 1, idx]
        # Deduplicate while preserving order, discard out-of-range
        seen = set()
        result = []
        for j in candidates:
            if 0 <= j < total and j not in seen:
                seen.add(j)
                result.append(j)
        return result

    def _iter_affected_spans(self, sid: int,
                             node: GeodesicSegment | None):
        """Yield ``(i, n0, n1)`` for every span whose endpoints depend
        on *node* (or every span when *node* is None).

        Shared iteration mechanics for ``_recompute_spans`` (blue +
        interp + didactic) and ``_submit_geodesic_spans`` (orange):

          * Resolves the node-to-span-index mapping via
            ``_adjacent_span_indices`` (or yields the full range).
          * Looks up each span's endpoint pair via ``_span_pair``.
          * Yields nothing when the spline is empty or *node* is not
            in the spline (avoids the duplicated ``try: idx = nodes.index(node)``
            block from the two consumers).

        Does **not** apply ``p_b is None`` / ``path_b is None``
        safety checks — those have caller-specific side effects
        (blue clears the actor; orange just skips submission), so
        the consumer keeps that branch.
        """
        nodes = self.splines[sid]
        total = self._span_count(sid)
        if total == 0:
            return
        if node is not None:
            try:
                idx = nodes.index(node)
            except ValueError:
                return
            indices = self._adjacent_span_indices(
                idx, total, self.splines_closed[sid])
        else:
            indices = range(total)
        for i in indices:
            n0, n1 = self._span_pair(sid, i)
            yield i, n0, n1

    def _recompute_spans(self, node=None, *, sid: int | None = None) -> None:
        """Recomputes Bézier spans for the active spline (or *sid* when given).

        When *node* is provided, only spans adjacent to that node are
        recomputed (exactly 2 for interior nodes, 1 for endpoints of
        open splines).  Otherwise all spans are recomputed.

        *sid* defaults to ``self.active_spline_idx``.  Pass it explicitly
        from callers that walk every spline (load, undo full restore,
        ``_rebuild_all_orange``) instead of mutating ``active_spline_idx``
        as a side channel — that pattern caused the
        ``audit fixes: cleanup leak, origin_cache invalidation`` regression.

        During drag preview (``node.is_dragging and node.is_preview``),
        affected spans use LOD sampling and are drawn with the lighter
        drag style (``SPAN_DRAG_COLOR``, thinner).  On consolidation
        (debounce sets ``is_preview=False``), the same spans are
        recomputed at full quality with normal appearance.
        """
        if sid is None:
            sid = self.active_spline_idx

        # The cheap fast-preview render (Euclidean H_out->H_in, no exact
        # path_12 solve, no secant pass) applies only while the drag is
        # actively previewing — i.e. ``is_preview``.  Consolidation
        # (``_fire_debounce`` on a mid-drag pause, or ``_on_release``)
        # clears ``is_preview`` *before* calling this, so those paths take
        # the exact branch even though ``is_dragging`` is still True.
        # Gating on ``is_dragging`` instead left a mid-drag pause on the
        # hybrid LOD while announcing "REFINED (EXACT)" — the exact solve
        # only ever ran on release, contradicting the documented
        # pause-to-consolidate behaviour.
        is_preview_drag = (node is not None and node.is_dragging
                           and node.is_preview)
        sc = self.scfg
        res = sc.DRAG_RESOLUTION if is_preview_drag else sc.RESOLUTION
        min_s = sc.DRAG_MIN_SAMPLES if is_preview_drag else sc.MIN_SAMPLES
        max_s = sc.DRAG_MAX_SAMPLES if is_preview_drag else sc.MAX_SAMPLES

        adaptive = sc.ADAPTIVE_SAMPLING
        ctrl = self._ctrl_scratch  # (4, 3) view; rows reused per span
        handleless: list[int] = []
        for i, n0, n1 in self._iter_affected_spans(sid, node):
            if n0.p_b is None or n1.p_a is None:
                # A handle is unsolvable (``compute_shoot`` returned
                # ``None`` — degenerate direction, boundary-adjacent
                # node) so this span has no curve at all.  Blank it, but
                # say so: a span silently vanishing mid-edit is the same
                # class of quiet failure the red fallback repaint exists
                # to prevent.  ``spline_export.compute_blue`` already
                # reports this exact condition; the editor was the mute
                # one.
                handleless.append(i)
                self._set_span(sid, i, None)
                continue
            ctrl[0] = n0.origin
            ctrl[1] = n0.p_b
            ctrl[2] = n1.p_a
            ctrl[3] = n1.origin
            n = self.geo.adaptive_samples(ctrl, res, min_s, max_s)
            t_vals = GeodesicMesh.curvature_adaptive_t_vals(ctrl, n) if adaptive else None
            # Two-mode Bezier: during drag use fast Euclidean+projection
            # for H_out->H_in (path_12=None); on consolidation compute the
            # exact geodesic path_12 via compute_endpoint_local for a
            # semi-geodesic curve (~25ms extra per span).  The solver
            # may return ``None`` (very rare — disconnected components,
            # all retries exhausted); treat that as "no extra accuracy
            # available" and fall back to the drag-style hybrid by
            # passing ``path_12=None``.
            path_12 = None
            if not is_preview_drag:
                path_12, was_fallback = self.geo.compute_endpoint_local(
                    n0.p_b, n1.p_a)
                if path_12 is not None and len(path_12) < 2:
                    path_12 = None
                # Track fallbacks only on consolidation.  During the fast
                # preview the hybrid skips the solver entirely so there is
                # nothing to flag.
                self._mark_span_degraded((sid, i), was_fallback)
            pts = self.geo.hybrid_de_casteljau_curve(
                ctrl, n0.path_b, n1.path_a, n, fast=is_preview_drag,
                t_vals=t_vals, path_12=path_12)
            # Phase 2 refinement: only on consolidation (no time pressure)
            if adaptive and not is_preview_drag and len(pts) >= 3:
                t2 = GeodesicMesh.refine_t_vals_by_curvature(pts, t_vals)
                if len(t2) > len(t_vals):
                    pts = self.geo.hybrid_de_casteljau_curve(
                        ctrl, n0.path_b, n1.path_a, len(t2),
                        t_vals=t2, path_12=path_12)
            projected = self.geo.project_smooth_batch(pts)
            if not is_preview_drag:
                projected = self.geo.subdivide_secant_chords(
                    projected, tol=self._secant_tol,
                    max_depth=self.scfg.SECANT_MAX_DEPTH)
            self._set_span(sid, i, projected, dragging=is_preview_drag)

        # Report skipped spans once per consolidation.  Gated on
        # ``is_preview_drag`` so a drag through the condition does not
        # repaint the HUD at frame rate; the log line is unconditional.
        if handleless:
            log.warning(
                "spline %d: %d span(s) have no curve (handles missing or "
                "unsolvable): %s", sid, len(handleless), handleless)
            if not is_preview_drag:
                self._set_hud(_t("span_no_handles", sid=sid,
                                 n=len(handleless)), 'yellow')

        # Interpolation curve tracks node origins — recompute on every call.
        self._recompute_interp_curve(sid, is_dragging=is_preview_drag)

        # Didactic scaffold (key 'd') — interactive refresh:
        #   * Only recompute when the drag actually affects the LAST
        #     span (the only one the scaffold visualises).  Drags on
        #     any other node leave the scaffold geometry unchanged so
        #     re-running ``_compute_didactic`` would be ~5-10 ms of
        #     pure waste per frame.  ``node is None`` is the
        #     full-recompute path (load / undo / structural change)
        #     and always refreshes.
        #   * During drag of an affected node: recompute with
        #     ``fast=True`` (Euclidean lines + ``project_smooth_batch``,
        #     ~5-10 ms — same approximation blue uses).  The scaffold
        #     follows the cursor live.
        #   * On consolidation (``is_dragging=False``): recompute with
        #     ``fast=False`` (exact geodesics via
        #     ``compute_endpoint_local``, ~75-125 ms).  The lines
        #     "snap" from the approximation to the exact geodesic —
        #     the visible snap is itself a teaching moment.
        if self._didactic_visible:
            if node is None or self._is_node_in_last_span(node):
                self._compute_didactic(fast=is_preview_drag)
        else:
            self._didactic_dirty = True

    # --- Interpolation curve (scipy B-spline through nodes, black) ---

    def _set_interp_curve(self, sid: int, pts: np.ndarray | None) -> None:
        """Updates the interpolation curve actor for spline *sid*."""
        if pts is None or len(pts) < 2:
            entry = self._interp_cache.get(sid)
            if entry is not None:
                if entry[1].GetVisibility():
                    self._hover_curve_dirty = True
                # Clear geometry so stale data can't reappear when
                # ``_toggle_layer`` / ``_refresh_visuals`` blanket
                # re-show the actor (same rationale as _set_geo_span).
                entry[0].points = np.zeros((0, 3), dtype=float)
                entry[0].Modified()
                entry[1].SetVisibility(False)
            return

        if sid not in self._interp_cache:
            pd = pv.PolyData()
            actor = self.plotter.add_mesh(pd, lighting=False, pickable=False)
            sc = self.scfg
            prop = actor.GetProperty()
            prop.SetColor(*sc.INTERP_COLOR)
            prop.SetLineWidth(sc.INTERP_LINE_WIDTH)
            prop.SetOpacity(sc.INTERP_OPACITY)
            self._set_depth_priority(actor, sc.DEPTH_INTERP)
            self._interp_cache[sid] = (pd, actor)

        pd, actor = self._interp_cache[sid]
        update_line_inplace(pd, pts)
        actor.SetVisibility(self._layer_visible['interp'])
        # Geometry changed — let hover detection rebuild on next idle move.
        self._hover_curve_dirty = True

    def _recompute_interp_curve(self, sid: int | None = None,
                                is_dragging: bool = False) -> None:
        """Recomputes the scipy B-spline interpolation curve for spline *sid*.

        Fits a B-spline (``splprep`` with ``s=0``) through the node
        origins, evaluates it at adaptively-sampled parameter values,
        projects all points onto the surface, and (on consolidation)
        subdivides secant chords.  For closed splines, ``per=True``
        produces a periodic (wrap-around) curve.

        During drag (*is_dragging=True*) the expensive secant chord
        subdivision is skipped — the curve stays fast (~1-3 ms).  On
        consolidation, the full refinement runs (~5-10 ms).

        When *sid* is None, recomputes all splines.

        Visibility gating
        -----------------
        Skipped when the interp layer is hidden (the default at startup).
        This is *unlike* the orange layer, which always computes via its
        ``_SpanWorkManager`` workers regardless of visibility — orange
        runs in child processes, so background work is free.  Interp
        runs synchronously on the main thread inside the drag-event
        loop, so computing it while invisible would cost 1-15 ms per
        frame stolen from the visible layers.  When the user toggles
        the layer ON, ``_toggle_layer`` triggers a one-shot recompute
        so the curve appears immediately at full quality.
        """
        if not self._layer_visible[LayerKind.INTERP]:
            return
        if sid is None:
            for s in range(len(self.splines)):
                self._recompute_interp_curve(s, is_dragging=is_dragging)
            return

        nodes = self.splines[sid]
        n_nodes = len(nodes)
        if n_nodes < 2:
            self._set_interp_curve(sid, None)
            self._interp_result_cache.pop(sid, None)
            return

        # Reuse a per-spline (N, 3) origins buffer.  Grows when N
        # changes; otherwise filled in place — eliminates the
        # ``np.array([n.origin for n in nodes])`` allocation each
        # drag frame.
        buf = self._interp_origins_buf.get(sid)
        if buf is None or buf.shape[0] != n_nodes:
            buf = np.empty((n_nodes, 3), dtype=float)
            self._interp_origins_buf[sid] = buf
        for ni, node in enumerate(nodes):
            buf[ni] = node.origin
        origins = buf
        closed = self.splines_closed[sid]

        # Need at least k+1 points for degree k
        k = min(3, n_nodes - 1)
        if closed and n_nodes < k + 1:
            self._set_interp_curve(sid, None)
            self._interp_result_cache.pop(sid, None)
            return

        # Content fingerprint: when origins + flags are bit-identical
        # to the previous successful call, reuse the projected
        # polyline directly.  Saves ~3-10 ms on no-op recomputes
        # (e.g. `_recompute_spans` runs interp every drag frame even
        # when only ANOTHER spline's node moved).
        fp = (origins.tobytes(), bool(closed), bool(is_dragging))
        cached = self._interp_result_cache.get(sid)
        if cached is not None and cached[0] == fp:
            self._set_interp_curve(sid, cached[1])
            return

        # scipy's ``per=True`` requires the last input point to
        # duplicate the first; when it does not, ``splprep`` silently
        # OVERWRITES the last point with the first, so the fitted curve
        # missed the true last node entirely.  Fit on an explicitly
        # wrapped copy for closed splines.
        if closed:
            pts_fit = np.vstack([origins, origins[:1]])
        else:
            pts_fit = origins
        try:
            tck, u = splprep(
                [pts_fit[:, 0], pts_fit[:, 1], pts_fit[:, 2]],
                s=0, k=k, per=closed)
        except Exception as exc:  # noqa: BLE001 — scipy raises bare Exception
            log.debug("splprep failed for spline %d: %s", sid, exc)
            self._set_interp_curve(sid, None)
            self._interp_result_cache.pop(sid, None)
            return

        # High base sample count -- the 3D B-spline has no geodesic
        # awareness, so shorter initial chords reduce surface deviation.
        # During drag, ~4x downsample keeps the cost ~1 ms instead of
        # ~5 ms per frame.  The overall shape is still recognisable
        # because the polyline is projected onto the surface; the user
        # gets the precise version on consolidation when secant
        # subdivision also runs.
        sc = self.scfg
        if is_dragging:
            n = max(sc.INTERP_DRAG_SAMPLES, k + 2)
        else:
            n = max(sc.INTERP_MIN_SAMPLES,
                    self.geo.adaptive_samples(origins, sc.RESOLUTION,
                                              sc.INTERP_MIN_SAMPLES, 500))

        u_fine = np.linspace(0.0, 1.0, n)
        x, y, z = splev(u_fine, tck)
        raw_pts = np.column_stack((x, y, z))

        projected = self.geo.project_smooth_batch(raw_pts)
        # Per-rendered-point splprep ``u`` parameter, propagated through
        # secant subdivision so node-insertion can locate the picked
        # segment in parametric space (robust on self-intersecting
        # splines, which 3-D distance can mis-attribute at the
        # crossing).  ``u_at_nodes = u`` gives each input origin its
        # exact parameter value directly from splprep — no need to
        # search for the closest polyline vertex per origin.
        u_per_pt = u_fine.copy()
        # Secant subdivision only on consolidation — too slow for drag
        if not is_dragging:
            mean_edge = float(np.sqrt(self.geo._face_edge_len2.mean()))
            interp_tol = mean_edge * sc.INTERP_SECANT_TOL_FACTOR
            projected, u_per_pt = self.geo.subdivide_secant_chords(
                projected, tol=interp_tol,
                max_depth=sc.INTERP_SECANT_MAX_DEPTH,
                labels=u_per_pt)

        u_at_nodes = np.asarray(u, dtype=float)
        if closed:
            # Drop the wrap duplicate's parameter (``u[-1] == 1.0``,
            # equivalent to node 0) so ``u_at_nodes`` keeps exactly one
            # entry per node — the shape ``_insert_node_from_interp``
            # expects.
            u_at_nodes = u_at_nodes[:-1]
        self._interp_result_cache[sid] = (fp, projected, u_at_nodes, u_per_pt)
        self._set_interp_curve(sid, projected)

    # --- Background curve layer (fully geodesic orange) ---

    def _set_geo_span(self, sid: int, i: int, pts: np.ndarray | None,
                      computing: bool = False) -> None:
        """Updates the orange geodesic-curve actor for span *(sid, i)*.

        Mirrors ``_set_span`` but operates on ``_geo_span_cache`` with the
        geodesic visual style (orange, thick).

        When *pts* is None, clears the PolyData geometry AND hides the
        actor.  This prevents stale data from reappearing if
        ``_refresh_visuals`` later re-shows the actor.

        *computing* controls the visual style used while the worker
        is still producing points:

          - **Color**: ``GEO_COLOR_COMPUTING`` (dimmer orange) while
            computing, ``GEO_COLOR`` (full orange) on consolidation.
            Clear binary signal for "still working / done".
          - **Dashed** (optional, ``GEO_DASHED_WHILE_COMPUTING``): when
            True, the partial curve is rendered as alternating odd
            1-indexed segments so the polyline looks like a dashed
            line while refining.  Switches to a solid polyline at
            consolidation.  Disable the flag for a solid-dimmer look
            without dashes.

        Degraded spans (geodesic fell back to a straight line) override
        both and use ``SPAN_FALLBACK_COLOR`` — a failure signal that
        should dominate any progress indicator.
        """
        key = (sid, i)
        if pts is None or len(pts) < 2:
            entry = self._geo_span_cache.get(key)
            if entry is not None:
                if entry[1].GetVisibility():
                    self._hover_curve_dirty = True
                # Clear geometry so stale data can't reappear
                entry[0].points = np.zeros((0, 3), dtype=float)
                entry[0].Modified()
                entry[1].SetVisibility(False)
            return

        if key not in self._geo_span_cache:
            pd = pv.PolyData()
            actor = self.plotter.add_mesh(pd, lighting=False, pickable=False)
            sc = self.scfg
            prop = actor.GetProperty()
            prop.SetColor(*sc.GEO_COLOR)
            prop.SetLineWidth(sc.GEO_LINE_WIDTH)
            prop.SetOpacity(sc.GEO_OPACITY)

            self._set_depth_priority(actor, sc.DEPTH_ORANGE)
            self._geo_span_cache[key] = (pd, actor)

        pd, actor = self._geo_span_cache[key]
        sc = self.scfg
        # Geometry of an orange span changed — invalidate hover cache.
        self._hover_curve_dirty = True
        use_dashed = computing and sc.GEO_DASHED_WHILE_COMPUTING
        if use_dashed:
            update_dashed_line_inplace(pd, pts)
        else:
            update_line_inplace(pd, pts)

        # Color priority: fallback > computing > final.
        prop = actor.GetProperty()
        if key in self._degraded_spans:
            prop.SetColor(*sc.SPAN_FALLBACK_COLOR)
        elif computing:
            prop.SetColor(*sc.GEO_COLOR_COMPUTING)
        else:
            prop.SetColor(*sc.GEO_COLOR)
        actor.SetVisibility(self._layer_visible['orange'])

    def _submit_geodesic_spans(self, node: GeodesicSegment | None = None,
                               *, sid: int | None = None) -> None:
        """Submits affected spans for background orange de Casteljau computation.

        Walks the same span set as ``_recompute_spans`` via the shared
        ``_iter_affected_spans`` generator (adjacent to *node*, or all
        spans if *node* is None).  For each span with complete control
        points, submits a worker to ``_work_mgr``.  Hides the orange
        actor while computation is in progress — the Master Clock
        timer will progressively reveal it as points arrive.

        *sid* defaults to ``self.active_spline_idx``; see ``_recompute_spans``
        for why callers that walk every spline should pass it explicitly.
        """
        if sid is None:
            sid = self.active_spline_idx
        sc = self.scfg
        for i, n0, n1 in self._iter_affected_spans(sid, node):
            span_key = (sid, i)
            if n0.p_b is None or n1.p_a is None:
                continue
            ctrl = [n0.origin, n0.p_b, n1.p_a, n1.origin]
            if n0.path_b is None or len(n0.path_b) < 2:
                continue
            if n1.path_a is None or len(n1.path_a) < 2:
                continue
            # Hide orange actor while recomputing
            self._set_geo_span(sid, i, None)
            # Submit the orange worker.  The manager owns the batch
            # counter — no per-call bookkeeping here.
            self._work_mgr.submit_span(
                span_key, ctrl,
                n0.path_b, n1.path_a[::-1],
                sc.GEO_SAMPLES, adaptive=sc.ADAPTIVE_SAMPLING,
                deviation_mode=sc.ORANGE_DEVIATION_MODE,
                subdiv_tol_factor=sc.ORANGE_SUBDIV_TOL_FACTOR,
                subdiv_max_depth=sc.ORANGE_SUBDIV_MAX_DEPTH,
                chord_bridging=sc.ORANGE_CHORD_BRIDGING,
                submesh_subdiv=sc.ORANGE_SUBMESH_SUBDIV,
                use_full_mesh=sc.ORANGE_USE_FULL_MESH)

    def _rebuild_all_orange(self) -> None:
        """Resubmits the fully-geodesic (orange) workers for **every** span
        across **every** spline, even when the user has not dragged.

        Bound to the ``r`` key.  Useful after toggling ``ADAPTIVE_SAMPLING``,
        loading a session, or recovering from a worker crash — situations
        where the orange polylines are stale or absent and a manual rebuild
        is faster than orchestrating dummy drags.

        Cancels any in-flight workers first so the batch counter starts
        fresh and the HUD reflects the new total.
        """
        self._work_mgr.cancel_all()
        for sid in range(len(self.splines)):
            self._submit_geodesic_spans(sid=sid)
        self._set_hud(_t("orange_rebuilt"), 'orange')
        self.plotter.render()

    # --- Didactic visualization (key 'd') ---

    def _toggle_didactic(self) -> None:
        """Press 'd': toggle the de Casteljau scaffold for the last span.

        Activates the four-line preview of the orange curve's de
        Casteljau construction at parameter ``self._didactic_t``
        (slider-controlled, default 0.5): path_12 (H_out↔H_in),
        path_c0 (b01↔b12), path_c1 (b12↔b23), path_final (c0↔c1).
        See ``_compute_didactic`` for the geometric interpretation
        and the per-actor docstring in ``__init__``.

        On the OFF→ON transition we lazy-create the t slider (see
        ``_ensure_didactic_slider``) and trigger a synchronous
        ``_compute_didactic()`` *unconditionally* so the editor feels
        frozen for ~75-125 ms once.  (The per-slot ``_didactic_geo_cache``
        inside ``_compute_didactic`` still short-circuits the t-invariant
        pieces, but the toggle itself does not gate the call.)  The OFF
        transition only hides the actors + disables the slider.
        """
        self._didactic_visible = not self._didactic_visible
        if self._didactic_visible:
            self._ensure_didactic_slider()
            self._didactic_slider.SetEnabled(1)
            self._compute_didactic()
            self._set_hud("DIDACTIC ON", 'white', sticky_seconds=1.5)
        else:
            self._hide_didactic_actors()
            if self._didactic_slider is not None:
                self._didactic_slider.SetEnabled(0)
            # Cancel any in-flight slider-tick consolidation — would
            # otherwise fire ~100 ms later and pay the cost of an
            # exact recompute whose result is never rendered.
            self.state.pending_debounces.pop('didactic_t', None)
            self._set_hud("DIDACTIC OFF", 'grey', sticky_seconds=1.5)
        self.plotter.render()

    # --- Imported guide polylines (Ctrl+X load, X toggle) ---

    def _clear_guides(self) -> None:
        """Removes every imported guide actor from the plotter.

        Called by ``_on_load_guides`` before importing a fresh set so
        re-imports do not accumulate stale actors.  ``safe_remove_actor``
        tolerates already-detached actors (e.g. after a partial cleanup).
        """
        for actor in self._guide_actors:
            safe_remove_actor(self.plotter, actor)
        self._guide_pds.clear()
        self._guide_actors.clear()

    def _on_load_guides(self) -> None:
        """Ctrl+X: import one or more VTK polydata files as guide curves.

        Opens a multi-select file dialog (anything pyvista can read:
        .vtk, .vtp, .ply, .stl, .obj).  Replaces any previously-loaded
        guides.  Only **line cells** are rendered — polygonal cells in
        the same file are dropped via ``pv.PolyData(points=..., lines=...)``
        reconstruction, so a mesh file with a few annotation polylines
        will not display the mesh surface as a wireframe.

        Container types handled (chosen by ``pv.read`` based on the file
        header, not the file extension — many tools write legacy ``.vtk``
        with a 1-D ``UnstructuredGrid`` header when the data is purely
        line cells):
          * ``PolyData``        — used directly.
          * ``UnstructuredGrid``— converted via ``vtkGeometryFilter``
            (preserves all line cells, drops volumetric cells).
          * ``MultiBlock``      — first PolyData block is taken; if the
            block is itself an UnstructuredGrid the same conversion
            applies.

        A file with zero line cells is skipped with a HUD warning; the
        rest of the selection still loads.
        """
        import tkinter as tk
        from tkinter import filedialog

        # ``finally``: without it an exception out of the dialog (a
        # ``TclError`` on a headless / broken display is the realistic
        # one) leaked a live Tcl interpreter per attempt.
        root = tk.Tk()
        try:
            root.withdraw()
            fpaths = filedialog.askopenfilenames(
                title="Load guide polylines (cell type 3 / 4)",
                filetypes=[
                    ("PyVista-readable", "*.vtk *.vtp *.ply *.stl *.obj"),
                    ("All files", "*.*"),
                ],
            )
        finally:
            root.destroy()
        if not fpaths:
            return

        self._clear_guides()

        # Always start a fresh import in the "visible (low opacity)"
        # state, regardless of where the previous toggle left
        # ``_guide_visible``.  Cancel any in-flight hold/fade so the
        # new actors don't inherit an ongoing animation aimed at the
        # actors we just discarded.
        self._guide_visible = True
        self._x_hold_was_visible = None
        self._guides_fade_start_t = None
        self.state.pending_debounces.pop('guides_fade', None)

        sc = self.scfg
        total_segs = 0
        loaded_files = 0
        last_err: str | None = None
        for fpath in fpaths:
            try:
                pd_raw = pv.read(fpath)
            except (OSError, RuntimeError, ValueError) as exc:
                log.error("guide load failed for %s: %s", fpath, exc)
                self._set_hud(_t("guides_load_failed",
                                 fname=os.path.basename(fpath),
                                 err=type(exc).__name__),
                              'red', sticky_seconds=4.0)
                last_err = str(exc)
                continue

            # Flatten MultiBlock — many "PolyData" .vtk files actually
            # arrive wrapped in a single-block container.
            if isinstance(pd_raw, pv.MultiBlock):
                blocks = [b for b in pd_raw
                          if isinstance(b, (pv.PolyData, pv.UnstructuredGrid))]
                if not blocks:
                    self._set_hud(_t("guides_empty",
                                     fname=os.path.basename(fpath)),
                                  'red', sticky_seconds=4.0)
                    continue
                pd_raw = blocks[0]

            # Promote UnstructuredGrid → PolyData via vtkGeometryFilter.
            # Tools that write legacy ``.vtk`` with only line cells often
            # emit a ``vtkUnstructuredGrid`` header even when the content
            # could legally be a ``vtkPolyData``.  ``vtkGeometryFilter``
            # is the canonical conversion: line / triangle / poly cells
            # are preserved as the corresponding PolyData cell types,
            # while 3-D cells (tetra, hexa, ...) are dropped to their
            # boundary faces — fine for our use, which then strips
            # everything except line cells two steps below.
            if isinstance(pd_raw, pv.UnstructuredGrid):
                gf = vtk.vtkGeometryFilter()
                gf.SetInputData(pd_raw)
                gf.Update()
                pd_raw = pv.wrap(gf.GetOutput())

            if not isinstance(pd_raw, pv.PolyData) or pd_raw.lines.size == 0:
                self._set_hud(_t("guides_empty",
                                 fname=os.path.basename(fpath)),
                              'red', sticky_seconds=4.0)
                continue

            # Strip everything except line cells so polygonal data in
            # the source file does not render as a wireframe overlay.
            pd_lines = pv.PolyData()
            pd_lines.points = pd_raw.points
            pd_lines.lines = pd_raw.lines
            n_segs = pd_lines.n_cells

            actor = self.plotter.add_mesh(
                pd_lines, color=sc.GUIDE_COLOR_HEX,
                line_width=sc.GUIDE_LINE_WIDTH, opacity=sc.GUIDE_OPACITY,
                lighting=False, pickable=False,
                # Index-prefixed: PyVista's ``name=`` *replaces* any
                # existing actor with the same name, so keying on the
                # basename alone silently dropped one of two same-named
                # guides picked from different directories in a single
                # multi-select (``left/curves.vtk`` + ``right/curves.vtk``)
                # while the HUD still reported both loaded.
                name=f"guide_{len(self._guide_actors)}_{os.path.basename(fpath)}",
            )
            self._set_depth_priority(actor, sc.DEPTH_GUIDE)
            actor.SetVisibility(self._guide_visible)
            self._guide_pds.append(pd_lines)
            self._guide_actors.append(actor)
            total_segs += n_segs
            loaded_files += 1

        if loaded_files:
            self._set_hud(_t("guides_loaded",
                             n_files=loaded_files, n_segs=total_segs),
                          'lime', sticky_seconds=3.0)
            # Curves+guides changed the on-screen line set — invalidate
            # the curve-hover cache so guide segments are not picked up
            # as hoverable spline points.
            self._hover_curve_dirty = True
        elif last_err is None:
            # Nothing loaded and no error path triggered — every file
            # was skipped via the "empty" branch; HUD already explained.
            pass

        self.plotter.render()

    # --- Node-index labels (visible while 'n' key is held) ---

    def _ensure_node_labels(self) -> None:
        """Resize the label pool to match the current node count and
        update every label's position + text.

        Multi-spline scenes get a ``"s{sid}:{n_idx}"`` prefix so the
        user can tell two same-indexed nodes apart; with a single
        spline the label is just the node index.  Pooling avoids the
        per-press churn of creating / destroying ``vtkActor``s as the
        user repeatedly checks node IDs.
        """
        sc = self.scfg
        all_nodes = list(self._iter_all_nodes())
        n_needed = len(all_nodes)
        n_have = len(self._node_labels)
        multi_spline = len(self.splines) > 1
        # Resolve label colour once (config stores hex; vtkTextProperty
        # wants an RGB triple in [0, 1]).
        rgb = pv.Color(sc.NODE_LABEL_COLOR_HEX).float_rgb

        # Grow the pool — each new actor inherits the same font / colour
        # so the only per-frame work is position + text.
        for _ in range(n_needed - n_have):
            actor = vtk.vtkBillboardTextActor3D()
            actor.SetDisplayOffset(*sc.NODE_LABEL_OFFSET_PX)
            prop = actor.GetTextProperty()
            prop.SetFontSize(sc.NODE_LABEL_FONT_SIZE)
            prop.SetColor(*rgb)
            prop.SetBold(True)
            prop.SetJustificationToCentered()
            prop.SetVerticalJustificationToBottom()
            actor.SetVisibility(False)
            # Attach to the overlay renderer (layer 1) — see
            # ``__init__`` for the rationale.  The main renderer
            # never owns the label actors, so they cannot be occluded
            # pixel-wise by the mesh in the z-buffer.
            self._overlay_renderer.AddViewProp(actor)
            self._node_labels.append(actor)

        # Trim a grossly oversized pool.  Pooling exists to absorb the
        # ±1 churn of adding / removing nodes, so a small surplus is
        # kept deliberately; without any trim, though, the actor count
        # was a session-lifetime high-water mark (load a 500-node
        # session, then a 5-node one, and 495 actors stayed attached to
        # the overlay renderer) — which the docstring above already
        # claimed did not happen.
        surplus = n_have - n_needed
        if surplus > _NODE_LABEL_POOL_SLACK:
            for label in self._node_labels[n_needed:]:
                self._overlay_renderer.RemoveViewProp(label)
            del self._node_labels[n_needed:]

        # Update or hide pre-existing actors.
        for i, label in enumerate(self._node_labels):
            if i >= n_needed:
                label.SetVisibility(False)
                continue
            s_idx, n_idx, node = all_nodes[i]
            label.SetPosition(node.origin[0], node.origin[1], node.origin[2])
            # 1-based indexing for human readability — matches the
            # ``// node N`` comments emitted in the session JSON.  Code-
            # facing references (HUD messages, span keys, log lines)
            # remain 0-based; this is purely a presentation choice for
            # the label overlay.
            if multi_spline:
                label.SetInput(f"s{s_idx + 1}:{n_idx + 1}")
            else:
                label.SetInput(str(n_idx + 1))

    def _refresh_node_label_visibility(self) -> None:
        """Toggles each label on/off according to the matching node's
        occlusion state in the main renderer's z-buffer.

        Labels are rendered in an overlay layer that ignores depth,
        so without this filter every label would show even for nodes
        on the far side of the mesh — defeating the "find node #4 on
        what I'm looking at" use case.  Ray-cast via
        ``_is_marker_occluded`` (already used for handle visibility):
        if the first mesh hit between camera and node is closer than
        the node itself, the node is hidden by the mesh and we drop
        its label.  Cost: one VTK locator query per node — same hot
        path the handle hover already uses, so it scales.
        """
        all_nodes = list(self._iter_all_nodes())
        for i, label in enumerate(self._node_labels):
            if i >= len(all_nodes):
                label.SetVisibility(False)
                continue
            _, _, node = all_nodes[i]
            if self._is_marker_occluded(node.origin):
                label.SetVisibility(False)
            else:
                label.SetVisibility(True)

    def _show_node_labels(self) -> None:
        """Make every visible node's index label appear (called on 'n'
        KeyPress).

        Idempotent — repeated calls from OS key-repeat refresh
        positions (so labels track a node being dragged while 'n' is
        held) AND re-evaluate per-label occlusion, which matters if
        the user pans / orbits while keeping 'n' pressed.
        """
        self._ensure_node_labels()
        self._refresh_node_label_visibility()
        self._node_labels_visible = True
        self.plotter.render()

    def _hide_node_labels(self) -> None:
        """Hide every node-index label (called on 'n' KeyRelease).

        Idempotent.  Does nothing when no labels have ever been
        created — the first 'n' press lazily populates the pool.
        """
        if not self._node_labels_visible:
            return
        for label in self._node_labels:
            label.SetVisibility(False)
        self._node_labels_visible = False
        self.plotter.render()

    def _set_guides_opacity(self, alpha: float) -> None:
        """Applies *alpha* to every imported guide actor.  Cheap (one
        ``SetOpacity`` per actor); does not issue a render — the caller
        decides when to flush."""
        for actor in self._guide_actors:
            actor.GetProperty().SetOpacity(alpha)

    def _on_key_press_guides(self, obj, event) -> None:
        """Raw VTK KeyPress handler for the 'x' hold-to-preview shortcut.

        First press of a hold cycle (``_x_hold_was_visible is None``)
        captures the logical visibility, cancels any in-flight fade,
        and forces every guide actor to opacity 1.0 + visible.  OS
        key-repeats arrive on this same handler — they are gated by
        the captured-state check so the snapshot is taken exactly once.
        Skips when a modifier is held so Ctrl+X (import) and Shift+X
        stay free.
        """
        iren = self.plotter.iren.interactor
        if iren.GetControlKey() or iren.GetAltKey() or iren.GetShiftKey():
            return
        key = iren.GetKeySym()
        if key not in ('x', 'X'):
            return
        if not self._guide_actors:
            # Match the legacy single-press feedback: tell the user
            # there is nothing to preview.  Idempotent under key-repeat
            # (HUD just stays sticky).
            self._set_hud(_t("guides_none"), 'grey', sticky_seconds=2.0)
            self.plotter.render()
            return
        if self._x_hold_was_visible is not None:
            return  # OS key-repeat — initial press already handled
        self._x_hold_was_visible = self._guide_visible
        # Cancel any in-flight fade so the press snaps to opaque
        # without a competing animation tick overwriting the alpha.
        self.state.pending_debounces.pop('guides_fade', None)
        self._guides_fade_start_t = None
        self._guide_visible = True
        for actor in self._guide_actors:
            actor.SetVisibility(True)
        self._set_guides_opacity(1.0)
        self.plotter.render()

    def _on_key_release_guides(self, obj, event) -> None:
        """Raw VTK KeyRelease handler — finalises the X hold cycle.

        Reads the captured pre-press state:
          - was *visible*  →  hide the guides outright (state toggles
            to hidden), reset opacity back to ``GUIDE_OPACITY`` so the
            next show starts from the resting alpha.
          - was *hidden*  →  keep visible and start a 500 ms ease-out
            fade from 1.0 down to ``GUIDE_OPACITY``.

        Deliberately does **not** re-check the modifier keys.  The press
        path already refuses to capture while a modifier is held, so
        ``_x_hold_was_visible is None`` is the correct and sufficient
        gate for "this release does not belong to a hold cycle" —
        Ctrl+X still cannot reach the toggle.  Re-checking here instead
        swallowed legitimate releases: press ``x``, then press Shift
        while still holding ``x``, and the release was dropped, leaving
        the guides pinned at opacity 1.0 with the cycle still open (the
        following ``x`` tap was then eaten by the captured-state guard
        in the press handler).  Same shape as ``_on_key_release_labels``.
        """
        iren = self.plotter.iren.interactor
        key = iren.GetKeySym()
        if key not in ('x', 'X'):
            return
        if self._x_hold_was_visible is None:
            return  # release without a captured press (e.g. modifier was held)
        was_visible = self._x_hold_was_visible
        self._x_hold_was_visible = None
        if was_visible:
            # Toggle to hidden.  Reset the resting alpha so the next
            # show / load cycle starts at ``GUIDE_OPACITY``.
            self._guide_visible = False
            for actor in self._guide_actors:
                actor.SetVisibility(False)
            self._set_guides_opacity(self.scfg.GUIDE_OPACITY)
            self._set_hud(_t("guides_off"), 'grey', sticky_seconds=1.5)
        else:
            # Toggle to visible — fade from the current opaque preview
            # back down to ``GUIDE_OPACITY`` so the appearance is
            # smooth, not a sudden dim.
            self._guide_visible = True
            self._guides_fade_start_t = time.perf_counter()
            self.state.pending_debounces['guides_fade'] = (
                self._guides_fade_start_t,
                self._tick_guides_fade,
            )
            self._set_hud(_t("guides_on"), 'lime', sticky_seconds=1.5)
        self.plotter.render()

    def _tick_guides_fade(self) -> None:
        """Master Clock callback that animates the post-release fade.

        Lerps every guide actor's opacity from 1.0 towards
        ``GUIDE_OPACITY`` over ``GUIDE_FADE_DURATION_SEC`` using an
        ease-out quadratic (1-(1-t)²).  Reschedules itself on the
        Master Clock until the elapsed fraction reaches 1.0; the poll
        timer issues ``render()`` after every fired callback so the
        animation runs at the heartbeat cadence (~50 ms ≈ 10 frames).
        Cancelling the fade is just ``pending_debounces.pop`` plus
        clearing ``_guides_fade_start_t``.
        """
        start = self._guides_fade_start_t
        if start is None:
            return
        duration = self.scfg.GUIDE_FADE_DURATION_SEC
        t = (time.perf_counter() - start) / duration if duration > 0 else 1.0
        if t >= 1.0:
            self._set_guides_opacity(self.scfg.GUIDE_OPACITY)
            self._guides_fade_start_t = None
            return
        if t < 0.0:
            t = 0.0
        # Ease-out quadratic: 1 - (1-t)²
        eased = 1.0 - (1.0 - t) * (1.0 - t)
        target = self.scfg.GUIDE_OPACITY
        alpha = 1.0 + (target - 1.0) * eased
        self._set_guides_opacity(alpha)
        # Reschedule on the next heartbeat tick.  Deadline == now means
        # ``_on_poll_timer`` fires us on its next iteration (~50 ms).
        self.state.pending_debounces['guides_fade'] = (
            time.perf_counter(),
            self._tick_guides_fade,
        )

    def _resolve_didactic_sid(self) -> int:
        """Spline index whose last span the didactic scaffold visualises.

        Walks backward from ``active_spline_idx`` looking for the first
        spline with at least 2 nodes.  This lets the scaffold remain
        anchored on the *previous* span right after the user inserts a
        break — ``Dbl-click R`` and ``LOOP CLOSED + BREAK`` both create
        a new empty spline and switch ``active_spline_idx`` to it, so
        without the walk-back the scaffold would vanish at the moment
        the user is most likely to want to discuss what they just
        finished.

        Returns -1 when no spline has a last span yet (fresh editor,
        or every spline reduced to a placeholder).
        """
        sid = self.active_spline_idx
        while sid >= 0:
            if sid < len(self.splines) and len(self.splines[sid]) >= 2:
                return sid
            sid -= 1
        return -1

    def _is_node_in_last_span(self, node: GeodesicSegment) -> bool:
        """True iff ``node`` is one of the two endpoints of the span the
        didactic scaffold currently visualises.

        The didactic span is resolved via ``_resolve_didactic_sid`` —
        typically the active spline's last span, but falls back to the
        previous spline when active is a post-break placeholder.

        Match rule:

          * Open spline of N nodes: ``nodes[N-2]`` and ``nodes[N-1]``.
          * Closed spline: wrap-around endpoints ``nodes[N-1]`` and
            ``nodes[0]``.

        Lookups use ``is`` (identity), not ``==``, since
        ``GeodesicSegment`` instances are referenced by identity
        throughout the editor.  Returns False when no didactic span
        exists — recomputing would be a no-op.
        """
        sid = self._resolve_didactic_sid()
        if sid < 0:
            return False
        nodes = self.splines[sid]
        if self.splines_closed[sid]:
            return node is nodes[-1] or node is nodes[0]
        return node is nodes[-2] or node is nodes[-1]

    def _hide_didactic_actors(self) -> None:
        """Hide all didactic actors — the four cascade lines and the
        level-3 collapse-point sphere — idempotently.
        """
        for actor in (*self._didactic_actors, self._didactic_point_actor):
            if actor.GetVisibility():
                actor.SetVisibility(False)

    def _cheap_geodesic(self, p0: np.ndarray, p1: np.ndarray,
                        n_samples: int = 16) -> np.ndarray:
        """Approximate geodesic from *p0* to *p1*: Euclidean line
        projected onto the surface.

        This is the same approximation ``hybrid_de_casteljau_curve``
        uses for the H_out↔H_in segment of the blue curve when it is
        called with ``path_12=None`` (i.e. during drag).  Cost is one
        ``project_smooth_batch`` call (Numba-JIT) — sub-millisecond on
        typical meshes.

        Used by ``_compute_didactic(fast=True)`` to keep the scaffold
        live during drag.  After release the scaffold re-renders with
        ``fast=False`` and the lines snap to the exact geodesic — the
        "snap" itself is informative (it shows the user how far the
        Euclidean approximation can drift on curved surfaces).
        """
        line = np.linspace(p0, p1, n_samples)
        return self.geo.project_smooth_batch(line)

    def _ensure_didactic_slider(self) -> None:
        """Lazy-create the t slider for the didactic scaffold.

        PyVista's ``add_slider_widget`` is non-trivial — it builds a
        ``vtkSliderWidget`` with its own representation, adds the
        observer, and ties it to the renderer.  Tearing it down on
        every toggle-off and rebuilding on every toggle-on would be
        wasteful (and visually flickers on some OpenGL drivers), so
        we build once, then enable / disable per toggle.

        Position: a horizontal slider in the bottom-left, just above
        the surface-opacity slider (``y = 0.10`` vs ``0.04``), 15%
        wide.  Same ``modern`` style for visual consistency.  The
        ``always`` interaction event makes the cascade refresh
        continuously while sliding (each tick costs ~75-125 ms; on
        modern hardware this still feels live).
        """
        if self._didactic_slider is not None:
            return
        self._didactic_slider = self.plotter.add_slider_widget(
            self._on_didactic_t_change,
            [0.0, 1.0], value=self._didactic_t,
            title="t", pointa=(0.0, 0.10), pointb=(0.15, 0.10),
            style='modern', fmt="%.2f", interaction_event='always',
        )

    def _on_didactic_t_change(self, value: float) -> None:
        """Slider callback: re-run the cascade at the new ``t``.

        Always uses ``fast=False`` (exact geodesic).  Rationale:
        ``path_12`` is t-INVARIANT and held in the ``'exact'`` slot
        of ``_didactic_geo_cache`` (along with the cumulative-length
        tables for path_b / path_a / path_12), so it is *not*
        recomputed per tick.  But the cache covers only the level-1
        middle segment: the level-2/3 chords (``path_c0``, ``path_c1``,
        ``path_final``) depend on ``t`` and are rebuilt every tick via
        three fresh ``compute_endpoint_local`` calls.  So each tick after
        the first still costs roughly ~75-300 ms (three local-submesh
        solves) — the cache saves only the fourth (path_12) solve, not
        the whole cascade.  The first tick pays all four.

        An earlier implementation alternated ``fast=True`` on the
        live tick with ``fast=False`` on a 100 ms debounce.  That
        produced a **visual jump** in the path_12 segment between
        each tick → debounce transition because the Euclidean
        projection used by ``fast`` differs from the true geodesic
        — same H_out/H_in, two different polylines.  Slider
        movement is fundamentally different from node-drag: only
        ``t`` changes, never the geometry, so the ``fast`` mode's
        purpose (cheap approximation while geometry mutates) does
        not apply here.
        """
        self._didactic_t = float(value)
        if not self._didactic_visible:
            return
        self._compute_didactic(fast=False)
        self.plotter.render()
        # Cancel any obsolete debounce from the legacy fast/exact
        # alternating implementation — defensive cleanup.
        self.state.pending_debounces.pop('didactic_t', None)

    def _compute_didactic(self, fast: bool = False) -> None:
        """Build the 4 auxiliary geodesic lines for the resolved
        didactic span at parameter ``self._didactic_t``.

        Span resolution: ``_resolve_didactic_sid`` returns the active
        spline's index when it has ≥ 2 nodes, otherwise walks backward
        to the previous spline.  This keeps the scaffold anchored on
        the most recent fully-formed span across breaks (``Dbl-click R``
        and ``LOOP CLOSED + BREAK`` both produce an empty active
        spline).

        "Last span" within the resolved spline depends on the
        open/closed flag:
          * Open spline of N nodes: span between ``nodes[N-2]`` and
            ``nodes[N-1]``.
          * Closed spline: the wrap-around span between ``nodes[N-1]``
            and ``nodes[0]``.

        When no spline has ≥ 2 nodes, or the relevant
        ``path_a`` / ``path_b`` is missing (e.g. the user just inserted
        a single node), the actors are hidden and a brief HUD note
        explains why.

        Two compute modes:
          * ``fast=False`` (default, used on toggle, slider, and
            consolidation post-drag): four ``compute_endpoint_local``
            calls — exact geodesics matching what the orange curve
            uses.  ~75-125 ms total.
          * ``fast=True`` (used while a node is being dragged):
            Euclidean line + ``project_smooth_batch`` for each of the
            four scaffold segments.  ~5-10 ms total — same approximation
            ``hybrid_de_casteljau_curve`` uses for blue's ``path_12``
            during drag (see ``_recompute_spans``), so during drag the
            scaffold is visually consistent with blue.  On debounce
            release the scaffold re-renders with ``fast=False`` and
            "snaps" to the exact geodesic — the visible difference
            between the two passes is itself didactic.

        The cascade as a function of ``t`` is what the slider exposes
        (callback ``_on_didactic_t_change``); each slider tick re-fires
        this method.
        """
        sid = self._resolve_didactic_sid()
        if sid < 0:
            self._hide_didactic_actors()
            self._didactic_dirty = True
            self._set_hud("DIDACTIC: no last span", 'grey', sticky_seconds=2.0)
            return
        nodes = self.splines[sid]

        if self.splines_closed[sid]:
            n0, n1 = nodes[-1], nodes[0]
        else:
            n0, n1 = nodes[-2], nodes[-1]

        if (n0.p_b is None or n1.p_a is None
                or n0.path_b is None or len(n0.path_b) < 2
                or n1.path_a is None or len(n1.path_a) < 2):
            self._hide_didactic_actors()
            self._didactic_dirty = True
            return

        H_out, H_in = n0.p_b, n1.p_a
        path_b = n0.path_b
        path_a_rev = n1.path_a[::-1]

        # ``_pair_path(p0, p1)`` resolves to either the exact geodesic
        # (slow, accurate) or a Euclidean line projected onto the
        # surface (fast, approximate — same trick the blue curve uses
        # via ``hybrid_de_casteljau_curve``'s ``path_12=None`` branch
        # during drag).  Closing over ``fast`` here keeps the four
        # resolution sites below readable.
        # Use the same submesh subdivision level as the orange
        # worker so the didactic scaffold's collapse point lands
        # exactly on the rendered orange curve.  Without this, the
        # orange (worker, with subdiv) and the didactic (main thread,
        # without subdiv) would compute slightly different geodesics
        # in coarse regions of the mesh.
        sub_lvl = int(self.scfg.ORANGE_SUBMESH_SUBDIV)
        def _pair_path(p0, p1):
            if fast:
                return self._cheap_geodesic(p0, p1)
            path, _ = self.geo.compute_endpoint_local(
                p0, p1, submesh_subdiv=sub_lvl)
            if path is None or len(path) < 2:
                return np.array([p0, p1])
            return path

        # --- Level 1: middle segment H_out -> H_in ---
        # path_12 (and the cumulative lengths of path_b / path_a /
        # path_12) are INVARIANT to ``t``: they only change when the
        # last-span endpoint nodes move.  Cache them so dragging the
        # didactic slider does not retrigger ``compute_endpoint_local``
        # (~75-125 ms per tick) every frame.
        #
        # Two slots: ``fast`` and ``exact``.  The slider tick path uses
        # ``fast`` (Euclidean lerp + projection); the debounce
        # consolidation 100 ms after the last tick uses ``exact``
        # (compute_endpoint_local).  An earlier single-slot cache
        # alternated between the two on every tick → debounce → tick
        # cycle, recomputing path_12 each time *and* showing a visible
        # jump because the two approximations differ on curved
        # surfaces.  Keeping both entries lets each path hit the cache
        # independently while the geometry is stable.
        #
        # Cache key (per slot) is by object identity of the geometry
        # buffers — every handle/origin drag produces fresh ndarrays in
        # ``SegmentData.update_*``, so ``id(...)`` flipping is the
        # natural invalidation trigger.  HAZARD: CPython recycles
        # addresses after GC, so two distinct objects can share an
        # ``id()`` over time.  We pin all 8 keyed objects via the
        # ``'refs'`` field below — they cannot be freed while the
        # cache entry is live, so ``id`` collisions are impossible.
        cache_key = (
            id(n0), id(n1),
            id(n0.origin), id(n0.p_b), id(n0.path_b),
            id(n1.origin), id(n1.p_a), id(n1.path_a),
        )
        if self._didactic_geo_cache is None:
            self._didactic_geo_cache = {}
        slot = 'fast' if fast else 'exact'
        cached = self._didactic_geo_cache.get(slot)
        if cached is not None and cached['key'] == cache_key:
            path_12 = cached['path_12']
            cum_b, total_b = cached['cum_b'], cached['total_b']
            cum_a, total_a = cached['cum_a'], cached['total_a']
            cum_12, total_12 = cached['cum_12'], cached['total_12']
        else:
            path_12 = _pair_path(H_out, H_in)
            cum_b, total_b = GeodesicMesh.compute_path_lengths(path_b)
            cum_a, total_a = GeodesicMesh.compute_path_lengths(path_a_rev)
            cum_12, total_12 = GeodesicMesh.compute_path_lengths(path_12)
            self._didactic_geo_cache[slot] = {
                'key': cache_key,
                # Strong refs to every object whose id() is in ``key`` —
                # prevents GC + address reuse from creating false hits.
                'refs': (n0, n1,
                         n0.origin, n0.p_b, n0.path_b,
                         n1.origin, n1.p_a, n1.path_a),
                'path_12': path_12,
                'cum_b': cum_b, 'total_b': total_b,
                'cum_a': cum_a, 'total_a': total_a,
                'cum_12': cum_12, 'total_12': total_12,
            }

        # Clamp defensively — the slider should already constrain the
        # value to [0, 1], but a hand-set ``self._didactic_t`` could
        # arrive out of range and break geodesic_lerp's path indexing.
        t = float(np.clip(self._didactic_t, 0.0, 1.0))
        b01 = GeodesicMesh.geodesic_lerp(path_b, t, cum_b, total_b)
        b12 = GeodesicMesh.geodesic_lerp(path_12, t, cum_12, total_12)
        b23 = GeodesicMesh.geodesic_lerp(path_a_rev, t, cum_a, total_a)

        # Level 2: two chords between consecutive level-1 midpoints.
        path_c0 = _pair_path(b01, b12)
        path_c1 = _pair_path(b12, b23)

        cum_c0, total_c0 = GeodesicMesh.compute_path_lengths(path_c0)
        cum_c1, total_c1 = GeodesicMesh.compute_path_lengths(path_c1)
        c0 = GeodesicMesh.geodesic_lerp(path_c0, t, cum_c0, total_c0)
        c1 = GeodesicMesh.geodesic_lerp(path_c1, t, cum_c1, total_c1)

        # Level 3: the chord that, evaluated at t, IS the orange curve
        # sample at parameter t.  Drawing it makes the cascade collapse
        # visually obvious.
        path_final = _pair_path(c0, c1)

        for pd, path in zip(self._didactic_pds,
                            (path_12, path_c0, path_c1, path_final), strict=False):
            update_line_inplace(pd, path)

        # Evaluate the cascade's collapse point — geodesic_lerp on
        # path_final at the same ``t``.  Place the level-3 marker
        # sphere there.  This point is, by construction, the orange
        # curve's sample at parameter ``t``; the marker visually
        # confirms the orange curve passes through it.
        cum_f, total_f = GeodesicMesh.compute_path_lengths(path_final)
        final_pt = GeodesicMesh.geodesic_lerp(path_final, t, cum_f, total_f)
        self._didactic_point_buf[0] = final_pt
        self._didactic_point_pd.points = self._didactic_point_buf
        self._didactic_point_pd.Modified()

        # All actors share visibility — flip on at the end so a
        # mid-compute exception leaves them in a clean state.  Local
        # import of ``gizmo`` mirrors ``_cycle_gizmo_opacity``: the
        # only callers that touch ``GIZMO_OPACITY`` are the toggle
        # path and this compute method, so deferring the import keeps
        # geo_splines start-up fast.
        import gizmo
        op = gizmo.GIZMO_OPACITY
        for actor in (*self._didactic_actors, self._didactic_point_actor):
            actor.SetVisibility(True)
            # Keep opacity in sync with the global handle opacity
            # (cycled via the 't' key inside ``_cycle_gizmo_opacity``).
            actor.GetProperty().SetOpacity(op)

        self._didactic_dirty = False

    def _cancel_geodesic_spans(self, node: GeodesicSegment) -> None:
        """Cancels running workers and hides geodesic actors for spans
        adjacent to *node*.  Called at the start of a drag.

        Cancellation works by closing the read end of the ``mp.Pipe``
        for each affected span (via ``_work_mgr.cancel_span``).  The
        worker's next ``send()`` raises ``BrokenPipeError`` and exits.
        """
        sid = self.active_spline_idx
        nodes = self.splines[sid]
        total = self._span_count(sid)
        if total == 0:
            return
        try:
            idx = nodes.index(node)
        except ValueError:
            return
        for j in self._adjacent_span_indices(
                idx, total, self.splines_closed[sid]):
            span_key = (sid, j)
            self._work_mgr.cancel_all_for_span(span_key)
            # Orange: hide immediately (too inaccurate to show stale)
            self._set_geo_span(sid, j, None)

    def _on_poll_timer(self, obj, event) -> None:
        """VTK timer observer wrapper — must not propagate exceptions."""
        try:
            self._on_poll_timer_impl(obj, event)
        except Exception:  # noqa: BLE001 — VTK observer must not propagate
            log.exception("poll timer failed")

    def _on_poll_timer_impl(self, obj, event) -> None:
        """Master Clock heartbeat — orchestrator only.

        The actual work is split across small helpers so each
        responsibility (drain, HUD, camera, render of progressive
        results, dead-worker cleanup) is testable in isolation:

          1. ``_apply_worker_fallbacks``  — degraded-span flag merging.
          2. ``_update_orange_hud``       — progress text in the HUD.
          3. ``_refresh_arrows_on_camera_change`` — fixed-screen-size arrows.
          4. ``_apply_orange_progress``   — progressive polyline updates.
          5. ``_clear_dead_orange_spans`` — actors orphaned by worker death.
        """
        super()._on_poll_timer(obj, event)
        has_worker_results = self._work_mgr.drain_queue()

        self._apply_worker_fallbacks()
        self._apply_shoot_truncation_hud()
        self._update_orange_hud()

        needs_render = self._refresh_arrows_on_camera_change()

        if has_worker_results and self._apply_orange_progress():
            needs_render = True
        # Dead spans can be flagged OUTSIDE drain_queue too —
        # ``submit_span``'s double-BrokenProcessPool give-up path adds
        # the key directly, with no pipe message to make drain_queue
        # return True — so the cleanup must run even on quiet ticks or
        # the key lingers until an unrelated worker message arrives.
        if self._work_mgr.dead_spans and self._clear_dead_orange_spans():
            needs_render = True

        if needs_render:
            self.plotter.render()

    def _apply_worker_fallbacks(self) -> None:
        """Reconciles the app-level ``_degraded_spans`` with the workers.

        For every span that emitted a ``'done'`` this tick, mirror the
        worker's final verdict: mark it degraded when a solver hit a
        straight-line fallback, and — crucially — *clear* it when the
        worker finished clean.  The old code merged only ``True`` flags,
        so a span painted red once stayed red forever: rebuilding it
        (``R``) with a clean result never propagated the clear, because
        the manager's clean-``done`` ``discard`` only emptied its own
        set (which the merge then skipped via an early return).  Runs
        before ``_apply_orange_progress`` so the repaint below reads the
        reconciled flag.  ``_mark_span_degraded`` is a no-op when the
        state is unchanged, so re-running each tick costs nothing and
        never re-flashes the HUD.
        """
        done = self._work_mgr.done_spans
        if not done:
            return
        degraded = self._work_mgr.degraded_spans
        for span_key in done:
            self._mark_span_degraded(span_key, span_key in degraded)
        self._work_mgr.degraded_spans.clear()

    def _apply_shoot_truncation_hud(self) -> None:
        """Surface a HUD warning when ``compute_shoot`` bailed out via
        the non-2-manifold fan safeguard since the previous tick.

        The truncation happens inside the JIT inner loop, which has no
        direct access to the HUD; ``compute_shoot`` increments a
        monotonic counter on the ``GeodesicMesh`` and we diff it here
        to detect new events.  Sticky for ~3 s so a single shoot during
        a fast drag still leaves a visible message.
        """
        cur = getattr(self.geo, '_shoot_truncation_count', 0)
        if cur != self._shoot_truncation_seen:
            self._shoot_truncation_seen = cur
            self._set_hud(_t("shoot_truncated"), 'red', sticky_seconds=3.0)

    def _update_orange_hud(self) -> None:
        """Translates the manager's batch counters into a HUD line."""
        done, total = self._work_mgr.progress()
        if self._work_mgr.active_spans:
            self._set_hud(_t("computing_orange", done=done, total=total), 'orange')
            self._orange_hud_active = True
        elif self._orange_hud_active:
            self._work_mgr.maybe_reset_progress()
            self._orange_hud_active = False
            self._set_hud(_t("orange_done"), 'lime')

    def _refresh_arrows_on_camera_change(self) -> bool:
        """Refreshes fixed-screen-size visuals on camera movement.

        Two things depend on camera distance / orientation:
          - **Handle arrows** (A / B cones) — fixed screen-size, so the
            world-space scale must update when the camera zooms / orbits.
          - **Curve-hover quad** — oriented per frame to face the camera
            with one axis on the curve tangent.  If the user grabs the
            view (right-drag) while hovering over a curve, the quad
            would otherwise stay frozen in the old orientation and go
            edge-on to the camera.

        Runs irrespective of whether workers produced results — the user
        may simply be orbiting after finishing edits.  Returns True when
        a render is required.
        """
        cam = self.plotter.camera.position
        if cam == self._last_cam_pos:
            return False
        self._last_cam_pos = cam
        for _, _, node in self._iter_all_nodes():
            node.refresh_arrows(self.plotter)
        # Re-orient the hover quad against the new camera position.
        # Cheap (a handful of vector ops on a 4-point buffer); skip
        # when no hover is active to avoid touching VTK state needlessly.
        hover_state = self._curve_hover_state
        if (hover_state is not None
                and self._curve_hover_circle_actor.GetVisibility()):
            self._orient_hover_marker(hover_state['pt'])
        # Re-evaluate per-label occlusion while 'n' is held: as the
        # user orbits, nodes that move behind the mesh in the new view
        # need their labels hidden, and vice-versa.  Cheap (one locator
        # query per node, same path used by hover detection).
        if self._node_labels_visible:
            self._refresh_node_label_visibility()
        return True

    def _apply_orange_progress(self) -> bool:
        """Pushes new orange points into their actors.

        Returns True if at least one polyline was updated.  Excludes
        dead spans — their worker terminated abnormally and any cached
        partial data should be discarded by ``_clear_dead_orange_spans``
        instead of being rendered.

        The polyline returned by ``_work_mgr.get_points`` is already in
        its final form — phase 1 + phase 2 of the worker produce
        cascade-faithful samples, and phase 3 (when enabled) replaces
        the inter-sample chords with exact mesh geodesics.  No
        ``subdivide_secant_chords`` is run here: that legacy post-pass
        was projecting *chord midpoints* to the surface, which moved
        the orange polyline off the cascade and was the visible cause
        of the orange-vs-didactic mismatch.  The worker now handles
        both density (cascade samples in problem regions) and surface-
        hugging (geodesic chord-bridging) directly.
        """
        dirty_orange = self._work_mgr.dirty_spans - self._work_mgr.dead_spans
        self._work_mgr.dirty_spans = set()
        rendered = False
        for span_key in dirty_orange:
            sid, i = span_key
            # Read + discard the done-flag up front (not after the guards
            # below): a done span that is out of range or has no points
            # still finished, and leaving its key in done_spans leaks it
            # across every future tick (it would re-enter
            # _apply_worker_fallbacks forever).
            is_done = span_key in self._work_mgr.done_spans
            self._work_mgr.done_spans.discard(span_key)
            if sid >= len(self.splines) or i >= self._span_count(sid):
                continue
            pts = self._work_mgr.get_points(span_key)
            if pts is None:
                continue
            self._set_geo_span(*span_key, pts, computing=not is_done)
            rendered = True
        return rendered

    def _clear_dead_orange_spans(self) -> bool:
        """Removes geometry of spans whose worker died unexpectedly."""
        dead = self._work_mgr.dead_spans
        if not dead:
            return False
        self._work_mgr.dead_spans = set()
        for span_key in dead:
            self._set_geo_span(*span_key, None)
        return True

    # --- Save / Load ---

    def _next_session_filename(self, ext: str) -> str:
        """Return an unused filename in CWD with the requested extension.

        Strategy depends on ``self._session_name``:

          * **Set** (a session JSON has been loaded): base = session
            stem.  Always probes ``<stem>_NN<ext>`` starting at
            ``NN=01`` so the originally-loaded JSON is preserved
            verbatim — saves and VTK exports never overwrite it.
            Skipping the no-suffix probe is intentional and the
            difference vs the timestamp branch.
          * **None** (fresh session): base =
            ``yyyymmdd_HHMMSS`` per the legacy strategy.  Probes the
            no-suffix filename first; only adds ``_NN`` on collision
            (matters when the user holds ``S`` and key autorepeat
            fires multiple saves within a second).

        Both branches share the suffix-search loop so behaviour is
        symmetric where it can be: caller doesn't need to know which
        branch ran.

        Parameters
        ----------
        ext
            Extension including the leading dot (``'.json'``,
            ``'.vtk'``).
        """
        if self._session_name:
            base = self._session_name
            suffix = 1
            while True:
                fname = f"{base}_{suffix:02d}{ext}"
                if not Path(fname).exists():
                    return fname
                suffix += 1
        else:
            base = datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = f"{base}{ext}"
            suffix = 1
            while Path(fname).exists():
                fname = f"{base}_{suffix:02d}{ext}"
                suffix += 1
            return fname

    def _on_save(self) -> None:
        """Saves all splines to a timestamped JSON file (atomic, UTF-8).

        Format: ``yyyymmdd_HHMMSS.json`` in the current directory.

        **v2 schema** (current).  Each node persists three 3-D points:

          - ``origin`` -- node position on the surface.
          - ``p_a``    -- handle A endpoint (or ``null`` if not yet set).
          - ``p_b``    -- handle B endpoint (or ``null`` if not yet set).

        Why these three and not the v1 ``(origin, tangent)`` layout?
        When the user drags a handle the editor calls
        ``compute_endpoint_from_origin`` (the EdgeFlipGeodesicSolver
        path), which curves to land **exactly** on the dragged
        position.  v1 stored only direction × magnitude, and on reload
        rebuilt the path with ``compute_shoot`` (a parallel-transport
        ray that goes straight in the requested direction).  On
        curved surfaces the ray cannot reproduce the solver's
        curving, so handles drifted ~0.1-0.2 units away from the user's
        choice on every save / load cycle.  Storing ``p_a`` / ``p_b``
        literally and reloading with the same solver call guarantees a
        bit-for-bit round-trip (within the solver's deterministic
        precision).

        Backward compatibility: v1 sessions (with ``tangent``) still
        load via the legacy branch in ``_apply_record_to_node``; new
        saves are always v2.

        Atomicity: the JSON is first written to ``<name>.tmp`` and then
        ``os.replace``-d into place.  A crash mid-write therefore leaves
        either the previous file untouched or no .json at all -- never a
        truncated half-written one.  Disk-full / permission errors are
        reported on the HUD instead of being silently swallowed.
        """
        data = {
            'version': 2,
            'mesh_file': self.mesh_label,
            'splines': [],
        }
        for sid, nodes in enumerate(self.splines):
            spline_data = {
                'closed': self.splines_closed[sid],
                'nodes': [],
            }
            for nid, node in enumerate(nodes):
                # ``id`` is purely cosmetic: 1-based, matches the
                # node-index labels shown under the 'n' hot-key.  It is
                # NOT consumed by ``_load_from_data`` — the loader uses
                # list position to assign indices, so the field can be
                # missing (legacy v1 / v2 sessions) or stale (user
                # edited the JSON by hand) without changing behaviour.
                # Stored as the first key per node so manually skimming
                # the file is straightforward.
                spline_data['nodes'].append({
                    'id': nid + 1,
                    'origin': node.origin.tolist(),
                    'p_a': node.p_a.tolist() if node.p_a is not None else None,
                    'p_b': node.p_b.tolist() if node.p_b is not None else None,
                })
            data['splines'].append(spline_data)

        # numpy .tolist() produces Python floats which json.dump writes
        # with full repr precision (~17 significant digits) by default.
        fname = self._next_session_filename('.json')
        try:
            self._atomic_write_json(fname, data)
        except OSError as exc:
            log.error("save failed: %s", exc)
            self._set_hud(_t("save_failed", err=str(exc)), 'red',
                          sticky_seconds=4.0)
            self.plotter.render()
            return
        n_nodes = sum(len(s) for s in self.splines)
        self._set_hud(_t("saved", n=n_nodes, fname=fname), 'gold')
        log.info("saved %d nodes across %d splines to %s",
                 n_nodes, len(self.splines), fname)
        self.plotter.render()


    @staticmethod
    def _atomic_write_json(fname: str, data: dict) -> None:
        """Writes *data* as JSON to *fname* atomically (UTF-8).

        Strategy: dump to a sibling ``*.tmp`` file, fsync, then
        ``os.replace`` it onto the target.  ``os.replace`` is atomic on
        both POSIX and Windows for files on the same volume — the user
        sees either the old file or the new one, never a partial write.

        Output uses ``_format_session_json``'s compact layout (one
        node per line, inline 3-float coordinate arrays).  Falls
        back to verbose ``json.dump(indent=2)`` if the dict shape
        is unrecognised.

        On any failure (encoder raising on an unsupported type,
        disk-full mid-flush, missing target dir on the replace step,
        ...) the partial ``*.tmp`` is removed so the user's directory
        does not accumulate orphan tmp files across repeated save
        attempts.
        """
        target = Path(fname)
        text = _format_session_json(data)
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                    'w', encoding='utf-8',
                    dir=target.parent if str(target.parent) else None,
                    prefix=target.stem + '.', suffix='.tmp',
                    delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(text)
                tmp.flush()
                try:
                    os.fsync(tmp.fileno())
                except OSError:
                    # Some filesystems (network, exotic) don't support
                    # fsync.  Replace below is still atomic at the
                    # inode level.
                    pass
            os.replace(tmp_path, fname)
            tmp_path = None
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # --- VTK export (key 'v') ---

    def _on_export_vtk(self) -> None:
        """Press 'v': export the orange (fully-geodesic) curve to a binary
        legacy ``.vtk`` file (UnstructuredGrid).

        Filename follows the same ``yyyymmdd_HHMMSS`` pattern as JSON
        save, with ``.vtk`` extension, written to the current working
        directory.  Functionally equivalent to running::

            python spline_export.py session.json --vtk \\
                --samples ${EXPORT_VTK_SAMPLES}

        with the same ``LL.vtk`` mesh — the contract is bit-for-bit
        parity with the CLI tool when both use the same sample count.

        Reuse vs recompute
        ------------------
        If ``EXPORT_VTK_SAMPLES >= GEO_SAMPLES`` AND the live orange
        cache contains polylines for every span (no orange workers
        still active), the live polylines are reused as-is — they
        already include the worker's phase-2 cascade densification
        and phase-3 chord-bridging, so the exported curve matches
        exactly what is on screen.  ``EXPORT_VTK_SAMPLES`` acts as a
        *minimum quality* threshold: any value at or above
        ``GEO_SAMPLES`` triggers the (free) reuse path; below it,
        the export falls back to a fresh ``compute_orange`` with
        only the requested sample count and no densification —
        useful for ultra-light exports where the curve only needs
        coarse landmarks.

        Per-spline semantics:
          * 0 nodes → skipped (placeholder break).
          * 1 node  → exported as a ``VTK_VERTEX`` landmark cell at the
                       node's origin (user-marked point).
          * ≥2 nodes → orange de Casteljau samples for each span,
                       written as ``VTK_LINE`` segments.

        Blocking: this runs synchronously on the main thread.  On
        large meshes the recompute can take 30-90 seconds; the HUD
        shows ``EXPORTING VTK ...`` before the work begins so the
        editor doesn't appear hung.
        """
        # spline_export imports lazily — keeps geo_splines start-up
        # path clean for users who never press 'v'.
        from spline_export import write_vtk

        fname = self._next_session_filename('.vtk')
        n_splines = len(self.splines)
        # Sticky long enough that the message survives any render
        # batching during the blocking compute.  10 minutes is a generous
        # ceiling — the next HUD update on completion supersedes it.
        self._set_hud(
            f"EXPORTING VTK ({n_splines} splines)...",
            'orange', sticky_seconds=600.0)
        self.plotter.render()  # paint the HUD before we block

        n_samples = self.scfg.EXPORT_VTK_SAMPLES
        # Reuse-from-cache is safe iff (a) the requested sample count
        # is at least the worker's grid (``GEO_SAMPLES``) — the cached
        # polylines already contain phase-1 samples + phase-2
        # densification + phase-3 chord-bridging, so any
        # ``n_samples >= GEO_SAMPLES`` is satisfied for free with
        # higher fidelity than a fresh compute would give — and
        # (b) no workers are still producing data.  Active workers may
        # have populated some spans but not others; rather than mix
        # sources we just recompute when any worker is in flight.
        can_reuse_live = (
            n_samples >= self.scfg.GEO_SAMPLES
            and not self._work_mgr.active_spans
        )

        # Disable the interactor during the blocking compute so
        # mouse / keyboard events queue up cleanly instead of being
        # processed against the about-to-stale state.  ``Disable``
        # ignores all input until ``Enable`` runs again — clearer to
        # the user than a silently-frozen UI.  Wrapped in try/finally
        # so a worker exception still re-enables interaction.
        iren = self.plotter.iren.interactor
        iren.Disable()
        try:
            spline_points_list, landmarks = self._gather_vtk_export_data(
                n_samples, can_reuse_live)
            write_vtk(fname, spline_points_list, landmarks=landmarks)
        except (OSError, RuntimeError, ValueError) as exc:
            log.error("vtk export failed: %s", exc)
            self._set_hud(
                f"VTK EXPORT FAILED: {exc}",
                'red', sticky_seconds=4.0)
            self.plotter.render()
            return
        finally:
            iren.Enable()

        n_spans = sum(len(s) for s in spline_points_list)
        n_lm = len(landmarks)
        self._set_hud(
            f"VTK EXPORTED -> {fname} ({n_spans} spans, {n_lm} landmarks)",
            'lime', sticky_seconds=4.0)
        log.info("vtk export: %d spans + %d landmarks -> %s",
                 n_spans, n_lm, fname)
        self.plotter.render()

    def _gather_vtk_export_data(self, n_samples: int, can_reuse_live: bool
                                ) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
        """Builds the (spline_points_list, landmarks) tuple that
        ``write_vtk`` consumes.

        Iterates ``self.splines`` and dispatches by node count:
          * 0 → empty list (placeholder, contributes nothing).
          * 1 → landmark; the inner spans list is empty.
          * ≥2 → orange spans, either pulled from
                 ``self._geo_span_cache`` (when ``can_reuse_live``) or
                 recomputed via ``spline_export.compute_orange``.

        ``compute_orange`` expects each node as a dict with the keys
        ``origin`` / ``face_idx`` / ``p_a`` / ``p_b`` / ``path_a`` /
        ``path_b`` — the same shape ``rebuild_mesh_and_nodes`` produces
        from a JSON.  We synthesize that view from each
        ``GeodesicSegment`` on the fly without copying the path arrays.
        """
        from spline_export import compute_orange  # local import (see _on_export_vtk)

        spline_points_list: list[list[np.ndarray]] = []
        landmarks: list[np.ndarray] = []

        for sid, nodes in enumerate(self.splines):
            n_nodes = len(nodes)
            if n_nodes == 0:
                spline_points_list.append([])
                continue
            if n_nodes == 1:
                landmarks.append(np.asarray(nodes[0].origin, dtype=float))
                spline_points_list.append([])
                continue

            closed = self.splines_closed[sid]
            spans = self._collect_orange_spans_for_export(
                sid, nodes, closed, n_samples, can_reuse_live, compute_orange)
            spline_points_list.append(spans)

        return spline_points_list, landmarks

    def _collect_orange_spans_for_export(
            self, sid: int, nodes: list[GeodesicSegment], closed: bool,
            n_samples: int, can_reuse_live: bool, compute_orange) -> list[np.ndarray]:
        """Returns the list of ``(M, 3)`` polyline arrays for spline *sid*'s
        orange spans, either reusing live cache data or recomputing.

        The reuse path requires every span of the spline to be already
        rendered (cache hit + ≥2 points).  If even one span is missing
        we fall through to the recompute branch — mixing partial live
        data (which has full phase-1+2+3 fidelity) with a fresh
        recompute (which doesn't) would write a polyline whose
        density jumps inconsistently at span boundaries.
        """
        n_spans = len(nodes) if closed else len(nodes) - 1
        if n_spans == 0:
            return []

        if can_reuse_live:
            cached_spans: list[np.ndarray] = []
            all_present = True
            for i in range(n_spans):
                entry = self._geo_span_cache.get((sid, i))
                if entry is None:
                    all_present = False
                    break
                pts = np.asarray(entry[0].points, dtype=float)
                if len(pts) < 2:
                    all_present = False
                    break
                cached_spans.append(pts)
            if all_present:
                return cached_spans
            log.debug("vtk export: live cache incomplete for spline %d, "
                      "recomputing", sid)

        # Recompute path.  Synthesize the per-node dict layout
        # ``compute_orange`` expects (matches rebuild_mesh_and_nodes).
        nodes_dict = [
            {
                'origin': n.origin,
                'face_idx': n.face_idx,
                'p_a': n.p_a, 'p_b': n.p_b,
                'path_a': n.path_a, 'path_b': n.path_b,
            }
            for n in nodes
        ]
        return compute_orange(self.geo, nodes_dict, closed, n_samples,
                              adaptive=self.scfg.ADAPTIVE_SAMPLING)

    def _on_load(self) -> None:
        """Loads splines from a JSON file, replacing all current splines.

        Opens a file dialog defaulting to the most recent ``*.json`` in
        the current directory.  Validates schema version and per-node
        shape (3-element ``origin`` and ``tangent``) before mutating any
        state — a malformed JSON cannot leave the editor in a partially
        loaded state.
        """
        import tkinter as tk
        from tkinter import filedialog

        # Find the most recent JSON for the default
        jsons = sorted(glob.glob('*.json'), reverse=True)
        initial_file = jsons[0] if jsons else ''

        root = tk.Tk()
        try:
            root.withdraw()
            fpath = filedialog.askopenfilename(
                title="Load splines",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=initial_file)
        finally:
            root.destroy()

        if not fpath:
            return

        try:
            with open(fpath, encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            log.error("invalid JSON in %s: line %d col %d: %s",
                      fpath, exc.lineno, exc.colno, exc.msg)
            hint = _json_decode_hint(exc)
            self._set_hud(
                _t("load_failed_json", line=exc.lineno, col=exc.colno,
                   msg=exc.msg, hint=hint),
                'red', sticky_seconds=5.0)
            self.plotter.render()
            return
        except OSError as exc:
            log.error("failed to read %s: %s", fpath, exc)
            self._set_hud(_t("load_failed"), 'red', sticky_seconds=4.0)
            self.plotter.render()
            return

        version = data.get('version')
        if version not in (1, 2):
            log.error("unknown JSON version: %s", version)
            self._set_hud(_t("load_failed_version"), 'red', sticky_seconds=4.0)
            self.plotter.render()
            return

        try:
            _validate_session_dict(data)
        except ValueError as exc:
            log.error("invalid session JSON %s: %s", fpath, exc)
            self._set_hud(_t("load_failed_format"), 'red', sticky_seconds=4.0)
            self.plotter.render()
            return

        self._push_undo()
        n_nodes = self._load_from_data(data)
        # Inherit the JSON's stem as the session name so subsequent
        # saves / VTK exports share the base filename.
        self._session_name = Path(fpath).stem
        self._set_hud(_t("loaded", n=n_nodes, fname=fpath), 'lime')
        log.info("loaded %d nodes across %d splines from %s",
                 n_nodes, len(self.splines), fpath)
        self.plotter.render()

    def _clear_all_curve_caches(self) -> None:
        """Cancels all background workers and removes every curve actor.

        Used by ``_load_from_data`` (full replace) and ``cleanup``
        (window close).  Leaves ``self.segments`` untouched — callers
        decide whether to clear that list as well.
        """
        self._work_mgr.cancel_all()
        for cache in (self._span_cache, self._geo_span_cache):
            for _pd, actor in cache.values():
                safe_remove_actor(self.plotter, actor)
            cache.clear()
        for _pd, actor in self._interp_cache.values():
            safe_remove_actor(self.plotter, actor)
        self._interp_cache.clear()
        self._interp_origins_buf.clear()
        self._interp_result_cache.clear()
        self._span_drag_state.clear()
        self._degraded_spans.clear()
        # The stitch cache is keyed by id(last_node); after a full replace
        # a recycled id could match a node built around a different origin,
        # so drop it here.
        self._invalidate_stitch_cache()
        # All curve actors gone — hover cache must be rebuilt next time.
        self._hover_curve_dirty = True
        self._hover_curve_items_cached = []
        # Didactic cache pins references to the active spline's last-span
        # nodes; releasing it here ensures a structural change can't
        # mistake a recycled object for a still-valid keyed entry.
        self._didactic_geo_cache = None

    def _load_from_data(self, data: dict) -> int:
        """Replaces all splines with those described in *data*.

        Clears existing state (workers, actors, caches), reconstructs
        each node from its saved fields (v2: origin + p_a + p_b handle
        endpoints; legacy v1: origin + tangent — dispatched by
        ``_node_from_record``), and recomputes all derived geometry from
        those.  Always leaves ``self.splines``
        with at least one (possibly empty) entry so downstream code can
        rely on that invariant.

        Returns the total number of nodes loaded.
        """
        # --- Clear existing splines ---
        # A load can arrive mid-drag ('l' key, or an undo/redo that
        # takes the full-rebuild path).  Nulling ``active_seg`` alone
        # left the camera locked forever: the eventual mouse release
        # hit ``_on_release``'s early return and ``_unlock_camera``
        # never ran.  Unwind the whole gesture instead.
        self._abort_active_drag()
        self._clear_all_curve_caches()
        for seg in list(self.segments):
            seg.clear_actors(self.plotter)
        self.segments.clear()
        self.state.hover_seg = None
        self.state.hover_marker = None
        self.state.pending_hover_revert_seg = None
        self.state.pending_debounces.pop('hover_revert', None)
        self._hover_dirty = True

        # --- Rebuild from JSON ---
        self.splines = []
        self.splines_closed = []

        for spline_data in data['splines']:
            nodes: list[GeodesicSegment] = []
            for nd in spline_data['nodes']:
                # ``nd`` is either a v1 ({origin, tangent}) or v2
                # ({origin, p_a, p_b}) record — _node_from_record dispatches
                # on the keys present.
                seg = self._node_from_record(nd)
                nodes.append(seg)
                self.segments.append(seg)
                seg.update_visuals(self.plotter)

            self.splines.append(nodes)
            self.splines_closed.append(bool(spline_data.get('closed', False)))

        # Invariant: there is always at least one spline (the active
        # editable target).  Callers that pass empty data still end up
        # with a usable editor.
        if not self.splines:
            self.splines.append([])
            self.splines_closed.append(False)

        self.active_spline_idx = 0
        self._prev_active_spline_idx = 0
        self._rebuild_node_index()
        self._refresh_visuals()
        # Walk EVERY spline explicitly — the sid-less defaults operate
        # on the active spline only (see ``_recompute_spans``'s
        # docstring), which left splines 1..N-1 without blue spans or
        # orange workers after a load / full undo-redo rebuild.
        for sid in range(len(self.splines)):
            self._recompute_spans(sid=sid)
            self._submit_geodesic_spans(sid=sid)
        self._update_stitch()

        return sum(len(s) for s in self.splines)

    def cleanup(self) -> None:
        """Shuts down background workers and clears all curve-layer actors.

        Also restores the global VTK polygon-offset state set in
        ``__init__`` so that other apps importing this module are not
        affected by leftover state.  Wraps actor removal in try/except
        via ``safe_remove_actor`` because the plotter may already be
        closed (window X button) when cleanup runs.

        In addition to the per-span / per-spline curve actors handled
        by ``_clear_all_curve_caches``, the editor owns several
        single-instance "auxiliary" actors created in ``__init__``:
        the curve-hover marker, the snap indicator, the stitch
        preview, the three coord-edit preview actors (input sphere,
        projected sphere, connector line) and the five didactic
        scaffold actors (four lines + collapse-point sphere).  In a
        normal session the plotter window closes and the OS reclaims
        them, but for repeated-instance flows (notebooks, tests,
        interactive exploration that creates several editors per
        process) leaving them dangling leaks references to vtkPolyData
        + vtkActor on the plotter.  We unregister them explicitly here.
        """
        self._work_mgr.shutdown()
        self._clear_all_curve_caches()

        # PyVista ``add_key_event`` callbacks bind lambdas that close
        # over ``self`` — they would keep the app alive after the
        # window closes if left attached.  ``clear_key_event_callbacks``
        # drops the whole map (no per-key tag API exists).  Same
        # neighbour-friendly trade-off as the parent's
        # ``RemoveObservers`` choice: this only fires at teardown.
        try:
            iren = self.plotter.iren
            if iren is not None:
                iren.clear_key_event_callbacks()
        except (AttributeError, RuntimeError):
            pass

        # Auxiliary single-instance actors — collected as one list so
        # the iteration is obvious and easy to extend if more are added
        # later.  ``safe_remove_actor`` already swallows the post-close
        # ValueError / AttributeError from VTK so this is safe even if
        # the plotter has already torn down.
        aux_actors = [
            self._snap_indicator_actor,
            self._stitch_actor,
            self._coord_preview_actor,
            self._coord_preview_input_actor,
            self._coord_preview_line_actor,
            self._didactic_point_actor,
        ]
        aux_actors.extend(self._didactic_actors)
        # Curve-hover marker actors live in the overlay renderer, so
        # ``safe_remove_actor`` (which targets ``self.plotter.remove_actor``)
        # would silently miss them.  Detach explicitly before the
        # overlay-renderer teardown below.
        overlay = getattr(self, '_overlay_renderer', None)
        if overlay is not None:
            for hover_actor in (self._curve_hover_circle_actor,
                                self._curve_hover_cross_actor):
                try:
                    overlay.RemoveViewProp(hover_actor)
                except (AttributeError, RuntimeError):
                    pass
        # Node-index labels share the overlay renderer's lifetime —
        # drain the pool AND detach the overlay renderer from the
        # render window so a re-instantiated app does not inherit
        # phantom billboards / a stale layer-1 renderer.  They were
        # added via ``overlay_renderer.AddViewProp`` (not
        # ``add_mesh``), so ``safe_remove_actor`` would silently skip
        # them — explicit ``RemoveViewProp`` is required.
        overlay = getattr(self, '_overlay_renderer', None)
        if overlay is not None:
            for label in self._node_labels:
                try:
                    overlay.RemoveViewProp(label)
                except (AttributeError, RuntimeError):
                    pass
            try:
                rwin = self.plotter.render_window
                if rwin is not None:
                    rwin.RemoveRenderer(overlay)
            except (AttributeError, RuntimeError):
                pass
        self._node_labels.clear()
        for actor in aux_actors:
            safe_remove_actor(self.plotter, actor)

        # Restore VTK global state to defaults — ``__init__`` flips the
        # mapper resolution to PolygonOffset to keep curves above the
        # surface.  Other applications running in the same interpreter
        # (e.g. tests, notebooks) shouldn't inherit that decision.
        try:
            vtk.vtkMapper.SetResolveCoincidentTopologyToDefault()
        except AttributeError:
            # Older VTK versions: best-effort restore.
            try:
                vtk.vtkMapper.SetResolveCoincidentTopologyToOff()
            except Exception as exc:  # noqa: BLE001
                log.debug("VTK polygon-offset restore failed: %s", exc)
        super().cleanup()

    # --- Visuals ---

    def _refresh_visuals(self) -> None:
        """Resets visual state for affected splines only.

        Tracks the previously active spline index to avoid iterating all
        nodes across all splines.  Only nodes in the old and new active
        splines are visited; span visibility is toggled only for relevant
        cache entries.  Falls back to a full sweep when the previous index
        is out of range (after spline deletion).
        """
        prev = self._prev_active_spline_idx
        curr = self.active_spline_idx
        self._prev_active_spline_idx = curr

        # Determine which spline indices need updating
        if prev == curr:
            affected = {curr}
        elif 0 <= prev < len(self.splines):
            affected = {prev, curr}
        else:
            # prev is stale (spline was deleted) — full sweep
            affected = None

        if affected is not None:
            for s_idx in affected:
                for node in self.splines[s_idx]:
                    changed = node.is_dimmed or not node.is_active
                    node.is_dimmed = False
                    node.is_active = True
                    if changed:
                        node.update_visuals(self.plotter)
            layer_vis = self._layer_visible
            for cache, layer in ((self._span_cache, 'blue'),
                                 (self._geo_span_cache, 'orange')):
                vis = layer_vis[layer]
                for (sid, _), (_, actor) in cache.items():
                    if sid in affected:
                        actor.SetVisibility(vis)
            interp_vis = layer_vis['interp']
            for sid, (_, actor) in self._interp_cache.items():
                if sid in affected:
                    actor.SetVisibility(interp_vis)
        else:
            for _, _, node in self._iter_all_nodes():
                changed = node.is_dimmed or not node.is_active
                node.is_dimmed = False
                node.is_active = True
                if changed:
                    node.update_visuals(self.plotter)
            layer_vis = self._layer_visible
            for cache, layer in ((self._span_cache, 'blue'),
                                 (self._geo_span_cache, 'orange')):
                vis = layer_vis[layer]
                for _, actor in cache.values():
                    actor.SetVisibility(vis)
            interp_vis = layer_vis['interp']
            for _, actor in self._interp_cache.values():
                actor.SetVisibility(interp_vis)
        # Per-spline actor visibility may have changed — invalidate the
        # hover cache so the next idle move rebuilds it from the new
        # visible-actor set.
        self._hover_curve_dirty = True


def _make_icosahedron(radius: float = 10.0, subdivisions: int = 2) -> pv.PolyData:
    """Creates a subdivided icosahedron with flat faces.

    Starts from a regular icosahedron (12 vertices, 20 faces) and applies
    *subdivisions* rounds of linear midpoint subdivision (each tri → 4).
    Midpoints are placed at the EDGE midpoint (linear interpolation),
    NOT re-projected to a sphere — so the surface stays polyhedral with
    flat triangular faces.  2 subdivisions give 80 faces.
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    verts *= radius
    F = np.array([
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
    ], dtype=int)
    V = verts
    for _ in range(subdivisions):
        edge_mids = {}
        new_V = list(V)
        new_F = []
        for f in F:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            mids = []
            for i, j in [(a, b), (b, c), (c, a)]:
                key = (min(i, j), max(i, j))
                if key not in edge_mids:
                    # Linear midpoint — NO sphere projection → flat faces
                    mid = (V[i] + V[j]) * 0.5
                    edge_mids[key] = len(new_V)
                    new_V.append(mid)
                mids.append(edge_mids[key])
            ab, bc, ca = mids
            new_F.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        V = np.array(new_V)
        F = np.array(new_F, dtype=int)
    n = len(F)
    pv_faces = np.column_stack([np.full(n, 3, dtype=int), F]).ravel()
    return pv.PolyData(V, faces=pv_faces).triangulate().clean()


# Sentinel mesh label kept for backwards compatibility with v1 JSON
# files saved before the prefixed form ``__builtin__:icosahedron`` was
# introduced.  Loading still accepts the legacy plain string; new saves
# use ``BUILTIN_ICOSAHEDRON``.
ICOSAHEDRON = BUILTIN_ICOSAHEDRON  # legacy alias


def _is_icosahedron_label(label: str) -> bool:
    """Returns True for any historical or current icosahedron sentinel."""
    return label in (BUILTIN_ICOSAHEDRON, _LEGACY_ICOSAHEDRON)


def _resolve_mesh(arg: str | None,
                  mesh_override: str | None = None
                  ) -> tuple[object, str, str | None]:
    """Resolves the CLI ``arg`` into ``(mesh_or_path, mesh_label, json_path)``.

    Behaviour:
      - ``None`` -> default mesh ``fandisk.obj`` if present in the
        current directory, otherwise the in-memory icosahedron.
      - ``*.json`` -> reads the session.  When *mesh_override* is given
        the session's ``mesh_file`` is replaced with it before loading;
        otherwise the session's own ``mesh_file`` is used.
      - any other path -> treated as a mesh file (PyVista handles
        ``.ply``, ``.obj``, ``.stl``, ``.vtk`` and other VTK-supported
        formats).  *mesh_override* is rejected here — you can't override
        a mesh with another mesh, only a session's mesh reference.

    *mesh_override* lets the user export / inspect a session against a
    different geometry from the one it was saved against — e.g. the
    same anatomy resampled at higher resolution, or a registered
    counterpart.  Sessions persist positions, not vertex indices, so
    the splines remap onto the alternate surface via the same
    ``project_to_surface`` path used during load.  Quality of the
    result depends on how close the two meshes are; nothing in the
    schema enforces compatibility.
    """
    if arg is None:
        if mesh_override is not None:
            log.error("mesh override given without a session JSON; "
                      "the first arg must be a *.json file when using the override")
            sys.exit(1)
        if os.path.exists(DEFAULT_MESH_FILENAME):
            log.info("default mesh: %s", DEFAULT_MESH_FILENAME)
            return DEFAULT_MESH_FILENAME, DEFAULT_MESH_FILENAME, None
        log.info("default mesh '%s' not found, falling back to icosahedron",
                 DEFAULT_MESH_FILENAME)
        return _make_icosahedron(radius=10.0), BUILTIN_ICOSAHEDRON, None

    if arg.lower().endswith('.json'):
        if not os.path.exists(arg):
            log.error("JSON file not found: %s", arg)
            sys.exit(1)
        try:
            with open(arg, encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            log.error("invalid JSON in %s: line %d col %d: %s%s",
                      arg, exc.lineno, exc.colno, exc.msg,
                      _json_decode_hint(exc))
            sys.exit(1)
        if mesh_override is not None:
            if not os.path.exists(mesh_override):
                log.error("override mesh not found: %s", mesh_override)
                sys.exit(1)
            log.info("mesh override: %s (replacing session's '%s')",
                     mesh_override, data.get('mesh_file', '<unset>'))
            label = mesh_override
        else:
            label = data.get('mesh_file', '')
        if _is_icosahedron_label(label):
            return _make_icosahedron(radius=10.0), BUILTIN_ICOSAHEDRON, arg
        if not label or not os.path.exists(label):
            log.error("mesh file referenced by session not found: %s", label)
            sys.exit(1)
        return label, label, arg

    if mesh_override is not None:
        log.error("mesh override given but first arg is not a session JSON: %s", arg)
        sys.exit(1)
    if not os.path.exists(arg):
        log.error("mesh file not found: %s", arg)
        sys.exit(1)
    return arg, arg, None


def _print_env_banner() -> None:
    """Print runtime interpreter + key dependency versions at startup.

    The editor's projection / picking / JIT behaviour can drift silently
    between Python installs on the same machine when ``numpy`` / ``vtk``
    / ``pyvista`` resolve to different majors.  Surfacing the env
    up-front catches "tested in 3.12 but launched in 3.10" mismatches
    in one glance.  Import failures are swallowed — the banner is
    diagnostic, not a hard requirement.
    """
    import importlib.metadata as _md
    print(f"python  : {sys.executable} {sys.version.split()[0]}")
    for label, modname, distname in (
        ('vtk',     'vtk',          'vtk'),
        ('pyvista', 'pyvista',      'pyvista'),
        ('numpy',   'numpy',        'numpy'),
        ('scipy',   'scipy',        'scipy'),
        ('numba',   'numba',        'numba'),
        ('pp3d',    'potpourri3d',  'potpourri3d'),
    ):
        try:
            mod = __import__(modname)
        except ImportError:
            print(f"{label:<8}: NOT INSTALLED")
            continue
        ver = getattr(mod, '__version__', None)
        if ver is None:
            try:
                ver = _md.version(distname)
            except _md.PackageNotFoundError:
                ver = '?'
        print(f"{label:<8}: {ver}")


def _cli_main() -> None:
    """Entry point for the ``geo-splines`` console script.

    Usage::

        python geo_splines.py
        python geo_splines.py <mesh.{obj,ply,stl,vtk}>
        python geo_splines.py <session.json>
        python geo_splines.py <session.json> <mesh.{obj,ply,stl,vtk}>

    The four forms cover the typical workflows:

      1. No args — opens with ``fandisk.obj`` if present, else a
         built-in subdivided icosahedron.  Useful for a quick
         "is the editor running" sanity check.
      2. One mesh arg — opens the editor on that mesh, no splines.
      3. One JSON arg — loads the session and its referenced mesh
         (``mesh_file`` field in the JSON).
      4. JSON + mesh — loads the session BUT replaces the JSON's
         ``mesh_file`` with the explicit second argument.  Use when
         you want to view / edit the same splines on a different
         geometry (registered counterpart, higher-res version, etc.).
         Splines are persisted as 3-D positions, not vertex indices,
         so they remap onto the alternate surface via the normal
         load-time projection.  Quality depends on how close the two
         meshes are.

    Help: pass ``-h`` or ``--help`` to print this usage block.
    """
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help'):
        # argparse-style help without pulling in argparse for two
        # positional args.  Same body as the docstring above so the
        # console help and ``pydoc`` stay in sync without duplication.
        print(_cli_main.__doc__.strip())  # type: ignore[union-attr]
        sys.exit(0)

    _print_env_banner()
    log.info(
        "Usage: python geo_splines.py "
        "[<mesh.{obj,ply,stl,vtk}> | <session.json> [<mesh.{obj,ply,stl,vtk}>]]"
    )
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    mesh_override = sys.argv[2] if len(sys.argv) > 2 else None
    if len(sys.argv) > 3:
        log.error("too many arguments; expected at most "
                  "<session.json> <mesh_override>")
        sys.exit(1)
    mesh_or_path, mesh_label, json_path = _resolve_mesh(arg, mesh_override)

    app: GeodesicSplineApp | None = None
    try:
        app = GeodesicSplineApp(mesh_or_path, mesh_label=mesh_label)

        if json_path is not None:
            try:
                with open(json_path, encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError as exc:
                log.error("invalid JSON in %s: line %d col %d: %s%s",
                          json_path, exc.lineno, exc.colno, exc.msg,
                          _json_decode_hint(exc))
                data = None
            if data is None:
                pass  # error already logged; skip the load step
            elif data.get('version') not in (1, 2):
                log.error("unknown JSON version: %s", data.get('version'))
            else:
                try:
                    _validate_session_dict(data)
                except ValueError as exc:
                    log.error("invalid session JSON %s: %s", json_path, exc)
                else:
                    n_nodes = app._load_from_data(data)
                    # CLI session-load: same name-inheritance as ``L``.
                    app._session_name = Path(json_path).stem
                    log.info("loaded %d nodes from %s", n_nodes, json_path)

        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure workers AND VTK resources are cleaned up even if
        # init/load was interrupted.  ``app.cleanup()`` removes the
        # interactor observers, master-clock timer, segment actors,
        # and resets the global polygon-offset state — without this,
        # programmatic reuse of the module (notebook, test harness)
        # leaks observers and pins GPU memory.
        if app is not None:
            try:
                app.cleanup()
            except Exception as exc:  # noqa: BLE001 — teardown best-effort
                log.debug("app cleanup: %s", exc)
        if app is not None and hasattr(app, '_work_mgr'):
            try:
                app._work_mgr.shutdown()
            except Exception as exc:  # noqa: BLE001 — teardown best-effort
                log.debug("worker shutdown: %s", exc)


if __name__ == "__main__":
    _cli_main()
