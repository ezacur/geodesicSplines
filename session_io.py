"""Session-format helpers shared by the editor and the CLI exporter.

Pure, dependency-light validation / diagnostics for the spline session
JSON schema.  This module deliberately imports **only** the standard
library (``json``) — no numpy, pyvista, vtk, or potpourri3d — so both
``geo_splines`` (the GUI editor) and ``spline_export`` (the headless
CLI) can validate a session without dragging in the heavy geometry /
rendering stack.  Splitting these out of ``geo_splines`` also removed
the import inversion where the CLI imported the 7k-line GUI module just
to reach the validator.
"""
from __future__ import annotations

import json


def _json_decode_hint(exc: json.JSONDecodeError) -> str:
    """Best-effort hint for the most common hand-edit JSON mistakes.

    Returns either an empty string or a leading-space-prefixed hint
    like ``" (trailing comma?)"`` so the caller can splat it into a
    HUD / log line without conditional branching.

    Detection is purely on ``exc.msg``: the standard library's parser
    reports ``Expecting value`` when a comma is followed immediately
    by a closing bracket, and ``Expecting property name enclosed in
    double quotes`` for the equivalent inside an object.  Both
    patterns are the canonical trailing-comma signature.
    """
    m = exc.msg
    if 'Expecting value' in m or 'Expecting property name' in m:
        return ' (likely a trailing comma — JSON does not allow them)'
    return ''


def _validate_session_dict(data: dict) -> None:
    """Schema check for a deserialized spline session.

    Raises ``ValueError`` with a precise location when the structure or
    a node's ``origin`` / ``tangent`` does not match the expected shape.
    Done before any state mutation so a malformed file never leaves the
    editor in a half-loaded state.

    Degenerate-spline rules
    -----------------------
    The interactive editor enforces these invariants implicitly.  When
    they are violated by a loaded session (manually edited JSON, or a
    bug in a future writer) the downstream renderer reaches code paths
    that assume them and crashes obscurely.  Catching them here turns a
    runtime crash into a clean rejection at load time:

      - **Open spline**: any node count is allowed, including 0 (a
        "break" placeholder created by ``Dbl-click R``) and 1 (a
        single-point spline mid-construction).  No span constraint.
      - **Closed spline**: requires at least 3 nodes.  A 2-node closed
        loop has both spans coincident on the same chord and renders as
        zero curvature; the wrap-around bezier is degenerate.  A 1- or
        0-node closed loop has no spans at all.  None of these can be
        produced by ``_on_close_spline`` (which itself enforces ≥ 3),
        so a closed flag with < 3 nodes can only come from a
        hand-edited or corrupted save.
    """
    if not isinstance(data, dict):
        raise ValueError("top-level value is not an object")
    splines = data.get('splines')
    if not isinstance(splines, list):
        raise ValueError("'splines' missing or not a list")
    # Schema dispatch is per-node, not per-file: a session is allowed
    # to mix v1 and v2 records (handy when manually concatenating
    # sessions or migrating piecemeal).  Each record is valid if it
    # has either ``tangent`` (v1) or both ``p_a`` and ``p_b`` (v2).
    # ``x != x`` catches NaN; the inf comparisons reject ±Infinity
    # (which Python's ``json.load`` otherwise parses silently because
    # the JSON spec extension ``Infinity`` is enabled by default).
    _POS_INF = float('inf')
    _NEG_INF = float('-inf')

    def _validate_3vec_or_none(label, v, allow_none):
        if v is None and allow_none:
            return
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            raise ValueError(f"{label} must be a 3-element list")
        for j, x in enumerate(v):
            # ``bool`` is a subclass of ``int``; rejecting it explicitly
            # prevents corrupt JSON like ``"origin": [true, 0, 0]`` from
            # silently passing validation and crashing in ``find_face``.
            if (isinstance(x, bool)
                    or not isinstance(x, (int, float))
                    or x != x  # NaN
                    or x == _POS_INF or x == _NEG_INF):
                raise ValueError(f"{label}[{j}] must be a finite number")

    for si, sd in enumerate(splines):
        if not isinstance(sd, dict):
            raise ValueError(f"splines[{si}] is not an object")
        if 'closed' in sd and not isinstance(sd['closed'], (bool, int)):
            raise ValueError(f"splines[{si}].closed must be bool")
        nodes = sd.get('nodes')
        if not isinstance(nodes, list):
            raise ValueError(f"splines[{si}].nodes missing or not a list")
        for ni, nd in enumerate(nodes):
            if not isinstance(nd, dict):
                raise ValueError(f"splines[{si}].nodes[{ni}] is not an object")
            base = f"splines[{si}].nodes[{ni}]"
            _validate_3vec_or_none(f"{base}.origin", nd.get('origin'),
                                   allow_none=False)
            has_v2 = 'p_a' in nd and 'p_b' in nd
            has_v1 = 'tangent' in nd
            if not (has_v1 or has_v2):
                raise ValueError(
                    f"{base} must have either 'tangent' (v1) "
                    f"or both 'p_a' and 'p_b' (v2)")
            if has_v2:
                # ``p_a`` / ``p_b`` may be null for placeholder nodes
                # (e.g. a freshly added single node before the second
                # node sets up symmetric tangents).
                _validate_3vec_or_none(f"{base}.p_a", nd['p_a'],
                                       allow_none=True)
                _validate_3vec_or_none(f"{base}.p_b", nd['p_b'],
                                       allow_none=True)
            else:
                _validate_3vec_or_none(f"{base}.tangent", nd['tangent'],
                                       allow_none=False)
        # Closed loops require >= 3 nodes (interactive editor enforces
        # this in _on_close_spline; loaded sessions might violate it).
        if bool(sd.get('closed', False)) and len(nodes) < 3:
            raise ValueError(
                f"splines[{si}].closed=true requires at least 3 nodes "
                f"(got {len(nodes)})")
