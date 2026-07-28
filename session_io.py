"""Session-format helpers shared by the editor and the CLI exporter.

Pure, dependency-light handling of the spline session JSON schema —
**both** directions:

  - ``_validate_session_dict`` / ``_json_decode_hint`` — the read side.
  - ``_format_session_json`` — the write side (the compact aligned
    layout the editor saves).

This module deliberately imports **only** the standard library
(``json``) — no numpy, pyvista, vtk, or potpourri3d — so both
``geo_splines`` (the GUI editor) and ``spline_export`` (the headless
CLI) can round-trip a session without dragging in the heavy geometry /
rendering stack.  Splitting these out of ``geo_splines`` also removed
the import inversion where the CLI imported the 7k-line GUI module just
to reach the validator.

Keeping the writer here rather than on ``GeodesicSplineApp`` matters for
more than symmetry: it is what lets the writer's tests run in CI's slim
matrix across every supported Python.  While it lived in
``geo_splines`` its tests needed ``importorskip("vtk")`` and therefore
only ever executed on the single full-dependency job.
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
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise ValueError(f"{label}[{j}] must be a finite number")
            # Python ints are arbitrary-precision, so a hand-edited
            # 400-digit coordinate is a genuine ``int`` that is neither
            # NaN nor ±inf and sails past the checks below.  It then
            # raises ``OverflowError`` much later, inside
            # ``np.asarray(record['origin'], dtype=float)`` — by which
            # point ``_load_from_data`` has already aborted the drag,
            # cleared every curve cache and emptied ``self.splines``,
            # i.e. exactly the half-loaded state this function exists
            # to prevent.  Do the float64 conversion the loader will do
            # here, where rejecting is still free.
            try:
                xf = float(x)
            except (OverflowError, ValueError):
                raise ValueError(
                    f"{label}[{j}] is out of range for a 64-bit float"
                ) from None
            if xf != xf or xf == _POS_INF or xf == _NEG_INF:  # NaN / ±inf
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


def _format_session_json(data: dict) -> str:
    """Render the session dict with aligned per-node blocks::

        {
          "version": 2,
          "mesh_file": "LL.vtk",
          "splines": [
            {
              "closed": false,
              "nodes": [
                {"id":     1,
                 "origin": [x, y, z],
                 "p_a":    [x, y, z],
                 "p_b":    [x, y, z]},
                {"id":     2,
                 "origin": [x, y, z],
                 "p_a":    [x, y, z],
                 "p_b":    [x, y, z]}
              ]
            }
          ]
        }

    Each node spans one line per field with the colons aligned
    (``"origin":`` is the longest known key; shorter keys like
    ``"id"`` / ``"p_a"`` / ``"p_b"`` get extra spaces after the
    colon to match).  Continuation lines align under the opening
    ``{`` of the first line so the values form a clean visual
    column.  ``id`` (when present) is rendered first as an inline
    scalar; it is purely cosmetic — see ``geo_splines._on_save`` for the
    rationale.

    Compared to ``json.dump(indent=2)``'s 12 lines per node, this
    emits 4 lines per node and keeps the coordinate triplets
    inline — typical sessions shrink ~3×.

    The output is still valid JSON: every value goes through ``_j``
    (``json.dumps`` with ``allow_nan=False``) so quoting / escaping
    / float repr (full-precision ~17-digit ``repr(float(x))``) is
    unchanged, and a non-finite value anywhere raises ``ValueError``
    instead of emitting the non-RFC-8259 ``NaN`` / ``Infinity``
    literals.  Round-trip via ``json.loads`` reproduces the original
    ``data`` exactly — locked by ``tests/test_session_writer.py``.

    Defensive fallback: if the dict shape diverges from the
    v1/v2 session schema, returns ``json.dumps(data, indent=2)``
    unchanged so we never emit malformed output.
    """
    # Validate the structure we know how to compact-format; on any
    # surprise, fall back to the verbose default.  Using EAFP
    # rather than full schema validation: ``_validate_session_dict``
    # is the canonical schema check; here we just need the shape
    # to traverse safely.
    try:
        splines = data['splines']
        if not isinstance(splines, list):
            raise TypeError
        for s in splines:
            if not isinstance(s, dict):
                raise TypeError
            if not isinstance(s.get('nodes', []), list):
                raise TypeError
    except (KeyError, TypeError):
        return json.dumps(data, indent=2, allow_nan=False)

    def _j(value) -> str:
        """``json.dumps`` with the non-finite escape hatch closed.

        Python's default ``allow_nan=True`` emits the bare literals
        ``NaN`` / ``Infinity`` / ``-Infinity``, which are **not**
        RFC 8259 — Python reads them back happily, every other JSON
        parser rejects the file.  A session that silently stops
        being portable is worse than one that fails to save, so
        every value in this writer goes through here.  (The
        off-schema fallback above already passes
        ``allow_nan=False``; this keeps the compact path equally
        strict instead of laxer than its own fallback.)
        """
        return json.dumps(value, allow_nan=False)

    def _arr(vec) -> str:
        """Inline JSON array of floats — single line, comma-space."""
        return '[' + ', '.join(_j(float(x)) for x in vec) + ']'

    # Canonical key order inside a node.  ``id`` (when present) is
    # rendered first as a single inline key so the human eye lands
    # on the node identifier before the coordinate triplets.
    # Anything not in this tuple is appended at the end (forward-
    # compat for future schema extensions).
    NODE_CANON = ('id', 'origin', 'tangent', 'p_a', 'p_b')

    def _node_lines(node: dict, indent: str, is_last: bool) -> list[str]:
        """Render *node* as N lines (one per key) with colons
        aligned.  The first line opens with ``{`` immediately
        after *indent*; subsequent lines continue at *indent + 1*
        so all keys form a vertical column under the first key.
        The closing brace + optional trailing comma is appended
        to the last line.
        """
        keys: list[str] = [k for k in NODE_CANON if k in node]
        for k in node:
            if k not in NODE_CANON:
                keys.append(k)
        tail = '' if is_last else ','
        if not keys:
            return [f'{indent}{{}}{tail}']

        key_reprs = [json.dumps(k) for k in keys]
        # ljust width = longest "key": + 1 space → values align.
        pad_to = max(len(kr) for kr in key_reprs) + 2  # +1 colon, +1 space
        cont = indent + ' '   # +1 to align under '{' contents

        lines: list[str] = []
        n = len(keys)
        for i, (k, kr) in enumerate(zip(keys, key_reprs, strict=False)):
            val = node[k]
            if val is None:
                rendered = 'null'
            elif k in NODE_CANON and isinstance(val, (list, tuple)):
                # Coordinate triplet (origin / tangent / p_a / p_b).
                rendered = _arr(val)
            else:
                # Scalar canonical keys (``id``) and forward-compat
                # extras both fall through to default JSON encoding
                # — single-token output keeps the per-node block
                # within its aligned column.
                rendered = _j(val)
            prefix = (kr + ':').ljust(pad_to)
            if i == 0:
                line = f'{indent}{{{prefix}{rendered}'
            else:
                line = f'{cont}{prefix}{rendered}'
            if i < n - 1:
                line += ','
            else:
                line += '}' + tail
            lines.append(line)
        return lines

    out: list[str] = ['{']
    # Top-level: version + mesh_file first (canonical), then any
    # other forward-compat keys, then splines last.
    TOP_HEAD = ('version', 'mesh_file')
    for key in TOP_HEAD:
        if key in data:
            out.append(f'  {json.dumps(key)}: {_j(data[key])},')
    for key in data:
        if key in TOP_HEAD or key == 'splines':
            continue
        out.append(f'  {json.dumps(key)}: {_j(data[key])},')
    out.append('  "splines": [')

    for si, spline in enumerate(splines):
        out.append('    {')
        if 'closed' in spline:
            out.append(f'      "closed": {_j(spline["closed"])},')
        # Forward-compat: emit any other keys before nodes
        # (matters because nodes is the open-ended block).
        for key in spline:
            if key in ('closed', 'nodes'):
                continue
            out.append(f'      {json.dumps(key)}: {_j(spline[key])},')
        nodes = spline.get('nodes', [])
        if not nodes:
            out.append('      "nodes": []')
        else:
            out.append('      "nodes": [')
            last_n = len(nodes) - 1
            for ni, node in enumerate(nodes):
                out.extend(_node_lines(
                    node, '        ', is_last=(ni == last_n)))
            out.append('      ]')
        spline_close = '    }' + (',' if si < len(splines) - 1 else '')
        out.append(spline_close)

    out.append('  ]')
    out.append('}')
    return '\n'.join(out) + '\n'
