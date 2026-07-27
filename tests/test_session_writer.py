"""Round-trip tests for the session *writer*, ``_format_session_json``.

``test_session_format.py`` covers the reader (``_validate_session_dict``)
thoroughly; the writer had no tests at all — 169 lines of hand-rolled
JSON layout with a silent ``json.dumps`` fallback, on the path that
persists the user's work.  A writer bug is the one failure mode in this
codebase that destroys data the user cannot recover.

The contract, per its own docstring:
  * output is valid JSON and ``json.loads`` reproduces the input dict
    *exactly* (including full float precision);
  * the reader must accept whatever the writer emits;
  * NaN / ±Inf raise ``ValueError`` rather than emitting the
    non-RFC-8259 ``NaN`` / ``Infinity`` literals;
  * an off-schema dict falls back to ``json.dumps(indent=2)`` rather
    than emitting something malformed;
  * the compact layout is ~4 lines per node, not 12.
"""
import json
import math

import pytest

pytest.importorskip("vtk")
pytest.importorskip("pyvista")

from geo_splines import GeodesicSplineApp  # noqa: E402
from session_io import _validate_session_dict  # noqa: E402

fmt = GeodesicSplineApp._format_session_json


def _v2_session():
    return {
        'version': 2,
        'mesh_file': 'fandisk.obj',
        'splines': [
            {'closed': False,
             'nodes': [
                 {'id': 1,
                  'origin': [0.1, -2.5, 3.25],
                  'p_a': [1.0, 0.0, 0.0],
                  'p_b': [-1.0, 0.0, 0.0]},
                 {'id': 2,
                  'origin': [4.0, 5.0, 6.0],
                  'p_a': None,
                  'p_b': None},
             ]},
            {'closed': True,
             'nodes': [
                 {'origin': [0.0, 0.0, 0.0], 'tangent': [1.0, 0.0, 0.0]},
                 {'origin': [1.0, 0.0, 0.0], 'tangent': [0.0, 1.0, 0.0]},
                 {'origin': [0.0, 1.0, 0.0], 'tangent': [0.0, 0.0, 1.0]},
             ]},
        ],
    }


# --------------------------------------------------------- round-tripping

def test_output_is_valid_json():
    json.loads(fmt(_v2_session()))


def test_round_trip_reproduces_the_input_exactly():
    data = _v2_session()
    assert json.loads(fmt(data)) == data


def test_reader_accepts_what_the_writer_emits():
    """The two halves of persistence must agree — this is the check the
    suite was missing entirely."""
    _validate_session_dict(json.loads(fmt(_v2_session())))


@pytest.mark.parametrize("value", [
    0.1,                     # not exactly representable
    1.0 / 3.0,
    1e-17,
    -2.2250738585072014e-308,   # smallest normal double
    1.7976931348623157e308,     # max double
    123456789.123456789,
])
def test_full_float_precision_survives(value):
    """Node coordinates are surface positions; a truncated repr moves
    the spline.  The writer promises ~17-digit ``repr(float(x))``."""
    data = {'version': 2, 'mesh_file': 'm.obj', 'splines': [
        {'closed': False, 'nodes': [
            {'origin': [value, 0.0, 0.0], 'tangent': [1.0, 0.0, 0.0]}]}]}
    got = json.loads(fmt(data))['splines'][0]['nodes'][0]['origin'][0]
    assert got == value


def test_integer_coordinates_become_floats_and_still_round_trip():
    """``_arr`` coerces via ``float(x)``, so an int coordinate is
    emitted as a float — the value must still compare equal."""
    data = {'version': 2, 'splines': [
        {'closed': False, 'nodes': [
            {'origin': [1, 2, 3], 'tangent': [0, 0, 1]}]}]}
    node = json.loads(fmt(data))['splines'][0]['nodes'][0]
    assert node['origin'] == [1.0, 2.0, 3.0]
    _validate_session_dict(json.loads(fmt(data)))


# ------------------------------------------------------------ edge shapes

def test_empty_session():
    data = {'version': 2, 'mesh_file': 'm.obj', 'splines': []}
    assert json.loads(fmt(data)) == data
    _validate_session_dict(json.loads(fmt(data)))


def test_spline_with_no_nodes():
    data = {'version': 2, 'splines': [{'closed': False, 'nodes': []}]}
    assert json.loads(fmt(data)) == data


def test_null_handles_preserved():
    data = {'version': 2, 'splines': [{'closed': False, 'nodes': [
        {'origin': [0.0, 0.0, 0.0], 'p_a': None, 'p_b': None}]}]}
    node = json.loads(fmt(data))['splines'][0]['nodes'][0]
    assert node['p_a'] is None and node['p_b'] is None


def test_forward_compat_keys_are_not_dropped():
    """Unknown keys at every level must survive — the layout appends
    them rather than discarding them."""
    data = {
        'version': 2,
        'mesh_file': 'm.obj',
        'future_top': {'a': 1},
        'splines': [{
            'closed': True,
            'future_spline': 'xyz',
            'nodes': [{'origin': [0.0, 0.0, 0.0],
                       'tangent': [1.0, 0.0, 0.0],
                       'future_node': 7}] * 3,
        }],
    }
    assert json.loads(fmt(data)) == data


def test_node_with_no_keys_at_all():
    data = {'version': 2, 'splines': [{'closed': False, 'nodes': [{}]}]}
    assert json.loads(fmt(data)) == data


# ------------------------------------------------------- non-finite guard

@pytest.mark.parametrize("bad", [float('nan'), float('inf'), float('-inf')])
def test_non_finite_coordinate_raises_instead_of_emitting_bad_json(bad):
    """``NaN`` / ``Infinity`` are not RFC 8259.  Emitting them would
    produce a file Python re-reads happily but every other JSON parser
    rejects — a session that silently stops being portable."""
    data = {'version': 2, 'splines': [{'closed': False, 'nodes': [
        {'origin': [bad, 0.0, 0.0], 'tangent': [1.0, 0.0, 0.0]}]}]}
    with pytest.raises(ValueError):
        fmt(data)


@pytest.mark.parametrize("data", [
    # top level
    {'version': 2, 'bogus': math.inf, 'splines': []},
    # spline level
    {'version': 2, 'splines': [{'closed': False, 'bogus': math.nan,
                                'nodes': []}]},
    # node scalar (not a coordinate triplet, so it misses ``_arr``)
    {'version': 2, 'splines': [{'closed': False, 'nodes': [
        {'origin': [0.0, 0.0, 0.0], 'tangent': [1.0, 0.0, 0.0],
         'bogus': math.inf}]}]},
], ids=['top', 'spline', 'node-scalar'])
def test_non_finite_anywhere_raises_not_just_in_coordinates(data):
    """Only ``_arr`` (the coordinate triplets) used to pass
    ``allow_nan=False``.  Every other value went through a bare
    ``json.dumps`` and would emit ``Infinity`` / ``NaN`` — leaving the
    compact path laxer than its own off-schema fallback, which was
    already strict."""
    with pytest.raises(ValueError):
        fmt(data)


# ---------------------------------------------------------------- fallback

@pytest.mark.parametrize("data", [
    {'no_splines_key': True},
    {'splines': 'not a list'},
    {'splines': ['not a dict']},
    {'splines': [{'nodes': 'not a list'}]},
])
def test_off_schema_input_falls_back_to_plain_json(data):
    """The fallback exists so a shape drift never emits malformed
    output.  It must still be valid, round-tripping JSON."""
    text = fmt(data)
    assert json.loads(text) == data


# ------------------------------------------------------------ the layout

def test_layout_is_four_lines_per_node():
    """The whole point of the hand-rolled writer: ~4 lines per node
    instead of ``json.dump(indent=2)``'s 12.  If this regresses, the
    formatter has silently fallen back."""
    data = {'version': 2, 'mesh_file': 'm.obj', 'splines': [
        {'closed': False, 'nodes': [
            {'id': i + 1,
             'origin': [0.0, 0.0, 0.0],
             'p_a': [1.0, 0.0, 0.0],
             'p_b': [-1.0, 0.0, 0.0]} for i in range(10)]}]}

    text = fmt(data)
    verbose = json.dumps(data, indent=2)
    assert len(text.splitlines()) < len(verbose.splitlines()) / 2
    # One line per rendered key, four keys per node.
    assert text.count('"origin":') == 10
    assert text.count('"id":') == 10


def test_colons_are_aligned_within_a_node():
    data = {'version': 2, 'splines': [{'closed': False, 'nodes': [
        {'id': 1, 'origin': [0.0, 0.0, 0.0],
         'p_a': [1.0, 0.0, 0.0], 'p_b': [-1.0, 0.0, 0.0]}]}]}
    lines = [ln for ln in fmt(data).splitlines()
             if any(k in ln for k in ('"id":', '"origin":', '"p_a":', '"p_b":'))]
    assert len(lines) == 4
    # Values start at the same column on every line of the block.
    starts = {len(ln) - len(ln.lstrip(' ')) + ln.strip().index(':') for ln in lines}
    value_cols = set()
    for ln in lines:
        after = ln.index(':') + 1
        value_cols.add(after + (len(ln[after:]) - len(ln[after:].lstrip(' '))))
    assert len(value_cols) == 1, f"values not column-aligned: {value_cols}"
    assert len(starts) >= 1


def test_output_ends_with_a_newline():
    assert fmt(_v2_session()).endswith('\n')
