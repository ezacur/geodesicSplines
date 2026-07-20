"""Regression tests for the CLI exporter's input guards and file formats.

- **version gate** (mirrors the editor): sessions with a missing or
  unknown ``version`` exit with code 2 instead of silently exporting.
- **mesh_file guard**: a session without ``mesh_file`` exits with the
  documented code-2 diagnostic instead of a raw ``KeyError``.
- **OBJ format**: polylines are written as ``l`` records (the OBJ
  polyline element) — 2-vertex ``f`` faces are invalid OBJ and are
  rejected outright by vtkOBJReader/ParaView.  One ``l`` chain per
  span, so skipped spans leave a gap instead of a fabricated bridge.
- **CSV gaps**: a NaN row is emitted where consecutive spans do not
  share an endpoint (a span was skipped), and between splines.
"""
import io
import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from spline_export import (  # noqa: E402
    load_json,
    rebuild_mesh_and_nodes,
    write_csv,
    write_obj,
)


def _valid_session(**overrides):
    data = {
        'version': 2,
        'mesh_file': 'mesh.obj',
        'splines': [{
            'closed': False,
            'nodes': [
                {'origin': [0.0, 0.0, 0.0], 'p_a': None, 'p_b': None},
                {'origin': [1.0, 0.0, 0.0], 'p_a': None, 'p_b': None},
            ],
        }],
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Input guards
# ---------------------------------------------------------------------------

def _write_session(tmp_path, data):
    path = tmp_path / "session.json"
    path.write_text(json.dumps(data), encoding='utf-8')
    return str(path)


def test_load_json_accepts_known_versions(tmp_path):
    path = _write_session(tmp_path, _valid_session())
    assert load_json(path)['version'] == 2


@pytest.mark.parametrize("version", [None, 3, "2"])
def test_load_json_rejects_unknown_version(tmp_path, version):
    data = _valid_session()
    if version is None:
        del data['version']
    else:
        data['version'] = version
    path = _write_session(tmp_path, data)

    with pytest.raises(SystemExit) as exc:
        load_json(path)
    assert exc.value.code == 2


def test_rebuild_without_mesh_file_exits_cleanly():
    data = _valid_session()
    del data['mesh_file']

    with pytest.raises(SystemExit) as exc:   # was: raw KeyError traceback
        rebuild_mesh_and_nodes(data)
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# OBJ format
# ---------------------------------------------------------------------------

def _spans(*chains):
    return [np.asarray(c, dtype=float) for c in chains]


def test_write_obj_emits_polyline_records(tmp_path):
    span1 = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    span2 = [[2, 0, 0], [3, 0, 0]]           # shares span1's endpoint
    path = tmp_path / "out.obj"

    write_obj(str(path), [_spans(span1, span2)])

    lines = path.read_text(encoding='utf-8').splitlines()
    assert not any(ln.startswith('f ') for ln in lines), \
        "2-vertex 'f' records are invalid OBJ"
    l_records = [ln for ln in lines if ln.startswith('l ')]
    assert l_records == ['l 1 2 3', 'l 4 5']  # one chain per span
    assert sum(ln.startswith('v ') for ln in lines) == 5
    assert 'g spline_0' in lines


def test_write_obj_output_readable_by_vtk(tmp_path):
    vtk = pytest.importorskip("vtk")
    path = tmp_path / "out.obj"
    write_obj(str(path), [_spans([[0, 0, 0], [1, 0, 0], [1, 1, 0]])])

    reader = vtk.vtkOBJReader()
    reader.SetFileName(str(path))
    reader.Update()
    out = reader.GetOutput()
    # The old 'f'-record output made vtkOBJReader error out and return
    # an EMPTY dataset (0 points).
    assert out.GetNumberOfPoints() == 3
    assert out.GetNumberOfCells() >= 1


# ---------------------------------------------------------------------------
# CSV gaps
# ---------------------------------------------------------------------------

def _csv_rows(spline_points_list):
    stream = io.StringIO()
    write_csv(spline_points_list, stream)
    return stream.getvalue().splitlines()


def _nan_count(rows):
    return sum('NaN' in r or 'nan' in r for r in rows)


def test_csv_contiguous_spans_have_no_break():
    span1 = [[0, 0, 0], [1, 0, 0]]
    span2 = [[1, 0, 0], [2, 0, 0]]           # shared endpoint
    rows = _csv_rows([_spans(span1, span2)])
    assert _nan_count(rows) == 0
    assert len(rows) == 4


def test_csv_skipped_span_gap_emits_break():
    span1 = [[0, 0, 0], [1, 0, 0]]
    span3 = [[5, 0, 0], [6, 0, 0]]           # gap: span2 was skipped
    rows = _csv_rows([_spans(span1, span3)])
    assert _nan_count(rows) == 1
    # The break sits between the two chains.
    assert 'NaN' in rows[2] or 'nan' in rows[2]


def test_csv_break_between_splines():
    spline_a = _spans([[0, 0, 0], [1, 0, 0]])
    spline_b = _spans([[9, 0, 0], [10, 0, 0]])
    rows = _csv_rows([spline_a, spline_b])
    assert _nan_count(rows) == 1
