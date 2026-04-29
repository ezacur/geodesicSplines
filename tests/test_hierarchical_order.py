"""Tests for ``_hierarchical_inner_order`` — pure function, no deps.

The function returns the inner indices (excluding the two endpoints)
in progressive-refinement order: midpoint first, then quarters, then
eighths, etc.  Verified for the canonical 33 = 2^5 + 1 case (the
worker's default ``GEO_SAMPLES``) and a few off-power sizes.
"""
from __future__ import annotations

import ast
from pathlib import Path


def _load_fn():
    src = (Path(__file__).resolve().parent.parent / "geo_splines.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_hierarchical_inner_order"
        ):
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            ns: dict = {}
            exec(compile(module, "geo_splines.py", "exec"), ns)
            return ns["_hierarchical_inner_order"]
    raise RuntimeError("_hierarchical_inner_order not found")


order = _load_fn()


def test_endpoints_excluded():
    """Endpoints (0 and total-1) are pre-seeded by the main thread and
    must never appear in the worker's plan."""
    for total in (3, 5, 9, 17, 33, 65, 7, 13):
        seq = order(total)
        assert 0 not in seq
        assert total - 1 not in seq


def test_covers_all_inner_indices():
    """Every interior index must be visited exactly once."""
    for total in (3, 5, 9, 17, 33, 65, 7, 13, 50):
        seq = order(total)
        assert sorted(seq) == list(range(1, total - 1))
        assert len(seq) == len(set(seq))  # no duplicates


def test_midpoint_first_for_power_of_two_plus_one():
    """For total = 2^k + 1 the first emitted index is the midpoint."""
    for k in range(2, 7):  # 5, 9, 17, 33, 65
        total = (1 << k) + 1
        seq = order(total)
        assert seq[0] == total // 2


def test_canonical_33_count_matches_geo_samples():
    """``GEO_SAMPLES = 33`` means 31 inner points (= 33 - 2 endpoints)."""
    assert len(order(33)) == 31
