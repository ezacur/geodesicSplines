"""Tests for ``GeodesicSplineApp._parse_coordinates``.

The parser is the input layer for the right-double-click coordinate
dialog.  It must accept three flexible textual formats and reject
anything that would crash ``find_face`` downstream (NaN, inf, wrong
arity).  Loaded via AST extraction so the test suite stays pure-Python
(no PyVista / VTK import).
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest


def _load_parser():
    """Extract the static method body without importing the heavy module."""
    src = (Path(__file__).resolve().parent.parent / "geo_splines.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name != "GeodesicSplineApp":
            continue
        for node in cls.body:
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "_parse_coordinates"
            ):
                # Strip the @staticmethod decorator so we can exec it directly.
                node.decorator_list = []
                module = ast.Module(body=[node], type_ignores=[])
                ast.fix_missing_locations(module)
                ns: dict = {}
                exec(compile(module, "geo_splines.py", "exec"), ns)
                return ns["_parse_coordinates"]
    raise RuntimeError("_parse_coordinates not found")


parse = _load_parser()


# --- Accepted formats -----------------------------------------------------


def test_bracketed_with_commas():
    assert parse("[1.23, 4.56, 7.89]") == (1.23, 4.56, 7.89)


def test_bracketed_with_spaces_inside():
    assert parse("[ 1.23 , 4.56 , 7.89 ]") == (1.23, 4.56, 7.89)


def test_comma_separated_no_brackets():
    assert parse("1.23, 4.56, 7.89") == (1.23, 4.56, 7.89)


def test_whitespace_separated_no_brackets():
    assert parse("1.23 4.56 7.89") == (1.23, 4.56, 7.89)


def test_mixed_separators():
    """Commas and spaces can be mixed freely."""
    assert parse("1.23,  4.56\t7.89") == (1.23, 4.56, 7.89)


def test_negative_and_signed_floats():
    assert parse("-1.0, +2.5, -3.14e2") == (-1.0, 2.5, -314.0)


def test_integer_tokens_accepted_as_floats():
    assert parse("1 2 3") == (1.0, 2.0, 3.0)


def test_leading_trailing_whitespace_stripped():
    assert parse("    [1, 2, 3]    ") == (1.0, 2.0, 3.0)


def test_user_provided_examples():
    """The exact strings the user listed in the spec."""
    assert parse("[1.23241,2.241251,5.12412131]") == (1.23241, 2.241251, 5.12412131)
    assert parse(" 1.234124, 1.12351,  6.213414  ") == (1.234124, 1.12351, 6.213414)
    assert parse("1.1231214  5.231231       8.123125  ") == (
        1.1231214,
        5.231231,
        8.123125,
    )


# --- Rejected inputs ------------------------------------------------------


def test_empty_string_rejected():
    assert parse("") is None


def test_whitespace_only_rejected():
    assert parse("   \t  ") is None


def test_two_tokens_rejected():
    assert parse("1, 2") is None


def test_four_tokens_rejected():
    assert parse("1, 2, 3, 4") is None


def test_non_numeric_token_rejected():
    assert parse("1, 2, hello") is None


def test_letters_inside_number_rejected():
    assert parse("1.0, 2.0, 3a") is None


def test_nan_rejected():
    assert parse(f"1, 2, {math.nan}") is None
    assert parse("nan nan nan") is None


def test_inf_rejected():
    assert parse("inf 0 0") is None
    assert parse("-inf, 1, 2") is None


def test_unbalanced_brackets_rejected():
    """Only matching '[...]' is stripped; lone bracket leaves a junk token."""
    assert parse("[1, 2, 3") is None
    assert parse("1, 2, 3]") is None
