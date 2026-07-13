"""Tests for the JSON session schema validator.

These tests exercise ``session_io._validate_session_dict`` directly —
no PyVista, no VTK, no mesh.  They guard the invariants that the
interactive editor relies on at load time:

  - Top-level shape is an object with a ``splines`` list.
  - Each node has a 3-element ``origin`` and ``tangent``, both finite.
  - Closed splines have at least 3 nodes.

The validator now lives in the stdlib-only ``session_io`` module (moved
out of ``geo_splines`` to break the CLI's pyvista dependency), so it
imports directly — no more AST-extraction shim to dodge the heavy deps.
"""
from __future__ import annotations

import math

import pytest

from session_io import _validate_session_dict as validate

# --- Happy path ----------------------------------------------------------


def test_empty_session_accepted():
    validate({"splines": []})


def test_single_open_spline_with_one_node_accepted():
    validate(
        {
            "splines": [
                {
                    "closed": False,
                    "nodes": [
                        {"origin": [0.0, 0.0, 0.0], "tangent": [1.0, 0.0, 0.0]},
                    ],
                }
            ]
        }
    )


def test_closed_spline_with_three_nodes_accepted():
    validate(
        {
            "splines": [
                {
                    "closed": True,
                    "nodes": [
                        {"origin": [0.0, 0.0, 0.0], "tangent": [1.0, 0.0, 0.0]},
                        {"origin": [1.0, 0.0, 0.0], "tangent": [0.0, 1.0, 0.0]},
                        {"origin": [0.5, 1.0, 0.0], "tangent": [-1.0, 0.0, 0.0]},
                    ],
                }
            ]
        }
    )


# --- Rejection paths ----------------------------------------------------


def test_top_level_must_be_dict():
    with pytest.raises(ValueError):
        validate([])  # type: ignore[arg-type]


def test_splines_must_be_list():
    with pytest.raises(ValueError):
        validate({"splines": "nope"})


def test_node_origin_wrong_shape():
    with pytest.raises(ValueError):
        validate(
            {
                "splines": [
                    {"closed": False, "nodes": [{"origin": [0.0], "tangent": [1, 0, 0]}]}
                ]
            }
        )


def test_node_tangent_with_nan_rejected():
    with pytest.raises(ValueError):
        validate(
            {
                "splines": [
                    {
                        "closed": False,
                        "nodes": [
                            {"origin": [0.0, 0.0, 0.0], "tangent": [math.nan, 0.0, 0.0]}
                        ],
                    }
                ]
            }
        )


def test_closed_spline_with_two_nodes_rejected():
    with pytest.raises(ValueError, match="closed=true"):
        validate(
            {
                "splines": [
                    {
                        "closed": True,
                        "nodes": [
                            {"origin": [0.0, 0.0, 0.0], "tangent": [1.0, 0.0, 0.0]},
                            {"origin": [1.0, 0.0, 0.0], "tangent": [0.0, 1.0, 0.0]},
                        ],
                    }
                ]
            }
        )


def test_closed_spline_with_zero_nodes_rejected():
    with pytest.raises(ValueError):
        validate({"splines": [{"closed": True, "nodes": []}]})


def test_open_spline_with_zero_nodes_accepted():
    """Empty splines are valid placeholders for "break" (Dbl-click R)."""
    validate({"splines": [{"closed": False, "nodes": []}]})


# --- v2 schema (origin + p_a + p_b) -------------------------------------


def test_v2_node_with_explicit_handles_accepted():
    validate(
        {
            "version": 2,
            "splines": [
                {
                    "closed": False,
                    "nodes": [
                        {
                            "origin": [0.0, 0.0, 0.0],
                            "p_a": [-1.0, 0.0, 0.0],
                            "p_b": [1.0, 0.0, 0.0],
                        }
                    ],
                }
            ],
        }
    )


def test_v2_node_with_null_handles_accepted():
    """Null p_a / p_b is valid for placeholder single-node splines."""
    validate(
        {
            "version": 2,
            "splines": [
                {
                    "closed": False,
                    "nodes": [
                        {"origin": [0.0, 0.0, 0.0], "p_a": None, "p_b": None}
                    ],
                }
            ],
        }
    )


def test_v2_node_with_p_a_nan_rejected():
    with pytest.raises(ValueError):
        validate(
            {
                "version": 2,
                "splines": [
                    {
                        "closed": False,
                        "nodes": [
                            {
                                "origin": [0.0, 0.0, 0.0],
                                "p_a": [math.nan, 0.0, 0.0],
                                "p_b": [1.0, 0.0, 0.0],
                            }
                        ],
                    }
                ],
            }
        )


def test_node_missing_tangent_and_handles_rejected():
    """A node must have either v1 'tangent' or v2 'p_a' + 'p_b'."""
    with pytest.raises(ValueError, match="tangent"):
        validate(
            {
                "splines": [
                    {"closed": False, "nodes": [{"origin": [0.0, 0.0, 0.0]}]}
                ]
            }
        )


def test_node_with_only_p_a_rejected():
    """Half-v2 (p_a without p_b) is rejected — it's neither schema."""
    with pytest.raises(ValueError):
        validate(
            {
                "splines": [
                    {
                        "closed": False,
                        "nodes": [
                            {"origin": [0.0, 0.0, 0.0], "p_a": [1.0, 0.0, 0.0]}
                        ],
                    }
                ]
            }
        )


def test_mixed_v1_v2_nodes_in_same_spline_accepted():
    """The validator dispatches per-node, so a session can mix schemas."""
    validate(
        {
            "splines": [
                {
                    "closed": False,
                    "nodes": [
                        {"origin": [0.0, 0.0, 0.0], "tangent": [1.0, 0.0, 0.0]},
                        {
                            "origin": [1.0, 0.0, 0.0],
                            "p_a": [0.5, 0.0, 0.0],
                            "p_b": [1.5, 0.0, 0.0],
                        },
                    ],
                }
            ]
        }
    )
