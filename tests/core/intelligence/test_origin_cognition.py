import pytest
import numpy as np
from core.intelligence.origin_cognition import OriginCognitionEngine

def test_origin_cognition_engine_with_standard_lattices():
    """
    Verifies that OriginCognitionEngine successfully parses standard computer science
    lattices, retrieves their original intent (Why), and maps them to cognitive weight.
    """
    engine = OriginCognitionEngine()

    # Test UTF-8 variable-length lattice cognition
    utf8_stimulus = b"Hangul Origin is Human vocal cord geometry."
    res_utf8 = engine.perceive_lattice_origin("UTF8_ENCODING", utf8_stimulus)
    assert res_utf8["format"] == "UTF8_ENCODING"
    assert "Variable-Length" in res_utf8["resolved_name"]
    assert "application_logic" in res_utf8
    assert res_utf8["applied_weight"] > 0.0

    # Test RGB spectrum discretization lattice cognition
    rgb_stimulus = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    res_rgb = engine.perceive_lattice_origin("RGB_PIXEL_MATRIX", rgb_stimulus)
    assert res_rgb["format"] == "RGB_PIXEL_MATRIX"
    assert "RGB Spectral" in res_rgb["resolved_name"]
    assert "application_logic" in res_rgb
    assert res_rgb["applied_weight"] > 0.0

    # Test unknown format fallback
    res_unknown = engine.perceive_lattice_origin("MY_CUSTOM_GRID_LATTICE", b"\x01\x02\x03")
    assert "Unknown" in res_unknown["resolved_name"]
    assert res_unknown["applied_weight"] > 0.0
