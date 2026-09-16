r"""
Unit Tests for Recursive Causal Evolution Loop
=============================================

Tests Component (\Delta_c), Principle (\Delta_p), and Structural (\Delta_s) fractal tension
propagation, meta-principle evolution, dynamic memory deconstruction, and zero-convergence feedback.
"""

import pytest
import numpy as np

from core.engine.recursive_causal_evolution_loop import (
    RecursiveCausalEvolutionLoop,
    MetaCausalPrinciple,
    FractalTensionReport,
)
from core.ingestion.raw_byte_causal_sensor import StructuralViolationError


def test_component_level_zero_convergence():
    """Tests authentic UTF-8 input reaching component level zero-convergence."""
    loop = RecursiveCausalEvolutionLoop(dimensions=16, principle_threshold=0.5)

    valid_utf8 = "세상의 빛과 인과적 연결성을 대조하는 완벽한 규격 스트림".encode("utf-8")
    report = loop.process_cycle(
        raw_input=valid_utf8,
        cycle_id="Test_Cycle_ValidUTF8",
        expected_format="UTF-8",
        enforce_strict_byte=True,
    )

    assert isinstance(report, FractalTensionReport)
    assert report.delta_c < 1e-3
    assert report.details["cycle_count"] == 1


def test_principle_evolution_on_persistent_friction():
    """Tests meta-principle evolution (\Theta shift) when principle tension (\Delta_p) exceeds threshold."""
    loop = RecursiveCausalEvolutionLoop(dimensions=16, principle_threshold=0.3)
    initial_version = loop.principle.evolution_version

    # Input orthogonal feature vector creating high principle dissonance
    novel_feature = np.zeros(16, dtype=np.float32)
    novel_feature[15] = 1.0  # High activation at dimension 15, orthogonal to initial ones vector

    report = loop.process_cycle(
        raw_input=novel_feature,
        cycle_id="Test_Cycle_NovelFeature",
    )

    assert report.principle_evolved is True
    assert loop.principle.evolution_version == initial_version + 1
    assert len(loop.principle.evolution_history) == 1
    # Check that principle parameter vector evolved towards novel feature
    assert loop.principle.param_vector[15] > 0.1


def test_structural_level_deconstruction_and_feedback_convergence():
    """Tests structural memory deconstruction and top-down zero-convergence feedback relaxation."""
    loop = RecursiveCausalEvolutionLoop(dimensions=16, principle_threshold=0.3, structural_threshold=0.4)

    # Cycle 1: Create initial base memory unit
    base_input = b"BASE_SYSTEM_SPECIFICATION_HEADER_0000"
    report1 = loop.process_cycle(
        raw_input=base_input,
        cycle_id="Base_Cycle",
        expected_format="ASCII",
    )
    unit_id = report1.details["created_unit_id"]

    # Cycle 2: Introduce contradictory impulse targeting base memory unit
    contradictory_feature = np.zeros(16, dtype=np.float32)
    contradictory_feature[0] = -1.0
    contradictory_feature[8] = 2.0

    report2 = loop.process_cycle(
        raw_input=contradictory_feature,
        cycle_id="Contradictory_Cycle",
        target_memory_unit_id=unit_id,
    )

    assert report2.memory_deconstructed is True
    # Verify that target memory unit was marked deconstructed in deconstruction engine
    target_unit = loop.deconstruction_engine.memory_units[unit_id]
    assert target_unit.is_deconstructed is True
    # Verify top-down feedback relaxation was applied
    assert report2.delta_s < 0.5


def test_fractal_multi_cycle_evolution_loop():
    """Tests a full multi-cycle evolution loop sequence."""
    loop = RecursiveCausalEvolutionLoop(dimensions=16)

    data_stream = [
        ("VALID_HEADER_DATA_1".encode("ascii"), "ASCII"),
        ("VALID_HEADER_DATA_2".encode("ascii"), "ASCII"),
        ("변형된 한국어 UTF8 데이터 스트림", "UTF-8"),
    ]

    for idx, (raw_data, fmt) in enumerate(data_stream):
        report = loop.process_cycle(
            raw_input=raw_data,
            cycle_id=f"MultiCycle_{idx}",
            expected_format=fmt,
        )
        assert report.total_fractal_tension >= 0.0

    assert loop.cycle_count == 3
