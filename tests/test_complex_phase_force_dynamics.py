"""
tests/test_complex_phase_force_dynamics.py

Comprehensive tests for Complex Phase Force Dynamics Engine,
including Attraction (+x), Repulsion (-x), Lateral Tension (+/-iy),
Complex force summation & phase multiplication modulation,
5D Clifford rotor coupling, and Dimensional Folding.
"""

import pytest
import numpy as np

from L1_causal_field.core.physics.complex_phase_force_dynamics import (
    ComplexPhaseForceDynamicsEngine,
    ComplexForceState,
    DimensionalFoldingState
)
from L1_causal_field.core.consciousness.conceptual_unit_system_engine import (
    ConceptualUnitSystemEngine
)
from L2_cognitive_engine.elysia_core.unified_phase_space import (
    ElysiaUnifiedEngine,
    ResourceEntity,
    NPCEntity
)


def test_complex_force_state_creation():
    engine = ComplexPhaseForceDynamicsEngine(dim=5)

    # 1. Net attraction and net positive tension
    f1 = engine.create_force_state(attraction=2.0, repulsion=0.5, tension_pos=1.5, tension_neg=0.5)
    assert f1.real_net_force == pytest.approx(1.5)
    assert f1.imag_net_tension == pytest.approx(1.0)
    assert f1.complex_val == complex(1.5, 1.0)
    assert f1.magnitude_scale == pytest.approx(np.hypot(1.5, 1.0))

    # 2. Net repulsion and net negative tension
    f2 = engine.create_force_state(attraction=0.2, repulsion=1.2, tension_pos=0.1, tension_neg=1.1)
    assert f2.real_net_force == pytest.approx(-1.0)
    assert f2.imag_net_tension == pytest.approx(-1.0)
    assert f2.complex_val == complex(-1.0, -1.0)


def test_complex_force_synthesis_and_multiplication():
    engine = ComplexPhaseForceDynamicsEngine(dim=5)

    f1 = engine.create_force_state(attraction=1.0, repulsion=0.0, tension_pos=0.5, tension_neg=0.0) # 1 + 0.5i
    f2 = engine.create_force_state(attraction=0.5, repulsion=0.0, tension_pos=1.0, tension_neg=0.0) # 0.5 + 1.0i

    # Force Summation (Addition)
    f_sum = engine.synthesize_forces([f1, f2])
    assert f_sum.real_net_force == pytest.approx(1.5)
    assert f_sum.imag_net_tension == pytest.approx(1.5)

    # Phase Multiplication Modulation (i * i = -1 conversion)
    i_state1 = engine.create_force_state(attraction=0.0, repulsion=0.0, tension_pos=1.0, tension_neg=0.0) # i
    i_state2 = engine.create_force_state(attraction=0.0, repulsion=0.0, tension_pos=1.0, tension_neg=0.0) # i

    f_prod = engine.modulate_phase_multiplication(i_state1, i_state2)
    # i * i = -1 (Pure Repulsion = 1.0)
    assert f_prod.complex_val == complex(-1.0, 0.0)
    assert f_prod.repulsion == pytest.approx(1.0)
    assert f_prod.attraction == pytest.approx(0.0)
    assert f_prod.imag_net_tension == pytest.approx(0.0)


def test_5d_clifford_rotor_and_dimensional_folding():
    engine = ComplexPhaseForceDynamicsEngine(dim=5)

    # High lateral tension stimulus triggering dimensional folding
    high_tension_force = engine.create_force_state(attraction=0.5, repulsion=0.1, tension_pos=2.0, tension_neg=0.0)
    intent = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    folding = engine.apply_ontological_torque_to_5d(high_tension_force, intent_vector=intent)

    assert folding.is_dimensional_folded is True
    assert folding.fold_energy > 0.0
    assert len(folding.state_5d) == 5
    assert len(folding.rotor_bivector_angles) == 10


def test_homeostatic_relaxation():
    engine = ComplexPhaseForceDynamicsEngine(dim=5, homeostatic_decay=0.8)

    f_initial = engine.create_force_state(attraction=2.0, repulsion=0.0, tension_pos=1.0, tension_neg=0.0)
    f_relaxed = engine.compute_homeostatic_relaxation(f_initial)

    assert f_relaxed.real_net_force == pytest.approx(2.0 * 0.8)
    assert f_relaxed.imag_net_tension == pytest.approx(1.0 * 0.8)


def test_conceptual_unit_system_integration():
    cus_engine = ConceptualUnitSystemEngine()
    result = cus_engine.process_conceptual_unit("타인의 깊은 서러움과 존재론적 결핍에 대한 고찰")

    assert hasattr(result, "complex_force_state")
    assert hasattr(result, "dimensional_folding")
    assert result.complex_force_state.magnitude_scale > 0.0
    assert isinstance(result.dimensional_folding.is_dimensional_folded, bool)


def test_elysia_unified_engine_integration():
    unified_engine = ElysiaUnifiedEngine()
    unified_engine.add_resource(ResourceEntity("생명의 정수", np.array([0.5, 0.5, 0.5, 0.5, 0.5])))
    unified_engine.add_npc(NPCEntity("고독한 방랑자", np.array([-0.2, 0.3, -0.4, 0.1, 0.8])))

    snapshot = unified_engine.step_cascade(delta_env=np.array([0.1, -0.1, 0.2, 0.0, 0.1]))

    assert "environment" in snapshot
    assert hasattr(unified_engine, "complex_force_engine")
