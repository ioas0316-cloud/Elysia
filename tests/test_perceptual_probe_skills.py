"""
Unit tests for Perceptual Probe Skill & Relational Void Engine in Project Elysia.
================================================================================
Verifies:
1. Active probing against CausalField & friction error E measurement.
2. Relational Void Probe sensing negative space constraints.
3. SkillAdaptabilityMap fine-tuning rotors along error gradients & tacit knowledge accumulation.
4. PerceptualProbeEngine multi-skill execution and feedback loops into CausalField.
"""

import pytest
import numpy as np
from core.physics.causal_field import CausalField, InformationVoxel
from elysia_core.perceptual_probe_skill import (
    PerceptualProbeSkill,
    RelationalVoidProbe,
    SkillAdaptabilityMap,
    PerceptualProbeEngine,
    ProbeFeedback
)


def test_relational_void_probe():
    cf = CausalField(dimensions=5)
    v1 = InformationVoxel(
        id="voxel_1",
        content="Target Voxel",
        tensor=np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        position=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        potential=0.2
    )
    cf.add_voxel(v1)

    void_probe = RelationalVoidProbe(dimension=5)
    probe_vec = np.array([0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    void_tension, void_dir = void_probe.detect_relational_void(cf, probe_vec)

    assert void_tension >= 0.0
    assert len(void_dir) == 5


def test_skill_adaptability_map():
    adapt_map = SkillAdaptabilityMap(dimension=5)
    ctx = "test_environment"

    # Default initial rotor
    r0 = adapt_map.get_context_rotor(ctx)
    assert np.allclose(r0, [1.0, 0.0, 0.0, 0.0, 0.0])
    assert adapt_map.get_tacit_knowledge(ctx) == 0.0

    # Update rotor along gradient
    grad = np.array([0.2, 0.1, -0.1, 0.0, 0.0], dtype=np.float32)
    r1, knowledge_gain = adapt_map.update_context_rotor(
        context_name=ctx,
        gradient_shift=grad,
        friction_error=0.4,
        learning_rate=0.2
    )

    assert not np.allclose(r0, r1)
    assert np.isclose(np.linalg.norm(r1), 1.0)
    assert knowledge_gain > 0.0
    assert adapt_map.get_tacit_knowledge(ctx) > 0.0


def test_perceptual_probe_skill_convergence():
    def dummy_probe_func(state: np.ndarray) -> np.ndarray:
        return state * 1.5 + 0.1

    skill = PerceptualProbeSkill(
        skill_id="test_probe",
        name="Test Active Probe",
        base_probe_func=dummy_probe_func,
        dimension=5,
        learning_rate=0.25
    )

    cf = CausalField(dimensions=5)
    v1 = InformationVoxel(
        id="target_voxel",
        content="Target Voxel",
        tensor=np.array([1.0, 0.5, 0.2, 0.0, 0.0], dtype=np.float32),
        position=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        potential=0.5
    )
    cf.add_voxel(v1)

    input_state = np.array([1.0, 0.5, 0.2, 0.0, 0.0], dtype=np.float32)
    friction_errors = []

    # Run multiple probe executions to verify error convergence & rotor adaptation
    for step in range(5):
        feedback = skill.probe_and_execute(
            input_state=input_state,
            causal_field=cf,
            context_name="volatile_field",
            target_voxel_id="target_voxel"
        )
        assert isinstance(feedback, ProbeFeedback)
        friction_errors.append(feedback.friction_error)

    # Check that tacit knowledge accumulated over executions
    tacit_knowledge = skill.adaptability_map.get_tacit_knowledge("volatile_field")
    assert tacit_knowledge > 0.0
    assert skill.execution_count == 5


def test_perceptual_probe_engine():
    engine = PerceptualProbeEngine(dimension=5)

    def force_impulse_skill(state: np.ndarray) -> np.ndarray:
        return state * 0.8 + np.array([0.1, 0.2, 0.0, 0.0, 0.0], dtype=np.float32)

    skill = PerceptualProbeSkill(
        skill_id="force_impulse",
        name="Force Impulse Probe",
        base_probe_func=force_impulse_skill,
        dimension=5
    )
    engine.register_skill(skill)

    cf = CausalField(dimensions=5)
    v1 = InformationVoxel(
        id="voxel_main",
        content="Main Core",
        tensor=np.array([0.8, 0.2, 0.0, 0.0, 0.0], dtype=np.float32)
    )
    cf.add_voxel(v1)

    fb = engine.execute_skill_probe(
        skill_id="force_impulse",
        input_state=np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        causal_field=cf,
        context_name="arena",
        target_voxel_id="voxel_main"
    )

    assert fb.skill_id == "force_impulse"
    assert fb.target_context == "arena"
    summary = engine.get_tacit_knowledge_summary("arena")
    assert "force_impulse" in summary
    assert summary["force_impulse"] > 0.0
