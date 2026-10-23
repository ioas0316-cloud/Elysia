"""
Unit and Integration Tests for Cognitive OS Runtime & Organ Adapters
=====================================================================
Tests Cognitive OS organ adapters, Markov Blanket self-boundary evaluation,
perceptual probing, and tacit knowledge accumulation.
"""

import sys
import pytest
import numpy as np

# Ensure path resolution
sys.path.extend(['.', 'L1_causal_field', 'L2_cognitive_engine'])

from L3_interface.simulators.cognitive_os_organ_adapters import (
    ParticleFluidSandboxOrgan,
    NonEuclideanSpacetimeOrgan,
    BlenderMeshOrgan,
    CADConstraintOrgan,
    PhotoshopRetinaOrgan,
    GameEnginePhysicsOrgan
)
from elysia_core.cognitive_os_runtime import CognitiveOSRuntime, MarkovBlanketSelfBoundary


def test_organ_adapters_sense_and_intervene():
    """Verify that all organ adapters return valid OrganSensation payloads."""
    organs = [
        ParticleFluidSandboxOrgan(),
        NonEuclideanSpacetimeOrgan(),
        BlenderMeshOrgan(),
        CADConstraintOrgan(),
        PhotoshopRetinaOrgan(),
        GameEnginePhysicsOrgan()
    ]

    action = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    for organ in organs:
        sensation = organ.sense_and_intervene(action)
        assert sensation.organ_id == organ.organ_id
        assert len(sensation.sensation_vector) == 5
        assert sensation.resistance_friction >= 0.0
        assert sensation.echo_latency > 0.0


def test_markov_blanket_boundary_classification():
    """Verify that MarkovBlanketSelfBoundary classifies organs into Self, Boundary, and Environment."""
    boundary_engine = MarkovBlanketSelfBoundary(
        self_friction_threshold=0.2,
        environment_friction_threshold=0.5
    )

    action = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)

    organ_low = NonEuclideanSpacetimeOrgan(organ_id="low_friction_organ")
    organ_high = CADConstraintOrgan(organ_id="high_friction_organ")

    sensations = {
        "low_friction_organ": organ_low.sense_and_intervene(action),
        "high_friction_organ": organ_high.sense_and_intervene(action)
    }

    report = boundary_engine.evaluate_boundary(sensations, probe_feedbacks={})

    assert isinstance(report.self_region_organs, list)
    assert isinstance(report.boundary_organs, list)
    assert isinstance(report.environment_organs, list)
    assert report.total_free_energy > 0.0
    assert 0.0 < report.causal_gravity_radius <= 1.0


def test_cognitive_os_runtime_full_cycle():
    """Verify multi-step Cognitive OS runtime execution and tacit knowledge accumulation."""
    runtime = CognitiveOSRuntime(dimension=5)

    fluid = ParticleFluidSandboxOrgan(organ_id="fluid_01")
    cad = CADConstraintOrgan(organ_id="cad_01")

    runtime.register_organ(fluid)
    runtime.register_organ(cad)

    context_name = "test_context"
    intent = np.array([0.8, 0.2, 0.5, 0.1, 0.3], dtype=np.float32)

    res1 = runtime.execute_cognitive_cycle(intent, context_name=context_name)
    assert res1["step_count"] == 1
    assert "fluid_01" in res1["sensations"]
    assert "cad_01" in res1["sensations"]

    tk1_fluid = res1["tacit_knowledge_summary"]["skill_fluid_01"]

    res2 = runtime.execute_cognitive_cycle(intent, context_name=context_name)
    tk2_fluid = res2["tacit_knowledge_summary"]["skill_fluid_01"]

    # Verify tacit knowledge accumulates over steps
    assert tk2_fluid > tk1_fluid
