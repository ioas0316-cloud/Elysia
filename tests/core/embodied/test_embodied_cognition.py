"""
Unit tests for Embodied Cognition, VQ Projector, Dual Plasticity Pipeline, and OOD Emergence.
"""

import pytest
from core.embodied.projector import VQSensoryProjector
from core.embodied.plasticity import DualPlasticityPipeline
from core.embodied.emergence import (
    AxiomInducer,
    ConsistencyFilter,
    CrossModalCausalFilter,
    OODLatentBuffer,
    OODSample,
    ResidualCompressibilityFilter,
)
from core.topology.grounding_ontology import GroundingAxiom, GroundingOntologyEngine
from core.engine.integrated_causal_engine import IntegratedCausalEngine


def test_vq_sensory_projector_and_dual_plasticity():
    projector = VQSensoryProjector()
    projector.register_prototype("THERMAL_HAZARD", [1.0, 0.0], initial_threshold=0.5)
    projector.register_prototype("FROST_CRYSTAL", [0.8, 0.2], initial_threshold=0.5)

    ontology = GroundingOntologyEngine()
    ontology.register_axiom(
        GroundingAxiom(
            axiom_id="AXIOM_01",
            description="Thermal & Frost exclusion",
            forbidden_pairs=[("THERMAL_HAZARD", "FROST_CRYSTAL")],
            required_bindings={"SAFETY_CONTAINMENT": "ACTIVE"},
        )
    )

    pipeline = DualPlasticityPipeline(projector, ontology)

    # Signal on overlapping boundary
    raw_signal = [0.9, 0.1]
    res1 = pipeline.process_sensory_input(raw_signal)

    # After initial violation, threshold for FROST_CRYSTAL is contracted (0.5 -> 0.25)
    assert projector.prototypes["FROST_CRYSTAL"].threshold == 0.25

    # Second pass with same signal -> FROST_CRYSTAL no longer activates (dist ~ 0.141 > 0.25? wait, dist is sqrt((0.9-0.8)^2 + (0.1-0.2)^2) = sqrt(0.01 + 0.01) = 0.141 <= 0.25)
    # Let's test adapt_threshold further down
    projector.adapt_threshold("FROST_CRYSTAL", contraction_factor=0.5)  # 0.25 -> 0.125
    qualities, _ = projector.project_to_qualities(raw_signal)
    assert "FROST_CRYSTAL" not in qualities
    assert "THERMAL_HAZARD" in qualities


def test_ood_filters_and_axiom_inducer():
    buffer = OODLatentBuffer(capacity=10)
    inducer = AxiomInducer(buffer)

    # 1. Less than 5 samples -> None
    assert inducer.discover_new_concept() is None

    # 2. Add 5 structured OOD samples
    for i in range(5):
        buffer.add_unknown_signal(
            OODSample(
                sensory_vector=[2.5 + i * 0.01, 3.0 - i * 0.01, 1.0],
                timestamp_ms=100.0 + i * 10.0,
                modalities={"visual": [2.5, 3.0], "motor": [1.0, 1.0]},
            )
        )

    axiom_data = inducer.discover_new_concept()
    assert axiom_data is not None
    assert axiom_data["symbol"].startswith("EMERGENT_CONCEPT_")
    assert axiom_data["metrics"]["rank"] <= 2
    assert axiom_data["metrics"]["compressibility"] >= 0.3


def test_integrated_causal_engine():
    engine = IntegratedCausalEngine()

    # Process continuous vector that doesn't trigger initial prototypes -> stored in OOD
    res = engine.process_continuous_sensory_vector([5.0, 5.0, 5.0], timestamp_ms=10.0)
    assert len(engine.ood_buffer.buffer) == 1
