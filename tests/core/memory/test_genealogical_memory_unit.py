import pytest
import numpy as np
from core.physics.semantic_mass_engine import RawPerturbationImpulse, SemanticMassEngine
from core.memory.genealogical_memory_unit import (
    StructuralProvenanceTrace,
    JustificationTensor,
    GenealogicalMemoryUnit,
    DynamicDeconstructionEngine,
)

def test_structural_provenance_trace():
    trace = StructuralProvenanceTrace(
        initial_tension_state=np.ones(16, dtype=np.float32),
        raw_perturbation_impulse="Raw_Vibration_01",
        refraction_delta_vector=np.ones(16, dtype=np.float32) * 0.5,
        causal_antecedent_ids=["Node_00"],
        contrast_resonance_matrix=np.eye(4, dtype=np.float32)
    )

    data = trace.to_dict()
    assert data["raw_perturbation_impulse"] == "Raw_Vibration_01"
    assert data["causal_antecedent_ids"] == ["Node_00"]
    assert len(data["initial_tension_state"]) == 16

def test_justification_tensor():
    jt = JustificationTensor(
        concept_gravity_proof=2.5,
        trinitarian_contrast_delta=1.2,
        label_necessity_score=0.85,
        justification_matrix=np.eye(4, dtype=np.float32)
    )

    proof = jt.verify_label_necessity(threshold=0.5)
    assert proof["is_justified"] is True
    assert proof["label_necessity_score"] == 0.85

def test_dynamic_deconstruction_engine_creation_and_deconstruction():
    engine = SemanticMassEngine(dimensions=16)
    deconstruction_engine = DynamicDeconstructionEngine(deconstruction_threshold=0.4)

    raw1 = RawPerturbationImpulse(impulse_id="impulse_1", raw_signal="Wind Blowing North", intensity=1.5)

    # 1. Create unit
    unit1 = deconstruction_engine.create_and_register_unit(
        unit_id="Memory_Unit_01",
        label="North_Wind_Phenomenon",
        raw_friction=raw1,
        semantic_engine=engine,
        causal_antecedent_ids=[]
    )

    assert unit1.unit_id == "Memory_Unit_01"
    assert unit1.is_deconstructed is False
    assert len(unit1.provenance_trace.refraction_delta_vector) == 16

    genealogy_proof = unit1.prove_genealogy()
    assert genealogy_proof["unit_id"] == "Memory_Unit_01"
    assert genealogy_proof["justification_proof"]["is_justified"] is True

    # 2. Introduce contradicting raw friction
    raw_contradict = RawPerturbationImpulse(
        impulse_id="impulse_2",
        raw_signal="Opposing Force Shockwave",
        intensity=2.0,
        frequency_signature=-1.0 * unit1.provenance_trace.refraction_delta_vector
    )

    is_deconstructed, new_unit, report = deconstruction_engine.evaluate_and_deconstruct(
        target_unit_id="Memory_Unit_01",
        new_raw_friction=raw_contradict,
        semantic_engine=engine,
        new_label_if_rewoven="Rewoven_South_Shockwave"
    )

    assert is_deconstructed is True
    assert unit1.is_deconstructed is True
    assert new_unit is not None
    assert new_unit.label == "Rewoven_South_Shockwave"
    assert "Memory_Unit_01" in new_unit.provenance_trace.causal_antecedent_ids
    assert report["deconstruction_triggered"] is True
