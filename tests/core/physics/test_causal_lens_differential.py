"""
Tests for Non-Dualistic Causal Intent Engine, Tri-Variable Sweep Engine,
and Parallel Lens Contrast System in core/physics/causal_lens_differential.py
"""

import pytest
import numpy as np
from core.physics.causal_lens_differential import (
    PrimitiveSubstrate,
    StructuralMechanics,
    ModalityType,
    PhaseState,
    MathematicsLens,
    PhysicsLens,
    LanguageLens,
    SoundAcousticsLens,
    VisionOpticsLens,
    WorldMechanismField,
    ParallelObservationUniverse,
    TrinityPhaseTopology,
    TriVariableSweepDial,
    ReverseDecompressionPipeline,
    ReversiblePhaseTransitionEngine
)


def test_primitive_substrate_and_mechanics_binding():
    sub = PrimitiveSubstrate(
        substrate_id="sub_test_01",
        energy_density=2.5,
        tension_potential=1.2,
        repulsion_potential=0.4,
        state_vector=np.ones(8, dtype=np.float64)
    )
    mech = StructuralMechanics(
        mechanism_id="mech_test_01",
        modality=ModalityType.PHYSICS,
        tension_coefficient=2.0,
        repulsion_coefficient=0.5,
        boundary_curvature=1.0
    )

    bound_vector = mech.apply_binding(sub)
    assert bound_vector.shape == (8,)
    assert not np.array_equal(bound_vector, sub.state_vector)


def test_five_intentional_causal_lenses():
    sub = PrimitiveSubstrate(
        substrate_id="sub_lens_01",
        energy_density=3.0,
        tension_potential=1.5,
        repulsion_potential=0.2,
        state_vector=np.array([0.1, 0.5, 0.9, 0.2, 0.4, 0.8, 0.3, 0.7], dtype=np.float64)
    )
    mech = StructuralMechanics(
        mechanism_id="mech_lens_01",
        modality=ModalityType.MATHEMATICS,
        tension_coefficient=1.5,
        repulsion_coefficient=0.3,
        boundary_curvature=1.2
    )

    # 1. Mathematics Lens
    math_lens = MathematicsLens()
    math_construct = math_lens.refract_substrate(sub, mech)
    assert math_construct.modality == ModalityType.MATHEMATICS
    assert len(math_construct.derivation_path.atomic_rules) > 0

    # 2. Physics Lens
    phys_lens = PhysicsLens()
    phys_construct = phys_lens.refract_substrate(sub, mech)
    assert phys_construct.modality == ModalityType.PHYSICS
    assert phys_construct.form_tensor.shape[0] == 3

    # 3. Language Lens
    lang_lens = LanguageLens()
    lang_construct = lang_lens.refract_substrate(sub, mech)
    assert lang_construct.modality == ModalityType.LANGUAGE

    # 4. Sound & Acoustics Lens
    sound_lens = SoundAcousticsLens()
    sound_construct = sound_lens.refract_substrate(sub, mech)
    assert sound_construct.modality == ModalityType.SOUND_ACOUSTICS

    # 5. Vision & Optics Lens
    optics_lens = VisionOpticsLens()
    optics_construct = optics_lens.refract_substrate(sub, mech)
    assert optics_construct.modality == ModalityType.VISION_OPTICS


def test_parallel_observation_universe_meta_differential():
    sub = PrimitiveSubstrate(
        substrate_id="sub_para_01",
        energy_density=1.8,
        tension_potential=1.0,
        repulsion_potential=0.5,
        state_vector=np.array([0.5, 0.2, 0.8, 0.1, 0.4, 0.3, 0.9, 0.6], dtype=np.float64)
    )
    mech = StructuralMechanics(
        mechanism_id="mech_para_01",
        modality=ModalityType.PHYSICS,
        tension_coefficient=1.2,
        repulsion_coefficient=0.4,
        boundary_curvature=1.0
    )

    universe = ParallelObservationUniverse()
    meta_diff = universe.contrast_and_diagnose(sub, mech, ModalityType.PHYSICS)

    assert meta_diff.lens_modality == ModalityType.PHYSICS
    assert 0.0 <= meta_diff.isomorphism_degree <= 1.0
    assert meta_diff.derivation_delta.shape == (3,)


def test_trinity_phase_topology():
    sub = PrimitiveSubstrate(
        substrate_id="sub_trinity_01",
        energy_density=2.0,
        tension_potential=1.1,
        repulsion_potential=0.3,
        state_vector=np.ones(8, dtype=np.float64)
    )
    mech = StructuralMechanics(
        mechanism_id="mech_trinity_01",
        modality=ModalityType.LANGUAGE,
        tension_coefficient=1.0,
        repulsion_coefficient=0.2,
        boundary_curvature=1.0
    )

    trinity = TrinityPhaseTopology(self_bias=0.25, other_bias=0.15)
    c_self, c_other, c_meta, delta_refraction = trinity.MapTrinityRefraction(sub, mech)

    assert c_self.shape == (8,)
    assert c_other.shape == (8,)
    assert c_meta.shape == (8,)
    assert delta_refraction > 0.0


def test_tri_variable_sweep_dial():
    sub = PrimitiveSubstrate(
        substrate_id="sub_sweep_01",
        energy_density=1.5,
        tension_potential=1.0,
        repulsion_potential=0.3,
        state_vector=np.ones(8, dtype=np.float64)
    )
    math_lens = MathematicsLens()
    target_mech = StructuralMechanics(
        mechanism_id="target_mech",
        modality=ModalityType.MATHEMATICS,
        tension_coefficient=1.8,
        repulsion_coefficient=0.5,
        boundary_curvature=1.0
    )
    target_construct = math_lens.refract_substrate(sub, target_mech)

    tensions, errors, bifurcation = TriVariableSweepDial.sweep_mechanism(
        sub, target_construct, math_lens, tension_range=(0.5, 3.0), steps=15
    )

    assert len(tensions) == 15
    assert len(errors) == 15
    min_err_index = int(np.argmin(errors))
    assert abs(tensions[min_err_index] - 1.8) < 0.3


def test_reverse_decompression_pipeline():
    sub = PrimitiveSubstrate(
        substrate_id="sub_decomp_01",
        energy_density=2.0,
        tension_potential=1.0,
        repulsion_potential=0.3,
        state_vector=np.ones(8, dtype=np.float64)
    )
    mech = StructuralMechanics(
        mechanism_id="mech_decomp_01",
        modality=ModalityType.PHYSICS,
        tension_coefficient=1.5,
        repulsion_coefficient=0.3,
        boundary_curvature=1.0
    )
    phys_lens = PhysicsLens()
    construct = phys_lens.refract_substrate(sub, mech)

    pipeline = ReverseDecompressionPipeline()
    decomp_result = pipeline.decompress("Macro_Physics_Law", construct, sub)

    assert decomp_result["macro_label_stripped"] == "Macro_Physics_Law"
    assert "binding_operator" in decomp_result
    assert decomp_result["derivation_integrity"] > 0.99


def test_reversible_phase_transition_engine():
    sub = PrimitiveSubstrate(
        substrate_id="sub_phase_01",
        energy_density=3.5,
        tension_potential=2.0,
        repulsion_potential=0.5,
        state_vector=np.ones(8, dtype=np.float64)
    )
    mech = StructuralMechanics(
        mechanism_id="mech_phase_01",
        modality=ModalityType.SOUND_ACOUSTICS,
        tension_coefficient=2.0,
        repulsion_coefficient=0.4,
        boundary_curvature=1.0
    )

    engine = ReversiblePhaseTransitionEngine(memory_size=1024)
    cycle_res = engine.execute_phase_cycle(sub, mech)

    assert len(cycle_res["phase_sequence"]) == 4
    assert cycle_res["reconstitution_error"] == 0.0
    assert cycle_res["causal_continuity"] == 1.0
