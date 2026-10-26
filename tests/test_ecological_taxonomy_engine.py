"""
Unit & Integration Tests for Ecological Taxonomy & Scale Coupling Engine
========================================================================

Tests:
1. Modal Roots Initialization & Taxonomy Node Growth
2. Substrate Unity & State/Phase Transitions (H2O: Ice -> Water -> Steam)
3. Scale Coupling Cascade & Bifurcation (Hair/Nail Damped vs. Blood/Bone Core Breach)
4. Topological Homeostasis & Autonomous Local Immune Quarantine
5. Cross-Modal Isomorphic Association (Surface Friction <-> Non-Linear Equation)
6. CausalField Integration (process_ecological_stimulus)
"""

import pytest
import numpy as np
from core.topology.ecological_taxonomy_engine import (
    EcologicalTaxonomyEngine,
    SensoryModality,
    SubstrateIdentity,
    SpectrumCoordinate,
    CascadeMode
)
from core.physics.causal_field import CausalField


def test_modal_roots_and_node_growth():
    engine = EcologicalTaxonomyEngine(target_dimension=8)

    # Verify roots are initialized
    for mod in SensoryModality:
        root = engine.modal_roots.get_root(mod)
        assert root is not None
        assert root.modality == mod

    # Sprout new node under Tactile Root
    substrate_sand = SubstrateIdentity(
        substrate_id="sub_sand",
        chemical_or_logical_formula="SiO2",
        core_invariants=["silicon_dioxide_lattice", "surface_friction"],
        current_phase_state="solid_granular"
    )

    node_sand = engine.add_taxonomy_node(
        label="Granular Sand Surface",
        modality=SensoryModality.TACTILE,
        substrate=substrate_sand,
        scale_level=0.1
    )

    assert node_sand.node_id in engine.nodes
    assert node_sand.modality == SensoryModality.TACTILE
    assert node_sand.parent_id == engine.modal_roots.get_root(SensoryModality.TACTILE).node_id


def test_substrate_unity_and_phase_transitions():
    engine = EcologicalTaxonomyEngine(target_dimension=8)

    # Create H2O Substrate Identities across 3 state phases
    h2o_ice = SubstrateIdentity(
        substrate_id="h2o_ice",
        chemical_or_logical_formula="H2O",
        core_invariants=["hydrogen_bonding", "dipole_moment", "molecular_weight_18"],
        current_phase_state="solid_ice",
        spectrum_axes={"temperature": SpectrumCoordinate("temperature", -10.0)}
    )

    h2o_water = SubstrateIdentity(
        substrate_id="h2o_water",
        chemical_or_logical_formula="H2O",
        core_invariants=["hydrogen_bonding", "dipole_moment", "molecular_weight_18"],
        current_phase_state="liquid_water",
        spectrum_axes={"temperature": SpectrumCoordinate("temperature", 25.0)}
    )

    h2o_steam = SubstrateIdentity(
        substrate_id="h2o_steam",
        chemical_or_logical_formula="H2O",
        core_invariants=["hydrogen_bonding", "dipole_moment", "molecular_weight_18"],
        current_phase_state="gaseous_steam",
        spectrum_axes={"temperature": SpectrumCoordinate("temperature", 120.0)}
    )

    disruptive_sub = SubstrateIdentity(
        substrate_id="n2_gas",
        chemical_or_logical_formula="N2",
        core_invariants=["triple_covalent_bond", "diatomic_nitrogen"],
        current_phase_state="gaseous"
    )

    # Evaluate H2O state transitions: Substrate Unity holds!
    is_same_phase1, msg1 = engine.substrate_engine.evaluate_state_transition(h2o_ice, h2o_water)
    assert is_same_phase1 is True
    assert "Substrate Unity Preserved" in msg1

    is_same_phase2, msg2 = engine.substrate_engine.evaluate_state_transition(h2o_water, h2o_steam)
    assert is_same_phase2 is True
    assert "Substrate Unity Preserved" in msg2

    # Disrupted substrate test
    is_same_disrupted, msg3 = engine.substrate_engine.evaluate_state_transition(h2o_water, disruptive_sub)
    assert is_same_disrupted is False
    assert "Substrate Disruption" in msg3


def test_scale_coupling_cascade_bifurcation():
    engine = EcologicalTaxonomyEngine(target_dimension=8)

    sub = SubstrateIdentity("sub_biological", "CELLULAR_MATRIX", ["membrane_integrity"], "normal")

    # Node A: Hair/Nail Shedding (Scale 0.1, low perturbation)
    node_hair = engine.add_taxonomy_node(
        label="Hair Shedding",
        modality=SensoryModality.TACTILE,
        substrate=sub,
        scale_level=0.1
    )

    res_hair = engine.process_incoming_stimulus(
        target_node_id=node_hair.node_id,
        stimulus_energy=0.2,
        stimulus_friction=0.1,
        direct_core_damage=False
    )

    assert res_hair["cascade"].cascade_mode == CascadeMode.LOCAL_DAMPED
    assert res_hair["cascade"].global_alarm_triggered is False
    assert res_hair["discernment"] == "LOCAL_BALANCED"

    # Node B: Blood/Bone/Organ Damage (Direct core damage or massive energy)
    node_organ = engine.add_taxonomy_node(
        label="Organ Vascular Fracture",
        modality=SensoryModality.TACTILE,
        substrate=sub,
        scale_level=1.0
    )

    res_organ = engine.process_incoming_stimulus(
        target_node_id=node_organ.node_id,
        stimulus_energy=5.0,
        stimulus_friction=0.9,
        direct_core_damage=True
    )

    assert res_organ["cascade"].cascade_mode == CascadeMode.CORE_BREACH
    assert res_organ["cascade"].global_alarm_triggered is True
    assert res_organ["discernment"] == "CORE_ALERT"


def test_topological_homeostasis_quarantine():
    engine = EcologicalTaxonomyEngine(target_dimension=8)
    sub = SubstrateIdentity("sub_cellular", "DNA_REPLICATION", ["genomic_stability"], "active")

    node_mutant = engine.add_taxonomy_node(
        label="Pathogenic Local Mutation",
        modality=SensoryModality.TACTILE,
        substrate=sub,
        scale_level=0.2
    )

    # Inject high friction stimulus repeatedly
    for _ in range(5):
        res = engine.process_incoming_stimulus(
            target_node_id=node_mutant.node_id,
            stimulus_energy=0.1,
            stimulus_friction=1.5
        )

    # Node should be quarantined locally by autonomous white blood cell/homeostatic defense
    assert node_mutant.quarantined is True
    assert node_mutant.node_id in engine.homeostasis_engine.quarantined_nodes


def test_cross_modal_isomorphic_association():
    engine = EcologicalTaxonomyEngine(target_dimension=8)

    # Physical tactile surface friction node
    phase_tactile = np.array([0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0], dtype=np.float32)
    sub_tactile = SubstrateIdentity("sub_friction", "SURFACE_FRICTION", ["turbulent_shear"], "physical")
    node_tactile = engine.add_taxonomy_node(
        label="Rough Sandpaper Friction",
        modality=SensoryModality.TACTILE,
        substrate=sub_tactile,
        scale_level=0.5,
        phase_vector=phase_tactile
    )

    # Abstract non-linear differential equation node (Symbolic modality)
    phase_symbolic = np.array([0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0], dtype=np.float32)
    sub_symbolic = SubstrateIdentity("sub_eqn", "NAVIER_STOKES_EQN", ["turbulent_shear"], "symbolic")
    node_symbolic = engine.add_taxonomy_node(
        label="Turbulence Differential Equation",
        modality=SensoryModality.SYMBOLIC,
        substrate=sub_symbolic,
        scale_level=1.5,
        phase_vector=phase_symbolic
    )

    beam = engine.associate_cross_modal_nodes(
        node_id_a=node_tactile.node_id,
        node_id_b=node_symbolic.node_id,
        invariants_a=["turbulent_shear", "non_linear_friction"],
        invariants_b=["turbulent_shear", "boundary_layer_dissipation"]
    )

    assert beam is not None
    assert beam.isomorphism_score >= 0.5
    assert "turbulent_shear" in beam.shared_topological_invariants
    assert beam.bidirectional is True


def test_causal_field_ecological_integration():
    cf = CausalField(dimensions=3)

    sub_h2o = SubstrateIdentity(
        substrate_id="h2o_liquid",
        chemical_or_logical_formula="H2O",
        core_invariants=["hydrogen_bond"],
        current_phase_state="liquid"
    )

    res = cf.process_ecological_stimulus(
        label="Water Flow Wave",
        modality=SensoryModality.TACTILE,
        substrate=sub_h2o,
        energy=1.2,
        friction=0.2,
        scale_level=0.5
    )

    assert "voxel_id" in res
    assert res["voxel_id"] in cf.voxels
    assert res["discernment"] == "LOCAL_BALANCED"
