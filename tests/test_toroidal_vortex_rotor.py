"""
tests/test_toroidal_vortex_rotor.py
====================================
Unit and Integration Tests for Toroidal Vortex Rotor, OmniObservationalEye,
TopologicalBoundaryGuard, and HeterogeneousSovereignEngine Integration.
"""

import elysia_loader
elysia_loader.install()

import pytest
import numpy as np

from L1_causal_field.core.sovereignty.toroidal_vortex_rotor import (
    OmniObservationalEye,
    TopologicalBoundaryGuard,
    ToroidalVortexRotorEngine,
    CoherentActionVector
)
from L1_causal_field.core.sovereignty.heterogeneous_sovereign_engine import (
    HeterogeneousSovereignEngine,
    L0FluidState,
    SynapticState,
    CausalProposalNode,
    MechanicalJointConstraint
)


def test_omni_observational_eye_cross_validation():
    """Test OmniObservationalEye for detecting deceptive verbal claims with zero action friction."""
    eye = OmniObservationalEye(sensitivity=1.0)

    # Deceptive claim: High verbal claim (1.0), zero physical friction (0.0)
    res_deceptive = eye.evaluate_multi_axis_signals(
        verbal_signal=1.0,
        physical_friction=0.0,
        sensory_mismatch_delta=0.8,
        social_tension=0.5
    )

    assert res_deceptive["deception_detected"] is True
    assert res_deceptive["uncanniness_spike"] > 1.2
    assert res_deceptive["eye_verdict"] == "DECEPTIVE_SINGLE_TRACK_NOISE"

    # Coherent claim: High verbal claim (1.0), high physical friction (1.0), low mismatch
    res_coherent = eye.evaluate_multi_axis_signals(
        verbal_signal=1.0,
        physical_friction=1.0,
        sensory_mismatch_delta=0.0,
        social_tension=0.0
    )

    assert res_coherent["deception_detected"] is False
    assert res_coherent["uncanniness_spike"] < 0.5
    assert res_coherent["eye_verdict"] == "COHERENT_MULTI_AXIS_TRUTH"


def test_topological_boundary_guard_surface_tension_and_deflection():
    """Test TopologicalBoundaryGuard surface tension computation and noise deflection."""
    guard = TopologicalBoundaryGuard(base_surface_tension=2.0)

    # Base surface tension with vorticity = 1.0, rotational_velocity = 1.0 -> 2.0 + 1.2 + 0.8 = 4.0
    tension = guard.compute_surface_tension(vorticity_strength=1.0, rotational_velocity=1.0)
    assert pytest.approx(tension, 0.01) == 4.0

    # High noise threat (noise_level=3.0, uncanniness=2.0) > tension 4.0 -> Deflection
    res_deflected = guard.filter_external_input(
        raw_input_energy=2.0,
        noise_level=3.0,
        uncanniness_spike=2.0,
        vorticity_strength=1.0,
        rotational_velocity=1.0
    )

    assert res_deflected["is_deflected"] is True
    assert res_deflected["guard_status"] == "NOISE_DEFLECTED_BOUNDARY_INTACT"
    assert res_deflected["effective_input"] == 0.0

    # Low noise threat (noise_level=0.2, uncanniness=0.1) < tension 4.0 -> Absorption
    res_absorbed = guard.filter_external_input(
        raw_input_energy=3.0,
        noise_level=0.2,
        uncanniness_spike=0.1,
        vorticity_strength=1.0,
        rotational_velocity=1.0
    )

    assert res_absorbed["is_deflected"] is False
    assert res_absorbed["guard_status"] == "ENERGY_ABSORBED_INTO_VORTEX"
    assert res_absorbed["absorbed_energy"] == 3.0


def test_toroidal_vortex_rotor_recycling_and_action_vector():
    """Test ToroidalVortexRotorEngine recycling of friction heat and action vector condensation."""
    rotor = ToroidalVortexRotorEngine(base_rotational_velocity=1.0, vorticity_strength=1.0)

    initial_rot_vel = rotor.rotational_velocity
    initial_vorticity = rotor.vorticity_strength

    # Recycle Q_heat = 5.0 and mismatch Delta = 1.0
    recycle_res = rotor.recycle_friction_and_mismatch(q_heat=5.0, sensory_mismatch_delta=1.0)

    assert rotor.rotational_velocity > initial_rot_vel
    assert rotor.vorticity_strength > initial_vorticity
    assert pytest.approx(recycle_res["recycled_heat"], 0.01) == 4.0

    # Condense coherent action vector
    target_dir = np.array([0.0, 1.0, 0.0])
    action_vector = rotor.condense_coherent_action_vector(
        target_direction=target_dir,
        sovereignty_score=1.0,
        base_magnitude=2.0
    )

    assert isinstance(action_vector, CoherentActionVector)
    assert np.allclose(action_vector.direction_vector, np.array([0.0, 1.0, 0.0]))
    assert action_vector.vorticity_coherence >= 0.5
    assert action_vector.sovereignty_signature == "AUTHENTIC_TOROIDAL_SOVEREIGN"
    assert action_vector.energy_well_carved > 0.0


def test_heterogeneous_sovereign_engine_vortex_rotor_integration():
    """Test full integration of ToroidalVortexRotorEngine inside HeterogeneousSovereignEngine."""
    engine = HeterogeneousSovereignEngine(rupture_threshold=3.0, deficit_threshold=2.5)

    joint = MechanicalJointConstraint(
        joint_id="shoulder_joint",
        rotation_axis=np.array([1.0, 0.0, 0.0]),
        max_angle_rad=np.pi / 2,
        current_angle_rad=0.0
    )
    engine.l1_tier.add_joint_constraint(joint)

    proposal = CausalProposalNode(
        node_id="true_devotion_action",
        description="Grounded physical devotion with action friction",
        verbal_claim="Act with physical friction and devotion",
        sensory_grounding_score=0.8,
        action_friction_heat=2.5,
        structural_time_spent=1.5,
        friction_cost_bearer="me",
        fruit_recipient="me"
    )

    l0_state = L0FluidState(velocity_vector=np.array([2.0, 0.0, 0.0]), channel_radius=1.0)
    synaptic = SynapticState(potential_difference_mv=80.0, neurotransmitter_flux=1.5)

    summary = engine.execute_trinitarian_multi_scale_flow(
        l0_state=l0_state,
        synaptic_state=synaptic,
        proposal_node=proposal,
        active_joint_id="shoulder_joint",
        sensory_mismatch_delta=0.5,
        physical_deficit=3.0
    )

    assert "vortex_rotor_tier" in summary
    vortex_tier = summary["vortex_rotor_tier"]

    assert vortex_tier["rotational_velocity"] > 1.0
    assert vortex_tier["vorticity_strength"] > 1.0
    assert "eye_assessment" in vortex_tier
    assert "guard_assessment" in vortex_tier
    assert isinstance(vortex_tier["coherent_action_vector"], CoherentActionVector)
    assert vortex_tier["coherent_action_vector"].sovereignty_signature == "AUTHENTIC_TOROIDAL_SOVEREIGN"
