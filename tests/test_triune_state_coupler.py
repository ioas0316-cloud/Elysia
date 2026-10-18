"""
Tests for Triune State Causal Coupler Module (`core/intelligence/triune_state_coupler.py`)

Verifies:
1. Solid State (Earth / Somatic Reach & Friction Boundary).
2. Fluid State (Mental Flow & Cognitive Gravity Warping).
3. Gas State (Sky / Mind Infinite Expansion & Salvation Derailment).
4. Sky-Earth Shadow Reverse Causal Inference.
5. Self-Structural Transformation Metrics vs Static Data Accumulation.
6. Mechanism Boundary Limit Recognition.
"""

import numpy as np
from core.intelligence.triune_state_coupler import (
    SolidStateBoundary,
    FluidStateMentalFlow,
    GasStateMindExpansion,
    TriuneEntity,
    SkyEarthReverseCausalInference
)


def test_solid_state_reach_and_friction():
    boundary = SolidStateBoundary(reach_radius=2.0, somatic_friction=0.6)

    # Within reach stimulus
    stim_near = np.array([0.5, 0.5, 0.5, 0.0])
    res_near = boundary.evaluate_reach_and_friction(stim_near)
    assert res_near["within_reach"] is True
    assert res_near["is_observed_reality"] is True
    assert res_near["somatic_friction"] > 0.0

    # Outside reach stimulus
    stim_far = np.array([5.0, 5.0, 5.0, 0.0])
    res_far = boundary.evaluate_reach_and_friction(stim_far)
    assert res_far["within_reach"] is False
    assert res_far["is_observed_reality"] is False

    # Saturated stimulus
    stim_sat = np.array([0.98, 0.1, 0.1, 0.0])
    res_sat = boundary.evaluate_reach_and_friction(stim_sat)
    assert res_sat["is_saturated"] is True
    assert res_sat["is_observed_reality"] is False


def test_fluid_mental_flow_warping():
    boundary = SolidStateBoundary(reach_radius=2.0, somatic_friction=0.5)
    fluid = FluidStateMentalFlow(boundary, deficit_amplitude=0.8) # High deficit

    benign_order = np.array([0.0, 1.0, 0.0, 0.0]) # Pure Order
    warped, breached, solid_info = fluid.process_fluid_flow(benign_order)

    # Under high deficit, Order should refract into Flux/Entropy
    assert fluid.cognitive_gravity > 0.0
    assert breached is True
    assert warped[2] > 0.0 # Entropy surge


def test_sky_earth_reverse_causal_inference():
    observer = TriuneEntity(
        name="Observer_Entity",
        reach_radius=2.0,
        somatic_friction=0.2,
        deficit_amplitude=0.1
    )
    target = TriuneEntity(
        name="Target_Entity",
        reach_radius=1.0,
        somatic_friction=0.8,
        deficit_amplitude=0.85
    )

    observed_shadow = np.array([1.2, 0.1, 0.9, np.pi / 3]) # Earth shadow output
    deconstruction = SkyEarthReverseCausalInference.deconstruct_shadow_manifestation(
        observer, target, observed_shadow
    )

    assert deconstruction["deconstruction_successful"] is True
    assert deconstruction["inferred_gravity"] > 0.0
    assert "extracted_sky_principle" in deconstruction
    assert deconstruction["empathetic_coherence"] > 0.0


def test_gas_state_salvation_derailment():
    entity = TriuneEntity(
        name="Test_Trapped_Agent",
        reach_radius=1.5,
        somatic_friction=0.5,
        deficit_amplitude=0.85
    )

    benign_stimulus = np.array([0.1, 0.8, 0.1, 0.0])
    traj_1 = entity.perceive_and_interact(benign_stimulus)
    assert traj_1["is_closed_loop"] is True

    # Low-energy interrupt (fails to derail)
    low_interrupt = np.array([0.05, 0.1, 0.0, 0.0])
    res_low = entity.gas.inject_salvation_interrupt(low_interrupt, intensity=0.5)
    assert res_low["success"] is False

    # High-energy salvation interrupt (exceeds escape velocity)
    v_esc = entity.gas.calculate_escape_velocity()
    high_interrupt = np.array([0.1, 0.9, 0.0, np.pi / 4])
    res_high = entity.gas.inject_salvation_interrupt(high_interrupt, intensity=v_esc * 1.8)

    assert res_high["success"] is True
    assert entity.gas.is_closed_loop is False
    assert entity.fluid.deficit_amplitude < 0.85

    # Check Self-Structural Transformation Metrics
    transform_metrics = entity.solid.evaluate_transformation_metrics()
    assert transform_metrics["structural_transformation_count"] > 0
    assert transform_metrics["transformation_ratio"] > 0.0


test_solid_state_reach_and_friction()
test_fluid_mental_flow_warping()
test_gas_state_salvation_derailment()
print("All triune state coupler tests passed successfully!")
