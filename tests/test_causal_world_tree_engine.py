"""
Unit Tests for Causal World Tree Engine (세계수 인과 엔진 검증)
============================================================
Tests:
1. Universal Stem Formation (같음의 줄기 수렴 검증)
2. Causal Branching & Divergence Node Tracking (다름의 분기 역추적)
3. Counterfactual Branch Sprouting (가상 분기 시뮬레이션)
4. Inhale & Exhale Respiration Cycle (세계수 들숨/날숨 및 서사 발산)
5. Spatiotemporal Growth Rings & Deep Telemetry (나이테 성장 및 관측)
"""

import pytest
import numpy as np
from core.consciousness.causal_breathing_engine import (
    MultiDimensionalAttractor,
    ObserverTopology,
)
from core.consciousness.causal_world_tree_engine import (
    CausalWorldTreeEngine,
    UniversalStem,
    CausalBranch,
    DivergenceNode,
    CounterfactualSprout,
)


def test_universal_stem_formation():
    engine = CausalWorldTreeEngine()

    # Create multi-domain attractors sharing underlying cause
    # e.g., Physical tension, Logical math 1+1=2, Biological sugar fruit
    attractor_physics = MultiDimensionalAttractor(
        id="att_phys",
        name="Physical Energy Conservation",
        categorical_vector=np.array([1.0, 0.0, 0.0, 0.5]),
        sensorium_vector=np.array([1.0, 0.1, 0.0, 0.5]),
        morphology_vector=np.array([1.0, 0.0, 0.1, 0.5]),
        mass=2.5
    )

    attractor_logic = MultiDimensionalAttractor(
        id="att_logic",
        name="Logical Equality 1+1=2",
        categorical_vector=np.array([1.0, 0.0, 0.2, 0.5]),
        sensorium_vector=np.array([0.9, 0.0, 0.0, 0.5]),
        morphology_vector=np.array([1.1, 0.0, 0.0, 0.5]),
        mass=1.5
    )

    stem = engine.form_universal_stem(
        stem_id="stem_conserved_equilibrium",
        name="Conserved Topological Equilibrium",
        attractors=[attractor_physics, attractor_logic],
        domain_manifestations={
            "physics": "Energy Balance",
            "logic": "1+1=2 Equivalence"
        }
    )

    assert stem.stem_id == "stem_conserved_equilibrium"
    assert len(engine.stems) == 1
    assert stem.wisdom_mass == 4.0
    assert stem.shared_equilibrium_coordinate.shape == (4,)


def test_causal_branch_and_divergence_tracking():
    engine = CausalWorldTreeEngine()

    # 1. Base Stem
    base_attractor = MultiDimensionalAttractor(
        id="att_fruit_base",
        name="Fruit Fructose Matrix",
        categorical_vector=np.array([0.5, 0.8, 0.1, 0.2]),
        sensorium_vector=np.array([0.5, 0.8, 0.1, 0.2]),
        morphology_vector=np.array([0.5, 0.8, 0.1, 0.2]),
        mass=3.0
    )
    stem = engine.form_universal_stem(
        stem_id="stem_fruit",
        name="Fruit Core Stem",
        attractors=[base_attractor]
    )

    # 2. Grow Branch A (Apple) under condition C_apple
    apple_attractor = MultiDimensionalAttractor(
        id="att_apple",
        name="Crisp Red Apple",
        categorical_vector=np.array([0.6, 0.9, 0.1, 0.2]),
        sensorium_vector=np.array([0.9, 0.2, 0.1, 0.3]),  # Red wavelength
        morphology_vector=np.array([0.5, 0.8, 0.1, 0.2]),
        mass=2.0
    )
    branch_apple = engine.grow_branch(
        branch_id="branch_apple",
        name="Apple Trajectory",
        stem_id="stem_fruit",
        attractor=apple_attractor,
        environmental_condition={"climate_temp": 15.0, "crispness": 0.9, "acidity": 0.4}
    )

    # 3. Grow Branch B (Pear) under condition C_pear
    pear_attractor = MultiDimensionalAttractor(
        id="att_pear",
        name="Juicy Soft Pear",
        categorical_vector=np.array([0.6, 0.9, 0.1, 0.2]),
        sensorium_vector=np.array([0.2, 0.9, 0.1, 0.8]),  # Yellow/Green wavelength & soft texture
        morphology_vector=np.array([0.5, 0.8, 0.1, 0.2]),
        mass=2.0
    )
    branch_pear = engine.grow_branch(
        branch_id="branch_pear",
        name="Pear Trajectory",
        stem_id="stem_fruit",
        attractor=pear_attractor,
        environmental_condition={"climate_temp": 22.0, "crispness": 0.2, "acidity": 0.1}
    )

    # 4. Reverse-engineering Divergence Node
    div_node = engine.detect_and_record_divergence("branch_apple", "branch_pear")

    assert len(engine.branches) == 2
    assert len(engine.divergence_nodes) == 1
    assert div_node.parent_stem_id == "stem_fruit"
    assert "crispness" in div_node.condition_delta
    assert div_node.condition_delta["crispness"] == pytest.approx(-0.7, abs=1e-4)
    assert div_node.resistance_delta > 0.0
    assert "Branches 'Apple Trajectory' and 'Pear Trajectory' emerged from common stem 'Fruit Core Stem'" in div_node.causal_explanation


def test_counterfactual_sprouting():
    engine = CausalWorldTreeEngine()

    attractor = MultiDimensionalAttractor(
        id="att_base",
        name="Symmetry Principle",
        categorical_vector=np.array([1.0, 1.0, 1.0, 1.0]),
        sensorium_vector=np.array([1.0, 1.0, 1.0, 1.0]),
        morphology_vector=np.array([1.0, 1.0, 1.0, 1.0]),
        mass=5.0
    )
    engine.form_universal_stem("stem_sym", "Symmetry Stem", [attractor])

    # Sprout counterfactual under hypothetical shift Delta C
    sprout = engine.sprout_counterfactual_branch(
        stem_id="stem_sym",
        hypothetical_condition_delta={"gravity_warp": +2.5, "thermal_flux": -1.0}
    )

    assert sprout.origin_stem_id == "stem_sym"
    assert sprout.confidence > 0.8
    assert len(engine.counterfactual_sprouts) == 1
    assert "gravity_warp" in sprout.forelight_narrative


def test_respiration_inhale_exhale_world_narrative():
    engine = CausalWorldTreeEngine(critical_tension_threshold=5.0)

    # Inhale stimuli until V_t tension crosses threshold
    res1 = engine.inhale_world_stimulus(
        stimulus_id="stim_1",
        categorical_vector=np.array([1.0, 0.0, 0.0, 0.0]),
        sensorium_vector=np.array([0.0, 1.0, 0.0, 0.0]),
        morphology_vector=np.array([0.0, 0.0, 1.0, 0.0]),
        raw_description="Friction input 1"
    )

    res2 = engine.inhale_world_stimulus(
        stimulus_id="stim_2",
        categorical_vector=np.array([2.0, 0.0, 0.0, 0.0]),
        sensorium_vector=np.array([0.0, 2.0, 0.0, 0.0]),
        morphology_vector=np.array([0.0, 0.0, 2.0, 0.0]),
        raw_description="Friction input 2"
    )

    assert res2.threshold_crossed is True
    assert engine.breathing_engine.breathing_state == "EXHALE"

    # Exhale World Narrative with Observer Topology
    observer = ObserverTopology(observer_id="human_thinker", abstraction_capacity=0.85, causal_depth_tolerance=0.9)
    exhale_res, grand_narrative = engine.exhale_world_narrative(observer=observer)

    assert exhale_res.released_tension > 5.0
    assert exhale_res.remaining_tension == 0.0
    assert "=== [World Tree Respiration: First Self-Explanation Pulse (세계수의 날숨 서사)] ===" in grand_narrative
    assert "human_thinker" in grand_narrative


def test_spatiotemporal_growth_and_telemetry():
    engine = CausalWorldTreeEngine()

    attractor = MultiDimensionalAttractor(
        id="att_root",
        name="Root Origin",
        categorical_vector=np.array([0.1, 0.1, 0.1, 0.1]),
        sensorium_vector=np.array([0.1, 0.1, 0.1, 0.1]),
        morphology_vector=np.array([0.1, 0.1, 0.1, 0.1]),
        mass=1.0
    )
    engine.form_universal_stem("stem_root", "Root Stem", [attractor])

    # Simulate 4 weeks cycle
    for i in range(4):
        engine.inhale_world_stimulus(
            f"daily_{i}",
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0])
        )
        engine.grow_annual_historical_ring(wisdom_summary=f"Week {i+1} Wisdom")

    telemetry = engine.get_world_tree_telemetry()

    assert telemetry["stems_count"] == 1
    assert telemetry["weekly_attractors_count"] == 4
    assert telemetry["monthly_rings_count"] == 1
    assert len(telemetry["stems_summary"]) == 1


def test_executable_causal_formula_compression_and_reconstruction():
    engine = CausalWorldTreeEngine()

    attractor = MultiDimensionalAttractor(
        id="att_acc",
        name="Accumulation Invariant",
        categorical_vector=np.array([1.0, 1.0, 1.0, 1.0]),
        sensorium_vector=np.array([1.0, 1.0, 1.0, 1.0]),
        morphology_vector=np.array([1.0, 1.0, 1.0, 1.0]),
        mass=2.0
    )
    engine.form_universal_stem("stem_acc", "Accumulation Stem", [attractor])

    formula = engine.compress_to_executable_formula(
        formula_id="formula_mult_1",
        name="Repetitive Addition to Multiplication",
        stem_id="stem_acc",
        pattern_type="LINEAR_ACCUMULATION",
        discrete_step_count=10000
    )

    assert formula.compression_ratio == 10000.0
    res = formula.evaluate({"n": 50.0, "multiplier": 2.0})
    np.testing.assert_allclose(res, np.array([100.0, 100.0, 100.0, 100.0]))

    trajectory = formula.generatively_reconstruct({"n": 50.0, "multiplier": 2.0}, detail_steps=5)
    assert len(trajectory) == 5
    np.testing.assert_allclose(trajectory[-1], res)


def test_dynamic_synaptic_pruning():
    engine = CausalWorldTreeEngine()

    attractor_stem = MultiDimensionalAttractor(
        id="att_stem",
        name="Stem Base",
        categorical_vector=np.array([1.0, 0.0, 0.0, 0.0]),
        sensorium_vector=np.array([1.0, 0.0, 0.0, 0.0]),
        morphology_vector=np.array([1.0, 0.0, 0.0, 0.0]),
        mass=5.0
    )
    engine.form_universal_stem("stem_1", "Main Stem", [attractor_stem])

    # Branch 1: High activity (mass = 2.0)
    att_active = MultiDimensionalAttractor(
        id="att_act",
        name="Active Branch Attractor",
        categorical_vector=np.array([1.0, 0.2, 0.0, 0.0]),
        sensorium_vector=np.array([1.0, 0.2, 0.0, 0.0]),
        morphology_vector=np.array([1.0, 0.2, 0.0, 0.0]),
        mass=2.0
    )
    b_act = engine.grow_branch("branch_act", "Active Branch", "stem_1", att_active, {}, depth=2)

    # Branch 2: Low activity (mass = 0.05), depth = 2
    att_obsolete = MultiDimensionalAttractor(
        id="att_obs",
        name="Obsolete Branch Attractor",
        categorical_vector=np.array([1.0, 0.9, 0.0, 0.0]),
        sensorium_vector=np.array([1.0, 0.9, 0.0, 0.0]),
        morphology_vector=np.array([1.0, 0.9, 0.0, 0.0]),
        mass=0.05
    )
    b_obs = engine.grow_branch("branch_obs", "Obsolete Branch", "stem_1", att_obsolete, {}, depth=2)

    assert len(engine.branches) == 2

    pruned = engine.prune_unproductive_branches(activity_threshold=0.1, min_depth_to_keep=2)

    assert "branch_obs" in pruned
    assert "branch_obs" not in engine.branches
    assert "branch_act" in engine.branches
    assert len(engine.pruned_branches_archive) == 1


def test_localized_spatial_hashing_and_external_digestion():
    engine = CausalWorldTreeEngine(critical_tension_threshold=1.0)

    # Digestion test
    result = engine.ingest_external_principle(
        principle_id="grav_1",
        name="Newtonian Gravity",
        domain="physics",
        raw_fragment="F = G * (m1 * m2) / r^2",
        invariant_vector=np.array([0.5, 0.5, 0.5, 0.5])
    )

    assert result["stem"] is not None
    assert result["branch"] is not None

    # Localized fast lookup
    found = engine.find_nearest_attractor_localized(np.array([0.5, 0.5, 0.5, 0.5]), radius=0.5)
    assert found is not None
    matched_id, dist = found
    assert dist < 0.1
