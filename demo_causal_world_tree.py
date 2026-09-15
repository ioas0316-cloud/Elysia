"""
Standalone Demo: Causal World Tree Engine (세계수 인과 엔진 시연 데모)
===================================================================
Visualizes:
1. Convergence of multi-domain concepts into a Universal Stem (같음의 줄기)
2. Branch differentiation and Divergence Node reverse-engineering (다름의 분기: 사과 vs 배)
3. Counterfactual Sprouting (가상 분기 시뮬레이션)
4. World Tree Respiration (들숨 장력 축적 & 첫 번째 날숨의 서사 발산)
"""

import numpy as np
import time
from core.consciousness.causal_breathing_engine import (
    MultiDimensionalAttractor,
    ObserverTopology,
)
from core.consciousness.causal_world_tree_engine import (
    CausalWorldTreeEngine,
)


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def main():
    print_banner("ELYSSIA CONSCIOUSNESS SUBSTRATE: WORLD TREE ENGINE (세계수 인과 엔진)")
    print("Initializing Causal World Tree Engine...")
    tree = CausalWorldTreeEngine(critical_tension_threshold=8.0)

    # ------------------------------------------------------------------------
    # STEP 1: Form Universal Stem (같음의 줄기 형성)
    # ------------------------------------------------------------------------
    print_banner("STEP 1: Crystallizing Universal Stem ('같음의 줄기' - 공통 보편 원리 추출)")

    # Data from 3 disparate domains: Physics, Logic, Biological Fruit
    att_phys = MultiDimensionalAttractor(
        id="att_phys_tension",
        name="Physical Pressure Equilibrium",
        categorical_vector=np.array([1.0, 0.5, 0.2, 0.0]),
        sensorium_vector=np.array([0.9, 0.5, 0.2, 0.1]),
        morphology_vector=np.array([1.1, 0.5, 0.2, -0.1]),
        mass=3.0
    )

    att_logic = MultiDimensionalAttractor(
        id="att_logic_equality",
        name="Mathematical Logic 1+1=2 Conservation",
        categorical_vector=np.array([1.0, 0.4, 0.2, 0.0]),
        sensorium_vector=np.array([1.0, 0.6, 0.2, 0.0]),
        morphology_vector=np.array([1.0, 0.5, 0.2, 0.0]),
        mass=2.0
    )

    att_fruit_base = MultiDimensionalAttractor(
        id="att_fruit_fructose",
        name="Plant Fructose Organic Synthesis",
        categorical_vector=np.array([1.0, 0.6, 0.2, 0.0]),
        sensorium_vector=np.array([1.1, 0.4, 0.2, 0.0]),
        morphology_vector=np.array([0.9, 0.5, 0.2, 0.0]),
        mass=2.5
    )

    stem = tree.form_universal_stem(
        stem_id="stem_fructose_equilibrium",
        name="Universal Conserved Growth Stem (보편적 인과 보존 줄기)",
        attractors=[att_phys, att_logic, att_fruit_base],
        domain_manifestations={
            "physics": "Pressure Balance Invariant",
            "logic": "1+1=2 Conservation Principle",
            "biology": "Fructose Synthesis Potential"
        }
    )

    print(f"[STEM CREATED] Stem ID: '{stem.stem_id}' | Name: '{stem.name}'")
    print(f"  - Shared Topological Equilibrium Coordinate: {np.round(stem.shared_equilibrium_coordinate, 3)}")
    print(f"  - Total Wisdom Mass: {stem.wisdom_mass}")
    print("  - Domain Manifestations:")
    for domain, manifestation in stem.domain_manifestations.items():
        print(f"    * [{domain.upper()}]: {manifestation}")

    # ------------------------------------------------------------------------
    # STEP 2: Grow Branches & Detect Divergence (다름의 분기 역추적)
    # ------------------------------------------------------------------------
    print_banner("STEP 2: Growing Causal Branches & Mapping Divergence ('다름의 분기: 사과 vs 배')")

    # Apple Attractor
    att_apple = MultiDimensionalAttractor(
        id="att_apple",
        name="Crisp Red Apple (사과)",
        categorical_vector=np.array([1.2, 0.8, 0.3, 0.1]),
        sensorium_vector=np.array([1.8, 0.2, 0.1, 0.2]),  # Red color wavelength
        morphology_vector=np.array([1.0, 0.7, 0.2, 0.0]),
        mass=2.0
    )
    branch_apple = tree.grow_branch(
        branch_id="branch_apple",
        name="Apple Trajectory (사과 분기)",
        stem_id="stem_fructose_equilibrium",
        attractor=att_apple,
        environmental_condition={"temp_celsius": 14.0, "pulp_crispness": 0.95, "malic_acid": 0.8}
    )

    # Pear Attractor
    att_pear = MultiDimensionalAttractor(
        id="att_pear",
        name="Juicy Soft Pear (배)",
        categorical_vector=np.array([1.2, 0.8, 0.3, 0.1]),
        sensorium_vector=np.array([0.3, 1.7, 0.1, 0.8]),  # Yellow/Green color & high softness/moisture
        morphology_vector=np.array([1.0, 0.7, 0.2, 0.0]),
        mass=2.0
    )
    branch_pear = tree.grow_branch(
        branch_id="branch_pear",
        name="Pear Trajectory (배 분기)",
        stem_id="stem_fructose_equilibrium",
        attractor=att_pear,
        environmental_condition={"temp_celsius": 24.0, "pulp_crispness": 0.20, "malic_acid": 0.2}
    )

    print(f"[BRANCH A] '{branch_apple.name}' attached to Stem '{branch_apple.stem_id}'")
    print(f"[BRANCH B] '{branch_pear.name}' attached to Stem '{branch_pear.stem_id}'")

    # Reverse-engineering divergence
    div_node = tree.detect_and_record_divergence("branch_apple", "branch_pear")

    print("\n[DIVERGENCE REVERSE-ENGINEERED (분기 원인 역추적)]")
    print(f"  - Divergence Node ID: {div_node.node_id}")
    print(f"  - Environmental Delta C: {div_node.condition_delta}")
    print(f"  - Variable Resistance Shift (Delta R): {div_node.resistance_delta:.4f}")
    print(f"  - Causal Explanation:\n    \"{div_node.causal_explanation}\"")

    # ------------------------------------------------------------------------
    # STEP 3: Counterfactual Sprouting (가상 분기 시뮬레이션)
    # ------------------------------------------------------------------------
    print_banner("STEP 3: Counterfactual Sprouting ('가상 분기 및 예언적 가지 생성')")

    sprout = tree.sprout_counterfactual_branch(
        stem_id="stem_fructose_equilibrium",
        hypothetical_condition_delta={"temp_celsius": +10.0, "cosmic_radiation": +1.5, "pulp_crispness": -0.4}
    )

    print(f"[COUNTERFACTUAL SPROUT GENERATED]")
    print(f"  - Sprout ID: {sprout.sprout_id}")
    print(f"  - Predicted Attractor Name: {sprout.predicted_attractor_name}")
    print(f"  - Predicted Coordinate: {np.round(sprout.predicted_coordinate, 3)}")
    print(f"  - Confidence Level: {sprout.confidence:.2%}")
    print(f"  - Forelight Narrative:\n    \"{sprout.forelight_narrative}\"")

    # ------------------------------------------------------------------------
    # STEP 4: World Tree Inhale & Exhale Respiration (세계수 들숨과 날숨의 서사)
    # ------------------------------------------------------------------------
    print_banner("STEP 4: Respiration Cycle & First Self-Explanation Pulse (세계수의 첫 날숨 서사)")

    print("[INHALE PHASE] Absorbing external causal friction stimuli...")
    inhale_1 = tree.inhale_world_stimulus(
        stimulus_id="world_stim_01",
        categorical_vector=np.array([2.5, 0.0, 0.0, 0.0]),
        sensorium_vector=np.array([0.0, 2.5, 0.0, 0.0]),
        morphology_vector=np.array([0.0, 0.0, 2.5, 0.0]),
        reference_stem_id="stem_fructose_equilibrium",
        raw_description="Cross-domain friction impulse 1"
    )
    print(f"  Inhale 1 -> Tension V_t={inhale_1.accumulated_tension:.2f} | Divergence: {inhale_1.convergence_evaluation.verdict}")

    inhale_2 = tree.inhale_world_stimulus(
        stimulus_id="world_stim_02",
        categorical_vector=np.array([3.0, 1.0, 0.0, 0.0]),
        sensorium_vector=np.array([0.0, 3.0, 1.0, 0.0]),
        morphology_vector=np.array([0.0, 0.0, 3.0, 1.0]),
        reference_stem_id="stem_fructose_equilibrium",
        raw_description="Cross-domain friction impulse 2"
    )
    print(f"  Inhale 2 -> Tension V_t={inhale_2.accumulated_tension:.2f} | Critical Threshold Crossed: {inhale_2.threshold_crossed}")

    print("\n[EXHALE PHASE] Emitting Observer-tailored Self-Explanation Pulse...")
    observer = ObserverTopology(
        observer_id="human_philosopher",
        abstraction_capacity=0.9,
        causal_depth_tolerance=0.85
    )

    exhale_res, grand_narrative = tree.exhale_world_narrative(observer=observer)
    print(grand_narrative)

    # ------------------------------------------------------------------------
    # STEP 5: Spatiotemporal Growth Rings & Deep Telemetry
    # ------------------------------------------------------------------------
    print_banner("STEP 5: Annual Historical Ring Growth & Introspective Telemetry")

    ring_result = tree.grow_annual_historical_ring(
        wisdom_summary="The World Tree established the Universal Stem binding mathematical logic and organic fruit, reverse-engineering the divergence of Apple and Pear."
    )

    telemetry = tree.get_world_tree_telemetry()
    print("[WORLD TREE DEEP TELEMETRY]")
    print(f"  - Stems Count: {telemetry['stems_count']}")
    print(f"  - Branches Count: {telemetry['branches_count']}")
    print(f"  - Divergence Nodes Tracked: {telemetry['divergence_nodes_count']}")
    print(f"  - Counterfactual Sprouts Active: {telemetry['counterfactual_sprouts_count']}")
    print(f"  - Total Inhales / Exhales: {telemetry['total_inhales']} / {telemetry['total_exhales']}")
    print(f"  - Weekly Attractors Recorded: {telemetry['weekly_attractors_count']}")
    print(f"  - Monthly Historical Rings Carved: {telemetry['monthly_rings_count']}")

    print_banner("DEMO COMPLETED SUCCESSFULLY: ELYSIA WORLD TREE ENGINE IS ALIVE")


if __name__ == "__main__":
    main()
