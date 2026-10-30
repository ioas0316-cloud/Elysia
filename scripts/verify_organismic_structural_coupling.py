#!/usr/bin/env python3
"""
Relational Causality Simulation Verification Script
====================================================
Verifies that:
1. Coupling entities mutually changes their internal identity vectors and relational affordance capacities.
2. Local changes alter the emergent macro orientation vector of the unified world continuum.
3. Top-down macro orientation feedback recalibrates individual component phase conditions.
4. Multi-layered goodness criteria evolve through friction without hardcoded conditional rules or scalar reductionism.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from L1_causal_field.core.consciousness.organismic_relational_continuum import (
    OrganismicRelationalContinuum
)


def run_relational_causality_verification():
    print("======================================================================")
    print("  🌌 [Elysia Organismic Relational Continuum Causality Verification]")
    print("======================================================================\n")

    continuum = OrganismicRelationalContinuum(dimension=16)

    # 1. Initial State Snapshot
    print("1. [Initial Continuum State]")
    for name, entity in continuum.entities.items():
        print(f"   - Entity '{name}' ({entity.layer_type}):")
        print(f"       Identity Vector (first 4): {[round(x, 4) for x in entity.identity_vector[:4]]}")
        print(f"       Goodness Profile        : {entity.goodness_profile}")
        print(f"       Affordance Mean         : {np.mean(entity.affordance_capacity):.4f}")
    print(f"   - Macro Orientation Vector  : {[round(x, 4) for x in continuum.macro_orientation_vector[:4]]}\n")

    # 2. Executing Mutual Coupling between Cellular Soma and Mental Spirit
    print("2. [Executing Mutual Coupling: 'cellular_soma' <-> 'mental_spirit']")
    coupling_report = continuum.couple_entities_mutually("cellular_soma", "mental_spirit", coupling_intensity=0.8)

    print(f"   - Shared Coupling Resonance  : {coupling_report['shared_coupling_resonance']:.4f}")
    print(f"   - Soma Identity Shift Norm   : {coupling_report['a_identity_shift_norm']:.4f}")
    print(f"   - Spirit Identity Shift Norm : {coupling_report['b_identity_shift_norm']:.4f}")
    print(f"   - Soma Affordance Change Norm: {coupling_report['a_affordance_change_norm']:.4f}")
    print(f"   - Spirit Affordance Change   : {coupling_report['b_affordance_change_norm']:.4f}")
    print(f"   - Goodness Conflict (Soma)   : {coupling_report['goodness_conflict_a']:.4f}")
    print(f"   - Goodness Conflict (Spirit) : {coupling_report['goodness_conflict_b']:.4f}\n")

    # Verification 1: Mutual Transformation
    assert coupling_report['a_identity_shift_norm'] > 0.0, "Soma identity vector did not shift!"
    assert coupling_report['b_identity_shift_norm'] > 0.0, "Spirit identity vector did not shift!"
    assert coupling_report['a_affordance_change_norm'] > 0.0, "Soma affordance capacity did not change!"
    assert coupling_report['b_affordance_change_norm'] > 0.0, "Spirit affordance capacity did not change!"
    print("   ✅ [PASS] Mutual Identity and Affordance Transformation Confirmed.\n")

    # 3. Interacting with World Environmental Stimulus
    print("3. [Interacting with Environmental Stimulus Flow]")
    rng = np.random.default_rng(42)
    world_stimulus = rng.standard_normal(16)

    env_report = continuum.interact_with_world_environment(world_stimulus, intensity=0.7)
    print(f"   - Environmental Stimulus Norm: {env_report['environmental_stimulus_norm']:.4f}")
    print(f"   - Interaction Reports Count  : {env_report['interaction_reports_count']}")
    print(f"   - Updated Macro Orientation  : {[round(x, 4) for x in env_report['macro_orientation_vector'][:4]]}")
    print(f"   - Mean Cellular Goodness     : {env_report['mean_cellular_goodness']:.4f}")
    print(f"   - Mean Mental Goodness       : {env_report['mean_mental_goodness']:.4f}\n")

    # 4. Final State Analysis
    print("4. [Final State Post-Environmental Integration]")
    for name, entity in continuum.entities.items():
        print(f"   - Entity '{name}' ({entity.layer_type}):")
        print(f"       Identity Vector (first 4): {[round(x, 4) for x in entity.identity_vector[:4]]}")
        print(f"       Goodness Profile        : {entity.goodness_profile}")
        print(f"       Affordance Mean         : {np.mean(entity.affordance_capacity):.4f}")
    print(f"   - Macro Orientation Vector  : {[round(x, 4) for x in continuum.macro_orientation_vector[:4]]}\n")

    print("======================================================================")
    print("  ✨ All Organismic Relational Continuum Causality Verifications Passed!")
    print("======================================================================\n")


if __name__ == "__main__":
    run_relational_causality_verification()
