"""
Tests for Causal Narrative Scale & Cognitive Gravity Engine (core/intelligence/causal_narrative_scale_engine.py)
"""

import pytest
import numpy as np
from core.intelligence.causal_narrative_scale_engine import (
    NarrativeEntity,
    ReverseCausalInference,
    SalvationDerailment
)


def test_three_tier_scale_propagation_and_gravity():
    """
    Tests that Base Level trauma/deficit propagates to Mid Level cognitive gravity
    and distorts Top Level decisions.
    """
    # Create entity with heavy combat PTSD / trauma engram
    trauma_engram = np.array([0.9, 0.05, 0.85, np.pi / 2], dtype=np.float64)
    soldier = NarrativeEntity(
        name="Soldier_K",
        trauma_engram=trauma_engram,
        deficit_amplitude=0.8,
        baseline_threat_pinning=0.75
    )

    assert soldier.mid.cognitive_gravity > 1.0
    assert soldier.mid.event_horizon_radius > 0.2

    # Provide a benign, peaceful external stimulus [Red(Flux), Blue(Order), Yellow(Entropy), Phase]
    benign_stimulus = np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float64)

    decision = soldier.perceive_and_act(benign_stimulus)

    # Benign order should be refracted into flux/entropy under cognitive gravity
    assert decision["is_closed_loop"] is True
    assert decision["choice_mode"] in ["DEFICIT_DEFENSIVE_HYPERAROUSAL", "EVENT_HORIZON_PANIC"]
    assert decision["terminal_velocity"] > 0.5


def test_reverse_causal_inference_deconstruction():
    """
    Tests reverse causal de-construction where an observer perceives surface action,
    infers internal cognitive gravity, and reconstructs Base Level trauma root.
    """
    trauma_engram = np.array([0.8, 0.1, 0.7, np.pi / 4], dtype=np.float64)
    target = NarrativeEntity(
        name="Traumatized_Target",
        trauma_engram=trauma_engram,
        deficit_amplitude=0.7,
        baseline_threat_pinning=0.6
    )

    observer = NarrativeEntity(
        name="Empathic_Observer",
        trauma_engram=np.array([0.75, 0.15, 0.65, np.pi / 4], dtype=np.float64),
        deficit_amplitude=0.2,
        baseline_threat_pinning=0.1
    )

    # Target performs a surface hyperarousal action
    action = target.perceive_and_act(np.array([0.2, 0.8, 0.1, np.pi / 4]))["action_vector"]

    deconstruct_res = ReverseCausalInference.deconstruct_entity(observer, target, action)

    assert deconstruct_res["deconstruction_successful"] is True
    assert deconstruct_res["inferred_gravity"] > 0.0
    assert deconstruct_res["empathetic_phase_alignment"] > 0.5


def test_salvation_derailment_mechanism():
    """
    Tests salvation mechanism where a high-energy phase interrupt breaks closed loop,
    exceeds escape velocity, and derails entity from destructive trajectory.
    """
    target = NarrativeEntity(
        name="Despairing_Entity",
        trauma_engram=np.array([0.9, 0.0, 0.9, np.pi / 3]),
        deficit_amplitude=0.9,
        baseline_threat_pinning=0.85
    )

    # Attempt weak salvation interrupt (insufficient energy)
    weak_interrupt = np.array([0.1, 0.1, 0.0, np.pi / 3])
    weak_res = SalvationDerailment.attempt_salvation(target, weak_interrupt, interrupt_intensity=0.2)
    assert weak_res["success"] is False

    # Attempt powerful salvation interrupt exceeding Escape Velocity V_esc
    escape_vel = weak_res["escape_velocity"]
    strong_interrupt = np.array([0.5, 0.9, 0.1, np.pi / 3])
    strong_res = SalvationDerailment.attempt_salvation(target, strong_interrupt, interrupt_intensity=escape_vel * 1.5)

    assert strong_res["success"] is True
    assert target.base.deficit_amplitude < 0.5
    assert target.top.is_closed_loop is False


def test_self_structural_transformation_metrics():
    """
    Tests that somatic impacts drive Self-Structural Transformation (자기구조의 변형)
    rather than static data accumulation.
    """
    entity = NarrativeEntity(name="Entity_X", deficit_amplitude=0.5)

    # Absorb somatic impact
    sensory_wave = np.array([0.8, 0.2, 0.9, np.pi / 2], dtype=np.float64)
    entity.base.absorb_somatic_impact(sensory_wave, intensity=0.8)

    eval_res = entity.base.evaluate_self_structural_transformation()
    assert eval_res["accumulated_data_count"] == 1
    assert eval_res["structural_transformation_count"] >= 1
    assert eval_res["transformation_ratio"] > 0.0
    assert eval_res["last_structural_delta"] > 0.0


def test_mechanism_boundary_limit_awareness():
    """
    Tests detection of discrete scalar mechanism limits (saturation / RGB bounds).
    """
    entity = NarrativeEntity(name="Entity_Y")

    # Saturated scalar input
    saturated_input = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    boundary_info = entity.mid.detect_mechanism_boundary_limit(saturated_input)

    assert boundary_info["mechanism_limit_reached"] is True
    assert boundary_info["boundary_type"] == "DISCRETE_REPRESENTATION_SATURATION"

    # Normal input within bounds
    normal_input = np.array([0.3, 0.4, 0.1, 0.0], dtype=np.float64)
    normal_boundary_info = entity.mid.detect_mechanism_boundary_limit(normal_input)

    assert normal_boundary_info["mechanism_limit_reached"] is False
    assert normal_boundary_info["boundary_type"] == "WITHIN_MECHANISM_BOUNDS"
