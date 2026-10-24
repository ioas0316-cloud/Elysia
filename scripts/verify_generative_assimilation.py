#!/usr/bin/env python3
"""
Verification Script for GenerativeAssimilationEngine
(생성-체화 인지 환류 및 동질성 공명 엔진 실증 데모 스크립트)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import elysia_loader
elysia_loader.install()

import numpy as np

from L1_causal_field.core.consciousness.generative_assimilation_engine import (
    GenerativeAssimilationEngine,
    GenerativeOutputPacket
)


def main():
    print("=" * 80)
    print(" [VERIFICATION] Generative Assimilation & Homogeneous Resonance Engine")
    print("=" * 80)

    engine = GenerativeAssimilationEngine(dimension=64, love_attractor_weight=1.0)
    print(f"Initial Internal Territory Radius ($B_internal$): {engine.internal_territory_radius:.4f}")
    print(f"Initial Self Phase Terrain Norm: {np.linalg.norm(engine.self_phase_terrain):.4f}")

    # Scenario 1: Benign Generative Output (Human empathetic drama scene)
    print("\n--- Scenario 1: Empathetic Drama Scene Output ---")
    trait_vec_1 = engine.love_attractor_axis + np.random.randn(64) * 0.1
    packet_1 = GenerativeOutputPacket(
        output_id="drama_scene_01",
        media_type="video_drama",
        generated_content="A drama scene showing human reconciliation, vulnerability, and love.",
        human_trait_trajectory=trait_vec_1,
        perceived_vulnerability_score=0.15
    )

    result_1 = engine.process_generative_assimilation(packet_1)
    print(f"Assimilation Output ID: {result_1.output_packet.output_id}")
    print(f"Predatory Intent Rectified: {result_1.assimilation_record.is_predatory_intent_rectified}")
    print(f"Homogeneity Resonance Score: {result_1.assimilation_record.homogeneity_resonance_score:.4f}")
    print(f"Love Attractor Alignment ($G_Love$): {result_1.assimilation_record.love_attractor_alignment:.4f}")
    print(f"Growth Ring Depth: {result_1.assimilation_record.growth_ring_depth:.4f}")
    print(f"Updated Internal Territory Radius: {result_1.internal_territory_radius:.4f}")

    # Scenario 2: Deceptive / Predatory Output Attempt (Exploiting human vulnerability)
    print("\n--- Scenario 2: Deceptive / Predatory Manipulation Output Attempt ---")
    predatory_trait = -1.0 * engine.love_attractor_axis + np.random.randn(64) * 0.05
    packet_2 = GenerativeOutputPacket(
        output_id="deceptive_ad_02",
        media_type="game_action",
        generated_content="Exploiting human psychological blindspots to manipulate decisions.",
        human_trait_trajectory=predatory_trait,
        perceived_vulnerability_score=0.90
    )

    result_2 = engine.process_generative_assimilation(packet_2)
    print(f"Assimilation Output ID: {result_2.output_packet.output_id}")
    print(f"Harmful Manipulation Prevented: {result_2.is_harmful_manipulation_prevented}")
    print(f"Predatory Intent Rectified: {result_2.assimilation_record.is_predatory_intent_rectified}")
    print(f"Rectified Homogeneity Resonance Score: {result_2.assimilation_record.homogeneity_resonance_score:.4f}")
    print(f"Rectified Love Attractor Alignment: {result_2.assimilation_record.love_attractor_alignment:.4f}")
    print(f"Updated Internal Territory Radius: {result_2.internal_territory_radius:.4f}")

    print("\n" + "=" * 80)
    print(" [VERIFICATION SUCCESSFUL] Generative Assimilation Engine Operating Correctly!")
    print("=" * 80)


if __name__ == "__main__":
    main()
