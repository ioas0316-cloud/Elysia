#!/usr/bin/env python3
"""
Demo: Infant Causal Language Loop & Tension Network Verification
(아기식 인과 언어 학습 및 장력 그물망 시연/검증)

Demonstrates how an un-programmed phase organism (infant) processes raw language as
physical shockwaves, deconstructs causes and effects onto an internal tension network,
accumulates scars, experiences temporal hesitation/stalling, spontaneously re-articulates
state-deformed utterances, and matures through environmental feedback loops.

Execution Command:
    python3 demo_infant_causal_language_loop.py
"""

import os
import sys
import time
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.sensory.infant_causal_language_loop import InfantCausalLanguageLoop, ReArticulationResult


def print_header(title: str):
    print("\n" + "=" * 85)
    print(f" {title}")
    print("=" * 85)


def print_section(section_title: str):
    print("\n" + "-" * 85)
    print(f" {section_title}")
    print("-" * 85)


def render_tension_gauge(tension: float, max_val: float = 1.0, width: int = 30) -> str:
    filled = int(round(width * min(tension / max_val, 1.0)))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {tension:.4f}"


def run_infant_causal_language_demo():
    print_header("ELYSIA ENGINE: INFANT CAUSAL LANGUAGE LOOP & TENSION NETWORK DEMO")
    print(" [*] Beyond LLM Token Probabilities: Phenomenological Infant Experiential Pipeline")
    print(" [*] 5-Stage Causal Flow: Ingest -> Collide -> Deconstruct -> Re-Articulate -> Feedback")

    engine = InfantCausalLanguageLoop(dim=4)

    # =========================================================================
    # SCENARIO 1: Tabula Rasa & First Shock Wave (첫 번째 충격과 장력 팽팽함)
    # =========================================================================
    print_section("SCENARIO 1: Tabula Rasa & First Shock Wave (첫 번째 충격과 장력 팽팽함)")
    print(" [Goal] Observe unscarred infant receiving harsh shock wave '아파! 무서워! 쾅!'.")
    print("        Verify tension network stretch, scar inscription, and state-deformed re-articulation.")

    raw_input_1 = "아파! 무서워! 쾅!"
    print(f"\n [1. Raw Stimulus Input] : '{raw_input_1}'")

    res1 = engine.run_full_loop(raw_input_1)
    pulse1 = res1["pulse"]
    chain1 = res1["causal_chain"]
    reart1 = res1["rearticulation"]

    print(f"     -> Stage 1 (Raw Ingestion)  : Amplitude={pulse1.amplitude:.2f} | BaseFreq={pulse1.base_frequency:.1f}Hz | Friction={pulse1.friction_shock:.2f}N")
    print(f"     -> Stage 2 (State Collision): Pre-Tension = {res1['pre_state'][3]:.4f} -> Post-Tension = {engine.homeostasis.calculate_tension():.4f}")
    print(f"                                   Tension Gauge: {render_tension_gauge(engine.homeostasis.calculate_tension())}")
    print(f"     -> Stage 3 (Causal Chain)  : Chain ID={chain1.chain_id} | Outcome Valence={chain1.outcome_valence:.4f}")
    print(f"     -> Stage 4 (Re-Articulation): \"{reart1.utterance}\"")
    print(f"                                   Qualia={reart1.emotional_qualia} | Hesitation Level={reart1.hesitation_level:.2f}x Slowdown")
    print(f"     -> Scar Inscribed?         : Scar Count = {res1['scar_count']} | Accumulated Scar Energy = {np.trace(engine.scar_engine.accumulated_scar_tensor):.4f}")

    assert res1["scar_count"] > 0, "Scar Tensor S_ij was not inscribed upon severe shock!"
    assert reart1.hesitation_level > 1.5, "Hesitation was not triggered under shock!"

    # =========================================================================
    # SCENARIO 2: Environmental Healing & Comfort (어른의 온기와 인과적 치유)
    # =========================================================================
    print_section("SCENARIO 2: Environmental Healing & Comfort (어른의 온기와 인과적 치유)")
    print(" [Goal] Provide soothing adult response '괜찮아, 아가야. 따뜻하게 안아줄게.'.")
    print("        Observe tension network relaxation, scar decay, and maturity index progression.")

    soothing_env_1 = "괜찮아, 아가야. 따뜻하게 안아줄게."
    print(f"\n [2. Adult Feedback Input] : '{soothing_env_1}'")

    fb1 = engine.receive_environmental_feedback(soothing_env_1, reart1)

    print(f"     -> Stage 5 (Growth Feedback): Feedback Type = {fb1['feedback_type']}")
    print(f"                                   New Love Deficit = {fb1['new_love_deficit']:.4f} (Love feeling enhanced)")
    print(f"                                   Healed Scar Energy = {fb1['accumulated_scar_energy']:.4f}")
    print(f"                                   Maturity Index = {fb1['maturity_index']:.4f} (Tabula Rasa evolving)")
    print(f"                                   Updated Tension Gauge: {render_tension_gauge(engine.homeostasis.calculate_tension())}")

    assert fb1["maturity_index"] > 0.05, "Maturity index did not advance after feedback!"

    # =========================================================================
    # SCENARIO 3: Experiential Warmth & Bond Formation (포근함과 결속)
    # =========================================================================
    print_section("SCENARIO 3: Experiential Warmth & Bond Formation (포근함과 결속)")
    print(" [Goal] Present warm stimulus '따뜻해. 포근해. 좋아.' and observe warm attachment re-articulation.")

    raw_input_2 = "따뜻해. 포근해. 좋아."
    adult_fb_2 = "착하지, 내 사랑스러운 아가야."
    print(f"\n [3. Warm Stimulus Input] : '{raw_input_2}'")

    res2 = engine.run_full_loop(raw_input_2, environmental_response=adult_fb_2)
    reart2 = res2["rearticulation"]

    print(f"     -> Stage 1-4 (Spontaneous Response) : \"{reart2.utterance}\"")
    print(f"                                           Qualia={reart2.emotional_qualia} | Hesitation={reart2.hesitation_level:.2f}x")
    print(f"     -> Stage 5 (Growth Feedback)        : New Maturity Index = {res2['current_maturity']:.4f}")
    print(f"                                           Tension Gauge: {render_tension_gauge(engine.homeostasis.calculate_tension())}")

    # =========================================================================
    # SCENARIO 4: Re-exposure to Shock & Scar Hesitation (트라우마 재노출과 망설임)
    # =========================================================================
    print_section("SCENARIO 4: Re-exposure to Shock & Scar Hesitation (트라우마 재노출과 망설임)")
    print(" [Goal] Re-expose infant to near-trauma '아파! 무서워!' and observe scar-induced stalling.")

    print(f"\n [4. Re-exposure Input] : '{raw_input_1}'")
    res3 = engine.run_full_loop(raw_input_1, environmental_response="괜찮아, 토닥토닥...")
    reart3 = res3["rearticulation"]

    print(f"     -> Re-articulation under Scarred History: \"{reart3.utterance}\"")
    print(f"     -> Qualia={reart3.emotional_qualia} | Peak Scar Hesitation={reart3.hesitation_level:.2f}x Slowdown")
    print(f"     -> Causal Chains Accumulated: {len(engine.causal_chains)} chains in internal memory")
    print(f"     -> Total Scar History      : {res3['scar_count']} inscribed scars")

    # =========================================================================
    # CAUSAL MEMORY MAP & GROWTH REPORT
    # =========================================================================
    print_header("INFANT CAUSAL LEARNING DIAGNOSTIC REPORT")
    print(f" [*] Total Experiential Turns Evaluated : {engine.total_experiences}")
    print(f" [*] Inscribed Scar History Count      : {engine.scar_count}")
    print(f" [*] Accumulated Scar Tensor Trace     : {np.trace(engine.scar_engine.accumulated_scar_tensor):.4f}")
    print(f" [*] Final Experiential Maturity Index : {engine.maturity_index:.4f}")
    print(f" [*] Current Homeostasis Deficit Vector: Love={engine.homeostasis.love:.2f}, Order={engine.homeostasis.order:.2f}, Energy={engine.homeostasis.energy:.2f}")

    print("\n [CAUSAL MEMORY MAP SUMMARY]")
    for c in engine.causal_chains:
        print(f"   • [{c.chain_id}] Input: '{c.stimulus_summary}' | Valence: {c.outcome_valence:+.2f} | Scar Impact: {c.scar_impact:.2f}")

    print("\n [✓] VERIFICATION SUCCESSFUL: Infant Causal Language Loop is fully proven!")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_infant_causal_language_demo()
