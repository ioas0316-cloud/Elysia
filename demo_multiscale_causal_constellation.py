"""
demo_multiscale_causal_constellation.py
======================================
CLI Demonstration Script for Elysia Multi-Scale Cosmological Causal Engine & SCM-NN.
Demonstrates:
1. 3-Tier Cosmological Hierarchy (Tier 1 Primordial -> Tier 2 Pantheon -> Tier 3 Lesser)
2. Causal Subcontracting & Causal Tax
3. Pearl's Do-Calculus Graph Surgery & Average Causal Effect (ACE)
4. PyTorch Neural SCM Backprop & Causal Loss Minimization
5. Temporal Triad (Spirit/Future/Superego, Mind/Present/Ego, Body/Past/Id) & LLM Prompt Decoding
6. Usurpation & Ascension Mechanics (Tier 3 -> Tier 2 Usurpation)
"""

import torch

from modules.causal_game_engine.alignment_field import (
    AlignmentVector,
    AlignmentType,
    HeroAlignmentState,
    AlignmentTensorField
)
from modules.causal_game_engine.multiscale_constellation import (
    MultiscaleCosmologicalEngine,
    MultiscaleConstellationNode,
    ConstellationTier,
    Tier3AffiliationType
)
from modules.causal_game_engine.do_calculus_engine import StructuralCausalModel
from modules.causal_game_engine.causal_scm_nn import DifferentiableSCM, CausalLossCalculator
from modules.causal_game_engine.causal_prompt_decoder import CausalPromptDecoder
from modules.causal_game_engine.bidirectional_causal_loop import IntegratedBidirectionalCausalLoop


def run_demo():
    print("==========================================================================")
    print("  ELYSIA MULTI-SCALE COSMOLOGICAL CAUSAL ENGINE & SCM-NN DEMONSTRATION")
    print("==========================================================================")

    # 1. Multi-scale Cosmology Setup
    tensor_field = AlignmentTensorField()
    cosmo_engine = MultiscaleCosmologicalEngine(tensor_field)

    print("\n[1] COSMOLOGICAL HIERARCHY INITIALIZED:")
    for cid, node in cosmo_engine.constellations.items():
        print(f"  • [{node.tier.value}] {node.name} (Share: {node.causal_share:.2f}, Power: {node.causal_power_pool:.1f})")

    # Register Hero
    hero = HeroAlignmentState(
        hero_id="hero_siegfried",
        name="지그프리트",
        current_alignment=AlignmentVector(0.4, 0.5),
        base_anchor_alignment=AlignmentVector(0.5, 0.5),
        spi_stat=60.0
    )
    tensor_field.register_hero(hero)

    # 2. Causal Subcontracting & Tax
    print("\n[2] CAUSAL SUBCONTRACTING & REVELATION TAX MECHANICS:")
    sub_res = cosmo_engine.execute_causal_subcontract("t2_pantheon_lg", "t3_vassal_iron", 300.0)
    print(f"  • Subcontract Transfer: Tier 2 -> Tier 3 Transferred {sub_res['transferred_power']} Causal Power.")

    tax_res = cosmo_engine.process_revelation_with_tax("t3_vassal_iron", "hero_siegfried", 50.0)
    print(f"  • Revelation Cast with Tax: Power Cost=50.0, Tax Paid to Tier 2={tax_res['causal_tax_paid']:.1f}")

    # 3. Do-Calculus Graph Surgery
    print("\n[3] PEARL'S DO-CALCULUS GRAPH SURGERY & ACE COMPUTATION:")
    scm = StructuralCausalModel()
    scm.add_causal_edge("event_shock", "hero_alignment_x", 0.6)
    scm.add_causal_edge("constellation_gravity", "hero_alignment_x", 0.4)

    print("  • Executing do(event_shock = -0.8) Intervention...")
    surgered_scm = scm.apply_do_intervention("event_shock", -0.8)
    print(f"  • Incoming edges to 'event_shock' severed! Surgered Hero Alignment X: {surgered_scm.nodes['hero_alignment_x'].value:.4f}")

    ace = scm.calculate_average_causal_effect("event_shock", "hero_alignment_x", val_a=1.0, val_b=-1.0)
    print(f"  • Average Causal Effect (ACE) of Event Shock on Hero Alignment X: {ace:.4f}")

    # 4. Bidirectional Causal Loop & Neural SCM Backprop
    print("\n[4] INTEGRATED BIDIRECTIONAL CAUSAL LOOP STEP:")
    loop = IntegratedBidirectionalCausalLoop()
    loop.alignment_field.register_hero(hero)

    loop_res = loop.execute_bidirectional_step(
        hero_id="hero_siegfried",
        external_action_log="영웅 지그프리트에게 지하 암시장의 금지된 계약서를 건넨다.",
        intervention_vector=(-0.6, -0.5)
    )

    print(f"  • Hero Drift Vector: dH/dt = {loop_res['drift_result']['dh_vector']}")
    print(f"  • New Hero Alignment: H(t) = {loop_res['drift_result']['new_alignment']}")
    print(f"  • PyTorch Causal Loss Backprop: Loss = {loop_res['causal_loss']:.4f}")

    print("\n[5] DECODED LLM AGENT PROMPT CONSTRAINTS & TEMPORAL TRIAD:")
    prompt_payload = loop_res["agent_prompt_payload"]
    print("  --- SYSTEM PROMPT BLOCK ---")
    print(prompt_payload["system_prompt"])
    print("  --- SAMPLING CONFIG ---")
    print(f"  Temp: {prompt_payload['parameters']['temperature']}, Top_P: {prompt_payload['parameters']['top_p']}")

    # 5. Ascension / Usurpation
    print("\n[6] CONSTELLATION ASCENSION & USURPATION MECHANIC:")
    asc_res = cosmo_engine.check_and_trigger_ascension("const_player", share_boost=0.30)
    print(f"  • Usurpation Event: {asc_res['description']}")
    print(f"  • Player Constellation New Tier: {cosmo_engine.constellations['const_player'].tier.value}")

    print("\n==========================================================================")
    print("  DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("==========================================================================")


if __name__ == "__main__":
    run_demo()
