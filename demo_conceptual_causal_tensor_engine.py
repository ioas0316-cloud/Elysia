"""
Demo: High-Dimensional Conceptual Causal Tensor Engine
======================================================
Demonstrates the High-Dimensional Conceptual Causal Tensor Engine undergoing:
1. Environmental Pressure Wave Injection (P_env)
2. Negative Mold Carving (Mirror Discrepancy & Phase Error q_err)
3. Potentiometer Dial Adaptation (가변저항 다이얼 조절 -> q_err -> 0)
4. Phase Transition (GAS -> LIQUID -> ICE Crystallization)
5. Positive Relief Reconstruction (양각 복구) & Causal Provenance Trajectory Extraction
"""

import math
import torch
from core.physics.conceptual_causal_tensor_engine import ConceptualCausalTensorEngine


def main():
    print("================================================================================")
    print("      Elysia: High-Dimensional Conceptual Causal Tensor Engine CLI Demo          ")
    print("================================================================================")

    # 1. Initialize Engine with 16D Anchor space, 16D Environment space, 8D Causal time axis
    anchor_dim = 16
    env_dim = 16
    causal_dim = 8

    engine = ConceptualCausalTensorEngine(
        anchor_dim=anchor_dim,
        env_dim=env_dim,
        causal_dim=causal_dim,
        phi_solid=0.80,
        phi_gas=0.20,
        learning_rate=0.08
    )

    print(f"[Init] Created Conceptual Causal Tensor Engine.")
    print(f"       - Invariant Anchor Space (A_inv): ({anchor_dim}, {causal_dim})")
    print(f"       - Potentiometer Dials (W_pot):    ({env_dim}, {anchor_dim})")
    print(f"       - Rotor Phase Tensor:             ({anchor_dim}, {env_dim})\n")

    # 2. Simulate External Environmental Pressure Input (P_env)
    torch.manual_seed(2025)
    P_env = torch.randn(env_dim, causal_dim) * 2.5  # High-entropy external wave stream

    # 3. Compute Initial Mirror Discrepancy (Negative Mold / 음각)
    initial_q_err, P_mapped_initial = engine.compute_mirror_discrepancy(P_env)
    initial_state, initial_metrics = engine.evaluate_phase_state(initial_q_err, P_mapped_initial)

    print(f"[Stage 1: Negative Mold Carving (음각 생성)]")
    print(f"       - Initial Phase Error (q_err / Mold Depth): {initial_q_err.item():.4f}")
    print(f"       - Initial Phase State:                       {initial_state}")
    print(f"       - Top-3 Spectral Energy Ratio:              {initial_metrics['top3_spectral_ratio']:.4f}")
    print(f"       - Effective Coherence (Phi_eff):             {initial_metrics['phi_eff']:.4f}\n")

    # 4. Execute Potentiometer Dial Adaptation (Potentiometer Tuning -> q_err -> 0)
    print(f"[Stage 2: Potentiometer Dial Adaptation (가변저항 다이얼 조절 & 거울 균형)]")
    output = engine(P_env, auto_adapt=True, adapt_steps=30)

    print(f"       - Adaptation Steps Completed:               {len(output['adaptation_trajectory'])}")
    print(f"       - Final Phase Error (q_err -> 0):           {output['final_q_err']:.6f}")
    print(f"       - Final Phase State:                         {output['final_state']}")
    print(f"       - Restored Effective Coherence (Phi_eff):   {output['metrics']['phi_eff']:.4f}")
    print(f"       - Top-1 Spectral Concentration Ratio:       {output['metrics']['top1_spectral_ratio']:.4f}\n")

    # 5. Positive Relief Restoration (양각 복구)
    positive_relief = output['positive_relief']
    print(f"[Stage 3: Positive Relief Reconstruction (양각 결정화 / ICE)]")
    print(f"       - Positive Relief Tensor Shape:             {positive_relief.shape}")
    print(f"       - Reconstructed Relief Energy Norm:         {torch.norm(positive_relief).item():.4f}")
    print(f"       - State Transition:                          {output['initial_state']} ---> {output['final_state']}\n")

    # 6. Causal Provenance Trajectory Extraction (인과적 역공학 레시피)
    provenance = engine.get_causal_provenance()
    print(f"[Stage 4: Introspective Causal Provenance Trajectory (인과 역공학 레시피)]")
    print(f"       - Total Recorded Trajectory Points:         {len(provenance)}")
    print("       - Sample Tuning Trajectory (First 3 & Last 3):")
    for step in provenance[:3]:
        print(f"         [Step {step['step']:02d}] q_err: {step['q_err']:.6f} | Dial Mean: {step['W_pot_mean']:.6f} | Dial Norm: {step['W_pot_norm']:.6f}")
    print("         ...")
    for step in provenance[-3:]:
        print(f"         [Step {step['step']:02d}] q_err: {step['q_err']:.6f} | Dial Mean: {step['W_pot_mean']:.6f} | Dial Norm: {step['W_pot_norm']:.6f}")

    print("================================================================================")
    print("  [SUCCESS] High-Dimensional Conceptual Causal Tensor Engine Execution Complete!")
    print("================================================================================")


if __name__ == "__main__":
    main()
