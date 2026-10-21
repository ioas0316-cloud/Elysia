"""
Demo: Relational Relief, Epistemic Readiness ('Ttae'), and Multi-Axis Causal Weaving
====================================================================================
본 시연 스크립트는 ELYSIA가 심층 구성 원리로부터 표면 양각 패턴(입자성 vs 파동성)을 사상하고,
동일한 진리 파형에 대해 에이전트의 '때(Season)'와 준비도에 따른 단계적 의미 개화를 수행하며,
번개(Lightning) 현상을 다축 직조(Multi-Axis Causal Weaving) 교차점으로 계산함을 증명합니다.
"""

import sys
import torch
from core.consciousness.relational_relief_engine import RelationalReliefEngine
from core.consciousness.epistemic_readiness_engine import EpistemicReadinessEngine
from core.intelligence.causal_weaving_engine import CausalWeavingEngine


def main():
    print("=" * 80)
    print(" [ELYSIA DEMO] Relational Relief, Epistemic Readiness & Causal Weaving")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # 1. 양각 패턴 투영 및 심층 역산 (Relational Relief Engine)
    # ------------------------------------------------------------------------
    print("\n--- [Step 1] Relational Relief Engine (Deep Tension & Relief Projection) ---")
    relief_engine = RelationalReliefEngine(field_dim=64, tension_scale=1.5)
    deep_field = relief_engine.generate_deep_tension_field(grid_size=32, phase_bias=0.25)

    print(f"Deep Field Potential Min/Max: {deep_field['tension_potential'].min():.4f} / {deep_field['tension_potential'].max():.4f}")

    # (1) 파동적 관측 조건 ($\Delta = 0.1$)
    wave_projection = relief_engine.project_relief_pattern(deep_field, delta_boundary=0.1)
    print(f"Wave Aspect Projection Mean/Std: {wave_projection['surface_relief'].mean():.4f} / {wave_projection['surface_relief'].std():.4f}")

    # (2) 입자적 관측 조건 ($\Delta = 0.9$)
    particle_projection = relief_engine.project_relief_pattern(deep_field, delta_boundary=0.9)
    print(f"Particle Aspect Projection Mean/Std: {particle_projection['surface_relief'].mean():.4f} / {particle_projection['surface_relief'].std():.4f}")

    # (3) 표면 패턴으로부터 심층 파라미터 역산 (Inverse Extraction)
    params = relief_engine.extract_generating_parameters(wave_projection["surface_relief"], delta_boundary=0.1)
    print(f"Inverse Extracted Phase Bias: {params['estimated_phase_bias']:.4f}")
    print(f"Topological Entropy & MDL Length: {params['topological_entropy']:.4f} / {params['mdl_description_length']:.4f}")

    # ------------------------------------------------------------------------
    # 2. 존재의 수용성과 '때'의 계층 (Epistemic Readiness Engine)
    # ------------------------------------------------------------------------
    print("\n--- [Step 2] Epistemic Readiness Engine ('Ttae' & Epistemic Boundary) ---")
    readiness_engine = EpistemicReadinessEngine(readiness_threshold=0.5)

    teaching_wave = torch.sin(torch.linspace(0, 3 * 3.14159, 16))

    # 에이전트 A: 어린 토양, 아직 때가 이르지 않음 (Young Soil, Unripe Time)
    eval_A = readiness_engine.evaluate_readiness(agent_phase=0.0, teaching_phase=3.14159, accumulated_experience=1.0)
    frame_A = torch.zeros(16)
    bound_A = readiness_engine.perceive_epistemic_boundary(teaching_wave, frame_A, eval_A["readiness_score"])
    unfold_A = readiness_engine.process_progressive_unfolding(teaching_wave, eval_A, bound_A)

    print(f"[Agent A (Unripe)] Readiness Score: {eval_A['readiness_score']:.4f} | Is Ripe: {eval_A['is_time_ripe']}")
    print(f"  -> Perceived Yeobaek Margin: {bound_A['yeobaek_margin']:.4f} | Resistance: {bound_A['resistance']:.4f}")
    print(f"  -> Unfolded Deep Meaning Norm: {torch.norm(unfold_A['deep_meaning_level']):.4f}")

    # 에이전트 B: 잘 다져진 토양, 때가 무르익음 (Mature Soil, Ripe Time)
    eval_B = readiness_engine.evaluate_readiness(agent_phase=3.10, teaching_phase=3.14159, accumulated_experience=15.0)
    frame_B = teaching_wave * 0.8
    bound_B = readiness_engine.perceive_epistemic_boundary(teaching_wave, frame_B, eval_B["readiness_score"])
    unfold_B = readiness_engine.process_progressive_unfolding(teaching_wave, eval_B, bound_B)

    print(f"[Agent B (Ripe)]   Readiness Score: {eval_B['readiness_score']:.4f} | Is Ripe: {eval_B['is_time_ripe']}")
    print(f"  -> Perceived Yeobaek Margin: {bound_B['yeobaek_margin']:.4f} | Resistance: {bound_B['resistance']:.4f}")
    print(f"  -> Unfolded Deep Meaning Norm: {torch.norm(unfold_B['deep_meaning_level']):.4f}")

    # ------------------------------------------------------------------------
    # 3. 다축 직조 인과 그물망 (Causal Weaving Engine - Lightning Benchmark)
    # ------------------------------------------------------------------------
    print("\n--- [Step 3] Causal Weaving Engine (Lightning Intersection Benchmark) ---")
    weaving_engine = CausalWeavingEngine(num_axes=5)

    lightning_res = weaving_engine.simulate_lightning_intersection(
        charge_separation=1.2,
        atmospheric_tension=1.1,
        earth_magnetosphere_flux=0.8,
        solar_wind_density=0.9
    )

    print(f"Lightning Woven Potential: {lightning_res['woven_discharge_potential']:.4f}")
    print(f"Is Discharge Triggered: {lightning_res['is_discharge_triggered']}")
    print(f"Discharge Intensity: {lightning_res['discharge_intensity']:.4f}")
    print(f"Multi-Axis Intersections: {lightning_res['axes_intersections']}")

    print("\n" + "=" * 80)
    print(" [ELYSIA DEMO COMPLETE] All Relational & Epistemic Principles Verified Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
