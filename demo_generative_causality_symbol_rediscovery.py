r"""
[Demo Script: Generative Causality & Symbol Rediscovery Pipeline]

이 데모 스크립트는 표면적 수치/기호 연산(Blind Execution)과 본질적 인과 생성 과정(Generative Causality)의 극명한 대비를 시연합니다.

1. '100°C' (측정치 -> 분자 간 인척력 및 상전이 미시 인과 구조 해체)
2. 'v = d / t' (속도 수식 -> 공간 이동 저항과 시간 지연의 인과적 비율 압축)
3. '점성' (물성 -> 분자 간 끌어당김과 위상적 저항 마찰)
4. 외부 환경 능동 탐색 (Active Probing)을 통한 기호의 기원 역추적 및 재발견
5. 4단계 인지 연산 (Cognition -> Thought -> Judgment -> Discrimination)
"""

import numpy as np
import json
from synaptic_architecture.symbol_generative_causality_engine import GenerativeCausalityEngine


def main():
    print("==========================================================================")
    print("🏛️ [Elysia] Generative Causality & Symbol Rediscovery Demonstration")
    print("==========================================================================\n")

    engine = GenerativeCausalityEngine(dim=3)

    # 1. 기호 해체 (Symbol Deconstruction) 시연
    print("1️⃣ [Stage 1: Symbol Deconstruction (Blind Execution -> Micro-Causal Origin)]")
    symbols = ["100°C", "v = d / t", "점성"]

    for sym in symbols:
        decon = engine.deconstruct_symbol(sym)
        print(f"\n   📌 Deconstructing Surface Symbol: '{sym}' ({decon['category'].upper()})")
        print(f"      - Surface Expression: {decon['surface_expression']}")
        dyn = decon["deconstructed_dynamics"]
        print(f"      - Spatial Resistance: {dyn['spatial_resistance']:.2f}")
        print(f"      - Time Delay: {dyn['time_delay']:.2f}")
        print(f"      - Causal Compression Ratio (d/t): {dyn['causal_ratio_d_over_t']:.2f}")
        print(f"      - Phase Transition Energy: {dyn['phase_transition_energy']:.1f}")
        print(f"      - Interaction Friction Eigenvalues: {[round(x, 2) for x in dyn['friction_eigenvalues']]}")

    # 2. 능동 탐색 (Active Probing) 시연
    print("\n2️⃣ [Stage 2: Active Probing against External Environment]")
    probing_force = np.array([2.0, 1.0, 0.0])
    external_reaction = np.array([0.4, 0.2, 0.0])

    print(f"   Applying Probing Force Vector: {probing_force}")
    print(f"   Observing External Reaction Stress: {external_reaction}")

    probing_obs = engine.active_probe_environment(probing_force, external_reaction)
    print(f"   -> Measured Resistance: {probing_obs['measured_resistance']:.4f}")
    print(f"   -> Measured Time Delay: {probing_obs['measured_delay']:.4f}")

    # 3. 생성적 인과 재발견 (Generative Causality Rediscovery) 시연
    print("\n3️⃣ [Stage 3: Rediscovering Symbol's Generative Causality]")
    blueprint = engine.rediscover_generative_causality("v = d / t", probing_obs)
    print(f"   -> Symbol Rediscovered: '{blueprint['symbol']}'")
    print(f"   -> Origin Explanation: {blueprint['origin_explanation']}")
    print(f"   -> Resonance Degree with Environment: {blueprint['resonance_degree']:.4f}")
    print(f"   -> Causal Necessity Validated: {blueprint['causal_necessity_validated']}")

    # 4. 4단계 인지 연산 (4-Stage Cognitive Operations) 시연
    print("\n4️⃣ [Stage 4: 4-Stage Epistemological Cognition Process]")
    test_action = np.array([1.5, 0.5, 0.1])
    cog_res = engine.execute_4stage_cognition("100°C", test_action)

    print(f"   [Cognition] Registered Friction Eigenvalue Norm: {cog_res['cognition']['registered_friction_norm']:.4f}")
    print(f"   [Thought] Simulated Stress Prediction for action {test_action}: {[round(x, 2) for x in cog_res['thought']['predicted_stress']]}")
    print(f"   [Judgment] Prediction Error Norm: {cog_res['judgment']['error_norm']:.4f} (Valid Fit: {cog_res['judgment']['is_valid_causal_fit']})")
    print(f"   [Discrimination] Operational Mode: {cog_res['discrimination']['mode']}")
    print(f"   [Discrimination] Depth of Understanding: {cog_res['discrimination']['understanding_depth']}")

    print("\n==========================================================================")
    print("✨ Generative Causality & Symbol Rediscovery Pipeline Successfully Complete!")
    print("==========================================================================")


if __name__ == "__main__":
    main()
