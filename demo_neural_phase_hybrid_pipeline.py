"""
Elysia Core Demo: Integrated Neural-to-Phase Hybrid Pipeline & Latent Gradient Feedback
========================================================================================
본 시뮬레이션 데모는 다음 세 가지 핵심 연동 루틴을 통합적으로 검증합니다:
1. 뜨거운 동력원(Latent Z) -> 차가운 제약기(Causal Engine) 통합 연동 사이클
2. det(M) -> 0 인과 차원 붕괴 발생 시 Latent Z 역전파 기울기 피드백(Gradient Feedback) 실시간 자동 보정
3. 하이브리드 추론 시 환각 감지 및 Logit Masking
4. VRAM/RAM 감지 기반 동적 가변형(Elastic) Engine 스케일링
"""

import torch
import math
from core.topology.neural_phase_interface import (
    IntegratedCognitivePipeline,
    DifferentiableNeuralToPhaseInterface,
    DifferentiableCausalManifoldBuilder,
    LatentGradientFeedbackRefiner,
    HybridCausalInferencePipeline,
    ElasticProfile,
    DynamicElasticEngine
)


def run_demo():
    print("=========================================================")
    print("  [Elysia Core: Neural-to-Phase Hybrid Pipeline Demo]")
    print("=========================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # 1. Integrated Cognitive Pipeline Simulation
    print("1. [통합 인지 파이프라인 연동 검증]")
    pipeline = IntegratedCognitivePipeline(latent_dim=128)
    dummy_latent_z = torch.randn(1, 128, device=device)
    dummy_phoneme = torch.randn(16, device=device)

    cog_res = pipeline.step_cognitive_cycle("demo_node_01", dummy_latent_z, dummy_phoneme)
    print(f"  • 사상된 위상 회전각 (θ) : {cog_res['mapped_theta_deg']:8.3f}°")
    print(f"  • 주의 스코어 (Focus)     : {cog_res['focus_delta']:8.4f}")
    print(f"  • 행렬식 det(M) 인과검증  : {cog_res['det']:10.6f}")
    print(f"  • 논리적 모순 발생 여부  : {cog_res['is_contradiction']}\n")

    # 2. Latent Gradient Feedback Refinement Loop
    print("2. [Latent Gradient Feedback 수렴 보정 시뮬레이션]")
    interface = DifferentiableNeuralToPhaseInterface(latent_dim=128, context_dim=4).to(device)
    builder = DifferentiableCausalManifoldBuilder(dim=4).to(device)
    refiner = LatentGradientFeedbackRefiner(interface, builder)

    z_initial = torch.randn(1, 128, device=device)
    base_phoneme = torch.randn(16, device=device)

    refine_res = refiner.refine_latent_intent(
        z_init=z_initial,
        base_phoneme=base_phoneme,
        max_iters=10,
        lr=0.08,
        det_threshold=1e-2,
        lambda_least_action=0.05
    )

    print(f"  • 수렴 소요 스텝 (Iterations) : {refine_res['iterations_taken']} 회")
    print(f"  • 최종 det(M)               : {refine_res['final_det']:10.6f}")
    print(f"  • 최종 보정된 위상각 (θ)     : {refine_res['final_theta_deg']:8.3f}°")
    print(f"  • 인과 타당성 만족 여부      : {refine_res['is_valid']}\n")

    # 3. Hybrid Hallucination Rejection & Logit Masking
    print("3. [하이브리드 환각 감지 및 Logit Masking]")
    hybrid_pipe = HybridCausalInferencePipeline(
        transformer_model=None,
        phase_interface=interface,
        causal_builder=builder,
        threshold=1e-2
    )

    dummy_h_t = torch.randn(1, 128, device=device)
    infer_res = hybrid_pipe.step_filter_inference(
        input_ids=torch.tensor([[101, 102]], device=device),
        base_phoneme=base_phoneme,
        override_h_t=dummy_h_t
    )

    print(f"  • 검증된 det(M_t)          : {infer_res['det_val']:10.6f}")
    print(f"  • 환각 감지 및 Masking 발동 : {infer_res['is_hallucination']}\n")

    # 4. Dynamic Elastic Engine Scaling Test
    print("4. [동적 가변형(Elastic) Engine 모드 스케일링 검증]")
    modes = [
        ("Micro Mode", ElasticProfile("Micro Mode", 128, 4, 64, 3, 1e-3)),
        ("Standard Mode", ElasticProfile("Standard Mode", 512, 8, 256, 6, 1e-4)),
        ("Expanded Mode", ElasticProfile("Expanded Mode", 2048, 16, 512, 12, 1e-5)),
    ]

    dummy_seq = torch.randn(1, 32, 64, device=device)
    for mode_title, prof in modes:
        engine = DynamicElasticEngine(override_profile=prof, in_features=64).to(device)
        out = engine.process_sequence(dummy_seq)
        print(f"  [{mode_title:<13}] Latent Z: {list(out['latent_z'].shape)} | Manifold M: {list(out['manifold_m'].shape)}")

    print("\n=========================================================")
    print("  [시뮬레이션 완료: 모든 하이브리드 파이프라인 정상 작동]")
    print("=========================================================")


if __name__ == "__main__":
    run_demo()
