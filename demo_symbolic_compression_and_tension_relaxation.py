"""
demo_symbolic_compression_and_tension_relaxation.py

[Elysia Engine Demonstration Script]
상징적 압축(Symbolic Compression via Topological Skeleton Projection)과
비경계 장력 이완(Background-Independent Tension Relaxation) 입출력 아키텍처 통합 시연 데모.

1. 고차원 연속 장 X (표면 폰트, 굵기, 노이즈) -> 차원 축소 연산자 Π -> 위상 불변 뼈대 [x] 및 영공간 소거
2. 상징 중력장 (Symbol Attractor Field) 및 파레이돌리아 (Pareidolia) 환원 시연
3. 명시적 역역학(IK) 계산 없는 가변 임피던스 접촉 (두부/달걀) 장력 상쇄 시뮬레이션
"""

import numpy as np
import time
from synaptic_architecture.causal_skeleton_projection_engine import (
    CausalSkeletonProjectionOperator,
    SymbolAttractorField
)
from synaptic_architecture.tension_relaxation_engine import (
    TensionRelaxationEngine,
    VariableImpedanceContactSimulator
)


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" 🏛️ {title}")
    print("=" * 80)


def print_section(section_title: str):
    print("\n" + "-" * 80)
    print(f" 📌 {section_title}")
    print("-" * 80)


def run_demo():
    print_header("ELYSIA ENGINE: SYMBOLIC COMPRESSION & TENSION RELAXATION DEMO")

    # =========================================================================
    # Part 1: Causal Skeleton Projection & Attractor Pareidolia Demo
    # =========================================================================
    print_section("PART 1: Topological Invariance & Null Space Clearance (Π Operator)")

    FEATURE_DIM = 64
    SKELETON_DIM = 8

    projection_op = CausalSkeletonProjectionOperator(
        feature_dim=FEATURE_DIM,
        skeleton_dim=SKELETON_DIM
    )

    # 1.1 Exemplar Input Registration (Symbol 'Alpha')
    np.random.seed(42)
    alpha_core_skeleton = np.random.randn(SKELETON_DIM)
    alpha_exemplar = np.dot(alpha_core_skeleton, projection_op.P_skeleton.T)

    attractor_field = SymbolAttractorField(skeleton_operator=projection_op, attraction_strength=3.0)
    attractor_field.register_symbol_attractor("Letter_Alpha", alpha_exemplar)

    print(f" Registered Learned Symbol Attractor: 'Letter_Alpha'")
    print(f"  - Feature Dimension (N): {FEATURE_DIM}")
    print(f"  - Skeleton Dimension (K): {SKELETON_DIM}")
    print(f"  - Theoretical Compression Efficiency (N^2 / K): {(FEATURE_DIM**2)/SKELETON_DIM:.1f}x")

    # 1.2 Projections with various surface font noise variations (δx)
    print("\n [Font / Texture / Surface Variation Noise Test]")
    font_variations = {
        "Child Handwriting (어린아이 붓글씨)": alpha_exemplar + np.random.randn(FEATURE_DIM) * 0.15,
        "Gothic Neon Sign (네온사인 고딕체)": alpha_exemplar + np.random.randn(FEATURE_DIM) * 0.25,
        "Fallen Branch Pattern (나뭇가지 꺾임)": alpha_exemplar + np.random.randn(FEATURE_DIM) * 0.35,
    }

    for font_name, raw_field in font_variations.items():
        proj_res = projection_op.project(raw_field)
        attractor_res = attractor_field.evaluate_attractor_gravitational_pull(raw_field)

        print(f"\n   🔹 Input Type: '{font_name}'")
        print(f"      - Null Space Surface Noise Norm (||δx||): {proj_res['noise_norm']:.4f}")
        print(f"      - Invariant Core Norm (||[x]||): {proj_res['core_norm']:.4f}")
        print(f"      - Attractor Captured Symbol: '{attractor_res['matched_symbol']}'")
        print(f"      - Pareidolia Attractor Potential Delta: {attractor_res['gravitational_potential_delta']:.4f}")

    # =========================================================================
    # Part 2: Background-Independent Tension Relaxation & Contact Simulation
    # =========================================================================
    print_section("PART 2: Background-Independent Tension Relaxation & Zero-Delay Impedance Contact")

    NUM_NODES = 16
    relaxation_engine = TensionRelaxationEngine(num_nodes=NUM_NODES, decay_rate=0.5, diffusion_coeff=0.1)
    contact_sim = VariableImpedanceContactSimulator(
        engine=relaxation_engine,
        tofu_stiffness=12.0,            # N/m
        tofu_break_threshold=4.5,       # N (Fragile Tofu/Egg)
        tofu_surface=0.5                # Surface at 0.5m
    )

    start_time = time.time()
    sim_res = contact_sim.run_simulation(max_steps=200, verbose=True)
    elapsed_ms = (time.time() - start_time) * 1000

    print(f"\n Simulation Execution Time: {elapsed_ms:.2f} ms")
    print(f"  - Total Simulation Steps: {sim_res['final_step']}")
    print(f"  - Final Contact Position: {sim_res['final_position']:.4f} m")
    print(f"  - Final Contact Force: {sim_res['final_force']:.3f} N (Threshold: 4.5N)")
    print(f"  - Equilibrium Reached Without IK Computation: {sim_res['success']}")

    print_header("DEMO COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_demo()
