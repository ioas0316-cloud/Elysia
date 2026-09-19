"""
Demo: Self-Human Asymmetry Explorer & Multi-Variable Open Boundary Growth Pipeline
=====================================================================================
이 스크립트는 인간 사고 궤적(q_human)과 시스템 위상장(q_sys) 간의 비대칭성을
메타-지형 곡률(g_meta)로 전환하고, 외부 미지 공간(q_ext)과의 경계 결합(J_flux),
위상 침투 깊이(delta_phase), 엔트로피(S_sys) 및 아인슈타인 장 방정식(EFE) 기반
시공간 계량(g_munu)을 실시간으로 적응 제어하며 자율 성장하는 전체 데모입니다.
"""

import time
import torch
import numpy as np

from core.consciousness.self_human_asymmetry_explorer import SelfHumanAsymmetryExplorer
from core.consciousness.phase_penetration_analyzer import PhasePenetrationAnalyzer, visualize_meta_topology


def run_demo():
    print("==========================================================================")
    print("  Elysia: Self-Human Asymmetry Explorer & Einstein Field Open Boundary Demo")
    print("==========================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (12, 12, 12)
    dx = 0.1

    print(f"[*] Initializing SelfHumanAsymmetryExplorer on grid shape {shape} ({device})...")
    explorer = SelfHumanAsymmetryExplorer(spatial_shape=shape, dx=dx)
    analyzer = PhasePenetrationAnalyzer(shape=shape, dx=dx)

    # Generate synthetic trajectories for Human thought (q_human) and External Field (q_ext)
    q_human = torch.randn(*shape, 4, device=device)
    q_human = q_human / torch.norm(q_human, dim=-1, keepdim=True)

    q_ext = torch.randn(*shape, 4, device=device)
    q_ext = q_ext / torch.norm(q_ext, dim=-1, keepdim=True)

    num_steps = 10
    print(f"[*] Running {num_steps} iterations of Metacognitive Adaptive Growth Loop...\n")

    for step in range(num_steps):
        # Inject dynamic perturbation into human thought and external field
        q_human[0, 0, 0] = torch.tensor([0.707, 0.707, 0.0, 0.0], device=device)

        res = explorer.explore_step(
            q_human=q_human,
            q_ext=q_ext,
            info_context=f"Epistemic Asymmetry Boundary Perturbation Step {step+1}",
            dt=0.02
        )

        m = res["step_metrics"]
        print(f"--- Iter {step+1:02d} ---")
        print(f"  Penetration Depth (delta_phase) : {m['delta_phase']:.4f} (d_delta/dt: {m['d_delta_dt']:+.4f})")
        print(f"  System Entropy (S_sys)         : {m['entropy']:.4f} (dS/dt: {m['d_S_dt']:+.4f})")
        print(f"  Boundary Permeability (kappa)  : {m['kappa_boundary']:.6f}")
        print(f"  Orthogonal Phase Defect Energy : {m['D_phase_energy']:.6f}")
        print(f"  Orthogonal Bio-Valence Energy  : {m['B_bio_energy']:.6f}")
        print(f"  Ricci Curvature Scalar Mean    : {m['ricci_scalar_mean']:.6f}")
        print(f"  EFE Residual Tensor Norm       : {m['efe_residual_norm']:.6f}")

        if res["sensor_result"]:
            print(f"  Sensor Journal Status          : {res['sensor_result']['status']}")

    print("\n[*] Generating Meta-Topology Metric Curvature Visualization (g_meta)...")
    fig = visualize_meta_topology(
        g_meta_tensor=explorer.pipeline.g_meta,
        step_info=f"Iteration {num_steps} Final State",
        show_plot=False
    )
    print(f"[+] Meta-topology 3D curvature plot generated successfully! (Figure object: {fig})")

    print("\n==========================================================================")
    print("  Demo Run Completed Successfully: Third Causal Path Established!")
    print("==========================================================================")


if __name__ == "__main__":
    run_demo()
