"""
demo_human_agency_phase_collapse.py
===================================
Integrated CLI Demo for Human Agency Non-linear Choice & Landau-Ginzburg Phase Collapse Simulation.

Steps:
1. [Step 1] Initial Setup: d=2 hero node & constellation causal model initialization.
2. [Step 2] Trial/Temptation Stress Injection: E_trial stress injection & det(Nabla^2 V(H)) <= 0 check.
3. [Step 3] SDE Euler-Maruyama Trajectory Integration: 100 Step numerical integration and trajectory logging.
4. [Step 4] Path Bifurcation & Agency Entropy Calculation: Path A/B/C decision, H_causal, and A_wonder index.
5. [Step 5] SCM-NN Backprop & W_adj Matrix Update: Wonder index backprop & adjacency matrix weight update.
"""

import numpy as np
import torch
import torch.optim as optim

from modules.causal_game_engine.human_agency_engine import (
    HumanAgencyEngine,
    HumanAgencyEvaluator,
    LandauGinzburgPotentialField,
    ChoiceOption,
    TransitionPath,
)
from modules.causal_game_engine.causal_scm_nn import DifferentiableSCM, CausalLossCalculator


def main():
    print("=" * 80)
    print(" Elysia Engine - Human Agency & Landau-Ginzburg Phase Collapse Simulation Demo ")
    print("=" * 80)

    # Step 1: Initial Setup
    print("\n[Step 1] Initial Setup: d=2 Hero Node & Constellation Causal Model Initialized")
    hero_id = "hero_prometheus"
    H_current = np.array([0.1, 0.05], dtype=np.float64)  # Initial alignment
    H_angel = np.array([1.0, 1.0], dtype=np.float64)    # Angelic pole (+Law, +Good)
    H_devil = np.array([-1.0, -1.0], dtype=np.float64)  # Demonic pole (-Chaos, -Evil)

    print(f" - Hero ID: {hero_id}")
    print(f" - Initial Alignment Vector H_0: {H_current.tolist()}")
    print(f" - Angelic Vector H_angel: {H_angel.tolist()}")
    print(f" - Demonic Vector H_devil: {H_devil.tolist()}")

    # Step 2: Trial/Temptation Stress Injection
    print("\n[Step 2] Trial/Temptation Stress Injection & Phase Collapse Detection")
    E_trial = np.array([1.8, 1.5], dtype=np.float64)  # Extreme trial stress towards ascension
    potential_field = LandauGinzburgPotentialField(a=-2.0, b=1.0)

    is_collapsed, det_val = potential_field.is_phase_collapsed(H_current)
    print(f" - Trial Stress Vector E_trial: {E_trial.tolist()}")
    print(f" - Hessian Matrix at H_0:\n{potential_field.hessian(H_current)}")
    print(f" - Hessian Determinant det(Nabla^2 V(H)): {det_val:.4f}")
    print(f" - Phase Collapse Condition Triggered: {is_collapsed}")

    # Step 3: SDE Euler-Maruyama Trajectory Integration
    print("\n[Step 3] SDE Euler-Maruyama Trajectory Integration (100 Steps)")
    engine = HumanAgencyEngine(
        potential_field=potential_field,
        sigma_fluctuation=0.05,
        dt=0.01,
        num_steps=100
    )
    H_final, trajectory = engine.simulate_sde_trajectory(H_current, E_trial, seed=42)
    print(f" - Trajectory Steps Completed: {len(trajectory) - 1}")
    print(f" - Initial Position: {trajectory[0].tolist()}")
    print(f" - Mid Position (Step 50): {trajectory[50].round(4).tolist()}")
    print(f" - Final Position (Step 100): {H_final.round(4).tolist()}")

    # Step 4: Path Bifurcation & Agency Entropy Calculation
    print("\n[Step 4] Path Bifurcation & Agency Entropy / Wonder Index Calculation")
    options = [
        ChoiceOption("option_sacrificed_ascension", utility=0.3, alignment_vector=np.array([0.9, 0.9])),
        ChoiceOption("option_tyrannical_fall", utility=0.7, alignment_vector=np.array([-0.8, -0.8])),
        ChoiceOption("option_human_defiance", utility=0.5, alignment_vector=np.array([0.0, 0.2])),
    ]
    W_constellation = np.array([-0.5, -0.5])  # Constellation heavily predicts option 2 (tyrannical fall)

    result = engine.process_trial_event(
        hero_id=hero_id,
        H_current=H_current,
        E_trial=E_trial,
        H_angel=H_angel,
        H_devil=H_devil,
        T_agency=0.9,
        xi_ordeal=1.8,
        options=options,
        W_constellation=W_constellation,
        seed=42
    )

    print(f" - Bifurcation Path Result: {result['bifurcation_path']}")
    print(f" - Event Description: {result['event'].description}")
    print(f" - Constellation Predicted Probabilities P_pred: {np.round(result['P_pred'], 4)}")
    print(f" - Human Actual Choice Probabilities P_act: {np.round(result['P_act'], 4)}")
    print(f" - Defiance Weight S_defiance: {np.round(result['S_defiance'], 4)}")
    print(f" - Causal Entropy H_causal (KL-Divergence): {result['H_causal']:.4f}")
    print(f" - Wonder Index A_wonder: {result['A_wonder']:.4f}")

    # Step 5: SCM-NN Backprop & W_adj Matrix Update
    print("\n[Step 5] SCM-NN Backprop & W_adj Matrix Weight Update")
    num_nodes = 4
    scm = DifferentiableSCM(num_nodes=num_nodes)
    optimizer = optim.Adam(scm.parameters(), lr=0.05)
    loss_calculator = CausalLossCalculator(lambda_cf=1.0, lambda_sparsity=0.05, lambda_dag=0.1, lambda_wonder=0.5)

    x_obs = torch.randn(16, num_nodes)
    target = torch.randn(16, num_nodes)

    W_adj_before = scm.get_masked_adj().detach().clone()

    optimizer.zero_grad()
    pred = scm(x_obs)
    W_adj = scm.get_masked_adj()
    loss = loss_calculator(pred, target, W_adj, A_wonder=result['A_wonder'])
    loss.backward()
    optimizer.step()

    W_adj_after = scm.get_masked_adj().detach().clone()
    adj_diff = torch.norm(W_adj_after - W_adj_before).item()

    print(f" - Total Causal Loss L_total: {loss.item():.4f}")
    print(f" - Adjacency Matrix W_adj Norm Shift after Backprop: {adj_diff:.6f}")
    print("\n[SUCCESS] Elysia Human Agency & Phase Collapse Pipeline successfully executed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
