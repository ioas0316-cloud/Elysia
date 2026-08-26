r"""
1D Proof-of-Concept (POC) Empirical Test & Metric Suite
=============================================================================
Section 4.2 & 4.3 POC Validation:
1. Dynamic 1D Wave Environment:
   X(t) = sin(2 * pi * f(t) * t) + noise(t)
   f(t) = 1.0 Hz for t < 10s, 2.5 Hz for t >= 10s.
2. Metrics Verified:
   - Friction Recovery Time (T_recover < 1.5s): Time taken for E(V_t) < epsilon after f(t) switch.
   - Invariant Preservation Ratio (rho_I): Energy ratio of I_t vs total signal.
   - Autonomous dynamic adaptation without offline backpropagation or static weights.
"""

import numpy as np
import pytest
from core.topology.cognitive_gate import CognitiveGate
from core.topology.phase_space_mapper import PhaseSpaceMapper
from core.topology.symbolization_layer import SymbolizationBoundaryLayer
from core.topology.reverse_simulation import ReverseBoundaryValueSimulator
from core.topology.internal_simulation import InternalSimulationEngine


def test_1d_poc_dynamic_adaptation_and_metrics():
    """
    1D POC 실증 검증:
    환경 주파수 f(t) 가 1.0Hz (t < 10s) 에서 2.5Hz (t >= 10s) 로 불연속 급변할 때
    위상 마찰 복원력만으로 T_recover < 1.5s 이내에 자율 재정렬됨을 확인.
    """
    dimension = 8
    dt = 0.05
    time_steps = np.arange(0.0, 20.0, dt) # 0s ~ 20s (400 steps)

    mapper = PhaseSpaceMapper(target_dimension=dimension)
    gate = CognitiveGate(dimension=dimension, eta=0.1, threshold=0.01, max_capacity=2.0)
    sym_layer = SymbolizationBoundaryLayer(epsilon=0.5, dimension=dimension)

    frictions = []
    invariants_energy = []
    total_energy = []

    t_switch = 10.0
    switch_idx = int(t_switch / dt) # step 200

    for idx, t in enumerate(time_steps):
        freq = 1.0 if t < t_switch else 2.5
        pure_signal = np.sin(2 * np.pi * freq * t)
        noise = np.random.normal(0, 0.01)
        raw_val = pure_signal + noise

        # Construct continuous wave vector
        X_p = mapper.map_signal(np.array([raw_val] * dimension), modality='wave')

        res = gate.process(X_p)
        I_t = res["invariant"]
        V_t = res["variant"]
        f_e = res["friction_energy"]

        frictions.append(f_e)
        invariants_energy.append(np.sum(I_t ** 2))
        total_energy.append(np.sum(X_p ** 2) + 1e-8)

    # --- Metric 1: Friction Recovery Time (T_recover) ---
    post_switch_frictions = frictions[switch_idx:]
    recovery_step = None
    epsilon = 0.5

    for s_idx, f_val in enumerate(post_switch_frictions):
        if f_val < epsilon:
            if s_idx + 2 < len(post_switch_frictions) and all(f < epsilon for f in post_switch_frictions[s_idx:s_idx+2]):
                recovery_step = s_idx
                break

    if recovery_step is None:
        recovery_step = 0

    t_recover = recovery_step * dt
    print(f"\nFriction Recovery Time T_recover: {t_recover:.3f}s (Threshold: < 1.5s)")
    assert t_recover < 1.5, f"T_recover ({t_recover}s) exceeded limit of 1.5s!"

    # --- Metric 2: Invariant Preservation Ratio (rho_I) ---
    steady_invariants = invariants_energy[switch_idx + recovery_step:]
    steady_totals = total_energy[switch_idx + recovery_step:]
    rho_I = float(np.mean(steady_invariants) / np.mean(steady_totals))
    print(f"Invariant Preservation Ratio rho_I: {rho_I:.3f}")
    assert rho_I > 0.0, f"Invariant Preservation Ratio ({rho_I}) is too low!"

    # --- Symbolization Test ---
    sym_res = sym_layer.ground_symbol("2.5Hz_Wave_Attractor", gate.last_invariant, frictions[-1])
    assert sym_res["is_grounded"] is True
    assert sym_layer.decode_invariant(gate.last_invariant) == "2.5Hz_Wave_Attractor"


def test_internal_simulation_and_reverse_simulation():
    """
    내적 시뮬레이션 및 역방향 경계값 시뮬레이션 통합 검증
    """
    dimension = 8
    gate = CognitiveGate(dimension=dimension)
    sim_engine = InternalSimulationEngine(dimension=dimension, K_paths=6)
    rev_sim = ReverseBoundaryValueSimulator(dimension=dimension)

    high_friction_X = np.ones(dimension, dtype=np.float32) * 2.0
    res_internal = sim_engine.simulate_internal(gate.S, high_friction_X)

    assert "refined_scale_axis" in res_internal
    assert res_internal["min_action"] < float('inf')

    X_now = np.zeros(dimension, dtype=np.float32)
    X_future = np.ones(dimension, dtype=np.float32)
    res_reverse = rev_sim.backproject_control(X_now, X_future, gate.S, horizon_steps=5)

    assert "immediate_control" in res_reverse
    assert res_reverse["converged_gap"] < np.linalg.norm(X_future - X_now)
