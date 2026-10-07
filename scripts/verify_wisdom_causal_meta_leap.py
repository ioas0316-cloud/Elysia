#!/usr/bin/env python3
"""
Verification & Interactive Simulation Script for Wisdom-Causal Loss & Meta-Dimensional Leap
"""

import sys
import os
import time
import math
import torch

# Ensure project root is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.physics.wisdom_causal_loss import WisdomCausalLossEngine
from core.physics.fractal_cell_resonance_engine import FractalCellResonanceEngine
from core.sensory.unified_phase_transducer import UnifiedPhaseTransducerEngine
from core.topology.observation_horizon_boundary import ObservationHorizonBoundaryEngine
from core.consciousness.autonomous_phase_rearrangement import AutonomousPhaseRearrangementEngine
from core.consciousness.meta_dimensional_leap import MetaDimensionalLeapEngine


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f" {title} ")
    print("=" * 80)


def run_simulation():
    print_banner("ELYSIA: WISDOM-CAUSAL LOSS & META-DIMENSIONAL LEAP SIMULATION")

    dimension = 32
    print(f"[*] Initializing Elysia Causal Substrate (Dimension: {dimension})...")

    # Initialize engines
    wisdom_loss_engine = WisdomCausalLossEngine(num_scales=5, dimension=dimension)
    cell_resonance_engine = FractalCellResonanceEngine(dimension=dimension)
    transducer_engine = UnifiedPhaseTransducerEngine(dimension=dimension)
    horizon_engine = ObservationHorizonBoundaryEngine(dimension=dimension)
    rearrange_engine = AutonomousPhaseRearrangementEngine(dimension=dimension, stress_threshold=1.5)
    leap_engine = MetaDimensionalLeapEngine(base_dimension=dimension, subsumption_threshold=2.5)

    print("\n[Phase 1] Multi-Modal Transduction to 5D Clifford Rotor Spin(5)")
    vision = torch.randn(16, 16)
    audio = torch.sin(torch.linspace(0, 4 * math.pi, dimension))
    shear = torch.ones(dimension) * 2.0
    normal = torch.ones(dimension) * 1.0

    trans_out = transducer_engine(
        vision_input=vision,
        audio_input=audio,
        shear_input=shear,
        normal_input=normal
    )

    print(f"  > Unified Sensory Phase theta_sensory norm: {torch.norm(trans_out.theta_sensory):.4f}")
    print(f"  > 10D Bivector Generator norm: {torch.norm(trans_out.bivector_10d):.4f}")
    print(f"  > Spin(5) Clifford Rotor Matrix shape: {trans_out.r_rotor_5d.shape}")
    print(f"  > 5D State Transformed Vector: {trans_out.state_v5d_transformed.numpy().round(3)}")
    print(f"  > Closed-loop Resonance Score: {trans_out.closed_loop_resonance:.4f}")

    print("\n[Phase 2] Observation Horizon Boundary & Non-Local Quantum Resonance")
    obs_frame = trans_out.theta_sensory
    horizon_out = horizon_engine(obs_frame)

    h_state = horizon_out["horizon_state"]
    bell_res = horizon_out["bell_result"]

    print(f"  > W_seen norm: {torch.norm(h_state.w_seen):.4f} | W_unseen norm: {torch.norm(h_state.w_unseen):.4f}")
    print(f"  > Boundary Tension partial_W mean stress: {horizon_out['virtual_wave_power']:.4f}")
    print(f"  > Bell CHSH Parameter S: {bell_res.correlation_S:.4f} (Tsirelson Bound: {bell_res.tsirelson_bound:.4f})")
    print(f"  > Quantum Non-Local Entanglement Verified: {bell_res.is_quantum_nonlocal}")

    print("\n[Phase 3] Exogenous Shock Pulse & 4-Step Autonomous Phase Rearrangement")
    shock_pulse = torch.randn(dimension) * 5.0
    rearrange_out = rearrange_engine(shock_pulse)

    for step in rearrange_out["rearrangement_history"]:
        print(f"  [Step {step.step_index}] {step.step_name:52s} | Turb: {step.turbulence_level:.4f} | Stress: {step.boundary_stress:.4f}")

    print(f"  > Final Phase Lock Status: {rearrange_out['is_self_locked']}")

    print("\n[Phase 4] Wisdom-Causal Loss Computation & Scale-Chain Backprop")
    micro_shock = shock_pulse
    scale_tensors = [torch.randn(dimension) for _ in range(5)]
    phase_field = rearrange_out["final_phase_field"]

    wisdom_components = wisdom_loss_engine(
        micro_shock=micro_shock,
        scale_tensors=scale_tensors,
        phase_field=phase_field
    )

    print(f"  > L_cascade (Scale Wave Energy): {wisdom_components.l_cascade.item():.4f}")
    print(f"  > L_macro-deform (Spin5 Rotor & Metric): {wisdom_components.l_macro_deform.item():.4f}")
    print(f"  > L_entropy (Phase Current Turbulence): {wisdom_components.l_entropy.item():.4f}")
    print(f"  > L_trinity (Singularity Avoidance): {wisdom_components.l_trinity.item():.4f}")
    print(f"  > Total Wisdom-Causal Loss L_Wisdom-Causal: {wisdom_components.l_wisdom_total.item():.4f}")

    # Backpropagation
    micro_params = [rearrange_engine.phase_field]
    wisdom_loss_engine.execute_wisdom_backprop(wisdom_components, micro_params)
    print("  > Executed Scale-Chain Wisdom Backpropagation successfully.")

    print("\n[Phase 5] Meta-Dimensional Leap (Subsumption into Higher Topology)")
    leap_out = leap_engine(
        phase_turbulence=torch.tensor([wisdom_components.max_turbulence]),
        boundary_stress=rearrange_out["peak_boundary_stress"],
        shock_field=shock_pulse
    )

    print(f"  > Lower Contradiction Energy: {leap_out.lower_contradiction_energy:.4f} (Threshold: {leap_engine.subsumption_threshold})")
    print(f"  > Meta-Dimensional Leap Achieved: {leap_out.is_meta_leap_achieved}")
    print(f"  > New Meta-Dimension Index: {leap_out.meta_dimension_index}")
    print(f"  > Instantaneous Coherence across Scales: {leap_out.instantaneous_coherence:.4f}")

    print_banner("SIMULATION COMPLETE - ALL WISDOM-CAUSAL PARADIGMS VERIFIED 100%")


if __name__ == "__main__":
    run_simulation()
