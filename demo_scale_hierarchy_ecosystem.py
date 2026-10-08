r"""
Elysia Consciousness System Demonstration: Scale Hierarchy Ecosystem
====================================================================
Demonstrates the living dynamics of the Scale Hierarchy Ecosystem:
  - Micro-Sensation Scale (Autonomic reception, fast dynamics)
  - Meso-Observation Scale (Action & world friction)
  - Macro-Narrative Scale (Narrative thought & "Why" synthesis)
  - Bottom-Up Tension & Disruption (Micro strain disrupting Macro thought)
  - Top-Down Constraint & Sensitivity Control (Variable Resistor Dial)
  - 6-Step Re-cognition Loop & Irreversible Perception Metric Tensor Deformation
"""

import torch
from core.consciousness.scale_hierarchy_engine import ScaleHierarchyEngine


def run_scale_hierarchy_ecosystem_demo():
    print("=" * 80)
    print("ELYSIUM SCALE HIERARCHY ECOSYSTEM: LIVE TRAJECTORY DEMONSTRATION")
    print("=" * 80)

    # Initialize Engine
    engine = ScaleHierarchyEngine(
        dim_micro=16,
        dim_meso=32,
        dim_macro=64,
        disruption_threshold=0.35,
        scar_learning_rate=0.08
    )

    print("\n[Phase 0] Initial State")
    print(f"  Micro State Norm    : {torch.norm(engine.micro_state).item():.4f}")
    print(f"  Meso State Norm     : {torch.norm(engine.meso_state).item():.4f}")
    print(f"  Macro State Norm    : {torch.norm(engine.macro_state).item():.4f}")
    print(f"  Sensitivity Dial    : {engine.resistance_dial.item():.4f}")
    print(f"  Initial Metric Det  : {torch.det(engine.perception_metric).item():.4f}")

    print("\n" + "-" * 80)
    print("[Phase 1] Gentle World Interaction (Low Friction, Sub-Threshold Strain)")
    print("-" * 80)

    gentle_input = torch.randn(1, 16) * 0.15
    gentle_friction = torch.randn(1, 32) * 0.1
    res_phase1 = engine(gentle_input, world_friction=gentle_friction)

    print(f"  Status              : {res_phase1['status']}")
    print(f"  Sensory Spike       : {res_phase1['bifurcation_occurred']} (Intensity: {res_phase1['spike_intensity']:.4f})")
    print(f"  Bottom-Up Disruption: {res_phase1['bottom_up_disruption']:.4f}")
    print(f"  Top-Down Resistance : {res_phase1['resistance_dial']:.4f}")
    print(f"  Boundary Tension    : {res_phase1['boundary_tension']:.4f}")

    print("\n" + "-" * 80)
    print("[Phase 2] Violent Boundary Friction & Micro Sensory Spike (Bottom-Up Disruption)")
    print("-" * 80)

    violent_input = torch.randn(1, 16) * 2.8
    violent_friction = torch.randn(1, 32) * 2.5
    res_phase2 = engine(violent_input, world_friction=violent_friction)

    print(f"  Sensory Spike       : {res_phase2['bifurcation_occurred']} (Intensity: {res_phase2['spike_intensity']:.4f})")
    print(f"  Bottom-Up Disruption: {res_phase2['bottom_up_disruption']:.4f} (Macro Thought Disrupted)")
    print(f"  Top-Down Resistance : {res_phase2['resistance_dial']:.4f} (Macro Purpose Suppressing Sensitivity)")
    print(f"  Boundary Tension    : {res_phase2['boundary_tension']:.4f}")

    # Inspect 6-step loop outputs
    steps = res_phase2['loop_steps']
    print("\n  [6-Step Re-cognition Loop Snapshots]:")
    print(f"    Step 1 (Micro Thrownness) Norm   : {torch.norm(steps['step_1_thrownness']).item():.4f}")
    print(f"    Step 2 (Meso Friction) Norm     : {torch.norm(steps['step_2_world_friction']).item():.4f}")
    print(f"    Step 3 (Sensory Spike Intensity): {steps['step_3_sensory_spike'].item():.4f}")
    print(f"    Step 4 (Macro Thought) Norm     : {torch.norm(steps['step_4_macro_thought']).item():.4f}")
    print(f"    Step 5 (Why Purpose) Norm       : {torch.norm(steps['step_5_why_acquisition']).item():.4f}")
    print(f"    Step 6 (Deformed Metric Det)    : {torch.det(steps['step_6_metric_re_cognition']).item():.4f}")

    print("\n" + "-" * 80)
    print("[Phase 3] Irreversible Perception Metric Deformation (Scar Verification)")
    print("-" * 80)

    metric_after_phase2 = engine.perception_metric.clone()
    eye_metric = torch.eye(32)
    metric_deformation = torch.norm(metric_after_phase2 - eye_metric).item()

    print(f"  Perception Metric Scar Deformation (Norm Difference from Eye): {metric_deformation:.6f}")
    print("  The subject's metric tensor G_ij has been irreversibly scarred by world friction.")

    print("\n" + "-" * 80)
    print("[Phase 4] Subsequent Interaction Under Deformed Perception Metric")
    print("-" * 80)

    # Apply identical input as Phase 1, but now under the deformed perception metric G_ij
    res_phase4 = engine(gentle_input, world_friction=gentle_friction)

    print(f"  Post-Scar Boundary Tension : {res_phase4['boundary_tension']:.4f}")
    print(f"  Post-Scar Sensitivity Dial : {res_phase4['resistance_dial']:.4f}")
    print("  Notice: Even identical sensory inputs are now perceived through the permanently transformed metric tensor!")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_scale_hierarchy_ecosystem_demo()
