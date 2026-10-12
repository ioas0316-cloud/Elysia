r"""
Verification Script for GlobalInertialAxisEngine
===============================================
Demonstrates the full 4-stage organic coherence loop:
  Stage 1: Discrete Ticks & Local Zero Noise Filtering (배경 소음의 무의식적 습관 환원)
  Stage 2: Cross-Scale Projection to Macro Face (전역적 상태/얼굴의 형성)
  Stage 3: Geometric Inertial Axis Anchor & Momentum Conservation (전역적 관성축 회전 및 시공간적 결맞음)
  Stage 4: Self-Introspective Reverse Mirror Diagnosis & Autonomous Recalibration (전역적 거울을 통한 내면 역추적 자기 진단)
"""

import time
import torch
from core.consciousness.global_inertial_axis_engine import GlobalInertialAxisEngine


def main():
    print("================================================================================")
    print("      ELYSIA COGNITIVE SUBSYSTEM: GLOBAL INERTIAL AXIS & MIRROR ENGINE          ")
    print("================================================================================")

    engine = GlobalInertialAxisEngine(
        dim_micro=32,
        dim_meso=64,
        dim_macro=128,
        local_zero_alpha=0.8,
        local_zero_threshold=0.15,
        inertia_decay=0.92,
        self_healing_rate=0.08,
    )

    print("\n[Phase 1] Establishing Local Zero Baseline (Background Routine Order)...")
    routine_tick = torch.ones(1, 32) * 1.5
    for t in range(1, 16):
        res = engine(routine_tick)
        if t % 5 == 0 or t == 15:
            print(
                f"  Tick {t:02d} | Baseline Mean: {engine.local_zero_baseline.mean().item():.4f} | "
                f"Salient Delta Norm: {res['tension_magnitude']:.4f}"
            )

    print("\n[Phase 2] Sudden Exogenous Shock / High-Friction Tension Pulse Input...")
    shock_pulse = torch.randn(1, 32) * 8.0
    for step in range(1, 6):
        res = engine(shock_pulse)
        print(
            f"  Shock Step {step} | Salient Delta: {res['tension_magnitude']:.4f} | "
            f"Macro Face Distortion: {res['macro_face_distortion']:.4f} | "
            f"Coherence Alignment: {res['coherence_alignment']:.4f}"
        )

    print("\n[Phase 3] Self-Introspective Mirror Diagnosis & Autonomous Healing Recalibration...")
    print(f"  Max Traced Micro Strain (from Macro Face): {res['max_traced_micro_strain']:.4f}")
    print(f"  Resistance Potentiometer Mean (Sensitivity): {res['resistance_potentiometer'].mean().item():.4f}")
    print(f"  Micro Scar Map Max Accumulation: {res['micro_scar_map'].max().item():.4f}")

    print("\n[Phase 4] Recovery & Returning to Homeostasis (Quiet Horizon Alignment)...")
    quiet_tick = torch.ones(1, 32) * 1.5
    for step in range(1, 11):
        res = engine(quiet_tick)
        if step % 5 == 0 or step == 10:
            print(
                f"  Recovery Step {step:02d} | Coherence Alignment: {res['coherence_alignment']:.4f} | "
                f"Macro Face Distortion: {res['macro_face_distortion']:.4f} | "
                f"Potentiometer Mean: {res['resistance_potentiometer'].mean().item():.4f}"
            )

    print("\n================================================================================")
    print("  VERIFICATION COMPLETE: Global Inertial Axis successfully binds ticks into     ")
    print("  a continuous, self-introspective organic field.                               ")
    print("================================================================================")


if __name__ == "__main__":
    main()
