r"""
Elysia Engine: Emergent Phase Viscosity & Thermodynamic Operator Demo
=====================================================================
Demonstrates:
1. Reinterpretation of mathematical operators (\sum, \log, \nabla, \int) as dynamic S^3 state space transformers.
2. Self-emergent viscosity and shear resistance without hardcoded viscosity constants.
3. Continuous thermodynamic phase transitions (Solid \Phi ~ 1, Liquid \Phi ~ 0.5, Gas \Phi ~ 0) driven by Langevin thermal noise T.
4. Non-Newtonian regime behaviors (shear-thinning and shear-thickening).
5. Unified Optical Shading & Raymarching shader mapping.
"""

import torch
from core.physics.emergent_phase_viscosity import (
    EmergentPhaseViscosityEngine,
    generate_glsl_volume_raymarch_shader,
    generate_hlsl_compute_shader,
    generate_max_mipmaps_3d
)


def run_demo():
    print("=====================================================================")
    print("      Elysia Engine: Emergent Phase Viscosity & Operator Engine")
    print("=====================================================================\n")

    # Domain Setup: 3D Grid [Batch=1, Channels=3, H=8, W=8, D=8]
    B, H, W, D = 1, 8, 8, 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Running on: {device}\n")

    # 1. Initialize Engine
    engine = EmergentPhaseViscosityEngine(K_0=15.0, D_R=0.2, shear_mode="newtonian")

    # 2. Setup Initial Velocity Field with Shear Flow (v_x = y)
    V = torch.zeros(B, 3, H, W, D, device=device)
    y_profile = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, W, 1).expand(B, H, W, D)
    V[:, 0, :, :, :] = y_profile  # Shear gradient

    print("[1] Reinterpreted Mathematical Operators in Action:")
    print(r"    - Derivative (\nabla \times): Extracting spatial shear vorticity gradient")
    print(r"    - Sigma (\sum_{phase}): S^3 Kuramoto quaternion synchronization torque")
    print(r"    - Log (\log_{scale}): Logarithmic energy scale compression and dissipation")
    print(r"    - Integral (\int / Div): Unified stress tensor momentum feedback")
    print()

    # 3. Simulate Thermodynamic Regime Transitions across Temperature Spectrum
    temperatures = [0.001, 5.0, 50.0]
    regime_names = ["Solid Regime (Low T)", "Liquid Regime (Mid T)", "Gas Regime (High T)"]

    for T_val, regime in zip(temperatures, regime_names):
        T_field = torch.ones(B, 1, H, W, D, device=device) * T_val
        # Add thermal random rotor alignment corresponding to T
        if T_val > 1.0:
            Q_state = torch.randn(B, 4, H, W, D, device=device) * (T_val ** 0.5)
            Q_state[:, 0] += 1.0
            Q_state = torch.nn.functional.normalize(Q_state, p=2, dim=1)
        else:
            Q_state = torch.zeros(B, 4, H, W, D, device=device)
            Q_state[:, 0] = 1.0

        V_step, Q_step, metrics = engine.step(V, Q_state, T_field, dt=0.01)

        print(f"--- {regime} (T = {T_val}) ---")
        print(f"  Order Parameter Phi:  {metrics['mean_order_parameter_phi']:.4f}")
        print(f"  Solid Fraction:       {metrics['solid_fraction'] * 100:.1f}%")
        print(f"  Liquid Fraction:      {metrics['liquid_fraction'] * 100:.1f}%")
        print(f"  Gas Fraction:         {metrics['gas_fraction'] * 100:.1f}%")
        print(f"  Sync Torque magnitude:{metrics['mean_torque_sync']:.4f}")
        print(f"  Dissipation alpha:    {metrics['mean_dissipation_alpha']:.4f}\n")

    # 4. Demonstrate Non-Newtonian Behavior Comparison
    print("[2] Non-Newtonian Rheological Behavior Comparison under High Shear:")
    thin_engine = EmergentPhaseViscosityEngine(K_0=15.0, shear_mode="shear_thinning")
    thick_engine = EmergentPhaseViscosityEngine(K_0=15.0, shear_mode="shear_thickening")

    T_fluid = torch.ones(B, 1, H, W, D, device=device) * 1.0
    Q_init = torch.zeros(B, 4, H, W, D, device=device)
    Q_init[:, 0] = 1.0

    _, _, thin_metrics = thin_engine.step(V * 3.0, Q_init, T_fluid)
    _, _, thick_metrics = thick_engine.step(V * 3.0, Q_init, T_fluid)

    print(f"  Shear-Thinning Dissipation (Shampoo/Paint):  {thin_metrics['mean_dissipation_alpha']:.4f}")
    print(f"  Shear-Thickening Dissipation (Oobleck):      {thick_metrics['mean_dissipation_alpha']:.4f}\n")

    # 5. Empty Space Skipping Mipmaps & Optical Shader Code Generation
    print("[3] Unified Optical Shader Generation & Space Skipping Mipmaps:")
    glsl_shader = generate_glsl_volume_raymarch_shader()
    hlsl_shader = generate_hlsl_compute_shader()

    print(f"  GLSL Volume Shader lines generated: {len(glsl_shader.splitlines())}")
    print(f"  HLSL Compute Shader lines generated: {len(hlsl_shader.splitlines())}")

    # Compute Max Mipmap
    Phi_sample = torch.rand(B, 1, 8, 8, 8, device=device)
    Phi_mip = generate_max_mipmaps_3d(Phi_sample, brick_size=4)
    print(f"  Hierarchical 3D Max-Mipmap shape generated: {list(Phi_mip.shape)}")

    print("\n=====================================================================")
    print("      Elysia Engine: Emergent Phase Viscosity Demo Complete!")
    print("=====================================================================")


if __name__ == "__main__":
    run_demo()
