"""
scripts/verify_step3_hyper_dimensional_blister_nebula.py

Step 3 Verification & Simulation Script:
Hyper-Dimensional Conceptual-Optical Volume Nebula & Optical Hyper-Slicer Projection
===================================================================================
Simulates:
1. 4D Volumetric Non-linear Blister Snap & Distance-vs-Force Curve past Omega_c = 0.25 (>10x ratio).
2. Phase Interference (cos Delta phi): Constructive Meniscus Snapping vs Destructive Repulsion.
3. 4D Nebula Multi-Scale Evolution:
   - Micro-depth (w < 0): dispersed spectral rotors & wavelets.
   - Transition Snap (w -> 0): rapid non-linear clumping.
   - Macro-surface (w > 0): cohesive 16K-scale resolution-independent volumetric structure.
4. Optical Hyper-Slicer & Prism Projections across R_obs rotation angles.
5. Intent Potential Shock (Delta V) Injection & Phase Rupture -> Organic Self-Reorganization.
6. Generates high-resolution visualization artifact `step3_hyper_dimensional_blister_nebula.png`.
"""

import os
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import elysia_loader
elysia_loader.install()

from elysia_engine.core.hyper_dimensional_blister_nebula import (
    Clifford4DRotor,
    HyperDimensionalOpticalRotor,
    HyperDimensionalBlisterKernel,
    OpticalHyperSlicer,
    HyperDimensionalBlisterNebulaEngine
)


def sweep_4d_distance_vs_force(kernel: HyperDimensionalBlisterKernel) -> Dict[str, np.ndarray]:
    """Sweeps 4D Euclidean distance to verify non-linear snap past Omega_c."""
    distances = np.linspace(0.2, 3.2, 150)
    overlaps = []
    forces_total = []
    forces_snap = []

    r1 = HyperDimensionalOpticalRotor(wavelength_nm=550.0, position_4d=[0.0, 0.0, 0.0, 0.0])
    r2 = HyperDimensionalOpticalRotor(wavelength_nm=550.0, position_4d=[1.0, 0.0, 0.0, 0.0])

    r_sum = r1.penumbra_radius + r2.penumbra_radius

    for d in distances:
        r2.position = np.array([d, 0.0, 0.0, 0.0], dtype=np.float32)
        res = kernel.evaluate_interaction(r1, r2)
        overlaps.append(res["overlap"])
        forces_total.append(res["force_total"])
        forces_snap.append(res["snap_force"])

    return {
        "distances": distances,
        "overlaps": np.array(overlaps),
        "forces_total": np.array(forces_total),
        "forces_snap": np.array(forces_snap),
        "r_sum": r_sum,
        "critical_distance": r_sum * (1.0 - kernel.omega_c)
    }


def sweep_4d_phase_interference(kernel: HyperDimensionalBlisterKernel) -> Dict[str, np.ndarray]:
    """Sweeps phase difference Delta phi from -pi to +pi."""
    delta_phases = np.linspace(-np.pi, np.pi, 120)
    forces = []
    interferences = []

    r1 = HyperDimensionalOpticalRotor(wavelength_nm=600.0, position_4d=[0.0, 0.0, 0.0, 0.0])
    r2 = HyperDimensionalOpticalRotor(wavelength_nm=600.0, position_4d=[0.8, 0.0, 0.0, 0.0])

    for phi in delta_phases:
        r1.phase_theta = 0.0
        r2.phase_theta = phi
        res = kernel.evaluate_interaction(r1, r2)
        forces.append(res["force_total"])
        interferences.append(res["interference"])

    return {
        "delta_phases": delta_phases,
        "forces": np.array(forces),
        "interferences": np.array(interferences)
    }


def simulate_nebula_evolution_and_intent_shock(
    steps: int = 150,
    dt: float = 0.02
) -> Tuple[HyperDimensionalBlisterNebulaEngine, Dict[str, List[float]]]:
    """
    Initializes a cluster of 4D cognitive-optical nodes spanning
    micro-depth (w < 0), transition (w ~ 0), and macro (w > 0).
    Runs evolution, injects intent shock at t=1.5s (step 75), and records trajectories.
    """
    kernel = HyperDimensionalBlisterKernel(omega_c=0.25, gamma_power=2.5, kappa_0=1.0, kappa_1=14.0)
    slicer = OpticalHyperSlicer(resolution_3d=36, resolution_2d=128)
    engine = HyperDimensionalBlisterNebulaEngine(kernel=kernel, slicer=slicer)

    # Populate 12 nodes across scale w with varying wavelengths
    wavelengths = [420.0, 480.0, 520.0, 550.0, 580.0, 640.0, 680.0, 710.0, 450.0, 530.0, 600.0, 670.0]
    w_coords = np.linspace(-0.8, 0.8, 12)

    np.random.seed(42)
    for i, (wl, w) in enumerate(zip(wavelengths, w_coords)):
        # Spatial positions dispersed
        pos = np.array([
            0.6 * math.cos(i * 0.5) + np.random.uniform(-0.1, 0.1),
            0.6 * math.sin(i * 0.5) + np.random.uniform(-0.1, 0.1),
            0.3 * math.sin(i * 1.2),
            w
        ], dtype=np.float32)
        node = HyperDimensionalOpticalRotor(
            wavelength_nm=wl,
            position_4d=pos,
            intensity=1.0,
            intent_potential=0.0
        )
        engine.add_node(node)

    history = {
        "time": [],
        "coherence": [],
        "active_blisters": [],
        "mean_distance": [],
        "intent_shock_marker": []
    }

    shock_applied = False
    for step in range(steps):
        t = step * dt

        # Inject intent shock at step 75
        if step == 75:
            # Apply shock and phase inversion to nodes 0..4 (micro-scale nodes)
            engine.inject_intent_potential_shock([0, 1, 2, 3], delta_potential=3.0, phase_inversion=True)
            shock_applied = True

        res = engine.step(dt=dt)

        # Compute mean distance between nodes
        dists = []
        for i in range(len(engine.nodes)):
            for j in range(i + 1, len(engine.nodes)):
                dists.append(np.linalg.norm(engine.nodes[i].position - engine.nodes[j].position))
        mean_d = float(np.mean(dists)) if dists else 0.0

        history["time"].append(t)
        history["coherence"].append(res["coherence"])
        history["active_blisters"].append(res["active_blisters"])
        history["mean_distance"].append(mean_d)
        history["intent_shock_marker"].append(1.0 if step >= 75 and step <= 80 else 0.0)

    return engine, history


def render_verification_dashboard(
    data_force: Dict[str, np.ndarray],
    data_phase: Dict[str, np.ndarray],
    history: Dict[str, List[float]],
    engine: HyperDimensionalBlisterNebulaEngine,
    output_path: str = "step3_hyper_dimensional_blister_nebula.png"
):
    """Renders a comprehensive 6-panel verification dashboard."""
    fig = plt.figure(figsize=(18, 12), dpi=140)
    fig.patch.set_facecolor("#0e1117")

    title_text = r"Step 3: Hyper-Dimensional Conceptual-Optical Volume Nebula ($C\ell(4,0)$ & Optical Hyper-Slicer)"
    plt.suptitle(title_text, fontsize=16, fontweight="bold", color="#f0f6fc", y=0.98)

    # Panel 1: 4D Distance vs Non-Linear Blister Snap Force
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor("#161b22")
    dists = data_force["distances"]
    f_tot = data_force["forces_total"]
    f_snap = data_force["forces_snap"]
    crit_d = data_force["critical_distance"]

    ax1.plot(dists, f_tot, label="Total Force $F_{total}$", color="#58a6ff", lw=2.2)
    ax1.plot(dists, f_snap, label="Non-linear Snap $F_{snap}$", color="#f778ba", ls="--", lw=1.8)
    ax1.axvline(crit_d, color="#ff7b72", ls="-.", alpha=0.8, label=r"Critical Overlap $\Omega_c=0.25$")
    ax1.set_title("4D Volumetric Blister Snap (>10x Ratio)", color="#f0f6fc", fontsize=11)
    ax1.set_xlabel("4D Euclidean Distance $r_{4d}$", color="#8b949e")
    ax1.set_ylabel("Attraction Force", color="#8b949e")
    ax1.tick_params(colors="#8b949e")
    ax1.grid(True, color="#30363d", alpha=0.5)
    ax1.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

    # Panel 2: Phase Interference (Constructive Meniscus vs Destructive Repulsion)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor("#161b22")
    phases = data_phase["delta_phases"]
    forces_p = data_phase["forces"]

    ax2.plot(phases, forces_p, color="#3fb950", lw=2.2, label=r"Interference $F(\Delta\phi)$")
    ax2.axhline(0.0, color="#8b949e", ls=":", alpha=0.6)
    ax2.fill_between(phases, forces_p, 0, where=(forces_p > 0), color="#238636", alpha=0.3, label="Meniscus Snap (Attraction)")
    ax2.fill_between(phases, forces_p, 0, where=(forces_p < 0), color="#da3633", alpha=0.3, label="Purification (Repulsion)")
    ax2.set_title(r"Phase Interference Modulation $\cos(\Delta\phi)$", color="#f0f6fc", fontsize=11)
    ax2.set_xlabel(r"Phase Difference $\Delta\phi$ (rad)", color="#8b949e")
    ax2.set_ylabel("Total Force", color="#8b949e")
    ax2.tick_params(colors="#8b949e")
    ax2.grid(True, color="#30363d", alpha=0.5)
    ax2.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

    # Panel 3: Dynamic Intent Potential Shock & Re-equilibration
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor("#161b22")
    t = history["time"]
    coherence = history["coherence"]
    active_b = history["active_blisters"]

    ax3.plot(t, coherence, color="#79c0ff", lw=2.2, label="Global Phase Coherence $R$")
    ax3.axvline(1.5, color="#d29922", ls="--", lw=1.8, label=r"Intent Shock $\Delta V$ (t=1.5s)")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(t, active_b, color="#ffa657", ls=":", lw=1.6, label="Active Blisters")
    ax3_twin.set_ylabel("Active Blister Pairs", color="#ffa657")
    ax3_twin.tick_params(colors="#ffa657")
    ax3.set_title("Intent Potential Shock & Organic Recovery", color="#f0f6fc", fontsize=11)
    ax3.set_xlabel("Time (s)", color="#8b949e")
    ax3.set_ylabel("Coherence Order $R$", color="#79c0ff")
    ax3.tick_params(colors="#8b949e")
    ax3.grid(True, color="#30363d", alpha=0.5)
    ax3.legend(loc="lower left", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

    # Panel 4: Optical Hyper-Slicer Cross Section 1 (Micro-depth w = -0.4)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor("#161b22")
    slice_micro = engine.slicer.slice_4d_to_3d_volume(engine.nodes, engine.kernel, w_slice=-0.4, rotor_obs=engine.observer_rotor)
    proj_micro = engine.slicer.project_to_2d_prism(slice_micro)
    ax4.imshow(proj_micro["rgb_2d"])
    ax4.set_title("Micro-Depth Slice ($w=-0.4$, Dispersed Wavelets)", color="#f0f6fc", fontsize=11)
    ax4.axis("off")

    # Panel 5: Optical Hyper-Slicer Cross Section 2 (Macro-surface w = +0.2)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor("#161b22")
    slice_macro = engine.slicer.slice_4d_to_3d_volume(engine.nodes, engine.kernel, w_slice=0.2, rotor_obs=engine.observer_rotor)
    proj_macro = engine.slicer.project_to_2d_prism(slice_macro)
    ax5.imshow(proj_macro["rgb_2d"])
    ax5.set_title("Macro-Surface Slice ($w=+0.2$, Fused Meniscus)", color="#f0f6fc", fontsize=11)
    ax5.axis("off")

    # Panel 6: Edge Sharpness Gradient Magnitude (Blister vs Gaussian Blur)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor("#161b22")
    # Compare blister edge sharpness against blurred representation
    grad_blister = proj_macro["grad_mag"]
    from scipy.ndimage import gaussian_filter
    blurred_density = gaussian_filter(proj_macro["density_2d"], sigma=2.0)
    gy_b, gx_b = np.gradient(blurred_density)
    grad_blur = np.sqrt(gx_b**2 + gy_b**2)

    im6 = ax6.imshow(grad_blister, cmap="inferno")
    fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04).ax.tick_params(colors="#8b949e")
    max_b = np.max(grad_blister)
    max_l = np.max(grad_blur)
    ax6.set_title(f"Edge Sharpness Gradient (Peak Blister {max_b:.2f} vs Blur {max_l:.2f})", color="#f0f6fc", fontsize=10)
    ax6.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[+] Step 3 Verification Dashboard successfully exported to {output_path}")


def main():
    print("=" * 70)
    print("Starting Step 3: Hyper-Dimensional Blister Nebula Verification")
    print("=" * 70)

    kernel = HyperDimensionalBlisterKernel()
    data_force = sweep_4d_distance_vs_force(kernel)
    data_phase = sweep_4d_phase_interference(kernel)

    print("[*] Running 4D Nebula Multi-Scale Evolution & Intent Shock simulation...")
    engine, history = simulate_nebula_evolution_and_intent_shock(steps=150, dt=0.02)

    print("[*] Rendering verification dashboard artifact...")
    output_png = "step3_hyper_dimensional_blister_nebula.png"
    render_verification_dashboard(data_force, data_phase, history, engine, output_path=output_png)

    # Verification checks
    # 1. Non-linear snap ratio
    overlaps = data_force["overlaps"]
    f_snap = data_force["forces_snap"]
    f_tot = data_force["forces_total"]
    sub_idx = np.where((overlaps > 0.1) & (overlaps < 0.24))[0][0]
    super_idx = np.where(overlaps > 0.8)[0][0]
    snap_ratio = f_tot[super_idx] / f_tot[sub_idx]
    print(f"[Check 1] Snap ratio super/sub: {snap_ratio:.2f}x (Expect >10x)")

    # 2. In-phase vs anti-phase
    in_idx = np.argmin(np.abs(data_phase["delta_phases"]))
    anti_idx = np.argmin(np.abs(data_phase["delta_phases"] - np.pi))
    print(f"[Check 2] In-phase force: {data_phase['forces'][in_idx]:.2f}, Anti-phase force: {data_phase['forces'][anti_idx]:.2f}")

    # 3. Post-shock recovery
    post_shock_coherence = history["coherence"][-1]
    print(f"[Check 3] Final post-shock coherence order R: {post_shock_coherence:.4f}")

    print("=" * 70)
    print("Step 3 Verification Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
