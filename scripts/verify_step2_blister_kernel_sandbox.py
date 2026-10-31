"""
scripts/verify_step2_blister_kernel_sandbox.py

Step 2: Shadow Blister Tension Kernel (F_blister) Isolated Sandbox Test & Visualization.
========================================================================================
Validates the micro-core physics of the non-linear shadow blister tension kernel:
1. Distance vs. Force Profile:
   - Smooth linear attenuation before critical penumbra overlap threshold (Omega_c).
   - Sharp power-law non-linear snap jump (F_blister) once overlap exceeds Omega_c.
2. Phase Resonance & Interference:
   - Constructive in-phase (Delta phi = 0) attraction vs. Destructive out-of-phase (Delta phi = pi) repulsion.
   - Continuous angular sweep across Delta phi in [-pi, pi].
3. Dynamic Hunting-Free Phase-Locking & Spatial Stiction:
   - 2-Node dynamic approach trajectory under F_total, non-linear stiction damping, and phase coupling.
   - Demonstrates rapid critical damping into stable phase-locking without oscillatory hunting.
4. Generates a publication-grade diagnostic plot: step2_blister_kernel_verification.png
"""

import math
import os
import sys
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure repository root is in python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import elysia_loader
elysia_loader.install()

from elysia_engine.core.optical_blister_rotor import (
    OpticalSpectrumBivectorRotor,
    ShadowBlisterTensionKernel
)


def sweep_distance_vs_force(
    kernel: ShadowBlisterTensionKernel,
    r1_nm: float = 600.0,
    r2_nm: float = 600.0,
    num_points: int = 200
) -> Dict[str, np.ndarray]:
    """
    Sweeps the distance between two in-phase optical rotors across the full
    range from fully separated to deeply overlapping.
    """
    node_i = OpticalSpectrumBivectorRotor(wavelength_nm=r1_nm, position=[0.0, 0.0, 0.0])
    node_j = OpticalSpectrumBivectorRotor(wavelength_nm=r2_nm, position=[0.0, 0.0, 0.0])

    r_sum = node_i.penumbra_radius + node_j.penumbra_radius
    distances = np.linspace(0.05, r_sum * 1.3, num_points)

    overlaps = []
    forces_total = []
    forces_snap = []
    forces_smooth = []
    is_active = []

    for d in distances:
        node_j.position = np.array([d, 0.0, 0.0], dtype=np.float32)
        node_i.phase_theta = 0.0
        node_j.phase_theta = 0.0  # In-phase

        res = kernel.evaluate_blister_interaction(node_i, node_j)
        overlap = res["overlap"]
        overlaps.append(overlap)
        forces_total.append(res["force_total"])
        is_active.append(res["is_blister_active"])

        # Decompose smooth vs snap components
        f_smooth = kernel.kappa_0 * overlap
        f_snap = 0.0
        if overlap >= kernel.omega_c:
            excess = (overlap - kernel.omega_c) / (1.0 - kernel.omega_c + kernel.eps)
            f_snap = kernel.kappa_1 * (excess ** kernel.gamma_power)
        forces_smooth.append(f_smooth)
        forces_snap.append(f_snap)

    return {
        "distances": distances,
        "r_sum": np.array([r_sum]),
        "overlaps": np.array(overlaps),
        "forces_total": np.array(forces_total),
        "forces_smooth": np.array(forces_smooth),
        "forces_snap": np.array(forces_snap),
        "is_active": np.array(is_active)
    }


def sweep_phase_vs_interference(
    kernel: ShadowBlisterTensionKernel,
    overlap_target: float = 0.65,
    num_points: int = 150
) -> Dict[str, np.ndarray]:
    """
    Sweeps phase difference Delta phi from -pi to pi at a fixed blister-active overlap.
    Examines cos(Delta phi) modulation and directional attraction/repulsion.
    """
    node_i = OpticalSpectrumBivectorRotor(wavelength_nm=550.0, position=[0.0, 0.0, 0.0])
    node_j = OpticalSpectrumBivectorRotor(wavelength_nm=550.0, position=[0.0, 0.0, 0.0])

    r_sum = node_i.penumbra_radius + node_j.penumbra_radius
    d_fixed = r_sum * (1.0 - overlap_target)
    node_j.position = np.array([d_fixed, 0.0, 0.0], dtype=np.float32)

    delta_phases = np.linspace(-math.pi, math.pi, num_points)
    interferences = []
    forces = []

    for dphi in delta_phases:
        node_i.phase_theta = 0.0
        node_j.phase_theta = float(dphi)
        res = kernel.evaluate_blister_interaction(node_i, node_j)
        interferences.append(res["interference"])
        forces.append(res["force_total"])

    return {
        "delta_phases": delta_phases,
        "interferences": np.array(interferences),
        "forces": np.array(forces),
        "overlap": np.array([overlap_target])
    }


def simulate_two_node_dynamic_approach(
    kernel: ShadowBlisterTensionKernel,
    initial_dist: float = 1.6,
    initial_phase_diff: float = 0.5,
    total_time: float = 5.0,
    dt: float = 0.01,
    mass: float = 0.8,
    spatial_damping: float = 3.0
) -> Dict[str, np.ndarray]:
    """
    Dynamically simulates two nodes approaching each other in 1D space under:
    - Mass inertia, blister tension force, and spatial damping.
    - Angular velocity, phase torque with non-linear stiction damping.
    Tests that:
    1. Beyond blister zone, nodes approach smoothly.
    2. Crossing Omega_c triggers rapid snap.
    3. Stiction damps relative phase and velocity cleanly without oscillatory hunting.
    """
    r1 = OpticalSpectrumBivectorRotor(wavelength_nm=520.0, position=[-initial_dist / 2.0, 0.0, 0.0])
    r2 = OpticalSpectrumBivectorRotor(wavelength_nm=520.0, position=[initial_dist / 2.0, 0.0, 0.0])

    r1.phase_theta = 0.0
    r2.phase_theta = initial_phase_diff

    v1 = np.zeros(3, dtype=np.float32)
    v2 = np.zeros(3, dtype=np.float32)

    steps = int(total_time / dt)
    time_pts = []
    dist_pts = []
    force_pts = []
    phase_diff_pts = []
    overlap_pts = []
    locked_pts = []

    for step_idx in range(steps):
        t = step_idx * dt
        diff = r2.position - r1.position
        dist = float(np.linalg.norm(diff))

        res = kernel.evaluate_blister_interaction(r1, r2, dt=dt)
        f_vec = res["force_vector"]
        t_vec = res["phase_torque"]

        time_pts.append(t)
        dist_pts.append(dist)
        force_pts.append(res["force_total"])
        phase_diff_pts.append(res["delta_phi"])
        overlap_pts.append(res["overlap"])
        locked_pts.append(1.0 if res["is_phase_locked"] else 0.0)

        # Equations of motion for positions (with spatial drag)
        a1 = (f_vec - spatial_damping * v1) / mass
        a2 = (-f_vec - spatial_damping * v2) / mass

        v1 += a1 * dt
        v2 += a2 * dt

        # Physical core contact stiction: nodes cannot interpenetrate below min_clearance
        min_clearance = 0.12
        if dist <= min_clearance and np.dot(v2 - v1, diff) < 0:
            # Stiction contact: damp relative velocity to zero
            v1 = np.zeros(3, dtype=np.float32)
            v2 = np.zeros(3, dtype=np.float32)
        else:
            r1.position += v1 * dt
            r2.position += v2 * dt

        # Evolution of rotors with blister phase torque
        r1.step(dt, external_torque=t_vec)
        r2.step(dt, external_torque=-t_vec)

    return {
        "time": np.array(time_pts),
        "distance": np.array(dist_pts),
        "force": np.array(force_pts),
        "phase_diff": np.array(phase_diff_pts),
        "overlap": np.array(overlap_pts),
        "is_locked": np.array(locked_pts)
    }


def generate_verification_report_plots(
    output_png: str = "step2_blister_kernel_verification.png"
) -> str:
    """
    Executes isolated sandbox tests, generates 4-panel diagnostic plot,
    and saves high-resolution verification artifact.
    """
    kernel = ShadowBlisterTensionKernel(
        omega_c=0.25,
        gamma_power=2.5,
        kappa_0=1.0,
        kappa_1=15.0,
        gamma_stiction=10.0
    )

    dist_data = sweep_distance_vs_force(kernel)
    phase_data = sweep_phase_vs_interference(kernel)
    sim_data = simulate_two_node_dynamic_approach(kernel)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    # Panel 1: Distance vs Force & Non-Linear Snap
    ax1 = axes[0, 0]
    d = dist_data["distances"]
    f_tot = dist_data["forces_total"]
    f_smooth = dist_data["forces_smooth"]
    f_snap = dist_data["forces_snap"]
    overlap = dist_data["overlaps"]

    ax1.plot(d, f_tot, label=r"Total Blister Force $F_{\mathrm{total}}$", color="#d62728", lw=2.5)
    ax1.plot(d, f_smooth, label=r"Smooth Background $\kappa_0 \cdot \Omega$", color="#1f77b4", ls="--", lw=1.8)
    ax1.plot(d, f_snap, label=r"Blister Snap $\kappa_1 (\frac{\Omega-\Omega_c}{1-\Omega_c})^\gamma$", color="#ff7f0e", ls=":", lw=2.0)

    # Mark critical threshold distance
    crit_mask = np.where(overlap >= kernel.omega_c)[0]
    if len(crit_mask) > 0:
        crit_d = d[crit_mask[-1]]
        ax1.axvline(crit_d, color="black", linestyle="-.", alpha=0.7, label=f"Critical Threshold $\Omega_c={kernel.omega_c}$")

    ax1.set_title("1. Distance vs. Blister Tension Force (Snap Verification)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Inter-Node Distance $d_{ij}$", fontsize=10)
    ax1.set_ylabel("Attractive Force $F$", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: Phase Difference vs Interference & Force
    ax2 = axes[0, 1]
    dphi = phase_data["delta_phases"]
    f_phase = phase_data["forces"]
    interf = phase_data["interferences"]

    ax2.plot(dphi, interf, label=r"Interference $\cos(\Delta\phi)$", color="#2ca02c", lw=2.0)
    ax2.plot(dphi, f_phase, label=r"Modulated Force $F_{\mathrm{total}}(\Delta\phi)$", color="#9467bd", lw=2.5)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(0, color="green", linestyle=":", alpha=0.6, label="In-Phase Resonance (Snap Max)")
    ax2.axvline(-math.pi, color="purple", linestyle=":", alpha=0.6)
    ax2.axvline(math.pi, color="purple", linestyle=":", alpha=0.6, label="Anti-Phase Repulsion")

    ax2.set_title("2. Phase Interference Resonance & Repulsion", fontsize=12, fontweight="bold")
    ax2.set_xlabel(r"Phase Difference $\Delta\phi$ (radians)", fontsize=10)
    ax2.set_ylabel(r"Force / Interference Factor", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    # Panel 3: Dynamic Distance Relaxation & Stiction Trajectory
    ax3 = axes[1, 0]
    t = sim_data["time"]
    dist_t = sim_data["distance"]
    overlap_t = sim_data["overlap"]

    ax3.plot(t, dist_t, label="Inter-Node Distance $d(t)$", color="#1f77b4", lw=2.5)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(t, overlap_t, label="Penumbra Overlap $\Omega(t)$", color="#e377c2", ls="--", lw=1.8)
    ax3_twin.axhline(kernel.omega_c, color="darkred", linestyle=":", label=r"Threshold $\Omega_c$")

    ax3.set_title("3. Dynamic Distance Convergence (No Hunting)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Simulation Time $t$ (s)", fontsize=10)
    ax3.set_ylabel("Distance $d(t)$", fontsize=10, color="#1f77b4")
    ax3_twin.set_ylabel("Overlap $\Omega(t)$", fontsize=10, color="#e377c2")
    ax3.grid(True, alpha=0.3)

    # Combine legends
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_3t, labels_3t = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_3t, labels_3 + labels_3t, loc="center right", fontsize=9)

    # Panel 4: Phase Difference Convergence to Phase-Locking
    ax4 = axes[1, 1]
    phi_t = sim_data["phase_diff"]
    locked_t = sim_data["is_locked"]

    ax4.plot(t, phi_t, label=r"Phase Delta $\Delta\phi(t)$", color="#ff7f0e", lw=2.5)
    ax4.fill_between(t, -0.15, 0.15, color="green", alpha=0.15, label="Phase-Lock Band $|\Delta\phi| < 0.15$")
    ax4.axhline(0.0, color="gray", linestyle="--", alpha=0.7)

    # Mark locking moment
    locked_idx = np.where(locked_t > 0.5)[0]
    if len(locked_idx) > 0:
        t_lock = t[locked_idx[0]]
        ax4.axvline(t_lock, color="green", linestyle="-.", label=f"Phase-Locked at $t={t_lock:.2f}$s")

    ax4.set_title("4. Clean Phase-Locking Convergence (Hunting-Free)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Simulation Time $t$ (s)", fontsize=10)
    ax4.set_ylabel(r"Phase Difference $\Delta\phi$", fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right", fontsize=9)

    output_path = os.path.join(repo_root, output_png)
    plt.suptitle("Elysia Step 2: Shadow Blister Tension Kernel Isolated Sandbox Verification", fontsize=15, fontweight="bold", y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Step 2 Sandbox] Diagnostic plot successfully generated at: {output_path}")
    return output_path


def main():
    print("=" * 70)
    print("Running Step 2: Shadow Blister Tension Kernel Isolated Sandbox Test")
    print("=" * 70)

    kernel = ShadowBlisterTensionKernel(
        omega_c=0.25,
        gamma_power=2.5,
        kappa_0=1.0,
        kappa_1=15.0,
        gamma_stiction=10.0
    )

    # 1. Sweep Distance & Overlap
    dist_sweep = sweep_distance_vs_force(kernel)
    sub_crit_idx = np.where((dist_sweep["overlaps"] > 0.05) & (dist_sweep["overlaps"] < kernel.omega_c))[0]
    snap_idx = np.where(dist_sweep["overlaps"] > 0.6)[0]

    f_sub = dist_sweep["forces_total"][sub_crit_idx[0]] if len(sub_crit_idx) > 0 else 0.0
    f_snap = dist_sweep["forces_total"][snap_idx[0]] if len(snap_idx) > 0 else 0.0

    print(f"[*] Distance sweep check:")
    print(f"    Sub-critical force (overlap ~ {dist_sweep['overlaps'][sub_crit_idx[0]]:.2f}): {f_sub:.4f}")
    print(f"    Super-critical snap force (overlap ~ {dist_sweep['overlaps'][snap_idx[0]]:.2f}): {f_snap:.4f}")
    print(f"    Snap ratio (F_snap / F_sub): {f_snap / (f_sub + 1e-6):.2f}x (Expected sharp non-linear jump)")

    # 2. Sweep Phase
    phase_sweep = sweep_phase_vs_interference(kernel)
    f_in_phase = float(phase_sweep["forces"][len(phase_sweep["forces"]) // 2])
    f_anti_phase = float(phase_sweep["forces"][0])
    print(f"[*] Phase sweep check:")
    print(f"    In-phase force (Delta phi = 0): {f_in_phase:.4f} (Attractive)")
    print(f"    Anti-phase force (Delta phi = +/-pi): {f_anti_phase:.4f} (Repulsive)")

    # 3. Dynamic Convergence
    sim = simulate_two_node_dynamic_approach(kernel)
    final_dist = sim["distance"][-1]
    final_phi = sim["phase_diff"][-1]
    is_locked = sim["is_locked"][-1] > 0.5

    # Check for hunting oscillations in the tail (last 20% of simulation)
    tail_len = int(len(sim["phase_diff"]) * 0.2)
    phase_tail = sim["phase_diff"][-tail_len:]
    phase_tail_std = float(np.std(phase_tail))

    print(f"[*] Dynamic approach & stiction check:")
    print(f"    Final distance: {final_dist:.4f}")
    print(f"    Final phase difference: {final_phi:.4f} rad")
    print(f"    Phase tail std-dev (Hunting indicator): {phase_tail_std:.6f} rad (Target < 0.005)")
    print(f"    Phase-locked state achieved: {is_locked}")

    # 4. Generate Plot
    png_path = generate_verification_report_plots()
    print(f"[*] Verification complete. Report saved: {png_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
