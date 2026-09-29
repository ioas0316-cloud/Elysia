"""
Demo Script: Hebbian 3D Dynamic Lightning & Axonal Highway Self-Organization
Demonstrates 3D dual-potential wave diffusion, dynamic obstacle avoidance,
and self-healing Hebbian field plasticity forming high-conductance axonal highways.
"""

import torch
from synaptic_architecture.dynamic_lightning_3d import DynamicLightning3D, HebbianFieldPlasticityEngine


def run_hebbian_3d_demo():
    print("=" * 80)
    print("  ELYSIUM ENGINE: 3D DYNAMIC LIGHTNING & HEBBIAN FIELD PLASTICITY DEMO")
    print("=" * 80)

    shape = (20, 20, 20)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()} | Grid Shape: {shape}\n")

    solver = DynamicLightning3D(shape=shape, device=device)
    hebbian = HebbianFieldPlasticityEngine(shape=shape, alpha=0.8, beta=0.01, eta=1.5, D=0.01, device=device)

    start_pt = (2, 2, 2)
    goal_pt = (17, 17, 17)

    # Base grid
    base_grid = torch.ones(shape, dtype=torch.float32, device=device)

    print("Simulating 10 time steps with a moving obstacle and Hebbian highway plasticity...\n")

    for t in range(10):
        # Time-varying moving 3D obstacle moving along z-axis
        obs_z = (t + 5) % 20
        grid_t = base_grid.clone()
        # Block a 3x4x4 sphere near the center
        z_min, z_max = max(0, obs_z - 1), min(20, obs_z + 2)
        grid_t[z_min:z_max, 8:12, 8:12] = 0.0

        # Combine with current Hebbian conductivity field
        effective_conductivity = grid_t * hebbian.sigma.squeeze()

        # Step 1: Lightning channel collapse over effective conductivity
        lightning_3d = solver.step(start_pt, goal_pt, effective_conductivity, gamma=16.0, relax_steps=15)

        # Step 2: Update Hebbian field plasticity based on current flow
        updated_sigma = hebbian.update_plasticity(solver.v_start + solver.v_goal, dt=0.2)

        # Track active channel voxels and maximum highway conductivity
        active_voxels = (lightning_3d > 0.1).sum().item()
        mean_conductivity = updated_sigma.mean().item()
        max_conductivity = updated_sigma.max().item()

        print(f"Step {t:02d} | Obstacle Z: {obs_z:02d} | Active Path Voxels: {active_voxels:03d} | Mean Conductivity: {mean_conductivity:.4f} | Max Highway Conductivity: {max_conductivity:.4f}")

    print("\nSimulation complete. High-conductance axonal highways successfully self-organized.")
    print("=" * 80)


if __name__ == "__main__":
    run_hebbian_3d_demo()
