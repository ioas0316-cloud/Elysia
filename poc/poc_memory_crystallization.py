import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def simulate_memory_crystallization():
    """
    Simulates the Memory Crystallization and Reminiscence in the 3D Tensor Field.
    - Dynamic Time Field
    - Friction (Reactive Power magnitude & change rate)
    - Crystallization at Emotional Threshold
    - Decay (Forgetting)
    - Evolving Crystals (Living memory)
    - Reminiscence / Flashback Mode
    """
    print("Starting Memory Crystallization Simulation...")
    # 1. Setup the 3D Space
    grid_size = 15
    x = torch.linspace(-5, 5, grid_size)
    y = torch.linspace(-5, 5, grid_size)
    z = torch.linspace(-5, 5, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # Base states
    complex_field = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.complex64)
    prev_imag_field = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.float32)

    # Crystal Memory Storage: stores the crystallized complex values
    crystal_field = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.complex64)
    is_crystal = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.bool)

    time_steps = 50
    wave_number = 1.5
    decay_rate = 0.85  # Slightly slower decay
    threshold = 1.2   # Increased threshold to make crystals sparse and meaningful
    crystal_evolution_rate = 0.05 # How much crystals change when hit by new waves

    # Stimulus sources (Events)
    events = [
        {"time": 5, "pos": (0.0, 0.0, 0.0), "energy": 1.5, "phase": 0.0},
        {"time": 15, "pos": (2.0, -2.0, 1.0), "energy": 1.2, "phase": np.pi/3},
        {"time": 25, "pos": (-2.0, 3.0, -1.0), "energy": 0.6, "phase": np.pi/2}, # Weak event (should decay, not crystallize)
        {"time": 35, "pos": (1.0, 1.0, -2.0), "energy": 2.5, "phase": np.pi},   # Strong event (should create dense crystals near source)
    ]

    history_max_friction = []
    history_crystal_count = []

    snapshots = {}

    for t in range(time_steps):
        # Apply decay to the current field (forgetting)
        complex_field = complex_field * decay_rate

        # Inject new events
        for event in events:
            if event["time"] == t:
                # Calculate distance from event
                R = torch.sqrt((X - event["pos"][0])**2 + (Y - event["pos"][1])**2 + (Z - event["pos"][2])**2)
                # Create wave. Decrease energy slightly over distance to localize the effect
                wave = (event["energy"] / (1.0 + 0.2*R)) * torch.exp(1j * (wave_number * R - t * 0.5 + event["phase"]))
                # Add to field
                complex_field += wave

        # 1. Calculate Friction
        current_imag = complex_field.imag
        imag_change_rate = torch.abs(current_imag - prev_imag_field)
        # Friction = Combination of absolute imaginary part and its rate of change
        friction = torch.abs(current_imag) * 0.4 + imag_change_rate * 0.6

        # 2. Crystallization
        # Find new spots that cross the threshold
        new_crystals_mask = (friction > threshold) & (~is_crystal)

        # Lock in the crystals
        is_crystal = is_crystal | new_crystals_mask
        # Store the current state as the crystal's memory
        crystal_field[new_crystals_mask] = complex_field[new_crystals_mask]

        # 3. Evolving Crystals (Living Memory)
        # Existing crystals interact with the current field and slightly shift
        existing_crystals_mask = is_crystal & (~new_crystals_mask)
        crystal_field[existing_crystals_mask] = crystal_field[existing_crystals_mask] * (1 - crystal_evolution_rate) + complex_field[existing_crystals_mask] * crystal_evolution_rate

        # Ensure the field itself retains the crystallized state strongly
        complex_field[is_crystal] = crystal_field[is_crystal]

        # Update previous state for next step
        prev_imag_field = current_imag.clone()

        # Record metrics
        history_max_friction.append(friction.max().item())
        history_crystal_count.append(is_crystal.sum().item())

        # Save snapshots
        if t == 10:
            snapshots['early'] = (complex_field.clone(), is_crystal.clone())
        elif t == 40:
            snapshots['late'] = (complex_field.clone(), is_crystal.clone())

    # 4. Reminiscence / Flashback Mode
    print("Initiating Flashback Mode...")
    flashback_field = torch.zeros_like(complex_field)

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                if is_crystal[i, j, k]:
                    pos = (x[i].item(), y[j].item(), z[k].item())
                    R = torch.sqrt((X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2)
                    R = R + 1e-5

                    crystal_val = crystal_field[i, j, k]
                    energy = torch.abs(crystal_val)
                    phase = torch.angle(crystal_val)

                    emission = (energy / (R + 1.0)) * torch.exp(1j * (wave_number * R + phase))
                    flashback_field += emission

    snapshots['flashback'] = (flashback_field.clone(), is_crystal.clone())

    # --- Visualization ---
    print("Generating Visualizations...")
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Elysia Memory Crystallization & Flashback", fontsize=18)

    z_slice = grid_size // 2

    # Plot 1: Metrics over time
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(history_max_friction, label="Max Friction", color='orange')
    ax1.axhline(threshold, color='red', linestyle='--', label="Threshold")
    ax1.set_title("Friction over Time")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Friction (Energy)")
    ax1.legend()

    ax2 = fig.add_subplot(2, 3, 4)
    ax2.plot(history_crystal_count, label="Crystal Count", color='purple')
    ax2.set_title("Memory Crystals Formation")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Number of Crystals")
    ax2.legend()

    def plot_slice(ax, field, crystal_mask, title):
        mag = torch.abs(field[:, :, z_slice]).numpy()
        c = ax.contourf(X[:, :, z_slice].numpy(), Y[:, :, z_slice].numpy(), mag, levels=20, cmap='magma')
        fig.colorbar(c, ax=ax)

        cx, cy = [], []
        for i in range(grid_size):
            for j in range(grid_size):
                if crystal_mask[i, j, z_slice]:
                    cx.append(X[i, j, z_slice].item())
                    cy.append(Y[i, j, z_slice].item())
        if cx:
            ax.scatter(cx, cy, color='cyan', marker='*', s=100, label='Crystals (Memories)', edgecolor='black')
            ax.legend(loc='upper right')

        ax.set_title(title)
        ax.set_xlabel("X Space")
        ax.set_ylabel("Y Space")

    # Plot 2: Early state
    ax3 = fig.add_subplot(2, 3, 2)
    plot_slice(ax3, snapshots['early'][0], snapshots['early'][1], "State at T=10 (Early Waves & Crystals)")

    # Plot 3: Late state
    ax4 = fig.add_subplot(2, 3, 3)
    plot_slice(ax4, snapshots['late'][0], snapshots['late'][1], "State at T=40 (Accumulated Living Crystals)")

    # Plot 4: Flashback
    ax5 = fig.add_subplot(2, 3, 5)
    plot_slice(ax5, snapshots['flashback'][0], snapshots['flashback'][1], "Flashback Mode (Crystals Emitting)")

    # Plot 5: 3D Crystal Distribution
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    cx_3d, cy_3d, cz_3d = [], [], []
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                if is_crystal[i, j, k]:
                    cx_3d.append(X[i, j, k].item())
                    cy_3d.append(Y[i, j, k].item())
                    cz_3d.append(Z[i, j, k].item())

    if cx_3d:
        ax6.scatter(cx_3d, cy_3d, cz_3d, c='cyan', marker='*', s=50, edgecolor='k')
    ax6.set_title(f"3D Distribution of Core Memories\n(Total: {is_crystal.sum().item()})")
    ax6.set_xlabel("X")
    ax6.set_ylabel("Y")
    ax6.set_zlabel("Z")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_file = 'memory_crystallization_simulation.png'
    plt.savefig(output_file)
    print(f"Simulation complete. Visualization saved to {output_file}")

if __name__ == "__main__":
    simulate_memory_crystallization()
