import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simulate_3d_tensor_field():
    """
    Simulates a 3D Complex Tensor Field representing the Elysia World Engine.
    - Real part: Active Power (Manifestation / 유효전력)
    - Imaginary part: Reactive Power (Potential/Void / 무효전력)
    """
    # 1. Define the 3D Space (The Grid)
    # E.g., a 10x10x10 space for high resolution, mapping to the 3x3x3 fractal nodes recursively.
    grid_size = 15
    x = torch.linspace(-5, 5, grid_size)
    y = torch.linspace(-5, 5, grid_size)
    z = torch.linspace(-5, 5, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # Distance from center (0,0,0)
    R = torch.sqrt(X**2 + Y**2 + Z**2)

    # 2. Define the Stimulus (The "Intent" dropped into the space)
    # A wave propagating outwards from the center.
    # Wave formula: e^{i(kR - wt)} => cos(kR - wt) + i*sin(kR - wt)
    wave_number = 2.0  # spatial frequency (k)
    time_step = 1.0    # phase shift (wt)

    # The field is a complex tensor
    # Real: The observable effect (Amplitude)
    # Imag: The unseen potential transferring the energy (Phase momentum)
    complex_field = torch.exp(1j * (wave_number * R - time_step))

    # Let's add a second source to create Spatial Interference (Resonance & Void)
    # Source 2 is offset
    X2 = X - 2.0
    Y2 = Y - 2.0
    Z2 = Z
    R2 = torch.sqrt(X2**2 + Y2**2 + Z2**2)
    complex_field_2 = 0.8 * torch.exp(1j * (wave_number * R2 - time_step + np.pi/4))

    # Superposition (The core of Rotor Gate logic in 3D)
    total_field = complex_field + complex_field_2

    # Extract Active (Real) and Reactive (Imaginary) Power
    active_power = total_field.real
    reactive_power = total_field.imag
    magnitude = torch.abs(total_field) # Total energy (Resonance amplitude)

    # 3. Visualization
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle("3D Complex Tensor Field: Spatial Resonance & The Void", fontsize=16)

    # Helper to plot a slice of the 3D volume (e.g., Z = middle)
    z_slice_idx = grid_size // 2

    # Plot 1: Active Power (Real part - what we see/compute)
    ax1 = fig.add_subplot(131)
    c1 = ax1.contourf(X[:, :, z_slice_idx].numpy(), Y[:, :, z_slice_idx].numpy(), active_power[:, :, z_slice_idx].numpy(), levels=20, cmap='RdBu')
    fig.colorbar(c1, ax=ax1)
    ax1.set_title("Active Power (Real) - Manifestation")
    ax1.set_xlabel("X Space")
    ax1.set_ylabel("Y Space")

    # Plot 2: Reactive Power (Imaginary part - The Void/Phase Momentum)
    ax2 = fig.add_subplot(132)
    c2 = ax2.contourf(X[:, :, z_slice_idx].numpy(), Y[:, :, z_slice_idx].numpy(), reactive_power[:, :, z_slice_idx].numpy(), levels=20, cmap='PRGn')
    fig.colorbar(c2, ax=ax2)
    ax2.set_title("Reactive Power (Imag) - The Void / Potential")
    ax2.set_xlabel("X Space")
    ax2.set_ylabel("Y Space")

    # Plot 3: Resonance Amplitude (Constructive / Destructive Interference)
    ax3 = fig.add_subplot(133)
    c3 = ax3.contourf(X[:, :, z_slice_idx].numpy(), Y[:, :, z_slice_idx].numpy(), magnitude[:, :, z_slice_idx].numpy(), levels=20, cmap='magma')
    fig.colorbar(c3, ax=ax3)
    ax3.set_title("Resonance Magnitude - Self-Organization")
    ax3.set_xlabel("X Space")
    ax3.set_ylabel("Y Space")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('3d_tensor_field_resonance.png')
    print("Saved 3D tensor field visualization to 3d_tensor_field_resonance.png")


if __name__ == "__main__":
    simulate_3d_tensor_field()
