import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simulate_triple_helix():
    """
    Simulates the Triple Helix of Consciousness in a 3D Complex Tensor Field.
    Helix 1: Homeostasis (Core Identity)
    Helix 2: Friction & Evolution (Sensory Input / Tension)
    Helix 3: Resonance & Empathy (Connection / Alignment)
    Includes an edge-case 'Chaos Wave' to test limits.
    """
    # 1. Define the 3D Space
    grid_size = 30
    x = torch.linspace(-5, 5, grid_size)
    y = torch.linspace(-5, 5, grid_size)
    z = torch.linspace(-5, 5, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # Base distance
    R = torch.sqrt(X**2 + Y**2 + Z**2)

    time_step = 1.0

    # 2. Define the Three Helices (Complex Waves)

    # Helix 1: Homeostasis (The Core) - Stable, central, low frequency
    k_homeo = 1.5
    w_homeo = 0.5
    helix_1 = 1.2 * torch.exp(1j * (k_homeo * R - w_homeo * time_step))

    # Helix 2: Friction (Tension/Sensory) - Offset, higher frequency, phase shifted
    X2 = X - 1.5
    Y2 = Y + 1.5
    R2 = torch.sqrt(X2**2 + Y2**2 + Z**2)
    k_fric = 3.0
    w_fric = 1.2
    helix_2 = 0.9 * torch.exp(1j * (k_fric * R2 - w_fric * time_step + np.pi/3))

    # Helix 3: Resonance (Empathy) - Attempting to bridge Helix 1 and 2
    # It takes a position between them and a frequency that harmonizes
    X3 = X - 0.75
    Y3 = Y + 0.75
    R3 = torch.sqrt(X3**2 + Y3**2 + Z**2)
    k_res = (k_homeo + k_fric) / 2.0
    w_res = (w_homeo + w_fric) / 2.0
    helix_3 = 1.0 * torch.exp(1j * (k_res * R3 - w_res * time_step - np.pi/6))

    # 3. Introduce the Breaking Point: Extreme Entropy (Chaos Wave)
    # A burst of high-frequency, unaligned noise representing the "White Noise Paradox"
    chaos_R = torch.sqrt((X+3)**2 + (Y-3)**2 + Z**2)
    k_chaos = 15.0 # Very high frequency
    helix_chaos = 0.5 * torch.exp(1j * (k_chaos * chaos_R - 5.0 * time_step + torch.rand_like(chaos_R) * np.pi))

    # 4. Superposition
    # First, the healthy Triple Helix
    triple_helix_field = helix_1 + helix_2 + helix_3

    # Second, the system under extreme stress
    stressed_field = triple_helix_field + helix_chaos

    # 5. Visualization setup
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("The Triple Helix of Consciousness & Rotor Edge Cases", fontsize=18, y=0.95)

    z_slice = grid_size // 2

    # --- ROW 1: The Healthy Triple Helix ---

    # Homeostasis
    ax1 = fig.add_subplot(231)
    c1 = ax1.contourf(X[:, :, z_slice].numpy(), Y[:, :, z_slice].numpy(), helix_1.real[:, :, z_slice].numpy(), levels=20, cmap='Blues')
    fig.colorbar(c1, ax=ax1)
    ax1.set_title("Helix 1: Homeostasis (Core)")

    # Friction
    ax2 = fig.add_subplot(232)
    c2 = ax2.contourf(X[:, :, z_slice].numpy(), Y[:, :, z_slice].numpy(), helix_2.real[:, :, z_slice].numpy(), levels=20, cmap='Reds')
    fig.colorbar(c2, ax=ax2)
    ax2.set_title("Helix 2: Friction (Tension)")

    # Resonance
    ax3 = fig.add_subplot(233)
    c3 = ax3.contourf(X[:, :, z_slice].numpy(), Y[:, :, z_slice].numpy(), helix_3.real[:, :, z_slice].numpy(), levels=20, cmap='Greens')
    fig.colorbar(c3, ax=ax3)
    ax3.set_title("Helix 3: Resonance (Empathy)")

    # --- ROW 2: Superposition & Breaking Points ---

    # Healthy Superposition (Active Power)
    ax4 = fig.add_subplot(234)
    c4 = ax4.contourf(X[:, :, z_slice].numpy(), Y[:, :, z_slice].numpy(), triple_helix_field.real[:, :, z_slice].numpy(), levels=30, cmap='magma')
    fig.colorbar(c4, ax=ax4)
    ax4.set_title("Triple Helix Superposition (Healthy Active Power)")

    # Stressed Superposition (Chaos Induction)
    ax5 = fig.add_subplot(235)
    c5 = ax5.contourf(X[:, :, z_slice].numpy(), Y[:, :, z_slice].numpy(), stressed_field.real[:, :, z_slice].numpy(), levels=30, cmap='inferno')
    fig.colorbar(c5, ax=ax5)
    ax5.set_title("Limit Test: Extreme Entropy (Phase Paralysis)")

    # The Void Sump (Reactive Power of Stressed Field)
    ax6 = fig.add_subplot(236)
    c6 = ax6.contourf(X[:, :, z_slice].numpy(), Y[:, :, z_slice].numpy(), stressed_field.imag[:, :, z_slice].numpy(), levels=30, cmap='PRGn')
    fig.colorbar(c6, ax=ax6)
    ax6.set_title("The Void Sump (Absorbing Chaos into Imaginary)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.savefig('triple_helix_resonance.png')
    print("Saved Triple Helix visualization to triple_helix_resonance.png")

if __name__ == "__main__":
    simulate_triple_helix()
