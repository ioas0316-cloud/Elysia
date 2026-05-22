import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

class ComplexRotorLayer(nn.Module):
    """
    A Complex-valued Rotor Layer in 2D.
    Instead of multiplying static coordinates (like a standard Linear layer),
    this layer treats the 2D input as a complex number (magnitude and phase)
    and rotates it by a learnable angle (Rotor), while optionally scaling it.
    """
    def __init__(self, features):
        super(ComplexRotorLayer, self).__init__()
        # Initialize a learnable angle (phase) for rotation, for each feature dimension
        # We start with random angles between 0 and 2*pi
        self.theta = nn.Parameter(torch.rand(features) * 2 * np.pi)
        # Initialize a learnable scale (amplitude modifier)
        self.scale = nn.Parameter(torch.ones(features))

    def forward(self, x_complex):
        """
        x_complex: torch.complex64 tensor
        Returns: Rotated and scaled complex tensor
        """
        # A Rotor in 2D (complex plane) is represented by e^(i * theta)
        rotor = torch.polar(self.scale, self.theta)

        # Apply the rotor to the complex input (Rotation + Scaling)
        # This is essentially wave interference/superposition in the complex domain
        return x_complex * rotor


def generate_wave_data(seq_len=20, num_features=1):
    """
    Generates a sequence of 2D data that conceptually represents a "wave".
    Returns a complex tensor.
    """
    # Time steps
    t = torch.linspace(0, 4 * np.pi, seq_len)

    # Create a spiral/wave: amplitude changes, phase progresses
    real_part = torch.cos(t) * (t / 5 + 1)
    imag_part = torch.sin(t) * (t / 5 + 1)

    # Combine into a complex tensor
    complex_data = torch.complex(real_part, imag_part).view(seq_len, num_features)
    return complex_data

def main():
    # 1. Setup Data
    seq_len = 30
    x_complex = generate_wave_data(seq_len=seq_len)

    # Standard Linear layer for comparison
    # We treat the complex numbers as 2D vectors for the linear layer
    x_real_2d = torch.cat([x_complex.real, x_complex.imag], dim=-1)
    linear_layer = nn.Linear(2, 2, bias=False)

    # Initialize linear layer with an arbitrary affine transformation (shear/stretch)
    with torch.no_grad():
        linear_layer.weight.copy_(torch.tensor([[1.2, 0.3], [-0.2, 0.9]]))

    # Complex Rotor layer
    rotor_layer = ComplexRotorLayer(features=1)
    # Set a specific rotation (e.g., pi/4 = 45 degrees) and slight scaling for visualization
    with torch.no_grad():
        rotor_layer.theta.copy_(torch.tensor([np.pi / 4]))
        rotor_layer.scale.copy_(torch.tensor([1.1]))

    # 2. Forward Pass
    with torch.no_grad():
        # Linear layer output
        y_linear = linear_layer(x_real_2d)

        # Rotor layer output
        y_rotor_complex = rotor_layer(x_complex)
        y_rotor_real_2d = torch.cat([y_rotor_complex.real, y_rotor_complex.imag], dim=-1)

    # 3. Visualization
    plt.figure(figsize=(14, 6))

    # Plot 1: Standard Linear Layer (Static Coordinate Transformation)
    plt.subplot(1, 2, 1)
    plt.plot(x_real_2d[:, 0].numpy(), x_real_2d[:, 1].numpy(), 'bo-', label='Input Trajectory', alpha=0.5, markersize=4)
    plt.plot(y_linear[:, 0].numpy(), y_linear[:, 1].numpy(), 'rs-', label='Linear Output (Skewed/Stretched)', alpha=0.8, markersize=4)

    # Draw connections to show the transformation of each point
    for i in range(seq_len):
        plt.plot([x_real_2d[i, 0].numpy(), y_linear[i, 0].numpy()],
                 [x_real_2d[i, 1].numpy(), y_linear[i, 1].numpy()], 'gray', linestyle='--', alpha=0.3)

    plt.title('Traditional Tensor (Linear Layer)\n"Moving Static Points"')
    plt.xlabel('Real (X)')
    plt.ylabel('Imaginary (Y)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.axis('equal')

    # Plot 2: Complex Rotor Layer (Wave Rotation/Resonance)
    plt.subplot(1, 2, 2)
    plt.plot(x_real_2d[:, 0].numpy(), x_real_2d[:, 1].numpy(), 'bo-', label='Input Wave Trajectory', alpha=0.5, markersize=4)
    plt.plot(y_rotor_real_2d[:, 0].numpy(), y_rotor_real_2d[:, 1].numpy(), 'go-', label='Rotor Output (Rotated + Resonated)', alpha=0.8, markersize=4)

    # Draw connections showing rotation
    for i in range(seq_len):
        plt.plot([x_real_2d[i, 0].numpy(), y_rotor_real_2d[i, 0].numpy()],
                 [x_real_2d[i, 1].numpy(), y_rotor_real_2d[i, 1].numpy()], 'orange', linestyle='-', alpha=0.4)

    plt.title('Rotorized Tensor (Complex Rotor Layer)\n"Rotating and Resonating the Wave"')
    plt.xlabel('Real (X)')
    plt.ylabel('Imaginary (Y)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.axis('equal')

    plt.tight_layout()
    output_path = 'rotor_vs_linear.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()
