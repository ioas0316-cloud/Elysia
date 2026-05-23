import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_complex_wave(frequency, phase_shift, amplitude, time_steps):
    """
    Generates a 1D complex wave over time: z(t) = A * e^{i(wt + phi)}
    """
    t = torch.linspace(0, 4 * np.pi, time_steps)
    # The real part is cosine, the imaginary part is sine.
    # We create a complex tensor directly.
    wave = amplitude * torch.exp(1j * (frequency * t + phase_shift))
    return t, wave

class RotorGate(torch.nn.Module):
    """
    A logical gate based on Wave Interference rather than binary states.
    It takes two complex inputs and combines them.
    """
    def __init__(self):
        super().__init__()

    def forward(self, wave_a, wave_b):
        """
        The gate operation is simple superposition (addition of complex vectors).
        - If phases align, magnitude increases (Constructive Interference / 양각).
        - If phases are opposite, magnitude cancels out (Destructive Interference / 음각 / Void).
        """
        return wave_a + wave_b

def run_simulation():
    time_steps = 500
    frequency = 1.0
    amplitude = 1.0

    # 1. Constructive Interference (양각 / Phase Alignment)
    # Both waves have the exact same phase.
    t, wave_a_constructive = generate_complex_wave(frequency, phase_shift=0.0, amplitude=amplitude, time_steps=time_steps)
    _, wave_b_constructive = generate_complex_wave(frequency, phase_shift=0.0, amplitude=amplitude, time_steps=time_steps)

    # 2. Destructive Interference (음각 / Phase Opposition / Void)
    # wave_b is pi radians (180 degrees) out of phase with wave_a
    t, wave_a_destructive = generate_complex_wave(frequency, phase_shift=0.0, amplitude=amplitude, time_steps=time_steps)
    _, wave_b_destructive = generate_complex_wave(frequency, phase_shift=np.pi, amplitude=amplitude, time_steps=time_steps)

    gate = RotorGate()

    # Apply the gate (superposition)
    output_constructive = gate(wave_a_constructive, wave_b_constructive)
    output_destructive = gate(wave_a_destructive, wave_b_destructive)

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Rotor Gate: Complex Wave Interference", fontsize=16)

    # Helper function to plot waves
    def plot_wave(ax, t, w1, w2, out, title, description):
        # We plot the Real part of the wave to visualize it as a standard 2D wave over time
        ax.plot(t, w1.real.numpy(), label='Input A (Real)', linestyle='--', alpha=0.7, color='blue')
        ax.plot(t, w2.real.numpy(), label='Input B (Real)', linestyle=':', alpha=0.7, color='green')
        ax.plot(t, out.real.numpy(), label='Output (Real)', linewidth=2.5, color='red')

        ax.set_title(title)
        ax.set_ylim(-3, 3)
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax.legend(loc='upper right')
        ax.text(0.5, -2.5, description, style='italic', fontsize=10)

    plot_wave(axs[0, 0], t, wave_a_constructive, wave_b_constructive, output_constructive,
              "Constructive Interference (Yang-gak / Resonance)",
              "Phases match perfectly. Energy amplifies (1 + 1 = Resonance).")

    plot_wave(axs[1, 0], t, wave_a_destructive, wave_b_destructive, output_destructive,
              "Destructive Interference (Eum-gak / The Void)",
              "Phases are exactly opposite. Waves cancel into a calm equilibrium (The Void).")

    # Polar plot to show phase space
    def plot_polar(ax, w1, w2, out, title):
        # Extract a single point in time to show the phase vectors
        idx = time_steps // 8 # Pick an arbitrary point

        origin = [0], [0]
        ax.quiver(*origin, w1[idx].real.item(), w1[idx].imag.item(), color='blue', scale=5, label='Input A Vector')
        ax.quiver(*origin, w2[idx].real.item(), w2[idx].imag.item(), color='green', scale=5, label='Input B Vector')
        ax.quiver(*origin, out[idx].real.item(), out[idx].imag.item(), color='red', scale=5, label='Output Vector')

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_title(title)
        # ax.legend()

    plot_polar(axs[0, 1], wave_a_constructive, wave_b_constructive, output_constructive, "Phase Vectors (Constructive)")
    plot_polar(axs[1, 1], wave_a_destructive, wave_b_destructive, output_destructive, "Phase Vectors (Destructive / Void)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('rotor_gate_interference.png')
    print("Visualization saved to rotor_gate_interference.png")

if __name__ == "__main__":
    run_simulation()
