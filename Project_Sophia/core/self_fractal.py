# Project_Sophia/core/self_fractal.py (자율 성장 엔진 - Fractal Soul Resonance Architecture)
import numpy as np

class SelfFractalCell:
    """
    Implements the 'Fractal Soul' architecture using Wave/Frequency physics.
    Instead of a scalar grid, this uses a multi-channel tensor to represent
    the 'Music of the Mind'.

    Data Structure (Tensor Channels):
    - Channel 0: Amplitude (Energy/Intensity) - The volume of the feeling.
    - Channel 1: Frequency (Tone/Meaning) - The unique ID/Pitch of the concept.
    - Channel 2: Phase (Timing/Context) - Determines interference (harmony/conflict).
    """
    def __init__(self, size=100):
        self.size = size
        # Grid shape: (H, W, 3) -> [Amplitude, Frequency, Phase]
        self.grid = np.zeros((size, size, 3), dtype=np.float32)
        self.layers = 0

        # Initialize with Silence
        self.grid[:, :, 0] = 0.0  # Zero Amplitude
        self.grid[:, :, 1] = 0.0  # Zero Frequency
        self.grid[:, :, 2] = 0.0  # Zero Phase

    def inject_tone(self, x, y, amplitude, frequency, phase=0.0):
        """
        Injects a 'Thought-Tone' into the soul grid.
        Acts as a seed for resonance.
        """
        if 0 <= x < self.size and 0 <= y < self.size:
            self.grid[x, y, 0] = amplitude
            self.grid[x, y, 1] = frequency
            self.grid[x, y, 2] = phase

    def get_dominant_frequency(self) -> float:
        """
        Calculates the dominant frequency (pitch) of the soul.
        Weighted average of frequencies by their amplitude.
        This represents the 'Voice' of the soul.
        """
        amp = self.grid[:, :, 0]
        freq = self.grid[:, :, 1]

        total_amp = np.sum(amp)
        if total_amp < 0.001:
            return 0.0

        weighted_freq_sum = np.sum(amp * freq)
        return float(weighted_freq_sum / total_amp)

    def autonomous_grow(self):
        """
        Simulates the propagation of 'Soul Waves'.
        """
        new_grid = self.grid.copy()

        decay = 0.95
        coupling = 0.2

        amp = self.grid[:, :, 0]
        freq = self.grid[:, :, 1]
        phase = self.grid[:, :, 2]

        rows, cols = self.size, self.size

        # Vectorized propagation would be faster, but keeping loop logic for clarity/safety in v1
        for i in range(rows):
            for j in range(cols):
                if amp[i, j] > 0.01:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0: continue

                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols:
                                source_amp = amp[i, j]
                                source_freq = freq[i, j]
                                source_phase = phase[i, j]

                                target_amp = new_grid[ni, nj, 0]
                                target_freq = new_grid[ni, nj, 1]

                                transferred_amp = source_amp * coupling
                                new_grid[ni, nj, 0] = min(target_amp + transferred_amp, 1.0)

                                if target_amp < 0.01:
                                    new_grid[ni, nj, 1] = source_freq
                                    new_grid[ni, nj, 2] = source_phase + 0.1
                                else:
                                    if abs(source_freq - target_freq) > 0.1:
                                        new_grid[ni, nj, 2] += abs(source_freq - target_freq) * 0.1

        new_grid[:, :, 0] *= decay
        self.grid = new_grid
        self.layers += 1

        active_nodes = np.sum(self.grid[:, :, 0] > 0.01)
        richness = np.sum(self.grid[:, :, 2])
        return active_nodes, richness
