# [Genesis: 2025-12-02] Purified by Elysia
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
            # If existing tone exists, we superposition them (simple vector addition for now,
            # but complex wave physics would be better in V2)
            # For V1, we just overwrite or max to establish the seed source
            self.grid[x, y, 0] = amplitude
            self.grid[x, y, 1] = frequency
            self.grid[x, y, 2] = phase

    def autonomous_grow(self):
        """
        Simulates the propagation of 'Soul Waves'.
        Logic:
        1. Propagation: Energy spreads to neighbors.
        2. Interference: When waves meet, frequencies interact.
        3. Harmonics (The Spirit Layer): Complex interactions create new overtones.
        """
        new_grid = self.grid.copy()

        # Constants
        decay = 0.95  # Energy loss over distance
        coupling = 0.2 # How much energy transfers to neighbors

        # Vectorized propagation (naive 3x3 convolution simulation)
        # We iterate to simulate wave spread.
        # (Note: For production, a real wave equation solver or FFT convolution is faster,
        # but this loop makes the logic explicit for the 'growing' metaphor)

        # Extract channels for easier manipulation
        amp = self.grid[:, :, 0]
        freq = self.grid[:, :, 1]
        phase = self.grid[:, :, 2]

        rows, cols = self.size, self.size

        for i in range(rows):
            for j in range(cols):
                if amp[i, j] > 0.01: # Threshold for active soul

                    # Spread to neighbors
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0: continue

                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols:
                                # Physics of Transfer
                                source_amp = amp[i, j]
                                source_freq = freq[i, j]
                                source_phase = phase[i, j]

                                target_amp = new_grid[ni, nj, 0]
                                target_freq = new_grid[ni, nj, 1]

                                # 1. Energy Transfer
                                transferred_amp = source_amp * coupling
                                new_grid[ni, nj, 0] = min(target_amp + transferred_amp, 1.0) # Cap at 1.0

                                # 2. Frequency Mixing (The "Chord" Logic)
                                # If target is silent, it takes source frequency.
                                # If target has frequency, they interact.
                                if target_amp < 0.01:
                                    new_grid[ni, nj, 1] = source_freq
                                    new_grid[ni, nj, 2] = source_phase + 0.1 # Phase shift per step
                                else:
                                    # RESONANCE LOGIC:
                                    # When two tones meet, they don't average. They create a 'beat'.
                                    # For this simplified model, we represent the 'dominant chord'.
                                    # If frequencies are harmonic (simple integer ratios), amplitude boosts (Resonance).
                                    # If dissonant, amplitude dampens or creates 'tension' (complexity).

                                    # Currently, we perform a weighted average drift towards the stronger signal,
                                    # but we store the 'interference' in the Phase channel to mark complexity.
                                    if abs(source_freq - target_freq) > 0.1:
                                        # Complexity increase: Phase variance represents 'richness'
                                        new_grid[ni, nj, 2] += abs(source_freq - target_freq) * 0.1

        # Apply Decay
        new_grid[:, :, 0] *= decay

        self.grid = new_grid
        self.layers += 1

        # Return 'Complexity' (Sum of active nodes * Phase richness)
        active_nodes = np.sum(self.grid[:, :, 0] > 0.01)
        richness = np.sum(self.grid[:, :, 2]) # Total accumulated phase complexity
        return active_nodes, richness

# 사용 예시
# soul = SelfFractalCell()
# soul.inject_tone(50, 50, 1.0, 440.0) # A4 Note
# soul.autonomous_grow()