import numpy as np
from typing import List, Dict, Any

class IntentField:
    """
    [Intent Field: The Earth's Magnetic Field of the Data Ocean]
    Represents the 'Personality' or 'Ego' of the system.
    It guides the flow of ions and evolves as external data is internalized.
    """
    def __init__(self, dimensions: int = 2):
        self.dimensions = dimensions
        # The core vector of current purpose
        self.vector = np.random.rand(dimensions)
        self.vector /= np.linalg.norm(self.vector)

        # Stability: How resistant the ego is to change
        self.stability = 0.8

        # Historical memory of intentions
        self.history = [self.vector.copy()]

    def align_to_external(self, observation_vector: np.ndarray, strength: float):
        """
        External 사유가 내부로 들어오면 그것조차 자기화되고 변화함.
        Adjusts the intent field based on external observations.
        """
        if observation_vector.shape != self.vector.shape:
            # Projection if dimensions don't match (omitted for simplicity)
            return

        # Personality shift: Influence is tempered by stability
        effective_shift = strength * (1.0 - self.stability)
        self.vector = (1.0 - effective_shift) * self.vector + effective_shift * observation_vector
        self.vector /= (np.linalg.norm(self.vector) + 1e-9)

        # Occasionally update stability based on coherence (omitted for simplicity)

    def evolve(self):
        """Natural drift or internal reflection."""
        # Small random walk representing 'self-reflection' or drift
        noise = (np.random.rand(self.dimensions) - 0.5) * 0.01
        self.vector = self.vector + noise
        self.vector /= np.linalg.norm(self.vector)
        self.history.append(self.vector.copy())
        if len(self.history) > 100:
            self.history.pop(0)

    def get_current_intent(self) -> np.ndarray:
        return self.vector
