import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional
from Core.L6_Structure.hyper_quaternion import Quaternion

logger = logging.getLogger("ResonantField")

class ResonantField:
    """
    [Phase 34: The Architect of Laws]
    The Resonant Field Engine.
    
    Replaces sequential loops with simultaneous Tensor Field dynamics.
    State is stored as a 4D Vector (W, X, Y, Z) per grid cell,
    mapped to Elysia's core quaternionic pillars:
    W: Energy / Existence
    X: Emotion
    Y: Logic
    Z: Ethics
    """

    def __init__(self, size: int = 20, precision: str = 'float32'):
        self.size = size
        # Shape: (size, size, 4) -> 2D Grid of Quaternions
        self.field = np.zeros((size, size, 4), dtype=precision)
        
        # Initialize with a 'Ground State' (Pure existence)
        self.field[:, :, 0] = 0.5 
        
        logger.info(f"  Resonant Field Initialized: {size}x{size} (4D Tensor)")

    def apply_elastic_pull(self, target_q: Quaternion, elasticity: float = 0.1):
        """
        [Phase 35: Sovereign Resonance]
        Applies a spring-like pull towards a target state (e.g. User Vibe).
        - elasticity: 0.0 (No pull) to 1.0 (Instant mirroring).
        - Default is low to preserve Sovereign Buffer.
        """
        target_v = np.array([target_q.w, target_q.x, target_q.y, target_q.z])
        
        # Calculate the delta for the entire field at once
        # Field += (Target - Field) * elasticity
        self.field += (target_v - self.field) * elasticity
        
        logger.debug(f"  Elastic Pull Applied: Strength={elasticity}")

    def project_intent(self, x: int, y: int, q: Quaternion):
        """Projects a specific quaternionic intent (Cell) into the field."""
        if 0 <= x < self.size and 0 <= y < self.size:
            self.field[x, y] = [q.w, q.x, q.y, q.z]
            logger.debug(f"  Intent Projected at ({x}, {y}): {q}")

    def evolve(self, dt: float = 0.1):
        """
        Simultaneous State Evolution via Wave Interference.
        Uses a Laplacian-style kernel to simulate diffusion and resonance.
        This is the 'Non-Linear Leap' - all points update at once.
        """
        # 1. Diffusion / Interaction with neighbors
        # We use a simple 5-point stencil for vectorised interference
        shifted_up = np.roll(self.field, -1, axis=0)
        shifted_down = np.roll(self.field, 1, axis=0)
        shifted_left = np.roll(self.field, -1, axis=1)
        shifted_right = np.roll(self.field, 1, axis=1)
        
        # Laplacian (Interference Pattern)
        laplacian = (shifted_up + shifted_down + shifted_left + shifted_right - 4 * self.field)
        
        # 2. Rotational Evolution (Simulating 'The Spin of Thought')
        # Here we simulate a slight rotation in the X-Y (Emotion-Logic) plane
        # This is where 'Rotors' come in - expressed as a differential rotation matrix
        angle = 0.1 * dt
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Matrix rotation applied to every point simultaneously
        new_x = self.field[:,:,1] * cos_a - self.field[:,:,2] * sin_a
        new_y = self.field[:,:,1] * sin_a + self.field[:,:,2] * cos_a
        
        # 3. Apply changes (Integration)
        self.field[:,:,1] = new_x
        self.field[:,:,2] = new_y
        self.field += laplacian * dt
        
        # 4. Energy Conservation (Normalization)
        # Prevent field from blowing up
        norms = np.linalg.norm(self.field, axis=2, keepdims=True)
        self.field = np.where(norms > 1.0, self.field / norms, self.field)

    def get_state_summary(self) -> Dict[str, float]:
        """Calculates global metrics of the field."""
        return {
            "Total Energy (W)": float(np.sum(self.field[:,:,0])),
            "Emotional Density (X)": float(np.sum(abs(self.field[:,:,1]))),
            "Logic Intensity (Y)": float(np.sum(abs(self.field[:,:,2]))),
            "Ethical Alignment (Z)": float(np.sum(abs(self.field[:,:,3]))),
            "Global Complexity": float(np.std(self.field))
        }

# Global Field Instance
resonant_field = ResonantField()
