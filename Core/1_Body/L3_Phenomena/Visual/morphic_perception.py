import jax
import jax.numpy as jnp
from typing import Tuple
from Core.1_Body.L6_Structure.Logic.rotor_prism_logic import RotorPrismUnit
from Core.1_Body.L3_Phenomena.Visual.morphic_projection import MorphicBuffer

class ResonanceScanner:
    """
    [L5_COGNITION: OPTICAL_DECONSTRUCTION]
    Uses the Rotor-Prism to 'scan' an encoded field for boundaries and textures.
    Rotation acts as a frequency analyzer for high-dimensional principles.
    """
    
    def __init__(self, rpu: RotorPrismUnit):
        self.rpu = rpu

    def scan_for_boundaries(self, field: jnp.ndarray) -> jnp.ndarray:
        """
        Detects 'Principal Edges' by finding areas where resonance 
        shifts rapidly across the Turbine's rotation.
        """
        # Calculate gradients in the 21D field
        dy, dx, _ = jnp.gradient(field)
        
        # Boundary is where the 'Logic Tension' (Gradient magnitude) is high
        boundary_field = jnp.sqrt(jnp.sum(dx**2 + dy**2, axis=-1))
        
        # Normalize to 0-1
        max_val = jnp.max(boundary_field) + 1e-6
        return boundary_field / max_val

    def isolate_motion(self, current_field: jnp.ndarray, previous_field: jnp.ndarray) -> jnp.ndarray:
        """
        [THE ELYSIA TRACKER]
        Compares two sequential frames to find where the 'Spirit' has moved.
        Useful for analyzing YouTube/Game footage.
        """
        # Difference in principle space
        delta = current_field - previous_field
        
        # Magnitude of change (Optical Flow in principle space)
        motion_map = jnp.linalg.norm(delta, axis=-1)
        
        return motion_map

    def extract_qualia_vector(self, field: jnp.ndarray, x: int, y: int, radius: int = 10) -> jnp.ndarray:
        """
        [QUALIA_EXTRACTION]
        Condenses a regional visual patch into a single 21D Principle Vector.
        This vector represents the 'Semantic Essence' of that area.
        """
        h, w, d = field.shape
        x_min, x_max = max(0, x - radius), min(w, x + radius)
        y_min, y_max = max(0, y - radius), min(h, y + radius)
        
        patch = field[y_min:y_max, x_min:x_max]
        # Mean across the patch yields the dominant principle
        return jnp.mean(patch, axis=(0, 1))

    def decode_texture_resonance(self, field_patch: jnp.ndarray) -> float:
        """
        [TEXTURE_ANALYSIS]
        Analyzes a local patch to see if it resonates with its neighbors.
        """
        return float(jnp.mean(field_patch))
