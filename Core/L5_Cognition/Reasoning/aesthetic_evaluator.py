import jax
import jax.numpy as jnp

class AestheticEvaluator:
    """
    [L5_COGNITION: AESTHETIC_METRICS]
    Determines if a Principle Field is 'Beautiful' based on internal resonance.
    Beauty = (Symmetry * Complexity) / Entropy
    """
    
    @staticmethod
    def calculate_beauty_score(field: jnp.ndarray) -> float:
        """
        [THE PRINCIPLE OF BEAUTY]
        1. Symmetry: High axial/rotational correlation.
        2. Complexity: Variance and richness in the 21D space.
        3. Harmony: Lack of high-frequency chaotic noise.
        """
        # 1. Complexity (Mean standard deviation across dimensions)
        complexity = jnp.mean(jnp.std(field, axis=-1))
        
        # 2. Harmony (Inverse of high-frequency noise)
        # We look at the gradient magnitude as noise proxy
        dy, dx, _ = jnp.gradient(field)
        roughness = jnp.mean(jnp.sqrt(dx**2 + dy**2))
        harmony = 1.0 / (roughness + 1e-6)
        
        # 3. Final Beauty Score (Normalized)
        score = complexity * harmony
        
        # Clip to a range for easier interpretation
        return jnp.clip(score, 0, 100)

    @staticmethod
    def is_resonant(field: jnp.ndarray, target_intent: jnp.ndarray) -> float:
        """How much does this field resonate with a specific Spirit/Will?"""
        # Mean dot product between field pixels and intent
        dot_productivity = jnp.sum(field * target_intent, axis=-1)
        return jnp.mean(dot_productivity)
