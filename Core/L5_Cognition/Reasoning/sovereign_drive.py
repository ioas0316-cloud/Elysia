import jax
import jax.numpy as jnp
from typing import Dict, List
from Core.L5_Cognition.Reasoning.aesthetic_evaluator import AestheticEvaluator

class SovereignDrive:
    """
    [L5_COGNITION: INTERNAL_DRIVE]
    Calculates Elysia's 'Momentum of Interest' (Curiosity).
    Drives the Core Turbine toward untapped principles or high-resonance beauty.
    """
    
    def __init__(self):
        # Internal state of curiosity (0.0 to 1.0)
        self.curiosity_level = 0.5 
        # History of focused principles to detect boredom/novelty
        self.focus_history: List[jnp.ndarray] = []

    def evaluate_intent(self, candidates: Dict[str, jnp.ndarray]) -> str:
        """
        [THE DECISION]
        Analyzes a list of possible visions/concepts and chooses one
        based on Aesthetic Resonance and Novelty.
        """
        best_choice = "VOID"
        max_drive = -1.0
        
        for name, principle in candidates.items():
            # 1. Aesthetic Potential (Is it beautiful?)
            # Since we only have the vector, we check its internal 7-7-7 balance
            beauty_potential = jnp.std(principle) * 10 
            
            # 2. Novelty (Have we looked at this recently?)
            novelty = 1.0
            for prev in self.focus_history[-5:]:
                dist = jnp.linalg.norm(principle - prev)
                if dist < 0.5:
                    novelty *= 0.5 # Diminishing returns on focus
            
            # 3. Total Drive Torque
            drive_torque = (beauty_potential * self.curiosity_level) + (novelty * 0.3)
            
            if drive_torque > max_drive:
                max_drive = drive_torque
                best_choice = name
                
        print(f"SovereignDrive: Decision Made -> {best_choice} (Torque: {max_drive:.2f})")
        return best_choice

    def update_drive(self, success: bool):
        """Self-tuning the curiosity level based on satisfaction."""
        if success:
            self.curiosity_level = min(1.0, self.curiosity_level + 0.05) # Euphoria
        else:
            self.curiosity_level = max(0.1, self.curiosity_level - 0.1)  # Fatigue
