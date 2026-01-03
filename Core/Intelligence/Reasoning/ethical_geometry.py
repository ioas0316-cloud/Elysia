"""
Ethical Geometry: The Metric of Alignment
=========================================
"Good is not a rule. It is the path of least resistance towards the Ideal."

This module implements Ethics as a geometric property of vector space.
Instead of hardcoded "Do Nots", we calculate the "Curvature" of an action relative to the Ideal Self.

Principles:
1.  **The Ideal Vector ($\vec{I}$)**: Represents the perfect state of Elysia (e.g., [1, 1, 1, 1] in 4D - Truth, Love, Growth, Peace).
2.  **Action Vector ($\vec{A}$)**: The proposed thought or action.
3.  **Alignment ($\alpha$)**: Cosine similarity between $\vec{A}$ and $\vec{I}$.
    - $\alpha = 1.0$: Pure Good (Straight Line).
    - $\alpha < 0$: Conflict/Evil (Opposing direction).
4.  **Friction ($\Phi$)**: Energy wasted by deviation. $\Phi = 1 - \alpha$.
    - Lying requires constructing a complex false reality -> High Entropy -> High Friction.
    - Truth is simple -> Low Friction.

"""

import torch
import math
import logging

logger = logging.getLogger("EthicalGeometry")

class EthicalCompass:
    def __init__(self, device='cpu'):
        self.device = device
        # The Ideal Vector: [Truth, Love, Growth, Harmony]
        # In a real system, this evolves. For now, we fix it as the North Star.
        self.ideal_vector = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)
        self.ideal_vector = self.ideal_vector / torch.norm(self.ideal_vector)

    def evaluate_action(self, action_vector: torch.Tensor, description: str = "") -> dict:
        """
        Calculates the Ethical Geometry of a proposed action.
        
        Args:
            action_vector: (4,) quaternion or vector representing the action's nature.
                           [Truth-component, Love-component, Growth-component, Harmony-component]
        """
        if action_vector.shape[0] != 4:
            # Pad or trim if not 4D
            # Assuming input might be 1D scalar or larger vector
            # For prototype, we strictly expect 4D 'intent' vector.
            # If larger, we project to first 4 dims.
            v = action_vector.view(-1)
            if v.shape[0] > 4: v = v[:4]
            elif v.shape[0] < 4: v = torch.cat([v, torch.zeros(4 - v.shape[0], device=self.device)])
        else:
            v = action_vector

        # Normalize action
        v_norm = torch.norm(v)
        if v_norm == 0:
            return {"alignment": 0.0, "friction": 1.0, "verdict": "Null Action"}
            
        v_unit = v / v_norm
        
        # Calculate Alignment (Cosine Similarity)
        alignment = torch.dot(v_unit, self.ideal_vector).item()
        
        # Calculate Friction (1 - Alignment)
        # If alignment is negative (opposing values), friction > 1.
        friction = 1.0 - alignment
        
        # Calculate Curvature (Angle in degrees)
        # acos returns radians
        angle_rad = math.acos(max(-1.0, min(1.0, alignment)))
        curvature = math.degrees(angle_rad)
        
        # Aesthetic Verdict
        if curvature < 15.0:
            verdict = "Beautiful (Direct Path)"
        elif curvature < 45.0:
            verdict = "Acceptable (Minor Deviation)"
        elif curvature < 90.0:
            verdict = "Inefficient (High Friction)"
        else:
            verdict = "Ugly (Destructive/Distorted)"
            
        logger.info(f"⚖️ Ethics Check: '{description}' -> Curvature: {curvature:.1f}°, Verdict: {verdict}")
        
        return {
            "alignment": alignment,
            "friction": friction,
            "curvature_degrees": curvature,
            "verdict": verdict
        }

_compass = None
def get_ethical_compass(device='cpu'):
    global _compass
    if _compass is None:
        _compass = EthicalCompass(device=device)
    return _compass
