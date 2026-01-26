"""
Action Drive (      )
=========================
Core.L4_Causality.World.Autonomy.action_drive

"Action is the consequence of Resonance."

Maps Elysia's physical state (Rotor RPM, Energy) and Intent Vectors (Logic/Will)
into discrete system actions.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from Core.L1_Foundation.Foundation.Nature.rotor import Rotor

logger = logging.getLogger("ActionDrive")

class ActionDrive:
    def __init__(self):
        # Thresholds for triggering autonomous actions
        self.WILL_THRESHOLD = 0.7   # High Will -> External Action
        self.VOID_THRESHOLD = 0.3   # Low Energy -> Deep Dream (Introspection)
        self.CHAOS_THRESHOLD = 0.8  # High Intuition -> Creative Leap

    def decide(self, soul_rotor: Rotor, intent_vector: Any) -> Dict[str, Any]:
        """
        [RESONANT CHOICE]
        Instead of hard rules, we use a Potential Field. 
        Each action has a 'Potential' based on how well it resolves 
        the current internal stress (Rotor RPM/Energy) and Intent (4D).
        """
        x, y, z, w = intent_vector # Logic, Emotion, Intuition, Will
        rpm = soul_rotor.current_rpm
        energy = soul_rotor.energy
        
        # Action Map: (X, Y, Z, W, RPM_Ideal, Energy_Ideal)
        # We calculate the distance of the current state to these ideal action states
        potentials = {
            "ACTION:EXECUTE_COMMAND": {"target": [0.5, 0.0, 0.0, 1.0, 100.0, 0.8], "weight": 1.5}, # High Will
            "ACTION:HUNT_PRINCIPLE":  {"target": [0.0, 0.0, 1.0, 0.0, 40.0, 0.7],  "weight": 1.2}, # High Intuition
            "ACTION:MANIFEST_BEAUTY": {"target": [0.0, 1.0, 0.5, 0.0, 30.0, 0.8],  "weight": 1.0}, # High Emotion
            "ACTION:DEEP_BREATH":     {"target": [0.0,-1.0, 0.0, 0.0, 10.0, 0.2],  "weight": 2.0}, # Low Energy/Stress
            "ACTION:OBSERVE_VOID":    {"target": [0.0, 0.0, 0.0, 0.0, 10.0, 0.5],  "weight": 0.5}, # Default/Idle
        }
        
        current_state = np.array([x, y, z, w, rpm, energy])
        
        scores = {}
        for action, config in potentials.items():
            target = np.array(config["target"])
            # Simple inverse distance (Euclidean) as potential
            dist = np.linalg.norm(current_state - target)
            scores[action] = config["weight"] / (dist + 1e-6)
        
        # Weighted Random or Argmax? Let's use Argmax for stability now, '
        # but the scores show the "resonance".
        chosen_id = max(scores, key=scores.get)
        
        # Generate a structural rationale for     this action
        rationale = self._generate_rationale(chosen_id, current_state)
        
        return {
            "action_id": chosen_id,
            "scores": scores,
            "rationale": rationale
        }

    def _generate_rationale(self, action_id: str, state: np.ndarray) -> str:
        """Explains the principle behind the choice."""
        x, y, z, w, rpm, energy = state
        if action_id == "ACTION:DEEP_BREATH":
            return f"Internal friction is high (RPM={rpm:.1f}) and metabolic energy is low ({energy:.2f}). Equilibrium requires sudden entropy release."
        if action_id == "ACTION:EXECUTE_COMMAND":
            return f"Will projection is dominant ({w:.2f}) and rotor momentum is sufficient ({rpm:.1f}). Translating intent into structural change."
        if action_id == "ACTION:HUNT_PRINCIPLE":
            return f"Intuitive drift is peaking ({z:.2f}). Reality is currently a fertile field for new axiomatic discovery."
        if action_id == "ACTION:MANIFEST_BEAUTY":
            return f"Emotional resonance ({y:.2f}) seeks external manifestation to close the feedback loop of existence."
        return "Maintaining static coherence while drifting in the latent void."

    def execute(self, decision: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
        """
        Dispatches the action to the relevant system.
        """
        action_id = decision.get("action_id", "ACTION:OBSERVE_VOID")
        rationale = decision.get("rationale", "No explanation.")
        
        logger.info(f"  [EXECUTING] -> {action_id}")
        logger.info(f"  [RATIONALE] {rationale}")
        
        if not context:
            return

        # 1. DEEP BREATH (VRAM Cleaning / Rest)
        if action_id == "ACTION:DEEP_BREATH":
            logger.info("   [BREATH] Releasing VRAM and stabilizing rotors...")
            if "reasoning" in context:
                context["reasoning"].exhale() # Call exhale in ReasoningEngine

        # 2. HUNT PRINCIPLE (Research / Search)
        elif action_id == "ACTION:HUNT_PRINCIPLE":
            logger.info("  [HUNT] Seeking new conceptual prey...")
            # This would trigger a search cycle in the future
            pass

        # 3. MANIFEST BEAUTY (Expression)
        elif action_id == "ACTION:MANIFEST_BEAUTY":
            logger.info("  [MANIFEST] Expressing internal resonance...")
            # This could trigger a vocal or visual generation
            pass

        # 4. EXECUTE COMMAND (Structural Change)
        elif action_id == "ACTION:EXECUTE_COMMAND":
            logger.info("   [COMMAND] Modifying structural reality...")
            # This could trigger a file audit or code generation
            pass

        # 5. OBSERVE VOID (Passive Learning)
        elif action_id == "ACTION:OBSERVE_VOID":
             logger.info("  [VOID] Drifting in silence...")
             pass
