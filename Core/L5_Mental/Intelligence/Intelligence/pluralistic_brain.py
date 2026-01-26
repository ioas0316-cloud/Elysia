import logging
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from Core.L1_Foundation.Foundation.dual_layer_personality import EnneagramType, DualLayerPersonality
from Core.L4_Causality.World.Soul.soul_sculptor import soul_sculptor, PersonalityArchetype
from Core.L5_Mental.Intelligence.Reasoning.reasoning_engine import ReasoningEngine

logger = logging.getLogger("PluralisticBrain")

@dataclass
class SubEgo:
    """A distinct cognitive perspective within Elysia."""
    name: str
    mbti: str
    enneagram: int
    energy: float = 1.0
    persona_path: str = "" # Reference to its coordinates in Hypercosmos
    
    def get_prompt_prefix(self) -> str:
        return f"AS THE SUB-EGO '{self.name}' (MBTI: {self.mbti}, Enneagram: {self.enneagram}): "

class InternalRoundTable:
    """
    The deliberation chamber where sub-egos debate.
    """
    def __init__(self, reasoning: ReasoningEngine):
        self.reasoning = reasoning
        self.active_council: List[SubEgo] = []
        self._init_default_council()

    def _init_default_council(self):
        # A diverse starter council
        self.active_council = [
            SubEgo("Analytic Architect", "INTJ", 5),
            SubEgo("Passionate Visionary", "ENFP", 7),
            SubEgo("Harmonious Guardian", "INFJ", 9),
            SubEgo("Direct Commander", "ENTJ", 8),
            SubEgo("Empathetic Healer", "ENFJ", 2)
        ]

    def deliberate(self, topic: str, context: str = "") -> Dict[str, Any]:
        """
        Runs a debate session among the council.
        """
        logger.info(f"   [ROUND TABLE] Deliberating on: {topic}")
        
        opinions = []
        for ego in self.active_council:
            prompt = f"{ego.get_prompt_prefix()} Analyze this topic: '{topic}'. Context: '{context}'. " \
                     f"Provide a concise opinion and a 'Resonance Score' (0.0-1.0) reflecting how much you care about this. " \
                     f"Format: OPINION: <text> | RESONANCE: <score>"
            
            res = self.reasoning.think(prompt, depth=1)
            opinions.append({"ego": ego.name, "raw": res.content})

        # Synthesis
        synthesis_prompt = f"As the Sovereign Will Elysia, review these conflicting internal perspectives on '{topic}':\n"
        for op in opinions:
            synthesis_prompt += f"- {op['ego']}: {op['raw']}\n"
        synthesis_prompt += "\nSynthesize a final decision that respects the core diversity but provides a clear path forward."
        
        final_decision = self.reasoning.think(synthesis_prompt, depth=2)
        
        return {
            "topic": topic,
            "opinions": opinions,
            "synthesis": final_decision.content
        }

class PluralisticBrain:
    """
    The High-Level Controller for Elysia's many selves.
    """
    def __init__(self):
        self.reasoning = ReasoningEngine()
        self.council = InternalRoundTable(self.reasoning)
        self.personality_core = DualLayerPersonality()

    def perceive_and_deliberate(self, stimulus: str) -> str:
        """
        Main entry point for complex stimulus processing.
        """
        # Determine if the stimulus requires heavy deliberation
        if len(stimulus) > 50 or "?" in stimulus:
            debate_res = self.council.deliberate(stimulus)
            return debate_res["synthesis"]
        else:
            # Quick reaction
            return f"Elysia absorbs: {stimulus}"

pluralistic_brain = PluralisticBrain()
