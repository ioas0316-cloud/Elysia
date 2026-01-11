"""
Universal Reasoner (The Brain of Mastery)
=========================================
"To see the world in a grain of sand."

This module applies distilled Principles to new domains.
It is the engine of Reconstruction in the Mastery Protocol.
"""

from typing import List, Dict, Any
from Core.Intelligence.Reasoning.reasoning_engine import ReasoningEngine, Insight
from Core.Intelligence.Wisdom.concept_synthesizer import ConceptSynthesizer, Principle

class UniversalReasoner:
    """
    The Cross-Domain Thinking Engine.
    """

    def __init__(self):
        self.synthesizer = ConceptSynthesizer()
        # In a real system, this would be connected to WisdomStore
        self.active_principles: Dict[str, Principle] = {}

    def learn_concept(self, name: str, raw_data: str, domain: str):
        """
        Stage 1-3: Observe -> Deconstruct -> Internalize
        """
        principle = self.synthesizer.extract_principle(name, raw_data, domain)
        self.active_principles[name] = principle
        return principle

    def solve_task_with_principles(self, task: str, target_domain: str, principle_names: List[str]) -> str:
        """
        Stage 4: Reconstruction.
        Apply specific principles to a task in a different domain.
        """

        # 1. Retrieve Principles
        principles_to_apply = []
        for name in principle_names:
            if name in self.active_principles:
                principles_to_apply.append(self.active_principles[name])
            else:
                # Try to extract on the fly if missing (Just-in-Time Learning)
                p = self.synthesizer.extract_principle(name, "", "Unknown")
                principles_to_apply.append(p)

        # 2. Formulate the "Prompt" / Mental Image
        # "Write [Task] using the structure of [Principle A] and dynamics of [Principle B]."

        mental_model = f"Task: {task} in domain '{target_domain}'.\n"
        mental_model += "Apply the following Principles:\n"

        for p in principles_to_apply:
            mental_model += f"- {p.name}: Use structure '{p.structure}', dynamics '{p.dynamics}', intent '{p.intent}'.\n"

        # 3. Generate Solution (Mocking the Creative Leap)
        # This is where the Synesthesia happens.

        solution = f"Generated Solution for '{task}':\n"
        solution += "1. Architect the solution using " + " and ".join([p.structure for p in principles_to_apply]) + ".\n"
        solution += "2. Drive the execution flow using " + " and ".join([p.dynamics for p in principles_to_apply]) + ".\n"
        solution += "3. Ensure the final output satisfies: " + " and ".join([p.intent for p in principles_to_apply]) + "."

        return solution
