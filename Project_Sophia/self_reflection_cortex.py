import json
from typing import Dict, Any, List, Optional

from Project_Sophia.core_memory import CoreMemory
from tools.kg_manager import KGManager

class SelfReflectionCortex:
    """
    A module dedicated to Elysia's metacognition. It allows her to analyze her own
    internal states (thoughts, emotions, knowledge) to identify gaps, inconsistencies,
    and opportunities for growth.
    """

    def __init__(self, core_memory: CoreMemory, kg_manager: KGManager):
        """
        Initializes the SelfReflectionCortex.

        Args:
            core_memory: An instance of CoreMemory to access past experiences and identity.
            kg_manager: An instance of KGManager to interact with the knowledge graph.
        """
        self.core_memory = core_memory
        self.kg_manager = kg_manager

    def analyze_internal_state(self, echo: Dict[str, float], emotional_state: Any) -> List[str]:
        """
        The main entry point for self-reflection. It orchestrates the analysis of
        Elysia's current thoughts and feelings to generate learning goals.

        Args:
            echo: The current echo of activated concepts in the cognition pipeline.
            emotional_state: The current emotional state of Elysia.

        Returns:
            A list of learning goals (as strings) to be added to the learning queue.
        """
        knowledge_gaps = self._identify_knowledge_gaps(echo)
        # In the future, we can add more analysis methods here, e.g., emotional analysis.

        learning_goals = self._generate_learning_goals(knowledge_gaps)

        return learning_goals

    def _identify_knowledge_gaps(self, echo: Dict[str, float]) -> List[str]:
        """
        Identifies concepts in the echo that are not well-defined or connected
        in the knowledge graph.

        Args:
            echo: The current echo of activated concepts.

        Returns:
            A list of concepts that represent knowledge gaps.
        """
        gaps = []
        for concept_id, activation_score in echo.items():
            # A simple heuristic: if a concept is highly active but has few connections,
            # it might be a knowledge gap.
            if activation_score > 0.5: # High activation threshold
                node = self.kg_manager.get_node(concept_id)
                if node:
                    connections = self.kg_manager.get_connections(concept_id)
                    if len(connections) < 2: # Low connection threshold
                        gaps.append(concept_id)
                else:
                    # The concept is active but doesn't exist in the KG.
                    gaps.append(concept_id)
        return list(set(gaps)) # Return unique gaps

    def _generate_learning_goals(self, knowledge_gaps: List[str]) -> List[str]:
        """
        Translates identified knowledge gaps into concrete learning goals.

        Args:
            knowledge_gaps: A list of concepts that are not well understood.

        Returns:
            A list of learning goals as strings.
        """
        goals = []
        for gap in knowledge_gaps:
            goals.append(f"Define and understand the concept of '{gap}' and its relationship to other concepts.")
        return goals
