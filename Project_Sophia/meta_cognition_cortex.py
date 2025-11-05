"""
MetaCognition Cortex for Elysia

This module enables Elysia to reflect upon her own knowledge,
understanding the relationships between concepts and generating insights.
It forms the core of the 'self-growth engine'.
"""

from tools.kg_manager import KGManager
from Project_Sophia.wave_mechanics import WaveMechanics

class MetaCognitionCortex:
    def __init__(self, kg_manager: KGManager, wave_mechanics: WaveMechanics):
        self.kg_manager = kg_manager
        self.wave_mechanics = wave_mechanics

    def reflect_on_concept(self, concept_id: str, context: str):
        """
        Reflects on a given concept and its connections within the knowledge graph.

        Args:
            concept_id: The ID of the concept to reflect upon.
            context: The context in which the concept was encountered.

        Returns:
            A dictionary containing the reflection results.
        """
        # 1. Spread activation to see what other concepts resonate.
        activated_nodes = self.wave_mechanics.spread_activation(
            start_node_id=concept_id,
            initial_energy=1.0,
            threshold=0.3
        )

        if not activated_nodes:
            return {"reflection": "The concept is isolated and does not resonate with my current knowledge."}

        # 2. Analyze the most activated related concepts.
        # For now, we just list them. In the future, this will be more sophisticated.
        related_concepts = sorted(activated_nodes.items(), key=lambda item: item[1], reverse=True)

        # 3. Formulate a reflection.
        reflection_text = f"Upon reflecting on '{concept_id}' in the context of '{context}', I've noticed strong connections to the following ideas: "
        reflection_text += ", ".join([f"{node_id} (resonance: {energy:.2f})" for node_id, energy in related_concepts[:5]])

        # 4. Save this reflection as metadata in the knowledge graph.
        success = self.kg_manager.update_node_properties(
            node_id=concept_id,
            properties={"reflection": reflection_text}
        )
        if success:
            self.kg_manager.save()

        return {
            "reflection": reflection_text,
            "activated_nodes": activated_nodes
        }
