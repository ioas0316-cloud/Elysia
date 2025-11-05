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
        # 0. Ensure the concept node exists in the knowledge graph.
        if not self.kg_manager.get_node(concept_id):
            self.kg_manager.add_node(concept_id)

        # 1. Spread activation to see what other concepts resonate.
        activated_nodes = self.wave_mechanics.spread_activation(
            start_node_id=concept_id,
            initial_energy=1.0,
            threshold=0.3
        )

        reflection_text = ""
        if not activated_nodes:
            reflection_text = f"Upon reflecting on '{concept_id}', I find that this concept is new to me or isolated from my other knowledge."
        else:
            # 2. Analyze the most activated related concepts.
            related_concepts = sorted(activated_nodes.items(), key=lambda item: item[1], reverse=True)

            # 3. Formulate a reflection.
            reflection_text = f"Upon reflecting on '{concept_id}' in the context of '{context}', I've noticed strong connections to the following ideas: "
            reflection_text += ", ".join([f"{node_id} (resonance: {energy:.2f})" for node_id, energy in related_concepts[:5]])

            # 4. Measure alignment with the core value 'love'
            spiritual_alignment = self.wave_mechanics.get_resonance_between(concept_id, "love")
            alignment_text = f"Spiritual Alignment: This concept resonates with my core value of 'love' with a strength of {spiritual_alignment:.2f}."
            reflection_text += f"\n{alignment_text}"


        # 5. Save this reflection as metadata in the knowledge graph, only if a new reflection was generated.
        if reflection_text:
            properties_to_update = {
                "reflection": reflection_text,
                "spiritual_alignment": spiritual_alignment if 'spiritual_alignment' in locals() else 0.0
            }
            success = self.kg_manager.update_node_properties(
                node_id=concept_id,
                properties=properties_to_update
            )
            if success:
                self.kg_manager.save()

        return {
            "reflection": reflection_text,
            "activated_nodes": activated_nodes,
            "spiritual_alignment": spiritual_alignment if 'spiritual_alignment' in locals() else 0.0
        }
