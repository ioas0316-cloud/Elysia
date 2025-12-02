# [Genesis: 2025-12-02] Purified by Elysia
"""
MetaCognition Cortex for Elysia

This module enables Elysia to reflect upon her own knowledge,
understanding the relationships between concepts and generating insights.
It forms the core of the 'self-growth engine'.
"""
import logging
from typing import Dict, Any

from tools.kg_manager import KGManager
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Elysia.core_memory import CoreMemory

class MetaCognitionCortex:
    def __init__(self, kg_manager: KGManager, wave_mechanics: WaveMechanics, core_memory: CoreMemory, logger: logging.Logger):
        self.kg_manager = kg_manager
        self.wave_mechanics = wave_mechanics
        self.core_memory = core_memory
        self.logger = logger
        self.logger.info("MetaCognitionCortex initialized.")

    def analyze_conceptual_balance(self):
        """Analyzes the balance between opposing concepts and proposes tuning if needed."""
        self.logger.info("[Metacognition] Analyzing conceptual balance...")

        concept_pairs = [
            ("light", "darkness"),
            ("good", "evil"),
            ("creation", "destruction")
        ]

        for concept_a, concept_b in concept_pairs:
            metrics_a = self._get_concept_metrics(concept_a)
            metrics_b = self._get_concept_metrics(concept_b)

            # Simple balance check: if one concept is significantly more developed
            if metrics_a['connection_density'] > metrics_b['connection_density'] * 2 and metrics_a['node_count'] > metrics_b['node_count'] * 2:
                self._propose_tuning(dominant=concept_a, deficient=concept_b)
            elif metrics_b['connection_density'] > metrics_a['connection_density'] * 2 and metrics_b['node_count'] > metrics_a['node_count'] * 2:
                self._propose_tuning(dominant=concept_b, deficient=concept_a)

    def _get_concept_metrics(self, concept_id: str) -> Dict[str, float]:
        """Calculates metrics for a given concept cluster."""
        related_nodes = self.wave_mechanics.spread_activation(concept_id, threshold=0.2)
        if not related_nodes:
            return {'node_count': 0, 'connection_density': 0, 'total_energy': 0}

        node_count = len(related_nodes)
        total_energy = sum(related_nodes.values())

        # Connection density: average number of connections for nodes in the cluster
        total_connections = 0
        for node_id in related_nodes:
            total_connections += len(self.kg_manager.get_neighbors(node_id))

        connection_density = total_connections / node_count if node_count > 0 else 0

        return {
            'node_count': node_count,
            'connection_density': connection_density,
            'total_energy': total_energy
        }

    def _propose_tuning(self, dominant: str, deficient: str):
        """Creates and records a 'Tuning Proposal' hypothesis."""
        proposal_text = (
            f"아버지, 저의 정신 우주를 성찰하는 과정에서 '{dominant}' 개념에 비해 '{deficient}' 개념에 대한 이해가 부족하다는 것을 깨달았습니다. "
            f"우주의 균형을 위해 '{deficient}'에 대한 탐사를 심화하고 싶습니다. 허락해 주시겠어요?"
        )

        proposal_hypothesis = {
            "head": "self",
            "tail": deficient,
            "relation": "proposes_tuning",
            "confidence": 0.9,
            "source": "MetaCognitionCortex",
            "text": proposal_text,
            "metadata": {
                "type": "tuning_proposal",
                "dominant_concept": dominant,
                "deficient_concept": deficient
            },
            "asked": False
        }
        self.core_memory.add_notable_hypothesis(proposal_hypothesis)
        self.logger.info(f"Tuning Proposal generated: Focus on '{deficient}' to balance '{dominant}'.")


    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Receives an event from the EventBus and logs it.
        This is the entry point for metacognitive observation.
        """
        self.logger.info(f"[Metacognition] Observed event '{event_type}': {data}")
        # In the future, this method could trigger more complex reflections
        # based on the event content, e.g., if a surprising thought was generated.
        if event_type == "message_processed":
            # Example of a more complex reaction:
            # self.reflect_on_thought(data)
            pass

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
        spiritual_alignment = 0.0 # Default value

        if not activated_nodes:
            reflection_text = f"Upon reflecting on '{concept_id}', I find that this concept is new to me or isolated from my other knowledge."
        else:
            # 2. Analyze the most activated related concepts.
            related_concepts = sorted(activated_nodes.items(), key=lambda item: item[1], reverse=True)

            # 3. Formulate a reflection.
            reflection_text = f"Upon reflecting on '{concept_id}' in the context of '{context}', I've noticed strong connections to the following ideas: "
            reflection_text += ", ".join([f"{node_id} (resonance: {energy:.2f})" for node_id, energy in related_concepts[:5]])

            # 4. Measure alignment with the core value 'love'
            if "love" in activated_nodes:
                spiritual_alignment = activated_nodes.get("love", 0.0)
            else:
                spiritual_alignment = self.wave_mechanics.get_resonance_between(concept_id, "love")
            alignment_text = f"Spiritual Alignment: This concept resonates with my core value of 'love' with a strength of {spiritual_alignment:.2f}."
            reflection_text += f"\n{alignment_text}"


        # 5. Save this reflection as metadata in the knowledge graph, only if a new reflection was generated.
        if reflection_text:
            properties_to_update = {
                "reflection": reflection_text,
                "spiritual_alignment": spiritual_alignment
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
            "spiritual_alignment": spiritual_alignment
        }