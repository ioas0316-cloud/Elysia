"""
MemoryWeaver for Elysia

This module is responsible for weaving together disparate memories (experiences)
from CoreMemory to form new insights. These insights are then integrated into
the Knowledge Graph, allowing Elysia to learn and grow from her experiences autonomously.
"""

from Project_Elysia.core_memory import CoreMemory
from tools.kg_manager import KGManager

class MemoryWeaver:
    def __init__(self, core_memory: CoreMemory, kg_manager: KGManager):
        """
        Initializes the MemoryWeaver.

        Args:
            core_memory: An instance of CoreMemory to access experiences.
            kg_manager: An instance of KGManager to write new insights to.
        """
        self.core_memory = core_memory
        self.kg_manager = kg_manager

    def weave_memories(self):
        """
        The main process of weaving memories into insights.
        This process should be triggered periodically (e.g., during an idle cycle).
        """
        # Step 1: Get unprocessed experiences from CoreMemory.
        # (This method needs to be implemented in CoreMemory)
        new_experiences = self.core_memory.get_unprocessed_experiences()

        if not new_experiences:
            print("[MemoryWeaver] No new experiences to weave.")
            return

        print(f"[MemoryWeaver] Found {len(new_experiences)} new experiences to weave.")

        # Step 2: Find connections between experiences (e.g., using embeddings or keywords).
        # (Implementation pending)
        related_clusters = self._find_related_clusters(new_experiences)

        # Step 3: Generate insights from each cluster.
        # (Implementation pending)
        insights = self._generate_insights(related_clusters)

        # Step 4: Add insights to the Knowledge Graph.
        # (Implementation pending)
        self._add_insights_to_kg(insights)

        # Step 5: Mark the original experiences as processed.
        # (This method needs to be implemented in CoreMemory)
        self.core_memory.mark_experiences_as_processed(new_experiences)

        print(f"[MemoryWeaver] Successfully generated and stored {len(insights)} new insights.")

    def _find_related_clusters(self, experiences: list) -> list:
        # Placeholder for clustering logic
        # For now, we can return a single cluster for simplicity
        return [experiences] if experiences else []

    def _generate_insights(self, clusters: list) -> list:
        # Placeholder for insight generation logic
        insights = []
        for cluster in clusters:
            if len(cluster) > 1:
                # Simple insight: connect the content of the first two experiences
                insight_text = f"The memory '{cluster[0].content}' seems related to the memory '{cluster[1].content}'."
                insights.append({"text": insight_text, "evidence": [exp.timestamp for exp in cluster]})
        return insights

    def _add_insights_to_kg(self, insights: list):
        # Placeholder for KG integration logic
        for insight in insights:
            insight_node_id = f"Insight_{insight['evidence'][0]}"
            self.kg_manager.add_node(insight_node_id, properties={"type": "insight", "text": insight['text']})
            for timestamp in insight['evidence']:
                # Assuming experience nodes are identifiable by their timestamp
                experience_node_id = f"Experience_{timestamp}"
                self.kg_manager.add_edge(insight_node_id, experience_node_id, "derived_from")
        self.kg_manager.save()

if __name__ == '__main__':
    # Example usage for testing
    print("Testing MemoryWeaver functionality...")
    mock_core_memory = CoreMemory()
    mock_kg_manager = KGManager(filepath='data/test_kg_weaver.json')

    # Add some mock experiences
    from Project_Elysia.core_memory import Memory, EmotionalState
    mock_core_memory.add_experience(Memory(timestamp="2025-01-01T12:00:00", content="I learned about black holes today.", emotional_state=EmotionalState(0.5, 0.5, 0.2, "curiosity", [])))
    mock_core_memory.add_experience(Memory(timestamp="2025-01-01T13:00:00", content="Gravity is a fascinating force.", emotional_state=EmotionalState(0.5, 0.5, 0.2, "curiosity", [])))

    # Mock the get_unprocessed_experiences method
    def get_mock_experiences():
        return mock_core_memory.get_experiences()
    mock_core_memory.get_unprocessed_experiences = get_mock_experiences
    mock_core_memory.mark_experiences_as_processed = lambda exps: print(f"Marked {len(exps)} experiences as processed.")

    weaver = MemoryWeaver(mock_core_memory, mock_kg_manager)
    weaver.weave_memories()

    print("\nFinal Knowledge Graph:")
    print(mock_kg_manager.kg)
