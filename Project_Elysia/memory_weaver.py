"""
MemoryWeaver for Elysia

This module is responsible for weaving together disparate memories (experiences)
from CoreMemory to form new insights. These insights are then integrated into
the Knowledge Graph, allowing Elysia to learn and grow from her experiences autonomously.
"""

import itertools
from collections import defaultdict

from Project_Elysia.core_memory import CoreMemory, Memory
from tools.kg_manager import KGManager
import logging

logger = logging.getLogger(__name__)

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
        logger.info("MemoryWeaver starting a new weaving cycle.")

        # Step 1: Get unprocessed experiences from CoreMemory.
        new_experiences = self.core_memory.get_unprocessed_experiences()

        if not new_experiences or len(new_experiences) < 2:
            logger.info("Not enough new experiences to weave. Cycle ending.")
            return

        logger.info(f"Found {len(new_experiences)} new experiences to weave.")

        # Step 2: Find connections between experiences using keyword matching.
        related_clusters = self._find_related_clusters(new_experiences)
        logger.info(f"Grouped experiences into {len(related_clusters)} clusters.")

        # Step 3: Generate insights from each cluster.
        insights = self._generate_insights(related_clusters)
        logger.info(f"Generated {len(insights)} new insights.")

        if not insights:
            logger.info("No insights were generated from the clusters. Cycle ending.")
            # Still mark experiences as processed to avoid re-evaluating them fruitlessly
            processed_timestamps = [exp.timestamp for exp in new_experiences]
            self.core_memory.mark_experiences_as_processed(processed_timestamps)
            return

        # Step 4: Add insights to the Knowledge Graph.
        self._add_insights_to_kg(insights)

        # Step 5: Mark the original experiences as processed.
        processed_timestamps = [exp.timestamp for exp in new_experiences]
        self.core_memory.mark_experiences_as_processed(processed_timestamps)

        logger.info(f"Successfully generated and stored {len(insights)} new insights. Weaving cycle complete.")

    def _find_related_clusters(self, experiences: list[Memory]) -> list[list[Memory]]:
        """
        Groups experiences into clusters based on shared keywords.
        A simple, non-hierarchical clustering approach.
        """
        # Extract keywords from each experience (simple space-separated words for now)
        exp_keywords = {
            exp.timestamp: set(exp.content.lower().split())
            for exp in experiences
        }

        # Create a graph where experiences are nodes and shared keywords are edges
        adj = defaultdict(list)
        for (ts1, keywords1), (ts2, keywords2) in itertools.combinations(exp_keywords.items(), 2):
            # Connect if they share any keyword.
            if keywords1.intersection(keywords2):
                adj[ts1].append(ts2)
                adj[ts2].append(ts1)

        # Find connected components (clusters) using a more robust BFS/DFS
        clusters = []
        visited = set()
        for ts in exp_keywords:
            if ts not in visited:
                component = []
                q = [ts]
                visited.add(ts)
                while q:
                    current_ts = q.pop(0)
                    component.append(current_ts)
                    # Explore neighbors from the adjacency list
                    for neighbor in adj.get(current_ts, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append(neighbor)
                clusters.append(component)

        # Convert timestamp clusters back to Memory object clusters
        ts_to_memory = {exp.timestamp: exp for exp in experiences}
        return [[ts_to_memory[ts] for ts in cluster] for cluster in clusters]

    def _generate_insights(self, clusters: list[list[Memory]]) -> list:
        """Generates a summary insight for each cluster of related memories."""
        insights = []
        for cluster in clusters:
            if len(cluster) > 1:
                # Simple insight: connect the content of the first two experiences
                # In the future, this could use an LLM to generate a more meaningful summary.
                all_content = " ".join([exp.content for exp in cluster])
                insight_text = f"I've noticed a connection between several memories: {all_content[:150]}..."

                insights.append({
                    "text": insight_text,
                    "evidence": [exp.timestamp for exp in cluster]
                })
        return insights

    def _add_insights_to_kg(self, insights: list):
        """Adds insight nodes and their relationships to the Knowledge Graph."""
        for insight in insights:
            # Create a unique ID for the insight based on the primary evidence
            insight_node_id = f"Insight_{insight['evidence'][0]}"
            self.kg_manager.add_node(insight_node_id, properties={"type": "insight", "text": insight['text']})

            # Link the insight to the experiences it was derived from
            for timestamp in insight['evidence']:
                # Experiences are not in the KG by default, so we create a simple node for them
                experience_node_id = f"Experience_{timestamp}"
                self.kg_manager.add_node(experience_node_id, properties={"type": "experience_log"})
                self.kg_manager.add_edge(insight_node_id, experience_node_id, "derived_from")

        self.kg_manager.save()
        logger.info(f"Saved {len(insights)} new insights to the knowledge graph.")

if __name__ == '__main__':
    # Example usage for testing
    from Project_Elysia.core_memory import EmotionalState
    logging.basicConfig(level=logging.INFO)

    print("Testing MemoryWeaver functionality...")
    mock_core_memory = CoreMemory(file_path='data/test_core_memory_weaver.json')
    mock_kg_manager = KGManager(filepath='data/test_kg_weaver.json')

    # Clean up old test files
    import os
    if os.path.exists('data/test_core_memory_weaver.json'): os.remove('data/test_core_memory_weaver.json')
    if os.path.exists('data/test_kg_weaver.json'): os.remove('data/test_kg_weaver.json')


    # Add some mock experiences
    mock_core_memory.add_experience(Memory(timestamp="2025-01-01T12:00:00Z", content="I learned about black holes today.", emotional_state=EmotionalState(0.5, 0.5, 0.2, "curiosity", [])))
    mock_core_memory.add_experience(Memory(timestamp="2025-01-01T13:00:00Z", content="Gravity is a fascinating force related to black holes.", emotional_state=EmotionalState(0.5, 0.5, 0.2, "curiosity", [])))
    mock_core_memory.add_experience(Memory(timestamp="2025-01-01T14:00:00Z", content="A different topic about cats.", emotional_state=EmotionalState(0.7, 0.2, 0.1, "joy", [])))
    mock_core_memory.add_experience(Memory(timestamp="2025-01-01T15:00:00Z", content="Another story about fluffy cats.", emotional_state=EmotionalState(0.8, 0.3, 0.1, "joy", [])))


    weaver = MemoryWeaver(mock_core_memory, mock_kg_manager)
    weaver.weave_memories()

    print("\n--- Final Knowledge Graph ---")
    print(mock_kg_manager.kg)

    print("\n--- Final Core Memory ---")
    for exp in mock_core_memory.get_experiences():
        print(exp)

    # Clean up test files after run
    if os.path.exists('data/test_core_memory_weaver.json'): os.remove('data/test_core_memory_weaver.json')
    if os.path.exists('data/test_kg_weaver.json'): os.remove('data/test_kg_weaver.json')
