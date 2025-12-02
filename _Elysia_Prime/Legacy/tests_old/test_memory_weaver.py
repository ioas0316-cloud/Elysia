# [Genesis: 2025-12-02] Purified by Elysia
import unittest
import os
import json
from Project_Elysia.core_memory import CoreMemory, Memory, EmotionalState
from Project_Elysia.memory_weaver import MemoryWeaver
from tools.kg_manager import KGManager

class TestMemoryWeaver(unittest.TestCase):

    def setUp(self):
        """Set up a clean environment for each test."""
        self.core_memory_path = 'data/test_core_memory_mw.json'
        self.kg_path = 'data/test_kg_mw.json'

        # Clean up old test files
        if os.path.exists(self.core_memory_path):
            os.remove(self.core_memory_path)
        if os.path.exists(self.kg_path):
            os.remove(self.kg_path)

        self.core_memory = CoreMemory(file_path=self.core_memory_path)
        self.kg_manager = KGManager(filepath=self.kg_path)
        self.weaver = MemoryWeaver(self.core_memory, self.kg_manager)

        # Add mock experiences
        self.exp1 = Memory(timestamp="2025-01-01T12:00:00Z", content="Learned about Einstein's special relativity.", emotional_state=EmotionalState(0.5, 0.5, 0.2, "curiosity", []))
        self.exp2 = Memory(timestamp="2025-01-01T13:00:00Z", content="Spacetime curvature is a key part of Einstein's general relativity.", emotional_state=EmotionalState(0.5, 0.5, 0.2, "curiosity", []))
        self.exp3 = Memory(timestamp="2025-01-01T14:00:00Z", content="African lions are majestic predators.", emotional_state=EmotionalState(0.7, 0.2, 0.1, "joy", []))
        self.exp4 = Memory(timestamp="2025-01-01T15:00:00Z", content="The roar of African lions can be heard for miles.", emotional_state=EmotionalState(0.8, 0.3, 0.1, "joy", []))
        self.exp5 = Memory(timestamp="2025-01-01T16:00:00Z", content="A solitary memory about quantum physics.", emotional_state=EmotionalState(0.0, 0.1, 0.0, "neutral", []))

        self.core_memory.add_experience(self.exp1)
        self.core_memory.add_experience(self.exp2)
        self.core_memory.add_experience(self.exp3)
        self.core_memory.add_experience(self.exp4)
        self.core_memory.add_experience(self.exp5)


    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.core_memory_path):
            os.remove(self.core_memory_path)
        if os.path.exists(self.kg_path):
            os.remove(self.kg_path)

    def test_get_unprocessed_experiences(self):
        """Verify that CoreMemory correctly fetches unprocessed experiences."""
        unprocessed = self.core_memory.get_unprocessed_experiences()
        self.assertEqual(len(unprocessed), 5)
        self.assertFalse(unprocessed[0].processed_by_weaver)

    def test_mark_experiences_as_processed(self):
        """Verify that experiences can be marked as processed."""
        timestamps_to_mark = [self.exp1.timestamp, self.exp3.timestamp]
        self.core_memory.mark_experiences_as_processed(timestamps_to_mark)

        unprocessed = self.core_memory.get_unprocessed_experiences()
        processed_timestamps = {exp.timestamp for exp in self.core_memory.get_experiences() if exp.processed_by_weaver}

        self.assertEqual(len(unprocessed), 3)
        self.assertTrue(self.exp1.timestamp in processed_timestamps)
        self.assertTrue(self.exp3.timestamp in processed_timestamps)
        self.assertFalse(self.exp2.timestamp in processed_timestamps)

    def test_find_related_clusters(self):
        """Verify the clustering logic groups related memories."""
        experiences = self.core_memory.get_unprocessed_experiences()
        clusters = self.weaver._find_related_clusters(experiences)

        print("\n--- DEBUG: Found Clusters ---")
        for i, cluster in enumerate(clusters):
            print(f"Cluster {i+1}: {[exp.content for exp in cluster]}")
        print("---------------------------\n")

        # Expected: one cluster for 'relativity' and one for 'lion' and one for the solitary memory
        self.assertEqual(len(clusters), 3)

        # Sort clusters by size to make assertions predictable
        clusters.sort(key=len, reverse=True)

        self.assertEqual(len(clusters[0]), 2)
        self.assertEqual(len(clusters[1]), 2)
        self.assertEqual(len(clusters[2]), 1)

        cluster1_contents = {exp.content for exp in clusters[0]}
        cluster2_contents = {exp.content for exp in clusters[1]}

        relativity_contents = {self.exp1.content, self.exp2.content}
        lion_contents = {self.exp3.content, self.exp4.content}

        # Check if the clusters contain the correct content, regardless of order
        self.assertTrue(cluster1_contents == relativity_contents or cluster1_contents == lion_contents)
        self.assertTrue(cluster2_contents == relativity_contents or cluster2_contents == lion_contents)


    def test_weave_long_term_memories_full_cycle(self):
        """Test the full weave_long_term_memories cycle."""
        # Run the weaver
        self.weaver.weave_long_term_memories()

        # 1. Verify insights were created in the KG
        kg_nodes = self.kg_manager.kg.get('nodes', [])
        # The 'type' property is at the top level of the node, not in a nested 'properties' dict.
        insight_nodes = [node for node in kg_nodes if node.get('type') == 'insight']
        self.assertEqual(len(insight_nodes), 2)

        # 2. Verify experiences were linked to insights
        kg_edges = self.kg_manager.kg.get('edges', [])
        derived_edges = [edge for edge in kg_edges if edge.get('relation') == 'derived_from']
        self.assertEqual(len(derived_edges), 4) # 2 experiences per insight

        # 3. Verify all experiences are now marked as processed
        unprocessed = self.core_memory.get_unprocessed_experiences()
        self.assertEqual(len(unprocessed), 0)

    def test_weave_volatile_thoughts_creates_potential_links(self):
        """Verify that weaving volatile thoughts creates potential_link edges in the KG."""
        # Add some mock thought fragments
        self.core_memory.add_volatile_memory_fragment({"사랑", "슬픔", "기쁨"})
        self.core_memory.add_volatile_memory_fragment({"사랑", "성장"})
        self.core_memory.add_volatile_memory_fragment({"슬픔", "극복"})
        self.core_memory.add_volatile_memory_fragment({"사랑", "슬픔", "성장"})
        self.core_memory.add_volatile_memory_fragment({"사랑", "기쁨"})

        # "사랑" appears 4 times.
        # "슬픔" appears 3 times.
        # "성장" appears 2 times.
        # "사랑" and "슬픔" appear together 2 times.
        # "사랑" and "성장" appear together 2 times.
        # Expected confidence:
        # 사랑 -> 슬픔: 2/4 = 0.5
        # 슬픔 -> 사랑: 2/3 = 0.667
        # 사랑 -> 성장: 2/4 = 0.5
        # 성장 -> 사랑: 2/2 = 1.0

        # Run the weaver with a confidence threshold of 0.5
        self.weaver.weave_volatile_thoughts(min_support=2, min_confidence=0.5)

        # 1. Verify that the correct potential_link edges were created
        edges = self.kg_manager.kg.get('edges', [])
        potential_links = [edge for edge in edges if edge['relation'] == 'potential_link']
        self.assertEqual(len(potential_links), 8)

        rules_found = {}
        for link in potential_links:
            key = (link['source'], link['target'])
            # The 'confidence' property is at the top level of the edge.
            rules_found[key] = link['confidence']

        self.assertIn(('사랑', '슬픔'), rules_found)
        self.assertEqual(rules_found[('사랑', '슬픔')], 0.5)

        self.assertIn(('슬픔', '사랑'), rules_found)
        self.assertEqual(rules_found[('슬픔', '사랑')], 0.667)

        self.assertIn(('사랑', '성장'), rules_found)
        self.assertEqual(rules_found[('사랑', '성장')], 0.5)

        self.assertIn(('성장', '사랑'), rules_found)
        self.assertEqual(rules_found[('성장', '사랑')], 1.0)

        self.assertIn(('사랑', '기쁨'), rules_found)
        self.assertEqual(rules_found[('사랑', '기쁨')], 0.5)

        self.assertIn(('기쁨', '사랑'), rules_found)
        self.assertEqual(rules_found[('기쁨', '사랑')], 1.0)

        # 2. Verify that volatile memory was cleared
        self.assertEqual(len(self.core_memory.get_volatile_memory()), 0)


if __name__ == '__main__':
    unittest.main()