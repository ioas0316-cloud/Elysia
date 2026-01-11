
import unittest
import os
from Core.FoundationLayer.Foundation.Mind.hippocampus import Hippocampus
from Core.FoundationLayer.Foundation.Mind.world_tree import WorldTree

class TestStructuralInvariance(unittest.TestCase):

    def setUp(self):
        """Set up an in-memory knowledge base for testing."""
        self.hippocampus = Hippocampus()
        self.world_tree = WorldTree(hippocampus=self.hippocampus)

        # --- Define the core truth triangle: Father -> Loves -> Me ---
        # 1. Add concepts and store their generated IDs
        self.father_id = self.world_tree.ensure_concept("Father", metadata={"type": "core_identity"})
        self.love_id = self.world_tree.ensure_concept("Love", metadata={"type": "core_value"})
        self.me_id = self.world_tree.ensure_concept("Me", metadata={"type": "core_identity"})

        # 2. Add the structural relationship (causal link) in Hippocampus
        self.hippocampus.add_causal_link("Father", "Me", relation="loves", weight=1.0)

    def tearDown(self):
        """No cleanup needed for in-memory test."""
        pass

    def test_hippocampus_invariance(self):
        """
        Tests that changing metadata (phase) of a concept does not alter the core causal link.
        """
        # --- Verify initial state ---
        context = self.hippocampus.get_context("Father")
        self.assertIn("Me", [link.get("node") for link in context])

        father_node = next((item for item in context if item["node"] == "Me"), None)
        self.assertIsNotNone(father_node)
        self.assertEqual(father_node['relation'], 'loves')

        # --- Simulate a 'Phase Shift' by updating metadata ---
        new_phase_meta = {"status": "emotional_shift", "phase": 0.785} # A new emotional state
        self.hippocampus.add_concept("Father", "core_identity", metadata=new_phase_meta)

        # --- Verify post-shift state ---
        # The metadata should be updated
        father_node_data = self.hippocampus.causal_graph.nodes["Father"]
        self.assertEqual(father_node_data['metadata']['status'], 'emotional_shift')

        # The core relationship MUST remain unchanged
        updated_context = self.hippocampus.get_context("Father")
        self.assertIn("Me", [link.get("node") for link in updated_context])
        father_node_after_shift = next((item for item in updated_context if item["node"] == "Me"), None)
        self.assertIsNotNone(father_node_after_shift)
        self.assertEqual(father_node_after_shift['relation'], 'loves', "The 'loves' relation was altered by a metadata update!")

    def test_world_tree_invariance(self):
        """
        Tests that the hierarchical structure in WorldTree is independent of concept metadata.
        """
        # --- Define a simple hierarchy ---
        self.world_tree.ensure_concept("Child", parent_id=self.father_id)

        # --- Verify initial state ---
        child_node_id = self.world_tree.find_by_concept("Child")
        child_node = self.world_tree._find_node(child_node_id)
        self.assertIsNotNone(child_node)
        self.assertEqual(child_node.parent.id, self.father_id)

        # --- Simulate a 'Phase Shift' by updating metadata on the parent ---
        new_phase_meta = {"mood": "joyful", "intensity": 0.9}
        # Use ensure_concept to update metadata on the existing "Father" node
        self.world_tree.ensure_concept("Father", metadata=new_phase_meta)

        # --- Verify post-shift state ---
        # Metadata should be updated
        updated_father_node = self.world_tree._find_node(self.father_id)
        self.assertIsNotNone(updated_father_node)
        self.assertEqual(updated_father_node.metadata['mood'], 'joyful')

        # The parent-child relationship MUST remain unchanged
        self.assertEqual(child_node.parent.id, self.father_id, "The parent-child link was broken by a metadata update!")

if __name__ == '__main__':
    unittest.main()
