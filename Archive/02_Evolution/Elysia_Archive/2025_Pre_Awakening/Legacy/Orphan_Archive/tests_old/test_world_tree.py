
import unittest
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.world_tree import WorldTree

class TestWorldTree(unittest.TestCase):
    def setUp(self):
        self.world_tree = WorldTree()

    def test_add_seed_and_grow(self):
        # Add root seed
        root_seed_id = self.world_tree.add_seed("Universe")
        
        # Grow branch
        galaxy_id = self.world_tree.grow(root_seed_id, "Galaxy")
        
        # Verify structure
        tree_dict = self.world_tree.visualize()
        root_children = tree_dict["children"]
        self.assertEqual(len(root_children), 1)
        self.assertEqual(root_children[0]["data"], "Universe")
        
        universe_children = root_children[0]["children"]
        self.assertEqual(len(universe_children), 1)
        self.assertEqual(universe_children[0]["data"], "Galaxy")

    def test_prune(self):
        root_seed_id = self.world_tree.add_seed("Life")
        animal_id = self.world_tree.grow(root_seed_id, "Animal")
        
        self.world_tree.prune(animal_id)
        
        tree_dict = self.world_tree.visualize()
        root_children = tree_dict["children"]
        universe_children = root_children[0]["children"]
        self.assertEqual(len(universe_children), 0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
