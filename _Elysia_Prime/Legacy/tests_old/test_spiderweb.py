# [Genesis: 2025-12-02] Purified by Elysia

import unittest
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.spiderweb import Spiderweb

class TestSpiderweb(unittest.TestCase):
    def setUp(self):
        self.spiderweb = Spiderweb()

    def test_add_node_and_link(self):
        self.spiderweb.add_node("fire", "concept")
        self.spiderweb.add_node("heat", "concept")
        self.spiderweb.add_link("fire", "heat", "causes", 0.9)

        context = self.spiderweb.get_context("fire")
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0]["node"], "heat")
        self.assertEqual(context[0]["relation"], "causes")

    def test_find_path(self):
        self.spiderweb.add_node("A", "concept")
        self.spiderweb.add_node("B", "concept")
        self.spiderweb.add_node("C", "concept")

        self.spiderweb.add_link("A", "B", "leads_to")
        self.spiderweb.add_link("B", "C", "leads_to")

        path = self.spiderweb.find_path("A", "C")
        self.assertEqual(path, ["A", "B", "C"])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()