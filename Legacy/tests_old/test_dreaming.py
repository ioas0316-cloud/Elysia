
import unittest
import sys
import os
import logging
from unittest.mock import MagicMock
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.spiderweb import Spiderweb
from Project_Sophia.dreaming_cortex import DreamingCortex
from Project_Elysia.core_memory import Experience

class TestDreamingCortex(unittest.TestCase):
    def setUp(self):
        self.mock_core_memory = MagicMock()
        self.spiderweb = Spiderweb()
        self.dreaming_cortex = DreamingCortex(self.mock_core_memory, self.spiderweb)

    def test_dream_process(self):
        # Setup mock experiences
        exp1 = Experience(
            timestamp="2023-10-27T10:00:00",
            content="I saw a fire burning.",
            type="episode"
        )
        exp2 = Experience(
            timestamp="2023-10-27T10:05:00",
            content="The fire was hot.",
            type="episode"
        )
        
        self.mock_core_memory.get_unprocessed_experiences.return_value = [exp1, exp2]
        
        # Run dream
        self.dreaming_cortex.dream()
        
        # Verify Spiderweb population
        # "fire" should be a node
        self.assertTrue(self.spiderweb.graph.has_node("fire"))
        
        # Verify that experiences were marked as processed
        self.mock_core_memory.mark_experiences_as_processed.assert_called()
        
        # Check for links (associative)
        # Both experiences mention "fire", so they should be linked via the "fire" concept
        path = self.spiderweb.find_path("event_2023-10-27T10:00:00", "event_2023-10-27T10:05:00")
        # Path should exist: event1 -> fire -> event2 (or similar indirect path)
        # Note: My simple implementation links event -> concept. 
        # So event1 -> fire, event2 -> fire. 
        # The graph is directed. event -> concept. 
        # To find path from event1 to event2, we need concept -> event links?
        # Ah, my implementation adds: self.spiderweb.add_link(neighbor["node"], event_id, relation="associative_link")
        # if neighbor is an event. 
        # Wait, neighbor["node"] comes from get_context(concept_id).
        # If event1 added "fire", then "fire" has incoming link from event1.
        # When event2 adds "fire", get_context("fire") will show event1 (incoming).
        # My code checks: if neighbor["node"].startswith("event_").
        # get_context returns both incoming and outgoing.
        # So it should find event1 and link event1 -> event2.
        
        # Let's verify the path
        self.assertTrue(len(path) > 0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
