
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from Core.Mind.hippocampus import Hippocampus

class TestVisualCortex(unittest.TestCase):
    def setUp(self):
        self.hippocampus = Hippocampus()
        
    def test_star_eating(self):
        # Simulate a video with 3 distinct "scenes" (frames)
        # We use strings as mock frames because our mock VisualCortex hashes them to create deterministic vectors
        video_id = "cat_video_01"
        frames = ["cat_sleeping", "cat_meowing", "cat_jumping"]
        
        # Ingest
        self.hippocampus.ingest_visual_experience(video_id, frames)
        
        # Verify Resonance Index grew
        # Initial index size depends on DB, but we added 3 frames
        # We can check if specific IDs exist in the VISUAL index
        self.assertIn("cat_video_01:0.00", self.hippocampus.resonance.visual_id_to_idx) # Frame 0
        self.assertIn("cat_video_01:0.03", self.hippocampus.resonance.visual_id_to_idx) # Frame 1 (1/30s)
        
    def test_visual_recall(self):
        video_id = "test_recall"
        frames = ["scene_A", "scene_B", "scene_C"]
        self.hippocampus.ingest_visual_experience(video_id, frames)
        
        # Query for "scene_B"
        # We need to generate the SAME vector that "scene_B" produced.
        # Our mock VisualCortex is deterministic if input is string.
        target_vector = self.hippocampus.visual_cortex.compress_frame("scene_B")
        
        # Search Resonance
        results = self.hippocampus.resonance.find_temporal_resonance(target_vector, k=1)
        
        print(f"DEBUG: Search Results: {results}")
        
        # Should find "test_recall:0.03" (Frame 1)
        # Note: Frame 0 is 0.00, Frame 1 is 1/30 = 0.0333...
        if not results:
            self.fail("No results found for visual recall.")
            
        found_id = results[0][0]
        self.assertTrue(found_id.startswith("test_recall"), f"Expected ID starting with 'test_recall', got '{found_id}'")
        
        # Check score
        self.assertAlmostEqual(results[0][1], 1.0, places=4)

if __name__ == "__main__":
    unittest.main()
