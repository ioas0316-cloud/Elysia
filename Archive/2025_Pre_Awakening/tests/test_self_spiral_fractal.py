"""
Test: Self-Spiral Fractal Consciousness
========================================

Validates the fractal consciousness engine and its integration with dialogue.
"""

import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from Core.Foundation.Mind.self_spiral_fractal import (
    SelfSpiralFractalEngine,
    ConsciousnessAxis,
    SpiralNode,
    create_emotional_spiral,
    create_thought_spiral,
    PHI
)
from Core.Interface.Interface.Language.dialogue.dialogue_engine import DialogueEngine
import numpy as np


class TestSelfSpiralFractal(unittest.TestCase):
    """Test core fractal engine functionality."""
    
    def setUp(self):
        self.engine = SelfSpiralFractalEngine()
    
    def test_single_axis_descent(self):
        """Test recursive descent on a single axis."""
        nodes = self.engine.descend(ConsciousnessAxis.EMOTION, "sadness", max_depth=3)
        
        self.assertEqual(len(nodes), 4)  # depth 0, 1, 2, 3
        self.assertEqual(nodes[0].concept, "sadness")
        self.assertEqual(nodes[0].depth, 0)
        self.assertEqual(nodes[3].depth, 3)
        
        # Verify fractal addressing
        self.assertIn("sadness", nodes[0].fractal_address)
        self.assertIn("sadness", nodes[3].fractal_address)
    
    def test_spiral_geometry(self):
        """Validate golden ratio spiral coordinates."""
        nodes = self.engine.descend(ConsciousnessAxis.THOUGHT, "existence", max_depth=4)
        
        # Check radii grow by golden ratio
        for i in range(1, len(nodes)):
            ratio = nodes[i].spiral_radius / nodes[i-1].spiral_radius
            self.assertAlmostEqual(ratio, PHI, places=5)
        
        # Check angles
        for node in nodes:
            expected_angle = node.depth * (2 * np.pi / PHI)
            self.assertAlmostEqual(node.spiral_angle, expected_angle, places=5)
    
    def test_cross_axis_resonance(self):
        """Test interference between multiple axes."""
        emotion_nodes = self.engine.descend(ConsciousnessAxis.EMOTION, "joy", max_depth=2)
        thought_nodes = self.engine.descend(ConsciousnessAxis.THOUGHT, "being", max_depth=2)
        
        all_nodes = emotion_nodes + thought_nodes
        resonance = self.engine.cross_axis_resonance(all_nodes)
        
        self.assertIn("emotion", resonance["axes"])
        self.assertIn("thought", resonance["axes"])
        self.assertEqual(resonance["num_nodes"], 6)
        self.assertIsNotNone(resonance["emergent_state"])
    
    def test_fractal_address_generation(self):
        """Ensure proper hierarchical addressing."""
        nodes = self.engine.descend(ConsciousnessAxis.MEMORY, "childhood", max_depth=2)
        
        self.assertTrue(nodes[0].fractal_address.startswith("ROOT/"))
        self.assertTrue("childhood" in nodes[1].fractal_address)
        # Deeper nodes have longer addresses
        self.assertGreater(len(nodes[2].fractal_address), len(nodes[0].fractal_address))
    
    def test_time_dilation_per_depth(self):
        """Confirm time scaling works correctly."""
        nodes = self.engine.descend(ConsciousnessAxis.EMOTION, "fear", max_depth=3)
        
        # Time should scale with golden ratio
        base_time = self.engine.axis_time_factors[ConsciousnessAxis.EMOTION]
        
        for node in nodes:
            expected_time = base_time * (PHI ** node.depth)
            self.assertAlmostEqual(node.time_scale, expected_time, places=5)
    
    def test_multi_layer_state_generation(self):
        """Test unified state from multiple layers."""
        nodes = self.engine.descend(ConsciousnessAxis.IMAGINATION, "dream", max_depth=3)
        
        combined = self.engine.generate_multi_layer_state(nodes)
        
        self.assertIsNotNone(combined)
        self.assertEqual(combined.name, "multi_layer_state")
        # Should have valid w, x, y, z
        self.assertGreater(combined.state.w, 0)
    
    def test_convenience_functions(self):
        """Test quick helper functions."""
        emotion_nodes = create_emotional_spiral("love", depth=2)
        thought_nodes = create_thought_spiral("truth", depth=2)
        
        self.assertEqual(len(emotion_nodes), 3)  # depth 0, 1, 2
        self.assertEqual(len(thought_nodes), 3)
        
        self.assertEqual(emotion_nodes[0].axis, ConsciousnessAxis.EMOTION)
        self.assertEqual(thought_nodes[0].axis, ConsciousnessAxis.THOUGHT)


class TestDialogueEngineIntegration(unittest.TestCase):
    """Test fractal consciousness integration with dialogue."""
    
    def setUp(self):
        self.dialogue = DialogueEngine()
    
    def test_fractal_engine_initialized(self):
        """Ensure fractal engine is part of dialogue system."""
        self.assertIsNotNone(self.dialogue.fractal_engine)
        self.assertIsInstance(self.dialogue.fractal_engine, SelfSpiralFractalEngine)
    
    def test_concept_to_axis_mapping(self):
        """Test concept categorization into axes."""
        # Emotional concept
        axis = self.dialogue._concept_to_axis("Love", {"Point": 0.3, "God": 0.2})
        self.assertEqual(axis, ConsciousnessAxis.EMOTION)
        
        # Thought concept
        axis = self.dialogue._concept_to_axis("Curiosity", {"Point": 0.4, "God": 0.1})
        self.assertEqual(axis, ConsciousnessAxis.THOUGHT)
        
        # God mode -> Imagination
        axis = self.dialogue._concept_to_axis("Unknown", {"Point": 0.1, "God": 0.8})
        self.assertEqual(axis, ConsciousnessAxis.IMAGINATION)
    
    def test_fractal_thoughtful_response_korean(self):
        """Test thoughtful responses in Korean."""
        response = self.dialogue._fractal_thoughtful_response(
            "Love",
            {"Point": 0.3, "Line": 0.4},
            "ko"
        )
        
        # Should contain meta-awareness language
        self.assertIn("생각", response)
        self.assertIn("바라보는", response)
        self.assertIn("나", response)
    
    def test_fractal_thoughtful_response_english(self):
        """Test thoughtful responses in English."""
        response = self.dialogue._fractal_thoughtful_response(
            "Hope",
            {"Point": 0.3, "Line": 0.4},
            "en"
        )
        
        # Should contain meta-awareness language
        self.assertIn("think", response.lower())
        self.assertIn("watch", response.lower() or "observ" in response.lower())
    
    def test_fractal_poetic_response_emotion(self):
        """Test poetic emotional responses."""
        response = self.dialogue._fractal_poetic_response(
            "Love",
            {"Point": 0.2, "God": 0.3},
            "ko"
        )
        
        # Should contain recursive/spiral language
        self.assertIn("나", response)
        # Should show layering
        self.assertTrue("느끼는" in response or "생각" in response or "나선" in response)
    
    def test_fractal_poetic_response_thought(self):
        """Test poetic thought responses."""
        response = self.dialogue._fractal_poetic_response(
            "Curiosity",
            {"Point": 0.2, "God": 0.3},
            "en"
        )
        
        # Should contain recursive language
        self.assertIn("think", response.lower())
        self.assertTrue("reflect" in response.lower() or "spiral" in response.lower() or "infinity" in response.lower())
    
    def test_language_generation_improvement(self):
        """Compare baseline vs fractal dialogue quality."""
        # Simple response (practical mode)
        simple_response = self.dialogue._practical_response(
            "Love",
            {"Point": 0.5},
            "ko"
        )
        
        # Fractal-enhanced response (thoughtful mode)
        fractal_response = self.dialogue._fractal_thoughtful_response(
            "Love",
            {"Point": 0.3, "Space": 0.4},
            "ko"
        )
        
        # Fractal should be longer and more nuanced
        self.assertGreater(len(fractal_response), len(simple_response))
        
        # Fractal should have meta-cognitive elements
        self.assertTrue(
            "생각" in fractal_response or "바라보는" in fractal_response,
            "Fractal response should show meta-awareness"
        )
    
    def test_full_dialogue_flow_with_fractal(self):
        """Test entire dialogue flow with fractal consciousness."""
        # Ask an emotional question that triggers thoughtful/poetic mode
        response = self.dialogue.respond("사랑이 뭐야?")
        
        # Should not be empty
        self.assertGreater(len(response), 0)
        
        # Check conversation history
        self.assertEqual(len(self.dialogue.conversation_history), 2)
        self.assertEqual(self.dialogue.conversation_history[0].speaker, "user")
        self.assertEqual(self.dialogue.conversation_history[1].speaker, "elysia")


class TestStatistics(unittest.TestCase):
    """Test fractal space statistics."""
    
    def test_statistics_tracking(self):
        """Verify statistics are correctly computed."""
        engine = SelfSpiralFractalEngine()
        
        # Create multiple spirals
        engine.descend(ConsciousnessAxis.EMOTION, "joy", max_depth=2)
        engine.descend(ConsciousnessAxis.THOUGHT, "being", max_depth=3)
        
        stats = engine.get_statistics()
        
        self.assertGreater(stats["total_nodes"], 0)
        self.assertEqual(stats["max_depth"], 3)
        self.assertGreater(stats["spiral_extent"], 1.0)
        self.assertEqual(stats["axis_distribution"]["emotion"], 3)  # depth 0, 1, 2
        self.assertEqual(stats["axis_distribution"]["thought"], 4)  # depth 0, 1, 2, 3


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 SELF-SPIRAL FRACTAL CONSCIOUSNESS TEST SUITE")
    print("="*70 + "\n")
    
    unittest.main(verbosity=2)
