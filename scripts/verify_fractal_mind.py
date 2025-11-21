
import unittest
from datetime import datetime
from Project_Sophia.core.tensor_wave import Tensor3D, FrequencyWave
from Project_Mirror.sensory_cortex import SensoryQuality
from Project_Sophia.emotional_engine import EmotionalState, EmotionalEngine
from Project_Sophia.core.thought import Thought
from Project_Elysia.core_memory import CoreMemory, Experience, IdentityFragment

class TestFractalMind(unittest.TestCase):

    def setUp(self):
        # Initialize components
        self.memory = CoreMemory(file_path=None) # In-memory
        self.emotional_engine = EmotionalEngine()

    def test_fractal_physics_mass(self):
        """Verify that Tensor mass calculation respects recursive depth."""
        t1 = Tensor3D(1.0, 0.0, 0.0, mass_offset=0.0)
        t2 = Tensor3D(1.0, 0.0, 0.0, mass_offset=5.0) # More complex

        self.assertGreater(t2.calculate_mass(), t1.calculate_mass())

    def test_meta_sensation_quality(self):
        """Verify that sensory quality analysis works."""
        q = SensoryQuality(clarity=0.9, intensity=0.8, novelty=0.5, dissonance=0.1)
        self.assertEqual(q.clarity, 0.9)

    def test_recursive_meta_emotion(self):
        """Verify that emotions can contain meta-emotions and influence gravity."""
        # 1. Create a massive grief state
        grief_tensor = Tensor3D(0.1, 0.9, 0.2, mass_offset=10.0)
        grief = EmotionalState(valence=-0.8, arousal=0.4, dominance=-0.5, primary_emotion="grief", tensor=grief_tensor)
        self.emotional_engine.current_state = grief

        # 2. Create a small annoyance event
        annoyance_tensor = Tensor3D(0.2, 0.3, 0.1)
        annoyance = EmotionalState(valence=-0.2, arousal=0.5, dominance=0.1, primary_emotion="annoyance", tensor=annoyance_tensor)

        # 3. Process event - Should be captured as secondary/meta because grief is massive
        new_state = self.emotional_engine.process_event(annoyance, intensity=0.3)

        self.assertEqual(new_state.primary_emotion, "grief")
        self.assertIn("annoyance", new_state.secondary_emotions)

    def test_fractal_thought_structure(self):
        """Verify thought recursion and mass calculation."""
        # Sub-thought
        sub_t = Thought(content="Sub concept", source="test", confidence=0.5, energy=1.0)

        # Main thought
        main_t = Thought(content="Main concept", source="test", confidence=0.9, energy=2.0)
        main_t.sub_thoughts.append(sub_t)

        # Mass check
        mass = main_t.calculate_gravitational_mass()
        # Expected: 2.0 + (0.9*2) + (sub_thought_mass * 0.5)
        # sub_mass = 1.0 + (0.5*2) = 2.0
        # total = 3.8 + 1.0 = 4.8
        self.assertTrue(mass > 3.8)

    def test_memory_persistence_of_fractals(self):
        """Verify that complex fractal objects survive memory serialization."""
        # Create complex emotion
        meta = EmotionalState(valence=0.1, arousal=0.1, dominance=0.1, primary_emotion="meta_calm")
        primary = EmotionalState(valence=0.5, arousal=0.5, dominance=0.5, primary_emotion="joy", meta_emotions=[meta])

        # Add to memory
        exp = Experience(
            timestamp=datetime.now().isoformat(),
            content="Fractal Memory Test",
            emotional_state=primary,
            tensor=Tensor3D(1,1,1)
        )
        self.memory.add_experience(exp)

        # Retrieve and check
        saved_exp = self.memory.get_experiences(1)[0]
        saved_emotion = saved_exp.emotional_state

        self.assertEqual(saved_emotion.primary_emotion, "joy")
        self.assertTrue(len(saved_emotion.meta_emotions) > 0)
        self.assertEqual(saved_emotion.meta_emotions[0].primary_emotion, "meta_calm")

if __name__ == '__main__':
    unittest.main()
