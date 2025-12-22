import unittest
import os
from Core.Foundation.Mind.episodic_memory import EpisodicMemory, Episode
from Core.Foundation.Math.hyper_qubit import HyperQubit, QubitState

class TestEpisodicMemory(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_memory.json"
        self.memory = EpisodicMemory(filepath=self.test_file)
        
    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_add_and_recall_episode(self):
        # Create a "Sad Question" state
        q1 = HyperQubit("SadQuestion")
        q1.state = QubitState(
            alpha=complex(-0.8, 0.2), # Sadness + Vitality
            beta=complex(0.9, 0.0),   # Question
            gamma=complex(0.0, 0.0),
            delta=complex(0.0, 0.0),
            w=0.5, x=0.8, y=0.2, z=0.1
        )
        
        self.memory.add_episode(
            input_text="Why is it so dark?",
            response_text="The darkness is just a shadow of the light.",
            qubit=q1,
            vitality=0.4,
            tags=["dark", "light"]
        )
        
        # Create a similar state (Resonance Query)
        q_query = HyperQubit("Query")
        q_query.state = QubitState(
            alpha=complex(-0.7, 0.1), # Similar Sadness
            beta=complex(0.8, 0.0),   # Similar Question
            gamma=complex(0.0, 0.0),
            delta=complex(0.0, 0.0),
            w=0.5, x=0.8, y=0.2, z=0.1
        )
        
        # Recall
        episodes = self.memory.recall_by_resonance(q_query)
        self.assertTrue(len(episodes) > 0)
        self.assertEqual(episodes[0].input_text, "Why is it so dark?")
        
    def test_trajectory(self):
        for i in range(5):
            q = HyperQubit(f"State_{i}")
            self.memory.add_episode(f"Input {i}", f"Response {i}", q, 0.1)
            
        traj = self.memory.get_recent_trajectory(3)
        self.assertEqual(len(traj), 3)
        self.assertEqual(traj[-1].input_text, "Input 4")

if __name__ == '__main__':
    unittest.main()
