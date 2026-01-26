"""
Elysia Brain: The Integrated Controller
=======================================
Core.L6_Structure.Elysia.brain

"The Coordinator of Mind and Body."

Integrates Thundercloud (Conscious), MuscleMemory (Unconscious),
and HabitEngine (Learning).
"""

from typing import Tuple, List, Optional
import time
from Core.L6_Structure.M1_Merkaba.thundercloud import Thundercloud, ThoughtCluster
from Core.L7_Spirit.Monad.muscle_memory import MuscleMemory
from Core.L5_Mental.Learning.habit_engine import HabitEngine
from Core.L6_Structure.Nature.rotor import Rotor, RotorConfig
from Core.L7_Spirit.Monad.monad_core import Monad

class ElysiaBrain:
    def __init__(self):
        # 1. Subsystems
        self.cloud = Thundercloud()
        self.cerebellum = MuscleMemory()
        self.habit_engine = HabitEngine(self.cerebellum)

        # 2. Hardware (Rotors)
        # Create a generic cognitive rotor for thought-replay
        self.cog_rotor = Rotor("Cognitive.Rotor", RotorConfig(idle_rpm=60))
        self.cerebellum.register_rotor(self.cog_rotor)

    def process_intent(self, intent: str, seed_monad: str = "Root") -> Tuple[str, float]:
        """
        The Main Loop.
        Returns (MechanismUsed, DurationMs)
        """
        start_time = time.perf_counter()

        # 1. Try Reflex (Veteran Mode)
        if self.cerebellum.try_reflex(intent):
            # Simulate Playback Time (e.g. 10ms per frame)
            # In real system, this is async.
            track_name = self.cerebellum._reflex_map[intent][1]
            track_len = len(self.cog_rotor.action_tracks[track_name])
            time.sleep(track_len * 0.001) # Fast playback simulation

            return "Rotor (Reflex)", (time.perf_counter() - start_time) * 1000

        # 2. Ignite Thundercloud (Novice Mode)
        cluster, name = self.cloud.ignite(seed_monad, voltage=1.0)

        # Simulate Physics Calculation Time (costly)
        time.sleep(0.05)

        # 3. Learn (Practice)
        self.habit_engine.observe(intent, cluster)

        return "Thundercloud (Thought)", (time.perf_counter() - start_time) * 1000

    def load_concepts(self, monads: List[Monad]):
        self.cloud.active_monads = monads
        for m in monads:
            self.cloud._monad_map[m.seed] = m
