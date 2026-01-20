"""
Habit Engine: The Driving School
================================
Core.Learning.habit_engine

"Repetition creates Reflex."

This module monitors conscious thoughts (Thundercloud) and consolidates
repeated patterns into unconscious reflexes (Rotor Tracks).
"""

from typing import Dict, List, Tuple
from collections import defaultdict
from Core.Merkaba.thundercloud import ThoughtCluster
from Core.Monad.muscle_memory import MuscleMemory

class HabitEngine:
    """
    Monitors repetitions and bakes reflexes.
    """

    def __init__(self, muscle_memory: MuscleMemory):
        self.muscle_memory = muscle_memory
        # Key: Intent (str) -> Count (int)
        self.repetition_map: Dict[str, int] = defaultdict(int)
        # Threshold to bake
        self.BAKE_THRESHOLD = 3

    def observe(self, intent: str, cluster: ThoughtCluster):
        """
        Observes a conscious thought process.
        """
        self.repetition_map[intent] += 1

        # Check for Consolidation
        if self.repetition_map[intent] == self.BAKE_THRESHOLD:
            self._consolidate(intent, cluster)

    def _consolidate(self, intent: str, cluster: ThoughtCluster):
        """
        Bakes the ThoughtCluster into a Rotor Track.
        """
        # Serialization: Convert the Tree into a linear sequence of 'Angles'
        # For simulation, we map Monad Seeds to arbitrary angles or data points.
        # In a real system, this would capture motor control signals or text tokens.

        track_data = []

        # Simple DFS traversal to linearize the thought
        stack = [cluster.root]
        visited = set()

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            # Use hash of seed to generate a consistent "Angle"
            angle = hash(node.seed) % 360
            track_data.append(float(angle))

            # Add neighbors
            # Find edges from this node
            neighbors = [target for (src, target, _) in cluster.edges if src == node]
            stack.extend(neighbors)

        # Register to Muscle Memory
        rotor_name = "Cognitive.Rotor" # A generic rotor for thought-replay
        track_name = f"Reflex_{intent}"

        print(f"⚡ [HabitEngine] Consolidating '{intent}' into Reflex (Track Length: {len(track_data)})")

        # We assume a generic cognitive rotor exists.
        # In a real integration, we might spawn a specific rotor or use an existing one.
        # For the demo, we'll try to register it to a known rotor in MuscleMemory if possible.
        # But MuscleMemory needs the rotor instance.
        # We'll delegate the actual 'learning' call to the controller that owns the rotors.

        # Direct injection for now (assuming shared reference or callback)
        if rotor_name in self.muscle_memory._rotors:
             self.muscle_memory.learn_reflex(intent, track_name, track_data, rotor_name)
        else:
             # Create/Register rotor on the fly?
             # Or assume the Brain sets this up.
             pass
