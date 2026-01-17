
"""
Legion Manager (Propagation Engine)
===================================
Core.Intelligence.Legion.legion

Manages the swarm of Personas and the propagation of LightRays.
It implements the "Infinite Ray" logic where thoughts expand until energy dissipates.
"""

from typing import List, Generator
from .persona import Persona, LightRay
from .council import Council

class Legion:
    """
    The Swarm Manager.
    It holds the Personas and orchestrates the beam splitting.
    """

    def __init__(self):
        self.council = Council()
        self.personas = [
            Persona("Child", "Phenomenologist", {"SENSATION": 0.9}),
            Persona("Historian", "Memory", {"TIME": 0.9}),
            Persona("Architect", "Logic", {"STRUCTURE": 0.9}),
            Persona("Poet", "Emotion", {"EMOTION": 0.9})
        ]

    def propagate(self, seed_concept: str, initial_energy: float = 1.0) -> Generator[str, None, None]:
        """
        The Main Generator Loop.
        Instead of returning a value, it yields narrative chunks as the thought expands.

        Args:
            seed_concept: The starting thought.
            initial_energy: The starting momentum.
        """

        # 1. Initialize the First Ray
        initial_ray = LightRay(
            origin=seed_concept,
            current_thought=seed_concept,
            energy=initial_energy,
            trajectory=[seed_concept],
            narrative=f"The seed '{seed_concept}' enters the Prism."
        )

        # Queue of active rays (BFS Expansion)
        # In a real async system, these would be parallel tasks.
        active_rays = [initial_ray]

        # We use a set to prevent infinite loops on the exact same thought loop immediately,
        # but we allow re-visiting if the path is different (Concept of "Return").
        # For simplicity in this demo, we just track (thought, depth).
        visited = set()

        yield self.council.narrate_start(seed_concept)

        while active_rays:
            current_ray = active_rays.pop(0)

            # Stop if energy is too low (The Event Horizon)
            if current_ray.energy < 0.1:
                continue

            # 2. Beam Splitting: Pass the ray through all Personas
            # Each Persona refracts the ray differently.
            for persona in self.personas:

                # Refract
                new_rays = persona.refract(current_ray)

                for new_ray in new_rays:
                    # Check uniqueness to avoid trivial loops like Apple->Red->Apple->Red
                    # We allow loops if the narrative depth is significant.
                    visit_key = (new_ray.current_thought, len(new_ray.trajectory))
                    if visit_key in visited:
                        continue
                    visited.add(visit_key)

                    # 3. Yield the Narrative
                    # The Council observes this specific expansion.
                    narration = self.council.narrate_step(persona, current_ray, new_ray)
                    yield narration

                    # 4. Enqueue for further expansion
                    active_rays.append(new_ray)

        yield self.council.narrate_end()
