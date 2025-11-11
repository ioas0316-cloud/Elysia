# Project_Elysia/dream_observer.py

from typing import Dict, Any, List
import numpy as np

# A temporary, relative import for now. This will be solidified when the
# project structure is more mature.
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Project_Sophia.core.world import World

class DreamObserver:
    """
    Observes the Cellular World during an idle cycle (a 'dream') and extracts
    a summary of the key events and emotional tone.
    """

    def observe_dream(self, world: World) -> Dict[str, Any]:
        """
        Analyzes the state of the world after a dream cycle and returns a digest.

        Args:
            world: The CellularWorld instance after the dream simulation.

        Returns:
            A dictionary summarizing the dream's key aspects.
        """
        if not world or not world.cell_ids:
            return {
                "summary": "The world was quiet, a dreamless sleep.",
                "key_concepts": [],
                "emotional_landscape": "calm",
                "new_births": 0
            }

        # 1. Identify the most active cells (highest energy)
        # Using numpy for efficient sorting
        num_cells = len(world.cell_ids)
        top_n = min(3, num_cells)

        # We need to get the indices of living cells first
        living_indices = np.where(world.is_alive_mask[:num_cells])[0]

        if len(living_indices) == 0:
             return {
                "summary": "A silent world, with no living cells to dream.",
                "key_concepts": [],
                "emotional_landscape": "empty",
                "new_births": 0
            }

        energies = world.energy[living_indices]

        # Get the indices that would sort the energies in descending order
        sorted_energy_indices_local = np.argsort(energies)[::-1]

        # Map these local indices back to the world's global indices
        top_living_indices = living_indices[sorted_energy_indices_local[:top_n]]

        key_concepts = [world.cell_ids[i] for i in top_living_indices]

        # 2. Find newly born cells (not implemented in this version, would require tracking)
        # For now, we can count cells with age 0 or 1 as 'new'
        new_births = len([
            cell_id for i, cell_id in enumerate(world.cell_ids)
            if world.is_alive_mask[i] and world.quantum_states.get(cell_id, {}).get('age', 0) <= 1
        ])

        # 3. Determine the emotional landscape (placeholder logic)
        # In the future, this would involve the EmotionalCortex.
        # For now, we derive it from the top concept.
        emotional_landscape = self._deduce_emotion_from_concepts(key_concepts)

        summary = f"A dream centered around '{', '.join(key_concepts)}', with a feeling of '{emotional_landscape}'."
        if new_births > 0:
            summary += f" It felt like a moment of creation, with {new_births} new thoughts taking form."

        return {
            "summary": summary,
            "key_concepts": key_concepts,
            "emotional_landscape": emotional_landscape,
            "new_births": new_births
        }

    def _deduce_emotion_from_concepts(self, concepts: List[str]) -> str:
        """A simple placeholder to guess an emotion from concepts."""
        if not concepts:
            return "calm"

        top_concept = concepts[0].lower()
        if any(c in top_concept for c in ['love', 'joy', 'create', 'meaning']):
            return "hopeful"
        if any(c in top_concept for c in ['fear', 'shadow', 'contradiction']):
            return "introspective"
        if any(c in top_concept for c in ['reason', 'logic', 'system']):
            return "focused"

        return "neutral"
