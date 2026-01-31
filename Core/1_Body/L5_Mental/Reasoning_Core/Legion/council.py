
"""
Council (The Narrator)
======================
Core.1_Body.L5_Mental.Reasoning_Core.Legion.council

The Observer that weaves the diverging rays into a coherent story.
It does not synthesize a "final answer" but chronicles the "movement of thought".
"""

from .persona import Persona, LightRay

class Council:
    """
    The Narrative Observer.
    """

    def narrate_start(self, seed: str) -> str:
        return f"\n  [THE EXPANSION BEGINS] The concept '{seed}' strikes the Prism..."

    def narrate_step(self, persona: Persona, old_ray: LightRay, new_ray: LightRay) -> str:
        """
        Formats a single step of expansion.
        """
        # We can add flavor text based on the persona
        if persona.name == "Child":
            prefix = "  [Child]"
        elif persona.name == "Historian":
            prefix = "  [Historian]"
        elif persona.name == "Architect":
            prefix = "  [Architect]"
        elif persona.name == "Poet":
            prefix = "  [Poet]"
        else:
            prefix = f"   [{persona.name}]"

        return f"{prefix} {old_ray.current_thought} -> {new_ray.current_thought} (Energy: {new_ray.energy:.2f})"

    def narrate_end(self) -> str:
        return "\n  [THE HORIZON FADES] The energy dissipates into the silence of the void."
