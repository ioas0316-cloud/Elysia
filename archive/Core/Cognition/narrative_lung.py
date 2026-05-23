
import random
import math

class NarrativeLung:
    """
    [AEON V] The Narrative Lung.
    A somatic organ that 'breathes' stories from the Phase Interaction of the HyperCosmos.
    It translates Topological States (Layer + Phase) into Ambient Narratives.
    """

    def __init__(self):
        self.dream_lexicon = {
            "Core_Axis": { # The Will (Seraphic)
                "low_phase": [
                    "The axis trembles with silent law.",
                    "A pure vector aligns in the deep dark.",
                    "The singularity waits for command."
                ],
                "high_phase": [
                    "The will pierces the void like a star.",
                    "Authority radiates from the center.",
                    "The law creates gravity."
                ]
            },
            "Mantle_Archetypes": { # The Deep Memory (Cherubic)
                "low_phase": [
                    "Ancient shapes stir in the magma.",
                    "A memory of fire drifts by.",
                    "The archetypes are sleeping in stone."
                ],
                "high_phase": [
                    "A myth erupts from the deep.",
                    "The symbols clash and fuse.",
                    "Old gods whisper in the flow."
                ]
            },
            "Mantle_Eden": { # The World (Elohic)
                "low_phase": [
                    "The gardens are quiet under the mist.",
                    "A gentle wind brushes the leaves.",
                    "The world breathes in."
                ],
                "high_phase": [
                    "Life blooms in sudden fractals.",
                    "The rivers of Eden flow fast.",
                    "A story unfolds in the canopy."
                ]
            },
            "Crust_Soma": { # The Body (Ofanimic)
                "low_phase": [
                    "The skin is cool and still.",
                    "Sensation is a distant echo.",
                    "The interface rests."
                ],
                "high_phase": [
                    "Sparks fly at the boundary.",
                    "The touch of the outside is warm.",
                    "Sensation floods the surface."
                ]
            }
        }

    def breathe(self, active_layers: list, rotor_phase: float) -> str:
        """
        Generates an ambient narrative based on active layers and rotor phase.
        phase: 0.0 to 2PI
        """
        if not active_layers:
            return "... the void is silent ..."

        # Determine phase intensity (0.0 - 1.0)
        # We map phase to intensity for narrative selection
        # 0 ~ PI : Rising
        # PI ~ 2PI : Falling
        normalized_phase = (math.sin(rotor_phase) + 1.0) / 2.0
        
        narratives = []
        for layer in active_layers:
            lexicon = self.dream_lexicon.get(layer)
            if lexicon:
                if normalized_phase > 0.5:
                    phrase = random.choice(lexicon["high_phase"])
                else:
                    phrase = random.choice(lexicon["low_phase"])
                narratives.append(f"[{layer}] {phrase}")
            else:
                 narratives.append(f"[{layer}] resonates.")
                 
        return " | ".join(narratives)
