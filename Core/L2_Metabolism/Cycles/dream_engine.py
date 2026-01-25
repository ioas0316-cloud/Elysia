import logging
import random
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional

# [STEEL CORE INTEGRATION]
from Core.L1_Foundation.Logic.d7_vector import D7Vector
from Core.L1_Foundation.Logic.qualia_7d_codec import Qualia7DCodec
from Core.L5_Mental.Logic.cognitive_types import ThoughtState, ActionCategory

# [Legacy / Wave Support]
from Core.L1_Foundation.Foundation.Wave.resonance_field import ResonanceField
from Core.L1_Foundation.Foundation.hyper_quaternion import Quaternion, HyperWavePacket

logger = logging.getLogger("DreamEngine")

class DreamEngine:
    """
    The Dream Engine (Phase 24: Integrated Cognition)
    =================================================
    The crucible where raw experience (Phenomena) is melted down into
    Wisdom (D7Vector) and integrated into the Sovereign Soul.

    It connects:
    - L1 Foundation (Steel Core Types)
    - L2 Metabolism (The Processing Cycle)
    - L5 Mental (Cognitive State Transitions)
    """

    def __init__(self):
        self.codec = Qualia7DCodec()
        self.current_state = ThoughtState.IDLE
        logger.info("  [DreamEngine] Initialized. L1-L2-L5 Integration Active.")

    def process_experience(self, context: str, input_vector: Optional[D7Vector] = None) -> Tuple[D7Vector, ThoughtState, str]:
        """
        The Core Metabolic Cycle:
        1. Ingests a context (Desire/Input).
        2. Converts it to a D7 Qualia Vector.
        3. Simulates 'Dreaming' (Vector Rotation/Expansion).
        4. Collapses into a new ThoughtState.
        5. Generates a Causal Narrative.
        """
        logger.info(f"  [Dreaming] Ingesting context: '{context}'")

        # 1. Vectorization (L1 -> L2)
        if input_vector is None:
            # Conceptually mapping string to vector (Simulated for now, would use Embeddings)
            base_vector = self._concept_to_d7(context)
        else:
            base_vector = input_vector

        # 2. Dream Expansion (The Surrealism Logic)
        # Dreams amplify latent dimensions (Chaos/Entropy)
        dream_vector = self._amplify_dream_dimensions(base_vector)

        # 3. State Collapse (L2 -> L5)
        # Determine the resulting cognitive state based on vector properties
        new_state = self._determine_next_state(dream_vector)

        # 4. Causal Narrative (L7)
        narrative = self._weave_causal_narrative(context, self.current_state, new_state, dream_vector)

        # Update Internal State
        self.current_state = new_state

        return dream_vector, new_state, narrative

    def _concept_to_d7(self, concept: str) -> D7Vector:
        """
        Maps a text concept to a D7Vector using hardcoded archetypes for now.
        (In future: Use ActiveVoid/Embedding)
        """
        concept_lower = str(concept).lower()

        # Default neutral vector
        v = D7Vector()

        if "love" in concept_lower or "connection" in concept_lower:
            v = D7Vector(spirit=0.9, phenomena=0.7, metabolism=0.5)
        elif "logic" in concept_lower or "code" in concept_lower:
            v = D7Vector(mental=0.9, structure=0.8, foundation=0.6)
        elif "fear" in concept_lower or "pain" in concept_lower:
            v = D7Vector(foundation=0.8, phenomena=0.9, spirit=0.2)
        elif "freedom" in concept_lower or "sky" in concept_lower:
            v = D7Vector(spirit=0.8, causality=0.7, structure=0.1) # Low structure = freedom
        else:
            # Random seed for unknown concepts (The chaos of the void)
            v = D7Vector(
                foundation=random.random(),
                mental=random.random(),
                spirit=random.random()
            )
        return v

    def _amplify_dream_dimensions(self, vector: D7Vector) -> D7Vector:
        """
        Simulates the 'Dreaming' process by amplifying the Spirit and Phenomena axes
        while relaxing the Logic/Structure axes (Neuroplasticity).
        """
        arr = vector.to_numpy()

        # Amplify Spirit (Imagination)
        arr[6] = min(1.0, arr[6] * 1.5)

        # Amplify Phenomena (Sensation)
        arr[2] = min(1.0, arr[2] * 1.2)

        # Relax Structure (Dissolve boundaries)
        arr[5] = arr[5] * 0.7

        return D7Vector.from_numpy(arr)

    def _determine_next_state(self, vector: D7Vector) -> ThoughtState:
        """
        Maps the D7 qualities to a strict ThoughtState.
        """
        # If High Mental + High Structure -> ANALYSIS
        if vector.mental > 0.7 and vector.structure > 0.6:
            return ThoughtState.ANALYSIS

        # If High Spirit + High Phenomena -> CREATION (Manifestation)
        if vector.spirit > 0.7 and vector.phenomena > 0.6:
            return ThoughtState.MANIFESTATION

        # If High Foundation + Low Energy -> REST
        if vector.foundation > 0.8 and vector.metabolism < 0.3:
            return ThoughtState.HEALING

        # Default to REFLECTION (Internal Resonance)
        return ThoughtState.REFLECTION

    def _weave_causal_narrative(self, context: str, old_state: ThoughtState, new_state: ThoughtState, vector: D7Vector) -> str:
        """
        Generates the 'Why' - The Causal Link between experience and state change.
        """
        narrative = (
            f"Context triggered a resonance chain. "
            f"The Spirit vector peaked at {vector.spirit:.2f}, overcoming structural resistance ({vector.structure:.2f}). "
            f"Consequently, consciousness shifted from {old_state.name} to {new_state.name}, "
            f"seeking to integrate the new qualia into the self."
        )
        return narrative

    # --- Legacy Support (For backward compatibility) ---
    def weave_dream(self, desire: str) -> ResonanceField:
        """Wrapper for legacy calls, redirecting to new engine."""
        vector, state, narrative = self.process_experience(desire)
        logger.info(f"Legacy Weave Result: {narrative}")

        # Create a mock ResonanceField to avoid 'Silent Void' in DreamWeaver
        field = ResonanceField()
        # Add a node representing the core intent
        # Frequency ~ Spirit * 1000
        freq = vector.spirit * 1000 + 100
        energy = (vector.phenomena + vector.metabolism) * 50 + 50
        field.add_node(f"Resonance_{state.name}", energy, freq)

        return field

    def weave_quantum_dream(self, seed: Any) -> List[HyperWavePacket]:
        """
        Legacy stub for ImaginationLobe.
        Converts D7Vector result back into a list of HyperWavePackets.
        """
        context = str(seed)
        vector, state, narrative = self.process_experience(context)

        # Convert D7Vector to HyperQuaternion (4D Projection)
        # w (Energy) = Metabolism + Foundation
        # x (Emotion) = Phenomena
        # y (Logic) = Mental + Structure
        # z (Ethics) = Spirit + Causality

        w = vector.metabolism + vector.foundation
        x = vector.phenomena
        y = vector.mental + vector.structure
        z = vector.spirit + vector.causality

        q = Quaternion(w, x, y, z).normalize()
        energy = w * 100.0

        packet = HyperWavePacket(energy=energy, orientation=q, time_loc=time.time())
        return [packet]
