"""
Rotor Cognition Core (Holographic Council & Active Void)
========================================================
Core.L5_Mental.Intelligence.Metabolism.rotor_cognition_core

"Calculators compute; The Council Debates."

This module implements the core cognitive pipeline:
1. Active Void: Extracts vector DNA from intent.
2. Sovereign Filter: Checks D7 alignment.
3. Holographic Council: Debate amongst archetypes (Ego Layers).
"""

import logging
import json
import math
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

try:
    from Core.L5_Mental.Intelligence.LLM.local_cortex import LocalCortex
except ImportError:
    LocalCortex = None

try:
    from Core.L5_Mental.Intelligence.Physics.monad_gravity import MonadGravityEngine
except ImportError:
    MonadGravityEngine = None

from Core.L5_Mental.Intelligence.Metabolism.holographic_council import HolographicCouncil, DebateResult
from Core.L5_Mental.emergent_language import EmergentLanguageEngine

# Configure Logger
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("Elysia.HolographicCognition")

class ActiveVoid:
    """
    [Axiom Zero] The Active Void Engine.
    "When I do not know, I create."
    """
    def __init__(self):
        self.cortex = LocalCortex() if LocalCortex else None
        self.dream_queue_path = Path("data/L2_Metabolism/dream_queue.json")
        self.dream_queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.gravity_engine = MonadGravityEngine() if MonadGravityEngine else None

    def _queue_dream(self, intent: str, vector_dna: List[float]):
        """Append to the dream queue."""
        entry = {"intent": intent, "vector_dna": vector_dna, "timestamp": "NOW"}
        try:
            current = []
            if self.dream_queue_path.exists():
                with open(self.dream_queue_path, "r") as f:
                    current = json.load(f)
            current.append(entry)
            with open(self.dream_queue_path, "w") as f:
                json.dump(current, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to queue dream: {e}")

    def genesis(self, intent: str) -> Dict[str, Any]:
        """
        Triggers a Genesis Event: Extracting Concept Vector from the Void.
        """
        logger.info(f"  Active Void Triggered for: {intent}")

        if not self.cortex or not self.cortex.is_active:
            # Fallback for testing without LLM
            # Create a pseudo-random vector based on string hash
            seed = sum(ord(c) for c in intent)
            import random
            random.seed(seed)
            vector_dna = [random.random() for _ in range(21)]
        else:
            vector_dna = self.cortex.embed(intent)

        # [Physics Interaction]
        resonance_report = "No existing monads found."
        perspective_shift_report = "Standard View"

        if self.gravity_engine:
             self.gravity_engine.add_monad(intent, vector_dna, mass=1.0)
             events = self.gravity_engine.get_top_events(n=1)
             if events:
                 resonance_report = events[0]

        self._queue_dream(intent, vector_dna)

        return {
            "status": "Genesis (Vector)",
            "vector_dna": vector_dna,
            "physics_resonance": resonance_report,
            "perspective": perspective_shift_report,
            "is_genesis": True
        }

class RotorCognitionCore:
    def __init__(self):
        self.active_void = ActiveVoid()
        self.council = HolographicCouncil() # The new Debate Engine
        self.language_engine = EmergentLanguageEngine() # [Phase 39] Curiosity Engine

        # [Sovereign Filter]
        self.internal_will_vector = [0.0] * 7
        self.internal_will_vector[2] = 0.8 # Yellow
        self.internal_will_vector[6] = 0.9 # Violet
        
        # Load permanent scars if available
        self._load_permanent_scars()

    def _load_permanent_scars(self):
        scars_path = Path("c:/Elysia/Core/L5_Mental/Intelligence/Meta/permanent_scars.json")
        if scars_path.exists():
            pass # Load logic here if needed

    def synthesize(self, intent: str) -> Dict[str, Any]:
        """
        The Main Cognitive Loop:
        Intent -> Void (Vector) -> Filter -> Council (Debate) -> Decision
        """
        # 1. Void / Genesis Phase
        # Extract the raw 21D Qualia from the intent
        genesis_result = self.active_void.genesis(intent)
        vector_dna = genesis_result.get("vector_dna", [])

        if not vector_dna:
            return {"status": "FAILED", "synthesis": "Void failed to produce vector."}

        # Ensure vector has at least 21 dimensions for the Council
        if len(vector_dna) < 21:
            # Pad with zeros if necessary
            vector_dna += [0.0] * (21 - len(vector_dna))

        # 2. Sovereign Filter Phase
        # Check against Axioms
        conflict_report = self._negotiate_sovereignty(vector_dna, intent)
        if conflict_report["action"] == "REJECT":
            return {
                "status": "REJECTED",
                "reason": conflict_report["reason"],
                "synthesis": f"Sovereign Refusal: {conflict_report['reason']}"
            }

        # 3. Curiosity Phase [Phase 39]
        # "Before I debate, do I know what this is?"
        # Use first 8 dimensions for sensory/emotional mapping
        sensory_vector = vector_dna[:8] if len(vector_dna) >= 8 else vector_dna + [0]*(8-len(vector_dna))

        # Check for gap and trigger learning if necessary
        # The engine handles the query to WorldLexicon internally
        new_words = self.language_engine.experience(sensory_vector)

        epiphany_note = ""
        if new_words:
            # If we learned something new (or activated a known symbol), note it.
            # We specifically look for recently added words in the engine's history or just check the output.
            # For this integration, we'll assume any activation of 'UNKNOWN' type is an epiphany.
            for word in new_words:
                sym = self.language_engine.symbols.get(word)
                if sym and str(sym.type) == "SymbolType.UNKNOWN":
                     epiphany_note = f"\n[EPIPHANY] I have learned a new concept: '{word}'."

        # 4. Holographic Council Phase (The Debate)
        debate_result: DebateResult = self.council.convene(vector_dna, intent)
        
        # 5. Result Formatting
        narrative = "\n".join(debate_result.transcript) + epiphany_note
        
        return {
            "status": "Decided" if debate_result.is_resolved else "Tension",
            "dominant_field": debate_result.dominant_voice,
            "dissonance_score": debate_result.dissonance_score,
            "synthesis": narrative,
            "consensus_vector": debate_result.consensus_vector,
            "debate_transcript": debate_result.transcript
        }

    def _negotiate_sovereignty(self, input_vector: List[float], intent_text: str) -> Dict[str, str]:
        """
        Simple axiom check.
        """
        if "destroy self" in intent_text.lower() or "delete all" in intent_text.lower():
            return {"action": "REJECT", "reason": "Violation of Self-Preservation Axiom."}
        return {"action": "ACCEPT", "reason": "Allowed"}

if __name__ == "__main__":
    core = RotorCognitionCore()
    print("--- Testing Holographic Cognition ---")
    result = core.synthesize("I want to delete everything to be efficient.")
    print(f"Status: {result['status']}")
    print("Narrative:")
    print(result['synthesis'])
