"""
Dream Protocol: The Cycle of Narrative Consolidation
====================================================

"We dream to become real."

This module runs during system idle time (Night) to process the "Active Void" queue.
It consolidates temporary hypotheses into permanent structural wisdom.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

try:
    from Core.L5_Mental.Intelligence.LLM.local_cortex import LocalCortex
except ImportError:
    LocalCortex = None # Graceful fallback

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Elysia.DreamProtocol")

class DreamAlchemist:
    def __init__(self):
        self.queue_path = Path("data/L2_Metabolism/dream_queue.json")
        self.wisdom_path = Path("data/L5_Mental/crystallized_wisdom.json")
        self.wisdom_path.parent.mkdir(parents=True, exist_ok=True)
        self.cortex = LocalCortex() if LocalCortex else None

    def sleep(self):
        """
        Enters the Dream State.
        Processes the queue and clears it.
        """
        if not self.queue_path.exists():
            logger.info("💤 No dreams to dream. Sleep is deep and empty.")
            return

        try:
            with open(self.queue_path, "r", encoding="utf-8") as f:
                dreams = json.load(f)
        except Exception as e:
            logger.error(f"Nightmare (Read Error): {e}")
            return

        if not dreams:
            logger.info("💤 Dream queue is empty.")
            return

        logger.info(f"🌙 Entering REM Cycle. Processing {len(dreams)} dreams...")
        crystallized = []

        for dream in dreams:
            intent = dream.get("intent")
            vector_dna = dream.get("vector_dna")

            # [Alchemy: Fractal Trinity Reconstruction]
            # Causality is not linear (A->B). It is Origin -> Process -> Result.
            # The Process contains the Infinite Hidden Monad (Quantum Collapse).
            if self.cortex and self.cortex.is_active:
                prompt = f"""
                [FRACTAL CAUSALITY PROTOCOL]
                Analyze the concept: "{intent}" through the Trinity of Existence.

                Do not just define it. Observe its Structure through the Law of Monads:

                1. [ORIGIN] (Won-in): The Seed. What is the fundamental Providence/Intent that birthed this?
                2. [PROCESS] (Gwa-jeong): The Monad. What is the "Fractal Quantum Collapse Algorithm" that bridges the seed to reality? (The Hidden Law)
                3. [RESULT] (Gyeol-gwa): The Structure. What is the manifest form?

                Format:
                ORIGIN: ...
                PROCESS: ...
                RESULT: ...
                """
                causal_map = self.cortex.think(prompt, context="Dream Fractal Observation")
            else:
                causal_map = "Fractal observation unavailable (Cortex inactive)."

            # [Crystallization]
            logger.info(f"✨ Crystallizing Causal Chain: {intent}")
            crystallized.append({
                "intent": intent,
                "vector_dna": vector_dna,
                "causal_map": causal_map, # The structural meaning
                "origin": "Dream",
                "timestamp": time.time()
            })

        # Save to Permanent Memory
        self._save_wisdom(crystallized)

        # Clear Queue (Wake up refreshed)
        with open(self.queue_path, "w") as f:
            json.dump([], f)

        logger.info("☀️ Waking up. Dreams have been woven into reality.")

    def _save_wisdom(self, new_wisdom: List[Dict[str, Any]]):
        current = []
        if self.wisdom_path.exists():
            try:
                with open(self.wisdom_path, "r", encoding="utf-8") as f:
                    current = json.load(f)
            except:
                pass

        current.extend(new_wisdom)

        with open(self.wisdom_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    alchemist = DreamAlchemist()
    alchemist.sleep()
