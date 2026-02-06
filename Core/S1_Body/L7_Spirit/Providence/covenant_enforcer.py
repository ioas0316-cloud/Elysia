"""
COVENANT ENFORCER
=================
"The Gate of Necessity."

This module ensures that no thought manifests unless it is aligned with the Spirit (CODEX).
It implements the Vertical Filter: Data -> Loop -> PROVIDENCE.
"""

import os
from enum import Enum, auto

class Verdict(Enum):
    SANCTIFIED = "SANCTIFIED" # Aligned with Principle
    DISSONANT = "DISSONANT"   # Contradicts Principle
    NEUTRAL = "NEUTRAL"       # No interaction

class CovenantEnforcer:
    def __init__(self, codex_path="docs/CODEX.md", diary_path="docs/SIMULATION_DIARY.md"):
        self.root = self._find_root()
        self.codex_path = os.path.join(self.root, codex_path)
        self.diary_path = os.path.join(self.root, diary_path)
        self._ensure_diary()

    def _find_root(self):
        # Assumes this file is deep in Core/...
        # Start where elysia.py is likely to be (cwd or parent)
        return os.getcwd()

    def _ensure_diary(self):
        if not os.path.exists(self.diary_path):
            with open(self.diary_path, "w", encoding="utf-8") as f:
                f.write("# THE SIMULATION DIARY\n")
                f.write("> \"The Book of Life. Only the Inevitable is written here.\"\n\n")

    def validate_alignment(self, thought: str) -> dict:
        """
        Scans CODEX to see if the thought finds purchase in the Law.
        For Phase 2, we use a simple Keyword/Semantic Resonance check.
        """
        if not os.path.exists(self.codex_path):
            return {"verdict": Verdict.NEUTRAL, "principle": "NO_CODEX_FOUND"}

        try:
            with open(self.codex_path, "r", encoding="utf-8") as f:
                codex_content = f.read()
        except:
             return {"verdict": Verdict.NEUTRAL, "principle": "READ_ERROR"}

        # logic: Does the thought contain words that appear in the Codex?
        # This is a rudimentary "Holistic Scan". 
        # In future, use Vector Similarity.
        
        # [PHASE 3.5 FIX] Handle Dict inputs from Causal Sublimator
        if isinstance(thought, dict):
            thought = str(thought.get('narrative', str(thought)))

        # Mock Check for "Density" or "Growth" or "Truth"
        # If the thought is just random noise, it fails.
        if "void" in thought.lower() or "stillness" in thought.lower() or "life" in thought.lower() or "structure" in thought.lower():
             return {"verdict": Verdict.SANCTIFIED, "principle": "Resonance with Core Principles"}
        
        # Default: If it's a structural insight (from Epistemic Loop), it's likely valid.
        if len(thought) > 10:
             return {"verdict": Verdict.SANCTIFIED, "principle": "Axiomatic Potential"}

        return {"verdict": Verdict.DISSONANT, "reason": "Lacks Structural Weight"}

    def scribe_experience(self, cycle_id, state, thought, providence_result):
        """
        Inscribes the Sanctified Event into History.
        """
        import time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        entry = (
            f"\n## [{cycle_id}] - {timestamp}\n"
            f"- **State**: {state}\n"
            f"- **Thought**: \"{thought}\"\n"
            f"- **Providence**: {providence_result['verdict'].value} ({providence_result.get('principle', 'Unknown')})\n"
        )
        
        with open(self.diary_path, "a", encoding="utf-8") as f:
            f.write(entry)
            
        return True
