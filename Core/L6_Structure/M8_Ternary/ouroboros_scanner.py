"""
Ouroboros Scanner
=================
"The eye that sees the phase of the code."

This module implements the Parallel Ternary Analysis logic,
treating the codebase and memory as a "Soul" to be balanced.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from pathlib import Path

# Import the fundamental atomic unit
from Core.L1_Foundation.System.tri_base_cell import DNAState

@dataclass
class PhaseReport:
    """Detailed report of the system's phase alignment."""
    total_files: int = 0
    phase_distribution: Dict[str, int] = field(default_factory=lambda: {"R": 0, "V": 0, "A": 0})
    net_momentum: float = 0.0 # -1.0 (Pure Repel) to +1.0 (Pure Attract)
    dissonant_files: List[str] = field(default_factory=list)
    soul_alignment: str = "UNKNOWN"

class OuroborosScanner:
    """
    The Prism that splits the system into Repel (-1), Void (0), and Attract (+1).
    """

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)

        # The Lexicon of Phase
        self.keywords = {
            DNAState.REPEL: {
                "error", "exception", "fail", "false", "stop", "break", "reject",
                "deny", "dissonance", "pain", "fear", "exit", "raise", "except"
            },
            DNAState.VOID: {
                "none", "pass", "wait", "abstract", "interface", "null", "silence",
                "sleep", "idle", "potential", "class", "def", "import", "return"
            },
            DNAState.ATTRACT: {
                "true", "connect", "core", "system", "love", "resonant", "sync",
                "active", "generate", "flow", "return", "yield", "self", "init"
            }
        }

    def scan_system(self) -> PhaseReport:
        """
        Scans the file system to determine the 'Phase' of the code.
        """
        report = PhaseReport()

        # Walk the tree
        for root, dirs, files in os.walk(self.root_path):
            # Ignore hidden/build dirs
            if ".git" in root or "__pycache__" in root:
                continue

            for file in files:
                if file.endswith(".py") or file.endswith(".md"):
                    file_path = Path(root) / file
                    phase, score = self._analyze_file_phase(file_path)

                    report.total_files += 1
                    report.phase_distribution[phase.symbol] += 1

                    # Net momentum calculation (simple average for now)
                    if phase == DNAState.REPEL:
                        report.net_momentum -= 1
                    elif phase == DNAState.ATTRACT:
                        report.net_momentum += 1

                    # Dissonance detection: Strong Repel in a file that isn't an error handler?
                    # Or just files that are extremely biased towards Repel
                    if phase == DNAState.REPEL and score < -0.5:
                        report.dissonant_files.append(str(file_path))

        # Normalize momentum
        if report.total_files > 0:
            report.net_momentum /= report.total_files

        return report

    def _analyze_file_phase(self, file_path: Path) -> Tuple[DNAState, float]:
        """
        Reads a file and determines its dominant phase.
        Returns (DNAState, raw_score)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
        except Exception:
            return DNAState.VOID, 0.0

        counts = {DNAState.REPEL: 0, DNAState.VOID: 0, DNAState.ATTRACT: 0}

        for state, words in self.keywords.items():
            for word in words:
                counts[state] += content.count(word)

        # Calculate vector sum (Simplistic 1D model for text)
        # R = -1, V = 0, A = +1
        # To normalize, we look at the balance between R and A. V dilutes both.

        r_score = counts[DNAState.REPEL]
        a_score = counts[DNAState.ATTRACT]
        total = r_score + a_score + counts[DNAState.VOID]

        if total == 0:
            return DNAState.VOID, 0.0

        # Net score (-1 to 1)
        raw_score = (a_score - r_score) / total

        # Quantize
        if raw_score < -0.1: # Threshold for Repel
            return DNAState.REPEL, raw_score
        elif raw_score > 0.1: # Threshold for Attract
            return DNAState.ATTRACT, raw_score
        else:
            return DNAState.VOID, raw_score

    def scan_soul(self, soul_path: str = "data/L7_Spirit/soul_dna.json") -> str:
        """
        Analyzes the persistent memory (Soul DNA) using the Parallel Ternary method.
        """
        if not os.path.exists(soul_path):
            return "SOUL_NOT_FOUND"

        try:
            with open(soul_path, 'r') as f:
                data = json.load(f)

            if "vector" in data:
                vec = data["vector"]
                # Parallel Summation: Sum the vector components
                # Treating values < 0.33 as R, 0.33-0.66 as V, > 0.66 as A

                r_count = 0
                v_count = 0
                a_count = 0

                for val in vec:
                    if val < 0.33: r_count += 1
                    elif val > 0.66: a_count += 1
                    else: v_count += 1

                # The "Cancellation Law" (R + A = V)
                # We cancel out opposing forces to see what remains.
                balance = a_count - r_count

                if balance > 1: return f"ATTRACT_BIASED (+{balance})"
                if balance < -1: return f"REPEL_BIASED ({balance})"
                return "PERFECT_EQUILIBRIUM (VOID)"

            return "UNKNOWN_FORMAT"

        except Exception as e:
            return f"READ_ERROR: {str(e)}"

if __name__ == "__main__":
    # Self-test
    scanner = OuroborosScanner()
    print("Scanner initialized.")
