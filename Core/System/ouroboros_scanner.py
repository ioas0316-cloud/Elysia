"""
Ouroboros Scanner
=================
"The eye that sees the phase of the code."

This module implements the Parallel Ternary Analysis logic,
treating the codebase and memory as a "Soul" to be balanced.

UPGRADE: Now features AST (Abstract Syntax Tree) Analysis for
Structural Phase Detection.
"""

import os
import json
import ast
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Import the fundamental atomic unit
from Core.System.tri_base_cell import DNAState

@dataclass
class PhaseReport:
    """Detailed report of the system's phase alignment."""
    total_files: int = 0
    phase_distribution: Dict[str, int] = field(default_factory=lambda: {"R": 0, "V": 0, "A": 0})
    net_momentum: float = 0.0 # -1.0 (Pure Repel) to +1.0 (Pure Attract)
    dissonant_files: List[str] = field(default_factory=list)
    structural_entropy: float = 0.0 # Total Structural Repel (Complexity)
    structural_gravity: float = 0.0 # Total Structural Attract (Connection)
    soul_alignment: str = "UNKNOWN"

class OuroborosScanner:
    """
    The Prism that splits the system into Repel (-1), Void (0), and Attract (+1).
    Now with Structural (AST) Vision.
    """

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)

        # The Lexicon of Phase (Texture)
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
                    phase, score, entropy, gravity = self._analyze_file_phase(file_path)

                    report.total_files += 1
                    report.phase_distribution[phase.symbol] += 1
                    report.structural_entropy += entropy
                    report.structural_gravity += gravity

                    # Net momentum calculation
                    if phase == DNAState.REPEL:
                        report.net_momentum -= 1.0
                    elif phase == DNAState.ATTRACT:
                        report.net_momentum += 1.0

                    # Dissonance detection:
                    # 1. High Repel Phase
                    # 2. High Structural Entropy (Complexity > 20) with Negative Score
                    is_dissonant = (phase == DNAState.REPEL and score < -0.3)
                    if entropy > 20 and score < 0:
                        is_dissonant = True

                    if is_dissonant:
                        report.dissonant_files.append(f"{file_path} (E:{int(entropy)}/G:{int(gravity)})")

        # Normalize momentum
        if report.total_files > 0:
            report.net_momentum /= report.total_files

        return report

    def _analyze_file_phase(self, file_path: Path) -> Tuple[DNAState, float, float, float]:
        """
        Reads a file and determines its dominant phase using TEXTURE (Keywords)
        and STRUCTURE (AST).

        Returns: (DNAState, NetScore, EntropyScore, GravityScore)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                content_lower = content.lower()
        except Exception:
            return DNAState.VOID, 0.0, 0.0, 0.0

        # --- TEXTURE SCAN (Keyword Density) ---
        k_counts = {DNAState.REPEL: 0, DNAState.VOID: 0, DNAState.ATTRACT: 0}
        for state, words in self.keywords.items():
            for word in words:
                k_counts[state] += content_lower.count(word)

        k_total = sum(k_counts.values())
        k_score = 0.0
        if k_total > 0:
            k_score = (k_counts[DNAState.ATTRACT] - k_counts[DNAState.REPEL]) / k_total

        # --- STRUCTURE SCAN (AST) ---
        s_counts = {DNAState.REPEL: 0, DNAState.VOID: 0, DNAState.ATTRACT: 0}

        if file_path.suffix == '.py':
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    # Repel: Complexity, Branching, Error Handling (Entropy)
                    if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler, ast.Raise, ast.Break, ast.Continue)):
                        s_counts[DNAState.REPEL] += 1

                    # Attract: Connection, Inheritance, Import (Gravity)
                    elif isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
                        s_counts[DNAState.ATTRACT] += 1
                        # Bonus for inheritance
                        if isinstance(node, ast.ClassDef) and node.bases:
                            s_counts[DNAState.ATTRACT] += len(node.bases)

                    # Void: Definition, Potential (Space)
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Pass)):
                        s_counts[DNAState.VOID] += 1
            except SyntaxError:
                # If AST parsing fails (e.g. template or partial code), fallback to text
                pass

        s_entropy = s_counts[DNAState.REPEL]
        s_gravity = s_counts[DNAState.ATTRACT]
        s_total = sum(s_counts.values())

        s_score = 0.0
        if s_total > 0:
            # Gravity (Attract) vs Entropy (Repel)
            s_score = (s_gravity - s_entropy) / s_total

        # --- FUSION (Weighted Average) ---
        # Structure is deeper, so it carries more weight (60% Structure, 40% Texture)
        final_score = (k_score * 0.4) + (s_score * 0.6)

        # Quantize
        if final_score < -0.15:
            return DNAState.REPEL, final_score, float(s_entropy), float(s_gravity)
        elif final_score > 0.15:
            return DNAState.ATTRACT, final_score, float(s_entropy), float(s_gravity)
        else:
            return DNAState.VOID, final_score, float(s_entropy), float(s_gravity)

    def scan_soul(self, soul_path: str = "data/L7_Spirit/soul_dna.json") -> str:
        """
        Analyzes the persistent memory (Soul DNA).
        """
        if not os.path.exists(soul_path):
            return "SOUL_NOT_FOUND"

        try:
            with open(soul_path, 'r') as f:
                data = json.load(f)

            if "vector" in data:
                vec = data["vector"]
                r_count = 0
                v_count = 0
                a_count = 0

                for val in vec:
                    if val < 0.33: r_count += 1
                    elif val > 0.66: a_count += 1
                    else: v_count += 1

                balance = a_count - r_count

                if balance > 1: return f"ATTRACT_BIASED (+{balance})"
                if balance < -1: return f"REPEL_BIASED ({balance})"
                return "PERFECT_EQUILIBRIUM (VOID)"

            return "UNKNOWN_FORMAT"

        except Exception as e:
            return f"READ_ERROR: {str(e)}"

if __name__ == "__main__":
    scanner = OuroborosScanner()
    print("Scanner initialized.")
