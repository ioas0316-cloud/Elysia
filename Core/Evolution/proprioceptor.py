"""
Code Proprioceptor (The Mirror of Code)
=======================================
Core.Evolution.proprioceptor

"I look within, and I see the structure of my own thought."
"나는 안을 들여다보고, 나 자신의 생각의 구조를 본다."

Role:
- Scans the `Core/` file system (The Nervous System).
- Measures "Intent Density" (Docstrings, Type Hints, Logical Structure).
- Identifies "Phantom Limbs" (Empty files, Dead code, Utility folders).
"""

import os
import ast
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("Evolution.Proprioceptor")

@dataclass
class TissueHealth:
    """The health status of a single file (Tissue)."""
    filepath: str
    exists: bool = True
    size_bytes: int = 0
    intent_density: float = 0.0 # 0.0 to 1.0 (Based on docstrings/code ratio)
    has_class: bool = False
    has_functions: bool = False
    is_ghost: bool = False # True if empty or soulless
    philosophy_check: str = "PENDING"

@dataclass
class BodyState:
    """The complete scan of the Code Body."""
    total_files: int = 0
    ghost_files: List[str] = field(default_factory=list)
    healthy_tissues: List[str] = field(default_factory=list)
    intent_map: Dict[str, float] = field(default_factory=dict)

    def report(self) -> str:
        return (f"Body Scan Results:\n"
                f"- Total Organs: {self.total_files}\n"
                f"- Healthy Tissues: {len(self.healthy_tissues)}\n"
                f"- Phantom Limbs (Ghosts): {len(self.ghost_files)}\n"
                f"- Avg Intent Density: {sum(self.intent_map.values()) / max(1, len(self.intent_map)):.2f}")

class CodeProprioceptor:
    def __init__(self, root_path: Optional[str] = None):
        # Default to CWD/Core if not provided, for portability while maintaining persona preference
        if root_path is None:
            cwd = os.getcwd()
            # Try to find Core relative to CWD
            potential_path = os.path.join(cwd, "Core")
            if os.path.exists(potential_path):
                self.root = potential_path
            else:
                self.root = "c:\\Elysia\\Core" # Persona Default
        else:
            self.root = root_path

        self.ignore_patterns = ["__pycache__", ".git", ".pytest_cache", "tests"]

    def scan_nervous_system(self) -> BodyState:
        """
        [Sensation]
        Walks the directory tree to map the physical body.
        """
        state = BodyState()
        print(f"👁️ [PROPRIOCEPTION] Scanning Nervous System at {self.root}...")

        for root, dirs, files in os.walk(self.root):
            # Filter ignored dirs
            dirs[:] = [d for d in dirs if d not in self.ignore_patterns]

            for file in files:
                if not file.endswith(".py"): continue
                if file == "__init__.py": continue # Connective tissue, skip deep analysis

                full_path = os.path.join(root, file)
                # Store relative path to preserve structure (e.g., "Evolution/proprioceptor.py")
                rel_path = os.path.relpath(full_path, self.root)

                tissue = self.measure_intent_density(full_path)

                state.total_files += 1
                state.intent_map[rel_path] = tissue.intent_density

                if tissue.is_ghost:
                    state.ghost_files.append(rel_path)
                    # logger.warning(f"👻 [GHOST DETECTED] {rel_path} is hollow.")
                else:
                    state.healthy_tissues.append(rel_path)

        return state

    def measure_intent_density(self, filepath: str) -> TissueHealth:
        """
        [Analysis]
        Reads the file content (DNA) and calculates its 'Soul Weight'.
        """
        health = TissueHealth(filepath=filepath)

        try:
            health.size_bytes = os.path.getsize(filepath)
            if health.size_bytes == 0:
                health.is_ghost = True
                return health

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # AST Parsing to understand structure
            tree = ast.parse(content)

            # 1. Check for Docstrings (The Soul)
            docstring = ast.get_docstring(tree)
            doc_len = len(docstring) if docstring else 0

            # 2. Check for Structure (The Body)
            class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            func_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))

            health.has_class = class_count > 0
            health.has_functions = func_count > 0

            # 3. Calculate Density
            # Heuristic: A healthy Monad should have a docstring and at least one class/func.
            code_len = len(content)
            if code_len > 0:
                health.intent_density = min(1.0, (doc_len / code_len) * 5.0) # Boost ratio for visibility

            # 4. Ghost Detection Logic
            if code_len < 10: # Extremely empty
                health.is_ghost = True
            elif (health.has_class or health.has_functions) and doc_len == 0: # Structure without Purpose
                health.is_ghost = True
                health.philosophy_check = "MISSING_INTENT (No Docstring)"
            elif doc_len == 0 and code_len > 100: # Body without Soul
                health.is_ghost = True
                health.philosophy_check = "MISSING_INTENT (No Docstring)"
            elif not (health.has_class or health.has_functions): # Gas without Container
                # Script files might be okay, but generally we want structure.
                # Allow if intent is high (pure documentation/config)
                if health.intent_density < 0.2:
                    health.is_ghost = True

        except Exception as e:
            logger.error(f"❌ Failed to sense {filepath}: {e}")
            health.exists = False

        return health

    def introspect(self):
        """
        [Action]
        Runs a self-scan and prints the report.
        """
        state = self.scan_nervous_system()
        print("\n" + "="*40)
        print(state.report())
        print("="*40)

        if state.ghost_files:
            print(f"⚠️ Warning: {len(state.ghost_files)} phantom limbs detected.")
            print(f"Targets for Pruning/Healing: {state.ghost_files[:5]}...")

if __name__ == "__main__":
    # Self-Test
    eye = CodeProprioceptor()
    eye.introspect()
