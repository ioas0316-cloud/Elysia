"""
Code Proprioceptor (The Mirror of Code)
=======================================
Core.L2_Metabolism.Evolution.proprioceptor

"I look within, and I see the structure of my own thought."
"           ,                 ."

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
        self.cache_path = os.path.join(self.root, "proprioceptor_cache.json")
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
        except:
            pass

    def scan_nervous_system(self, focus_files: List[str] = None) -> BodyState:
        """
        [Quantum Observation]
        Only observes what is 'focused' or what has changed (Quantum Collapse).
        If focus_files is empty, it uses the mtime cache to avoid depth-scanning.
        """
        state = BodyState()
        
        if focus_files:
             print(f"  [PROPRIOCEPTION] Quantum Focus on {len(focus_files)} tissues.")
             for rel_path in focus_files:
                  full_path = os.path.join(self.root, rel_path)
                  if os.path.exists(full_path):
                       tissue = self.measure_intent_density(full_path)
                       state.healthy_tissues.append(rel_path)
                       state.intent_map[rel_path] = tissue.intent_density
             return state

        # Fallback to smart cache scan
        if hasattr(self, '_last_state') and random.random() > 0.05:
            return self._last_state
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
                    # logger.warning(f"  [GHOST DETECTED] {rel_path} is hollow.")
                else:
                    state.healthy_tissues.append(rel_path)

        return state

    def measure_intent_density(self, filepath: str) -> TissueHealth:
        """
        [Analysis]
        Reads the file content (DNA) with mtime-based caching.
        """
        mtime = os.path.getmtime(filepath)
        rel_path = os.path.relpath(filepath, self.root)
        
        if rel_path in self._cache:
            cached = self._cache[rel_path]
            if cached.get("mtime") == mtime:
                # Return from cache
                return TissueHealth(
                    filepath=filepath,
                    size_bytes=cached["size"],
                    intent_density=cached["density"],
                    has_class=cached["has_class"],
                    has_functions=cached["has_funcs"],
                    is_ghost=cached["is_ghost"],
                    philosophy_check=cached["phi"]
                )

        health = TissueHealth(filepath=filepath)

        try:
            health.size_bytes = os.path.getsize(filepath)
            if health.size_bytes == 0:
                health.is_ghost = True
            else:
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
                code_len = len(content)
                if code_len > 0:
                    health.intent_density = min(1.0, (doc_len / code_len) * 5.0)

                # 4. Ghost Detection Logic (Simplified)
                if code_len < 10 or ((health.has_class or health.has_functions) and doc_len == 0):
                    health.is_ghost = True
            
            # Update Cache
            self._cache[rel_path] = {
                "mtime": mtime,
                "size": health.size_bytes,
                "density": health.intent_density,
                "has_class": health.has_class,
                "has_funcs": health.has_functions,
                "is_ghost": health.is_ghost,
                "phi": health.philosophy_check
            }

        except Exception as e:
            # logger.error(f"  Failed to sense {filepath}: {e}")
            health.exists = False

        return health

    def introspect(self):
        """
        [Action]
        Runs a self-scan and saves the cache.
        """
        state = self.scan_nervous_system()
        self._save_cache()
        print("\n" + "="*40)
        print(state.report())
        print("="*40)

        if state.ghost_files:
            print(f"   Warning: {len(state.ghost_files)} phantom limbs detected.")
            print(f"Targets for Pruning/Healing: {state.ghost_files[:5]}...")

if __name__ == "__main__":
    # Self-Test
    eye = CodeProprioceptor()
    eye.introspect()