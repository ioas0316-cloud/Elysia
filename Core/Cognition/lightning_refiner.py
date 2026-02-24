"""
Lightning Refiner: The Sculptor of Flow
=======================================

"I do not fix the code. I remove the dam to let the river run."

This script identifies 'Logical Friction' in the codebase and proposes 
'Lightning Paths'—logical sequences with the least resistance.
"""

import sys
import ast
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from Core.Cognition.kg_manager import KGManager

class LightningRefiner:
    def __init__(self):
        self.kg = KGManager()

    def sense_friction(self, file_path: str):
        print(f"🌩️ [LIGHTNING] Sensing friction in '{file_path}'...")
        path = Path(file_path)
        if not path.exists(): return None

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        try:
            tree = ast.parse(content)
        except Exception as e:
            print(f"💥 [LIGHTNING] The medium is broken (Syntax Error): {e}")
            return None

        friction_points = []
        for node in ast.walk(tree):
            # 1. Deep Nesting (Resistance to flow)
            if isinstance(node, (ast.If, ast.For, ast.With, ast.While)):
                depth = self._get_nesting_depth(node)
                if depth > 3:
                    friction_points.append(f"Deep logical turbulence at L{node.lineno} (Depth {depth}). Intent is trapped.")

            # 2. Redundant Checks (Unnecessary Friction)
            # (Heuristic: consecutive Ifs on same variables)
            
            # 3. Rigid Gates (Hardcoded values in logic flow)
            if isinstance(node, ast.Compare):
                # Hardcoded logic gates offer resistance
                pass

        if not friction_points:
            print("🌊 [LIGHTNING] The medium is superconductive. No resistance felt.")
        else:
            print(f"🔥 [FRICTION] Found {len(friction_points)} barriers to flow.")
            for fp in friction_points:
                print(f"   - {fp}")

        return friction_points

    def _get_nesting_depth(self, node):
        depth = 0
        parent = node
        while hasattr(parent, 'parent'):
            if isinstance(parent, (ast.If, ast.For, ast.With, ast.While)):
                depth += 1
            parent = parent.parent
        return depth

if __name__ == "__main__":
    refiner = LightningRefiner()
    # Analyzing the 'Sovereign Monad' for friction
    refiner.sense_friction('Core/S1_Body/L6_Structure/M1_Merkaba/sovereign_monad.py')
