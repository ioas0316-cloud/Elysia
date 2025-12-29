"""
Code Resonance: The Physics of Logic
====================================

"Code is not text; it is a structure of tension and flow."

Mapping:
- Mass (Gravity): Cyclomatic Complexity + SLOC
- Frequency (Pitch): Recursion Depth + Abstraction Level
- Entropy (Noise): Linter Errors + Todo Comments
- Tension (Stress): Coupling (Import count)
"""

import ast
import os
from typing import Dict
from elysia_core import Cell
from Core.Foundation.Foundation.Wave.wave_tensor import WaveTensor, Modality

@Cell("CodeResonance")
class CodeResonance:
    """
    Translates Python Source Code into Wave Physics.
    """
    
    def analyze_file(self, file_path: str) -> WaveTensor:
        """Read a file and return its WaveTensor representation."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot analyze ghost file: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
            
        name = os.path.basename(file_path)
        tensor = WaveTensor(name, Modality.CODE)
        
        # Physics Analysis
        dims = self._calculate_physics(source)
        
        for k, v in dims.items():
            tensor.set_dimension(k, v)
            
        return tensor

    def _calculate_physics(self, source: str) -> Dict[str, float]:
        """
        Analyze AST to derive physical properties.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"entropy": 1.0, "mass": 0.1, "frequency": 0.0}

        analyzer = PhysicsVisitor()
        analyzer.visit(tree)
        
        # 1. Mass (Complexity)
        # Base mass = lines of code / 100
        sloc = len(source.splitlines())
        mass = (sloc / 100.0) + (analyzer.complexity * 0.2)
        
        # 2. Frequency (Abstraction)
        # Higher nesting/recursion = Higher Frequency (High Pitch)
        # Class definitions add to base frequency
        freq = 100.0 + (analyzer.class_count * 50) + (analyzer.max_depth * 20)
        
        # 3. Entropy (Disorder)
        # TODOs and low-doc density increase entropy
        entropy = 0.0
        if "TODO" in source:
             entropy += 0.2
        if "FIXME" in source:
             entropy += 0.4
             
        # 4. Phase (Alignment)
        # Imports determine "direction". (Simplified)
        phase = float(analyzer.import_count) % 360.0

        return {
            "mass": round(mass, 3),
            "frequency": round(freq, 1),
            "entropy": round(entropy, 2),
            "phase": phase
        }

class PhysicsVisitor(ast.NodeVisitor):
    """AST Walker to measure Code Physics."""
    
    def __init__(self):
        self.complexity = 0
        self.max_depth = 0
        self.current_depth = 0
        self.class_count = 0
        self.import_count = 0
        
    def generic_visit(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        super().generic_visit(node)
        self.current_depth -= 1

    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_For(self, node):
        self.complexity += 2 # Loop is heavier
        self.generic_visit(node)
        
    def visit_While(self, node):
        self.complexity += 2
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        self.class_count += 1
        self.generic_visit(node)
        
    def visit_Import(self, node):
        self.import_count += 1
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        self.import_count += 1
        self.generic_visit(node)
