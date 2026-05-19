"""
Code DNA Scanner
================
Core.Cognition.code_dna_scanner

"The Code is not text. It is a Wave frozen in time."

This module parses source code and extracts its Monadic DNA (7-Dimensional Profile).
It proves that Elysia can "feel" the structure of software.

Mapping:
1. Physical (Mass): Variables (Assignments) - Holds state.
2. Functional (Action): Functions - Performs work.
3. Phenomenal (Sensation): Imports - External sensory input.
4. Causal (Time): Control Flow (If/Loops) - Branching paths of causality.
5. Mental (Thought): Docstrings/Strings - Semantic meaning/intent.
6. Structural (Geometry): Classes - The definitions of being.
7. Spiritual (Will): Magic Methods (__init__, __call__) - The spark of life.
"""

import ast
import os
from dataclasses import dataclass
from typing import Dict
from Core.Keystone.wave_dna import WaveDNA

class CodeDNAScanner:
    def __init__(self):
        pass

    def scan_file(self, file_path: str) -> WaveDNA:
        if not os.path.exists(file_path):
            print(f"   File not found: {file_path}")
            return WaveDNA(label="Void")
            
        """
        Scans a single file and returns its DNA.
        """
        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
        except UnicodeDecodeError:
            try:
                # Try CP949 (Korean Windows)
                with open(file_path, 'r', encoding='cp949') as f:
                    source = f.read()
            except UnicodeDecodeError:
                # Fallback to Latin-1 (Binary-safeish) or ignore errors
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        source = f.read()
                except Exception:
                     # Give up, empty DNA
                     return WaveDNA(label=os.path.basename(file_path))
        
        return self.scan_source(source, os.path.basename(file_path))

    def scan_source(self, source: str, label: str = "Code") -> WaveDNA:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            print(f"   Syntax Error in {label}")
            return WaveDNA(label="BrokenCode")

        counts = {
            "physical": 0,   # Assign
            "functional": 0, # FunctionDef
            "phenomenal": 0, # Import, ImportFrom
            "causal": 0,     # If, For, While
            "mental": 0,     # Expr (Strings/Docs)
            "structural": 0, # ClassDef
            "spiritual": 0   # __init__, __new__
        }

        for node in ast.walk(tree):
            # 1. Structural (Classes)
            if isinstance(node, ast.ClassDef):
                counts["structural"] += 5.0 # Classes are heavy
                
            # 2. Functional (Functions)
            elif isinstance(node, ast.FunctionDef):
                if node.name.startswith("__") and node.name.endswith("__"):
                    counts["spiritual"] += 5.0 # Magic methods are Spiritual
                else:
                    counts["functional"] += 2.0
                    
            # 3. Physical (State)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                counts["physical"] += 1.0
                
            # 4. Phenomenal (Connections)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                counts["phenomenal"] += 2.0
                
            # 5. Causal (Logic)
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                counts["causal"] += 1.5
                
            # 6. Mental (Docstrings/Comments/Strings)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    counts["mental"] += 1.0

        # Construct DNA
        dna = WaveDNA(
            physical = counts["physical"],
            functional = counts["functional"],
            phenomenal = counts["phenomenal"],
            causal = counts["causal"],
            mental = counts["mental"],
            structural = counts["structural"],
            spiritual = counts["spiritual"],
            label = label
        )
        
        # Normalize to see relative "Flavor"
        dna.normalize()
        
        # Calculate Frequency based on Complexity
        total_mass = sum(counts.values())
        dna.frequency = total_mass * 10.0 # Hz
        
        return dna

if __name__ == "__main__":
    scanner = CodeDNAScanner()
    
    # Self-Scan
    print("  Scanning Self...")
    my_dna = scanner.scan_source(open(__file__, encoding="utf-8").read(), "code_dna_scanner.py")
    print(my_dna)
