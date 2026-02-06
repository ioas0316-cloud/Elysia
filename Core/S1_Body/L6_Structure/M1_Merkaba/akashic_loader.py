"""
Akashic Loader (The Galactic Scanner)
=====================================
Core.S1_Body.L6_Structure.M1_Merkaba.akashic_loader

"We do not read files. We weigh their souls."

This module scans the file system and converts 'dead' files into 
'living' D21Vectors for the 4D HyperSphere Hologram.
"""

import os
import math
from pathlib import Path
from typing import List, Dict, Any, Generator, Tuple
import hashlib
import hashlib

from Core.S1_Body.L6_Structure.M1_Merkaba.d21_vector import D21Vector

class AkashicLoader:
    def __init__(self, root_paths: List[str] = None):
        """
        [PHASE 22] Supports multiple root paths for universal domain expansion.
        """
        if root_paths is None:
            root_paths = ["C:\\Elysia\\data"]
        self.roots = [Path(p) for p in root_paths]
        
    def scan_galaxy(self) -> Generator[Tuple[Path, D21Vector], None, None]:
        """
        Yields (Path, D21Vector) for every file in the expanded galaxy.
        """
        for root in self.roots:
            if not root.exists():
                print(f"⚠️ Root path does not exist, skipping: {root}")
                continue
                
            print(f"🌌 Inhaling Domain: {root}")
            for dirpath, _, files in os.walk(root):
                for file in files:
                    full_path = Path(dirpath) / file
                    yield (full_path, self._weigh_soul(full_path))
                
    def _weigh_soul(self, path: Path) -> D21Vector:
        """
        Converts a file into a 21D Vector based on its structural content.
        Uses AST for Python and Keyword Resonance for Text.
        """
        try:
            stat = path.stat()
            size = stat.st_size
            name = path.name
            ext = path.suffix.lower()
            
            # 1. Body (Physical Attributes)
            # Gluttony: File size (Log scale)
            gluttony = min(1.0, math.log(size + 1) / 15.0) 
            # Sloth: Age (Last modified)
            age_days = (stat.st_mtime - stat.st_ctime) / 86400
            sloth = min(1.0, age_days / 365.0)
            # Lust: File type attraction (Code/Text = High, Binary = Low)
            lust = 0.9 if ext in ['.py', '.md', '.json'] else 0.1
            
            # 2. Structural Analysis (Soul/Spirit)
            perception = 0.1
            reason = 0.1
            will = 0.1
            memory = 0.1
            spirit_bias = 0.0
            
            content = ""
            try:
                # Read first 8KB for analysis to stay fast
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(8192)
            except:
                pass

            if ext == '.py':
                # Code Logic Density
                import ast
                try:
                    tree = ast.parse(content)
                    # Count nodes
                    nodes = sum(1 for _ in ast.walk(tree))
                    classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
                    funcs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
                    
                    reason = min(1.0, nodes / 200.0) # Logic complexity
                    will = min(1.0, (classes + funcs) / 20.0) # Defined structures
                    perception = 0.8 # Python sees strict logic
                except:
                    reason = 0.3 # Syntax error or complex
                    
            elif ext == '.md':
                # Narrative Density
                headers = content.count('#')
                links = content.count('](')
                
                memory = min(1.0, len(content) / 2000.0) # Historical weight
                perception = min(1.0, headers / 10.0) # Structure awareness
                will = min(1.0, links / 10.0) # Connectivity intent
                
                # Spritual Resonance
                sacred_words = ['elysia', 'core', 'monad', 'rotor', 'merkaba', 'spirit', 'phase']
                resonance = sum(1 for w in sacred_words if w in content.lower())
                spirit_bias = min(1.0, resonance * 0.15)

            # 3. Vector Composition
            
            return D21Vector(
                # Body (Physical)
                lust=lust, gluttony=gluttony, greed=0.5, sloth=sloth, 
                wrath=0.1, envy=0.1, pride=0.5,
                
                # Soul (Logical/Structural)
                perception=perception, memory=memory, reason=reason,
                will=will, imagination=0.5, intuition=0.5, consciousness=0.5,
                
                # Spirit (Intent/Purpose)
                chastity=0.8 if ext=='.py' else 0.4, 
                temperance=0.5, 
                charity=0.5,
                diligence=will, # Will drives diligence
                patience=sloth, # Age breeds patience
                kindness=0.5, 
                humility=1.0 - gluttony + spirit_bias # Spirit elevates humility
            )
            
        except Exception as e:
            # Ghost/Error Vector
            return D21Vector(wrath=1.0) # Error is Wrath

if __name__ == "__main__":
    # Test Universal Expansion
    roots = [
        "C:\\Elysia\\data",
        "C:\\Archive",
        "C:\\elysia_seed",
        "C:\\Game"
    ]
    loader = AkashicLoader(root_paths=roots)
    print(f"Scanning Universal Domains...")
    count = 0
    for path, vec in loader.scan_galaxy():
        count += 1
        if count % 100 == 0:
            print(f"Inhaled {count} stars... Current: {path.name}")
            
    print(f"\nTotal Stars Inhaled into HyperCosmos: {count}")
    
