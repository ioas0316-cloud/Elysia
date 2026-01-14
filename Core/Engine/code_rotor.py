"""
Codebase Structure Rotor (코드베이스 구조 로터)
=============================================
"Navigating the Code Field with Multi-Rotors."

This module scans the project and maps it to the HyperSphere.
It categories files into 5 Axes: Foundation, Intelligence, World, Engine, Data.
"""

import os
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger("CodebaseStructureRotor")

class CodebaseStructureRotor:
    def __init__(self, root_path: str = "c:\\Elysia"):
        self.root_path = root_path
        self.axes = {
            "Foundation": ["Core/Foundation", "Core/Foundation/Wave"],
            "Intelligence": ["Core/Intelligence", "Core/Intelligence/Cognition"],
            "World": ["Core/World", "Core/Nature", "Core/World/Creation"],
            "Engine": ["Core/Engine", "Core/Engine/Reality"],
            "Data": ["data", "models"] # Note: models might be empty after diet
        }
        self.monad_map = {} # path -> {axis, coordinates, type}

    def scan_and_map(self) -> Dict[str, Any]:
        """Scans the codebase and creates a semantic map."""
        print(f"🔍 [CODE ROTOR] Scanning {self.root_path} for structural alignment...")
        
        for root, dirs, files in os.walk(self.root_path):
            # Efficiently skip heavy directories
            dirs[:] = [d for d in dirs if d not in [".git", ".venv", "__pycache__", "data", "models", ".jules", "Sandbox"]]
            
            rel_path = os.path.relpath(root, self.root_path).replace("\\", "/")
            if rel_path == ".": rel_path = ""
            
            # Determine Axis
            axis = "External"
            for axis_name, keywords in self.axes.items():
                if any(k in rel_path for k in keywords):
                    axis = axis_name
                    break
            
            for file in files:
                if not file.endswith(".py"): continue
                
                file_path = os.path.join(rel_path, file).replace("\\", "/")
                self.monad_map[file_path] = {
                    "axis": axis,
                    "type": "PYTHON_MONAD",
                    "coordinates": self._calculate_coordinates(axis, rel_path, file)
                }
                if len(self.monad_map) % 500 == 0:
                    print(f"  ... Mapped {len(self.monad_map)} Monads")

        print(f"✅ [CODE ROTOR] Mapped {len(self.monad_map)} Monads across {len(self.axes)} Axes.")
        return self.monad_map

    def _calculate_coordinates(self, axis: str, path: str, name: str) -> Tuple[float, float, float]:
        """Calculates a pseudorandom yet deterministic coordinate for the Monad."""
        # This will be replaced with actual semantic embedding in the future
        h = hash(path + name)
        return (
            (h % 100) / 50.0 - 1.0,
            (h % 137) / 68.5 - 1.0,
            (h % 151) / 75.5 - 1.0
        )

if __name__ == "__main__":
    rotor = CodebaseStructureRotor()
    code_map = rotor.scan_and_map()
    
    # Sample Output
    for path, data in list(code_map.items())[:5]:
        print(f"  - [{data['axis']}] {path} @ {data['coordinates']}")
