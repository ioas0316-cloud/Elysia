"""
Merkava Unity Engine (          )
========================================
"Everything is World. The Project is the Reality."

Grand Orchestrator that unifies:
1. GovernanceEngine (Will/Rotor/Time)
2. RealityProjector (Space/Coagulation/Aesthetics)
3. CodebaseFieldEngine (Monad/Project/Structure)

This is the realization of 'Zero Latency' manifestation.
"""

import logging
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

# Add project root to sys.path for standalone testing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Core.L6_Structure.M5_Engine.governance_engine import GovernanceEngine
from Core.L6_Structure.M5_Engine.reality_projector import RealityProjector, CoagulationMode
from Core.L6_Structure.M5_Engine.code_field_engine import CodebaseFieldEngine
from Core.L6_Structure.M5_Engine.recursive_loop import RecursiveOptimizer

logger = logging.getLogger("MerkavaUnity")

class MerkavaUnity:
    def __init__(self, root: str = "c:\\Elysia"):
        self.root = root
        self.governance = GovernanceEngine()
        self.projector = RealityProjector(CoagulationMode.MESH)
        self.coder = CodebaseFieldEngine(root)
        self.evolver = RecursiveOptimizer()
        
        # The Living State
        self.world_state = {}

    def manifest_unity(self) -> Dict[str, Any]:
        """
        Collapses the entire project into a single 3D World manifestation.
        """
        print("   [MERKAVA] Initiating Grand Synthesis...")
        
        # 1. Sync Project Field (2000+ Monads)
        monad_map = self.coder.sense_neural_mass()
        monads = self.coder.monad_map
        
        # 2. Project Reality Parameters from Governance
        reality_params = self.projector.project(self.governance)
        
        # 3. Transmute Monads (Files) into World Entities (1:1 Mapping)
        entities = []
        for file_path, data in monads.items():
            # Treat each file as a 'Soul Fragment' or 'Monadic Pillar'
            entity = {
                "id": f"monad_{file_path.replace('/', '_').replace('.', '_')}",
                "type": "MONADIC_PILLAR",
                "pos": data["coordinates"], # Use the coordinates from CodeRotor
                "axis": data["axis"],
                "properties": {
                    "mass": 1.0,
                    "color": self._get_axis_color(data["axis"]),
                    "logic_ref": file_path
                }
            }
            entities.append(entity)
            
        print(f"  [MERKAVA] 1:1 Mapping Complete: {len(entities)} Monads manifested.")

        # 4. Final World Synthesis
        self.world_state = {
            "version": "MERKAVA_UNITY_V1",
            "world_name": "MerkavaUniverse",
            "global_time": reality_params["timestamp"],
            "parameters": reality_params["parameters"],
            "divine_parameters": reality_params.get("divine_parameters", {}),
            "monad_entities": entities,
            "evolution": self.evolver.evolve({"monad_entities": entities}), # The Recursive Cycle
            "manifested_at": datetime.now().isoformat()
        }
        
        return self.world_state

    def _get_axis_color(self, axis: str) -> str:
        colors = {
            "Foundation": "#ffffff", # Pure/Base
            "Intelligence": "#ff00ff", # Psionic/Purple
            "World": "#00ff00", # Nature/Green
            "Engine": "#00ffff", # Power/Cyan
            "Data": "#ffff00" # Knowledge/Yellow
        }
        return colors.get(axis, "#888888")

    def save_world(self, path: str = r"C:\game\merkava_world"):
        if not os.path.exists(path):
            os.makedirs(path)
            
        target = os.path.join(path, "world_state.json")
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.world_state, f, indent=4)
        print(f"  [MERKAVA] World saved to {target}")

if __name__ == "__main__":
    merkava = MerkavaUnity()
    world = merkava.manifest_unity()
    merkava.save_world()
