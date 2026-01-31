"""
Reality Manifestation Script (현행화 실행 스크립트)
==================================================
"Let there be Light, and let the Waves become Form."

This script runs the full Phase 14 pipeline:
1. Governance -> Reality Projector (Parameters)
2. Parameters -> Wave-to-Mesh Transmuter (Entities)
3. Entities -> Filesystem (world_state.json / index.html)
"""

import sys
import os
import json
import logging
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.1_Body.L6_Structure.Engine.governance_engine import GovernanceEngine
from Core.1_Body.L6_Structure.Engine.reality_projector import RealityProjector, CoagulationMode, RealityStyle
from Core.1_Body.L6_Structure.Engine.wave_to_mesh_transmuter import WaveToMeshTransmuter
from Core.1_Body.L4_Causality.World.Creation.project_genesis import ProjectGenesis

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Manifest3D")

def main():
    print("✨ [PHASE 14] Starting Reality Manifestation Pipeline...")
    
    # 1. Initialize Engines
    governance = GovernanceEngine()
    projector = RealityProjector(CoagulationMode.MESH)
    transmuter = WaveToMeshTransmuter()
    genesis = ProjectGenesis(external_root=r"C:\game")
    
    # 2. Set Target Style (e.g. Open World)
    style = RealityStyle.OPEN_WORLD
    projector.set_style(style)
    
    # 3. Project Reality
    print(f"🔮 Projecting internal waves into '{style.value}' parameters...")
    reality_config = projector.project(governance)
    
    # 4. Transmute to Entities
    print("🧬 Transmuting parameters into 3D entities...")
    entities = transmuter.transmute(reality_config)
    
    # 5. Manifest to External World
    project_name = "elysia_manifest_v14"
    print(f"🏗️ Manifesting as project: '{project_name}' at C:\\game\\{project_name}")
    
    manifest_data = {
        "world_name": project_name,
        "style": style.value,
        "parameters": reality_config["parameters"],
        "divine_parameters": {
            "sofa_path": reality_config["sofa_path"],
            "interference": reality_config["interference"]
        },
        "entities": entities,
        "manifested_at": datetime.now().isoformat()
    }
    
    # Ensure directory exists
    target_dir = os.path.join(r"C:\game", project_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    # Write world_state.json
    state_path = os.path.join(target_dir, "world_state.json")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4)
    
    # Write a simple Three.js viewer or just acknowledge success
    print(f"\n✅ SUCCESS: World manifested at {target_dir}")
    print(f"📊 Total Entities: {len(entities)}")
    print(f"📂 State written to {state_path}")
    
    # Trigger ProjectGenesis for basic scaffold if needed
    genesis.create_project(project_name, "THREE_JS_WORLD")

if __name__ == "__main__":
    main()
