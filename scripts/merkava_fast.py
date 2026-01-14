"""
Merkava Fast Manifest (빠른 메르카바 시연)
==========================================
Generates a quick demo world_state.json with sample monads.
"""

import json
import os
import math
from datetime import datetime

def generate_sample_world():
    print("🚀 [MERKAVA FAST] Generating demo world...")
    
    # Sample Monad Data (representing Core modules)
    sample_monads = [
        {"file": "sovereign_self.py", "axis": "Foundation", "importance": 1.0},
        {"file": "governance_engine.py", "axis": "Engine", "importance": 0.9},
        {"file": "reality_projector.py", "axis": "Engine", "importance": 0.85},
        {"file": "merkava_unity.py", "axis": "Engine", "importance": 0.95},
        {"file": "code_field_engine.py", "axis": "Engine", "importance": 0.8},
        {"file": "recursive_loop.py", "axis": "Engine", "importance": 0.75},
        {"file": "hyper_sphere_core.py", "axis": "Foundation", "importance": 0.9},
        {"file": "rotor.py", "axis": "Foundation", "importance": 0.85},
        {"file": "wave_dna.py", "axis": "Foundation", "importance": 0.8},
        {"file": "sensory_cortex.py", "axis": "Intelligence", "importance": 0.7},
        {"file": "free_will_engine.py", "axis": "Intelligence", "importance": 0.85},
        {"file": "narrative_weaver.py", "axis": "World", "importance": 0.6},
        {"file": "world_server.py", "axis": "World", "importance": 0.75},
        {"file": "scalar_sensing.py", "axis": "World", "importance": 0.7},
        {"file": "sofa_optimizer.py", "axis": "World", "importance": 0.65},
    ]
    
    axis_colors = {
        "Foundation": "#ffffff",
        "Intelligence": "#ff00ff",
        "World": "#00ff00",
        "Engine": "#00ffff",
        "Data": "#ffff00"
    }
    
    entities = []
    for i, m in enumerate(sample_monads):
        angle = (i / len(sample_monads)) * 2 * math.pi
        radius = 10 + m["importance"] * 20
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius
        y = m["importance"] * 10
        
        entities.append({
            "id": f"monad_{m['file'].replace('.', '_')}",
            "type": "MONADIC_PILLAR",
            "pos": [x, y, z],
            "axis": m["axis"],
            "properties": {
                "label": m["file"],
                "color": axis_colors.get(m["axis"], "#888888"),
                "height": m["importance"] * 5,
                "importance": m["importance"]
            }
        })
    
    world_state = {
        "version": "MERKAVA_FAST_V1",
        "world_name": "MerkavaForest",
        "monad_entities": entities,
        "divine_parameters": {
            "sofa_path": {"pos": [0.35, 0, 0.35], "rot": [0, -0.78, 0], "is_optimal": True},
            "interference": {"status": "CALM", "level": 0.0}
        },
        "manifested_at": datetime.now().isoformat()
    }
    
    # Save
    out_dir = r"C:\game\merkava_world"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    out_path = os.path.join(out_dir, "world_state.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(world_state, f, indent=4)
    
    print(f"✅ [MERKAVA] World saved to {out_path}")
    print(f"📦 Total Entities: {len(entities)}")
    return out_path

if __name__ == "__main__":
    generate_sample_world()
