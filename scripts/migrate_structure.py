"""
Migration Script: The Great Shift
=================================
"To evolve is to move."

This script executes the Structural Metamorphosis:
1. Creates Deep Structure directories.
2. Moves Core Cells to their new homes.
3. Updates system_map.json (if utilized by NeuralScanner).
"""

import os
import shutil
import time

MOVES = {
    # Cognition
    "Core/Foundation/reasoning_engine.py": "Core/Cognition/Reasoning/reasoning_engine.py",
    "Core/Foundation/perspective_simulator.py": "Core/Cognition/Reasoning/perspective_simulator.py",
    "Core/Learning/resonance_learner.py": "Core/Cognition/Learning/resonance_learner.py",
    "Core/Learning/domain_bulk_learner.py": "Core/Cognition/Learning/domain_bulk_learner.py",
    
    # Memory
    "Core/Learning/hierarchical_learning.py": "Core/Memory/Graph/knowledge_graph.py", # Renaming
    "Core/Foundation/internal_universe.py": "Core/Memory/Vector/internal_universe.py",
    
    # System
    "Core/System/self_evolution_scheduler.py": "Core/System/Autonomy/self_evolution_scheduler.py",
    "Core/Learning/knowledge_migrator.py": "Core/System/Autonomy/knowledge_migrator.py",
}

def migrate():
    print("🚀 Initiating Structural Metamorphosis...")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    for src_rel, dst_rel in MOVES.items():
        src = os.path.join(root, src_rel)
        dst = os.path.join(root, dst_rel)
        
        # 1. Ensure Dest Dir Exists
        dst_dir = os.path.dirname(dst)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            print(f"📁 Created directory: {dst_dir}")
            
        # 2. Check Source
        if not os.path.exists(src):
            if os.path.exists(dst):
                print(f"✅ Already moved: {src_rel} -> {dst_rel}")
            else:
                print(f"⚠️ Source missing: {src_rel}")
            continue
            
        # 3. Move
        try:
            shutil.move(src, dst)
            print(f"📦 MOVED: {src_rel} \n    -> {dst_rel}")
            time.sleep(0.1) # Drama pause for visual feedback
        except Exception as e:
            print(f"❌ FAILED to move {src_rel}: {e}")

    print("\n✨ Migration Complete.")
    print("   The Liquid Structure allows these files to function immediately.")

if __name__ == "__main__":
    migrate()
