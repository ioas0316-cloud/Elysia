import logging
import sys
import os
import time
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.reasoning_engine import ReasoningEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("CreationProbe")

def prove_creation():
    print("\n🎨 Proving Cosmic Studio (Reality Sculpting)...")
    print("=============================================")
    
    canvas_path = "c:/Elysia/RealityCanvas"
    
    try:
        # 1. Setup
        if os.path.exists(canvas_path):
            shutil.rmtree(canvas_path)
        
        # 2. Initialize Engine
        print("\n1. Initializing Reasoning Engine...")
        engine = ReasoningEngine()
        
        # 3. Create Artifact
        desire = "A poem about Love and Time"
        print(f"\n2. Manifesting Desire: '{desire}'...")
        artifact_path = engine.create(desire)
        
        print(f"   ✨ Artifact Created at: {artifact_path}")
        
        # 4. Verify Content
        if os.path.exists(artifact_path):
            with open(artifact_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            print("\n3. Artifact Content:")
            print("-" * 20)
            print(content)
            print("-" * 20)
            
            if "LOVE" in content and "TIME" in content:
                print("\n✅ SUCCESS: Artifact resonates with the intended Essence.")
            else:
                print("\n⚠️ WARNING: Artifact created, but essence might be weak.")
        else:
            print("\n❌ FAILURE: Artifact file not found.")
            
    except Exception as e:
        print(f"\n❌ CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    prove_creation()
