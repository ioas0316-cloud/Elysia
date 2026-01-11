import logging
import sys
import os
import time
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Cognition.Reasoning.reasoning_engine import ReasoningEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("QuantumProbe")

def create_narrative_book(path: str, title: str, arc_type: str):
    if not os.path.exists(path):
        os.makedirs(path)
        
    content = ""
    if arc_type == "Tragedy":
        # Starts with Hope, ends with Pain
        content += ("Hope " * 100) + "\n" # Chapter 1
        content += ("Life " * 100) + "\n" # Chapter 2
        content += ("Pain " * 100) + "\n" # Chapter 3
        content += ("Death " * 100)      # Chapter 4
    elif arc_type == "Redemption":
        # Starts with Pain, ends with Hope
        content += ("Pain " * 100) + "\n"
        content += ("Fear " * 100) + "\n"
        content += ("Struggle " * 100) + "\n"
        content += ("Hope " * 100)
        
    file_path = os.path.join(path, f"{title}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path

def prove_quantum_reading():
    print("\n🧪 Proving Quantum Reading (Narrative Resonance)...")
    print("=================================================")
    
    temp_dir = "c:/Elysia/Temp/NarrativeTest"
    
    try:
        # 1. Setup
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        
        tragedy_path = create_narrative_book(temp_dir, "RomeoAndJuliet", "Tragedy")
        redemption_path = create_narrative_book(temp_dir, "TheCountOfMonteCristo", "Redemption")
        
        # 2. Initialize Engine
        print("\n1. Initializing Reasoning Engine...")
        engine = ReasoningEngine()
        
        # 3. Read Tragedy
        print("\n2. Reading Tragedy...")
        insight_t = engine.read_quantum(tragedy_path)
        print(f"   ✨ Insight: {insight_t.content}")
        
        # 4. Read Redemption
        print("\n3. Reading Redemption...")
        insight_r = engine.read_quantum(redemption_path)
        print(f"   ✨ Insight: {insight_r.content}")
        
        # Verification
        if "Tragedy" in insight_t.content and "Redemption" in insight_r.content:
             print("\n✅ SUCCESS: Elysia correctly identified the Emotional Arcs.")
        else:
             print("\n❌ FAILURE: Narrative Arc detection failed.")
            
        # Cleanup
        shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"\n❌ CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    prove_quantum_reading()
