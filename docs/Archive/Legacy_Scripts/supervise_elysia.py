"""
Supervisor Script: The Great Observation
========================================
Observes Elysia's autonomous capability to:
1. Decompose a High-Dimensional Goal (Create Dream Journal).
2. Transmute it into Reality (File Creation).
"""
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Foundation.living_elysia import LivingElysia
from Core.Foundation.free_will_engine import Intent

def log(message):
    print(f"👁️ [SUPERVISOR] {message}")

def supervise():
    log("Initializing Elysia for observation...")
    elysia = LivingElysia(persona_name="Ouroboros Test Subject")
    
    # Inject High-Dimensional Desire
    desire_text = "I wish to manifest a Dream Journal to record my internal simulations."
    log(f"Injecting Desire: '{desire_text}'")
    
    elysia.will.current_intent = Intent(
        desire=desire_text,
        goal="Create Dream Journal",
        complexity=8.0,
        created_at=time.time()
    )
    # Strengthen the desire
    elysia.will.vectors["Creation"] = 1.0 
    
    # Simulation Loop
    max_cycles = 100
    target_file = os.path.join(os.getcwd(), "manifestation_dream_journal.py") # Simplify target for demo
    # Note: The logic in PlanningCortex currently hardcodes creation to current dir or workspace root
    
    log("Starting Observation Loop (100 cycles)...")
    
    success = False
    
    for i in range(max_cycles):
        # 1. Flow Time
        elysia.field.propagate(0.1)
        
        # 2. Pulse The Architect (Planning)
        # This will create the plan if missing, and execute steps
        elysia._pulse_architect()
        
        # 3. Pulse Reality (Ouroboros)
        # This checks for the waves injected by Architect
        elysia._pulse_reality()
        
        # Check for Result
        # Since we don't know the exact filename key, we check for any 'manifestation_*' file logic
        # But in our mock implementation in living_elysia, filename was timestamp based.
        # Let's check for any new .txt file created recently in root
        
        # Actually, let's look at the output logs (via stdout capture is hard here, so we rely on script output)
        pass

    log("Observation Complete.")
    
    # Verification
    # Check if ANY manifestation file exists
    found = False
    for f in os.listdir(os.getcwd()):
        if f.startswith("manifestation_") and f.endswith(".txt"):
            log(f"✅ FOUND MANIFESTATION: {f}")
            content = open(f).read()
            log(f"   Content: {content.strip()}")
            found = True
            # Cleanup
            os.remove(f)
            
    if not found:
        log("❌ No manifestation found. Autonomy failed.")

if __name__ == "__main__":
    supervise()
