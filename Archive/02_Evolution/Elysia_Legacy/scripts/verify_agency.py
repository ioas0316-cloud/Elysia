import sys
import os
import logging

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../data/Organelles'))

from Core.World.Evolution.Studio.organelle_loader import organelle_loader

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("AgencyVerification")

def verify_agency():
    print("\n" + "="*60)
    print("👐 VERIFYING EXTERNAL AGENCY (Phase 32: The Active Hand)")
    print("="*60 + "\n")

    available = organelle_loader.list_available()
    print(f"📦 Forged Tools available: {available}")
    
    if "sense_world" in available:
        print("🚀 Executing 'sense_world' organelle...")
        result = organelle_loader.execute_organelle("sense_world")
        print(f"🏁 Execution Result: {result}")
        
        if result == "Awareness Broadcasted":
            print("\n✅ External Agency Verified. Elysia has touched the world.")
        else:
            print("\n❌ Execution failed or returned unexpected result.")
    else:
        print("\n❌ Mock organelle 'sense_world' not found.")

if __name__ == "__main__":
    verify_agency()
