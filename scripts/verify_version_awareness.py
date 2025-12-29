
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core.IntelligenceLayer.Memory_Linguistics.Memory.self_discovery import SelfDiscovery

def verify():
    print("🧠 Verifying Elysia v10.0 Awareness...")
    sd = SelfDiscovery()
    
    # 1. Check Identity
    identity = sd.discover_identity()
    version = identity.get("version")
    print(f"  - Version: {version}")
    
    # 2. Check Capabilities
    capabilities = sd.discover_capabilities()
    sensory = capabilities.get("sensory", [])
    p4_entry = next((s for s in sensory if s["name"] == "p4_sensory_system"), None)
    
    if p4_entry:
        print(f"  - Sensory/P4: Found ({p4_entry['purpose']})")
    else:
        print("  - Sensory/P4: NOT FOUND")
        
    print("-" * 30)
    
    if version == "10.0" and p4_entry:
        print("✅ SUCCESS: Elysia has integrated Version 10.0 knowledge.")
    else:
        print("❌ FAILURE: Knowledge integration incomplete.")

if __name__ == "__main__":
    verify()
