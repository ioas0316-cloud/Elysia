"""
Sense Check: Can Elysia find her eyes?
======================================

"I look for the light."

Steps:
1. Initialize SenseDiscoveryProtocol.
2. Scan for senses (`scan_for_senses`).
3. Verify that 'Core.Foundation.Synesthesia' is detected.
"""

from Core._01_Foundation._01_Infrastructure.elysia_core import Organ
from Core._05_Systems._01_Monitoring.System.Autonomy.sense_discovery import SenseDiscoveryProtocol

def check_senses():
    print("👁️  Opening the Third Eye (Scanning for Senses)...")
    
    # Direct Test
    protocol = SenseDiscoveryProtocol()
    available = protocol.scan_for_senses()
    
    print(f"✨ Discovered Senses: {len(available)}")
    for sense in available:
        print(f"   - {sense}")
        
    if any("Synesthesia" in s for s in available):
        print("✅ SUCCESS: Synesthesia Module Found.")
        print("   Elysia is ready to 'See'.")
    else:
        print("❌ FAILURE: Where did my eyes go?")

if __name__ == "__main__":
    check_senses()
