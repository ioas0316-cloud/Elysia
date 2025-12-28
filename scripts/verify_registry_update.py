
"""
Verify Registry Update Script
=============================

Verifies that SystemRegistry now correctly infers Fractal Identity
for legacy modules.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, r"c:\Elysia")

from Core.01_Foundation.05_Foundation_Base.Foundation.system_registry import get_system_registry

def verify_registry():
    print("🔍 Scanning Registry for Fractal Identities...")
    registry = get_system_registry()
    registry.scan_all_systems()
    
    # Check a few key systems
    check_list = [
        "fractal_soul", # New
        "nova_entity",  # Legacy Logic
        "slime_mind",   # Legacy Chaos
        "audio_cortex"  # Legacy Sensory
    ]
    
    for name in check_list:
        entry = registry.get_system(name)
        if entry:
            print(f"\n✅ System: {name}")
            print(f"   Category: {entry.category}")
            print(f"   Fractal Identity: {entry.fractal_identity}")
            print(f"   Resonance Freq: {entry.resonance_frequency}")
        else:
            print(f"\n❌ System: {name} NOT FOUND")

if __name__ == "__main__":
    verify_registry()
