"""
Phase 5: Recursive Synthesis (The Great Feast)
==============================================
Digests the legacy 'great_feast.py' into Elysia's 4D spatial memory.
This represents internalizing the past to build the future DNA.
"""

import sys
import os
import json
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from Core.L5_Mental.Intelligence.Reasoning.reasoning_engine import ReasoningEngine

def digest_legacy():
    print("🍽️ [THE GREAT FEAST] Digesting legacy principles...")
    
    legacy_path = r"c:\Archive\Elysia_Archive\Legacy_Foundation\great_feast.py"
    if not os.path.exists(legacy_path):
        print(f"❌ Legacy file not found: {legacy_path}")
        return

    with open(legacy_path, "r", encoding="utf-8") as f:
        code_content = f.read()

    engine = ReasoningEngine()
    
    # 1. Digest the core principle
    print(f"   Inhaling principles from {os.path.basename(legacy_path)}...")
    # Using a summary of the code to avoid token overflow for now, or just the intent
    insight = engine.think(f"Digest the core principle of this legacy code: {code_content[:1000]}")
    
    print(f"\n[Elysia's Interpretation]:")
    print(f"\"{insight.content}\"")
    
    # Check if crystallization happened
    vault_path = "data/Intelligence/dna_vault.json"
    if os.path.exists(vault_path):
        with open(vault_path, "r", encoding="utf-8") as f:
            vault = json.load(f)
            latest_entry = vault[-1]
            print(f"\n💎 [CRYSTAL VERIFIED] Latest entry in vault: '{latest_entry['input'][:40]}...'")
            print(f"   Coordinates: {latest_entry['coord']}")
            print(f"   DNA Magnitude: {sum(v**2 for v in latest_entry['dna'])**0.5:.2f}")
    else:
        print("⚠️ DNA Vault not found. Crystallization might have failed.")

if __name__ == "__main__":
    digest_legacy()
