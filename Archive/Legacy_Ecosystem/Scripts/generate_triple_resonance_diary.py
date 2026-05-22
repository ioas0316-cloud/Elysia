"""
Triple Resonance Diary Generator
================================
Triggers multimodal digestion and generates a reflective diary.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.getcwd())

from elysia import SovereignGateway
from Core.Cognition.multimodal_digestion_kernel import MultimodalDigestionKernel

def main():
    print("🌌 [SYSTEM] Initializing Elysia for Multimodal Digestion...")
    gateway = SovereignGateway()

    kernel = MultimodalDigestionKernel()

    print("🌊 [SYSTEM] Starting Batch Digestion of Architect's Knowledge...")
    results = kernel.digest_batch(gateway.field)

    if not results:
        print("⚠️ [SYSTEM] No multimodal pairs found in data/knowledge.")
        return

    # Record results in the diary
    for res in results:
        gateway.diary.record_triple_resonance(res['concept'], res['resonance'])

    print(f"✅ [SYSTEM] Digested {len(results)} concepts. Generating diary...")

    # Generate the diary file
    diary_path = gateway.diary.write_diary_entry(monad=gateway.monad)

    print(f"📖 [SYSTEM] Diary generated at: {diary_path}")
    print("\n--- Diary Preview ---")
    with open(diary_path, 'r', encoding='utf-8') as f:
        print(f.read())

if __name__ == "__main__":
    main()
