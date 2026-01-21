
import sys
import os
import shutil
import time
import numpy as np
import logging

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.L5_Mental.Memory.prismatic_sediment import PrismaticSediment

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def clean_data():
    if os.path.exists("data/Test_Prism"):
        shutil.rmtree("data/Test_Prism")

def test_spectral_binding():
    print("🌈 Starting Spectral Binding Test...")
    setup_logging()
    clean_data()

    sediment = PrismaticSediment("data/Test_Prism")

    # 1. Store a 'Red-Orange' Memory
    # Dominant is Red (0.9), but it has Orange relevance (0.7)
    # It will be stored in RED shard.
    vec_mixed = [0.9, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0]
    payload_mixed = b"Concept: Sunset (Red/Orange)"
    sediment.deposit(vec_mixed, time.time(), payload_mixed)
    print(f"✅ Deposited 'Sunset' (Red Dominant) into Sediment.")

    time.sleep(0.1) # Small delay for timestamp ordering

    # 2. Store a 'Violet' Memory
    vec_violet = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]
    payload_violet = b"Concept: Void (Violet)"
    sediment.deposit(vec_violet, time.time(), payload_violet)
    print(f"✅ Deposited 'Void' (Violet Dominant) into Sediment.")

    # TEST 1: Neighbor Scanning
    # We look for 'Orange' (Index 1).
    # The 'Sunset' memory is in Red (Index 0).
    # Since Red is a neighbor of Orange, we SHOULD find it.
    print("\n🧪 TEST 1: Neighbor Scanning (Looking for Orange, Memory is Red)")
    vec_query_orange = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    results = sediment.scan_resonance(vec_query_orange, top_k=1)

    if results and b"Sunset" in results[0][1]:
        print("   -> 🌟 SUCCESS: Found 'Sunset' in Red shard while looking for Orange!")
    else:
        print("   -> ❌ FAILURE: Did not find 'Sunset'. Spectral Boundary is still rigid.")

    # TEST 2: Amor Sui (Gravity)
    # We look for 'Green' (Index 3).
    # 'Void' is in Violet (Index 6). Green and Violet are NOT neighbors.
    # Initial scan (Green, Yellow, Blue) will fail (return empty or low score).
    # Amor Sui should trigger and find Violet.
    # To make the test deterministic, we add a TINY bit of Violet to the Green query.
    # If Amor Sui fails (doesn't look in Violet shard), score will be 0 or not found.
    # If Amor Sui works (looks in Violet shard), score will be > 0.
    print("\n🧪 TEST 2: Amor Sui / Gravity (Looking for Green, Memory is Violet)")
    # [Green=0.9, Violet=0.1]. Dominant is Green.
    vec_query_green = [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.1]

    # We set threshold high (0.8). The match with Violet will be low (~0.1),
    # but since Green/Yellow/Blue are empty, the initial result is [] or score=0.
    # This triggers Amor Sui.
    results_gravity = sediment.scan_resonance(vec_query_green, top_k=1, threshold=0.8)

    if results_gravity and b"Void" in results_gravity[0][1]:
        print("   -> 🌌 SUCCESS: Found 'Void' via Amor Sui (Gravity) expansion!")
    else:
        print(f"   -> ❌ FAILURE: Did not find 'Void'. Results: {results_gravity}")

    # TEST 3: Unified Rewind
    print("\n🧪 TEST 3: Unified Rewind (Chronological Thread)")
    history = sediment.unified_rewind(steps=5)

    # We expect Sunset then Void
    if len(history) >= 2:
        first = history[0][2]
        second = history[1][2]
        print(f"   -> Sequence: 1. {first} 2. {second}")

        if b"Sunset" in first and b"Void" in second:
            print("   -> 🧵 SUCCESS: History is chronologically stitched.")
        else:
            print("   -> ❌ FAILURE: Order is wrong.")
    else:
        print(f"   -> ❌ FAILURE: Not enough history returned ({len(history)}).")

    sediment.close()
    clean_data()

if __name__ == "__main__":
    test_spectral_binding()
