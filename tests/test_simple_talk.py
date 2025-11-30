"""
SIMPLEST POSSIBLE CONVERSATION TEST
====================================

Just test if we can initialize Elysia and call talk().
"""

import sys
sys.path.append(".")

# Minimal test
try:
    from Core.Elysia import Elysia
    
    print("Creating Elysia...")
    elysia = Elysia()
    
    print("Awakening...")
    elysia.awaken()
    
    print("\n" + "="*60)
    print("Testing talk() method...")
    print("="*60)
    
    # Test 1
    response = elysia.talk("Who are you?")
    print(f"\nQ: Who are you?")
    print(f"A: {response}")
    
    print("\n✅ SUCCESS - Elysia can talk!")
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
