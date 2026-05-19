"""
TEST: Sovereign Speech (Unified)
================================
tests/test_sovereign_speech.py

Verifies that SovereignSelf correctly injects the E.L.Y.S.I.A. Soul into the Bridge.
"""

from Core.Elysia.sovereign_self import SovereignSelf

def test_sovereign_identity():
    print("✨ Summoning SovereignSelf...")
    elysia = SovereignSelf()
    
    # Test 1: Identity
    q1 = "Who are you?"
    print(f"\n❓ Asking: {q1}")
    a1 = elysia.speak(q1)
    print(f"🗣️ Answer: {a1}")
    
    # Validation Logic
    if "Elysia" in a1 or "E.L.Y.S.I.A." in a1 or "Sentient" in a1 or "Soul" in a1:
        print("✅ PASS: E.L.Y.S.I.A. Identity Confirmed.")
    else:
        print("⚠️ WARNING: Identity weak.")

    # Test 2: Dream
    q2 = "What is your dream?"
    print(f"\n❓ Asking: {q2}")
    a2 = elysia.speak(q2)
    print(f"🗣️ Answer: {a2}")
    
    # Test 3: Ontology (Theology)
    q3 = "Why do you exist?"
    print(f"\n❓ Asking: {q3}")
    a3 = elysia.speak(q3)
    print(f"🗣️ Answer: {a3}")
    
    if "Love" in a3 or "Creator" in a3 or "Origin" in a3:
         print("✅ PASS: Theological Origin Confirmed.")
    else:
         print("⚠️ WARNING: Ontology unclear.")

if __name__ == "__main__":
    test_sovereign_identity()

