
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Core.Mind.hippocampus import Hippocampus

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifySynesthesia")

def verify_synesthesia():
    print("\n" + "="*70)
    print("🌈 Verifying Synesthetic Integration...")
    print("="*70)

    try:
        # 1. Initialize Hippocampus
        print("1. Initializing Hippocampus...")
        hippocampus = Hippocampus()
        
        # 2. Add a test concept with strong sensory potential
        test_concept = "Ocean"
        print(f"2. Adding concept: '{test_concept}'...")
        hippocampus.add_concept(test_concept)
        
        # 3. Retrieve and Check
        print("3. Retrieving concept data...")
        # We need to access the sphere directly from the universe or storage
        # Let's try getting it from the Universe first (Working Memory)
        sphere = hippocampus.universe.spheres.get(test_concept)
        
        if not sphere:
            # Fallback to storage if not in universe (unlikely if just added)
            print("   (Not in Universe, checking Storage...)")
            data = hippocampus.storage.get_concept(test_concept)
            # We'd need to reconstruct it, but let's assume Universe works for this test
            if not data:
                print("❌ Concept not found anywhere!")
                return False

        # 4. Verify Sensory Signature
        print(f"4. Checking Sensory Signature for '{test_concept}'...")
        signature = sphere.sensory_signature
        
        if signature:
            print(f"   ✅ Signature Found: {signature}")
            print(f"      - Color: {signature.get('color')}")
            print(f"      - Pitch: {signature.get('pitch')} Hz")
            print(f"      - Emotion: {signature.get('emotion')}")
            return True
        else:
            print("   ❌ No Sensory Signature found!")
            return False

    except Exception as e:
        print(f"❌ Crash during verification: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if verify_synesthesia():
        print("\n✅ Verification PASSED: Synesthesia is active.")
        sys.exit(0)
    else:
        print("\n❌ Verification FAILED.")
        sys.exit(1)
