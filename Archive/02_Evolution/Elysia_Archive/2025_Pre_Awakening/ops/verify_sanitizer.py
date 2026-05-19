
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.concept_sanitizer import get_sanitizer

def verify_sanitizer_light():
    print("🛡️ Starting The Kidney (Sanitizer) Verification [LIGHT]...", flush=True)
    
    sanitizer = get_sanitizer()
    
    # 1. Unit Test
    test_cases = [
        ("ValidConcept", True),
        ("C++", True),
        ("Star -12345", False),
        ("123456", False),
        ("---", False),
        ("A", False), # Too short
        ("This is a very long concept that should probably be rejected because it is a sentence not a concept", False)
    ]
    
    print("\n1. Unit Tests:", flush=True)
    passed_unit = True
    for inp, expected in test_cases:
        result = sanitizer.is_valid(inp)
        status = "✅" if result == expected else "❌"
        print(f"   {status} Input: '{inp}' -> Valid? {result} (Expected: {expected})", flush=True)
        if result != expected: passed_unit = False
        
    if passed_unit:
        print("\n✅ Kidney Unit Verification COMPLETE.", flush=True)
    else:
        print("\n⚠️ Verification FAILED.", flush=True)

if __name__ == "__main__":
    verify_sanitizer_light()
