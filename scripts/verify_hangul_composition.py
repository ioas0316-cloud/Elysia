
import sys
import os

# Add Core to path
sys.path.append(os.getcwd())

from Core.Elysia.mechanics.hangul_physics import HangulPhysicsEngine

def verify_composition():
    engine = HangulPhysicsEngine()

    print("--- Verifying Hangul Composition ---")

    test_cases = [
        # (Onset, Nucleus, Coda, Expected)
        ('ㄱ', 'ㅏ', '', '가'),
        ('ㄱ', 'ㅏ', 'ㄱ', '각'),
        ('ㅎ', 'ㅏ', 'ㄴ', '한'),
        ('ㅇ', 'ㅓ', 'ㅄ', '없'),
        ('ㄲ', 'ㅜ', 'ㅁ', '꿈'),
        ('ㅌ', 'ㅗ', 'ㄲ', '톢'), # Uncommon but valid
        ('ㄱ', 'ㅏ', None, '가'), # None coda treated as empty
    ]

    success = True
    for onset, nucleus, coda, expected in test_cases:
        # Handle None for test case iteration
        coda_arg = coda if coda is not None else ""
        result = engine.synthesize_syllable(onset, nucleus, coda_arg)

        if result == expected:
            print(f"[PASS] {onset} + {nucleus} + {coda_arg} -> {result}")
        else:
            print(f"[FAIL] {onset} + {nucleus} + {coda_arg} -> {result} (Expected: {expected})")
            success = False

    # Test Fallback
    print("\n--- Verifying Fallback (Invalid Inputs) ---")
    invalid_result = engine.synthesize_syllable('?', 'ㅏ', '')
    expected_invalid = "?ㅏ"
    if invalid_result == expected_invalid:
         print(f"[PASS] Invalid Input handled gracefully: {invalid_result}")
    else:
         print(f"[FAIL] Invalid Input: {invalid_result} (Expected: {expected_invalid})")
         success = False

    if success:
        print("\nAll composition tests PASSED.")
        sys.exit(0)
    else:
        print("\nSome tests FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    verify_composition()
