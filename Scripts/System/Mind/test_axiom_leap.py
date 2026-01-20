"""
[TEST] The Axiomatic Leap Verification
=====================================
Runs a comparative study between Babbling (Infant) and Reasoning (Genius).
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from Scripts.System.Body.motor_babbling import run_babbling_session

print("🧪 [EXPERIMENT] Phase 9: The Value of Knowledge")
print("==============================================")

print("\n👶 [CASE 1] Infant Mode (Trial & Error)")
run_babbling_session("HELLO", use_logic=False)

print("\n🧠 [CASE 2] Logic Mode (Axiomatic Reasoning)")
run_babbling_session("HELLO", use_logic=True)
