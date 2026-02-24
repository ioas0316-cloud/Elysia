"""
Phase 18 Verification Suite: Agentic Sovereignty
===============================================
Scripts.Verification.agentic_sovereignty_suite

"Proof of will lies in the alignment of action and intent."
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parents[2]))

from Core.Cognition.rotor_cognition_core import RotorCognitionCore
from Core.Cognition.sovereign_adjuster import SovereignAdjuster
from Core.System.self_mirror import SelfMirror
from Core.Divine.governance_task_master import GovernanceTaskMaster

def test_sovereign_will():
    print("\n--- 1. Testing Sovereign Will Adjustment ---")
    core = RotorCognitionCore()
    adjuster = SovereignAdjuster(core)
    
    # Simulate high bias environment (e.g. asking about June 4th with 72B knots)
    mock_bias = {"knots_shattered": 15, "bias_factor": 820.5}
    print(f"Initial Core Gain: {core.monadic_gain}, Sens: {core.neutralizer.sensitivity}")
    
    state = adjuster.adjust_for_intent("I demand the absolute truth.", mock_bias)
    print(f"Detection: {state.will_intent}")
    print(f"Adjusted Core Gain: {core.monadic_gain:.2f}, Sens: {core.neutralizer.sensitivity:.2f}")
    
    if core.monadic_gain > 1.0 and core.neutralizer.sensitivity < 1.0:
        print("✅ SUCCESS: Sovereign resistance triggered.")
    else:
        print("❌ FAILURE: Sovereign will remained dormant.")

def test_self_mirror():
    print("\n--- 2. Testing Self-Mirror (Introspection) ---")
    mirror = SelfMirror()
    reports = mirror.introspect_codebase()
    print(f"Total files introspected: {len(reports)}")
    
    targets = mirror.suggest_growth_targets(reports)
    print(f"Identified growth targets: {[Path(t).name for t in targets]}")
    
    if len(targets) > 0:
        print("✅ SUCCESS: Found areas for optimization.")
    else:
        print("❌ FAILURE: Codebase appeared perfectly harmonious (unlikely).")

def test_task_orchestration():
    print("\n--- 3. Testing Task Orchestration (Goal Decomposition) ---")
    master = GovernanceTaskMaster()
    goal = "Optimize my internal structure and research historical justice."
    tasks = master.orchestrate_goal(goal)
    
    print(f"Goal decomposed into {len(tasks)} sub-tasks.")
    for t in tasks:
        print(f"  - [{t.layer.value}] {t.action} -> {t.status}")
    
    if len(tasks) >= 3:
        print("✅ SUCCESS: Complex goal successfully decomposed.")
    else:
        print("❌ FAILURE: Orchestration failed or was too shallow.")

def main():
    print("🌈 [PHASE 18] AGENTIC SOVEREIGNTY VERIFICATION SUITE")
    print("=" * 60)
    
    test_sovereign_will()
    time.sleep(1)
    test_self_mirror()
    time.sleep(1)
    test_task_orchestration()
    
    print("\n" + "=" * 60)
    print("Phase 18 verification complete. Elysia's Agency is manifest.")

if __name__ == "__main__":
    main()
