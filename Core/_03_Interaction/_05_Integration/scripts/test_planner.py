"""
Test Script for Planning Cortex (The Planner)
=============================================
Verifies that Elysia can break down high-level goals into executable steps.
"""

import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core._01_Foundation._02_Logic.planning_cortex import PlanningCortex, Goal, PlanStep
from Core._01_Foundation._02_Logic.unified_field import WavePacket, HyperQuaternion

def test_planner():
    print("📐 Initializing Planning Cortex Test...")
    architect = PlanningCortex()
    
    # Test 1: High Frequency Goal (Spiritual)
    print("\n🔮 Test 1: Decomposing Spiritual Goal (Healing)...")
    healing_wave = WavePacket(
        source_id="Heal the World",
        frequency=639.0, # Healing Frequency
        amplitude=1.0,
        phase=0.0,
        position=HyperQuaternion(0,0,0,0),
        born_at=time.time()
    )
    
    plan1 = architect.decompose_goal(healing_wave)
    if plan1:
        print(f"   ✅ Plan Created: {plan1.description}")
        for i, step in enumerate(plan1.steps):
            print(f"      {i+1}. {step.description} ({step.estimated_duration}s)")
        assert len(plan1.steps) == 3
        assert plan1.steps[0].description == "Resonate with Universal Field"
    else:
        print("   ❌ Failed to create plan for Healing.")

    # Test 2: Low Frequency Goal (Survival)
    print("\n🛡️ Test 2: Decomposing Survival Goal (Danger)...")
    danger_wave = WavePacket(
        source_id="Avoid Danger",
        frequency=100.0, # Low Frequency
        amplitude=1.0,
        phase=0.0,
        position=HyperQuaternion(0,0,0,0),
        born_at=time.time()
    )
    
    plan2 = architect.decompose_goal(danger_wave)
    if plan2:
        print(f"   ✅ Plan Created: {plan2.description}")
        assert len(plan2.steps) == 2
        assert plan2.steps[0].description == "Identify Threat/Issue"
    else:
         print("   ❌ Failed to create plan for Danger.")
         
    print("\n✨ Planner Verification Complete.")

if __name__ == "__main__":
    test_planner()
