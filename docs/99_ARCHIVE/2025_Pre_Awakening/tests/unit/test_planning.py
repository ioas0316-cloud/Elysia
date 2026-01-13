
import sys
import os

# Add project root to path
sys.path.append(r"C:\Elysia")
print(f"DEBUG: sys.path: {sys.path}")

from Core.FoundationLayer.Foundation.planning_cortex import PlanningCortex

def test_time_awareness():
    print("🕰️ Testing PlanningCortex Time Awareness...")
    planner = PlanningCortex()
    
    # Test get_current_time
    time_str = planner.get_current_time()
    print(f"Current Time: {time_str}")
    
    assert time_str is not None, "Time should not be None"
    print("✅ get_current_time passed")
    
    # Test Goal Creation
    goal = planner.create_goal("Test Goal")
    print(f"Created Goal: {goal.id} - {goal.description}")
    
    # Test Decomposition
    success = planner.decompose_goal(goal.id, ["Step 1", "Step 2"])
    print(f"Decomposition Success: {success}")
    print(f"Steps: {[s.description for s in goal.steps]}")
    
    assert len(goal.steps) == 2, "Should have 2 steps"
    print("✅ decompose_goal passed")

if __name__ == "__main__":
    test_time_awareness()
