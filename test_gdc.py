import os
import sys

# Add the project root to the Python path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.goal_decomposition_cortex import GoalDecompositionCortex

def run_test():
    """
    Directly invokes the GoalDecompositionCortex to isolate and test its functionality.
    """
    print("--- Starting Direct Cortex Invocation Test ---")

    cortex = GoalDecompositionCortex()

    goal = "웹에서 '인공지능의 역사'에 대한 정보를 찾아줘"
    print(f"Testing with goal: \"{goal}\"")

    plan = cortex.decompose_goal(goal)

    print("\n--- Test Results ---")
    if plan:
        print("Successfully generated plan:")
        for i, step in enumerate(plan):
            print(f"  Step {i+1}: Tool='{step.get('tool_name')}', Parameters={step.get('parameters')}")
    else:
        print("Failed to generate a plan.")

    print("--- Test Complete ---")

if __name__ == "__main__":
    run_test()
