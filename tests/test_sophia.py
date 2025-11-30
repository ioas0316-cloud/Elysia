from Project_Sophia.planning_cortex import PlanningCortex
import os

# Create dummy file
with open("dummy_code.py", "w") as f:
    f.write("# Dummy Code")

planner = PlanningCortex()
print("Plan:", planner.formulate_plan("Please improve your own code", {}))

# Execute steps
while planner.current_plan:
    step = planner.current_plan[0]
    if step == "apply_optimization":
        # Hack to redirect to dummy file for test
        planner._modify_file("dummy_code.py", "# Optimized by Sophia", mode="append")
        planner.current_plan.pop(0)
    else:
        print(planner.execute_next_step())

# Verify
with open("dummy_code.py", "r") as f:
    content = f.read()
    print("\nFile Content:")
    print(content)
