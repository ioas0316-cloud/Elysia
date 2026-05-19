from Core.FoundationLayer.Foundation.planning_cortex import PlanningCortex
import os

print("--- Starting Test ---")
# Create dummy file
with open("dummy_code.py", "w") as f:
    f.write("# Dummy Code")

planner = PlanningCortex()
plan = planner.formulate_plan("Please improve your own code", {})
print(f"Plan generated: {plan}")

# Execute steps manually to avoid loop issues
print("Executing Step 1...")
print(planner.execute_next_step()) # scan_self

print("Executing Step 2...")
print(planner.execute_next_step()) # identify_improvement

print("Executing Step 3 (Optimization)...")
# We need to ensure execute_next_step calls _modify_file
# In the real class, execute_next_step calls _modify_file for 'apply_optimization'
# But it uses "Core/Elysia.py" hardcoded.
# We need to override that for the test, or just let it try to modify Core/Elysia.py and see if it fails.
# But I want to verify it works.
# I will monkeypatch _modify_file to write to dummy_code.py
original_modify = planner._modify_file
def mock_modify(filepath, content, mode="append"):
    print(f"Mock modifying {filepath} -> dummy_code.py")
    original_modify("dummy_code.py", content, mode)

planner._modify_file = mock_modify
print(planner.execute_next_step()) # apply_optimization

# Verify
with open("dummy_code.py", "r") as f:
    content = f.read()
    print("\nFile Content:")
    print(content)
print("--- End Test ---")
