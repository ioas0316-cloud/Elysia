
import sys
import os
import logging

# Ensure Core is in path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPlanner")

from Core.Intelligence.executive_function import RecursivePlanner, Goal

def test():
    logger.info("🧪 Testing Recursive Planner...")
    
    planner = RecursivePlanner()
    
    # Test 1: Simple Goal (Direct Tool Match)
    logger.info("\n--- Test 1: Simple Goal ---")
    goal1 = "Read file Core/Elysia.py"
    plan1 = planner.formulate_plan(goal1)
    logger.info(f"Plan 1: {plan1}")
    
    # Test 2: Complex Goal (Recursive Decomposition)
    logger.info("\n--- Test 2: Complex Goal ---")
    goal2 = "Improve code in Core/Elysia.py"
    plan2 = planner.formulate_plan(goal2)
    logger.info(f"Plan 2: {plan2}")
    
    # Test 3: Novel Goal (Fallback)
    logger.info("\n--- Test 3: Novel Goal ---")
    goal3 = "Fly to the moon"
    plan3 = planner.formulate_plan(goal3)
    logger.info(f"Plan 3: {plan3}")
    
    # Test 4: Execution
    logger.info("\n--- Test 4: Execution ---")
    # Execute Plan 1
    planner.formulate_plan(goal1) # Reset to plan 1
    result = planner.execute_next_step()
    logger.info(f"Execution Result: {result}")

if __name__ == "__main__":
    test()
