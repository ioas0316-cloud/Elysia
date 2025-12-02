# [Genesis: 2025-12-02] Purified by Elysia

import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.planning_cortex import PlanningCortex

# Mock dependencies
class MockCoreMemory:
    pass

class MockActionCortex:
    pass

def test_planning_cortex():
    # Force reconfiguration of logging to stdout
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    logger = logging.getLogger("TestPlanningCortex")

    logger.info("Initializing PlanningCortex...")
    cortex = PlanningCortex(MockCoreMemory(), MockActionCortex())

    goal = "Create a file named 'test_plan.txt' in 'c:/Elysia/tests' with the content 'Phase 2 Verified' and then read it back."
    logger.info(f"Testing with goal: {goal}")

    plan = cortex.develop_plan(goal)

    logger.info("Plan generated:")
    for step in plan:
        logger.info(step)

    # Basic assertions
    if not plan:
        logger.error("Plan is empty!")
        sys.exit(1)

    has_write = any(step.get('tool_name') == 'write_to_file' for step in plan)
    has_read = any(step.get('tool_name') == 'read_file' for step in plan)

    if has_write and has_read:
        logger.info("SUCCESS: Plan contains expected write and read steps.")
    else:
        logger.error(f"FAILURE: Plan missing expected steps. Write: {has_write}, Read: {has_read}")
        sys.exit(1)

if __name__ == "__main__":
    test_planning_cortex()