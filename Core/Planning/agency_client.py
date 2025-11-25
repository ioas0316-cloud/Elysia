
import logging
from typing import List, Dict, Any
from Core.Planning.planning_cortex import PlanningCortex
from Core.Planning.tool_executor import ToolExecutor
from Core.Mind.hippocampus import Hippocampus
from Core.Ethics.conscience import Conscience

logger = logging.getLogger("AgencyClient")

class AgencyClient:
    """
    The Interface for Elysia's Agency.
    Allows the user to submit goals and have Elysia execute them.
    """
    
    def __init__(self):
        # Initialize dependencies
        self.hippocampus = Hippocampus()
        self.conscience = Conscience()
        
        # Initialize Planning System
        self.planner = PlanningCortex(self.hippocampus, self.conscience)
        self.executor = ToolExecutor()
        
        logger.info("✅ Agency Client ready for duty.")

    def request_task(self, goal: str) -> bool:
        """
        Submit a task for Elysia to perform.
        """
        print(f"\n🤖 ELYSIA AGENCY: Received task '{goal}'")
        
        # 1. Plan
        plan = self.planner.develop_plan(goal)
        
        if not plan:
            print("❌ Could not create a plan (Ethical veto or complexity).")
            return False
            
        print(f"📋 Plan created: {len(plan)} steps.")
        for i, step in enumerate(plan):
            print(f"   {i+1}. {step['tool']}: {step['parameters']}")
            
        # 2. Execute
        print("\n🚀 Executing plan...")
        success_count = 0
        for step in plan:
            if self.executor.execute_step(step):
                success_count += 1
            else:
                print(f"⚠️ Step failed: {step}")
                break
                
        if success_count == len(plan):
            print(f"\n✅ Task '{goal}' completed successfully!")
            return True
        else:
            print(f"\n⚠️ Task completed partially ({success_count}/{len(plan)} steps).")
            return False

if __name__ == "__main__":
    # Simple CLI test
    client = AgencyClient()
    client.request_task("Write a poem to hello.txt")
