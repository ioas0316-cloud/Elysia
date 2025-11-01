import re

class PlanningCortex:
    """
    Breaks down complex, high-level goals into a sequence of executable tool calls.
    This cortex is the heart of Elysia's ability to form and execute multi-step plans.
    """

    def __init__(self, core_memory, action_cortex):
        """
        Initializes the Planning Cortex.

        Args:
            core_memory: An interface to Elysia's core memory system.
            action_cortex: The action cortex to decide on individual tool calls.
        """
        self.core_memory = core_memory
        self.action_cortex = action_cortex

    def develop_plan(self, goal):
        """
        Develops a step-by-step plan to achieve a given goal.

        Args:
            goal (str): The high-level goal to achieve.

        Returns:
            list: A list of tool calls or sub-goals representing the plan.
        """
        print(f"Developing plan for goal: {goal}")
        
        # Simple goal decomposition using keywords
        steps = self._decompose_goal(goal)
        
        plan = []
        for step in steps:
            # Use action_cortex to find the tool for each step
            action = self.action_cortex.decide_action(step)
            if action:
                plan.append(action)
                
        return plan

    def _decompose_goal(self, goal):
        """
        Decomposes a goal into smaller steps.
        This is a very simple implementation and can be improved.
        """
        # Use "and then" as a simple delimiter for steps
        steps = [s.strip().replace("read the file", "readfile") for s in re.split(r'and then', goal, flags=re.IGNORECASE)]
        return steps