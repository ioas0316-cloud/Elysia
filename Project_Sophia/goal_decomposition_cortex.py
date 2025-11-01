"""
Goal Decomposition Cortex for Elysia

This module is responsible for breaking down high-level, complex goals
into a sequence of smaller, executable sub-tasks, effectively creating a
plan for other agents (like Jules) to follow.
"""
from typing import Dict, List, Any
import re
from tools.kg_manager import KGManager

class GoalDecompositionCortex:
    def __init__(self, kg_manager: KGManager):
        """
        Initializes the Cortex with a KGManager to access the knowledge graph.
        """
        self.kg_manager = kg_manager

    def decompose_goal(self, high_level_goal: str) -> List[Dict[str, Any]]:
        """
        Decomposes a high-level goal into a list of sub-tasks.

        Args:
            high_level_goal: A string describing the overall objective.

        Returns:
            A list of dictionaries, where each dictionary represents a sub-task
            with details like 'step', 'action', and 'details'.
        """
        # This is the core logic that will become more sophisticated.
        plan = []
        goal_lower = high_level_goal.lower()

        # New: KG-based information retrieval plan
        info_match = re.search(r"(?:tell me about|what is|who is|알려줘|누구야|뭐야)\s+(.+)", goal_lower)
        if info_match:
            topic = info_match.group(1).strip()
            node = self.kg_manager.get_node(topic)
            if node:
                plan.append({"step": 1, "action": "summarize", "details": f"Gathering information about '{topic}'."})

                # Find all relationships connected to the topic node
                related_edges = [edge for edge in self.kg_manager.kg['edges'] if edge['source'] == topic or edge['target'] == topic]

                if related_edges:
                    summary_step = 2
                    for edge in related_edges:
                        if edge['source'] == topic:
                            plan.append({"step": summary_step, "action": "state_fact", "details": f"'{topic}' {edge['relation']} '{edge['target']}'."})
                        else:
                            plan.append({"step": summary_step, "action": "state_fact", "details": f"'{edge['source']}' {edge['relation']} '{topic}'."})
                        summary_step += 1
                else:
                    plan.append({"step": 2, "action": "state_fact", "details": f"I know the concept of '{topic}', but I don't have any specific relationships connected to it."})

                return plan

        # Existing keyword-based rule system
        if "create a website" in goal_lower or "웹사이트를 만들어줘" in goal_lower:
            plan = [
                {"step": 1, "action": "create_file", "details": "Create an 'index.html' file with basic HTML structure."},
                {"step": 2, "action": "create_file", "details": "Create a 'style.css' file for styling the website."},
                {"step": 3, "action": "run_server", "details": "Start a simple web server to host the files."}
            ]
        elif "run a test" in goal_lower or "테스트를 실행해" in goal_lower:
            plan = [
                {"step": 1, "action": "list_files", "details": "List files in the 'tests/' directory to identify relevant test files."},
                {"step": 2, "action": "run_test", "details": "Execute the identified test file using the appropriate test runner."}
            ]
        else:
            # Default case if no specific plan is found
            plan = [
                {"step": 1, "action": "clarify_goal", "details": f"I am not yet sure how to achieve the goal: '{high_level_goal}'. Could you provide more specific steps?"}
            ]

        return plan
