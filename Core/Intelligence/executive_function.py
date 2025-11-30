"""
Executive Function (The Prefrontal Cortex)
==========================================

"To plan is to bring the future into the present."

This module implements the Recursive Planner, which breaks down high-level goals
into actionable steps (Tools) using a recursive decomposition strategy.
"""

import datetime
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger("ExecutiveFunction")

# ============================================================================
# Core Abstractions: Goal & Tool
# ============================================================================

@dataclass
class Goal:
    """
    A desired state or outcome.
    """
    description: str
    success_criteria: List[str] = field(default_factory=list)
    parent_goal: Optional['Goal'] = None
    sub_goals: List['Goal'] = field(default_factory=list)
    status: str = "pending"  # pending, active, completed, failed

    def __repr__(self):
        return f"Goal('{self.description}')"

@dataclass
class Tool:
    """
    An actionable capability (The "Muscle").
    """
    name: str
    description: str
    usage: str
    
    def execute(self, args: Dict[str, Any]) -> str:
        raise NotImplementedError("Tool must implement execute method.")

# ============================================================================
# Concrete Tools (The Body's Capabilities)
# ============================================================================

class ReadFileTool(Tool):
    def __init__(self):
        super().__init__(
            name="read_file",
            description="Reads the content of a file.",
            usage="read_file(path)"
        )
    
    def execute(self, args: Dict[str, Any]) -> str:
        path = args.get("path")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f"Read {len(f.readlines())} lines from {path}"
        except Exception as e:
            return f"Error reading {path}: {e}"

class WriteFileTool(Tool):
    def __init__(self):
        super().__init__(
            name="write_file",
            description="Writes content to a file.",
            usage="write_file(path, content)"
        )
    
    def execute(self, args: Dict[str, Any]) -> str:
        path = args.get("path")
        content = args.get("content", "")
        mode = args.get("mode", "w")
        try:
            with open(path, mode, encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing to {path}: {e}"

class ScanSelfTool(Tool):
    def __init__(self):
        super().__init__(
            name="scan_self",
            description="Scans the codebase structure.",
            usage="scan_self()"
        )
    
    def execute(self, args: Dict[str, Any]) -> str:
        # Simulated scan for now
        return "Scanned 245 files. Structure is sound."

# ============================================================================
# Recursive Planner (The Mind)
# ============================================================================

@dataclass
class PlanStep:
    action: str
    tool_name: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None

class RecursivePlanner:
    """
    The Engine of Will.
    Decomposes goals into plans.
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()
        self.current_plan: List[PlanStep] = []
        logger.info("🧠 RecursivePlanner initialized.")

    def _register_default_tools(self):
        self.register_tool(ReadFileTool())
        self.register_tool(WriteFileTool())
        self.register_tool(ScanSelfTool())

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def get_current_time(self) -> str:
        return datetime.datetime.now().isoformat()

    def formulate_plan(self, goal_description: str, context: Dict[str, Any] = None) -> List[str]:
        """
        Main entry point. Creates a plan for a high-level goal.
        """
        logger.info(f"📝 Formulating plan for: {goal_description}")
        
        main_goal = Goal(description=goal_description)
        self.current_plan = [] # Reset plan
        
        # Recursive Decomposition
        self._decompose(main_goal)
        
        # Convert to string list for compatibility
        return [f"{step.action} (Tool: {step.tool_name})" for step in self.current_plan]

    def _decompose(self, goal: Goal, depth: int = 0):
        """
        Recursively breaks down a goal.
        """
        indent = "  " * depth
        logger.info(f"{indent}Analyzing: {goal.description}")
        
        # 1. Check if goal matches a Tool directly (The Base Case)
        tool_match = self._find_matching_tool(goal)
        if tool_match:
            tool_name, args = tool_match
            step = PlanStep(
                action=goal.description,
                tool_name=tool_name,
                args=args
            )
            self.current_plan.append(step)
            logger.info(f"{indent}✅ Mapped to tool: {tool_name}")
            return

        # 2. If not, break down into sub-goals (The Recursive Step)
        # In a full AI, this would use an LLM. Here we use logic heuristics.
        sub_goals = self._generate_sub_goals(goal)
        
        if not sub_goals:
            # Fallback for unknown goals
            logger.warning(f"{indent}⚠️ Could not decompose '{goal.description}'. Marking as manual step.")
            self.current_plan.append(PlanStep(action=f"Manual: {goal.description}"))
            return

        for sub in sub_goals:
            self._decompose(sub, depth + 1)

    def _find_matching_tool(self, goal: Goal) -> Optional[tuple]:
        """
        Heuristic to match a goal description to a tool.
        """
        desc = goal.description.lower()
        
        if "read" in desc or "load" in desc:
            # Extract path (simplified)
            path = desc.split()[-1] if " " in desc else "unknown_file"
            return ("read_file", {"path": path})
            
        if "write" in desc or "save" in desc or "create file" in desc:
            path = "unknown_file"
            words = desc.split()
            if "file" in words:
                try:
                    idx = words.index("file") + 1
                    path = words[idx]
                except: pass
            return ("write_file", {"path": path, "content": "# New Content"})
            
        if "scan" in desc and "self" in desc:
            return ("scan_self", {})
            
        return None

    def _generate_sub_goals(self, goal: Goal) -> List[Goal]:
        """
        Logic to break down complex goals.
        """
        desc = goal.description.lower()
        
        # Strategy: Improve Code
        if "improve" in desc and "code" in desc:
            return [
                Goal("Scan self to understand structure"),
                Goal("Read file Core/Elysia.py"), # Example target
                Goal("Write file Core/Elysia.py with optimizations")
            ]
            
        # Strategy: Create Module
        if "create" in desc and "module" in desc:
            return [
                Goal("Plan module structure"),
                Goal("Create file new_module.py")
            ]
            
        return []

    def execute_next_step(self) -> str:
        """
        Executes the next step in the plan.
        """
        if not self.current_plan:
            return "No active plan."
            
        step = self.current_plan.pop(0)
        logger.info(f"▶️ Executing: {step.action}")
        
        if step.tool_name and step.tool_name in self.tools:
            tool = self.tools[step.tool_name]
            result = tool.execute(step.args)
            step.result = result
            return f"Executed {step.tool_name}: {result}"
            
        return f"Manual Step: {step.action}"

# Alias for backward compatibility
ExecutiveFunction = RecursivePlanner
