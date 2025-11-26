"""
Planning Cortex (The Bridge)
============================
Protocol-41: "Intent must become Action through Time"

This module acts as the "Concept Protocol OS" Kernel.
It bridges the gap between Elysia's abstract "Will" (from ResonanceEngine)
and concrete "Action" (Tool calls/Code execution).

Key Responsibilities:
1. Time Perception (The Anchor)
2. Intent Synthesis (Resonance -> Goal)
3. Plan Generation (Goal -> Steps)
4. Plan Execution (Steps -> Actions)
"""

import logging
import datetime
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from Core.Mind.hippocampus import Hippocampus
from Core.Ethics.conscience import Conscience

# Configure logger
logger = logging.getLogger("PlanningCortex")
logger.setLevel(logging.INFO)

@dataclass
class PlanStep:
    """A single step in a plan."""
    step_id: int
    action: str
    description: str
    estimated_duration: float  # in seconds
    required_tools: List[str]
    status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "description": self.description,
            "estimated_duration": self.estimated_duration,
            "required_tools": self.required_tools,
            "status": self.status
        }

@dataclass
class Plan:
    """A sequence of steps to achieve an intent."""
    intent: str
    created_at: datetime.datetime
    steps: List[PlanStep]
    status: str = "pending"  # pending, in_progress, completed, failed, cancelled
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "steps": [s.to_dict() for s in self.steps]
        }

class PlanningCortex:
    """
    The Prefrontal Cortex of Elysia.
    Responsible for breaking down high-level goals into executable plans.
    Integrates with Memory (Hippocampus) and Ethics (Conscience).
    """
    
    def __init__(self, hippocampus: Optional[Hippocampus] = None, conscience: Optional[Conscience] = None):
        self.hippocampus = hippocampus
        self.conscience = conscience
        self.active_plans: List[Plan] = []
        self.history: List[Plan] = []
        self.logger = logger
        self.logger.info("✅ Planning Cortex initialized (Concept Protocol OS Kernel)")

    def perceive_time(self) -> datetime.datetime:
        """
        Returns the current time.
        This is the fundamental anchor for all planning.
        """
        now = datetime.datetime.now()
        return now

    def synthesize_intent(self, resonance_pattern: Dict[str, float]) -> str:
        """
        Synthesizes a high-level intent from the resonance pattern.
        """
        if not resonance_pattern:
            return "Rest"

        # Find the concept with the highest resonance
        dominant_concept = max(resonance_pattern, key=resonance_pattern.get)
        intensity = resonance_pattern[dominant_concept]
        
        self.logger.info(f"🌊 Resonance Wave Analysis: Dominant={dominant_concept} (Intensity={intensity:.2f})")

        # Simple mapping logic (to be expanded with LLM/ConceptNet later)
        intent_map = {
            "Hunger": "Find Energy Source",
            "Energy": "Find Energy Source",
            "Curiosity": "Explore Unknown Area",
            "Experiment": "Explore Unknown Area",
            "Fear": "Seek Safety",
            "Pain": "Heal Self",
            "Social": "Communicate with Others",
            "Speak": "Communicate with Others",
            "Creation": "Build New Structure",
            "Rest": "Consolidate Memory",
            "사랑": "Express Affection", # Love
            "기쁨": "Share Joy", # Joy
            "빛": "Seek Enlightenment", # Light
        }
        
        return intent_map.get(dominant_concept, f"Focus on {dominant_concept}")

    def generate_plan(self, intent: str) -> Plan:
        """
        Decomposes a high-level intent into a structured plan.
        """
        self.logger.info(f"📝 Generating plan for intent: '{intent}'")
        
        steps = []
        
        # Rule-based planning (Placeholder for future LLM-based planning)
        if intent == "Find Energy Source":
            steps = [
                PlanStep(1, "scan_environment", "Scan for nearby resources", 5.0, ["vision"]),
                PlanStep(2, "move_to_target", "Move towards nearest energy source", 10.0, ["locomotion"]),
                PlanStep(3, "consume", "Consume resource", 2.0, ["metabolism"])
            ]
        elif intent == "Explore Unknown Area":
            steps = [
                PlanStep(1, "pick_random_direction", "Choose a random vector", 1.0, ["rng"]),
                PlanStep(2, "move", "Move in chosen direction", 15.0, ["locomotion"]),
                PlanStep(3, "record_observation", "Log new findings", 2.0, ["memory"])
            ]
        elif intent == "Communicate with Others":
             steps = [
                PlanStep(1, "scan_for_agents", "Look for nearby agents", 3.0, ["vision"]),
                PlanStep(2, "compose_message", "Compose a greeting or query", 2.0, ["language"]),
                PlanStep(3, "broadcast", "Send message", 1.0, ["speech"])
            ]
        elif intent == "Consolidate Memory":
             steps = [
                PlanStep(1, "enter_sleep_mode", "Reduce energy consumption", 1.0, ["metabolism"]),
                PlanStep(2, "replay_memories", "Process recent events", 10.0, ["hippocampus"]),
                PlanStep(3, "wake_up", "Return to active state", 1.0, ["metabolism"])
            ]
        else:
            # Generic fallback plan
            # Merging the heuristic logic from the old develop_plan
            heuristic_steps = self._heuristic_planner(intent)
            if heuristic_steps:
                for i, step_dict in enumerate(heuristic_steps):
                    steps.append(PlanStep(
                        step_id=i+1,
                        action=step_dict["tool"],
                        description=f"Execute {step_dict['tool']} with {step_dict['parameters']}",
                        estimated_duration=5.0,
                        required_tools=[step_dict["tool"]]
                    ))
            else:
                steps = [
                    PlanStep(1, "observe", f"Observe state related to {intent}", 2.0, ["perception"]),
                    PlanStep(2, "evaluate", "Evaluate best course of action", 1.0, ["reasoning"])
                ]
            
        plan = Plan(
            intent=intent,
            created_at=self.perceive_time(),
            steps=steps
        )
        
        self.active_plans.append(plan)
        return plan

    def execute_plan(self, plan: Plan) -> bool:
        """
        Executes the given plan (Simulation).
        """
        self.logger.info(f"🚀 Executing Plan: {plan.intent}")
        plan.status = "in_progress"
        
        try:
            for step in plan.steps:
                self.logger.info(f"  ▶ Step {step.step_id}: {step.action} - {step.description}")
                step.status = "in_progress"
                
                # Simulate execution time
                # time.sleep(step.estimated_duration * 0.1) 
                
                step.status = "completed"
                
            plan.status = "completed"
            self.history.append(plan)
            if plan in self.active_plans:
                self.active_plans.remove(plan)
            self.logger.info(f"✅ Plan Completed: {plan.intent}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Plan Failed: {e}")
            plan.status = "failed"
            self.history.append(plan)
            if plan in self.active_plans:
                self.active_plans.remove(plan)
            return False

    def develop_plan(self, goal: str) -> List[Dict[str, Any]]:
        """
        Legacy method for backward compatibility.
        Wraps generate_plan.
        """
        # 1. Ethical Check (if conscience is available)
        if self.conscience and not self.conscience.evaluate_action("planning", {"goal": goal}):
            self.logger.warning(f"⛔ Plan rejected by Conscience: {goal}")
            return []

        # 2. Generate Plan using new logic
        plan = self.generate_plan(goal)
        
        # 3. Convert to legacy format (List[Dict])
        legacy_plan = []
        for step in plan.steps:
            legacy_plan.append({
                "tool": step.action,
                "parameters": {"description": step.description}, # Simplified mapping
                "step_id": step.step_id
            })
            
        return legacy_plan

    def _heuristic_planner(self, goal: str) -> List[Dict[str, Any]]:
        """
        A simple heuristic planner for demonstration purposes.
        Preserved from legacy implementation.
        """
        plan = []
        goal_lower = goal.lower()
        
        if "write" in goal_lower and (".txt" in goal_lower or "file" in goal_lower):
            # Example: "Write a poem to test.txt"
            filename = "test_output.txt"
            # Try to extract filename from goal
            words = goal.split()
            for word in words:
                if ".txt" in word:
                    filename = word
                    
            plan.append({
                "tool": "write_to_file",
                "parameters": {
                    "filename": filename,
                    "content": "This is a test file generated by Elysia's Planning Cortex."
                }
            })
        elif "search" in goal_lower or "research" in goal_lower:
            # Example: "Research quantum physics"
            plan.append({
                "tool": "web_search",
                "parameters": {
                    "query": goal.replace("research", "").replace("search", "").strip()
                }
            })
            plan.append({
                "tool": "write_to_file",
                "parameters": {
                    "filename": "research_notes.txt",
                    "content": "Summary of research..."
                }
            })
            
        return plan

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the Cortex."""
        return {
            "current_time": self.perceive_time().isoformat(),
            "active_plans": [p.to_dict() for p in self.active_plans],
            "history_count": len(self.history)
        }
