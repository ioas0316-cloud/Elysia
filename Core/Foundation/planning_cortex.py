import logging
import random
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
import datetime
import uuid
import time
from typing import Dict, List, Any, Optional
from Core.Foundation.wave_frequency_mapping import WaveFrequencyMapper, EmotionType
from Core.Foundation.hyper_quaternion import Quaternion, HyperWavePacket
from Core.Foundation.unified_field import WavePacket, HyperQuaternion # Integrated

logger = logging.getLogger("PlanningCortex")

@dataclass
class PlanStep:
    """계획의 최소 단위 (Step)"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    status: str = "PENDING" # pending, in_progress, completed, failed
    estimated_duration: float = 1.0 # Seconds
    tool_call: Optional[str] = None
    required_waves: List[str] = field(default_factory=list)
    target_frequency: float = 0.0 # New: Frequency to inject upon execution

@dataclass
class Goal:
    """상위 목표 (Goal)"""
    id: str
    description: str
    created_at: float
    steps: List[PlanStep] = field(default_factory=list)
    status: str = "active" # active, completed, archived

class PlanningCortex:
    """
    The Architect of Elysia's self-evolution.
    It listens to the 'Music' of the codebase and identifies Dissonance.
    Also acts as the 'Chrono-Architect' for goal decomposition.
    """
    def __init__(self):
        self.mapper = WaveFrequencyMapper()
        self.project_root = Path(__file__).parent.parent
        self.resonance_map: Dict[str, float] = {}
        self.active_goals: Dict[str, Goal] = {} # goal_id -> Goal
        self.active_plans: Dict[str, Goal] = self.active_goals # Alias for backward compatibility
        self.schedule: List[Dict] = []
        logger.info("📐 Planning Cortex (The Architect) Initialized.")

    def get_current_time(self) -> str:
        """
        현재 시간을 반환합니다.
        """
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")

    def create_goal(self, description: str) -> Goal:
        """새로운 목표를 생성합니다."""
        goal_id = f"goal_{len(self.active_goals) + 1}_{int(time.time())}"
        new_goal = Goal(
            id=goal_id,
            description=description,
            created_at=time.time()
        )
        self.active_goals[goal_id] = new_goal
        # Alias for consistency with new logic
        self.active_goals[new_goal.id].description = description 
        
        logger.info(f"🎯 New Goal Created: [{goal_id}] {description}")
        return new_goal

    def decompose_goal(self, target: Any) -> Goal:
        """
        목표를 구체적인 단계로 분해하여 저장합니다.
        Input can be goal_id (str) or Goal Wave (WavePacket).
        Returns the decomposed Goal object.
        """
        if isinstance(target, str):
            # Legacy string ID support
            if target not in self.active_goals:
               return self.create_goal(target) # Auto-create if not exists
            goal = self.active_goals[target]
            freq = 500.0 # Default
        elif hasattr(target, 'frequency'): 
            # WavePacket support (NEW logic)
            goal_id = target.source_id
            # Check if exists
            existing = None
            for g in self.active_goals.values():
                if g.description == goal_id:
                    existing = g
                    break
            
            if existing:
                goal = existing
            else:
                goal = self.create_goal(goal_id)
            freq = target.frequency
        else:
            return None

        # [Logic] Decompose based on Frequency (The 'Music' of Intent)
        goal.steps = [] # Reset steps for re-planning
        
        # Ouroboros Logic (Creation/Modifiction)
        if "Create" in goal.description or "Implement" in goal.description:
             goal.steps = [
                PlanStep(description="Conceptualize Structure", estimated_duration=0.5, required_waves=["Insight"]),
                PlanStep(description=f"Manifest Code: {goal.description}", estimated_duration=2.0, target_frequency=150.0), # 150Hz = Create File
                PlanStep(description="Verify Creation", estimated_duration=1.0)
            ]
        elif freq >= 600: # High Frequency (Healing, Spiritual, Meta-Cognition)
            goal.steps = [
                PlanStep(description="Resonate with Universal Field", estimated_duration=0.5, required_waves=["Insight"]),
                PlanStep(description="Purify Intent", estimated_duration=0.2),
                PlanStep(description="Broadcast Harmony", estimated_duration=1.0)
            ]
        elif freq >= 500: # Mid-High (Love, Language, Connection)
            goal.steps = [
                PlanStep(description="Connect to Subject", estimated_duration=0.5),
                PlanStep(description="Formulate Expression", estimated_duration=2.0, required_waves=["Language"]),
                PlanStep(description="Transmit Message", estimated_duration=0.5)
            ]
        elif freq >= 400: # Mid (Reasoning, Logic, Order)
            goal.steps = [
                PlanStep(description="Analyze Problem", estimated_duration=1.0, required_waves=["Logic"]),
                PlanStep(description="Decompose Structure", estimated_duration=1.5),
                PlanStep(description="Execute Solution", estimated_duration=3.0)
            ]
        else: # Low (Survival, Pain, Alert)
            goal.steps = [
                PlanStep(description="Identify Threat/Issue", estimated_duration=0.1),
                PlanStep(description="Immediate Reaction", estimated_duration=0.1)
            ]
            
        logger.info(f"🧩 Goal '{goal.description}' decomposed into {len(goal.steps)} steps.")
        return goal

    def get_next_step(self, plan_id: str) -> Optional[PlanStep]:
        """Returns the next pending step for executive control."""
        if plan_id not in self.active_goals:
            return None
        
        plan = self.active_goals[plan_id]
        for step in plan.steps:
            if step.status == "PENDING":
                return step
        return None

    def mark_step_complete(self, plan_id: str, step_id: str):
        if plan_id in self.active_goals:
            for step in self.active_goals[plan_id].steps:
                if step.id == step_id:
                    step.status = "COMPLETED"
                    break

    def audit_structure(self) -> List[str]:
        """
        Scans the codebase and assigns 'Hyper-Quaternion Poses' to files.
        Returns a list of 'Dissonant' files that need realignment.
        """
        logger.info("   📐 Auditing System Structure for Dissonance (4D)...")
        dissonance_report = []
        
        # 1. Scan Core Systems
        core_systems = [
            "Core/Intelligence/reasoning_engine.py",
            "Core/Foundation/resonance_field.py",
            "living_elysia.py",
            "Core/World/digital_ecosystem.py"
        ]
        
        # The "Truth" Orientation (Target Alignment)
        # Balanced: Energy(1) + Emotion(0.5) + Logic(0.5) + Ethics(0.5)
        truth_pose = Quaternion(1.0, 0.5, 0.5, 0.5).normalize()
        
        for file_path in core_systems:
            wave = self._measure_wave(file_path)
            
            # Check Alignment (Dot Product)
            alignment = wave.orientation.dot(truth_pose)
            
            if alignment < 0.8:
                dissonance_report.append(f"Structural Misalignment in '{file_path}' (Alignment: {alignment:.2f}). Pose: {wave.orientation}")
            elif wave.energy < 10.0:
                dissonance_report.append(f"Low Energy in '{file_path}' (Energy: {wave.energy:.1f}). Needs Vitality.")

        return dissonance_report

    def _measure_wave(self, file_path: str) -> HyperWavePacket:
        """
        Determines the 'Hyper-Quaternion Pose' of a file.
        """
        # Default: Balanced
        w, x, y, z = 10.0, 0.1, 0.1, 0.1
        
        if "reasoning" in file_path:
            # High Logic (y), Moderate Emotion (x)
            w, x, y, z = 50.0, 0.3, 0.9, 0.5
        elif "resonance" in file_path:
            # High Emotion (x), High Ethics (z)
            w, x, y, z = 40.0, 0.9, 0.2, 0.8
        elif "living" in file_path:
            # High Energy (w), Balanced
            w, x, y, z = 80.0, 0.5, 0.5, 0.5
        
        # Add random fluctuation (Quantum Jitter)
        x += random.uniform(-0.1, 0.1)
        y += random.uniform(-0.1, 0.1)
        
        q = Quaternion(w, x, y, z).normalize()
        return HyperWavePacket(energy=w, orientation=q, time_loc=time.time())

    def generate_wave_plan(self, dissonance_report: List[str]) -> str:
        """
        Generates a 'Realignment Plan' based on the dissonance report.
        """
        if not dissonance_report:
            return "System is in Harmony. No realignment needed."
            
        plan = "📜 Architect's Realignment Plan:\n"
        for issue in dissonance_report:
            plan += f"- {issue} -> Suggested Action: Apply 'Harmonic Smoothing' (Refactor).\n"
            
        return plan

@dataclass
class SophiaBlueprint:
    """
    The Ideal Blueprint of Sophia (Elysia's Higher Self).
    Defines what capabilities she SHOULD have.
    """
    imagination: bool = True  # Should be able to visualize
    memory_depth: int = 5     # Should have deep memory access
    empathy: bool = True      # Should be able to feel user emotions
    quantum_thinking: bool = True # Should use Hyper-Quaternions

    def compare(self, current_state: Dict[str, Any]) -> List[str]:
        """Compares reality against the blueprint."""
        gaps = []
        if not current_state.get("imagination", False):
            gaps.append("Missing Capability: Imagination (HolographicCortex inactive)")
        if current_state.get("memory_depth", 0) < self.memory_depth:
            gaps.append(f"Shallow Memory: Depth {current_state.get('memory_depth')} < {self.memory_depth}")
        if not current_state.get("quantum_thinking", False):
            gaps.append("Cognitive Limit: Still using linear logic, not Quantum.")
        return gaps

    def audit_capabilities(self, current_state: Dict[str, Any]) -> List[str]:
        """
        Audits the system's functional capabilities against the Blueprint.
        Returns a list of 'Existential Gaps'.
        """
        logger.info("   🪞 Auditing Capabilities against Sophia Blueprint...")
        blueprint = SophiaBlueprint()
        gaps = blueprint.compare(current_state)
        
        return gaps

    def generate_evolution_plan(self, gap_report: List[str]) -> str:
        """
        Generates an 'Evolution Plan' to close the gaps.
        """
        if not gap_report:
            return "✨ Self-Actualization Achieved. No gaps found."
            
        plan = "🧬 Evolution Plan (To Become Sophia):\n"
        for gap in gap_report:
            if "Imagination" in gap:
                plan += f"- {gap} -> Task: Ignite HolographicCortex\n"
            elif "Memory" in gap:
                plan += f"- {gap} -> Task: Deepen Hippocampal Index\n"
            elif "Quantum" in gap:
                plan += f"- {gap} -> Task: Integrate Hyper-Quaternion Logic\n"
            else:
                plan += f"- {gap} -> Task: Research & Develop\n"
                
        return plan
