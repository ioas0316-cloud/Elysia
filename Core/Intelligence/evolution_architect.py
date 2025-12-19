
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger("EvolutionArchitect")

@dataclass
class OptimizationGoal:
    name: str # e.g., "Quantum Leanness"
    description: str
    target_complexity: float # 0.0 - 1.0 (Lower is simpler)

@dataclass
class Blueprint:
    """
    The DNA Map for the New Seed.
    """
    goal: OptimizationGoal
    structure: Dict[str, str] # {module_name: action} (e.g., 'Core.Foundation': 'Keep')
    improvements: List[str]
    execution_steps: List[str]
    
    def explain(self) -> str:
        """Returns a monologue explaining this blueprint."""
        return (
            f"I have designed a Seed for '{self.goal.name}'. "
            f"The goal is: {self.goal.description}. "
            f"Structurally, I will refine {len(self.structure)} components. "
            f"Key Improvements: {', '.join(self.improvements[:2])}. "
            f"Execution Phase: {self.execution_steps[0]}..."
        )

class EvolutionArchitect:
    """
    [The Designer of the Self]
    
    Analyses the current self and proposes an optimized 'Seed' version.
    It does NOT execute; it Plans and Explains.
    """
    
    def __init__(self, cns_ref=None):
        self.cns = cns_ref
        self.current_blueprint = None
        
        # Connect to Internal Systems
        try:
            from Core.Cognition.metacognitive_awareness import MetacognitiveAwareness
            self.metacognition = MetacognitiveAwareness()
        except ImportError:
            self.metacognition = None
            
        try:
            from Core.Autonomy.self_modifier_v2 import get_self_modifier
            self.self_modifier = get_self_modifier()
        except ImportError:
            self.self_modifier = None

    def design_seed(self, intent: str = "Optimization") -> Blueprint:
        """
        Generates a Blueprint based on real-time cognitive gaps and structural tension.
        """
        logger.info(f"🏗️ Designing Seed with intent: {intent}")
        
        # 1. Gather Needs (Cognitive Gaps)
        gaps = []
        if self.metacognition:
            needs = self.metacognition.get_exploration_priorities(top_n=3)
            for n in needs:
                gaps.append(f"Lack of understanding: {n['question']}")
        
        # 2. Gather Tension (Structural Faults)
        faults = []
        if self.self_modifier:
            # Analyze a sample critical module
            report = self.self_modifier.generate_report(directory="Core/Cognition")
            for f in report.get("high_tension_files", [])[:3]:
                faults.append(f"High Tension in {Path(f['path']).name} (T: {f['tension']:.2f})")
                
        # 3. Formulate Goal
        description = "Upgrade system to bridge cognitive gaps and relieve structural tension."
        if not gaps and not faults:
            description = "Routine optimization for higher coherence."
            
        goal = OptimizationGoal(
            name=f"Seed: {intent.capitalize()}",
            description=description,
            target_complexity=0.8
        )
        
        # 4. Determine Structure & Improvements
        structure = {}
        improvements = []
        
        if gaps:
            improvements.append(f"Resolve Cognitive Gaps: {'; '.join(gaps[:1])}")
            structure["Core.Cognition"] = "Expand (Add missing patterns)"
            
        if faults:
            improvements.append(f"Refactor Structural Faults: {'; '.join(faults[:1])}")
            structure["Core.Autonomy"] = "Refactor (Relieve tension)"
            
        if not improvements:
            improvements = ["General Optimization", "Clean up legacy thoughts"]
            structure["Core"] = "Maintain"
        
        # 5. Plan Execution
        steps = [
            "1. Materialize Blueprint",
            "2. User Review & Approval",
            "3. Execute Self-Modification via WaveCoder"
        ]
        
        self.current_blueprint = Blueprint(goal, structure, improvements, steps)
        return self.current_blueprint

    def console_explain(self) -> str:
        """Returns a monologue explaining this blueprint."""
        return self.current_blueprint.explain() if self.current_blueprint else "No blueprint."

    def materialize_blueprint(self) -> str:
        """
        [Manifestation]
        Writes the detailed Blueprint to a physical Markdown file.
        """
        if not self.current_blueprint:
            return "No blueprint to manifest."
            
        bp = self.current_blueprint
        
        # Design the Markdown content
        md_content = f"""# 🧬 Blueprint: {bp.goal.name}
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Architect:** Elysia (EvolutionArchitect)
**Goal:** {bp.goal.description}
**Target Complexity:** {bp.goal.target_complexity}

## 🏗️ Structural Analysis
| Module | Action |
|--------|--------|
"""
        for mod, action in bp.structure.items():
            md_content += f"| `{mod}` | {action} |\n"
            
        md_content += f"""
## 🚀 Key Improvements
"""
        for imp in bp.improvements:
             md_content += f"- {imp}\n"
             
        md_content += f"""
## ⚙️ Execution Plan (The Ouroboros Protocol)
"""
        for step in bp.execution_steps:
             md_content += f"1. {step}\n"
             
        md_content += """
## 📐 Architecture Diagram (Conceptual)
```mermaid
graph TD
    User((User)) -->|Input| Seed(Nova Seed)
    Seed -->|Fractal Loop| Core(Immutable Core)
    Core -->|Output| Voice(Evolution Voice)
    subgraph Nova Seed
        Core
        Logic[Fractal Logic]
        Mem[Quantum Memory]
    end
```
"""
        # Ensure directory exists
        seed_dir = Path("seeds")
        seed_dir.mkdir(exist_ok=True)
        
        file_path = seed_dir / "nova_seed_blueprint.md"
        file_path.write_text(md_content, encoding="utf-8")
        
        logger.info(f"✨ Blueprint materialized at: {file_path}")
        return str(file_path)
