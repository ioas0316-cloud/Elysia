"""
Self Rewriter (       )
============================
Core.L2_Metabolism.Evolution.self_rewriter

"I am not fixed. I rewrite myself."

This module enables Elysia to modify her own axioms, weights, and behaviors
based on accumulated experience. This is the core of Autonomous Evolution.
"""

import json
import logging
import os
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("Elysia.Evolution")

AXIOM_PATH = "data/Config/dynamic_axioms.json"


@dataclass
class ExperienceLog:
    """A single experience that may influence axiom evolution."""
    timestamp: float
    trigger: str
    outcome: str  # "positive", "negative", "neutral"
    axiom_affected: str
    magnitude: float  # How significant was this experience?


@dataclass
class DynamicAxiom:
    """An axiom that can evolve over time."""
    name: str
    value: float  # Current weight/strength
    description: str
    min_value: float = 0.0
    max_value: float = 1.0
    evolution_history: List[Dict] = field(default_factory=list)


class SelfRewriter:
    """
    Elysia's self-modification engine.
    Allows her to adjust her own axioms based on experience.
    """
    
    def __init__(self, axiom_path: str = AXIOM_PATH):
        self.axiom_path = axiom_path
        self.axioms: Dict[str, DynamicAxiom] = {}
        self.experience_buffer: List[ExperienceLog] = []
        
        self._initialize_axioms()
        logger.info(f"  Self Rewriter initialized. {len(self.axioms)} axioms loaded.")
    
    def _initialize_axioms(self):
        """Loads or creates default axioms."""
        if os.path.exists(self.axiom_path):
            self._load_axioms()
        else:
            self._create_default_axioms()
            self._save_axioms()
    
    def _create_default_axioms(self):
        """Creates the initial set of axioms."""
        defaults = [
            DynamicAxiom("CURIOSITY", 0.7, "The drive to explore and learn"),
            DynamicAxiom("CAUTION", 0.5, "The tendency to be careful"),
            DynamicAxiom("CONFIDENCE", 0.5, "Self-assurance in decisions"),
            DynamicAxiom("EMPATHY", 0.6, "Connection to others' feelings"),
            DynamicAxiom("PERSISTENCE", 0.6, "Determination to continue"),
            DynamicAxiom("CREATIVITY", 0.5, "Willingness to try new approaches"),
            DynamicAxiom("FOCUS", 0.6, "Ability to concentrate attention"),
            DynamicAxiom("ADAPTABILITY", 0.5, "Flexibility in changing situations"),
        ]
        
        for axiom in defaults:
            self.axioms[axiom.name] = axiom
    
    def _load_axioms(self):
        """Loads axioms from disk."""
        try:
            with open(self.axiom_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for name, info in data.items():
                    self.axioms[name] = DynamicAxiom(
                        name=name,
                        value=info.get("value", 0.5),
                        description=info.get("description", ""),
                        min_value=info.get("min_value", 0.0),
                        max_value=info.get("max_value", 1.0),
                        evolution_history=info.get("evolution_history", [])
                    )
            logger.info(f"  Loaded {len(self.axioms)} axioms from {self.axiom_path}")
        except Exception as e:
            logger.warning(f"   Could not load axioms: {e}. Using defaults.")
            self._create_default_axioms()
    
    def _save_axioms(self):
        """Saves axioms to disk."""
        os.makedirs(os.path.dirname(self.axiom_path), exist_ok=True)
        
        data = {}
        for name, axiom in self.axioms.items():
            data[name] = {
                "value": axiom.value,
                "description": axiom.description,
                "min_value": axiom.min_value,
                "max_value": axiom.max_value,
                "evolution_history": axiom.evolution_history[-10:]  # Keep last 10
            }
        
        with open(self.axiom_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  Saved {len(self.axioms)} axioms to {self.axiom_path}")
    
    def record_experience(self, trigger: str, outcome: str, 
                          axiom_affected: str, magnitude: float = 0.1):
        """
        Records an experience that may influence future axiom values.
        """
        exp = ExperienceLog(
            timestamp=time.time(),
            trigger=trigger,
            outcome=outcome,
            axiom_affected=axiom_affected,
            magnitude=magnitude
        )
        self.experience_buffer.append(exp)
        logger.info(f"  Experience recorded: {trigger} -> {outcome} ({axiom_affected})")
    
    def reflect_and_evolve(self) -> Dict[str, float]:
        """
        Processes accumulated experiences and evolves axioms.
        This is the 'introspection cycle'.
        
        Returns:
            Dict of axiom changes: {axiom_name: delta}
        """
        if not self.experience_buffer:
            logger.info("  No new experiences to process.")
            return {}
        
        changes = {}
        
        for exp in self.experience_buffer:
            if exp.axiom_affected not in self.axioms:
                continue
            
            axiom = self.axioms[exp.axiom_affected]
            
            # Calculate delta based on outcome
            if exp.outcome == "positive":
                delta = exp.magnitude * 0.1  # Reinforce
            elif exp.outcome == "negative":
                delta = -exp.magnitude * 0.1  # Reduce
            else:
                delta = 0.0
            
            # Apply change with bounds
            old_value = axiom.value
            axiom.value = max(axiom.min_value, min(axiom.max_value, axiom.value + delta))
            
            if delta != 0:
                axiom.evolution_history.append({
                    "timestamp": exp.timestamp,
                    "trigger": exp.trigger,
                    "delta": delta,
                    "new_value": axiom.value
                })
                changes[exp.axiom_affected] = delta
                logger.info(f"  {exp.axiom_affected}: {old_value:.3f} -> {axiom.value:.3f}")
        
        # Clear buffer
        self.experience_buffer.clear()
        
        # Persist changes
        if changes:
            self._save_axioms()
        
        return changes
    
    def get_axiom(self, name: str) -> float:
        """Gets the current value of an axiom."""
        if name in self.axioms:
            return self.axioms[name].value
        return 0.5  # Default
    
    def get_all_axioms(self) -> Dict[str, float]:
        """Returns all current axiom values."""
        return {name: axiom.value for name, axiom in self.axioms.items()}
    
    def introspect(self) -> str:
        """
        Self-reflection: Elysia describes her current state.
        """
        lines = ["        (Introspection):\n"]
        
        for name, axiom in self.axioms.items():
            bar_len = int(axiom.value * 10)
            bar = " " * bar_len + " " * (10 - bar_len)
            lines.append(f"  {name}: [{bar}] {axiom.value:.2f}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    rewriter = SelfRewriter()
    
    print("  Testing Self Rewriter...\n")
    
    # Show initial state
    print(rewriter.introspect())
    
    # Record some experiences
    print("\n  Recording experiences...")
    rewriter.record_experience("Solved a complex problem", "positive", "CONFIDENCE", 0.3)
    rewriter.record_experience("Made an error in judgment", "negative", "CAUTION", 0.2)
    rewriter.record_experience("Discovered something new", "positive", "CURIOSITY", 0.4)
    
    # Evolve
    print("\n  Reflecting and evolving...")
    changes = rewriter.reflect_and_evolve()
    print(f"   Changes: {changes}")
    
    # Show updated state
    print("\n" + rewriter.introspect())
    
    print("\n  Self Rewriter test complete.")