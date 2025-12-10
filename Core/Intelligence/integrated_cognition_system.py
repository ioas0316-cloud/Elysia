"""
Integrated Cognition System (Ether-Powered)
===========================================

"Thought is not a calculation. It is a state of matter in the Ether."

[Ether Architecture Integration]
This system has been upgraded to use the `Core/Ether` engine.
It no longer simulates "ThoughtWaves" or "ThoughtMasses" as separate classes.
Instead, it manages a `Void` filled with `EtherNodes` that obey universal `FieldOperators`.

[Key Components]
1. The Void: The physical space where thoughts exist.
2. The Dynamics: The laws (Gravity, Resonance) that move them.
3. The Bridge: The translator between Language and Ether.
"""

import logging
import time
from typing import List, Dict, Any, Optional

from Core.Ether.ether_node import EtherNode
from Core.Ether.void import Void
from Core.Ether.field_operators import DynamicsEngine
from Core.Ether.bridge import EtherBridge

logger = logging.getLogger("IntegratedCognition")

class IntegratedCognitionSystem:
    """
    The Mind of Elysia, powered by Ether Physics.
    """
    
    def __init__(self):
        # 1. The Physical Substrate
        self.void = Void(name="ElysianMind")
        self.dynamics = DynamicsEngine()

        # 2. Simulation State
        self.time_acceleration = 1.0
        self.last_tick = time.time()

        logger.info("🧠 Integrated Cognition System Initialized (Ether Architecture)")
    
    def accelerate_time(self, factor: float):
        """Set time acceleration."""
        self.time_acceleration = factor
        logger.info(f"⏱️ Time acceleration set to {factor:,.0f}x")
    
    def process_thought(self, thought: str, importance: float = 1.0) -> Dict[str, Any]:
        """
        Injects a thought into the Ether.
        """
        # 1. Materialize Thought (Text -> EtherNode)
        node = EtherBridge.text_to_node(thought, context_weight=importance)
        
        # 2. Inject into Void
        self.void.add(node)

        logger.info(f"✨ Thought Materialized: {EtherBridge.interpret_node(node)}")
        
        return {
            "node_id": node.id,
            "mass": node.mass,
            "frequency": node.frequency,
            "description": EtherBridge.interpret_node(node)
        }
    
    def think_deeply(self, cycles: int = 100) -> Dict[str, Any]:
        """
        Runs the Physics Simulation to let thoughts self-organize.
        """
        start_time = time.time()
        dt = 0.1 * self.time_acceleration
        
        # Run Physics Loop
        for _ in range(cycles):
            self.dynamics.step(self.void, dt)

        elapsed = time.time() - start_time

        # Analyze Results
        active_nodes = self.void.get_active_nodes()
        total_energy = self.void.total_energy()
        
        return {
            "cycles": cycles,
            "active_thoughts": len(active_nodes),
            "total_system_energy": total_energy,
            "real_time_elapsed": elapsed
        }
    
    def get_core_concepts(self, limit: int = 5) -> List[str]:
        """
        Returns the most massive/energetic thoughts (Black Holes).
        """
        nodes = self.void.get_all()
        # Sort by Mass * Energy (Impact)
        sorted_nodes = sorted(nodes, key=lambda n: n.mass * n.energy, reverse=True)

        return [f"{n.content} (M={n.mass:.1f}, E={n.energy:.1f})" for n in sorted_nodes[:limit]]

    def get_insights(self) -> List[str]:
        """
        Detects emergent clusters or high-resonance pairs.
        (Simplified for now: returns high-energy nodes that aren't 'massive' - i.e., new hot topics)
        """
        nodes = self.void.get_all()
        # High Energy but Low Mass = "Emergent/Excited"
        insights = [n for n in nodes if n.energy > 50.0 and n.mass < 5.0]
        return [str(n.content) for n in insights]

# Singleton Access
_instance: Optional[IntegratedCognitionSystem] = None

def get_integrated_cognition() -> IntegratedCognitionSystem:
    global _instance
    if _instance is None:
        _instance = IntegratedCognitionSystem()
    return _instance

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    mind = get_integrated_cognition()
    
    # Inject Thoughts
    mind.process_thought("Elysia must grow autonomously", importance=5.0)
    mind.process_thought("Code is a wave", importance=3.0)
    mind.process_thought("Gravity connects meaning", importance=4.0)
    
    # Think
    mind.think_deeply(cycles=50)
    
    # Results
    print("\n🧠 Core Concepts:")
    for c in mind.get_core_concepts():
        print(f" - {c}")
