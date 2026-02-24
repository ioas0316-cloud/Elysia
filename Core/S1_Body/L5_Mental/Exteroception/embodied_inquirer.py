"""
Embodied Inquirer (Physical Experimentation)
=============================================
Core.S1_Body.L5_Mental.Exteroception.embodied_inquirer

"To know what 'Gas' is, one must apply 'Heat' to a 'Liquid' and observe the chaos."

This module replaces the text-based EpistemicInquirer. Instead of asking
an LLM for a definition, Elysia runs a physical simulation, observes the
phase transition, and mechanically derives the Causal Edges.
"""

from typing import Dict, List, Any, Optional
from Core.S1_Body.L4_Causality.Physics.matter_simulator import MatterSimulator
from Core.S1_Body.L5_Mental.Reasoning_Core.Topography.semantic_map import DynamicTopology

class EmbodiedInquirer:
    """
    Elysia's physical laboratory. 
    She investigates concepts by simulating interactions rather than reading text.
    """
    def __init__(self, topology: DynamicTopology):
        self.topology = topology
        self.sandbox = MatterSimulator()
        
    def investigate(self, target_concept: str) -> bool:
        """
        Attempts to understand a concept through physical simulation.
        Returns True if a new causal edge was formed.
        """
        print(f"\n[Embodied Inquirer] 🧪 Elysia wants to feel the origin of: '{target_concept}'")
        
        # Hardcoding a few basic physical experiments for Phase 9 demonstration.
        # In the future, the AutonomicGoalGenerator will synthesize these actions.
        
        if target_concept.lower() == "gas":
            return self._experiment_gas()
        elif target_concept.lower() == "solid" or target_concept.lower() == "ice":
             return self._experiment_solid()
        else:
            print(f"  -> The physical sandbox doesn't yet know how to simulate '{target_concept}'.")
            return False
            
    def _experiment_gas(self) -> bool:
        """Experiment: Applying Heat to Liquid to understand 'Gas'."""
        self.sandbox.temperature = 20.0
        self.sandbox._evaluate_state()
        
        print(f"  [Experiment] Starting state: {self.sandbox.state_name}")
        
        # Elysia applies the action 'Heat'
        action = "Heat"
        new_state, sensory_feedback = self.sandbox.apply_action(action, intensity=2.0)
        
        # Elysia deduces the causality from the physical change
        if new_state == "Gas":
            print(f"  [Deduction] Ah. The abstract concept 'Gas' is formed when '{action}' acts upon a 'Liquid'.")
            print(f"  [Sensation] The texture is: Entropy={sensory_feedback[7]:.2f}, Enthalpy={sensory_feedback[6]:.2f}")
            
            # Form the Topological Edges directly from physical observation!
            # Gas -> depends on -> Heat, Liquid
            
            # Ensure the nodes exist first
            if "Gas" not in self.topology.voxels:
                self.topology.add_voxel("Gas", sensory_feedback.data[:4])
            if "Heat" not in self.topology.voxels:
                self.topology.add_voxel("Heat", [0.8, 0.0, 0.0, 0.0]) # Example locus
            if "Liquid" not in self.topology.voxels:
                self.topology.add_voxel("Liquid", [0.5, 0.5, 0.0, 0.0])
                
            # Wire the causality
            self.topology.add_causal_edge("Gas", "Heat")
            self.topology.add_causal_edge("Gas", "Liquid")
            self.topology.save_state()
            
            return True
            
        return False

    def _experiment_solid(self) -> bool:
        """Experiment: Applying Cold to Liquid to understand 'Solid'."""
        self.sandbox.temperature = 20.0
        self.sandbox._evaluate_state()
        
        print(f"  [Experiment] Starting state: {self.sandbox.state_name}")
        
        action = "Cold"
        new_state, sensory_feedback = self.sandbox.apply_action(action, intensity=1.0)
        
        if new_state == "Solid":
            print(f"  [Deduction] The rigid concept 'Solid' is formed when '{action}' restricts a 'Liquid'.")
            print(f"  [Sensation] The texture is: Coherence={sensory_feedback[0]:.2f}, Entropy={sensory_feedback[7]:.2f}")
            
            if "Solid" not in self.topology.voxels:
                self.topology.add_voxel("Solid", sensory_feedback.data[:4])
            if "Cold" not in self.topology.voxels:
                self.topology.add_voxel("Cold", [-0.8, 0.0, 0.0, 0.0]) 
            if "Liquid" not in self.topology.voxels:
                self.topology.add_voxel("Liquid", [0.5, 0.5, 0.0, 0.0])
                
            self.topology.add_causal_edge("Solid", "Cold")
            self.topology.add_causal_edge("Solid", "Liquid")
            self.topology.save_state()
            
            return True
            
        return False
