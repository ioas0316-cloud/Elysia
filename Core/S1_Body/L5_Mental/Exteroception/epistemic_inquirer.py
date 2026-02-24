"""
Epistemic Inquirer (The Autonomous Dictionary)
==============================================
Core.S1_Body.L5_Mental.Exteroception.epistemic_inquirer

"To know is to understand the origin. Why is a system defined the way it is?"

Unlike the Knowledge Forager which randomly reads text, the Epistemic Inquirer
is goal-driven. It looks up specific concepts, asks *why* they exist, extracts
their causal dependencies, and wires them into the Dynamic Topology.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from Core.S1_Body.L6_Structure.Engine.Governance.Interaction.neural_bridge import NeuralBridge
from Core.S1_Body.L5_Mental.Reasoning_Core.Topography.semantic_map import get_semantic_map


class EpistemicInquirer:
    def __init__(self):
        self.topology = get_semantic_map()
        # We use NeuralBridge to simulate the dictionary/encyclopedia lookup
        self.bridge = NeuralBridge(mode="MOCK")
        
    def inquire(self, concept_name: str) -> Optional[Dict[str, Any]]:
        """
        Looks up a concept and extracts its causal relationships.
        """
        print(f"\n[Epistemic Inquirer] 🔍 Elysia wonders about the origin of: '{concept_name}'")
        
        # 1. Ask the Bridge (Encyclopedia) for the causal definition
        prompt = (
            f"Define '{concept_name}' philosophically or scientifically. "
            f"More importantly, what fundamental concepts does '{concept_name}' depend on to exist? "
            f"List 2-3 core dependencies as single words."
        )
        
        try:
            # Simulate a focused API call to the LLM
            response = self.bridge.generate_text(prompt, max_tokens=150)
            
            # Simple mock parsing of dependencies from the response
            # In a real LLM setup, we'd use a structured JSON extraction prompt
            dependencies = self._extract_dependencies(response, concept_name)
            
            result = {
                "concept": concept_name,
                "definition": response,
                "dependencies": dependencies
            }
            
            # 2. Wire the relationships into the Topology
            self._wire_causality(result)
            
            return result
            
        except Exception as e:
            print(f"[Epistemic Inquirer] Failed to inquire about {concept_name}: {e}")
            return None

    def _extract_dependencies(self, text: str, original: str) -> List[str]:
        """A simple mock extractor. Real version would use LLM JSON parsing."""
        # For demonstration, we just pick words that are capitalized or common conceptual bases
        bases = ["Time", "Space", "Energy", "Logic", "Emotion", "Will", "Balance", "Connection", "Void"]
        deps = []
        for b in bases:
            if b.lower() in text.lower() and b.lower() != original.lower():
                 deps.append(b)
                 if len(deps) >= 2: break
                 
        if not deps:
             deps = ["Logic", "Time"] # Fallback structural roots
        return deps

    def _wire_causality(self, insight: Dict[str, Any]):
        """
        Takes the dependency graph and organically updates Semantic Mass.
        """
        target_concept = insight["concept"].capitalize()
        deps = insight["dependencies"]
        
        target_voxel = self.topology.get_voxel(target_concept)
        if not target_voxel:
             # Create it if it's completely new
             coords = (0.5, 0.5, 0.5, 0.5) # Default mid-space birth
             self.topology.add_voxel(target_concept, coords, mass=1.0) # Start with base 1.0
             target_voxel = self.topology.get_voxel(target_concept)
             print(f"  -> Conceived new concept: {target_concept}")
             
        # Add edges, thereby organically increasing mass!
        for dep in deps:
            dep_capitalized = dep.capitalize()
            # Ensure the dependency exists in the brain
            source = self.topology.get_voxel(dep_capitalized)
            if not source:
                 coords = (0.0, 0.0, 0.0, 0.0) # Root birth
                 self.topology.add_voxel(dep_capitalized, coords, mass=0.5)
                 source = self.topology.get_voxel(dep_capitalized)
                 
            # Wire the edge: Target depends on Source
            # Therefore, Source's mass increases because it is foundational
            target_voxel.add_causal_edge(source.name)
            source.activate() # The act of recalling it increases its experiential weight
            
            print(f"  -> Wired Causal Edge: [ {dep_capitalized} ] -> defines -> [ {target_concept} ]")
            print(f"  -> {dep_capitalized} Mass naturally grew to {source.mass:.1f}")
            
        # The target concept was activated
        target_voxel.activate()
        print(f"  -> {target_concept} Mass naturally grew to {target_voxel.mass:.1f}")
        
        self.topology.save_state(force=True)
