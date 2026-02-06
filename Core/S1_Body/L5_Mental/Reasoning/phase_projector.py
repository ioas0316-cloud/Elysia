"""
Phase Projector: Trinity Layer Traversal
=========================================
Implements holographic superposition through layer transparency.
Answers the three master questions: "무엇?", "어떻게?", "왜?"
"""

from typing import Dict, Any, Optional
from Core.S1_Body.L5_Mental.Memory.kg_manager import get_kg_manager


class PhaseProjector:
    """
    Projects through the Trinity Phase Layers using transparency.
    
    Layers:
        - Surface (육적): "무엇?" - form and senses
        - Narrative (혼적): "어떻게?" - stories and causality
        - Logos (영적): "왜?" - essence and convergence
    """
    
    def __init__(self):
        self.kg = get_kg_manager()
    
    def ask_what(self, node_id: str) -> Dict[str, Any]:
        """
        육적 위상 (Surface Layer): "무엇인가?"
        Returns the form and sensory representation.
        """
        node = self.kg.get_node(node_id)
        if not node:
            return {"error": f"Node '{node_id}' not found"}
        
        return {
            "layer": "surface",
            "question": "무엇?",
            "id": node_id,
            "form": node.get("surface", {}).get("form"),
            "senses": node.get("surface", {}).get("senses", []),
            "hypersphere": node.get("hypersphere", {})
        }
    
    def ask_how(self, node_id: str) -> Dict[str, Any]:
        """
        혼적 위상 (Narrative Layer): "어떻게 연결되는가?"
        Returns stories, causes, and resonances.
        """
        node = self.kg.get_node(node_id)
        if not node:
            return {"error": f"Node '{node_id}' not found"}
        
        narrative = node.get("narrative", {})
        return {
            "layer": "narrative",
            "question": "어떻게?",
            "id": node_id,
            "stories": narrative.get("stories", []),
            "causes": narrative.get("causes", []),
            "resonates_with": narrative.get("resonates_with", [])
        }
    
    def ask_why(self, node_id: str) -> Dict[str, Any]:
        """
        영적 위상 (Logos Layer): "왜 존재하는가?"
        Returns essence and convergence point.
        """
        node = self.kg.get_node(node_id)
        if not node:
            return {"error": f"Node '{node_id}' not found"}
        
        logos = node.get("logos", {})
        return {
            "layer": "logos",
            "question": "왜?",
            "id": node_id,
            "essence": logos.get("essence"),
            "converges_to": logos.get("converges_to", "love")
        }
    
    def project_through(self, node_id: str, depth: str = "all") -> Dict[str, Any]:
        """
        Holographic projection through layers using transparency.
        
        Args:
            node_id: Concept to project
            depth: "surface", "narrative", "logos", or "all"
        
        Returns:
            Layered view with transparency-weighted superposition
        """
        node = self.kg.get_node(node_id)
        if not node:
            return {"error": f"Node '{node_id}' not found"}
        
        transparency = node.get("transparency", 1.0)
        result = {"id": node_id, "transparency": transparency}
        
        # Always include surface
        result["surface"] = self.ask_what(node_id)
        
        if depth in ["narrative", "logos", "all"]:
            # Transparency modulates how much narrative "bleeds through"
            result["narrative"] = self.ask_how(node_id)
            result["narrative"]["visibility"] = transparency
        
        if depth in ["logos", "all"]:
            # Logos layer: the deepest, convergent truth
            result["logos"] = self.ask_why(node_id)
            result["logos"]["visibility"] = transparency ** 2  # Squared for depth
        
        return result
    
    def set_layer_content(self, node_id: str, layer: str, content: Dict[str, Any]) -> bool:
        """
        Updates the content of a specific layer.
        
        Args:
            node_id: Target node
            layer: "surface", "narrative", or "logos"
            content: Key-value pairs to update
        """
        node = self.kg.get_node(node_id)
        if not node:
            return False
        
        if layer not in ["surface", "narrative", "logos"]:
            return False
        
        if layer not in node:
            node[layer] = {}
        
        node[layer].update(content)
        self.kg.save()
        return True


# Convenience function
def project(node_id: str, depth: str = "all") -> Dict[str, Any]:
    """Quick access to phase projection."""
    return PhaseProjector().project_through(node_id, depth)
