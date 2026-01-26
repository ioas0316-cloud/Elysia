"""
FRACTAL CAUSAL TOPOLOGY
=======================
Core.L5_Mental.Intelligence.Metabolism.causal_graph

"Why is the apple red? The answer is a loop, not a line."

This module traces the deep narrative structure of concepts.
It uses the LLM to recursively ask "Why?" and "How?", building a Causal Graph.
"""

import json
import logging
import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import ollama

logger = logging.getLogger("CausalTopology")

@dataclass
class CausalNode:
    concept: str
    depth: int # 0 for root, negative for cause, positive for effect
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    vector: Optional[List[float]] = None

class CausalDepthSounder:
    def __init__(self, model_name: str = "qwen2.5:0.5b"):
        self.model = model_name
        self.nodes: Dict[str, CausalNode] = {}
        logger.info(f"   Causal Depth Sounder initialized with {model_name}")

    def trace_root(self, concept: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Traces the full causal tree for a concept up to max_depth.
        """
        logger.info(f"  Tracing Causality for: [{concept}]")
        self.nodes = {} # Reset
        
        # Create Root
        self.nodes[concept] = CausalNode(concept=concept, depth=0)
        
        # Recursive Trace
        self._trace_antecedents(concept, current_depth=0, max_depth=max_depth)
        self._trace_consequences(concept, current_depth=0, max_depth=max_depth)
        
        return self._export_graph()

    def _trace_antecedents(self, current_concept: str, current_depth: int, max_depth: int):
        if abs(current_depth) >= max_depth: return

        # Ask "Why?"
        prompt = f"What is the single most fundamental cause or prerequisite for '{current_concept}'? Answer with one single word."
        cause = self._query_llm(prompt)
        
        if cause and cause not in self.nodes:
            logger.info(f"   Why({current_concept}) -> {cause}")
            # Link
            self.nodes[current_concept].parents.append(cause)
            
            # Create Node
            self.nodes[cause] = CausalNode(concept=cause, depth=current_depth - 1)
            self.nodes[cause].children.append(current_concept) # Bi-directional logic
            
            # Recurse
            self._trace_antecedents(cause, current_depth - 1, max_depth)

    def _trace_consequences(self, current_concept: str, current_depth: int, max_depth: int):
        if abs(current_depth) >= max_depth: return

        # Ask "How?"
        prompt = f"What is the single most direct consequence or effect of '{current_concept}'? Answer with one single word."
        effect = self._query_llm(prompt)
        
        if effect and effect not in self.nodes:
            logger.info(f"   How({current_concept}) -> {effect}")
            # Link
            self.nodes[current_concept].children.append(effect)
            
            # Create Node
            self.nodes[effect] = CausalNode(concept=effect, depth=current_depth + 1)
            self.nodes[effect].parents.append(current_concept)
            
            # Recurse
            self._trace_consequences(effect, current_depth + 1, max_depth)

    def _query_llm(self, prompt: str) -> str:
        try:
            response = ollama.generate(model=self.model, prompt=prompt)
            # Clean output: simple word extraction
            text = response['response'].strip().split('\n')[0]
            # Remove punctuation
            text = ''.join(e for e in text if e.isalnum())
            return text if text else "Void"
        except Exception as e:
            logger.error(f"LLM Query Failed: {e}")
            return "Void"

    def _export_graph(self) -> Dict[str, Any]:
        """Exports the graph structure."""
        roots = [d.concept for d in self.nodes.values() if d.depth == 0]
        return {
            "root": roots[0] if roots else "Unknown",
            "nodes": [
                {
                    "concept": k,
                    "depth": v.depth,
                    "parents": v.parents,
                    "children": v.children
                }
                for k, v in self.nodes.items()
            ]
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tracer = CausalDepthSounder()
    # Test with 'Justice'
    graph = tracer.trace_root("Justice", max_depth=2)
    print(json.dumps(graph, indent=2))
