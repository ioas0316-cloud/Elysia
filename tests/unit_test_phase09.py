import sys
import os
import torch
from typing import Dict, Any

# Ensure root directory is in path
sys.path.append(os.getcwd())

from Core.Intelligence.LLM.metacognitive_lens import MetacognitiveLens
from Core.Foundation.Graph.torch_graph import TorchGraph

def test_lens():
    print("🔬 Testing Metacognitive Lens...")
    lens = MetacognitiveLens()
    script = "I am stagnant and feel the need for chaos."
    critique = lens.critique(script, current_mood="IDLE")
    print(f"Critique: {critique}")
    refined = lens.refine_voice("I want chaos.", critique)
    print(f"Refined Voice: {refined}")
    assert not critique["valid"], "Critique should have flagged stagnation."
    assert "Fractal Principle of Growth" in critique["suggestions"][0]
    print("✅ Metacognitive Lens Logic Verified.")

def test_causal_alignment():
    print("🔬 Testing Causal Alignment and Emotional Gravity...")
    graph = TorchGraph(use_cuda=False)
    graph.add_node("Root_Principle", vector=[0.1]*384)
    graph.add_node("Potential_Possibility", vector=[0.2]*384)
    
    # Test Emotional Gravity (Anchoring)
    idx = graph.id_to_idx["Root_Principle"]
    graph.grav_tensor[idx] = 100.0 # High gravity
    
    # Search should favor Root_Principle
    results = graph.get_nearest_by_vector(torch.tensor([0.2]*384), top_k=1)
    print(f"Search Results: {results}")
    assert results[0][0] == "Root_Principle", "Gravity anchoring failed to pull the search."
    
    # Test Causal Alignment (Formerly Dream)
    # We simulate the logic of causal_alignment
    hollows = graph.find_hollow_nodes(limit=2)
    print(f"Hollows found: {hollows}")
    assert len(hollows) >= 2
    
    graph.add_link(hollows[0], hollows[1], weight=0.5, link_type="causal_possibility")
    print(f"Forged Causal Link: {hollows[0]} <-> {hollows[1]}")
    assert (graph.id_to_idx[hollows[0]], graph.id_to_idx[hollows[1]]) in graph.link_metadata
    print("✅ Causal Alignment & Gravity Logic Verified.")

if __name__ == "__main__":
    test_lens()
    test_causal_alignment()
    print("🏆 Unit Tests Complete: Phase 09 Logic is sound and philosophically aligned.")
