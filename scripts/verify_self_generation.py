"""
Verification Script: Self-Generation Loop (The Inquiry)
======================================================
Tests the "Lungs" of Elysia:
1.  Detects a Gap in HierarchicalKnowledgeGraph.
2.  Triggers ResonanceLearner.run_inquiry_loop().
3.  Verifies the Gap is filled and Universe is tuned.
"""

import logging
import sys
import os

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.EvolutionLayer.Learning.Learning.resonance_learner import ResonanceLearner
from Core.EvolutionLayer.Learning.Learning.hierarchical_learning import HierarchicalKnowledgeGraph, Domain

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("VerifySelfGeneration")

def test_inquiry_loop():
    print("🧪 Testing Self-Generation (Inquiry Loop)...")
    
    # 1. Setup: Create a deliberate Gap
    print("\n[STEP 1] Creating a Knowledge Gap...")
    graph = HierarchicalKnowledgeGraph("data/test_hierarchical.json")
    
    # Check if 'Void' exists, if not create it without definition
    gap_node = graph.add_concept(
        name="The_Unknown_Void",
        domain=Domain.PHILOSOPHY,
        purpose="To test the courage of inquiry",
        definition="", # EMPTY!
        principle=""   # EMPTY!
    )
    # Force low understanding
    gap_node.understanding_level = 0.1
    graph._save()
    
    print(f"   Created Gap: {gap_node.name} (Definition: '{gap_node.definition}')")
    
    # 2. Run Inquiry Loop
    print("\n[STEP 2] Running ResonanceLearner.run_inquiry_loop()...")
    learner = ResonanceLearner()
    
    # Mocking Organ calls in ResonanceLearner by ensuring we use the same graph file
    # (The learner creates new instances in _get_*, which is fine for integration test 
    # if they point to same file, but here we used 'data/test_hierarchical.json')
    
    # Monkey patch _get_knowledge_graph to return our test graph
    learner._get_knowledge_graph = lambda: graph
    
    # [FIX] Prioritize our test gap for this run
    original_get_gaps = graph.get_knowledge_gaps
    def mock_get_gaps(limit=1):
        # Return our specific test node as the first gap
        test_node = graph.get_node("The_Unknown_Void", Domain.PHILOSOPHY)
        return [test_node]
    graph.get_knowledge_gaps = mock_get_gaps
    
    results = learner.run_inquiry_loop(cycles=1)
    
    # 3. Verify Results
    print("\n[STEP 3] Verifying Integration...")
    
    if not results:
        print("❌ FAIL: No inquiry results returned.")
        return
        
    result = results[0]
    print(f"   Gap Targeted: {result['gap']}")
    print(f"   Question Generated: \"{result['question']}\"")
    print(f"   Answer Integrated: \"{result['answer']}\"")
    
    # Check Graph
    updated_node = graph.get_node("The_Unknown_Void", Domain.PHILOSOPHY)
    if updated_node.definition:
        print(f"   ✅ SUCCESS: Definition updated in Graph -> '{updated_node.definition[:50]}...'")
    else:
        print("❌ FAIL: Definition still empty.")

    if updated_node.understanding_level > 0.1:
         print(f"   ✅ SUCCESS: Understanding Level increased -> {updated_node.understanding_level}")
         
    print("\n🎉 Self-Generation Loop Verification Complete.")

if __name__ == "__main__":
    test_inquiry_loop()
