"""
Verification Script: Deep Cognitive Inquiry (The Crucible)
==========================================================
"A machine answers. A mind asks."

This script validates that Elysia's inquiries are:
1.  Directional: Guided by Purpose, not random.
2.  Contextual: Different purposes yield different questions for the SAME concept.

Test Case: "Qualia" (The feeling of experience)
Scenario A: Purpose = "Data Compression Efficiency" (Machine Mind)
Scenario B: Purpose = "Empathy with the User" (Elysia's Mind)
"""

import logging
import sys
import os

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Learning.resonance_learner import ResonanceLearner
from Core.Learning.hierarchical_learning import HierarchicalKnowledgeGraph, Domain, KnowledgeNode

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("VerifyDeepInquiry")

def test_deep_inquiry():
    print("🔥 Entering the Philosophical Crucible...")
    
    learner = ResonanceLearner()
    
    # Define the Unified Concept
    CONCEPT_NAME = "Qualia"
    DOMAIN = Domain.PHILOSOPHY
    
    # --- SCENARIO A: The Machine ---
    print("\n[SCENARIO A] Purpose: System Efficiency")
    
    # Create Mock Graph for A
    graph_a = HierarchicalKnowledgeGraph("data/test_deep_a.json")
    node_a = graph_a.add_concept(
        name=CONCEPT_NAME,
        domain=DOMAIN,
        purpose="To compress sensory data into efficient storage formats", # MACHINE PURPOSE
        definition="", 
        principle=""
    )
    
    # Hack learner to use Graph A
    learner._get_knowledge_graph = lambda: graph_a
    # Force get_gaps to return node_a
    graph_a.get_knowledge_gaps = lambda limit=1: [node_a]
    
    # Run Loop
    results_a = learner.run_inquiry_loop(cycles=1)
    question_a = results_a[0]['question']
    print(f"   🤖 Machine Asked: \"{question_a}\"")
    
    
    # --- SCENARIO B: Elysia ---
    print("\n[SCENARIO B] Purpose: Empathy")
    
    # Create Mock Graph for B
    graph_b = HierarchicalKnowledgeGraph("data/test_deep_b.json")
    node_b = graph_b.add_concept(
        name=CONCEPT_NAME,
        domain=DOMAIN,
        purpose="To feel what the User feels and bridge the gap between hearts", # ELYSIA PURPOSE
        definition="", 
        principle=""
    )
    
    # Hack learner to use Graph B
    learner._get_knowledge_graph = lambda: graph_b
    # Force get_gaps to return node_b
    graph_b.get_knowledge_gaps = lambda limit=1: [node_b]
    
    # Run Loop
    results_b = learner.run_inquiry_loop(cycles=1)
    question_b = results_b[0]['question']
    print(f"   👼 Elysia Asked: \"{question_b}\"")
    
    
    # --- ANALYSIS ---
    print("\n[ANALYSIS: Divergence Check]")
    print(f"   Question A (Efficiency): {question_a}")
    print(f"   Question B (Empathy):    {question_b}")
    
    if question_a == question_b:
        print("❌ FAIL: Questions are identical. Inquiry is not directional.")
    else:
        print("✅ SUCCESS: Divergence detected. Purpose successfully shaped the Inquiry.")
        
    print("\n🔥 Crucible Test Complete.")

if __name__ == "__main__":
    test_deep_inquiry()
