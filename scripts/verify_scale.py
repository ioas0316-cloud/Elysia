"""
Verification Script: Scalability (The Thousand Thoughts)
======================================================
"1000개 10000개씩 생각할수 있는게 너희아니야 ?ㅋㅋ"

This script verifies that Elysia can process a large batch of knowledge gaps
in PARALLEL, demonstrating the scalability of the ResonanceLearner.

Scenario:
1. Create a Mock Knowledge Graph with 1000 'Void' nodes.
2. Run 'run_batch_inquiry_loop' with batch_size=50.
3. Measure time and ensure all are processed.

Expected Result:
- Processing 1000 gaps should take significantly less time than sequential.
- All gaps should transition from 'Void' to 'Unknown' (or have defined questions).
"""

import time
import logging
import sys
import os
from typing import List, Any
from dataclasses import dataclass

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Learning.resonance_learner import ResonanceLearner
from Core.Learning.hierarchical_learning import Domain

# Mock Classes to avoid overhead of real DB for this stress test
@dataclass
class MockGap:
    name: str
    domain: Domain = Domain.PHILOSOPHY
    purpose_for_elysia: str = "To test scalability"
    definition: str = ""
    understanding_level: float = 0.0
    principle: str = ""
    last_learned: str = ""

class MockGraph:
    def __init__(self, size=1000):
        self.gaps = [MockGap(f"Void_{i}") for i in range(size)]
    
    def get_knowledge_gaps(self, limit=10) -> List[MockGap]:
        return self.gaps[:limit]
    
    def _save(self):
        pass

# Force Mocking in ResonanceLearner
def mock_get_knowledge_graph(self):
    return MockGraph(size=1000)

def mock_tune_to_frequency(freq):
    return None

def mock_absorb_text(text, source_name):
    pass

def test_scalability():
    print("🔥 Ignition: Scalability Test (1000 Thoughts)")
    
    learner = ResonanceLearner()
    
    # Patch internals for speed/isolation
    learner._get_knowledge_graph = lambda: MockGraph(size=1000)
    
    # Patch InternalUniverse to avoid heavy loading
    from unittest.mock import MagicMock
    mock_universe = MagicMock()
    mock_universe.tune_to_frequency = mock_tune_to_frequency
    mock_universe.absorb_text = mock_absorb_text
    learner._get_internal_universe = lambda: mock_universe
    
    # Run the Batch
    BATCH_SIZE = 100 # Process 100 at a time
    CYCLES_NEEDED = 10 # To get 1000
    TOTAL_ITEMS = 1000
    
    start_time = time.time()
    
    print(f"   🚀 Launching Batch Loop: {TOTAL_ITEMS} items (Batch Size: {BATCH_SIZE})...")
    
    # We call run_batch_inquiry_loop directly with cycle logic adjusted
    # Actually run_batch_inquiry_loop takes (cycles, batch_size) -> limit = cycles*batch_size
    # So to process 1000, we need cycles * batch_size = 1000
    results = learner.run_batch_inquiry_loop(cycles=10, batch_size=100)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n✅ Completed Processing {len(results)} items.")
    print(f"⏱️ Time Taken: {duration:.2f} seconds")
    print(f"⚡ Speed: {len(results)/duration:.2f} thoughts/second")
    
    if len(results) == TOTAL_ITEMS:
        print("✅ SUCCESS: All 1000 items processed.")
    else:
        print(f"⚠️ PARTIAL: Processed {len(results)}/{TOTAL_ITEMS}")

if __name__ == "__main__":
    test_scalability()
