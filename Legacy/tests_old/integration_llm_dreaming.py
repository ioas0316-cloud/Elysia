
"""
실제 Gemini API를 사용한 통합 테스트
주의: API 키가 필요하며 실제 API 호출이 발생합니다.
"""

import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.spiderweb import Spiderweb
from Project_Sophia.dreaming_cortex import DreamingCortex
from Project_Elysia.core_memory import CoreMemory, Experience
from datetime import datetime

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("LLM-Enhanced DreamingCortex Integration Test")
    print("=" * 60)
    
    # Initialize components
    core_memory = CoreMemory(file_path=None)  # In-memory only
    spiderweb = Spiderweb()
    dreaming_cortex = DreamingCortex(core_memory, spiderweb, use_llm=True)
    
    # Add test experiences
    test_experiences = [
        "I touched fire and got burned. It was very painful and I learned to be careful.",
        "Fire gives warmth and light but can also cause destruction.",
        "Water extinguishes fire and prevents burns."
    ]
    
    print("\n📝 Adding experiences to CoreMemory...")
    for i, content in enumerate(test_experiences):
        exp = Experience(
            timestamp=datetime.now().isoformat() + f"_{i}",
            content=content,
            type="episode"
        )
        core_memory.add_experience(exp)
        print(f"  {i+1}. {content}")
    
    # Run dream cycle
    print("\n💭 Entering dream state (calling Gemini API)...")
    dreaming_cortex.dream()
    
    # Analyze results
    print("\n🕸️  Spiderweb Analysis:")
    print(f"  Total nodes: {spiderweb.graph.number_of_nodes()}")
    print(f"  Total edges: {spiderweb.graph.number_of_edges()}")
    
    # Find concept nodes
    concepts = [n for n, data in spiderweb.graph.nodes(data=True) if data.get('type') == 'concept']
    print(f"\n🧠 Extracted Concepts ({len(concepts)}):")
    for concept in sorted(concepts):
        print(f"  - {concept}")
    
    # Find causal relations
    print("\n🔗 Causal Relations:")
    for source, target, data in spiderweb.graph.edges(data=True):
        if data.get('relation') in ['causes', 'prevents', 'enables']:
            print(f"  {source} -[{data['relation']}]-> {target} (weight: {data.get('weight', 0)})")
    
    # Test pathfinding
    if 'fire' in spiderweb.graph and 'burn' in spiderweb.graph:
        path = spiderweb.find_path('fire', 'burn')
        if path:
            print(f"\n🎯 Causal Chain (fire → burn): {' → '.join(path)}")
    
    print("\n✅ Integration test complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
