"""
Demo: Deep Reasoning with Spiderweb (Simplified)
=================================================
This is a simplified demonstration of Elysia's deep reasoning capability using Spiderweb.
We manually populate the knowledge graph to demonstrate multi-hop reasoning clearly.
"""

import sys
import os

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force UTF-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from Core.FoundationLayer.Foundation.spiderweb import Spiderweb
import networkx as nx

def run_simulation():
    print("=== Elysia: Deep Reasoning Demonstration ===")
    print("Building Elysia's knowledge graph...\n")
    
    # Initialize Spiderweb
    spiderweb = Spiderweb()
    
    # Manually populate with knowledge from our corpus
    # This demonstrates what DreamingCortex would ideally extract
    
    print("📚 Populating Knowledge Graph...")
    
    # Add concepts
    concepts = [
        "사랑", "고통", "희생", "배움", "진실", "자유",
        "시간", "변화", "성장", "연결", "존재", "의식"
    ]
    
    for concept in concepts:
        spiderweb.add_node(concept, type="concept")
    
    # Add relationships (from our corpus)
    relationships = [
        ("사랑", "희생", "requires"),    # 사랑은 희생이다
        ("사랑", "고통", "causes"),      # 사랑은 고통을 낳는다
        ("고통", "배움", "becomes"),     # 고통은 배움이다
        ("고통", "성장", "enables"),     # 고통은 성장을 만든다
        ("진실", "자유", "gives"),       # 진실은 자유를 준다
        ("시간", "변화", "creates"),     # 시간은 변화를 만든다
        ("변화", "성장", "is_a"),        # 변화는 성장이다
        ("연결", "존재", "defines"),     # 연결은 존재를 정의한다
        ("의식", "존재", "is_a"),        # 의식은 존재이다
        ("희생", "고통", "is_a"),        # 희생은 고통이다
    ]
    
    for source, target, relation in relationships:
        spiderweb.add_link(source, target, relation)
    
    stats = spiderweb.get_statistics()
    print(f"✅ Knowledge Graph Built!")
    print(f"   - Concepts: {stats['node_count']}")
    print(f"   - Relations: {stats['edge_count']}\n")
    
    # Show the graph structure
    print("📊 Knowledge Graph Structure:")
    for concept in ["사랑", "고통", "진실"]:
        if concept in spiderweb.graph.nodes():
            neighbors = list(spiderweb.graph.successors(concept))
            if neighbors:
                print(f"   {concept} → {', '.join(neighbors)}")
    
    print("\n" + "=" * 60)
    print("Deep Reasoning Tests")
    print("=" * 60)
    
    # Test 1: Direct query
    print("\n--- Test 1: What does Elysia know about '사랑'? ---")
    concept = "사랑"
    if concept in spiderweb.graph.nodes():
        context = spiderweb.get_context(concept)
        print(f"👤 You: Tell me about '{concept}'")
        print(f"🤖 Elysia's Thought Process:")
        print(f"   Searching graph for '{concept}'...")
        
        outgoing = [c for c in context if c['direction'] == 'outgoing']
        if outgoing:
            relations_str = ', '.join([f"{c['node']} ({c['relation']})" for c in outgoing])
            print(f"   Found: {concept} connects to {relations_str}")
            
            # Synthesize response
            concepts_connected = [c['node'] for c in outgoing]
            response = f"{concept}은 {', '.join(concepts_connected)}과 연결되어 있다"
            print(f"🤖 Elysia: {response}")
    
    # Test 2: Multi-hop reasoning
    print("\n--- Test 2: Path from 사랑 to 배움 ---")
    start = "사랑"
    end = "배움"
    print(f"👤 You: Is there a connection between '{start}' and '{end}'?")
    print(f"🤖 Elysia's Thought Process:")
    print(f"   Searching for path: {start} → ... → {end}")
    
    try:
        path = nx.shortest_path(spiderweb.graph, start, end)
        print(f"   Found path: {' → '.join(path)}")
        
        # Generate natural language explanation
        path_explanation = []
        for i in range(len(path) - 1):
            edge_data = spiderweb.graph.get_edge_data(path[i], path[i+1])
            relation = edge_data.get('relation', 'leads_to')
            path_explanation.append(f"{path[i]} {relation} {path[i+1]}")
        
        print(f"   Reasoning: {', '.join(path_explanation)}")
        print(f"🤖 Elysia: 네, {start}은 {' 그리고 '.join(path[1:-1])}을 거쳐 {end}과 연결됩니다")
    except nx.NetworkXNoPath:
        print(f"   No path found")
        print(f"🤖 Elysia: 나는 직접적인 연결을 찾지 못했다")
    
    # Test 3: Inference (transitive reasoning)
    print("\n--- Test 3: Transitive Inference ---")
    print(f"👤 You: If 사랑 requires 희생, and 희생 is 고통, what can you infer?")
    print(f"🤖 Elysia's Thought Process:")
    
    # Check if both paths exist
    path1_exists = spiderweb.graph.has_edge("사랑", "희생")
    path2_exists = spiderweb.graph.has_edge("희생", "고통")
    
    if path1_exists and path2_exists:
        print(f"   Step 1: 사랑 → 희생 (verified)")
        print(f"   Step 2: 희생 → 고통 (verified)")
        print(f"   Inference: Therefore, 사랑 → 고통")
        print(f"🤖 Elysia: 사랑은 희생을 필요로 하고, 희생은 고통이다. 따라서 사랑은 고통을 낳는다고 추론할 수 있다")
    else:
        print(f"🤖 Elysia: 나는 그 추론을 확인할 수 없다")
    
    # Test 4: Complex query
    print("\n--- Test 4: What leads to 성장? ---")
    target = "성장"
    print(f"👤 You: What are all the paths that lead to '{target}'?")
    print(f"🤖 Elysia's Thought Process:")
    
    # Find all predecessors
    predecessors = list(spiderweb.graph.predecessors(target))
    if predecessors:
        print(f"   Direct paths to {target}: {', '.join(predecessors)}")
        
        # Find 2-hop paths
        two_hop = set()
        for pred in predecessors:
            for pred2 in spiderweb.graph.predecessors(pred):
                two_hop.add(f"{pred2} → {pred} → {target}")
        
        if two_hop:
            print(f"   Multi-hop paths:")
            for path in list(two_hop)[:3]:  # Show first 3
                print(f"      {path}")
        
        print(f"🤖 Elysia: {target}으로 가는 여러 경로가 있다: {', '.join(predecessors)} 등을 통해 도달할 수 있다")
    else:
        print(f"🤖 Elysia: 나는 {target}으로 가는 경로를 모른다")
    
    print("\n=== Demonstration Complete ===")
    print(f"\nThis shows Elysia's potential for:")
    print(f"  ✅ Knowledge graph navigation")
    print(f"  ✅ Multi-hop reasoning (A → B → C)")
    print(f"  ✅ Transitive inference")
    print(f"  ✅ Path finding between concepts")

if __name__ == "__main__":
    run_simulation()
