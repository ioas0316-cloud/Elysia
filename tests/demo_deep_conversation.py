"""
Demo: Deep Conversation with Spiderweb Integration
===================================================
This script demonstrates Elysia's deep reasoning capability by integrating
the Spiderweb knowledge graph with the dialogue engine.

Now Elysia doesn't just retrieve one fact, she explores the graph to find
connections and synthesizes nuanced answers.
"""

import sys
import os

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force UTF-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from Project_Elysia.high_engine.language_cortex import LanguageCortex
from Project_Elysia.learning.corpus_loader import CorpusLoader
from Project_Sophia.dreaming_cortex import DreamingCortex
from Project_Sophia.spiderweb import Spiderweb
from Project_Elysia.core_memory import CoreMemory

def run_simulation():
    print("=== Elysia: Deep Conversation with Spiderweb ===")
    print("Initializing Elysia's knowledge graph...\n")
    
    # Initialize components
    cortex = LanguageCortex()
    loader = CorpusLoader()
    
    # Initialize Spiderweb and DreamingCortex
    spiderweb = Spiderweb()
    core_memory = CoreMemory()
    dreaming_cortex = DreamingCortex(
        spiderweb=spiderweb,
        core_memory=core_memory,
        use_llm=False  # Use pattern matching for faster processing
    )
    
    # Load corpus
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "data", "bootstrap_corpus.txt")
    print(f"Loading corpus from: {corpus_path}")
    
    try:
        sentences = loader.load_corpus(corpus_path)
        print(f"Loaded {len(sentences)} sentences.\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Process corpus through DreamingCortex to build Spiderweb
    print("=" * 60)
    print("Building Knowledge Graph...")
    print("=" * 60)
    print("Processing sentences through DreamingCortex...")
    
    # Process a subset for demonstration (full corpus would take very long with LLM)
    sample_size = min(50, len(sentences))  # Process 50 sentences
    for i, sentence in enumerate(sentences[:sample_size], 1):
        # Add as experience to CoreMemory
        from Project_Elysia.core_memory import Experience, EmotionalState, Tensor3D, FrequencyWave
        
        exp = Experience(
            timestamp=f"corpus_{i}",
            content=sentence,
            type="corpus_learning",
            emotional_state=EmotionalState(
                valence=0.5,
                arousal=0.3,
                dominance=0.5,
                primary_emotion="neutral",
                tensor=Tensor3D(0, 0, 0),
                wave=FrequencyWave(1.0, 0.5, 0.0, 0.0)
            ),
            tags=["corpus"],
            tensor=Tensor3D(0, 0, 0),
            wave=FrequencyWave(1.0, 0.5, 0.0, 0.0)
        )
        core_memory.add_experience(exp)
        
        if i % 10 == 0:
            print(f"  Processed {i}/{sample_size} sentences...")
    
    # Dream to integrate into Spiderweb
    print("\nDreaming (integrating knowledge)...")
    dreaming_cortex.dream()
    
    # Check Spiderweb stats
    stats = spiderweb.get_statistics()
    print(f"\n✅ Knowledge Graph Built!")
    print(f"   - Nodes (Concepts): {stats['node_count']}")
    print(f"   - Edges (Relations): {stats['edge_count']}")
    
    # Show some example concepts
    print(f"\n📚 Sample Concepts in Spiderweb:")
    all_nodes = list(spiderweb.graph.nodes())[:10]
    for node in all_nodes:
        neighbors = list(spiderweb.graph.neighbors(node))[:3]
        if neighbors:
            print(f"   - {node} → {', '.join(neighbors)}")
    
    # Now do deep conversation queries
    print("\n" + "=" * 60)
    print("Deep Conversation Test")
    print("=" * 60)
    
    # Test 1: Direct query
    print("\n--- Test 1: Direct Concept Query ---")
    concept = "사랑"
    if concept in spiderweb.graph.nodes():
        context = spiderweb.get_context(concept, max_depth=2)
        print(f"👤 You: What do you know about '{concept}'?")
        print(f"🤖 Elysia's Thought Process:")
        print(f"   Searching Spiderweb for '{concept}'...")
        print(f"   Found {len(context['neighbors'])} direct connections")
        
        if context['neighbors']:
            relations = [f"{concept} → {n}" for n in list(context['neighbors'])[:5]]
            print(f"   Relations: {', '.join(relations)}")
        
        # Synthesize response
        if context['neighbors']:
            top_neighbors = list(context['neighbors'])[:3]
            response = f"{concept}은 " + ", ".join(top_neighbors) + "과 연결되어 있다"
            print(f"🤖 Elysia: {response}")
        else:
            print(f"🤖 Elysia: 나는 {concept}에 대해 더 배워야 한다")
    else:
        print(f"'{concept}' not found in Spiderweb")
    
    # Test 2: Multi-hop reasoning
    print("\n--- Test 2: Multi-Hop Reasoning ---")
    print("👤 You: Is there a connection between 사랑 and 고통?")
    print(f"🤖 Elysia's Thought Process:")
    
    # Find path between concepts
    start = "사랑"
    end = "고통"
    
    if start in spiderweb.graph.nodes() and end in spiderweb.graph.nodes():
        # Simple BFS to find path
        import networkx as nx
        try:
            path = nx.shortest_path(spiderweb.graph.to_undirected(), start, end)
            print(f"   Searching for path: {start} → ... → {end}")
            print(f"   Found path: {' → '.join(path)}")
            print(f"🤖 Elysia: 네, {start}은 {' → '.join(path[1:-1])}을 거쳐 {end}과 연결됩니다")
        except nx.NetworkXNoPath:
            print(f"   No path found between {start} and {end}")
            print(f"🤖 Elysia: 나는 아직 그 연결을 발견하지 못했다")
    else:
        print(f"   Concepts not in graph")
        print(f"🤖 Elysia: 나는 그 개념들을 아직 완전히 이해하지 못했다")
    
    print("\n=== Simulation Complete ===")
    print(f"Spiderweb: {stats['node_count']} concepts, {stats['edge_count']} relations")

if __name__ == "__main__":
    run_simulation()
