"""
Test: Concrete Learning with Graph Node Creation
=================================================

This tests whether the learning system now creates:
1. Actual TorchGraph nodes (not just coordinate_map entries)
2. Real content stored in nodes (definitions, not just labels)
3. Edges connecting to existing knowledge
"""

import sys
import os
sys.path.insert(0, "c:\\Elysia")
os.chdir("c:\\Elysia")

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

def test_concrete_learning():
    print("\n" + "="*60)
    print("TEST: Concrete Learning (Graph Nodes + Content)")
    print("="*60)
    
    # 1. Check initial graph state
    try:
        from Core.01_Foundation.05_Foundation_Base.Foundation.Graph.torch_graph import get_torch_graph
        graph = get_torch_graph()
        initial_nodes = len(graph.id_to_idx)
        print(f"\n1. Initial graph nodes: {initial_nodes}")
    except Exception as e:
        print(f"Could not load graph: {e}")
        return
    
    # 2. Learn a specific concept with real content
    print(f"\n2. Learning a concept...")
    
    from Core.01_Foundation.05_Foundation_Base.Foundation.external_data_connector import ExternalDataConnector
    
    connector = ExternalDataConnector()
    
    # Real content (like from Wikipedia)
    concept = "Neural_Plasticity"
    content = """
    Neural plasticity, also known as neuroplasticity or brain plasticity, 
    is the ability of neural networks in the brain to change through growth 
    and reorganization. These changes range from individual neuron pathways 
    making new connections, to systematic adjustments like cortical remapping.
    
    Neuroplasticity was once thought by neuroscientists to manifest only 
    during childhood, but research in the latter half of the 20th century 
    showed that many aspects of the brain can be altered even through adulthood.
    
    Key mechanisms include synaptic plasticity (strengthening/weakening of 
    synapses), neurogenesis (creation of new neurons), and structural changes.
    """
    
    result = connector.internalize_from_text(concept, content)
    
    print(f"\n3. Learning result:")
    print(f"   Concept: {result['concept']}")
    print(f"   Graph stored: {result.get('graph_stored', False)}")
    print(f"   Related concepts: {result.get('related_concepts', [])}")
    print(f"   Text length: {result['text_length']}")
    
    # 3. Check graph after learning
    final_nodes = len(graph.id_to_idx)
    print(f"\n4. Final graph nodes: {final_nodes}")
    print(f"   New nodes added: {final_nodes - initial_nodes}")
    
    # 4. Verify node has content
    if concept in graph.id_to_idx:
        idx = graph.id_to_idx[concept]
        if hasattr(graph, 'node_metadata') and concept in graph.node_metadata:
            meta = graph.node_metadata[concept]
            print(f"\n5. Node content verification:")
            print(f"   Type: {meta.get('type', 'unknown')}")
            print(f"   Content preview: {meta.get('content', '')[:100]}...")
            print(f"   Source: {meta.get('source', 'unknown')}")
        else:
            print(f"\n5. Node exists but no metadata")
    else:
        print(f"\n5. Node NOT in graph - learning failed")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_concrete_learning()
