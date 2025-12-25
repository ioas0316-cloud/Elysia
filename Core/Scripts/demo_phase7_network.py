"""
Phase 7 Demo: Collective Intelligence Network

Demonstrates:
1. Multi-node collaboration
2. Knowledge sharing and validation
3. Role specialization and load balancing
4. Integrated network intelligence
"""

import asyncio
from Core.03_Interaction.04_Network.Network import (
    ElysiaNode, ElysiaNetwork, KnowledgeSync, 
    SpecializationManager, Discovery, Role
)


async def demo_1_network_collaboration():
    """Demo 1: Multi-node collaborative problem solving."""
    print("\n" + "="*70)
    print("DEMO 1: Multi-Node Collaborative Problem Solving")
    print("="*70)
    
    # Create a network with different specialized nodes
    network = ElysiaNetwork(topology="mesh")
    
    # Create nodes with different specializations
    nodes = [
        ElysiaNode(specialization="logic"),
        ElysiaNode(specialization="creativity"),
        ElysiaNode(specialization="emotion"),
        ElysiaNode(specialization="pattern"),
        ElysiaNode(specialization="integration")
    ]
    
    # Add all nodes to network
    for node in nodes:
        network.add_node(node)
    
    print(f"\nCreated network with {len(nodes)} specialized nodes")
    print(f"Topology: {network.topology}")
    
    # Test collaborative problem solving
    problem = "Design an AI system that is both logical, creative, and emotionally intelligent"
    
    print(f"\n📝 Problem: {problem}")
    print("\nSolving collaboratively...")
    
    solution = await network.collaborative_problem_solving(problem)
    
    print(f"\n✅ Integrated Solution:")
    print(f"   {solution['integrated_solution']}")
    print(f"\n📊 Metrics:")
    print(f"   Confidence: {solution['confidence']:.2f}")
    print(f"   Contributors: {solution['num_perspectives']} nodes")
    print(f"   Processing Time: {solution['processing_time']:.3f}s")
    print(f"   Subproblems: {len(solution['subproblems'])}")
    
    return network


async def demo_2_knowledge_sharing():
    """Demo 2: Knowledge sharing and validation across network."""
    print("\n" + "="*70)
    print("DEMO 2: Knowledge Sharing and Validation")
    print("="*70)
    
    # Create knowledge sync system
    knowledge_sync = KnowledgeSync()
    
    # Create some validator nodes
    validators = [
        ElysiaNode(specialization="logic"),
        ElysiaNode(specialization="pattern"),
        ElysiaNode(specialization="knowledge")
    ]
    
    # Create discoveries to share
    discoveries = [
        Discovery(
            content={"insight": "Combining logic and creativity leads to innovation"},
            source_node_id=validators[0].node_id,
            confidence=0.8,
            category="insight"
        ),
        Discovery(
            content={"pattern": "High emotional intelligence correlates with better collaboration"},
            source_node_id=validators[1].node_id,
            confidence=0.75,
            category="pattern"
        ),
        Discovery(
            content={"best_practice": "Always validate knowledge through multiple perspectives"},
            source_node_id=validators[2].node_id,
            confidence=0.9,
            category="best_practice"
        )
    ]
    
    print(f"\nSharing {len(discoveries)} discoveries with validation...")
    
    # Share each discovery
    for i, discovery in enumerate(discoveries, 1):
        print(f"\n📤 Discovery {i}: {discovery.category}")
        print(f"   Content: {list(discovery.content.values())[0]}")
        print(f"   Initial Confidence: {discovery.confidence:.2f}")
        
        # Share and validate
        accepted = await knowledge_sync.share_discovery(discovery, validators)
        
        if accepted:
            print(f"   ✅ Accepted after validation")
            print(f"   Validations: {len(discovery.validations)}")
            final_confidence = knowledge_sync.evaluate_confidence(discovery)
            print(f"   Final Confidence: {final_confidence:.2f}")
        else:
            print(f"   ❌ Rejected - did not meet consensus threshold")
    
    # Show knowledge stats
    stats = knowledge_sync.get_stats()
    print(f"\n📊 Knowledge Base Stats:")
    print(f"   Total Shared Knowledge: {stats['total_knowledge']}")
    print(f"   By Category: {stats['by_category']}")
    print(f"   Average Confidence: {stats['avg_confidence']:.2f}")
    
    return knowledge_sync


async def demo_3_role_specialization():
    """Demo 3: Dynamic role assignment and load balancing."""
    print("\n" + "="*70)
    print("DEMO 3: Role Specialization and Load Balancing")
    print("="*70)
    
    # Create specialization manager
    spec_manager = SpecializationManager()
    
    # Create nodes with varied strengths
    nodes = [
        ElysiaNode(specialization="logic"),
        ElysiaNode(specialization="creativity"),
        ElysiaNode(specialization="emotion"),
        ElysiaNode(specialization="pattern"),
        ElysiaNode(specialization="knowledge"),
        ElysiaNode(specialization="integration")
    ]
    
    print(f"\nCreated {len(nodes)} nodes with different specializations")
    
    # Assign roles based on strengths
    print("\n🎯 Assigning roles based on node strengths...")
    spec_manager.assign_roles(nodes)
    
    print("\n📋 Role Assignments:")
    for node in nodes:
        role = spec_manager.get_node_role(node)
        if role:
            print(f"   Node ({node.specialization}): {role.value}")
    
    # Show distribution
    distribution = spec_manager.get_role_distribution()
    print(f"\n📊 Role Distribution:")
    for role_name, count in distribution.items():
        if count > 0:
            print(f"   {role_name}: {count} node(s)")
    
    # Check if balanced
    is_balanced = spec_manager.is_balanced()
    print(f"\n⚖️  Network Balance: {'✅ Balanced' if is_balanced else '❌ Needs Rebalancing'}")
    
    # Simulate load and rebalance
    print("\n🔄 Simulating load increase and rebalancing...")
    
    # Artificially increase load on one role
    spec_manager.role_loads[Role.LOGIC_VALIDATOR] = 15  # Over threshold
    
    print(f"   Overloaded roles: {[r.value for r in spec_manager.identify_overloaded_roles()]}")
    
    # Rebalance
    await spec_manager.dynamic_rebalancing(nodes)
    
    print(f"   ✅ Rebalancing complete")
    print(f"   New balance status: {'✅ Balanced' if spec_manager.is_balanced() else '❌ Still unbalanced'}")
    
    return spec_manager


async def demo_4_integrated_network():
    """Demo 4: Integrated network with all systems working together."""
    print("\n" + "="*70)
    print("DEMO 4: Integrated Collective Intelligence Network")
    print("="*70)
    
    # Create integrated network with all components
    network = ElysiaNetwork(topology="mesh")
    knowledge_sync = KnowledgeSync()
    spec_manager = SpecializationManager()
    
    # Create diverse node network
    nodes = [
        ElysiaNode(specialization="logic"),
        ElysiaNode(specialization="creativity"),
        ElysiaNode(specialization="emotion"),
        ElysiaNode(specialization="pattern"),
        ElysiaNode(specialization="knowledge"),
        ElysiaNode(specialization="integration")
    ]
    
    # Add nodes to network
    for node in nodes:
        network.add_node(node)
    
    # Assign specialized roles
    spec_manager.assign_roles(nodes)
    
    print(f"\n🌐 Integrated Network Status:")
    net_status = network.get_network_status()
    print(f"   Nodes: {net_status['num_nodes']}")
    print(f"   Topology: {net_status['topology']}")
    print(f"   Specializations: {net_status['specializations']}")
    
    # Complex multi-faceted problem
    complex_problem = """
    Create a system that can learn from experiences, generate creative solutions,
    understand emotions, recognize patterns, and integrate all these capabilities
    """
    
    print(f"\n🎯 Complex Problem:")
    print(f"   {complex_problem.strip()}")
    
    # Solve collaboratively
    print("\n💡 Solving with full network collaboration...")
    solution = await network.collaborative_problem_solving(complex_problem)
    
    print(f"\n✅ Solution:")
    print(f"   {solution['integrated_solution']}")
    print(f"\n📊 Solution Metrics:")
    print(f"   Confidence: {solution['confidence']:.2f}")
    print(f"   Perspectives: {solution['num_perspectives']}")
    print(f"   Processing Time: {solution['processing_time']:.3f}s")
    
    # Extract insights as discoveries
    print("\n📚 Extracting insights as shared knowledge...")
    discovery = Discovery(
        content={"solution": solution['integrated_solution']},
        source_node_id="integrated_network",
        confidence=solution['confidence'],
        category="insight"
    )
    
    # Share with network
    accepted = await knowledge_sync.share_discovery(discovery, nodes[:3])
    
    if accepted:
        print(f"   ✅ Solution validated and added to collective knowledge")
    
    # Final stats
    print(f"\n📈 Final Network Statistics:")
    print(f"   Problems Solved: {net_status['total_problems_solved']}")
    print(f"   Shared Knowledge Items: {knowledge_sync.get_stats()['total_knowledge']}")
    print(f"   Role Balance: {spec_manager.is_balanced()}")
    
    print("\n✨ Collective intelligence network demonstration complete!")


async def main():
    """Run all Phase 7 demonstrations."""
    print("\n" + "="*70)
    print("PHASE 7: COLLECTIVE INTELLIGENCE NETWORK - COMPREHENSIVE DEMO")
    print("Demonstrating multi-instance collaboration and knowledge sharing")
    print("="*70)
    
    # Run all demos
    await demo_1_network_collaboration()
    await demo_2_knowledge_sharing()
    await demo_3_role_specialization()
    await demo_4_integrated_network()
    
    print("\n" + "="*70)
    print("ALL PHASE 7 DEMONSTRATIONS COMPLETE ✅")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
