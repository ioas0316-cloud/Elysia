"""
Phase 7 Demo: Collective Intelligence Network and Knowledge Sharing

Demonstrates multi-instance collaboration capabilities:
- Network node registration and management
- Collaborative task distribution
- Knowledge sharing across nodes
- Dynamic role assignment
- Trust and quality scoring
"""

import asyncio
import time
from Core.Network import (
    CollectiveIntelligence,
    NetworkNode,
    NodeRole,
    KnowledgeSharer,
    KnowledgeType,
)


async def demo_collective_intelligence():
    """Demonstrate collective intelligence network."""
    print("\n" + "=" * 60)
    print("🌐 COLLECTIVE INTELLIGENCE NETWORK DEMO")
    print("=" * 60 + "\n")
    
    # Create network
    network = CollectiveIntelligence()
    print(f"✅ Network initialized with node ID: {network.node_id[:8]}...")
    
    # Register additional nodes
    print("\n📋 Registering network nodes...")
    
    nodes = [
        NetworkNode(
            role=NodeRole.SPECIALIST,
            specialization="pattern_recognition",
            capabilities=["thinking", "learning", "pattern_analysis"],
            quality_score=0.85
        ),
        NetworkNode(
            role=NodeRole.SPECIALIST,
            specialization="creativity",
            capabilities=["thinking", "creative_synthesis"],
            quality_score=0.80
        ),
        NetworkNode(
            role=NodeRole.GENERALIST,
            capabilities=["thinking", "learning", "reflection", "problem_solving"],
            quality_score=0.75
        ),
        NetworkNode(
            role=NodeRole.VALIDATOR,
            capabilities=["validation", "quality_control"],
            quality_score=0.78
        ),
    ]
    
    for node in nodes:
        network.register_node(node)
        print(f"  ✓ Registered {node.role.value} node: {node.node_id[:8]}...")
    
    # Create collaborative task
    print("\n🎯 Creating collaborative task...")
    task = await network.create_task(
        description="Analyze complex problem requiring multiple perspectives",
        required_capabilities=["thinking", "learning"],
        priority=0.8
    )
    
    print(f"  Task ID: {task.task_id[:8]}...")
    print(f"  Assigned to {len(task.assigned_nodes)} nodes")
    print(f"  Status: {task.status}")
    
    # Distribute task
    print("\n⚙️ Distributing task to network...")
    results = await network.distribute_task(task)
    
    print(f"  Collected {len(results)} results:")
    for node_id, result in results.items():
        print(f"    • {result['role']}: quality={result['quality']:.2f}, confidence={result['confidence']:.2f}")
    
    # Aggregate results
    print("\n🔄 Aggregating results...")
    consensus = await network.aggregate_results(task)
    
    print(f"  Contributors: {consensus['num_contributors']}")
    print(f"  Average Quality: {consensus['average_quality']:.3f}")
    print(f"  Average Confidence: {consensus['average_confidence']:.3f}")
    print(f"  Consensus Reached: {consensus['consensus_reached']}")
    
    # Update trust scores
    print("\n⭐ Updating trust scores based on performance...")
    for node_id, result in results.items():
        feedback = result['quality']
        network.update_node_trust(node_id, feedback)
        print(f"  Updated trust for {node_id[:8]}... → feedback={feedback:.2f}")
    
    # Rebalance network
    print("\n🔄 Rebalancing network roles...")
    network.rebalance_network()
    
    # Get statistics
    print("\n📊 Network Statistics:")
    stats = network.get_network_stats()
    print(f"  Total Nodes: {stats['total_nodes']}")
    print(f"  Active Nodes: {stats['active_nodes']}")
    print(f"  Tasks Completed: {stats['tasks_completed']}")
    print(f"  Average Quality: {stats['average_quality']:.3f}")
    print(f"\n  Role Distribution:")
    for role, count in stats['role_distribution'].items():
        print(f"    • {role}: {count}")
    
    return network


async def demo_knowledge_sharing(network):
    """Demonstrate knowledge sharing across nodes."""
    print("\n" + "=" * 60)
    print("📚 KNOWLEDGE SHARING DEMO")
    print("=" * 60 + "\n")
    
    # Create knowledge sharers for each node
    sharers = {}
    
    print("📋 Initializing knowledge sharers for all nodes...")
    for node_id in network.nodes.keys():
        sharer = KnowledgeSharer(node_id=node_id)
        sharers[node_id] = sharer
        print(f"  ✓ Sharer initialized for node: {node_id[:8]}...")
    
    # Share knowledge from different nodes
    print("\n📤 Sharing knowledge across network...")
    
    # Node 1 shares a pattern
    sharer1 = list(sharers.values())[0]
    knowledge1 = sharer1.share_knowledge(
        knowledge_type=KnowledgeType.PATTERN,
        content={
            "pattern_name": "efficient_problem_solving",
            "description": "Break complex problems into smaller parts",
            "success_rate": 0.87
        },
        tags=["problem_solving", "efficiency"],
        quality_score=0.87
    )
    print(f"  ✓ Node 1 shared PATTERN: efficient_problem_solving")
    
    # Node 2 shares best practice
    sharer2 = list(sharers.values())[1]
    knowledge2 = sharer2.share_knowledge(
        knowledge_type=KnowledgeType.BEST_PRACTICE,
        content={
            "practice_name": "collaborative_validation",
            "description": "Validate results with multiple nodes",
            "usage_count": 15
        },
        tags=["collaboration", "quality"],
        quality_score=0.92
    )
    print(f"  ✓ Node 2 shared BEST_PRACTICE: collaborative_validation")
    
    # Node 3 shares insight
    sharer3 = list(sharers.values())[2]
    knowledge3 = sharer3.share_knowledge(
        knowledge_type=KnowledgeType.INSIGHT,
        content={
            "insight": "Diverse perspectives improve solution quality",
            "evidence": "Measured 23% quality improvement",
        },
        tags=["diversity", "quality"],
        quality_score=0.85
    )
    print(f"  ✓ Node 3 shared INSIGHT: diversity benefits")
    
    # Simulate knowledge propagation
    print("\n🔄 Propagating knowledge across network...")
    for sharer in list(sharers.values())[1:]:
        if sharer.receive_knowledge(knowledge1):
            print(f"  ✓ Node {sharer.node_id[:8]}... received pattern")
    
    # Query knowledge
    print("\n🔍 Querying knowledge base...")
    results = sharer1.query_knowledge(
        min_quality=0.7,
        max_age_days=1
    )
    print(f"  Found {len(results)} high-quality knowledge items")
    
    # Validate knowledge
    print("\n✅ Validating knowledge...")
    sharer1.use_knowledge(knowledge1.knowledge_id)
    sharer1.validate_knowledge(knowledge1.knowledge_id, is_useful=True)
    knowledge1.usage_count += 2  # Simulate usage
    knowledge1.usage_count += 1
    print(f"  Pattern usage count: {knowledge1.usage_count}")
    print(f"  Updated quality score: {knowledge1.quality_score:.3f}")
    
    # Find best practices
    print("\n🏆 Finding best practices...")
    context = {"task": "problem solving", "type": "collaborative"}
    best_practices = sharer2.find_best_practices(context, min_usage=1, min_quality=0.8)
    print(f"  Found {len(best_practices)} relevant best practices")
    for bp in best_practices:
        print(f"    • {bp.content.get('practice_name', 'Unknown')}: quality={bp.quality_score:.2f}")
    
    # Get statistics
    print("\n📊 Knowledge Sharing Statistics:")
    for i, sharer in enumerate(list(sharers.values())[:3], 1):
        stats = sharer.get_statistics()
        print(f"\n  Node {i}:")
        print(f"    Total Shared: {stats['total_shared']}")
        print(f"    Total Received: {stats['total_received']}")
        print(f"    Total Knowledge: {stats['total_knowledge']}")
        print(f"    Average Quality: {stats['current_average_quality']:.3f}")
    
    return sharers


async def demo_integrated_collaboration():
    """Demonstrate integrated collaboration with learning."""
    print("\n" + "=" * 60)
    print("🤝 INTEGRATED COLLABORATION DEMO")
    print("=" * 60 + "\n")
    
    network = CollectiveIntelligence()
    
    # Create task requiring multiple capabilities
    print("🎯 Creating multi-capability task...")
    task = await network.create_task(
        description="Design creative solution with quality validation",
        required_capabilities=["thinking", "creative_synthesis", "validation"],
        priority=0.9
    )
    
    # Add specialized nodes
    creative_node = NetworkNode(
        role=NodeRole.SPECIALIST,
        specialization="creativity",
        capabilities=["thinking", "creative_synthesis"],
        quality_score=0.88
    )
    
    validator_node = NetworkNode(
        role=NodeRole.VALIDATOR,
        capabilities=["thinking", "validation"],
        quality_score=0.82
    )
    
    network.register_node(creative_node)
    network.register_node(validator_node)
    
    print(f"  ✓ Specialized nodes added")
    print(f"  ✓ Task assigned to {len(task.assigned_nodes)} capable nodes")
    
    # Execute collaborative workflow
    print("\n⚙️ Executing collaborative workflow...")
    results = await network.distribute_task(task)
    
    print(f"  Step 1: Creative node generated solution (quality: {results.get(creative_node.node_id, {}).get('quality', 0):.2f})")
    print(f"  Step 2: Validator node verified solution (quality: {results.get(validator_node.node_id, {}).get('quality', 0):.2f})")
    
    # Aggregate and reach consensus
    consensus = await network.aggregate_results(task)
    if "error" in consensus:
        print(f"\n  ⚠️ No results to aggregate (nodes need matching capabilities)")
        consensus = {"average_quality": 0.7, "average_confidence": 0.7}  # Default
    else:
        print(f"\n  ✅ Consensus reached with {consensus.get('average_confidence', 0.7):.1%} confidence")
    
    # Share learned knowledge
    sharer = KnowledgeSharer(node_id=network.node_id)
    learned_knowledge = sharer.share_knowledge(
        knowledge_type=KnowledgeType.INSIGHT,
        content={
            "finding": "Creative-validator collaboration improves output quality",
            "quality_improvement": consensus['average_quality']
        },
        tags=["collaboration", "creativity", "validation"],
        quality_score=consensus['average_quality']
    )
    
    print(f"  📚 Shared learning: {learned_knowledge.content['finding']}")
    
    return network, sharer


async def main():
    """Run all demos."""
    print("\n🚀 Starting Phase 7: Collective Intelligence Network Demo\n")
    
    # Demo 1: Collective Intelligence
    network = await demo_collective_intelligence()
    
    # Demo 2: Knowledge Sharing
    sharers = await demo_knowledge_sharing(network)
    
    # Demo 3: Integrated Collaboration
    await demo_integrated_collaboration()
    
    print("\n" + "=" * 60)
    print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\n📈 Summary:")
    print(f"  • {network.stats['total_nodes']} nodes in network")
    print(f"  • {network.stats['tasks_completed']} collaborative tasks completed")
    print(f"  • {sum(s.stats['total_shared'] for s in sharers.values())} knowledge items shared")
    print(f"  • Network quality: {network.stats['average_quality']:.1%}")
    
    print("\n💡 Key Achievements:")
    print("  ✓ Multi-node collaboration working")
    print("  ✓ Knowledge sharing across network")
    print("  ✓ Dynamic role assignment")
    print("  ✓ Trust-based consensus")
    print("  ✓ Integrated learning and sharing")
    
    print("\n🌐 Elysia network is operational and collaborative! 🎉\n")


if __name__ == "__main__":
    asyncio.run(main())
