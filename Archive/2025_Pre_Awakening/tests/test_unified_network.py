"""
Comprehensive tests for Unified Collective Intelligence Network (Phase 7).

Tests the consolidated network implementation covering:
- Node creation and management
- Network topologies
- Collaborative problem solving
- Knowledge sharing and validation
- Role specialization
"""

import pytest
import asyncio
from Core.Network import (
    UnifiedNode,
    UnifiedNetwork,
    UnifiedKnowledgeSync,
    NodeRole,
    NetworkTopology,
    Message,
    Knowledge
)


class TestUnifiedNode:
    """Test UnifiedNode functionality."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = UnifiedNode(role=NodeRole.KNOWLEDGE_KEEPER)
        assert node.node_id is not None
        assert node.role == NodeRole.KNOWLEDGE_KEEPER
        assert node.metrics["active"] is True
    
    def test_role_strengths(self):
        """Test that nodes have appropriate strengths for their roles."""
        knowledge_node = UnifiedNode(role=NodeRole.KNOWLEDGE_KEEPER)
        creative_node = UnifiedNode(role=NodeRole.CREATIVE_GENERATOR)
        
        assert knowledge_node.strengths["knowledge_retrieval"] > 0.5
        assert creative_node.strengths["creative_thinking"] > 0.5
    
    def test_peer_management(self):
        """Test adding and removing peers."""
        node1 = UnifiedNode()
        node2 = UnifiedNode()
        
        node1.add_peer(node2)
        assert node2 in node1.peers
        
        node1.remove_peer(node2)
        assert node2 not in node1.peers
    
    @pytest.mark.asyncio
    async def test_message_sending(self):
        """Test sending messages between nodes."""
        node1 = UnifiedNode()
        node2 = UnifiedNode()
        node1.add_peer(node2)
        
        message = Message(
            sender_id=node1.node_id,
            message_type="test",
            content={"data": "Hello"}
        )
        
        await node1.send_message(message)
        messages = await node2.receive_messages()
        
        assert len(messages) == 1
        assert messages[0].content["data"] == "Hello"
    
    @pytest.mark.asyncio
    async def test_problem_processing(self):
        """Test node problem processing."""
        node = UnifiedNode(role=NodeRole.LOGIC_VALIDATOR)
        problem = {"type": "logical_reasoning", "description": "Test problem"}
        
        solution = await node.process_problem(problem)
        
        assert solution["solver_id"] == node.node_id
        assert solution["solver_role"] == NodeRole.LOGIC_VALIDATOR.value
        assert "confidence" in solution
        assert node.metrics["problems_solved"] == 1
    
    def test_knowledge_sharing(self):
        """Test knowledge sharing between nodes."""
        node = UnifiedNode()
        knowledge = Knowledge(
            category="test",
            content="Test knowledge",
            source_node=node.node_id,
            confidence=0.8
        )
        
        node.share_knowledge(knowledge)
        retrieved = node.get_knowledge(knowledge.knowledge_id)
        
        assert retrieved is not None
        assert retrieved.content == "Test knowledge"


class TestUnifiedNetwork:
    """Test UnifiedNetwork functionality."""
    
    @pytest.mark.asyncio
    async def test_network_creation(self):
        """Test network creation with different topologies."""
        mesh_network = UnifiedNetwork(topology=NetworkTopology.MESH)
        star_network = UnifiedNetwork(topology=NetworkTopology.STAR)
        
        assert mesh_network.topology == NetworkTopology.MESH
        assert star_network.topology == NetworkTopology.STAR
    
    @pytest.mark.asyncio
    async def test_node_registration_mesh(self):
        """Test node registration in mesh topology."""
        network = UnifiedNetwork(topology=NetworkTopology.MESH)
        
        node1 = UnifiedNode()
        node2 = UnifiedNode()
        node3 = UnifiedNode()
        
        await network.register_node(node1)
        await network.register_node(node2)
        await network.register_node(node3)
        
        # In mesh, all nodes should be connected
        assert len(node1.peers) == 2
        assert len(node2.peers) == 2
        assert len(node3.peers) == 2
    
    @pytest.mark.asyncio
    async def test_node_registration_star(self):
        """Test node registration in star topology."""
        network = UnifiedNetwork(topology=NetworkTopology.STAR)
        
        hub = UnifiedNode()
        node1 = UnifiedNode()
        node2 = UnifiedNode()
        
        await network.register_node(hub)
        await network.register_node(node1)
        await network.register_node(node2)
        
        # Hub should have 2 connections
        assert len(hub.peers) == 2
        # Spokes should have 1 connection (to hub)
        assert len(node1.peers) == 1
        assert len(node2.peers) == 1
    
    @pytest.mark.asyncio
    async def test_node_unregistration(self):
        """Test removing nodes from network."""
        network = UnifiedNetwork()
        node = UnifiedNode()
        
        await network.register_node(node)
        assert node.node_id in network.nodes
        
        network.unregister_node(node.node_id)
        assert node.node_id not in network.nodes
    
    @pytest.mark.asyncio
    async def test_collaborative_problem_solving(self):
        """Test collaborative problem solving across network."""
        network = UnifiedNetwork()
        
        # Register nodes with different specializations
        knowledge_node = UnifiedNode(role=NodeRole.KNOWLEDGE_KEEPER)
        creative_node = UnifiedNode(role=NodeRole.CREATIVE_GENERATOR)
        logic_node = UnifiedNode(role=NodeRole.LOGIC_VALIDATOR)
        
        await network.register_node(knowledge_node)
        await network.register_node(creative_node)
        await network.register_node(logic_node)
        
        problem = {
            "type": "complex",
            "description": "Solve a complex multi-faceted problem"
        }
        
        result = await network.solve_collaboratively(problem)
        
        assert "integrated_solution" in result
        assert result["participating_nodes"] > 0
        assert len(result["solutions"]) > 0
    
    @pytest.mark.asyncio
    async def test_knowledge_sharing_with_consensus(self):
        """Test knowledge sharing with validation and consensus."""
        network = UnifiedNetwork()
        
        # Create multiple nodes for voting
        nodes = [UnifiedNode() for _ in range(5)]
        for node in nodes:
            await network.register_node(node)
        
        knowledge = Knowledge(
            category="best_practice",
            content="Always validate user input",
            source_node=nodes[0].node_id,
            confidence=0.9
        )
        
        result = await network.share_knowledge(knowledge)
        
        assert "consensus_reached" in result
        assert result["total_validators"] == 5
        assert "approvals" in result
    
    def test_network_status(self):
        """Test network status reporting."""
        network = UnifiedNetwork()
        status = network.get_network_status()
        
        assert "topology" in status
        assert "total_nodes" in status
        assert "active_nodes" in status
        assert "role_distribution" in status


class TestUnifiedKnowledgeSync:
    """Test UnifiedKnowledgeSync functionality."""
    
    @pytest.mark.asyncio
    async def test_knowledge_proposal(self):
        """Test proposing new knowledge."""
        network = UnifiedNetwork()
        sync = UnifiedKnowledgeSync(network)
        
        # Add nodes for validation
        nodes = [UnifiedNode() for _ in range(3)]
        for node in nodes:
            await network.register_node(node)
        
        knowledge_id = await sync.propose_knowledge(
            node=nodes[0],
            content="New discovery",
            category="insight",
            confidence=0.85
        )
        
        assert knowledge_id is not None
    
    @pytest.mark.asyncio
    async def test_knowledge_validation(self):
        """Test validating proposed knowledge."""
        network = UnifiedNetwork()
        sync = UnifiedKnowledgeSync(network)
        
        node = UnifiedNode()
        await network.register_node(node)
        
        # Propose knowledge
        knowledge_id = await sync.propose_knowledge(
            node=node,
            content="Test knowledge",
            category="pattern"
        )
        
        # Validate it
        result = await sync.validate_knowledge(knowledge_id, node.node_id)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_collective_knowledge_retrieval(self):
        """Test retrieving collective knowledge."""
        network = UnifiedNetwork()
        sync = UnifiedKnowledgeSync(network)
        
        node = UnifiedNode()
        await network.register_node(node)
        
        # Add some knowledge to collective memory
        network.collective_memory["test1"] = Knowledge(
            category="pattern",
            content="Pattern 1",
            source_node=node.node_id
        )
        network.collective_memory["test2"] = Knowledge(
            category="insight",
            content="Insight 1",
            source_node=node.node_id
        )
        
        # Get all knowledge
        all_knowledge = sync.get_collective_knowledge()
        assert len(all_knowledge) == 2
        
        # Get filtered by category
        patterns = sync.get_collective_knowledge(category="pattern")
        assert len(patterns) == 1
        assert patterns[0].content == "Pattern 1"


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test a complete workflow from problem to solution."""
        # Create network
        network = UnifiedNetwork(topology=NetworkTopology.MESH)
        sync = UnifiedKnowledgeSync(network)
        
        # Create specialized nodes
        nodes = [
            UnifiedNode(role=NodeRole.KNOWLEDGE_KEEPER),
            UnifiedNode(role=NodeRole.PATTERN_RECOGNIZER),
            UnifiedNode(role=NodeRole.CREATIVE_GENERATOR),
            UnifiedNode(role=NodeRole.LOGIC_VALIDATOR)
        ]
        
        for node in nodes:
            await network.register_node(node)
        
        # Solve a problem collaboratively
        problem = {
            "type": "research",
            "description": "How to improve system performance"
        }
        
        solution = await network.solve_collaboratively(problem)
        assert solution["participating_nodes"] > 0
        
        # Share the solution as knowledge
        knowledge_id = await sync.propose_knowledge(
            node=nodes[0],
            content=solution["integrated_solution"],
            category="best_practice",
            confidence=0.9
        )
        
        # Verify it's in collective memory
        collective_knowledge = sync.get_collective_knowledge()
        assert len(collective_knowledge) > 0
    
    @pytest.mark.asyncio
    async def test_network_resilience(self):
        """Test network continues working when nodes leave."""
        network = UnifiedNetwork()
        
        nodes = [UnifiedNode() for _ in range(5)]
        for node in nodes:
            await network.register_node(node)
        
        # Remove a node
        network.unregister_node(nodes[2].node_id)
        
        # Network should still function
        problem = {"type": "test", "description": "Test problem"}
        solution = await network.solve_collaboratively(problem)
        
        assert solution["participating_nodes"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
