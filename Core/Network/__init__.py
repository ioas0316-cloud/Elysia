"""
Network module for Elysia - Collective Intelligence and Communication.

This module enables Elysia instances to communicate, collaborate, and share knowledge
across a distributed network.

Phase 7: Collective Intelligence Network Implementation (CONSOLIDATED)
"""

# Unified implementation (recommended)
from .unified_collective_intelligence import (
    UnifiedNode,
    UnifiedNetwork,
    UnifiedKnowledgeSync,
    NodeRole,
    NetworkTopology,
    Message,
    Knowledge
)

# Legacy implementations (for backward compatibility)
from .collective_intelligence import CollectiveIntelligence, NetworkNode
from .knowledge_sharing import KnowledgeSharer, SharedKnowledge, KnowledgeType
from .elysia_node import ElysiaNode
from .elysia_network import ElysiaNetwork
from .knowledge_sync import KnowledgeSync, Discovery
from .role_specialization import SpecializationManager, Role

__all__ = [
    # Unified (recommended)
    "UnifiedNode",
    "UnifiedNetwork",
    "UnifiedKnowledgeSync",
    "NodeRole",
    "NetworkTopology",
    "Message",
    "Knowledge",
    # Legacy (backward compatibility)
    "CollectiveIntelligence",
    "NetworkNode",
    "KnowledgeSharer",
    "SharedKnowledge",
    "KnowledgeType",
    "ElysiaNode",
    "ElysiaNetwork",
    "KnowledgeSync",
    "Discovery",
    "SpecializationManager",
    "Role",
]
