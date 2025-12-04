"""
Network module for Elysia - Collective Intelligence and Communication.

This module enables Elysia instances to communicate, collaborate, and share knowledge
across a distributed network.

Phase 7: Collective Intelligence Network Implementation
"""

# Original implementations
from .collective_intelligence import CollectiveIntelligence, NetworkNode, NodeRole
from .knowledge_sharing import KnowledgeSharer, SharedKnowledge, KnowledgeType

# Phase 7 implementations
from .elysia_node import ElysiaNode, Message
from .elysia_network import ElysiaNetwork
from .knowledge_sync import KnowledgeSync, Discovery
from .role_specialization import SpecializationManager, Role

__all__ = [
    # Original
    "CollectiveIntelligence",
    "NetworkNode",
    "NodeRole",
    "KnowledgeSharer",
    "SharedKnowledge",
    "KnowledgeType",
    # Phase 7
    "ElysiaNode",
    "Message",
    "ElysiaNetwork",
    "KnowledgeSync",
    "Discovery",
    "SpecializationManager",
    "Role",
]
