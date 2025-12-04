"""
Network module for Elysia - Collective Intelligence and Communication.

This module enables Elysia instances to communicate, collaborate, and share knowledge
across a distributed network.
"""

from .collective_intelligence import CollectiveIntelligence, NetworkNode, NodeRole
from .knowledge_sharing import KnowledgeSharer, SharedKnowledge, KnowledgeType

__all__ = [
    "CollectiveIntelligence",
    "NetworkNode",
    "NodeRole",
    "KnowledgeSharer",
    "SharedKnowledge",
    "KnowledgeType",
]
