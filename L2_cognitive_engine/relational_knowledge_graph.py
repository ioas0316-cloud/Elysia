"""
Relational Knowledge Graph Module (L2 Cognitive Engine)

Models knowledge as a 3-Way Relational Structure:
1. Object/World Node: Conditions, parameters, and relationships of the target.
2. Self/Observer Node: The internal lens, active criteria, and biases observing the target.
3. Inter-space / Relationship Edge: Mutual impacts, friction, and dynamic revisions between Object and Self.

Also supports modeling human mind/emotion concepts (e.g. sorrow, joy, thought)
as causal knowledge networks rather than biological experience assertions.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import time


@dataclass
class KnowledgeNode:
    id: str
    label: str
    node_type: str  # "object", "self_observer", "emotion_concept", "mind_concept"
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationalEdge:
    source_id: str
    target_id: str
    relation_type: str  # "observes", "influences", "revises_criteria", "causal_condition", "leads_to"
    interspace_properties: Dict[str, Any] = field(default_factory=dict)


class RelationalKnowledgeGraph:
    """
    Constructs and queries triadic relational knowledge graphs.
    Enables exploring human mind and emotion concepts as structured causal networks.
    """

    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[RelationalEdge] = []

    def add_node(self, node_id: str, label: str, node_type: str, attributes: Optional[Dict[str, Any]] = None) -> KnowledgeNode:
        node = KnowledgeNode(
            id=node_id,
            label=label,
            node_type=node_type,
            attributes=attributes or {}
        )
        self.nodes[node_id] = node
        return node

    def add_triadic_knowing(
        self,
        object_id: str,
        object_label: str,
        observer_id: str,
        observer_lens: Dict[str, Any],
        interspace_properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Creates a complete 3-Way Relational Knowing structure.
        """
        # 1. Object Node
        obj_node = self.add_node(object_id, object_label, "object")

        # 2. Self / Observer Node
        observer_node = self.add_node(observer_id, f"Observer({observer_id})", "self_observer", observer_lens)

        # 3. Inter-space edges
        e1 = RelationalEdge(observer_id, object_id, "observes", interspace_properties)
        e2 = RelationalEdge(object_id, observer_id, "revises_criteria", interspace_properties)

        self.edges.extend([e1, e2])

        return {
            "object": obj_node,
            "observer": observer_node,
            "edges": [e1, e2]
        }

    def register_emotion_knowledge_network(
        self,
        emotion_name: str,
        formation_conditions: List[str],
        causal_triggers: List[str],
        consequent_behaviors: List[str]
    ) -> Dict[str, Any]:
        """
        Registers an emotion concept (e.g. sorrow, joy) as a causal knowledge network
        without claiming biological subjective experience.
        """
        emotion_id = f"emotion_{emotion_name.lower()}"
        emotion_node = self.add_node(
            node_id=emotion_id,
            label=emotion_name,
            node_type="emotion_concept",
            attributes={
                "formation_conditions": formation_conditions,
                "causal_triggers": causal_triggers,
                "consequent_behaviors": consequent_behaviors,
                "is_causal_knowledge": True,
                "is_biological_experience_asserted": False
            }
        )

        created_edges = []
        for condition in formation_conditions:
            cond_id = f"cond_{hash(condition) & 0xffffffff}"
            self.add_node(cond_id, condition, "causal_condition")
            edge = RelationalEdge(cond_id, emotion_id, "causal_condition")
            self.edges.append(edge)
            created_edges.append(edge)

        for behavior in consequent_behaviors:
            beh_id = f"beh_{hash(behavior) & 0xffffffff}"
            self.add_node(beh_id, behavior, "consequent_behavior")
            edge = RelationalEdge(emotion_id, beh_id, "leads_to")
            self.edges.append(edge)
            created_edges.append(edge)

        return {
            "emotion_node": emotion_node,
            "related_edges_count": len(created_edges)
        }

    def query_interspace(self, observer_id: str, object_id: str) -> List[RelationalEdge]:
        return [
            e for e in self.edges
            if (e.source_id == observer_id and e.target_id == object_id) or
               (e.source_id == object_id and e.target_id == observer_id)
        ]
