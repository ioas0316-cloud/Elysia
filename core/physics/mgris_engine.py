"""
M-GRIS: Molecular Graph Rewriting Inference System Engine.

This module implements a non-numerical, topological inference engine based on
molecular graph rewriting rules, bitwise complementary binding (sticky ends),
enzymatic rewrite operators (Polymerase, Helicase, Ligase, Restriction Enzyme),
and concept operators for self-referential semantic feedback / 2-category meta-rewriting.

References:
    - THE_ABSOLUTE_COMMANDMENT.md: "Do not calculate, let it flow."
    - AGENTS.md: Continuous Causal Intelligence Principles.
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union, Any


class Polarity(Enum):
    """Polarity of a Sticky End (analogous to 5' Donor vs 3' Acceptor in DNA strands)."""
    DONOR = 0
    ACCEPTOR = 1

    def is_complementary_to(self, other: "Polarity") -> bool:
        return self != other


class StickyEnd:
    """
    Representing a 64-bit topological sticky end (key-lock shape) for O(1) bitwise matching.

    Complementarity rule:
        1. Polarities must be opposite (DONOR <-> ACCEPTOR).
        2. Bit patterns must be bitwise NOT of each other (pattern_a ^ pattern_b == 0xFFFFFFFFFFFFFFFF).
    """
    MASK64: int = 0xFFFFFFFFFFFFFFFF

    def __init__(self, polarity: Polarity, pattern: int):
        self.polarity = polarity
        self.pattern = pattern & self.MASK64

    def can_bind_with(self, other: "StickyEnd") -> bool:
        if not self.polarity.is_complementary_to(other.polarity):
            return False
        return (self.pattern ^ other.pattern) == self.MASK64

    def invert(self) -> "StickyEnd":
        """Inverts the pattern bitwise (topological negation)."""
        new_pol = Polarity.ACCEPTOR if self.polarity == Polarity.DONOR else Polarity.DONOR
        return StickyEnd(new_pol, ~self.pattern & self.MASK64)

    def __repr__(self) -> str:
        pol_str = "Donor" if self.polarity == Polarity.DONOR else "Acceptor"
        return f"StickyEnd({pol_str}, pattern=0x{self.pattern:016X})"


class Node:
    """
    A Knowledge Node representing a concept or operator in the molecular graph.

    Attributes:
        id: Unique integer identifier.
        label: Semantic label (e.g. "Socrates", "Human", "Mortality").
        sticky_ends: List of exposed sticky ends for binding.
        valence_limit: Maximum allowed external bonds (prevents graph explosion).
        is_operator: Whether this concept node functions as a ConceptOperator (modifies other structures).
        constraint_mask: Bitmask used by RestrictionEnzyme for contradiction/conflict detection.
    """
    def __init__(
        self,
        node_id: int,
        label: str,
        sticky_ends: List[StickyEnd],
        valence_limit: int = 4,
        is_operator: bool = False,
        constraint_mask: int = 0,
    ):
        self.id = node_id
        self.label = label
        self.sticky_ends = sticky_ends
        self.valence_limit = valence_limit
        self.is_operator = is_operator
        self.constraint_mask = constraint_mask & StickyEnd.MASK64

    def invert_sticky_ends(self) -> None:
        """Applies 2-morphism topological inversion to all sticky ends of this node."""
        self.sticky_ends = [se.invert() for se in self.sticky_ends]

    def __repr__(self) -> str:
        op_flag = " [Op]" if self.is_operator else ""
        return f"Node(id={self.id}, label='{self.label}'{op_flag}, ends={len(self.sticky_ends)})"


class Bond:
    """A covalent-like/hydrogen-like topological edge joining two sticky ends of two nodes."""
    def __init__(self, node_a: int, end_a_idx: int, node_b: int, end_b_idx: int):
        self.node_a = node_a
        self.end_a_idx = end_a_idx
        self.node_b = node_b
        self.end_b_idx = end_b_idx

    def connects(self, node_id: int) -> bool:
        return self.node_a == node_id or self.node_b == node_id

    def connects_end(self, node_id: int, end_idx: int) -> bool:
        return (self.node_a == node_id and self.end_a_idx == end_idx) or (
            self.node_b == node_id and self.end_b_idx == end_idx
        )

    def other_node(self, node_id: int) -> Optional[int]:
        if self.node_a == node_id:
            return self.node_b
        if self.node_b == node_id:
            return self.node_a
        return None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bond):
            return False
        return (
            (self.node_a == other.node_a and self.end_a_idx == other.end_a_idx and
             self.node_b == other.node_b and self.end_b_idx == other.end_b_idx) or
            (self.node_a == other.node_b and self.end_a_idx == other.end_b_idx and
             self.node_b == other.node_a and self.end_b_idx == other.end_a_idx)
        )

    def __repr__(self) -> str:
        return f"Bond(Node#{self.node_a}[{self.end_a_idx}] <-> Node#{self.node_b}[{self.end_b_idx}])"


class MolecularGraph:
    """
    Molecular Graph representation holding nodes, bonds, and execution constraints (ATP step budget).
    """
    def __init__(self, atp_budget: int = 100):
        self.nodes: Dict[int, Node] = {}
        self.bonds: List[Bond] = []
        self.atp_budget = atp_budget
        self.atp_consumed = 0

    def add_node(self, node: Node) -> int:
        self.nodes[node.id] = node
        return node.id

    def consume_atp(self, cost: int = 1) -> bool:
        """Consumes step budget. Returns False if budget exhausted."""
        if self.atp_consumed + cost > self.atp_budget:
            return False
        self.atp_consumed += cost
        return True

    def get_node_valence(self, node_id: int) -> int:
        """Returns the number of active bonds connected to this node."""
        return sum(1 for b in self.bonds if b.connects(node_id))

    def is_end_bound(self, node_id: int, end_idx: int) -> bool:
        """Checks if a specific sticky end of a node is currently bound."""
        return any(b.connects_end(node_id, end_idx) for b in self.bonds)

    def get_unbound_sticky_ends(self, node_id: int) -> List[Tuple[int, StickyEnd]]:
        """Returns list of (end_idx, sticky_end) that are not bound."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        unbound = []
        for idx, end in enumerate(node.sticky_ends):
            if not self.is_end_bound(node_id, idx):
                unbound.append((idx, end))
        return unbound

    # =========================================================================
    # ENZYMATIC REWRITE OPERATORS
    # =========================================================================

    def polymerase_extend(
        self,
        target_node_id: int,
        target_end_idx: int,
        knowledge_pool: List[Node]
    ) -> Optional[int]:
        """
        Polymerase Operator: Extends knowledge chain via complementary binding (Forward Chaining).

        Searches knowledge_pool for a node whose sticky end is complementary to target sticky end.
        If found and valence limits permit, adds cloned node to graph and creates Bond.
        """
        if not self.consume_atp(1):
            return None

        target_node = self.nodes.get(target_node_id)
        if not target_node:
            return None

        if self.is_end_bound(target_node_id, target_end_idx):
            return None

        if self.get_node_valence(target_node_id) >= target_node.valence_limit:
            return None

        target_end = target_node.sticky_ends[target_end_idx]

        for pool_node in knowledge_pool:
            for pool_end_idx, pool_end in enumerate(pool_node.sticky_ends):
                if target_end.can_bind_with(pool_end):
                    # Assign a new unique node ID in this graph
                    new_id = max(self.nodes.keys(), default=-1) + 1
                    cloned_node = Node(
                        node_id=new_id,
                        label=pool_node.label,
                        sticky_ends=[
                            StickyEnd(e.polarity, e.pattern) for e in pool_node.sticky_ends
                        ],
                        valence_limit=pool_node.valence_limit,
                        is_operator=pool_node.is_operator,
                        constraint_mask=pool_node.constraint_mask,
                    )
                    self.add_node(cloned_node)
                    self.bonds.append(Bond(target_node_id, target_end_idx, new_id, pool_end_idx))
                    return new_id

        return None

    def helicase_unzip(self, node_a_id: int, node_b_id: int) -> int:
        """
        Helicase Operator: Unzips / breaks bonds between two nodes to decouple edges.
        """
        if not self.consume_atp(1):
            return 0

        initial_count = len(self.bonds)
        self.bonds = [
            b for b in self.bonds
            if not ((b.node_a == node_a_id and b.node_b == node_b_id) or
                    (b.node_a == node_b_id and b.node_b == node_a_id))
        ]
        return initial_count - len(self.bonds)

    def ligase_seal(self, node_a_id: int, end_a_idx: int, node_b_id: int, end_b_idx: int) -> bool:
        """
        Ligase Operator: Seals internal open sticky ends into a closed causal loop or edge
        without introducing new nodes.
        """
        if not self.consume_atp(1):
            return False

        if node_a_id not in self.nodes or node_b_id not in self.nodes:
            return False

        node_a = self.nodes[node_a_id]
        node_b = self.nodes[node_b_id]

        if self.is_end_bound(node_a_id, end_a_idx) or self.is_end_bound(node_b_id, end_b_idx):
            return False

        if self.get_node_valence(node_a_id) >= node_a.valence_limit:
            return False
        if self.get_node_valence(node_b_id) >= node_b.valence_limit:
            return False

        end_a = node_a.sticky_ends[end_a_idx]
        end_b = node_b.sticky_ends[end_b_idx]

        if end_a.can_bind_with(end_b):
            self.bonds.append(Bond(node_a_id, end_a_idx, node_b_id, end_b_idx))
            return True

        return False

    def restriction_enzyme_prune(self, recognition_mask: int) -> int:
        """
        Restriction Enzyme Operator: Scans for contradiction recognition sites
        (nodes or bonds conflicting with recognition_mask) and prunes invalid subgraphs.

        Rule: If (node.constraint_mask & recognition_mask) != 0, node is pruned along with its bonds.
        """
        if not self.consume_atp(1):
            return 0

        mask64 = recognition_mask & StickyEnd.MASK64
        if mask64 == 0:
            return 0

        pruned_nodes: Set[int] = set()
        for node_id, node in self.nodes.items():
            if (node.constraint_mask & mask64) != 0:
                pruned_nodes.add(node_id)

        if not pruned_nodes:
            return 0

        # Remove bonds connected to pruned nodes
        self.bonds = [b for b in self.bonds if b.node_a not in pruned_nodes and b.node_b not in pruned_nodes]

        # Remove pruned nodes
        for nid in pruned_nodes:
            del self.nodes[nid]

        return len(pruned_nodes)


class MGRISInferenceEngine:
    """
    High-level M-GRIS Inference Engine orchestrating molecular graph rewriting,
    concept operator execution, and self-referential semantic feedback / 2-morphism transformations.
    """
    def __init__(self, atp_budget: int = 100):
        self.atp_budget = atp_budget

    def execute_inference_cycle(
        self,
        query_strand: Node,
        knowledge_pool: List[Node],
        max_depth: int = 5,
        contradiction_masks: Optional[List[int]] = None
    ) -> Tuple[MolecularGraph, List[str]]:
        """
        Runs a full M-GRIS execution cycle:
        1. Inject query strand into graph.
        2. Polymerase chain extension up to max_depth.
        3. Concept Operator & 2-Morphism reification pass (Self-Referential Feedback).
        4. Ligase pass: internal loop closure on remaining complementary ends.
        5. Restriction enzyme pass: pruning on contradiction masks.
        6. Read out topological inference narrative.
        """
        graph = MolecularGraph(atp_budget=self.atp_budget)
        graph.add_node(query_strand)

        active_nodes = [query_strand.id]
        depth = 0

        while active_nodes and depth < max_depth and graph.atp_consumed < graph.atp_budget:
            next_active = []
            for curr_id in active_nodes:
                unbound_ends = graph.get_unbound_sticky_ends(curr_id)
                for end_idx, _ in unbound_ends:
                    new_id = graph.polymerase_extend(curr_id, end_idx, knowledge_pool)
                    if new_id is not None:
                        next_active.append(new_id)

                        # Self-Referential Concept Operator Reification:
                        # If newly attached node is an operator (e.g., Negation Operator),
                        # execute 2-morphism meta-rewriting on adjacent nodes.
                        attached_node = graph.nodes[new_id]
                        if attached_node.is_operator:
                            self._execute_2morphism_operator(graph, new_id, curr_id)

            active_nodes = next_active
            depth += 1

        # Ligase pass: attempt internal loop closure among unbound sticky ends
        all_node_ids = list(graph.nodes.keys())
        for i, id_a in enumerate(all_node_ids):
            unbound_a = graph.get_unbound_sticky_ends(id_a)
            for end_a_idx, _ in unbound_a:
                for id_b in all_node_ids[i+1:]:
                    unbound_b = graph.get_unbound_sticky_ends(id_b)
                    for end_b_idx, _ in unbound_b:
                        graph.ligase_seal(id_a, end_a_idx, id_b, end_b_idx)

        # Restriction Enzyme pass: prune contradictions if masks provided
        if contradiction_masks:
            for c_mask in contradiction_masks:
                graph.restriction_enzyme_prune(c_mask)

        # Extract topological inference narrative
        narrative = self._extract_narrative(graph)
        return graph, narrative

    def _execute_2morphism_operator(self, graph: MolecularGraph, op_node_id: int, target_node_id: int) -> None:
        """
        2-Morphism Meta-Rewriting:
        Concept operator node modifies the target node's binding rules (sticky end inversion).
        """
        target_node = graph.nodes.get(target_node_id)
        if target_node:
            target_node.invert_sticky_ends()

    def _extract_narrative(self, graph: MolecularGraph) -> List[str]:
        narrative = []
        for bond in graph.bonds:
            node_a = graph.nodes.get(bond.node_a)
            node_b = graph.nodes.get(bond.node_b)
            if node_a and node_b:
                narrative.append(f"{node_a.label} -> {node_b.label}")
        return narrative


class MGRISCausalBridge:
    """
    Bridge interface connecting M-GRIS Engine with CausalField & CognitiveEngine.
    Translates causal queries into sticky-end bitmasks and returns topological causal chains.
    """
    @staticmethod
    def concept_to_pattern(concept_name: str) -> int:
        """Generates a deterministic 64-bit pattern from a concept string."""
        import hashlib
        digest = hashlib.sha256(concept_name.encode('utf-8')).digest()
        pattern = int.from_bytes(digest[:8], byteorder='big')
        return pattern & StickyEnd.MASK64

    @classmethod
    def create_concept_node(
        cls,
        node_id: int,
        concept: str,
        complementary_concept: Optional[str] = None,
        valence: int = 4,
        is_operator: bool = False,
        constraint_mask: int = 0
    ) -> Node:
        pattern = cls.concept_to_pattern(concept)
        if complementary_concept:
            # Complementary pattern is bitwise NOT
            pattern_b = ~cls.concept_to_pattern(complementary_concept) & StickyEnd.MASK64
        else:
            pattern_b = pattern

        sticky_ends = [
            StickyEnd(Polarity.DONOR, pattern),
            StickyEnd(Polarity.ACCEPTOR, pattern_b)
        ]
        return Node(
            node_id=node_id,
            label=concept,
            sticky_ends=sticky_ends,
            valence_limit=valence,
            is_operator=is_operator,
            constraint_mask=constraint_mask
        )
