"""
Tests for M-GRIS (Molecular Graph Rewriting Inference System) Engine.

Covers:
- Bitwise sticky end complementarity matching (`pattern_a ^ pattern_b == 0xFFFFFFFFFFFFFFFF`).
- Polymerase forward-chaining knowledge synthesis.
- Helicase unzipping / bond decoupling.
- Ligase internal loop closure without adding new nodes.
- Restriction Enzyme contradiction site pruning.
- Valence limits and ATP step budget bounds.
- Concept Operator / 2-morphism meta-rewriting (Self-Referential Feedback).
- Integration test with CausalBridge and Cognitive reasoning pipeline.
"""

import pytest
from core.physics.mgris_engine import (
    Polarity,
    StickyEnd,
    Node,
    Bond,
    MolecularGraph,
    MGRISInferenceEngine,
    MGRISCausalBridge,
)


def test_bit_complementarity():
    pattern_a = 0x1122334455667788
    pattern_b = ~pattern_a & StickyEnd.MASK64

    end_donor_a = StickyEnd(Polarity.DONOR, pattern_a)
    end_acceptor_b = StickyEnd(Polarity.ACCEPTOR, pattern_b)
    end_donor_b = StickyEnd(Polarity.DONOR, pattern_b)

    # Opposite polarities + bitwise NOT pattern -> True
    assert end_donor_a.can_bind_with(end_acceptor_b) is True
    # Same polarity -> False
    assert end_donor_a.can_bind_with(end_donor_b) is False
    # Non-complementary pattern -> False
    non_comp = StickyEnd(Polarity.ACCEPTOR, pattern_a)
    assert end_donor_a.can_bind_with(non_comp) is False


def test_polymerase_chain():
    p_socrates = 0x00FF00FF00FF00FF
    p_human = ~p_socrates & StickyEnd.MASK64
    p_mortal_in = 0xAA55AA55AA55AA55
    p_mortal = ~p_mortal_in & StickyEnd.MASK64

    query = Node(0, "Socrates", [StickyEnd(Polarity.DONOR, p_socrates)])
    knowledge_pool = [
        Node(99, "Human", [
            StickyEnd(Polarity.ACCEPTOR, p_human),
            StickyEnd(Polarity.DONOR, p_mortal_in)
        ]),
        Node(100, "Mortality", [
            StickyEnd(Polarity.ACCEPTOR, p_mortal)
        ])
    ]

    engine = MGRISInferenceEngine(atp_budget=20)
    graph, narrative = engine.execute_inference_cycle(query, knowledge_pool, max_depth=3)

    assert "Socrates -> Human" in narrative
    assert "Human -> Mortality" in narrative
    assert len(graph.nodes) == 3


def test_helicase_unzip():
    graph = MolecularGraph(atp_budget=10)
    node1 = Node(1, "A", [StickyEnd(Polarity.DONOR, 0x01)])
    node2 = Node(2, "B", [StickyEnd(Polarity.ACCEPTOR, ~0x01 & StickyEnd.MASK64)])
    graph.add_node(node1)
    graph.add_node(node2)

    # Manually add a bond
    graph.bonds.append(Bond(1, 0, 2, 0))
    assert len(graph.bonds) == 1

    # Helicase unzips bond between node 1 and 2
    unzipped_count = graph.helicase_unzip(1, 2)
    assert unzipped_count == 1
    assert len(graph.bonds) == 0


def test_ligase_closure():
    graph = MolecularGraph(atp_budget=10)
    pat = 0xFF00FF00FF00FF00
    comp_pat = ~pat & StickyEnd.MASK64

    # Node A and Node B already in graph with unbound complementary sticky ends
    node_a = Node(0, "Start", [StickyEnd(Polarity.DONOR, pat)])
    node_b = Node(1, "End", [StickyEnd(Polarity.ACCEPTOR, comp_pat)])
    graph.add_node(node_a)
    graph.add_node(node_b)

    # Ligase seals the open ends into a bond
    success = graph.ligase_seal(0, 0, 1, 0)
    assert success is True
    assert len(graph.bonds) == 1
    assert graph.bonds[0].node_a == 0 and graph.bonds[0].node_b == 1


def test_restriction_enzyme_pruning():
    graph = MolecularGraph(atp_budget=10)
    conflict_mask = 0b1000

    node_valid = Node(0, "ValidConcept", [StickyEnd(Polarity.DONOR, 0x11)], constraint_mask=0b0001)
    node_invalid = Node(1, "Contradiction", [StickyEnd(Polarity.ACCEPTOR, ~0x11 & StickyEnd.MASK64)], constraint_mask=0b1001)

    graph.add_node(node_valid)
    graph.add_node(node_invalid)
    graph.bonds.append(Bond(0, 0, 1, 0))

    pruned_count = graph.restriction_enzyme_prune(conflict_mask)
    assert pruned_count == 1
    assert 1 not in graph.nodes
    assert 0 in graph.nodes
    assert len(graph.bonds) == 0


def test_valence_and_budget_limits():
    # Test Valence Limit
    graph = MolecularGraph(atp_budget=10)
    # Node with max valence limit = 1
    node_limited = Node(0, "Limited", [StickyEnd(Polarity.DONOR, 0x55)], valence_limit=1)
    node_b1 = Node(1, "B1", [StickyEnd(Polarity.ACCEPTOR, ~0x55 & StickyEnd.MASK64)])
    node_b2 = Node(2, "B2", [StickyEnd(Polarity.ACCEPTOR, ~0x55 & StickyEnd.MASK64)])

    graph.add_node(node_limited)
    graph.bonds.append(Bond(0, 0, 1, 0)) # Uses 1 valence

    # Further polymerase extend should fail due to valence limit
    res = graph.polymerase_extend(0, 0, [node_b2])
    assert res is None

    # Test ATP Budget Limit
    graph_exhausted = MolecularGraph(atp_budget=1)
    assert graph_exhausted.consume_atp(1) is True
    assert graph_exhausted.consume_atp(1) is False # Budget exhausted


def test_concept_operator_2morphism():
    # Test Negation Operator that inverts sticky ends of target node
    p_a = 0x1234567890ABCDEF
    p_comp = ~p_a & StickyEnd.MASK64

    query = Node(0, "StatementA", [StickyEnd(Polarity.DONOR, p_a)])
    negation_op = Node(
        node_id=99,
        label="NegationOperator",
        sticky_ends=[StickyEnd(Polarity.ACCEPTOR, p_comp)],
        is_operator=True
    )

    engine = MGRISInferenceEngine(atp_budget=20)
    graph, narrative = engine.execute_inference_cycle(query, [negation_op], max_depth=2)

    assert "StatementA -> NegationOperator" in narrative
    # The statement's sticky ends should have been inverted by the 2-morphism meta-rewriting
    assert query.sticky_ends[0].polarity == Polarity.ACCEPTOR
    assert query.sticky_ends[0].pattern == p_comp


def test_mgris_causal_integration():
    bridge = MGRISCausalBridge()

    node_cause = bridge.create_concept_node(0, "Rain", complementary_concept="WetGround")
    node_effect = bridge.create_concept_node(100, "WetGround")

    engine = MGRISInferenceEngine(atp_budget=20)
    graph, narrative = engine.execute_inference_cycle(node_cause, [node_effect], max_depth=2)

    assert len(narrative) > 0
    assert "Rain -> WetGround" in narrative
