"""
Unit tests for Relational Semantic Tensor & Context-Driven Cognition
(core/topology/relational_semantic_tensor.py)
"""

import pytest
from core.topology.relational_semantic_tensor import (
    SymbolicTerm,
    SymbolicTensor,
    ContextOperator,
    BackgroundFieldMasker,
    TopologicalBalanceSolver,
    RelationKind
)


def test_symbolic_tensor_creation_and_indexing():
    tensor = SymbolicTensor(shape=(2, 2), name="TestTensor")

    apple = SymbolicTerm(
        id="term_apple",
        label="Apple",
        category_hierarchy={"taxonomy": ["Fruit", "Organism"]}
    )

    tensor.set_element((0, 0), apple)
    retrieved = tensor.get_element((0, 0))

    assert retrieved is not None
    assert retrieved.id == "term_apple"
    assert retrieved.label == "Apple"


def test_context_operator_refraction():
    apple = SymbolicTerm(
        id="term_apple",
        label="Apple",
        category_hierarchy={
            "taxonomy": ["Fruit", "Plantae", "Organism"],
            "color": ["Red"]
        }
    )
    monkey = SymbolicTerm(
        id="term_monkey",
        label="Monkey",
        category_hierarchy={
            "taxonomy": ["Primate", "Mammal", "Animalia", "Organism"],
            "color": ["Brown"]
        }
    )

    taxonomy_operator = ContextOperator(name="taxonomy", active_axes={"biological_kingdom"})
    terms = [apple, monkey]

    refracted = taxonomy_operator.refract(terms)
    assert len(refracted) == 2
    assert "Fruit -> Plantae -> Organism" in refracted
    assert "Primate -> Mammal -> Animalia -> Organism" in refracted


def test_relational_evaluator_branching_and_subsumption():
    tensor = SymbolicTensor(shape=(2, 2))

    apple = SymbolicTerm(
        id="term_apple",
        label="Apple",
        category_hierarchy={"taxonomy": ["Fruit", "Organism"]}
    )
    monkey = SymbolicTerm(
        id="term_monkey",
        label="Monkey",
        category_hierarchy={"taxonomy": ["Animal", "Organism"]}
    )
    fruit = SymbolicTerm(
        id="term_fruit",
        label="Fruit",
        category_hierarchy={"taxonomy": ["Organism"]}
    )

    tensor.set_element((0, 0), apple)
    tensor.set_element((0, 1), monkey)
    tensor.set_element((1, 0), fruit)

    # 1. Branching check (Apple vs Monkey branch from Organism)
    eval_branching = tensor.evaluate_relational_pair((0, 0), (0, 1), context_name="taxonomy")
    assert eval_branching.relation_kind == RelationKind.BRANCHING
    assert eval_branching.common_ancestor == "Organism"
    assert eval_branching.disparity_score == 0.4

    # 2. Subsumption check (Apple subsumed under Fruit)
    eval_subsumption = tensor.evaluate_relational_pair((0, 0), (1, 0), context_name="taxonomy")
    assert eval_subsumption.relation_kind == RelationKind.SUBSUMPTION
    assert eval_subsumption.common_ancestor == "Fruit"


def test_relational_evaluator_contradiction():
    tensor = SymbolicTensor(shape=(1, 2))

    statement_a = SymbolicTerm(
        id="stmt_a",
        label="Light is Wave",
        category_hierarchy={},
        invariant_rules={"CONTRADICTS_stmt_b"}
    )
    statement_b = SymbolicTerm(
        id="stmt_b",
        label="Light is Particle",
        category_hierarchy={},
        invariant_rules={"CONTRADICTS_stmt_a"}
    )

    tensor.set_element((0, 0), statement_a)
    tensor.set_element((0, 1), statement_b)

    eval_contradiction = tensor.evaluate_relational_pair((0, 0), (0, 1), context_name="physics")
    assert eval_contradiction.relation_kind == RelationKind.CONTRADICTION
    assert eval_contradiction.disparity_score == 1.0


def test_background_field_masker_o1_bypass():
    masker = BackgroundFieldMasker()
    # Register frozen axiom (e.g. Conservation Law)
    masker.register_frozen_axiom("conservation_law", lambda state: state.get("energy_balance") == True)

    assert masker.is_axiom_satisfied("conservation_law", {"energy_balance": True}) is True
    assert masker.is_axiom_satisfied("conservation_law", {"energy_balance": False}) is False
    assert masker.bypass_count == 2


def test_topological_balance_solver_equivalence():
    solver = TopologicalBalanceSolver()
    # 4x + 1 = 2x + 13
    result = solver.simplify_linear_equation(lhs_coeff=4, lhs_const=1, rhs_coeff=2, rhs_const=13)

    assert result["solution"] == 6.0
    assert len(result["reduction_steps"]) == 4
    assert result["method"] == "TOPOLOGICAL_EQUIVALENCE_SIMPLIFICATION"


def test_space_state_graph_and_cognitive_time_transition():
    from core.topology.relational_semantic_tensor import SpaceStateNode, SpaceStateGraph

    graph = SpaceStateGraph(name="TestCognitiveTime")

    node_0 = SpaceStateNode(
        state_id="space_s0",
        tensor_snapshot_name="InitialUnbalancedTensor",
        active_axioms={"conservation_law", "symmetry_rule"},
        disparity_entropy=0.8
    )
    node_1 = SpaceStateNode(
        state_id="space_s1",
        tensor_snapshot_name="EquilibratedTensor",
        active_axioms={"conservation_law", "symmetry_rule", "context_refraction"},
        disparity_entropy=0.1
    )

    graph.add_space_state(node_0)
    edge = graph.transition(node_1, operator_used="EQUILIBRIUM_REACTION", invariant_preservation_ratio=1.0)

    assert edge.source_state_id == "space_s0"
    assert edge.target_state_id == "space_s1"
    assert edge.disparity_reduction == pytest.approx(0.7)
    assert len(graph.get_trajectory_history()) == 1


def test_invariant_trace_tensor():
    from core.topology.relational_semantic_tensor import InvariantTraceTensor

    trace = InvariantTraceTensor()
    trace.register_invariant("conservation_law")
    trace.register_invariant("causal_continuity")

    source_axioms = {"conservation_law", "causal_continuity", "temp_axiom_a"}
    target_axioms = {"conservation_law", "causal_continuity", "temp_axiom_b"}

    ratio = trace.compute_preservation_ratio(source_axioms, target_axioms)
    assert ratio == 1.0


def test_cognitive_vector_field_convergence():
    from core.topology.relational_semantic_tensor import CognitiveVectorField

    field = CognitiveVectorField(target_equilibrium_disparity=0.0)
    res = field.calculate_convergence_vector(current_disparity=0.5, structural_friction=0.1)

    assert res["disparity_gap"] == 0.5
    assert res["restoring_force"] == pytest.approx(0.4)
    assert res["net_convergence_velocity"] == pytest.approx(0.38)
    assert res["is_equilibrated"] is False


def test_higher_dimensional_meta_observer():
    from core.topology.relational_semantic_tensor import (
        SpaceStateNode,
        SpaceStateGraph,
        HigherDimensionalMetaObserver
    )

    graph = SpaceStateGraph(name="SubspaceGraph")

    n0 = SpaceStateNode("state_0", "snap0", {"ax1"}, disparity_entropy=0.9)
    n1 = SpaceStateNode("state_1", "snap1", {"ax1"}, disparity_entropy=0.2)
    n2 = SpaceStateNode("state_2", "snap2", {"ax1"}, disparity_entropy=0.95)

    graph.add_space_state(n0)
    graph.transition(n1, operator_used="OPERATOR_GOOD", invariant_preservation_ratio=1.0)
    graph.transition(n2, operator_used="OPERATOR_BAD", invariant_preservation_ratio=0.5)

    observer = HigherDimensionalMetaObserver(observer_id="MetaObserver_Alpha")
    observer.attach_graph("subgraph_1", graph)

    res = observer.overview_and_prune_graph("subgraph_1", max_disparity_threshold=0.5)

    assert res["total_edges_observed"] == 2
    assert res["valid_edges_retained"] == 1
    assert res["pruned_unstable_edges"] == 1
    assert res["overview_effect"] == "HIGH_DIMENSIONAL_RESTRUCTURE_COMPLETE"
