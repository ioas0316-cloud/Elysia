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
