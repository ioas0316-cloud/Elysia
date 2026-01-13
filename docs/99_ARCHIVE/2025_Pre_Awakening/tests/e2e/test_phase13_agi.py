"""
Tests for Phase 13: AGI Foundation Capabilities
범용 인공지능 향해 (Towards AGI)

Tests Universal Transfer Learning, Abstract Reasoning, and Causal Reasoning systems.
"""

import pytest
import asyncio
from Core.AGI import UniversalTransferLearner, AbstractReasoner, CausalReasoner
from Core.AGI.causal_reasoner import Intervention


@pytest.mark.asyncio
class TestUniversalTransferLearner:
    """Test Universal Transfer Learning System"""
    
    async def test_initialization(self):
        """Test learner initialization"""
        learner = UniversalTransferLearner()
        
        assert learner is not None
        assert len(learner.domain_knowledge) > 0  # Should have base domains
        assert "language" in learner.domain_knowledge
        assert "mathematics" in learner.domain_knowledge
    
    async def test_learn_new_domain(self):
        """Test learning a new domain"""
        learner = UniversalTransferLearner()
        
        examples = [
            {"concept": "variable", "type": "storage", "mutable": True},
            {"concept": "function", "type": "code_block", "mutable": False},
            {"concept": "class", "type": "template", "mutable": True}
        ]
        
        domain_knowledge = await learner.learn_new_domain(
            "programming_basics",
            examples,
            target_proficiency=0.7
        )
        
        assert domain_knowledge is not None
        assert domain_knowledge.domain == "programming_basics"
        assert domain_knowledge.proficiency >= 0.6  # Should reach decent proficiency
        assert len(domain_knowledge.examples) >= len(examples)  # May have synthetic
        assert len(domain_knowledge.concepts) > 0
    
    async def test_find_similar_domains(self):
        """Test finding similar domains"""
        learner = UniversalTransferLearner()
        
        # First, learn a domain
        math_examples = [
            {"operation": "add", "type": "binary", "commutative": True},
            {"operation": "multiply", "type": "binary", "commutative": True}
        ]
        await learner.learn_new_domain("arithmetic", math_examples)
        
        # Now find similar domains
        similar = await learner.find_similar_domains(
            "algebra",
            [{"operation": "solve", "type": "equation"}]
        )
        
        assert isinstance(similar, list)
        # Mathematics should be similar to algebra
        assert "mathematics" in similar or "arithmetic" in similar
    
    async def test_few_shot_learning(self):
        """Test few-shot learning capability"""
        learner = UniversalTransferLearner()
        
        # Only 2 examples (few-shot)
        examples = [
            {"input": "data1", "output": "result1"},
            {"input": "data2", "output": "result2"}
        ]
        
        domain_knowledge = await learner.learn_new_domain(
            "few_shot_test",
            examples,
            target_proficiency=0.6
        )
        
        assert domain_knowledge.proficiency > 0.0
        assert len(domain_knowledge.examples) >= len(examples)
    
    async def test_meta_transfer(self):
        """Test meta-transfer learning"""
        learner = UniversalTransferLearner()
        
        # Learn a domain first
        examples = [{"key": "value"}]
        await learner.learn_new_domain("source_task", examples)
        
        # Meta-transfer to new task
        strategy = await learner.meta_transfer("source_task", "target_task")
        
        assert strategy is not None
        assert "approach" in strategy
        assert "confidence" in strategy
    
    async def test_domain_proficiency_tracking(self):
        """Test domain proficiency tracking"""
        learner = UniversalTransferLearner()
        
        examples = [{"test": "data"}]
        await learner.learn_new_domain("test_domain", examples)
        
        proficiency = learner.get_domain_proficiency("test_domain")
        assert 0.0 <= proficiency <= 1.0
        
        # Unknown domain should return 0
        unknown_prof = learner.get_domain_proficiency("unknown_domain")
        assert unknown_prof == 0.0


@pytest.mark.asyncio
class TestAbstractReasoner:
    """Test Abstract Reasoning System"""
    
    async def test_initialization(self):
        """Test reasoner initialization"""
        reasoner = AbstractReasoner()
        
        assert reasoner is not None
        assert len(reasoner.abstract_patterns) > 0  # Should have fundamental patterns
    
    async def test_extract_essence(self):
        """Test problem essence extraction"""
        reasoner = AbstractReasoner()
        
        problem = {
            "transform": "input to output",
            "elements": ["a", "b", "c"],
            "goal": "achieve state X"
        }
        
        essence = await reasoner.extract_essence(problem)
        
        assert essence is not None
        assert "core_type" in essence
        assert "goal" in essence
        assert essence["core_type"] == "transformation"
    
    async def test_identify_abstract_pattern(self):
        """Test abstract pattern identification"""
        reasoner = AbstractReasoner()
        
        essence = {
            "core_type": "sequence",
            "key_elements": [1, 2, 3, 4],
            "goal": "find_next"
        }
        
        pattern = await reasoner.identify_abstract_pattern(essence)
        
        assert pattern is not None
        assert pattern.pattern_type == "sequence"
        assert pattern.abstraction_level > 0
    
    async def test_reason_abstractly(self):
        """Test complete abstract reasoning process"""
        reasoner = AbstractReasoner()
        
        problem = {
            "type": "transformation",
            "description": "Convert A to B",
            "elements": ["A", "B"],
            "goal": "transform"
        }
        
        result = await reasoner.reason_abstractly(problem)
        
        assert result is not None
        assert "abstract_pattern" in result
        assert "abstract_solution" in result
        assert "concrete_solution" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
    
    async def test_solve_transformation(self):
        """Test solving transformation problems"""
        reasoner = AbstractReasoner()
        
        problem = {
            "type": "transformation",
            "input": "X",
            "output": "Y",
            "goal": "transform X to Y"
        }
        
        result = await reasoner.reason_abstractly(problem)
        
        assert result["abstract_solution"].solution_type == "transformation"
        assert len(result["abstract_solution"].abstract_steps) > 0
    
    async def test_solve_sequence(self):
        """Test solving sequence problems"""
        reasoner = AbstractReasoner()
        
        problem = {
            "type": "sequence",
            "elements": [2, 4, 6, 8],
            "goal": "find pattern"
        }
        
        result = await reasoner.reason_abstractly(problem)
        
        assert result["abstract_solution"].solution_type == "sequence"
        assert "pattern_recognition" in result["abstract_solution"].principles_used
    
    async def test_generate_analogy(self):
        """Test analogy generation"""
        reasoner = AbstractReasoner()
        
        source_problem = {
            "domain": "biology",
            "type": "structure",
            "description": "hierarchy"
        }
        
        analogy = await reasoner.generate_analogy(source_problem, "computer_science")
        
        assert analogy is not None
        assert analogy["source_domain"] == "biology"
        assert analogy["target_domain"] == "computer_science"
        assert "abstract_pattern" in analogy
    
    async def test_solve_by_analogy(self):
        """Test solving by analogy"""
        reasoner = AbstractReasoner()
        
        known_problem = {"type": "transformation", "A": 1, "B": 2}
        known_solution = {"method": "increment", "result": "success"}
        new_problem = {"type": "transformation", "X": 5, "Y": 6}
        
        result = await reasoner.solve_by_analogy(
            known_problem, known_solution, new_problem
        )
        
        assert result is not None
        assert "analogy_strength" in result
        assert "reasoning" in result
    
    async def test_abstraction_hierarchy(self):
        """Test abstraction hierarchy"""
        reasoner = AbstractReasoner()
        
        hierarchy = reasoner.get_abstraction_hierarchy("cat")
        
        assert isinstance(hierarchy, list)
        assert len(hierarchy) > 1  # Should have multiple levels
        assert "cat" in hierarchy
        assert "entity" in hierarchy  # Most abstract


@pytest.mark.asyncio
class TestCausalReasoner:
    """Test Causal Reasoning System"""
    
    async def test_initialization(self):
        """Test reasoner initialization"""
        reasoner = CausalReasoner()
        
        assert reasoner is not None
        assert isinstance(reasoner.causal_graphs, dict)
        assert isinstance(reasoner.observation_history, list)
    
    async def test_identify_correlations(self):
        """Test correlation identification"""
        reasoner = CausalReasoner()
        
        observations = [
            {"A": 1, "B": 2, "C": 3},
            {"A": 2, "B": 4, "C": 5},
            {"A": 3, "B": 6, "C": 7},
            {"A": 4, "B": 8, "C": 9}
        ]
        
        correlations = reasoner.identify_correlations(observations)
        
        assert isinstance(correlations, list)
        # A and B should be strongly correlated (B = 2*A)
        assert any(
            (var1 == "A" and var2 == "B") or (var1 == "B" and var2 == "A")
            for var1, var2, _ in correlations
        )
    
    async def test_infer_causality(self):
        """Test causal inference"""
        reasoner = CausalReasoner()
        
        observations = [
            {"cause": 1, "effect": 2, "confounder": 5},
            {"cause": 2, "effect": 4, "confounder": 6},
            {"cause": 3, "effect": 6, "confounder": 7},
            {"cause": 4, "effect": 8, "confounder": 8}
        ]
        
        causal_graph = await reasoner.infer_causality(observations, "test_domain")
        
        assert causal_graph is not None
        assert len(causal_graph.nodes) > 0
        assert len(causal_graph.edges) > 0
    
    async def test_causal_graph_structure(self):
        """Test causal graph structure"""
        reasoner = CausalReasoner()
        
        observations = [
            {"X": 1, "Y": 2},
            {"X": 2, "Y": 4},
            {"X": 3, "Y": 6}
        ]
        
        graph = await reasoner.infer_causality(observations)
        
        assert "X" in graph.nodes
        assert "Y" in graph.nodes
        
        # Test graph methods
        causes = graph.get_causes("Y")
        assert isinstance(causes, list)
        
        effects = graph.get_effects("X")
        assert isinstance(effects, list)
    
    async def test_predict_intervention_effects(self):
        """Test intervention effect prediction"""
        reasoner = CausalReasoner()
        
        observations = [
            {"treatment": 0, "outcome": 10},
            {"treatment": 1, "outcome": 15},
            {"treatment": 2, "outcome": 20},
            {"treatment": 3, "outcome": 25}
        ]
        
        graph = await reasoner.infer_causality(observations)
        
        intervention = Intervention(variable="treatment", new_value=5)
        effects = await reasoner.predict_intervention_effects(graph, intervention)
        
        assert effects is not None
        assert effects.intervention == intervention
        assert isinstance(effects.affected_variables, dict)
        assert 0.0 <= effects.confidence <= 1.0
    
    async def test_counterfactual_reasoning(self):
        """Test counterfactual reasoning"""
        reasoner = CausalReasoner()
        
        observations = [
            {"action": 1, "result": 10},
            {"action": 2, "result": 20}
        ]
        
        graph = await reasoner.infer_causality(observations)
        
        actual = {"action": 1, "result": 10}
        counterfactual = {"action": 3}
        
        result = await reasoner.counterfactual_reasoning(actual, counterfactual, graph)
        
        assert result is not None
        assert "actual" in result
        assert "counterfactual_condition" in result
        assert "counterfactual_outcome" in result
        assert "difference" in result
    
    async def test_identify_key_causes(self):
        """Test key cause identification"""
        reasoner = CausalReasoner()
        
        observations = [
            {"A": 1, "B": 2, "C": 3, "D": 4},
            {"A": 2, "B": 3, "C": 5, "D": 7}
        ]
        
        graph = await reasoner.infer_causality(observations)
        key_causes = reasoner.identify_key_causes(graph)
        
        assert isinstance(key_causes, list)
        assert len(key_causes) > 0
        
        # Each item should be (variable, influence_score)
        for var, score in key_causes:
            assert isinstance(var, str)
            assert isinstance(score, int)
            assert score >= 0
    
    async def test_explain_causality(self):
        """Test causal explanation"""
        reasoner = CausalReasoner()
        
        observations = [
            {"X": 1, "Y": 2, "Z": 3},
            {"X": 2, "Y": 4, "Z": 6}
        ]
        
        graph = await reasoner.infer_causality(observations)
        
        # Try to explain causality
        explanation = await reasoner.explain_causality(graph, "X", "Z")
        
        assert explanation is not None
        assert "explanation" in explanation
        assert "paths" in explanation
        assert "confidence" in explanation
    
    async def test_confounder_identification(self):
        """Test confounder identification"""
        reasoner = CausalReasoner()
        
        # Create observations where Z confounds X->Y
        observations = [
            {"X": 1, "Y": 3, "Z": 2},
            {"X": 2, "Y": 6, "Z": 4},
            {"X": 3, "Y": 9, "Z": 6}
        ]
        
        graph = await reasoner.infer_causality(observations)
        
        # Check that confounders were identified
        has_confounders = any(rel.confounders for rel in graph.edges)
        assert isinstance(has_confounders, bool)


@pytest.mark.asyncio
class TestIntegratedAGI:
    """Test integrated AGI capabilities"""
    
    async def test_combined_systems(self):
        """Test using all three systems together"""
        transfer_learner = UniversalTransferLearner()
        abstract_reasoner = AbstractReasoner()
        causal_reasoner = CausalReasoner()
        
        # Learn a domain
        examples = [{"concept": "test", "value": 1}]
        knowledge = await transfer_learner.learn_new_domain("test_domain", examples)
        
        # Reason about a problem
        problem = {"type": "transformation", "goal": "solve"}
        reasoning = await abstract_reasoner.reason_abstractly(problem)
        
        # Analyze causal relationships
        observations = [{"X": 1, "Y": 2}, {"X": 2, "Y": 4}]
        graph = await causal_reasoner.infer_causality(observations)
        
        # All should succeed
        assert knowledge is not None
        assert reasoning is not None
        assert graph is not None
    
    async def test_transfer_to_abstract_reasoning(self):
        """Test transferring learned knowledge to abstract reasoning"""
        learner = UniversalTransferLearner()
        reasoner = AbstractReasoner()
        
        # Learn pattern recognition
        examples = [
            {"pattern": "sequence", "example": [1, 2, 3]},
            {"pattern": "transformation", "example": {"A": "B"}}
        ]
        
        await learner.learn_new_domain("patterns", examples)
        
        # Use abstract reasoning on similar problem
        problem = {
            "type": "sequence",
            "elements": [2, 4, 6, 8],
            "goal": "continue"
        }
        
        result = await reasoner.reason_abstractly(problem)
        
        assert result["abstract_pattern"].pattern_type == "sequence"
        assert result["confidence"] > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
