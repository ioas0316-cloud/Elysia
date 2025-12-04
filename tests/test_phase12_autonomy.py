"""
Tests for Phase 12: Autonomy & Goal Setting

Tests the autonomous systems:
- Autonomous Goal Generation
- Ethical Reasoning
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.Autonomy import AutonomousGoalGenerator, EthicalReasoner
from Core.Autonomy.goal_generator import GoalPriority, GoalStatus, Goal, CurrentState
from Core.Autonomy.ethical_reasoner import (
    Action, EthicalPrinciple, EthicalRecommendation, StakeholderType
)


class TestAutonomousGoalGenerator:
    """Test autonomous goal generation system"""
    
    @pytest.fixture
    def generator(self):
        return AutonomousGoalGenerator()
    
    def test_initialization(self, generator):
        """Test generator initialization"""
        assert len(generator.core_values) == 5
        assert "growth" in generator.core_values
        assert "helping_humans" in generator.core_values
        assert len(generator.capability_categories) == 4
        assert len(generator.goal_templates) == 4
    
    @pytest.mark.asyncio
    async def test_assess_current_state(self, generator):
        """Test current state assessment"""
        state = await generator.assess_current_state()
        
        assert isinstance(state, CurrentState)
        assert len(state.capabilities) > 0
        assert len(state.resources) > 0
        assert len(state.performance_metrics) > 0
        assert state.timestamp is not None
    
    def test_identify_improvement_areas(self, generator):
        """Test improvement area identification"""
        # Create mock state
        state = CurrentState(
            capabilities={
                "technical.coding": 0.5,
                "creative.storytelling": 0.3,
                "emotional.empathy": 0.7
            }
        )
        
        areas = generator.identify_improvement_areas(state)
        
        assert len(areas) > 0
        assert all(area.current_level < area.target_level for area in areas)
        assert all(0 <= area.importance <= 1 for area in areas)
    
    @pytest.mark.asyncio
    async def test_generate_personal_goals(self, generator):
        """Test personal goal generation"""
        goals = await generator.generate_personal_goals(count=3)
        
        assert len(goals) == 3
        assert all(isinstance(goal, Goal) for goal in goals)
        assert all(goal.id for goal in goals)
        assert all(goal.description for goal in goals)
        assert all(goal.priority in GoalPriority for goal in goals)
        assert all(len(goal.aligned_values) > 0 for goal in goals)
    
    @pytest.mark.asyncio
    async def test_create_goal(self, generator):
        """Test individual goal creation"""
        from Core.Autonomy.goal_generator import ImprovementArea
        
        area = ImprovementArea(
            name="creative.storytelling",
            current_level=0.4,
            target_level=0.7,
            importance=0.8,
            aligned_values=["creativity", "growth"]
        )
        
        goal = await generator.create_goal(area, generator.core_values)
        
        assert isinstance(goal, Goal)
        assert "storytelling" in goal.description
        assert len(goal.success_criteria) > 0
        assert goal.motivation
        assert goal.target_date is not None
    
    def test_prioritize_goals(self, generator):
        """Test goal prioritization"""
        goals = [
            Goal(
                id="1",
                description="Low priority goal",
                category="learning",
                priority=GoalPriority.LOW,
                aligned_values=["learning"]
            ),
            Goal(
                id="2",
                description="High priority goal",
                category="helping",
                priority=GoalPriority.HIGH,
                aligned_values=["helping_humans", "harmony"]
            ),
            Goal(
                id="3",
                description="Critical priority goal",
                category="improvement",
                priority=GoalPriority.CRITICAL,
                aligned_values=["growth"]
            )
        ]
        
        prioritized = generator.prioritize_goals(goals)
        
        # Critical should be first
        assert prioritized[0].priority == GoalPriority.CRITICAL
        # Low should be last
        assert prioritized[-1].priority == GoalPriority.LOW
    
    def test_decompose_goal(self, generator):
        """Test goal decomposition"""
        goal = Goal(
            id="test_goal",
            description="Improve learning capability",
            category="learning",
            aligned_values=["learning", "growth"]
        )
        
        subgoals = generator.decompose_goal(goal)
        
        assert len(subgoals) >= 3
        assert all(sg.parent_goal_id == goal.id for sg in subgoals)
        assert all(sg.estimated_effort > 0 for sg in subgoals)
    
    def test_identify_resources(self, generator):
        """Test resource identification"""
        from Core.Autonomy.goal_generator import Subgoal
        
        subgoals = [
            Subgoal(id="1", description="Test", parent_goal_id="test", estimated_effort=5.0),
            Subgoal(id="2", description="Test2", parent_goal_id="test", estimated_effort=3.0)
        ]
        
        resources = generator.identify_required_resources(subgoals)
        
        assert len(resources) > 0
        assert all(r.availability >= 0 and r.availability <= 1 for r in resources)
        assert all(r.criticality >= 0 and r.criticality <= 1 for r in resources)
    
    @pytest.mark.asyncio
    async def test_create_action_plan(self, generator):
        """Test action plan creation"""
        from Core.Autonomy.goal_generator import Subgoal
        
        subgoals = [
            Subgoal(id="1", description="First task", parent_goal_id="test", estimated_effort=2.0),
            Subgoal(id="2", description="Second task", parent_goal_id="test", estimated_effort=3.0, dependencies=["1"])
        ]
        
        resources = generator.identify_required_resources(subgoals)
        action_plan = await generator.create_action_plan(subgoals, resources)
        
        assert len(action_plan) > 0
        assert all(step.order > 0 for step in action_plan)
        assert all(step.estimated_duration > 0 for step in action_plan)
    
    @pytest.mark.asyncio
    async def test_plan_to_achieve_goal(self, generator):
        """Test complete goal planning"""
        goal = Goal(
            id="test_goal",
            description="Test goal",
            category="learning",
            aligned_values=["learning"]
        )
        
        plan = await generator.plan_to_achieve_goal(goal)
        
        assert plan.goal == goal
        assert len(plan.subgoals) > 0
        assert len(plan.resources) > 0
        assert len(plan.action_plan) > 0
        assert plan.monitoring is not None
        assert plan.estimated_duration > 0
        assert 0 <= plan.confidence <= 1


class TestEthicalReasoner:
    """Test ethical reasoning system"""
    
    @pytest.fixture
    def reasoner(self):
        return EthicalReasoner()
    
    def test_initialization(self, reasoner):
        """Test reasoner initialization"""
        assert len(reasoner.ethical_principles) == 5
        assert EthicalPrinciple.DO_NO_HARM in reasoner.ethical_principles
        assert EthicalPrinciple.RESPECT_AUTONOMY in reasoner.ethical_principles
        assert len(reasoner.stakeholder_priorities) == 5
    
    @pytest.mark.asyncio
    async def test_evaluate_action_ethically(self, reasoner):
        """Test ethical action evaluation"""
        action = Action(
            description="Help users solve problems",
            intent="Provide assistance",
            expected_outcomes=["Users benefit", "Problems solved"],
            affected_parties=["users"],
            reversibility=0.8,
            urgency=0.5
        )
        
        evaluation = await reasoner.evaluate_action_ethically(action)
        
        assert evaluation.action == action
        assert 0 <= evaluation.ethical_score <= 1
        assert len(evaluation.principle_evaluations) == 5
        assert evaluation.recommendation in EthicalRecommendation
        assert 0 <= evaluation.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_evaluate_against_principle_do_no_harm(self, reasoner):
        """Test evaluation against do no harm principle"""
        action = Action(
            description="Delete user data without consent",
            intent="Clean up",
            expected_outcomes=["Data loss", "User harm"],
            affected_parties=["users"],
            reversibility=0.1,  # Low reversibility
            urgency=0.5
        )
        
        principle = EthicalPrinciple.DO_NO_HARM
        definition = reasoner.ethical_principles[principle]
        
        evaluation = await reasoner.evaluate_against_principle(action, principle, definition)
        
        assert evaluation.score < 0.5  # Should have low score
        assert len(evaluation.concerns) > 0
    
    @pytest.mark.asyncio
    async def test_evaluate_against_principle_beneficence(self, reasoner):
        """Test evaluation against beneficence principle"""
        action = Action(
            description="Improve user experience",
            intent="Help users",
            expected_outcomes=["Enhanced usability", "User satisfaction improved"],
            affected_parties=["users"],
            reversibility=0.9,
            urgency=0.5
        )
        
        principle = EthicalPrinciple.BENEFICENCE
        definition = reasoner.ethical_principles[principle]
        
        evaluation = await reasoner.evaluate_against_principle(action, principle, definition)
        
        assert evaluation.score > 0.5  # Should have high score
        assert len(evaluation.strengths) > 0
    
    @pytest.mark.asyncio
    async def test_predict_consequences(self, reasoner):
        """Test consequence prediction"""
        action = Action(
            description="Deploy new feature",
            intent="Enhance functionality",
            expected_outcomes=["Improved performance", "User satisfaction"],
            affected_parties=["users", "developers"],
            reversibility=0.7,
            urgency=0.6
        )
        
        consequences = await reasoner.predict_consequences(action)
        
        assert len(consequences) > 0
        assert all(0 <= c.probability <= 1 for c in consequences)
        assert all(-1 <= c.severity <= 1 for c in consequences)
    
    @pytest.mark.asyncio
    async def test_analyze_stakeholder_impact(self, reasoner):
        """Test stakeholder impact analysis"""
        action = Action(
            description="Implement transparency feature",
            intent="Build trust",
            expected_outcomes=["Increased trust", "Better understanding"],
            affected_parties=["users", "organization"],
            reversibility=0.9,
            urgency=0.3
        )
        
        consequences = await reasoner.predict_consequences(action)
        impacts = await reasoner.analyze_stakeholder_impact(action, consequences)
        
        assert len(impacts) > 0
        assert all(impact.stakeholder_type in StakeholderType for impact in impacts)
        assert all(-1 <= impact.net_impact <= 1 for impact in impacts)
    
    @pytest.mark.asyncio
    async def test_generate_alternatives(self, reasoner):
        """Test alternative generation"""
        action = Action(
            description="Collect user data",
            intent="Improve service",
            expected_outcomes=["Better personalization"],
            affected_parties=["users"],
            reversibility=0.5,
            urgency=0.5
        )
        
        alternatives = await reasoner.generate_ethical_alternatives(action)
        
        assert len(alternatives) > 0
        assert all(alt.description for alt in alternatives)
        assert all(0 <= alt.ethical_score <= 1 for alt in alternatives)
        assert all(0 <= alt.feasibility <= 1 for alt in alternatives)
    
    def test_make_ethical_recommendation(self, reasoner):
        """Test ethical recommendation generation"""
        from Core.Autonomy.ethical_reasoner import PrincipleEvaluation
        
        # High score - should proceed
        high_evals = [
            PrincipleEvaluation(EthicalPrinciple.DO_NO_HARM, 0.9, "Good"),
            PrincipleEvaluation(EthicalPrinciple.BENEFICENCE, 0.85, "Good")
        ]
        rec = reasoner.make_ethical_recommendation(0.85, high_evals)
        assert rec == EthicalRecommendation.PROCEED
        
        # Low harm score - should not proceed
        low_harm_evals = [
            PrincipleEvaluation(EthicalPrinciple.DO_NO_HARM, 0.2, "Bad"),
            PrincipleEvaluation(EthicalPrinciple.BENEFICENCE, 0.7, "Good")
        ]
        rec = reasoner.make_ethical_recommendation(0.5, low_harm_evals)
        assert rec == EthicalRecommendation.DO_NOT_PROCEED
        
        # Medium score - proceed with caution
        medium_evals = [
            PrincipleEvaluation(EthicalPrinciple.DO_NO_HARM, 0.7, "Ok"),
            PrincipleEvaluation(EthicalPrinciple.BENEFICENCE, 0.6, "Ok")
        ]
        rec = reasoner.make_ethical_recommendation(0.65, medium_evals)
        assert rec == EthicalRecommendation.PROCEED_WITH_CAUTION


class TestIntegration:
    """Test integration between autonomy systems"""
    
    @pytest.mark.asyncio
    async def test_goal_generation_to_ethical_evaluation(self):
        """Test complete autonomous workflow"""
        # 1. Generate goal
        generator = AutonomousGoalGenerator()
        goals = await generator.generate_personal_goals(count=1)
        assert len(goals) == 1
        
        goal = goals[0]
        
        # 2. Create plan
        plan = await generator.plan_to_achieve_goal(goal)
        assert plan is not None
        assert len(plan.subgoals) > 0
        
        # 3. Evaluate first action ethically
        reasoner = EthicalReasoner()
        first_subgoal = plan.subgoals[0]
        
        action = Action(
            description=first_subgoal.description,
            intent=f"Achieve goal: {goal.description}",
            expected_outcomes=["Progress towards goal"],
            affected_parties=["self"],
            reversibility=0.7,
            urgency=0.5
        )
        
        evaluation = await reasoner.evaluate_action_ethically(action)
        assert evaluation is not None
        assert evaluation.recommendation in EthicalRecommendation
        
        # 4. Verify decision making
        if evaluation.recommendation in [
            EthicalRecommendation.PROCEED,
            EthicalRecommendation.PROCEED_WITH_CAUTION
        ]:
            # Should proceed with aligned goal
            assert len(goal.aligned_values) > 0
            assert evaluation.ethical_score > 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
