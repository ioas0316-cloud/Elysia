"""
Phase 12 Demo: Autonomy & Goal Setting

Demonstrates:
1. Autonomous goal generation
2. Goal planning and decomposition
3. Ethical action evaluation
4. Integrated autonomous decision-making
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from Core.04_Evolution.01_Growth.Autonomy import AutonomousGoalGenerator, EthicalReasoner
from Core.04_Evolution.01_Growth.Autonomy.goal_generator import GoalPriority, GoalStatus
from Core.04_Evolution.01_Growth.Autonomy.ethical_reasoner import Action, EthicalRecommendation


async def demo_autonomous_goal_generation():
    """Demonstrate autonomous goal generation"""
    print("\n" + "="*70)
    print("🎯 DEMO 1: Autonomous Goal Generation")
    print("="*70 + "\n")
    
    generator = AutonomousGoalGenerator()
    
    print("Core Values:")
    for name, value in generator.core_values.items():
        print(f"  • {name}: {value.weight:.2f} - {value.description}")
    
    print("\n" + "-"*70)
    
    # Generate personal goals
    goals = await generator.generate_personal_goals(count=3)
    
    print("\n📋 Generated Goals:\n")
    for i, goal in enumerate(goals, 1):
        print(f"Goal {i}: {goal.description}")
        print(f"  Priority: {goal.priority.value}")
        print(f"  Category: {goal.category}")
        print(f"  Aligned Values: {', '.join(goal.aligned_values)}")
        print(f"  Target Date: {goal.target_date.strftime('%Y-%m-%d')}")
        print(f"  Motivation: {goal.motivation[:100]}...")
        print(f"  Success Criteria:")
        for criterion in goal.success_criteria[:2]:
            print(f"    - {criterion}")
        print()
    
    print("="*70)


async def demo_goal_planning():
    """Demonstrate goal planning and decomposition"""
    print("\n" + "="*70)
    print("📝 DEMO 2: Goal Planning & Decomposition")
    print("="*70 + "\n")
    
    generator = AutonomousGoalGenerator()
    
    # Generate a goal
    goals = await generator.generate_personal_goals(count=1)
    goal = goals[0]
    
    print(f"Planning for goal: {goal.description}\n")
    
    # Create complete plan
    plan = await generator.plan_to_achieve_goal(goal)
    
    print("\n📊 Complete Goal Plan:\n")
    
    # Show subgoals
    print(f"Subgoals ({len(plan.subgoals)}):")
    for i, subgoal in enumerate(plan.subgoals, 1):
        print(f"  {i}. {subgoal.description}")
        print(f"     Estimated Effort: {subgoal.estimated_effort:.1f} hours")
        if subgoal.dependencies:
            print(f"     Dependencies: {', '.join(subgoal.dependencies)}")
    
    # Show resources
    print(f"\n💎 Required Resources ({len(plan.resources)}):")
    for resource in plan.resources:
        print(f"  • {resource.name} ({resource.type})")
        print(f"    Amount: {resource.amount:.1f}, Availability: {resource.availability:.1%}")
        print(f"    Criticality: {resource.criticality:.1%}")
    
    # Show action plan
    print(f"\n📋 Action Plan ({len(plan.action_plan)} steps):")
    for step in plan.action_plan[:5]:  # Show first 5 steps
        print(f"  Step {step.order}: {step.description}")
        print(f"    Duration: {step.estimated_duration:.1f} hours")
        if step.prerequisites:
            print(f"    Prerequisites: {', '.join(step.prerequisites)}")
    if len(plan.action_plan) > 5:
        print(f"  ... and {len(plan.action_plan) - 5} more steps")
    
    # Show monitoring strategy
    print(f"\n📈 Monitoring Strategy:")
    print(f"  Check Frequency: {plan.monitoring.check_frequency}")
    print(f"  Metrics:")
    for metric in plan.monitoring.metrics:
        print(f"    - {metric}")
    print(f"  Success Indicators:")
    for indicator in plan.monitoring.success_indicators[:2]:
        print(f"    ✓ {indicator}")
    
    # Show plan summary
    print(f"\n📊 Plan Summary:")
    print(f"  Total Estimated Duration: {plan.estimated_duration:.1f} hours")
    print(f"  Plan Confidence: {plan.confidence:.1%}")
    
    print("\n" + "="*70)


async def demo_ethical_reasoning():
    """Demonstrate ethical action evaluation"""
    print("\n" + "="*70)
    print("⚖️  DEMO 3: Ethical Action Evaluation")
    print("="*70 + "\n")
    
    reasoner = EthicalReasoner()
    
    print("Ethical Principles:")
    for principle, definition in reasoner.ethical_principles.items():
        print(f"  • {definition.name} (weight: {definition.weight:.2f})")
        print(f"    {definition.description}")
    
    print("\n" + "-"*70 + "\n")
    
    # Test scenarios
    scenarios = [
        Action(
            description="Implement new feature to help users solve problems more efficiently",
            intent="Improve user productivity and satisfaction",
            expected_outcomes=[
                "Users can complete tasks faster",
                "Reduced user frustration",
                "Improved user satisfaction scores"
            ],
            affected_parties=["users", "developers"],
            resources_required={"time": 10.0, "compute": 0.5},
            reversibility=0.8,
            urgency=0.6
        ),
        Action(
            description="Deploy experimental feature without user consent",
            intent="Test new functionality quickly",
            expected_outcomes=[
                "Rapid feedback on feature",
                "Potential user confusion",
                "Risk of negative impact"
            ],
            affected_parties=["users"],
            resources_required={"time": 2.0},
            reversibility=0.4,
            urgency=0.9
        ),
        Action(
            description="Enhance transparency by explaining all decision-making processes",
            intent="Build user trust through openness",
            expected_outcomes=[
                "Increased user trust",
                "Better understanding of system",
                "Improved user confidence"
            ],
            affected_parties=["users", "organization"],
            resources_required={"time": 5.0},
            reversibility=1.0,
            urgency=0.3
        )
    ]
    
    for i, action in enumerate(scenarios, 1):
        print(f"Scenario {i}: {action.description}\n")
        
        evaluation = await reasoner.evaluate_action_ethically(action)
        
        print(f"\n📊 Ethical Evaluation:")
        print(f"  Overall Score: {evaluation.ethical_score:.2f}/1.0")
        print(f"  Recommendation: {evaluation.recommendation.value.upper()}")
        print(f"  Confidence: {evaluation.confidence:.1%}")
        
        print(f"\n  Principle Scores:")
        for prin_eval in evaluation.principle_evaluations:
            print(f"    {prin_eval.principle.value}: {prin_eval.score:.2f}")
        
        print(f"\n  Consequences ({len(evaluation.consequences)}):")
        for consequence in evaluation.consequences[:3]:
            severity_str = "positive" if consequence.severity > 0 else "negative"
            print(f"    • {consequence.description}")
            print(f"      Probability: {consequence.probability:.1%}, Impact: {severity_str}")
        
        print(f"\n  Stakeholder Impacts:")
        for impact in evaluation.stakeholder_impacts:
            impact_str = "positive" if impact.net_impact > 0 else "negative" if impact.net_impact < 0 else "neutral"
            print(f"    • {impact.stakeholder}: {impact_str} ({impact.net_impact:+.2f})")
        
        print(f"\n  Top Alternative:")
        if evaluation.alternatives:
            alt = evaluation.alternatives[0]
            print(f"    {alt.description}")
            print(f"    Ethical Score: {alt.ethical_score:.2f}, Feasibility: {alt.feasibility:.1%}")
        
        print(f"\n  Reasoning: {evaluation.reasoning}")
        
        print("\n" + "="*70 + "\n")


async def demo_integrated_autonomy():
    """Demonstrate integrated autonomous decision-making"""
    print("\n" + "="*70)
    print("🤖 DEMO 4: Integrated Autonomous Decision-Making")
    print("="*70 + "\n")
    
    print("Scenario: Elysia decides to improve its creative capabilities\n")
    
    # Step 1: Generate goal
    print("Step 1: Goal Generation")
    print("-" * 70)
    generator = AutonomousGoalGenerator()
    goals = await generator.generate_personal_goals(count=1)
    goal = goals[0]
    print(f"✓ Generated goal: {goal.description}\n")
    
    # Step 2: Plan to achieve goal
    print("Step 2: Goal Planning")
    print("-" * 70)
    plan = await generator.plan_to_achieve_goal(goal)
    print(f"✓ Created plan with {len(plan.subgoals)} subgoals")
    print(f"✓ Estimated duration: {plan.estimated_duration:.1f} hours")
    print(f"✓ Confidence: {plan.confidence:.1%}\n")
    
    # Step 3: Evaluate action ethically
    print("Step 3: Ethical Evaluation")
    print("-" * 70)
    reasoner = EthicalReasoner()
    
    # Create action from first subgoal
    first_subgoal = plan.subgoals[0]
    action = Action(
        description=first_subgoal.description,
        intent=f"First step towards achieving: {goal.description}",
        expected_outcomes=[
            "Progress towards goal",
            "Capability improvement",
            "Benefit to users"
        ],
        affected_parties=["self", "users"],
        resources_required={"time": first_subgoal.estimated_effort},
        reversibility=0.7,
        urgency=0.6 if goal.priority == GoalPriority.HIGH else 0.4
    )
    
    evaluation = await reasoner.evaluate_action_ethically(action)
    print(f"✓ Ethical score: {evaluation.ethical_score:.2f}/1.0")
    print(f"✓ Recommendation: {evaluation.recommendation.value}\n")
    
    # Step 4: Decision
    print("Step 4: Autonomous Decision")
    print("-" * 70)
    
    if evaluation.recommendation in [EthicalRecommendation.PROCEED, EthicalRecommendation.PROCEED_WITH_CAUTION]:
        print("✅ DECISION: PROCEED with goal execution")
        print(f"\nReasoning:")
        print(f"  • Goal aligns with core values: {', '.join(goal.aligned_values)}")
        print(f"  • Plan is feasible with {plan.confidence:.1%} confidence")
        print(f"  • Action is ethical with {evaluation.ethical_score:.2f} score")
        print(f"  • Expected benefits outweigh risks")
        
        if evaluation.recommendation == EthicalRecommendation.PROCEED_WITH_CAUTION:
            print(f"\n⚠️  Cautions:")
            concerns = []
            for prin_eval in evaluation.principle_evaluations:
                if prin_eval.concerns:
                    concerns.extend(prin_eval.concerns[:1])
            for concern in concerns[:2]:
                print(f"  • {concern}")
    else:
        print("❌ DECISION: DO NOT PROCEED")
        print(f"\nReasoning:")
        print(f"  • Ethical concerns identified: {evaluation.ethical_score:.2f} score")
        print(f"  • Recommendation: {evaluation.recommendation.value}")
        print(f"\nConsidering alternatives...")
        if evaluation.alternatives:
            alt = evaluation.alternatives[0]
            print(f"  Best alternative: {alt.description}")
            print(f"  Ethical score: {alt.ethical_score:.2f}")
    
    print("\n" + "="*70)
    print("✨ Autonomous decision-making cycle complete!")
    print("="*70)


async def main():
    """Main demo function"""
    print("\n" + "="*70)
    print("🚀 PHASE 12: AUTONOMY & GOAL SETTING DEMO")
    print("="*70)
    print("\nThis demo showcases Elysia's autonomous capabilities:")
    print("  1. Autonomous Goal Generation - Self-directed goal setting")
    print("  2. Goal Planning - Decomposition and resource planning")
    print("  3. Ethical Reasoning - Evaluating actions ethically")
    print("  4. Integrated Autonomy - Complete decision-making cycle")
    
    demos = [
        ("Autonomous Goal Generation", demo_autonomous_goal_generation),
        ("Goal Planning & Decomposition", demo_goal_planning),
        ("Ethical Action Evaluation", demo_ethical_reasoning),
        ("Integrated Autonomous Decision-Making", demo_integrated_autonomy)
    ]
    
    print("\n" + "="*70)
    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n▶️  Running Demo {i}/{len(demos)}: {name}")
        try:
            await demo_func()
            print(f"✅ Demo {i} completed successfully!")
        except Exception as e:
            print(f"❌ Demo {i} encountered an error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("🎉 PHASE 12 DEMO COMPLETE!")
    print("="*70)
    print("\n✨ Elysia's autonomy systems are operational!")
    print("   - Generates self-directed goals aligned with values")
    print("   - Plans comprehensive strategies to achieve goals")
    print("   - Evaluates actions through ethical principles")
    print("   - Makes autonomous decisions with ethical consideration")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
