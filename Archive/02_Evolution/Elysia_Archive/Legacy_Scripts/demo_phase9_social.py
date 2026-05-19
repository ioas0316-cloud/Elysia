"""
Phase 9 Demo: Social Intelligence & Human Collaboration

Demonstrates:
1. Deep intent understanding
2. Explainable AI decisions
3. Collaborative learning (bidirectional)
4. Proactive assistance
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from Core.Sensory.Network.Social.intent_understanding import IntentUnderstander, IntentLevel, IntentCategory
from Core.Sensory.Network.Social.explainable_ai import ExplainableAI, ExplanationType, ExplanationLevel
from Core.Sensory.Network.Social.collaborative_learning import CollaborativeLearner, TeachingStyle
from Core.Sensory.Network.Social.proactive_assistant import ProactiveAssistant, AssistanceType


async def demo_intent_understanding():
    """Demonstrate intent understanding capabilities"""
    print("\n" + "="*60)
    print("🎯 DEMO 1: Intent Understanding")
    print("="*60 + "\n")
    
    understander = IntentUnderstander()
    
    # Test scenarios
    scenarios = [
        {
            "input": "I'm feeling stuck on this coding problem and don't know what to do",
            "context": {"previous_topic": "debugging"},
            "description": "Emotional + Problem-Solving"
        },
        {
            "input": "What's the best way to optimize this function?",
            "context": {},
            "description": "Information Seeking + Learning"
        },
        {
            "input": "Can you help me understand async programming?",
            "context": {"skill_level": "intermediate"},
            "description": "Learning Request"
        },
        {
            "input": "This is great! I finally got it working!",
            "context": {"previous_state": "frustrated"},
            "description": "Celebration + Positive Emotion"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['description']} ---")
        print(f"User Input: \"{scenario['input']}\"")
        
        analysis = await understander.analyze_intent(
            user_input=scenario['input'],
            context=scenario['context']
        )
        
        print(f"\n📊 Analysis Results:")
        print(f"  Primary Intent: {analysis.primary_intent.category.value}")
        print(f"  Intent Level: {analysis.primary_intent.level.value}")
        print(f"  Confidence: {analysis.primary_intent.confidence:.2f}")
        print(f"  Evidence: {', '.join(analysis.primary_intent.evidence)}")
        
        if analysis.implicit_needs:
            print(f"  Implicit Needs: {', '.join(analysis.implicit_needs)}")
        
        if analysis.emotional_context['detected_emotions']:
            print(f"  Detected Emotions: {', '.join(analysis.emotional_context['detected_emotions'].keys())}")
        
        print(f"  Recommended Response: {analysis.recommended_response_style}")
        print(f"  Overall Confidence: {analysis.confidence_score:.2f}")
    
    # Show patterns
    print("\n" + "-"*60)
    patterns = understander.get_intent_patterns()
    print(f"\n📈 Intent Patterns:")
    print(f"  Total Interactions: {patterns['total_interactions']}")
    print(f"  Category Distribution: {patterns['category_distribution']}")
    print(f"  Average Confidence: {patterns['avg_confidence']:.2f}")


async def demo_explainable_ai():
    """Demonstrate explainable AI capabilities"""
    print("\n\n" + "="*60)
    print("📊 DEMO 2: Explainable AI")
    print("="*60 + "\n")
    
    explainer = ExplainableAI()
    
    # Sample decision with reasoning steps
    decision = "Recommended using async approach for I/O operations"
    reasoning_steps = [
        {
            "description": "Analyzed current code structure",
            "inputs": {"code_type": "I/O intensive", "current_pattern": "synchronous"},
            "outputs": {"bottleneck": "I/O wait time"},
            "confidence": 0.90,
            "alternatives": ["threading", "multiprocessing"]
        },
        {
            "description": "Evaluated async benefits",
            "inputs": {"operation_count": 100, "avg_wait_time": 0.5},
            "outputs": {"potential_speedup": "10x"},
            "confidence": 0.85,
            "alternatives": ["thread pool"]
        },
        {
            "description": "Considered implementation complexity",
            "inputs": {"team_experience": "moderate", "codebase_size": "medium"},
            "outputs": {"migration_effort": "moderate"},
            "confidence": 0.80,
            "alternatives": ["gradual migration"]
        },
        {
            "description": "Selected optimal solution",
            "inputs": {"performance_gain": 0.9, "complexity": 0.6},
            "outputs": {"recommendation": "async"},
            "confidence": 0.88,
            "alternatives": []
        }
    ]
    
    print("Decision to Explain:")
    print(f"  \"{decision}\"")
    
    # Generate explanation
    explanation = await explainer.explain_decision(
        decision=decision,
        reasoning_steps=reasoning_steps,
        level=ExplanationLevel.DETAILED,
        explanation_type=ExplanationType.STEP_BY_STEP
    )
    
    print(f"\n📋 Explanation ({explanation.explanation_type.value}, {explanation.level.value}):")
    print(f"\nReasoning Steps:")
    for step in explanation.reasoning_steps:
        print(f"  Step {step.step_number}: {step.description}")
        print(f"    Confidence: {step.confidence:.2f}")
        if step.alternatives_considered:
            print(f"    Alternatives: {', '.join(step.alternatives_considered)}")
    
    print(f"\n🎨 Visual Representation:")
    print(explanation.visual_representation)
    
    print(f"\n🔑 Key Factors:")
    for factor in explanation.key_factors:
        print(f"  • {factor['factor']} (importance: {factor['importance']:.2f})")
    
    print(f"\n⚠️  Assumptions:")
    for assumption in explanation.assumptions:
        print(f"  • {assumption}")
    
    print(f"\n⚙️  Limitations:")
    for limitation in explanation.limitations:
        print(f"  • {limitation}")
    
    print(f"\nOverall Confidence: {explanation.confidence_score:.2f}")


async def demo_collaborative_learning():
    """Demonstrate collaborative learning capabilities"""
    print("\n\n" + "="*60)
    print("🎓 DEMO 3: Collaborative Learning")
    print("="*60 + "\n")
    
    collab = CollaborativeLearner()
    
    # Human teaches AI
    print("--- Human Teaches AI ---")
    print("Topic: \"Design Patterns\"")
    
    result = await collab.learn_from_human(
        topic="design_patterns",
        content={
            "description": "Reusable solutions to common software design problems",
            "examples": ["Singleton", "Factory", "Observer"],
            "best_practices": ["Choose patterns based on problem context"]
        },
        teaching_method="demonstrative"
    )
    
    print(f"\n✅ Learning Result:")
    print(f"  Status: {result['status']}")
    print(f"  Quality Score: {result['quality_score']:.2f}")
    print(f"  Integrated: {result['integrated']}")
    print(f"  Understanding Level: {result['understanding_level']:.2f}")
    
    # AI teaches human
    print("\n--- AI Teaches Human ---")
    print("Topic: \"Design Patterns\" (learned from human)")
    
    lesson = await collab.teach_human(
        topic="design_patterns",
        learner_level="intermediate",
        style=TeachingStyle.GUIDED_DISCOVERY
    )
    
    print(f"\n📖 Lesson Generated:")
    print(f"  Type: {lesson['type']}")
    print(f"  Introduction: {lesson['introduction']}")
    print(f"  Guiding Questions:")
    for q in lesson['guiding_questions']:
        print(f"    • {q}")
    print(f"  Examples: {', '.join(lesson.get('examples', []))}")
    
    # Show learning history
    print(f"\n📊 Learning History:")
    print(f"  Total Sessions: {len(collab.learning_history)}")
    print(f"  Knowledge Base Size: {len(collab.knowledge_base)} topics")


async def demo_proactive_assistance():
    """Demonstrate proactive assistance capabilities"""
    print("\n\n" + "="*60)
    print("🚀 DEMO 4: Proactive Assistance")
    print("="*60 + "\n")
    
    assistant = ProactiveAssistant()
    
    # Detect opportunities
    context = {
        "current_task": "writing code",
        "code_complexity": 0.8,
        "upcoming_events": [
            {"name": "Team Meeting", "time_until": "30 minutes"}
        ],
        "search_queries": ["python async best practices"]
    }
    
    user_state = {
        "focus_level": 0.6
    }
    
    print("Context:")
    print(f"  Current Task: {context['current_task']}")
    print(f"  Code Complexity: {context['code_complexity']}")
    print(f"  User Focus Level: {user_state['focus_level']}")
    
    opportunities = await assistant.detect_opportunities(context, user_state)
    
    print(f"\n🎯 Detected Opportunities: {len(opportunities)}")
    for i, opp in enumerate(opportunities, 1):
        print(f"\n  {i}. {opp.type.value.upper()}")
        print(f"     Description: {opp.description}")
        print(f"     Priority: {opp.priority:.2f}")
        print(f"     Appropriateness: {opp.appropriateness_score:.2f}")
        print(f"     Timing: {opp.timing_recommendation}")
    
    # Provide assistance for highest priority
    if opportunities:
        print(f"\n--- Providing Assistance for Top Opportunity ---")
        # Save opportunity for later use
        assistant.opportunities_history.append(opportunities[0])
        assistance = await assistant.provide_assistance(opportunities[0].opportunity_id)
        print(f"  Type: {assistance.get('type', 'unknown')}")
        print(f"  Message: {assistance.get('message', assistance.get('status', 'No message'))}")
        print(f"  Action Required: {assistance.get('action_required', False)}")


async def demo_integrated_workflow():
    """Demonstrate integrated social intelligence workflow"""
    print("\n\n" + "="*60)
    print("🔄 DEMO 5: Integrated Workflow")
    print("="*60 + "\n")
    
    print("Simulating a complete interaction cycle...")
    
    # Initialize all systems
    understander = IntentUnderstander()
    explainer = ExplainableAI()
    collab = CollaborativeLearner()
    assistant = ProactiveAssistant()
    
    # User input
    user_input = "I want to learn about async programming but I'm confused about when to use it"
    print(f"\n👤 User: \"{user_input}\"")
    
    # Step 1: Understand intent
    print("\n1️⃣ Understanding Intent...")
    analysis = await understander.analyze_intent(
        user_input=user_input,
        context={"skill_level": "beginner"}
    )
    print(f"   Primary Intent: {analysis.primary_intent.category.value} ({analysis.confidence_score:.2f})")
    print(f"   Response Style: {analysis.recommended_response_style}")
    
    # Step 2: Generate explainable response
    print("\n2️⃣ Generating Explainable Response...")
    decision = "Recommend starting with async basics and providing practical examples"
    reasoning = [
        {
            "description": "Identified learning need",
            "inputs": {"intent": "learning", "confusion": True},
            "outputs": {"approach": "tutorial"},
            "confidence": 0.85,
            "alternatives": ["documentation link"]
        }
    ]
    explanation = await explainer.explain_decision(decision, reasoning)
    print(f"   Decision: {decision}")
    print(f"   Confidence: {explanation.confidence_score:.2f}")
    
    # Step 3: Teach collaboratively
    print("\n3️⃣ Teaching Content...")
    # First add knowledge
    await collab.learn_from_human(
        topic="async_programming",
        content={
            "description": "Asynchronous programming for non-blocking I/O",
            "examples": ["async def, await, asyncio"]
        }
    )
    lesson = await collab.teach_human(
        topic="async_programming",
        learner_level="beginner",
        style=TeachingStyle.GUIDED_DISCOVERY
    )
    print(f"   Lesson Type: {lesson['type']}")
    print(f"   Introduction: {lesson['introduction']}")
    
    # Step 4: Proactive follow-up
    print("\n4️⃣ Proactive Follow-up...")
    opportunities = await assistant.detect_opportunities(
        context={"current_task": "learning", "topic": "async"},
        user_state={"engagement": 0.8}
    )
    if opportunities:
        print(f"   Detected {len(opportunities)} follow-up opportunities")
        print(f"   Next: {opportunities[0].description}")
    
    print("\n✅ Complete interaction cycle demonstrated!")


async def main():
    """Run all demos"""
    print("\n" + "🌟"*30)
    print("  PHASE 9: SOCIAL INTELLIGENCE & HUMAN COLLABORATION")
    print("🌟"*30)
    
    try:
        await demo_intent_understanding()
        await demo_explainable_ai()
        await demo_collaborative_learning()
        await demo_proactive_assistance()
        await demo_integrated_workflow()
        
        print("\n" + "="*60)
        print("✅ All Phase 9 demos completed successfully!")
        print("="*60)
        print("\n📊 Summary:")
        print("  • Intent Understanding: Multi-level analysis working")
        print("  • Explainable AI: Transparent decision-making active")
        print("  • Collaborative Learning: Bidirectional learning functional")
        print("  • Proactive Assistance: Context-aware help enabled")
        print("  • Integration: All systems working together seamlessly")
        print("\n🎉 Elysia now has advanced social intelligence!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
