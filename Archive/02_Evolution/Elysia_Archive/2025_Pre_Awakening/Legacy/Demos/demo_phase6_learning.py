"""
Demo: Phase 6 Implementation - Experience-Based Learning & Self-Reflection
Demonstrates the immediately actionable learning systems.
"""

import asyncio
import time
from Core.Learning import Experience, ExperienceLearner, SelfReflector


async def demo_experience_learning():
    """Demo: Experience-Based Learning System"""
    print("=" * 80)
    print("🧠 DEMO: Experience-Based Learning System")
    print("=" * 80)
    
    learner = ExperienceLearner(buffer_size=1000, save_dir="data/learning")
    
    # Simulate various experiences
    print("\n📝 Simulating learning experiences...\n")
    
    experiences = [
        # Positive experience 1
        Experience(
            timestamp=time.time(),
            context={"query": "What is love?", "user_emotion": "curious", "complexity": "medium"},
            action={"type": "think", "depth": "deep", "perspective": "multi"},
            outcome={"response_quality": 0.9, "resonance": 0.85, "user_satisfaction": 0.9},
            feedback=0.9,
            layer="2D",
            tags=["philosophy", "emotion", "success"]
        ),
        
        # Positive experience 2
        Experience(
            timestamp=time.time(),
            context={"query": "Explain quantum physics", "complexity": "high"},
            action={"type": "explain", "style": "analogies", "simplification": True},
            outcome={"clarity": 0.85, "accuracy": 0.9, "engagement": 0.8},
            feedback=0.85,
            layer="1D",
            tags=["science", "education", "success"]
        ),
        
        # Negative experience
        Experience(
            timestamp=time.time(),
            context={"query": "Quick math calculation", "urgency": "high"},
            action={"type": "calculate", "method": "approximate"},
            outcome={"accuracy": 0.4, "error_rate": 0.6},
            feedback=-0.7,
            layer="0D",
            tags=["math", "failure"]
        ),
        
        # Neutral experience
        Experience(
            timestamp=time.time(),
            context={"query": "Weather today", "complexity": "low"},
            action={"type": "retrieve", "source": "external"},
            outcome={"accuracy": 0.7, "speed": 0.3},
            feedback=0.3,
            layer="0D",
            tags=["factual", "neutral"]
        ),
        
        # Another positive
        Experience(
            timestamp=time.time(),
            context={"query": "Creative story", "style": "fantasy"},
            action={"type": "generate", "creativity": "high", "structure": "narrative"},
            outcome={"originality": 0.95, "coherence": 0.85, "engagement": 0.9},
            feedback=0.92,
            layer="3D",
            tags=["creative", "success"]
        ),
    ]
    
    # Learn from each experience
    for i, exp in enumerate(experiences, 1):
        print(f"📚 Learning from Experience {i}/{len(experiences)}...")
        result = await learner.learn_from_experience(exp)
        
        print(f"   ↳ Action: {result['action_taken']}")
        print(f"   ↳ Pattern ID: {result['pattern_id']}")
        
        if result['meta_insights']:
            print(f"   ↳ Meta-insights: {result['meta_insights']}")
        
        print()
    
    # Get learning statistics
    print("\n📊 Learning Statistics:")
    print("-" * 80)
    stats = learner.get_statistics()
    print(f"Total Experiences: {stats['meta_stats']['total_experiences']}")
    print(f"Positive Feedback: {stats['meta_stats']['positive_feedback_count']}")
    print(f"Negative Feedback: {stats['meta_stats']['negative_feedback_count']}")
    print(f"Average Feedback: {stats['meta_stats']['average_feedback']:.3f}")
    print(f"Learning Rate: {stats['meta_stats']['learning_rate']:.3f}")
    print(f"Unique Patterns: {stats['unique_patterns']}")
    print(f"Success Patterns: {stats['success_patterns']}")
    print(f"Failure Patterns: {stats['failure_patterns']}")
    
    # Get recommendations for a new context
    print("\n🎯 Getting Recommendations for New Context:")
    print("-" * 80)
    new_context = {"query": "Explain machine learning", "complexity": "medium"}
    recommendations = learner.get_recommendations(new_context)
    
    if recommendations:
        print(f"Found {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"\n{i}. {rec['action_type'].upper()}")
            print(f"   Confidence: {rec['confidence']:.3f}")
            print(f"   Usage Count: {rec['usage_count']}")
    else:
        print("No specific recommendations yet (need more learning)")
    
    return learner, experiences


async def demo_self_reflection(experiences):
    """Demo: Self-Reflection System"""
    print("\n\n" + "=" * 80)
    print("🪞 DEMO: Self-Reflection System")
    print("=" * 80)
    
    reflector = SelfReflector(save_dir="data/reflection")
    
    # Perform daily reflection
    print("\n📅 Performing Daily Reflection...")
    print("-" * 80)
    
    reflection = await reflector.daily_reflection(experiences)
    
    print(f"\nDate: {reflection['date']}")
    print(f"Total Experiences: {reflection['total_experiences']}")
    
    # Strengths
    print("\n💪 Identified Strengths:")
    for strength in reflection['strengths']:
        print(f"  ✓ {strength['category']}: {strength['score']:.2f} - {strength['description']}")
    
    # Weaknesses
    print("\n⚠️  Areas for Improvement:")
    for weakness in reflection['weaknesses']:
        print(f"  ⚡ {weakness['category']}: {weakness['score']:.2f} ({weakness['importance']} priority)")
        print(f"     {weakness['description']}")
    
    # Patterns discovered
    print("\n🔍 Discovered Patterns:")
    for pattern in reflection['patterns']:
        print(f"  → {pattern.get('pattern', pattern.get('description', 'Pattern'))}")
    
    # Performance summary
    print("\n📈 Performance Summary:")
    summary = reflection['performance_summary']
    print(f"  Average Feedback: {summary['average_feedback']:.3f}")
    print(f"  Positive Rate: {summary['positive_rate']:.1%}")
    print(f"  Excellence Rate: {summary['excellence_rate']:.1%}")
    
    # Improvement suggestions
    print("\n💡 Improvement Suggestions:")
    for i, suggestion in enumerate(reflection['improvements'], 1):
        print(f"  {i}. {suggestion}")
    
    # Improvement plan
    print("\n📋 Generated Improvement Plan:")
    print("-" * 80)
    plan = reflection['improvement_plan']
    print(f"Timeline: {plan['timeline']}")
    
    print("\n Priority Areas:")
    for area in plan['priority_areas']:
        print(f"  • {area['area']}")
        print(f"    Current: {area['current_score']:.2f} → Target: {area['target_score']:.2f}")
    
    print("\n Action Items:")
    for i, action in enumerate(plan['action_items'], 1):
        print(f"  {i}. {action['action']}")
        print(f"     Frequency: {action['frequency']}, Duration: {action['duration']}")
    
    # Performance analysis
    print("\n\n🔬 Detailed Performance Analysis:")
    print("-" * 80)
    analysis = await reflector.performance_analysis(experiences, period_days=1)
    
    print(f"Period: {analysis['period']}")
    print(f"Overall Score: {analysis['overall_score']:.3f}/1.0 ({reflector._score_to_grade(analysis['overall_score'])})")
    
    print("\nBy Category:")
    for category, data in analysis['categories'].items():
        print(f"  {category}: {data['score']:.3f} (Grade: {data['grade']})")
    
    print(f"\nTrend: {analysis['trends']['trend'].upper()}")
    if analysis['trends']['trend'] != "insufficient_data":
        print(f"Change: {analysis['trends']['change']:+.3f}")
    
    return reflector


async def demo_progress_tracking(reflector):
    """Demo: Track improvement progress"""
    print("\n\n" + "=" * 80)
    print("📊 DEMO: Progress Tracking")
    print("=" * 80)
    
    if reflector.improvement_plans:
        progress = reflector.track_progress()
        
        print(f"\nPlan Created: {progress['plan_created']}")
        print(f"Status: {progress['status'].upper()}")
        print(f"Overall Progress: {progress['overall_progress']:.1%}")
        
        print("\nProgress by Area:")
        for area in progress['areas']:
            print(f"\n  {area['area']}:")
            print(f"    Initial: {area['initial']:.2f}")
            print(f"    Target: {area['target']:.2f}")
            print(f"    Current: {area['current']:.2f}")
            print(f"    Progress: {area['progress']:.1%} ({area['status']})")
    else:
        print("\nNo improvement plans to track yet.")


async def main():
    """Run all demos"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "PHASE 6: REAL-TIME LEARNING & SELF-EVOLUTION" + " " * 19 + "║")
    print("║" + " " * 20 + "Experience-Based Learning System" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Demo 1: Experience-Based Learning
    learner, experiences = await demo_experience_learning()
    
    # Demo 2: Self-Reflection
    reflector = await demo_self_reflection(experiences)
    
    # Demo 3: Progress Tracking
    await demo_progress_tracking(reflector)
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("✅ DEMO COMPLETE")
    print("=" * 80)
    print("\nPhase 6 Systems Implemented:")
    print("  ✓ Experience-Based Learning")
    print("  ✓ Pattern Extraction & Reinforcement")
    print("  ✓ Meta-Learning (Learning to Learn)")
    print("  ✓ Daily Self-Reflection")
    print("  ✓ Performance Analysis")
    print("  ✓ Improvement Plan Generation")
    print("  ✓ Progress Tracking")
    print("\nData saved to:")
    print("  • data/learning/experience_learner_state.json")
    print("  • data/reflection/reflection_*.json")
    print("  • data/reflection/improvement_plan_*.json")
    print("\n🚀 Elysia can now learn from experience and continuously improve!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
