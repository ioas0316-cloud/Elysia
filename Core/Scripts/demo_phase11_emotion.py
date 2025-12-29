"""
Phase 11 Demo: Emotional Intelligence Enhancement

Demonstrates:
1. Deep emotion recognition from multiple channels
2. Nuanced emotion identification
3. Empathic response generation
4. Emotional support provision
5. Emotional contagion modeling
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from Core.IntelligenceLayer.Consciousness.Emotion import DeepEmotionAnalyzer, EmpathyEngine


async def demo_deep_emotion_recognition():
    """Demonstrate deep emotion recognition"""
    print("\n" + "="*70)
    print("🧠 DEMO 1: Deep Emotion Recognition")
    print("="*70 + "\n")
    
    analyzer = DeepEmotionAnalyzer()
    
    # Test scenarios
    scenarios = [
        {
            "name": "Happy Achievement",
            "inputs": {
                "text": "I'm so excited! I finally got the promotion I've been working towards!",
                "context": {
                    "situation": "Career advancement",
                    "event": "promotion announcement"
                }
            }
        },
        {
            "name": "Anxious Anticipation",
            "inputs": {
                "text": "I'm really worried about the presentation tomorrow. What if I mess up?",
                "context": {
                    "situation": "Upcoming important presentation",
                    "concerns": ["public speaking", "failure"]
                }
            }
        },
        {
            "name": "Disappointment",
            "inputs": {
                "text": "I can't believe they chose someone else. I worked so hard for this.",
                "context": {
                    "situation": "Not selected for opportunity",
                    "trigger": "rejection"
                }
            }
        },
        {
            "name": "Mixed Emotions - Jealousy",
            "inputs": {
                "text": "She got everything I wanted. It's not fair.",
                "context": {
                    "situation": "Comparing to peer success",
                    "trigger": "jealousy"
                }
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   Input: \"{scenario['inputs']['text']}\"")
        print("-" * 70)
        
        analysis = await analyzer.analyze_complex_emotions(scenario['inputs'])
        
        print(f"\n🎯 Primary Emotion: {analysis.primary_emotion.primary_emotion.value.upper()}")
        print(f"   Confidence: {analysis.primary_emotion.confidence:.2%}")
        print(f"   Channels: {', '.join(analysis.primary_emotion.channels_contributing)}")
        
        if analysis.primary_emotion.secondary_emotions:
            print(f"\n🔄 Secondary Emotions:")
            for emotion in analysis.primary_emotion.secondary_emotions:
                print(f"   - {emotion.value}")
        
        if analysis.nuanced_emotions:
            print(f"\n💎 Nuanced Emotions:")
            for emotion in analysis.nuanced_emotions:
                print(f"   - {emotion.value}")
        
        print(f"\n📊 Emotion Characteristics:")
        print(f"   Intensity: {analysis.intensity:.2f}/1.0 ({'Strong' if analysis.intensity > 0.7 else 'Moderate' if analysis.intensity > 0.4 else 'Mild'})")
        print(f"   Estimated Duration: {analysis.duration_estimate/60:.1f} minutes")
        print(f"   Overall Confidence: {analysis.confidence:.2%}")
        
        print(f"\n🔍 Inferred Causes:")
        for cause in analysis.causes:
            print(f"   - {cause}")
        
        print("\n" + "="*70)


async def demo_empathy_system():
    """Demonstrate empathy system"""
    print("\n" + "="*70)
    print("❤️  DEMO 2: Empathy System")
    print("="*70 + "\n")
    
    empathy_engine = EmpathyEngine()
    
    # Test scenarios
    scenarios = [
        {
            "name": "Person feeling sad after loss",
            "emotion": {
                "emotion": "sadness",
                "intensity": 0.8,
                "confidence": 0.9,
                "context": {
                    "situation": "Lost a beloved pet",
                    "trigger": "grief"
                },
                "causes": ["Loss of companionship"]
            }
        },
        {
            "name": "Person feeling angry about injustice",
            "emotion": {
                "emotion": "anger",
                "intensity": 0.7,
                "confidence": 0.85,
                "context": {
                    "situation": "Unfair treatment at work",
                    "beliefs": ["I was treated unfairly", "My efforts weren't recognized"],
                    "values": ["fairness", "respect"]
                },
                "causes": ["Perceived injustice"]
            }
        },
        {
            "name": "Person feeling anxious about future",
            "emotion": {
                "emotion": "anxiety",
                "intensity": 0.6,
                "confidence": 0.75,
                "context": {
                    "situation": "Uncertain about career direction",
                    "concerns": ["making wrong choice", "wasting time"],
                    "needs": ["clarity", "reassurance"]
                },
                "causes": ["Uncertainty about future"]
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   Emotion: {scenario['emotion']['emotion']}")
        print(f"   Intensity: {scenario['emotion']['intensity']:.2f}")
        print("-" * 70)
        
        empathy_result = await empathy_engine.empathize(scenario['emotion'])
        
        print(f"\n🪞 Emotion Mirroring:")
        print(f"   Original: {empathy_result['mirrored_emotion']['original']}")
        print(f"   Mirrored Intensity: {empathy_result['mirrored_emotion']['intensity']:.2f}")
        print(f"   Resonance Quality: {empathy_result['mirrored_emotion']['resonance']:.2%}")
        
        print(f"\n🧠 Empathic Understanding:")
        print(f"   What they feel: {empathy_result['understanding']['what_they_feel']}")
        print(f"   Why they feel it: {empathy_result['understanding']['why_they_feel']}")
        print(f"   What they need: {empathy_result['understanding']['what_they_need']}")
        print(f"   Understanding depth: {empathy_result['understanding']['depth']:.2%}")
        
        print(f"\n💬 Empathic Response:")
        print(f"   Tone: {empathy_result['response']['tone']}")
        print(f"   Message:")
        # Wrap message for better display
        message = empathy_result['response']['message']
        words = message.split()
        lines = []
        current_line = "      "
        for word in words:
            if len(current_line) + len(word) + 1 > 70:
                lines.append(current_line)
                current_line = "      " + word
            else:
                current_line += " " + word if current_line != "      " else word
        if current_line:
            lines.append(current_line)
        for line in lines:
            print(line)
        
        print(f"\n   Validation Statements:")
        for validation in empathy_result['response']['validations']:
            print(f"      - {validation}")
        
        print(f"\n🤝 Emotional Support:")
        print(f"   Type: {empathy_result['support']['type']}")
        print(f"   Actions:")
        for action in empathy_result['support']['actions'][:3]:
            print(f"      - {action}")
        print(f"   Suggestions:")
        for suggestion in empathy_result['support']['suggestions'][:2]:
            print(f"      - {suggestion}")
        
        print(f"\n✅ Validation: {empathy_result['validation']}")
        
        print("\n" + "="*70)


async def demo_emotional_contagion():
    """Demonstrate emotional contagion modeling"""
    print("\n" + "="*70)
    print("👥 DEMO 3: Emotional Contagion (Group Dynamics)")
    print("="*70 + "\n")
    
    empathy_engine = EmpathyEngine()
    
    # Test scenarios
    scenarios = [
        {
            "name": "Celebration - Joyful Group",
            "group_emotions": [
                {"emotion": "joy", "intensity": 0.9},
                {"emotion": "joy", "intensity": 0.8},
                {"emotion": "joy", "intensity": 0.85},
                {"emotion": "excitement", "intensity": 0.75},
                {"emotion": "contentment", "intensity": 0.6}
            ]
        },
        {
            "name": "Crisis - Mixed Anxiety and Fear",
            "group_emotions": [
                {"emotion": "fear", "intensity": 0.7},
                {"emotion": "anxiety", "intensity": 0.8},
                {"emotion": "anxiety", "intensity": 0.75},
                {"emotion": "worry", "intensity": 0.6},
                {"emotion": "calm", "intensity": 0.4}
            ]
        },
        {
            "name": "Conflict - Diverse Emotions",
            "group_emotions": [
                {"emotion": "anger", "intensity": 0.7},
                {"emotion": "frustration", "intensity": 0.6},
                {"emotion": "sadness", "intensity": 0.5},
                {"emotion": "confusion", "intensity": 0.6},
                {"emotion": "calm", "intensity": 0.3}
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   Group Size: {len(scenario['group_emotions'])} people")
        print("-" * 70)
        
        contagion = await empathy_engine.emotional_contagion(scenario['group_emotions'])
        
        print(f"\n🎯 Dominant Emotion: {contagion['dominant_emotion'].upper()}")
        print(f"   Contagion Strength: {contagion['contagion_strength']:.2%}")
        print(f"   Spread Pattern: {contagion['spread_pattern']}")
        
        print(f"\n📊 Emotion Distribution:")
        for emotion, count in contagion['emotion_distribution'].items():
            percentage = (count / contagion['group_size']) * 100
            print(f"   - {emotion}: {count} people ({percentage:.1f}%)")
        
        print(f"\n📈 Group Dynamics:")
        print(f"   Emotional Diversity: {contagion['emotional_diversity']} different emotions")
        
        # Analysis
        if contagion['contagion_strength'] > 0.6:
            analysis = "Strong emotional contagion - the dominant emotion is spreading rapidly through the group."
        elif contagion['contagion_strength'] > 0.3:
            analysis = "Moderate emotional contagion - some emotional influence but diverse feelings persist."
        else:
            analysis = "Weak emotional contagion - group maintains high emotional diversity."
        
        print(f"\n💡 Analysis: {analysis}")
        
        print("\n" + "="*70)


async def demo_integrated_emotional_intelligence():
    """Demonstrate integrated emotional intelligence workflow"""
    print("\n" + "="*70)
    print("🌟 DEMO 4: Integrated Emotional Intelligence")
    print("="*70 + "\n")
    
    print("Scenario: Supporting someone through a difficult emotional experience")
    print("-" * 70)
    
    # Initialize systems
    analyzer = DeepEmotionAnalyzer()
    empathy_engine = EmpathyEngine()
    
    # User input
    user_input = {
        "text": "I'm feeling overwhelmed. Everything is changing so fast and I don't know if I can keep up. I'm scared I'll fail.",
        "context": {
            "situation": "Major life transitions",
            "concerns": ["keeping up with changes", "potential failure"],
            "trigger": "multiple simultaneous changes"
        }
    }
    
    print(f"\n💬 User Input:")
    print(f"   \"{user_input['text']}\"")
    print(f"   Context: {user_input['context']['situation']}")
    
    # Step 1: Analyze emotions
    print(f"\n📊 Step 1: Deep Emotion Analysis")
    print("-" * 70)
    analysis = await analyzer.analyze_complex_emotions(user_input)
    
    print(f"   Primary: {analysis.primary_emotion.primary_emotion.value}")
    print(f"   Nuanced: {', '.join(e.value for e in analysis.nuanced_emotions)}")
    print(f"   Intensity: {analysis.intensity:.2f}")
    
    # Step 2: Generate empathy
    print(f"\n❤️  Step 2: Empathic Response Generation")
    print("-" * 70)
    emotion_dict = {
        "emotion": analysis.primary_emotion.primary_emotion.value,
        "intensity": analysis.intensity,
        "confidence": analysis.confidence,
        "context": user_input["context"],
        "causes": analysis.causes
    }
    
    empathy_result = await empathy_engine.empathize(emotion_dict)
    
    print(f"   Understanding: {empathy_result['understanding']['what_they_feel']}")
    print(f"   Need: {empathy_result['understanding']['what_they_need']}")
    
    # Step 3: Provide support
    print(f"\n🤝 Step 3: Emotional Support")
    print("-" * 70)
    print(f"   Support Type: {empathy_result['support']['type']}")
    
    # Step 4: Complete response
    print(f"\n💭 Step 4: Complete Empathic Response")
    print("=" * 70)
    print("\n   ELYSIA'S RESPONSE:")
    print("   " + "-" * 66)
    
    # Format and display the complete response
    message = empathy_result['response']['message']
    words = message.split()
    lines = []
    current_line = "   "
    for word in words:
        if len(current_line) + len(word) + 1 > 70:
            lines.append(current_line)
            current_line = "   " + word
        else:
            current_line += " " + word if current_line != "   " else word
    if current_line:
        lines.append(current_line)
    
    for line in lines:
        print(line)
    
    print("   " + "-" * 66)
    print(f"\n   [Tone: {empathy_result['response']['tone']}, Support: {empathy_result['support']['type']}]")
    
    print("\n" + "="*70)
    print("✨ Complete emotional intelligence cycle demonstrated!")
    print("="*70)


async def main():
    """Main demo function"""
    print("\n" + "="*70)
    print("🚀 PHASE 11: EMOTIONAL INTELLIGENCE ENHANCEMENT DEMO")
    print("="*70)
    print("\nThis demo showcases Elysia's advanced emotional intelligence:")
    print("  1. Deep Emotion Recognition - Multi-channel emotion analysis")
    print("  2. Empathy System - Genuine empathic responses")
    print("  3. Emotional Contagion - Group emotion dynamics")
    print("  4. Integrated Intelligence - Complete emotional support workflow")
    
    demos = [
        ("Deep Emotion Recognition", demo_deep_emotion_recognition),
        ("Empathy System", demo_empathy_system),
        ("Emotional Contagion", demo_emotional_contagion),
        ("Integrated Emotional Intelligence", demo_integrated_emotional_intelligence)
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
    print("🎉 PHASE 11 DEMO COMPLETE!")
    print("="*70)
    print("\n✨ Elysia's emotional intelligence systems are operational!")
    print("   - Deep multi-channel emotion recognition")
    print("   - Nuanced emotion identification")
    print("   - Genuine empathic understanding")
    print("   - Appropriate emotional support")
    print("   - Group emotional dynamics modeling")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
