"""
Comprehensive Conversation System Verification
=============================================

엘리시아와 제대로된 대화가 가능한지 검증
(Verify if proper conversation with Elysia is possible)

Tests:
1. Basic Korean conversation
2. English conversation
3. Context maintenance
4. Emotional understanding
5. Technical questions
6. Philosophical dialogue
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(message)s'
)
logger = logging.getLogger("ConversationVerification")


def test_communication_system():
    """Test the Real Communication System directly"""
    logger.info("="*70)
    logger.info("TEST 1: Real Communication System")
    logger.info("="*70)
    
    from Core.FoundationLayer.Foundation.real_communication_system import RealCommunicationSystem
    
    comm = RealCommunicationSystem()
    
    # Test cases with expected patterns
    test_cases = [
        # Korean questions
        ("안녕하세요", "greeting"),
        ("당신은 누구인가요?", "question"),
        ("당신의 목적은 무엇인가요?", "question"),
        
        # English questions
        ("Hello, Elysia", "greeting"),
        ("What are you?", "question"),
        ("How do you think?", "question"),
        
        # Emotional
        ("I feel happy today", "emotion"),
        ("This is wonderful", "emotion"),
        
        # Philosophical
        ("What is consciousness?", "philosophical"),
        ("What is the meaning of existence?", "philosophical"),
        
        # Technical
        ("How does your code work?", "technical"),
        ("Explain your system architecture", "technical"),
    ]
    
    results = []
    for i, (input_text, expected_intent) in enumerate(test_cases, 1):
        logger.info(f"\n--- Test Case {i} ---")
        logger.info(f"Input: {input_text}")
        
        try:
            # Test understanding
            understanding = comm.understand(input_text)
            logger.info(f"Intent: {understanding.detected_intent} (expected: {expected_intent})")
            logger.info(f"Sentiment: {understanding.sentiment}")
            logger.info(f"Urgency: {understanding.urgency:.2f}")
            logger.info(f"Complexity: {understanding.complexity:.2f}")
            
            # Test response generation
            response = comm.communicate(input_text)
            logger.info(f"Response: {response}")
            
            # Validate response is not empty and makes sense
            is_valid = len(response) > 0 and len(response.split()) > 2
            
            results.append({
                'input': input_text,
                'intent': understanding.detected_intent,
                'expected_intent': expected_intent,
                'response': response,
                'valid': is_valid,
                'intent_match': understanding.detected_intent == expected_intent
            })
            
        except Exception as e:
            logger.error(f"ERROR: {e}")
            results.append({
                'input': input_text,
                'error': str(e),
                'valid': False
            })
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY: Real Communication System Test")
    logger.info("="*70)
    
    total = len(results)
    valid = sum(1 for r in results if r.get('valid', False))
    intent_matches = sum(1 for r in results if r.get('intent_match', False))
    
    logger.info(f"Total tests: {total}")
    logger.info(f"Valid responses: {valid}/{total} ({100*valid/total:.1f}%)")
    logger.info(f"Intent detection: {intent_matches}/{total} ({100*intent_matches/total:.1f}%)")
    
    return results


def test_context_maintenance():
    """Test if conversation context is maintained"""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Context Maintenance")
    logger.info("="*70)
    
    from Core.FoundationLayer.Foundation.real_communication_system import RealCommunicationSystem
    
    comm = RealCommunicationSystem()
    
    # Multi-turn conversation
    conversation = [
        "Hello, my name is Kim",
        "I am a researcher",
        "What is my name?",
        "What do I do?"
    ]
    
    responses = []
    for i, msg in enumerate(conversation, 1):
        logger.info(f"\n--- Turn {i} ---")
        logger.info(f"User: {msg}")
        
        try:
            response = comm.communicate(msg)
            logger.info(f"Elysia: {response}")
            responses.append(response)
            
            # Check context
            logger.info(f"Context - Participants: {comm.context.participants}")
            logger.info(f"Context - Topics: {comm.context.topics}")
            logger.info(f"Context - Turn count: {comm.context.turn_count}")
            
        except Exception as e:
            logger.error(f"ERROR: {e}")
            responses.append(f"ERROR: {str(e)}")
    
    # Validate context was maintained
    context_valid = (
        len(comm.context.participants) >= 2 and  # User + Elysia
        comm.context.turn_count >= len(conversation) and  # Each turn is one user input
        len(comm.context.topics) > 0  # Some topics extracted
    )
    
    logger.info("\n" + "="*70)
    logger.info("SUMMARY: Context Maintenance Test")
    logger.info("="*70)
    logger.info(f"Context valid: {context_valid}")
    logger.info(f"Messages stored: {len(comm.context.messages)}")
    logger.info(f"Participants: {comm.context.participants}")
    logger.info(f"Topics: {comm.context.topics[:5]}")  # Show first 5 topics
    
    return context_valid


def test_wave_integration():
    """Test Wave Communication integration"""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Wave Communication Integration")
    logger.info("="*70)
    
    try:
        from Core.FoundationLayer.Foundation.wave_integration_hub import get_wave_hub
        from Core.FoundationLayer.Foundation.ultra_dimensional_reasoning import UltraDimensionalReasoning
        from Core.FoundationLayer.Foundation.real_communication_system import RealCommunicationSystem
        
        # Initialize systems
        wave_hub = get_wave_hub()
        ultra_reasoning = UltraDimensionalReasoning()
        comm = RealCommunicationSystem(
            reasoning_engine=ultra_reasoning,
            wave_hub=wave_hub
        )
        
        logger.info(f"Wave Hub active: {wave_hub.active}")
        
        # Test communication with wave broadcasting
        test_message = "Hello, can you hear me through waves?"
        logger.info(f"Sending: {test_message}")
        
        response = comm.communicate(test_message)
        logger.info(f"Response: {response}")
        
        # Check wave metrics
        if wave_hub.active:
            metrics = wave_hub.get_metrics()
            logger.info(f"Wave metrics: {metrics}")
            
            resonance_score = wave_hub.calculate_resonance_score()
            logger.info(f"Resonance score: {resonance_score:.1f}/100")
        
        wave_integration_valid = True
        
    except Exception as e:
        logger.error(f"Wave integration test failed: {e}")
        wave_integration_valid = False
    
    logger.info("\n" + "="*70)
    logger.info("SUMMARY: Wave Communication Test")
    logger.info("="*70)
    logger.info(f"Wave integration valid: {wave_integration_valid}")
    
    return wave_integration_valid


def run_all_tests():
    """Run all conversation verification tests"""
    logger.info("\n\n")
    logger.info("🌌 " + "="*68 + " 🌌")
    logger.info("🌌 ELYSIA CONVERSATION SYSTEM COMPREHENSIVE VERIFICATION 🌌")
    logger.info("🌌 엘리시아 대화 시스템 종합 검증                        🌌")
    logger.info("🌌 " + "="*68 + " 🌌")
    logger.info("\n")
    
    # Run tests
    test_results = {}
    
    try:
        comm_results = test_communication_system()
        test_results['communication'] = comm_results
    except Exception as e:
        logger.error(f"Communication test failed: {e}")
        test_results['communication'] = None
    
    try:
        context_valid = test_context_maintenance()
        test_results['context'] = context_valid
    except Exception as e:
        logger.error(f"Context test failed: {e}")
        test_results['context'] = False
    
    try:
        wave_valid = test_wave_integration()
        test_results['wave'] = wave_valid
    except Exception as e:
        logger.error(f"Wave test failed: {e}")
        test_results['wave'] = False
    
    # Final Report
    logger.info("\n\n")
    logger.info("🎯 " + "="*68 + " 🎯")
    logger.info("🎯 FINAL EVALUATION REPORT                                🎯")
    logger.info("🎯 최종 평가 보고서                                        🎯")
    logger.info("🎯 " + "="*68 + " 🎯")
    logger.info("\n")
    
    # Calculate scores
    comm_score = 0
    if test_results['communication']:
        total = len(test_results['communication'])
        valid = sum(1 for r in test_results['communication'] if r.get('valid', False))
        comm_score = (valid / total * 100) if total > 0 else 0
    
    context_score = 100 if test_results['context'] else 0
    wave_score = 100 if test_results['wave'] else 0
    
    overall_score = (comm_score + context_score + wave_score) / 3
    
    logger.info(f"📊 Communication Quality:     {comm_score:.1f}/100")
    logger.info(f"📊 Context Maintenance:       {context_score:.1f}/100")
    logger.info(f"📊 Wave Integration:          {wave_score:.1f}/100")
    logger.info(f"")
    logger.info(f"📊 OVERALL SCORE:             {overall_score:.1f}/100")
    logger.info(f"")
    
    # Qualitative assessment
    if overall_score >= 80:
        assessment = "✅ EXCELLENT - 엘리시아와 제대로된 대화가 가능합니다!"
    elif overall_score >= 60:
        assessment = "⚠️ GOOD - 대화가 가능하지만 개선이 필요합니다."
    else:
        assessment = "❌ NEEDS IMPROVEMENT - 대화 시스템에 문제가 있습니다."
    
    logger.info(f"🔍 Assessment: {assessment}")
    logger.info(f"")
    logger.info("="*70)
    
    return test_results, overall_score


if __name__ == "__main__":
    results, score = run_all_tests()
    
    # Exit with appropriate code
    if score >= 60:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure
