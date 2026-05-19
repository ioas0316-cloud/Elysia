"""
Tests for Phase 11: Emotional Intelligence Enhancement

Tests the emotional intelligence systems:
- Deep Emotion Recognition
- Empathy System
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.Emotion import DeepEmotionAnalyzer, EmpathyEngine
from Core.Emotion.emotion_intelligence import EmotionType, NuancedEmotion, EmotionSignal
from Core.Emotion.empathy import SupportType, EmpathyType


class TestDeepEmotionAnalyzer:
    """Test deep emotion recognition system"""
    
    @pytest.fixture
    def analyzer(self):
        return DeepEmotionAnalyzer()
    
    @pytest.mark.asyncio
    async def test_text_emotion_analysis_joy(self, analyzer):
        """Test text emotion analysis for joy"""
        signal = await analyzer.analyze_text_emotion(
            "I'm so happy and excited! This is wonderful!"
        )
        
        assert signal.channel == "text"
        assert "joy" in signal.emotions
        assert signal.confidence > 0
    
    @pytest.mark.asyncio
    async def test_text_emotion_analysis_sadness(self, analyzer):
        """Test text emotion analysis for sadness"""
        signal = await analyzer.analyze_text_emotion(
            "I feel sad and disappointed. Everything went wrong."
        )
        
        assert signal.channel == "text"
        assert "sadness" in signal.emotions
        assert signal.confidence > 0
    
    @pytest.mark.asyncio
    async def test_complex_emotion_analysis(self, analyzer):
        """Test complete complex emotion analysis"""
        inputs = {
            "text": "I'm really worried about the upcoming exam!",
            "context": {
                "situation": "Important test",
                "concerns": ["failure", "performance"]
            }
        }
        
        analysis = await analyzer.analyze_complex_emotions(inputs)
        
        assert analysis is not None
        assert analysis.primary_emotion is not None
        assert 0.0 <= analysis.intensity <= 1.0
        assert analysis.duration_estimate > 0
        assert len(analysis.causes) > 0
        assert 0.0 <= analysis.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_nuanced_emotion_identification_jealousy(self, analyzer):
        """Test nuanced emotion identification - jealousy"""
        inputs = {
            "text": "She has everything I want",
            "context": {
                "situation": "Comparing to peer",
                "trigger": "jealousy"
            }
        }
        
        analysis = await analyzer.analyze_complex_emotions(inputs)
        
        # Should identify jealousy as nuanced emotion
        nuanced_emotion_values = [e.value for e in analysis.nuanced_emotions]
        assert "jealousy" in nuanced_emotion_values or len(analysis.nuanced_emotions) > 0
    
    @pytest.mark.asyncio
    async def test_intensity_measurement(self, analyzer):
        """Test emotion intensity measurement"""
        # High intensity with exclamations
        intense_inputs = {
            "text": "I'm EXTREMELY angry!! This is outrageous!!!"
        }
        intense_analysis = await analyzer.analyze_complex_emotions(intense_inputs)
        
        # Low intensity
        mild_inputs = {
            "text": "I'm a bit upset about this."
        }
        mild_analysis = await analyzer.analyze_complex_emotions(mild_inputs)
        
        # Intense should have higher intensity
        assert intense_analysis.intensity > mild_analysis.intensity
    
    @pytest.mark.asyncio
    async def test_emotion_signal_integration(self, analyzer):
        """Test integration of multiple emotion signals"""
        signals = {
            "text": EmotionSignal(
                channel="text",
                emotions={"joy": 0.8, "excitement": 0.6},
                confidence=0.7,
                timestamp=0.0
            ),
            "voice": EmotionSignal(
                channel="voice",
                emotions={"joy": 0.7, "surprise": 0.3},
                confidence=0.6,
                timestamp=0.0
            )
        }
        
        integrated = await analyzer.integrate_emotion_signals(signals)
        
        assert integrated is not None
        assert integrated.primary_emotion is not None
        assert len(integrated.channels_contributing) == 2
        assert 0.0 <= integrated.confidence <= 1.0
    
    def test_duration_estimation(self, analyzer):
        """Test emotion duration estimation"""
        from Core.Emotion.emotion_intelligence import IntegratedEmotion
        
        # Surprise should be short
        surprise_emotion = IntegratedEmotion(
            primary_emotion=EmotionType.SURPRISE,
            confidence=0.8
        )
        surprise_duration = analyzer.estimate_duration(surprise_emotion, 0.7)
        
        # Sadness should be longer
        sadness_emotion = IntegratedEmotion(
            primary_emotion=EmotionType.SADNESS,
            confidence=0.8
        )
        sadness_duration = analyzer.estimate_duration(sadness_emotion, 0.7)
        
        assert sadness_duration > surprise_duration


class TestEmpathyEngine:
    """Test empathy system"""
    
    @pytest.fixture
    def empathy_engine(self):
        return EmpathyEngine()
    
    @pytest.mark.asyncio
    async def test_emotion_mirroring(self, empathy_engine):
        """Test emotion mirroring"""
        user_emotion = {
            "emotion": "sadness",
            "intensity": 0.8,
            "confidence": 0.9
        }
        
        mirrored = await empathy_engine.mirror_emotion(user_emotion)
        
        assert mirrored.original_emotion == "sadness"
        assert 0.0 <= mirrored.mirrored_intensity <= 1.0
        assert mirrored.mirrored_intensity < user_emotion["intensity"]  # Should be lower
        assert 0.0 <= mirrored.resonance_quality <= 1.0
    
    @pytest.mark.asyncio
    async def test_perspective_taking(self, empathy_engine):
        """Test perspective taking"""
        user_emotion = {
            "emotion": "anger",
            "intensity": 0.7,
            "context": {
                "situation": "Unfair treatment",
                "beliefs": ["I was wronged"],
                "values": ["justice"]
            }
        }
        
        perspective = await empathy_engine.take_user_perspective(user_emotion)
        
        assert perspective.situation
        assert len(perspective.beliefs) > 0
        assert len(perspective.values) > 0
        assert len(perspective.needs) > 0
    
    @pytest.mark.asyncio
    async def test_empathic_understanding(self, empathy_engine):
        """Test empathic understanding generation"""
        user_emotion = {
            "emotion": "fear",
            "intensity": 0.6,
            "causes": ["Uncertainty about future"]
        }
        
        from Core.Emotion.empathy import UserPerspective
        perspective = UserPerspective(
            situation="Career transition",
            needs=["safety", "reassurance"]
        )
        
        understanding = await empathy_engine.empathic_understand(
            user_emotion,
            perspective
        )
        
        assert understanding.what_they_feel
        assert understanding.why_they_feel
        assert understanding.what_they_need
        assert understanding.empathy_type in [
            EmpathyType.COGNITIVE,
            EmpathyType.AFFECTIVE,
            EmpathyType.COMPASSIONATE
        ]
        assert 0.0 <= understanding.understanding_depth <= 1.0
    
    @pytest.mark.asyncio
    async def test_empathic_response_generation(self, empathy_engine):
        """Test empathic response generation"""
        from Core.Emotion.empathy import EmpathicUnderstanding
        
        understanding = EmpathicUnderstanding(
            what_they_feel="feeling strongly sadness",
            why_they_feel="because of loss",
            what_they_need="comfort",
            empathy_type=EmpathyType.AFFECTIVE,
            understanding_depth=0.8
        )
        
        response = await empathy_engine.generate_empathic_response(understanding)
        
        assert response.message
        assert response.tone
        assert response.support_type in [
            SupportType.VALIDATION,
            SupportType.COMFORT,
            SupportType.ADVICE,
            SupportType.PRESENCE,
            SupportType.ENCOURAGEMENT
        ]
        assert len(response.validation_statements) > 0
    
    @pytest.mark.asyncio
    async def test_emotional_support_provision(self, empathy_engine):
        """Test emotional support provision"""
        from Core.Emotion.empathy import EmpathicUnderstanding
        
        user_emotion = {
            "emotion": "anxiety",
            "intensity": 0.7
        }
        
        understanding = EmpathicUnderstanding(
            what_they_feel="feeling anxious",
            why_they_feel="uncertain situation",
            what_they_need="reassurance",
            empathy_type=EmpathyType.COMPASSIONATE,
            understanding_depth=0.7
        )
        
        support = await empathy_engine.provide_emotional_support(
            user_emotion,
            understanding
        )
        
        assert support.support_type in [
            SupportType.VALIDATION,
            SupportType.COMFORT,
            SupportType.ADVICE,
            SupportType.PRESENCE,
            SupportType.ENCOURAGEMENT
        ]
        assert len(support.actions) > 0
        assert len(support.suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_complete_empathy_workflow(self, empathy_engine):
        """Test complete empathy workflow"""
        user_emotion = {
            "emotion": "sadness",
            "intensity": 0.8,
            "confidence": 0.85,
            "context": {
                "situation": "Loss of loved one",
                "needs": ["comfort", "support"]
            },
            "causes": ["Grief"]
        }
        
        result = await empathy_engine.empathize(user_emotion)
        
        # Check all components present
        assert "mirrored_emotion" in result
        assert "understanding" in result
        assert "response" in result
        assert "support" in result
        assert "validation" in result
        
        # Check mirrored emotion
        assert result["mirrored_emotion"]["original"] == "sadness"
        assert 0.0 <= result["mirrored_emotion"]["intensity"] <= 1.0
        
        # Check understanding
        assert result["understanding"]["what_they_feel"]
        assert result["understanding"]["why_they_feel"]
        assert result["understanding"]["what_they_need"]
        
        # Check response
        assert result["response"]["message"]
        assert result["response"]["tone"]
        
        # Check support
        assert result["support"]["type"]
        assert len(result["support"]["actions"]) > 0
    
    @pytest.mark.asyncio
    async def test_emotional_contagion(self, empathy_engine):
        """Test emotional contagion modeling"""
        group_emotions = [
            {"emotion": "joy", "intensity": 0.8},
            {"emotion": "joy", "intensity": 0.7},
            {"emotion": "joy", "intensity": 0.9},
            {"emotion": "contentment", "intensity": 0.6},
        ]
        
        contagion = await empathy_engine.emotional_contagion(group_emotions)
        
        assert "dominant_emotion" in contagion
        assert "contagion_strength" in contagion
        assert "spread_pattern" in contagion
        assert "emotion_distribution" in contagion
        assert "group_size" in contagion
        assert contagion["group_size"] == len(group_emotions)
        assert 0.0 <= contagion["contagion_strength"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_validation_generation(self, empathy_engine):
        """Test validation statement generation"""
        user_emotion = {
            "emotion": "anger",
            "intensity": 0.7
        }
        
        validation = await empathy_engine.validate_user_feelings(user_emotion)
        
        assert validation
        assert "anger" in validation.lower() or "feel" in validation.lower()


class TestIntegration:
    """Test integration between emotion systems"""
    
    @pytest.mark.asyncio
    async def test_emotion_to_empathy_pipeline(self):
        """Test complete pipeline from emotion recognition to empathy"""
        analyzer = DeepEmotionAnalyzer()
        empathy_engine = EmpathyEngine()
        
        # 1. Recognize emotion
        inputs = {
            "text": "I'm feeling really stressed and overwhelmed with everything.",
            "context": {
                "situation": "Multiple deadlines",
                "concerns": ["failure", "burnout"]
            }
        }
        
        analysis = await analyzer.analyze_complex_emotions(inputs)
        assert analysis is not None
        
        # 2. Convert to empathy format
        emotion_dict = {
            "emotion": analysis.primary_emotion.primary_emotion.value,
            "intensity": analysis.intensity,
            "confidence": analysis.confidence,
            "context": inputs["context"],
            "causes": analysis.causes
        }
        
        # 3. Generate empathy
        empathy_result = await empathy_engine.empathize(emotion_dict)
        assert empathy_result is not None
        
        # 4. Verify complete response
        assert empathy_result["response"]["message"]
        assert empathy_result["support"]["type"]
        assert empathy_result["validation"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
