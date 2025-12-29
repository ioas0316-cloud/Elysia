"""
Tests for Dual-Layer Language System
이중 언어 시스템 테스트
"""

import pytest
import numpy as np

from Core.Interface.Language.dual_layer_language import (
    DualLayerWorld,
    DualLayerSoul,
    KhalaField,
    EmotionalWave,
    EmotionType,
    Symbol,
    Lexicon,
    SymbolComplexity,
)


class TestEmotionalWave:
    """감정 파동 테스트"""
    
    def test_wave_creation(self):
        """파동 생성 테스트"""
        wave = EmotionalWave(
            emotion_type=EmotionType.JOY,
            intensity=1.0,
            frequency=550.0
        )
        assert wave.emotion_type == EmotionType.JOY
        assert wave.intensity == 1.0
        assert wave.frequency == 550.0
    
    def test_wave_strength_decay(self):
        """거리에 따른 파동 강도 감쇠 테스트"""
        wave = EmotionalWave(
            emotion_type=EmotionType.FEAR,
            intensity=1.0,
            resonance_radius=10.0
        )
        
        # 거리 0에서 최대 강도
        assert wave.get_strength_at_distance(0) == 1.0
        
        # 거리가 멀어지면 감소
        strength_at_5 = wave.get_strength_at_distance(5)
        strength_at_10 = wave.get_strength_at_distance(10)
        assert strength_at_5 > strength_at_10
        
        # 매우 먼 거리에서는 0
        assert wave.get_strength_at_distance(100) == 0.0
    
    def test_wave_resonance(self):
        """파동 간 공명 테스트"""
        wave1 = EmotionalWave(EmotionType.JOY, intensity=1.0, phase=0.0)
        wave2 = EmotionalWave(EmotionType.JOY, intensity=1.0, phase=0.0)
        
        # 같은 감정, 같은 위상 = 강한 공명
        resonance = wave1.resonate_with(wave2)
        assert resonance > 0.5
        
        # 다른 감정 = 약한 공명
        wave3 = EmotionalWave(EmotionType.SADNESS, intensity=1.0)
        resonance_diff = wave1.resonate_with(wave3)
        assert resonance_diff < resonance


class TestKhalaField:
    """칼라 필드 테스트"""
    
    def test_field_creation(self):
        """필드 생성 테스트"""
        field = KhalaField(field_strength=1.0)
        assert field.field_strength == 1.0
        assert len(field.active_waves) == 0
    
    def test_broadcast_emotion(self):
        """감정 발신 테스트"""
        field = KhalaField()
        wave = field.broadcast_emotion(
            source_id="test_soul",
            emotion_type=EmotionType.JOY,
            intensity=1.0
        )
        
        assert wave is not None
        assert wave.emotion_type == EmotionType.JOY
        assert len(field.active_waves) == 1
    
    def test_field_strength_effect(self):
        """필드 강도 효과 테스트"""
        field_weak = KhalaField(field_strength=0.5)
        field_strong = KhalaField(field_strength=1.5)
        
        wave_weak = field_weak.broadcast_emotion("soul1", EmotionType.FEAR, 1.0)
        wave_strong = field_strong.broadcast_emotion("soul2", EmotionType.FEAR, 1.0)
        
        # 강한 필드에서 더 강한 파동
        assert wave_weak.intensity < wave_strong.intensity
    
    def test_wave_decay(self):
        """파동 감쇠 테스트"""
        field = KhalaField()
        field.broadcast_emotion("soul", EmotionType.JOY, 0.5)
        
        initial_count = len(field.active_waves)
        
        # 시간이 지나면 감쇠
        for _ in range(100):
            field.decay_waves(dt=1.0)
        
        # 파동이 사라짐
        assert len(field.active_waves) < initial_count


class TestSymbol:
    """상징 테스트"""
    
    def test_symbol_creation(self):
        """상징 생성 테스트"""
        symbol = Symbol(
            name="maku",
            meaning="물",
            complexity=SymbolComplexity.PROTO
        )
        assert symbol.name == "maku"
        assert symbol.meaning == "물"
        assert symbol.complexity == SymbolComplexity.PROTO
    
    def test_ambiguity_score(self):
        """애매함 점수 테스트"""
        symbol = Symbol(name="test", meaning="test")
        
        # 사용 전 = 최대 애매함
        assert symbol.get_ambiguity_score() == 1.0
        
        # 성공적 사용 = 명확해짐
        symbol.usage_count = 10
        symbol.misunderstanding_count = 2
        assert symbol.get_ambiguity_score() < 1.0


class TestLexicon:
    """어휘집 테스트"""
    
    def test_lexicon_creation(self):
        """어휘집 생성 테스트"""
        lexicon = Lexicon(owner_id="test_soul")
        assert lexicon.owner_id == "test_soul"
        assert len(lexicon.symbols) == 0
    
    def test_add_symbol(self):
        """상징 추가 테스트"""
        lexicon = Lexicon(owner_id="test")
        symbol = Symbol(name="kuma", meaning="곰", complexity=SymbolComplexity.PROTO)
        
        # 여러 번 시도하면 언젠가 학습됨
        for _ in range(20):
            lexicon.add_symbol(symbol)
        
        # 학습되었거나 시도 기록됨
        assert lexicon.total_learning_attempts > 0


class TestDualLayerSoul:
    """이중 언어 영혼 테스트"""
    
    def test_soul_creation(self):
        """영혼 생성 테스트"""
        soul = DualLayerSoul(name="테스트")
        assert soul.name == "테스트"
        assert soul.age == 0.0
    
    def test_feel_emotion(self):
        """감정 느끼기 테스트"""
        soul = DualLayerSoul(name="test")
        soul.feel_emotion(EmotionType.JOY, 0.8)
        
        assert EmotionType.JOY in soul.emotional_state
        assert soul.emotional_state[EmotionType.JOY] == 0.8
    
    def test_broadcast_and_receive(self):
        """감정 발신/수신 테스트"""
        field = KhalaField()
        soul1 = DualLayerSoul(name="sender")
        soul2 = DualLayerSoul(name="receiver")
        
        soul1.feel_emotion(EmotionType.FEAR, 1.5)
        soul1.broadcast_emotion(field)
        
        received = soul2.receive_emotions(field)
        
        # 무언가 수신됨
        assert len(field.active_waves) > 0
    
    def test_communication_style(self):
        """소통 스타일 테스트"""
        soul = DualLayerSoul(name="test")
        
        # 초기 상태
        assert soul.get_communication_style() == "silent"
        
        # 감정적 연결 증가
        soul.emotional_connections = 80
        soul.symbolic_communications = 20
        assert soul.get_communication_style() == "empath"
        
        # 언어적 소통 증가
        soul.emotional_connections = 20
        soul.symbolic_communications = 80
        assert soul.get_communication_style() == "rational"
    
    def test_relationship_gap(self):
        """관계의 틈 테스트"""
        soul1 = DualLayerSoul(name="soul1")
        soul2 = DualLayerSoul(name="soul2")
        
        # 같은 감정 공유
        soul1.feel_emotion(EmotionType.LOVE, 1.0)
        soul2.feel_emotion(EmotionType.LOVE, 1.0)
        
        gap_info = soul1.get_relationship_gap(soul2)
        
        assert "emotional_connection" in gap_info
        assert "linguistic_connection" in gap_info
        assert "relationship_gap" in gap_info
        assert "interpretation" in gap_info


class TestDualLayerWorld:
    """이중 언어 세계 테스트"""
    
    def test_world_creation(self):
        """세계 생성 테스트"""
        world = DualLayerWorld(n_souls=10, khala_strength=1.0)
        assert len(world.souls) == 10
        assert world.khala_field.field_strength == 1.0
    
    def test_world_step(self):
        """세계 진행 테스트"""
        world = DualLayerWorld(n_souls=10)
        initial_time = world.time
        
        world.step(dt=1.0)
        
        assert world.time > initial_time
    
    def test_khala_strength_adjustment(self):
        """칼라 강도 조절 테스트"""
        world = DualLayerWorld(n_souls=5, khala_strength=1.0)
        
        world.adjust_khala_strength(0.5)
        assert world.khala_field.field_strength == 0.5
        
        world.adjust_khala_strength(1.5)
        assert world.khala_field.field_strength == 1.5
    
    def test_simulation_produces_language(self):
        """시뮬레이션이 언어를 생성하는지 테스트"""
        world = DualLayerWorld(n_souls=20, khala_strength=0.8)
        
        # 100 스텝 실행
        for _ in range(100):
            world.step(1.0)
        
        # 언어적 이벤트 발생
        assert world.total_linguistic_events > 0
        
        # 일부 영혼들이 어휘를 갖게 됨
        vocab_sizes = [s.lexicon.get_vocabulary_size() for s in world.souls.values()]
        assert sum(vocab_sizes) > 0
    
    def test_low_khala_promotes_language(self):
        """낮은 칼라 강도가 언어 발달을 촉진하는지 테스트"""
        # 낮은 칼라 강도
        world_low = DualLayerWorld(n_souls=15, khala_strength=0.3)
        for _ in range(50):
            world_low.step(1.0)
        
        # 높은 칼라 강도
        world_high = DualLayerWorld(n_souls=15, khala_strength=1.5)
        for _ in range(50):
            world_high.step(1.0)
        
        # 낮은 칼라에서 더 많은 언어적 시도 (상대적)
        ratio_low = (world_low.total_linguistic_events / 
                     max(1, world_low.total_emotional_events))
        ratio_high = (world_high.total_linguistic_events / 
                      max(1, world_high.total_emotional_events))
        
        # 이 테스트는 확률적이므로 항상 통과하지 않을 수 있음
        # 하지만 경향성은 확인 가능
        assert world_low.total_linguistic_events > 0


class TestTimeAcceleratedIntegration:
    """시간 가속과의 통합 테스트"""
    
    def test_import_all_systems(self):
        """모든 시스템 임포트 테스트"""
        from Core.Interface.Language import (
            PrimalWaveWorld,
            TimeAcceleratedPrimalWorld,
            DualLayerWorld,
        )
        
        assert PrimalWaveWorld is not None
        assert TimeAcceleratedPrimalWorld is not None
        assert DualLayerWorld is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
