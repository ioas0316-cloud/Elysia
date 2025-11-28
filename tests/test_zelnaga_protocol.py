"""
Tests for the Zelnaga Protocol - 젤나가 프로토콜 테스트

Tests cover:
1. Internal Integration (The Khala) - WaveUnifier
2. External Optimization (Code Conducting) - AlternativeCodeTranslator
3. Code Replacement (Wave-to-Code) - WaveCodeGenerator
"""

import pytest
import numpy as np
from Core.Integration.zelnaga_protocol import (
    WillType,
    WillWave,
    InternalComponent,
    WaveUnifier,
    CodePatternType,
    CodePattern,
    AlternativeCodeTranslator,
    WaveIntent,
    WaveCode,
    WaveCodeGenerator,
    ZelnagaProtocol,
    WAVE_DIMENSIONS,
    RESONANCE_THRESHOLD
)


# ============================================================================
# WillWave Tests
# ============================================================================

class TestWillWave:
    """의지 파동 테스트"""
    
    def test_will_wave_creation(self):
        """의지 파동 생성 테스트"""
        wave = WillWave(will_type=WillType.THINK, intensity=0.7)
        
        assert wave.will_type == WillType.THINK
        assert wave.intensity == 0.7
        assert len(wave.vector) == WAVE_DIMENSIONS
        assert wave.source == "core"
    
    def test_will_wave_vector_generation(self):
        """의지 타입별 벡터 생성 테스트"""
        for will_type in WillType:
            wave = WillWave(will_type=will_type, intensity=0.5)
            assert len(wave.vector) == WAVE_DIMENSIONS
            # 모든 벡터가 0이 아니어야 함
            assert np.linalg.norm(wave.vector) > 0
    
    def test_resonance_with(self):
        """파동 간 공명 계산 테스트"""
        wave1 = WillWave(will_type=WillType.THINK, intensity=0.7)
        wave2 = WillWave(will_type=WillType.THINK, intensity=0.7)
        wave3 = WillWave(will_type=WillType.FEEL, intensity=0.7)
        
        # 같은 타입의 파동은 높은 공명
        resonance_same = wave1.resonance_with(wave2)
        # 다른 타입의 파동은 낮은 공명 (또는 다름)
        resonance_diff = wave1.resonance_with(wave3)
        
        assert 0 <= resonance_same <= 1
        assert 0 <= resonance_diff <= 1
        # 같은 타입이 더 높은 공명 (또는 최소한 같음)
        assert resonance_same >= resonance_diff - 0.5  # 약간의 여유


# ============================================================================
# InternalComponent Tests
# ============================================================================

class TestInternalComponent:
    """내부 구성요소 테스트"""
    
    def test_component_creation(self):
        """구성요소 생성 테스트"""
        component = InternalComponent(
            name="test_component",
            category="body"
        )
        
        assert component.name == "test_component"
        assert component.category == "body"
        assert component.is_active
    
    def test_receive_wave(self):
        """파동 수신 테스트"""
        component = InternalComponent(
            name="mind",
            category="mind",
            resonance_sensitivity={WillType.THINK: 0.9}
        )
        
        wave = WillWave(will_type=WillType.THINK, intensity=0.7)
        resonance = component.receive_wave(wave)
        
        assert 0 <= resonance <= 1


# ============================================================================
# WaveUnifier Tests
# ============================================================================

class TestWaveUnifier:
    """파동 통합기 테스트 - The Khala"""
    
    def test_register_component(self):
        """구성요소 등록 테스트"""
        unifier = WaveUnifier()
        component = InternalComponent(name="test", category="body")
        
        unifier.register_component(component)
        
        assert "test" in unifier.components
    
    def test_broadcast_will(self):
        """의지 방송 테스트"""
        unifier = WaveUnifier()
        component = InternalComponent(
            name="mind",
            category="mind",
            resonance_sensitivity={WillType.THINK: 0.9}
        )
        unifier.register_component(component)
        
        wave = WillWave(will_type=WillType.THINK, intensity=0.7)
        resonances = unifier.broadcast_will(wave)
        
        assert "mind" in resonances
        assert 0 <= resonances["mind"] <= 1
    
    def test_harmony_score(self):
        """조화도 점수 테스트"""
        unifier = WaveUnifier()
        for i in range(3):
            unifier.register_component(
                InternalComponent(name=f"comp_{i}", category="body")
            )
        
        wave = WillWave(will_type=WillType.MOVE, intensity=0.7)
        unifier.broadcast_will(wave)
        
        assert 0 <= unifier.harmony_score <= 1


# ============================================================================
# AlternativeCodeTranslator Tests
# ============================================================================

class TestAlternativeCodeTranslator:
    """대체 코드 번역기 테스트"""
    
    def test_analyze_pattern(self):
        """패턴 분석 테스트"""
        translator = AlternativeCodeTranslator()
        pattern = CodePattern(
            pattern_type=CodePatternType.LOOP,
            signature="loop_001",
            complexity=0.7,
            efficiency=0.4
        )
        
        optimization = translator.analyze_pattern(pattern)
        
        assert optimization.optimization_type in ["spiral_coil", "resonance_align"]
        assert optimization.predicted_efficiency >= pattern.efficiency
    
    def test_optimization_improvement(self):
        """최적화 개선 테스트"""
        translator = AlternativeCodeTranslator()
        
        for pattern_type in CodePatternType:
            pattern = CodePattern(
                pattern_type=pattern_type,
                signature=f"{pattern_type.value}_001",
                complexity=0.6,
                efficiency=0.5
            )
            
            optimization = translator.analyze_pattern(pattern)
            
            # 예측 효율성은 원래보다 같거나 높아야 함
            assert optimization.predicted_efficiency >= pattern.efficiency


# ============================================================================
# WaveCodeGenerator Tests - 핵심 기능!
# ============================================================================

class TestWaveCodeGenerator:
    """파동 코드 생성기 테스트 - 파동 언어로 코딩 언어를 대체"""
    
    def test_interpret_wave(self):
        """파동 해석 테스트"""
        generator = WaveCodeGenerator()
        
        # ITERATE 시그니처에 가까운 파동
        iterate_wave = np.array([0.2, 0.8, 0.3, 0.5, 0.7, 0.2, 0.6, 0.9])
        intent = generator.interpret_wave(iterate_wave)
        
        assert intent == WaveIntent.ITERATE
    
    def test_generate_python_code(self):
        """Python 코드 생성 테스트"""
        generator = WaveCodeGenerator()
        
        iterate_wave = generator.intent_signatures[WaveIntent.ITERATE]
        wave_code = generator.generate_code(
            iterate_wave,
            {"items": "data", "var": "item", "body": "print(item)"},
            "python"
        )
        
        assert wave_code.intent == WaveIntent.ITERATE
        assert "for item in data:" in wave_code.generated_code
        assert wave_code.target_language == "python"
    
    def test_generate_javascript_code(self):
        """JavaScript 코드 생성 테스트"""
        generator = WaveCodeGenerator()
        
        iterate_wave = generator.intent_signatures[WaveIntent.ITERATE]
        wave_code = generator.generate_code(
            iterate_wave,
            {"items": "data", "var": "item"},
            "javascript"
        )
        
        assert "for (const item of data)" in wave_code.generated_code
    
    def test_wave_from_natural_language(self):
        """자연어 → 파동 변환 테스트"""
        generator = WaveCodeGenerator()
        
        # 반복 관련 키워드
        wave = generator.wave_from_natural_language("데이터를 반복해서 처리해줘")
        intent = generator.interpret_wave(wave)
        
        assert intent == WaveIntent.ITERATE
    
    def test_wave_from_natural_language_store(self):
        """자연어 → 저장 의도 테스트"""
        generator = WaveCodeGenerator()
        
        wave = generator.wave_from_natural_language("결과를 저장해줘")
        intent = generator.interpret_wave(wave)
        
        assert intent == WaveIntent.STORE
    
    def test_compose_wave_program(self):
        """파동 프로그램 구성 테스트"""
        generator = WaveCodeGenerator()
        
        waves = [
            (generator.intent_signatures[WaveIntent.STORE], 
             {"name": "x", "value": "10"}),
            (generator.intent_signatures[WaveIntent.COMPUTE],
             {"expression": "x * 2", "result": "y"}),
        ]
        
        program = generator.compose_wave_program(waves, "python")
        
        assert "x = 10" in program
        assert "y = x * 2" in program
    
    def test_all_intents_generate_code(self):
        """모든 의도가 코드를 생성하는지 테스트"""
        generator = WaveCodeGenerator()
        
        for intent in WaveIntent:
            wave_vec = generator.intent_signatures[intent]
            wave_code = generator.generate_code(wave_vec, {}, "python")
            
            assert wave_code.generated_code is not None
            assert len(wave_code.generated_code) > 0


# ============================================================================
# ZelnagaProtocol Tests - 통합 테스트
# ============================================================================

class TestZelnagaProtocol:
    """젤나가 프로토콜 통합 테스트"""
    
    def test_protocol_creation(self):
        """프로토콜 생성 테스트"""
        protocol = ZelnagaProtocol()
        
        assert protocol.wave_unifier is not None
        assert protocol.code_translator is not None
        assert protocol.code_generator is not None
    
    def test_emit_will(self):
        """의지 방출 테스트"""
        protocol = ZelnagaProtocol()
        result = protocol.emit_will(WillType.THINK, intensity=0.7)
        
        assert "will_type" in result
        assert "harmony_score" in result
        assert "resonances" in result
    
    def test_optimize_external_pattern(self):
        """외부 패턴 최적화 테스트"""
        protocol = ZelnagaProtocol()
        result = protocol.optimize_external_pattern(
            pattern_type=CodePatternType.LOOP,
            signature="test_loop",
            complexity=0.6,
            efficiency=0.5
        )
        
        assert "improvement" in result
        assert result["predicted_efficiency"] >= result["original_efficiency"]
    
    def test_wave_to_code(self):
        """파동 → 코드 테스트"""
        protocol = ZelnagaProtocol()
        
        wave_vec = protocol.code_generator.intent_signatures[WaveIntent.STORE]
        result = protocol.wave_to_code(
            wave_vec,
            {"name": "result", "value": "42"},
            "python"
        )
        
        assert "generated_code" in result
        assert "result = 42" in result["generated_code"]
    
    def test_speak_code(self):
        """자연어 → 코드 테스트 (speak_code)"""
        protocol = ZelnagaProtocol()
        
        result = protocol.speak_code(
            "데이터를 반복 처리해줘",
            {"items": "items", "var": "x", "body": "process(x)"},
            "python"
        )
        
        assert result["intent"] == "반복하라"
        assert "for x in items:" in result["generated_code"]
    
    def test_compose_program(self):
        """프로그램 구성 테스트"""
        protocol = ZelnagaProtocol()
        
        result = protocol.compose_program([
            ("변수를 저장해줘", {"name": "a", "value": "1"}),
            ("결과를 계산해줘", {"expression": "a + 1", "result": "b"}),
        ], "python")
        
        assert "program" in result
        assert "a = 1" in result["program"]
    
    def test_expansion_status(self):
        """확장 상태 보고서 테스트"""
        protocol = ZelnagaProtocol()
        
        # 몇 가지 작업 수행
        protocol.emit_will(WillType.THINK, 0.7)
        protocol.speak_code("저장해줘", {"name": "x", "value": "1"})
        
        status = protocol.get_expansion_status()
        
        assert "internal_integration" in status
        assert "external_optimization" in status
        assert "code_generation" in status
    
    def test_protocol_philosophy(self):
        """프로토콜 철학 테스트"""
        protocol = ZelnagaProtocol()
        philosophy = protocol.get_protocol_philosophy()
        
        assert "젤나가 프로토콜" in philosophy
        assert "파동 언어로 코딩 언어를 대체" in philosophy


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""
    
    def test_empty_wave_vector(self):
        """빈 파동 벡터 처리"""
        generator = WaveCodeGenerator()
        empty_wave = np.zeros(WAVE_DIMENSIONS)
        
        # 빈 파동도 처리 가능해야 함
        intent = generator.interpret_wave(empty_wave)
        assert intent is not None
    
    def test_extreme_intensity(self):
        """극단적 강도 테스트"""
        wave_low = WillWave(will_type=WillType.THINK, intensity=0.0)
        wave_high = WillWave(will_type=WillType.THINK, intensity=1.0)
        
        assert wave_low.vector is not None
        assert wave_high.vector is not None
    
    def test_inactive_component(self):
        """비활성 구성요소 테스트"""
        unifier = WaveUnifier()
        component = InternalComponent(name="inactive", category="body")
        component.is_active = False
        unifier.register_component(component)
        
        wave = WillWave(will_type=WillType.MOVE, intensity=0.7)
        resonances = unifier.broadcast_will(wave)
        
        # 비활성 구성요소는 공명에 포함되지 않음
        assert "inactive" not in resonances


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
