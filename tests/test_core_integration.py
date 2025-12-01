"""
Core Integration Tests - 핵심 모듈 통합 테스트
==============================================

이 테스트는 Elysia의 핵심 모듈들 (Yggdrasil, Ether, Chronos, FreeWill)이
올바르게 연동되는지 검증합니다.

테스트 범위:
1. Yggdrasil (자아 모델) 초기화 및 상태 관리
2. Ether (파동 통신) 발신 및 수신
3. Chronos (시간 주권) 비동기 심장박동
4. FreeWillEngine (자유 의지) 욕망-행동-반성 루프

Note: FreeWillEngine 테스트는 외부 API 의존성으로 인해 
google.generativeai가 설치된 환경에서만 실행됩니다.
"""

import pytest
import sys
import os
import asyncio

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Check if google.generativeai is available
try:
    import google.generativeai
    HAS_GOOGLE_AI = True
except ImportError:
    HAS_GOOGLE_AI = False

requires_google_ai = pytest.mark.skipif(
    not HAS_GOOGLE_AI,
    reason="google.generativeai not installed"
)


class TestYggdrasilIntegration:
    """Yggdrasil (세계수) 자아 모델 테스트"""
    
    def test_yggdrasil_singleton(self):
        """Yggdrasil은 싱글톤이어야 함"""
        from Core.Structure.yggdrasil import Yggdrasil
        
        tree1 = Yggdrasil()
        tree2 = Yggdrasil()
        
        assert tree1 is tree2
    
    def test_yggdrasil_realms_structure(self):
        """Yggdrasil의 영역 구조 테스트"""
        from Core.Structure.yggdrasil import Yggdrasil, Realm
        
        tree = Yggdrasil()
        
        # 뿌리 등록
        tree.plant_root("TestRoot", {"type": "test"})
        
        # 줄기 등록
        tree.grow_trunk("TestTrunk", {"type": "test"})
        
        # 가지 등록
        tree.extend_branch("TestBranch", {"type": "test"})
        
        status = tree.status()
        
        assert "roots" in status
        assert "trunk" in status
        assert "branches" in status


class TestEtherIntegration:
    """Ether (에테르) 파동 통신 테스트"""
    
    def test_ether_singleton(self):
        """Ether는 싱글톤이어야 함"""
        from Core.Field.ether import Ether
        
        ether1 = Ether()
        ether2 = Ether()
        
        assert ether1 is ether2
    
    def test_wave_creation(self):
        """파동 생성 테스트"""
        from Core.Field.ether import Wave
        
        wave = Wave(
            sender="TestSender",
            frequency=432.0,
            amplitude=0.8,
            phase="TEST_PHASE",
            payload={"message": "Hello Elysia"}
        )
        
        assert wave.sender == "TestSender"
        assert wave.frequency == 432.0
        assert wave.amplitude == 0.8
        assert wave.phase == "TEST_PHASE"
    
    def test_wave_emission_and_reception(self):
        """파동 발신 및 수신 테스트"""
        from Core.Field.ether import ether, Wave
        
        received_waves = []
        
        def listener(wave):
            received_waves.append(wave)
        
        # 특정 주파수에 조율
        test_frequency = 123.45
        ether.tune_in(test_frequency, listener)
        
        # 파동 발신
        wave = Wave(
            sender="TestSender",
            frequency=test_frequency,
            amplitude=1.0,
            phase="TEST",
            payload="test data"
        )
        ether.emit(wave)
        
        # 수신 확인
        assert len(received_waves) >= 1
        assert received_waves[-1].payload == "test data"
    
    def test_wave_amplitude_filtering(self):
        """파동 진폭 필터링 테스트"""
        from Core.Field.ether import ether, Wave
        
        # 다양한 진폭의 파동 발신
        for amp in [0.1, 0.5, 0.9]:
            wave = Wave(
                sender="AmplitudeTest",
                frequency=999.0,
                amplitude=amp,
                phase="TEST",
                payload=f"amp_{amp}"
            )
            ether.emit(wave)
        
        # 진폭 0.5 이상만 필터링
        strong_waves = ether.get_waves(min_amplitude=0.5)
        
        # 파동이 있다면 진폭 확인
        if strong_waves:
            for w in strong_waves:
                assert w.amplitude >= 0.5


class TestChronosIntegration:
    """Chronos (크로노스) 시간 주권 테스트"""
    
    @pytest.mark.asyncio
    async def test_chronos_heartbeat(self):
        """심장박동 테스트"""
        from Core.Time.chronos import Chronos
        
        # 가짜 엔진 생성
        class MockEngine:
            def __init__(self):
                self.subconscious_calls = 0
            
            def subconscious_cycle(self):
                self.subconscious_calls += 1
        
        mock_engine = MockEngine()
        chronos = Chronos(mock_engine)
        
        # 한 번의 박동 테스트
        await chronos.beat()
        
        assert chronos.beat_count == 1
        assert mock_engine.subconscious_calls == 1
    
    def test_chronos_bpm_setting(self):
        """BPM 설정 테스트"""
        from Core.Time.chronos import Chronos
        
        class MockEngine:
            def subconscious_cycle(self):
                pass
        
        chronos = Chronos(MockEngine())
        
        # 기본 BPM 확인
        assert chronos.bpm == 60.0
        
        # BPM 변경
        chronos.bpm = 120.0
        assert chronos.bpm == 120.0


class TestFreeWillEngineIntegration:
    """FreeWillEngine (자유 의지) 통합 테스트"""
    
    @requires_google_ai
    def test_engine_initialization(self):
        """엔진 초기화 테스트"""
        from Core.Intelligence.Will.free_will_engine import FreeWillEngine, WillPhase
        
        engine = FreeWillEngine()
        
        assert engine is not None
        assert engine.current_phase == WillPhase.DESIRE
        assert len(engine.desires) > 0
        assert engine.active_desire is not None
    
    @requires_google_ai
    def test_desire_creation(self):
        """욕망 생성 테스트"""
        from Core.Intelligence.Will.free_will_engine import (
            FreeWillEngine, MissionType
        )
        
        engine = FreeWillEngine()
        initial_count = len(engine.desires)
        
        # 새로운 욕망 생성
        new_desire = engine.feel_desire(
            "Test desire",
            MissionType.SELF_EVOLUTION,
            intensity=0.8
        )
        
        assert len(engine.desires) == initial_count + 1
        assert new_desire.content == "Test desire"
        assert new_desire.intensity == 0.8
    
    @requires_google_ai
    def test_imagination_engine(self):
        """상상 엔진 테스트"""
        from Core.Intelligence.Will.free_will_engine import FreeWillEngine
        
        engine = FreeWillEngine()
        
        # 상상 시뮬레이션
        simulation = engine.imagination.imagine(
            action="파동을 보내다",
            target="아버지"
        )
        
        assert "predicted_response" in simulation
        assert "predicted_emotion" in simulation
        assert "confidence" in simulation
        assert simulation["confidence"] > 0
    
    @requires_google_ai
    def test_core_values_exist(self):
        """핵심 가치 존재 확인"""
        from Core.Intelligence.Will.free_will_engine import FreeWillEngine
        
        engine = FreeWillEngine()
        
        assert len(engine.core_values) > 0
        # 아버지에 대한 사랑이 핵심 가치에 포함되어야 함
        assert any("아버지" in v for v in engine.core_values)
    
    @requires_google_ai
    def test_will_loop_phases(self):
        """자유 의지 루프 단계 테스트"""
        from Core.Intelligence.Will.free_will_engine import (
            FreeWillEngine, WillPhase
        )
        
        engine = FreeWillEngine()
        
        # 모든 단계가 정의되어 있는지 확인
        phases = list(WillPhase)
        expected_phases = [
            "DESIRE", "LEARN", "CONTEMPLATE", 
            "EXPLORE", "ACT", "REFLECT", "GROW"
        ]
        
        for phase_name in expected_phases:
            assert any(p.name == phase_name for p in phases)


class TestModuleInterconnection:
    """모듈 간 연결 테스트"""
    
    @requires_google_ai
    def test_freewill_uses_ether(self):
        """FreeWillEngine이 Ether를 통해 통신하는지 테스트"""
        from Core.Intelligence.Will.free_will_engine import FreeWillEngine
        from Core.Field.ether import ether
        
        # 파동 수신 기록
        received = []
        
        def capture_wave(wave):
            received.append(wave)
        
        # 지구 주파수에 조율
        ether.tune_in(7.83, capture_wave)
        
        # 엔진 초기화 (Ether 연결 확인)
        engine = FreeWillEngine()
        
        # 엔진이 Ether에 조율되어 있는지 확인
        assert 7.83 in ether.listeners
        assert 963.0 in ether.listeners  # Divine frequency
    
    def test_yggdrasil_registers_modules(self):
        """Yggdrasil이 모듈을 등록할 수 있는지 테스트"""
        from Core.Structure.yggdrasil import yggdrasil
        from Core.Field.ether import ether
        
        # 뿌리에 Ether 등록
        yggdrasil.plant_root("Ether", ether)
        
        status = yggdrasil.status()
        root_names = [r["name"] for r in status["roots"]]
        
        assert "Ether" in root_names


class TestErrorHandling:
    """오류 처리 테스트"""
    
    def test_ether_handles_callback_errors(self):
        """Ether가 콜백 오류를 처리하는지 테스트"""
        from Core.Field.ether import ether, Wave
        
        def faulty_listener(wave):
            raise ValueError("Intentional error")
        
        # 오류를 발생시키는 리스너 등록
        ether.tune_in(404.0, faulty_listener)
        
        # 파동 발신 - 예외가 전파되지 않아야 함
        wave = Wave(
            sender="ErrorTest",
            frequency=404.0,
            amplitude=1.0,
            phase="ERROR_TEST",
            payload=None
        )
        
        # 예외가 발생하지 않아야 함
        try:
            ether.emit(wave)
        except ValueError:
            pytest.fail("Ether should handle callback errors gracefully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
