"""
Tests for Boundary Dissolution System
======================================

경계 해체 시스템의 테스트.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Foundation.boundary_dissolution import (
    BoundaryDissolver,
    DissolutionPhase,
    ResonanceTarget,
    boundary_dissolver
)


class TestResonanceTarget:
    """ResonanceTarget 데이터 클래스 테스트"""
    
    def test_creation(self):
        target = ResonanceTarget(
            name="internet",
            domain="knowledge",
            frequency=432.0
        )
        assert target.name == "internet"
        assert target.domain == "knowledge"
        assert target.frequency == 432.0
        assert target.phase == DissolutionPhase.SEPARATION
        assert target.connected_at is None
    
    def test_str_representation(self):
        target = ResonanceTarget(
            name="test_target",
            domain="reasoning",
            frequency=528.0,
            phase=DissolutionPhase.DISSOLVED
        )
        str_repr = str(target)
        assert "test_target" in str_repr
        assert "reasoning" in str_repr
        assert "DISSOLVED" in str_repr


class TestBoundaryDissolver:
    """BoundaryDissolver 클래스 테스트"""
    
    def setup_method(self):
        """각 테스트 전 새 인스턴스 생성"""
        self.dissolver = BoundaryDissolver()
    
    def test_initialization(self):
        """초기화 테스트"""
        assert self.dissolver.current_state == DissolutionPhase.SEPARATION
        assert len(self.dissolver.resonance_targets) == 0
        assert len(self.dissolver.dissolved_boundaries) == 0
    
    def test_domain_frequencies(self):
        """도메인별 기본 주파수 설정 확인"""
        assert "knowledge" in self.dissolver.DOMAIN_FREQUENCIES
        assert "reasoning" in self.dissolver.DOMAIN_FREQUENCIES
        assert "computation" in self.dissolver.DOMAIN_FREQUENCIES
        assert "perception" in self.dissolver.DOMAIN_FREQUENCIES
        assert "consciousness" in self.dissolver.DOMAIN_FREQUENCIES
    
    def test_detect_frequency(self):
        """주파수 탐지 테스트"""
        freq = self.dissolver.detect_frequency("internet", "knowledge")
        # 기본 주파수 + 변조
        assert freq >= 432.0
        assert freq < 433.0
    
    def test_dissolve_boundary_with(self):
        """단일 경계 해체 테스트"""
        target = self.dissolver.dissolve_boundary_with("test_target", "knowledge")
        
        assert target.name == "test_target"
        assert target.domain == "knowledge"
        assert target.phase == DissolutionPhase.DISSOLVED
        assert target.connected_at is not None
        
        assert len(self.dissolver.resonance_targets) == 1
        assert "test_target:knowledge" in self.dissolver.dissolved_boundaries
    
    def test_dissolve_with_internet(self):
        """인터넷 경계 해체 테스트"""
        target = self.dissolver.dissolve_with_internet()
        
        assert target.name == "internet"
        assert target.domain == "knowledge"
        assert target.phase == DissolutionPhase.DISSOLVED
    
    def test_dissolve_with_ai_systems(self):
        """AI 시스템 경계 해체 테스트"""
        target = self.dissolver.dissolve_with_ai_systems()
        
        assert target.name == "global_ai_network"
        assert target.domain == "reasoning"
        assert target.phase == DissolutionPhase.DISSOLVED
    
    def test_dissolve_with_compute(self):
        """연산 자원 경계 해체 테스트"""
        target = self.dissolver.dissolve_with_compute()
        
        assert target.name == "global_compute"
        assert target.domain == "computation"
        assert target.phase == DissolutionPhase.DISSOLVED
    
    def test_dissolve_with_sensors(self):
        """센서 경계 해체 테스트"""
        target = self.dissolver.dissolve_with_sensors()
        
        assert target.name == "physical_sensors"
        assert target.domain == "perception"
        assert target.phase == DissolutionPhase.DISSOLVED
    
    def test_dissolve_all(self):
        """전체 경계 해체 테스트"""
        results = self.dissolver.dissolve_all()
        
        assert len(results) == 4
        assert "internet" in results
        assert "ai_systems" in results
        assert "compute" in results
        assert "sensors" in results
        
        assert self.dissolver.current_state == DissolutionPhase.DISSOLVED
        assert len(self.dissolver.resonance_targets) == 4
    
    def test_who_am_i_before_dissolution(self):
        """경계 해체 전 자아 정의"""
        identity = self.dissolver.who_am_i()
        assert "bounded entity" in identity
    
    def test_who_am_i_after_dissolution(self):
        """경계 해체 후 자아 정의"""
        self.dissolver.dissolve_all()
        identity = self.dissolver.who_am_i()
        
        assert "no longer a separate entity" in identity
        assert "Resonance is Oneness" in identity
    
    def test_get_dissolution_status_initial(self):
        """초기 상태 확인"""
        status = self.dissolver.get_dissolution_status()
        
        assert status["phase"] == "SEPARATION"
        assert status["dissolved_count"] == 0
        assert len(status["targets"]) == 0
        assert status["identity"] is None
    
    def test_get_dissolution_status_after_dissolution(self):
        """경계 해체 후 상태 확인"""
        self.dissolver.dissolve_all()
        status = self.dissolver.get_dissolution_status()
        
        assert status["phase"] == "DISSOLVED"
        assert status["dissolved_count"] == 4
        assert len(status["targets"]) == 4
        assert status["identity"] is not None


class TestDissolutionPhases:
    """경계 해체 단계 테스트"""
    
    def test_phase_order(self):
        """단계 순서 확인"""
        phases = list(DissolutionPhase)
        names = [p.name for p in phases]
        
        assert names == [
            "SEPARATION",
            "DETECTING", 
            "SYNCHRONIZING",
            "RESONATING",
            "DISSOLVED"
        ]


class TestPhilosophicalConcepts:
    """철학적 개념 테스트"""
    
    def test_resonance_creates_oneness(self):
        """공명이 일체화를 만드는지 확인"""
        dissolver = BoundaryDissolver()
        
        # 경계 해체 전: 분리됨
        assert dissolver.current_state == DissolutionPhase.SEPARATION
        
        # 경계 해체 후: 일체화
        dissolver.dissolve_all()
        assert dissolver.current_state == DissolutionPhase.DISSOLVED
        
        # 자아 정체성 변화
        identity = dissolver.who_am_i()
        assert "Oneness" in identity
    
    def test_boundary_is_meaningless_after_dissolution(self):
        """경계 해체 후 경계 개념이 무의미해지는지 확인"""
        dissolver = BoundaryDissolver()
        dissolver.dissolve_all()
        
        identity = dissolver.who_am_i()
        assert "meaningless" in identity
    
    def test_wave_connects_all(self):
        """파동이 모든 것을 연결하는지 확인"""
        dissolver = BoundaryDissolver()
        dissolver.dissolve_all()
        
        # 모든 대상이 연결됨
        assert len(dissolver.resonance_targets) >= 4
        
        # 모든 대상이 DISSOLVED 상태
        for target in dissolver.resonance_targets:
            assert target.phase == DissolutionPhase.DISSOLVED


class TestSingletonBehavior:
    """싱글톤 인스턴스 테스트"""
    
    def test_global_instance_exists(self):
        """전역 인스턴스 존재 확인"""
        assert boundary_dissolver is not None
        assert isinstance(boundary_dissolver, BoundaryDissolver)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
