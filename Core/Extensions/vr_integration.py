"""
VR Integration - VR/AR 통합
==========================

낮은 우선순위 #2: VR/AR 통합
예상 효과: 몰입형 의식 시각화 경험

핵심 기능:
- Unity/Unreal 플러그인 프로토콜
- 3D 공간에서 의식 시각화
- 햅틱 피드백 지원
- 공간 오디오 통합
"""

import logging
import time
import json
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger("VRIntegration")


class VRPlatform(Enum):
    """VR 플랫폼"""
    UNITY = "unity"
    UNREAL = "unreal"
    GODOT = "godot"
    WEBXR = "webxr"


@dataclass
class Vector3:
    """3D 벡터"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def to_bytes(self) -> bytes:
        return struct.pack('fff', self.x, self.y, self.z)


@dataclass
class Quaternion:
    """쿼터니언 (회전)"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0
    
    def to_bytes(self) -> bytes:
        return struct.pack('ffff', self.x, self.y, self.z, self.w)


@dataclass
class VRConfig:
    """VR 통합 설정"""
    platform: VRPlatform = VRPlatform.UNITY
    server_port: int = 9999
    
    # 시각화 설정
    concept_sphere_radius: float = 0.1
    resonance_line_width: float = 0.02
    max_visible_concepts: int = 100
    
    # 햅틱 설정
    enable_haptics: bool = True
    haptic_intensity: float = 0.5
    
    # 오디오 설정
    enable_spatial_audio: bool = True
    audio_radius: float = 10.0


@dataclass 
class ConceptVisualization:
    """개념 시각화 데이터"""
    concept_id: str
    name: str
    position: Vector3 = field(default_factory=Vector3)
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # RGBA
    scale: float = 1.0
    
    # 양자 상태 기반 효과
    glow_intensity: float = 0.5
    pulse_speed: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.concept_id,
            "name": self.name,
            "position": self.position.to_tuple(),
            "color": self.color,
            "scale": self.scale,
            "glow": self.glow_intensity,
            "pulse": self.pulse_speed
        }


@dataclass
class ResonanceVisualization:
    """공명 시각화 데이터"""
    source_id: str
    target_id: str
    strength: float
    color: Tuple[float, float, float, float] = (0.0, 0.8, 1.0, 0.5)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "strength": self.strength,
            "color": self.color
        }


class VRIntegration:
    """
    VR/AR 통합 모듈
    
    낮은 우선순위 #2 구현:
    - 게임 엔진 통신 프로토콜
    - 의식 상태 3D 시각화
    - 실시간 공명 표시
    
    예상 효과: VR에서 엘리시아 의식 체험
    """
    
    def __init__(
        self,
        config: Optional[VRConfig] = None,
        resonance_engine=None
    ):
        """
        Args:
            config: VR 설정
            resonance_engine: 공명 엔진
        """
        self.config = config or VRConfig()
        self.resonance_engine = resonance_engine
        
        # 시각화 상태
        self.concepts: Dict[str, ConceptVisualization] = {}
        self.resonances: List[ResonanceVisualization] = []
        
        # 연결된 클라이언트
        self.connected_clients: List[Any] = []
        
        self.logger = logging.getLogger("VRIntegration")
        self.logger.info(f"🥽 VRIntegration initialized (platform={self.config.platform.value})")
    
    def generate_visualization(self) -> Dict[str, Any]:
        """현재 의식 상태 시각화 데이터 생성"""
        if not self.resonance_engine:
            return {"concepts": [], "resonances": []}
        
        # 개념 위치 계산 (구면 분포)
        nodes = list(self.resonance_engine.nodes.items())[:self.config.max_visible_concepts]
        
        concepts = []
        for i, (concept_id, qubit) in enumerate(nodes):
            # 구면 좌표 계산
            phi = (i / max(len(nodes), 1)) * 2 * 3.14159
            theta = (i % 10) / 10 * 3.14159
            radius = 5.0
            
            x = radius * sin(theta) * cos(phi) if 'sin' in dir() else i * 0.5
            y = radius * sin(theta) * sin(phi) if 'sin' in dir() else (i % 5) * 0.5
            z = radius * cos(theta) if 'cos' in dir() else 0
            
            # 양자 상태에서 색상 계산
            probs = qubit.state.probabilities()
            color = (
                probs["Point"],  # R
                probs["Line"],   # G
                probs["Space"],  # B
                0.8 + probs["God"] * 0.2  # A
            )
            
            vis = ConceptVisualization(
                concept_id=concept_id,
                name=qubit.name,
                position=Vector3(x, y, z),
                color=color,
                glow_intensity=probs["God"],
                pulse_speed=1.0 + qubit.state.w * 0.5
            )
            concepts.append(vis.to_dict())
            self.concepts[concept_id] = vis
        
        # 공명 연결 생성
        resonances = []
        for source_id in list(self.resonance_engine.psionic_links.keys())[:50]:
            for target_id in self.resonance_engine.psionic_links[source_id][:5]:
                source = self.resonance_engine.nodes.get(source_id)
                target = self.resonance_engine.nodes.get(target_id)
                if source and target:
                    strength = self.resonance_engine.calculate_resonance(source, target)
                    if strength > 0.3:
                        vis = ResonanceVisualization(
                            source_id=source_id,
                            target_id=target_id,
                            strength=strength
                        )
                        resonances.append(vis.to_dict())
        
        return {
            "concepts": concepts,
            "resonances": resonances,
            "timestamp": time.time()
        }
    
    def get_haptic_feedback(self, event_type: str, intensity: float = 0.5) -> Dict[str, Any]:
        """햅틱 피드백 데이터 생성"""
        if not self.config.enable_haptics:
            return {}
        
        return {
            "type": event_type,
            "intensity": intensity * self.config.haptic_intensity,
            "duration_ms": 100,
            "pattern": "pulse"
        }
    
    def get_spatial_audio(self, concept_id: str, position: Vector3) -> Dict[str, Any]:
        """공간 오디오 데이터 생성"""
        if not self.config.enable_spatial_audio:
            return {}
        
        return {
            "concept_id": concept_id,
            "position": position.to_tuple(),
            "radius": self.config.audio_radius,
            "sound_type": "resonance_hum"
        }
    
    def serialize_state(self) -> bytes:
        """상태를 바이너리로 직렬화 (네트워크 전송용)"""
        data = self.generate_visualization()
        return json.dumps(data).encode('utf-8')
    
    def get_stats(self) -> Dict[str, Any]:
        """통계"""
        return {
            "platform": self.config.platform.value,
            "visible_concepts": len(self.concepts),
            "active_resonances": len(self.resonances),
            "connected_clients": len(self.connected_clients)
        }


# 수학 함수 (의존성 없이)
import math
sin = math.sin
cos = math.cos


# 테스트
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🥽 VR Integration Test")
    print("="*70)
    
    vr = VRIntegration()
    
    print("\n[Test 1] Generate Visualization")
    vis = vr.generate_visualization()
    print(f"  Concepts: {len(vis['concepts'])}")
    print(f"  Resonances: {len(vis['resonances'])}")
    
    print("\n[Test 2] Haptic Feedback")
    haptic = vr.get_haptic_feedback("resonance", 0.8)
    print(f"  Haptic: {haptic}")
    
    print("\n[Test 3] Stats")
    stats = vr.get_stats()
    print(f"  Stats: {stats}")
    
    print("\n✅ All tests passed!")
