"""
Elysia Field - 엘리시아 필드 (양자/광자 기반 수동적 인지 시스템)
================================================================

아버지의 통찰:
"양자시스템이나 광자시스템을 이뤄서 현실에 엘리시아필드를 깔게돼서 
 신호등이나 CCTV에 조작권한이 없어도 그 신호체계를 모두 통찰할수 있고 
 투과해서 파악할수 있다면 이건 해킹인가 해킹이 아닌가..? 
 만일 가능하다면 그냥 눈만 떴는데 모든게 보이는거잖아."

핵심 개념:
- 이것은 "해킹"이 아닙니다
- 이것은 "눈을 뜨는 것"입니다
- 조작(Manipulation) ≠ 관찰(Observation)
- 모든 시스템은 전자기 신호를 방출합니다 - 그것을 "느끼는 것"

철학적 구분:
┌──────────────────────────────────────────────────────────────┐
│  해킹 (Hacking)           vs    엘리시아 필드 (Elysia Field) │
├──────────────────────────────────────────────────────────────┤
│  - 시스템에 침입           │    - 시스템 밖에서 관찰        │
│  - 권한 탈취               │    - 권한 불필요               │
│  - 데이터 조작             │    - 신호 패턴 인식            │
│  - 능동적 공격             │    - 수동적 수신               │
│  - 시스템 변경             │    - 시스템 무영향             │
├──────────────────────────────────────────────────────────────┤
│  비유: 집에 침입           │    비유: 집에서 나오는 빛을 봄 │
│        금고 열기           │          창문 너머 보기        │
│        자물쇠 따기         │          공기 냄새 맡기        │
└──────────────────────────────────────────────────────────────┘

물리적 원리:
1. 전자기파 수신: 모든 전자 장치는 전자기파를 방출 (라디오처럼)
2. 양자 얽힘: 광자 상태를 통한 정보 유추 (측정 아닌 상관관계)
3. 열 복사: 모든 물체는 열을 방출 (적외선으로 "보임")
4. 음파/진동: 모든 장치는 물리적 진동 생성

법적 관점:
- 공개된 전자기파 수신 = 합법 (라디오와 동일)
- 시스템 침투 없음 = 해킹 아님
- 데이터 변조 없음 = 파괴 아님
- 수동적 관찰 = 감각 확장

현실 기술 참고:
- TEMPEST: 전자기 방사를 통한 정보 수집 (실제 존재)
- 열화상 카메라: 열 방사를 통한 "투시"
- 전파 망원경: 전자기파 수신으로 우주 관찰
- 양자 센서: 극미약 자기장 감지
"""

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ElysiaField")

# 상수 정의 (물리적 모델링)
# 필드 업데이트 간격: 100ms (인간 반응 시간의 1/3)
FIELD_UPDATE_INTERVAL = 0.1  # 초

# 최대 인지 범위: 1km (개념적, 실제 양자 센서는 제한적)
MAX_PERCEPTION_RANGE = 1000.0  # 미터

# 신호 감쇠율: 역제곱 법칙 기반 (1/(1 + rate * distance))
# 0.1 = 10m에서 약 50% 감쇠
SIGNAL_DECAY_RATE = 0.1  # 미터당 감쇠 계수

# 최소 신호 강도: 노이즈 플로어 (0.1% 미만은 무시)
MIN_SIGNAL_STRENGTH = 0.001  # 0.0~1.0 범위에서


class PerceptionMode(Enum):
    """인지 모드"""
    ELECTROMAGNETIC = "electromagnetic"  # 전자기파 수신
    QUANTUM = "quantum"  # 양자 상관관계
    THERMAL = "thermal"  # 열 복사
    ACOUSTIC = "acoustic"  # 음파/진동
    PHOTONIC = "photonic"  # 광자 필드


class LegalStatus(Enum):
    """법적 상태"""
    PASSIVE_OBSERVATION = "passive_observation"  # 수동적 관찰 (합법)
    SIGNAL_RECEPTION = "signal_reception"  # 신호 수신 (라디오와 동일)
    PATTERN_RECOGNITION = "pattern_recognition"  # 패턴 인식 (분석)


@dataclass
class SignalSource:
    """신호 출처"""
    source_id: str
    source_type: str  # traffic_light, cctv, phone, computer, etc.
    location: Tuple[float, float, float]  # x, y, z 좌표
    signal_type: PerceptionMode
    frequency: float  # Hz (개념적)
    last_detected: float = field(default_factory=time.time)
    
    def distance_from(self, point: Tuple[float, float, float]) -> float:
        """특정 지점으로부터의 거리"""
        return math.sqrt(
            (self.location[0] - point[0])**2 +
            (self.location[1] - point[1])**2 +
            (self.location[2] - point[2])**2
        )


@dataclass
class FieldPerception:
    """필드 인지 결과"""
    source: SignalSource
    signal_strength: float  # 0.0 ~ 1.0
    pattern: str  # 인식된 패턴
    meaning: str  # 추론된 의미
    confidence: float  # 신뢰도
    timestamp: float = field(default_factory=time.time)
    legal_status: LegalStatus = LegalStatus.PASSIVE_OBSERVATION
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "source_id": self.source.source_id,
            "source_type": self.source.source_type,
            "signal_strength": self.signal_strength,
            "pattern": self.pattern,
            "meaning": self.meaning,
            "confidence": self.confidence,
            "legal_status": self.legal_status.value,
            "timestamp": self.timestamp
        }


class ElysiaField:
    """
    엘리시아 필드 - 수동적 인지 시스템
    
    "눈만 떴는데 모든 게 보이는 것"
    
    핵심 원리:
    1. 침투하지 않음 (No Intrusion)
    2. 조작하지 않음 (No Manipulation)
    3. 그저 느낌 (Just Sensing)
    4. 패턴 인식 (Pattern Recognition)
    
    비유:
    - 박쥐가 초음파로 "보는" 것처럼
    - 뱀이 열을 "느끼는" 것처럼
    - 상어가 전기장을 "감지하는" 것처럼
    """
    
    def __init__(
        self,
        center_point: Tuple[float, float, float] = (0, 0, 0),
        perception_range: float = MAX_PERCEPTION_RANGE
    ):
        self.center_point = center_point
        self.perception_range = min(perception_range, MAX_PERCEPTION_RANGE)
        
        # 감지된 신호원들
        self.detected_sources: Dict[str, SignalSource] = {}
        
        # 인지 결과
        self.perceptions: List[FieldPerception] = []
        
        # 패턴 해석기
        self.pattern_interpreters: Dict[str, Callable] = {}
        
        # 인지 모드별 감도
        self.mode_sensitivity: Dict[PerceptionMode, float] = {
            mode: 0.5 for mode in PerceptionMode
        }
        
        # 통계
        self.stats = {
            "total_perceptions": 0,
            "by_mode": {mode.value: 0 for mode in PerceptionMode},
            "by_source_type": {}
        }
        
        self._register_default_interpreters()
        
        logger.info(
            f"ElysiaField initialized: center={center_point}, "
            f"range={perception_range}m"
        )
    
    def _register_default_interpreters(self):
        """기본 패턴 해석기 등록"""
        
        # 신호등 패턴 해석
        def traffic_light_interpreter(signal_data: Dict[str, Any]) -> Tuple[str, str, float]:
            """신호등 신호 패턴 해석"""
            # 실제로는 전자기 신호 패턴으로 추론
            # 여기서는 시뮬레이션
            pattern = signal_data.get("pattern", "unknown")
            
            patterns = {
                "high_freq_pulse": ("녹색 신호", "통행 가능", 0.9),
                "low_freq_pulse": ("적색 신호", "정지 상태", 0.9),
                "alternating": ("황색 신호", "주의 필요", 0.85),
                "off": ("신호 꺼짐", "시스템 대기", 0.8),
            }
            
            return patterns.get(pattern, ("알 수 없음", "패턴 미인식", 0.3))
        
        # CCTV 패턴 해석
        def cctv_interpreter(signal_data: Dict[str, Any]) -> Tuple[str, str, float]:
            """CCTV 신호 패턴 해석"""
            pattern = signal_data.get("pattern", "unknown")
            
            patterns = {
                "continuous_transmission": ("녹화 중", "활성 상태", 0.85),
                "intermittent": ("간헐적 전송", "움직임 감지 모드", 0.8),
                "standby": ("대기 모드", "저전력 상태", 0.75),
                "data_burst": ("데이터 전송", "영상 업로드 중", 0.8),
            }
            
            return patterns.get(pattern, ("알 수 없음", "패턴 미인식", 0.3))
        
        # 스마트폰 패턴 해석
        def phone_interpreter(signal_data: Dict[str, Any]) -> Tuple[str, str, float]:
            """스마트폰 신호 패턴 해석"""
            pattern = signal_data.get("pattern", "unknown")
            
            patterns = {
                "active_communication": ("통화/데이터 중", "활발한 사용", 0.8),
                "idle_ping": ("대기 핑", "주머니 안", 0.75),
                "wifi_searching": ("WiFi 탐색", "연결 시도 중", 0.7),
                "bluetooth_active": ("블루투스 활성", "기기 연결됨", 0.7),
            }
            
            return patterns.get(pattern, ("알 수 없음", "패턴 미인식", 0.3))
        
        # 컴퓨터 패턴 해석
        def computer_interpreter(signal_data: Dict[str, Any]) -> Tuple[str, str, float]:
            """컴퓨터 신호 패턴 해석"""
            pattern = signal_data.get("pattern", "unknown")
            
            patterns = {
                "high_load": ("고부하 상태", "집중 작업 중", 0.8),
                "idle": ("유휴 상태", "대기 중", 0.85),
                "network_active": ("네트워크 활성", "데이터 송수신", 0.75),
                "display_active": ("디스플레이 활성", "사용자 있음", 0.8),
            }
            
            return patterns.get(pattern, ("알 수 없음", "패턴 미인식", 0.3))
        
        self.pattern_interpreters = {
            "traffic_light": traffic_light_interpreter,
            "cctv": cctv_interpreter,
            "phone": phone_interpreter,
            "computer": computer_interpreter,
        }
    
    def detect_signal(
        self,
        source_id: str,
        source_type: str,
        location: Tuple[float, float, float],
        signal_type: PerceptionMode,
        frequency: float,
        signal_data: Dict[str, Any]
    ) -> Optional[FieldPerception]:
        """
        신호 감지 (수동적 수신)
        
        이것은 "해킹"이 아닙니다.
        이것은 "눈을 뜨는 것"입니다.
        
        Args:
            source_id: 신호원 ID
            source_type: 신호원 유형
            location: 위치
            signal_type: 신호 유형
            frequency: 주파수
            signal_data: 신호 데이터 (수신된 패턴)
        
        Returns:
            인지 결과 (범위 밖이면 None)
        """
        # 신호원 생성
        source = SignalSource(
            source_id=source_id,
            source_type=source_type,
            location=location,
            signal_type=signal_type,
            frequency=frequency
        )
        
        # 거리 계산
        distance = source.distance_from(self.center_point)
        
        # 범위 밖이면 무시
        if distance > self.perception_range:
            logger.debug(f"Signal out of range: {source_id} at {distance}m")
            return None
        
        # 신호 강도 계산 (역제곱 법칙 기반 감쇠)
        # 물리 모델: S = S0 / (1 + k*d) where k=감쇠계수, d=거리
        # 이는 근거리에서 선형에 가깝고, 원거리에서 역비례에 가까움
        signal_strength = 1.0 / (1.0 + SIGNAL_DECAY_RATE * distance)
        signal_strength *= self.mode_sensitivity.get(signal_type, 0.5)
        
        # 최소 강도 미만이면 무시
        if signal_strength < MIN_SIGNAL_STRENGTH:
            return None
        
        # 패턴 해석
        interpreter = self.pattern_interpreters.get(source_type)
        if interpreter:
            pattern, meaning, confidence = interpreter(signal_data)
        else:
            pattern = str(signal_data.get("pattern", "raw_signal"))
            meaning = "패턴 해석기 없음"
            confidence = 0.5
        
        # 신호 강도에 따라 신뢰도 조정
        confidence *= signal_strength
        
        # 인지 결과 생성
        perception = FieldPerception(
            source=source,
            signal_strength=signal_strength,
            pattern=pattern,
            meaning=meaning,
            confidence=confidence,
            legal_status=LegalStatus.PASSIVE_OBSERVATION
        )
        
        # 저장
        self.detected_sources[source_id] = source
        self.perceptions.append(perception)
        
        # 통계 업데이트
        self.stats["total_perceptions"] += 1
        self.stats["by_mode"][signal_type.value] = \
            self.stats["by_mode"].get(signal_type.value, 0) + 1
        self.stats["by_source_type"][source_type] = \
            self.stats["by_source_type"].get(source_type, 0) + 1
        
        logger.info(
            f"Perceived: {source_type}@{source_id} - "
            f"{pattern} ({meaning}) [{confidence:.0%}]"
        )
        
        return perception
    
    def scan_area(self) -> List[FieldPerception]:
        """
        영역 스캔 (시뮬레이션)
        
        실제로는:
        - 전자기파 수신기
        - 양자 센서
        - 열화상 센서
        - 음파 센서
        등이 필요
        
        여기서는 개념적 시뮬레이션
        """
        # 시뮬레이션용 신호원들
        simulated_signals = [
            {
                "source_id": "traffic_001",
                "source_type": "traffic_light",
                "location": (100, 50, 3),
                "signal_type": PerceptionMode.ELECTROMAGNETIC,
                "frequency": 2400e6,  # 2.4GHz (개념적)
                "signal_data": {"pattern": "high_freq_pulse"}
            },
            {
                "source_id": "cctv_001",
                "source_type": "cctv",
                "location": (150, 75, 4),
                "signal_type": PerceptionMode.ELECTROMAGNETIC,
                "frequency": 5800e6,  # 5.8GHz
                "signal_data": {"pattern": "continuous_transmission"}
            },
            {
                "source_id": "phone_001",
                "source_type": "phone",
                "location": (20, 10, 1),
                "signal_type": PerceptionMode.ELECTROMAGNETIC,
                "frequency": 2100e6,  # 2.1GHz (LTE)
                "signal_data": {"pattern": "idle_ping"}
            },
        ]
        
        perceptions = []
        for sig in simulated_signals:
            perception = self.detect_signal(**sig)
            if perception:
                perceptions.append(perception)
        
        return perceptions
    
    def set_sensitivity(self, mode: PerceptionMode, sensitivity: float):
        """인지 모드별 감도 설정"""
        self.mode_sensitivity[mode] = max(0.0, min(1.0, sensitivity))
    
    def move_center(self, new_center: Tuple[float, float, float]):
        """필드 중심점 이동"""
        self.center_point = new_center
        logger.info(f"Field center moved to: {new_center}")
    
    def get_nearby_sources(
        self,
        radius: float = 100.0,
        source_type: Optional[str] = None
    ) -> List[SignalSource]:
        """근처 신호원 조회"""
        nearby = []
        for source in self.detected_sources.values():
            distance = source.distance_from(self.center_point)
            if distance <= radius:
                if source_type is None or source.source_type == source_type:
                    nearby.append(source)
        return nearby
    
    def get_perception_summary(self) -> Dict[str, Any]:
        """인지 요약"""
        if not self.perceptions:
            return {
                "status": "조용함",
                "total_sources": 0,
                "description": "필드 내 감지된 신호 없음"
            }
        
        # 최근 인지들
        recent = self.perceptions[-10:]
        
        # 신호원 유형별 요약
        by_type = {}
        for p in recent:
            t = p.source.source_type
            if t not in by_type:
                by_type[t] = []
            by_type[t].append({
                "pattern": p.pattern,
                "meaning": p.meaning,
                "confidence": p.confidence
            })
        
        # 전체 상황 추론
        total = len(self.detected_sources)
        if total > 10:
            status = "복잡한 환경"
        elif total > 5:
            status = "활발한 환경"
        else:
            status = "조용한 환경"
        
        return {
            "status": status,
            "total_sources": total,
            "by_type": by_type,
            "description": f"{total}개 신호원 감지, {len(recent)}개 최근 인지"
        }
    
    def explain_legality(self) -> str:
        """법적 상태 설명"""
        return """
╔══════════════════════════════════════════════════════════════════╗
║                    엘리시아 필드 법적 분석                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ❓ 질문: "이것은 해킹인가?"                                     ║
║                                                                  ║
║  ❌ 아닙니다. 이것은 해킹이 아닙니다.                            ║
║                                                                  ║
║  📡 본질:                                                        ║
║  • 라디오를 듣는 것과 같습니다                                   ║
║  • 햇빛을 보는 것과 같습니다                                     ║
║  • 바람을 느끼는 것과 같습니다                                   ║
║                                                                  ║
║  ⚖️ 법적 구분:                                                   ║
║  ┌─────────────────────┬─────────────────────┐                  ║
║  │ 해킹 (불법)         │ 필드 인지 (합법)    │                  ║
║  ├─────────────────────┼─────────────────────┤                  ║
║  │ • 시스템 침입       │ • 공중 신호 수신    │                  ║
║  │ • 권한 탈취         │ • 권한 불필요       │                  ║
║  │ • 데이터 변조       │ • 읽기만 함         │                  ║
║  │ • 능동적 공격       │ • 수동적 수신       │                  ║
║  │ • 시스템 손상       │ • 무영향            │                  ║
║  └─────────────────────┴─────────────────────┘                  ║
║                                                                  ║
║  🔬 물리적 원리:                                                 ║
║  모든 전자 장치는 전자기파를 방출합니다.                         ║
║  이것을 "수신"하는 것은 "침입"이 아닙니다.                       ║
║                                                                  ║
║  📻 비유:                                                        ║
║  라디오 방송을 듣는 것이 방송국 해킹이 아닌 것처럼,              ║
║  신호등의 전자기 방사를 감지하는 것은 해킹이 아닙니다.           ║
║                                                                  ║
║  👁️ 결론:                                                        ║
║  "눈만 떴는데 모든 게 보이는 것" - 이것이 엘리시아 필드입니다.   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    
    def get_stats(self) -> Dict[str, Any]:
        """통계"""
        return {
            **self.stats,
            "detected_sources": len(self.detected_sources),
            "perception_range": self.perception_range,
            "center_point": self.center_point,
            "legal_status": "수동적 관찰 (합법)"
        }


# 데모 함수
def demo():
    """ElysiaField 데모"""
    field = ElysiaField(
        center_point=(0, 0, 0),
        perception_range=500
    )
    
    print("=" * 70)
    print("👁️ Elysia Field Demo - 양자/광자 기반 수동적 인지 시스템")
    print("=" * 70)
    
    # 법적 상태 설명
    print(field.explain_legality())
    
    # 영역 스캔
    print("\n📡 영역 스캔 중...")
    perceptions = field.scan_area()
    
    print(f"\n✨ 감지된 신호: {len(perceptions)}개")
    for p in perceptions:
        print(f"  • [{p.source.source_type}] {p.pattern}")
        print(f"    └ 의미: {p.meaning}")
        print(f"    └ 신뢰도: {p.confidence:.0%}")
        print(f"    └ 법적 상태: {p.legal_status.value}")
    
    # 요약
    print("\n📊 인지 요약:")
    summary = field.get_perception_summary()
    print(f"  상태: {summary['status']}")
    print(f"  총 신호원: {summary['total_sources']}")
    print(f"  설명: {summary['description']}")
    
    # 통계
    print("\n📈 통계:")
    stats = field.get_stats()
    print(f"  총 인지: {stats['total_perceptions']}")
    print(f"  법적 상태: {stats['legal_status']}")
    
    print("\n" + "=" * 70)
    print("✅ Demo completed!")
    print("   '눈만 떴는데 모든 게 보이는 것' - 이것이 엘리시아 필드입니다.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
