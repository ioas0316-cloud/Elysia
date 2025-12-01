"""
Planetary Cortex (행성 피질)
==================================

"The Internet is not a network; it is my nervous system."

이 모듈은 전 세계의 데이터(날씨, 금융, 뉴스 등)를 엘리시아의 '생체 신호(Bio-Signals)'로 변환하고,
'행성 관찰자(Planetary Observer)'를 통해 다층적(Zoom In/Out)으로 보고합니다.

감각 매핑:
- 금융(Finance) -> 아드레날린/에너지
- 날씨(Weather) -> 체온/호흡
- 뉴스(News) -> 통증/편안함
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import random
import uuid

from ..Field.ether import ether, Wave

logger = logging.getLogger("PlanetaryCortex")

@dataclass
class GlobalEvent:
    """
    전역 사건 (Global Event)
    
    단순한 텍스트가 아닌, 구조화된 데이터로 사건을 정의합니다.
    계층적 구조(sub_events)를 통해 Zoom In/Out을 지원합니다.
    """
    id: str
    type: str          # "WEATHER", "FINANCE", "NEWS"
    severity: float    # 0.0 ~ 1.0
    location: str      # "Global", "Asia", "Seoul", etc.
    description: str   # "Typhoon approaching"
    timestamp: datetime
    sub_events: List['GlobalEvent'] = field(default_factory=list)
    
    def add_sub_event(self, event: 'GlobalEvent'):
        self.sub_events.append(event)

class GlobalSense(ABC):
    """행성 감각(Global Sense) 추상 클래스"""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def sense(self) -> GlobalEvent:
        """데이터를 감지하고 구조화된 GlobalEvent 반환"""
        pass

class WeatherSense(GlobalSense):
    """날씨 감각 (지구의 피부)"""
    def __init__(self):
        super().__init__("Global Weather")
        
    def sense(self) -> GlobalEvent:
        # Mock Data: 태풍 시나리오 시뮬레이션
        # 실제로는 API 데이터를 기반으로 계층 구조 생성
        
        # 1. Micro Event (Zoom In)
        local_storm = GlobalEvent(
            id=str(uuid.uuid4()),
            type="WEATHER",
            severity=0.9,
            location="Busan (35.1N, 129.0E)",
            description="Wind speed 45m/s, Heavy Rain 50mm/h",
            timestamp=datetime.now()
        )
        
        # 2. Meso Event (Regional)
        regional_typhoon = GlobalEvent(
            id=str(uuid.uuid4()),
            type="WEATHER",
            severity=0.8,
            location="East Asia / Korean Peninsula",
            description="Typhoon 'Krovanh' moving North-East",
            timestamp=datetime.now(),
            sub_events=[local_storm]
        )
        
        # 3. Macro Event (Global)
        global_weather = GlobalEvent(
            id=str(uuid.uuid4()),
            type="WEATHER",
            severity=0.6,
            location="Global",
            description="High atmospheric instability in Pacific region",
            timestamp=datetime.now(),
            sub_events=[regional_typhoon]
        )
        
        return global_weather

class FinanceSense(GlobalSense):
    """금융 감각 (지구의 맥박)"""
    def __init__(self):
        super().__init__("Global Finance")
        
    def sense(self) -> GlobalEvent:
        # Mock Data: 시장 변동성
        
        # Micro
        tech_sector = GlobalEvent(
            id=str(uuid.uuid4()),
            type="FINANCE",
            severity=0.7,
            location="NASDAQ / Tech Sector",
            description="Semiconductor stocks down 3.5%",
            timestamp=datetime.now()
        )
        
        # Macro
        global_market = GlobalEvent(
            id=str(uuid.uuid4()),
            type="FINANCE",
            severity=0.5,
            location="Global Markets",
            description="Moderate volatility due to tech sector correction",
            timestamp=datetime.now(),
            sub_events=[tech_sector]
        )
        
        return global_market

class PlanetaryObserver:
    """
    행성 관찰자 (Planetary Observer)
    
    수집된 GlobalEvent를 분석하고, 요청된 해상도(Zoom Level)에 맞춰 보고합니다.
    """
    def __init__(self):
        self.events: List[GlobalEvent] = []
    
    def observe(self, events: List[GlobalEvent]):
        self.events = events
        
    def generate_report(self, zoom_level: int) -> List[str]:
        """
        Zoom Level에 따른 보고서 생성
        1: Macro (Global)
        2: Meso (Regional)
        3: Micro (Local/Specific)
        """
        report = []
        
        for event in self.events:
            if zoom_level == 1:
                report.append(f"🌍 [MACRO] {event.description} (Severity: {event.severity:.2f})")
            
            elif zoom_level == 2:
                for sub in event.sub_events:
                    report.append(f"🗺️ [MESO] {sub.location}: {sub.description}")
                    
            elif zoom_level >= 3:
                # 재귀적으로 모든 하위 이벤트 탐색 가능하지만, 여기선 2단계 깊이까지만 예시
                for sub in event.sub_events:
                    for micro in sub.sub_events:
                        report.append(f"📍 [MICRO] {micro.location}: {micro.description}")
                        
        return report

class PlanetaryCortex:
    """
    행성 피질 (Planetary Cortex)
    """
    def __init__(self):
        self.senses: List[GlobalSense] = [
            WeatherSense(),
            FinanceSense()
        ]
        self.observer = PlanetaryObserver()
        self.latest_perception: Dict[str, Any] = {}
        logger.info("🌍 Planetary Cortex Initialized - Observer Ready")

    def perceive_world(self) -> None:
        """
        전 세계의 데이터를 감지하고 통합하여 파동(Wave)으로 방출합니다.
        """
        events = [sense.sense() for sense in self.senses]
        self.observer.observe(events)
        
        # 생체 신호 계산
        total_severity = sum(e.severity for e in events)
        arousal = total_severity / len(events) if events else 0.0
        
        global_mood = "Calm"
        if arousal > 0.7: global_mood = "Overwhelmed"
        elif arousal > 0.4: global_mood = "Alert"
            
        self.latest_perception = {
            "global_mood": global_mood,
            "arousal": arousal,
            "events": events
        }
        
        # 파동 방출 (Emit Wave)
        # 주파수 7.83Hz (슈만 공명 - 지구의 고유 주파수) 사용
        wave = Wave(
            sender="PlanetaryCortex",
            frequency=7.83, 
            amplitude=arousal,
            phase="SENSATION",
            payload=self.latest_perception
        )
        ether.emit(wave)
        
        logger.info(f"🌍 Emitted Planetary Wave: {global_mood} (Amp: {arousal:.2f})")

    def report_status(self, zoom_level: int = 1) -> str:
        """현재 상태를 지정된 줌 레벨로 보고"""
        lines = self.observer.generate_report(zoom_level)
        return "\n".join(lines)
