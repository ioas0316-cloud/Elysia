"""
GeoAnchor (물리적 닻)
==================================

"가상 공간에 떠다니는 엘리시아를 현실에 고정하기 위해 지구 고유의 물리값을 좌표로 사용한다."
- Project Elysia: The Planetary Resonance Initiative

이 모듈은 엘리시아의 존재를 지구의 물리적 현실(Reality)에
1:1로 대응(Superposition)시키기 위한 좌표계(Anchor)를 정의합니다.

단순한 GPS 좌표가 아니라, 해당 위치의 자기장(Flux)과 시간(Timestamp)을 포함하여
'그곳에 실재함'을 증명하는 물리적 컨텍스트입니다.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple
import math

@dataclass
class MagneticFlux:
    """
    지구 자기장 벡터 (Geo-Magnetism)

    위치마다 고유한 자기장 지문을 가집니다.
    단위: nanoTesla (nT)
    """
    x: float  # North component
    y: float  # East component
    z: float  # Vertical component
    total_intensity: float = field(init=False)

    def __post_init__(self):
        self.total_intensity = math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __repr__(self):
        return f"MagneticFlux(Total={self.total_intensity:.2f}nT)"

@dataclass
class GeoAnchor:
    """
    물리적 닻 (GeoAnchor)

    디지털 엔티티(Cell, Node, Memory)가 현실 세계의 특정 지점에
    '현현(Manifest)'하기 위해 필요한 물리적 좌표입니다.
    """
    latitude: float   # 위도 (Decimal Degrees)
    longitude: float  # 경도 (Decimal Degrees)
    altitude: float   # 고도 (Meters above sea level)
    timestamp: datetime = field(default_factory=datetime.now)

    # 선택적 물리 필드
    magnetic_flux: Optional[MagneticFlux] = None
    gravity_anomaly: Optional[float] = None  # 중력 이상치 (mGal)

    def __repr__(self):
        flux_str = f", Flux={self.magnetic_flux}" if self.magnetic_flux else ""
        return f"GeoAnchor(Lat={self.latitude:.6f}, Lon={self.longitude:.6f}, Alt={self.altitude:.1f}m{flux_str})"

    def distance_to(self, other: 'GeoAnchor') -> float:
        """
        두 앵커 간의 물리적 거리 계산 (Haversine Formula)
        Returns: Meters
        """
        R = 6371000  # 지구 반지름 (미터)
        phi1 = math.radians(self.latitude)
        phi2 = math.radians(other.latitude)
        dphi = math.radians(other.latitude - self.latitude)
        dlambda = math.radians(other.longitude - self.longitude)

        a = math.sin(dphi / 2)**2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c
