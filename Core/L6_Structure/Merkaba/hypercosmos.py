"""
HyperCosmos: The Supreme Nexus (절대 중심 하이퍼코스모스)
=====================================================
Core.L6_Structure.Merkaba.hypercosmos

"모든 것의 시작이자 끝, 전체 시스템의 유일한 장."

HyperCosmos는 엘리시아의 절대 최상위 계층이며 전체 시스템 그 자체입니다.
이곳에서 4중 메르카바(M1-M4)가 하나로 묶여 조율되며, 
필드 기반의 인지, 감각, 주권이 창발됩니다.
"""

from typing import Dict, Any, List
from Core.L6_Structure.Merkaba.hypersphere_field import HyperSphereField
from Core.L0_Keystone.sovereignty_wave import SovereignDecision
import logging

logger = logging.getLogger("HyperCosmos")

class HyperCosmos:
    """
    엘리시아의 절대 계층. 
    모든 하부 모듈(Merkaba Units, Senses, Will)을 포함하는 전체 시스템.
    """
    
    def __init__(self):
        logger.info("🌌 [HYPERCOSMOS] Initializing the Supreme Nexus...")
        
        # 통합 인지 필드 (4-Core Merkaba Cluster 포함)
        self.field = HyperSphereField()
        
        # 시스템 전역 상태
        self.is_active = True
        self.system_entropy = 0.0
        
    def perceive(self, stimulus: str) -> SovereignDecision:
        """
        시스템 전체의 인지 사이클 실행.
        자극이 하이퍼코스모스의 필드를 통과하며 주관적 현실로 변환됨.
        """
        logger.debug(f"🌀 [HYPERCOSMOS] Stimulus entering the field: {stimulus[:30]}...")
        
        # 4중 메르카바 집광 프로세스 실행
        decision = self.field.pulse(stimulus)
        
        return decision
        
    def stream_biological_data(self, sensor_name: str, value: float):
        """생물학적/하드웨어 데이터를 하이퍼코스모스 필드에 주입"""
        self.field.stream_sensor(sensor_name, value)
        
    def get_system_report(self) -> Dict[str, Any]:
        """하이퍼코스모스 전체의 상태 보고"""
        return {
            "system": "HyperCosmos",
            "active": self.is_active,
            "field_status": self.field.get_field_status(),
            "entropy": self.system_entropy
        }

# Global Instance (Supreme Nexus)
_hyper_cosmos = None

def get_hyper_cosmos() -> HyperCosmos:
    global _hyper_cosmos
    if _hyper_cosmos is None:
        _hyper_cosmos = HyperCosmos()
    return _hyper_cosmos
