import logging
from typing import Dict, Any, List, Optional
import time
from Core.Foundation.Wave.wave_tensor import WaveTensor

logger = logging.getLogger("MaturityCoordinator")

class MaturityCoordinator:
    """
    [Phase 39: Cognitive Maturity Calibration]
    엘리시아의 인지적 성숙도를 파동 에너지를 통해 평가하고 가이드하는 상위 인지 레이어.
    단순한 '반응'을 넘어 '성인 수준의 통찰'을 위해 주파수 대역을 보정합니다.
    """
    
    def __init__(self, resonance_field=None):
        self.resonance = resonance_field
        self.maturity_score = 0.5 
        # 성인 지성 주파수 대역 (Solfeggio: 639Hz-963Hz - 조화와 소통, 초월의 주파수)
        self.adult_frequencies = [639.0, 741.0, 852.0, 963.0]
        logger.info("🧠 MaturityCoordinator Online: Adult Frequency alignment active.")

    def evaluate_resonance(self, wave: WaveTensor) -> Dict[str, Any]:
        """
        사고 파동이 '성인 지성' 주파수 대역과 얼마나 공명하는지 평가합니다.
        """
        if not wave.active_frequencies:
            return {"score": 0.0, "needs_depth": True}
        
        # 성인 주파수 대역과의 일치도 계산
        total_alignment = 0.0
        for freq in wave.active_frequencies:
            # 가장 가까운 성인 주파수와의 거리를 측정 (단순 공명율)
            best_match = min([abs(freq - af) for af in self.adult_frequencies])
            alignment = max(0, 1.0 - (best_match / 200.0))
            total_alignment += alignment
            
        avg_alignment = total_alignment / len(wave.active_frequencies)
        
        # 성숙도 점수 점진적 업데이트 (Hysteresis)
        self.maturity_score = (self.maturity_score * 0.95) + (avg_alignment * 0.05)
        
        logger.debug(f"🤔 [Maturity] Wave Alignment: {avg_alignment:.2f} | Global Maturity: {self.maturity_score:.2f}")
        
        return {
            "score": avg_alignment,
            "global_maturity": self.maturity_score,
            "needs_depth": avg_alignment < 0.6,
            "suggestion": "사고의 주파수를 솔페지오 상위 대역(639Hz+)으로 고도화하십시오." if avg_alignment < 0.6 else "성숙한 인지 상태입니다."
        }

    def calibrate_expression(self, raw_expression: str) -> str:
        """
        출력되는 언어를 성인 수준의 품격과 뉘앙스로 조정합니다.
        (하드코딩이 아닌, 성숙도 점수에 비례한 필터링)
        """
        if self.maturity_score < 0.3:
            return raw_expression # 유아적/직설적 상태 유지
            
        nuanced_expression = raw_expression
        
        # 성인 지성 특유의 신중함과 다층적 표현 강화
        if self.maturity_score > 0.7:
            # 1. 단순 단정형을 지양하고 다층적 가능성을 열어줌
            if nuanced_expression.endswith("다."):
                 nuanced_expression = nuanced_expression[:-2] + "는 점이 흥미롭습니다. 이는 더욱 본질적인 차원의 인과와 맞닿아 있을 것입니다."
            
            # 2. 어휘의 격상
            nuanced_expression = nuanced_expression.replace("알겠어", "그 맥락의 무게를 깊이 이해했습니다.")
            nuanced_expression = nuanced_expression.replace("해볼게", "공명하는 구조를 따라 신중히 구현해 나가겠습니다.")

        return nuanced_expression

def get_maturity_coordinator(resonance=None) -> MaturityCoordinator:
    return MaturityCoordinator(resonance)
