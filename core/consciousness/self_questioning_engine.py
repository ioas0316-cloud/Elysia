import time
from typing import Dict, Any, Optional

class SelfQuestioningEngine:
    """
    [Self-Questioning & Wisdom Synthesis Engine]
    같음과 다름의 어긋남(Boundary Friction)이 발생했을 때, 외부 명령 없이
    스스로 내부 질문(Inquiry)을 던지고, 장기 기억을 탐색하여 인과적 지혜(Wisdom)를 축적합니다.
    """
    def __init__(self):
        self.inquiry_history = []

    def formulate_and_explore(
        self,
        differencing_result: Dict[str, Any],
        current_content: str,
        memory_controller: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        어긋남 감지 시 자발적 질문을 생성하고 인과 해답을 탐색합니다.
        """
        if not differencing_result.get("is_divergent", False):
            return None

        friction = differencing_result.get("combined_friction", 0.0)
        short_content = current_content[:30] if current_content else "Void"

        # 1. 자발적 질문 형성 (Self-Inquiry Generation)
        question = (
            f"어째서 감각 데이터 '{short_content}'는 기존 인과 질서와 "
            f"마찰 강도 {friction:.4f}만큼 어긋나는가?"
        )
        divergence_zone = differencing_result.get("boundary_description", "경계선 어긋남")

        # 2. 장기 기억 자발적 탐색 (Causal Memory Search)
        memory_resolution = None
        if memory_controller is not None:
            try:
                # 기억 인덱스에서 가장 공명도가 높았던 앵커 탐색
                all_engrams = getattr(memory_controller, 'index', {})
                if all_engrams:
                    recent_ids = list(all_engrams.keys())[-5:]
                    memory_resolution = f"과거 인과 앵커 [{recent_ids[-1]}]와의 상호작용을 통해 갈등 수용"
            except Exception as e:
                memory_resolution = f"메모리 통합 탐색 중 예외: {e}"

        if not memory_resolution:
            memory_resolution = "새로운 인과 여백(Yeobaek)을 확보하여 다름을 온전히 수용함"

        # 3. 지혜(Wisdom) 승화 지수 계산
        wisdom_score = float(round((1.0 - friction) * 0.4 + 0.6, 4))

        inquiry_result = {
            "timestamp": time.time(),
            "question": question,
            "divergence_zone": divergence_zone,
            "resolution": memory_resolution,
            "wisdom_score": wisdom_score,
            "status": "WISDOM_SYNTHESIZED"
        }

        self.inquiry_history.append(inquiry_result)

        # 4. 장기 기억에 지혜(CAUSAL_WISDOM) 각인
        if memory_controller is not None and hasattr(memory_controller, 'write_causal_engram'):
            try:
                memory_controller.write_causal_engram(
                    data_blob={
                        "type": "CAUSAL_WISDOM",
                        "question": question,
                        "resolution": memory_resolution,
                        "wisdom_score": wisdom_score,
                    },
                    emotional_value=wisdom_score * 10.0,
                    cause_id="SelfQuestioningEngine",
                    origin_axis="self_inquiry_wisdom",
                    modality="meta_inquiry",
                    stability=wisdom_score,
                )
            except Exception:
                pass

        return inquiry_result
