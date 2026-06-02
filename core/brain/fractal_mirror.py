"""
프랙탈 위상 거울 (Fractal Topological Mirror)
태양(LLM)의 크기나 데이터를 복사하지 않고,
태양이 드리우는 그림자(출력 스트림의 궤적)만을 관측하여
엘리시아의 작은 막대기(가변 로터)들에 태양의 섭리를 위상 동기화시킨다.
"""
import hashlib
import math
from typing import List
from core.utils.math_utils import Multivector
from core.topological_universe import LivingUniverse

class FractalObserver:
    def __init__(self, universe: LivingUniverse, signature=(16, 0)):
        self.universe = universe
        self.signature = signature
        # 궤적을 그리기 위한 이전 시점의 관측 상태
        self.previous_shadow: Multivector = None

    def _generate_seed_rotor(self, concept: str) -> Multivector:
        """단어(막대기)의 초기 무작위 위상 (아무 섭리도 담기지 않은 맹목적 상태)"""
        h = hashlib.sha256(concept.encode('utf-8')).digest()
        data = {}
        for i in range(16):
            val = (h[i] - 128) / 128.0
            data[1 << i] = val * 0.1
        mv = Multivector(data, self.signature)
        norm_sq = (mv * mv.reverse()).data.get(0, 0.0)
        if norm_sq > 1e-9:
            return mv * (1.0 / math.sqrt(norm_sq))
        return mv

    def observe_shadow_stream(self, text_stream: List[str], learning_rate: float = 0.2):
        """
        태양이 뱉어내는 텍스트(그림자)가 흘러가는 궤적을 순차적으로 관측한다.
        어떤 숨겨진 벡터(Latent)도 계산하지 않는다.
        그저 단어가 출현하는 '순서'와 '흐름' 자체가 로터를 회전시키는 토크(Torque)가 된다.
        """
        for concept in text_stream:
            if not concept.strip():
                continue
                
            # 막대기(로터)가 없으면 꽂는다
            if concept not in self.universe._content_map:
                seed = self._generate_seed_rotor(concept)
                datum = self.universe.inject_rotor(concept, seed)
            else:
                datum = self.universe._content_map[concept]

            # [프랙탈 동기화] 그림자가 이동하는 궤적에 따라 막대기가 회전한다.
            if self.previous_shadow is not None:
                # 이전 단어의 위상이 현재 단어의 위상으로 흘러들어옴 (인과적 얽힘)
                # 계산(쐐기곱 폭발) 없이 선형적인 위상 덧셈(회전 변환)만으로 궤적을 누적
                new_data = {}
                for k, v in datum.echo.data.items():
                    new_data[k] = v
                for k, v in self.previous_shadow.data.items():
                    new_data[k] = new_data.get(k, 0.0) + (v * learning_rate)

                new_echo = Multivector(new_data, self.signature)
                
                # Normalize (막대기의 길이 유지)
                n_sq = (new_echo * new_echo.reverse()).data.get(0, 0.0)
                if n_sq > 1e-9:
                    datum.echo = new_echo * (1.0 / math.sqrt(n_sq))

            # 현재 막대기의 최종 상태가 다음 그림자의 원인이 됨
            self.previous_shadow = datum.echo
