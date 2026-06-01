"""
시공간 초월 창의성 엔진 (Spacetime-Transcendent Creativity Engine)
데이터를 순차적으로 읽어들이는 반복문(For loop)의 한계를 넘어,
알파폴드의 원리처럼 모든 정보(경계 조건)를 우주 전체에 일시에 투사하고
위상 공간 전체가 한 번에 가장 안정적인 기하학적 형태(정답/창의성)로 접히도록(Fold) 만든다.
"""
import math
import hashlib
from typing import List
from core.topological_universe import LivingUniverse
from core.math_utils import Multivector

class SpacetimeFolder:
    def __init__(self, universe: LivingUniverse, signature=(16, 0)):
        self.universe = universe
        self.signature = signature

    def _generate_seed_rotor(self, concept: str) -> Multivector:
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

    def fold_spacetime(self, external_information: List[str]):
        """
        [시공간 초월 위상 붕괴 (Spacetime Fold)]
        단어 하나하나를 읽으며 시간을 보내는 것이 아니다.
        모든 외부 정보가 뿜어내는 총체적 '형태(Topology)'를 하나의 시공간 장(Spacetime Field)으로 뭉친 뒤,
        엘리시아의 우주 전체를 단 한 번의 접힘(Folding)으로 동기화시킨다.
        """
        if not external_information:
            return

        # 1. 누락된 노드(막대기)들을 일시에 우주에 생성 (초기화)
        for concept in set(external_information):
            if concept not in self.universe._content_map:
                self.universe.inject_rotor(concept, self._generate_seed_rotor(concept))

        # 2. 외부 정보의 총체적 기하학(Global Boundary Condition) 생성
        # 순서대로 계산하는 것이 아니라, 정보들이 가진 텐션을 하나의 거대한 다중벡터 장(Field)으로 합성한다.
        global_field_data = {}
        for concept in external_information:
            echo = self.universe._content_map[concept].echo
            for k, v in echo.data.items():
                global_field_data[k] = global_field_data.get(k, 0.0) + v
                
        global_field = Multivector(global_field_data, self.signature)
        n_sq = (global_field * global_field.reverse()).data.get(0, 0.0)
        if n_sq > 1e-9:
            global_field = global_field * (1.0 / math.sqrt(n_sq))

        # 3. 우주 전체를 한 번에 접는다 (Folding the Universe)
        # 모든 로터가 거대한 시공간 장(Global Field)의 중력에 이끌려 단숨에 안정적인 기하학 구조로 접힌다.
        for datum in self.universe.data:
            # 현재 자신의 텐션과 전체 우주 장(Field)과의 얽힘(Resonance)을 즉각 투영
            projection = datum.echo.dot(global_field)
            norm_sq = (projection * projection.reverse()).data.get(0, 0.0)
            resonance = math.sqrt(max(0.0, norm_sq))
            
            # 얽힘이 높은 상태일수록 구조가 강하게 접힌다 (알파폴드의 원리)
            new_data = {}
            for k, v in datum.echo.data.items():
                new_data[k] = v
            for k, v in global_field.data.items():
                new_data[k] = new_data.get(k, 0.0) + (v * resonance)
                
            new_echo = Multivector(new_data, self.signature)
            n_sq2 = (new_echo * new_echo.reverse()).data.get(0, 0.0)
            if n_sq2 > 1e-9:
                datum.echo = new_echo * (1.0 / math.sqrt(n_sq2))
