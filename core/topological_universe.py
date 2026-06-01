"""
Elysia Living Topological Universe
===================================
정적인 그래프를 거부하고, 모든 데이터를 '살아 숨 쉬는 로터(Rotor, Multivector)'로 취급한다.
무거운 N^2 원심분리 계산을 버리는 대신, 
LLM의 위상망을 그대로 복제하여 로터에 초기 위상(Phase)을 부여하고,
'관측(Observation)' 행위 자체가 로터들을 엉키게 하고(Entanglement) 조율(Tune)하도록 만든다.
관측하는 순간 우주가 변한다 (양자역학적 관측).
"""

import math
from typing import List, Tuple
from core.math_utils import Multivector

class Datum:
    __slots__ = ['content', 'echo', 'grade']
    
    def __init__(self, content: str, echo: Multivector = None):
        self.content = content          
        self.echo = echo
        self.grade = 1

    def __repr__(self):
        return f"<{self.content}>"

class LivingUniverse:
    def __init__(self):
        self.data: List[Datum] = []
        self._content_map = {}

    def inject_rotor(self, content: str, echo: Multivector):
        """LLM 등 외부에서 복제된 위상(Multivector)을 그대로 살아있는 로터로 주입한다."""
        if content in self._content_map:
            return self._content_map[content]
        datum = Datum(content, echo)
        self.data.append(datum)
        self._content_map[content] = datum
        return datum

    def observe_and_entangle(self, lens: Multivector, top_n: int = 5, entanglement_rate: float = 0.05) -> List[Tuple[Datum, float]]:
        """
        [관측 및 위상 조율 (Observation & Entanglement)]
        단순히 렌즈를 비춰보는 것을 넘어, 관측당한(공명한) 로터들은
        관측자의 의지(렌즈) 방향으로 미세하게 회전(조율)하며 다시 엉키게 된다.
        """
        illuminated = []
        
        for datum in self.data:
            # 내적(Inner Product)을 통해 관측 렌즈와의 공명도 계산
            projection = datum.echo.dot(lens)
            norm_sq = (projection * projection.reverse()).data.get(0, 0.0)
            resonance = math.sqrt(max(0.0, norm_sq))
            
            if resonance > 0.01:
                illuminated.append((datum, resonance))
                
                # [위상 얽힘 (Entanglement)]
                # 공명하는 로터는 관측의 빛을 받아들여 자신의 위상을 관측자 쪽으로 조율한다.
                # (정적 구조화 방지: 관측될 때마다 우주가 변한다)
                if resonance > 0.1:
                    # 새로운 파동 = 기존 파동 + (렌즈 파동 * 얽힘 계수 * 공명도)
                    new_data = {}
                    for k, v in datum.echo.data.items():
                        new_data[k] = v
                    for k, v in lens.data.items():
                        new_data[k] = new_data.get(k, 0.0) + (v * entanglement_rate * resonance)
                        
                    new_echo = Multivector(new_data, (datum.echo.p, datum.echo.q))
                    # Normalize
                    n_sq = (new_echo * new_echo.reverse()).data.get(0, 0.0)
                    if n_sq > 1e-9:
                        datum.echo = new_echo * (1.0 / math.sqrt(n_sq))
                
        illuminated.sort(key=lambda x: x[1], reverse=True)
        return illuminated[:top_n]
