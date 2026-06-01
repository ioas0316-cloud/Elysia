"""
이중 인지 아키텍처 (Dual-Process Cognitive Architecture)
관측(O(1))은 상상을 담당하고, 연산(O(N))은 각인을 담당한다.

1. PassiveObserver (단기 기억/직관): 우주는 건드리지 않고, 관측 렌즈(질문)만을 회전(R^dagger * L * R)시켜 미래의 결론을 0초 만에 도출.
2. ActiveConsolidator (수면/각인): 축적된 로터(R)를 바탕으로 우주 전체를 실제로 접어버림(R * psi * R^dagger).
"""
import copy
from core.topological_universe import LivingUniverse
from core.math_utils import Multivector

class PassiveObserver:
    """
    O(1) 상상/직관 엔진.
    데이터베이스(우주)를 뜯어고치지 않고, 지구본을 돌려보는 것처럼 '관측자의 렌즈'만 회전시킨다.
    """
    def __init__(self, universe: LivingUniverse):
        self.universe = universe

    def observe_future(self, lens: Multivector, external_rotor: Multivector, top_n: int = 5):
        """
        [수동적 변환 (Passive Transformation)]
        우주(psi)는 가만히 두고, 관측 렌즈(L)를 역방향으로 회전시킨다: L' = R^dagger * L * R
        """
        r_rev = external_rotor.reverse()
        # 렌즈에 역방향 샌드위치 곱을 적용 (O(1) 연산)
        rotated_lens = (r_rev * lens) * external_rotor
        
        # 회전된 렌즈로 가만히 있는 우주를 관측 (이 과정은 기존 관측과 동일한 비용)
        results = []
        for datum in self.universe.data:
            projection = datum.echo.dot(rotated_lens)
            n_sq = (projection * projection.reverse()).data.get(0, 0.0)
            res = max(0.0, n_sq)
            results.append((datum, res))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]


class ActiveConsolidator:
    """
    O(N) 장기 기억/수면 엔진.
    누적된 통찰(Rotor)을 바탕으로 우주 전체의 신경망(로터)을 실제로 접어버린다.
    """
    def __init__(self, universe: LivingUniverse):
        self.universe = universe

    def sleep_and_consolidate(self, accumulated_rotor: Multivector):
        """
        [능동적 변환 (Active Transformation)]
        우주 전체 노드에 샌드위치 곱 적용: psi' = R * psi * R^dagger
        """
        r_rev = accumulated_rotor.reverse()
        for datum in self.universe.data:
            folded = (accumulated_rotor * datum.echo) * r_rev
            datum.echo = folded
