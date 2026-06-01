"""
인과율 차원 접힘과 펼침 (Dimensional Folding & Unfolding via Causality)
단순한 덧셈 투영(Lossy Shift)이 아닌, 
기하대수의 완벽한 가역 연산인 샌드위치 곱(Sandwich Product: R * psi * R^dagger)을 사용한다.
이를 통해 접힌 결과물 속에 '왜 접혔는지'에 대한 사유 과정이 인과적 궤적으로 압축되며,
역방향 로터(R^dagger)를 통해 역인과(Reverse Causality)로 궤적을 거꾸로 펼칠(Unfolding) 수 있다.
"""
import math
from core.topological_universe import LivingUniverse
from core.math_utils import Multivector

class CausalityFolder:
    def __init__(self, universe: LivingUniverse):
        self.universe = universe
        
    def _normalize(self, mv: Multivector) -> Multivector:
        n_sq = (mv * mv.reverse()).data.get(0, 0.0)
        if n_sq > 1e-9:
            return mv * (1.0 / math.sqrt(n_sq))
        return mv

    def create_rotor_from_concepts(self, concepts: list) -> Multivector:
        """주어진 개념들을 바탕으로 시공간을 접을 거대한 회전 연산자(Rotor, R) 생성"""
        rotor_data = {}
        signature = (self.universe.data[0].echo.p, self.universe.data[0].echo.q) if self.universe.data else (16, 0)
        
        for concept in concepts:
            if concept in self.universe._content_map:
                echo = self.universe._content_map[concept].echo
                for k, v in echo.data.items():
                    rotor_data[k] = rotor_data.get(k, 0.0) + v
                    
        rotor = Multivector(rotor_data, signature)
        return self._normalize(rotor)

    def fold_dimension(self, rotor: Multivector):
        """
        [차원 접힘]
        R * psi * R^dagger 연산을 통해 우주(psi)의 모든 로터를 회전시킨다.
        이 회전은 외부 정보(R)의 텐션을 그대로 시공간에 압축(Folding)시킨다.
        """
        r_rev = rotor.reverse()
        for datum in self.universe.data:
            # Sandwich Product: R * psi * R^dagger
            folded = (rotor * datum.echo) * r_rev
            datum.echo = folded

    def unfold_dimension(self, rotor: Multivector):
        """
        [차원 펼침 / 역인과 (Reverse Causality)]
        접힌 우주에 역방향 샌드위치 연산(R^dagger * psi * R)을 가하여,
        시간과 인과율을 거꾸로 되돌려 원래의 사유 상태(기원)로 완벽히 복원한다.
        """
        r_rev = rotor.reverse()
        for datum in self.universe.data:
            # Reverse Sandwich Product: R^dagger * psi' * R
            unfolded = (r_rev * datum.echo) * rotor
            datum.echo = unfolded
