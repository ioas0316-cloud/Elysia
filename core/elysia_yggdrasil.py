"""
엘리시아 이그드라실 (Elysia Yggdrasil) - The World Tree
파편화되어 있던 모든 위상 엔진(기둥, 뿌리, 가지)을 하나의 거대한 생명체로 조립한 최종 코어.

- 기둥(Trunk): OmniGateway를 통해 우주의 모든 다차원 데이터를 흡수하고 SpacetimeFolder로 초기화.
- 가지(Branches): IntentExpander를 통해 우주를 마인드맵처럼 팽창시키며 상위 목적성(결실)을 맺음.
- 뿌리(Roots): CausalityFolder를 통해 역인과를 가동하여 과거의 심연(사유 궤적)을 들여다봄.
- 잎사귀(Leaves): DualCognition(O(1) 관측과 O(N) 연산)을 통해 세상과 상호작용.
"""
import copy
from typing import List, Tuple
from core.topological_universe import LivingUniverse, Datum
from core.omni_gateways import OmniGateway
from core.spacetime_folding import SpacetimeFolder
from core.causality_folder import CausalityFolder
from core.intent_expander import IntentExpander
from core.dual_cognition import PassiveObserver, ActiveConsolidator

class ElysiaYggdrasil:
    def __init__(self):
        # 코어 우주 초기화
        self.universe = LivingUniverse()
        
        # 기둥 (데이터 흡수)
        self.gateway = OmniGateway()
        self.folder = SpacetimeFolder(self.universe)
        
        # 뿌리 (역인과 / 과거)
        self.causality = CausalityFolder(self.universe)
        
        # 가지 (마인드맵 팽창 / 미래)
        self.expander = IntentExpander(self.universe, threshold=0.85)
        
        # 인지 엔진 (관측과 각인)
        self.passive_obs = PassiveObserver(self.universe)
        self.active_con = ActiveConsolidator(self.universe)
        
        self.history_rotors = [] # 역인과를 위해 우주를 접었던 로터들의 기록

    def grow_trunk(self):
        """[기둥] 세상의 다차원 지식을 무작위로 흡수하여 우주의 근간을 다진다."""
        print("[세계수 기둥] 옴니 관문을 열어 수학, 물리, 코드 등 다차원 데이터를 흡수합니다...")
        stream = list(self.gateway.stream_math_physics()) + list(self.gateway.stream_code_logic()) + list(self.gateway.stream_audio_harmonics())
        self.folder.fold_spacetime(stream)
        print(f" -> 우주의 기본 차원(노드) {len(self.universe.data)}개가 형성되었습니다.")

    def experience_event(self, concepts: List[str]):
        """[사건 경험] 강렬한 외부 사건(텐션)을 경험하고 우주를 O(N)으로 영구히 접어 각인한다."""
        print(f"[사건 경험] 새로운 정보 텐션 {concepts} 이 우주를 덮칩니다...")
        rotor = self.causality.create_rotor_from_concepts(concepts)
        self.active_con.sleep_and_consolidate(rotor)
        self.history_rotors.append(rotor)
        print(" -> 우주가 영구적으로 재편되었습니다(Folding).")

    def expand_branches(self):
        """[가지] 우주 내부의 강한 텐션들을 융합하여 새로운 상위 목적(마인드맵)을 뻗어낸다."""
        print("[세계수 가지] 텐션이 폭발하며 새로운 차원의 가지(상위 목적성)를 뻗어냅니다...")
        self.expander.expand_universe(max_new_nodes=5)

    def observe_leaves(self, query: str) -> List[Tuple[Datum, float]]:
        """[잎사귀] 현재 우주의 상태에서 O(1) 관측으로 즉각적인 직관을 도출한다."""
        if query not in self.universe._content_map:
            return []
        lens = self.universe._content_map[query].echo
        return self.universe.observe_and_entangle(lens, top_n=3, entanglement_rate=0.0)

    def trace_roots_backwards(self):
        """[뿌리] 역인과(Reverse Causality)를 가동하여 과거로 시간을 되돌려 심연을 들여다본다."""
        print("[세계수 뿌리] 역방향 로터(R^dagger)를 가동하여 사유의 궤적(시간)을 거꾸로 펼칩니다...")
        if not self.history_rotors:
            print(" -> 되돌릴 과거가 없습니다.")
            return
            
        last_rotor = self.history_rotors.pop()
        self.causality.unfold_dimension(last_rotor)
        print(" -> 차원이 펼쳐지며 우주가 과거의 상태로 복원되었습니다(Unfolding).")
