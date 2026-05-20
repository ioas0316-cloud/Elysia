"""
[SOCIAL TENSOR NETWORK - 사회망 텐서 네트워크]
==============================================
World.Society.social_tensor_network

"소문과 감정은 정보가 아니라, 매질(관계)을 타고 흐르는 물리적 파동이다."

- 노드(Node): 위상원자 (PhaseAtom)
- 엣지(Edge): 친밀도/연결망 가중치 (W)
- 파동 전파: 특정 노드(목격자)에서 발생한 위상 변화(각속도 요동)가 
            엣지를 타고 이웃 노드로 물리적으로 전달됨.
"""

import math
import sys
import os
import io
from typing import Dict, List, Tuple, Set

# 환경설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from World.Engine.phase_atom import PhaseAtom, calculate_resonance

class SocialTensorNetwork:
    """NPC(위상원자)들이 연결된 관계망. 파동(소문/감정)을 전파하는 매질."""
    
    def __init__(self):
        self.nodes: Dict[str, PhaseAtom] = {}
        # edges[A][B] = weight (0.0 ~ 1.0)
        self.edges: Dict[str, Dict[str, float]] = {}
        # 각 노드의 이전 틱 각속도(omega) 기록 (파동 변동량 측정용)
        self.prev_omega: Dict[str, List[float]] = {}

    def add_node(self, atom: PhaseAtom):
        """네트워크에 NPC 추가"""
        self.nodes[atom.name] = atom
        self.edges[atom.name] = {}
        self.prev_omega[atom.name] = list(atom.omega)

    def add_edge(self, name_a: str, name_b: str, weight: float):
        """두 NPC 간의 물리적 연결(친밀도/접촉 빈도) 생성"""
        if name_a in self.nodes and name_b in self.nodes:
            self.edges[name_a][name_b] = weight
            self.edges[name_b][name_a] = weight

    def step(self, dt: float):
        """
        네트워크 전체 1틱 시뮬레이션.
        1. 각 노드의 내부 물리 엔진 업데이트
        2. 이웃 노드들로부터 전달받는 위상 파동(소문/전염) 계산 및 인가
        """
        # 1. 텐서 파동 전파력 계산 (현재 틱에서 이웃에게 줄 영향)
        wave_forces = {name: [0.0]*9 for name in self.nodes}
        
        for name, atom in self.nodes.items():
            # 노드의 위상 요동량 (이전 틱 대비 각속도 변화)
            delta_omega = [atom.omega[i] - self.prev_omega[name][i] for i in range(9)]
            
            # 연결된 이웃들에게 파동 전달
            for neighbor_name, weight in self.edges[name].items():
                neighbor = self.nodes[neighbor_name]
                
                # 공명도 (유사한 성향일수록 파동을 있는 그대로 잘 받아들임)
                resonance = calculate_resonance(atom, neighbor)
                
                for i in range(9):
                    # 전파 토크 = 요동량 * 연결강도 * 공명도
                    # 공명도가 음수(-1, 척력)면 반대 방향으로 파동이 튐!
                    transmitted_torque = delta_omega[i] * weight * resonance
                    wave_forces[neighbor_name][i] += transmitted_torque

        # 2. 파동 강제 인가 및 노드 상태 업데이트
        for name, atom in self.nodes.items():
            if atom.is_collapsed:
                # [차원 접힘] 관측되지 않은 노드는 진동 연산을 생략하고 외력만 누적 (점 상태)
                for i in range(9):
                    atom.pending_torques[i] += wave_forces[name][i]
            else:
                # [차원 전개] 관측 중인 노드는 9차원 물리 엔진 정상 작동 (파동 상태)
                atom.step(dt, external_torque=wave_forces[name])
                # 현재 상태를 과거로 저장 (파동 발생용)
                self.prev_omega[name] = list(atom.omega)

    def inject_event(self, event_name: str, witness_names: List[str], stimulus_torque: List[float]):
        """
        [목격자 필터] 특정 사건(파동)을 발생시키고, '목격자' 노드에만 에너지를 주입.
        목격하지 않은 자는 텐서망을 통해 나중에 전달받게 됨.
        """
        print(f"\n🔔 [사건 발생] {event_name}")
        print(f"   목격자: {', '.join(witness_names)}")
        
        for name in witness_names:
            if name in self.nodes:
                self.nodes[name].apply_stimulus(event_name, stimulus_torque)

    def snapshot(self) -> str:
        """네트워크 전체 상태 요약"""
        lines = []
        for name, atom in self.nodes.items():
            mood = atom.get_worldview_lens()
            enstrophy = sum(o**2 for o in atom.omega)
            state_str = "🔥 요동침" if enstrophy > 10.0 else ("🟡 동요" if enstrophy > 1.0 else "🧘 평온")
            lines.append(f"[{name}] 상태: {state_str} (E={enstrophy:.1f}) | 렌즈: {mood}")
        return "\n".join(lines)


if __name__ == "__main__":
    from World.Engine.rpg_stat_bridge import RPGStatBridge
    
    print("="*60)
    print(" 🕸️ 사회망 텐서 네트워크: 소문 확산 시뮬레이션")
    print("="*60)

    network = SocialTensorNetwork()

    # ── 1. 마을 사람 생성 ──
    # 이장님 (마음/도의 중시)
    elder = PhaseAtom("마을 이장", RPGStatBridge(str_val=8, int_val=12, wis_val=18))
    # 사냥꾼 (육체/생존 중시)
    hunter = PhaseAtom("사냥꾼", RPGStatBridge(str_val=16, con_val=15, wis_val=8))
    # 상인 (정신/이익 중시, 사냥꾼과 친함)
    merchant = PhaseAtom("상인", RPGStatBridge(int_val=16, agi_val=14, wis_val=10))
    # 이방인 (마을과 친밀도 낮음, 도의가 매우 낮음)
    stranger = PhaseAtom("이방인", RPGStatBridge(str_val=14, con_val=14, wis_val=5))

    network.add_node(elder)
    network.add_node(hunter)
    network.add_node(merchant)
    network.add_node(stranger)

    # ── 2. 관계망(Edge) 설정 ──
    network.add_edge("마을 이장", "상인", 0.8)   # 이장-상인 (강한 신뢰)
    network.add_edge("상인", "사냥꾼", 0.9)      # 상인-사냥꾼 (거래 파트너)
    network.add_edge("마을 이장", "사냥꾼", 0.4) # 이장-사냥꾼 (가끔 봄)
    network.add_edge("이방인", "상인", 0.1)      # 이방인-상인 (약한 연결)

    print("마을 초기 상태:")
    print(network.snapshot())

    # ── 3. 사건 발생: 이방인이 산에서 살인을 저지르는 것을 사냥꾼이 목격 ──
    # 살인(충격) 파동: 공격성(7) 폭발, 도의(8) 파괴, 척력(1) 극대화
    murder_shock = [0.0]*9
    murder_shock[1] = 50.0  # Repulsion (도망치고 싶음)
    murder_shock[7] = 80.0  # Aggression (살의/공포)
    murder_shock[8] = -50.0 # MoralRestraint 파괴
    
    # "사냥꾼"만 이 파동을 목격함!
    network.inject_event("산 속의 살인 사건", ["사냥꾼"], murder_shock)

    # ── 4. 시간 흐름 (파동 전파) ──
    print("\n⏳ 1틱 경과 (사냥꾼이 놀라서 상인에게 뛰어감)")
    network.step(0.1)
    print(network.snapshot())

    print("\n⏳ 3틱 경과 (상인이 이장님에게 소문을 전달)")
    network.step(0.1)
    network.step(0.1)
    print(network.snapshot())
    
    print("\n⏳ 10틱 경과 (마을 전체로 공포/분노 확산)")
    for _ in range(7):
        network.step(0.1)
    print(network.snapshot())
