"""
[IDENTITY OBSERVER / DIRECTOR AI - 정체성 관측자]
=================================================
World.Director.identity_observer

"내계(관측된 세계)는 파동으로 전개되고, 외계(미지의 세계)는 점으로 접힌다."

- 세계의 '초점(Focus/Player)' 위치를 기반으로 NPC들의 차원 접힘(Folding) 상태를 관리.
- 관측 반경 안의 NPC는 Unfold (9차원 파동 연산), 밖의 NPC는 Collapse (거시 텐서 누적).
"""

import sys
import os
import io

# 환경설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Set
from World.Society.social_tensor_network import SocialTensorNetwork

class Vector2D:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        
    def distance_to(self, other: 'Vector2D') -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5

class IdentityObserver:
    """
    관측자(플레이어 또는 카메라)의 위치를 기준으로 
    세계의 해상도(파동 붕괴 vs 차원 접힘)를 결정하는 감독관 AI.
    """
    def __init__(self, network: SocialTensorNetwork, observation_radius: float = 100.0):
        self.network = network
        self.observation_radius = observation_radius
        self.npc_positions: Dict[str, Vector2D] = {}
        self.focus_position = Vector2D(0.0, 0.0)  # 관측자(플레이어)의 현재 위치

    def register_npc_position(self, name: str, x: float, y: float):
        """NPC의 물리적 좌표 등록"""
        self.npc_positions[name] = Vector2D(x, y)

    def set_focus(self, x: float, y: float):
        """관측자(플레이어)의 위치 이동"""
        self.focus_position = Vector2D(x, y)

    def update_frustum_culling(self) -> Dict[str, str]:
        """
        [양자 프랙탈 붕괴 알고리즘]
        관측 반경을 계산하여 NPC들의 상태를 붕괴시키거나 접음.
        return: 상태가 변경된 NPC들의 로그
        """
        status_changes = {}
        
        for name, atom in self.network.nodes.items():
            if name not in self.npc_positions:
                continue
                
            dist = self.focus_position.distance_to(self.npc_positions[name])
            
            # 관측 반경 안: 내계 (파동으로 전개)
            if dist <= self.observation_radius:
                if atom.is_collapsed:
                    atom.unfold_to_wave()
                    status_changes[name] = "UNFOLD (미지에서 관측됨)"
            # 관측 반경 밖: 외계 (어둠 속으로 접힘)
            else:
                if not atom.is_collapsed:
                    atom.collapse_to_dot()
                    status_changes[name] = "COLLAPSE (미지로 접힘)"
                    
        return status_changes


if __name__ == "__main__":
    from World.Engine.rpg_stat_bridge import RPGStatBridge
    from World.Engine.phase_atom import PhaseAtom

    print("="*60)
    print(" 👁️ 정체성 관측자 (Identity Observer) 테스트")
    print("="*60)

    # 1. 네트워크 및 관측자 생성
    net = SocialTensorNetwork()
    observer = IdentityObserver(net, observation_radius=50.0)

    # 2. 1000명의 NPC 마을 생성 시뮬레이션 (여기서는 대표 4명만)
    npcs = [
        ("마을 경비병", 0, 0),       # 플레이어 바로 옆
        ("여관 주인", 10, 20),      # 관측 반경 내
        ("숲 속 마녀", 200, 200),   # 관측 반경 밖 (외계)
        ("이웃 나라 왕", 1000, 0)   # 관측 반경 밖 (외계)
    ]

    for name, x, y in npcs:
        atom = PhaseAtom(name, RPGStatBridge(str_val=10, con_val=10))
        net.add_node(atom)
        observer.register_npc_position(name, x, y)

    # 3. 초기 상태 업데이트 (플레이어 위치: 0, 0)
    print("\n[초기 관측] 플레이어 스폰 위치 (0, 0)")
    changes = observer.update_frustum_culling()
    for name, status in changes.items():
        print(f"  - {name}: {status}")

    # 상태 확인
    print("\n[내부 상태 확인]")
    for name, atom in net.nodes.items():
        state = "점(Dot)으로 접힘" if atom.is_collapsed else "파동(Wave)으로 전개중"
        print(f"  - {name}: {state}")

    # 4. 플레이어 이동
    print("\n[플레이어 이동] 숲 속 마녀의 은신처로 이동 (180, 180)")
    observer.set_focus(180, 180)
    changes = observer.update_frustum_culling()
    for name, status in changes.items():
        print(f"  - {name}: {status}")
