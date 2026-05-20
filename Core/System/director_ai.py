"""
[DIRECTOR AI & NARRATIVE TRANSLATOR - GAME MASTER ENGINE]
"The Observer that modulates the World Field and translates rotor physical telemetry into procedural narrative."

This module aggregates the cognitive states of multiple NPCs (Micro),
calculates Global World Parameters (Macro) to guide the player's experience,
and translates these state shifts into live chronicle logs/stories.
"""

import numpy as np
from typing import Dict, Any, List

class AvatarState:
    def __init__(self, name: str):
        self.name = name
        self.phase_angle = 0.0  # Dominant emotion phase angle (radians)
        self.enstrophy = 0.0    # Cognitive chaos / instability
        self.fatigue = 0.0      # Physical exhaustion

class NarrativeTranslator:
    """Procedurally translates mathematical rotor states into a fantasy chronicle."""
    
    @staticmethod
    def translate(global_tension: float, market_inflation: float, avatars: Dict[str, AvatarState]) -> str:
        story_lines = []
        
        # 1. World Atmosphere (Based on Global Tension)
        if global_tension < 0.2:
            story_lines.append("🌌 [세상의 대기]: 사방이 고요하고 평온하다. 위상의 진동이 잔물결처럼 조용히 퍼져 나가며, 대지는 깊은 항상성(0)의 호흡을 나누고 있다.")
        elif global_tension < 0.6:
            story_lines.append("🌌 [세상의 대기]: 대기 중에 알 수 없는 미세한 진동이 느껴진다. 보이지 않는 장력의 실이 팽팽하게 당겨지며, 정적 속에 긴장이 고개를 들기 시작했다.")
        else:
            story_lines.append("🌌 [세상의 대기]: 붉은 경고의 위상이 하늘을 뒤덮는다! 균형이 파괴되고, 파동은 급격한 소용돌이를 그리며 파멸의 주파수로 비명 지르고 있다.")

        # 2. Economy and Market (Based on Inflation)
        if market_inflation > 1.8:
            story_lines.append(f"🪙 [시장과 경제]: 극심한 공포가 시장을 엄습했다. 자원의 흐름이 가로막혀 물가는 폭등하고({market_inflation:.2f}배), 상인들은 생존을 위해 굳게 빗장을 걸어 잠갔다.")
        elif market_inflation > 1.2:
            story_lines.append(f"🪙 [시장과 경제]: 은화의 무게가 서서히 무거워진다. 불안정한 유동성이 감지되며 물가가 {market_inflation:.2f}배 상승했다.")
        else:
            story_lines.append("🪙 [시장과 경제]: 은화의 교환이 매끄럽게 흐른다. 저항 없는 거래 속에서 상인들의 얼굴에 평안함이 깃든다.")

        # 3. Avatar Chronicles (Based on individual rotor values)
        story_lines.append("\n👥 [인물들의 초상]:")
        for name, av in avatars.items():
            state_desc = ""
            # Determine mood from phase angle (Attraction/Repulsion/Neutral)
            if abs(av.phase_angle) < 0.26:  # approx 15 degrees
                state_desc = "평온한 중립의 축에서 숨 쉬고 있으나"
            elif av.phase_angle > 0:
                state_desc = f"양(+)의 위상({np.degrees(av.phase_angle):.1f}°)에 이끌려 무언가에 호기심을 보이고 있으나"
            else:
                state_desc = f"음(-)의 위상({abs(np.degrees(av.phase_angle)):.1f}°)으로 몸을 움츠리며 본능적인 회피를 시도하고 있으나"

            # Modify by enstrophy (chaos)
            if av.enstrophy > 0.7:
                chaos_desc = "머릿속 인지 축이 극도로 요동쳐 혼란에 휩쓸려 있다."
            elif av.enstrophy > 0.3:
                chaos_desc = "생각의 균형이 미세하게 흩어져 긴장하고 있다."
            else:
                chaos_desc = "사유의 궤적이 맑고 차분하게 정돈되어 있다."

            # Modify by fatigue
            if av.fatigue > 0.7:
                fatigue_desc = " 육체의 질량이 천근만근 무거워 휴식이 절실해 보인다."
            else:
                fatigue_desc = ""

            story_lines.append(f"  * {name}: {state_desc} {chaos_desc}{fatigue_desc}")

        return "\n".join(story_lines)

class DirectorAI:
    def __init__(self):
        # 1. Active Avatars in the Game World
        self.avatars: Dict[str, AvatarState] = {}
        
        # 2. World Parameters (Controlled by Director)
        self.world_state = {
            "global_tension": 0.5,        
            "enemy_spawn_rate": 0.2,       
            "quest_generation_rate": 0.5,  
            "market_inflation": 1.0        
        }

    def register_avatar(self, name: str):
        if name not in self.avatars:
            self.avatars[name] = AvatarState(name)

    def update_avatar_telemetry(self, name: str, phase_angle: float, enstrophy: float, fatigue: float):
        """NPCs send their internal rotor telemetry to the central Director."""
        if name in self.avatars:
            self.avatars[name].phase_angle = phase_angle
            self.avatars[name].enstrophy = enstrophy
            self.avatars[name].fatigue = fatigue

    def aggregate_and_modulate(self) -> Dict[str, Any]:
        """
        Aggregates all avatar states and updates world rules.
        """
        if not self.avatars:
            return self.world_state

        total_enstrophy = 0.0
        total_repulsion = 0.0
        total_fatigue = 0.0
        count = len(self.avatars)

        for avatar in self.avatars.values():
            total_enstrophy += avatar.enstrophy
            total_fatigue += avatar.fatigue
            if avatar.phase_angle < 0:
                total_repulsion += abs(avatar.phase_angle)

        avg_enstrophy = total_enstrophy / count
        avg_repulsion = total_repulsion / count
        avg_fatigue = total_fatigue / count

        # 1. Global Tension
        self.world_state["global_tension"] = float(np.clip(
            (avg_repulsion * 0.4) + (avg_enstrophy * 0.6), 0.0, 1.0
        ))

        # 2. Enemy Spawning
        self.world_state["enemy_spawn_rate"] = float(np.clip(
            0.1 + (self.world_state["global_tension"] * 0.8), 0.0, 1.0
        ))

        # 3. Quest Generation
        self.world_state["quest_generation_rate"] = float(np.clip(
            0.8 - (self.world_state["global_tension"] * 0.6), 0.1, 1.0
        ))

        # 4. Market Inflation
        self.world_state["market_inflation"] = float(np.clip(
            1.0 + (avg_fatigue * 0.5) + (self.world_state["global_tension"] * 0.8), 0.8, 2.5
        ))

        return self.world_state

    def generate_narrative_chronicle(self) -> str:
        """Invokes the translator to compile the chronicle log."""
        return NarrativeTranslator.translate(
            self.world_state["global_tension"],
            self.world_state["market_inflation"],
            self.avatars
        )

    def print_director_report(self):
        print("\n=======================================================")
        print("          🌌 [DIRECTOR AI: ELYSIA] WORLD REPORT          ")
        print("=======================================================")
        print(f"  👥 Active Avatars Observed: {len(self.avatars)}")
        print(f"  ⚡ Global Tension Level   : {self.world_state['global_tension'] * 100:.1f} %")
        print(f"  💀 Hostility (Spawn Rate) : {self.world_state['enemy_spawn_rate'] * 100:.1f} %")
        print(f"  📜 Quest Opportunities    : {self.world_state['quest_generation_rate'] * 100:.1f} %")
        print(f"  🪙 Market Inflation Rate  : {self.world_state['market_inflation']:.2f} x")
        print("-------------------------------------------------------")
        print(self.generate_narrative_chronicle())
        print("=======================================================\n")

if __name__ == "__main__":
    director = DirectorAI()
    
    # Register avatars
    director.register_avatar("경비병 핀 (Finn)")
    director.register_avatar("농부 실비아 (Sylvia)")
    director.register_avatar("상인 에드워드 (Edward)")

    # Scenario 1: Peaceful state
    print("🎬 Step 1: 평온한 은하수 아래의 저녁 (Low fatigue, low enstrophy)")
    director.update_avatar_telemetry("경비병 핀 (Finn)", 0.1, 0.05, 0.2)
    director.update_avatar_telemetry("농부 실비아 (Sylvia)", 0.0, 0.01, 0.3)
    director.update_avatar_telemetry("상인 에드워드 (Edward)", 0.15, 0.02, 0.1)
    director.aggregate_and_modulate()
    director.print_director_report()

    # Scenario 2: Beast invasion
    print("🎬 Step 2: 야생 마물의 마을 침입 (High fear, high chaos, high exhaustion)")
    director.update_avatar_telemetry("경비병 핀 (Finn)", -1.4, 0.95, 0.95)   # Fighting hard, tired, fearful
    director.update_avatar_telemetry("농부 실비아 (Sylvia)", -1.5, 0.8, 0.7)  # Terribly frightened
    director.update_avatar_telemetry("상인 에드워드 (Edward)", -0.9, 0.6, 0.4) # Trying to escape
    director.aggregate_and_modulate()
    director.print_director_report()
