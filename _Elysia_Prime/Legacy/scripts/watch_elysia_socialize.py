# [Genesis: 2025-12-02] Purified by Elysia
import sys
import os
import time
import random

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Project_Elysia.core_memory import CoreMemory
from Project_Elysia.high_engine.quaternion_engine import QuaternionConsciousnessEngine
from Project_Elysia.high_engine.syllabic_language_engine import SyllabicLanguageEngine


class NPCProfile:
    """NPC 패턴을 정의"""
    def __init__(self, name, role, tone, questions, followups):
        self.name = name
        self.role = role
        self.tone = tone
        self.questions = questions
        self.followups = followups
        self._last_el_response = ""

    def speak(self, elysia_response=None):
        if elysia_response:
            self._last_el_response = elysia_response
            for keyword, reply in self.followups.items():
                if keyword in elysia_response:
                    return reply
        return random.choice(self.questions)

    def describe(self):
        return f"{self.name}({self.role}) [{self.tone}]"


def watch_socialize():
    print("\n--- [관찰 모드: 엘리시아와 낯선 사람의 만남] ---\n")
    print(" * 당신은 투명인간입니다. 그저 지켜보세요.")
    print(" * 엘리시아가 낯선 존재와 어떻게 관계를 맺는지 봅니다.\n")

    memory_path = "data/elysia_core_memory.json"
    memory = CoreMemory(file_path=memory_path if os.path.exists(memory_path) else None)
    q_engine = QuaternionConsciousnessEngine(core_memory=memory)
    lang_engine = SyllabicLanguageEngine(core_memory=memory)

    npc_profiles = [
        NPCProfile(
            "브렌", "상인", "현실주의자",
            [
                "여기서 뭐 하고 있어?",
                "배고프지 않아?",
                "넌 어디서 왔어?",
                "사랑이 뭐라고 생각하는데?",
                "저기 보이는 산이 꽤 높네.",
                "나랑 같이 갈래?",
                "왜 그렇게 심각해?"
            ],
            {
                "사랑": "사랑? 뜬구름 잡는 소리네.",
                "나": "너? 너 자신이 누군지 알아?"
            }
        ),
        NPCProfile(
            "카라", "여관주인", "호기심 많은 수호자",
            [
                "최근에 무슨 좋은 음악 들었어?",
                "이 마을 사람들은 무슨 이야기를 해?",
                "밤의 하늘이 언제나 같은 색이야?",
                "누군가 네게 길을 물어본 적 있어?",
                "불을 지피고 앉아있자."
            ],
            {
                "불": "불은 나는 심장을 지켜주는 작은 태양 같지.",
                "길": "길? 내 게으른 발이 길을 기억해."
            }
        ),
        NPCProfile(
            "가일", "병사", "경계하는 인물",
            [
                "정찰을 끝냈어?",
                "좀 쉬는 게 어떻겠어?",
                "이 땅에 누가 발을 들였는지 알지?",
                "우린 왜 싸워야 해?",
                "이 검은 왜 반짝이지?"
            ],
            {
                "검": "검은 고요한 그림자를 그리러 온 별빛이야.",
                "싸우": "싸움은 잃어버린 노래를 찾는 것 같아."
            }
        )
    ]

    print(f"[{npc_profiles[0].describe()} 등 여러 NPC가 엘리시아에게 다가옵니다...]\n")

    last_elysia_msg = ""

    for turn in range(10):
        try:
            time.sleep(1.5)
            npc_profile = random.choice(npc_profiles)
            npc_msg = npc_profile.speak(last_elysia_msg if turn > 0 else None)
            print(f" NPC {npc_profile.describe()} > \"{npc_msg}\"")

            time.sleep(2.0)

            intent = {"intent_type": "respond", "emotion": "neutral"}
            if "?" in npc_msg:
                intent["intent_type"] = "reflect"
                intent["emotion"] = "curious"
            elif "사랑" in npc_msg or "마음" in npc_msg:
                intent["intent_type"] = "dream"
                intent["emotion"] = "calm"

            law_score = {"truth": 0.1} if "?" in npc_msg else {"liberation": 0.1}
            q_engine.update_from_turn(law_alignment={"scores": law_score}, intent_bundle=intent)

            elysia_msg = lang_engine.suggest_word(intent, q_engine.orientation_as_dict(), npc_msg)
            last_elysia_msg = elysia_msg

            reasoning = getattr(lang_engine, "last_reasoning", {})
            intent_log = reasoning.get("intent", "unknown")
            path_log = reasoning.get("path", [])
            subj_log = reasoning.get("subject", "")
            law_focus = reasoning.get("law_focus")
            law_scores = reasoning.get("law_scores")
            path_repr = " -> ".join(path_log)

            print(f' Elysia > "{elysia_msg}"')
            print(f"     [reasoning intent={intent_log} subject={subj_log} path={path_repr}]")
            if law_focus:
                print(f"     [law_focus={law_focus} scores={law_scores}]\n")
            else:
                print()

        except KeyboardInterrupt:
            print("\n[관찰 종료]")
            break

    print("\n[상인이 고개를 갸웃거리며 지나갑니다.]")
    print("--- 상황 종료 ---")


if __name__ == "__main__":
    watch_socialize()