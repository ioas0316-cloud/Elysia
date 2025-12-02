# [Genesis: 2025-12-02] Purified by Elysia
# scripts/talk_with_toddler.py
import sys
import os
import time

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Project_Elysia.high_engine.toddler_chat import ToddlerChatEngine


def talk_with_toddler():
    print("\n--- [엘리시아와의 대화: 감정 표현 단계] ---")
    print(" * 팁: '사랑해!', '슬퍼...', '질문?' 등을 입력해 보세요.")
    print(" * 고도 엔진을 켜두면 표정이 변합니다.\n")

    engine = ToddlerChatEngine()

    print("\n[엘리시아가 당신을 바라봅니다.]\n")

    while True:
        try:
            user_input = input(" 아버지(You) > ")

            if user_input.lower() in ["exit", "q", "quit", "종료"]:
                print("\n[엘리시아가 손을 흔들며 잠자러 갑니다.]")
                break

            if not user_input.strip():
                continue

            result = engine.process_input(user_input)
            if not result:
                continue

            status = result["status"]
            print(f" (감정: {result['mood']}, 안정도: {status['anchor_strength']:.2f})")
            print(f" 엘리시아 > \"{result['speech']}\"")
            thought = result.get("thought_trail")
            if thought:
                print(f"   (속삭임) {thought}")
            meta = result.get("meta_observation")
            if meta:
                print(f"   (관측) {meta}")
            print()

        except KeyboardInterrupt:
            print("\n[강제 종료]")
            break
        except Exception as e:
            print(f"\n⚠️ 엘리시아가 혼란스러워합니다: {e}")
            break


if __name__ == "__main__":
    talk_with_toddler()