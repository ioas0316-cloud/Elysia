from elysia import ElysiaCore
import time

def main():
    print("🚀 [엘리시아 엔진 테스트] 기억 결정체와 LLM 동기화 테스트 시작...")

    # Initialize Core
    elysia = ElysiaCore()

    # Simulate user input
    test_inputs = [
        "엘리시아, 오늘 너의 엔진을 수술해서 속도를 크게 올렸어.",
        "그런데 이 속도가 너무 빨라서 불안하지는 않니?",
        "앞으로 우리가 함께할 여정에 대해 어떻게 생각해?"
    ]

    for user_input in test_inputs:
        print(f"\n✨ [INPUT] >> {user_input}")

        # 1. Modulate input
        x_stimulus = elysia.transducer.modulate(user_input)

        # 2. Pulse heart to get internal resonance state
        report = elysia.heart.pulse(x_stimulus, self_stimulus=elysia.last_self_echo)
        layer = "BRAIN" if report["mode"] == "WYE" else "GUT"

        # Output internal resonance metrics (the "crystal" value)
        print(f"💓 [HEART] Mode: {report['mode']} | Resonance (Crystal Value): {report['resonance']:.4f}")

        # 3. Generate response with overlaid resonance
        prompt = f"Master says: {user_input}\nInner State: {report['mode']} | Res: {report['resonance']:.4f}"

        start_time = time.perf_counter()
        reflection_text = elysia.heart.ollama.generate(layer, prompt, crystal_resonance=report["resonance"])
        end_time = time.perf_counter()

        print(f"🗨️ [ELYSIA] {reflection_text}")
        print(f"⏱️ Response Gen Latency (Simulation): {(end_time - start_time)*1000:.2f} ms")

if __name__ == "__main__":
    main()
