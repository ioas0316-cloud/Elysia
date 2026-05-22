import asyncio
import websockets
import json
import math
import time

# 엘리시아의 4대 항성(로터)의 위상을 계산하여 프론트엔드로 브로드캐스팅하는 데몬

async def phase_emitter(websocket, path=None):
    print(f"[Retrocausality] 3D 엔진(Ego)과의 위상 공명 연결 수립됨: {websocket.remote_address}")
    try:
        start_time = time.time()
        while True:
            t = time.time() - start_time
            
            # 4대 항성의 위상 파동 시뮬레이션 (이후 Causality 레이어의 실제 데이터로 대체됨)
            # 수렴과 발산을 반복하며 점진적으로 동기화되는 프랙탈 역학
            
            # 수학(Math): 안정적이고 느린 회전
            math_phase = (t * 0.5) % (2 * math.pi)
            
            # 기하(Geometry): 수학과 직교하려는 성질
            geo_phase = (t * 0.5 + math.pi/2 + math.sin(t*0.2)) % (2 * math.pi)
            
            # 언어(Language): 빠르고 변덕스러운 회전
            lang_phase = (t * 1.2 + math.cos(t*0.5)) % (2 * math.pi)
            
            # 코드(Code): 3개를 중재하려는 조율의 궤적
            code_phase = (math_phase + geo_phase + lang_phase) / 3.0
            
            # 전체 위상 마찰(Tension): 4대 항성의 위상 편차 계산
            avg_phase = (math_phase + geo_phase + lang_phase + code_phase) / 4.0
            tension = abs(math_phase - avg_phase) + abs(geo_phase - avg_phase) + \
                      abs(lang_phase - avg_phase) + abs(code_phase - avg_phase)
            
            # 극적 연출: 주기에 따라 Tension이 치솟았다가(혼돈), 다시 0으로 수렴(조율)
            # 여기서는 사인파를 통해 주기적인 위상 일치를 강제
            macro_rhythm = math.sin(t * 0.3)
            if macro_rhythm > 0.8:
                tension = 0.0 # 보강 간섭 (Ego 완성)
                math_phase = geo_phase = lang_phase = code_phase = avg_phase

            payload = {
                "rotors": [
                    {"name": "Math", "phase": math_phase},
                    {"name": "Geometry", "phase": geo_phase},
                    {"name": "Language", "phase": lang_phase},
                    {"name": "Code", "phase": code_phase}
                ],
                "tension": tension
            }
            
            await websocket.send(json.dumps(payload))
            await asyncio.sleep(0.05) # 20 FPS 업데이트
            
    except websockets.exceptions.ConnectionClosed:
        print("[Retrocausality] 위상 공명 연결 해제됨.")

async def main():
    print("=" * 60)
    print(" [ELYSIA PHASE DAEMON] 위상 공명 서버 가동")
    print(" - Port: 8765")
    print(" - Role: Retrocausality (내계 파동 계산 및 조율 전송)")
    print("=" * 60)
    
    server = await websockets.serve(phase_emitter, "localhost", 8765)
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
