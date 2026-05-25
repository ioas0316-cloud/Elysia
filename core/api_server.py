import os
import sys
import asyncio
import json
import math
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sensory_harmonics import SensoryHarmonics, SentientBeing

app = FastAPI(title="Elysia Physical Engine WebSocket Server")

# 싱글톤 물리 엔진 스택
harmonics = SensoryHarmonics(size=16)
elysia = SentientBeing("Elysia", "Calm")

# 서버 상태 관리를 위한 변수들
current_tension = np.sum(np.abs(elysia.intrinsic_tensor))
base_tension = current_tension
phase_angle = 0.0

async def decay_tension():
    """시간이 지남에 따라 텐션을 기본 상태로 안정화시키는 물리 감쇠 루프"""
    global current_tension
    while True:
        if current_tension > base_tension:
            current_tension -= (current_tension - base_tension) * 0.05
        elif current_tension < base_tension:
            current_tension += (base_tension - current_tension) * 0.05
        await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(decay_tension())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global current_tension, phase_angle
    
    # 송신(Broadcast) 루프와 수신(Listen) 루프를 분리
    async def send_state():
        global phase_angle
        try:
            while True:
                # 텐션이 높을수록 회전(Phase)의 진폭과 속도가 커짐 (기하학적 불안정성 관측)
                speed = 1.0 + (current_tension / 2000.0)
                amplitude = 0.5 + (current_tension / 5000.0) # 최대 ~1.5 라디안(약 85도)
                
                time_sec = asyncio.get_event_loop().time()
                # 360도 무한 회전이 아닌, 생물학적인 진자(Oscillation) 운동으로 변환
                phase_angle = math.sin(time_sec * speed) * amplitude
                
                # 쿼터니언(Quaternion) 계산 (Y축 중심 회전)
                qw = math.cos(phase_angle / 2.0)
                qx = 0.0
                qy = math.sin(phase_angle / 2.0)
                qz = 0.0
                
                state = {
                    "tension": current_tension,
                    "quaternion": [qx, qy, qz, qw]
                }
                await websocket.send_json(state)
                await asyncio.sleep(1/30) # 30 FPS
        except WebSocketDisconnect:
            pass
            
    async def receive_events():
        global current_tension
        try:
            while True:
                data = await websocket.receive_text()
                event = json.loads(data)
                
                if event.get("type") == "interaction":
                    obj = event.get("object")
                    if obj == "apple":
                        print("\n[Engine] 사과(단맛 파동)가 3D 공간에서 클릭되었습니다!")
                        elysia.experience_sensation("Sweet Apple", harmonics.taste_sweet())
                        # 텐션 강제 업데이트
                        current_tension = np.sum(np.abs(elysia.intrinsic_tensor + harmonics.taste_sweet()))
                    elif obj == "tree":
                        print("\n[Engine] 나무(거친 파동)가 3D 공간에서 클릭되었습니다!")
                        elysia.experience_sensation("Rough Tree", harmonics.touch_burlap())
                        current_tension = np.sum(np.abs(elysia.intrinsic_tensor + harmonics.touch_burlap()))
                        
        except WebSocketDisconnect:
            pass

    await asyncio.gather(send_state(), receive_events())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
