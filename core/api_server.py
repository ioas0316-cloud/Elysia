import os
import sys
import asyncio
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="Elysia Physical Engine WebSocket Server")

DATA_DIR = r"c:\Elysia\data"
CORE_EGRESS_PATH = os.path.join(DATA_DIR, "core_egress_state.json")
INTERACTION_EVENTS_PATH = os.path.join(DATA_DIR, "interaction_events.json")

os.makedirs(DATA_DIR, exist_ok=True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # 송신(Broadcast) 루프와 수신(Listen) 루프를 분리
    async def send_state():
        try:
            while True:
                tension = 0.0
                quaternion = [0.0, 0.0, 0.0, 1.0]  # [qx, qy, qz, qw]
                is_sleeping = False
                sleep_factor = 0.0
                
                if os.path.exists(CORE_EGRESS_PATH):
                    try:
                        with open(CORE_EGRESS_PATH, "r", encoding="utf-8") as f:
                            egress_state = json.load(f)
                        tension = egress_state.get("tension", 0.0)
                        is_sleeping = egress_state.get("is_sleeping", False)
                        sleep_factor = egress_state.get("sleep_factor", 0.0)
                        phase_rotor = egress_state.get("phase_rotor", [])
                        if len(phase_rotor) >= 4:
                            qw, qx, qy, qz = phase_rotor[0:4]
                            # Web client App.jsx expects [qx, qy, qz, qw]
                            quaternion = [qx, qy, qz, qw]
                    except Exception:
                        pass
                
                state = {
                    "tension": tension,
                    "quaternion": quaternion,
                    "is_sleeping": is_sleeping,
                    "sleep_factor": sleep_factor
                }
                await websocket.send_json(state)
                await asyncio.sleep(1/30) # 30 FPS
        except WebSocketDisconnect:
            pass
            
    async def receive_events():
        try:
            while True:
                data = await websocket.receive_text()
                event = json.loads(data)
                
                if event.get("type") == "interaction":
                    obj = event.get("object")
                    print(f"\n[API Server] 브라우저 클릭 수신: '{obj}'")
                    
                    # interaction_events.json에 이벤트 기록하여 데몬에 전송
                    interaction_event = {
                        "timestamp": time.time(),
                        "object": obj
                    }
                    try:
                        with open(INTERACTION_EVENTS_PATH, "w", encoding="utf-8") as f:
                            json.dump(interaction_event, f)
                    except Exception as e:
                        print(f"⚠️ [API Server] interaction_events.json 쓰기 실패: {e}")
                        
        except WebSocketDisconnect:
            pass

    await asyncio.gather(send_state(), receive_events())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
