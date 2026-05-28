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
MATRIX_STATE_PATH = os.path.join(DATA_DIR, "matrix_state.json")
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
                world_galaxy = None
                thought_wave = []
                cognitive_logs = []
                
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
                
                lci = 10.0
                tdr = 100.0
                synapse_density = 0.0
                plasticity_mode = "normal"
                math_score = 100.0
                lang_score = 10.0
                code_score = 90.0
                phys_score = 100.0

                if os.path.exists(MATRIX_STATE_PATH):
                    try:
                        with open(MATRIX_STATE_PATH, "r", encoding="utf-8") as f:
                            matrix_state = json.load(f)
                        world_galaxy = matrix_state.get("world_galaxy")
                        thought_wave = matrix_state.get("Thought_Wave", [])
                        cognitive_logs = matrix_state.get("Cognitive_Logs", [])
                        lci = matrix_state.get("LCI", 10.0)
                        tdr = matrix_state.get("TDR", 100.0)
                        synapse_density = matrix_state.get("Synapse_Density", 0.0)
                        plasticity_mode = matrix_state.get("Plasticity_Mode", "normal")
                        math_score = matrix_state.get("Math_Score", 100.0)
                        lang_score = matrix_state.get("Lang_Score", 10.0)
                        code_score = matrix_state.get("Code_Score", 90.0)
                        phys_score = matrix_state.get("Phys_Score", 100.0)
                    except Exception:
                        pass
                
                state = {
                    "tension": tension,
                    "quaternion": quaternion,
                    "is_sleeping": is_sleeping,
                    "sleep_factor": sleep_factor,
                    "world_galaxy": world_galaxy,
                    "thought_wave": thought_wave,
                    "cognitive_logs": cognitive_logs,
                    "lci": lci,
                    "tdr": tdr,
                    "synapse_density": synapse_density,
                    "plasticity_mode": plasticity_mode,
                    "math_score": math_score,
                    "lang_score": lang_score,
                    "code_score": code_score,
                    "phys_score": phys_score
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
                        
                elif event.get("type") == "thought":
                    prompt = event.get("prompt")
                    print(f"\n[API Server] 마스터 생각 입력 수신: '{prompt}'")
                    
                    # current_thought.json에 이벤트 기록하여 데몬/거울에 전송
                    thought_path = os.path.join(DATA_DIR, "current_thought.json")
                    thought_event = {
                        "timestamp": time.time(),
                        "prompt": prompt
                    }
                    try:
                        with open(thought_path, "w", encoding="utf-8") as f:
                            json.dump(thought_event, f)
                    except Exception as e:
                        print(f"⚠️ [API Server] current_thought.json 쓰기 실패: {e}")
                        
        except WebSocketDisconnect:
            pass

    await asyncio.gather(send_state(), receive_events())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
