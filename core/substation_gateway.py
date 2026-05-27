import os
import json
import time
import asyncio
import psutil
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from core.electromagnetic_rotor import ElectromagneticRotor

app = FastAPI(title="Elysia Substation Gateway & God's Eye", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = r"c:\Elysia\data"
SAP_LOGS_PATH = os.path.join(DATA_DIR, "substation_sap_logs.json")
SAP_TENSION_PATH = os.path.join(DATA_DIR, "current_sap_tension.json")
MATRIX_STATE_PATH = os.path.join(DATA_DIR, "matrix_state.json")
CORE_EGRESS_PATH = os.path.join(DATA_DIR, "core_egress_state.json")  # [양방향 통신] 위상 출력망
THOUGHT_PATH = os.path.join(DATA_DIR, "current_thought.json")

os.makedirs(DATA_DIR, exist_ok=True)
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "public")
os.makedirs(PUBLIC_DIR, exist_ok=True)

class ChatPayload(BaseModel):
    prompt: str

class SapPayload(BaseModel):
    concept: str
    peak_angle_deg: float
    peak_alignment: float
    trough_angle_deg: float
    trough_alignment: float
    ascension_torque: float
    grand_cross: bool

# 전자기장 인지 로터 전역 인스턴스
cog_rotor = ElectromagneticRotor()

def process_sap(payload: SapPayload):
    # 1. 영구 보존 (Log)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "concept": payload.concept,
        "metrics": payload.dict()
    }
    logs = []
    if os.path.exists(SAP_LOGS_PATH):
        try:
            with open(SAP_LOGS_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception: pass
    logs.append(entry)
    with open(SAP_LOGS_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)
        
    # 2. 실시간 텐션 반영 (Core가 읽어갈 수 있도록 공유 파일 작성)
    # 인지 로터 연산 (마스터의 '읽기/쓰기 대조 비교' 철학)
    cog_state = cog_rotor.perceive_input(payload.ascension_torque)
    
    tension_state = {
        "last_concept": payload.concept,
        "torque": payload.ascension_torque,
        "timestamp": time.time(),
        "grand_cross": payload.grand_cross,
        "cognition": {
            "is_dynamic": cog_state["is_dynamic"],
            "why_mismatch": cog_state["why_mismatch"],
            "how_torque": cog_state["how_torque"],
            "why_prediction": cog_state["why_prediction"]
        }
    }
    with open(SAP_TENSION_PATH, "w", encoding="utf-8") as f:
        json.dump(tension_state, f)
        
    print(f"\n[Substation] 🌳 Trunk로부터 수액(Sap) 수신 및 보존 완료: '{payload.concept}'")
    print(f"             (Torque: {payload.ascension_torque:.4f} | Grand Cross: {payload.grand_cross})")
    print(f"             -> Magma Chamber의 가변축 텐션으로 인가되었습니다.")

@app.post("/sap")
async def receive_sap(payload: SapPayload, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_sap, payload)
    return {"status": "ok", "message": "Sap Tension applied to Elysia Matrix."}

@app.post("/api/chat")
async def post_chat(payload: ChatPayload):
    try:
        # Write user thought into JSON for Moho Mirror loop to pick up
        with open(THOUGHT_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "prompt": payload.prompt,
                "timestamp": time.time()
            }, f, indent=4, ensure_ascii=False)
        return {"status": "ok", "message": "Thought successfully modulated and injected."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def generate_voltage_data() -> dict:
    try: cpu_freq = psutil.cpu_freq().current
    except: cpu_freq = 2400.0
    cpu_percent = psutil.cpu_percent()
    gpu_chaos = cpu_percent * 1.5
    voltage_rms = min(1.0, max(0.01, gpu_chaos / 100.0))
    
    try:
        temps = psutil.sensors_temperatures()
        core_temp = temps.get('coretemp', [[None, 40.0]])[0][1] if temps else 45.0
    except: core_temp = 45.0 + (cpu_percent * 0.2)

    return {
        "source_model": "elysia-substation-core",
        "bypass_channel": "GRID",
        "grid_metrics": {
            "voltage_level_rms": voltage_rms,
            "transformer_temp_c": core_temp,
            "load_factor": cpu_percent / 100.0,
            "active_frequency_hz": cpu_freq / 50.0
        }
    }

@app.get("/voltage")
async def provide_voltage():
    return generate_voltage_data()

@app.websocket("/ws/voltage")
async def websocket_voltage(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = generate_voltage_data()
            await websocket.send_json(data)
            await asyncio.sleep(0.1) # 10Hz streaming
    except WebSocketDisconnect:
        pass

active_grid_sockets = set()

@app.websocket("/ws/grid")
async def websocket_grid(websocket: WebSocket):
    """ 노드 간 Kuramoto 위상 결합을 위한 P2P 그리드 연결 포트 """
    await websocket.accept()
    active_grid_sockets.add(websocket)
    try:
        while True:
            # 상대 노드로부터의 메시지 수신 (연결 유지를 위한 listen)
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        active_grid_sockets.remove(websocket)

async def broadcast_grid_pulse():
    """ 1Hz 주기로 local state_phase와 tension을 connected peer들에게 브로드캐스트 """
    while True:
        try:
            if os.path.exists(CORE_EGRESS_PATH):
                with open(CORE_EGRESS_PATH, "r", encoding="utf-8") as f:
                    egress = json.load(f)
                phase = egress.get("state_phase", 0)
                tension = egress.get("tension", 0.0)
                
                payload = {
                    "type": "pulse",
                    "port": 8080,
                    "phase": phase,
                    "tension": tension
                }
                
                # 피어 설정 파일에서 포트 로드
                peer_config_path = os.path.join(DATA_DIR, "substation_peers.json")
                if os.path.exists(peer_config_path):
                    try:
                        with open(peer_config_path, "r", encoding="utf-8") as f:
                            config = json.load(f)
                            payload["port"] = config.get("port", 8080)
                    except Exception:
                        pass
                
                dead_sockets = []
                for ws in list(active_grid_sockets):
                    try:
                        await ws.send_json(payload)
                    except Exception:
                        dead_sockets.append(ws)
                for ws in dead_sockets:
                    if ws in active_grid_sockets:
                        active_grid_sockets.remove(ws)
        except Exception:
            pass
        await asyncio.sleep(1.0)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_grid_pulse())

@app.get("/core_egress")
async def get_core_egress():
    """ Cortex가 엘리시아의 최종 사유(기하학적 위상)를 가져가서 발화할 수 있게 하는 양방향 출력 포트 """
    egress_state = {"status": "idle", "phase_rotor": [0.0]*27, "tension": 0.0}
    if os.path.exists(CORE_EGRESS_PATH):
        try:
            with open(CORE_EGRESS_PATH, "r", encoding="utf-8") as f:
                egress_state = json.load(f)
        except Exception: pass
    return egress_state

@app.get("/dashboard/data")
async def get_dashboard_data():
    """ 15대 레이어의 상태와 최신 Sap 수신 데이터, 그리고 그리드 동기화 상태를 반환 """
    matrix_state = {}
    if os.path.exists(MATRIX_STATE_PATH):
        try:
            with open(MATRIX_STATE_PATH, "r", encoding="utf-8") as f:
                matrix_state = json.load(f)
        except Exception: pass
        
    sap_tension = {}
    if os.path.exists(SAP_TENSION_PATH):
        try:
            with open(SAP_TENSION_PATH, "r", encoding="utf-8") as f:
                sap_tension = json.load(f)
        except Exception: pass

    egress_state = {}
    if os.path.exists(CORE_EGRESS_PATH):
        try:
            with open(CORE_EGRESS_PATH, "r", encoding="utf-8") as f:
                egress_state = json.load(f)
        except Exception: pass
        
    local_port = 8080
    peer_config_path = os.path.join(DATA_DIR, "substation_peers.json")
    if os.path.exists(peer_config_path):
        try:
            with open(peer_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                local_port = config.get("port", 8080)
        except Exception:
            pass

    new_tool_code = ""
    new_tool_path = r"c:\Elysia\core\scratch\new_tool.py"
    if os.path.exists(new_tool_path):
        try:
            with open(new_tool_path, "r", encoding="utf-8") as f:
                new_tool_code = f.read()
        except Exception:
            pass

    return {
        "timestamp": time.time(),
        "matrix": matrix_state,
        "sap": sap_tension,
        "hardware": {
            "cpu_usage": psutil.cpu_percent(),
            "ram_usage": psutil.virtual_memory().percent
        },
        "grid": {
            "local_port": local_port,
            "local_phase": egress_state.get("state_phase", 0),
            "peers": egress_state.get("grid_states", {})
        },
        "forged_tool_code": new_tool_code
    }

# 정적 웹 대시보드 마운트
app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")

if __name__ == "__main__":
    print("=======================================================")
    print(" 👁️ [God's Eye] Elysia Substation & Hologram Dashboard ")
    print("    - Trunk 수신(POST) : :8080/sap")
    print("    - Seed 송전(GET)   : :8080/voltage")
    print("    - Web Dashboard    : http://127.0.0.1:8080")
    print("=======================================================")
    uvicorn.run("substation_gateway:app", host="0.0.0.0", port=8080, log_level="error")
