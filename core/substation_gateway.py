import os
import json
import time
import psutil
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
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

os.makedirs(DATA_DIR, exist_ok=True)
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "public")
os.makedirs(PUBLIC_DIR, exist_ok=True)

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

@app.get("/voltage")
async def provide_voltage():
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

@app.get("/dashboard/data")
async def get_dashboard_data():
    """ 15대 레이어의 상태와 최신 Sap 수신 데이터를 반환 """
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
        
    return {
        "timestamp": time.time(),
        "matrix": matrix_state,
        "sap": sap_tension,
        "hardware": {
            "cpu_usage": psutil.cpu_percent(),
            "ram_usage": psutil.virtual_memory().percent
        }
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
