from fastapi import FastAPI, Body, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import sys
import uuid
import mmap
import struct

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from engine import find_resonance_angle, generate_symbolic_regression, evaluate_current_state, elysia_auto_observe_step
from core.memory.causal_controller import CausalMemoryController
from core.consciousness.autonomous_loop import ConsciousnessLoop
from core.consciousness.resonance_tracker import ResonanceTracker

app = FastAPI(title="Elysia Consciousness Engine", version="18.0")

# ── 공유 컴포넌트 (서버 기동 시 1회 초기화) ─────────────────────────
_BASE_DIR   = os.path.join(os.path.dirname(__file__), '..', '..')
_DATA_DIR   = os.path.abspath(os.path.join(_BASE_DIR, 'data'))
_CORPUS_DIR = os.path.abspath(os.path.join(_BASE_DIR, 'docs'))

_memory_controller: Optional[CausalMemoryController] = None
_consciousness_loop: Optional[ConsciousnessLoop]     = None
_resonance_tracker: Optional[ResonanceTracker]       = None

def _get_loop() -> ConsciousnessLoop:
    """ConsciousnessLoop 싱글턴 (지연 초기화)."""
    global _memory_controller, _consciousness_loop, _resonance_tracker
    if _consciousness_loop is None:
        _memory_controller  = CausalMemoryController(data_dir=_DATA_DIR)
        _consciousness_loop = ConsciousnessLoop(
            corpus_path=_CORPUS_DIR,
            memory_controller=_memory_controller,
            data_dir=_DATA_DIR,
        )
        _resonance_tracker  = _consciousness_loop.tracker
    return _consciousness_loop

# Mount static files
_STATIC_DIR = os.path.join(os.path.dirname(__file__), '..', 'static')
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))

@app.get("/dashboard")
def read_dashboard():
    """엘리시아 실시간 상태 대시보드 (Legacy)"""
    return FileResponse(os.path.join(_STATIC_DIR, "dashboard.html"))

@app.get("/manifestation")
def read_manifestation():
    """엘리시아 통합 시각화 대시보드 (Phase 3)"""
    return FileResponse(os.path.join(_STATIC_DIR, "manifestation.html"))

INGEST_DIR = "../../data/ingest"
os.makedirs(INGEST_DIR, exist_ok=True)

@app.post("/api/init_field")
def init_movement_field(input_data: TextInput):
    """단순 텍스트를 파일로 떨구어 엘리시아가 자율 섭취하게 둡니다."""
    filename = os.path.join(INGEST_DIR, f"text_{uuid.uuid4().hex}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(input_data.text)
    return {"status": "success", "message": "Text dropped into ingest folder."}

class ImageInput(BaseModel):
    image_base64: str

@app.post("/api/init_image_field")
def init_image_movement_field(input_data: ImageInput):
    """이미지 데이터를 파일로 떨구어 엘리시아가 자율 섭취하게 둡니다."""
    filename = os.path.join(INGEST_DIR, f"image_{uuid.uuid4().hex}.dat")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(input_data.image_base64)
    return {"status": "success", "message": "Image dropped into ingest folder."}

@app.get("/api/observe_field")
def observe_field():
    """파이썬 연산 없이 공유 메모리의 순수 텐션(Tension)만을 관측하여 좌표로 반환합니다."""
    points = []
    try:
        shm = mmap.mmap(0, 1024 * 1024 * 16, tagname="Local\\ElysiaTopologyField", access=mmap.ACCESS_READ)
        header_size = 12
        max_rotors = (1024 * 1024 * 16 - header_size) // 8
        
        # 전체를 스캔하면 브라우저가 느려지므로 압력이 있는(텐션이 높은) 의미있는 로터만 추출
        for i in range(0, max_rotors, 100):
            offset = header_size + (i * 8)
            shm.seek(offset)
            rotor_data = shm.read(8)
            if len(rotor_data) == 8:
                math_t, lang_t, spatial_t, temporal_t, light_mass, byte_val, pad = struct.unpack('<BBBBHBB', rotor_data)
                
                # 순수 전위차(Tension)가 3D 공간의 X, Y, Z가 됩니다 (자연 매핑)
                if math_t > 0 or lang_t > 0 or spatial_t > 0:
                    points.append({
                        "position": [math_t / 10.0, lang_t / 10.0, spatial_t / 10.0],
                        "velocity": [0, 0, 0], # 인위적 속도 없음
                        "phase": temporal_t / 255.0 * 3.14,
                        "token": chr(byte_val) if 32 <= byte_val <= 126 else "*"
                    })
        shm.close()
    except Exception as e:
        print("Mmap observe error:", e)
        pass
    
    return {"status": "success", "data": points}

class AlignRequest(BaseModel):
    points_data: list
    time_t: float

@app.post("/api/auto_align")
def auto_align_field(request_data: AlignRequest):
    """현재 점들의 상태를 바탕으로 공명 각도를 찾고 수식을 반환"""
    best_quat, min_var = find_resonance_angle(request_data.points_data, request_data.time_t)
    formula, r_squared = generate_symbolic_regression(request_data.points_data, best_quat, request_data.time_t)

    return {
        "status": "success",
        "quaternion": best_quat,
        "variance": min_var,
        "formula": formula,
        "r_squared": r_squared
    }


class EvaluateRequest(BaseModel):
    points_data: list
    time_t: float
    quaternion: list

@app.post("/api/evaluate_state")
def evaluate_state(request_data: EvaluateRequest):
    """현재 사용자의 조이스틱(쿼터니언)과 시간 상태를 바탕으로 공명 여부 평가"""
    variance, is_resonant, formula = evaluate_current_state(
        request_data.points_data,
        request_data.quaternion,
        request_data.time_t
    )

    return {
        "status": "success",
        "variance": variance,
        "is_resonant": is_resonant,
        "formula": formula
    }


class AutoObserveRequest(BaseModel):
    points_data: list
    time_t: float

@app.post("/api/auto_observe")
def auto_observe(request_data: AutoObserveRequest):
    """엘리시아가 스스로 쿼터니언을 조절하며 공명을 찾는 자율 관측 스텝"""
    next_quat, variance, is_resonant, formula = elysia_auto_observe_step(
        request_data.points_data,
        request_data.time_t
    )

    return {
        "status": "success",
        "quaternion": next_quat,
        "variance": variance,
        "is_resonant": is_resonant,
        "formula": formula
    }

# ══════════════════════════════════════════════════════════════════
# 의식 루프 제어 엔드포인트
# ══════════════════════════════════════════════════════════════════

@app.get("/api/status")
def get_status():
    """
    엘리시아의 현재 전체 상태 스냅샷.
    건강 요약 + 인지 파라미터 + 누적 통계.
    """
    loop = _get_loop()
    health = loop.tracker.get_health_summary()
    params = loop.memory.cognitive_params.copy()
    return {
        "status": "ok",
        "health": health,
        "cognitive_params": params,
        "session_crystals": loop.crystals_formed,
        "session_cycles":   loop.cycle_count,
        "log_path":         loop.tracker.get_log_path(),
    }


@app.get("/api/resonance_log")
def get_resonance_log(n: int = 100):
    """
    최근 N개 사이클의 공명/텐션 트렌드 데이터.
    대시보드 차트용.
    """
    if n < 1 or n > 2000:
        raise HTTPException(status_code=400, detail="n must be between 1 and 2000")
    loop   = _get_loop()
    trend  = loop.tracker.get_trend(n=n)
    health = loop.tracker.get_health_summary()
    return {
        "status": "ok",
        "count":  len(trend),
        "trend":  trend,
        "health_summary": {
            "emotional_state":  health["emotional_state"],
            "resonance_rate":   health["resonance_rate"],
            "avg_tension":      health["avg_tension"],
            "total_cycles":     health["total_cycles"],
            "crystals":         loop.crystals_formed,
        },
    }


@app.post("/api/run_cycle")
def run_single_cycle():
    """
    의식 사이클 1회 수동 실행.
    즉시 해당 사이클의 결과를 반환합니다.
    """
    loop   = _get_loop()
    result = loop.process_life_cycle()
    loop.memory.flush_index()
    return {
        "status": "ok",
        "cycle":  result,
        "health_snapshot": loop.tracker.get_health_summary(),
    }


@app.post("/api/run_loop/{n}")
def run_loop_n(n: int):
    """
    N회 의식 사이클 연속 실행.
    완료 후 전체 요약을 반환합니다.
    """
    if n < 1 or n > 500:
        raise HTTPException(status_code=400, detail="n must be between 1 and 500")
    loop    = _get_loop()
    summary = loop.run(cycles=n, verbose=False)
    return {
        "status":  "ok",
        "summary": summary,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
