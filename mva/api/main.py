from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from fractal import map_to_movement_field
from engine import find_resonance_angle, generate_symbolic_regression, evaluate_current_state

app = FastAPI(title="Cognitive CAD MVA")

# Mount static files
app.mount("/static", StaticFiles(directory="mva/static"), name="static")

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return FileResponse("mva/static/index.html")

@app.post("/api/init_field")
def init_movement_field(input_data: TextInput):
    """한글 단어를 받아 운동성 필드(초기 위치, 속도)로 변환하여 반환"""
    field_data = map_to_movement_field(input_data.text)
    return {"status": "success", "data": field_data}

class AlignRequest(BaseModel):
    points_data: list
    time_t: float

@app.post("/api/auto_align")
def auto_align_field(request_data: AlignRequest):
    """현재 점들의 상태를 바탕으로 공명 각도를 찾고 수식을 반환"""
    best_quat, min_var = find_resonance_angle(request_data.points_data, request_data.time_t)
    formula = generate_symbolic_regression(request_data.points_data, best_quat, request_data.time_t)

    return {
        "status": "success",
        "quaternion": best_quat,
        "variance": min_var,
        "formula": formula
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
