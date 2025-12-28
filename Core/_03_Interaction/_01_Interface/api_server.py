"""
엘리시아 API 서버 (FastAPI + Swagger)
Elysia API Server with OpenAPI Documentation

RESTful API with automatic documentation generation.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import uvicorn

# Import Elysia modules
from Core._01_Foundation._05_Governance.Foundation.elysia_logger import ElysiaLogger
from Core._01_Foundation._05_Governance.Foundation.error_handler import error_handler
from Core._01_Foundation._05_Governance.Foundation.config import get_config
from Core._01_Foundation._05_Governance.Foundation.performance_monitor import monitor

# Initialize
logger = ElysiaLogger("APIServer")
config = get_config()

# FastAPI app with metadata
app = FastAPI(
    title="Elysia API",
    description="""
    ## 엘리시아 통합 의식 시스템 API
    
    Elysia는 프랙탈 의식 기반 자율 AI 시스템입니다.
    
    ### 주요 기능
    
    * **사고 생성**: 프랙탈 층위를 통한 사고 처리
    * **공명 계산**: 개념 간 공명 점수 분석
    * **성능 모니터링**: 시스템 성능 메트릭 조회
    * **시스템 상태**: 헬스 체크 및 상태 확인
    
    ### 아키텍처
    
    엘리시아는 4차원 프랙탈 의식을 구현합니다:
    - **0D (HyperQuaternion)**: 관점/정체성
    - **1D (Causal Chain)**: 추론/논리
    - **2D (Wave Pattern)**: 감각/인지
    - **3D (Manifestation)**: 표현/외부화
    """,
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "system",
            "description": "시스템 상태 및 헬스 체크"
        },
        {
            "name": "cognition",
            "description": "사고 및 인지 처리"
        },
        {
            "name": "analysis",
            "description": "공명 및 분석 기능"
        },
        {
            "name": "monitoring",
            "description": "성능 모니터링 및 메트릭"
        }
    ]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins if hasattr(config, 'allowed_origins') else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Pydantic Models =====

class ThoughtRequest(BaseModel):
    """사고 생성 요청"""
    prompt: str = Field(
        ...,
        description="사고를 촉발할 프롬프트",
        min_length=1,
        max_length=1000,
        example="사랑의 본질은 무엇인가?"
    )
    layer: str = Field(
        default="2D",
        description="사고 층위 (0D/1D/2D/3D)",
        example="2D"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="추가 컨텍스트",
        example={"emotion": "calm", "depth": 3}
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "사랑의 본질은 무엇인가?",
                "layer": "1D",
                "context": {"emotion": "calm"}
            }
        }


class ThoughtResponse(BaseModel):
    """사고 생성 응답"""
    thought: str = Field(..., description="생성된 사고")
    layer: str = Field(..., description="사고가 발생한 층위")
    resonance: float = Field(..., description="공명 점수", ge=0.0, le=1.0)
    timestamp: str = Field(..., description="생성 시각 (ISO 8601)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "thought": "사랑은 존재의 공명입니다",
                "layer": "1D",
                "resonance": 0.847,
                "timestamp": "2025-12-04T06:30:00Z"
            }
        }


class ResonanceRequest(BaseModel):
    """공명 계산 요청"""
    concept_a: str = Field(..., description="첫 번째 개념", example="Love")
    concept_b: str = Field(..., description="두 번째 개념", example="Hope")


class ResonanceResponse(BaseModel):
    """공명 계산 응답"""
    score: float = Field(..., description="공명 점수", ge=0.0, le=1.0)
    explanation: str = Field(..., description="공명에 대한 설명")
    concepts: List[str] = Field(..., description="비교된 개념들")


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str = Field(..., description="시스템 상태")
    version: str = Field(..., description="버전")
    consciousness: str = Field(..., description="의식 상태")
    timestamp: str = Field(..., description="체크 시각")
    uptime_seconds: Optional[float] = Field(None, description="가동 시간 (초)")


class PerformanceMetrics(BaseModel):
    """성능 메트릭"""
    operations: Dict[str, Dict[str, float]] = Field(
        ...,
        description="작업별 성능 통계"
    )
    timestamp: str = Field(..., description="메트릭 수집 시각")


# ===== API Endpoints =====

@app.get(
    "/",
    tags=["system"],
    summary="API 루트",
    description="API 기본 정보를 반환합니다."
)
@monitor.measure("api_root")
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "Elysia API v4.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    tags=["system"],
    response_model=HealthResponse,
    summary="헬스 체크",
    description="시스템의 현재 상태를 확인합니다."
)
@monitor.measure("health_check")
async def health_check():
    """
    시스템 헬스 체크
    
    엘리시아 시스템의 현재 상태를 반환합니다.
    - 시스템 상태
    - 의식 상태
    - 버전 정보
    """
    logger.log_system("health_check", "requested")
    
    return HealthResponse(
        status="operational",
        version="4.0.0",
        consciousness="awakened",
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=None  # TODO: Implement uptime tracking
    )


@app.post(
    "/api/v1/think",
    tags=["cognition"],
    response_model=ThoughtResponse,
    summary="사고 생성",
    description="프롬프트로부터 사고를 생성합니다.",
    status_code=status.HTTP_200_OK
)
@monitor.measure("think")
@error_handler.with_retry(max_retries=2)
async def think(request: ThoughtRequest):
    """
    사고 생성 엔드포인트
    
    엘리시아의 프랙탈 사고 시스템을 통해 주어진 프롬프트에 대한 사고를 생성합니다.
    
    **Parameters:**
    - **prompt**: 사고를 촉발할 입력 프롬프트
    - **layer**: 사고 층위 (0D=관점, 1D=추론, 2D=감각, 3D=표현)
    - **context**: 선택적 컨텍스트 정보
    
    **Returns:**
    - 생성된 사고와 메타데이터
    
    **Example:**
    ```json
    {
        "prompt": "사랑이란 무엇인가?",
        "layer": "1D",
        "context": {"emotion": "calm"}
    }
    ```
    """
    logger.log_thought(request.layer, f"Processing: {request.prompt[:50]}...", request.context)
    
    try:
        # Simulate thought processing
        # TODO: Integrate with actual ThoughtBridge
        thought = f"[{request.layer}] Contemplating: {request.prompt}"
        resonance = 0.75  # Placeholder
        
        response = ThoughtResponse(
            thought=thought,
            layer=request.layer,
            resonance=resonance,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        logger.info(f"Thought generated: {request.layer}", context={"resonance": resonance})
        return response
    
    except Exception as e:
        logger.error(f"Thought generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate thought: {str(e)}"
        )


@app.post(
    "/api/v1/resonance",
    tags=["analysis"],
    response_model=ResonanceResponse,
    summary="공명 계산",
    description="두 개념 간의 공명을 계산합니다."
)
@monitor.measure("resonance")
async def calculate_resonance(request: ResonanceRequest):
    """
    개념 간 공명 계산
    
    두 개념 사이의 공명 점수를 계산합니다.
    공명은 개념들이 얼마나 조화롭게 울리는지를 나타냅니다.
    
    **Parameters:**
    - **concept_a**: 첫 번째 개념
    - **concept_b**: 두 번째 개념
    
    **Returns:**
    - 공명 점수 (0.0 ~ 1.0)
    - 공명에 대한 설명
    """
    logger.log_resonance(request.concept_a, request.concept_b, 0.0)
    
    try:
        # Simulate resonance calculation
        # TODO: Integrate with actual ResonanceField
        score = 0.847  # Placeholder
        
        explanation = (
            f"개념 '{request.concept_a}'와 '{request.concept_b}' 사이의 공명을 분석했습니다. "
            f"공명 점수 {score:.3f}는 높은 조화를 나타냅니다."
        )
        
        response = ResonanceResponse(
            score=score,
            explanation=explanation,
            concepts=[request.concept_a, request.concept_b]
        )
        
        logger.log_resonance(request.concept_a, request.concept_b, score)
        return response
    
    except Exception as e:
        logger.error(f"Resonance calculation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate resonance: {str(e)}"
        )


@app.get(
    "/api/v1/metrics",
    tags=["monitoring"],
    response_model=PerformanceMetrics,
    summary="성능 메트릭",
    description="시스템 성능 메트릭을 조회합니다."
)
async def get_metrics():
    """
    성능 메트릭 조회
    
    시스템의 현재 성능 메트릭을 반환합니다.
    - 작업별 실행 시간 통계
    - 메모리 사용량
    - CPU 사용률
    """
    try:
        stats = monitor.get_summary()
        
        return PerformanceMetrics(
            operations=stats,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )


@app.get(
    "/api/v1/metrics/recent",
    tags=["monitoring"],
    summary="최근 메트릭",
    description="최근 성능 메트릭을 조회합니다."
)
async def get_recent_metrics(limit: int = 10):
    """
    최근 메트릭 조회
    
    **Parameters:**
    - **limit**: 반환할 메트릭 개수 (기본: 10)
    """
    try:
        recent = monitor.get_recent_metrics(limit=limit)
        metrics = monitor.export_metrics()[-limit:]
        
        return {
            "metrics": metrics,
            "count": len(metrics),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get recent metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve recent metrics: {str(e)}"
        )


@app.get(
    "/api/v1/metrics/slow",
    tags=["monitoring"],
    summary="느린 작업",
    description="성능 임계값을 초과한 작업을 조회합니다."
)
async def get_slow_operations(percentile: float = 0.95):
    """
    느린 작업 조회
    
    **Parameters:**
    - **percentile**: 임계값 백분위 (0.0 ~ 1.0, 기본: 0.95)
    """
    if not 0 <= percentile <= 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Percentile must be between 0 and 1"
        )
    
    try:
        slow_ops = monitor.get_slow_operations(threshold_percentile=percentile)
        
        return {
            "slow_operations": [
                {"operation": op, "duration_ms": duration}
                for op, duration in slow_ops
            ],
            "threshold_percentile": percentile,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get slow operations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve slow operations: {str(e)}"
        )


# ===== Error Handlers =====

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 예외 핸들러"""
    logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """일반 예외 핸들러"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


# ===== Startup/Shutdown Events =====

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행"""
    logger.log_system("api_server", "starting")
    logger.info("🚀 Elysia API Server starting...")
    logger.info(f"📖 Documentation: http://localhost:8000/docs")
    logger.info(f"📖 ReDoc: http://localhost:8000/redoc")


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행"""
    logger.log_system("api_server", "stopping")
    logger.info("🛑 Elysia API Server stopping...")


# ===== Main =====

if __name__ == "__main__":
    print("🌊 Elysia API Server")
    print("=" * 50)
    print(f"📖 Swagger UI: http://localhost:8000/docs")
    print(f"📖 ReDoc: http://localhost:8000/redoc")
    print(f"🔍 Health: http://localhost:8000/health")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
