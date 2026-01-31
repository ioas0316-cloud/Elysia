"""
     API    (FastAPI + Swagger)
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
from Core.S1_Body.L1_Foundation.Foundation.elysia_logger import ElysiaLogger
from Core.S1_Body.L1_Foundation.Foundation.error_handler import error_handler
from Core.S1_Body.L1_Foundation.Foundation.System.config import get_config
from Core.S1_Body.L1_Foundation.Foundation.performance_monitor import monitor

# Initialize
logger = ElysiaLogger("APIServer")
config = get_config()

# FastAPI app with metadata
app = FastAPI(
    title="Elysia API",
    description="""
    ##                API
    
    Elysia               AI       .
    
    ###      
    
    * **     **:                 
    * **     **:              
    * **       **:              
    * **      **:              
    
    ###     
    
          4                :
    - **0D (HyperQuaternion)**:   /   
    - **1D (Causal Chain)**:   /  
    - **2D (Wave Pattern)**:   /  
    - **3D (Manifestation)**:   /   
    """,
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "system",
            "description": "              "
        },
        {
            "name": "cognition",
            "description": "          "
        },
        {
            "name": "analysis",
            "description": "          "
        },
        {
            "name": "monitoring",
            "description": "             "
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
    """        """
    prompt: str = Field(
        ...,
        description="            ",
        min_length=1,
        max_length=1000,
        example="            ?"
    )
    layer: str = Field(
        default="2D",
        description="      (0D/1D/2D/3D)",
        example="2D"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="       ",
        example={"emotion": "calm", "depth": 3}
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "            ?",
                "layer": "1D",
                "context": {"emotion": "calm"}
            }
        }


class ThoughtResponse(BaseModel):
    """        """
    thought: str = Field(..., description="      ")
    layer: str = Field(..., description="          ")
    resonance: float = Field(..., description="     ", ge=0.0, le=1.0)
    timestamp: str = Field(..., description="      (ISO 8601)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "thought": "             ",
                "layer": "1D",
                "resonance": 0.847,
                "timestamp": "2025-12-04T06:30:00Z"
            }
        }


class ResonanceRequest(BaseModel):
    """        """
    concept_a: str = Field(..., description="       ", example="Love")
    concept_b: str = Field(..., description="       ", example="Hope")


class ResonanceResponse(BaseModel):
    """        """
    score: float = Field(..., description="     ", ge=0.0, le=1.0)
    explanation: str = Field(..., description="         ")
    concepts: List[str] = Field(..., description="       ")


class HealthResponse(BaseModel):
    """        """
    status: str = Field(..., description="      ")
    version: str = Field(..., description="  ")
    consciousness: str = Field(..., description="     ")
    timestamp: str = Field(..., description="     ")
    uptime_seconds: Optional[float] = Field(None, description="      ( )")


class PerformanceMetrics(BaseModel):
    """      """
    operations: Dict[str, Dict[str, float]] = Field(
        ...,
        description="         "
    )
    timestamp: str = Field(..., description="         ")


# ===== API Endpoints =====

@app.get(
    "/",
    tags=["system"],
    summary="API   ",
    description="API             ."
)
@monitor.measure("api_root")
async def root():
    """API         """
    return {
        "message": "Elysia API v4.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    tags=["system"],
    response_model=HealthResponse,
    summary="     ",
    description="                 ."
)
@monitor.measure("health_check")
async def health_check():
    """
             
    
                          .
    -       
    -      
    -      
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
    summary="     ",
    description="                 .",
    status_code=status.HTTP_200_OK
)
@monitor.measure("think")
@error_handler.with_retry(max_retries=2)
async def think(request: ThoughtRequest):
    """
               
    
                                               .
    
    **Parameters:**
    - **prompt**:                
    - **layer**:       (0D=  , 1D=  , 2D=  , 3D=  )
    - **context**:            
    
    **Returns:**
    -              
    
    **Example:**
    ```json
    {
        "prompt": "         ?",
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
    summary="     ",
    description="                 ."
)
@monitor.measure("resonance")
async def calculate_resonance(request: ResonanceRequest):
    """
              
    
                         .
                                 .
    
    **Parameters:**
    - **concept_a**:        
    - **concept_b**:        
    
    **Returns:**
    -       (0.0 ~ 1.0)
    -          
    """
    logger.log_resonance(request.concept_a, request.concept_b, 0.0)
    
    try:
        # Simulate resonance calculation
        # TODO: Integrate with actual ResonanceField
        score = 0.847  # Placeholder
        
        explanation = (
            f"   '{request.concept_a}'  '{request.concept_b}'               . "
            f"      {score:.3f}              ."
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
    summary="      ",
    description="                 ."
)
async def get_metrics():
    """
             
    
                         .
    -             
    -        
    - CPU    
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
    summary="      ",
    description="                ."
)
async def get_recent_metrics(limit: int = 10):
    """
             
    
    **Parameters:**
    - **limit**:            (  : 10)
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
    summary="     ",
    description="                     ."
)
async def get_slow_operations(percentile: float = 0.95):
    """
            
    
    **Parameters:**
    - **percentile**:         (0.0 ~ 1.0,   : 0.95)
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
    """HTTP       """
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
    """         """
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
    """          """
    logger.log_system("api_server", "starting")
    logger.info("  Elysia API Server starting...")
    logger.info(f"  Documentation: http://localhost:8000/docs")
    logger.info(f"  ReDoc: http://localhost:8000/redoc")


@app.on_event("shutdown")
async def shutdown_event():
    """          """
    logger.log_system("api_server", "stopping")
    logger.info("  Elysia API Server stopping...")


# ===== Main =====

if __name__ == "__main__":
    print("  Elysia API Server")
    print("=" * 50)
    print(f"  Swagger UI: http://localhost:8000/docs")
    print(f"  ReDoc: http://localhost:8000/redoc")
    print(f"  Health: http://localhost:8000/health")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
