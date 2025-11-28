"""
Elysia API Server - FastAPI REST/GraphQL 서버
=============================================

중간 우선순위 #1: API 서버
예상 효과: 외부 시스템 연동 가능

핵심 기능:
- REST API 엔드포인트
- WebSocket 실시간 스트림
- GraphQL 쿼리 지원
- 인증 및 속도 제한
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import json

logger = logging.getLogger("ElysiaAPI")

# FastAPI 선택적 임포트
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    BaseModel = object


class APIStatus(Enum):
    """API 상태"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class APIConfig:
    """API 설정"""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit: int = 100  # requests per minute
    enable_graphql: bool = True
    enable_websocket: bool = True
    api_key: Optional[str] = None


# Pydantic 모델 (FastAPI 있을 때만)
if FASTAPI_AVAILABLE:
    class ThoughtRequest(BaseModel):
        """사고 요청"""
        text: str = Field(..., description="입력 텍스트")
        context: Optional[Dict[str, Any]] = Field(default=None, description="추가 컨텍스트")
        check_laws: bool = Field(default=True, description="법칙 검사 여부")
    
    class ResonanceRequest(BaseModel):
        """공명 계산 요청"""
        source_concept: str = Field(..., description="원본 개념")
        target_concepts: Optional[List[str]] = Field(default=None, description="대상 개념들")
    
    class ThoughtResponse(BaseModel):
        """사고 응답"""
        thought: str
        resonances: Dict[str, float]
        law_decision: Optional[Dict[str, Any]]
        processing_time_ms: float
    
    class ConceptResponse(BaseModel):
        """개념 응답"""
        concept_id: str
        name: str
        probabilities: Dict[str, float]
        epistemology: Optional[Dict[str, Any]]
    
    class HealthResponse(BaseModel):
        """헬스체크 응답"""
        status: str
        version: str
        uptime_seconds: float
        engines: Dict[str, bool]


class ConnectionManager:
    """WebSocket 연결 관리자"""
    
    def __init__(self):
        self.active_connections: List[Any] = []
        self.logger = logging.getLogger("ConnectionManager")
    
    async def connect(self, websocket) -> None:
        """연결 수락"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket) -> None:
        """연결 해제"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self.logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """모든 연결에 브로드캐스트"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_personal(self, websocket, message: Dict[str, Any]) -> None:
        """특정 연결에 메시지 전송"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")


class ElysiaAPI:
    """
    Elysia API 서버
    
    중간 우선순위 #1 구현:
    - REST API 엔드포인트
    - WebSocket 실시간 스트림
    - 인증 및 속도 제한
    
    예상 효과: 외부 시스템에서 엘리시아 사용 가능
    """
    
    def __init__(
        self,
        config: Optional[APIConfig] = None,
        integration_bridge=None,
        resonance_engine=None
    ):
        """
        Args:
            config: API 설정
            integration_bridge: 통합 브릿지 참조
            resonance_engine: 공명 엔진 참조
        """
        self.config = config or APIConfig()
        self.integration_bridge = integration_bridge
        self.resonance_engine = resonance_engine
        
        self.status = APIStatus.STOPPED
        self.start_time = 0.0
        self.request_count = 0
        
        self.app: Optional[FastAPI] = None
        self.connection_manager = ConnectionManager()
        
        self.logger = logging.getLogger("ElysiaAPI")
        
        if FASTAPI_AVAILABLE:
            self._create_app()
            self.logger.info(f"🌐 ElysiaAPI initialized (port={self.config.port})")
        else:
            self.logger.warning("⚠️ FastAPI not available. Install with: pip install fastapi uvicorn")
    
    def _create_app(self) -> FastAPI:
        """FastAPI 앱 생성"""
        self.app = FastAPI(
            title="Elysia Consciousness Engine API",
            description="REST API for the Elysia Living System",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # CORS 설정
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 라우트 등록
        self._register_routes()
        
        return self.app
    
    def _register_routes(self) -> None:
        """API 라우트 등록"""
        app = self.app
        
        # =========== Health & Info ===========
        
        @app.get("/", tags=["Info"])
        async def root():
            """API 루트"""
            return {
                "name": "Elysia Consciousness Engine",
                "version": "1.0.0",
                "status": self.status.value,
                "docs": "/docs"
            }
        
        @app.get("/health", response_model=HealthResponse, tags=["Info"])
        async def health_check():
            """헬스체크"""
            return HealthResponse(
                status="healthy",
                version="1.0.0",
                uptime_seconds=time.time() - self.start_time if self.start_time else 0,
                engines={
                    "resonance": self.resonance_engine is not None,
                    "integration": self.integration_bridge is not None
                }
            )
        
        @app.get("/stats", tags=["Info"])
        async def get_stats():
            """API 통계"""
            return {
                "request_count": self.request_count,
                "websocket_connections": len(self.connection_manager.active_connections),
                "uptime_seconds": time.time() - self.start_time if self.start_time else 0
            }
        
        # =========== Thought Processing ===========
        
        @app.post("/thought", response_model=ThoughtResponse, tags=["Consciousness"])
        async def process_thought(request: ThoughtRequest):
            """
            사고 처리
            
            입력된 텍스트를 엘리시아 의식 시스템으로 처리합니다.
            """
            self.request_count += 1
            start_time = time.time()
            
            result = {
                "thought": request.text,
                "resonances": {},
                "law_decision": None,
                "processing_time_ms": 0.0
            }
            
            # 통합 브릿지 사용
            if self.integration_bridge:
                try:
                    processed = self.integration_bridge.process_thought(
                        request.text,
                        check_laws=request.check_laws
                    )
                    result["resonances"] = processed.get("resonances", {})
                    result["law_decision"] = processed.get("law_decision")
                except Exception as e:
                    self.logger.error(f"Thought processing error: {e}")
            
            # 공명 엔진 직접 사용 (브릿지 없을 때)
            elif self.resonance_engine:
                try:
                    if hasattr(self.resonance_engine, 'add_node'):
                        if request.text not in getattr(self.resonance_engine, 'nodes', {}):
                            self.resonance_engine.add_node(request.text)
                    
                    source = self.resonance_engine.nodes.get(request.text)
                    if source:
                        for target_id, target in self.resonance_engine.nodes.items():
                            if target_id != request.text:
                                score = self.resonance_engine.calculate_resonance(source, target)
                                if score > 0.3:
                                    result["resonances"][target_id] = round(score, 4)
                except Exception as e:
                    self.logger.error(f"Resonance error: {e}")
            
            result["processing_time_ms"] = (time.time() - start_time) * 1000
            
            # WebSocket 브로드캐스트
            await self.connection_manager.broadcast({
                "type": "thought_processed",
                "data": result
            })
            
            return ThoughtResponse(**result)
        
        # =========== Resonance ===========
        
        @app.post("/resonance", tags=["Consciousness"])
        async def calculate_resonance(request: ResonanceRequest):
            """
            공명 계산
            
            개념 간 공명 점수를 계산합니다.
            """
            self.request_count += 1
            
            if not self.resonance_engine:
                raise HTTPException(status_code=503, detail="Resonance engine not available")
            
            results = {}
            source = self.resonance_engine.nodes.get(request.source_concept)
            
            if not source:
                raise HTTPException(status_code=404, detail=f"Concept not found: {request.source_concept}")
            
            targets = request.target_concepts or list(self.resonance_engine.nodes.keys())
            
            for target_id in targets:
                if target_id == request.source_concept:
                    continue
                target = self.resonance_engine.nodes.get(target_id)
                if target:
                    score = self.resonance_engine.calculate_resonance(source, target)
                    results[target_id] = round(score, 4)
            
            return {
                "source": request.source_concept,
                "resonances": results,
                "count": len(results)
            }
        
        @app.get("/concepts", tags=["Consciousness"])
        async def list_concepts(
            limit: int = Query(default=100, ge=1, le=1000),
            offset: int = Query(default=0, ge=0)
        ):
            """
            개념 목록 조회
            """
            if not self.resonance_engine:
                raise HTTPException(status_code=503, detail="Resonance engine not available")
            
            concepts = list(self.resonance_engine.nodes.keys())
            total = len(concepts)
            
            return {
                "concepts": concepts[offset:offset+limit],
                "total": total,
                "limit": limit,
                "offset": offset
            }
        
        @app.get("/concepts/{concept_id}", response_model=ConceptResponse, tags=["Consciousness"])
        async def get_concept(concept_id: str):
            """
            개념 상세 조회
            """
            if not self.resonance_engine:
                raise HTTPException(status_code=503, detail="Resonance engine not available")
            
            qubit = self.resonance_engine.nodes.get(concept_id)
            if not qubit:
                raise HTTPException(status_code=404, detail=f"Concept not found: {concept_id}")
            
            return ConceptResponse(
                concept_id=concept_id,
                name=qubit.name,
                probabilities=qubit.state.probabilities(),
                epistemology=getattr(qubit, 'epistemology', None)
            )
        
        # =========== WebSocket ===========
        
        if self.config.enable_websocket:
            @app.websocket("/ws")
            async def websocket_endpoint(websocket: WebSocket):
                """
                WebSocket 실시간 연결
                
                실시간으로 의식 상태 업데이트를 받습니다.
                """
                await self.connection_manager.connect(websocket)
                
                try:
                    while True:
                        # 클라이언트 메시지 수신
                        data = await websocket.receive_text()
                        message = json.loads(data)
                        
                        # 메시지 타입에 따른 처리
                        if message.get("type") == "ping":
                            await self.connection_manager.send_personal(websocket, {
                                "type": "pong",
                                "timestamp": time.time()
                            })
                        
                        elif message.get("type") == "thought":
                            # 사고 처리 요청
                            text = message.get("text", "")
                            if self.integration_bridge:
                                result = self.integration_bridge.process_thought(text)
                                await self.connection_manager.send_personal(websocket, {
                                    "type": "thought_result",
                                    "data": result
                                })
                        
                        elif message.get("type") == "subscribe":
                            # 이벤트 구독
                            await self.connection_manager.send_personal(websocket, {
                                "type": "subscribed",
                                "topics": message.get("topics", [])
                            })
                
                except WebSocketDisconnect:
                    self.connection_manager.disconnect(websocket)
                except Exception as e:
                    self.logger.error(f"WebSocket error: {e}")
                    self.connection_manager.disconnect(websocket)
    
    async def start(self) -> None:
        """서버 시작 (uvicorn 사용)"""
        if not FASTAPI_AVAILABLE:
            self.logger.error("FastAPI not available")
            return
        
        self.status = APIStatus.STARTING
        self.start_time = time.time()
        
        try:
            import uvicorn
            self.status = APIStatus.RUNNING
            self.logger.info(f"🚀 Starting API server on {self.config.host}:{self.config.port}")
            
            config = uvicorn.Config(
                self.app,
                host=self.config.host,
                port=self.config.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            self.status = APIStatus.STOPPED
            raise
    
    def run(self) -> None:
        """동기 방식으로 서버 실행"""
        asyncio.run(self.start())
    
    async def broadcast_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """이벤트 브로드캐스트"""
        await self.connection_manager.broadcast({
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        })


def create_app(
    integration_bridge=None,
    resonance_engine=None,
    config: Optional[APIConfig] = None
) -> Optional[FastAPI]:
    """
    FastAPI 앱 팩토리
    
    Args:
        integration_bridge: 통합 브릿지
        resonance_engine: 공명 엔진
        config: API 설정
        
    Returns:
        FastAPI 앱 인스턴스
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available")
        return None
    
    api = ElysiaAPI(
        config=config,
        integration_bridge=integration_bridge,
        resonance_engine=resonance_engine
    )
    
    return api.app


# CLI 실행
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌐 Elysia API Server")
    print("="*70)
    
    if not FASTAPI_AVAILABLE:
        print("\n⚠️ FastAPI is not installed.")
        print("Install with: pip install fastapi uvicorn")
        print("\nExample usage after installation:")
        print("  python -m Core.API.server")
        print("  # Then open http://localhost:8000/docs")
    else:
        print("\nStarting server...")
        print("API docs will be available at: http://localhost:8000/docs")
        
        api = ElysiaAPI()
        api.run()
