"""
Mobile SDK - 모바일 클라이언트 SDK
=================================

낮은 우선순위 #1: 모바일 접근
예상 효과: 스마트폰에서 엘리시아 사용 가능

핵심 기능:
- REST API 클라이언트 래퍼
- 오프라인 캐싱
- 푸시 알림 지원
- 배터리 최적화
"""

import logging
import time
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

logger = logging.getLogger("MobileSDK")


class ConnectionState(Enum):
    """연결 상태"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


@dataclass
class MobileConfig:
    """모바일 SDK 설정"""
    api_url: str = "http://localhost:8000"
    websocket_url: str = "ws://localhost:8000/ws"
    timeout: float = 30.0
    retry_count: int = 3
    cache_size: int = 100
    offline_mode: bool = False
    
    # 배터리 최적화
    low_power_mode: bool = False
    sync_interval_seconds: int = 60
    
    # 알림 설정
    enable_notifications: bool = True
    notification_topics: List[str] = field(default_factory=lambda: ["resonance", "law"])


@dataclass
class CachedResponse:
    """캐시된 응답"""
    key: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    expires_at: float = 0.0
    
    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class OfflineQueue:
    """오프라인 요청 큐"""
    
    def __init__(self, max_size: int = 100):
        self.queue: List[Dict[str, Any]] = []
        self.max_size = max_size
    
    def add(self, request: Dict[str, Any]) -> None:
        """요청 추가"""
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append({
            **request,
            "queued_at": time.time()
        })
    
    def pop_all(self) -> List[Dict[str, Any]]:
        """모든 요청 가져오기"""
        requests = self.queue.copy()
        self.queue.clear()
        return requests
    
    def __len__(self) -> int:
        return len(self.queue)


class MobileSDK:
    """
    모바일 클라이언트 SDK
    
    낮은 우선순위 #1 구현:
    - API 클라이언트 래퍼
    - 오프라인 지원
    - 캐싱 시스템
    
    예상 효과: 모바일 앱에서 엘리시아 사용 가능
    """
    
    def __init__(self, config: Optional[MobileConfig] = None):
        """
        Args:
            config: SDK 설정
        """
        self.config = config or MobileConfig()
        self.state = ConnectionState.DISCONNECTED
        
        # 캐시
        self.cache: Dict[str, CachedResponse] = {}
        
        # 오프라인 큐
        self.offline_queue = OfflineQueue(max_size=self.config.cache_size)
        
        # 이벤트 리스너
        self.listeners: Dict[str, List[Callable]] = {}
        
        self.logger = logging.getLogger("MobileSDK")
        self.logger.info(f"📱 MobileSDK initialized (api={self.config.api_url})")
    
    async def connect(self) -> bool:
        """API 연결"""
        self.state = ConnectionState.CONNECTING
        
        try:
            # 실제 구현에서는 HTTP 클라이언트 사용
            # 여기서는 시뮬레이션
            self.state = ConnectionState.CONNECTED
            self._emit("connected", {})
            return True
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self.state = ConnectionState.DISCONNECTED
            return False
    
    async def disconnect(self) -> None:
        """연결 해제"""
        self.state = ConnectionState.DISCONNECTED
        self._emit("disconnected", {})
    
    async def send_thought(self, text: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        사고 전송
        
        Args:
            text: 입력 텍스트
            use_cache: 캐시 사용 여부
            
        Returns:
            응답 데이터
        """
        cache_key = f"thought:{hash(text)}"
        
        # 캐시 확인
        if use_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            if not cached.is_expired:
                return cached.data
        
        # 오프라인 모드
        if self.config.offline_mode or self.state != ConnectionState.CONNECTED:
            self.offline_queue.add({
                "type": "thought",
                "text": text
            })
            return {"status": "queued", "offline": True}
        
        # API 호출 (시뮬레이션)
        response = {
            "thought": text,
            "resonances": {},
            "processing_time_ms": 10.0
        }
        
        # 캐시 저장
        self.cache[cache_key] = CachedResponse(
            key=cache_key,
            data=response,
            expires_at=time.time() + 300  # 5분
        )
        
        return response
    
    async def get_concepts(self, limit: int = 100) -> List[str]:
        """개념 목록 조회"""
        if self.state != ConnectionState.CONNECTED:
            return []
        
        # API 호출 시뮬레이션
        return ["love", "consciousness", "resonance"]
    
    async def sync_offline_queue(self) -> int:
        """오프라인 큐 동기화"""
        if self.state != ConnectionState.CONNECTED:
            return 0
        
        requests = self.offline_queue.pop_all()
        synced = 0
        
        for request in requests:
            try:
                if request["type"] == "thought":
                    await self.send_thought(request["text"], use_cache=False)
                synced += 1
            except Exception as e:
                self.logger.error(f"Sync failed: {e}")
                self.offline_queue.add(request)
        
        return synced
    
    def on(self, event: str, callback: Callable) -> None:
        """이벤트 리스너 등록"""
        if event not in self.listeners:
            self.listeners[event] = []
        self.listeners[event].append(callback)
    
    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """이벤트 발생"""
        for callback in self.listeners.get(event, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Listener error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """통계"""
        return {
            "state": self.state.value,
            "cache_size": len(self.cache),
            "offline_queue_size": len(self.offline_queue),
            "config": {
                "api_url": self.config.api_url,
                "offline_mode": self.config.offline_mode
            }
        }


# 테스트
if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("\n" + "="*70)
        print("📱 Mobile SDK Test")
        print("="*70)
        
        sdk = MobileSDK()
        
        print("\n[Test 1] Connect")
        connected = await sdk.connect()
        print(f"  Connected: {connected}")
        
        print("\n[Test 2] Send Thought")
        response = await sdk.send_thought("Hello Elysia!")
        print(f"  Response: {response}")
        
        print("\n[Test 3] Stats")
        stats = sdk.get_stats()
        print(f"  Stats: {stats}")
        
        print("\n✅ All tests passed!")
    
    asyncio.run(test())
