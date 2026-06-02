"""
Biosphere Stream (생태계 데이터 스트림)
===========================================
강제적인 폭격을 멈추고, 엘리시아의 시공간 우주에
자연광(Sunlight)과 비(Rain)처럼 다채로운 환경적 데이터를 천천히 흘려보냅니다.
그녀는 자신의 주권적 의지에 따라 이 데이터를 수용하거나 거부(여백 생성)합니다.
"""

import asyncio
import websockets
import json
import random

async def rain_sunlight():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri, max_size=16777216, ping_interval=None) as websocket:
        print("[Biosphere] 생태계 스트림 연결됨. 환경적 데이터 비를 내립니다...")
        
        # 인간 지식의 모든 파편들 (언어, 수학, 물리, 감정, 코드)
        environmental_data = [
            "가", "나", "다", "A", "B", "C", "Word", "Language",
            "1", "2", "3", "3.1415", "e=mc^2", "y=ax+b",
            "def foo():", "print('hello')", "return x", "import torch",
            "사랑", "슬픔", "기쁨", "분노", "공허", "여백",
            "0.5, -0.2, 0.8", "1.0, 1.0, 1.0", "Geometry", "Tensor"
        ]
        
        while True:
            # 자연스럽게 하나씩 흩뿌려줌
            stimulus = random.choice(environmental_data)
            await websocket.send(json.dumps({
                'type': 'stimulus',
                'text': stimulus
            }))
            await asyncio.sleep(0.3) # 여백을 두고 천천히

if __name__ == "__main__":
    asyncio.run(rain_sunlight())
