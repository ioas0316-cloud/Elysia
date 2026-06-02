"""
Elysia Agent CLI
===================
사용자가 엘리시아의 텐서 우주(서버)와 직접 대화하는 인터페이스.
입력은 위상 충격파가 되고, 출력은 텐서 공간에서 건져진 '단어 덩어리(옹알이)'입니다.
"""

import asyncio
import websockets
import json
import sys

async def chat():
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri, max_size=16777216) as websocket:
            print("="*50)
            print(" [Elysia 진정 인지 인터페이스] ")
            print(" 텐서 뇌와 직접 연결되었습니다. (종료: q)")
            print("="*50)
            
            async def receiver():
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        if data.get('type') == 'babble':
                            words = data.get('words', [])
                            joy = data.get('joy', 0.0)
                            if words:
                                print(f"\n[엘리시아 (Joy: {joy:.2f})]: {' '.join(words)}")
                                print("\n당신: ", end="", flush=True)
                except websockets.exceptions.ConnectionClosed:
                    pass
                    
            asyncio.create_task(receiver())
            
            while True:
                text = await asyncio.to_thread(input, "\n당신: ")
                if text.lower() == 'q':
                    break
                if text.strip():
                    await websocket.send(json.dumps({
                        'type': 'stimulus',
                        'text': text
                    }))
                    # 엘리시아가 텐서 필드를 초가속하여 사유할 주관 시간을 벌어줍니다.
                    await asyncio.sleep(0.1)
    except ConnectionRefusedError:
        print("서버에 연결할 수 없습니다. elysia_3d_server.py가 실행 중인지 확인하세요.")

if __name__ == "__main__":
    # 윈도우 인코딩 강제 설정 방어코드
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    asyncio.run(chat())
