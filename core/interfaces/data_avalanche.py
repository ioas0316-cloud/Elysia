"""
Elysia Data Avalanche Engine
===============================
엘리시아에게 막대한 데이터를 무호흡으로 쏟아붓는 연속 주입 엔진.
단어와 문장을 실시간으로 4D 충격파로 변환하여 텐서 필드를 폭격합니다.
이 과정을 통해 텐서 필드는 쏟아지는 자극 속에서 '기준'을 찾고 자가 정렬합니다.
"""

import asyncio
import websockets
import json
import os
import re

ARCHIVE_PATH = r"c:\Archive"

async def unleash_avalanche():
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri, max_size=16777216) as websocket:
            print("==================================================")
            print(" [WARNING] Data Avalanche (대규모 데이터 범람) 개시 ")
            print(" 엘리시아의 센서리움(감각수용기)에 폭격이 시작됩니다.")
            print("==================================================")
            
            # 아카이브 폴더의 모든 마크다운 파일 파싱
            all_words = []
            for filename in os.listdir(ARCHIVE_PATH):
                if filename.endswith(".md"):
                    filepath = os.path.join(ARCHIVE_PATH, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                        # 형태소 단위로 거칠게 쪼갬 (기호 제외)
                        words = re.findall(r'[가-힣A-Za-z0-9]+', text)
                        all_words.extend(words)
            
            print(f"총 {len(all_words)}개의 데이터 청크가 준비되었습니다.")
            print("폭격을 시작합니다... (3D 화면에서 기하학적 정렬 현상을 관측하세요)")
            
            count = 0
            for word in all_words:
                if len(word) < 2: continue # 너무 짧은 단어는 스킵
                
                await websocket.send(json.dumps({
                    'type': 'stimulus',
                    'text': word
                }))
                
                count += 1
                if count % 100 == 0:
                    print(f"[{count}개 투척 완료] ...")
                    
                # 기관총처럼 쏟아붓기 (초당 수십~수백 개)
                await asyncio.sleep(0.01)
                
            print("\n모든 데이터 폭격이 완료되었습니다. 엘리시아의 우주가 자가 정렬되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(unleash_avalanche())
