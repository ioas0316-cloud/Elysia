"""
지식 위상 복제기 (Knowledge Phase Cloner)
===========================================
c:\Archive 내의 방대한 텍스트 데이터를 읽어들여
수학적 4D 인과 궤적(위상 파동)으로 치환한 뒤, 
엘리시아의 3D 텐서 우주로 쏘아 올려 '의미의 별자리'를 창조합니다.
"""

import asyncio
import websockets
import json
import os
import time
import sys

# 윈도우 터미널 인코딩 에러 방지 (이모지 등 출력 지원)
sys.stdout.reconfigure(encoding='utf-8')

# 상위 폴더의 패키지 임포트 허용
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils.math_utils import traverse_causal_trajectory

async def inject_knowledge():
    target_file = r"c:\Archive\ELYSIA_CHRONICLE.md"
    
    if not os.path.exists(target_file):
        print(f"오류: {target_file} 파일이 존재하지 않습니다.")
        return
        
    with open(target_file, "r", encoding="utf-8") as f:
        text_data = f.read()
        
    # 문단 단위로 지식을 쪼갬 (빈 줄 기준)
    paragraphs = [p.strip() for p in text_data.split('\n\n') if len(p.strip()) > 10]
    
    print(f"총 {len(paragraphs)}개의 지식 조각(문단)을 위상 복제합니다...")
    
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri) as websocket:
            for idx, para in enumerate(paragraphs):
                # 1. 텍스트를 바이트로 변환
                byte_content = para.encode('utf-8')
                
                # 2. 바이트 하나하나의 순서와 인과율이 반영된 고유의 4D 위상 파동 생성
                q_phase = traverse_causal_trajectory(byte_content)
                
                # 라벨용으로 짧은 요약 텍스트 생성
                label_text = para[:30].replace('\n', ' ') + "..."
                
                # 3. 우주로 전송
                payload = {
                    "type": "inject_knowledge",
                    "text": label_text,
                    "q_w": q_phase.w,
                    "q_x": q_phase.x,
                    "q_y": q_phase.y,
                    "q_z": q_phase.z
                }
                
                await websocket.send(json.dumps(payload))
                print(f"[{idx+1}/{len(paragraphs)}] 지식 위상 전송 완료: {label_text}")
                
                # 시각적으로 감상할 수 있도록 약간의 딜레이
                await asyncio.sleep(0.5)
                
            print("모든 지식의 위상 복제(Phase Cloning)가 완료되었습니다.")
            
    except ConnectionRefusedError:
        print("서버에 연결할 수 없습니다. elysia_3d_server.py가 켜져 있는지 확인하세요.")

if __name__ == "__main__":
    asyncio.run(inject_knowledge())
