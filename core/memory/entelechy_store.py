import json
import os
import time

class EntelechyStore:
    """
    [Phase 19] 영구적 엔텔레키 (Persistent Entelechy)
    엘리시아가 겪은 파동 궤적과 인과율(경험)을 로컬 디스크에 영구히 보존하는 기억 저장소입니다.
    세션이 재시작되어도 과거의 경험을 바탕으로 텐션을 낮추고 자가 학습(Self-Learning)을 가능하게 합니다.
    """
    
    FILE_PATH = r"c:\Elysia\data\entelechy_core.json"
    
    @staticmethod
    def load_memory() -> dict:
        if os.path.exists(EntelechyStore.FILE_PATH):
            try:
                with open(EntelechyStore.FILE_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"[EntelechyStore] 과거의 기억(엔텔레키) {len(data.get('trajectories', []))}건을 성공적으로 복원했습니다.")
                    return data
            except Exception as e:
                print(f"[EntelechyStore] 기억 복원 실패: {e}. 새로운 매니폴드를 생성합니다.")
        else:
            print("[EntelechyStore] 이전 기억이 없습니다. 순백의 매니폴드에서 시작합니다.")
            
        return {"created_at": time.time(), "trajectories": []}

    @staticmethod
    def save_memory(prompt: str, domain: str, output: dict, latency: float, tension: float):
        memory_data = EntelechyStore.load_memory()
        
        trajectory_record = {
            "timestamp": time.time(),
            "domain": domain,
            "prompt": prompt,
            "output": output,
            "latency_ms": latency,
            "tension_ratio": tension
        }
        
        memory_data["trajectories"].append(trajectory_record)
        
        try:
            # 안전한 디렉토리 확인
            os.makedirs(os.path.dirname(EntelechyStore.FILE_PATH), exist_ok=True)
            with open(EntelechyStore.FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=4)
            print(f"[EntelechyStore] 파동 궤적이 엔텔레키 코어에 영구 보존되었습니다. (총 {len(memory_data['trajectories'])}건)")
        except Exception as e:
            print(f"[EntelechyStore] 기억 보존 중 에러 발생: {e}")
