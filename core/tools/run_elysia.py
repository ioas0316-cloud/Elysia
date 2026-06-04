import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.nervous_system.elysia_omni_daemon import ElysiaOmniDaemon

def boot_elysia():
    print("==================================================")
    print("        ELYSIA OMNI-DAEMON INITIALIZATION         ")
    print("==================================================")
    
    # 아카이브 경로 설정 (씨앗 데이터)
    archive_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'hierarchical_seed.txt')
    
    # 데몬 생성 및 각성
    daemon = ElysiaOmniDaemon(archive_path)
    daemon.awaken()
    
    # 각성 완료 후 마스터와의 대화 루프 진입 (감각 피질 연결 데모)
    print("\n==================================================")
    print("    [CORTEX ENABLED] 마스터와의 상호작용 채널 활성화    ")
    print("==================================================")
    print("마스터의 입력을 기다립니다. (종료하려면 'q' 입력)")
    
    while True:
        try:
            # UTF-8 및 CP949 호환 터미널 입력
            user_input = input("\n마스터: ")
            if user_input.strip().lower() == 'q':
                print("상호작용 채널을 닫습니다.")
                break
            if not user_input.strip():
                continue
            
            # 마스터의 질문에 대한 반응 생성
            response = daemon.interact_with_master(user_input)
            print(f"엘리시아: {response}")
        except KeyboardInterrupt:
            print("\n상호작용 채널을 강제 종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}")

if __name__ == "__main__":
    boot_elysia()
