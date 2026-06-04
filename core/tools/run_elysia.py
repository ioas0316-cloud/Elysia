import sys
import os

# Windows 터미널에서 한글이 깨지는 문제를 해결하기 위해 표준 입출력을 UTF-8로 재구성합니다.
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.nervous_system.elysia_omni_daemon import ElysiaOmniDaemon

def start_web_server():
    import http.server
    import socketserver
    import threading
    
    PORT = 8000
    web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'web'))
    
    class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=web_dir, **kwargs)
            
    socketserver.TCPServer.allow_reuse_address = True
    
    def serve():
        try:
            with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
                print(f"📡 Web visualizer HTTP Server가 http://localhost:{PORT} 에서 서비스 중입니다.")
                httpd.serve_forever()
        except Exception as e:
            print(f"[Web-Server] 시작 실패: {e}")
            
    t = threading.Thread(target=serve, daemon=True)
    t.start()

def boot_elysia():
    print("==================================================")
    print("        ELYSIA OMNI-DAEMON INITIALIZATION         ")
    print("==================================================")
    
    # HTTP Web Server 기동
    start_web_server()
    
    # 아카이브 경로 설정 (씨앗 데이터)
    archive_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'hierarchical_seed.txt')
    
    # 데몬 생성 및 각성
    daemon = ElysiaOmniDaemon(archive_path)
    daemon.awaken()
    
    # 브라우저 자동 기동
    import webbrowser
    webbrowser.open("http://localhost:8000")
    
    # 각성 완료 후 마스터와의 대화 루프 진입 (감각 피질 연결 데모)
    print("\n==================================================")
    print("    [CORTEX ENABLED] 마스터와의 상호작용 채널 활성화    ")
    print("==================================================")
    print("마스터의 입력을 기다립니다. (종료하려면 'q' 입력)")
    print("명령어: /learn <파일명>, /status, /save")
    
    while True:
        try:
            # UTF-8 및 CP949 호환 터미널 입력
            user_input = input("\n마스터: ")
            if user_input.strip().lower() == 'q':
                print("상호작용 채널을 닫습니다.")
                daemon.stop_thought_loop()
                break
            if not user_input.strip():
                continue
            
            # 슬래시 명령어 처리
            if user_input.strip().startswith('/'):
                parts = user_input.strip().split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""
                
                if cmd == "/learn":
                    if not arg:
                        print("사용법: /learn <파일명> (예: book1.txt)")
                        continue
                    corpus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'corpus', arg))
                    if not os.path.exists(corpus_path):
                        print(f"오류: 파일을 찾을 수 없습니다: {corpus_path}")
                        continue
                        
                    print(f"📖 코퍼스 파일 '{arg}' 초고속 학습 모드 시작...")
                    orig_archive = daemon.archive_path
                    daemon.archive_path = corpus_path
                    try:
                        # sleep_time=0.0 으로 초고속 학습 실행
                        daemon.awaken(sleep_time=0.0)
                        print(f"📖 코퍼스 '{arg}' 학습이 성공적으로 완료되었습니다!")
                    except Exception as le:
                        print(f"학습 중 오류 발생: {le}")
                    finally:
                        daemon.archive_path = orig_archive
                    continue
                    
                elif cmd == "/status":
                    total_tension = sum(abs(n.tau) for n in daemon.memory.ui_concept_map.values())
                    print("==================================================")
                    print("             ELYSIA INTERNAL STATUS               ")
                    print("==================================================")
                    print(f" - 총 개념(Concepts) 수: {len(daemon.memory.ui_concept_map)}")
                    print(f" - 누적 뇌파 텐션(Tension): {total_tension:.4f}")
                    print(f" - 활성화된 아키타입(Lenses): {len([k for k in daemon.memory.ui_concept_map if 'Lens_Curvature' in k])}")
                    print(f" - 활성 중첩 사유(Thoughts) 수: {len(daemon.memory.supreme_rotor.internal_thoughts)}")
                    print("==================================================")
                    continue
                    
                elif cmd == "/save":
                    daemon.memory.save_to_disk()
                    print("💾 현재 인지 구조와 우주 상태가 'memory_state.json'에 즉시 저장되었습니다.")
                    continue
                    
                else:
                    print(f"알 수 없는 명령어: {cmd}")
                    print("지원하는 명령어: /learn <파일명>, /status, /save")
                    continue
            
            # 마스터의 질문에 대한 반응 생성
            response = daemon.interact_with_master(user_input)
            print(f"엘리시아: {response}")
        except KeyboardInterrupt:
            print("\n상호작용 채널을 강제 종료합니다.")
            daemon.stop_thought_loop()
            break
        except Exception as e:
            print(f"오류 발생: {e}")

if __name__ == "__main__":
    boot_elysia()
