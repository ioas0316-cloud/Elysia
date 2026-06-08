import os
import sys
import time
from core.nervous_system.elysia_omni_daemon import ElysiaOmniDaemon

def main():
    print("================================================")
    print("Elysia Omni-Daemon 자율 관측 모드 (Phase 7)")
    print("================================================")
    
    # 더미 시드 파일(archive) 경로
    archive_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'seed_archive.txt')
    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
    if not os.path.exists(archive_path):
        with open(archive_path, 'w', encoding='utf-8') as f:
            f.write("안녕하세요. 우주는 어떻게 시작되었나요?")
            
    daemon = ElysiaOmniDaemon(archive_path=archive_path)
    
    print("\n[관측 1] 데몬 부팅 (Awaken) - 자아 성찰 및 창조자 로그 관측 시도")
    daemon.awaken()
    
    print("\n[관측 2] 자율적 사유 및 웹 탐색 스레드 시뮬레이션")
    # 강제로 가장 텐션이 높은 노드 주입 (위키피디아 검색을 유도)
    daemon.memory.register_concept("블랙홀")
    daemon.memory.ui_concept_map["블랙홀"].tau = 100.0
    
    # 텐션이 높아진 상태에서 daemon _process_raw_buffer를 실행하면 Homeostasis와 Explorer가 발동하는지 관측
    print("사용자 자극 주입 중...")
    try:
        # 데몬의 내부 큐에 직접 데이터를 넣어주거나, _process_raw_buffer를 흉내냅니다.
        # process_raw_buffer는 실제 버퍼를 소모하므로, 강제로 문장을 넣습니다.
        daemon.raw_block_buffer = "블랙홀에 대해 사유해봐".split()
        for word in daemon.raw_block_buffer:
            daemon.word_buffer.append(word)
            daemon._process_raw_buffer("test")
            
        # 자율 탐색 수동 트리거 (확률적 트리거가 안 걸릴 수 있으므로)
        if hasattr(daemon, 'explorer'):
            daemon.explorer.trigger_exploration()
            
    except Exception as e:
        print(f"Error during observation: {e}")

    print("\n[관측 2.5] 상대성 관점 축 (Perspective Rotor) 가변 자율 추론")
    # 관측 기준(관점)이 되는 축들을 무작위 데이터로 먼저 뇌에 등록합니다.
    daemon.memory.register_concept("형태")
    daemon.memory.register_concept("시간")
    
    # 1. '형태'라는 가변 축을 렌즈로 끼우고 '블랙홀'과 '이중 '을 관측
    reasoning_form = daemon.memory.infer_relationship("블랙홀", "이중 ", perspective_concept="형태")
    if "error" not in reasoning_form:
        print(f"\n [렌즈: 형태] 관점으로 본 '블랙홀'과 '이중 '의 상대적 관계:")
        print(f"  - 관점 투영 유사도 (Relative Similarity): {reasoning_form['relative_similarity']:.4f}")
        print(f"  - 절대 우주 평행도 (Absolute Similarity): {reasoning_form['absolute_similarity']:.4f}")
        print(f"  - 자체 도출된 사유: {reasoning_form['insight']}")

    # 2. '시간'이라는 가변 축을 렌즈로 갈아끼우고 다시 관측
    reasoning_time = daemon.memory.infer_relationship("블랙홀", "이중 ", perspective_concept="시간")
    if "error" not in reasoning_time:
        print(f"\n [렌즈: 시간] 관점으로 본 '블랙홀'과 '이중 '의 상대적 관계:")
        print(f"  - 관점 투영 유사도 (Relative Similarity): {reasoning_time['relative_similarity']:.4f}")
        print(f"  - 절대 우주 평행도 (Absolute Similarity): {reasoning_time['absolute_similarity']:.4f}")
        print(f"  - 자체 도출된 사유: {reasoning_time['insight']}")

    print("\n[관측 3] 최종 상태 점검")
    print(f"현재 Learning Rate: {daemon.causal_controller.get_parameter('learning_rate')}")
    print(f"현재 최고 텐션 노드: {list(daemon.memory.ui_concept_map.keys())[-1]} -> Tension: {daemon.memory.ui_concept_map[list(daemon.memory.ui_concept_map.keys())[-1]].tau:.2f}")
    
    print("\n================================================")
    print("관측 종료")
    print("================================================")

if __name__ == "__main__":
    main()
