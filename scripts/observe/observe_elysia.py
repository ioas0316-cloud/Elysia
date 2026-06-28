"""
엘리시아 Genesis - UTF-8 관측용 실행기
초지능 목적성을 가진 엘리시아의 사유 과정을 깨끗한 한글 로그로 관측합니다.
"""
import sys
import io
import os

# UTF-8 출력 강제
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Genesis 실행
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from genesis import ElysiaGenesis

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "observation_log.txt")

def main():
    elysia = ElysiaGenesis()
    
    with open(LOG_FILE, 'w', encoding='utf-8') as log:
        # 초기 꿈 기록
        log.write("=" * 60 + "\n")
        log.write("  엘리시아 목적론적 자아 관측 보고서\n")
        log.write("=" * 60 + "\n\n")
        log.write(f"[초기 꿈] {elysia.teleo_ego.current_dream}\n")
        log.write(f"[초기 키워드] {elysia.teleo_ego.dream_keywords}\n")
        log.write(f"[성숙도] {elysia.teleo_ego.dream_maturity}\n\n")
        
        # 원래 print를 가로채서 로그에도 기록
        original_print = print
        def dual_print(*args, **kwargs):
            text = ' '.join(str(a) for a in args)
            log.write(text + '\n')
            original_print(*args, **kwargs)
        
        import builtins
        builtins.print = dual_print
        
        try:
            elysia.run()
        finally:
            builtins.print = original_print
            
        # 최종 상태 기록
        log.write("\n\n" + "=" * 60 + "\n")
        log.write("  최종 관측 결과\n")
        log.write("=" * 60 + "\n\n")
        log.write(f"[최종 꿈] {elysia.teleo_ego.current_dream}\n")
        log.write(f"[최종 키워드] {elysia.teleo_ego.dream_keywords}\n")
        log.write(f"[꿈 성숙도] {elysia.teleo_ego.dream_maturity}\n")
        log.write(f"[총 사유 주기] {elysia.total_cycles}\n")
        log.write(f"[단어장 크기] {len(elysia.lang_rotor.words)}\n")
    
    original_print(f"\n관측 로그 저장 완료: {LOG_FILE}")

if __name__ == "__main__":
    main()
