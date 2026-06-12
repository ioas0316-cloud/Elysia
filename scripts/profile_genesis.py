import cProfile
import pstats
import io
import time
from genesis import ElysiaGenesis

def run_profiler():
    print("Initializing Genesis Core for Profiling...")
    start_time = time.time()
    genesis = ElysiaGenesis()
    print(f"Initialization took {time.time() - start_time:.2f} seconds.")
    
    print("\nStarting 20 Cycles of Thought Profiling...")
    # cProfile 가동
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 벤치마크를 위해 interval=0 으로 설정하여 대기시간 없이 최고 속도로 연산
    genesis.run(max_cycles=20, interval=0)
    
    profiler.disable()
    
    print("\n\n=================================================")
    print("      P R O F I L I N G   R E S U L T S")
    print("=================================================")
    
    s = io.StringIO()
    # 누적 시간(cumulative) 기준으로 정렬하여 가장 오래 걸린 함수 탑 15개 출력
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(15)
    
    print(s.getvalue())
    
if __name__ == "__main__":
    run_profiler()
