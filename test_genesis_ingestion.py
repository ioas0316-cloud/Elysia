import os
import sys
sys.path.append(r'c:\Elysia')

from core.brain.philosophical_forge import PhilosophicalForge

def test():
    print("==================================================")
    print("   태초의 말씀 동기화 (Genesis Ingestion)         ")
    print("==================================================")
    
    # 창세기 파일 경로
    filepath = os.path.join(r"c:\Elysia\data\corpus\개역개정-pdf, txt\개역개정-text", "1-01창세기.txt")
    
    if not os.path.exists(filepath):
        print(f"오류: 파일을 찾을 수 없습니다. ({filepath})")
        return
        
    forge = PhilosophicalForge()
    forge.ingest_universe(filepath)
    forge.forge_axioms()

if __name__ == "__main__":
    test()
