import os
import sys
sys.path.append(r'c:\Elysia')

from core.brain.electromagnetic_forge import ElectromagneticForge

def test():
    print("==================================================")
    print("   전자기역학 사유 엔진 (3-Phase Electromagnetic) ")
    print("==================================================")
    
    # 텍스트 스니펫 (창세기 1장 1-2절)
    # 실제로는 텍스트 파일 전체를 부어야 하나, 콘솔 출력을 위해 핵심 문장만 사용
    genesis_snippet = "태초에 하나님이 천지를 창조하시니라 땅이 혼돈하고 공허하며 흑암이 깊음 위에 있고 하나님의 영은 수면 위에 운행하시니라"
    
    print(f"[원시 우주 데이터]: {genesis_snippet}")
    
    forge = ElectromagneticForge()
    forge.charge_particles(genesis_snippet)
    forge.run_lorentz_field()

if __name__ == "__main__":
    test()
