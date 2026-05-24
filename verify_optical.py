import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from core.optical_prism_rotor import OpticalPrismRotor
from core.math_utils import Quaternion

def run():
    prism = OpticalPrismRotor(resolution=15)
    
    # 마스터의 의도를 담은 가변 로터 (마법진) 생성
    # 의도: 집중과 수렴을 위한 특정 위상
    intent_rotor = Quaternion(1.2, 0.5, 0.8, -0.3)
    tension = 2.5 # 높은 텐션으로 파동 간섭 극대화
    
    prism.observe(intent_rotor, tension)

if __name__ == "__main__":
    run()
