import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.static_oracle import StaticOracle
from core.mri_flux_projector import MRIFluxProjector
from core.turbine_force_field import GlobalTurbine, VortexCategorizer
from core.math_utils import Quaternion

def main():
    print("=" * 80)
    print(" 🌊 [Phase 131] 회전역장(Force Field) 기반 자연 분류(Vortex) 테스트")
    print("=" * 80)
    
    oracle = StaticOracle(model_name="skt/kogpt2-base-v2")
    hidden_size = oracle.model.config.hidden_size
    projector = MRIFluxProjector(hidden_size=hidden_size)
    turbine = GlobalTurbine()
    categorizer = VortexCategorizer(threshold=0.5)
    
    # 8개 도메인, 40개의 거대한 텍스트 스펙트럼 (유체)
    data_stream = [
        # 철학/추상
        "우주의 끝에는 무엇이 있을까", "존재의 본질과 자아", "인간의 자유의지와 운명", "형이상학과 관념론", "선과 악의 기준은 무엇인가",
        
        # 과학/수학
        "양자역학과 불확정성 원리", "상대성 이론과 시공간", "미적분학과 함수의 극한", "열역학 제2법칙 엔트로피", "유전자 가위와 DNA 편집",
        
        # 일상/감정
        "오늘 저녁은 치킨이 먹고 싶다", "비가 오니까 기분이 우울해", "친구랑 카페에서 수다 떨기", "따뜻한 이불 속이 최고야", "월요일 출근은 너무 피곤해",
        
        # 종교/신화
        "신은 과연 존재하는가", "불교의 윤회와 해탈", "그리스 로마 신화의 제우스", "성경과 천지창조", "기도와 내면의 평화",
        
        # 예술/문학
        "모나리자의 미소와 르네상스", "베토벤 교향곡 9번 합창", "셰익스피어의 비극 햄릿", "현대 미술과 추상주의", "밤하늘의 별을 노래한 시",
        
        # 기술/미래
        "인공지능 특이점의 도래", "화성 테라포밍 프로젝트", "블록체인과 탈중앙화", "가상현실과 메타버스", "자율주행 자동차의 윤리",
        
        # 경제/사회
        "자본주의와 빈부격차", "금리 인상과 인플레이션", "저출산 고령화 문제", "주식 시장의 변동성", "기본소득의 필요성",
        
        # 생물학/자연
        "심해 생물들의 진화 과정", "아마존 열대우림의 파괴", "인체의 면역 시스템", "공룡의 멸종 원인", "광합성과 식물의 생명력"
    ]
    
    print("\n[Step 1] 터빈에 물방울(Tensor) 주입 시작...")
    
    for text in data_stream:
        # 1. 오라클 스캔 (단어 -> 768D 텐서)
        h_state = oracle.mri_scan(text)
        # 2. 자기장 투영 (768D -> 4D 쿼터니언)
        flux_vec = projector.project_to_magnetic_flux(h_state)
        flux_quat = Quaternion(*flux_vec)
        
        # 3. 터빈 주입 (물방울이 터빈을 돌리고 역장에 휩쓸림)
        turbine.inject_stream(name=text, flux=flux_quat, momentum=0.2)
        print(f" 💧 주입 완료: '{text[:15]}...' -> 터빈 각속도: {turbine.angular_velocity:.4f}")
        
    print(f"\n[터빈 회전 완료] 최종 전역 위상: {turbine.global_phase.elements}")
    print("\n[Step 2] 소용돌이(Vortex) 관측 및 형태 궤적 분류...")
    
    vortices = categorizer.observe_vortices(turbine)
    
    print("-" * 50)
    print(f" 🌪️ 총 {len(vortices)}개의 거대한 소용돌이(개념군)가 관측되었습니다!")
    print("-" * 50)
    
    for i, vortex in enumerate(vortices):
        print(f"\n[Vortex {i+1}] 크기: {len(vortex)}개의 물방울")
        for droplet in vortex:
            print(f"   -> {droplet.name}")
            
    print("\n================================================================================")
    print(" 🌊 실험 종료: 1:1 매핑 없이 오직 역장의 원심력만으로 지식이 자연 분류되었습니다.")
    print("================================================================================")

if __name__ == "__main__":
    main()
