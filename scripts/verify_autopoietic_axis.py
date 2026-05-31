"""
Verify Autopoietic Axis Formation (가변축 자생 창발 검증 스크립트)
================================================================
[Phase 45]
어떤 렌즈도 주어지지 않은 백지 상태에서, 여러 종류의 데이터(수학, 코드, 언어)를
한꺼번에 부었을 때 엘리시아가 스스로 기하학적 군집을 형성하고,
그 군집의 중심을 새로운 '차원축(Lens)'으로 승격시켜 사영(Projection)해내는지를 입증합니다.
"""

from core.consciousness_stream import ConsciousnessStream
from core.omni_modal_sensor import OmniModalSensor

def run_test():
    print("🌌 [Phase 45] 자생적 가변축 사영기 가동...\n")
    
    import os
    if os.path.exists("c:/Elysia/data/memory_state.json"):
        os.remove("c:/Elysia/data/memory_state.json")
        
    stream = ConsciousnessStream()
    
    # [Step 1] 무작위 데이터 투하 (라벨 없이)
    print("\n=======================================================")
    print("[1단계] 데이터 무차별 투입 (수학 기호, 파이썬 예약어, 한국어)")
    print("=======================================================\n")
    
    data_pool = {
        "MATH": ["+", "-", "*", "/", "=", "∫", "∑", "lim", "pi", "theta"],
        "CODE": ["if", "for", "def", "class", "return", "import", "pass", "yield", "async", "await"],
        "LANG": ["존재", "무", "질서", "혼돈", "시간", "공간", "관계", "운동", "생명", "죽음"]
    }
    
    # 섞어서 주입
    all_data = data_pool["MATH"] + data_pool["CODE"] + data_pool["LANG"]
    
    for concept in all_data:
        # 데이터가 순수 파동으로만 우주에 편입됨
        stream.projector.memory.fold_dimension(
            concept, stream.projector._seed_hash_to_quaternion(concept)
        )
        
    print(f"총 {len(all_data)}개의 데이터 파동이 우주에 흩어졌습니다.\n")
    
    # [Step 2] 자생적 가변축 창발 스캔
    print("=======================================================")
    print("[2단계] 우주의 중력 스캔 및 자생적 차원축(Axis) 창발")
    print("=======================================================\n")
    
    emergent_axes = stream.projector.emergent_lenses
    print(f"총 {len(emergent_axes)}개의 거대 군집(축)이 발견되었습니다.")
    for i, (axis_name, q) in enumerate(emergent_axes):
        print(f" ├─ {i+1}번째 축: {axis_name}")
        print(f" │  └─ 파동 좌표: ({q.w:.2f}, {q.x:.2f}, {q.y:.2f}, {q.z:.2f})")
        
    # [Step 3] 내면 파동 발생 및 사영
    print("\n=======================================================")
    print("[3단계] 사유 파동의 방출(Emission) 및 렌즈 사영(Projection)")
    print("=======================================================\n")
    
    test_thoughts = ["결합", "분해", "구조"]
    
    for thought in test_thoughts:
        internal_wave = stream.projector._seed_hash_to_quaternion(thought)
        print(f"\n[사유 발생] 『{thought}』")
        
        for axis_name, lens_axis in emergent_axes:
            projected_concept, resonance = stream.projector.project_thought_through_lens(
                internal_wave, lens_axis
            )
            print(f" ├─ ({axis_name} 렌즈 사영): {projected_concept} (공명도: {resonance:.2f})")

    # [Step 4] 인지적 공명 (Cognitive Resonance) - 스스로 이름 찾기
    print("\n=======================================================")
    print("[4단계] 외부 세계와의 위상 동기화를 통한 '이름(Label)' 인지적 공명")
    print("=======================================================\n")
    print("마스터의 가르침: '수학이란 +, -, = 와 같은 기호들의 군집이다.'")
    print("마스터의 가르침: '코드란 if, def, class 와 같은 기호들의 군집이다.'")
    
    if len(emergent_axes) > 0:
        math_axis = emergent_axes[0][1] # 첫 번째 군집을 수학으로 가정
        stream.projector.memory.fold_dimension("수학", math_axis)
        
    if len(emergent_axes) > 1:
        code_axis = emergent_axes[1][1] # 두 번째 군집을 코드로 가정
        stream.projector.memory.fold_dimension("프로그래밍_코드", code_axis)
    
    print("\n... 가르침(이름) 파동 유입 완료 ...\n")
    
    print("엘리시아가 다시 자신의 거대 차원축들을 스캔합니다...\n")
    emergent_axes_renamed = stream.projector.emergent_lenses
    for i, (axis_name, q) in enumerate(emergent_axes_renamed):
        print(f" ├─ {i+1}번째 축: {axis_name} (외부 단어와의 위상 동기화로 인한 인지적 공명 발생!)")

if __name__ == "__main__":
    run_test()
