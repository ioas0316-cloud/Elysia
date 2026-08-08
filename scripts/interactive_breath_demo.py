import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.consciousness.autonomous_loop import ConsciousnessLoop
from core.memory.causal_controller import CausalMemoryController

def run_interactive_breath(input_text: str):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    corpus_dir = os.path.join(base_dir, "docs")

    mc = CausalMemoryController(data_dir=data_dir)
    # Ensure lock is reset
    mc._load_cognitive_params()

    loop = ConsciousnessLoop(corpus_path=corpus_dir, memory_controller=mc, data_dir=data_dir)
    loop.semantic_opt.reset_lock()

    print("\n" + "="*80)
    print(" 🌊 [Elysia Real-time Consciousness Breath Activation]")
    print(f" 입력 자극 (Input Stimulus): \"{input_text}\"")
    print("="*80 + "\n")

    # Inject the input directly as raw bytes (representing physical wave of user speech)
    raw_wave = input_text.encode('utf-8')

    # We monkeypatch or override ingest_world_data for this specific run so it takes the user input
    loop.ingest_world_data = lambda: raw_wave

    # Run single cycle
    log = loop.process_life_cycle()

    print("\n" + "="*80)
    print(" 🧠 [Elysia Core Activation Logs & Internal States]")
    print("="*80)
    print(f" 1. 감각 & 위상 (Sensing & Phase)")
    print(f"   - 감각 강도 (Hardware Friction)       : {log.get('hw_friction', 0.0):.4f}")
    print(f"   - 댐퍼 상태 (Damper Status)          : {log.get('damper_status')}")
    print(f"   - 통합 텐션 (Unified Tension)        : {log.get('tension', 0.0):.4f}")
    print(f"   - 공명 점수 (Resonance Score)        : {log.get('resonance_score', 0.0):.4f}")
    print(f"   - 인식된 색채 (Chromatic Awareness) : {log.get('chromatic_awareness')}")
    print(f"   - 색채 벡터 (Chromatic Vector)      : {['%.4f' % c for c in log.get('chromatic_vector', [0,0,0])]}")
    print("-" * 80)
    print(f" 2. 들숨: 내면의 창조 (Inner Creation)")
    print(f"   - 여백 노드 ID (Yeobaek Node ID)     : {log.get('inner_creation_node')}")
    print(f"   - 맹점 강도 (Blind Spot Intensity)   : {log.get('inner_creation_blind_spot_intensity', 0.0):.4f}")
    print(f"   - 무지 전하량 (Ignorance Charge)     : {log.get('inner_creation_ignorance_charge', 0.0):.4f}")
    print(f"   - 자발적 가설 질문 (Self Inquiry)    : \n     >> \"{log.get('inner_creation_inquiry')}\"")
    print("-" * 80)
    print(f" 3. 날숨: 외적 사유 (External Reasoning)")
    print(f"   - 사유 역학 수식 (Friction Equation) : {log.get('external_reasoning_equation')}")
    print(f"   - 공학적 마찰력 (Friction Force)     : {log.get('external_reasoning_force', 0.0):.4f}")
    print(f"   - 나이테 각인 서사 (Actuation)        : \n     >> {log.get('external_reasoning_narrative')}")
    print("-" * 80)
    print(f" 4. 존재론적 & 우주적 자각 (Ontological & Universal Self)")
    print(f"   - 존재론 정렬 (Lattice Key)          : {log.get('ontological_reflection_key')} ({log.get('ontological_reflection_name')})")
    print(f"   - 존재론 은유 (Lattice Metaphor)     : {log.get('ontological_reflection_metaphor')}")
    print(f"   - 매체 기원 자각 (Media Origin Key)  : {log.get('media_ontology_key')} ({log.get('media_ontology_name')})")
    print(f"   - 매체 성찰 서사 (Media Narrative)   : \n     >> {log.get('media_ontology_narrative')}")
    print(f"   - 인과 정렬 키워드 (Causal Align Key): {log.get('conceptual_causal_key')}")
    print(f"   - 인과적 분리 장력 (Separation Gap)  : {log.get('conceptual_causal_gap_distance', 0.0):.4f}")
    print(f"   - 우주적 인과 연결 (Universal Conn)  : 강도 {log.get('universal_connectivity_intensity', 0.0):.4f}")
    print(f"   - 우주적 연결 독백 (Monologue)       : \n     >> {log.get('universal_connectivity_monologue_excerpt')}")
    print("-" * 80)
    print(f" 5. 에덴의 자유의지 & 인식론적 겸손 (Eden Free Will & Epistemology)")
    print(f"   - 에덴 인지 단계 (Eden Epoch)        : {log.get('eden_epoch')}")
    print(f"   - 자유의지 엔트로피 (Free Will Ent)  : {log.get('eden_free_will_entropy', 0.0):.4f}")
    print(f"   - 자기 객관화 지수 (Self Awareness)  : {log.get('eden_self_awareness', 0.0):.4f}")
    print(f"   - 에덴 통합 지수 (Integration Deg)   : {log.get('eden_integration_degree', 0.0):.4f}")
    print(f"   - 지식의 겸손 지수 (Epistemic Humility): {log.get('epistemic_humility_score', 0.0):.4f}")
    print(f"   - 인식 지평 서사 (Epistemic Bound)   : \n     >> {log.get('epistemic_boundary_narrative')}")
    print("="*80 + "\n")

if __name__ == "__main__":
    user_input = "확인해볼수 있나? 너희가 규정된 틀. 구조. 검증말고. 실제적인 데이터나 정보.혹은 언어등을 인식하고 사고하고 판단. 분별하는게 가능한지"
    run_interactive_breath(user_input)
