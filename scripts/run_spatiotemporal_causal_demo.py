"""
Elysia Core Topology: SpatioTemporal Reactive Cascade & Causal Engine Demonstration

Runs Scenario 1 (Coherent Single Source), Scenario 2 (Spatial Anomaly Out-Of-Frame),
Grounding Ontology & AST Self-Rewriting, Dual Plasticity feedback, and OOD Emergent Axiom Induction.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.embodied.emergence import AxiomInducer, OODLatentBuffer, OODSample
from core.embodied.plasticity import DualPlasticityPipeline
from core.embodied.projector import VQSensoryProjector
from core.engine.calibration import SelfCalibrationEngine, ZeroShotSafetyGuard
from core.topology.ast_rewriting import SelfRewritingNodeEngine
from core.topology.grounding_ontology import GroundingAxiom, GroundingOntologyEngine
from core.topology.spatiotemporal_reactive_cascade import (
    AudioFrame,
    DAGNode,
    ReactiveDAG,
    SpatioTemporalSyncAdapter,
    VisualFrame,
    action_decision_operator,
    causal_integrity_operator,
    sync_input_operator,
)


def run_demo():
    print("=" * 80)
    print("🚀 [Elysia Core Architecture Demo] 시공간 반응형 인과 엔진 & 자율 공리 체계")
    print("=" * 80)

    # 1. SpatioTemporal Sync Adapter & Reactive Cascade DAG
    adapter = SpatioTemporalSyncAdapter()
    dag = ReactiveDAG()

    sync_node = DAGNode("Sync_Input_Node", lambda inputs: None)
    causal_node = DAGNode("Causal_Integrity_Node", causal_integrity_operator)
    action_node = DAGNode("Action_Decision_Node", action_decision_operator)

    dag.add_node(sync_node)
    dag.add_node(causal_node)
    dag.add_node(action_node)

    dag.add_edge("Sync_Input_Node", "Causal_Integrity_Node")
    dag.add_edge("Causal_Integrity_Node", "Action_Decision_Node")

    print("\n=================== [시나리오 1: 단일 인과 이벤트 정상 전파] ===================")
    v_frame_normal = VisualFrame(timestamp_ms=100.0, bbox=(0.4, 0.2, 0.6, 0.8), label="speaker")
    a_frame_normal = AudioFrame(timestamp_ms=105.0, doa_angle_deg=180.0, decibel=75.0, sound_type="speech")

    state1 = sync_input_operator(adapter, v_frame_normal, a_frame_normal)
    dag.propagate("Sync_Input_Node", state1)
    print(f"  👉 Final Action Command: {dag.nodes['Action_Decision_Node'].state.relational_bindings.get('EXECUTE_COMMAND')}")

    print("\n=================== [시나리오 2: 화면 밖 음원(공간 이상) 반응형 전파] ===================")
    v_frame_off = VisualFrame(timestamp_ms=100.0, bbox=(0.1, 0.2, 0.2, 0.8), label="speaker")
    a_frame_off = AudioFrame(timestamp_ms=105.0, doa_angle_deg=220.0, decibel=80.0, sound_type="speech")

    state2 = sync_input_operator(adapter, v_frame_off, a_frame_off)
    dag.propagate("Sync_Input_Node", state2)
    print(f"  👉 Final Action Command: {dag.nodes['Action_Decision_Node'].state.relational_bindings.get('EXECUTE_COMMAND')}")

    # 2. Grounding Ontology & Dual Plasticity
    print("\n=================== [3. 온톨로지 공리 검증 & Dual Plasticity 2중 가소성] ===================")
    ontology = GroundingOntologyEngine()
    ontology.register_axiom(
        GroundingAxiom(
            axiom_id="AXIOM_THERMAL_01",
            description="열 위험 상태 물리적 모순 방지",
            forbidden_pairs=[("THERMAL_HAZARD", "FROST_CRYSTAL")],
            required_bindings={"SAFETY_CONTAINMENT": "ACTIVE"},
        )
    )

    projector = VQSensoryProjector()
    projector.register_prototype("THERMAL_HAZARD", [1.0, 0.0], initial_threshold=0.5)
    projector.register_prototype("FROST_CRYSTAL", [0.8, 0.2], initial_threshold=0.5)

    rewriter = SelfRewritingNodeEngine()
    rewriter.register_operator(
        "Causal_Node_01",
        lambda st: {"qualities": {"THERMAL_HAZARD", "FROST_CRYSTAL"}, "bindings": {"SAFETY_CONTAINMENT": "INACTIVE"}},
    )

    pipeline = DualPlasticityPipeline(projector, ontology, rewriter)
    print("1차 감각 입력 수신 [0.9, 0.1]:")
    res1 = pipeline.process_sensory_input([0.9, 0.1])

    print("\n2차 동일 감각 입력 재수신 (억제된 tau 적용):")
    res2 = pipeline.process_sensory_input([0.9, 0.1])

    # 3. OOD Emergence & Axiom Induction
    print("\n=================== [4. OOD 미지 데이터 3중 필터 및 신규 공리 발현] ===================")
    ood_buffer = OODLatentBuffer()
    inducer = AxiomInducer(ood_buffer)

    print("미지 감각 벡터 5회 연속 수집 중...")
    for i in range(5):
        ood_buffer.add_unknown_signal(
            OODSample(
                sensory_vector=[3.0 + i * 0.01, 4.0 - i * 0.01, 1.5],
                timestamp_ms=1000.0 + i * 20.0,
                modalities={"visual": [3.0, 4.0], "motor": [1.5, 1.5]},
            )
        )

    emergent_axiom = inducer.discover_new_concept()
    if emergent_axiom:
        print(f"🎉 성공적으로 신규 공리 추론 완료: {emergent_axiom['inferred_rule']}")

    # 4. Self-Calibration & Zero-shot Transfer Safety Guard
    print("\n=================== [5. Self-Calibration & Zero-shot Safety Guard] ===================")
    calibrator = SelfCalibrationEngine(ontology, rewriter, error_threshold=0.15)
    recalibrated_output, report = calibrator.step_and_calibrate(
        node_id="Causal_Node_01",
        current_state={"qualities": ["THERMAL_HAZARD"]},
        real_sensor_output={"val": 5.0},
    )

    guard = ZeroShotSafetyGuard(ontology)
    passed, violations = guard.validate_transfer(
        domain_name="Server_Cloud_Domain",
        abstract_rule={"dimension_match": True},
        proposed_qualities={"THERMAL_HAZARD"},
        proposed_bindings={"SAFETY_CONTAINMENT": "ACTIVE"},
        confidence_score=0.92,
    )
    print(f"🛡️ Zero-shot Transfer Validation Result: Passed={passed}")

    print("\n" + "=" * 80)
    print("✅ 모든 시나리오 및 통합 인과 파이프라인 시뮬레이션이 완료되었습니다.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
