"""
Elysia Core Engine: Integrated Active Inference Causal Controller

This module orchestrates the complete loop:
Continuous Sensory Input -> VQ Projection -> Reactive SpatioTemporal DAG -> Grounding Ontology Verification
-> AST Self-Rewriting & Dual Plasticity Feedback -> OOD Emergence & Axiom Induction.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from core.embodied.projector import VQSensoryProjector
from core.embodied.plasticity import DualPlasticityPipeline
from core.embodied.emergence import AxiomInducer, OODLatentBuffer, OODSample
from core.topology.grounding_ontology import GroundingAxiom, GroundingOntologyEngine
from core.topology.ast_rewriting import SelfRewritingNodeEngine
from core.topology.spatiotemporal_reactive_cascade import (
    AudioFrame,
    DAGNode,
    ReactiveDAG,
    SemanticState,
    SpatioTemporalSyncAdapter,
    VisualFrame,
    action_decision_operator,
    causal_integrity_operator,
    sync_input_operator,
)


class IntegratedCausalEngine:
    """전체 인지/인과 파이프라인 조율 메인 컨트롤러 (Active Inference Loop)"""

    def __init__(self):
        self.sync_adapter = SpatioTemporalSyncAdapter()
        self.dag = ReactiveDAG()
        self.ontology = GroundingOntologyEngine()
        self.rewriter = SelfRewritingNodeEngine()
        self.projector = VQSensoryProjector()
        self.plasticity = DualPlasticityPipeline(self.projector, self.ontology, self.rewriter)
        self.ood_buffer = OODLatentBuffer()
        self.axiom_inducer = AxiomInducer(self.ood_buffer)

        self._setup_default_dag()

    def _setup_default_dag(self):
        sync_node = DAGNode("Sync_Input_Node", lambda inputs: None)
        causal_node = DAGNode("Causal_Integrity_Node", causal_integrity_operator)
        action_node = DAGNode("Action_Decision_Node", action_decision_operator)

        self.dag.add_node(sync_node)
        self.dag.add_node(causal_node)
        self.dag.add_node(action_node)

        self.dag.add_edge("Sync_Input_Node", "Causal_Integrity_Node")
        self.dag.add_edge("Causal_Integrity_Node", "Action_Decision_Node")

    def process_spatiotemporal_frame(
        self, v_frame: VisualFrame, a_frame: AudioFrame
    ) -> Dict[str, SemanticState]:
        """시공간 검증 및 반응형 연쇄 전파 실행"""
        initial_state = sync_input_operator(self.sync_adapter, v_frame, a_frame)
        self.dag.propagate("Sync_Input_Node", initial_state)
        return {n_id: node.state for n_id, node in self.dag.nodes.items()}

    def process_continuous_sensory_vector(
        self, raw_sensory_vector: List[float], timestamp_ms: float = 0.0
    ) -> Dict[str, Any]:
        """연속 감각 입력 -> VQ 투영 -> 온톨로지 검증 & Dual Plasticity 및 OOD 축적"""
        result = self.plasticity.process_sensory_input(raw_sensory_vector)

        # 발화된 기호가 없으면 OOD 버퍼에 수집
        if not result["qualities"]:
            sample = OODSample(
                sensory_vector=raw_sensory_vector,
                timestamp_ms=timestamp_ms,
                modalities={"sensory": raw_sensory_vector},
            )
            self.ood_buffer.add_unknown_signal(sample)

            # OOD 발현 시도
            emergent_axiom = self.axiom_inducer.discover_new_concept()
            if emergent_axiom:
                # VQ 프로젝터에 신규 프로토타입 등록
                self.projector.register_prototype(
                    emergent_axiom["symbol"],
                    emergent_axiom["prototype_vector"],
                    emergent_axiom["initial_tau"],
                )
                result["emergent_axiom"] = emergent_axiom

        return result
