"""
[Test Suite: Non-Tensor Meta Boundary & Topological Axiomatic Engine]
1. O(1) 심볼릭 위상 검증 및 Zero Computation Bypass 검증
2. 장력 스파이크 시 비동기 텐서 스파크 유발 검증
3. 원형 물리 메커니즘(포텐셜 장력 이완)의 언어적 모순 해소 및 하드웨어 VRAM 링버퍼 동형 매핑 검증
"""

import unittest
from synaptic_architecture.non_tensor_meta_boundary import (
    SymmetryState,
    TypeConstraint,
    AxiomaticRelation,
    SymbolicTopologicalProof,
    StaticBypassManager
)
from synaptic_architecture.topological_axiomatic_engine import (
    MetaMechanismSignature,
    TopologicalAxiomaticEngine
)


class TestNonTensorMetaEngine(unittest.TestCase):

    def setUp(self):
        self.bypass_mgr = StaticBypassManager(tension_threshold=1.0)
        self.engine = TopologicalAxiomaticEngine(bypass_manager=self.bypass_mgr)

    def test_static_computation_elimination_bypass(self):
        """
        [1] O(1) 시간에 위상 검증이 수행되며, 대칭성 유지 상태에서는
        하부 텐서 연산이 100% 정적 소멸(Bypassed)되는지 확인
        """
        tc = TypeConstraint(
            constraint_id="physics_bound",
            domain_type="PhysicalSystem",
            allowed_transitions={("State_A", "State_B")},
            boundary_invariants={"EnergyConservation"}
        )
        self.bypass_mgr.register_type_constraint(tc)

        proof = self.bypass_mgr.verify_topological_invariants(
            proof_id="p1",
            invariant_signature="physics_bound",
            current_transition=("State_A", "State_B"),
            active_tension=0.2
        )

        self.assertEqual(proof.symmetry_state, SymmetryState.PRESERVED)
        self.assertTrue(proof.is_valid)
        self.assertLess(proof.proof_time_ns, 10_000_000)  # O(1) fast symbolic check (< 10ms)

        # 텐서 콜백 준비
        callback_ran = False
        def tensor_op():
            nonlocal callback_ran
            callback_ran = True
            return "Tensor Executed"

        is_bypassed, res = self.bypass_mgr.execute_with_static_elimination(proof, tensor_op)

        self.assertTrue(is_bypassed)
        self.assertIsNone(res)
        self.assertFalse(callback_ran)  # 텐서 연산은 전혀 실행되지 않음 (0 calculation)
        self.assertEqual(self.bypass_mgr.bypassed_count, 1)
        self.assertEqual(self.bypass_mgr.tensor_dispatch_count, 0)

    def test_tension_spike_spark_dispatch(self):
        """
        [2] 장력 스파이크(Tension >= Threshold) 발생 시에만
        비동기 스파크 텐서 연산이 촉발되는지 검증
        """
        tc = TypeConstraint(
            constraint_id="physics_bound",
            domain_type="PhysicalSystem",
            allowed_transitions={("State_A", "State_B")}
        )
        self.bypass_mgr.register_type_constraint(tc)

        # 장력 스파이크 주입 (2.5 >= 1.0)
        proof = self.bypass_mgr.verify_topological_invariants(
            proof_id="p2_spike",
            invariant_signature="physics_bound",
            current_transition=("State_A", "State_B"),
            active_tension=2.5
        )

        self.assertEqual(proof.symmetry_state, SymmetryState.SPIKED)

        callback_ran = False
        def tensor_op():
            nonlocal callback_ran
            callback_ran = True
            return [1.0, 2.0, 3.0]  # Local spark tensor output

        is_bypassed, res = self.bypass_mgr.execute_with_static_elimination(proof, tensor_op)

        self.assertFalse(is_bypassed)
        self.assertTrue(callback_ran)
        self.assertEqual(res, [1.0, 2.0, 3.0])
        self.assertEqual(self.bypass_mgr.tensor_dispatch_count, 1)

    def test_cross_domain_isomorphic_mapping(self):
        """
        [3] 원형 메커니즘 Θ_meta (물리 포텐셜 장력 이완)를
        1) 언어적 모순 해소 도메인
        2) 하드웨어 VRAM 링버퍼 제어 도메인
        으로 동형 매핑하여 O(1) 정적 검증을 동일 통과시키는지 입증
        """
        # 1. 물리 도메인의 원형 메커니즘 Θ_meta 생성
        phys_dag = {"HighPotential": ["TensionRelease"], "TensionRelease": ["Equilibrium"]}
        phys_transitions = [("HighPotential", "TensionRelease"), ("TensionRelease", "Equilibrium")]
        phys_axioms = ["EnergyMinimization"]

        phys_sig = self.engine.extract_meta_signature_from_axioms(
            signature_id="theta_potential_relaxation",
            symmetry_group="SU(2)",
            axioms=phys_axioms,
            dag=phys_dag,
            transitions=phys_transitions
        )

        # 물리 도메인 O(1) 검증
        proof_phys, is_bypassed_phys = self.engine.verify_and_resolve_isomorphic_state(
            signature_id=phys_sig.signature_id,
            current_transition=("HighPotential", "TensionRelease"),
            tension_magnitude=0.1
        )
        self.assertTrue(proof_phys.is_valid)
        self.assertTrue(is_bypassed_phys)

        # 2. 언어 도메인으로의 동형 매핑 (Isomorphic Mapping)
        ling_entity_map = {
            "HighPotential": "LinguisticContradiction",
            "TensionRelease": "ContextualResolution",
            "Equilibrium": "CoherentMeaning"
        }
        ling_sig = self.engine.perform_isomorphic_mapping(
            source_signature_id="theta_potential_relaxation",
            target_domain_name="LinguisticDomain",
            domain_entity_mapping=ling_entity_map
        )

        # 언어 도메인 전이 O(1) 검증
        proof_ling, is_bypassed_ling = self.engine.verify_and_resolve_isomorphic_state(
            signature_id=ling_sig.signature_id,
            current_transition=("LinguisticContradiction", "ContextualResolution"),
            tension_magnitude=0.1
        )
        self.assertTrue(proof_ling.is_valid)
        self.assertTrue(is_bypassed_ling)  # 하부 연산 0으로 소멸

        # 3. 1060 3GB VRAM 링버퍼 제어 도메인으로의 동형 매핑
        vram_entity_map = {
            "HighPotential": "VRAMBufferOverflowTension",
            "TensionRelease": "RingBufferPageEviction",
            "Equilibrium": "ZeroCopyMemoryEquilibrium"
        }
        vram_sig = self.engine.perform_isomorphic_mapping(
            source_signature_id="theta_potential_relaxation",
            target_domain_name="HardwareVRAMRingBuffer",
            domain_entity_mapping=vram_entity_map
        )

        # VRAM 도메인 전이 O(1) 검증
        proof_vram, is_bypassed_vram = self.engine.verify_and_resolve_isomorphic_state(
            signature_id=vram_sig.signature_id,
            current_transition=("VRAMBufferOverflowTension", "RingBufferPageEviction"),
            tension_magnitude=0.05
        )
        self.assertTrue(proof_vram.is_valid)
        self.assertTrue(is_bypassed_vram)  # 하부 VRAM 텐서 연산 0으로 정적 소멸

        # 통계 확인: 물리, 언어, VRAM 도메인의 정상 상태 검증 결과 텐서 연산은 총 3회 소멸(Bypassed)
        self.assertEqual(self.bypass_mgr.bypassed_count, 3)


if __name__ == "__main__":
    unittest.main()
