"""
Integration Test Suite for Executable Causal Topology & Autonomous Active Inference Engine
==========================================================================================
Tests:
1. Executable DAG Topology, Topological Sorting, Dirty Flag Propagation & SoA Vectorized Batch Evaluator
2. SCM do-operator Graph Surgery & Sandbox Isolation
3. Compound Node Encapsulation, Component Pattern, JSON Serialization & Code Generators
4. Active Inference POMDP (F & G decomposition) & Continuous Active Inference Reflex Arc
5. Autonomous Causal Discovery (Slot Attention, Invariance, NOTEARS, MDL Pruning, Surprise Graph Rewriting)
6. InformationTopology Manifold Bridge to Executable SCM
"""

import unittest
import json
import numpy as np

from core.topology.executable_causal_topology import (
    StructuralCausalModel, ExecutableDAGNode, NodeType, OpCode,
    CausalCompiler, SoACausalEvaluator, CompoundNode, CausalComponent,
    CausalSerializer, CausalCodeGenerator
)
from core.topology.autonomous_causal_discovery import (
    SlotAttentionDisentangler, InvarianceDetector, ActiveInferenceAgent,
    ContinuousActiveInferenceReflexArc, DifferentiableCausalDiscovery,
    MDLPruner, SurpriseGraphRewriter
)
from core.topology.causal_structure import InformationTopology, CausalNumber, TopologyLink


class TestAutonomousCausalTopology(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_executable_causal_topology_dag_and_soa(self):
        scm = StructuralCausalModel("CombatDamageSCM")
        scm.add_node(ExecutableDAGNode(id="attack", node_type=NodeType.VALUE, op=OpCode.INPUT_VAR, default_value=50.0))
        scm.add_node(ExecutableDAGNode(id="multiplier", node_type=NodeType.VALUE, op=OpCode.INPUT_VAR, default_value=1.5))
        scm.add_node(ExecutableDAGNode(id="raw_dmg", node_type=NodeType.COMPUTE, op=OpCode.MULTIPLY, input_ids=["attack", "multiplier"]))
        scm.add_node(ExecutableDAGNode(id="final_dmg", node_type=NodeType.COMPUTE, op=OpCode.CLAMP_MIN, default_value=10.0, input_ids=["raw_dmg"]))

        # Compile SoA Program
        program = CausalCompiler.compile(scm)
        self.assertEqual(program.num_nodes, 4)

        # Batch Simulation
        evaluator = SoACausalEvaluator(program, batch_size=100)
        evaluator.execute_vectorized_batch()
        results = evaluator.get_node_result("final_dmg")
        self.assertEqual(len(results), 100)
        self.assertAlmostEqual(results[0], 75.0, places=4)

    def test_do_operator_graph_surgery(self):
        scm = StructuralCausalModel("OrigSCM")
        scm.add_node(ExecutableDAGNode(id="X", node_type=NodeType.VALUE, op=OpCode.INPUT_VAR, default_value=10.0))
        scm.add_node(ExecutableDAGNode(id="Y", node_type=NodeType.COMPUTE, op=OpCode.MULTIPLY, default_value=2.0, input_ids=["X"]))

        # Original evaluation (X=10 -> Y=10*2=20 or X*2)
        # Apply intervention do(X=50)
        do_scm = scm.do_intervention({"X": 50.0})

        # Ensure original SCM is unchanged
        self.assertEqual(scm.nodes["X"].default_value, 10.0)
        self.assertEqual(do_scm.nodes["X"].default_value, 50.0)
        self.assertEqual(len(do_scm.nodes["X"].input_ids), 0)

        # Evaluate sandbox snapshot
        prog = CausalCompiler.compile(do_scm)
        evaluator = SoACausalEvaluator(prog, batch_size=10)
        evaluator.execute_vectorized_batch()
        res = evaluator.get_node_result("X")
        self.assertAlmostEqual(res[0], 50.0, places=4)

    def test_compound_node_and_serialization(self):
        # Build inner subgraph
        inner1 = ExecutableDAGNode(id="in1", node_type=NodeType.VALUE, op=OpCode.INPUT_VAR, default_value=5.0)
        inner2 = ExecutableDAGNode(id="in2", node_type=NodeType.VALUE, op=OpCode.INPUT_VAR, default_value=3.0)
        inner3 = ExecutableDAGNode(id="out1", node_type=NodeType.COMPUTE, op=OpCode.ADD, input_ids=["in1", "in2"])

        compound = CompoundNode(
            node_id="compound_calc",
            sub_nodes=[inner1, inner2, inner3],
            input_pins={"pin_a": "in1", "pin_b": "in2"},
            output_pins={"pin_out": "out1"}
        )

        res = compound.evaluate_subgraph({"pin_a": 10.0, "pin_b": 20.0})
        self.assertAlmostEqual(res["pin_out"], 30.0, places=4)

        # Test Serialization
        scm = StructuralCausalModel("SerialSCM")
        scm.add_node(ExecutableDAGNode(id="a", node_type=NodeType.VALUE, op=OpCode.CONSTANT, default_value=7.0))
        json_str = CausalSerializer.serialize_scm(scm)
        restored_scm = CausalSerializer.deserialize_scm(json_str)
        self.assertEqual(restored_scm.nodes["a"].default_value, 7.0)

        # Test Code Generators
        cpp_hdr = CausalCodeGenerator.generate_cpp_header(scm)
        py_mod = CausalCodeGenerator.generate_python_module(scm)
        report = CausalCodeGenerator.generate_causal_report(scm)
        self.assertIn("struct SerialSCMSoAProgram", cpp_hdr)
        self.assertIn("class SerialSCMEvaluator", py_mod)
        self.assertIn("# Causal Flow Report: SerialSCM", report)

    def test_active_inference_pomdp_and_continuous_reflex(self):
        # Discrete POMDP Test
        A = np.array([[0.9, 0.1], [0.1, 0.9]]) # Likelihood
        B = np.zeros((2, 2, 2))                # Transition: [s_next, s_curr, action]
        B[:, :, 0] = np.eye(2)
        B[:, :, 1] = np.array([[0, 1], [1, 0]])
        C = np.array([0.0, 3.0])               # Strong preference for observation 1
        D = np.array([0.9, 0.1])               # Initial belief state 0

        agent = ActiveInferenceAgent(A, B, C, D)
        action = agent.select_action(obs_idx=0, policies=[[0], [1]])
        # Action 1 flips state to 1, satisfying strong preference for obs 1
        self.assertEqual(action, 1)

        # Continuous Reflex Arc Test
        reflex = ContinuousActiveInferenceReflexArc(dt=0.01)
        for _ in range(200):
            state = reflex.step(target_mu_d=5.0)
        self.assertGreater(state["x"], 2.0)
        self.assertGreater(state["mu"], 2.0)

    def test_autonomous_causal_discovery_and_rewriting(self):
        # 1. Slot Disentanglement
        disentangler = SlotAttentionDisentangler(num_slots=2, slot_dim=4)
        raw_obs = np.random.normal(0, 1, size=(4, 8))
        slots = disentangler.extract_slots(raw_obs)
        self.assertEqual(slots.shape, (4, 2, 4))

        # 2. Invariance Detector
        invariant_idx = InvarianceDetector.filter_invariant_features([slots, slots])
        self.assertIn(0, invariant_idx)

        # 3. Differentiable DAG Discovery & MDL Pruning
        data = np.random.normal(0, 1, size=(50, 3))
        data[:, 1] += 0.9 * data[:, 0] # Synthetic causal link 0 -> 1
        W = DifferentiableCausalDiscovery.discover_dag(data, max_iter=20)
        W_pruned = MDLPruner.prune_edges(W, data, mdl_penalty=0.1)
        self.assertEqual(W_pruned.shape, (3, 3))

        # 4. Surprise Graph Rewriting
        scm = StructuralCausalModel("SurpriseSCM")
        scm.add_node(ExecutableDAGNode(id="n1", default_value=1.0))
        scm.add_node(ExecutableDAGNode(id="n2", default_value=1.0))

        rewriter = SurpriseGraphRewriter(scm, surprise_threshold=0.1)
        obs_data = {"n1": 1.0, "n2": 10.0}
        pred_data = {"n1": 1.0, "n2": 1.0}

        modified = rewriter.evaluate_and_rewrite(obs_data, pred_data)
        self.assertTrue(modified)
        self.assertIn("n1", scm.nodes["n2"].input_ids)

    def test_information_topology_bridge(self):
        topo = InformationTopology("BridgeTopology")
        topo.add_number(CausalNumber(id="base", value=10.0, sequence_index=0, magnitude=10.0, gradient_tension=0.0, chromatic_vector=np.array([0.1, 0.1, 0.1])))
        topo.add_number(CausalNumber(id="bonus", value=5.0, sequence_index=1, magnitude=5.0, gradient_tension=0.0, chromatic_vector=np.array([0.1, 0.1, 0.1])))
        topo.add_number(CausalNumber(id="total", value=0.0, sequence_index=2, magnitude=0.0, gradient_tension=0.0, chromatic_vector=np.array([0.1, 0.1, 0.1])))

        topo.add_link(TopologyLink("base", "total", "causal", 1.0, 0.0))
        topo.add_link(TopologyLink("bonus", "total", "causal", 1.0, 0.0))

        scm = topo.to_executable_scm()
        self.assertEqual(set(scm.nodes["total"].input_ids), {"base", "bonus"})

        prog = CausalCompiler.compile(scm)
        evaluator = SoACausalEvaluator(prog, batch_size=1)
        evaluator.execute_vectorized_batch()
        res = evaluator.get_node_result("total")
        self.assertAlmostEqual(res[0], 15.0, places=4)


if __name__ == "__main__":
    unittest.main()
