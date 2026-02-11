import unittest
import threading

from Core.S1_Body.L4_Causality.M5_Logic.causal_admissibility_gate import (
    CausalAdmissibilityGate,
    CausalSignature,
    ResonanceLedger,
)
from Core.S1_Body.L4_Causality.causal_flow_engine import CausalFlowEngine
from Core.S1_Body.L6_Structure.M6_Architecture.holographic_memory import HolographicMemory


class TestCausalAdmissibilityGate(unittest.TestCase):
    def setUp(self):
        self.gate = CausalAdmissibilityGate()

    def _signature(self, **overrides):
        payload = dict(
            cause_id="cause-1",
            intent_vector=[1.0, 0.0, 0.5],
            result_vector=[0.9, 0.1, 0.4],
            phase_delta=0.2,
            energy_cost=0.6,
            trinary_state={"negative": 0.2, "neutral": 0.3, "positive": 0.5},
        )
        payload.update(overrides)
        return CausalSignature(**payload)

    def test_admissible_transition(self):
        rec = self.gate.evaluate(
            from_state="IGNITED",
            to_state="RESONATING",
            signature=self._signature(),
            resonance_score=0.7,
        )
        self.assertTrue(rec.admissible)
        self.assertEqual(rec.rejection_reasons, [])
        self.assertEqual(len(self.gate.quarantine), 0)

    def test_rejected_transition_goes_to_quarantine(self):
        rec = self.gate.evaluate(
            from_state="IGNITED",
            to_state="COLLAPSED",
            signature=self._signature(cause_id="", energy_cost=9.0),
        )
        self.assertFalse(rec.admissible)
        self.assertIn("missing_cause", rec.rejection_reasons)
        self.assertIn("energy_over_budget", rec.rejection_reasons)
        self.assertEqual(len(self.gate.quarantine), 1)
        self.assertGreater(self.gate.quarantine_ratio(), 0.0)

    def test_trinary_neutral_floor_enforced(self):
        rec = self.gate.evaluate(
            from_state="A",
            to_state="B",
            signature=self._signature(trinary_state={"negative": 0.49, "neutral": 0.01, "positive": 0.50}),
        )
        self.assertFalse(rec.admissible)
        self.assertIn("trinary_unstable", rec.rejection_reasons)


    def test_gate_is_thread_safe_under_parallel_evaluations(self):
        def _job(idx: int):
            self.gate.evaluate(
                from_state="IGNITED",
                to_state="RESONATING",
                signature=self._signature(cause_id=f"cause-{idx}"),
                resonance_score=0.5,
            )

        threads = [threading.Thread(target=_job, args=(i,)) for i in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        metrics = self.gate.snapshot()
        self.assertEqual(int(metrics["ledger_count"]), 32)
        self.assertEqual(int(metrics["quarantine_count"]), 0)

    def test_drain_quarantine(self):
        self.gate.evaluate(
            from_state="IGNITED",
            to_state="COLLAPSED",
            signature=self._signature(cause_id="", energy_cost=9.0),
        )
        drained = self.gate.drain_quarantine()
        self.assertEqual(len(drained), 1)
        self.assertEqual(len(self.gate.quarantine), 0)


class TestLedgerAndFlowIntegration(unittest.TestCase):
    def test_resonance_ledger_filters(self):
        gate = CausalAdmissibilityGate()
        ok = gate.evaluate(
            from_state="A",
            to_state="B",
            signature=CausalSignature(
                cause_id="c1",
                intent_vector=[1.0, 0.0],
                result_vector=[1.0, 0.0],
                phase_delta=0.1,
                energy_cost=0.3,
                trinary_state={"negative": 0.2, "neutral": 0.2, "positive": 0.6},
            ),
        )
        bad = gate.evaluate(
            from_state="B",
            to_state="C",
            signature=CausalSignature(
                cause_id="c2",
                intent_vector=[1.0, 0.0],
                result_vector=[0.0, 1.0],
                phase_delta=3.14,
                energy_cost=5.0,
                trinary_state={"negative": 0.8, "neutral": 0.0, "positive": 0.2},
            ),
        )

        ledger = ResonanceLedger([ok, bad])
        self.assertEqual(len(ledger.rejected()), 1)
        self.assertEqual(len(ledger.by_cause("c1")), 1)

    def test_causal_flow_engine_exposes_gate_evaluation(self):
        memory = HolographicMemory(dimension=8)
        engine = CausalFlowEngine(memory)
        rec = engine.evaluate_transition(
            from_state="IGNITED",
            to_state="RESONATING",
            cause_id="flow-cause",
            intent_vector=[1.0, 0.0, 0.0],
            result_vector=[1.0, 0.0, 0.0],
            phase_delta=0.2,
            energy_cost=0.4,
            trinary_state={"negative": 0.2, "neutral": 0.2, "positive": 0.6},
            resonance_score=0.9,
        )
        self.assertTrue(rec.admissible)
        self.assertEqual(rec.from_state, "IGNITED")

    def test_collapse_guarded_blocks_rejected_transition(self):
        memory = HolographicMemory(dimension=8)
        engine = CausalFlowEngine(memory)
        packet = {"seed": "BlockedIdea", "flow_type": "DISSONANCE", "amplitude": 0.2}

        out = engine.collapse_guarded(
            packet,
            cause_id="",
            intent_vector=[1.0, 0.0],
            result_vector=[0.0, 1.0],
            phase_delta=3.14,
            energy_cost=9.0,
        )
        self.assertIn("[QUARANTINE]", out)
        self.assertEqual(len(engine.gate.quarantine), 1)



    def test_quarantine_worker_pulls_records(self):
        memory = HolographicMemory(dimension=8)
        engine = CausalFlowEngine(memory)
        packet = {"seed": "BlockedIdea", "flow_type": "DISSONANCE", "amplitude": 0.2}

        engine.collapse_guarded(
            packet,
            cause_id="",
            intent_vector=[1.0, 0.0],
            result_vector=[0.0, 1.0],
            phase_delta=3.14,
            energy_cost=9.0,
        )
        moved = engine._drain_quarantine_once()
        self.assertEqual(moved, 1)
        pulled = engine.pull_quarantined()
        self.assertEqual(len(pulled), 1)
        self.assertFalse(pulled[0].admissible)



    def test_process_quarantine_batch_assigns_recovery_action(self):
        memory = HolographicMemory(dimension=8)
        engine = CausalFlowEngine(memory)
        packet = {"seed": "BlockedIdea", "flow_type": "DISSONANCE", "amplitude": 0.2}

        engine.collapse_guarded(
            packet,
            cause_id="",
            intent_vector=[1.0, 0.0],
            result_vector=[0.0, 1.0],
            phase_delta=3.14,
            energy_cost=9.0,
        )

        decisions = engine.process_quarantine_batch(max_items=4)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["action"], "await_cause_enrichment")
        self.assertIn("missing_cause", decisions[0]["reasons"])



    def test_process_quarantine_batch_auto_execute_recovers_energy_case(self):
        memory = HolographicMemory(dimension=8)
        engine = CausalFlowEngine(memory)
        packet = {"seed": "EnergySpike", "flow_type": "HARMONY", "amplitude": 0.95}

        out = engine.collapse_guarded(
            packet,
            cause_id="cause-energy",
            intent_vector=[1.0, 0.0],
            result_vector=[1.0, 0.0],
            phase_delta=0.1,
            energy_cost=1.5,
        )
        self.assertIn("[QUARANTINE]", out)

        decisions = engine.process_quarantine_batch(max_items=4, auto_execute=True, retry_budget=2)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["action"], "throttle_and_retry")
        self.assertEqual(decisions[0]["execution_status"], "recovered")
        self.assertTrue(decisions[0]["recovered"])

    def test_collapse_guarded_allows_admissible_transition(self):
        memory = HolographicMemory(dimension=8)
        engine = CausalFlowEngine(memory)
        memory.imprint("AlignedIdea", intensity=1.0)
        packet = {"seed": "AlignedIdea", "flow_type": "HARMONY", "amplitude": 0.95}

        out = engine.collapse_guarded(
            packet,
            cause_id="cause-ok",
            intent_vector=[1.0, 0.0],
            result_vector=[1.0, 0.0],
            phase_delta=0.2,
            energy_cost=0.3,
        )
        self.assertIn("[MANIFEST]", out)
        self.assertEqual(len(engine.gate.quarantine), 0)


if __name__ == "__main__":
    unittest.main()
