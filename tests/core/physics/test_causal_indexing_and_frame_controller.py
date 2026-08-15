import pytest
import numpy as np
from core.memory.state_dag import StateDAGManager
from core.physics.counterfactual_branching import CounterfactualBranchingEngine
from core.physics.frame_controller import ObservationalFrameController

def test_counterfactual_branching():
    dag = StateDAGManager({"temp": 20.0, "pressure": 1.0}, state_dim=16)
    engine = CounterfactualBranchingEngine(dag)

    s1 = dag.step({"temp": 25.0})
    inter_node = engine.apply_do_operator("pressure", 5.0, affected_dimensions=[0, 1])

    assert inter_node.get_state_chain()["pressure"] == 5.0

    cone_delta = engine.compute_causal_cone_delta(s1.id, inter_node.id)
    assert "pressure" in cone_delta["cone_delta_dict"]
    assert cone_delta["cone_delta_dict"]["pressure"]["baseline"] == 1.0
    assert cone_delta["cone_delta_dict"]["pressure"]["intervened"] == 5.0

def test_frame_controller_operations():
    dag = StateDAGManager({"temp": 20.0, "status": "NORMAL"}, state_dim=16)
    ctrl = ObservationalFrameController(dag)

    # Step with frame lock
    ctrl.frame_lock("status", "LOCKED")
    n1 = ctrl.observe_step({"temp": 30.0})
    assert n1.get_state_chain() == {"temp": 30.0, "status": "LOCKED"}

    # Pause
    ctrl.pause()
    n2 = ctrl.observe_step({"temp": 40.0})
    assert n2.id == n1.id  # Deferred step

    ctrl.resume()
    n3 = ctrl.observe_step({"temp": 40.0})
    assert n3.get_state_chain()["temp"] == 40.0

    # Horizon monitoring
    horizon_info = ctrl.monitor_causal_horizon(dag.root.id, n3.id)
    assert "horizon_distance" in horizon_info
    assert horizon_info["within_horizon"] is True
