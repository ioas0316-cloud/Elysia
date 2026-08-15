import pytest
import numpy as np
from core.memory.state_dag import StateDAGManager, PhysicalStateSlabPool
from core.memory.causal_gc import CausalAwareGC

def test_physical_slab_pool():
    pool = PhysicalStateSlabPool(capacity=100, dimension=16)
    delta1 = np.ones(16, dtype=np.float32) * 2.0
    off1 = pool.allocate_slab(delta1, parent_offset=-1)
    assert off1 == 0
    state1 = pool.get_slab_state(off1)
    assert np.allclose(state1, 2.0)

    delta2 = np.ones(16, dtype=np.float32) * 3.0
    off2 = pool.allocate_slab(delta2, bitmask=0xFFFF, parent_offset=off1)
    assert off2 == 1

def test_state_dag_manager_step_and_rewind():
    dag = StateDAGManager({"temp": 20.0, "status": "INIT"}, state_dim=16)
    root_id = dag.root.id

    s1 = dag.step({"temp": 25.0})
    assert s1.get_state_chain() == {"temp": 25.0, "status": "INIT"}

    s2 = dag.step({"status": "RUNNING"})
    assert s2.get_state_chain() == {"temp": 25.0, "status": "RUNNING"}

    # Rewind to root
    dag.rewind_to(root_id)
    assert dag.current_node.id == root_id
    assert dag.current_node.get_state_chain() == {"temp": 20.0, "status": "INIT"}

def test_do_intervention_branching():
    dag = StateDAGManager({"temp": 20.0, "status": "INIT"}, state_dim=16)
    s1 = dag.step({"temp": 25.0})
    s1_id = s1.id

    # Intervention
    inter = dag.do_intervention("status", "OVERRIDE")
    assert inter.get_state_chain() == {"temp": 25.0, "status": "OVERRIDE"}
    assert inter.parent.id == s1_id

def test_causal_aware_gc():
    dag = StateDAGManager({"temp": 20.0}, state_dim=16)
    gc = CausalAwareGC(dag, threshold_gc=50.0)

    s1 = dag.step({"temp": 22.0})
    dead_branch = dag.step({"temp": 22.01})

    dag.rewind_to(s1.id)
    active_branch = dag.step({"temp": 100.0})

    pruned = gc.run_cgc()
    assert pruned == 1
    assert dead_branch.id not in dag.nodes
    assert active_branch.id in dag.nodes
