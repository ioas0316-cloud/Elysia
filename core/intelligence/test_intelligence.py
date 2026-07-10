import pytest
import numpy as np
from core.intelligence.thought_element import ThoughtTransistor
from core.intelligence.thought_field import ThoughtField

def test_thought_transistor_activation():
    concept = np.array([1.0, 0.0])
    t = ThoughtTransistor("T1", concept)
    t.inject_energy(0.6)
    output = t.process()
    assert output > 0
    assert t.is_active is True
    assert t.energy == 0 # Discharged

def test_thought_transistor_lens_effect():
    concept = np.array([1.0, 0.0])
    t = ThoughtTransistor("T1", concept)
    t.inject_energy(0.3) # Below default 0.5 threshold

    # Without bias, it shouldn't fire
    output1 = t.process(context_bias=0.0)
    assert output1 == 0

    # With positive bias (lens effect), it should fire
    t.inject_energy(0.3)
    output2 = t.process(context_bias=0.3)
    assert output2 > 0

def test_thought_field_flow():
    field = ThoughtField()
    t1 = ThoughtTransistor("T1", np.array([1.0, 0.0]))
    t2 = ThoughtTransistor("T2", np.array([1.0, 0.0]))
    field.add_element(t1)
    field.add_element(t2)
    field.connect("T1", "T2")

    field.pulse({"T1": 1.0})
    results = field.step()

    assert "T1" in results
    # T2 should have received energy from T1
    # Note: T2 might have fired immediately if the energy exceeded threshold
    assert t2.energy > 0 or t2.is_active is True

def test_dynamic_rewiring():
    field = ThoughtField()
    t1 = ThoughtTransistor("T1", np.array([1.0, 0.0]))
    t2 = ThoughtTransistor("T2", np.array([1.0, 0.1])) # Resonant
    field.add_element(t1)
    field.add_element(t2)

    # Initially no connection
    assert "T2" not in t1.collectors

    # Step should trigger semantic coupling
    field.step()
    assert "T2" in t1.collectors
