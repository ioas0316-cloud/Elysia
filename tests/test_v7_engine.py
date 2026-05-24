from core.linguistic_axiom import LinguisticAxiomFilter
from core.hologram_sphere import HologramSphere
import numpy as np
from core.math_utils import Quaternion

def test_linguistic_axiom_korean():
    rotor = LinguisticAxiomFilter.analyze_text_axiom('안녕')
    assert isinstance(rotor, Quaternion)
    # Check normalization
    assert np.isclose(rotor.w**2 + rotor.x**2 + rotor.y**2 + rotor.z**2, 1.0)

def test_linguistic_axiom_english():
    rotor = LinguisticAxiomFilter.analyze_text_axiom('hello')
    assert isinstance(rotor, Quaternion)
    assert np.isclose(rotor.w**2 + rotor.x**2 + rotor.y**2 + rotor.z**2, 1.0)

def test_hologram_sphere():
    hs = HologramSphere(size=16)
    hs.populate_manifold('test data')
    grid, score = hs.condense_sphere('생명')

    assert grid.shape == (16, 16)
    assert 0 <= score <= 100
