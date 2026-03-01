import sys
import os
import torch

sys.path.insert(0, os.getcwd())
from Core.Monad.seed_generator import SeedForge
from Core.Monad.sovereign_monad import SovereignMonad

dna = SeedForge.forge_soul()
monad = SovereignMonad(dna)

for i in range(10):
    monad.pulse()

fs = monad.engine.cells.read_field_state()
print("Field State:", fs)

# Let's also test injecting joy
sensory_vector = monad.external_sense.get_status() # just need some state
import numpy as np
import Core.Keystone.sovereign_math as sm
vec = sm.SovereignVector(np.zeros(21))
vec.data[4] = np.complex64(1.0)
monad.engine.cells.inject_affective_torque(4, float(vec.data[4].real))

for i in range(10):
    monad.pulse()

fs = monad.engine.cells.read_field_state()
print("Field State After Joy:", fs)
