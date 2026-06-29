import numpy as np
from synaptic_architecture.field import CrystallizationField

cf = CrystallizationField(256)
pos = np.array([128, 128])
val = np.uint64(0x1111222233334444)
cf.crystallize_gene(pos, val)
print(f"Conductance at 128,128: {cf.conductance[128, 128]}")
print(f"Gene at 128,128: {hex(cf.bit_genes[128, 128])}")
idx = np.argmax(cf.conductance)
y, x = np.unravel_index(idx, cf.conductance.shape)
print(f"Max conductance at {y}, {x}: {cf.conductance[y, x]}")
