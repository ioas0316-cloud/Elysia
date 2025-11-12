from core.cell_world import CellWorld
from core.biome import classify_biome
from core.agents import Agents

def test_fields_and_biome():
    w = CellWorld(64,64)
    h,m,t = w.update_fields()
    b = classify_biome(h,m,t)
    assert b.shape==(64,64)

def test_agents_step():
    w = CellWorld(64,64)
    h,m,t = w.update_fields()
    b = classify_biome(h,m,t)
    a = Agents(1000,64,64)
    a.step(b, dt=1.0)
    assert a.x.shape[0]==1000
