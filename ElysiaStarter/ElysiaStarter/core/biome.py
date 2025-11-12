import numpy as np

BIOMES = [
    "ocean","tundra","taiga","grassland","forest","desert","tropical","alpine"
]

def classify_biome(height, moisture, temp, sea_level=0.45):
    h, w = height.shape
    out = np.full((h,w), -1, dtype=np.int16)
    ocean = height < sea_level
    out[ocean] = 0  # ocean

    land = ~ocean
    # 간단 규칙: 온도-수분으로 분류
    t = temp[land]; m = moisture[land]
    idx = np.zeros_like(t, dtype=np.int16)
    idx[(t<0.2) & (m<0.5)] = 1  # tundra
    idx[(t<0.4) & (m>=0.5)] = 2 # taiga
    idx[(t>=0.4) & (m<0.4)] = 3 # grassland
    idx[(t>=0.4) & (m>=0.6)] = 4 # forest
    idx[(t>=0.6) & (m<0.2)] = 5 # desert
    idx[(t>=0.7) & (m>=0.5)] = 6 # tropical
    idx[(height[land]>0.8)] = 7 # alpine (고도)
    out[land] = idx
    return out
