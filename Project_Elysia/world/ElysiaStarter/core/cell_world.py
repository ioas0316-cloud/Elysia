import numpy as np

class CellWorld:
    """
    연속장 생성: energy/phase/delta -> height/moisture/temp
    CPU/NumPy 레퍼런스. CUDA는 core/cuda/runtime.py 백엔드로 대체 가능.
    """
    def __init__(self, w=256, h=256, seed=42):
        self.w, self.h = w, h
        rng = np.random.default_rng(seed)
        # 초기 energy/phase ([-1,1] 범위)
        self.energy = rng.uniform(-1, 1, (h, w)).astype(np.float32)
        self.phase  = rng.uniform(-np.pi, np.pi, (h, w)).astype(np.float32)
        self.prev_energy = self.energy.copy()
        # 출력 필드
        self.height   = np.zeros((h, w), dtype=np.float32)
        self.moisture = np.zeros((h, w), dtype=np.float32)
        self.temp     = np.zeros((h, w), dtype=np.float32)
        # resource fields (derived from biome proxies)
        self.wood_density = np.zeros((h, w), dtype=np.float32)
        self.food_density = np.zeros((h, w), dtype=np.float32)
        # affective/visible energy fields
        self.aether = np.zeros((h, w), dtype=np.float32)
        self.mana = np.zeros((h, w), dtype=np.float32)
        self.is_night = False  # external driver may toggle this
        # optional semantic context
        self.context = None

    def laplacian(self, x):
        # 3x3 합성곱으로 근사 라플라시안 (주기 경계조건)
        up    = np.roll(x, -1, axis=0)
        down  = np.roll(x,  1, axis=0)
        left  = np.roll(x,  1, axis=1)
        right = np.roll(x, -1, axis=1)
        return (up + down + left + right - 4 * x)

    def update_fields(self, k1=0.8, k2=0.6, k3=0.4):
        # delta (변화율)로 시간 감각
        delta = self.energy - self.prev_energy
        self.prev_energy = self.energy.copy()

        # 간단한 내적 공명(이웃 평균으로 수렴)
        neighbor_avg = (np.roll(self.energy,1,0)+np.roll(self.energy,-1,0)+
                        np.roll(self.energy,1,1)+np.roll(self.energy,-1,1)) * 0.25
        self.energy += 0.05 * (neighbor_avg - self.energy)
        self.phase = np.sin(self.energy)

        # 연속장으로 사상
        self.height   = k1 * self.laplacian(self.energy)
        self.moisture = k2 * (1.0 - np.abs(self.phase))
        self.temp     = k3 * (-1.0 * delta)  # 변화율 낮음 -> 온난/안정

        # 정규화
        def norm(a):
            a = (a - a.min()) / (a.max() - a.min() + 1e-6)
            return a.astype(np.float32)
        self.height   = norm(self.height)
        self.moisture = norm(self.moisture)
        self.temp     = norm(self.temp)

        # derive simple resource maps from moisture/height/temp
        forest_mask = (self.moisture > 0.6) & (self.height > 0.45)
        self.wood_density = forest_mask.astype(np.float32) * (0.3 + 0.7 * self.moisture)
        self.food_density = (~forest_mask).astype(np.float32) * (0.2 + 0.5 * (1.0 - self.temp))

        # Aether (affect/romance energy): regen with night boost
        # regen uses moisture + temp stability + bias; decay light
        regen = 0.002 * (0.4 * self.moisture + 0.3 * (1.0 - np.abs(0.5 - self.temp)) + 0.3)
        if bool(getattr(self, "is_night", False)):
            regen = regen * 1.8
        self.aether = np.clip(self.aether + regen - 0.001, 0.0, 1.0)

        # Mana (game-like visible resource): low baseline spawn near forest/water
        mana_base = 0.0015 * (0.5 * self.moisture + 0.2 * (self.height > 0.45) + 0.3 * (np.sin(self.energy) * 0.5 + 0.5))
        self.mana = np.clip(self.mana + mana_base - 0.0008, 0.0, 1.0)

        return self.height, self.moisture, self.temp

    # Context injection: store guidance used by higher layers (optional)
    def inject_context(self, ctx: dict):
        self.context = ctx
        return True

    # --- Aether/Mana consumption helpers ---
    def consume_aether(self, x: float, y: float, amount: float = 0.05, r: int = 6) -> float:
        H, W = self.aether.shape
        xi, yi = int(x), int(y)
        x0, x1 = max(0, xi - r), min(W, xi + r + 1)
        y0, y1 = max(0, yi - r), min(H, yi + r + 1)
        patch = self.aether[y0:y1, x0:x1]
        used = min(amount, float(patch.mean()) if patch.size else 0.0)
        if patch.size:
            patch[...] = np.clip(patch - used, 0.0, 1.0)
        return used

    def spend_mana(self, x: float, y: float, amount: float = 0.1, r: int = 4) -> float:
        H, W = self.mana.shape
        xi, yi = int(x), int(y)
        x0, x1 = max(0, xi - r), min(W, xi + r + 1)
        y0, y1 = max(0, yi - r), min(H, yi + r + 1)
        patch = self.mana[y0:y1, x0:x1]
        have = float(patch.mean()) if patch.size else 0.0
        spent = min(amount, have)
        if patch.size:
            patch[...] = np.clip(patch - spent, 0.0, 1.0)
        return spent

    # Backward-compatibility: legacy 'consume_mana' maps to aether consumption
    def consume_mana(self, x: float, y: float, amount: float = 0.05, r: int = 6) -> float:
        return self.consume_aether(x, y, amount, r)
