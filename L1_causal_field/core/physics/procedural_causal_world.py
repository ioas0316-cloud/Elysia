"""
procedural_causal_world.py
===========================
Cosmic Planetary Procedural Causal World (거시 천체 로터 기반 프랙탈 인과 월드)

핵심 철학:
- "Do not calculate, let it flow."
- "상위의 거시적 경계조건(Macro Boundary Constraints)이 하위의 미시적 자유도(Micro Degrees of Freedom)를 결정한다."
- 인위적인 난수(Random Noise)를 완전히 배제하고, 태양/달/자전축의 거시 천체 로터(Cosmic Macro Rotor)부터
  하위 미시 로터까지 하향 분화되는 프랙탈 인과 로터 연쇄(Fractal Rotor Cascade: S1 -> S2 -> ... -> Sn)로
  세상 전체를 연속 파동(Wave Packet)과 위상 공명(Phase Resonance)으로 구성합니다.
- 수십만 번의 입자 기반 무차별 대입(Brute-force) 시뮬레이션을 지양하고,
  다층적 경계층(지각-수권-크로매틱 기후장)의 상호작용 평형 방정식 F(Phi, Theta) = 0을
  단 한 번의 대수적 장 연산(Closed-Form Analytical Evaluation)으로 도출합니다.
- 관측자의 관측 렌즈(Scale Focus)에 따라 로터 연쇄 깊이를 동적으로 조정하는 인과적 LOD(Causal Level of Detail)를 지원합니다.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from core.physics.causal_mmorpg_sandbox import ContinuousWorldManifold, CausalSandboxAgent


class CosmicMacroRotor:
    """
    [Cosmic Macro Rotor (S1)]
    태양, 달, 자전축의 SO(3) 쿼터니언 회전 위상차(Phase Shift)를 바탕으로
    가상 월드의 일주 주기, 계절, 기조력(밀물/썰물), 태양 복사에너지를 생성하는 근본 인과 시계.
    """
    def __init__(self, day_length: float = 86400.0, year_length: float = 365.25 * 86400.0, axial_tilt: float = 0.409):
        self.day_length = day_length
        self.year_length = year_length
        self.axial_tilt = axial_tilt # ~23.4 degrees in radians

    def get_cosmic_state(self, t: float) -> Dict[str, Any]:
        """
        시간 t에 따른 천체 로터의 위상 상태를 대수적으로 구합니다.
        """
        # Diurnal phase (0 to 2*pi)
        diurnal_phase = (2.0 * np.pi * (t % self.day_length)) / self.day_length
        # Seasonal phase (0 to 2*pi)
        seasonal_phase = (2.0 * np.pi * (t % self.year_length)) / self.year_length
        # Lunar phase (~29.53 days cycle)
        lunar_cycle = 29.53 * self.day_length
        lunar_phase = (2.0 * np.pi * (t % lunar_cycle)) / lunar_cycle

        # Solar vector in 3D
        sun_elevation = np.sin(diurnal_phase) * np.cos(self.axial_tilt * np.sin(seasonal_phase))
        sun_azimuth = np.cos(diurnal_phase)
        sun_vector = np.array([sun_azimuth, np.sin(diurnal_phase) * np.sin(self.axial_tilt * np.sin(seasonal_phase)), sun_elevation], dtype=np.float32)
        sun_vector_norm = np.linalg.norm(sun_vector)
        if sun_vector_norm > 1e-6:
            sun_vector /= sun_vector_norm

        # Solar radiation intensity (0 at night, max 1.0 at zenith)
        insolation = max(0.0, float(sun_elevation))

        # Tidal force magnitude (Combined solar and lunar gravity resonance)
        tidal_force = 0.7 * np.cos(lunar_phase) + 0.3 * np.cos(diurnal_phase)

        # Seasonal temperature shift (-1.0 to 1.0)
        seasonal_temp = np.sin(seasonal_phase)

        return {
            "diurnal_phase": float(diurnal_phase),
            "seasonal_phase": float(seasonal_phase),
            "lunar_phase": float(lunar_phase),
            "sun_vector": sun_vector.tolist(),
            "insolation": insolation,
            "tidal_force": float(tidal_force),
            "seasonal_temp": float(seasonal_temp)
        }


class FractalRotorCascade:
    """
    [Fractal Rotor Cascade (S1 -> S2 -> ... -> Sn)]
    난수(Random Noise)를 완전히 배제하고, 상위 거시 로터의 위상 오메가(omega)와 회전각 시타(theta)를
    하향 분화시키는 무난수 자기유사 결정론적 파동 연쇄 방정식:
    F_Total = sum_{k=1}^n A_k * cos(2^k * omega_k * (x * cos(theta_k) + y * sin(theta_k)) + phi_k(t))
    """
    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    def evaluate_rotor_wave(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        t: float = 0.0,
        scale_depth: int = 5,
        base_frequency: float = 0.005,
        persistence: float = 0.5,
        lacunarity: float = 2.0
    ) -> np.ndarray:
        """
        [Pure Self-Similar Deterministic Rotor Wave]
        난수 생성기(Random) 호출 없이, 시드 및 천체 파동 하모닉스에 바탕한
        위상 회전 합산으로 지형 및 포텐셜장을 생성합니다.
        """
        shape = x_grid.shape
        total_field = np.zeros(shape, dtype=np.float32)
        amplitude = 1.0
        frequency = base_frequency
        max_amplitude = 0.0

        for k in range(1, scale_depth + 1):
            # Deterministic phase angle derived from seed, scale layer k
            theta_k = ((self.seed * 13 + k * 101) % 360) * (np.pi / 180.0)
            phi_k = ((self.seed * 37 + k * 211) % 1000) / 1000.0 * 2.0 * np.pi + 0.1 * t * (1.0 / k)

            proj = x_grid * np.cos(theta_k) + y_grid * np.sin(theta_k)
            rotor_component = amplitude * np.cos(frequency * proj + phi_k)

            # Cross-phase modulation with orthogonal rotor component
            theta_ortho = theta_k + np.pi / 2.0
            proj_ortho = x_grid * np.cos(theta_ortho) + y_grid * np.sin(theta_ortho)
            rotor_cross = np.sin(frequency * proj_ortho * 0.7 + phi_k * 0.5)

            total_field += rotor_component * (1.0 + 0.3 * rotor_cross)
            max_amplitude += amplitude * 1.3

            amplitude *= persistence
            frequency *= lacunarity

        return total_field / (max_amplitude + 1e-6)

    def evaluate_domain_warping(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        t: float = 0.0,
        scale_depth: int = 4,
        warp_strength: float = 30.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applying continuous self-similar rotor wave domain warping.
        """
        q_x = self.evaluate_rotor_wave(x_grid, y_grid, t=t, scale_depth=scale_depth, base_frequency=0.003)
        q_y = self.evaluate_rotor_wave(x_grid + 100.0, y_grid + 100.0, t=t, scale_depth=scale_depth, base_frequency=0.003)

        warp_x = x_grid + warp_strength * q_x
        warp_y = y_grid + warp_strength * q_y

        return warp_x, warp_y


class PlanetaryDynamo:
    """
    [Planetary Dynamo]
    행성 내부 코어와 대기층의 전자기장/중력 결합 장.
    유체 대순환 및 외부 우주선 차단 프로필을 구동합니다.
    """
    def __init__(self, core_mass: float = 5.97e24, magnetic_field_strength: float = 1.0):
        self.core_mass = core_mass
        self.magnetic_field_strength = magnetic_field_strength

    def get_dynamo_field(self, pos_3d: np.ndarray, cosmic_state: Dict[str, Any]) -> Dict[str, float]:
        """
        좌표 및 천체 로터 상태에 대한 전자기/대기 구속 프로필을 산출합니다.
        """
        r = np.linalg.norm(pos_3d) + 1.0
        gravity_acc = 9.81 / (r ** 2)
        mag_shield = self.magnetic_field_strength * (1.0 / (1.0 + 0.01 * r))
        coriolis_effect = np.sin(pos_3d[1] * 0.01) * 0.1 # Latitude dependency
        return {
            "gravity": float(gravity_acc),
            "mag_shield": float(mag_shield),
            "coriolis": float(coriolis_effect)
        }


class TopologicalPhaseValidator:
    """
    [Topological & Phase Resonance Validator]
    생성된 결합 장(Coupled Field)의 위상 연속성, 경계 조건, 에너지 보존 법칙을 검증하고,
    비정상적 단층이나 불통과 이상 위상을 자동으로 사영 치유(Phase Healing)하는 이중 검증 레이어.
    """
    def __init__(self, max_slope_threshold: float = 2.5):
        self.max_slope_threshold = max_slope_threshold

    def validate_and_heal_field(
        self,
        heightmap: np.ndarray,
        flow_accumulation: np.ndarray,
        chromatic_field: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        지형 및 유량장의 경사도/위상 이상 상태를 검사하고,
        위상 치유(Phase Healing) 필터를 통해 평형을 유지한 지형 및 검증 리포트를 반환합니다.
        """
        healed_heightmap = heightmap.copy()

        # Calculate gradients
        gy, gx = np.gradient(healed_heightmap)
        slope_magnitude = np.sqrt(gx**2 + gy**2)

        # Detect cliff / step discontinuities exceeding max slope threshold
        invalid_slopes = slope_magnitude > self.max_slope_threshold
        num_invalid = int(np.sum(invalid_slopes))

        # Perform Phase Healing (Laplacian relaxation smoothing on invalid steep areas)
        if num_invalid > 0:
            kernel = np.array([[0.05, 0.1, 0.05],
                              [0.1,  0.4, 0.1],
                              [0.05, 0.1, 0.05]], dtype=np.float32)

            pad_h = np.pad(healed_heightmap, 1, mode='edge')
            smoothed = (
                pad_h[:-2, :-2]*kernel[0,0] + pad_h[:-2, 1:-1]*kernel[0,1] + pad_h[:-2, 2:]*kernel[0,2] +
                pad_h[1:-1, :-2]*kernel[1,0] + pad_h[1:-1, 1:-1]*kernel[1,1] + pad_h[1:-1, 2:]*kernel[1,2] +
                pad_h[2:, :-2]*kernel[2,0] + pad_h[2:, 1:-1]*kernel[2,1] + pad_h[2:, 2:]*kernel[2,2]
            )
            healed_heightmap[invalid_slopes] = smoothed[invalid_slopes]

        # Check chromatic resonance conservation (RGB sum normalized)
        chromatic_sums = np.sum(chromatic_field, axis=-1)
        resonance_valid = bool(np.all(np.abs(chromatic_sums - 1.0) < 1e-3))

        report = {
            "is_valid": bool(num_invalid == 0 and resonance_valid),
            "invalid_slope_count": num_invalid,
            "max_slope_observed": float(np.max(slope_magnitude)),
            "healed_cells_count": num_invalid,
            "chromatic_resonance_valid": resonance_valid
        }

        return healed_heightmap, report


class ProceduralCausalWorld:
    """
    [Procedural Causal World Generator]
    단일 고차원 시드(Seed)와 거시 천체 로터(Cosmic Macro Rotor) 및 프랙탈 로터 연쇄를 결합하여
    수십만 번의 입자 반복 없이 $O(1)$ 대수적 결합 평형 방정식 $F(\\Phi, \\Theta) = 0$으로
    온디맨드 인과 가상 월드(지각-수권-크로매틱 기후)를 일관되게 복원하는 가상 월드 엔진.
    """
    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.cosmic_rotor = CosmicMacroRotor()
        self.rotor_cascade = FractalRotorCascade(seed=self.seed)
        self.dynamo = PlanetaryDynamo()
        self.validator = TopologicalPhaseValidator()

    def evaluate_coupled_equilibrium(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        t: float = 0.0,
        causal_lod_depth: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        [Closed-Form Coupled Equilibrium Field Operator]
        3대 다층 경계층(지각-수권-크로매틱 기후) 및 천체 로터 상태의 결합 방정식을
        입자 시뮬레이션 없이 단 한 번의 대수적 행렬 연산으로 평가합니다.

        1) Cosmic & Dynamo Field Calculation
        2) Lithosphere Field (Height H, Gradient \\nabla H) with Domain Warping
        3) Hydrosphere Field (Analytical Drainage Accumulation A, Stream Power Erosion E)
        4) Chromatic Climate Field (Flux, Order, Entropy)
        5) Causal LOD Depth Scale Focus Integration
        """
        cosmic_state = self.cosmic_rotor.get_cosmic_state(t)
        insolation = cosmic_state["insolation"]
        tidal = cosmic_state["tidal_force"]
        seasonal_temp = cosmic_state["seasonal_temp"]

        # 1. Base Lithosphere Height Field via Fractal Rotor Cascade + Domain Warping
        warp_x, warp_y = self.rotor_cascade.evaluate_domain_warping(
            x_grid, y_grid, t=t, scale_depth=min(causal_lod_depth, 4), warp_strength=25.0
        )
        raw_height = self.rotor_cascade.evaluate_rotor_wave(
            warp_x, warp_y, t=t, scale_depth=causal_lod_depth, base_frequency=0.005
        )

        # Scale heightmap to physical potential altitude [0, 100.0]
        heightmap = (raw_height - np.min(raw_height)) / (np.ptp(raw_height) + 1e-6) * 100.0

        # 2. Gradient Calculation for Slope Vector \\nabla H
        gy, gx = np.gradient(heightmap)
        slope_norm = np.sqrt(gx**2 + gy**2)

        # 3. Closed-Form Hydrosphere Analytical Stream Power Erosion Field
        # Flow accumulation is algebraically derived from negative potential basins + tidal magnitude
        potential_basin = np.maximum(0.0, 100.0 - heightmap)
        flow_accumulation = (potential_basin ** 1.2) * (1.0 + 0.3 * tidal)

        # Stream Power Erosion Law: E = K * A^m * ||\\nabla H||^n
        K_erosion = 0.01
        erosion_field = K_erosion * (flow_accumulation ** 0.5) * (slope_norm ** 1.0)

        # Apply Closed-Form Compressed Field Erosion
        eroded_heightmap = np.maximum(0.0, heightmap - erosion_field)

        # Moisture field M(x, y)
        moisture_field = (flow_accumulation / (np.max(flow_accumulation) + 1e-6)) * (1.0 + 0.2 * seasonal_temp)

        # 4. Chromatic Climate Field (Flux, Order, Entropy)
        flux_channel = 0.33 + 0.3 * insolation + 0.2 * (slope_norm / (np.max(slope_norm) + 1e-6))
        order_channel = 0.33 + 0.3 * (eroded_heightmap / 100.0)
        entropy_channel = 0.34 + 0.3 * moisture_field

        chromatic_field = np.stack([flux_channel, order_channel, entropy_channel], axis=-1)
        chromatic_sum = np.sum(chromatic_field, axis=-1, keepdims=True)
        chromatic_field /= (chromatic_sum + 1e-6)

        # 5. Topological Phase Validation & Healing
        healed_heightmap, validation_report = self.validator.validate_and_heal_field(
            eroded_heightmap, flow_accumulation, chromatic_field
        )

        return {
            "heightmap": healed_heightmap,
            "gradient_x": gx,
            "gradient_y": gy,
            "slope_norm": slope_norm,
            "flow_accumulation": flow_accumulation,
            "erosion_field": erosion_field,
            "moisture_field": moisture_field,
            "chromatic_field": chromatic_field,
            "cosmic_state": cosmic_state,
            "validation_report": validation_report
        }

    def generate_chunk(
        self,
        chunk_x: int,
        chunk_y: int,
        chunk_size: float = 64.0,
        resolution: int = 32,
        t: float = 0.0,
        causal_lod_depth: int = 5
    ) -> Dict[str, Any]:
        """
        Generates a deterministic world chunk given chunk coordinates (chunk_x, chunk_y).
        """
        x_min = chunk_x * chunk_size
        x_max = (chunk_x + 1) * chunk_size
        y_min = chunk_y * chunk_size
        y_max = (chunk_y + 1) * chunk_size

        x_coords = np.linspace(x_min, x_max, resolution, dtype=np.float32)
        y_coords = np.linspace(y_min, y_max, resolution, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        field_data = self.evaluate_coupled_equilibrium(x_grid, y_grid, t=t, causal_lod_depth=causal_lod_depth)

        # Emergent Feature Placement (Poisson Disk Sampling & L-System Nodes)
        structures = self._place_emergent_structures(
            x_grid, y_grid,
            field_data["heightmap"],
            field_data["flow_accumulation"],
            field_data["chromatic_field"]
        )

        return {
            "chunk_coords": (chunk_x, chunk_y),
            "chunk_size": chunk_size,
            "resolution": resolution,
            "x_grid": x_grid,
            "y_grid": y_grid,
            "field_data": field_data,
            "emergent_structures": structures
        }

    def _place_emergent_structures(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        heightmap: np.ndarray,
        flow_acc: np.ndarray,
        chromatic_field: np.ndarray,
        min_distance: float = 8.0
    ) -> List[Dict[str, Any]]:
        """
        [Poisson Disk & L-System Emergent Placement]
        Places resources/structures naturally where chromatic flux and hydro resonance nodes align,
        guaranteeing minimum distance between nodes without arbitrary discrete conditional branching.
        """
        structures = []
        rows, cols = heightmap.shape

        # Resonance score for node emergence = Hydro Accumulation * Order Channel * Flux Channel
        resonance_nodes = flow_acc * chromatic_field[:, :, 0] * chromatic_field[:, :, 1]
        threshold = np.percentile(resonance_nodes, 90) # Top 10% resonance points

        candidates = []
        for r in range(rows):
            for c in range(cols):
                if resonance_nodes[r, c] >= threshold:
                    pos = np.array([x_grid[r, c], y_grid[r, c], heightmap[r, c]], dtype=np.float32)
                    candidates.append((resonance_nodes[r, c], pos, chromatic_field[r, c]))

        # Sort candidates by resonance score descending
        candidates.sort(key=lambda item: item[0], reverse=True)

        # Poisson Disk minimum distance filtering
        placed_positions = []
        for score, pos, chrom in candidates:
            too_close = False
            for prev_pos in placed_positions:
                if np.linalg.norm(pos[:2] - prev_pos[:2]) < min_distance:
                    too_close = True
                    break
            if not too_close:
                placed_positions.append(pos)
                # Structure type derived from chromatic dominant signature
                dom_color = int(np.argmax(chrom))
                type_names = ["FluxEnergyBeacon", "OrderMonolith", "EntropySanctuary"]
                structures.append({
                    "id": f"struct_{len(structures)}_{pos[0]:.1f}_{pos[1]:.1f}",
                    "type": type_names[dom_color],
                    "position": pos.tolist(),
                    "chromatic_vector": chrom.tolist(),
                    "resonance_score": float(score)
                })

        return structures

    def populate_manifold(
        self,
        manifold: ContinuousWorldManifold,
        chunk_x: int,
        chunk_y: int,
        chunk_size: float = 64.0,
        resolution: int = 32,
        t: float = 0.0
    ) -> Dict[str, Any]:
        """
        Injects the generated chunk potential nodes and emergent structures into an existing
        `ContinuousWorldManifold` from `causal_mmorpg_sandbox.py`.
        """
        chunk_data = self.generate_chunk(chunk_x, chunk_y, chunk_size, resolution, t)
        field_data = chunk_data["field_data"]
        structures = chunk_data["emergent_structures"]

        x_grid = chunk_data["x_grid"]
        y_grid = chunk_data["y_grid"]
        heightmap = field_data["heightmap"]
        flow_acc = field_data["flow_accumulation"]

        step = max(1, resolution // 4)
        nodes_added = 0
        for r in range(0, resolution, step):
            for c in range(0, resolution, step):
                pos_3d = np.array([x_grid[r, c], y_grid[r, c], heightmap[r, c]], dtype=np.float32)
                intensity = float(flow_acc[r, c] * 0.1)
                manifold.inject_potential(pos_3d, intensity, node_type="hydro_potential")
                nodes_added += 1

        for struct in structures:
            pos_3d = np.array(struct["position"], dtype=np.float32)
            manifold.inject_potential(pos_3d, struct["resonance_score"] * 0.5, node_type=struct["type"])

        return {
            "chunk_coords": (chunk_x, chunk_y),
            "potential_nodes_added": nodes_added,
            "structures_added": len(structures)
        }
