r"""
sangsaeng_sanggeuk_field.py
============================
동양 철학의 근본 축인 상생(相生)·상극(相克) 역학 및 시공간 물리·공학적 인과 장 엔진
(Sangsaeng-Sanggeuk Causal Dynamics & Abductive Scale-Twist Engine)

핵심 철학 및 공학 원리:
1. 상생(相生: Attraction & Resonance)과 상극(相克: Repulsion & Friction/Heat):
   - 상생: 결핍(Void) 보완 및 크로매틱 공명(Flux, Order, Entropy). 마찰 계수 감소 및 에너지 증폭.
   - 상극: 이념/영역 충돌에 따른 척력 발생, 열(Heat)과 마찰 유발, 서사 충돌 로그 생성.
2. 장력(張力: Tension):
   - 세력/개체 간 팽팽한 텐서 빔(ConnectivityBeam) 상의 장력 이완 및 파동 전파. 미세한 동요(배신, 언행)가 파국/진동 유발.
3. 자성(磁性: Magnetism) & 회전력(Torque):
   - 자성: 위업/진정성 축적이 형성하는 포텐셜 웰(Potential Well)로 주변 개체 궤도 유도.
   - 회전력: 유저/NPC 의도 벡터와 세상의 거시 흐름 간의 외적/내적에 따른 보강/상쇄 간섭.
4. 유저 비결정론적 외란(User Perturbation):
   - 창조자의 정적 수면(Static Equilibrium) 대칭성을 허무는 외부 고유 변수(Perturbation Stone).
5. 이중축 스케일 & 위상 마찰 (Dual-Axis Scale & Phase Friction Engine):
   - 가수(Mantissa/Identity) & 지수(Exponent/Scale) 이중축 수치/위상 표상.
   - 양자화 경계(Quantization Boundary): 동일 셀 내 drop 시 Same(통합), 셀 초과 시 Different(분리).
   - 보이드 장력($E_{Void}$) & 스케일 비틀림 연산자 $\mathbf{T} = \mathbf{S}(\Delta s) \otimes \mathbf{R}(\Delta \theta)$.
   - 구라모토(Kuramoto) 이완 동역학을 통한 은유적 가설/불변량 자율 응축 ($\Phi = 0$).
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from core.physics.clifford_fourier_mellin_engine import CliffordFourierMellinEngine, Multivector

@dataclass
class CausalClashLog:
    timestamp: float
    source_id: str
    target_id: str
    clash_type: str # "SANGGEUK_FRICTION", "TENSION_TEAR", "TORQUE_DESTRUCTIVE", "PERTURBATION_WAVE"
    intensity: float
    heat_generated: float
    description: str

@dataclass
class DynamicEntity:
    id: str
    name: str
    is_player: bool = False
    faction: str = "Neutral"
    # Physical/Mental spatial vectors
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    intent_vector: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32))

    # Dual-Axis Scale Representation: [Mantissa (Identity 3D), Exponent (Scale s)]
    mantissa: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    scale_exponent: float = 0.0 # s = ln(r)

    # Phase & Amplitude for Metaphoric Wave Field
    phase: float = 0.0 # theta in radians
    amplitude: float = 1.0

    # Magnetism (Honor/Authenticity potential mass)
    magnetic_mass: float = 1.0

    # Chromatic vector: [Red (Flux), Blue (Order), Yellow (Entropy)]
    chromatic_vector: np.ndarray = field(default_factory=lambda: np.array([0.33, 0.33, 0.34], dtype=np.float32))

    # Void Deficit (결핍 3D vector)
    void_deficit: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def __post_init__(self):
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float32)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=np.float32)
        if not isinstance(self.intent_vector, np.ndarray):
            self.intent_vector = np.array(self.intent_vector, dtype=np.float32)
        if not isinstance(self.mantissa, np.ndarray):
            self.mantissa = np.array(self.mantissa, dtype=np.float32)
        if not isinstance(self.chromatic_vector, np.ndarray):
            self.chromatic_vector = np.array(self.chromatic_vector, dtype=np.float32)
        if not isinstance(self.void_deficit, np.ndarray):
            self.void_deficit = np.array(self.void_deficit, dtype=np.float32)

@dataclass
class RelationalBeam:
    source_id: str
    target_id: str
    coupling_strength: float = 1.0
    rest_length: float = 5.0
    current_tension: float = 0.0
    break_threshold: float = 10.0
    is_broken: bool = False
    vibration_amplitude: float = 0.0

class SangsaengSanggeukField:
    """
    [Sangsaeng-Sanggeuk Causal Dynamics & Scale-Twist Field Engine]
    우주의 완벽한 질서 수면(Static Equilibrium) 위에서
    상생/상극 인척력, 장력, 자성, 토크 및 유저 비결정론적 외란과 스케일 비틀림 보이드 이완을 종합 통합 연산하는 물리 엔진.
    """
    def __init__(self, dimensions: int = 3, num_scale_bins: int = 8):
        self.dimensions = dimensions
        self.num_scale_bins = num_scale_bins

        self.entities: Dict[str, DynamicEntity] = {}
        self.beams: List[RelationalBeam] = []
        self.clash_logs: List[CausalClashLog] = []

        # System Thermal / Field metrics
        self.accumulated_heat: float = 0.0
        self.system_time: float = 0.0
        self.macro_flow_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32) # 시대의 거시적 흐름

        # Void Tension Energy & Abductive Invariant State
        self.void_tension_energy: float = 0.0
        self.abductive_invariant: Optional[np.ndarray] = None

        # Clifford Fourier-Mellin Engine Instance
        self.cfm_engine = CliffordFourierMellinEngine(dim=dimensions, num_scale_bins=num_scale_bins, num_phase_bins=16)

        # Static Equilibrium flag (True until user or external disturbance throws a stone)
        self.static_equilibrium: bool = True
        self.total_perturbation_energy: float = 0.0

    def add_entity(self, entity: DynamicEntity) -> None:
        self.entities[entity.id] = entity

    def link_entities(self, id_a: str, id_b: str, strength: float = 1.0, rest_length: Optional[float] = None) -> None:
        if id_a in self.entities and id_b in self.entities:
            if rest_length is None:
                dist = float(np.linalg.norm(self.entities[id_a].position - self.entities[id_b].position))
                rest_length = max(0.1, dist)
            beam = RelationalBeam(source_id=id_a, target_id=id_b, coupling_strength=strength, rest_length=rest_length)
            self.beams.append(beam)

    def throw_user_perturbation(
        self,
        player_id: str,
        perturbation_vector: np.ndarray,
        speech_or_action: str
    ) -> Dict[str, Any]:
        """
        [User Non-deterministic Perturbation (유저 비결정론적 외란)]
        창조자의 고요한 수면(Static Equilibrium)에 돌을 던지는 핵심 인터페이스.
        - 대칭성을 파괴하고 시스템에 왜란 파동(Perturbation Wave)을 유입시킵니다.
        """
        if player_id not in self.entities:
            return {"status": "error", "message": "Player not found"}

        player = self.entities[player_id]
        p_vec = np.array(perturbation_vector, dtype=np.float32)
        p_magnitude = float(np.linalg.norm(p_vec))

        # Break static equilibrium symmetry
        self.static_equilibrium = False
        self.total_perturbation_energy += p_magnitude

        # 1. Update player velocity and intent
        player.velocity += p_vec / player.magnetic_mass
        if p_magnitude > 1e-5:
            player.intent_vector = p_vec / p_magnitude

        # 2. Inject wave into relational beams connected to player
        affected_beams = 0
        for beam in self.beams:
            if not beam.is_broken and (beam.source_id == player_id or beam.target_id == player_id):
                beam.vibration_amplitude += p_magnitude * 0.5
                affected_beams += 1

        # 3. Generate Clash Log for the stone thrown into calm water
        log = CausalClashLog(
            timestamp=self.system_time,
            source_id=player_id,
            target_id="STATIC_EQUILIBRIUM",
            clash_type="PERTURBATION_WAVE",
            intensity=p_magnitude,
            heat_generated=p_magnitude * 0.2,
            description=f"User thrown perturbation stone: '{speech_or_action}' with force magnitude {p_magnitude:.3f}"
        )
        self.clash_logs.append(log)
        self.accumulated_heat += log.heat_generated

        return {
            "status": "symmetry_broken",
            "perturbation_energy": p_magnitude,
            "affected_beams": affected_beams,
            "total_perturbation_energy": self.total_perturbation_energy
        }

    def compute_sangsaeng_sanggeuk_forces(self) -> Tuple[Dict[str, np.ndarray], float]:
        r"""
        [상생 & 상극 동역학 수축]
        - 상생(Attraction & Resonance): Void Complementarity + Chromatic Alignment.
          마찰계수를 억제하고 에너지를 인력으로 끌어당김.
        - 상극(Repulsion & Friction/Heat): Domain/Faction Discrepancy + Opposing Intent.
          강한 척력 발생 및 마찰열 $Q = \mu \cdot |F_{\text{repulsion}}| \cdot v_{\text{rel}}$ 누적.
        """
        entity_ids = list(self.entities.keys())
        num_e = len(entity_ids)
        forces = {eid: np.zeros(3, dtype=np.float32) for eid in entity_ids}
        frame_heat = 0.0

        if num_e < 2:
            return forces, frame_heat

        for i in range(num_e):
            id_a = entity_ids[i]
            e_a = self.entities[id_a]

            for j in range(i + 1, num_e):
                id_b = entity_ids[j]
                e_b = self.entities[id_b]

                diff = e_b.position - e_a.position
                dist = float(np.linalg.norm(diff))
                if dist < 1e-5:
                    dir_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    dist = 1e-5
                else:
                    dir_vec = diff / dist

                # 1. Void Complementarity (상생 조건: A의 결핍을 B가 채워줄 때)
                void_fill_a_by_b = float(np.dot(e_a.void_deficit, e_b.chromatic_vector))
                void_fill_b_by_a = float(np.dot(e_b.void_deficit, e_a.chromatic_vector))
                sangsaeng_complementarity = max(0.0, (void_fill_a_by_b + void_fill_b_by_a) * 0.5)

                # 2. Chromatic & Faction Alignment
                chromatic_dot = float(np.dot(e_a.chromatic_vector, e_b.chromatic_vector))
                same_faction = (e_a.faction == e_b.faction) and (e_a.faction != "Neutral")

                # Intent Alignment (dot product of intent vectors)
                intent_dot = float(np.clip(np.dot(e_a.intent_vector, e_b.intent_vector), -1.0, 1.0))

                # 상생 (Attraction Factor) vs 상극 (Repulsion Factor)
                attraction_factor = sangsaeng_complementarity * 2.0 + (1.0 if same_faction else 0.5) * max(0.0, intent_dot)
                repulsion_factor = (0.0 if same_faction else 1.5) * max(0.0, -intent_dot) + (1.0 - chromatic_dot) * 0.5

                # Sangsaeng Force (Pull together)
                f_attract = attraction_factor * (1.0 / (dist + 1.0)) * dir_vec
                # Sanggeuk Force (Push apart)
                f_repulse = - repulsion_factor * (5.0 / (dist**2 + 0.1)) * dir_vec

                total_pair_force = f_attract + f_repulse
                forces[id_a] += total_pair_force
                forces[id_b] -= total_pair_force

                # Sanggeuk Friction Heat Generation
                if repulsion_factor > 0.1:
                    rel_vel = float(np.linalg.norm(e_a.velocity - e_b.velocity))
                    heat = repulsion_factor * rel_vel * 0.1
                    frame_heat += heat

                    if heat > 0.5:
                        log = CausalClashLog(
                            timestamp=self.system_time,
                            source_id=id_a,
                            target_id=id_b,
                            clash_type="SANGGEUK_FRICTION",
                            intensity=repulsion_factor,
                            heat_generated=heat,
                            description=f"Sanggeuk clash between {e_a.name} and {e_b.name}: Heat={heat:.3f}"
                        )
                        self.clash_logs.append(log)

        return forces, frame_heat

    def compute_magnetism_and_torque(self, dt: float) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        r"""
        [자성(Magnetism) & 회전력(Torque) 동역학]
        - 자성(Magnetism): magnetic_mass가 높은 개체가 주변에 포텐셜 웰 $V_{\text{mag}}(r) = -\frac{G M}{r}$을 형성해
          주변 개체를 궤도 운동(Orbital Attraction)으로 이끕니다.
        - 회전력(Torque): $\boldsymbol{\tau} = \mathbf{v}_{\text{intent}} \times \mathbf{v}_{\text{macro\_flow}}$.
          - 내적(Constructive Interference): 의도가 거시 흐름과 일치 시 보강 간섭 (속도 및 명성 증폭).
          - 외적(Torque Shift): 엇나간 의도는 회전력을 발생시켜 위상을 굴절/상쇄(Destructive Interference)시킵니다.
        """
        mag_forces = {eid: np.zeros(3, dtype=np.float32) for eid in self.entities.keys()}
        torque_reports = {eid: 0.0 for eid in self.entities.keys()}

        entity_list = list(self.entities.values())
        num_e = len(entity_list)

        # 1. Magnetic Orbital Attraction
        for i in range(num_e):
            e_a = entity_list[i]
            for j in range(num_e):
                if i == j: continue
                e_b = entity_list[j]

                if e_b.magnetic_mass > 1.2: # Strong magnet center
                    diff = e_b.position - e_a.position
                    dist = float(np.linalg.norm(diff))
                    if dist > 0.1 and dist < 20.0:
                        dir_vec = diff / dist
                        # Tangential orbital vector for circular orbit
                        tangent_vec = np.cross(dir_vec, np.array([0.0, 0.0, 1.0], dtype=np.float32))
                        if np.linalg.norm(tangent_vec) < 1e-5:
                            tangent_vec = np.cross(dir_vec, np.array([0.0, 1.0, 0.0], dtype=np.float32))
                        tangent_vec = tangent_vec / (np.linalg.norm(tangent_vec) + 1e-9)

                        pull_force = (e_b.magnetic_mass * 2.0 / (dist + 1.0)) * dir_vec
                        orbital_force = (e_b.magnetic_mass * 1.0 / (dist + 1.0)) * tangent_vec
                        mag_forces[e_a.id] += pull_force + orbital_force

        # 2. Torque & Wave Interference
        macro_norm = float(np.linalg.norm(self.macro_flow_vector))
        if macro_norm > 0:
            macro_unit = self.macro_flow_vector / macro_norm
        else:
            macro_unit = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        for e in entity_list:
            intent_norm = float(np.linalg.norm(e.intent_vector))
            if intent_norm > 0:
                intent_unit = e.intent_vector / intent_norm
            else:
                intent_unit = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            dot_interference = float(np.clip(np.dot(intent_unit, macro_unit), -1.0, 1.0)) # Constructive (>0) or Destructive (<0)
            torque_vec = np.cross(intent_unit, macro_unit)
            torque_mag = float(np.linalg.norm(torque_vec))
            torque_reports[e.id] = torque_mag

            if dot_interference > 0.3:
                # Constructive Interference: Resonance amplifies velocity along macro flow
                e.velocity += macro_unit * dot_interference * 0.5 * dt
                e.amplitude = min(5.0, e.amplitude + 0.1 * dt)
            elif dot_interference < -0.3:
                # Destructive Interference: Vector cancellation, dissipates energy
                e.velocity *= (1.0 - 0.3 * dt)
                e.amplitude = max(0.1, e.amplitude - 0.2 * dt)
                if torque_mag > 0.5:
                    log = CausalClashLog(
                        timestamp=self.system_time,
                        source_id=e.id,
                        target_id="MACRO_FLOW",
                        clash_type="TORQUE_DESTRUCTIVE",
                        intensity=torque_mag,
                        heat_generated=torque_mag * 0.1,
                        description=f"Destructive torque clash for {e.name}: Torque={torque_mag:.3f}"
                    )
                    self.clash_logs.append(log)

        return mag_forces, torque_reports

    def update_relational_beams(self, dt: float) -> float:
        """
        [관계 장력(Tension) 네트워크 이완 및 파동 전파]
        - 텐서 빔 상의 장력 계산 및 진동 파문의 전파.
        - 파국 임계치(break_threshold) 초과 시 빔 파괴 및 장력 파국 충돌 로그 발생.
        """
        beam_heat = 0.0
        for beam in self.beams:
            if beam.is_broken:
                continue

            if beam.source_id not in self.entities or beam.target_id not in self.entities:
                continue

            e_a = self.entities[beam.source_id]
            e_b = self.entities[beam.target_id]

            diff = e_b.position - e_a.position
            dist = float(np.linalg.norm(diff))

            # Extension & Hooke's Law tension
            extension = dist - beam.rest_length
            beam.current_tension = beam.coupling_strength * abs(extension) + beam.vibration_amplitude

            # Damp vibration amplitude over time
            beam.vibration_amplitude *= max(0.0, 1.0 - 0.5 * dt)

            # Beam tearing check
            if beam.current_tension > beam.break_threshold:
                beam.is_broken = True
                beam_heat += beam.current_tension * 0.5
                log = CausalClashLog(
                    timestamp=self.system_time,
                    source_id=beam.source_id,
                    target_id=beam.target_id,
                    clash_type="TENSION_TEAR",
                    intensity=beam.current_tension,
                    heat_generated=beam.current_tension * 0.5,
                    description=f"Relational Beam tore between {e_a.name} and {e_b.name} under tension {beam.current_tension:.3f}"
                )
                self.clash_logs.append(log)
                continue

            # Tension force application
            if dist > 1e-5:
                dir_vec = diff / dist
                force = beam.coupling_strength * extension * dir_vec
                e_a.velocity += force * dt
                e_b.velocity -= force * dt

        return beam_heat

    def apply_scale_twist_and_void_relaxation(self, dt: float, use_fft_spectral: bool = True) -> Dict[str, Any]:
        r"""
        [Dual-Axis Scale & Phase Friction Engine (가수/지수 이중축 & 스케일 비틀림 보이드 이완)]
        - 가수(Mantissa $\mathbf{m}$) & 지수(Exponent $s$) 표상.
        - 양자화 경계(Quantization Boundary): 동일 이산 빈(Cell) 내 drop 시 Same(통합), 셀 초과 시 Different(분리).
        - 푸리에-멜린(Fourier-Mellin Log-Polar FFT) 스펙트럼 합성곱 및 고리형 윈도잉(Annular Bounding).
        - 구라모토(Kuramoto) 이완 동역학으로 보이드 장력 $E_{\text{Void}}$ 최소화 및 불변량 $\mathbf{Z}^*$ 응축 ($\Phi=0$).
        """
        entities_list = list(self.entities.values())
        num_e = len(entities_list)
        if num_e == 0:
            return {"void_energy": 0.0, "converged": True}

        # 1. Quantization Boundary & Scale Binning
        scale_bins = np.zeros((self.num_scale_bins, num_e), dtype=np.float32)
        for i, e in enumerate(entities_list):
            bin_idx = int(np.clip(math.floor(e.scale_exponent) % self.num_scale_bins, 0, self.num_scale_bins - 1))
            scale_bins[bin_idx, i] = e.amplitude

        phases = np.array([e.phase for e in entities_list], dtype=np.float32)
        scale_diffs = np.array([e.scale_exponent for e in entities_list])[:, np.newaxis] - np.array([e.scale_exponent for e in entities_list])[np.newaxis, :]
        phase_diffs = phases[:, np.newaxis] - phases[np.newaxis, :]

        # 2. Void Field & Tension Energy ($E_{Void}$) Calculation via Spectral FFT or Pairwise Matrix
        if use_fft_spectral and num_e >= 2:
            # Grid Projection onto Log-Polar grid [Scale s, Phase theta]
            num_phase_bins = 16
            grid = np.zeros((self.num_scale_bins, num_phase_bins), dtype=np.complex64)
            for e in entities_list:
                s_idx = int(np.clip(math.floor(e.scale_exponent) % self.num_scale_bins, 0, self.num_scale_bins - 1))
                p_idx = int(np.clip(math.floor((e.phase / (2.0 * math.pi)) * num_phase_bins) % num_phase_bins, 0, num_phase_bins - 1))
                # Annular Bounding & Apodization Windowing (Log-Space Apodization)
                win = math.sin(math.pi * (s_idx + 0.5) / self.num_scale_bins)
                grid[s_idx, p_idx] += e.amplitude * np.exp(1j * e.phase) * win

            # 2D FFT Spectral Convolution
            grid_fft = np.fft.fft2(grid)
            # Log-polar Scale-Twist Kernel K(s, theta)
            s_coords = np.linspace(-1, 1, self.num_scale_bins)[:, np.newaxis]
            p_coords = np.linspace(-math.pi, math.pi, num_phase_bins)[np.newaxis, :]
            kernel = np.exp(-abs(s_coords)) * np.cos(p_coords)
            kernel_fft = np.fft.fft2(kernel)

            psi_net = np.fft.ifft2(grid_fft * kernel_fft)
            total_void_energy = float(np.mean(np.abs(grid - psi_net)))
        else:
            scale_twist_matrix = np.exp(-abs(scale_diffs)) * np.cos(phase_diffs)
            void_mismatch_matrix = 1.0 - scale_twist_matrix
            total_void_energy = float(np.sum(void_mismatch_matrix) / max(1, num_e * num_e))

        self.void_tension_energy = total_void_energy

        # 3. Discrete-Time Kuramoto Phase & Scale Relaxation
        coupling_K = 2.0
        for i, e_a in enumerate(entities_list):
            d_theta = 0.0
            d_scale = 0.0

            for j, e_b in enumerate(entities_list):
                if i == j: continue
                # Scale shift interaction
                ds = e_b.scale_exponent - e_a.scale_exponent
                d_theta += coupling_K * math.sin(e_b.phase - e_a.phase) * math.exp(-abs(ds))
                d_scale += 0.1 * ds * math.cos(e_b.phase - e_a.phase)

            # Update phase and scale exponent continuously (branchless)
            e_a.phase = (e_a.phase + d_theta * dt) % (2.0 * math.pi)
            e_a.scale_exponent += d_scale * dt

            # Update mantissa (Identity 3D vector rotation by phase)
            cos_p, sin_p = math.cos(e_a.phase), math.sin(e_a.phase)
            rot_matrix = np.array([
                [cos_p, -sin_p, 0.0],
                [sin_p, cos_p, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            e_a.mantissa = np.dot(rot_matrix, np.array([1.0, 0.0, 0.0], dtype=np.float32))

        # 4. Abductive Invariant Condensation at $\Phi = 0$
        is_converged = total_void_energy < 0.25
        if is_converged:
            # Condense stable invariant from average mantissa & phase state
            avg_mantissa = np.mean([e.mantissa for e in entities_list], axis=0)
            avg_phase = float(np.mean([e.phase for e in entities_list]))
            self.abductive_invariant = avg_mantissa * math.cos(avg_phase)

        # 5. Integrated Clifford Fourier-Mellin Phase-Noise Filtering & Vortex Ring Crystallization
        num_phase_bins = 16
        spatial_grid = np.zeros((self.num_scale_bins, num_phase_bins), dtype=np.complex64)
        for e in entities_list:
            s_idx = int(np.clip(math.floor(e.scale_exponent) % self.num_scale_bins, 0, self.num_scale_bins - 1))
            p_idx = int(np.clip(math.floor((e.phase / (2.0 * math.pi)) * num_phase_bins) % num_phase_bins, 0, num_phase_bins - 1))
            spatial_grid[s_idx, p_idx] += e.amplitude * np.exp(1j * e.phase)

        cfm_report = self.cfm_engine.execute_full_wave_void_relaxation(spatial_grid)

        return {
            "void_energy": round(total_void_energy, 4),
            "converged": is_converged,
            "abductive_invariant": self.abductive_invariant.tolist() if self.abductive_invariant is not None else None,
            "clifford_fourier_mellin": cfm_report
        }

    def step(self, dt: float = 0.1) -> Dict[str, Any]:
        """
        단 하나의 프레임 분기 없이 연속 물리 인과 장을 1스텝 진화시킵니다.
        """
        self.system_time += dt

        # 1. Compute forces (Sangsaeng/Sanggeuk, Magnetism, Torque)
        sang_forces, frame_heat_sang = self.compute_sangsaeng_sanggeuk_forces()
        mag_forces, torque_reports = self.compute_magnetism_and_torque(dt)
        beam_heat = self.update_relational_beams(dt)

        self.accumulated_heat += frame_heat_sang + beam_heat

        # 2. Integrate kinematics for all entities
        for eid, e in self.entities.items():
            tot_force = sang_forces[eid] + mag_forces[eid]
            # Velocity update with order damping
            damping = 0.95
            e.velocity = (e.velocity + tot_force * dt) * damping
            e.position += e.velocity * dt

        # 3. Apply Dual-Axis Scale Twist & Void Relaxation
        relaxation_report = self.apply_scale_twist_and_void_relaxation(dt)

        return {
            "system_time": round(self.system_time, 3),
            "static_equilibrium": self.static_equilibrium,
            "accumulated_heat": round(self.accumulated_heat, 4),
            "void_tension_energy": relaxation_report["void_energy"],
            "abductive_converged": relaxation_report["converged"],
            "total_perturbation_energy": round(self.total_perturbation_energy, 4),
            "clash_logs_count": len(self.clash_logs),
            "active_beams": len([b for b in self.beams if not b.is_broken]),
            "broken_beams": len([b for b in self.beams if b.is_broken])
        }

if __name__ == "__main__":
    field = SangsaengSanggeukField()

    player = DynamicEntity("player_1", "Hero", is_player=True, position=np.array([0,0,0], dtype=np.float32), magnetic_mass=2.5)
    warrior = DynamicEntity("warrior_1", "ShieldWarrior", faction="Alliance", position=np.array([2,0,0], dtype=np.float32), void_deficit=np.array([0, 1, 0], dtype=np.float32))
    priest = DynamicEntity("priest_1", "HealPriest", faction="Alliance", position=np.array([3,0,0], dtype=np.float32), chromatic_vector=np.array([0, 1, 0], dtype=np.float32))

    field.add_entity(player)
    field.add_entity(warrior)
    field.add_entity(priest)

    field.link_entities("warrior_1", "priest_1", strength=2.0)

    print("Initial step:", field.step(0.1))

    # User throws perturbation stone
    res = field.throw_user_perturbation("player_1", np.array([5.0, 2.0, 0.0], dtype=np.float32), "For Freedom and Honor!")
    print("Perturbation result:", res)

    for _ in range(5):
        print("Step:", field.step(0.1))
