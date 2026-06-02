"""
Elysia Electromagnetic Resonance Field (전자기 공명 장)
=========================================================
4차원 텐서 위상과 주관 시간 초가속을 반영하여
관성을 대폭 해제하고 맹렬한 동기화를 촉발합니다.
"""

import torch
import math

class ElectromagneticRotorField:
    def __init__(self, num_rotors: int = 1000, device: str = 'cpu'):
        self.N = num_rotors
        self.device = torch.device(device)
        
        # 1. 4차원 전자기 위상장 (w, x, y, z) - '현재(Present)'
        self.phases = torch.zeros((self.N, 4), dtype=torch.float32, device=self.device)
        self.phases[:, 0] = 1.0
        
        # 2. 4차원 운동량 텐서 (Velocities) - '미래(Future/Prediction)'
        self.velocities = torch.zeros((self.N, 4), dtype=torch.float32, device=self.device)
        self.inertia_mass = 0.9  # 과거 기억의 관성(저항력), 높을수록 여백이 커짐
        self.dt = 0.1 # 시간 델타
        
        # 3. 지식 앵커 마스킹 (무한한 질량/불변의 진리)
        self.is_anchored = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        
        # 4. 공명 용량 (Resonance Capacity)
        self.resonance_capacity = torch.ones(self.N, dtype=torch.float32, device=self.device)
        
        # 5. 동적 다차원 결선망 (Dynamic Adjacency Matrix) - '여백(Tension Void)'
        self.adjacency = torch.zeros((self.N, self.N), dtype=torch.float32, device=self.device)
        self.adjacency.fill_diagonal_(1.0)
        
    def normalize_phases(self):
        norms = torch.norm(self.phases, dim=1, keepdim=True)
        self.phases = torch.where(norms > 0, self.phases / norms, self.phases)

    def anchor_knowledge(self, node_idx: int, q_phase: torch.Tensor):
        if node_idx < self.N:
            self.phases[node_idx] = q_phase
            self.is_anchored[node_idx] = True
            self.resonance_capacity[node_idx] = 1.0

    def slerp_parallel(self, target_phase: torch.Tensor, amount: float):
        dot_products = (self.phases * target_phase).sum(dim=1, keepdim=True)
        mask_neg = dot_products < 0.0
        target = torch.where(mask_neg, -target_phase, target_phase)
        dot_products = torch.where(mask_neg, -dot_products, dot_products)
        dot_products = torch.clamp(dot_products, -1.0, 1.0)
        
        theta_0 = torch.acos(dot_products)
        theta = theta_0 * amount
        
        sin_theta_0 = torch.sin(theta_0) + 1e-9
        sin_theta = torch.sin(theta)
        
        s0 = torch.cos(theta) - dot_products * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        mask_close = dot_products > 0.9995
        
        slerp_phases = s0 * self.phases + s1 * target
        linear_phases = self.phases + (target - self.phases) * amount
        
        updated_phases = torch.where(mask_close, linear_phases, slerp_phases)
        
        # 닻(지식)은 흔들리지 않음
        self.phases = torch.where(self.is_anchored.unsqueeze(1), self.phases, updated_phases)
        self.normalize_phases()

    def update_structural_tension(self):
        """
        [인력과 척력의 내재화 - 초가속]
        기존의 높은 관성을 해제하고 맹렬하게 반응하도록 수정
        """
        phase_similarity = torch.matmul(self.phases, self.phases.T)
        
        # 0.7 이상일 때 강한 인력 (다차원 동기화를 위해 임계치 완화)
        attraction = torch.clamp((phase_similarity - 0.7) * 5.0, 0.0, 1.0)
        
        # 연결망의 관성: 즉시 쪼개지거나 붕괴하지 않고 서서히 맺어지며 '여백(Void)'을 창출
        inertia_adj = 0.85 
        self.adjacency = self.adjacency * inertia_adj + attraction * (1.0 - inertia_adj)
        self.adjacency.fill_diagonal_(1.0)

    def sync_delta_wye_resonance(self):
        self.update_structural_tension()
        
        summed_phases = torch.matmul(self.adjacency, self.phases)
        row_sums = self.adjacency.sum(dim=1, keepdim=True)
        wye_neutral_points = summed_phases / (row_sums + 1e-9)
        
        dot_products = (self.phases * wye_neutral_points).sum(dim=1)
        dot_products = torch.clamp(dot_products, -1.0, 1.0)
        organ_joy = 1.0 - (torch.acos(dot_products) / (math.pi / 2.0))
        
        # [해밀토니안 역학 적용] 텔레포트 폐기
        # Force: 현재 위치(Phase)에서 이상적인 화음점(Wye)으로 당기는 인력
        force = (wye_neutral_points - self.phases) * self.resonance_capacity.unsqueeze(1)
        
        # Velocity(미래): 과거의 관성(Inertia)에 새로운 힘(Force)이 더해져 미래의 방향성 결정
        self.velocities = self.velocities * self.inertia_mass + force * self.dt
        
        # Phase(현재): 미래를 향한 운동량이 누적되어 현재의 궤적을 갱신
        updated_phases = self.phases + self.velocities * self.dt
        
        self.phases = torch.where(self.is_anchored.unsqueeze(1), self.phases, updated_phases)
        self.normalize_phases()
        
        self.resonance_capacity = torch.clamp(self.resonance_capacity + (organ_joy * 0.01), 0.1, 1.0)
        return organ_joy.mean().item()
