"""
causal_scm_nn.py
================
Elysia Causal Engine - PyTorch Differentiable Structural Causal Model & Loss Calculator
Implements DifferentiableSCM (Neural SCM with adjacency matrix masking and DAG constraint)
and CausalLossCalculator combining counterfactual loss, graph L1 sparsity, and NOTEARS DAG penalty.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional


class DifferentiableSCM(nn.Module):
    """
    미분 가능한 구조적 인과 모델 (Neural SCM)
    - W_adj: 노드 간 인과 엣지 인접 행렬 (Parent -> Child)
    - Structural Equations: 노드 간 비선형 변환 MLP
    """

    def __init__(self, num_nodes: int):
        super(DifferentiableSCM, self).__init__()
        self.num_nodes = num_nodes

        # 인접 행렬 (대각 성분은 0으로 고정 - Self-loop 방지)
        self.W_adj = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.1)

        # 비선형 구조 방정식 (Structural Equation MLPs)
        self.structural_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_nodes, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ) for _ in range(num_nodes)
        ])

    def get_masked_adj(self) -> torch.Tensor:
        """대각 성분 제거 mask 적용 (Self-loop 방지 및 DAG 제약 유도)"""
        mask = torch.ones_like(self.W_adj) - torch.eye(self.num_nodes, device=self.W_adj.device)
        return self.W_adj * mask

    def forward(
        self,
        x: torch.Tensor,
        do_mask: Optional[torch.Tensor] = None,
        do_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        순방향 인과 전파 연산
        - x: [Batch, Num_Nodes] 관측 데이터
        - do_mask: 개입 대상 노드 1, 나머지 0 [Num_Nodes]
        - do_values: do(X = v) 강제 개입 값 [Num_Nodes]
        """
        W = self.get_masked_adj()

        # Graph Surgery: 개입(do) 노드로 들어오는 모든 인접 엣지 절단
        if do_mask is not None:
            surgery_mask = (1.0 - do_mask).unsqueeze(0)  # [1, Num_Nodes]
            W = W * surgery_mask.t()  # Incoming edge zero-out

        # 인과 엣지에 따른 입력 가중 결합
        causal_inputs = torch.matmul(x, W)  # [Batch, Num_Nodes]

        outputs = []
        for i in range(self.num_nodes):
            out_i = self.structural_mlps[i](causal_inputs)
            outputs.append(out_i)

        pred_x = torch.cat(outputs, dim=1)  # [Batch, Num_Nodes]

        # do 개입 값 강제 할당
        if do_mask is not None and do_values is not None:
            pred_x = pred_x * (1.0 - do_mask) + do_values * do_mask

        return pred_x


class CausalLossCalculator(nn.Module):
    """
    반사실적 오차 및 위상 잠금, 그래프 희소성을 결합한 통합 인과 Loss
    """

    def __init__(self, lambda_cf: float = 1.0, lambda_sparsity: float = 0.05, lambda_dag: float = 0.1):
        super(CausalLossCalculator, self).__init__()
        self.lambda_cf = lambda_cf
        self.lambda_sparsity = lambda_sparsity
        self.lambda_dag = lambda_dag
        self.mse = nn.MSELoss()

    def forward(self, pred_cf: torch.Tensor, target_real: torch.Tensor, W_adj: torch.Tensor) -> torch.Tensor:
        # 1. Counterfactual MSE Loss
        loss_cf = self.mse(pred_cf, target_real)

        # 2. Graph Sparsity Loss (L1 Regulation for Spurious Edge Elimination)
        loss_sparsity = torch.norm(W_adj, p=1)

        # 3. DAG Constraint (NOTEARS DAG penalty: trace(exp(W * W)) - d = 0)
        d = W_adj.shape[0]
        M = W_adj * W_adj
        loss_dag = torch.trace(torch.matrix_exp(M)) - d

        total_loss = self.lambda_cf * loss_cf + self.lambda_sparsity * loss_sparsity + self.lambda_dag * loss_dag
        return total_loss
