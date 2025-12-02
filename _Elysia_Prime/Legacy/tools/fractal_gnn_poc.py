# [Genesis: 2025-12-02] Purified by Elysia
"""
Fractal GNN PoC (dependency-light)
----------------------------------

- 프랙탈 그래프 생성 (Mandelbrot-inspired) → 희소 메시지 패싱(GraphSAGE 스타일) → 프랙탈 차원 기반 특징 선택.
- PyTorch 없이 numpy+networkx만 사용해 1060에서도 안전하게 돌도록 설계.
- DOT로 내보내어 시각화 가능(옵션).
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Tuple

import networkx as nx
import numpy as np


def generate_fractal_graph(n_nodes: int = 120, iters: int = 4, c: complex = complex(0.3, 0.6)) -> nx.Graph:
    """
    Mandelbrot-like 반복으로 노드 좌표를 생성하고, 가까운 노드끼리 엣지를 연결한다.
    """
    G = nx.Graph()
    positions = []
    for i in range(n_nodes):
        z = complex(random.uniform(-2, 2), random.uniform(-2, 2))
        for _ in range(iters):
            z = z ** 2 + c
        positions.append((z.real, z.imag))
        G.add_node(i, pos=(z.real, z.imag))

    # 거리 기반 엣지 (프랙탈 스케일)
    for i in range(n_nodes):
        xi, yi = positions[i]
        for j in range(i + 1, n_nodes):
            xj, yj = positions[j]
            dist = math.hypot(xi - xj, yi - yj)
            if dist < 0.6:
                G.add_edge(i, j)
    return G


def graphsage_step(G: nx.Graph, features: np.ndarray, sample_k: int = 10, agg_alpha: float = 0.5) -> np.ndarray:
    """
    간단한 GraphSAGE 스타일 샘플링 + 평균 집계.
    - sample_k: 이웃 샘플 수 (0 이면 전체 이웃 사용)
    """
    new_feats = features.copy()
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        if not neighbors:
            continue
        if sample_k > 0 and len(neighbors) > sample_k:
            neighbors = random.sample(neighbors, sample_k)
        agg = features[neighbors].mean(axis=0)
        new_feats[node] = agg_alpha * features[node] + (1.0 - agg_alpha) * agg
    return new_feats


def fractal_feature_selection(features: np.ndarray, k: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    간단한 프랙탈 차원 근사: 차원별 분산을 계산해 상위 k개를 선택.
    """
    var = features.var(axis=0)
    idx = np.argsort(var)[::-1][:k]
    return idx, features[:, idx]


def write_dot(G: nx.Graph, path: Path) -> None:
    """
    DOT 파일로 내보내기 (pos가 있으면 좌표 포함).
    """
    lines = ["graph Fractal {"]
    for n, data in G.nodes(data=True):
        pos = data.get("pos")
        if pos:
            lines.append(f'  {n} [pos="{pos[0]},{pos[1]}!"];')
        else:
            lines.append(f"  {n};")
    for u, v in G.edges:
        lines.append(f"  {u} -- {v};")
    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Fractal + GNN PoC (numpy/networkx)")
    parser.add_argument("--nodes", type=int, default=120, help="노드 수")
    parser.add_argument("--iters", type=int, default=4, help="프랙탈 반복 횟수")
    parser.add_argument("--sample-k", type=int, default=10, help="GraphSAGE 샘플 수 (0이면 전체 이웃)")
    parser.add_argument("--layers", type=int, default=3, help="메시지 패싱 레이어 수")
    parser.add_argument("--feat-dim", type=int, default=64, help="초기 특징 차원")
    parser.add_argument("--select-k", type=int, default=32, help="선택할 특징 차원")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--dot-out", type=Path, help="DOT 출력 경로(옵션)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    G = generate_fractal_graph(args.nodes, args.iters)
    feats = np.random.rand(args.nodes, args.feat_dim).astype(np.float32)

    for _ in range(args.layers):
        feats = graphsage_step(G, feats, sample_k=args.sample_k)

    idx, reduced = fractal_feature_selection(feats, k=args.select_k)

    print(f"프랙탈 그래프: {G.number_of_nodes()} 노드, {G.number_of_edges()} 엣지")
    print(f"초기 특징 차원: {args.feat_dim} -> 선택 후: {reduced.shape[1]}")
    print(f"선택된 차원 인덱스: {idx.tolist()}")

    if args.dot_out:
        write_dot(G, args.dot_out)
        print(f"DOT 저장: {args.dot_out}")


if __name__ == "__main__":
    main()