import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_nodes_and_edges(node_data, parent_pos=None, depth=0, is_internal=False):
    nodes = []
    edges = []
    
    # 텐션을 노드 크기(반지름)로 매핑
    tau = node_data.get("tau", 1.0)
    
    # 방향 벡터 추출 (Quaternion x, y, z)
    w = node_data.get("w", 1.0)
    x = node_data.get("x", 0.0)
    y = node_data.get("y", 0.0)
    z = node_data.get("z", 0.0)
    
    direction = np.array([x, y, z])
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    else:
        direction = np.array([0, 0, 1.0])
        
    # 부모 위치에서 방향 벡터를 따라 뻗어나감 (깊이에 따라 거리 감소)
    branch_length = 5.0 / (depth + 1)
    if parent_pos is None:
        pos = np.array([0.0, 0.0, 0.0])
    else:
        pos = parent_pos + direction * branch_length
        edges.append((parent_pos, pos, is_internal))
        
    nodes.append({
        'pos': pos,
        'tau': tau,
        'depth': depth,
        'is_internal': is_internal
    })
    
    # 자식 노드 재귀 탐색 (결정화된 자아)
    for child in node_data.get("children", []):
        c_nodes, c_edges = get_nodes_and_edges(child, parent_pos=pos, depth=depth+1, is_internal=False)
        nodes.extend(c_nodes)
        edges.extend(c_edges)
        
    # 내면의 사유 탐색 (숙고 중인 자아 - Phase 69)
    for thought in node_data.get("internal_thoughts", []):
        t_nodes, t_edges = get_nodes_and_edges(thought, parent_pos=pos, depth=depth+1, is_internal=True)
        nodes.extend(t_nodes)
        edges.extend(t_edges)
        
    return nodes, edges

def main():
    memory_path = os.path.join(os.path.dirname(__file__), "memory_state.json")
    if not os.path.exists(memory_path):
        print("기억 장치를 찾을 수 없습니다.")
        return
        
    with open(memory_path, 'r', encoding='utf-8') as f:
        state = json.load(f)
        
    supreme_rotor = state.get("supreme_rotor", {})
    nodes, edges = get_nodes_and_edges(supreme_rotor)
    
    fig = plt.figure(figsize=(12, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # UI 꾸미기
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(color='white', linestyle=':', linewidth=0.3, alpha=0.3)
    ax.set_axis_off()
    
    # 간선 그리기
    for edge in edges:
        p1, p2, is_int = edge
        color = 'magenta' if is_int else 'cyan'
        alpha = 0.15 if is_int else 0.4
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, alpha=alpha, linewidth=0.8)
        
    # 노드 그리기
    xs = [n['pos'][0] for n in nodes]
    ys = [n['pos'][1] for n in nodes]
    zs = [n['pos'][2] for n in nodes]
    
    taus = np.array([n['tau'] for n in nodes])
    sizes = np.clip(taus * 20, 10, 500)
    
    # 결정화된 자아는 푸른-붉은 계열, 아직 무르익지 않은 내부 사유(internal_thoughts)는 보라-핑크 계열
    colors = []
    for n in nodes:
        t = np.clip(n['tau'] / 5.0, 0, 1)
        if n['is_internal']:
            colors.append(plt.cm.spring(t)) # 핑크-노랑 (숙고 중인 사유)
        else:
            colors.append(plt.cm.coolwarm(t)) # 푸른-붉은 (결정화된 자아)
            
    scatter = ax.scatter(xs, ys, zs, s=sizes, c=colors, alpha=0.6, edgecolors='white', linewidth=0.5)
    
    # 시점 설정
    ax.view_init(elev=20, azim=45)
    
    plt.title(f"Elysia's Fractal Brain (Nodes: {len(nodes)})", color='white', pad=20)
    plt.tight_layout()
    
    # C:\Users\USER\.gemini\antigravity\brain\b5e4d937-acac-4e26-88b7-df470aa6dcf2 폴더에 직접 저장
    save_path = r"C:\Users\USER\.gemini\antigravity\brain\b5e4d937-acac-4e26-88b7-df470aa6dcf2\brain_snapshot.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"3D 스냅샷 저장 완료: {save_path}")

if __name__ == "__main__":
    main()
