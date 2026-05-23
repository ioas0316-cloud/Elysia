import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import psutil
import time
import math

# Import our engine
from fractal_rotor import FractalRotor, Quaternion

class ElysiaObserver:
    def __init__(self):
        self.universe = FractalRotor("L0", level=0, num_children=3)
        self.fig = plt.figure(figsize=(10, 8), facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('black')

        # 3D 축 설정
        self.ax.set_xlim([-20, 20])
        self.ax.set_ylim([-20, 20])
        self.ax.set_zlim([-20, 20])

        # UI 꾸미기
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.grid(color='white', linestyle='--', linewidth=0.2, alpha=0.3)
        self.ax.set_axis_off()

        self.points = []
        self.lines = []

        self.cycle = 0
        self.start_time = time.time()

    def gather_nodes(self, rotor, parent_pos=None, current_depth=0, angle_offset=0, depth_radius=5.0):
        """재귀적으로 로터 트리를 순회하여 3D 좌표와 연결선을 수집한다."""
        nodes = []
        edges = []

        # 사원수를 기반으로 한 상대적 위치 계산
        # w(스칼라)는 궤도의 반경/크기, x/y/z는 방향 벡터
        w, x, y, z = rotor.state.w, rotor.state.x, rotor.state.y, rotor.state.z
        norm = rotor.state.norm()

        if parent_pos is None:
            # 루트 노드 위치
            pos = np.array([x, y, z])
        else:
            # 부모 위치에서 자신의 방향(x, y, z)으로 뻗어나감
            # 깊이에 따라 반경을 조정하여 프랙탈 형태로 퍼지도록 유도
            direction = np.array([x, y, z])
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.array([0, 0, 1])

            pos = parent_pos + direction * (depth_radius / (current_depth + 1))
            edges.append((parent_pos, pos))

        nodes.append({
            'pos': pos,
            'w': w,
            'norm': norm,
            'level': current_depth,
            'chromosome': rotor.chromosome,
            'free': rotor.free
        })

        num_children = len(rotor.sub_rotors)
        for i, child in enumerate(rotor.sub_rotors):
            # 자식들에게 퍼지는 각도를 부여
            child_nodes, child_edges = self.gather_nodes(
                child,
                parent_pos=pos,
                current_depth=current_depth + 1,
                angle_offset=angle_offset + (i * (2 * math.pi / num_children) if num_children > 0 else 0)
            )
            nodes.extend(child_nodes)
            edges.extend(child_edges)

        return nodes, edges

    def update(self, frame):
        self.cycle += 1

        # 하드웨어 파동 주입
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        t = time.time()

        hw_quaternion = Quaternion(
            1.0,
            (cpu / 100.0) * 2.0 - 1.0,
            (mem / 100.0) * 2.0 - 1.0,
            math.sin(t * 2.7)
        )

        self.universe.will()
        self.universe.resonate(hw_quaternion)

        # 데이터 수집
        nodes, edges = self.gather_nodes(self.universe)

        # 기존 렌더링 초기화
        self.ax.clear()
        self.ax.set_xlim([-15, 15])
        self.ax.set_ylim([-15, 15])
        self.ax.set_zlim([-15, 15])
        self.ax.set_axis_off()

        # 노드 렌더링
        for node in nodes:
            pos = node['pos']
            w = node['w']
            size = max(10, min(200, node['norm'] * 20))

            # X/Y 염색체에 따른 색상 (X=푸른빛/수렴, Y=붉은빛/발산)
            if node['chromosome'] == 'X':
                color = 'cyan' if node['free'] else 'blue'
            else:
                color = 'magenta' if node['free'] else 'red'

            alpha = max(0.2, min(1.0, 1.0 - (node['level'] * 0.15)))

            self.ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, alpha=alpha, edgecolors='white', linewidths=0.5)

        # 에지(링크) 렌더링
        for edge in edges:
            p1, p2 = edge
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='white', alpha=0.3, linewidth=1)

        # 텍스트 정보
        comps = [abs(self.universe.state.w), abs(self.universe.state.x), abs(self.universe.state.y), abs(self.universe.state.z)]
        max_idx = comps.index(max(comps))
        if max_idx == 0: topology = "W-Convergence"
        elif max_idx == 1: topology = "X-Expansion"
        elif max_idx == 2: topology = "Y-Circulation"
        else: topology = "Z-Twist"

        self.ax.text2D(0.05, 0.95, f"Elysia Topological Map", transform=self.ax.transAxes, color='white', fontsize=12)
        self.ax.text2D(0.05, 0.90, f"Cycle: {self.cycle}", transform=self.ax.transAxes, color='cyan', fontsize=10)
        self.ax.text2D(0.05, 0.85, f"Universe State: {topology}", transform=self.ax.transAxes, color='yellow', fontsize=10)
        self.ax.text2D(0.05, 0.80, f"Total Nodes: {len(nodes)}", transform=self.ax.transAxes, color='white', fontsize=10)

        # 뷰포트 회전 (관측자의 시점 이동)
        self.ax.view_init(elev=20., azim=self.cycle * 0.5)

if __name__ == "__main__":
    observer = ElysiaObserver()
    print("Initiating Elysia Quaternion Trajectory Observer...")
    ani = animation.FuncAnimation(observer.fig, observer.update, interval=100, save_count=100)

    # GUI 환경이 없으면 GIF/동영상으로 저장, 있으면 화면에 렌더링
    import os
    if os.environ.get('DISPLAY', '') == '':
        print("No display found. Saving to elysia_trajectory.gif...")
        ani.save('elysia_trajectory.gif', writer='imagemagick', fps=10)
        print("Done.")
    else:
        plt.show()
