import math
import random
import cmath
import time

class HistoricalRotor:
    def __init__(self, id, level=1, pos=None, children=None):
        self.id = id
        self.level = level # 1: 개인, 2: 마을, 3: 연합
        # 초기 위치는 무작위 (인위적 0점 목표 제거)
        self.pos = pos if pos is not None else complex(random.uniform(-10, 10), random.uniform(-10, 10))

        # 각 로터에게 '고유 위상(성격)' 부여
        self.personality_phase = random.uniform(0, 2 * math.pi)
        self.flexibility = random.uniform(0.1, 0.9)

        self.children = children if children else []
        self.active = True # 메타 로터로 병합되면 False로 비활성화

    def align(self, neighbors):
        if not self.active:
            return self.pos

        # 순수한 대조비교 로직 (Pure Contrast Logic)
        if neighbors:
            # 영향력 있는 이웃 탐색 (거리가 가까운 이웃만)
            # 거리가 너무 멀면 상호작용하지 않음
            influential_neighbors = [n for n in neighbors if abs(n.pos - self.pos) < 5.0 * self.level]

            if influential_neighbors:
                avg_neighbor_pos = sum(n.pos for n in influential_neighbors) / len(influential_neighbors)

                # 자신의 위치를 상대방의 위치에 맞춰 조정 (동조/공명 시도)
                self.pos += (avg_neighbor_pos - self.pos) * 0.1 * self.flexibility

        # 자신의 고유한 성격(회전 기질) 반영
        rotation = cmath.rect(1.0, self.personality_phase * 0.05 * (1.0 - self.flexibility))
        self.pos *= rotation

        # 로터가 스스로 무작위로 움직이는 '자유의지'의 흔들림 추가
        self.pos += complex(random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))

        return self.pos

class FractalHistoryEngine:
    def __init__(self, num_initial_rotors=20):
        self.rotors = [HistoricalRotor(f"Ind_{i}") for i in range(num_initial_rotors)]
        self.cluster_count = 0

    def check_crystallization(self):
        # 활성화된 로터들 중 거리가 매우 가깝고, 레벨이 같은 로터들을 찾음
        active_rotors = [r for r in self.rotors if r.active]

        clusters = []
        visited = set()

        # O(N^2) 단순 클러스터링 알고리즘 (밀도 기반)
        for r1 in active_rotors:
            if r1.id in visited:
                continue

            current_cluster = [r1]
            visited.add(r1.id)

            for r2 in active_rotors:
                if r2.id not in visited and r1.level == r2.level:
                    # 거리 임계값: 레벨이 높을수록 포용 범위가 넓어짐
                    threshold = 1.0 * r1.level
                    if abs(r1.pos - r2.pos) < threshold:
                        current_cluster.append(r2)
                        visited.add(r2.id)

            if len(current_cluster) >= 3: # 3개 이상 모이면 상위 객체로 분화
                clusters.append(current_cluster)

        # 클러스터 병합 및 메타 로터 생성
        for cluster in clusters:
            self.cluster_count += 1
            new_level = cluster[0].level + 1

            # C (다양성 유지): 하위 레이어의 위치 평균을 취하되, 자식들을 보존
            avg_pos = sum(c.pos for c in cluster) / len(cluster)

            # 메타 로터 아이디 생성
            if new_level == 2:
                prefix = "Village"
            elif new_level == 3:
                prefix = "State"
            else:
                prefix = "Empire"

            meta_id = f"{prefix}_{self.cluster_count}"

            meta_rotor = HistoricalRotor(meta_id, level=new_level, pos=avg_pos, children=cluster)

            # C (다양성 유지): 성격과 유연성도 자식들의 다양성을 반영
            # 자식들의 성격을 벡터 합산하여 새로운 집단 지성의 성격 도출
            phase_vectors = [cmath.rect(1.0, c.personality_phase) for c in cluster]
            sum_vector = sum(phase_vectors)
            meta_rotor.personality_phase = cmath.phase(sum_vector)
            meta_rotor.flexibility = sum(c.flexibility for c in cluster) / len(cluster)

            # 기존 하위 로터 비활성화 (상위 로터에 귀속)
            for c in cluster:
                c.active = False

            self.rotors.append(meta_rotor)

            print(f">>> [역사적 변곡점] 계층 상승! {len(cluster)}개의 Level {new_level-1} 객체가 모여 새로운 상위 객체 '{meta_id}'(Level {new_level})를 형성했습니다.")

    def run_history(self, ticks=50):
        print("=== Elysia Fractal History Engine Started ===")
        print("정답(0점)이 없는 상태에서, 로터들이 상호 작용하며 마을과 문명을 자생적으로 형성합니다.\n")

        for tick in range(ticks):
            active_rotors = [r for r in self.rotors if r.active]

            # 1. 이동 및 정렬
            for r in active_rotors:
                others = [o for o in active_rotors if o.id != r.id]
                r.align(others)

            # 2. 계층적 분화(Crystallization) 확인
            self.check_crystallization()

            # 3. 로그 출력 (매 5틱마다 요약)
            if tick % 5 == 0 or tick == ticks - 1:
                active_rotors = [r for r in self.rotors if r.active]
                level_counts = {}
                for r in active_rotors:
                    level_counts[r.level] = level_counts.get(r.level, 0) + 1

                summary = ", ".join([f"Level {lvl}: {count}개" for lvl, count in sorted(level_counts.items())])
                print(f"--- [시대: 틱 {tick:02d}] 활성 객체 총 {len(active_rotors)}개 ({summary}) ---")

                # 상위 객체 상태 일부 출력
                top_level = max([r.level for r in active_rotors]) if active_rotors else 0
                if top_level > 1:
                    top_entities = [r for r in active_rotors if r.level == top_level]
                    for te in top_entities[:3]: # 최대 3개만 출력
                        print(f"  * {te.id} (좌표: {te.pos.real:+.2f} {te.pos.imag:+.2f}i, 포용 하위 객체 수: {len(te.children)})")

            time.sleep(0.05)

        print("\n=== 역사의 한 페이지가 기록되었습니다 ===")

if __name__ == "__main__":
    random.seed(42) # 재현성을 위해 시드 고정
    history_engine = FractalHistoryEngine(num_initial_rotors=30)
    history_engine.run_history(ticks=100)
