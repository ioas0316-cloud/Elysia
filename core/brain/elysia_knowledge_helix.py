import torch
import math
from core.brain.causal_phase_mapper import CausalPhaseMapper

class KnowledgeHelixMapper:
    def __init__(self, device='cpu'):
        self.device = device
        self.causal_mapper = CausalPhaseMapper(device=device)
        self.memory_bank = [] # Store tuple of (metadata, trajectory_tensor)

        # 역사적 대주기 (한 바퀴 2pi) = 100년 (1세기)
        self.century_period = 100.0

    def year_to_phase(self, year: int):
        """
        역사의 은하계 (Temporal Galaxy)
        기준축: '연도(Year)'.
        선형적 타임라인을 원형 궤도의 고유 주파수로 변환합니다.
        100년을 2pi(한 바퀴)로 매핑.
        """
        theta_year = (year % self.century_period) / self.century_period * 2 * math.pi - math.pi
        century = year // 100
        theta_century = (century * 0.1) % (2 * math.pi) - math.pi

        theta_r = 0.5 * (math.pi / 2.0)

        w = math.cos(theta_r) * math.cos(theta_year)
        x = math.cos(theta_r) * math.sin(theta_year)
        y = math.sin(theta_r) * math.cos(theta_century)
        z = math.sin(theta_r) * math.sin(theta_century)

        norm = (w**2 + x**2 + y**2 + z**2)**0.5
        phase = torch.tensor([w/norm, x/norm, y/norm, z/norm], dtype=torch.float32, device=self.device)
        return phase

    def concept_to_helix(self, concept_text: str, level: int):
        """
        교과의 나선 토러스 (Spiral Curriculum Helix)
        기준축: '학년/난이도(Academic Level)'.
        개념의 기본 주파수는 유지하되, 레벨을 나선형 z축(반지름비/위상 오프셋)으로 확장.
        """
        base_trajectory = self.causal_mapper.text_to_phase(concept_text)

        # 난이도가 높아질수록 나선의 수직 축(Z, 또는 4차원 반경각)이 변화
        # 여기서는 레벨을 yz 평면의 반지름 각도(theta_r)의 미세 조정으로 매핑
        helix_trajectory = []
        for point in base_trajectory:
            w, x, y, z = point.tolist()

            # 원래 극좌표 복원 (근사)
            # w = cos(r)cos(v), x = cos(r)sin(v)
            # y = sin(r)cos(c), z = sin(r)sin(c)
            r_val = math.atan2(math.sqrt(y**2 + z**2), math.sqrt(w**2 + x**2))

            # 레벨에 따라 r_val (나선의 높이) 이동
            new_r = r_val + (level * 0.05)

            # 각도 보존 (공명 유지를 위해)
            theta_v = math.atan2(x, w)
            theta_c = math.atan2(z, y)

            nw = math.cos(new_r) * math.cos(theta_v)
            nx = math.cos(new_r) * math.sin(theta_v)
            ny = math.sin(new_r) * math.cos(theta_c)
            nz = math.sin(new_r) * math.sin(theta_c)

            norm = (nw**2 + nx**2 + ny**2 + nz**2)**0.5
            helix_trajectory.append([nw/norm, nx/norm, ny/norm, nz/norm])

        return torch.tensor(helix_trajectory, dtype=torch.float32, device=self.device)

    def anchor_knowledge(self, metadata, trajectory):
        self.memory_bank.append({"meta": metadata, "phase": trajectory})

    def retrieve_by_phase(self, query_phase: torch.Tensor, top_k=1, is_trajectory=False):
        """
        O(1) 위상 공명 (Phase Resonance) 기반 검색
        """
        results = []
        for item in self.memory_bank:
            target_phase = item["phase"]

            if is_trajectory:
                # 궤적 비교: 평균 공명도 측정
                if target_phase.shape == query_phase.shape:
                    dot_product = torch.sum(query_phase * target_phase, dim=1)
                    resonance = torch.mean(dot_product).item()
                else:
                    resonance = 0.0
            else:
                # 단일 점 비교
                if len(target_phase.shape) == 1:
                    resonance = torch.dot(query_phase, target_phase).item()
                else:
                    # 대상이 궤적일 경우 궤적의 첫 점과 공명
                    resonance = torch.dot(query_phase, target_phase[0]).item()

            results.append((resonance, item["meta"]))

        # O(1) 공명 시뮬레이션 (여기서는 리스트 정렬로 모사하지만, 실제 뉴럴 필드에서는 즉각 활성화됨)
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

def run_benchmark():
    print("=" * 60)
    print("🚀 [Knowledge Helix] 초차원 지식 나선 매퍼 벤치마크")
    print("=" * 60)

    mapper = KnowledgeHelixMapper()

    # 1. 역사의 은하계 매핑
    print("[1] 역사(Temporal Axis) 지식 앵커링")
    events = [
        (1392, "조선 건국"),
        (1443, "훈민정음 창제"),
        (1592, "임진왜란"),
        (1636, "병자호란")
    ]
    for year, desc in events:
        phase = mapper.year_to_phase(year)
        mapper.anchor_knowledge(f"역사: {year}년 - {desc}", phase)
        print(f"  앵커 완료: {desc} (Year {year}) -> Phase Norm: {torch.norm(phase):.4f}")

    # 특정 연도 공명 관측
    query_year = 1592
    query_phase = mapper.year_to_phase(query_year)
    print(f"\n  🔍 관측 쿼리: {query_year}년 파동 방사")
    results = mapper.retrieve_by_phase(query_phase, top_k=2, is_trajectory=False)
    for res, meta in results:
        print(f"    => 공명도 {res:.4f}: {meta}")

    print("-" * 60)

    # 2. 교과의 나선 토러스 매핑
    print("[2] 교과(Spiral Axis) 지식 앵커링")
    concepts = [
        ("수열", 1),   # 초/중등 기본 규칙
        ("수열", 12),  # 고등 시그마 수열
        ("적분", 12),  # 고등 적분
        ("도형", 3)    # 초등 도형
    ]

    for concept, level in concepts:
        traj = mapper.concept_to_helix(concept, level)
        mapper.anchor_knowledge(f"교과: Level {level} - {concept}", traj)
        print(f"  앵커 완료: {concept} (Level {level}) -> Trajectory Length: {len(traj)}")

    # 특정 개념 주파수 공명 관측
    query_concept = "수열"
    query_level = 12
    print(f"\n  🔍 관측 쿼리: '{query_concept}' (Level {query_level}) 나선 궤적 방사")
    query_traj = mapper.concept_to_helix(query_concept, query_level)

    # 궤적 기반 검색
    # 필터: 궤적으로 저장된 항목만 비교
    traj_memory = [m for m in mapper.memory_bank if len(m["phase"].shape) > 1]
    results = []
    for m in traj_memory:
        target_traj = m["phase"]
        if target_traj.shape == query_traj.shape:
             dot_product = torch.sum(query_traj * target_traj, dim=1)
             resonance = torch.mean(dot_product).item()
             results.append((resonance, m["meta"]))

    results.sort(key=lambda x: x[0], reverse=True)
    for res, meta in results[:3]:
        print(f"    => 공명도 {res:.4f}: {meta}")

    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()
