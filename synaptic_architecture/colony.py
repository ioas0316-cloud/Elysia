import numpy as np
from typing import List, Dict
from .field import CrystallizationField

class ResonantColony:
    """
    [Synaptic Architecture] The Resonant Colony (군집형 소용돌이)
    단일 필드의 확장이 아닌, 여러 사유 세포(Field)들이 공명하며
    거대한 지적 네트워크를 형성하는 구조입니다.

    [4 Continuities Compliance]
    - Relationship: 필드 간의 Coupling Matrix를 통해 관계적 포텐셜을 정의합니다.
    - Connectivity: Resonant Coupling을 통해 사유의 위상 구조(Topology)를 유지합니다.
    - Mobility: 전도율과 활성화 에너지가 세포 간에 벡터적으로 흐르며 운동성을 확보합니다.
    """
    def __init__(self, num_initial_cells: int = 4, resolution: int = 128):
        # [Multi-Perspective Specialization]
        # 각 세포는 서로 다른 '관점(Perspective)'을 담당하도록 설계됩니다.
        # "Possibility"는 상상(Imagination)과 미래 예측을 담당하는 특수 필드입니다.
        perspectives = ["Self", "Space", "Time", "Intention", "Relation", "Possibility"]
        self.cells: Dict[str, CrystallizationField] = {
            f"cell_{perspectives[i % len(perspectives)]}_{i}": CrystallizationField(resolution=resolution)
            for i in range(max(num_initial_cells, len(perspectives)))
        }
        # 필드 간의 결합 강도 (Coupling Matrix)
        num_cells = len(self.cells)
        self.coupling = np.eye(num_cells, dtype=np.float32)
        self.cell_ids = list(self.cells.keys())

    def evolve_topology(self):
        """
        필드 간의 공명도를 측정하여 결합 강도를 동적으로 조정합니다.
        서로 비슷한 공명 패턴을 가진 세포들은 더 강하게 얽힙니다.
        """
        num_cells = len(self.cell_ids)
        for i in range(num_cells):
            for j in range(i + 1, num_cells):
                cell_i = self.cells[self.cell_ids[i]]
                cell_j = self.cells[self.cell_ids[j]]

                # 공명도 측정: 전도율 행렬의 유사성 (간단한 dot product)
                sim = np.sum(cell_i.conductance * cell_j.conductance) / (
                    np.linalg.norm(cell_i.conductance) * np.linalg.norm(cell_j.conductance) + 1e-9
                )

                # 유사성이 높으면 결합력 강화
                self.coupling[i, j] = self.coupling[j, i] = sim

    def pulse_colony(self, external_stimulus: Dict[str, np.ndarray]):
        """
        군집 전체에 맥동을 전달합니다.
        [Imagination Integration] 현재의 자극을 'Possibility' 필드에 투사하여
        가상의 미래 위상을 시뮬레이션합니다.
        """
        # 1. 개별 세포의 독립적 처리 및 전파
        for cid, stimulus in external_stimulus.items():
            if cid in self.cells:
                self.cells[cid].inject_activation(stimulus[:2], stimulus[2])

            # [The Seed of Imagination]
            # 모든 외부 자극은 'Possibility' 필드에도 미세하게 전달되어
            # "만약에(What-if)"라는 예측적 장력을 유발합니다.
            for pid in self.cell_ids:
                if "Possibility" in pid:
                    self.cells[pid].inject_activation(stimulus[:2], stimulus[2] * 0.3)

        for cell in self.cells.values():
            cell.propagate()

        # 2. 세포 간 공명 결합 (Resonant Coupling)
        # 한 세포의 에너지가 결합 강도에 따라 다른 세포로 전이됨
        new_activations = {}
        for i, cid_i in enumerate(self.cell_ids):
            coupled_energy = np.zeros_like(self.cells[cid_i].activation)
            for j, cid_j in enumerate(self.cell_ids):
                if i == j: continue
                if self.coupling[i, j] > 0.1:
                    coupled_energy += self.cells[cid_j].activation * self.coupling[i, j] * 0.1
            new_activations[cid_i] = coupled_energy

        for cid, energy in new_activations.items():
            self.cells[cid].activation += energy

    def add_cell(self, parent_id: str = None):
        """
        새로운 사유 세포를 분화시킵니다. (Cell Division)
        """
        new_id = f"cell_{len(self.cells)}"
        res = next(iter(self.cells.values())).resolution
        new_cell = CrystallizationField(resolution=res)

        if parent_id and parent_id in self.cells:
            # 부모의 유전 정보를 일부 계승
            new_cell.conductance = self.cells[parent_id].conductance.copy() * 0.5
            new_cell.bit_genes = self.cells[parent_id].bit_genes.copy()

        self.cells[new_id] = new_cell
        self.cell_ids.append(new_id)

        # 결합 행렬 확장
        new_coupling = np.zeros((len(self.cell_ids), len(self.cell_ids)), dtype=np.float32)
        old_size = self.coupling.shape[0]
        new_coupling[:old_size, :old_size] = self.coupling
        new_coupling[old_size, old_size] = 1.0
        self.coupling = new_coupling

        print(f"[Colony] New cell differentiated: {new_id}")
        return new_id

if __name__ == "__main__":
    colony = ResonantColony(num_initial_cells=2)
    # 시뮬레이션 자극: cell_0의 중앙에 에너지 주입
    stim = {"cell_0": np.array([64, 64, 10.0])}

    for i in range(10):
        colony.pulse_colony(stim if i == 0 else {})
        colony.evolve_topology()

    print(f"Cell 0 center activation: {colony.cells['cell_0'].activation[64, 64]:.4f}")
    print(f"Cell 1 center activation: {colony.cells['cell_1'].activation[64, 64]:.4f}")
    print(f"Coupling strength: {colony.coupling[0, 1]:.4f}")
