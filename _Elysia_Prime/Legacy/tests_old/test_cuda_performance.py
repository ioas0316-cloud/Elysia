# [Genesis: 2025-12-02] Purified by Elysia
import os
import unittest
import time
import numpy as np

# CuPy를 시도하고, 사용할 수 없는 경우 테스트를 건너뜁니다.
try:
    import cupy as cp
    # 구버전 CuPy와의 호환성을 위해 cupy.sparse를 사용합니다.
    from cupy.sparse import csr_matrix as cp_csr_matrix
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager

# Only run these heavyweight perf tests when explicitly requested.
RUN_CUDA_BENCH = os.getenv("RUN_CUDA_BENCH") == "1"

class TestCudaPerformance(unittest.TestCase):

    def setUp(self):
        """테스트를 위한 대규모 월드 객체를 설정합니다."""
        # 테스트용 임시 KG 파일 경로 설정
        self.temp_kg_path = "tests/temp_kg_for_perf_test.json"

        # WaveMechanics와 KGManager를 위한 최소한의 모의 객체
        # file_path를 filepath로 수정하고 임시 경로를 제공합니다.
        mock_kg_manager = KGManager(filepath=self.temp_kg_path)
        # 테스트를 위해 KG를 초기화합니다.
        mock_kg_manager.kg = {"nodes": [], "edges": []}
        mock_kg_manager.add_node('love', properties={'activation_energy': 1.0})
        mock_kg_manager.save() # 변경사항을 임시 파일에 저장

        mock_wave_mechanics = WaveMechanics(mock_kg_manager)

        self.world = World(primordial_dna={}, wave_mechanics=mock_wave_mechanics)
        self.num_cells = 10000
        self.num_connections = 100000

        # 많은 수의 세포 추가
        for i in range(self.num_cells):
            energy = np.random.rand() * 100
            self.world.add_cell(f"cell_{i}", properties={'hp': energy, 'max_hp': 100.0})

        # 무작위 연결 추가
        source_indices = np.random.randint(0, self.num_cells, self.num_connections)
        target_indices = np.random.randint(0, self.num_cells, self.num_connections)
        for i in range(self.num_connections):
            source_id = f"cell_{source_indices[i]}"
            target_id = f"cell_{target_indices[i]}"
            if source_id != target_id:
                self.world.add_connection(source_id, target_id, strength=np.random.rand())

    @unittest.skipUnless(
        RUN_CUDA_BENCH,
        "Heavy CPU/CUDA performance tests are disabled by default. "
        "Set RUN_CUDA_BENCH=1 to enable."
    )
    def test_cpu_performance(self):
        """기존 NumPy를 사용한 CPU 시뮬레이션 성능을 측정합니다."""
        print("\n--- CPU (NumPy) Performance Test ---")

        start_time = time.time()
        # 10번의 시뮬레이션 스텝 실행
        for _ in range(10):
            self.world.run_simulation_step()
        end_time = time.time()

        duration = end_time - start_time
        print(f"CPU (NumPy) A.I. 사고 시뮬레이션 시간: {duration:.4f} 초")
        # 이 테스트는 성능을 측정하는 것이므로, 특정 결과에 대한 assert는 필요 없습니다.
        self.assertTrue(True)

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy is not available, skipping GPU test.")
    @unittest.skipUnless(
        RUN_CUDA_BENCH,
        "Heavy CPU/CUDA performance tests are disabled by default. "
        "Set RUN_CUDA_BENCH=1 to enable."
    )
    def test_gpu_performance_prototype(self):
        """CuPy를 사용한 GPU 시뮬레이션 성능 프로토타입을 측정합니다."""
        print("\n--- GPU (CuPy) Performance Test ---")

        # 1. 데이터를 CPU에서 GPU로 전송
        energy_gpu = cp.asarray(self.world.energy)
        # lil_matrix를 CSR로 변환한 후 GPU로 전송
        adj_matrix_csr_cpu = self.world.adjacency_matrix.tocsr()
        adj_matrix_gpu = cp_csr_matrix((cp.asarray(adj_matrix_csr_cpu.data),
                                        cp.asarray(adj_matrix_csr_cpu.indices),
                                        cp.asarray(adj_matrix_csr_cpu.indptr)),
                                       shape=adj_matrix_csr_cpu.shape)

        # GPU 작업이 모두 준비될 때까지 기다림
        cp.cuda.Stream.null.synchronize()

        start_time = time.time()

        # 10번의 시뮬레이션 스텝 실행 (핵심 로직만)
        for _ in range(10):
            # run_simulation_step의 핵심 에너지 전파 로직을 CuPy로 구현
            transfer_rate = 0.1
            energy_out_matrix = adj_matrix_gpu.multiply(energy_gpu[:, cp.newaxis]) * transfer_rate

            total_energy_out = cp.array(energy_out_matrix.sum(axis=1)).flatten()
            total_energy_in = cp.array(energy_out_matrix.sum(axis=0)).flatten()

            # Law of Love는 복잡성으로 인해 이 프로토타입에서는 제외
            energy_boost = cp.zeros_like(energy_gpu)

            energy_deltas = total_energy_in - total_energy_out + energy_boost
            energy_gpu += energy_deltas

        # 모든 GPU 계산이 끝날 때까지 동기화
        cp.cuda.Stream.null.synchronize()
        end_time = time.time()

        duration = end_time - start_time
        print(f"GPU (CuPy) A.I. 사고 시뮬레이션 시간: {duration:.4f} 초")
        # 이 테스트는 성능을 측정하는 것이므로, 특정 결과에 대한 assert는 필요 없습니다.
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()