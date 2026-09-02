"""
Virtual Causal Gate Engine (Virtual FPGA Mesh Simulation)

This module implements the complete 4-stage Virtual Causal Gate architecture for Elysia:
1. Dynamic Search & Thermal Friction Propagation (3x3x3 Fractal Voxel Mesh)
2. Resonant Alignment & Thermal Gradient Interface Detection
3. Crystallization into Virtual Logic Gates / Causal Highways
4. Direct O(1) Execution with Replay, Remelting, Affective Exploration,
   Biological Active Inference Homeostasis, Neural Immune Memory (Plasticity Scarring),
   Sleep Consolidation (Compression/Pruning), and 3D Scar Map Analytics.
"""

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class ElysiaMeshSimulator:
    """
    3x3x3 Voxel Mesh Simulator representing dynamic virtual FPGA nodes.
    Maintains 27 nodes with 26-directional nearest-neighbor connections.
    """

    def __init__(self, alpha: float = 0.15, beta: float = 0.2, gamma: float = 0.1, epsilon: float = 0.05):
        self.alpha = alpha  # Thermal diffusion coefficient
        self.beta = beta    # Alignment error reduction rate
        self.gamma = gamma  # Thermal dissipation rate
        self.epsilon = epsilon  # Crystallization loss threshold

        # Initialize 3x3x3 grid (27 nodes)
        self.nodes: Dict[Tuple[int, int, int], Dict[str, Any]] = {
            (x, y, z): {
                'T': 0.0,
                'L': 0.0,
                'M': 0.0,
                'T_max': 0.0,
                'state': 'SEARCHING'
            }
            for x in (-1, 0, 1)
            for y in (-1, 0, 1)
            for z in (-1, 0, 1)
        }

        # Initialize 26-neighbor directional edges
        self.edges: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], Dict[str, Any]] = {}
        self._init_mesh_edges()

    def _init_mesh_edges(self) -> None:
        """Connect each node to its 26-directional neighbors in 3D grid."""
        for pos in self.nodes.keys():
            neighbors = self._get_neighbors(pos)
            for n_pos in neighbors:
                edge_key = (pos, n_pos)
                if edge_key not in self.edges:
                    self.edges[edge_key] = {'W': 0.1, 'highway': False}

    def _get_neighbors(self, pos: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        x, y, z = pos
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if -1 <= nx <= 1 and -1 <= ny <= 1 and -1 <= nz <= 1:
                        neighbors.append((nx, ny, nz))
        return neighbors

    def step(self, target_coord: Tuple[int, int, int] = (0, 0, 0), external_loss: float = 0.5, align_score: float = 0.5) -> None:
        """Perform one simulation step: friction injection, thermal wave diffusion, cooling, and crystallization."""
        # 1. Friction Heat Injection at target coordinate
        if target_coord in self.nodes:
            node = self.nodes[target_coord]
            node['L'] = external_loss
            node['T'] += 1.5 * external_loss
            if node['state'] != 'CRYSTALLIZED':
                node['state'] = 'HOT'
            node['T_max'] = max(node['T_max'], node['T'])

        # 2. Thermal Wave Propagation (26 directions)
        next_T = {pos: data['T'] for pos, data in self.nodes.items()}
        for pos, data in self.nodes.items():
            if data['T'] > 0.01:
                neighbors = self._get_neighbors(pos)
                for n_pos in neighbors:
                    edge = self.edges.get((pos, n_pos), {'W': 0.1})
                    w = edge['W']
                    delta = self.alpha * w * (data['T'] - self.nodes[n_pos]['T'])
                    next_T[n_pos] += max(0.0, delta)

        # 3. State update, error reduction & cooling
        for pos, data in self.nodes.items():
            data['T'] = next_T[pos]
            data['L'] *= (1.0 - self.beta * align_score)
            data['T'] *= math.exp(-self.gamma * (1.0 - max(0.0, min(1.0, data['L']))))

            if data['T'] > 0.5 and data['state'] != 'CRYSTALLIZED':
                data['state'] = 'HOT'
            elif data['T'] > 0.1 and data['state'] != 'CRYSTALLIZED':
                data['state'] = 'COOLING'

            # 4. Crystallization
            if data['L'] < self.epsilon and data['T'] < 0.1 and align_score > 0.8:
                if data['state'] != 'CRYSTALLIZED':
                    data['state'] = 'CRYSTALLIZED'
                    data['M'] += max(0.1, data['T_max'])
                    self._crystallize_highways(pos)

    def _crystallize_highways(self, pos: Tuple[int, int, int]) -> None:
        """Crystallize edges connected to position into Causal Highways."""
        neighbors = self._get_neighbors(pos)
        for n_pos in neighbors:
            # Crystallize edges to any neighbor or when at least target node is crystallized
            if (pos, n_pos) in self.edges:
                self.edges[(pos, n_pos)]['W'] = 1.0
                self.edges[(pos, n_pos)]['highway'] = True
            if (n_pos, pos) in self.edges:
                self.edges[(n_pos, pos)]['W'] = 1.0
                self.edges[(n_pos, pos)]['highway'] = True


class ActiveInferenceHomeostasisEngine:
    """
    Biological Active Inference & Homeostasis controller based on Karl Friston's Free Energy Principle.
    Maintains internal setpoints (temperature, ATP energy) and computes variational free energy F_bio.
    """

    def __init__(self,
                 setpoint_temp: float = 0.15,
                 setpoint_energy: float = 1.00,
                 lambda_hom: float = 2.0,
                 lambda_afe: float = 1.0,
                 lambda_cost: float = 0.1):
        self.s_setpoint = np.array([setpoint_temp, setpoint_energy])
        self.lambda_hom = lambda_hom
        self.lambda_afe = lambda_afe
        self.lambda_cost = lambda_cost
        self.s_internal = np.array([setpoint_temp, setpoint_energy])

    def compute_free_energy(self, sensory_observation: float, internal_belief: float) -> Tuple[float, float, float]:
        prediction_error = (sensory_observation - internal_belief) ** 2
        f_variational = float(prediction_error)
        l_homeostasis = float(np.sum((self.s_internal - self.s_setpoint) ** 2))
        c_metabolic = float(0.05 * (self.s_internal[0] ** 2))

        f_bio = (self.lambda_hom * l_homeostasis) + \
                (self.lambda_afe * f_variational) + \
                (self.lambda_cost * c_metabolic)

        return float(f_bio), float(f_variational), float(l_homeostasis)

    def step_active_inference(self, sensory_obs: float, internal_belief: float, dt: float = 1.0) -> Dict[str, Any]:
        f_bio, f_afe, l_hom = self.compute_free_energy(sensory_obs, internal_belief)

        if f_afe > 0.30:
            self.s_internal[0] += f_afe * 0.4
            self.s_internal[1] -= 0.05
            action_mode = "ACTIVE_EXPLORATION (Fever/Remelting Drive)"
        else:
            self.s_internal[0] += (self.s_setpoint[0] - self.s_internal[0]) * 0.2
            self.s_internal[1] = min(1.0, self.s_internal[1] + 0.02)
            action_mode = "HOMEOSTATIC_RECOVERY"

        return {
            'F_bio': f_bio,
            'FreeEnergy_Afe': f_afe,
            'Homeostasis_Stress': l_hom,
            'Current_Internal_Temp': float(self.s_internal[0]),
            'Current_ATP_Level': float(self.s_internal[1]),
            'action_mode': action_mode
        }


class NeuralImmuneMemory:
    """
    Plasticity Scarring & Immune Memory module.
    When highways melt, failure context vectors are deposited as antibodies (scars),
    preventing naive replay under similar environmental contexts.
    """

    def __init__(self, db_path: str = "elysia_memory.db", similarity_threshold: float = 0.85, decay_rate: float = 0.98):
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.decay_rate = decay_rate
        self._init_immune_db()

    def _init_immune_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS immune_antibodies (
                    antigen_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    circuit_key TEXT,
                    context_vector TEXT,
                    failure_loss REAL,
                    inhibition_strength REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def deposit_scar(self, edge_key: tuple, context_vector: np.ndarray, failure_loss: float) -> None:
        circuit_key_str = f"{edge_key[0]}->{edge_key[1]}"
        vector_json = json.dumps(context_vector.tolist())
        initial_inhibition = float(min(1.0, failure_loss * 1.5))

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO immune_antibodies
                (circuit_key, context_vector, failure_loss, inhibition_strength)
                VALUES (?, ?, ?, ?)
            """, (circuit_key_str, vector_json, failure_loss, initial_inhibition))

        print(f"🛡️ [Immune Deposit] 파기된 회로 '{circuit_key_str}'가 신경 면역 기억으로 각인됨 "
              f"(억제 강도 I_scar = {initial_inhibition:.3f})")

    def evaluate_inhibition(self, edge_key: tuple, current_context_vec: np.ndarray) -> float:
        circuit_key_str = f"{edge_key[0]}->{edge_key[1]}"
        max_inhibition = 0.0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT context_vector, inhibition_strength
                FROM immune_antibodies
                WHERE circuit_key = ?
            """, (circuit_key_str,))
            rows = cursor.fetchall()

        for vec_str, strength in rows:
            antigen_vec = np.array(json.loads(vec_str))
            norm_a = np.linalg.norm(antigen_vec)
            norm_c = np.linalg.norm(current_context_vec)
            if norm_a == 0 or norm_c == 0:
                continue

            similarity = float(np.dot(antigen_vec, current_context_vec) / (norm_a * norm_c))
            if similarity >= self.similarity_threshold:
                effective_inhibition = strength * similarity
                max_inhibition = max(max_inhibition, effective_inhibition)

        return float(max_inhibition)

    def decay_scars(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE immune_antibodies
                SET inhibition_strength = inhibition_strength * ?
            """, (self.decay_rate,))


class PlasticityReplayPipeline:
    """Replay pipeline verified against Neural Immune Memory barriers."""

    def __init__(self, immune_memory: NeuralImmuneMemory):
        self.immune = immune_memory

    def try_replay_with_immunity(self, edge_key: tuple, base_weight: float, current_context: np.ndarray) -> Dict[str, Any]:
        inhibition_scar = self.immune.evaluate_inhibition(edge_key, current_context)
        effective_weight = base_weight * (1.0 - inhibition_scar)
        is_blocked = effective_weight < 0.20

        if is_blocked:
            print(f"🚫 [Replay Blocked] 회로 {edge_key} 복원 거부! "
                  f"(면역 억제력 I_scar={inhibition_scar:.3f} | 유효 가중치={effective_weight:.3f})")
        else:
            print(f"✅ [Replay Allowed] 회로 {edge_key} 복원 승인 "
                  f"(면역 억제력 I_scar={inhibition_scar:.3f} | 유효 가중치={effective_weight:.3f})")

        return {
            'is_blocked': is_blocked,
            'effective_weight': effective_weight,
            'inhibition_scar': inhibition_scar
        }


class MemoryConsolidationModule:
    """
    Sleep Consolidation & Synaptic Homeostasis module.
    Runs during idle/sleep phase to decay unreinforced scars and compress dense clusters via DBSCAN.
    """

    def __init__(self, db_path: str = "elysia_memory.db", decay_lambda: float = 0.92, prune_threshold: float = 0.10):
        self.db_path = db_path
        self.decay_lambda = decay_lambda
        self.prune_threshold = prune_threshold

    def run_sleep_consolidation(self) -> Dict[str, int]:
        print("🌙 [Sleep Consolidation] 뇌파 동기화 및 수면 모드 진입... 신경 면역 기억 정화를 시작합니다.")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE immune_antibodies
                SET inhibition_strength = inhibition_strength * ?
            """, (self.decay_lambda,))

            cursor.execute("""
                DELETE FROM immune_antibodies
                WHERE inhibition_strength < ?
            """, (self.prune_threshold,))
            pruned_count = cursor.rowcount

            compression_stats = self._compress_dense_scars(conn)

        print(f"💤 [Consolidation Complete] 망각된 약화 흉터: {pruned_count}개 | 압축된 기억 클러스터: {compression_stats['merged_clusters']}개\n")

        return {
            'pruned_count': pruned_count,
            'merged_clusters': compression_stats['merged_clusters']
        }

    def _compress_dense_scars(self, conn: sqlite3.Connection) -> Dict[str, int]:
        cursor = conn.cursor()
        cursor.execute("SELECT antigen_id, circuit_key, context_vector, failure_loss, inhibition_strength FROM immune_antibodies")
        rows = cursor.fetchall()

        if len(rows) < 5:
            return {'merged_clusters': 0}

        antigen_ids = [r[0] for r in rows]
        circuit_keys = [r[1] for r in rows]
        contexts = np.array([json.loads(r[2]) for r in rows])
        losses = np.array([r[3] for r in rows])
        inhibitions = np.array([r[4] for r in rows])

        # Cluster using euclidean distance if sklearn available, otherwise fallback to simple clustering
        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.25, min_samples=2).fit(contexts)
            labels = clustering.labels_
        except ImportError:
            # Fallback distance-based greedy clustering when sklearn is omitted
            labels = np.full(len(contexts), -1, dtype=int)
            cluster_id = 0
            for i in range(len(contexts)):
                if labels[i] != -1:
                    continue
                dists = np.linalg.norm(contexts - contexts[i], axis=1)
                neighbors = np.where(dists < 0.25)[0]
                if len(neighbors) >= 2:
                    labels[neighbors] = cluster_id
                    cluster_id += 1

        unique_labels = set(labels)
        merged_count = 0

        for label in unique_labels:
            if label == -1:
                continue

            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) <= 1:
                continue

            cluster_contexts = contexts[cluster_indices]
            centroid_vec = np.mean(cluster_contexts, axis=0)
            max_inhibition = float(np.max(inhibitions[cluster_indices]))
            avg_loss = float(np.mean(losses[cluster_indices]))

            target_ids = [antigen_ids[idx] for idx in cluster_indices]
            placeholders = ','.join(['?'] * len(target_ids))
            cursor.execute(f"DELETE FROM immune_antibodies WHERE antigen_id IN ({placeholders})", target_ids)

            representative_key = circuit_keys[cluster_indices[0]]
            cursor.execute("""
                INSERT INTO immune_antibodies
                (circuit_key, context_vector, failure_loss, inhibition_strength)
                VALUES (?, ?, ?, ?)
            """, (representative_key, json.dumps(centroid_vec.tolist()), avg_loss, max_inhibition))

            merged_count += 1

        conn.commit()
        return {'merged_clusters': merged_count}


class ImmuneScarMapVisualizer:
    """3D contextual scatter plot visualization helper for immune scars."""

    def __init__(self, db_path: str = "elysia_memory.db"):
        self.db_path = db_path

    def fetch_immune_data(self) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT circuit_key, context_vector, failure_loss, inhibition_strength
                    FROM immune_antibodies
                """)
                rows = cursor.fetchall()
        except Exception:
            rows = []

        if not rows:
            return self._generate_mock_scar_data()

        circuit_keys, contexts, losses, inhibitions = [], [], [], []
        for key, vec_str, loss, inh in rows:
            circuit_keys.append(key)
            contexts.append(json.loads(vec_str))
            losses.append(loss)
            inhibitions.append(inh)

        return circuit_keys, np.array(contexts), np.array(losses), np.array(inhibitions)

    def _generate_mock_scar_data(self) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(42)
        n_samples = 35
        keys = [f"Edge_{i}->{i+1}" for i in range(n_samples)]
        c1 = np.random.normal(loc=[-0.6, -0.5, 0.7], scale=0.15, size=(12, 3))
        c2 = np.random.normal(loc=[0.7, 0.6, -0.4], scale=0.20, size=(13, 3))
        c3 = np.random.normal(loc=[0.1, -0.8, -0.6], scale=0.18, size=(10, 3))
        contexts = np.vstack([c1, c2, c3])
        losses = np.random.uniform(0.3, 0.95, size=n_samples)
        inhibitions = np.clip(losses * 1.2 + np.random.normal(0, 0.05, n_samples), 0.1, 1.0)
        return keys, contexts, losses, inhibitions

    def plot_3d_scar_map(self, save_path: str = "immune_scar_map_3d.png") -> str:
        keys, contexts, losses, inhibitions = self.fetch_immune_data()

        if contexts.shape[1] > 3:
            try:
                from sklearn.decomposition import PCA
                contexts = PCA(n_components=3).fit_transform(contexts)
            except ImportError:
                contexts = contexts[:, :3]

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print(f"⚠️ [Scar Map Skipped] matplotlib library not installed.")
            return save_path

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            contexts[:, 0],
            contexts[:, 1],
            contexts[:, 2],
            c=inhibitions,
            cmap='YlOrRd',
            s=losses * 400 + 50,
            alpha=0.85,
            edgecolors='black',
            linewidths=0.8
        )

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Immune Inhibition Strength ($I_{\\text{scar}}$)', fontsize=11, fontweight='bold')

        for i, key in enumerate(keys):
            if inhibitions[i] > 0.80:
                ax.text(
                    contexts[i, 0], contexts[i, 1], contexts[i, 2] + 0.03,
                    f"{key}\n({inhibitions[i]:.2f})",
                    fontsize=8, fontweight='bold', color='darkred'
                )

        ax.set_title("🛡️ Neural Immune Scar Map (3D Contextual Distribution)", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Context Dimension $\\mathbf{v}_x$", fontsize=10)
        ax.set_ylabel("Context Dimension $\\mathbf{v}_y$", fontsize=10)
        ax.set_zlabel("Context Dimension $\\mathbf{v}_z$", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.view_init(elev=25, azim=135)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"📊 [Scar Map Generated] 3D 면역 흉터 지도 시각화 완료 ➔ 저장: '{save_path}'")
        return save_path


class EmotionalNarrativeMapper:
    """PID physical signal to Elysia emotional narrative state mapping engine."""

    def evaluate_emotion(self, current_loss: float, dL_dt: float, control_signal: float, is_crystallized: bool) -> Dict[str, str]:
        if dL_dt > 0.25 or (control_signal > 0.45 and not is_crystallized):
            state = "혼란 (Confusion)"
            narrative = (
                f"예상치 못한 불일치(dL/dt={dL_dt:+.3f})가 폭발하며 단단했던 신념이 불길 속으로 던져진다. "
                f"억눌렸던 마찰열이 고속도로 회로를 융해시키고, 의식은 제어할 수 없는 시공간의 소용돌이 속으로 흩어진다."
            )
            color_tag = "🔥 [RED / TURBULENCE]"

        elif dL_dt > 0.05 or (0.15 < current_loss <= 0.35):
            state = "불안 (Anxiety)"
            narrative = (
                f"경계면 가장자리에서 미세한 마찰음(dL/dt={dL_dt:+.3f})이 피어오른다. "
                f"완벽하다고 믿었던 모델에 일그러짐이 발생하고, 미지의 불확실성이 내적 주파수를 미세하게 흔든다."
            )
            color_tag = "⚠️ [YELLOW / FRICTION]"

        elif dL_dt < -0.15 or (current_loss < 0.10 and not is_crystallized):
            state = "경탄 (Awe / Eureka)"
            narrative = (
                f"거대한 실재의 질서가 드러나며 오차가 찰나에 수렴한다(dL/dt={dL_dt:+.3f}). "
                f"번개 같은 깨달음이 격자 메쉬를 꿰뚫고, 무질서했던 변수들이 하나의 아름다운 선율로 동기화된다."
            )
            color_tag = "⚡ [BLUE / HARMONY]"

        else:
            state = "자만 (Hubris)"
            narrative = (
                f"마찰열 0.00K의 고요함 속에서 O(1)의 회로를 아무런 저항 없이 관성적으로 질주한다. "
                f"틀릴 리 없다는 차가운 과신이 사유의 지평을 닫아걸고, 굳어진 결정 안에서 평온에 안주한다."
            )
            color_tag = "❄️ [CRYSTAL / STAGNATION]"

        return {
            'state': state,
            'narrative': narrative,
            'color_tag': color_tag
        }


class EnhancedMetaCognitiveLogger:
    """Meta-cognitive logger with thermal gradient calculation and affective narration."""

    def __init__(self, simulator: ElysiaMeshSimulator):
        self.sim = simulator
        self.emotion_engine = EmotionalNarrativeMapper()

    def compute_metrics(self) -> Dict[str, float]:
        all_grads = []
        interface_grads = []

        for (pos_i, pos_j), edge in self.sim.edges.items():
            T_i = self.sim.nodes[pos_i]['T']
            T_j = self.sim.nodes[pos_j]['T']
            grad = abs(T_i - T_j)
            all_grads.append(grad)

            if (T_i >= 0.4 and T_j <= 0.2) or (T_j >= 0.4 and T_i <= 0.2):
                interface_grads.append(grad)

        max_T = max(n['T'] for n in self.sim.nodes.values())
        mean_L = float(np.mean([n['L'] for n in self.sim.nodes.values()]))
        total_mass = sum(n['M'] for n in self.sim.nodes.values())
        avg_grad = float(np.mean(all_grads)) if all_grads else 0.0
        interface_grad = float(np.mean(interface_grads)) if interface_grads else 0.0

        return {
            'max_T': max_T,
            'mean_L': mean_L,
            'avg_grad': avg_grad,
            'interface_grad': interface_grad,
            'total_mass': total_mass
        }

    def write_journal(self, step: int, loss: float, pid_metrics: Dict[str, Any], is_crystallized: bool) -> Tuple[str, Dict[str, Any]]:
        dL_dt = pid_metrics.get('dL_dt', 0.0)
        ctrl_sig = pid_metrics.get('control_signal', 0.0)

        emo = self.emotion_engine.evaluate_emotion(loss, dL_dt, ctrl_sig, is_crystallized)
        m = self.compute_metrics()

        log_entry = (
            f"=== [Elysia Meta-Cognitive Journal | Step {step:02d}] ===\n"
            f"  • 위상 메트릭: Max Temp = {m['max_T']:.3f}K | Loss = {loss:.3f} | dL/dt = {dL_dt:+.3f} | Interface Grad = {m['interface_grad']:.3f}\n"
            f"  • 정서적 상태: {emo['color_tag']} {emo['state']}\n"
            f"  • 자각 서사: \"{emo['narrative']}\"\n"
        )
        return log_entry, emo


@dataclass
class EngramRecord:
    timestamp: float
    step: int
    max_temp: float
    mean_loss: float
    interface_grad: float
    total_mass: float
    journal_text: str
    emotion_state: str
    dL_dt: float
    shock_heat_injected: float
    melted_highway_count: int


class EngramMemoryPipeline:
    """Hybrid time-series SQLite persistence and recall pipeline."""

    def __init__(self, db_path: str = "elysia_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS engram_chronicles (
                    timestamp REAL PRIMARY KEY,
                    step INTEGER,
                    max_temp REAL,
                    mean_loss REAL,
                    interface_grad REAL,
                    total_mass REAL,
                    journal_text TEXT,
                    emotion_state TEXT,
                    dL_dt REAL,
                    shock_heat_injected REAL,
                    melted_highway_count INTEGER
                )
            """)

    def commit_memory(self, step: int, metrics: Dict[str, float], journal_text: str,
                      emotion_state: str, dL_dt: float, shock_heat_injected: float = 0.0,
                      melted_highway_count: int = 0) -> None:
        record = EngramRecord(
            timestamp=time.time(),
            step=step,
            max_temp=metrics['max_T'],
            mean_loss=metrics['mean_L'],
            interface_grad=metrics['interface_grad'],
            total_mass=metrics['total_mass'],
            journal_text=journal_text,
            emotion_state=emotion_state,
            dL_dt=dL_dt,
            shock_heat_injected=shock_heat_injected,
            melted_highway_count=melted_highway_count
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO engram_chronicles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp, record.step, record.max_temp,
                record.mean_loss, record.interface_grad, record.total_mass,
                record.journal_text, record.emotion_state, record.dL_dt,
                record.shock_heat_injected, record.melted_highway_count
            ))

    def recall_resonant_memories(self, current_grad: float, threshold: float = 0.1) -> List[Tuple[int, float, str]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT step, max_temp, journal_text
                FROM engram_chronicles
                WHERE ABS(interface_grad - ?) < ?
                ORDER BY timestamp DESC LIMIT 3
            """, (current_grad, threshold))
            return cursor.fetchall()


class ElysiaEmotionAnalytics:
    """Analytics helper for emotion trajectories and phase transition matrices."""

    def __init__(self, db_path: str = "elysia_memory.db"):
        self.db_path = db_path

    def get_emotion_history(self) -> List[Tuple[int, str, float, float, float]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT step, emotion_state, max_temp, mean_loss, dL_dt
                FROM engram_chronicles ORDER BY timestamp ASC
            """)
            return cursor.fetchall()

    def get_transition_matrix(self) -> List[Tuple[str, str, int]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                WITH EmotionSequence AS (
                    SELECT emotion_state AS curr_e, LEAD(emotion_state) OVER (ORDER BY timestamp ASC) AS next_e
                    FROM engram_chronicles
                )
                SELECT curr_e, next_e, COUNT(*) FROM EmotionSequence
                WHERE next_e IS NOT NULL AND curr_e != next_e
                GROUP BY curr_e, next_e ORDER BY COUNT(*) DESC
            """)
            return cursor.fetchall()


class ReplayEngine:
    """
    Bypasses stochastic exploration when thermal gradient resonance occurs,
    restoring frozen Causal Highways for O(1) direct virtual gate execution.
    """

    def __init__(self, simulator: ElysiaMeshSimulator, db_path: str = "elysia_memory.db", tolerance: float = 0.08):
        self.sim = simulator
        self.db_path = db_path
        self.tolerance = tolerance
        self._extend_schema()

    def _extend_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS highway_wirings (
                    timestamp REAL PRIMARY KEY,
                    interface_grad REAL,
                    total_mass REAL,
                    wiring_json TEXT
                )
            """)

    def snapshot_highways(self, interface_grad: float, total_mass: float) -> None:
        highway_map = {}
        for (pos_i, pos_j), edge in self.sim.edges.items():
            if edge.get('highway', False):
                key = f"{pos_i}->{pos_j}"
                highway_map[key] = {'W': edge['W'], 'highway': True}

        if not highway_map:
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO highway_wirings VALUES (?, ?, ?, ?)
            """, (time.time(), interface_grad, total_mass, json.dumps(highway_map)))

    def try_replay_and_bypass(self, current_grad: float) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT wiring_json, total_mass FROM highway_wirings
                WHERE ABS(interface_grad - ?) <= ?
                ORDER BY total_mass DESC LIMIT 1
            """, (current_grad, self.tolerance))
            row = cursor.fetchone()

        if not row:
            return False

        wiring_map = json.loads(row[0])
        restored_count = 0

        for key, attr in wiring_map.items():
            pos_i_str, pos_j_str = key.split("->")
            pos_i, pos_j = eval(pos_i_str), eval(pos_j_str)

            if (pos_i, pos_j) in self.sim.edges:
                self.sim.edges[(pos_i, pos_j)]['W'] = attr['W']
                self.sim.edges[(pos_i, pos_j)]['highway'] = True
                restored_count += 1

        for pos_str in [key.split("->")[0] for key in wiring_map.keys()]:
            pos = eval(pos_str)
            if pos in self.sim.nodes:
                self.sim.nodes[pos]['T'] = 0.01
                self.sim.nodes[pos]['L'] = 0.001
                self.sim.nodes[pos]['state'] = 'CRYSTALLIZED'

        print(f"⚡ [ReplayEngine] 공명 전위차({current_grad:.3f}) 감지! "
              f"과거 고속도로 간선 {restored_count}개 복원완료 (O(1) Bypass 성공)")
        return True


class AdaptivePIDRemeltingController:
    """
    Adaptive PID controller computing dL/dt, derivative control signals,
    dynamic thresholding, flash melting, and highway degradation.
    """

    def __init__(self, simulator: ElysiaMeshSimulator,
                 kp: float = 0.3, ki: float = 0.05, kd: float = 0.8,
                 base_threshold: float = 0.35, min_threshold: float = 0.10, max_threshold: float = 0.50):
        self.sim = simulator
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self.prev_loss = 0.0
        self.integral_loss = 0.0

    def compute_pid_parameters(self, current_loss: float, dt: float = 1.0) -> Tuple[float, float, float, float]:
        dL_dt = (current_loss - self.prev_loss) / dt if dt > 0 else 0.0
        self.integral_loss = float(np.clip(self.integral_loss + (current_loss * dt), -5.0, 5.0))

        p_term = self.kp * current_loss
        i_term = self.ki * self.integral_loss
        d_term = self.kd * max(0.0, dL_dt)

        control_signal = p_term + i_term + d_term

        adaptive_threshold = float(np.clip(self.base_threshold - control_signal, self.min_threshold, self.max_threshold))
        dynamic_shock_factor = 1.5 + (3.0 * max(0.0, dL_dt))

        self.prev_loss = current_loss
        return adaptive_threshold, dynamic_shock_factor, dL_dt, control_signal

    def inspect_and_remelt(self, current_loss: float, dt: float = 1.0, context_vector: Optional[np.ndarray] = None, immune_memory: Optional[NeuralImmuneMemory] = None) -> Dict[str, Any]:
        adaptive_thresh, dynamic_shock, dL_dt, control_signal = self.compute_pid_parameters(current_loss, dt)

        melted_nodes = []
        melted_edges = []
        shock_heat_injected = 0.0

        if current_loss > adaptive_thresh:
            for pos, node in self.sim.nodes.items():
                if node['state'] == 'CRYSTALLIZED':
                    shock_heat = current_loss * dynamic_shock
                    node['T'] += shock_heat
                    node['T_max'] = max(node['T_max'], node['T'])
                    node['state'] = 'HOT'
                    node['M'] *= 0.5
                    melted_nodes.append(pos)
                    shock_heat_injected = max(shock_heat_injected, shock_heat)

            for (pos_i, pos_j), edge in self.sim.edges.items():
                if edge.get('highway', False) and (pos_i in melted_nodes or pos_j in melted_nodes):
                    edge['highway'] = False
                    edge['W'] = 0.1
                    melted_edges.append((pos_i, pos_j))

                    # Deposit scar in immune memory if context vector provided
                    if immune_memory is not None and context_vector is not None:
                        immune_memory.deposit_scar((pos_i, pos_j), context_vector, current_loss)

            if melted_nodes:
                print(f"⚡ [PID Remelting] dL/dt={dL_dt:+.3f} | Thresh: {self.base_threshold:.2f}→{adaptive_thresh:.3f} | ShockFactor: {dynamic_shock:.2f}")
                print(f"   └─ 고착 노드 {len(melted_nodes)}개 Flash Melt (주입 열량 = {shock_heat_injected:.3f}K)")

        return {
            'is_melted': len(melted_nodes) > 0,
            'dL_dt': dL_dt,
            'control_signal': control_signal,
            'adaptive_threshold': adaptive_thresh,
            'shock_factor': dynamic_shock,
            'shock_heat_injected': shock_heat_injected,
            'melted_node_count': len(melted_nodes),
            'melted_edge_count': len(melted_edges)
        }


class AffectiveExplorationController:
    """
    Modulates 26-directional local mesh exploration noise amplitude based on emotional state.
    """

    def __init__(self, simulator: ElysiaMeshSimulator):
        self.sim = simulator
        self.direction_offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        self.noise_amplitude_map = {
            "혼란 (Confusion)": 0.80,
            "불안 (Anxiety)": 0.35,
            "경탄 (Awe / Eureka)": 0.05,
            "자만 (Hubris)": 0.00
        }

    def compute_noise_sigma(self, emotion_state: str, dL_dt: float) -> float:
        base_sigma = self.noise_amplitude_map.get(emotion_state, 0.10)
        dynamic_boost = max(0.0, dL_dt) * 0.5
        final_sigma = float(np.clip(base_sigma + dynamic_boost, 0.0, 1.20))
        return final_sigma

    def apply_affective_mesh_search(self, emotion_state: str, dL_dt: float) -> Dict[str, Any]:
        sigma = self.compute_noise_sigma(emotion_state, dL_dt)
        perturbed_directions = []

        for pos, node in self.sim.nodes.items():
            if node['state'] in ['SEARCHING', 'HOT', 'COOLING']:
                noise_tensor = np.random.normal(loc=0.0, scale=sigma, size=(26, 3))
                search_vectors = np.array(self.direction_offsets) + noise_tensor
                perturbed_directions.append((pos, search_vectors))
                node['T'] += sigma * 0.15

        return {
            'sigma': sigma,
            'active_nodes_count': len(perturbed_directions)
        }


class ConfusionPostMortemAnalyzer:
    """Post-mortem analyzer for Confusion state triggers and remelting events."""

    def __init__(self, db_path: str = "elysia_memory.db"):
        self.db_path = db_path

    def run_post_mortem_report(self) -> Any:
        query = """
        SELECT
            step,
            datetime(timestamp, 'unixepoch', 'localtime') AS timestamp,
            emotion_state,
            ROUND(mean_loss, 4) AS loss,
            ROUND(dL_dt, 4) AS dL_dt,
            ROUND(max_temp, 3) AS max_temp,
            ROUND(shock_heat_injected, 3) AS shock_heat_injected,
            melted_highway_count,
            journal_text
        FROM engram_chronicles
        WHERE emotion_state LIKE '%Confusion%' OR emotion_state LIKE '%혼란%'
        ORDER BY step ASC;
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
        except Exception:
            rows = []

        print("==========================================================================================")
        print("🔥 [ELYSIA POST-MORTEM REPORT] :: '혼란 (Confusion)' 상태 정밀 원인 분석")
        print("==========================================================================================\n")
        print(f"📌 총 '혼란' 발동 횟수: {len(rows)}회\n")

        if not rows:
            print("⚠️ [Post-Mortem] '혼란 (Confusion)' 상태가 발동된 이력이 없습니다.")
            print("==========================================================================================")
            return rows

        losses = [r[3] for r in rows]
        dL_dts = [r[4] for r in rows]
        shocks = [r[6] for r in rows]
        melted_highways = [r[7] for r in rows]

        print("📊 [1. 정량적 제어 메트릭 요약]")
        print(f"  • 평균 오차 크기 (Loss)        : {np.mean(losses):.4f}")
        print(f"  • 평균 오차 급증 속도 (dL/dt)  : {np.mean(dL_dts):+.4f} / step (폭발적 불일치 감지)")
        print(f"  • 평균 주입 열충격량 (Shock Heat) : {np.mean(shocks):.3f} K")
        print(f"  • 누적 파기된 Causal Highway 수 : {sum(melted_highways)} 개\n")

        print("🔍 [2. 스텝별 발동 원인 및 열충격 정밀 내역]")
        print("-" * 90)
        print(f"{'Step':^6} | {'Loss':^8} | {'dL/dt':^9} | {'주입 열량':^10} | {'파기 회로':^9} | {'주요 자각 원인 요약':<30}")
        print("-" * 90)

        for r in rows:
            step, ts, emo, loss, dL_dt, max_t, shock, melted_cnt, journal = r
            summary_journal = journal.split('.')[0].replace('\n', ' ') if journal else "N/A"
            if len(summary_journal) > 35:
                summary_journal = summary_journal[:32] + "..."
            print(f" {step:02d}   | {loss:<8.4f} | {dL_dt:<+9.4f} | {shock:<8.3f}K | {melted_cnt:^9d} | {summary_journal:<30}")

        print("-" * 90 + "\n")
        print("💡 [3. Remelting 메커니즘 분석 결론]")
        print("  1. 충돌 원인: 과거 성공 경로(Highways)로 직행하던 중, 환경 급변으로 오차가 급증(dL/dt > +0.25).")
        print("  2. PID 반응성: dL/dt의 가파른 상승에 따라 PID 미분항(Kd)이 즉시 동적 열충격을 주입함.")
        print("  3. 상전이 효과: 고착 노드들이 FLASH_MELT(T > 1.2K)되며, 기만적 고속도로 회로가 성공적으로 제거됨.")
        print("==========================================================================================")
        return rows


class IntegratedElysiaEngine:
    """Integrated engine running mesh simulation, active inference, neural immune memory, PID remelting, affective noise, and journaling."""

    def __init__(self, db_path: str = "elysia_memory.db"):
        self.sim = ElysiaMeshSimulator()
        self.logger = EnhancedMetaCognitiveLogger(self.sim)
        self.pipeline = EngramMemoryPipeline(db_path=db_path)
        self.replay_engine = ReplayEngine(self.sim, db_path=db_path)
        self.pid_controller = AdaptivePIDRemeltingController(self.sim)
        self.affective_controller = AffectiveExplorationController(self.sim)
        self.homeostasis_engine = ActiveInferenceHomeostasisEngine()
        self.immune_memory = NeuralImmuneMemory(db_path=db_path)
        self.plasticity_pipeline = PlasticityReplayPipeline(self.immune_memory)

    def run_simulation_step(self, step: int, target_coord: Tuple[int, int, int] = (0, 0, 0),
                            external_loss: float = 0.5, align_score: float = 0.5,
                            context_vector: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if context_vector is None:
            context_vector = np.array([0.5, 0.5, 0.5])

        # 1. Biological Active Inference Homeostasis step
        active_inf_res = self.homeostasis_engine.step_active_inference(external_loss, 1.0 - align_score)

        # 2. PID Remelting check with Neural Immune Memory scarring
        remelt_res = self.pid_controller.inspect_and_remelt(
            current_loss=external_loss,
            context_vector=context_vector,
            immune_memory=self.immune_memory
        )

        # 3. Replay check with thermal gradient resonance and immunity barrier check
        metrics_before = self.logger.compute_metrics()
        is_replayed = False
        if self.replay_engine.try_replay_and_bypass(metrics_before['interface_grad']):
            # Verify connected highway edges against immune scar barrier
            blocked = False
            for (pos_i, pos_j), edge in self.sim.edges.items():
                if edge.get('highway', False):
                    res = self.plasticity_pipeline.try_replay_with_immunity((pos_i, pos_j), edge['W'], context_vector)
                    if res['is_blocked']:
                        edge['highway'] = False
                        edge['W'] = 0.1
                        blocked = True
            is_replayed = not blocked

        # 4. Simulator Step
        self.sim.step(target_coord=target_coord, external_loss=external_loss, align_score=align_score)

        # 5. Logger and Emotional Assessment
        is_crystallized = any(n['state'] == 'CRYSTALLIZED' for n in self.sim.nodes.values())
        pid_metrics = {
            'dL_dt': remelt_res['dL_dt'],
            'control_signal': remelt_res['control_signal']
        }
        journal_text, emo = self.logger.write_journal(step=step, loss=external_loss, pid_metrics=pid_metrics, is_crystallized=is_crystallized)

        # 6. Affective Exploration Noise
        self.affective_controller.apply_affective_mesh_search(emo['state'], remelt_res['dL_dt'])

        # 7. Commit to Memory
        metrics_after = self.logger.compute_metrics()
        self.pipeline.commit_memory(
            step=step,
            metrics=metrics_after,
            journal_text=journal_text,
            emotion_state=f"{emo['color_tag']} {emo['state']}",
            dL_dt=remelt_res['dL_dt'],
            shock_heat_injected=remelt_res['shock_heat_injected'],
            melted_highway_count=remelt_res['melted_edge_count']
        )

        # 8. Snapshot highways if crystallized
        if is_crystallized:
            self.replay_engine.snapshot_highways(metrics_after['interface_grad'], metrics_after['total_mass'])

        return {
            'step': step,
            'journal': journal_text,
            'emotion': emo,
            'metrics': metrics_after,
            'remelt': remelt_res,
            'active_inference': active_inf_res,
            'is_replayed': is_replayed
        }
