import numpy as np
from typing import List, Dict, Any, Tuple
from core.physics.causal_field import InformationVoxel

class PredictiveProcessingEngine:
    """
    [Predictive Processing & Coarse-Graining Engine]
    Simulates Top-Down prediction vs Bottom-Up sensory inputs.
    Computes "Prediction Error" (Phase misalignment).

    Implements a "Sliding Scale Lens" (Dynamic Threshold):
    - Low Prediction Error -> Zoom-out / Coarse-Graining (integrates sameness, suppresses noise)
    - High Prediction Error -> Zoom-in / Fine-Graining (details difference, adapts cognitive models)
    """
    def __init__(self, dimensions: int = 3, learning_rate: float = 0.15):
        self.dimensions = dimensions
        self.learning_rate = learning_rate

        # Top-Down expected coordinate/state vector
        self.expected_state = np.zeros(dimensions, dtype=np.float32)

        # Sliding threshold: controls coarse-graining resolution
        # Range: [0.01 (extremely fine) to 2.0 (extremely coarse)]
        self.sliding_threshold = 0.5
        self.prediction_error = 0.0

    def compute_prediction_error(self, actual_sensory_vector: np.ndarray) -> float:
        """
        [예측 오차 계측]
        Calculates the Euclidean distance (misalignment) between
        the Top-Down expectation and the Bottom-Up actual sensory input.
        """
        diff = actual_sensory_vector - self.expected_state
        self.prediction_error = float(np.linalg.norm(diff))
        return self.prediction_error

    def adapt_expectation(self, actual_sensory_vector: np.ndarray):
        """
        [탑다운 가설 학습/수정]
        Updates the expectation vector using active inference learning rate.
        Expected state slowly morphs towards sensory truth to minimize future prediction errors.
        """
        self.expected_state = (1.0 - self.learning_rate) * self.expected_state + self.learning_rate * actual_sensory_vector

    def adjust_scale_lens(self) -> float:
        """
        [동적 슬라이딩 임계치 조절]
        Shifts the threshold based on the current prediction error:
        - High Error (> 1.0) -> Decrease threshold -> Zoom-in / Fine-Graining (sensitive to differences)
        - Low Error (< 0.2) -> Increase threshold -> Zoom-out / Coarse-Graining (broadly generalizes sameness)
        """
        # Mapping prediction error to dynamic sliding threshold
        target_threshold = 1.0 / (1.0 + self.prediction_error * 2.0 + 1e-9)
        # Bounded between 0.05 (Fine) and 1.5 (Coarse)
        target_threshold = np.clip(target_threshold, 0.05, 1.5)

        # Smooth adaptation of threshold
        self.sliding_threshold = float(0.8 * self.sliding_threshold + 0.2 * target_threshold)
        return self.sliding_threshold

    def process_coarse_graining(self, voxels: List[InformationVoxel]) -> List[List[InformationVoxel]]:
        """
        [조상화 그룹핑 프로토콜]
        Groups voxels into "Sameness Clusters" based on the dynamic sliding threshold.
        If distance between voxels' tensors is less than self.sliding_threshold,
        they are coarse-grained (grouped) together, effectively suppressing minor differences as noise.
        """
        if not voxels:
            return []

        clusters: List[List[InformationVoxel]] = []
        for vox in voxels:
            added = False
            for cluster in clusters:
                # Compare voxel with cluster representative (first element)
                rep = cluster[0]
                dist = np.linalg.norm(vox.tensor[:self.dimensions] - rep.tensor[:self.dimensions])

                # If distance falls within the sliding threshold, treat them as "same"
                if dist < self.sliding_threshold:
                    cluster.append(vox)
                    added = True
                    break

            if not added:
                clusters.append([vox])

        return clusters
