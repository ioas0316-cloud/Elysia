import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from typing import List, Tuple

font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False

class OpticsQuaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def multiply(self, other):
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return OpticsQuaternion(w, x, y, z)

    def normalize(self):
        norm = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm > 0:
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm
        return self

    def conjugate(self):
        return OpticsQuaternion(self.w, -self.x, -self.y, -self.z)

    def rotate_point(self, point):
        p_quat = OpticsQuaternion(0, point[0], point[1], point[2])
        q_conj = self.conjugate()
        rotated = self.multiply(p_quat).multiply(q_conj)
        return np.array([rotated.x, rotated.y, rotated.z])

class HunminjeongeumEncoder:
    CHOSEONG_BASE = 0x1100
    JUNGSEONG_BASE = 0x1161
    JONGSEONG_BASE = 0x11A7
    BASE_CODE = 0xAC00
    CHOSEONG_COUNT = 19
    JUNGSEONG_COUNT = 21
    JONGSEONG_COUNT = 28

    @classmethod
    def decompose(cls, char):
        if not ('가' <= char <= '힣'):
            return (0, 0, 0)
        char_code = ord(char) - cls.BASE_CODE
        choseong_idx = char_code // (cls.JUNGSEONG_COUNT * cls.JONGSEONG_COUNT)
        jungseong_idx = (char_code % (cls.JUNGSEONG_COUNT * cls.JONGSEONG_COUNT)) // cls.JONGSEONG_COUNT
        jongseong_idx = char_code % cls.JONGSEONG_COUNT
        return (choseong_idx + 1, jungseong_idx + 1, jongseong_idx)

    @classmethod
    def text_to_flux(cls, text):
        flux = []
        for char in text:
            cho, jung, jong = cls.decompose(char)
            f_A = cho * 0.1
            f_B = jung * 0.1
            f_C = jong * 0.1 if jong > 0 else 0.05
            flux.append((f_A, f_B, f_C))
        return flux

class SensoryStreamReceptor:
    """Converts multimodal inputs into 3-phase Spatiotemporal Flux."""
    @staticmethod
    def process_multimodal(data_dict):
        # A simple metaphor:
        # Phase A (Audio), Phase B (Video/Spatial), Phase C (Text/Math)
        flux_stream = []

        audio = data_dict.get('audio', [])
        video = data_dict.get('video', [])
        text = data_dict.get('text', [])

        # Determine max length to synchronize streams
        max_len = max(len(audio), len(video), len(text))

        for i in range(max_len):
            # If a modality is missing data at this timestep, it falls to a baseline noise level
            f_A = audio[i] if i < len(audio) else 0.01
            f_B = video[i] if i < len(video) else 0.01

            f_C = 0.01
            if i < len(text) and isinstance(text, str):
                cho, jung, jong = HunminjeongeumEncoder.decompose(text[i])
                f_C = (cho + jung + jong) * 0.05

            flux_stream.append((f_A, f_B, f_C))

        return flux_stream

class ElysiaCognitiveSpine:
    def __init__(self, coherence_threshold=0.5, trajectory_tolerance=0.1):
        self.coherence_threshold = coherence_threshold
        self.trajectory_tolerance = trajectory_tolerance
        self.core_memory = [] # Engrams
        self.narrative_trajectories = [] # Locked-in spatiotemporal narratives

    def calculate_imbalance(self, f_A, f_B, f_C, t):
        theta_A = 2 * math.pi * f_A * t
        theta_B = 2 * math.pi * f_B * t + (2 * math.pi / 3)
        theta_C = 2 * math.pi * f_C * t + (4 * math.pi / 3)

        q_A = OpticsQuaternion(math.cos(theta_A/2), 0, 0, math.sin(theta_A/2)).normalize()
        q_B = OpticsQuaternion(math.cos(theta_B/2), 0, 0, math.sin(theta_B/2)).normalize()
        q_C = OpticsQuaternion(math.cos(theta_C/2), 0, 0, math.sin(theta_C/2)).normalize()

        vec_A = q_A.rotate_point([1, 0, 0])
        vec_B = q_B.rotate_point([1, 0, 0])
        vec_C = q_C.rotate_point([1, 0, 0])

        neutral_vector = vec_A + vec_B + vec_C
        imbalance = np.linalg.norm(neutral_vector)
        return imbalance, neutral_vector

    def process_multimodal_event(self, event_name, data_dict):
        print(f"\n--- Processing Multimodal Event: '{event_name}' ---")
        flux_stream = SensoryStreamReceptor.process_multimodal(data_dict)

        trajectory = []
        t = 0.0
        dt = 0.1

        for idx, (f_A, f_B, f_C) in enumerate(flux_stream):
            imbalance, neutral_vec = self.calculate_imbalance(f_A, f_B, f_C, t)

            # Store the trajectory points regardless, to capture the narrative flow
            trajectory.append((t, neutral_vec, imbalance))
            t += dt

        # Check if this narrative trajectory resonates with existing memories
        resonance_found = False
        for stored_name, stored_traj in self.narrative_trajectories:
            if self.compare_trajectories(trajectory, stored_traj):
                print(f"🔄 Phase Resonance Lock-in! '{event_name}' resonates with existing memory '{stored_name}'.")
                resonance_found = True
                break

        if not resonance_found:
            print(f"✨ New Spatiotemporal Narrative Locked In: '{event_name}'")
            self.narrative_trajectories.append((event_name, trajectory))

        return trajectory

    def compare_trajectories(self, traj1, traj2):
        # Compare the geometric signature (neutral vectors) over time
        if len(traj1) != len(traj2):
            return False # For simplicity in this simulation, lengths must match

        total_diff = 0
        for (_, vec1, _), (_, vec2, _) in zip(traj1, traj2):
            diff = np.linalg.norm(vec1 - vec2)
            total_diff += diff

        avg_diff = total_diff / len(traj1)
        return avg_diff <= self.trajectory_tolerance

    def visualize_trajectories(self, filename='cognitive_spine_multimodal_trajectory.png'):
        if not self.narrative_trajectories:
            print("No trajectories to visualize.")
            return

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        colors = ['b', 'g', 'r', 'c', 'm', 'y']

        for idx, (name, traj) in enumerate(self.narrative_trajectories):
            xs = [vec[0] for _, vec, _ in traj]
            ys = [vec[1] for _, vec, _ in traj]
            zs = [vec[2] for _, vec, _ in traj]

            color = colors[idx % len(colors)]
            ax.plot(xs, ys, zs, label=name, color=color, linewidth=2, marker='o')

        ax.set_title("Elysia Spatiotemporal Narrative Matrix (Y-Neutral Trajectories)")
        ax.set_xlabel("X (Phase Projection)")
        ax.set_ylabel("Y (Phase Projection)")
        ax.set_zlabel("Z (Phase Projection)")
        ax.legend()
        plt.savefig(filename)
        print(f"Trajectory visualization saved to '{filename}'")

if __name__ == "__main__":
    spine = ElysiaCognitiveSpine(trajectory_tolerance=0.5)

    # Event 1: Drawing the number '3' (Audio of scraping, Video trajectory, Intent)
    event_3_original = {
        'audio': [0.1, 0.2, 0.3, 0.2, 0.1],
        'video': [0.5, 0.8, 0.2, 0.8, 0.5],
        'text': "숫자삼"
    }
    spine.process_multimodal_event("Drawing Number '3'", event_3_original)

    # Event 2: A different, random event
    event_random = {
        'audio': [0.9, 0.1, 0.9, 0.1, 0.9],
        'video': [0.1, 0.9, 0.1, 0.9, 0.1],
        'text': "랜덤임"
    }
    spine.process_multimodal_event("Random Noise", event_random)

    # Event 3: Drawing the number '3' again, slightly different but narratively identical
    event_3_new = {
        'audio': [0.12, 0.18, 0.31, 0.19, 0.11], # Slight variations
        'video': [0.48, 0.82, 0.18, 0.79, 0.51],
        'text': "숫자삼"
    }
    spine.process_multimodal_event("Drawing Number '3' (New Attempt)", event_3_new)

    spine.visualize_trajectories()
