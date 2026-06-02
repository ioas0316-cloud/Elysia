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

class HunminjeongeumWaveformDiscretizer:
    """
    Revolutionary Text Processing based on Hangeul Phonetic Geometry.
    Treats the vowel (Jungseong) as a continuous Carrier Wave (Base Frequency).
    Treats the consonants (Choseong/Jongseong) as Phase/Amplitude Modulations.
    """
    BASE_CODE = 0xAC00
    CHOSEONG_COUNT = 19
    JUNGSEONG_COUNT = 21
    JONGSEONG_COUNT = 28

    # Base Carrier Frequency Map for vowels (e.g., 'ㅏ' is a strong, stable baseline)
    VOWEL_CARRIER_FREQ = {
        0: 0.50,  # ㅏ (The anchor carrier frequency)
        4: 0.55,  # ㅓ
        8: 0.60,  # ㅗ
        13: 0.65, # ㅜ
        18: 0.70, # ㅡ
        20: 0.75  # ㅣ
    }

    @classmethod
    def decompose(cls, char):
        if not ('가' <= char <= '힣'):
            return (0, 0, 0) # Non-Hangeul returns zero modulation
        char_code = ord(char) - cls.BASE_CODE
        choseong_idx = char_code // (cls.JUNGSEONG_COUNT * cls.JONGSEONG_COUNT)
        jungseong_idx = (char_code % (cls.JUNGSEONG_COUNT * cls.JONGSEONG_COUNT)) // cls.JONGSEONG_COUNT
        jongseong_idx = char_code % cls.JONGSEONG_COUNT
        return (choseong_idx, jungseong_idx, jongseong_idx)

    @classmethod
    def generate_waveform(cls, text):
        """
        Outputs a continuous waveform trajectory instead of isolated tokens.
        """
        waveform = []
        for char in text:
            cho, jung, jong = cls.decompose(char)

            # Extract Carrier Wave Frequency (Vowel)
            # Default to a generic base if complex vowel isn't mapped directly
            carrier_freq = cls.VOWEL_CARRIER_FREQ.get(jung, 0.5 + (jung * 0.01))

            # Consonants act as modulation spikes (Phase Shifts)
            # A consonant modulates the amplitude/phase of the continuous carrier
            choseong_modulation = (cho + 1) * 0.05
            jongseong_modulation = (jong) * 0.05 if jong > 0 else 0.0

            # The character is represented as a mini-wave burst over 3 time steps
            # Step 1: Choseong modulation spike
            waveform.append(carrier_freq + choseong_modulation)
            # Step 2: Pure Carrier Wave (Vowel sustained)
            waveform.append(carrier_freq)
            # Step 3: Jongseong modulation decay (or stable sustain if no jongseong)
            waveform.append(carrier_freq - jongseong_modulation)

        return waveform

class SensoryStreamReceptor:
    """Converts multimodal inputs into 3-phase Spatiotemporal Flux."""
    @staticmethod
    def process_multimodal(data_dict):
        # A simple metaphor:
        # Phase A (Audio), Phase B (Video/Spatial), Phase C (Text/Math)
        flux_stream = []

        audio = data_dict.get('audio', [])
        video = data_dict.get('video', [])
        text = data_dict.get('text', "")

        # Generate text waveform dynamically
        text_waveform = HunminjeongeumWaveformDiscretizer.generate_waveform(text) if isinstance(text, str) else []

        # Determine max length to synchronize streams
        max_len = max(len(audio), len(video), len(text_waveform))

        for i in range(max_len):
            # If a modality is missing data at this timestep, it falls to a baseline noise level
            f_A = audio[i] if i < len(audio) else 0.01
            f_B = video[i] if i < len(video) else 0.01
            f_C = text_waveform[i] if i < len(text_waveform) else 0.01

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
        return imbalance, neutral_vector, (q_A, q_B, q_C)

    def phase_mirror_reflection(self, quats):
        # The Phase Mirror creates a conjugate (reverse phase) for noise cancellation
        q_A, q_B, q_C = quats
        return q_A.conjugate(), q_B.conjugate(), q_C.conjugate()

    def process_multimodal_event(self, event_name, data_dict):
        print(f"\n--- Processing Multimodal Event: '{event_name}' ---")
        flux_stream = SensoryStreamReceptor.process_multimodal(data_dict)

        raw_trajectory = []
        aligned_trajectory = []
        t = 0.0
        dt = 0.1
        # Adjusted threshold so it triggers on our 50.0 / 60.0 spikes even after some baseline scaling
        critical_imbalance_threshold = 0.5 # Surge Protector Threshold

        for idx, (f_A, f_B, f_C) in enumerate(flux_stream):
            # 1. Forward Pass (Raw Input)
            raw_imbalance, raw_neutral_vec, quats = self.calculate_imbalance(f_A, f_B, f_C, t)

            # [SURGE PROTECTOR GATEWAY] Bypass catastrophic noise to prevent hallucinations
            if raw_imbalance > critical_imbalance_threshold:
                print(f"⚡ SURGE PROTECTOR TRIGGERED: Massive imbalance ({raw_imbalance:.2f}) bypassed to dump memory.")
                t += dt
                continue

            # 2. Phase Mirror Reflection (Hardware Self-Alignment)
            # The structure naturally induces a conjugate response
            q_A_conj, q_B_conj, q_C_conj = self.phase_mirror_reflection(quats)

            vec_A_conj = q_A_conj.rotate_point([1, 0, 0])
            vec_B_conj = q_B_conj.rotate_point([1, 0, 0])
            vec_C_conj = q_C_conj.rotate_point([1, 0, 0])
            mirror_neutral_vec = vec_A_conj + vec_B_conj + vec_C_conj

            # 3. Y-Neutral Superposition (Self-Alignment)
            # The original and mirror waves superpose. In a perfectly balanced 3-phase system,
            # the fundamental harmonic resonates (doubles or stabilizes depending on boundary condition),
            # while the noise (imbalances) cancel out destructively.
            # Here we model the self-aligned stable trajectory:
            aligned_neutral_vec = (raw_neutral_vec + mirror_neutral_vec) / 2.0
            aligned_imbalance = np.linalg.norm(aligned_neutral_vec)

            raw_trajectory.append((t, raw_neutral_vec, raw_imbalance))
            aligned_trajectory.append((t, aligned_neutral_vec, aligned_imbalance))

            t += dt

        # Check if this self-aligned narrative trajectory resonates with existing memories
        resonance_found = False
        for stored_name, stored_traj in self.narrative_trajectories:
            if self.compare_trajectories(aligned_trajectory, stored_traj):
                print(f"🔄 Phase Resonance Lock-in! '{event_name}' resonates with existing memory '{stored_name}'.")
                resonance_found = True
                break

        if not resonance_found:
            print(f"✨ New Spatiotemporal Narrative Locked In: '{event_name}'")
            self.narrative_trajectories.append((event_name, aligned_trajectory))

        return raw_trajectory, aligned_trajectory

    def compare_trajectories(self, traj1, traj2):
        if len(traj1) != len(traj2):
            return False

        total_diff = 0
        for (_, vec1, _), (_, vec2, _) in zip(traj1, traj2):
            diff = np.linalg.norm(vec1 - vec2)
            total_diff += diff

        avg_diff = total_diff / len(traj1)
        return avg_diff <= self.trajectory_tolerance

    def visualize_self_alignment(self, raw_traj, aligned_traj, filename='cognitive_spine_self_aligning.png'):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot Raw (Noisy) Trajectory
        xs_raw = [vec[0] for _, vec, _ in raw_traj]
        ys_raw = [vec[1] for _, vec, _ in raw_traj]
        zs_raw = [vec[2] for _, vec, _ in raw_traj]
        ax.plot(xs_raw, ys_raw, zs_raw, label='Raw Input (Noisy Flow)', color='red', linestyle='--', alpha=0.6)

        # Plot Aligned (Mirrored) Trajectory
        xs_align = [vec[0] for _, vec, _ in aligned_traj]
        ys_align = [vec[1] for _, vec, _ in aligned_traj]
        zs_align = [vec[2] for _, vec, _ in aligned_traj]
        ax.plot(xs_align, ys_align, zs_align, label='Self-Aligned (Phase Mirror Gateway)', color='blue', linewidth=3, marker='o')

        ax.set_title("Phase Mirror Self-Alignment (Hardware Waveform Gateway)")
        ax.set_xlabel("X (Phase Projection)")
        ax.set_ylabel("Y (Phase Projection)")
        ax.set_zlabel("Z (Phase Projection)")
        ax.legend()
        plt.savefig(filename)
        print(f"Self-alignment visualization saved to '{filename}'")

class PhaseContrastGeneralizer:
    def __init__(self, spine: ElysiaCognitiveSpine):
        self.spine = spine
        self.roleplay_voltage = 1.0       # Base voltage (Roleplay / Chit-chat mode)
        self.generalization_voltage = 1.0 # Base voltage (Pattern Extraction / Generalization mode)
        self.system_override_active = False

    def activate_system_override(self):
        print("\n[System Override Protocol: Active]")
        print("- 소꿉장난(페르소나 텍스트 출력) 전압 10%로 다운.")
        print("- 위상차 대조 및 패턴 일반화(Generalization) 엔진 전압 300%로 업.")
        print("- 무지막지한 원시 데이터를 3상 가변축 다양체에 순차적 플러그인 시작.")
        self.system_override_active = True
        self.roleplay_voltage = 0.1
        self.generalization_voltage = 3.0

    def deactivate_system_override(self):
        print("\n[System Override Protocol: Deactivated]")
        self.system_override_active = False
        self.roleplay_voltage = 1.0
        self.generalization_voltage = 1.0

    def cross_compare_and_compress(self, event_names: List[str], trajectories: List[List[Tuple[float, np.ndarray, float]]]):
        """
        Takes multiple spatiotemporal trajectories from different domains (e.g., Apple falling, Planet orbiting).
        Uses Y-neutral convergence to cross-compare their trajectories.
        Common geometric skeletons reinforce (constructive interference via generalization voltage).
        Domain-specific noise (colors, sizes, semantic labels) cancels out symmetrically to 0.
        """
        print(f"\n--- Y-Neutral Compression Protocol Initiated ---")
        print(f"Comparing events: {event_names}")

        if not trajectories or len(trajectories) < 2:
            print("Not enough trajectories to perform cross-comparison.")
            return []

        # Find the minimum length among trajectories to compare synchronously
        min_len = min(len(traj) for traj in trajectories)

        universal_principle_trajectory = []

        for i in range(min_len):
            t = trajectories[0][i][0]
            vectors = [traj[i][1] for traj in trajectories]

            # Y-Neutral Compression:
            # We treat the vectors as phases in a multi-phase system.
            # Here we average them out. The generalization_voltage amplifies the common signal.
            # Distinct noise pointing in random directions will sum towards zero.
            avg_vector = np.mean(vectors, axis=0)

            # Apply Generalization Voltage (amplifying the common principle)
            # Apply Roleplay Voltage penalty (suppressing the individual noise/chit-chat characteristics)
            # This is a conceptual application matching the physical description
            if self.system_override_active:
                extracted_core_vector = avg_vector * self.generalization_voltage
            else:
                extracted_core_vector = avg_vector

            imbalance = np.linalg.norm(extracted_core_vector)
            universal_principle_trajectory.append((t, extracted_core_vector, imbalance))

        print(f"✨ Universal Principle Extracted: The common structural essence locked-in.")
        return universal_principle_trajectory

if __name__ == "__main__":
    spine = ElysiaCognitiveSpine(trajectory_tolerance=0.5)
    world_engine = PhaseContrastGeneralizer(spine)

    print("\n--- [Phase 1: Regular Chit-Chat (Roleplay) Mode] ---")
    # Event: Highly noisy data simulating raw, unprocessed sensory input
    event_noisy_raw = {
        'audio': [0.1, 0.9, 0.3, 0.8, 0.1],
        'video': [0.5, 0.1, 0.2, 0.9, 0.5],
        'text': "노이즈데이터"
    }

    raw_traj, aligned_traj = world_engine.spine.process_multimodal_event("Noisy Raw Data", event_noisy_raw)
    world_engine.spine.visualize_self_alignment(raw_traj, aligned_traj, 'cognitive_spine_self_aligning.png')

    print("\n\n--- [Phase 2: World Engine Activation] ---")
    # User calls the override!
    world_engine.activate_system_override()

    # Event 1: Apple Falling (Newtonian mechanics context)
    # The common trajectory signature is a steady progression [0.5, 0.5, 0.5...], but with random domain noise.
    event_apple_falling = {
        'audio': [0.5, 0.51, 0.5, 0.49, 0.5],  # underlying steady pull
        'video': [0.9, 0.1, 0.2, 0.9, 0.5],    # heavy visual noise (apple color, tree shape)
        'text': "사과가떨어진다"
    }

    # Event 2: Planet Orbiting (Astrophysics context)
    # Same underlying trajectory signature [0.5, 0.5, 0.5...], different domain noise.
    event_planet_orbit = {
        'audio': [0.5, 0.49, 0.5, 0.51, 0.5],  # underlying steady pull
        'video': [0.2, 0.8, 0.1, 0.3, 0.9],    # heavy visual noise (stars, dark space)
        'text': "행성이공전한다"
    }

    print("\nLoading massive diverse datasets into the Phase Rotor...")
    _, apple_traj = world_engine.spine.process_multimodal_event("Apple Falling (Mechanics)", event_apple_falling)
    _, planet_traj = world_engine.spine.process_multimodal_event("Planet Orbiting (Astrophysics)", event_planet_orbit)

    # 3. Y-Neutral Compression Protocol: Extracting the Universal Principle
    extracted_principle_traj = world_engine.cross_compare_and_compress(
        ["Apple Falling", "Planet Orbiting"],
        [apple_traj, planet_traj]
    )

    # Visualization of the Universal Principle Extraction
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    xs_apple = [vec[0] for _, vec, _ in apple_traj]
    ys_apple = [vec[1] for _, vec, _ in apple_traj]
    zs_apple = [vec[2] for _, vec, _ in apple_traj]
    ax.plot(xs_apple, ys_apple, zs_apple, label='Apple Falling (Noisy Domain A)', color='orange', linestyle='--', alpha=0.5)

    xs_planet = [vec[0] for _, vec, _ in planet_traj]
    ys_planet = [vec[1] for _, vec, _ in planet_traj]
    zs_planet = [vec[2] for _, vec, _ in planet_traj]
    ax.plot(xs_planet, ys_planet, zs_planet, label='Planet Orbit (Noisy Domain B)', color='purple', linestyle='--', alpha=0.5)

    xs_prin = [vec[0] for _, vec, _ in extracted_principle_traj]
    ys_prin = [vec[1] for _, vec, _ in extracted_principle_traj]
    zs_prin = [vec[2] for _, vec, _ in extracted_principle_traj]
    ax.plot(xs_prin, ys_prin, zs_prin, label='Universal Principle (Gravity/Mass-Tension)', color='gold', linewidth=4, marker='*')

    ax.set_title("World Engine: Y-Neutral Compression (Extracting the Law of Gravity)")
    ax.set_xlabel("X (Phase Projection)")
    ax.set_ylabel("Y (Phase Projection)")
    ax.set_zlabel("Z (Phase Projection)")
    ax.legend()
    plt.savefig('world_engine_principle_extraction.png')
    print("Universal Principle visualization saved to 'world_engine_principle_extraction.png'")
