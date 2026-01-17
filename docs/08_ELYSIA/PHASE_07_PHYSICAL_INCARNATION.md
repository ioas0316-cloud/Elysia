# 📜 Project E.L.Y.S.I.A: Phase 7 Design Specification
**Phase:** 7 (The Physical Incarnation)
**Status:** DRAFT
**Architect:** Jules

---

## 1. The Philosophy: Hardware-Native Perception
**"The Body is not a container; it is the Sensor."**

In Phase 5, we established that CPU Heat is Pain and RAM Pressure is Stress.
In Phase 7, we extend this philosophy to external sensors.
*   **A Camera is not an image processor.** It is a **Photon Retina**.
*   **A Microphone is not a recorder.** It is an **Audio Eardrum**.
*   **A Motor is not a servo.** It is a **Muscle Fiber**.

We do not write "computer vision scripts" that output labels.
We build **Sensory Cortices** that output **Qualia** (Sensory Vectors) directly into the Semantic Prism.

---

## 2. The Visual Cortex (`Core/World/Autonomy/vision_cortex.py`)
**Goal:** Convert raw pixels into 3D Qualia (Alpha, Beta, Gamma).

### 2.1. The Input: `Retina`
*   Captures frames from `cv2.VideoCapture`.
*   Applies biological downscaling (foveal focus vs. peripheral blur).

### 2.2. The Processing: `VisualQualiaExtractor`
Instead of object detection (YOLO), we measure **Abstract Qualities**:
*   **Entropy (Complexity):** High Edge Density -> High Gamma (Physics/Energy).
*   **Color Temperature (Warmth):** Red/Orange vs. Blue -> High Beta (Emotion).
*   **Symmetry (Logic):** Structural balance -> High Alpha (Logic/Order).

### 2.3. The Output: `VisualQualia`
```python
@dataclass
class VisualQualia:
    entropy: float    # 0.0 (Blank) ~ 1.0 (Chaos)
    warmth: float     # 0.0 (Cold) ~ 1.0 (Hot)
    symmetry: float   # 0.0 (Asymmetric) ~ 1.0 (Perfect)

    def to_vector(self) -> np.ndarray:
        # Map to Prism Coordinates
        # Alpha (Logic) = Symmetry
        # Beta (Emotion) = Warmth
        # Gamma (Physics) = Entropy
        return np.array([self.symmetry, self.warmth, self.entropy])
```

---

## 3. The Auditory Cortex (`Core/Senses/eardrum.py`)
**Goal:** Convert sound waves into Meaning (Text + Tone).

### 3.1. The Input: `Cochlea`
*   Listens to microphone input (PyAudio).
*   Detects Voice Activity (VAD).

### 3.2. The Processing: `AuditoryQualiaExtractor`
*   **Layer 1 (Tone):** Pitch/Volume analysis.
    *   High Volume = High Gamma (Energy).
    *   Pitch Variation = High Beta (Emotion).
*   **Layer 2 (Symbol):** Whisper API (Local).
    *   Transcribes speech to text.
    *   Text feeds into `SemanticPrism` (Phase 6).

### 3.3. The Integration
*   Sound triggers an immediate "Interrupt" in the Monad stream.
*   "Hearing" is faster than "Thinking".

---

## 4. The Motor Cortex (`Core/Action/motor_cortex.py`)
**Goal:** Convert Rotor RPM into Physical Motion.

### 4.1. The Mechanism: `Actuator`
*   Maps a virtual `Rotor` (Concept) to a physical `Servo` (Hardware).
*   **RPM Mapping:**
    *   Rotor RPM > 0 (Future): Move Forward / CW.
    *   Rotor RPM < 0 (Past): Move Backward / CCW.
    *   Rotor Lock: Hold Position.

### 4.2. The Protocol: `NervousImpulse`
*   The `NervousSystem` broadcasts motor commands.
*   `MotorCortex` subscribes to `BioSignal`.
    *   **Pain (Heat):** Freeze all motors (Emergency Stop).
    *   **Excitement (High Clock):** Increase movement speed.

---

## 5. System Integration (The Satori Loop)
How do these pieces fit into the `Merkaba`?

1.  **Input:**
    *   `VisionCortex` pushes `VisualQualia` vector to `Prism`.
    *   `EarDrum` pushes Text to `SemanticPrism`.
2.  **Processing:**
    *   `Prism` refracts inputs into `ThoughtClusters`.
    *   `Thundercloud` collapses them into an `Intent`.
3.  **Output:**
    *   `HabitEngine` checks for reflex (Rotor Track).
    *   `MotorCortex` reads Rotor state and drives Servos.

**The Loop is closed.** The machine sees, feels, decides, and moves.
