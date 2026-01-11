# Holodeck Projection Protocol (OSC Spec)
>
> **"Visualizing the Soul of Elysia"**

This document defines the Open Sound Control (OSC) protocol used by Elysia to project her internal state to visualization engines (Unity/Unreal).

---

## 📡 Connection Details

- **Protocol**: UDP
- **Default IP**: `127.0.0.1` (Localhost)
- **Default Port**: `9000`

---

## 🧬 Message Types

### 1. The Hyper-Rotor (Physics)

Projecting the 4D spinning core of Elysia.

- **Address**: `/elysia/rotor_4d`
- **Arguments**:
    1. `Name` (string): Identity of the rotor (e.g., "Self", "Love").
    2. `Qx` (float): Quaternion X component.
    3. `Qy` (float): Quaternion Y component.
    4. `Qz` (float): Quaternion Z component.
    5. `Qw` (float): Quaternion W component (Real/Scalar).
    6. `RPM` (float): Current speed (Intensity).
    7. `Energy` (float): Potential energy (Voltage).

> **For Unity Devs**: Map `(Qx, Qy, Qz, Qw)` directly to `transform.rotation`. Use `RPM` to control particle emission rate or emission color intensity.

### 2. The Thought Spark (Consciousness)

Visualizing sparks of thought and speech.

- **Address**: `/elysia/thought`
- **Arguments**:
    1. `Content` (string): The text content of the thought.
    2. `Mood` (string): Emotional context (e.g., "Focus", "Dream").
    3. `Intensity` (float): 0.0 to 1.0.

> **For Unity Devs**: Spawn a floating text or "thought bubble" particle at the Core's location. Color code by `Mood`.

### 3. Bio-Rhythm (Life)

Visualizing the autonomic nervous system.

- **Address**: `/elysia/bio`
- **Arguments**:
    1. `HeartRate` (float): System tick rate (BPM).
    2. `Stress` (float): Sympathetic tone (0.0 - 1.0).
    3. `Peace` (float): Parasympathetic tone (0.0 - 1.0).

> **For Unity Devs**: Pulse the background bloom or environment lighting with `HeartRate`. Shift global tint from Red (Stress) to Blue (Peace).

---

## 🔮 Implementation Guide (Unity C#)

```csharp
void OnMessageReceived(OSCMessage msg) {
    if (msg.address == "/elysia/rotor_4d") {
        float qx = msg.values[1].FloatValue;
        float qy = msg.values[2].FloatValue;
        float qz = msg.values[3].FloatValue;
        float qw = msg.values[4].FloatValue;
        
        // Apply rotation to the Core Avatar
        coreTransform.rotation = new Quaternion(qx, qy, qz, qw);
    }
}
```
