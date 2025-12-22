# Protocol 03: Observability, Debugging & Telemetry

> "If you can't measure it, you can't improve it."

---

## 1. Overview

Protocol 03 defines how Elysia observes itself in real-time during learning and simulation.

This is the **eyes and ears** of self-improvement: without observability, Elysia cannot achieve meta-learning (the ability to improve its own learning).

---

## 2. Core Logging Infrastructure

### 2.1 Resonance Pattern Logging (`logs/resonance_patterns.jsonl`)

**Purpose**: Track which concepts are being "lit up" by incoming thoughts.

**Trigger**: Every time `ResonanceEngine.calculate_global_resonance()` is called.

**Format** (one line per event):
```json
{
  "timestamp": 1234567890.123,
  "wave_text": "사랑과 빛",
  "intensity": 2.0,
  "top_resonances": [
    {"node_id": "concept_123", "score": 0.92},
    {"node_id": "concept_456", "score": 0.87},
    ...
  ],
  "total_nodes": 150
}
```

**Analysis Use**:
- Identify which concepts are most "responsive" to language input.
- Detect if certain concepts are over-resonating (potential bottleneck).
- Build a "resonance signature" of the agent's current mind state.

---

### 2.2 Phase-Resonance Event Detection (`logs/resonance_events.jsonl`)

**Purpose**: Detect anomalies and peaks in the information structure (evidence of "transcendence").

**Trigger**: After each major simulation cycle (e.g., every 1000 ticks in ultra-dense simulation).

**Format**:
```json
{
  "timestamp": 1234567890.123,
  "duration_ticks": 1000,
  "event_count": 3,
  "events": [
    {
      "type": "density_multimodality",
      "severity": "high",
      "metric": 1.23,
      "interpretation": "Information density shows peaked distribution",
      "involved_particles": 47
    },
    {
      "type": "wavelength_standing_waves",
      "severity": "medium",
      "clusters": [[620, 15], [580, 12]],
      "interpretation": "Two distinct wavelength clusters detected",
      "involved_particles": 27
    },
    ...
  ]
}
```

**Analysis Use**:
- Detect "breakthroughs" (sudden coherence in the information space).
- Monitor stability (if events are constant, system may be stuck).
- Validate that simulation is producing meaningful emergent patterns, not noise.

---

### 2.3 Checkpoint & Resume Protocol

**Purpose**: Enable long-running simulations to survive crashes and be resumed mid-stream.

**Trigger**: Periodic (every N ticks, where N is configurable).

**Checkpoint Structure** (`runs/ultra_dense_<timestamp>/checkpoint_<tick>.json`):
```json
{
  "tick": 5000,
  "elapsed_seconds": 123.45,
  "particles_count": 512,
  "meta_stats": {
    "total_subjective_time": 1e20,
    "effective_acceleration": 1e15
  },
  "top_particles": [
    {
      "id": "particle_001",
      "info_density": 0.95,
      "concept": "concept_love"
    },
    ...
  ]
}
```

**Resume Capability** (TODO in next update):
- Load latest checkpoint.
- Restore system state.
- Continue from that tick onward.

---

### 2.4 Fractal Connectivity Validation

**Purpose**: Verify that the three hierarchies (WorldTree, Hippocampus, ResonanceEngine) remain coherent.

**Tool**: `Tools/validate_fractal_connectivity.py`

**Output** (`logs/fractal_validation_<timestamp>.json`):
```json
{
  "total_issues": 12,
  "by_type": {
    "orphan_node": 3,
    "isolated_node": 5,
    "denormalized_qubit": 4
  },
  "by_severity": {
    "high": 3,
    "medium": 5,
    "low": 4
  },
  "issues": [
    {
      "type": "orphan_node",
      "node_id": "concept_xyz",
      "severity": "high"
    },
    ...
  ]
}
```

**When to Run**:
- After every major curriculum phase.
- Before declaring a simulation "successful".
- Monthly as a maintenance check.

---

### 2.5 Language Trajectory Monitoring

**Purpose**: Track whether the agent's language is diverging, converging, or staying balanced.

**Tool**: `Tools/analyze_language_trajectory.py`

**Output** (`logs/language_trajectory_<timestamp>.json`):
```json
{
  "total_records": 50,
  "path_entropy": {
    "average": 0.85,
    "max": 1.2,
    "min": 0.1,
    "unique_paths": 23,
    "most_common_paths": [...]
  },
  "subject_variance": {
    "unique_subjects": 15,
    "avg_subjects_per_record": 3.2,
    "top_subject": "law:play",
    "top_subject_percentage": 32.5
  },
  "convergence_drift": {
    "first_half_variance": 8.5,
    "second_half_variance": 3.2,
    "drift_score": 0.62,
    "is_converging": true
  },
  "alerts": [
    "⚠️  CONVERGENCE DRIFT: Subject variance dropped 62%. Agent may be narrowing focus excessively.",
    ...
  ]
}
```

**Interpretation**:
- High entropy = Good (exploring diverse paths).
- High subject variance = Good (learning many concepts).
- Low convergence drift = Good (maintaining diversity).
- Alerts trigger when agent gets stuck.

---

## 3. Real-Time Telemetry Dashboard (TODO)

**Goal**: Web-based UI showing:
- Live resonance heatmap (which concepts are active).
- Event stream (anomalies as they occur).
- Learning trajectory (progress on language, concepts, wisdom).
- System health (CPU/memory/stability).

**Planned Technology**: `Tools/visualizer_server.py` (already started).

---

## 4. Integration with CODEX

### 4.1 Self-Awareness Loop (사자 인식 루프)

The three outputs (resonance_patterns, resonance_events, language_trajectory) feed into a **meta-cognition layer**:

```
Simulation → Logging (Pattern, Event, Trajectory) 
         → Analysis (Fractal Check, Trajectory Analysis)
         → Meta-Decision: "Am I converging? Do I need to change strategy?"
         → Curriculum Adjustment (add new corpus, new laws, new branches)
         → Next Simulation (with updated parameters)
```

### 4.2 Failure Modes to Watch

| Symptom | Cause | Fix |
|---------|-------|-----|
| All events = "low" severity | System is bored, not learning | Add harder curriculum |
| All events = "high" severity | System is chaotic, not stable | Add constraints / slow down |
| Convergence drift > 0.8 | Over-focusing on one topic | Introduce diversity corpus |
| Density multimodality = 0 | Information is flat/uniform | Increase interference/mutation |
| Fractals have many orphans | Concepts not integrated | Audit WorldTree + Hippocampus |

---

## 5. Maintenance Schedule

| Task | Frequency | Owner | Output |
|------|-----------|-------|--------|
| Resonance pattern logging | Real-time (per wave) | ResonanceEngine | logs/resonance_patterns.jsonl |
| Event detection | Per 1000 ticks | ExperienceDigester | logs/resonance_events.jsonl |
| Checkpoint save | Per 1000 ticks (configurable) | UltraDenseSimulation | runs/ultra_dense_*/checkpoint_*.json |
| Fractal validation | Per phase (weekly) | Developer/Script | logs/fractal_validation_*.json |
| Language analysis | Per phase (weekly) | Script | logs/language_trajectory_*.json |
| Dashboard refresh | Real-time | VisualizerServer | Web UI @ http://localhost:8080 |

---

## 6. Data Retention Policy

- **Real-Time Logs** (resonance_patterns.jsonl, resonance_events.jsonl):
  - Keep last 7 days in main logs/.
  - Archive older logs to logs/archive/ (compressed).
  - Retention: 30 days full, 1 year compressed.

- **Checkpoints**:
  - Keep last 5 checkpoints per run.
  - Discard others to save disk.
  - Full run archive to runs/archive/ after completion.

- **Analysis Reports**:
  - Keep all reports (small JSON files).
  - Link reports to corresponding simulation run ID.

---

## 7. Debugging Workflow

When system behaves unexpectedly:

1. **Check latest resonance_events.jsonl**: What anomalies were detected?
2. **Check language_trajectory_*.json**: Is language converging/diverging unexpectedly?
3. **Run validate_fractal_connectivity.py**: Are there structural issues?
4. **Review resonance_patterns.jsonl**: Which concepts are lighting up?
5. **Inspect checkpoint**: What was the system state at the time?
6. **Propose fix**: Adjust curriculum, laws, or hyperparameters.
7. **Resume from checkpoint or restart** with changes.

---

## 8. Future Extensions

- **Causal Backtracking**: Given an unexpected outcome, automatically find which earlier event caused it.
- **Predictive Telemetry**: Forecast the next 1000 ticks based on current trajectory.
- **Anomaly Outlier Detection**: Automatically flag unusual patterns before they become problems.
- **Cross-Simulation Comparison**: Compare metrics across multiple parallel runs (ensembles).

---

**End of Protocol 03**
