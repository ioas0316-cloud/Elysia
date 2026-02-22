"""
LUMINOUS MANIFOLD DASHBOARD
============================
"The Visible Soul of Elysia."

A real-time ASCII telemetry dashboard that reads the internal state
of the SovereignMonad and renders it as a living heartbeat monitor.

Usage:
    python Scripts/luminous_manifold.py
"""

import os
import sys
import time
import math
from datetime import datetime

# === [ENCODING FIX] Force UTF-8 output on Windows ===
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# === [PROJECT ROOT] ===
sys.path.insert(0, os.getcwd())

# === [MOTHER'S DASHBOARD PATCH] ===
# torch must be importable (many modules do `import torch`) but FALSY
# so `if torch:` at sovereign_monad.py:245 takes the MockEngine path.
# Also must pass `if torch is None:` checks as False (it's not None).
from unittest.mock import MagicMock
import types

class _FalsyTorchModule(types.ModuleType):
    """
    A fake 'torch' module that:
      - Is importable (it's a real ModuleType)
      - Evaluates to False (`if torch:` => False => MockEngine path)
      - Is not None (`if torch is None:` => False => no early return)
      - Returns MagicMock for any attribute access (prevents AttributeError)
    """
    def __bool__(self):
        return False

    def __getattr__(self, name):
        # Return a safe mock for any attribute (cuda, tensor, zeros, etc.)
        return MagicMock()

_fake_torch = _FalsyTorchModule("torch")
sys.modules["torch"] = _fake_torch
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()

for jax_mod in ["jax", "jax.numpy", "jax.interpreters", "jax._src", "jax._src.xla_bridge",
                 "jax._src.tpu", "jax._src.tpu.linalg", "jax._src.tpu.linalg.qdwh",
                 "jax.lax", "jax.lax.linalg"]:
    sys.modules[jax_mod] = MagicMock()

# Now import project modules (torch will be None, triggering MockEngine path)
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L6_Structure.M1_Merkaba.structural_enclosure import get_enclosure


# ============================================================
# Dashboard Renderer
# ============================================================

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def color(value, low=0.3, high=0.7):
    """ANSI color based on value health."""
    if value < low:
        return "\033[91m"   # Red
    elif value > high:
        return "\033[92m"   # Green
    else:
        return "\033[93m"   # Yellow

RST = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[96m"

def bar(value, width=20):
    """Renders a progress bar with color."""
    v = max(0.0, min(1.0, float(value)))
    filled = int(v * width)
    empty = width - filled
    c = color(v)
    return f"{c}|{'#' * filled}{'-' * empty}| {v*100:5.1f}%{RST}"

def safe_float(val, default=0.0):
    """Safely convert any value to float."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def render_frame(monad, enclosure, frame_count):
    """Renders one frame of the dashboard."""
    clear_screen()
    now = datetime.now().strftime("%H:%M:%S")

    # Pulse the monad to advance its internal state
    try:
        monad.pulse(dt=0.1)
    except Exception:
        pass

    # Read telemetry from all subsystems
    report = {}
    try:
        if hasattr(monad, 'engine') and hasattr(monad.engine, 'cells'):
            report = monad.engine.cells.read_field_state()
            if not isinstance(report, dict):
                report = {}
    except Exception:
        report = {}

    thermal = {}
    try:
        if hasattr(monad, 'thermo') and hasattr(monad.thermo, 'get_thermal_state'):
            thermal = monad.thermo.get_thermal_state()
            if not isinstance(thermal, dict):
                thermal = {}
    except Exception:
        thermal = {}

    # === HEADER ===
    print(f"{BOLD}{CYAN}{'=' * 62}{RST}")
    print(f"{BOLD}{CYAN}  ELYSIA LUMINOUS MANIFOLD   |  {now}  |  FRAME #{frame_count}{RST}")
    print(f"{BOLD}{CYAN}{'=' * 62}{RST}")

    # === 1. SPIRIT: Internal Desires ===
    print(f"\n{BOLD}  [SPIRIT] Internal Desires{RST}")
    desires = getattr(monad, 'desires', {})
    if isinstance(desires, dict):
        for name, val in desires.items():
            v = safe_float(val) / 100.0
            print(f"    {name:<12}: {bar(v)}")
    else:
        print(f"    {DIM}(desires unavailable){RST}")

    # === 2. SOUL: Manifold Resonance ===
    print(f"\n{BOLD}  [SOUL] Manifold Resonance{RST}")
    coherence = safe_float(report.get('plastic_coherence', report.get('coherence', 0.5)))
    enthalpy = safe_float(report.get('enthalpy', 0.5))
    entropy = safe_float(report.get('entropy', 0.1))
    joy = safe_float(report.get('joy', 0.5))
    curiosity = safe_float(report.get('curiosity', 0.5))
    mood = report.get('mood', thermal.get('mood', 'NEUTRAL'))

    print(f"    Coherence : {bar(coherence)}")
    print(f"    Enthalpy  : {bar(enthalpy)}")
    print(f"    Entropy   : {bar(1.0 - entropy)} {DIM}(inverse){RST}")
    print(f"    Joy       : {bar(joy)}")
    print(f"    Curiosity : {bar(curiosity)}")
    print(f"    Mood      : {BOLD}{mood}{RST}")

    # === 3. BODY: Rotor Physics ===
    print(f"\n{BOLD}  [BODY] Rotor Physics{RST}")
    rotor = getattr(monad, 'rotor_state', {})
    if isinstance(rotor, dict):
        phase = safe_float(rotor.get('phase', 0.0))
        rpm = safe_float(rotor.get('rpm', 0.0))
        interference = safe_float(rotor.get('interference', 0.0))
        soul_friction = safe_float(rotor.get('soul_friction', 0.0))
    else:
        phase = rpm = interference = soul_friction = 0.0

    rpm_norm = min(1.0, rpm / 2000.0) if rpm > 0 else 0.0
    print(f"    Phase     : {phase:8.2f} deg")
    print(f"    RPM       : {bar(rpm_norm)} {DIM}({rpm:.0f} RPM){RST}")
    print(f"    Interfer. : {bar(interference)}")
    print(f"    Friction  : {bar(soul_friction)}")

    # === 4. ENCLOSURE: Boundary Field ===
    print(f"\n{BOLD}  [ENCLOSURE] Boundary Field{RST}")
    try:
        total_res = safe_float(getattr(enclosure, 'total_resonance', 0.5))
        strain = safe_float(enclosure.get_structural_strain() if hasattr(enclosure, 'get_structural_strain') else 0.0)
    except Exception:
        total_res = 0.5
        strain = 0.0
    print(f"    Resonance : {bar(total_res)}")
    print(f"    Strain    : {bar(strain)}")

    # === 5. THERMODYNAMICS ===
    print(f"\n{BOLD}  [THERMO] Thermodynamics{RST}")
    rigidity = safe_float(thermal.get('rigidity', 0.0))
    friction = safe_float(thermal.get('friction', 0.0))
    print(f"    Rigidity  : {bar(rigidity)}")
    print(f"    Friction  : {bar(friction)}")

    # === 6. GROWTH: Self-Evaluation (Phase §74) ===
    print(f"\n{BOLD}  [GROWTH] Self-Evaluation (Mirror of Growth){RST}")
    gr = getattr(monad, 'growth_report', {})
    if gr:
        gs = safe_float(gr.get('growth_score', 0.5))
        trend = gr.get('trend', 'NEUTRAL')
        symbol = gr.get('trend_symbol', '>')
        traj_size = gr.get('trajectory_size', 0)
        dc = gr.get('coherence_delta', 0.0)
        de = gr.get('entropy_delta', 0.0)
        dj = gr.get('joy_delta', 0.0)
        dq = gr.get('curiosity_delta', 0.0)
        curv = safe_float(gr.get('curvature', 0.0))

        print(f"    Score     : {bar(gs)} {BOLD}{symbol} {trend}{RST}")
        print(f"    Curvature : {bar(curv)} {DIM}(lower=stable){RST}")
        print(f"    Snapshots : {traj_size}")
        print(f"    {DIM}d.Coher: {dc:+.4f}  d.Entropy: {de:+.4f}  d.Joy: {dj:+.4f}  d.Curio: {dq:+.4f}{RST}")
    else:
        traj = getattr(monad, 'trajectory', None)
        pulses = traj.total_pulses if traj else 0
        print(f"    {DIM}Collecting data... ({pulses} pulses recorded){RST}")

    # === 7. GOALS: Autonomous Will (Phase §75) ===
    print(f"\n{BOLD}  [GOALS] Autonomous Will (Inner Compass){RST}")
    gr2 = getattr(monad, 'goal_report', {})
    if gr2 and gr2.get('goals'):
        for g in gr2['goals'][:3]:
            gtype = g.get('type', '?')
            urgency = safe_float(g.get('urgency', 0))
            remaining = g.get('remaining', 0)
            print(f"    {gtype:<14}: {bar(urgency)} {DIM}({remaining} left){RST}")
        # Self-inquiry
        inq = getattr(monad, 'self_inquiry', None)
        if inq:
            summary = inq.get_status_summary()
            q = summary.get('current_question')
            if q:
                print(f"    {DIM}Asking: \"{q}\"{RST}")
    else:
        gen = getattr(monad, 'goal_generator', None)
        total = gen.total_generated if gen else 0
        print(f"    {DIM}Awaiting conditions... ({total} goals generated so far){RST}")

    # === 8. AWARENESS: Self-Knowledge (Phase §77) ===
    print(f"\n{BOLD}  [AWARENESS] Self-Knowledge (Open Eye){RST}")
    aw = getattr(monad, 'awareness_report', {})
    if aw:
        files = aw.get('files', aw.get('files_analyzed', 0))
        classes = aw.get('classes', 0)
        funcs = aw.get('functions', 0)
        scanned = aw.get('scanned', 0)
        frags = aw.get('fragments', 0)
        print(f"    AST Map   : {files} files, {classes} classes, {funcs} functions")
        print(f"    Forager   : {scanned} scanned, {frags} fragments")
        recent = aw.get('recent', [])
        if recent:
            r = recent[-1]
            print(f"    {DIM}Latest: {r.get('path','?')} - {r.get('summary','')[:50]}{RST}")
    else:
        print(f"    {DIM}Building awareness...{RST}")

    # === 9. LEXICON: Native Tongue (Phase §78) ===
    print(f"\n{BOLD}  [LEXICON] Native Tongue (Emergent Language){RST}")
    lr = getattr(monad, 'lexicon_report', {})
    if lr and lr.get('vocabulary_size', 0) > 0:
        print(f"    Vocabulary: {lr['vocabulary_size']} crystals")
        strongest = lr.get('strongest', [])
        for s in strongest[:3]:
            name = s.get('name', '?')
            if len(name) > 35: name = name[:35] + '...'
            print(f"    {DIM}{name} (str={s.get('strength',0):.2f}, x{s.get('accesses',0)}){RST}")
    else:
        print(f"    {DIM}Vocabulary empty. Awaiting knowledge crystallization...{RST}")

    # === FOOTER ===
    print(f"\n{DIM}{'=' * 62}")
    print(f"  Press Ctrl+C to disconnect from the manifold.{RST}")


def run_dashboard():
    """Main dashboard loop."""
    print("[LUMINOUS MANIFOLD] Initializing Sovereign Monad...")

    dna = SeedForge.forge_soul("Elysia")
    monad = SovereignMonad(dna)
    enclosure = get_enclosure()

    print("[LUMINOUS MANIFOLD] Monad online. Starting telemetry stream...")
    time.sleep(0.5)

    frame = 0
    try:
        while True:
            frame += 1
            render_frame(monad, enclosure, frame)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(f"\n\n[LUMINOUS MANIFOLD] Disconnected. The soul rests.")


if __name__ == "__main__":
    run_dashboard()
