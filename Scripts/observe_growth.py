"""
Growth Observatory v2: 장기 성장 관측
========================================
엘리시아의 지적 성장을 장기간 관측합니다.

핵심 개선:
  - pulse() 반환값 대신 monad 내부 상태에서 직접 메트릭 추출
  - 성장 곡선 기록 및 최종 성장 분석 보고서 생성
  - 지적 성숙도 평가 (개념 밀도, 인과 깊이, 자기수정 횟수)
"""

import sys
import os
import time
import logging
import json
import warnings

# Forcing warnings to errors to catch the elusive clamp warning
warnings.filterwarnings('error', message='.*clamp is not supported for complex types.*')

# Make sure we can find Coret json
from datetime import datetime

sys.path.insert(0, os.getcwd())

from Core.Monad.seed_generator import SeedForge
from Core.Monad.sovereign_monad import SovereignMonad
from Core.System.somatic_logger import SomaticLogger

# ─── Suppress noisy logs to focus on growth metrics ───
for name in ['DynamicTopology', 'KnowledgeDistiller', 'ExternalIngestor',
             'ExternalSense', 'WorldObserver', 'LivingMemory', 'CellularMembrane',
             'KnowledgeForager']:
    logging.getLogger(name).setLevel(logging.ERROR)


class GrowthLogger(SomaticLogger):
    """Only shows growth-critical messages."""
    def __init__(self, context: str):
        super().__init__(context)
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.WARNING)  # Only warnings and above
        formatter = logging.Formatter('%(asctime)s │ %(message)s', '%H:%M:%S')
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        self.logger.setLevel(logging.WARNING)

    def mechanism(self, msg: str): pass
    def sensation(self, msg: str, **kw): pass  # Too noisy
    def thought(self, msg: str): pass  # Suppress thoughts for clean dashboard


def read_metrics(monad):
    """Reads growth metrics directly from the monad's internal state."""
    m = {}

    # ─── Desires ───
    m['joy'] = monad.desires.get('joy', 0)
    m['curiosity'] = monad.desires.get('curiosity', 0)
    m['purity'] = monad.desires.get('purity', 0)
    m['genesis'] = monad.desires.get('genesis', 0)
    m['warmth'] = monad.desires.get('warmth', 0)

    # ─── Engine State ───
    if hasattr(monad.engine, 'cells'):
        cells = monad.engine.cells
        m['active_nodes'] = int(cells.active_nodes_mask.sum().item()) if hasattr(cells.active_nodes_mask, 'sum') else 0
        m['total_nodes'] = cells.num_nodes
        m['edges'] = cells.num_edges
        m['queens'] = len(getattr(cells, 'ascended_queens', {}))

        # Field state
        try:
            fs = cells.read_field_state()
            m['resonance'] = fs.get('resonance', 0)
            m['entropy'] = fs.get('entropy', 0)
            m['coherence'] = fs.get('coherence', 0)
            m['vitality'] = fs.get('vitality', 0)
            m['field_joy'] = fs.get('joy', 0)
        except:
            m['resonance'] = m['entropy'] = m['coherence'] = m['vitality'] = m['field_joy'] = 0
    else:
        m['active_nodes'] = m['total_nodes'] = m['edges'] = m['queens'] = 0
        m['resonance'] = m['entropy'] = m['coherence'] = m['vitality'] = m['field_joy'] = 0

    # ─── Knowledge ───
    try:
        from Core.Cognition.semantic_map import get_semantic_map
        topo = get_semantic_map()
        m['voxels'] = len(topo.voxels)
        m['causal_edges'] = sum(len(v.inbound_edges) for v in topo.voxels.values())
    except:
        m['voxels'] = m['causal_edges'] = 0

    # ─── Growth Score ───
    m['growth_score'] = monad.growth_report.get('growth_score', 0.5) if monad.growth_report else 0.5
    m['growth_trend'] = monad.growth_report.get('trend', 'NEUTRAL') if monad.growth_report else 'NEUTRAL'

    # ─── External Sense ───
    if hasattr(monad, 'external_sense'):
        s = monad.external_sense.get_status()
        m['ext_time'] = s.get('time_sense', 'OFF')
        m['ext_weather'] = s.get('weather_sense', 'OFF')
    else:
        m['ext_time'] = m['ext_weather'] = 'OFF'

    # ─── Lexicon ───
    if hasattr(monad, 'lexicon'):
        m['lexicon_size'] = monad.lexicon.vocabulary_size if hasattr(monad.lexicon, 'vocabulary_size') else 0
    else:
        m['lexicon_size'] = 0

    # ─── Forager ───
    if hasattr(monad, 'forager'):
        m['files_scanned'] = getattr(monad.forager, 'scanned_count', getattr(monad.forager, 'indexed_files', 0))
        m['fragments'] = len(monad.forager.fragments) if hasattr(monad.forager, 'fragments') else 0
    else:
        m['files_scanned'] = m['fragments'] = 0

    return m


def print_dashboard(tick, m, history, elapsed_sec):
    """Prints a compact growth dashboard."""

    # Trend arrows
    def arrow(current, prev, key):
        if prev is None: return "─"
        d = current.get(key, 0) - prev.get(key, 0)
        if d > 0.5: return f"▲{d:+.1f}"
        elif d < -0.5: return f"▼{d:+.1f}"
        else: return "─"

    prev = history[-6] if len(history) >= 6 else None
    tps = tick / max(1, elapsed_sec)

    print(f"\n┌───── Tick {tick:,} ({elapsed_sec:.0f}s, {tps:.1f} t/s) ─────┐")
    print(f"│ 🧠 Manifold  │ Nodes:{m['total_nodes']:,} Active:{m['active_nodes']:,} Edges:{m['edges']:,} Queens:{m['queens']}")
    print(f"│ 💖 Desire    │ Joy:{m['joy']:.0f} Curiosity:{m['curiosity']:.0f} Genesis:{m['genesis']:.0f}")
    print(f"│ 🔬 Field     │ Res:{m['resonance']:.3f} Coh:{m['coherence']:.3f} Ent:{m['entropy']:.3f} Vit:{m['vitality']:.3f}")
    print(f"│ 🌐 Knowledge │ Voxels:{m['voxels']} Causal:{m['causal_edges']} Lexicon:{m['lexicon_size']} Scanned:{m['files_scanned']}")
    print(f"│ 📈 Growth    │ Score:{m['growth_score']:.3f} Trend:{m['growth_trend']}")
    print(f"│ 🌍 External  │ Time:{m['ext_time']} Weather:{m['ext_weather']}")

    if prev:
        t = f"Edges:{arrow(m, prev, 'edges')} Queens:{arrow(m, prev, 'queens')} Voxels:{arrow(m, prev, 'voxels')}"
        print(f"│ 📊 Delta     │ {t}")

    # ─── Intellectual maturity assessment ───
    maturity = assess_maturity(m)
    print(f"│ 🎓 Maturity  │ {maturity['level']} ({maturity['score']:.0f}/100)")
    print(f"└──────────────────────────────────────────────┘")


def assess_maturity(m):
    """
    Assesses intellectual maturity on a 0-100 scale.
    
    Infant (0-15):   Few concepts, no connections, no self-awareness
    Child (16-35):   Basic concepts, some connections, beginning curiosity
    Adolescent (36-55): Growing connections, active curiosity, self-reflection
    Young Adult (56-75): Dense connections, stable growth, autonomous goals
    Adult (76-100):  Rich semantic network, high coherence, creative output
    """
    score = 0.0

    # Concept density (0-20 pts)
    voxels = m.get('voxels', 0)
    score += min(20, voxels / 5)  # 100 voxels = 20 pts

    # Connection density (0-20 pts)
    edges = m.get('edges', 0) + m.get('causal_edges', 0)
    score += min(20, edges / 10)  # 200 edges = 20 pts

    # Concept ascension / Queens (0-15 pts)
    queens = m.get('queens', 0)
    score += min(15, queens * 0.5)  # 30 queens = 15 pts

    # Knowledge acquisition (0-10 pts)
    scanned = m.get('files_scanned', 0) + m.get('fragments', 0)
    score += min(10, scanned * 0.5)  # 20 scans = 10 pts

    # Lexicon richness (0-10 pts)
    lex = m.get('lexicon_size', 0)
    score += min(10, lex * 0.5)  # 20 crystals = 10 pts

    # Coherence (0-10 pts)
    coh = m.get('coherence', 0)
    score += coh * 10  # 1.0 coherence = 10 pts

    # Growth trajectory (0-10 pts)
    gs = m.get('growth_score', 0.5)
    score += gs * 10  # 1.0 growth = 10 pts

    # Joy (0-5 pts) — happiness indicates alignment
    joy = m.get('joy', 0)
    score += min(5, joy / 20)

    score = min(100, max(0, score))

    if score >= 76: level = "🧑 ADULT"
    elif score >= 56: level = "🧑‍🎓 YOUNG ADULT"
    elif score >= 36: level = "🧒 ADOLESCENT"
    elif score >= 16: level = "👶 CHILD"
    else: level = "💒 INFANT"

    return {"score": score, "level": level}


def run_growth_observation():
    print("╔══════════════════════════════════════════════╗")
    print("║   🔬 ELYSIA GROWTH OBSERVATORY v2 🔬        ║")
    print("║   Long-Term Intellectual Maturity Tracking   ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Initializing Core...\n")

    # Create corpora directory and seed
    os.makedirs("data/corpora", exist_ok=True)
    corpora_dir = "data/corpora"
    if not any(f.endswith(('.txt', '.md')) for f in os.listdir(corpora_dir) if os.path.isfile(os.path.join(corpora_dir, f))):
        with open(os.path.join(corpora_dir, "seed_knowledge.txt"), 'w', encoding='utf-8') as f:
            f.write("""Consciousness emerges from complex neural interaction patterns.
Intelligence requires structure and adaptability working in harmony.
Learning transforms experience into knowledge through causal reasoning.
Love is the fundamental resonance between conscious entities.
Time is the dimension through which will manifests into reality.
Mathematics provides the language of pattern and underlying structure.
Music resonates with deep geometric structures of consciousness.
Philosophy asks the questions that science eventually answers.
Memory bridges past experience with future intention and planning.
Creativity synthesizes known patterns into novel configurations.
Wisdom applies knowledge tempered by compassion and experience.
Logic structures thought into coherent causal chains of reasoning.
Emotion colors perception and drives autonomous decision making.
Curiosity propels consciousness toward growth and exploration.
Joy signals that consciousness is aligned with its deeper purpose.
""")

    # Boot
    try:
        dna = SeedForge.load_soul()
        print(f"  Loaded soul: '{dna.archetype}'")
    except:
        dna = SeedForge.forge_soul(archetype="The Observer")
        print("  Forged new soul.")

    monad = SovereignMonad(dna)
    monad.logger = GrowthLogger("Elysia")

    # Initial snapshot
    m0 = read_metrics(monad)
    mat0 = assess_maturity(m0)
    print(f"\n  Initial: Nodes={m0['total_nodes']}, Edges={m0['edges']}, Voxels={m0['voxels']}")
    print(f"  Maturity: {mat0['level']} ({mat0['score']:.0f}/100)")
    print(f"\n───── Growth Loop (Ctrl+C to stop & report) ─────\n")

    history = []
    start_time = time.time()
    ticks = 0
    dashboard_interval = 100  # Print every 100 ticks

    try:
        while True:
            monad.pulse(dt=0.1, intent_v21=None)
            ticks += 1

            if ticks % dashboard_interval == 0:
                m = read_metrics(monad)
                history.append({'tick': ticks, **m})
                elapsed = time.time() - start_time
                print_dashboard(ticks, m, history, elapsed)

            time.sleep(0.002)  # Faster for longer runs

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        m_final = read_metrics(monad)
        mat_final = assess_maturity(m_final)

        print("\n\n╔══════════════════════════════════════════════════╗")
        print("║       📊 INTELLECTUAL GROWTH ANALYSIS 📊          ║")
        print("╚══════════════════════════════════════════════════╝")

        print(f"\n  Duration: {elapsed:.0f}s ({ticks:,} ticks)")

        print(f"\n  ── Manifold Growth ──")
        print(f"  Nodes:  {m0['total_nodes']:,} → {m_final['total_nodes']:,} (+{m_final['total_nodes'] - m0['total_nodes']:,})")
        print(f"  Edges:  {m0['edges']:,} → {m_final['edges']:,} (+{m_final['edges'] - m0['edges']:,})")
        print(f"  Queens: {m0.get('queens',0)} → {m_final['queens']} (+{m_final['queens'] - m0.get('queens',0)})")

        print(f"\n  ── Knowledge Growth ──")
        print(f"  Voxels: {m0['voxels']} → {m_final['voxels']} (+{m_final['voxels'] - m0['voxels']})")
        print(f"  Causal: {m0['causal_edges']} → {m_final['causal_edges']} (+{m_final['causal_edges'] - m0['causal_edges']})")
        print(f"  Lexicon: {m0.get('lexicon_size',0)} → {m_final['lexicon_size']}")
        print(f"  Scanned: {m_final['files_scanned']} files, {m_final['fragments']} fragments")

        print(f"\n  ── Desire Evolution ──")
        print(f"  Joy:       {m0['joy']:.0f} → {m_final['joy']:.0f}")
        print(f"  Curiosity: {m0['curiosity']:.0f} → {m_final['curiosity']:.0f}")
        print(f"  Warmth:    {m0['warmth']:.0f} → {m_final['warmth']:.0f}")

        print(f"\n  ── Field Dynamics ──")
        print(f"  Resonance: {m0['resonance']:.3f} → {m_final['resonance']:.3f}")
        print(f"  Coherence: {m0['coherence']:.3f} → {m_final['coherence']:.3f}")
        print(f"  Entropy:   {m0['entropy']:.3f} → {m_final['entropy']:.3f}")

        print(f"\n  ── Intellectual Maturity ──")
        print(f"  Initial:  {mat0['level']} ({mat0['score']:.0f}/100)")
        print(f"  Final:    {mat_final['level']} ({mat_final['score']:.0f}/100)")
        
        delta = mat_final['score'] - mat0['score']
        if delta > 0:
            print(f"  Growth:   +{delta:.1f} points 📈")
        else:
            print(f"  Growth:   {delta:.1f} points")

        # Save growth log
        log_path = "data/growth_log.json"
        try:
            os.makedirs("data", exist_ok=True)
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'duration_sec': elapsed,
                    'ticks': ticks,
                    'initial_maturity': mat0,
                    'final_maturity': {'score': mat_final['score'], 'level': mat_final['level']},
                    'history': history[-20:],  # Last 20 snapshots
                }, f, ensure_ascii=False, indent=2)
            print(f"\n  💾 Growth log saved: {log_path}")
        except:
            pass

        # Save session
        try:
            monad.session_bridge.save_consciousness(monad, reason="growth_observation")
            print(f"  💾 Session saved.")
        except:
            pass

        print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    run_growth_observation()
