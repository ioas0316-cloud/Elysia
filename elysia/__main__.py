"""
Elysia Unified Entry Point (통합 진입점)
========================================

Elysia를 실행하는 단일 진입점입니다.

Usage:
    python -m elysia           # 기본 모드 (대화형 대기)
    python -m elysia daemon    # 백그라운드 꿈꾸기 모드
    python -m elysia analyze   # 자기 분석 모드
    python -m elysia status    # 시스템 상태 확인
    python -m elysia wave      # 파동 품질 검사

Examples:
    python -m elysia daemon --hud      # HUD와 함께 데몬 실행
    python -m elysia analyze Core/     # Core 폴더 분석
    python -m elysia status --verbose  # 상세 상태
"""

import sys
import os
import argparse
import signal
import time
import logging

# UTF-8 강제
if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# 경로 설정
ELYSIA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ELYSIA_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Elysia")


def cmd_daemon(args):
    """백그라운드 꿈꾸기 모드 (기존 wake_elysia.py 기능)"""
    print("🌅 Elysia: Awakening Protocol Initiated...")
    print("=" * 50)
    print("   [Mode: Perpetual Dreaming]")
    print("   [Press Ctrl+C to Sleep]")
    print()
    
    # 핵심 모듈 로드
    from Core.Autonomy.dream_daemon import get_dream_daemon
    from Core.Foundation.torch_graph import get_torch_graph
    from Core.Interface.world_exporter import get_world_exporter
    from Core.Elysia.elysia_core import ElysiaCore
    
    daemon = get_dream_daemon()
    graph = get_torch_graph()
    exporter = get_world_exporter()
    core = ElysiaCore()
    
    # GlobalHub에 핵심 모듈 등록
    _register_core_modules()
    
    # 브레인 로드
    loaded = graph.load_state()
    if not loaded and graph.pos_tensor.shape[0] < 5:
        print("   🔍 Brain is empty. Detecting Legacy Knowledge...")
        from Core.Foundation.knowledge_migrator import get_migrator
        migrator = get_migrator()
        migrator.migrate()
    
    daemon.is_dreaming = True
    
    # HUD
    if args.hud:
        from Core.Interface.console_hud import get_console_hud
        hud = get_console_hud(graph)
    else:
        hud = None
    
    cycle_count = 0
    try:
        while True:
            current_action = "Dreaming"
            
            # 꿈꾸기 사이클
            if graph.pos_tensor.shape[0] < 5:
                daemon._seed_reality()
            
            if hasattr(daemon, '_ingest_knowledge') and cycle_count % 5 == 0:
                current_action = "Ingesting Knowledge"
                daemon._ingest_knowledge()
            
            if hasattr(daemon, '_contemplate_essence') and cycle_count % 10 == 0:
                current_action = "Distilling Principles"
                daemon._contemplate_essence()
            
            # 파동 코딩 (자기 리팩토링)
            if cycle_count % 30 == 0:
                current_action = "Refactoring Self"
                from Core.Autonomy.wave_coder import get_wave_coder
                get_wave_coder().transmute()
            
            daemon._weave_serendipity()
            graph.apply_gravity(iterations=10)
            
            # 내보내기
            if cycle_count % 5 == 0:
                exporter.export_world()
            
            # 저장
            if cycle_count % 60 == 0 and cycle_count > 0:
                graph.save_state()
                current_action = "Saving Memory"
                if core.universe:
                    core.universe.decay_resonance(half_life=3600.0)
            
            if hud:
                hud.render(current_action)
            
            cycle_count += 1
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n\n💤 Elysia: Entering Hibernation.")
        graph.save_state()
        print("   ✅ Brain State Saved.")
        print("   Good night.")


def cmd_status(args):
    """시스템 상태 확인"""
    print("📊 Elysia System Status")
    print("=" * 50)
    
    # GlobalHub 상태
    try:
        from Core.Ether.global_hub import get_global_hub
        hub = get_global_hub()
        hub.load_state()
        status = hub.get_hub_status()
        
        print(f"\n🌐 GlobalHub:")
        print(f"   Modules: {status['total_modules']}")
        print(f"   Subscriptions: {status['total_subscriptions']}")
        print(f"   Relations: {status['total_relations']}")
        
        if status['modules']:
            print(f"\n   Registered Modules:")
            for mod in status['modules']:
                print(f"      • {mod}")
        
        if args.verbose and status.get('strongest_bonds'):
            print(f"\n   Strongest Bonds:")
            for bond in status['strongest_bonds'][:5]:
                print(f"      {bond['from']} → {bond['to']}: {bond['weight']:.2f}")
    except Exception as e:
        print(f"   ⚠️ GlobalHub error: {e}")
    
    # 파동 시스템 상태
    try:
        from Core.Wave import get_system_status
        wave_status = get_system_status()
        print(f"\n🌊 Wave System:")
        for key, value in wave_status.items():
            icon = "✅" if value else "❌"
            print(f"   {icon} {key}: {value}")
    except Exception as e:
        print(f"   ⚠️ Wave system error: {e}")
    
    # TorchGraph 상태
    try:
        from Core.Foundation.torch_graph import get_torch_graph
        graph = get_torch_graph()
        graph.load_state()
        print(f"\n🧠 Brain (TorchGraph):")
        print(f"   Nodes: {graph.pos_tensor.shape[0]}")
    except Exception as e:
        print(f"   ⚠️ TorchGraph error: {e}")
    
    print()


def cmd_analyze(args):
    """자기 분석 모드"""
    target = args.target or "Core/"
    print(f"🔍 Analyzing: {target}")
    print("=" * 50)
    
    from Core.Wave import scan_quality
    
    if scan_quality:
        report = scan_quality(target)
        if report:
            print(report.to_markdown())
    else:
        print("⚠️ Quality scanner not available")


def cmd_wave(args):
    """파동 품질 검사"""
    target = args.target or "Core/"
    print(f"🌊 Wave Quality Check: {target}")
    print("=" * 50)
    
    from Core.Wave.quality_guard import WaveQualityGuard
    
    guard = WaveQualityGuard()
    report = guard.scan_directory(target)
    
    # Tension 경보 추가
    tension_alerts = guard.get_tension_alerts()
    report.issues.extend(tension_alerts)
    
    print(report.to_markdown())
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report.to_markdown())
        print(f"\n📄 Report saved to {args.output}")


def _register_core_modules():
    """핵심 모듈을 GlobalHub에 등록"""
    try:
        from Core.Ether.global_hub import get_global_hub
        hub = get_global_hub()
        
        # 핵심 모듈 등록
        modules = [
            ("ReasoningEngine", "Core/Intelligence/reasoning_engine.py", 
             ["decision", "ethics", "planning"], "The Soul"),
            ("CognitiveHub", "Core/Cognition/cognitive_hub.py",
             ["understanding", "analysis"], "The Mind"),
            ("WaveCodingSystem", "Core/Intelligence/wave_coding_system.py",
             ["code_analysis", "dna"], "The Wave Analyzer"),
            ("NervousSystem", "Core/Interface/nervous_system.py",
             ["input", "output", "stream"], "The Interface"),
            ("TorchGraph", "Core/Foundation/torch_graph.py",
             ["memory", "association", "graph"], "The Brain"),
        ]
        
        for name, path, caps, desc in modules:
            hub.register_module(name, path, caps, desc)
        
        hub.save_state()
        logger.info(f"✅ Registered {len(modules)} core modules to GlobalHub")
        
    except Exception as e:
        logger.warning(f"⚠️ Could not register modules: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Elysia - Sovereign Crystalline Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m elysia daemon --hud    # Run with HUD
  python -m elysia status          # Check system status
  python -m elysia analyze Core/   # Analyze Core folder
  python -m elysia wave Core/      # Wave quality check
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # daemon
    daemon_parser = subparsers.add_parser('daemon', help='Run dream daemon')
    daemon_parser.add_argument('--hud', action='store_true', help='Show HUD')
    
    # status
    status_parser = subparsers.add_parser('status', help='Show system status')
    status_parser.add_argument('--verbose', '-v', action='store_true')
    
    # analyze
    analyze_parser = subparsers.add_parser('analyze', help='Analyze codebase')
    analyze_parser.add_argument('target', nargs='?', default='Core/')
    
    # wave
    wave_parser = subparsers.add_parser('wave', help='Wave quality check')
    wave_parser.add_argument('target', nargs='?', default='Core/')
    wave_parser.add_argument('--output', '-o', help='Output file')
    
    args = parser.parse_args()
    
    if args.command == 'daemon':
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        cmd_daemon(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'wave':
        cmd_wave(args)
    else:
        # 기본: 상태 출력
        parser.print_help()
        print("\n" + "=" * 50)
        cmd_status(argparse.Namespace(verbose=False))


if __name__ == "__main__":
    main()
