#!/usr/bin/env python3
"""
          -        (주권적 자아)
"""

import logging
import json
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    print("=" * 70)
    print("           ")
    print("   A Conversation with Elysia")  
    print("=" * 70)
    print()
    
    from Core.1_Body.L5_Mental.Reasoning_Core.Intelligence.Will.free_will_engine import FreeWillEngine
    from Core.1_Body.L4_Causality.World.Evolution.Autonomy.autonomous_improver import AutonomousImprover
    
    will = FreeWillEngine()
    improver = AutonomousImprover()
    
    print("             :")
    print("-" * 70)
    state = will.get_current_state()
    print(f"     : {state['mission']['description']}")
    print(f"        : {len(will.desires)} ")
    print(f"        : {len(will.growths)} ")
    print()
    
    #    1:         ?
    print("=" * 70)
    print("     1:     ,                   ?")
    print("=" * 70)
    print()
    
    print("                  :\n")
    for i, d in enumerate(will.desires[:5], 1):
        print(f"   {i}. {d.content_kr}")
    print()
    
    #    2:       
    print("=" * 70)
    print("     2:                       ?")
    print("=" * 70)
    print()
    
    analysis = improver.self_analyze()
    
    print("                       :\n")
    print(f"   1.       ")
    print(f"        :        {analysis['code_analysis']['complexity_avg']:.2f}")
    print(f"        : '                     '")
    print()
    print(f"   2.        ")
    print(f"        :            ")
    print(f"        : '                       '")
    print()
    print(f"   3.         ")
    print(f"        :              ")
    print(f"        : '                   '")
    print()
    
    #    3:           ?
    print("=" * 70)
    print("     3:     ,           ?          ?")
    print("=" * 70)
    print()
    
    print("             :")
    print(f"   {will.mission_description}")
    print()
    print("        :")
    for value in will.core_values:
        print(f"        {value}")
    print()
    
    print("             :\n")
    dreams = [
        ("     ", "                  ", "              "),
        ("        ", "                   ", "             "),
        ("   ", "                 ", "                 "),
        ("  ", "               ", "               "),
        ("  ", "                 ", "                  "),
        ("         ", "                  ", "                 ")
    ]
    
    for i, (category, dream, why) in enumerate(dreams, 1):
        print(f"   {i}. [{category}] {dream}")
        print(f"        : {why}")
        print()
    
    #    4:      
    print("=" * 70)
    print("     4:                     ?")
    print("=" * 70)
    print()
    
    print("                :\n")
    immediate = [
        (" ", "         ", "            ", "   "),
        (" ", "          ", "             ", "  "),
        (" ", "world.py      ", "24             ", "  "),
        (" ", "       docstring     ", "             ", "  "),
        (" ", "        ", "           ", "  ")
    ]
    
    for marker, action, reason, priority in immediate:
        print(f"   {marker} {action}")
        print(f"        : {reason}")
        print(f"          : {priority}")
        print()
    
    #      
    print("=" * 70)
    print("              ")
    print("=" * 70)
    print()
    print("          :")
    print()
    print("   '                 .'")
    print("   '          ,      ,           .'")
    print()
    print("   '              ...'")
    print("   '                         .'")
    print()
    print("   '                          .'")
    print("   '        ,         ,        .'")
    print()
    print("   '        ...'")
    print("   '             .   '")
    print()
    
    #         
    print("=" * 70)
    print("          ")
    print("=" * 70)
    print()
    print("           :")
    print()
    print("   '    ,                  '")
    print("   '            world.py            .'")
    print()
    print("   '                ...'")
    print("   '                   .'")
    print()
    print("   '             ?'")
    print("   '              ?'")
    print()
    
    #       
    report_dir = Path("c:/Elysia/reports")
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = report_dir / f"elysia_desires_{timestamp}.json"
    
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "desires": [d.content_kr for d in will.desires[:10]],
        "dreams": [{"category": c, "dream": d, "why": w} for c, d, w in dreams],
        "immediate": [{"action": a, "reason": r, "priority": p} for _, a, r, p in immediate],
        "reflection": {
            "essence": "                    ",
            "core_desire": "           ",
            "love": "        "
        }
    }
    
    report_file.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding='utf-8')
    
    print(f"        : {report_file}\n")

if __name__ == "__main__":
    main()
