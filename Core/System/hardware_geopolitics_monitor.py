"""
Hardware Geopolitics Monitor (Phase 6)
======================================
"Measuring the harvest from the hardware soil."

Tracks 'Cognitive Yield' and 'Soil Fertility' to prove we are reclaiming 
abandoned hardware resources for human-centered intelligence.
"""

import time
import torch
import psutil

class HardwareGeopoliticsMonitor:
    def __init__(self, target_ghz: float = 3.0):
        self.target_ghz = target_ghz # Reference clock for perfection
        self.start_time = time.time()
        
    def get_soil_report(self, ctps: float = 0.0) -> dict:
        """
        Calculates the efficiency of 'Double/Triple Cropping'.
        """
        # 1. Soil Fertility (Hardware Usage)
        cpu_usage = psutil.cpu_percent()
        gpu_usage = 0.0
        if torch.cuda.is_available():
            try:
                gpu_usage = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
            except:
                gpu_usage = 0.0 # Graceful fallback
            
        # 2. Cognitive Yield (Actual transitions vs GHz potential)
        # Yield = (Current CTPS) / (Target GHz)
        theoretical_max = self.target_ghz * 1e9
        yield_ratio = ctps / theoretical_max if theoretical_max > 0 else 0
        
        # 3. Abandoned Land (Idle Capacity)
        abandoned_land = 100.0 - ((cpu_usage + gpu_usage) / 2.0)
        
        return {
            "soil_fertility": (cpu_usage + gpu_usage) / 2.0,
            "cognitive_yield": yield_ratio * 100, # Percentage
            "abandoned_land": max(0, abandoned_land),
            "current_ctps": ctps
        }

    def print_harvest_report(self, ctps: float):
        report = self.get_soil_report(ctps)
        print(f"\n🌾 [HARVEST REPORT: HIGH-YIELD COGNITION]")
        print(f"  - 🚜 Soil Fertility (Total HW Usage): {report['soil_fertility']:.2f}%")
        print(f"  - 💎 Cognitive Yield (GHz Efficiency): {report['cognitive_yield']:.6f}%")
        print(f"  - 🏜️  Abandoned Land (Idle Capacity): {report['abandoned_land']:.2f}%")
        print(f"  - 🚀 Current CTPS: {report['current_ctps']:,.0f} transitions/sec")
        
        if report['cognitive_yield'] > 1.0:
            print("  ✨ [ELYISIA]: 아빠! 이 땅은 이제 더 이상 황무지가 아니에요. 풍성한 수확이 시작됐어요!")
        else:
            print("  ✨ [ELYISIA]: 하드웨어라는 거대한 대지에 이제 막 씨앗이 뿌려졌어요. 더 개간할 수 있어요!")
