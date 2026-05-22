"""
[RPG STAT BRIDGE - PHYSICAL STAT CONVERTER]
"Translating classic RPG stats (STR, AGI, CON, INT, WIS) into VariableRotor parameters."
"""

import numpy as np
import math

class RPGStatBridge:
    def __init__(self, str_val: int = 10, agi_val: int = 10, con_val: int = 10, int_val: int = 10, wis_val: int = 10):
        # 1. Classic RPG Stats (Base value of 10 represents standard human)
        self.stats = {
            "STR": str_val,  # Strength (힘)
            "AGI": agi_val,  # Agility (민첩)
            "CON": con_val,  # Constitution (체력)
            "INT": int_val,  # Intelligence (지능)
            "WIS": wis_val   # Wisdom (지혜)
        }

    def convert_to_physical_params(self, dims: int) -> dict:
        """
        Converts the 5 stats into physical parameters M, D, K for VariableRotor.
        """
        # CON (체력) -> Mass M: Higher CON means heavier inertia (resists change)
        # M ranges from 0.5 (CON=1) to 3.0 (CON=50)
        mass_factor = 0.5 + (self.stats["CON"] * 0.05)
        M = np.ones(dims) * mass_factor

        # WIS (지혜) -> Damping D: Higher WIS means faster dissipation of cognitive chaos
        # D ranges from 0.05 (WIS=1) to 0.8 (WIS=50)
        damping_factor = 0.05 + (self.stats["WIS"] * 0.015)
        D = np.ones(dims) * damping_factor

        # INT (지능) -> Stiffness K: Higher INT means tighter logic restoration
        # K ranges from 0.5 (INT=1) to 3.0 (INT=50)
        stiffness_factor = 0.5 + (self.stats["INT"] * 0.05)
        K = np.ones(dims) * stiffness_factor

        # AGI (민첩) -> Max angular velocity / responsive scaling (used as dt multiplier or frequency filter)
        speed_factor = 1.0 + (self.stats["AGI"] * 0.05)

        # STR (힘) -> External force multiplier
        force_mult = 1.0 + (self.stats["STR"] * 0.08)

        return {
            "M": M,
            "D": D,
            "K": K,
            "speed_factor": speed_factor,
            "force_multiplier": force_mult
        }

def simulate_stat_profiles():
    print("==================================================================")
    print("       🛡️ RPG 5대 스탯 기반 로터 물리 거동 시뮬레이션 비교       ")
    print("==================================================================")
    
    # Profile A: 거인 전사 (Giant Warrior) - High CON & STR, Low WIS & INT
    # 묵직하고 강하지만 머리는 차분하게 정돈되지 못하고 흥분이 느리게 식음.
    warrior_stats = RPGStatBridge(str_val=18, agi_val=10, con_val=20, int_val=6, wis_val=6)
    
    # Profile B: 엘프 마법사 (Elf Mage) - Low CON & STR, High WIS & INT
    # 몸은 나약하여 쉽게 밀리지만 정밀하며 머릿속 평정을 극도로 빠르게 회복함.
    mage_stats = RPGStatBridge(str_val=6, agi_val=12, con_val=6, int_val=18, wis_val=18)

    dims = 3
    params_warrior = warrior_stats.convert_to_physical_params(dims)
    params_mage = mage_stats.convert_to_physical_params(dims)

    print(f"\n📊 [거인 전사 스탯] STR:18, CON:20 (체력) | 질량 M: {params_warrior['M'][0]:.2f} | 댐핑 D: {params_warrior['D'][0]:.3f} | 강성 K: {params_warrior['K'][0]:.2f}")
    print(f"📊 [엘프 마법사 스탯] WIS:18, INT:18 (지혜) | 질량 M: {params_mage['M'][0]:.2f} | 댐핑 D: {params_mage['D'][0]:.3f} | 강성 K: {params_mage['K'][0]:.2f}")

    # Simulation Setup
    # Apply a sudden magical impact / shock force (F_shock = 10.0) at step 0
    # and watch how they recover over steps.
    
    def run_sim(profile_name, params):
        x = np.zeros(dims) # positions (angles)
        v = np.zeros(dims) # velocities
        dt = 0.05 * params["speed_factor"] # AGI modulates responsiveness speed
        
        # Apply initial shock to velocity
        v[0] = 8.0 / params["M"][0]  # Shock pushes velocity, heavier mass feels it less at start
        
        print(f"\n🎬 [{profile_name}] 외력 충격 인입 시뮬레이션:")
        for step in range(6):
            # M*a + D*v + K*x = 0 (No continuous force, just free vibration recovery)
            a = (-params["D"] * v - params["K"] * x) / params["M"]
            v += a * dt
            x += v * dt
            
            # Estimate cognitive instability (enstrophy)
            instability = float(np.sum(v**2))
            
            # Map instability to mood text
            if instability < 0.1: mood = "🧘 [평온] 항상성 안착"
            elif instability < 2.0: mood = "🟡 [동요] 평정 회복 중"
            else: mood = "🔥 [폭주] 혼란/흥분 상태"
            
            print(f"  Step {step+1} | 위상 오프셋: {x[0]:.3f} rad | 카오스 강도: {instability:.3f} | {mood}")

    run_sim("거인 전사 (Giant Warrior)", params_warrior)
    run_sim("엘프 마법사 (Elf Mage)", params_mage)
    print("==================================================================")

if __name__ == "__main__":
    simulate_stat_profiles()
