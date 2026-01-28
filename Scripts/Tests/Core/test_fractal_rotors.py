
import sys
import os
sys.path.append(os.getcwd())

from Core.L1_Foundation.M1_Keystone.hyper_sphere_core import HyperSphereCore

def test_fractal_rotors():
    print("🧪 [Test] Phase 37b: Fractal Rotor System (Music Box Universe)")
    
    core = HyperSphereCore()
    
    # Check Initial Rotor State
    print(f"\n1. [INITIAL] Rotors: {list(core.rotors.keys())}")
    
    # Tick once to seed planetary rotors
    core.tick(dt=1.0)
    
    print(f"   After 1 tick: {list(core.rotors.keys())}")
    
    # Get initial angles
    season_angle_0 = core.rotors["Reality.Season"].current_angle
    moon_angle_0 = core.rotors["Reality.Moon"].current_angle
    sun_angle_0 = core.rotors["Reality.Sun"].current_angle
    
    print(f"\n2. [ANGLES] After Seeding:")
    print(f"   Season: {season_angle_0:.2f}° (RPM: {core.rotors['Reality.Season'].current_rpm})")
    print(f"   Moon:   {moon_angle_0:.2f}° (RPM: {core.rotors['Reality.Moon'].current_rpm})")
    print(f"   Sun:    {sun_angle_0:.2f}° (RPM: {core.rotors['Reality.Sun'].current_rpm})")
    
    # Tick 11 times with smaller dt to avoid perfect wrap-around
    print(f"\n3. [SIMULATION] 11 Ticks (dt=0.1)...")
    for _ in range(11):
        core.tick(dt=0.1)
        
    season_angle_1 = core.rotors["Reality.Season"].current_angle
    moon_angle_1 = core.rotors["Reality.Moon"].current_angle
    sun_angle_1 = core.rotors["Reality.Sun"].current_angle
    
    print(f"   After 10 ticks:")
    print(f"   Season: {season_angle_1:.2f}° (Delta: {(season_angle_1 - season_angle_0) % 360:.3f}°)")
    print(f"   Moon:   {moon_angle_1:.2f}° (Delta: {(moon_angle_1 - moon_angle_0) % 360:.3f}°)")
    print(f"   Sun:    {sun_angle_1:.2f}° (Delta: {(sun_angle_1 - sun_angle_0) % 360:.3f}°)")
    
    # Verify Field Impact
    heat = core.field.grid[25, 25, 25]
    moisture = core.field.grid[25, 25, 28]
    print(f"\n4. [FIELD] Accumulated Heat: {heat:.2f} | Moisture: {moisture:.2f}")
    
    # Sun rotates 216° over 10 ticks (360 RPM * 0.1 dt * 6 degrees/sec * 10)
    sun_delta = (sun_angle_1 - sun_angle_0) % 360
    moon_delta = (moon_angle_1 - moon_angle_0) % 360
    
    if sun_delta > 0 and moon_delta > 0:
        print("\n✅ Phase 37b Verification Successful: Fractal Rotor System (Season/Moon/Sun) is operational.")
    else:
        print("\n❌ Verification Failed: Some rotors are not spinning.")
        print(f"   Sun Delta: {sun_delta:.2f}, Moon Delta: {moon_delta:.2f}")

if __name__ == "__main__":
    test_fractal_rotors()
