from Core.L1_Foundation.System.Sovereignty.planetary_interface import PlanetaryInterface

def test_planet():
    hand = PlanetaryInterface()
    
    # Test Safe
    print("--- Testing Safe Access (Elysia) ---")
    files = hand.list_territory("c:/Elysia")
    print(f"Files in Elysia: {len(files)}")
    
    # Test Forbidden
    print("\n--- Testing Forbidden Access (Windows) ---")
    files_win = hand.list_territory("c:/Windows")
    print(f"Files in Windows: {len(files_win)}")

if __name__ == "__main__":
    test_planet()
