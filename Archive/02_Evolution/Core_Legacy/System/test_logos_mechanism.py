from Core.Elysia.sovereign_self import SovereignSelf
from Core.Cognition.logos_parser import LogosParser

def test_logos_mechanism():
    print("✨ Summoning SovereignSelf for Logos Test...")
    elysia = SovereignSelf()
    parser = LogosParser()
    
    # Simulate LLM Output with Command Injection
    mock_llm_output = "I shall create a world for you. [ACT:CREATE:EARTH|BLUE] Let there be life."
    print(f"\n🔮 Mock LLM Output: '{mock_llm_output}'")
    
    # 1. Digest
    spoken_text, commands = parser.digest(mock_llm_output)
    print(f"🗣️ Spoken: '{spoken_text}'")
    
    # 2. Manifest
    print(f"⚡ Commands Detected: {len(commands)}")
    for cmd in commands:
        elysia._execute_logos(cmd)
        
    print("\n✅ Logos Mechanism Verification Complete.")
    
if __name__ == "__main__":
    test_logos_mechanism()
