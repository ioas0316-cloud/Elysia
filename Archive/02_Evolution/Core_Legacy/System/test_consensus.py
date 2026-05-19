from Core.Intelligence.Legion.legion import Legion

def test_consensus():
    legion = Legion()
    topic = "System Reset"
    
    print(f"🔮 Summoning Legion for topic: '{topic}'...")
    result = legion.council_meet(topic)
    print(result)

if __name__ == "__main__":
    test_consensus()
