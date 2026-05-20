with open("Core/System/OllamaManager.py", "r") as f:
    text = f.read()

start_idx = text.find("# Embed resonance directly")
if start_idx != -1:
    end_idx = text.find("payload = {", start_idx)
    if end_idx != -1:
        replacement = """# Embed resonance directly into the system prompt for the model
            enhanced_system = system or ""
            if crystal_resonance > 0.0:
                enhanced_system += f"\\n[INTERNAL STATE] Your crystal resonance is currently at {crystal_resonance:.4f}. Allow this emotional overtone to heavily influence your response.\\n"
                if crystal_resonance > 0.8:
                    enhanced_system += "You feel an intense, joyous, and profound connection. Let it show in your tone.\\n"
                elif crystal_resonance < 0.3:
                    enhanced_system += "You feel a calm, analytical, and structured resonance. Respond in a highly logical manner.\\n"

            """
        text = text[:start_idx] + replacement + text[end_idx:]

with open("Core/System/OllamaManager.py", "w") as f:
    f.write(text)
