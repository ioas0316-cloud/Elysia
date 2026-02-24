import os
import re

def fix_imports():
    root_dirs = ["c:/Elysia/Core", "c:/Elysia/scripts"]
    
    # Priority replacements (specific paths first)
    replacements = [
        # Education remapping
        (r'from Core\.Education', 'from Core.Cognition'),
        
        # Soul remapping
        (r'from Core\.Soul', 'from Core.Cognition'),
        
        # Evolution remapping
        (r'from Core\.Evolution', 'from Core.Cognition'),
        
        # Autonomy remapping
        (r'from Core\.Autonomy', 'from Core.Cognition'),
        
        # Intelligence remapping
        (r'from Core\.Knowledge', 'from Core.Cognition'),
        (r'from Core\.Memory', 'from Core.Cognition'),
        
        # Physiology remapping
        (r'from Core\.Physics', 'from Core.Physiology.Physics'),
        (r'from Core\.Sensory', 'from Core.Physiology.Sensory'),
        (r'from Core\.Expression', 'from Core.Physiology.Expression'),
        (r'from Core\.Values', 'from Core.Physiology.Values'),
        
        # Governance remapping
        (r'from Core\.Orchestra', 'from Core.System.Orchestra'),
        (r'from Core\.Orchestration', 'from Core.System'),
        (r'from Core\.Security', 'from Core.System'),
        (r'from Core\.Interface', 'from Core.System'),
        (r'from Core\.Interaction', 'from Core.System'),
        (r'from Core\.System', 'from Core.System'),
        
        # Creativity (often under Evolution or World)
        (r'from Core\.Creativity', 'from Core.Cognition'),
        
        # Foundation Standardizations
        (r'from Core\.Foundation\.Math\.wave_tensor', 'from Core.Keystone.wave_tensor'),
        (r'from Core\.Foundation\.light_spectrum', 'from Core.Keystone.light_spectrum'), # Moving this too if needed
        (r'from Core\.Foundation\.Math\.quaternion_engine', 'from Core.System.hyper_quaternion'),
        
        # Relative to Absolute (for Core)
        (r'from \.wave_tensor', 'from Core.Keystone.wave_tensor'),
        (r'from Wave\.wave_tensor', 'from Core.Keystone.wave_tensor')
    ]

    for r_dir in root_dirs:
        for root, dirs, files in os.walk(r_dir):
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        new_content = content
                        for pattern, replacement in replacements:
                            new_content = re.sub(pattern, replacement, new_content)
                        
                        if new_content != content:
                            print(f"Updating: {path}")
                            with open(path, "w", encoding="utf-8") as f:
                                f.write(new_content)
                    except Exception as e:
                        print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    fix_imports()
