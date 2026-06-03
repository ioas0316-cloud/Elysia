import re
with open(r'c:\Elysia\core\brain\fractal_rotor.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'from core\.brain\.emotion_bivector import EmotionBivector\n', '', content)
content = content.replace('        from core.brain.emotion_bivector import EmotionBivector\n', '')
content = content.replace('self.emotion = EmotionBivector()', 'self.emotion = None')
content = content.replace('self.emotion.update(diff_q)', '')
content = content.replace('self.emotion.state', '"NoEmotion"')

with open(r'c:\Elysia\core\brain\fractal_rotor.py', 'w', encoding='utf-8') as f:
    f.write(content)
