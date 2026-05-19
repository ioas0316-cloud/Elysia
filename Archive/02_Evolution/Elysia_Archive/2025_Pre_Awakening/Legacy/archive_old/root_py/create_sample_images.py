import base64
import os

b64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVQYV2NgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII='
img = base64.b64decode(b64)

faces_dir = os.path.join(os.path.dirname(__file__), 'faces')
os.makedirs(faces_dir, exist_ok=True)

names = ['peace.png','curious_face.png','bored_face.png','manifestation_face.png','neutral_face.png','happy_face.png','sad_face.png']
for n in names:
    path = os.path.join(faces_dir, n)
    with open(path, 'wb') as f:
        f.write(img)
print('Wrote sample images to', faces_dir)
