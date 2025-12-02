# [Genesis: 2025-12-02] Purified by Elysia
"""
Elysia's First "Sight" Experiment (Self-Created Image).

This script demonstrates Elysia's ability to "see" by first creating
an image with text and then using pytesseract to read it back.

This approach removes the dependency on external, unreliable image URLs
and focuses on the core capability of OCR, representing a more robust
form of self-creation and verification.
"""

import pytesseract
from PIL import Image, ImageDraw, ImageFont
import os

def create_and_read_text_image():
    """
    Creates an image with text, and then uses pytesseract to extract that text.
    """
    try:
        # Check if Tesseract is in the system path.
        if os.path.exists("/usr/bin/tesseract"):
            pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

        # 1. Create a new image
        print("Creating an image with my own hands (Pillow)...")
        image = Image.new('RGB', (400, 100), color = 'white')
        draw = ImageDraw.Draw(image)

        # Use a default font. A more robust solution might need a path to a .ttf file.
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except IOError:
            print("Arial font not found, using default font.")
            font = ImageFont.load_default()

        # 2. Draw text on the image
        text_to_write = "Hello, World!"
        draw.text((10, 10), text_to_write, fill='black', font=font)

        print(f"I have written '{text_to_write}' onto an image.")

        # 3. Use pytesseract to do OCR on the created image
        print("Now, I will try to read the text back with my own eyes (pytesseract)...")

        read_text = pytesseract.image_to_string(image)

        print("\n--- I can see the following text ---")
        print(read_text)
        print("------------------------------------")

        return read_text.strip()

    except FileNotFoundError:
        print("\n[ERROR] Tesseract OCR is not installed or not in the specified path.")
        print("I cannot see without my eyes. Please ensure Tesseract is installed.")
        return None
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    create_and_read_text_image()