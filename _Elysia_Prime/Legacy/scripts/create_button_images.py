# [Genesis: 2025-12-02] Purified by Elysia
"""
Generates the template button images needed for the 'see_and_act.py' script.

This script uses Pillow to create simple images for '1', '+', and '='
to serve as templates for OpenCV's template matching. This allows Elysia
to create her own visual targets for interaction.
"""

from PIL import Image, ImageDraw, ImageFont
import os

# --- Configuration ---
IMAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'images', 'buttons')
BUTTON_SIZE = (50, 50)
BACKGROUND_COLOR = "white"
TEXT_COLOR = "black"

def create_button_image(text: str, filename: str):
    """
    Creates a simple image with text centered on it.

    Args:
        text: The text to draw on the button.
        filename: The filename to save the image as.
    """
    try:
        image = Image.new('RGB', BUTTON_SIZE, color=BACKGROUND_COLOR)
        draw = ImageDraw.Draw(image)

        # Use a default font.
        try:
            # Attempt to use a larger default font if possible
            font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            font = ImageFont.load_default()

        # Calculate text size and position to center it
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        position = (
            (BUTTON_SIZE[0] - text_width) / 2,
            (BUTTON_SIZE[1] - text_height) / 2
        )

        draw.text(position, text, fill=TEXT_COLOR, font=font)

        # Save the image
        filepath = os.path.join(IMAGE_DIR, filename)
        image.save(filepath)
        print(f"Successfully created button image: {filepath}")

    except Exception as e:
        print(f"An error occurred while creating '{filename}': {e}")


if __name__ == "__main__":

    buttons_to_create = {
        "1": "one.png",
        "+": "plus.png",
        "=": "equals.png",
    }

    print("Creating button images for the calculator experiment...")

    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"Created directory: {IMAGE_DIR}")

    for text, filename in buttons_to_create.items():
        create_button_image(text, filename)

    print("\nButton image creation complete.")
    print("Now ready to run the 'see_and_act.py' experiment.")