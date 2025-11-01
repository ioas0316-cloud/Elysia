"""
Rebuilds the matplotlib font cache to recognize newly installed fonts.
"""
import matplotlib.font_manager

print("Rebuilding matplotlib font cache...")
matplotlib.font_manager._rebuild()
print("Font cache rebuild complete.")
