import os
import re
import sys
import unittest
from pathlib import Path
import shutil
import tempfile

class TestWebtoonWeaverIsolated(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for outputs to avoid polluting the repo
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.test_dir.name) / "outputs" / "comic"

        # Load the class dynamically
        self.WebtoonWeaver = self._load_class_from_source()

    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

    def _load_class_from_source(self):
        # Read the source file
        source_path = Path("Core/Creativity/webtoon_weaver.py")
        with open(source_path, "r", encoding="utf-8") as f:
            source_code = f.read()

        # Remove imports to avoid dependency errors
        lines = source_code.split('\n')
        modified_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                modified_lines.append("# " + line)
            elif "logging." in line:
                 modified_lines.append(line)
            else:
                modified_lines.append(line)

        modified_code = "\n".join(modified_lines)

        # Prepare namespace with mocks
        class MockLogger:
            def info(self, msg): pass
            def warning(self, msg): pass
            def error(self, msg): pass

        mock_logging = type('logging', (), {'getLogger': lambda x: MockLogger(), 'basicConfig': lambda **k: None, 'INFO': 10})

        namespace = {
            'logging': mock_logging,
            'Path': Path,
            'time': type('time', (), {'time': lambda: 1234567890, 'sleep': lambda x: None}),
            'random': type('random', (), {'uniform': lambda a,b: a, 'randint': lambda a,b: a}),
            'HAS_COMFY': False
        }

        try:
            exec(modified_code, namespace)
        except Exception as e:
            print(f"Error executing source: {e}")
            raise e

        return namespace['WebtoonWeaver']

    def test_publish_html_behavior(self):
        print("\nTesting _publish_html behavior via isolated class...")

        # Instantiate
        weaver = self.WebtoonWeaver.__new__(self.WebtoonWeaver)
        weaver.comfy = None

        # Data
        title = "Isolated Test Story"
        panel_map = {1: ("test_img.png", "Hello World")}
        episode_num = 888

        # Override the path logic inside _publish_html implies we might need to change where it writes?
        # The code hardcodes `Path("outputs/comic")`.
        # To make it write to our temp dir, we need to mock Path inside the namespace or change directory.
        # Changing directory is safer for this isolated test.

        original_cwd = os.getcwd()
        os.chdir(self.test_dir.name)

        try:
            # Call method
            weaver._publish_html(title, panel_map, episode_num)

            # Assertions
            latest_path = Path("outputs/comic/latest_episode.html")

            self.assertTrue(latest_path.exists(), "latest_episode.html should be created")

            with open(latest_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check current behavior (Redirect)
            target_filename = f"episode_{episode_num:03d}.html"

            if '<meta http-equiv="refresh"' in content and f'url={target_filename}' in content:
                print("[STATE: REDIRECT] Redirect mechanism is ACTIVE.")
            else:
                print("[STATE: COPY] Content is copied (Standard/Old behavior).")
                self.fail("Expected Redirect mechanism, but found Copy behavior.")

        finally:
            os.chdir(original_cwd)

if __name__ == "__main__":
    unittest.main()
