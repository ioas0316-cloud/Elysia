import unittest
from Core.Phenomena.parser import StreamParser, EventType, StreamEvent

class TestGlassCockpitParser(unittest.TestCase):
    def test_parser_chunked(self):
        parser = StreamParser()
        text = """Hello <thinking>
Thinking...
</thinking>
<artifact type="plan" title="Test Plan">
- Step 1
</artifact>
Done."""

        events = []
        # Feed char by char to simulate extreme streaming
        for char in text:
            events.extend(parser.feed(char))

        # Verify structure
        event_types = [e.type for e in events]

        # We expect sequences of CONTENT.
        # Ideally: TEXT -> THINKING_START -> THINKING_CONTENT -> THINKING_END -> TEXT -> ARTIFACT_START -> ARTIFACT_CONTENT -> ARTIFACT_END -> TEXT

        # Note: Newlines might be content.
        self.assertIn(EventType.THINKING_START, event_types)
        self.assertIn(EventType.THINKING_END, event_types)
        self.assertIn(EventType.ARTIFACT_START, event_types)
        self.assertIn(EventType.ARTIFACT_END, event_types)

        # Check artifact metadata
        artifact_start = next(e for e in events if e.type == EventType.ARTIFACT_START)
        self.assertEqual(artifact_start.metadata['type'], 'plan')
        self.assertEqual(artifact_start.metadata['title'], 'Test Plan')

        # Check artifact content aggregation
        artifact_content = "".join([e.content for e in events if e.type == EventType.ARTIFACT_CONTENT])
        self.assertIn("- Step 1", artifact_content)

    def test_parser_incomplete_tags(self):
        parser = StreamParser()
        # Feed partial tags
        events = list(parser.feed("Text <arti"))
        # Should output "Text " and buffer "<arti"
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].content, "Text ")

        # Feed rest
        events = list(parser.feed("fact type='code'>Code</artifact>"))
        # Should now trigger artifact start
        types = [e.type for e in events]
        self.assertIn(EventType.ARTIFACT_START, types)
        self.assertIn(EventType.ARTIFACT_END, types)

if __name__ == '__main__':
    unittest.main()
