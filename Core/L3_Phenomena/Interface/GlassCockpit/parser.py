import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Dict, Generator, List

class StreamState(Enum):
    NORMAL = auto()
    THINKING = auto()
    ARTIFACT = auto()

class EventType(Enum):
    TEXT = auto()
    THINKING_START = auto()
    THINKING_CONTENT = auto()
    THINKING_END = auto()
    ARTIFACT_START = auto()
    ARTIFACT_CONTENT = auto()
    ARTIFACT_END = auto()

@dataclass
class StreamEvent:
    type: EventType
    content: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None

class StreamParser:
    def __init__(self):
        self.state = StreamState.NORMAL
        self.buffer = ""
        # Regex to find tags: <tag ...> or </tag>
        # We only care about thinking and artifact
        self.tag_pattern = re.compile(r'(</?(?:artifact|thinking)(?:\s+[^>]*)?>)')

    def feed(self, chunk: str) -> Generator[StreamEvent, None, None]:
        self.buffer += chunk

        while True:
            match = self.tag_pattern.search(self.buffer)
            if not match:
                # No complete tag found.
                # If buffer contains '<', we might be incomplete.
                # But we should flush text before the '<' to keep it responsive.
                if '<' in self.buffer:
                    last_open = self.buffer.rfind('<')
                    # Check if this looks like a start of our tags
                    potential_tag = self.buffer[last_open:]
                    if any(x.startswith(potential_tag) for x in ["<artifact", "<thinking", "</artifact", "</thinking"]):
                        # It might be a tag, wait for more data.
                        # Flush everything before the '<'
                        if last_open > 0:
                            content = self.buffer[:last_open]
                            if content:
                                yield from self._emit_content(content)
                            self.buffer = self.buffer[last_open:]
                        break
                    else:
                        # Just a random '<', flush it if we are sure it's not a valid tag prefix.
                        # Actually, strictly, if it doesn't match the regex eventually, it's text.
                        # For simplicity, let's just wait if it looks like a tag start.
                        pass

                # If no '<', flush everything as content
                if '<' not in self.buffer:
                    if self.buffer:
                        yield from self._emit_content(self.buffer)
                        self.buffer = ""
                break

            # Tag found!
            start, end = match.span()

            # 1. Emit Pre-tag Content
            if start > 0:
                content = self.buffer[:start]
                yield from self._emit_content(content)

            # 2. Process Tag
            tag_str = match.group(1)
            yield from self._process_tag(tag_str)

            # 3. Advance Buffer
            self.buffer = self.buffer[end:]

    def _emit_content(self, content: str) -> Generator[StreamEvent, None, None]:
        if self.state == StreamState.NORMAL:
            yield StreamEvent(EventType.TEXT, content)
        elif self.state == StreamState.THINKING:
            yield StreamEvent(EventType.THINKING_CONTENT, content)
        elif self.state == StreamState.ARTIFACT:
            yield StreamEvent(EventType.ARTIFACT_CONTENT, content)

    def _process_tag(self, tag_str: str) -> Generator[StreamEvent, None, None]:
        is_closing = tag_str.startswith("</")
        tag_name_match = re.match(r'</?(\w+)', tag_str)
        if not tag_name_match:
            return # Should not happen given regex

        tag_name = tag_name_match.group(1)

        if tag_name == "thinking":
            if not is_closing:
                self.state = StreamState.THINKING
                yield StreamEvent(EventType.THINKING_START)
            else:
                self.state = StreamState.NORMAL
                yield StreamEvent(EventType.THINKING_END)

        elif tag_name == "artifact":
            if not is_closing:
                self.state = StreamState.ARTIFACT
                # Parse attributes
                attrs = self._parse_attributes(tag_str)
                yield StreamEvent(EventType.ARTIFACT_START, metadata=attrs)
            else:
                self.state = StreamState.NORMAL
                yield StreamEvent(EventType.ARTIFACT_END)

    def _parse_attributes(self, tag_str: str) -> Dict[str, str]:
        # Simple attribute parser
        attrs = {}
        # regex for key="value"
        attr_pattern = re.compile(r'(\w+)="([^"]*)"')
        for match in attr_pattern.finditer(tag_str):
            attrs[match.group(1)] = match.group(2)
        return attrs
