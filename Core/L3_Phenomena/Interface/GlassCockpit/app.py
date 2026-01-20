from textual.app import App, ComposeResult
from textual.containers import Grid, Vertical, Horizontal, VerticalScroll
from textual.widgets import Header, Footer, Input, Static, TabbedContent, TabPane, Markdown, Label
from textual.worker import Worker
from textual import work
from textual.reactive import reactive
from rich.syntax import Syntax
import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from Core.L6_Structure.Merkaba.merkaba import Merkaba
from Core.L3_Phenomena.Interface.GlassCockpit.parser import StreamParser, EventType, StreamEvent, StreamState

class ChatBubble(Static):
    """A widget to display a single chat message."""
    pass

class ArtifactView(Static):
    """Displays artifact content (Code or Plan)."""
    def update_content(self, content: str, language: str = "text"):
        if language == "markdown" or language == "plan":
            self.update(Markdown(content))
        else:
            self.update(Syntax(content, language, theme="monokai", line_numbers=True))

class GlassCockpitApp(App):
    CSS = """
    #chat-sidebar {
        width: 50%;
        height: 100%;
        border-right: solid green;
        padding: 1;
    }
    #artifact-area {
        width: 50%;
        height: 100%;
        padding: 1;
    }
    Input {
        dock: bottom;
    }
    .user-msg {
        color: cyan;
        text-style: bold;
        margin-bottom: 1;
    }
    .system-msg {
        color: white;
        margin-bottom: 1;
    }
    .thinking-msg {
        color: yellow;
        text-style: italic;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            VerticalScroll(
                Static("System Online. Glass Cockpit Active.", classes="system-msg"),
                id="chat-scroll"
            ),
            Vertical(
                TabbedContent(id="artifact_tabs"),
                id="artifact-area"
            )
        )
        yield Input(placeholder="Command Elysia...")
        yield Footer()

    def on_mount(self) -> None:
        self.merkaba = Merkaba("GlassCockpit")
        try:
            self.merkaba.run_lifecycle()
        except Exception as e:
            self.notify(f"Lifecycle Init Warning: {e}", severity="warning")

        self.parser = StreamParser()
        self.current_response_widget = None
        self.current_artifact_content = ""
        self.current_artifact_meta = {}
        self.current_artifact_view = None
        self.is_streaming = False

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        if self.is_streaming:
            self.notify("System is processing...", severity="error")
            return

        user_input = message.value
        if not user_input.strip():
            return

        message.input.value = ""

        # Add User Message
        scroll = self.query_one("#chat-scroll")
        scroll.mount(Static(f"USER: {user_input}", classes="user-msg"))
        scroll.scroll_end()

        self.is_streaming = True
        self.process_input(user_input)

    @work
    async def process_input(self, text: str) -> None:
        try:
            # 1. Get response from Merkaba
            # We simulate the delay of 'Thinking' by just calling it.
            # In simulation, pulse returns the full string instantly.
            full_response = self.merkaba.pulse(text)

            # Prepare for streaming
            self.parser = StreamParser() # Reset parser state

            # Create a widget for the incoming system response
            scroll = self.query_one("#chat-scroll")
            self.current_response_widget = Static("", classes="system-msg")
            await scroll.mount(self.current_response_widget)

            # 2. Stream it
            chunk_size = 3 # Characters per tick
            for i in range(0, len(full_response), chunk_size):
                chunk = full_response[i:i+chunk_size]
                await asyncio.sleep(0.01) # Typing effect

                # Feed parser
                events = self.parser.feed(chunk)
                for event in events:
                    self.call_from_thread(self.handle_event, event)

            self.call_from_thread(self.finalize_stream)

        except Exception as e:
            self.notify(f"Error: {e}", severity="error")
            self.is_streaming = False

    def handle_event(self, event: StreamEvent):
        scroll = self.query_one("#chat-scroll")

        if event.type == EventType.TEXT:
            # Append to current chat bubble
            if self.current_response_widget:
                # Textual Static update replaces content. We need to append.
                # Efficiently, we should keep a buffer.
                # But here we just get the text from widget (if possible) or maintain state.
                # Current simple approach:
                old_text = str(self.current_response_widget.renderable)
                if old_text == "": old_text = ""
                # Actually renderable is a complex object.
                # Let's keep the buffer in the App instance for the *active* widget.
                if not hasattr(self, 'response_buffer'):
                    self.response_buffer = ""

                self.response_buffer += event.content
                self.current_response_widget.update(self.response_buffer)
                scroll.scroll_end()

        elif event.type == EventType.THINKING_START:
            # Maybe change color of current widget?
            if self.current_response_widget:
                self.current_response_widget.add_class("thinking-msg")
                self.response_buffer = "[Thinking]\n"
                self.current_response_widget.update(self.response_buffer)

        elif event.type == EventType.THINKING_CONTENT:
            self.response_buffer += event.content
            self.current_response_widget.update(self.response_buffer)
            scroll.scroll_end()

        elif event.type == EventType.THINKING_END:
            self.current_response_widget.remove_class("thinking-msg")
            self.response_buffer += "\n[End Thinking]\n"
            self.current_response_widget.update(self.response_buffer)
            # Create new widget for subsequent text
            self.current_response_widget = Static("", classes="system-msg")
            scroll.mount(self.current_response_widget)
            self.response_buffer = ""

        elif event.type == EventType.ARTIFACT_START:
            self.current_artifact_meta = event.metadata or {}
            self.current_artifact_content = ""

            title = self.current_artifact_meta.get("title", "Artifact")
            type_ = self.current_artifact_meta.get("type", "text")

            # Create Tab
            tabs = self.query_one("#artifact_tabs")

            # Check if tab exists? No, just add new one for history.
            # But prompt says "Right panel shows code".
            # Let's add a tab.
            tab_id = f"tab_{len(tabs.children)}"
            self.current_artifact_view = ArtifactView("")

            # Add pane (Textual 0.86+ might vary, assuming standard TabbedContent)
            # If add_pane doesn't exist, we might crash.
            # Safe way: use mount on content?
            # TabbedContent usually takes TabPane.
            tabs.add_pane(TabPane(title, self.current_artifact_view, id=tab_id))
            tabs.active = tab_id

            self.notify(f"Started Artifact: {title}")

        elif event.type == EventType.ARTIFACT_CONTENT:
            self.current_artifact_content += event.content
            # Update View
            lang = self.current_artifact_meta.get("language", "python")
            if self.current_artifact_meta.get("type") == "plan":
                lang = "plan"

            if self.current_artifact_view:
                self.current_artifact_view.update_content(self.current_artifact_content, lang)

        elif event.type == EventType.ARTIFACT_END:
            # Persistence
            self.save_current_artifact()
            self.current_artifact_view = None

    def save_current_artifact(self):
        title = self.current_artifact_meta.get("title", "untitled")
        content = self.current_artifact_content

        # Sanitize filename
        safe_title = "".join([c for c in title if c.isalnum() or c in "._- "]).strip().replace(" ", "_")
        if not safe_title: safe_title = "artifact"

        workspace = "workspace"
        os.makedirs(workspace, exist_ok=True)

        base_path = os.path.join(workspace, safe_title)
        ext = ""
        # Guess extension if not present
        if "." not in safe_title:
            lang = self.current_artifact_meta.get("language", "")
            if lang == "python": ext = ".py"
            elif lang == "markdown" or self.current_artifact_meta.get("type") == "plan": ext = ".md"
            else: ext = ".txt"

        # Versioning
        path = f"{base_path}{ext}"
        counter = 1
        while os.path.exists(path):
            path = f"{base_path}_v{counter}{ext}"
            counter += 1

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        self.notify(f"Saved artifact to: {path}", title="Persistence", severity="information", timeout=5.0)

    def finalize_stream(self):
        self.is_streaming = False
        self.response_buffer = ""

if __name__ == "__main__":
    app = GlassCockpitApp()
    app.run()
