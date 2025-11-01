"""
Tool Manager for Elysia.

This module is responsible for discovering, loading, and providing information
about the available tools in the 'tools' directory. It allows other components,
like the GoalDecompositionCortex, to get a dynamic list of available tool
signatures for planning.
"""
import os
import inspect
import importlib
import logging

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

class ToolManager:
    """
    Dynamically discovers and manages available tools.
    """
    def __init__(self):
        self.tools_directory = os.path.dirname(__file__)
        self.available_tools = self._discover_tools()

    def _discover_tools(self):
        """
        Scans the tools directory to find and load tool classes.
        A file is considered a tool if it contains a class that is not this
        ToolManager class itself. This is a simple heuristic.
        """
        tools = {}
        for filename in os.listdir(self.tools_directory):
            if filename.endswith('.py') and filename != '__init__.py' and filename != 'tool_manager.py':
                module_name = f"tools.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if obj.__module__ == module_name:
                            tools[name] = obj
                except Exception as e:
                    # Log the error but continue discovering other tools
                    logger.warning(f"Error discovering tool in {filename}: {e}", exc_info=False)
        return tools

    def get_tool_signatures(self):
        """
        Returns a list of simplified signatures for the available tools.
        This method now provides a more robust, hardcoded list to ensure
        the planner always knows about critical tools, regardless of discovery success.
        """
        # Hardcode the most critical tool signatures for stability.
        # This prevents discovery errors (like the one in query_tool.py)
        # from breaking the entire planning system.
        stable_signatures = [
            "search_web(query: str) - Searches the web for up-to-date information on a topic.",
            "read_file(filepath: str) - Reads the content of a specified file.",
            "write_file(filepath: str, content: str) - Writes content to a specified file.",
            "list_files(directory: str) - Lists all files in a specified directory.",
            "summarize_text(text: str) - Summarizes a long piece of text into key points.",
            "query_knowledge_graph(query: str) - Queries Elysia's internal knowledge base for facts and relationships.",
            "create_image(prompt: str) - Creates an image based on a descriptive prompt."
        ]

        # We can still attempt to dynamically add more, but the core ones are guaranteed.
        discovered_signatures = []
        tool_function_map = {
            "QueryTool": "search_web(query: str) - Searches the web for up-to-date information on a topic.",
            "CanvasTool": "create_image(prompt: str) - Creates an image based on a descriptive prompt.",
            "KGManager": "query_knowledge_graph(query: str) - Queries Elysia's internal knowledge.",
            "CausalLearner": "find_causal_links(data: dict) - Analyzes data to find causal relationships."
        }
        for name in self.available_tools:
            if name in tool_function_map and tool_function_map[name] not in stable_signatures:
                discovered_signatures.append(tool_function_map[name])

        return stable_signatures + discovered_signatures

    def get_tool(self, name):
        """
        Retrieves a tool class by its name.
        """
        return self.available_tools.get(name)
