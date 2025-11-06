
from Project_Sophia.safety_guardian import SafetyGuardian

class WebSearchCortex:
    def __init__(self, search_tool=None):
        """
        Initializes the WebSearchCortex.
        This cortex is responsible for interfacing with the web.

        Args:
            search_tool: The tool to use for searching. If None, it will default to google_search.
        """
        self.guardian = SafetyGuardian()
        self.search_tool = search_tool

    def search(self, query: str):
        """
        Performs a web search for the given query using the configured search tool.
        The results are then filtered by the SafetyGuardian.

        Args:
            query: The search query string.

        Returns:
            The filtered search results.
        """
        if self.search_tool is None:
            from agents.tools import google_search
            self.search_tool = google_search

        print(f"[WebSearchCortex] Performing search for: {query}")
        try:
            # Call the configured search tool
            search_results = self.search_tool(query=query)

            # Filter the results for harmful content
            filtered_results = self.guardian.filter_search_results(search_results)

            return filtered_results
        except Exception as e:
            print(f"[WebSearchCortex] An error occurred during the search: {e}")
            return {"error": f"Failed to execute search for '{query}'. Reason: {e}"}
