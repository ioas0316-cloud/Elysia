from typing import List, Dict, Optional, Callable
import logging

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
web_search_logger = logging.getLogger(__name__)

class WebSearchCortex:
    """
    Provides Elysia with the ability to search the internet for information.
    It uses available tools to perform a web search and retrieve content.
    """

    def __init__(self, google_search_func: Callable, view_website_func: Callable):
        """
        Initializes the WebSearchCortex with dependency-injected search functions.
        """
        self.google_search = google_search_func
        self.view_text_website = view_website_func

    def search(self, query: str) -> Optional[str]:
        """
        Searches the web for a given query and returns the content of the most relevant result.

        Args:
            query: The search query.

        Returns:
            The text content of the most relevant webpage, or None if no results are found.
        """
        web_search_logger.info(f"Performing web search for query: '{query}'")
        try:
            # Step 1: Use the injected google_search to get a list of URLs
            search_results = self.google_search(query=query)

            if not search_results:
                web_search_logger.warning("Google search returned no results.")
                return None

            # Step 2: Select the top result (most relevant)
            top_result_url = search_results[0].get('url')
            if not top_result_url:
                web_search_logger.warning("Top search result had no URL.")
                return None

            web_search_logger.info(f"Found top URL: {top_result_url}")

            # Step 3: Use the injected view_text_website to retrieve the content
            content = self.view_text_website(url=top_result_url)

            if not content:
                web_search_logger.warning(f"Could not retrieve content from URL: {top_result_url}")
                return None

            web_search_logger.info(f"Successfully retrieved content of length {len(content)}.")
            return content

        except Exception as e:
            web_search_logger.error(f"An error occurred during web search for query '{query}': {e}", exc_info=True)
            return None

if __name__ == '__main__':
    # Example usage for direct testing
    search_cortex = WebSearchCortex()
    test_query = "What is a black hole?"
    print(f"--- Testing WebSearchCortex with query: '{test_query}' ---")
    result = search_cortex.search(test_query)
    if result:
        print("\n--- Search Result ---")
        # Print a snippet of the result
        print(result[:500] + "...")
        print("\n--- End of Snippet ---")
    else:
        print("\n--- No result found. ---")
