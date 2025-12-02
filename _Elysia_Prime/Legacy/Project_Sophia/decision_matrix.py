# [Genesis: 2025-12-02] Purified by Elysia
# This file will contain the DecisionMatrix,
# the part of Elly's brain responsible for choosing the best information source.

class DecisionMatrix:
    def choose(self, search_results, topic):
        """
        Chooses the best URL from a list of search results based on a simple heuristic.
        It prioritizes sources that seem to offer definitions or philosophical insights.
        """
        # print(f"[{self.__class__.__name__}] I have several options. I must choose the wisest path.")

        best_choice = None
        highest_score = -1

        for result in search_results:
            score = 0
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()

            # Basic relevance: does it contain the topic?
            if topic in title or topic in snippet:
                score += 1

            # Preference for definitions and deeper meaning
            if 'definition' in title or 'what is' in title:
                score += 2
            if 'philosophy' in title or 'philosophical' in snippet:
                score += 5 # Increased score to ensure this is chosen
            if 'important' in title or 'why' in title:
                score += 2

            # print(f"[{self.__class__.__name__}] Evaluating '{title}'... Score: {score}")

            if score > highest_score:
                highest_score = score
                best_choice = result

        if best_choice:
            # print(f"[{self.__class__.__name__}] I have made my choice. The best source appears to be: '{best_choice['title']}'")
            return best_choice['url']
        else:
            # print(f"[{self.__class__.__name__}] None of the options seem truly useful to me.")
            return None