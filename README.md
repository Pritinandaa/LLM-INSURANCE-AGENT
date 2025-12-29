import json
import os
import time
import requests
from crewai.tools import BaseTool


class SearchInternetTool(BaseTool):

    def __init__(self):
        super().__init__(
            name="search_internet",
            description="Search the internet for information about a given topic and return relevant results."
        )

    def _run(self, query: str) -> str:
        """
        Perform a search query on the internet using the Serper.dev API.

        Args:
            query (str): The search query.

        Returns:
            str: The formatted search results.
        """
        print(f"[SERPER] Searching internet for: {query}")
        top_result_to_return = 4
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': os.getenv('SERPER_API_KEY'),
            'content-type': 'application/json'
        }

        # retry on transient errors; return graceful message on 401/403
        max_attempts = 3
        backoff = 1
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(url, headers=headers, data=payload, timeout=15)
            except requests.RequestException:
                response = None
            if response is None:
                time.sleep(backoff)
                backoff *= 2
                continue
            if response.status_code == 200:
                print(f"[SERPER] Successfully retrieved {len(response.json().get('organic', []))} search results")
                break
            if response.status_code in (401, 403):
                print(f"[SERPER] API key invalid or rate-limited (status {response.status_code})")
                return f"Search unavailable: API key invalid or rate-limited (status {response.status_code})."
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 2
                continue
            # other non-retryable errors
            return f"Search failed with status code: {response.status_code}"

        results = response.json().get('organic', []) if response is not None else []
        formatted_results = []

        for result in results[:top_result_to_return]:
            try:
                formatted_results.append('\n'.join([
                    f"Title: {result['title']}",
                    f"Link: {result['link']}",
                    f"Snippet: {result['snippet']}",
                    "\n-----------------"
                ]))
            except KeyError:
                continue

        return '\n'.join(formatted_results)


class SearchNewsTool(BaseTool):

    def __init__(self):
        super().__init__(
            name="search_news",
            description="Search for news about a company, stock, or other topics and return relevant results."
        )

    def _run(self, query: str) -> str:
        """
        Perform a news search query using the Serper.dev API.

        Args:
            query (str): The news query.

        Returns:
            str: The formatted news results.
        """
        print(f"[SERPER] Searching news for: {query}")
        top_result_to_return = 4
        url = "https://google.serper.dev/news"
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': os.getenv('SERPER_API_KEY'),
            'content-type': 'application/json'
        }

        # retry on transient errors; return graceful message on 401/403
        max_attempts = 3
        backoff = 1
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(url, headers=headers, data=payload, timeout=15)
            except requests.RequestException:
                response = None
            if response is None:
                time.sleep(backoff)
                backoff *= 2
                continue
            if response.status_code == 200:
                print(f"[SERPER] Successfully retrieved {len(response.json().get('news', []))} news results")
                break
            if response.status_code in (401, 403):
                print(f"[SERPER] News API key invalid or rate-limited (status {response.status_code})")
                return f"News search unavailable: API key invalid or rate-limited (status {response.status_code})."
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 2
                continue
            return f"News fetch failed with status code: {response.status_code}"

        results = response.json().get('news', []) if response is not None else []
        formatted_results = []

        for result in results[:top_result_to_return]:
            try:
                formatted_results.append('\n'.join([
                    f"Title: {result['title']}",
                    f"Link: {result['link']}",
                    f"Snippet: {result['snippet']}",
                    "\n-----------------"
                ]))
            except KeyError:
                continue

        return '\n'.join(formatted_results)
