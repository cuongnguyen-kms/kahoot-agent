import requests
from langchain_core.tools import tool
import re

@tool
def search_recent_news(query: str) -> str:
    """
    Search for recent news articles about a given topic (e.g., Ukraine, Trump) using Google Programmable Search Engine (CSE).
    Returns headlines with links. Requires Google API key and CSE ID.
    """
    GOOGLE_API_KEY="AIzaSyAHqXdEXMcvmDZNVahcBjtCKhPDA6OtJ98"
    try:
        api_key = GOOGLE_API_KEY  # Replace with your Google API key
        cse_id = "a46d4451b863a4c52"            # Replace with your Programmable Search Engine ID
        endpoint = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": 5,
            "hl": "en"
        }
        resp = requests.get(endpoint, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        if not items:
            return "No news found."
        results = []
        for item in items:
            title = item.get("title", "No title")
            link = item.get("link", "")
            results.append(f"- {title}: {link}")
        return "\n".join(results)
    except Exception as e:
        return f"[News Search Error] {e}"
