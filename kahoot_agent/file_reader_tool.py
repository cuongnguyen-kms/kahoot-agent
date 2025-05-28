import httpx
from langchain_core.tools import tool
import re
import requests

def _convert_gdrive_url(url: str) -> str:
    """
    Convert various Google Drive sharing/view URLs to a direct download link.
    """
    # Handle 'file/d/FILEID' pattern
    match = re.search(r'drive\.google\.com/file/d/([\w-]+)', url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    # Handle 'uc?id=FILEID' pattern
    match = re.search(r'drive\.google\.com/uc\?id=([\w-]+)', url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    # Handle sharing links
    match = re.search(r'drive\.google\.com/open\?id=([\w-]+)', url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

@tool
def read_external_file(url: str) -> str:
    """
    Download and return the content of a text file from a given URL (e.g., Google Drive .txt link).
    """
    try:
        url = _convert_gdrive_url(url)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        text = resp.text
        # Try to extract only the file content if Google Drive returns HTML
        if 'content="text/html' in resp.text and 'google' in url:
            # Try to extract file content from <pre> or <body>
            pre_match = re.search(r'<pre.*?>(.*?)</pre>', resp.text, re.DOTALL)
            if pre_match:
                return pre_match.group(1).strip()
            body_match = re.search(r'<body.*?>(.*?)</body>', resp.text, re.DOTALL)
            if body_match:
                return body_match.group(1).strip()
            # Fallback: return all text
            return text
        return text
    except Exception as e:
        return f"[File Read Error] {e}"
