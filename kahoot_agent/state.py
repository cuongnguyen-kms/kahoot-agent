from typing import TypedDict, Any, Optional, List
from playwright.async_api import Browser, Page

class KahootChoice(TypedDict, total=False):
    text: Optional[str]
    img_url: Optional[str]
    img_base64: Optional[str]

class KahootAgentState(TypedDict, total=False):
    browser: Browser
    page: Page
    step: Optional[str]
    question: Optional[str]
    choices: Optional[List[KahootChoice]]