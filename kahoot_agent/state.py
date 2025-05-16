from typing import TypedDict, Any, Optional
from playwright.async_api import Browser, Page

class KahootAgentState(TypedDict, total=False):
    browser: Browser
    page: Page
    step: Optional[str]
    question: Optional[str]
    choices: Optional[list[str]]