import asyncio
import functools
from typing import TYPE_CHECKING
from playwright.async_api import async_playwright
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from .state import KahootAgentState
from . import selectors
from .helpers import (
  extract_text,
  extract_attribute,
  extract_choices_with_images,
  find_confident_kms_match,
  build_gpt_input_blocks,
  build_openai_prompt
)
from .file_reader_tool import read_external_file
from .constants import SYSTEM_PROMPT

if TYPE_CHECKING:
    from playwright.async_api import Browser, Page

# ---- Agent Building ----
def build_react_agent(
    openai_api_key: str,
    tools: list,
    gpt_model: str = "gpt-4o-mini",
    system_prompt_override: str = None,
    use_azure: bool = False,
    azure_endpoint: str = None,
    azure_deployment: str = None,
    azure_api_version: str = None
):
    """
    Build a React agent using LangGraph's prebuilt create_react_agent, with provided tools and OpenAI or Azure OpenAI model.
    Optionally override the default system prompt.
    Set use_azure=True and provide azure_endpoint, azure_deployment, azure_api_version for Azure OpenAI.
    """
    if use_azure:
        llm = ChatOpenAI(
            api_key=openai_api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment or gpt_model,
            api_version=azure_api_version or "2024-02-15-preview",
            model=gpt_model,
        )
    else:
        llm = ChatOpenAI(api_key=openai_api_key, model=gpt_model)
    agent = create_react_agent(llm, tools)
    return agent

# --- Kahoot game loop using React agent ---
async def kahoot_game_loop_with_react_agent(
    openai_api_key: str,
    kahoot_url: str,
    gpt_model: str,
    nickname: str,
    tools: list,
    system_prompt_override: str = None,
    use_azure: bool = False,
    azure_endpoint: str = None,
    azure_deployment: str = None,
    azure_api_version: str = None
):
    """
    Main Kahoot game loop using the React agent for answering questions.
    Supports Azure OpenAI if use_azure and Azure params are provided.
    """
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    page = await browser.new_page()
    await page.goto(kahoot_url)
    await page.fill(selectors.NICKNAME_INPUT_SELECTOR, nickname)
    await page.click(selectors.JOIN_BUTTON_SELECTOR)
    print(f"[*] Joined game as {nickname}")

    # Build React agent with system prompt
    agent = build_react_agent(
        openai_api_key,
        tools,
        gpt_model,
        system_prompt_override,
        use_azure=use_azure,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        azure_api_version=azure_api_version
    )

    # Main loop: wait for questions and answer
    while True:
        try:
            print("[*] Waiting for question...")
            await page.wait_for_selector(selectors.QUESTION_SELECTOR, timeout=0)
            question = await extract_text(page, selectors.QUESTION_SELECTOR)
            answer_buttons = await page.query_selector_all(selectors.ANSWER_BUTTONS_SELECTOR)
            choices, _ = await extract_choices_with_images(answer_buttons)
            choice_texts = [c["text"] for c in choices]
            print(f"Question: {question}")
            print(f"Choices: {choice_texts}")

            # Use React agent to get answer
            user_content = f"Question: {question}\nChoices: {', '.join(choice_texts)}"
            result = await agent.ainvoke({
                "messages": [
                    {"role": "system", "content": system_prompt_override or SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                "choices": choice_texts,
                "max_tokens": 8,
                "temperature": 0.5
            })
            model_answer = ""
            if "messages" in result and result["messages"]:
                last_msg = result["messages"][-1]
                model_answer = getattr(last_msg, "content", "") or (last_msg.get("content", "") if isinstance(last_msg, dict) else "")
            model_answer = model_answer.strip()
            print(f"[Agent] Model selected: {model_answer}")

            # Click the answer that matches model output
            selected_idx = next((i for i, c in enumerate(choice_texts) if model_answer.lower() in (c or '').lower()), 0)
            await answer_buttons[selected_idx].click()
            print(f"[*] Clicked answer: {choice_texts[selected_idx]}")

            # Wait for next question or end
            try:
                await page.wait_for_selector(selectors.QUESTION_SELECTOR, timeout=0)
            except Exception:
                print("[*] Game ended or timed out.")
                break
        except Exception as e:
            print(f"[!] Error in game loop: {e}")
            break
    await browser.close()
    print("[*] Game loop finished.")