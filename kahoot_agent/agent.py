import asyncio
import os
import httpx
import functools
from typing import TYPE_CHECKING, List, Dict, Optional, Any
from playwright.async_api import async_playwright
from openai import AsyncOpenAI
from langgraph.graph import StateGraph, END

from .state import KahootAgentState
from . import selectors
from .helpers import (
  fetch_image_base64,
  fetch_images_base64,
  extract_text,
  extract_attribute,
  extract_choices_with_images,
  find_confident_kms_match,
  build_gpt_input_blocks,
  build_openai_prompt
)
from .constants import KMS_KEYWORDS

if TYPE_CHECKING:
    from playwright.async_api import Browser, Page

# ---- Node Steps ----
async def join_game_node(state: KahootAgentState, kahoot_url: str) -> KahootAgentState:
  """
  Launch browser and join kahoot game.
  """
  print("[*] Launching browser and joining Kahoot...")
  playwright = await async_playwright().start()
  browser = await playwright.chromium.launch(headless=False)
  page = await browser.new_page()
  await page.goto(kahoot_url)
  state["browser"] = browser
  state["page"] = page
  state["step"] = "nick"
  return state

async def enter_nickname_node(state: KahootAgentState, nickname: str) -> KahootAgentState:
  """
  Enter the nickname for the Kahoot game.
  """
  print("[*] Entering nickname...")
  page = state["page"]
  await page.fill(selectors.NICKNAME_INPUT_SELECTOR, nickname)
  await page.click(selectors.JOIN_BUTTON_SELECTOR)
  state["step"] = "wait_lobby"
  return state

async def wait_for_question_node(state: KahootAgentState) -> KahootAgentState:
  """
  Wait for the question to start.
  """
  page = state["page"]
  print("[*] Waiting for question to start...")
  await page.wait_for_selector(selectors.QUESTION_SELECTOR, timeout=0)
  state["step"] = "question"
  return state

async def answer_question_node(
  state: KahootAgentState,
  openai_api_key: str,
  gpt_model: str
) -> KahootAgentState:
  """
  Answer the question using OpenAI API or KMS knowledge base.
  """
  print("[*] Extracting question and choices...")
  page = state["page"]
  await asyncio.sleep(0.5)

  # Extract question and choices
  question = await extract_text(page, selectors.QUESTION_SELECTOR)
  question_img_url = await extract_attribute(page, selectors.QUESTION_IMAGE_SELECTOR, "src")
  answer_buttons = await page.query_selector_all(selectors.ANSWER_BUTTONS_SELECTOR)
  choices, image_urls = await extract_choices_with_images(answer_buttons)

  # Fetch images as base64 if present
  all_image_urls = image_urls + ([question_img_url] if question_img_url else [])
  url_to_base64 = await fetch_images_base64(all_image_urls)

  print(f"Question: {question}")
  print(f"Choices: {choices}")

  # Try KMS knowledge base first
  confident_match_idx, kb_results = await find_confident_kms_match(question, choices, answer_buttons)
  if confident_match_idx is not None:
      print(f"[Agent] Confident match found in knowledge base: {choices[confident_match_idx]['text']}")
      await answer_buttons[confident_match_idx].click()
      state.update({
          "step": "wait_next",
          "question": question,
          "choices": [c["text"] for c in choices]
      })
      return state

  # Prepare GPT input
  content_blocks = build_gpt_input_blocks(question, question_img_url, choices, url_to_base64)

  # Fallback to OpenAI
  system_prompt, user_content = build_openai_prompt(kb_results, content_blocks)
  client = AsyncOpenAI(api_key=openai_api_key)
  response = await client.chat.completions.create(
      model=gpt_model,
      messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_content}
      ],
      max_tokens=8,
      temperature=0.5
  )
  model_answer = response.choices[0].message.content.strip().strip('.').strip()
  print(f"Model response: {model_answer}")

  # Select the answer that matches model output
  selected_idx = next((i for i, c in enumerate(choices) if model_answer.lower() in (c["text"] or "").lower()), 0)
  await answer_buttons[selected_idx].click()

  state.update({
      "step": "wait_next",
      "question": question,
      "choices": [c["text"] for c in choices]
  })
  return state

async def wait_next_question_node(state: KahootAgentState) -> KahootAgentState:
  """
  Wait for the next question or game over.
  """
  print("[*] Waiting for next question (or end)...")
  page = state["page"]
  try:
      await page.wait_for_selector(selectors.QUESTION_SELECTOR, timeout=0)
      state["step"] = "question"
  except Exception:
      print("[*] Game ended or timed out.")
      state["step"] = END
  return state

# ---- Agent Building ----
def build_agent(
  openai_api_key: str,
  kahoot_url: str,
  gpt_model: str,
  nickname: str
):
  """
  Returns the compiled StateGraph agent.
  """
  sg = StateGraph(KahootAgentState)
  sg.add_node("join_game", functools.partial(join_game_node, kahoot_url=kahoot_url))
  sg.add_node("nickname", functools.partial(enter_nickname_node, nickname=nickname))
  sg.add_node("wait_question", wait_for_question_node)
  sg.add_node("answer_question", functools.partial(answer_question_node, openai_api_key=openai_api_key, gpt_model=gpt_model))
  sg.add_node("wait_next", wait_next_question_node)

  sg.set_entry_point("join_game")
  sg.add_edge("join_game", "nickname")
  sg.add_edge("nickname", "wait_question")
  sg.add_edge("wait_question", "answer_question")
  sg.add_edge("answer_question", "wait_next")
  sg.add_edge("wait_next", "answer_question")
  return sg.compile()