import httpx
import base64
from typing import Optional

async def fetch_image_base64(url: str, timeout: float = 2.0) -> Optional[str]:
  """
  Fetches an image at the given URL and return its base64-encoded string
  """
  try:
    async with httpx.AsyncClient(timeout=timeout) as client:
      response = await client.get(url)
      response.raise_for_status()
      return base64.b64encode(response.content).decode('utf-8')
  except Exception as e:
    print(f"Error fetching image: {e}")
    return None

async def extract_text(page, selector: str) -> str:
    el = await page.query_selector(selector)
    return await el.inner_text() if el else ""

async def extract_attribute(page, selector: str, attr: str) -> Optional[str]:
  el = await page.query_selector(selector)
  return await el.get_attribute(attr) if el else None

async def extract_choices_with_images(buttons) -> Tuple[List[Dict[str, Any]], List[str]]:
  choices = []
  image_urls = []
  for btn in buttons:
      text_el = await btn.query_selector(selectors.CHOICE_TEXT_SELECTOR)
      text = await text_el.inner_text() if text_el else ""
      img_el = await btn.query_selector('img')
      img_url = await img_el.get_attribute("src") if img_el else None
      if img_url:
          image_urls.append(img_url)
      choices.append({"text": text, "img_url": img_url})
  return choices, image_urls

def build_gpt_input_blocks(question: str, question_img_url: Optional[str], choices: List[Dict[str, Any]], url_to_base64: Dict[str, str]) -> List[Dict[str, Any]]:
  blocks = [{"type": "text", "text": f"Question: {question}"}]

  if question_img_url and question_img_url in url_to_base64:
    blocks.append({
      "type": "image_url",
      "image_url": {"url": f"data:image/png;base64,{url_to_base64[question_img_url]}"}
    })

  for i, choice in enumerate(choices):
    label = f"Choice {i + 1}: {choice['text'] or '[IMAGE]'}"
    blocks.append({"type": "text", "text": label})
    if choice["img_url"] and choice["img_url"] in url_to_base64:
      blocks.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{url_to_base64[choice['img_url']]}"},
      })

  blocks.append({"type": "text", "text": "Reply with just the correct answer text."})
  return blocks