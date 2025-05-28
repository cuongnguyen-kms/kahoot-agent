from langchain_core.tools import tool
from PIL import Image
import requests
import pytesseract
from io import BytesIO

@tool
def ocr_image_tool(image_url: str) -> str:
    """
    Extract text from an image URL using OCR.
    """
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"[OCR Error] {e}"
