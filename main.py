import os
import asyncio
from dotenv import load_dotenv

from kahoot_agent.agent import kahoot_game_loop_with_react_agent
from kahoot_agent.file_reader_tool import read_external_file
from kahoot_agent.search_tool import search_knowledge
from kahoot_agent.image_tools import ocr_image_tool
from kahoot_agent.search_recent_news_tool import search_recent_news

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
NICKNAME = os.environ.get("NICKNAME", "AI_Player")
KAHOOT_URL = os.environ.get("KAHOOT_URL")
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-4o-mini")

# Azure OpenAI support
USE_AZURE = os.environ.get("AZURE_OPENAI", "false").lower() == "true"
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

tools = [read_external_file, search_knowledge, ocr_image_tool, search_recent_news]  # Added search_recent_news tool

async def main():
    await kahoot_game_loop_with_react_agent(
        OPENAI_API_KEY,
        KAHOOT_URL,
        GPT_MODEL,
        NICKNAME,
        tools,
        use_azure=USE_AZURE,
        azure_endpoint=AZURE_ENDPOINT,
        azure_deployment=AZURE_DEPLOYMENT,
        azure_api_version=AZURE_API_VERSION
    )
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(main())